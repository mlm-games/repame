//! TCP transport: `std::net`, a reader thread per connection plus one writer
//! thread, `u32`-length-prefixed frames.
//!
//! TCP already delivers bytes reliably and in order, so a frame carries no
//! sequence numbers and every channel behaves as reliable-ordered no matter
//! what reliability the [`ChannelTable`] records for it. That degradation is
//! deliberate; use [`crate::transport::udp::UdpTransport`] when an unreliable
//! class is actually needed.
//!
//! `send` never touches the socket: it queues to the writer thread, so a slow
//! or stalled peer cannot block the game loop. A write that fails (including
//! hitting the write timeout) ends that peer and is reported as
//! [`TransportEvent::Error`] followed by [`TransportEvent::Disconnected`].

use std::collections::BTreeMap;
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use std::collections::BTreeSet;

use crate::channel::{ChannelId, ChannelTable};
use crate::error::NetError;
use crate::protocol::{FramedMessage, decode_frame_body, encode_frame, frame_length};
use crate::transport::{DisconnectReason, PeerId, Transport, TransportEvent};

/// How often a blocking thread wakes to notice shutdown.
const TICK: Duration = Duration::from_millis(20);

/// Write timeout: a peer that cannot accept a frame this long is dropped.
const WRITE_TIMEOUT: Duration = Duration::from_secs(5);

/// How long a blocking read waits before re-checking shutdown.
const READ_TIMEOUT: Duration = Duration::from_millis(50);

enum Outbound {
    Register {
        peer: PeerId,
        stream: TcpStream,
    },
    Send {
        peer: PeerId,
        channel: ChannelId,
        payload: Vec<u8>,
    },
    Close,
}

/// TCP transport owning a listener (server) or one connection (client).
///
/// A client talks to peer `1`; a server assigns peers `1, 2, 3, ...` in
/// accept order. Peer ids are only valid for this transport's lifetime.
pub struct TcpTransport {
    incoming: Receiver<Vec<TransportEvent>>,
    outgoing: Sender<Outbound>,
    local_addr: SocketAddr,
    channels: Arc<ChannelTable>,
    peers: Arc<Mutex<BTreeSet<PeerId>>>,
    shutdown: Arc<AtomicBool>,
    writer: Option<JoinHandle<()>>,
    accept: Option<JoinHandle<()>>,
    closed: bool,
}

impl TcpTransport {
    /// Connect to `remote` and talk to it as peer `1`.
    pub fn client(remote: impl ToSocketAddrs, channels: ChannelTable) -> Result<Self, NetError> {
        let stream = TcpStream::connect(remote)?;
        stream.set_nodelay(true)?;
        let local_addr = stream.local_addr()?;
        let (events_tx, events_rx) = std::sync::mpsc::channel();
        let (out_tx, out_rx) = std::sync::mpsc::channel();
        let shutdown = Arc::new(AtomicBool::new(false));
        let peers = Arc::new(Mutex::new(BTreeSet::new()));
        let table = Arc::new(channels.clone());
        let writer = {
            let shutdown = shutdown.clone();
            let peers = peers.clone();
            let events = events_tx.clone();
            thread::Builder::new()
                .name("repame-net-tcp-writer".to_string())
                .spawn(move || writer_loop(out_rx, events, peers, shutdown))
                .map_err(NetError::Io)?
        };
        let reader_stream = stream.try_clone()?;
        spawn_reader(reader_stream, 1, events_tx, shutdown.clone())?;
        // The writer registers the stream in order, but mark peer 1 known here
        // so a send issued right after `client` cannot race the writer.
        if let Ok(mut set) = peers.lock() {
            set.insert(1);
        }
        out_tx
            .send(Outbound::Register { peer: 1, stream })
            .map_err(|_| NetError::Closed)?;
        Ok(Self {
            incoming: events_rx,
            outgoing: out_tx,
            local_addr,
            channels: table,
            peers,
            shutdown,
            writer: Some(writer),
            accept: None,
            closed: false,
        })
    }

    /// Serve every connection the listener accepts.
    pub fn server(listener: TcpListener, channels: ChannelTable) -> Result<Self, NetError> {
        let local_addr = listener.local_addr()?;
        listener.set_nonblocking(true)?;
        let (events_tx, events_rx) = std::sync::mpsc::channel();
        let (out_tx, out_rx) = std::sync::mpsc::channel();
        let shutdown = Arc::new(AtomicBool::new(false));
        let peers = Arc::new(Mutex::new(BTreeSet::new()));
        let table = Arc::new(channels.clone());
        let writer = {
            let shutdown = shutdown.clone();
            let peers = peers.clone();
            let events = events_tx.clone();
            thread::Builder::new()
                .name("repame-net-tcp-writer".to_string())
                .spawn(move || writer_loop(out_rx, events, peers, shutdown))
                .map_err(NetError::Io)?
        };
        let accept = {
            let shutdown = shutdown.clone();
            let events = events_tx.clone();
            let accept_tx = out_tx.clone();
            let next_peer = Arc::new(AtomicU32::new(1));
            thread::Builder::new()
                .name("repame-net-tcp-accept".to_string())
                .spawn(move || accept_loop(listener, accept_tx, events, next_peer, shutdown))
                .map_err(NetError::Io)?
        };
        Ok(Self {
            incoming: events_rx,
            outgoing: out_tx,
            local_addr,
            channels: table,
            peers,
            shutdown,
            writer: Some(writer),
            accept: Some(accept),
            closed: false,
        })
    }

    /// Bind `addr` and serve it. `addr` may use port `0`.
    pub fn bind(addr: impl ToSocketAddrs, channels: ChannelTable) -> Result<Self, NetError> {
        let listener = TcpListener::bind(addr)?;
        Self::server(listener, channels)
    }
}

impl Transport for TcpTransport {
    fn send(&mut self, peer: PeerId, channel: ChannelId, payload: &[u8]) -> Result<(), NetError> {
        self.channels.reliability(channel)?;
        let known = self
            .peers
            .lock()
            .map(|set| set.contains(&peer))
            .unwrap_or(false);
        if !known {
            return Err(NetError::UnknownPeer(peer));
        }
        self.outgoing
            .send(Outbound::Send {
                peer,
                channel,
                payload: payload.to_vec(),
            })
            .map_err(|_| NetError::Closed)
    }

    fn poll(&mut self) -> Result<Vec<TransportEvent>, NetError> {
        let mut events = Vec::new();
        loop {
            match self.incoming.try_recv() {
                Ok(batch) => events.extend(batch),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    if events.is_empty() {
                        return Err(NetError::Closed);
                    }
                    break;
                }
            }
        }
        Ok(events)
    }

    fn close(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        self.shutdown.store(true, Ordering::Relaxed);
        let _ = self.outgoing.send(Outbound::Close);
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
        if let Some(accept) = self.accept.take() {
            let _ = accept.join();
        }
    }

    fn local_addr(&self) -> Option<SocketAddr> {
        Some(self.local_addr)
    }
}

impl Drop for TcpTransport {
    fn drop(&mut self) {
        self.close();
    }
}

fn spawn_reader(
    stream: TcpStream,
    peer: PeerId,
    events: Sender<Vec<TransportEvent>>,
    shutdown: Arc<AtomicBool>,
) -> Result<(), NetError> {
    thread::Builder::new()
        .name(format!("repame-net-tcp-reader-{peer}"))
        .spawn(move || reader_loop(stream, peer, events, shutdown))
        .map(|_| ())
        .map_err(NetError::Io)
}

/// Push one event to the game as a single-event batch.
fn send_event(events: &Sender<Vec<TransportEvent>>, event: TransportEvent) {
    let _ = events.send(vec![event]);
}

/// Fill `buf` completely. `Ok(false)` means a clean end of stream or shutdown.
fn read_full(stream: &mut TcpStream, buf: &mut [u8], shutdown: &AtomicBool) -> io::Result<bool> {
    let mut filled = 0;
    while filled < buf.len() {
        if shutdown.load(Ordering::Relaxed) {
            return Ok(false);
        }
        match stream.read(&mut buf[filled..]) {
            Ok(0) => return Ok(false),
            Ok(n) => filled += n,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock
                        | io::ErrorKind::TimedOut
                        | io::ErrorKind::Interrupted
                ) => {}
            Err(error) => return Err(error),
        }
    }
    Ok(true)
}

fn reader_loop(
    mut stream: TcpStream,
    peer: PeerId,
    events: Sender<Vec<TransportEvent>>,
    shutdown: Arc<AtomicBool>,
) {
    let _ = stream.set_read_timeout(Some(READ_TIMEOUT));
    let mut header = [0u8; 4];
    loop {
        if shutdown.load(Ordering::Relaxed) {
            return;
        }
        match read_full(&mut stream, &mut header, &shutdown) {
            Ok(true) => {}
            Ok(false) => {
                let _ = send_event(&events, TransportEvent::Disconnected {
                    peer,
                    reason: DisconnectReason::PeerClosed,
                });
                return;
            }
            Err(error) => {
                let _ = send_event(&events, TransportEvent::Error {
                    peer: Some(peer),
                    error: NetError::Io(error),
                });
                let _ = send_event(&events, TransportEvent::Disconnected {
                    peer,
                    reason: DisconnectReason::Io("read failed".to_string()),
                });
                return;
            }
        }
        let length = match frame_length(header) {
            Ok(length) => length,
            Err(error) => {
                let reason = DisconnectReason::Protocol(error.to_string());
                let _ = send_event(
                    &events,
                    TransportEvent::Error {
                        peer: Some(peer),
                        error: NetError::Codec(error),
                    },
                );
                let _ = send_event(
                    &events,
                    TransportEvent::Disconnected { peer, reason },
                );
                return;
            }
        };
        let mut body = vec![0u8; length];
        match read_full(&mut stream, &mut body, &shutdown) {
            Ok(true) => {}
            Ok(false) => {
                let _ = send_event(&events, TransportEvent::Disconnected {
                    peer,
                    reason: DisconnectReason::PeerClosed,
                });
                return;
            }
            Err(error) => {
                let _ = send_event(&events, TransportEvent::Error {
                    peer: Some(peer),
                    error: NetError::Io(error),
                });
                return;
            }
        }
        match decode_frame_body(&body) {
            Ok(messages) => {
                let batch: Vec<TransportEvent> = messages
                    .into_iter()
                    .map(|message| TransportEvent::Message {
                        peer,
                        channel: message.channel,
                        payload: message.payload,
                    })
                    .collect();
                if !batch.is_empty() && events.send(batch).is_err() {
                    return;
                }
            }
            Err(error) => {
                let reason = DisconnectReason::Protocol(error.to_string());
                let _ = send_event(
                    &events,
                    TransportEvent::Error {
                        peer: Some(peer),
                        error: NetError::Codec(error),
                    },
                );
                let _ = send_event(
                    &events,
                    TransportEvent::Disconnected { peer, reason },
                );
                return;
            }
        }
    }
}

fn writer_loop(
    outgoing: Receiver<Outbound>,
    events: Sender<Vec<TransportEvent>>,
    known: Arc<Mutex<BTreeSet<PeerId>>>,
    shutdown: Arc<AtomicBool>,
) {
    let mut streams: BTreeMap<PeerId, TcpStream> = BTreeMap::new();
    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }
        match outgoing.recv_timeout(TICK) {
            Ok(Outbound::Register { peer, stream }) => {
                let _ = stream.set_write_timeout(Some(WRITE_TIMEOUT));
                let _ = stream.set_nodelay(true);
                streams.insert(peer, stream);
                if let Ok(mut set) = known.lock() {
                    set.insert(peer);
                }
                let _ = send_event(&events, TransportEvent::Connected { peer });
            }
            Ok(Outbound::Send {
                peer,
                channel,
                payload,
            }) => {
                let frame = match encode_frame(&[FramedMessage { channel, payload }]) {
                    Ok(frame) => frame,
                    Err(error) => {
                        let _ = send_event(&events, TransportEvent::Error {
                            peer: Some(peer),
                            error: NetError::Codec(error),
                        });
                        continue;
                    }
                };
                match streams.get_mut(&peer) {
                    None => {
                        let _ = send_event(&events, TransportEvent::Error {
                            peer: Some(peer),
                            error: NetError::UnknownPeer(peer),
                        });
                    }
                    Some(stream) => {
                        if let Err(error) = stream.write_all(&frame).and_then(|()| stream.flush()) {
                            streams.remove(&peer);
                            if let Ok(mut set) = known.lock() {
                                set.remove(&peer);
                            }
                            let _ = send_event(&events, TransportEvent::Error {
                                peer: Some(peer),
                                error: NetError::Io(error),
                            });
                            let _ = send_event(&events, TransportEvent::Disconnected {
                                peer,
                                reason: DisconnectReason::Io("write failed".to_string()),
                            });
                        }
                    }
                }
            }
            Ok(Outbound::Close) | Err(RecvTimeoutError::Disconnected) => break,
            Err(RecvTimeoutError::Timeout) => {}
        }
    }
    if let Ok(mut set) = known.lock() {
        set.clear();
    }
    for peer in streams.keys() {
        let _ = send_event(&events, TransportEvent::Disconnected {
            peer: *peer,
            reason: DisconnectReason::LocalClosed,
        });
    }
}

fn accept_loop(
    listener: TcpListener,
    outgoing: Sender<Outbound>,
    events: Sender<Vec<TransportEvent>>,
    next_peer: Arc<AtomicU32>,
    shutdown: Arc<AtomicBool>,
) {
    loop {
        if shutdown.load(Ordering::Relaxed) {
            return;
        }
        match listener.accept() {
            Ok((stream, _addr)) => {
                let _ = stream.set_nodelay(true);
                let peer = next_peer.fetch_add(1, Ordering::Relaxed);
                let writer_stream = match stream.try_clone() {
                    Ok(writer_stream) => writer_stream,
                    Err(error) => {
                        let _ = send_event(&events, TransportEvent::Error {
                            peer: None,
                            error: NetError::Io(error),
                        });
                        continue;
                    }
                };
                if outgoing
                    .send(Outbound::Register {
                        peer,
                        stream: writer_stream,
                    })
                    .is_err()
                {
                    return;
                }
                if spawn_reader(stream, peer, events.clone(), shutdown.clone()).is_err() {
                    return;
                }
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(TICK);
            }
            Err(error) => {
                let _ = send_event(&events, TransportEvent::Error {
                    peer: None,
                    error: NetError::Io(error),
                });
                return;
            }
        }
    }
}
