//! A reliable-ordered channel over UDP, plus the [`UdpTransport`] that drives
//! it.
//!
//! [`ReliableEndpoint`] is the whole protocol and is transport-agnostic: it is
//! fed datagrams and produces datagrams, so the algorithm can be tested under
//! simulated loss and reordering without a socket (see the tests). The
//! [`UdpTransport`] is the socket side: it owns one endpoint per peer and runs
//! the retransmit clock on a worker thread.
//!
//! # Guarantees
//!
//! * `ReliableOrdered` — exactly-once and in sequence order. The receiver
//!   buffers a gap and releases the run once the gap is filled, so a message
//!   never overtakes an earlier one. Retransmission is what fills the gap.
//! * `ReliableUnordered` — exactly-once, delivered as each message arrives;
//!   a retransmit of an already delivered sequence is suppressed.
//! * `Unreliable` — at most once per transmission, and genuinely allowed to be
//!   lost, duplicated or reordered. No sequence number is assigned, so a
//!   duplicate is *not* suppressed: that is the documented contract.
//!
//! Duplicate suppression uses a per-channel, per-direction 32-entry receive
//! window; sequences older than the window are treated as duplicates.
//! Sequence numbers wrap at `2^32` messages per channel and direction (about
//! 2.3 years at 60 messages/s); across a wrap, ordering breaks.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::net::{SocketAddr, UdpSocket};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::channel::{ChannelId, ChannelTable, Reliability};
use crate::error::{ChannelError, NetError};
use crate::protocol::{
    ACK_WINDOW, MAX_DATAGRAM_BYTES, MAX_RECORD_OVERHEAD, Record, decode_packet, encode_packet,
};
use crate::transport::{DisconnectReason, PeerId, Transport, TransportEvent};

/// Default retransmit timeout in milliseconds.
pub const DEFAULT_RETRANSMIT_MS: u32 = 120;

/// Default retransmission budget per reliable message.
pub const DEFAULT_MAX_RETRIES: u32 = 12;

/// Largest exponent applied to the base timeout when backing off.
const MAX_BACKOFF_SHIFT: u32 = 5;

/// How often the socket worker wakes to check for shutdown and retransmits.
const TICK: Duration = Duration::from_millis(2);

#[derive(Debug, Clone)]
struct Pending {
    payload: Vec<u8>,
    last_sent_ms: Option<u64>,
    attempts: u32,
}

/// Receive-side bookkeeping for one channel: the next in-order sequence, the
/// out-of-order buffer and the acknowledgement window.
#[derive(Debug, Default, Clone)]
struct RecvState {
    expected: u32,
    buffered: BTreeMap<u32, Vec<u8>>,
    highest: u32,
    bits: u32,
    have_any: bool,
}

impl RecvState {
    fn is_seen(&self, seq: u32) -> bool {
        if !self.have_any {
            return false;
        }
        if seq == self.highest {
            return true;
        }
        if seq < self.highest {
            let delta = self.highest - seq;
            return delta <= ACK_WINDOW && self.bits & (1 << (delta - 1)) != 0;
        }
        false
    }

    fn mark_seen(&mut self, seq: u32) {
        if !self.have_any {
            self.have_any = true;
            self.highest = seq;
            self.bits = 0;
            return;
        }
        if seq > self.highest {
            let shift = seq - self.highest;
            self.bits = if shift < ACK_WINDOW {
                (self.bits << shift) | (1 << (shift - 1))
            } else {
                0
            };
            self.highest = seq;
        } else if seq < self.highest {
            let delta = self.highest - seq;
            if delta <= ACK_WINDOW {
                self.bits |= 1 << (delta - 1);
            }
        }
    }
}

/// Whether `seq` is covered by a cumulative acknowledgement plus its window.
fn is_acked(seq: u32, ack_seq: u32, ack_bits: u32) -> bool {
    if seq <= ack_seq {
        return true;
    }
    let delta = seq - ack_seq;
    delta <= ACK_WINDOW && ack_bits & (1 << (delta - 1)) != 0
}

/// One message handed up by [`ReliableEndpoint::receive`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inbound {
    /// Channel the message arrived on.
    pub channel: ChannelId,
    /// Message bytes.
    pub payload: Vec<u8>,
}

/// The reliability algorithm for one connection, independent of any socket.
///
/// Feed received datagrams to [`receive`](Self::receive) and call
/// [`flush`](Self::flush) on a timer; it returns the datagrams to put on the
/// wire. Dropping or reordering those datagrams is exactly packet loss, so the
/// algorithm is testable in memory.
#[derive(Debug)]
pub struct ReliableEndpoint {
    channels: ChannelTable,
    next_seq: Vec<u32>,
    in_flight: Vec<BTreeMap<u32, Pending>>,
    recv: Vec<RecvState>,
    outgoing: Vec<Record>,
    pending_acks: BTreeSet<ChannelId>,
    retransmit_ms: u32,
    max_retries: u32,
}

impl ReliableEndpoint {
    /// Endpoint with the default timeout and retry budget.
    pub fn new(channels: ChannelTable) -> Self {
        Self::with_config(channels, DEFAULT_RETRANSMIT_MS, DEFAULT_MAX_RETRIES)
    }

    /// Endpoint with an explicit base retransmit timeout and retry budget.
    pub fn with_config(channels: ChannelTable, retransmit_ms: u32, max_retries: u32) -> Self {
        let count = channels.len();
        Self {
            channels,
            next_seq: vec![0; count],
            in_flight: vec![BTreeMap::new(); count],
            recv: vec![RecvState::default(); count],
            outgoing: Vec::new(),
            pending_acks: BTreeSet::new(),
            retransmit_ms,
            max_retries: max_retries.max(1),
        }
    }

    /// The channel table this endpoint was built with.
    pub fn channels(&self) -> &ChannelTable {
        &self.channels
    }

    /// Largest payload one message may carry without exceeding the datagram
    /// budget.
    pub fn max_payload(&self) -> usize {
        MAX_DATAGRAM_BYTES - MAX_RECORD_OVERHEAD - 2
    }

    /// Queue one message for the next [`flush`](Self::flush).
    ///
    /// Reliable channels assign a sequence number and keep the message until
    /// it is acknowledged. Fails on an unknown channel or a payload larger
    /// than [`max_payload`](Self::max_payload).
    pub fn queue(&mut self, channel: ChannelId, payload: Vec<u8>) -> Result<(), NetError> {
        let reliability = self.channels.reliability(channel)?;
        let max = self.max_payload();
        if payload.len() > max {
            return Err(NetError::PayloadTooLarge {
                size: payload.len(),
                max,
            });
        }
        let index = channel as usize;
        match reliability {
            Reliability::Unreliable => self.outgoing.push(Record::Unreliable { channel, payload }),
            Reliability::ReliableOrdered | Reliability::ReliableUnordered => {
                let seq = self.next_seq[index];
                self.next_seq[index] = seq.wrapping_add(1);
                self.in_flight[index].insert(
                    seq,
                    Pending {
                        payload: payload.clone(),
                        last_sent_ms: None,
                        attempts: 0,
                    },
                );
                self.outgoing.push(Record::Reliable {
                    channel,
                    seq,
                    payload,
                });
            }
        }
        Ok(())
    }

    /// Consume one datagram and return every message it delivers.
    ///
    /// A record whose reliability class contradicts its channel is a
    /// [`NetError::Protocol`] failure. Acks and window state are updated
    /// before delivery, so a duplicate is suppressed for reliable channels.
    pub fn receive(&mut self, datagram: &[u8]) -> Result<Vec<Inbound>, NetError> {
        let records = decode_packet(datagram)?;
        let mut delivered = Vec::new();
        for record in records {
            match record {
                Record::Ack {
                    channel,
                    ack_seq,
                    ack_bits,
                } => self.apply_ack(channel, ack_seq, ack_bits)?,
                Record::Reliable {
                    channel,
                    seq,
                    payload,
                } => self.receive_reliable(channel, seq, payload, &mut delivered)?,
                Record::Unreliable { channel, payload } => {
                    if self.channels.reliability(channel)? != Reliability::Unreliable {
                        return Err(NetError::Protocol(format!(
                            "unreliable record on reliable channel {channel}"
                        )));
                    }
                    delivered.push(Inbound { channel, payload });
                }
            }
        }
        Ok(delivered)
    }

    /// Datagrams to send now: queued messages, acknowledgements and any
    /// retransmission whose backoff has elapsed.
    ///
    /// Returns [`NetError::RetryLimit`] when a reliable message has been
    /// retransmitted `max_retries` times without an acknowledgement; the
    /// caller is expected to drop the connection rather than keep retrying.
    pub fn flush(&mut self, now_ms: u64) -> Result<Vec<Vec<u8>>, NetError> {
        let mut records: Vec<Record> = Vec::new();
        for channel in std::mem::take(&mut self.pending_acks) {
            let state = &self.recv[channel as usize];
            records.push(Record::Ack {
                channel,
                ack_seq: state.highest,
                ack_bits: state.bits,
            });
        }
        records.append(&mut self.outgoing);
        let retransmit_ms = self.retransmit_ms;
        let max_retries = self.max_retries;
        for index in 0..self.in_flight.len() {
            let channel = index as ChannelId;
            let mut retransmit = Vec::new();
            for (seq, pending) in self.in_flight[index].iter_mut() {
                match pending.last_sent_ms {
                    None => {
                        pending.last_sent_ms = Some(now_ms);
                        pending.attempts = 1;
                    }
                    Some(sent) => {
                        if now_ms.saturating_sub(sent) >= Self::backoff_ms(retransmit_ms, pending.attempts) {
                            if pending.attempts >= max_retries {
                                return Err(NetError::RetryLimit {
                                    channel,
                                    seq: *seq,
                                });
                            }
                            pending.attempts += 1;
                            pending.last_sent_ms = Some(now_ms);
                            retransmit.push(*seq);
                        }
                    }
                }
            }
            for seq in retransmit {
                let payload = self.in_flight[index][&seq].payload.clone();
                records.push(Record::Reliable {
                    channel,
                    seq,
                    payload,
                });
            }
        }
        Ok(self.chunk(records)?)
    }

    /// Reliable messages that have not been acknowledged yet.
    pub fn unacked(&self) -> usize {
        self.in_flight.iter().map(BTreeMap::len).sum()
    }

    fn apply_ack(
        &mut self,
        channel: ChannelId,
        ack_seq: u32,
        ack_bits: u32,
    ) -> Result<(), NetError> {
        let index = self.index(channel)?;
        self.in_flight[index].retain(|seq, _| !is_acked(*seq, ack_seq, ack_bits));
        Ok(())
    }

    fn receive_reliable(
        &mut self,
        channel: ChannelId,
        seq: u32,
        payload: Vec<u8>,
        delivered: &mut Vec<Inbound>,
    ) -> Result<(), NetError> {
        let reliability = self.channels.reliability(channel)?;
        if !reliability.is_reliable() {
            return Err(NetError::Protocol(format!(
                "reliable record on unreliable channel {channel}"
            )));
        }
        let index = channel as usize;
        self.pending_acks.insert(channel);
        let seen = self.recv[index].is_seen(seq);
        self.recv[index].mark_seen(seq);
        if reliability.preserves_order() {
            if seq >= self.recv[index].expected {
                self.recv[index].buffered.entry(seq).or_insert(payload);
            }
            let mut expected = self.recv[index].expected;
            while let Some(payload) = self.recv[index].buffered.remove(&expected) {
                delivered.push(Inbound { channel, payload });
                expected = expected.wrapping_add(1);
            }
            self.recv[index].expected = expected;
        } else if !seen {
            delivered.push(Inbound { channel, payload });
        }
        Ok(())
    }

    fn index(&self, channel: ChannelId) -> Result<usize, NetError> {
        let index = channel as usize;
        if index < self.channels.len() {
            Ok(index)
        } else {
            Err(ChannelError::UnknownId(channel).into())
        }
    }

/// Exponential backoff on the base retransmit timeout, capped at 32x.
fn backoff_ms(base_ms: u32, attempts: u32) -> u64 {
    let shift = attempts.saturating_sub(1).min(MAX_BACKOFF_SHIFT);
    u64::from(base_ms.saturating_mul(1u32 << shift))
}

    fn chunk(&self, records: Vec<Record>) -> Result<Vec<Vec<u8>>, NetError> {
        let mut datagrams = Vec::new();
        let mut batch: Vec<Record> = Vec::new();
        let mut size = 2usize;
        for record in records {
            let record_len = record.encoded_len();
            let full = batch.len() >= u8::MAX as usize
                || (!batch.is_empty() && size + record_len > MAX_DATAGRAM_BYTES);
            if full {
                datagrams.push(encode_packet(&batch)?);
                batch.clear();
                size = 2;
            }
            size += record_len;
            batch.push(record);
        }
        if !batch.is_empty() {
            datagrams.push(encode_packet(&batch)?);
        }
        Ok(datagrams)
    }
}

/// How a [`UdpTransport`] is addressed.
#[derive(Debug, Clone, Copy)]
enum UdpMode {
    /// One fixed remote address; datagrams from anywhere else are ignored.
    Client { server: SocketAddr },
    /// Any source address that sends a valid datagram becomes a peer.
    Server,
}

struct UdpPeerState {
    addr: SocketAddr,
    endpoint: ReliableEndpoint,
}

enum Outbound {
    Send {
        peer: PeerId,
        channel: ChannelId,
        payload: Vec<u8>,
    },
    Close,
}

/// UDP transport: one [`ReliableEndpoint`] per peer on a worker thread.
///
/// A client is created with [`UdpTransport::connect`] and talks to a single
/// peer (`1`). A server is created with [`UdpTransport::bind`] and assigns
/// peers in the order source addresses first send a valid datagram.
///
/// A datagram that cannot be parsed ends that peer: the transport emits an
/// [`TransportEvent::Error`] and then
/// [`TransportEvent::Disconnected`], and a fresh peer is only created if the
/// address sends again. Because sequence numbers are per peer and do not
/// reset with it, a session does not resume after that; both sides time out
/// and reconnect. Same for a client datagram that arrives from an address
/// other than the server's: it is ignored, since trusting it would let anyone
/// inject state.
pub struct UdpTransport {
    incoming: Receiver<Vec<TransportEvent>>,
    outgoing: Sender<Outbound>,
    local_addr: SocketAddr,
    channels: Arc<ChannelTable>,
    peers: Arc<Mutex<BTreeSet<PeerId>>>,
    shutdown: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    closed: bool,
}

impl UdpTransport {
    /// Bind `local` and talk to `server`. `local` may use port `0`.
    pub fn connect(
        local: SocketAddr,
        server: SocketAddr,
        channels: ChannelTable,
    ) -> Result<Self, NetError> {
        let socket = UdpSocket::bind(local)?;
        Self::spawn(socket, UdpMode::Client { server }, channels)
    }

    /// Bind and accept datagrams from any source address.
    pub fn bind(bind: SocketAddr, channels: ChannelTable) -> Result<Self, NetError> {
        let socket = UdpSocket::bind(bind)?;
        Self::spawn(socket, UdpMode::Server, channels)
    }

    fn spawn(socket: UdpSocket, mode: UdpMode, channels: ChannelTable) -> Result<Self, NetError> {
        let local_addr = socket.local_addr()?;
        socket.set_read_timeout(Some(TICK))?;
        let (events_tx, events_rx) = std::sync::mpsc::channel();
        let (out_tx, out_rx) = std::sync::mpsc::channel();
        let shutdown = Arc::new(AtomicBool::new(false));
        let peers = Arc::new(Mutex::new(BTreeSet::new()));
        let table = Arc::new(channels.clone());
        if matches!(mode, UdpMode::Client { .. }) {
            // The client's single peer is known before the worker starts, so a
            // send issued right after `connect` cannot race the worker.
            if let Ok(mut set) = peers.lock() {
                set.insert(1);
            }
        }
        let worker = {
            let shutdown = shutdown.clone();
            let peers = peers.clone();
            thread::Builder::new()
                .name("repame-net-udp".to_string())
                .spawn(move || worker_loop(socket, mode, channels, peers, events_tx, out_rx, shutdown))
                .map_err(NetError::Io)?
        };
        Ok(Self {
            incoming: events_rx,
            outgoing: out_tx,
            local_addr,
            channels: table,
            peers,
            shutdown,
            worker: Some(worker),
            closed: false,
        })
    }

    fn is_connected(peers: &Mutex<BTreeSet<PeerId>>, peer: PeerId) -> bool {
        peers.lock().map(|set| set.contains(&peer)).unwrap_or(false)
    }
}

impl Transport for UdpTransport {
    fn send(&mut self, peer: PeerId, channel: ChannelId, payload: &[u8]) -> Result<(), NetError> {
        self.channels.reliability(channel)?;
        if !Self::is_connected(&self.peers, peer) {
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
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }

    fn local_addr(&self) -> Option<SocketAddr> {
        Some(self.local_addr)
    }
}

impl Drop for UdpTransport {
    fn drop(&mut self) {
        self.close();
    }
}

fn worker_loop(
    socket: UdpSocket,
    mode: UdpMode,
    channels: ChannelTable,
    known: Arc<Mutex<BTreeSet<PeerId>>>,
    events: Sender<Vec<TransportEvent>>,
    outgoing: Receiver<Outbound>,
    shutdown: Arc<AtomicBool>,
) {
    let started = Instant::now();
    let server_filter = match mode {
        UdpMode::Client { server } => Some(server),
        UdpMode::Server => None,
    };
    let mut peers: BTreeMap<PeerId, UdpPeerState> = BTreeMap::new();
    let mut addr_to_peer: HashMap<SocketAddr, PeerId> = HashMap::new();
    let mut next_peer: PeerId = 1;
    let mut queue: Vec<TransportEvent> = Vec::new();

    if let Some(server) = server_filter {
        peers.insert(
            1,
            UdpPeerState {
                addr: server,
                endpoint: ReliableEndpoint::new(channels.clone()),
            },
        );
        addr_to_peer.insert(server, 1);
        next_peer = 2;
        if let Ok(mut set) = known.lock() {
            set.insert(1);
        }
        queue.push(TransportEvent::Connected { peer: 1 });
    }

    let mut buffer = vec![0u8; 65_536];
    'run: loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }
        loop {
            match outgoing.try_recv() {
                Ok(Outbound::Send {
                    peer,
                    channel,
                    payload,
                }) => match peers.get_mut(&peer) {
                    Some(state) => {
                        if let Err(error) = state.endpoint.queue(channel, payload) {
                            queue.push(TransportEvent::Error {
                                peer: Some(peer),
                                error,
                            });
                        }
                    }
                    None => queue.push(TransportEvent::Error {
                        peer: Some(peer),
                        error: NetError::UnknownPeer(peer),
                    }),
                },
                Ok(Outbound::Close) | Err(TryRecvError::Disconnected) => {
                    shutdown.store(true, Ordering::Relaxed);
                    break;
                }
                Err(TryRecvError::Empty) => break,
            }
        }
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        loop {
            let (len, from) = match socket.recv_from(&mut buffer) {
                Ok(received) => received,
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                    ) =>
                {
                    break;
                }
                Err(error) => {
                    queue.push(TransportEvent::Error {
                        peer: None,
                        error: NetError::Io(error),
                    });
                    break;
                }
            };
            if let Some(server) = server_filter {
                if from != server {
                    continue;
                }
            }
            let peer = match addr_to_peer.get(&from) {
                Some(peer) => *peer,
                None => {
                    let peer = next_peer;
                    next_peer = next_peer.wrapping_add(1).max(1);
                    peers.insert(
                        peer,
                        UdpPeerState {
                            addr: from,
                            endpoint: ReliableEndpoint::new(channels.clone()),
                        },
                    );
                    addr_to_peer.insert(from, peer);
                    if let Ok(mut set) = known.lock() {
                        set.insert(peer);
                    }
                    queue.push(TransportEvent::Connected { peer });
                    peer
                }
            };
            let received = peers
                .get_mut(&peer)
                .map(|state| state.endpoint.receive(&buffer[..len]));
            match received {
                Some(Ok(inbound)) => {
                    for message in inbound {
                        queue.push(TransportEvent::Message {
                            peer,
                            channel: message.channel,
                            payload: message.payload,
                        });
                    }
                }
                Some(Err(error)) => {
                    let reason = DisconnectReason::Protocol(error.to_string());
                    queue.push(TransportEvent::Error {
                        peer: Some(peer),
                        error,
                    });
                    peers.remove(&peer);
                    addr_to_peer.remove(&from);
                    if let Ok(mut set) = known.lock() {
                        set.remove(&peer);
                    }
                    queue.push(TransportEvent::Disconnected { peer, reason });
                }
                None => {}
            }
        }

        let now_ms = started.elapsed().as_millis() as u64;
        let mut dead: Vec<(PeerId, SocketAddr, DisconnectReason)> = Vec::new();
        let ids: Vec<PeerId> = peers.keys().copied().collect();
        for peer in ids {
            let Some(state) = peers.get_mut(&peer) else {
                continue;
            };
            let addr = state.addr;
            let datagrams = match state.endpoint.flush(now_ms) {
                Ok(datagrams) => datagrams,
                Err(error) => {
                    let reason = match error {
                        NetError::RetryLimit { channel, seq } => {
                            DisconnectReason::RetryLimit { channel, seq }
                        }
                        other => DisconnectReason::Protocol(other.to_string()),
                    };
                    dead.push((peer, addr, reason));
                    continue;
                }
            };
            for datagram in datagrams {
                if let Err(error) = socket.send_to(&datagram, addr) {
                    queue.push(TransportEvent::Error {
                        peer: Some(peer),
                        error: NetError::Io(error),
                    });
                }
            }
        }
        for (peer, addr, reason) in dead {
            peers.remove(&peer);
            addr_to_peer.remove(&addr);
            if let Ok(mut set) = known.lock() {
                set.remove(&peer);
            }
            queue.push(TransportEvent::Disconnected { peer, reason });
        }

        if !queue.is_empty() && events.send(std::mem::take(&mut queue)).is_err() {
            break 'run;
        }
    }

    if let Ok(mut set) = known.lock() {
        set.clear();
    }
    for peer in peers.keys() {
        let _ = events.send(vec![TransportEvent::Disconnected {
            peer: *peer,
            reason: DisconnectReason::LocalClosed,
        }]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::ChannelSpec;

    /// xorshift64*: deterministic on every target, so the loss/reorder pattern
    /// below is part of the test's identity.
    fn next_rand(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        *state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn table() -> ChannelTable {
        ChannelTable::new(&[
            ChannelSpec {
                name: "reliable",
                reliability: Reliability::ReliableOrdered,
            },
            ChannelSpec {
                name: "unreliable",
                reliability: Reliability::Unreliable,
            },
        ])
        .unwrap()
    }

    fn decode_u32(payload: &[u8]) -> u32 {
        u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]])
    }

    #[test]
    fn reliable_ordered_delivery_survives_loss_duplication_and_reordering() {
        const COUNT: u32 = 48;
        let mut sender = ReliableEndpoint::with_config(table(), 10, 64);
        let mut receiver = ReliableEndpoint::new(table());
        for i in 0..COUNT {
            sender.queue(0, i.to_le_bytes().to_vec()).unwrap();
        }

        let mut wire: Vec<Vec<u8>> = Vec::new();
        let mut received: Vec<u32> = Vec::new();
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut now = 0u64;
        let mut steps = 0u32;

        while received.len() < COUNT as usize && steps < 50_000 {
            steps += 1;
            now += 5;
            for datagram in sender.flush(now).unwrap() {
                let roll = next_rand(&mut rng);
                if roll % 3 == 0 {
                    continue;
                }
                if roll % 7 == 0 {
                    wire.push(datagram.clone());
                }
                wire.push(datagram);
            }
            if wire.len() >= 2 && next_rand(&mut rng) % 2 == 0 {
                wire.swap(0, 1);
            }
            let batch: Vec<Vec<u8>> = wire.drain(..).collect();
            for datagram in batch {
                for inbound in receiver.receive(&datagram).unwrap() {
                    assert_eq!(inbound.channel, 0);
                    received.push(decode_u32(&inbound.payload));
                }
            }
            for datagram in receiver.flush(now).unwrap() {
                sender.receive(&datagram).unwrap();
            }
        }

        assert!(steps < 50_000, "the reliable channel never converged");
        assert_eq!(
            received,
            (0..COUNT).collect::<Vec<u32>>(),
            "reliable-ordered must deliver exactly the sent sequence"
        );
        assert_eq!(sender.unacked(), 0, "every message was acknowledged");
    }
}
