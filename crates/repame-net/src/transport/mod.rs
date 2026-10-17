//! Transports: how channel messages get onto and off the wire.
//!
//! [`Transport`] is the seam every game talks to. Two real implementations
//! live here:
//!
//! * [`tcp::TcpTransport`] — `std::net`, one background reader thread per
//!   connection plus one writer thread, `u32`-length-prefixed frames.
//! * [`udp::UdpTransport`] — datagram transport whose reliability lives in
//!   [`udp::ReliableEndpoint`]: per-channel sequence numbers, cumulative
//!   acknowledgements with a 32-bit window, retransmission with exponential
//!   backoff, ordered delivery on reliable-ordered channels and an
//!   unreliable class that is genuinely best-effort.
//!
//! Neither socket implementation blocks the caller: `send` hands a message to
//! a worker thread, and `poll` drains whatever the worker has produced. A
//! transport fault is never swallowed — it arrives as
//! [`TransportEvent::Error`] or [`TransportEvent::Disconnected`].

pub mod tcp;
pub mod udp;

use std::net::SocketAddr;

use crate::channel::ChannelId;
use crate::error::NetError;

/// Identifies one remote endpoint inside a transport.
///
/// A client transport talks to exactly one peer (`1`). A server transport
/// assigns `1, 2, 3, ...` in the order peers appear: on `accept` for TCP, on
/// the first datagram from a source address for UDP.
pub type PeerId = u32;

/// Why a peer went away.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DisconnectReason {
    /// This side closed the transport.
    LocalClosed,
    /// The peer closed the connection or stopped sending.
    PeerClosed,
    /// A socket operation failed.
    Io(String),
    /// A reliable message exhausted its retransmit budget.
    RetryLimit {
        /// Channel that gave up.
        channel: ChannelId,
        /// Sequence number that was never acknowledged.
        seq: u32,
    },
    /// The peer sent data this build cannot parse.
    Protocol(String),
}

impl std::fmt::Display for DisconnectReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LocalClosed => write!(f, "closed locally"),
            Self::PeerClosed => write!(f, "peer closed"),
            Self::Io(message) => write!(f, "io: {message}"),
            Self::RetryLimit { channel, seq } => {
                write!(f, "retransmit budget exhausted on channel {channel} seq {seq}")
            }
            Self::Protocol(message) => write!(f, "protocol: {message}"),
        }
    }
}

/// Something the transport worker produced for the game.
#[derive(Debug)]
pub enum TransportEvent {
    /// A peer became addressable and can now be sent to.
    Connected {
        /// Newly available peer.
        peer: PeerId,
    },
    /// A peer went away; no further events will name it.
    Disconnected {
        /// Peer that went away.
        peer: PeerId,
        /// Why it went away.
        reason: DisconnectReason,
    },
    /// One channel message arrived, already decoded and de-duplicated.
    Message {
        /// Peer that sent it.
        peer: PeerId,
        /// Channel it arrived on.
        channel: ChannelId,
        /// Message bytes.
        payload: Vec<u8>,
    },
    /// A fault that did not yet end the peer's life (a send failed, a stray
    /// datagram was unparseable, ...).
    Error {
        /// Peer the fault belongs to, when known.
        peer: Option<PeerId>,
        /// What went wrong.
        error: NetError,
    },
}

/// A message transport with at-least-once `poll` delivery of channel events.
///
/// Implementations must be usable from a single owning thread; the sockets run
/// on their own worker threads internally.
pub trait Transport: Send {
    /// Queue one message to `peer` on `channel`.
    ///
    /// Fails with [`NetError::UnknownPeer`] if the transport does not know the
    /// peer, and with [`NetError::Channel`] if the channel is not in the
    /// table. Returns as soon as the message is queued, not sent.
    fn send(&mut self, peer: PeerId, channel: ChannelId, payload: &[u8]) -> Result<(), NetError>;

    /// Drain everything the worker has produced so far, without blocking.
    ///
    /// Returns [`NetError::Closed`] once the worker thread is gone; every
    /// per-peer fault meanwhile comes back as an event, never as an `Err`.
    fn poll(&mut self) -> Result<Vec<TransportEvent>, NetError>;

    /// Ask the worker to stop and release the socket.
    fn close(&mut self);

    /// Local socket address, when the transport owns a socket.
    fn local_addr(&self) -> Option<SocketAddr>;
}
