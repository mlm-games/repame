//! Game networking for repame: transports, channels, replication and RPC.
//!
//! The crate is layered so a game can use as much or as little as it needs:
//!
//! ```text
//! replicate  rpc          game-level state and calls
//!    \       /
//!    channel table         named channels + reliability per channel
//!         |
//!    Transport (TCP | UDP + ReliableEndpoint)
//!         |
//!      protocol            versioned packet/frame bytes (codec)
//! ```
//!
//! # Reliability
//!
//! Reliability is a property of the channel, not of a call site, exactly as in
//! Godot's `MultiplayerPeer`. [`Reliability::ReliableOrdered`] is exactly-once
//! and in order, [`Reliability::ReliableUnordered`] is exactly-once but
//! unblocked by gaps, and [`Reliability::Unreliable`] is best effort and may
//! be lost, duplicated and reordered. Over TCP every channel behaves as
//! reliable-ordered because the stream already guarantees it; over UDP the
//! guarantees come from
//! [`ReliableEndpoint`](transport::udp::ReliableEndpoint), which is real
//! retransmission with acknowledgements, not a simulated one.
//!
//! # Wire format
//!
//! [`WIRE_VERSION`] leads every packet and frame, and every field layout is
//! documented in [`protocol`] and [`codec`]: little-endian fixed-width
//! scalars, `u32`-length-prefixed blobs, and `BTreeMap` iteration order, so
//! the same message always produces the same bytes on every target. Decoding
//! rejects a wrong version, trailing bytes, oversized lengths and unknown tags
//! with a [`CodecError`]; nothing is skipped silently.
//!
//! # Determinism
//!
//! The codec is deterministic: no `usize`, no hashing, no map iteration order
//! leaks into the bytes. Reliable delivery is exactly-once per channel and
//! direction. That is what makes a replicated world reproducible across a
//! reconnect, but note the limits: timestamps never enter the wire, sequence
//! numbers wrap at `2^32` per channel and direction, and a
//! [`RetryLimit`](NetError::RetryLimit) failure is surfaced rather than
//! hidden, so a game must decide what to do with a peer that stopped
//! acknowledging.

pub mod channel;
pub mod codec;
pub mod error;
pub mod protocol;
pub mod replicate;
pub mod rpc;
pub mod transport;

pub use bevy_ecs;
pub use channel::{ChannelId, ChannelSpec, ChannelTable, Reliability, names as channel_names};
pub use codec::{
    Decode, Encode, MAX_COLLECTION_LEN, Reader, Writer, decode_from_slice, encode_to_vec,
};
pub use error::{
    ChannelError, CodecError, NetError, RPC_HANDLER_FAILED, RPC_NO_HANDLER, ReplicationError,
    RpcCallError, RpcError,
};
pub use protocol::{
    ACK_WINDOW, FramedMessage, MAX_DATAGRAM_BYTES, MAX_FRAME_BYTES, Record, RecordTag,
    decode_frame, encode_frame, frame_length,
};
pub use replicate::{
    CompOp, ComponentState, ReplicaEvent, ReplicaId, ReplicaRegistry, ReplicaTypeId, Replicated,
    ReplicationClient, ReplicationMessage, ReplicationServer, SnapshotEntry,
};
pub use rpc::{
    RpcDispatcher, RpcEffect, RpcHandler, RpcId, RpcMessage, RpcMode, RpcSpec, RpcTable,
};
pub use transport::tcp::TcpTransport;
pub use transport::udp::{DEFAULT_MAX_RETRIES, DEFAULT_RETRANSMIT_MS, Inbound, ReliableEndpoint,
    UdpTransport};
pub use transport::{DisconnectReason, PeerId, Transport, TransportEvent};

/// Wire format version, the first byte of every packet and frame.
///
/// A peer speaking another version is rejected at the first byte, before any
/// field is parsed, so a mismatch is a clean protocol error rather than a
/// misread message.
pub const WIRE_VERSION: u8 = 1;

/// Convenience alias for a transport that has been boxed for a game's own
/// session type.
pub type BoxedTransport = Box<dyn Transport>;
