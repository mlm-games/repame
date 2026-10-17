//! Typed errors for the codec, channels, transports, replication and RPC.
//!
//! Every layer returns its own error enum and the top-level [`NetError`]
//! wraps them, so a caller can match on the layer that failed instead of
//! parsing a message. Nothing in this crate drops a failure silently: an
//! unrecoverable transport fault is reported as `TransportEvent::Error`
//! or `TransportEvent::Disconnected`, never as a missing message.

use std::fmt;
use std::io;

use crate::channel::ChannelId;
use crate::transport::PeerId;

/// Malformed or truncated wire data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodecError {
    /// Ran out of bytes in the middle of a field.
    UnexpectedEof {
        /// Bytes the field needed.
        needed: usize,
        /// Bytes actually left in the reader.
        remaining: usize,
    },
    /// Bytes remained after a complete decode.
    TrailingBytes(usize),
    /// The packet's leading version byte is not `WIRE_VERSION`.
    BadVersion {
        /// Version found on the wire.
        found: u8,
        /// Version this build speaks.
        expected: u8,
    },
    /// A length or count field exceeded its cap.
    Oversize {
        /// Name of the offending field, for diagnostics.
        field: &'static str,
        /// Value found on the wire.
        size: usize,
        /// Largest accepted value.
        max: usize,
    },
    /// A string field was not valid UTF-8.
    InvalidUtf8,
    /// A tag byte is not a known record, message or replication op.
    UnknownTag(u8),
    /// A bool field held something other than `0` or `1`.
    InvalidBool(u8),
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof { needed, remaining } => {
                write!(f, "truncated: needed {needed} bytes, {remaining} left")
            }
            Self::TrailingBytes(n) => write!(f, "{n} trailing bytes after a complete decode"),
            Self::BadVersion { found, expected } => {
                write!(f, "wire version {found}, this build speaks {expected}")
            }
            Self::Oversize { field, size, max } => {
                write!(f, "{field} size {size} exceeds the cap of {max}")
            }
            Self::InvalidUtf8 => write!(f, "string field is not valid UTF-8"),
            Self::UnknownTag(tag) => write!(f, "unknown tag {tag}"),
            Self::InvalidBool(v) => write!(f, "bool byte {v} is neither 0 nor 1"),
        }
    }
}

impl std::error::Error for CodecError {}

/// Channel table construction and lookup failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChannelError {
    /// More channels than a `u8` channel id can address.
    TooManyChannels(usize),
    /// The same channel name was registered twice.
    DuplicateName(String),
    /// No channel is registered under that name.
    UnknownName(String),
    /// No channel is registered with that id.
    UnknownId(ChannelId),
}

impl fmt::Display for ChannelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyChannels(n) => write!(f, "{n} channels exceeds the 256 channel limit"),
            Self::DuplicateName(name) => write!(f, "channel {name:?} registered twice"),
            Self::UnknownName(name) => write!(f, "no channel named {name:?}"),
            Self::UnknownId(id) => write!(f, "no channel with id {id}"),
        }
    }
}

impl std::error::Error for ChannelError {}

/// Replication protocol failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplicationError {
    /// The component type is already registered under another id.
    DuplicateComponentType(&'static str),
    /// More replicated component types than a `u16` type id can address.
    TooManyComponentTypes(usize),
    /// The wire carried a component type id that is not registered.
    UnknownComponentType(u16),
    /// The entity is already tracked by this server.
    AlreadyTracked,
    /// The entity is not tracked by this server.
    NotTracked,
    /// The replica id was never announced by a spawn.
    UnknownReplica(u32),
    /// A spawn arrived for a replica id that is already live on the client.
    DuplicateReplica(u32),
}

impl fmt::Display for ReplicationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateComponentType(name) => write!(f, "component {name:?} already registered"),
            Self::TooManyComponentTypes(n) => {
                write!(f, "{n} component types exceeds the 65536 type limit")
            }
            Self::UnknownComponentType(id) => write!(f, "component type id {id} is not registered"),
            Self::AlreadyTracked => write!(f, "entity is already tracked for replication"),
            Self::NotTracked => write!(f, "entity is not tracked for replication"),
            Self::UnknownReplica(id) => write!(f, "no live replica {id}"),
            Self::DuplicateReplica(id) => write!(f, "replica {id} was spawned twice"),
        }
    }
}

impl std::error::Error for ReplicationError {}

/// A remote procedure failure, carried to the caller over the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RpcError {
    /// Stable machine-readable code.
    pub code: u16,
    /// Human-readable explanation.
    pub message: String,
}

/// The callee has no handler registered for the invoked procedure.
pub const RPC_NO_HANDLER: u16 = 1;
/// The procedure was invoked but the handler returned an error.
pub const RPC_HANDLER_FAILED: u16 = 2;

impl RpcError {
    /// Build an error from a stable code and an explanation.
    pub fn new(code: u16, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl fmt::Display for RpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "rpc error {}: {}", self.code, self.message)
    }
}

impl std::error::Error for RpcError {}

/// RPC layer failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RpcCallError {
    /// The procedure name was registered twice.
    DuplicateName(String),
    /// No procedure is registered under that name.
    UnknownName(String),
    /// No procedure is registered with that id.
    UnknownId(u16),
    /// A response arrived for a call id this side never issued.
    UnknownCall(u32),
    /// The call id space is exhausted.
    TooManyCalls,
}

impl fmt::Display for RpcCallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateName(name) => write!(f, "procedure {name:?} registered twice"),
            Self::UnknownName(name) => write!(f, "no procedure named {name:?}"),
            Self::UnknownId(id) => write!(f, "no procedure with id {id}"),
            Self::UnknownCall(id) => write!(f, "no pending call with id {id}"),
            Self::TooManyCalls => write!(f, "pending call id space exhausted"),
        }
    }
}

impl std::error::Error for RpcCallError {}

/// Top-level error for every repame-net operation.
#[derive(Debug)]
pub enum NetError {
    /// A socket operation failed.
    Io(io::Error),
    /// Wire data was malformed.
    Codec(CodecError),
    /// Channel table construction or lookup failed.
    Channel(ChannelError),
    /// Replication state or message failed.
    Replication(ReplicationError),
    /// A remote procedure failed.
    Rpc(RpcError),
    /// RPC bookkeeping failed locally.
    RpcCall(RpcCallError),
    /// A send targeted a peer this transport does not know.
    UnknownPeer(PeerId),
    /// The transport has not finished connecting yet.
    NotConnected,
    /// The transport's worker thread is gone; no further events will arrive.
    Closed,
    /// A message could not be encoded into one datagram.
    PayloadTooLarge {
        /// Payload size in bytes.
        size: usize,
        /// Largest payload one datagram can carry.
        max: usize,
    },
    /// A reliable message exhausted its retransmit budget.
    RetryLimit {
        /// Channel that gave up.
        channel: ChannelId,
        /// Sequence number that was never acknowledged.
        seq: u32,
    },
    /// The peer sent something the protocol does not allow.
    Protocol(String),
}

impl fmt::Display for NetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io: {e}"),
            Self::Codec(e) => write!(f, "codec: {e}"),
            Self::Channel(e) => write!(f, "channel: {e}"),
            Self::Replication(e) => write!(f, "replication: {e}"),
            Self::Rpc(e) => write!(f, "rpc: {e}"),
            Self::RpcCall(e) => write!(f, "rpc: {e}"),
            Self::UnknownPeer(peer) => write!(f, "unknown peer {peer}"),
            Self::NotConnected => write!(f, "transport is not connected yet"),
            Self::Closed => write!(f, "transport is closed"),
            Self::PayloadTooLarge { size, max } => {
                write!(f, "payload of {size} bytes exceeds the {max} byte datagram budget")
            }
            Self::RetryLimit { channel, seq } => {
                write!(f, "channel {channel} gave up on seq {seq} after the retry budget")
            }
            Self::Protocol(m) => write!(f, "protocol violation: {m}"),
        }
    }
}

impl std::error::Error for NetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Codec(e) => Some(e),
            Self::Channel(e) => Some(e),
            Self::Replication(e) => Some(e),
            Self::Rpc(e) => Some(e),
            Self::RpcCall(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for NetError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<CodecError> for NetError {
    fn from(value: CodecError) -> Self {
        Self::Codec(value)
    }
}

impl From<ChannelError> for NetError {
    fn from(value: ChannelError) -> Self {
        Self::Channel(value)
    }
}

impl From<ReplicationError> for NetError {
    fn from(value: ReplicationError) -> Self {
        Self::Replication(value)
    }
}

impl From<RpcError> for NetError {
    fn from(value: RpcError) -> Self {
        Self::Rpc(value)
    }
}

impl From<RpcCallError> for NetError {
    fn from(value: RpcCallError) -> Self {
        Self::RpcCall(value)
    }
}
