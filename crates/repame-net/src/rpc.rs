//! Named remote procedures over channels.
//!
//! A procedure is registered by name once and looked up by a `u16` id on the
//! wire. Every procedure names the channel it rides, so the reliability
//! options are exactly the channel's: a reliable-ordered `rpc` procedure
//! cannot be lost or reordered, an unreliable channel's procedure can. That
//! mirrors Godot's `rpc_config` with `transfer_mode` + channel, minus the
//! reflection-based method-name lookup, which repame's `bevy_ecs` does not
//! have.
//!
//! # Wire layout
//!
//! `RpcMessage` is a tag byte plus fields, all through [`crate::codec`]:
//!
//! ```text
//! tag 0 CALL    : id:u16, call_id:u32, args:bytes
//! tag 1 REPLY   : call_id:u32, payload:bytes
//! tag 2 FAILURE : call_id:u32, code:u16, message:string
//! ```
//!
//! # Round trip
//!
//! 1. [`begin_call`](RpcDispatcher::begin_call) looks the name up, allocates a
//!    call id and returns the `Call` to send on the procedure's channel.
//! 2. The peer decodes it and calls
//!    [`dispatch`](RpcDispatcher::dispatch), which runs the registered handler
//!    and returns the `Reply`/`Failure` to send back on the same channel.
//! 3. The caller decodes that and calls `dispatch` again, which returns
//!    [`RpcEffect::Resolved`] because the call id is one it issued.
//!
//! A handler that is missing or errors produces a `Failure` message rather
//! than dropping the call, and a response for an unknown call id is a
//! [`NetError`], never silently ignored.

use std::collections::HashMap;

use repame_sim::bevy_ecs::world::World;

use crate::channel::ChannelId;
use crate::codec::{Decode, Encode, Reader, Writer};
use crate::error::{CodecError, NetError, RpcCallError, RpcError};
use crate::transport::PeerId;

/// Registry slot of one procedure.
pub type RpcId = u16;

/// Which side may invoke a procedure.
///
/// Enforced on receipt when the dispatcher knows its own role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RpcMode {
    /// Only the server may invoke it (it runs on a client).
    Server,
    /// Only a client may invoke it (it runs on the server).
    Client,
    /// Either side may invoke it.
    Any,
}

impl RpcMode {
    /// Whether a call from `caller` is allowed.
    pub fn allows(self, caller: RpcMode) -> bool {
        matches!(self, Self::Any) || self == caller
    }
}

/// One registered procedure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RpcSpec {
    /// Registry slot.
    pub id: RpcId,
    /// Stable procedure name.
    pub name: &'static str,
    /// Channel the call and its reply travel on; the channel carries the
    /// reliability mode.
    pub channel: ChannelId,
    /// Which side may invoke it.
    pub mode: RpcMode,
}

/// Procedure names and their channels.
#[derive(Debug, Clone, Default)]
pub struct RpcTable {
    specs: Vec<RpcSpec>,
    by_name: HashMap<&'static str, RpcId>,
}

impl RpcTable {
    /// Empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a procedure and return its id.
    pub fn register(
        &mut self,
        name: &'static str,
        channel: ChannelId,
        mode: RpcMode,
    ) -> Result<RpcId, NetError> {
        if self.by_name.contains_key(name) {
            return Err(RpcCallError::DuplicateName(name.to_string()).into());
        }
        let id = u16::try_from(self.specs.len()).map_err(|_| RpcCallError::TooManyCalls)?;
        self.specs.push(RpcSpec {
            id,
            name,
            channel,
            mode,
        });
        self.by_name.insert(name, id);
        Ok(id)
    }

    /// Id of a procedure, by name.
    pub fn id(&self, name: &str) -> Result<RpcId, NetError> {
        self.by_name
            .get(name)
            .copied()
            .ok_or_else(|| RpcCallError::UnknownName(name.to_string()).into())
    }

    /// Spec of a procedure, by id.
    pub fn spec(&self, id: RpcId) -> Option<&RpcSpec> {
        self.specs.get(id as usize)
    }

    /// Channel a procedure travels on.
    pub fn channel(&self, id: RpcId) -> Result<ChannelId, NetError> {
        self.spec(id)
            .map(|spec| spec.channel)
            .ok_or_else(|| RpcCallError::UnknownId(id).into())
    }

    /// Number of registered procedures.
    pub fn len(&self) -> usize {
        self.specs.len()
    }

    /// Whether no procedure is registered.
    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }
}

/// A remote procedure message on the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RpcMessage {
    /// Invoke a procedure.
    Call {
        /// Registered procedure id.
        id: RpcId,
        /// Caller-assigned id, echoed in the reply.
        call_id: u32,
        /// Pre-encoded argument bytes.
        args: Vec<u8>,
    },
    /// The handler returned successfully.
    Reply {
        /// Call id being answered.
        call_id: u32,
        /// Return value bytes.
        payload: Vec<u8>,
    },
    /// The handler was missing or failed.
    Failure {
        /// Call id being answered.
        call_id: u32,
        /// Stable machine-readable code.
        code: u16,
        /// Human-readable explanation.
        message: String,
    },
}

impl RpcMessage {
    /// Call id this message belongs to, for routing.
    pub fn call_id(&self) -> u32 {
        match self {
            Self::Call { call_id, .. } | Self::Reply { call_id, .. } | Self::Failure { call_id, .. } => {
                *call_id
            }
        }
    }
}

impl Encode for RpcMessage {
    fn encode(&self, out: &mut Writer) -> Result<(), CodecError> {
        match self {
            Self::Call { id, call_id, args } => {
                out.put_u8(0);
                out.put_u16(*id);
                out.put_u32(*call_id);
                out.put_bytes(args)
            }
            Self::Reply { call_id, payload } => {
                out.put_u8(1);
                out.put_u32(*call_id);
                out.put_bytes(payload)
            }
            Self::Failure {
                call_id,
                code,
                message,
            } => {
                out.put_u8(2);
                out.put_u32(*call_id);
                out.put_u16(*code);
                out.put_str(message)
            }
        }
    }
}

impl Decode for RpcMessage {
    fn decode(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        match input.get_u8()? {
            0 => Ok(Self::Call {
                id: input.get_u16()?,
                call_id: input.get_u32()?,
                args: input.get_bytes()?.to_vec(),
            }),
            1 => Ok(Self::Reply {
                call_id: input.get_u32()?,
                payload: input.get_bytes()?.to_vec(),
            }),
            2 => Ok(Self::Failure {
                call_id: input.get_u32()?,
                code: input.get_u16()?,
                message: input.get_string()?,
            }),
            tag => Err(CodecError::UnknownTag(tag)),
        }
    }
}

/// A registered procedure body.
///
/// The handler receives the local world, the peer that called and the raw
/// argument bytes (decoded with the game's own codec), and returns the raw
/// reply bytes or an [`RpcError`] to forward to the caller.
pub type RpcHandler = Box<dyn FnMut(&mut World, PeerId, &[u8]) -> Result<Vec<u8>, RpcError> + Send>;

/// What [`RpcDispatcher::dispatch`] decided the caller should do next.
#[derive(Debug)]
pub enum RpcEffect {
    /// Send this message back on the same channel.
    Respond(RpcMessage),
    /// A response to a call this side issued.
    Resolved {
        /// Call id that completed.
        call_id: u32,
        /// Reply payload, or the remote failure.
        result: Result<Vec<u8>, RpcError>,
    },
}

/// Procedures, their handlers and outstanding calls.
///
/// The same dispatcher can be used by a server or a client; set
/// [`local_mode`](Self::local_mode) so the receiver enforces
/// [`RpcSpec::mode`].
pub struct RpcDispatcher {
    table: RpcTable,
    local_mode: RpcMode,
    handlers: Vec<Option<RpcHandler>>,
    pending: HashMap<u32, RpcId>,
    next_call: u32,
}

impl RpcDispatcher {
    /// Dispatcher over `table`, enforcing `local_mode`.
    pub fn new(table: RpcTable, local_mode: RpcMode) -> Self {
        Self {
            table,
            local_mode,
            handlers: Vec::new(),
            pending: HashMap::new(),
            next_call: 0,
        }
    }

    /// The procedure table.
    pub fn table(&self) -> &RpcTable {
        &self.table
    }

    /// This side's role, used by [`RpcMode`] enforcement.
    pub fn local_mode(&self) -> RpcMode {
        self.local_mode
    }

    /// Attach the body for a registered procedure.
    pub fn register(&mut self, id: RpcId, handler: RpcHandler) -> Result<(), NetError> {
        if self.table.spec(id).is_none() {
            return Err(RpcCallError::UnknownId(id).into());
        }
        if self.handlers.len() <= id as usize {
            self.handlers.resize_with(id as usize + 1, || None);
        }
        if self.handlers[id as usize].is_some() {
            return Err(RpcCallError::DuplicateName(
                self.table
                    .spec(id)
                    .map(|spec| spec.name.to_string())
                    .unwrap_or_default(),
            )
            .into());
        }
        self.handlers[id as usize] = Some(handler);
        Ok(())
    }

    /// Allocate a call id and build the `Call` to send on the procedure's
    /// channel.
    pub fn begin_call(&mut self, name: &str, args: &[u8]) -> Result<(RpcId, RpcMessage), NetError> {
        let id = self.table.id(name)?;
        let call_id = self.next_call;
        self.next_call = self.next_call.checked_add(1).ok_or(RpcCallError::TooManyCalls)?;
        if self.pending.insert(call_id, id).is_some() {
            return Err(RpcCallError::UnknownCall(call_id).into());
        }
        Ok((
            id,
            RpcMessage::Call {
                id,
                call_id,
                args: args.to_vec(),
            },
        ))
    }

    /// Handle one inbound message.
    ///
    /// A `Call` runs the handler and yields the reply to send back. A
    /// `Reply`/`Failure` completes one of our calls and yields
    /// [`RpcEffect::Resolved`]. An unknown procedure id or call id is a
    /// [`NetError`].
    pub fn dispatch(
        &mut self,
        world: &mut World,
        peer: PeerId,
        message: &RpcMessage,
    ) -> Result<RpcEffect, NetError> {
        match message {
            RpcMessage::Call { id, call_id, args } => {
                let spec = self
                    .table
                    .spec(*id)
                    .ok_or_else(|| RpcCallError::UnknownId(*id))?;
                if self.local_mode != RpcMode::Any {
                    let caller = match self.local_mode {
                        RpcMode::Server => RpcMode::Client,
                        _ => RpcMode::Server,
                    };
                    if !spec.mode.allows(caller) {
                        return Err(NetError::Protocol(format!(
                            "procedure {:?} is not callable by {:?}",
                            spec.name, caller
                        )));
                    }
                }
                let Some(handler) = self
                    .handlers
                    .get_mut(*id as usize)
                    .and_then(Option::as_mut)
                else {
                    return Ok(RpcEffect::Respond(RpcMessage::Failure {
                        call_id: *call_id,
                        code: crate::error::RPC_NO_HANDLER,
                        message: format!("no handler for {:?}", spec.name),
                    }));
                };
                match handler(world, peer, args) {
                    Ok(payload) => Ok(RpcEffect::Respond(RpcMessage::Reply {
                        call_id: *call_id,
                        payload,
                    })),
                    Err(error) => Ok(RpcEffect::Respond(RpcMessage::Failure {
                        call_id: *call_id,
                        code: error.code,
                        message: error.message,
                    })),
                }
            }
            RpcMessage::Reply { call_id, payload } => {
                self.finish_call(*call_id)?;
                Ok(RpcEffect::Resolved {
                    call_id: *call_id,
                    result: Ok(payload.clone()),
                })
            }
            RpcMessage::Failure {
                call_id,
                code,
                message,
            } => {
                self.finish_call(*call_id)?;
                Ok(RpcEffect::Resolved {
                    call_id: *call_id,
                    result: Err(RpcError::new(*code, message.clone())),
                })
            }
        }
    }

    /// Outstanding calls that have not been answered.
    pub fn pending_calls(&self) -> usize {
        self.pending.len()
    }

    fn finish_call(&mut self, call_id: u32) -> Result<(), NetError> {
        self.pending
            .remove(&call_id)
            .map(|_| ())
            .ok_or_else(|| RpcCallError::UnknownCall(call_id).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{decode_from_slice, encode_to_vec};

    #[test]
    fn rpc_round_trip_through_the_wire_format() {
        let mut table = RpcTable::new();
        let id = table.register("add", 1, RpcMode::Any).unwrap();
        let mut dispatcher = RpcDispatcher::new(table, RpcMode::Any);
        dispatcher
            .register(
                id,
                Box::new(|_world, _peer, args| {
                    let mut reader = Reader::new(args);
                    let a = reader.get_i32().map_err(|error| RpcError::new(3, error.to_string()))?;
                    let b = reader.get_i32().map_err(|error| RpcError::new(3, error.to_string()))?;
                    let mut writer = Writer::new();
                    writer.put_i32(a + b);
                    Ok(writer.finish())
                }),
            )
            .unwrap();

        let mut args = Writer::new();
        args.put_i32(3);
        args.put_i32(4);

        let (_id, call) = dispatcher.begin_call("add", args.as_slice()).unwrap();
        assert_eq!(dispatcher.pending_calls(), 1);

        let wire = encode_to_vec(&call).unwrap();
        let decoded: RpcMessage = decode_from_slice(&wire).unwrap();
        assert_eq!(decoded, call);

        let mut world = World::new();
        let reply = match dispatcher.dispatch(&mut world, 7, &decoded).unwrap() {
            RpcEffect::Respond(reply) => reply,
            other => panic!("expected a reply, got {other:?}"),
        };

        let wire = encode_to_vec(&reply).unwrap();
        let decoded: RpcMessage = decode_from_slice(&wire).unwrap();
        match dispatcher.dispatch(&mut world, 7, &decoded).unwrap() {
            RpcEffect::Resolved { result: Ok(payload), call_id } => {
                assert_eq!(call_id, call.call_id());
                assert_eq!(decode_from_slice::<i32>(&payload).unwrap(), 7);
            }
            other => panic!("expected a resolved call, got {other:?}"),
        }
        assert_eq!(dispatcher.pending_calls(), 0);
    }
}
