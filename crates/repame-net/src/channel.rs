//! Named channels and their reliability modes.
//!
//! A channel is the unit of ordering and reliability, exactly as in Godot's
//! `MultiplayerPeer`/ENet mapping (`SYSCH_RELIABLE`, `SYSCH_UNRELIABLE`, plus
//! user channels). Every channel has its own sequence space on each
//! direction, so a reliable channel never head-of-line blocks another.
//!
//! Ids are assigned in registration order, `0..len`, and are stable for the
//! lifetime of the table. Both peers must build the same table or the wire
//! carries ids that mean different reliability on the two sides; the codec
//! version check does not catch that, so games should derive the table from
//! one shared definition (see [`ChannelTable::standard`]).

use std::collections::HashMap;

use crate::error::{ChannelError, CodecError, NetError};
use crate::codec::{Decode, Encode, Reader, Writer};

/// Channel identifier. `u8`, so a table holds at most 256 channels.
pub type ChannelId = u8;

/// Delivery guarantee for one channel.
///
/// [`Reliability::Unreliable`] is the only mode where loss is acceptable by
/// contract; the two reliable modes retransmit until acknowledged or the
/// retry budget runs out, which is reported as
/// [`NetError::RetryLimit`] rather than dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reliability {
    /// Exactly-once, in sequence-number order. Gaps are buffered until filled.
    ReliableOrdered,
    /// Exactly-once, delivered as it arrives. Gaps do not block later messages.
    ReliableUnordered,
    /// Best effort. May be lost, duplicated or reordered; no retransmission.
    Unreliable,
}

impl Reliability {
    /// Whether the sender keeps the message until it is acknowledged.
    pub fn is_reliable(self) -> bool {
        matches!(self, Self::ReliableOrdered | Self::ReliableUnordered)
    }

    /// Whether the receiver may hand messages up out of sequence order.
    pub fn preserves_order(self) -> bool {
        matches!(self, Self::ReliableOrdered)
    }
}

/// One channel definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelSpec {
    /// Stable, unique name. Games look channels up by this.
    pub name: &'static str,
    /// Delivery guarantee for this channel.
    pub reliability: Reliability,
}

/// Name-to-id table with the reliability of every channel.
#[derive(Debug, Clone, Default)]
pub struct ChannelTable {
    specs: Vec<ChannelSpec>,
    by_name: HashMap<&'static str, ChannelId>,
}

/// The logical channels [`ChannelTable::standard`] registers.
pub mod names {
    /// Replication traffic: spawns, despawns and state snapshots.
    pub const REPLICATION: &str = "replication";
    /// Remote procedure calls and their replies.
    pub const RPC: &str = "rpc";
    /// High-frequency, non-authoritative state (positions, cosmetic ticks).
    pub const STATE: &str = "state";
}

impl ChannelTable {
    /// Build a table, assigning ids in slice order.
    ///
    /// Fails on a duplicate name or more than 256 channels.
    pub fn new(specs: &[ChannelSpec]) -> Result<Self, NetError> {
        if specs.len() > u8::MAX as usize + 1 {
            return Err(NetError::Channel(ChannelError::TooManyChannels(
                specs.len(),
            )));
        }
        let mut by_name = HashMap::with_capacity(specs.len());
        for (id, spec) in specs.iter().enumerate() {
            if by_name.insert(spec.name, id as ChannelId).is_some() {
                return Err(NetError::Channel(ChannelError::DuplicateName(
                    spec.name.to_string(),
                )));
            }
        }
        Ok(Self {
            specs: specs.to_vec(),
            by_name,
        })
    }

    /// The three channels every repame session starts with: reliable-ordered
    /// `replication` (0), reliable-ordered `rpc` (1) and unreliable `state` (2).
    pub fn standard() -> Self {
        Self::new(&[
            ChannelSpec {
                name: names::REPLICATION,
                reliability: Reliability::ReliableOrdered,
            },
            ChannelSpec {
                name: names::RPC,
                reliability: Reliability::ReliableOrdered,
            },
            ChannelSpec {
                name: names::STATE,
                reliability: Reliability::Unreliable,
            },
        ])
        .expect("the standard three-channel table is always valid")
    }

    /// Look an id up by name.
    pub fn id(&self, name: &str) -> Result<ChannelId, NetError> {
        self.by_name
            .get(name)
            .copied()
            .ok_or_else(|| ChannelError::UnknownName(name.to_string()).into())
    }

    /// Look a name up by id.
    pub fn name(&self, id: ChannelId) -> Option<&'static str> {
        self.spec(id).map(|spec| spec.name)
    }

    /// Look a spec up by id.
    pub fn spec(&self, id: ChannelId) -> Option<&ChannelSpec> {
        self.specs.get(id as usize)
    }

    /// Delivery guarantee of one channel.
    pub fn reliability(&self, id: ChannelId) -> Result<Reliability, NetError> {
        self.spec(id)
            .map(|spec| spec.reliability)
            .ok_or_else(|| ChannelError::UnknownId(id).into())
    }

    /// Number of registered channels.
    pub fn len(&self) -> usize {
        self.specs.len()
    }

    /// Whether the table has no channels.
    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }

    /// Every channel in id order.
    pub fn iter(&self) -> impl Iterator<Item = (ChannelId, &ChannelSpec)> {
        self.specs
            .iter()
            .enumerate()
            .map(|(id, spec)| (id as ChannelId, spec))
    }
}

impl Encode for Reliability {
    fn encode(&self, out: &mut Writer) -> Result<(), CodecError> {
        out.put_u8(match self {
            Self::ReliableOrdered => 0,
            Self::ReliableUnordered => 1,
            Self::Unreliable => 2,
        });
        Ok(())
    }
}

impl Decode for Reliability {
    fn decode(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        match input.get_u8()? {
            0 => Ok(Self::ReliableOrdered),
            1 => Ok(Self::ReliableUnordered),
            2 => Ok(Self::Unreliable),
            tag => Err(CodecError::UnknownTag(tag)),
        }
    }
}
