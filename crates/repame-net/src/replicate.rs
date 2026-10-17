//! Server-authoritative replication: spawns, despawns and component state
//! snapshots applied into a `bevy_ecs` world (repame-sim's
//! [`World`](repame_sim::bevy_ecs::world::World)).
//!
//! Replication rides the `replication` channel from
//! [`ChannelTable::standard`](crate::ChannelTable::standard) (reliable-ordered),
//! so messages are exactly-once and ordered; the layer below never reorders
//! them.
//!
//! # Model
//!
//! The server [`track`](ReplicationServer::track)s entities. Tracking is
//! explicit because repame has no reflection (see the workspace plan,
//! Tier 2 item 5), so the server cannot discover "replicated components" on
//! its own; it is told which entities to replicate and which component types
//! are replicable via [`ReplicaRegistry`].
//!
//! Each replica has a state: component type id -> encoded component bytes,
//! keyed in type-id order. A snapshot is a *delta against the last
//! acknowledged state*, not against the last sent state: a component whose
//! bytes equal the acked bytes is omitted, one whose bytes differ is sent in
//! full, and one that vanished is sent as a removal. Granularity is the whole
//! component (byte equality), because field-level delta needs reflection the
//! workspace does not have yet.
//!
//! The client acknowledges a snapshot sequence number; acknowledgement is
//! cumulative (`on_ack(seq)` releases every pending snapshot up to `seq`),
//! which is sound because the transport keeps snapshots ordered. Until an
//! acknowledgement arrives, the server keeps the state each snapshot
//! described, so the next delta is computed against the acked one.
//!
//! # Wire layout
//!
//! `ReplicationMessage` is a tag byte plus fields, all through
//! [`crate::codec`]:
//!
//! ```text
//! tag 0 SPAWN    : replica:u32, state        (state = count:u32, (type:u16, bytes)*)
//! tag 1 DESPAWN  : replica:u32
//! tag 2 SNAPSHOT : seq:u32, entries          (entries = count:u32, entry*)
//!                  entry = replica:u32, ops   (ops = count:u32, op*)
//!                  op tag 0 SET    : type:u16, bytes
//!                  op tag 1 REMOVE : type:u16
//! tag 3 ACK      : seq:u32
//! ```

use std::collections::{BTreeMap, HashMap};

use repame_sim::bevy_ecs::prelude::*;

use crate::codec::{Decode, Encode, Reader, Writer};
use crate::error::{CodecError, NetError, ReplicationError};

/// Server-assigned identity of one replicated entity.
pub type ReplicaId = u32;

/// Registry slot of one replicated component type.
pub type ReplicaTypeId = u16;

/// Every replicated component of one entity: type id -> encoded bytes.
pub type ComponentState = BTreeMap<ReplicaTypeId, Vec<u8>>;

/// A component that can be replicated.
///
/// Implement it for any `bevy_ecs` component that also has a wire encoding;
/// `NAME` is the stable identity both peers must agree on.
pub trait Replicated: Component + Encode + Decode {
    /// Stable type name, shared by both peers.
    const NAME: &'static str;
}

#[derive(Clone)]
struct ReplicaType {
    name: &'static str,
    encode: fn(&World, Entity) -> Result<Option<Vec<u8>>, NetError>,
    insert: fn(&mut World, Entity, &[u8]) -> Result<(), NetError>,
    remove: fn(&mut World, Entity),
}

/// Replicated component types, in registration order.
///
/// Register the same types in the same order on both peers; a type id is
/// positional. Clone the registry: a server and a client normally each hold
/// one built from the same list.
#[derive(Clone, Default)]
pub struct ReplicaRegistry {
    types: Vec<ReplicaType>,
    by_name: HashMap<&'static str, ReplicaTypeId>,
}

impl ReplicaRegistry {
    /// Empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a component type and return its id.
    pub fn register<C: Replicated>(&mut self) -> Result<ReplicaTypeId, NetError> {
        if self.by_name.contains_key(C::NAME) {
            return Err(ReplicationError::DuplicateComponentType(C::NAME).into());
        }
        let id = u16::try_from(self.types.len())
            .map_err(|_| ReplicationError::TooManyComponentTypes(self.types.len()))?;
        self.types.push(ReplicaType {
            name: C::NAME,
            encode: |world, entity| {
                world
                    .get::<C>(entity)
                    .map(|component| {
                        let mut writer = Writer::new();
                        component
                            .encode(&mut writer)
                            .map(|()| writer.finish())
                            .map_err(NetError::Codec)
                    })
                    .transpose()
            },
            insert: |world, entity, bytes| {
                let mut reader = Reader::new(bytes);
                let component = C::decode(&mut reader).map_err(NetError::Codec)?;
                reader.expect_end().map_err(NetError::Codec)?;
                world.entity_mut(entity).insert(component);
                Ok(())
            },
            remove: |world, entity| {
                world.entity_mut(entity).remove::<C>();
            },
        });
        self.by_name.insert(C::NAME, id);
        Ok(id)
    }

    /// Id of a registered type, by name.
    pub fn type_id(&self, name: &str) -> Option<ReplicaTypeId> {
        self.by_name.get(name).copied()
    }

    /// Name of a registered type, by id.
    pub fn type_name(&self, id: ReplicaTypeId) -> Option<&'static str> {
        self.types.get(id as usize).map(|ty| ty.name)
    }

    /// Number of registered types.
    pub fn len(&self) -> usize {
        self.types.len()
    }

    /// Whether no type is registered.
    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }

    /// Read every registered component of `entity`.
    pub fn encode_entity(&self, world: &World, entity: Entity) -> Result<ComponentState, NetError> {
        let mut state = ComponentState::new();
        for (id, ty) in self.types.iter().enumerate() {
            if let Some(bytes) = (ty.encode)(world, entity)? {
                state.insert(id as ReplicaTypeId, bytes);
            }
        }
        Ok(state)
    }

    /// Insert one decoded component onto `entity`.
    pub fn insert_component(
        &self,
        world: &mut World,
        entity: Entity,
        type_id: ReplicaTypeId,
        bytes: &[u8],
    ) -> Result<(), NetError> {
        let ty = self
            .types
            .get(type_id as usize)
            .ok_or(ReplicationError::UnknownComponentType(type_id))?;
        (ty.insert)(world, entity, bytes)
    }

    /// Remove one component from `entity`, reporting an unregistered type.
    pub fn remove_component(
        &self,
        world: &mut World,
        entity: Entity,
        type_id: ReplicaTypeId,
    ) -> Result<(), NetError> {
        let ty = self
            .types
            .get(type_id as usize)
            .ok_or(ReplicationError::UnknownComponentType(type_id))?;
        (ty.remove)(world, entity);
        Ok(())
    }

    /// Component-level delta from `previous` to `current`.
    ///
    /// Ops come out in type-id order: sets for present components whose bytes
    /// changed, then removals for components that disappeared. Byte equality
    /// is the change test.
    pub fn diff(previous: &ComponentState, current: &ComponentState) -> Vec<CompOp> {
        let mut ops = Vec::new();
        for (type_id, data) in current {
            if previous.get(type_id) != Some(data) {
                ops.push(CompOp::Set {
                    type_id: *type_id,
                    data: data.clone(),
                });
            }
        }
        for type_id in previous.keys() {
            if !current.contains_key(type_id) {
                ops.push(CompOp::Remove { type_id: *type_id });
            }
        }
        ops
    }
}

/// One change inside a snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompOp {
    /// The component is present with these bytes.
    Set {
        /// Registered component type id.
        type_id: ReplicaTypeId,
        /// Full encoded component value.
        data: Vec<u8>,
    },
    /// The component is gone.
    Remove {
        /// Registered component type id.
        type_id: ReplicaTypeId,
    },
}

impl Encode for CompOp {
    fn encode(&self, out: &mut Writer) -> Result<(), CodecError> {
        match self {
            Self::Set { type_id, data } => {
                out.put_u8(0);
                out.put_u16(*type_id);
                out.put_bytes(data)
            }
            Self::Remove { type_id } => {
                out.put_u8(1);
                out.put_u16(*type_id);
                Ok(())
            }
        }
    }
}

impl Decode for CompOp {
    fn decode(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        match input.get_u8()? {
            0 => Ok(Self::Set {
                type_id: input.get_u16()?,
                data: input.get_bytes()?.to_vec(),
            }),
            1 => Ok(Self::Remove {
                type_id: input.get_u16()?,
            }),
            tag => Err(CodecError::UnknownTag(tag)),
        }
    }
}

/// One replica's changes inside a snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotEntry {
    /// Replica the ops apply to.
    pub replica: ReplicaId,
    /// Changes, in type-id order.
    pub ops: Vec<CompOp>,
}

impl Encode for SnapshotEntry {
    fn encode(&self, out: &mut Writer) -> Result<(), CodecError> {
        out.put_u32(self.replica);
        self.ops.encode(out)
    }
}

impl Decode for SnapshotEntry {
    fn decode(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        Ok(Self {
            replica: input.get_u32()?,
            ops: Vec::<CompOp>::decode(input)?,
        })
    }
}

/// A replication message on the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplicationMessage {
    /// A replica exists; its full state follows.
    Spawn {
        /// Server-assigned replica id.
        replica: ReplicaId,
        /// Full component state.
        state: ComponentState,
    },
    /// A replica was destroyed.
    Despawn {
        /// Server-assigned replica id.
        replica: ReplicaId,
    },
    /// Component deltas against the last acknowledged state.
    Snapshot {
        /// Monotonic snapshot sequence number, starting at `0`.
        seq: u32,
        /// One entry per replica that changed.
        entries: Vec<SnapshotEntry>,
    },
    /// Client acknowledgement: every snapshot up to and including `seq`.
    Ack {
        /// Highest applied snapshot sequence number.
        seq: u32,
    },
}

impl ReplicationMessage {
    /// A despawn for `replica`, convenient for a server-side teardown path.
    pub fn despawn(replica: ReplicaId) -> Self {
        Self::Despawn { replica }
    }
}

impl Encode for ReplicationMessage {
    fn encode(&self, out: &mut Writer) -> Result<(), CodecError> {
        match self {
            Self::Spawn { replica, state } => {
                out.put_u8(0);
                out.put_u32(*replica);
                state.encode(out)
            }
            Self::Despawn { replica } => {
                out.put_u8(1);
                out.put_u32(*replica);
                Ok(())
            }
            Self::Snapshot { seq, entries } => {
                out.put_u8(2);
                out.put_u32(*seq);
                entries.encode(out)
            }
            Self::Ack { seq } => {
                out.put_u8(3);
                out.put_u32(*seq);
                Ok(())
            }
        }
    }
}

impl Decode for ReplicationMessage {
    fn decode(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        match input.get_u8()? {
            0 => Ok(Self::Spawn {
                replica: input.get_u32()?,
                state: ComponentState::decode(input)?,
            }),
            1 => Ok(Self::Despawn {
                replica: input.get_u32()?,
            }),
            2 => Ok(Self::Snapshot {
                seq: input.get_u32()?,
                entries: Vec::<SnapshotEntry>::decode(input)?,
            }),
            3 => Ok(Self::Ack {
                seq: input.get_u32()?,
            }),
            tag => Err(CodecError::UnknownTag(tag)),
        }
    }
}

/// What happened when a client applied a replication message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplicaEvent {
    /// A replica was spawned and mapped to a local entity.
    Spawned {
        /// Server-assigned replica id.
        replica: ReplicaId,
        /// Local entity.
        entity: Entity,
    },
    /// A replica and its local entity were destroyed.
    Despawned {
        /// Server-assigned replica id.
        replica: ReplicaId,
    },
    /// A replica's components were updated.
    Updated {
        /// Server-assigned replica id.
        replica: ReplicaId,
    },
    /// The message named a replica this client never spawned.
    UnknownReplica {
        /// Server-assigned replica id.
        replica: ReplicaId,
    },
    /// A snapshot older than one already applied; it was skipped.
    StaleSnapshot {
        /// Sequence number that arrived.
        seq: u32,
        /// Highest sequence number already applied.
        last: u32,
    },
}

/// Server side: tracks entities and produces deltas.
///
/// Register the same [`ReplicaRegistry`] on both peers, then
/// [`track`](Self::track) whenever an entity becomes networked and send the
/// [`spawn_message`](Self::spawn_message) once. Call
/// [`snapshot`](Self::snapshot) every replication tick on the `replication`
/// channel and feed client [`Ack`](ReplicationMessage::Ack)s back into
/// [`on_ack`](Self::on_ack).
pub struct ReplicationServer {
    registry: ReplicaRegistry,
    next_replica: ReplicaId,
    entities: HashMap<Entity, ReplicaId>,
    replicas: BTreeMap<ReplicaId, Entity>,
    acked: BTreeMap<ReplicaId, ComponentState>,
    pending: BTreeMap<u32, BTreeMap<ReplicaId, ComponentState>>,
    next_snapshot: u32,
    /// Unacknowledged snapshots kept before the server gives up on the client.
    pub max_pending_snapshots: usize,
}

impl ReplicationServer {
    /// Server over `registry`.
    pub fn new(registry: ReplicaRegistry) -> Self {
        Self {
            registry,
            next_replica: 1,
            entities: HashMap::new(),
            replicas: BTreeMap::new(),
            acked: BTreeMap::new(),
            pending: BTreeMap::new(),
            next_snapshot: 0,
            max_pending_snapshots: 128,
        }
    }

    /// The component registry both peers share.
    pub fn registry(&self) -> &ReplicaRegistry {
        &self.registry
    }

    /// Start replicating `entity`, assigning it a replica id.
    pub fn track(&mut self, entity: Entity) -> Result<ReplicaId, NetError> {
        if self.entities.contains_key(&entity) {
            return Err(ReplicationError::AlreadyTracked.into());
        }
        let replica = self.next_replica;
        self.next_replica = self.next_replica.wrapping_add(1).max(1);
        self.entities.insert(entity, replica);
        self.replicas.insert(replica, entity);
        Ok(replica)
    }

    /// Stop replicating `entity`; send [`ReplicationMessage::despawn`] to clients.
    pub fn untrack(&mut self, entity: Entity) -> Result<ReplicaId, NetError> {
        let replica = self
            .entities
            .remove(&entity)
            .ok_or(ReplicationError::NotTracked)?;
        self.replicas.remove(&replica);
        self.acked.remove(&replica);
        Ok(replica)
    }

    /// Replica id of a tracked entity.
    pub fn replica_of(&self, entity: Entity) -> Option<ReplicaId> {
        self.entities.get(&entity).copied()
    }

    /// Entity of a replica id.
    pub fn entity_of(&self, replica: ReplicaId) -> Option<Entity> {
        self.replicas.get(&replica).copied()
    }

    /// Number of tracked replicas.
    pub fn len(&self) -> usize {
        self.replicas.len()
    }

    /// Whether nothing is tracked.
    pub fn is_empty(&self) -> bool {
        self.replicas.is_empty()
    }

    /// Full state of one replica, to send once when it spawns.
    pub fn spawn_message(
        &self,
        world: &World,
        replica: ReplicaId,
    ) -> Result<ReplicationMessage, NetError> {
        let entity = self
            .replicas
            .get(&replica)
            .ok_or(ReplicationError::UnknownReplica(replica))?;
        Ok(ReplicationMessage::Spawn {
            replica,
            state: self.registry.encode_entity(world, *entity)?,
        })
    }

    /// Build the next delta snapshot and record what it describes.
    ///
    /// Only replicas whose state differs from the last acknowledged state
    /// produce ops. Snapshots with no changes are still numbered and sent (the
    /// sequence number is what the client acknowledges), but carry no entries.
    /// An untracked or despawned entity is skipped; call
    /// [`untrack`](Self::untrack) to stop replicating it.
    pub fn snapshot(&mut self, world: &World) -> Result<ReplicationMessage, NetError> {
        if self.pending.len() > self.max_pending_snapshots {
            return Err(NetError::Protocol(format!(
                "replication backlog of {} unacknowledged snapshots exceeded the cap of {}",
                self.pending.len(),
                self.max_pending_snapshots
            )));
        }
        let seq = self.next_snapshot;
        self.next_snapshot = seq.wrapping_add(1);
        let mut entries = Vec::new();
        let mut full = BTreeMap::new();
        for (replica, entity) in &self.replicas {
            if !world.entities().contains(*entity) {
                continue;
            }
            let current = self.registry.encode_entity(world, *entity)?;
            let ops = match self.acked.get(replica) {
                Some(previous) => ReplicaRegistry::diff(previous, &current),
                None => ReplicaRegistry::diff(&ComponentState::new(), &current),
            };
            if !ops.is_empty() {
                entries.push(SnapshotEntry {
                    replica: *replica,
                    ops,
                });
            }
            full.insert(*replica, current);
        }
        if !full.is_empty() {
            self.pending.insert(seq, full);
        }
        Ok(ReplicationMessage::Snapshot { seq, entries })
    }

    /// Apply a cumulative acknowledgement; returns how many snapshots it released.
    pub fn on_ack(&mut self, seq: u32) -> usize {
        let released: Vec<u32> = self.pending.range(..=seq).map(|(key, _)| *key).collect();
        for key in &released {
            if let Some(states) = self.pending.remove(key) {
                for (replica, state) in states {
                    if self.replicas.contains_key(&replica) {
                        self.acked.insert(replica, state);
                    }
                }
            }
        }
        released.len()
    }

    /// Snapshots sent but not yet acknowledged.
    pub fn pending_snapshots(&self) -> usize {
        self.pending.len()
    }
}

/// Client side: applies snapshots into a local world.
///
/// Feed it decoded [`ReplicationMessage`]s from the `replication` channel and
/// send the [`ack_message`](Self::ack_message) back after applying each
/// snapshot. Unknown replica ids are reported as events, not dropped
/// silently; malformed data is a [`NetError`].
pub struct ReplicationClient {
    registry: ReplicaRegistry,
    replicas: BTreeMap<ReplicaId, Entity>,
    last_seq: Option<u32>,
}

impl ReplicationClient {
    /// Client over the shared `registry`.
    pub fn new(registry: ReplicaRegistry) -> Self {
        Self {
            registry,
            replicas: BTreeMap::new(),
            last_seq: None,
        }
    }

    /// The component registry both peers share.
    pub fn registry(&self) -> &ReplicaRegistry {
        &self.registry
    }

    /// Local entity of a replica id.
    pub fn entity(&self, replica: ReplicaId) -> Option<Entity> {
        self.replicas.get(&replica).copied()
    }

    /// Number of live replicas.
    pub fn len(&self) -> usize {
        self.replicas.len()
    }

    /// Whether no replica is live.
    pub fn is_empty(&self) -> bool {
        self.replicas.is_empty()
    }

    /// Highest snapshot sequence number applied so far.
    pub fn last_seq(&self) -> Option<u32> {
        self.last_seq
    }

    /// The acknowledgement to send back, once a snapshot has been applied.
    pub fn ack_message(&self) -> Option<ReplicationMessage> {
        self.last_seq.map(|seq| ReplicationMessage::Ack { seq })
    }

    /// Apply one message to `world`.
    ///
    /// `Spawn` creates an entity and inserts every component; `Despawn`
    /// destroys it; `Snapshot` writes the ops. A snapshot at or below the last
    /// applied sequence is skipped as [`ReplicaEvent::StaleSnapshot`]. An op
    /// naming an unregistered component type is an error, never ignored.
    pub fn apply(
        &mut self,
        world: &mut World,
        message: &ReplicationMessage,
    ) -> Result<Vec<ReplicaEvent>, NetError> {
        let mut events = Vec::new();
        match message {
            ReplicationMessage::Spawn { replica, state } => {
                if self.replicas.contains_key(replica) {
                    return Err(ReplicationError::DuplicateReplica(*replica).into());
                }
                let entity = world.spawn_empty().id();
                for (type_id, bytes) in state {
                    self.registry.insert_component(world, entity, *type_id, bytes)?;
                }
                self.replicas.insert(*replica, entity);
                events.push(ReplicaEvent::Spawned {
                    replica: *replica,
                    entity,
                });
            }
            ReplicationMessage::Despawn { replica } => match self.replicas.remove(replica) {
                Some(entity) => {
                    world.despawn(entity);
                    events.push(ReplicaEvent::Despawned { replica: *replica });
                }
                None => events.push(ReplicaEvent::UnknownReplica { replica: *replica }),
            },
            ReplicationMessage::Snapshot { seq, entries } => {
                if let Some(last) = self.last_seq {
                    if *seq <= last {
                        events.push(ReplicaEvent::StaleSnapshot { seq: *seq, last });
                        return Ok(events);
                    }
                }
                for entry in entries {
                    let Some(entity) = self.replicas.get(&entry.replica).copied() else {
                        events.push(ReplicaEvent::UnknownReplica {
                            replica: entry.replica,
                        });
                        continue;
                    };
                    for op in &entry.ops {
                        match op {
                            CompOp::Set { type_id, data } => {
                                self.registry
                                    .insert_component(world, entity, *type_id, data)?;
                            }
                            CompOp::Remove { type_id } => {
                                self.registry.remove_component(world, entity, *type_id)?;
                            }
                        }
                    }
                    events.push(ReplicaEvent::Updated {
                        replica: entry.replica,
                    });
                }
                self.last_seq = Some(*seq);
            }
            ReplicationMessage::Ack { .. } => {}
        }
        Ok(events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{decode_from_slice, encode_to_vec};

    #[derive(Component, Debug, Clone, PartialEq)]
    struct Health(u32);

    impl Encode for Health {
        fn encode(&self, out: &mut Writer) -> Result<(), CodecError> {
            out.put_u32(self.0);
            Ok(())
        }
    }

    impl Decode for Health {
        fn decode(input: &mut Reader<'_>) -> Result<Self, CodecError> {
            Ok(Self(input.get_u32()?))
        }
    }

    impl Replicated for Health {
        const NAME: &'static str = "Health";
    }

    #[derive(Component, Debug, Clone, PartialEq)]
    struct Position {
        x: f32,
        y: f32,
    }

    impl Encode for Position {
        fn encode(&self, out: &mut Writer) -> Result<(), CodecError> {
            out.put_f32(self.x);
            out.put_f32(self.y);
            Ok(())
        }
    }

    impl Decode for Position {
        fn decode(input: &mut Reader<'_>) -> Result<Self, CodecError> {
            Ok(Self {
                x: input.get_f32()?,
                y: input.get_f32()?,
            })
        }
    }

    impl Replicated for Position {
        const NAME: &'static str = "Position";
    }

    fn registry() -> ReplicaRegistry {
        let mut registry = ReplicaRegistry::new();
        registry.register::<Health>().unwrap();
        registry.register::<Position>().unwrap();
        registry
    }

    #[test]
    fn delta_snapshot_round_trip_preserves_component_values() {
        let mut server_world = World::new();
        let entity = server_world.spawn((Health(100), Position { x: 1.5, y: -2.25 })).id();
        let mut server = ReplicationServer::new(registry());
        let replica = server.track(entity).unwrap();

        let spawn = encode_to_vec(&server.spawn_message(&server_world, replica).unwrap()).unwrap();
        let mut client = ReplicationClient::new(registry());
        let mut client_world = World::new();
        let applied = client
            .apply(&mut client_world, &decode_from_slice(&spawn).unwrap())
            .unwrap();
        assert!(matches!(applied[0], ReplicaEvent::Spawned { .. }));
        let local = client.entity(replica).unwrap();
        assert_eq!(*client_world.get::<Health>(local).unwrap(), Health(100));
        assert_eq!(
            *client_world.get::<Position>(local).unwrap(),
            Position { x: 1.5, y: -2.25 }
        );

        let first = server.snapshot(&server_world).unwrap();
        let first_seq = match &first {
            ReplicationMessage::Snapshot { seq, .. } => *seq,
            other => panic!("expected a snapshot, got {other:?}"),
        };
        let bytes = encode_to_vec(&first).unwrap();
        client
            .apply(&mut client_world, &decode_from_slice(&bytes).unwrap())
            .unwrap();
        assert_eq!(server.on_ack(first_seq), 1);

        server_world.get_mut::<Health>(entity).unwrap().0 = 42;
        let delta = server.snapshot(&server_world).unwrap();
        match &delta {
            ReplicationMessage::Snapshot { entries, .. } => {
                assert_eq!(entries.len(), 1, "only the changed replica is sent");
                assert_eq!(entries[0].ops.len(), 1, "only Health changed");
                assert_eq!(
                    entries[0].ops[0],
                    CompOp::Set {
                        type_id: 0,
                        data: encode_to_vec(&Health(42)).unwrap()
                    }
                );
            }
            other => panic!("expected a snapshot, got {other:?}"),
        }
        let bytes = encode_to_vec(&delta).unwrap();
        client
            .apply(&mut client_world, &decode_from_slice(&bytes).unwrap())
            .unwrap();
        assert_eq!(*client_world.get::<Health>(local).unwrap(), Health(42));
        assert_eq!(
            *client_world.get::<Position>(local).unwrap(),
            Position { x: 1.5, y: -2.25 },
            "an omitted component keeps its value"
        );

        let unchanged = server.snapshot(&server_world).unwrap();
        match &unchanged {
            ReplicationMessage::Snapshot { entries, .. } => {
                assert!(entries.is_empty(), "no delta against the acked state");
            }
            other => panic!("expected a snapshot, got {other:?}"),
        }
    }
}
