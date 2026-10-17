use std::collections::BTreeMap;

use bevy_ecs::prelude::{Entity, World};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::ReflectError;
use crate::registry::Registry;

/// Format version written into every snapshot this crate produces.
///
/// [`load_world`] rejects any other version with [`ReflectError::Version`].
pub const WORLD_FORMAT: u32 = 1;

/// A RON world snapshot: the document [`save_world`] writes and [`load_world`]
/// reads.
///
/// `entities` is sorted ascending by [`Entity`] bits, and each entity's
/// `components` is a [`BTreeMap`], so the serialized bytes are a pure function
/// of the registered set and the stored component values.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorldSnapshot {
    /// [`WORLD_FORMAT`] at the time of writing.
    pub format: u32,
    /// The reflected entities, ordered by id.
    #[serde(default)]
    pub entities: Vec<EntitySnapshot>,
}

/// One entity in a [`WorldSnapshot`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EntitySnapshot {
    /// The source entity's [`Entity::to_bits`] value.
    pub id: u64,
    /// Component values keyed by registered name, ordered by name.
    #[serde(default)]
    pub components: BTreeMap<String, Value>,
}

/// What [`load_world`] did: the source-id to spawned-entity map and how many
/// components were applied.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoadReport {
    /// Mapping from each document entity id to the entity spawned for it.
    pub entities: BTreeMap<u64, Entity>,
    /// Total components inserted across every spawned entity.
    pub components: usize,
}

impl LoadReport {
    /// The entity spawned for the document id `old`, if any.
    pub fn entity(&self, old: u64) -> Option<Entity> {
        self.entities.get(&old).copied()
    }
}

/// Capture the reflected state of `world` as a structured document.
///
/// Entities carrying no registered component are omitted. A component that
/// fails to serialize is [`ReflectError::BadValue`], not a dropped field.
pub fn capture_world(world: &World, registry: &Registry) -> Result<WorldSnapshot, ReflectError> {
    let mut entities = Vec::new();
    for source in world.iter_entities() {
        let mut components = BTreeMap::new();
        for (name, _) in registry.iter() {
            if let Some(value) = registry.serialize(world, source.id(), name)? {
                components.insert(name.to_string(), value);
            }
        }
        if components.is_empty() {
            continue;
        }
        entities.push(EntitySnapshot {
            id: source.id().to_bits(),
            components,
        });
    }
    entities.sort_by_key(|entity| entity.id);
    Ok(WorldSnapshot {
        format: WORLD_FORMAT,
        entities,
    })
}

/// Serialize the reflected state of `world` into a RON document.
///
/// Deterministic: two saves of an unchanged world produce identical bytes,
/// because entities are ordered by id and components by name.
pub fn save_world(world: &World, registry: &Registry) -> Result<String, ReflectError> {
    let snapshot = capture_world(world, registry)?;
    let mut text = ron::ser::to_string_pretty(&snapshot, Default::default())
        .map_err(|source| ReflectError::Format { source })?;
    text.push('\n');
    Ok(text)
}

/// Spawn the entities of a RON document into `world`.
///
/// Every component name is checked against `registry` before anything is
/// spawned, so an unregistered name is [`ReflectError::UnknownType`] and
/// leaves `world` untouched. Existing entities are preserved; the returned
/// [`LoadReport`] maps each document id to the entity spawned for it.
pub fn load_world(
    world: &mut World,
    registry: &Registry,
    text: &str,
) -> Result<LoadReport, ReflectError> {
    let snapshot: WorldSnapshot =
        ron::from_str(text).map_err(|source| ReflectError::Format { source: source.into() })?;
    if snapshot.format != WORLD_FORMAT {
        return Err(ReflectError::Version {
            found: snapshot.format,
            expected: WORLD_FORMAT,
        });
    }
    for entity in &snapshot.entities {
        for name in entity.components.keys() {
            if !registry.contains(name) {
                return Err(ReflectError::UnknownType { name: name.clone() });
            }
        }
    }

    let mut report = LoadReport::default();
    for entity in &snapshot.entities {
        let target = world.spawn_empty().id();
        for (name, value) in &entity.components {
            registry.insert(world, target, name, value)?;
            report.components += 1;
        }
        report.entities.insert(entity.id, target);
    }
    Ok(report)
}
