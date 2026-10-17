use bevy_ecs::prelude::{Entity, World};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::ReflectError;
use crate::registry::Registry;

/// One component of an entity, as a debug panel reads it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ComponentView {
    /// Registered component name.
    pub name: String,
    /// The component's field tree.
    pub value: Value,
}

/// Entities carrying at least one registered component, ordered by id.
///
/// A panel binds its entity list to this. Entities with no reflected component
/// (including `bevy_ecs`'s internal resource entities) are omitted, and ids are
/// sorted rather than left in archetype order so the list is stable.
pub fn entities(world: &World, registry: &Registry) -> Vec<Entity> {
    let mut entities: Vec<Entity> = world
        .iter_entities()
        .map(|entity| entity.id())
        .filter(|entity| registry.any_present(world, *entity))
        .collect();
    entities.sort_by_key(|entity| entity.to_bits());
    entities
}

/// The registered components `entity` carries, ordered by name.
///
/// A component that fails to serialize is reported with a `log` warning and
/// omitted from the returned tree; use [`set`]/[`value`] when a typed error is
/// needed.
pub fn inspect(world: &World, registry: &Registry, entity: Entity) -> Vec<ComponentView> {
    let mut views = Vec::new();
    for (name, _) in registry.iter() {
        match registry.serialize(world, entity, name) {
            Ok(Some(value)) => views.push(ComponentView {
                name: name.to_string(),
                value,
            }),
            Ok(None) => {}
            Err(error) => log::warn!("{error}"),
        }
    }
    views
}

/// Whether `entity` carries the registered component `name`.
///
/// An unregistered `name` is `false`.
pub fn has(world: &World, registry: &Registry, entity: Entity, name: &str) -> bool {
    registry
        .find(name)
        .is_some_and(|vtable| (vtable.contains)(world, entity))
}

/// One component value, or [`ReflectError::MissingComponent`] when absent.
pub fn value(
    world: &World,
    registry: &Registry,
    entity: Entity,
    name: &str,
) -> Result<Value, ReflectError> {
    match registry.serialize(world, entity, name)? {
        Some(value) => Ok(value),
        None => Err(ReflectError::MissingComponent {
            entity,
            name: name.to_string(),
        }),
    }
}

/// Deserialize `value` into the registered component `name` and insert it,
/// replacing any existing value.
///
/// An unregistered name is [`ReflectError::NotRegistered`], a value that does
/// not fit the type is [`ReflectError::BadValue`], and a missing entity is
/// [`ReflectError::MissingComponent`].
pub fn set(
    world: &mut World,
    registry: &Registry,
    entity: Entity,
    name: &str,
    value: Value,
) -> Result<(), ReflectError> {
    registry.require(name)?;
    if world.get_entity_mut(entity).is_err() {
        return Err(ReflectError::MissingComponent {
            entity,
            name: name.to_string(),
        });
    }
    registry.insert(world, entity, name, &value)
}

/// Remove the registered component `name` from `entity`.
///
/// [`ReflectError::MissingComponent`] when the entity does not carry it.
pub fn remove(
    world: &mut World,
    registry: &Registry,
    entity: Entity,
    name: &str,
) -> Result<(), ReflectError> {
    let vtable = registry.require(name)?;
    if !(vtable.contains)(world, entity) {
        return Err(ReflectError::MissingComponent {
            entity,
            name: name.to_string(),
        });
    }
    (vtable.remove)(world, entity);
    Ok(())
}
