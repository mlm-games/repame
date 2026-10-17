use std::any::TypeId;
use std::collections::BTreeMap;

use bevy_ecs::prelude::{Component, Entity, World};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::error::{ReflectError, bad_value};

/// Monomorphized `serialize component -> value`. `None` when the entity does
/// not carry the component; `Err` when the component itself fails to
/// serialize.
pub(crate) type SerializeFn = fn(&World, Entity) -> Result<Option<Value>, serde_json::Error>;

/// Monomorphized `deserialize value -> insert`. The caller guarantees the
/// entity exists (it is checked by every public entry point).
pub(crate) type InsertFn = fn(&mut World, Entity, &Value) -> Result<(), serde_json::Error>;

/// Monomorphized presence check.
pub(crate) type ContainsFn = fn(&World, Entity) -> bool;

/// Monomorphized removal.
pub(crate) type RemoveFn = fn(&mut World, Entity);

/// One reflected component type, as four monomorphized function pointers.
#[derive(Clone, Copy, Debug)]
pub(crate) struct VTable {
    pub(crate) serialize: SerializeFn,
    pub(crate) insert: InsertFn,
    pub(crate) contains: ContainsFn,
    pub(crate) remove: RemoveFn,
}

/// A registry of reflected component types, keyed by an explicit, stable name.
///
/// A type registers once under one name; a second registration of the same type
/// or an already-used name is [`ReflectError::AlreadyRegistered`]. Names are
/// caller-supplied strings (for example `"repame::Transform"`), never
/// `std::any::type_name`, so a document naming a component keeps its meaning
/// across refactors.
///
/// The registry borrows nothing and owns only function pointers, so it is
/// cheap to clone-by-reference and safe to share with a debug panel.
#[derive(Default)]
pub struct Registry {
    types: BTreeMap<String, VTable>,
    by_type: BTreeMap<TypeId, String>,
}

impl Registry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `C` under `name`.
    ///
    /// Errors with [`ReflectError::BadName`] if `name` is empty, padded, or
    /// holds control characters, and with [`ReflectError::AlreadyRegistered`]
    /// if `C` or `name` is already present.
    pub fn register<C>(&mut self, name: impl Into<String>) -> Result<(), ReflectError>
    where
        C: Component + Serialize + DeserializeOwned + Clone + Send + Sync + 'static,
    {
        let name = name.into();
        validate_name(&name)?;
        if let Some(existing) = self.by_type.get(&TypeId::of::<C>()) {
            return Err(ReflectError::AlreadyRegistered {
                name: existing.clone(),
            });
        }
        if self.types.contains_key(&name) {
            return Err(ReflectError::AlreadyRegistered { name });
        }
        self.by_type.insert(TypeId::of::<C>(), name.clone());
        self.types.insert(
            name,
            VTable {
                serialize: serialize_component::<C>,
                insert: insert_component::<C>,
                contains: contains_component::<C>,
                remove: remove_component::<C>,
            },
        );
        Ok(())
    }

    /// Whether `name` is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.types.contains_key(name)
    }

    /// Registered names, sorted. The documented ordering of snapshots.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.types.keys().map(String::as_str)
    }

    /// Number of registered types.
    pub fn len(&self) -> usize {
        self.types.len()
    }

    /// Whether nothing is registered.
    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }

    /// Registered types in name order, paired with their name.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&str, &VTable)> {
        self.types.iter().map(|(name, vtable)| (name.as_str(), vtable))
    }

    /// The vtable for `name`, or [`ReflectError::NotRegistered`].
    pub(crate) fn require(&self, name: &str) -> Result<&VTable, ReflectError> {
        self.types
            .get(name)
            .ok_or_else(|| ReflectError::NotRegistered {
                name: name.to_string(),
            })
    }

    /// The vtable for `name`, if registered.
    pub(crate) fn find(&self, name: &str) -> Option<&VTable> {
        self.types.get(name)
    }

    /// Whether `entity` carries any registered component.
    pub(crate) fn any_present(&self, world: &World, entity: Entity) -> bool {
        self.types
            .values()
            .any(|vtable| (vtable.contains)(world, entity))
    }

    /// Serialize one named component into a value.
    pub(crate) fn serialize(
        &self,
        world: &World,
        entity: Entity,
        name: &str,
    ) -> Result<Option<Value>, ReflectError> {
        let vtable = self.require(name)?;
        (vtable.serialize)(world, entity).map_err(|source| bad_value(name, source))
    }

    /// Deserialize a value and insert it as the named component.
    pub(crate) fn insert(
        &self,
        world: &mut World,
        entity: Entity,
        name: &str,
        value: &Value,
    ) -> Result<(), ReflectError> {
        let vtable = self.require(name)?;
        (vtable.insert)(world, entity, value).map_err(|source| bad_value(name, source))
    }
}

impl std::fmt::Debug for Registry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Registry")
            .field("types", &self.types.keys().collect::<Vec<_>>())
            .finish()
    }
}

fn validate_name(name: &str) -> Result<(), ReflectError> {
    if name.is_empty() || name.trim() != name || name.chars().any(char::is_control) {
        return Err(ReflectError::BadName {
            name: name.to_string(),
        });
    }
    Ok(())
}

fn serialize_component<C>(world: &World, entity: Entity) -> Result<Option<Value>, serde_json::Error>
where
    C: Component + Serialize,
{
    match world.get::<C>(entity) {
        Some(component) => serde_json::to_value(component).map(Some),
        None => Ok(None),
    }
}

fn insert_component<C>(
    world: &mut World,
    entity: Entity,
    value: &Value,
) -> Result<(), serde_json::Error>
where
    C: Component + DeserializeOwned,
{
    let component: C = serde_json::from_value(value.clone())?;
    world.entity_mut(entity).insert(component);
    Ok(())
}

fn contains_component<C: Component>(world: &World, entity: Entity) -> bool {
    world
        .get_entity(entity)
        .is_ok_and(|entity| entity.contains::<C>())
}

fn remove_component<C: Component>(world: &mut World, entity: Entity) {
    if let Ok(mut entity) = world.get_entity_mut(entity) {
        entity.remove::<C>();
    }
}
