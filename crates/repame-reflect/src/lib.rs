//! ECS reflection for repame: a component registry, a RON world snapshot, and
//! an inspectable component tree.
//!
//! Games register the component types they want to persist or expose with
//! [`Registry::register`], giving each an explicit, stable name. From there:
//!
//! - [`save_world`] / [`load_world`] round-trip the registered components of a
//!   [`World`] through a RON [`WorldSnapshot`].
//! - [`entities`], [`inspect`], [`has`], [`value`], [`set`] and [`remove`] are
//!   the reflection surface a debug panel binds to. They return plain data
//!   ([`ComponentView`], [`serde_json::Value`]) and depend on no renderer.
//!
//! This crate is renderer-free: it depends on `bevy_ecs` and serde only, so a
//! Repose panel renders the tree while the sim owns it. It adds no dependency
//! beyond the published workspace set.
//!
//! # Data model
//!
//! A component's fields are carried as a [`serde_json::Value`] tree; the
//! document that holds those values is RON. The component type itself only has
//! to be `Component + Serialize + DeserializeOwned + Clone`, and it decides
//! its own field shape, so an enum field, a nested struct, and a `Vec` all
//! round-trip without extra wiring.
//!
//! # Determinism
//!
//! [`save_world`] is a pure function of the registered set and the stored
//! component values: entities are emitted in ascending [`Entity::to_bits`]
//! order and each entity's components in name order (a `BTreeMap`), so two
//! saves of an unchanged world are byte-identical. `serde_json`'s default
//! object map is ordered too, so nested objects do not depend on insertion.
//! Loading then saving is stable in content, though not in entity ids: the
//! entity bits of the target world are whatever `spawn_empty` hands out, and
//! [`LoadReport`] is how a caller recovers the mapping.
//!
//! One fidelity limit is inherited from the value model: non-finite floats
//! (`NaN`, infinities) are not representable as JSON numbers and become
//! `null`. A component that must preserve them is out of scope for this format.
//!
//! # Failures
//!
//! Every failure is a [`ReflectError`]. In particular, a document naming a
//! component that is not registered is [`ReflectError::UnknownType`] and is
//! rejected before anything is spawned, never skipped.

mod error;
mod inspector;
mod registry;
mod snapshot;

pub use error::ReflectError;
pub use inspector::{ComponentView, entities, has, inspect, remove, set, value};
pub use registry::Registry;
pub use snapshot::{
    EntitySnapshot, LoadReport, WORLD_FORMAT, WorldSnapshot, capture_world, load_world, save_world,
};

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::prelude::{Component, World};
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    #[derive(Component, Clone, Debug, PartialEq, Serialize, Deserialize)]
    struct Health {
        current: i32,
        max: i32,
    }

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    struct Position {
        x: f32,
        y: f32,
    }

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    enum Stance {
        Idle,
        Walk { speed: f32 },
    }

    #[derive(Component, Clone, Debug, PartialEq, Serialize, Deserialize)]
    struct Agent {
        name: String,
        stance: Stance,
        home: Position,
        path: Vec<[i32; 2]>,
    }

    fn registry() -> Registry {
        let mut registry = Registry::new();
        registry.register::<Health>("game::Health").unwrap();
        registry.register::<Agent>("game::Agent").unwrap();
        registry
    }

    fn agent() -> Agent {
        Agent {
            name: "scout".to_string(),
            stance: Stance::Walk { speed: 1.5 },
            home: Position { x: 1.0, y: -2.0 },
            path: vec![[0, 0], [4, 7]],
        }
    }

    #[test]
    fn snapshot_round_trip_is_byte_stable() {
        let registry = registry();
        let mut world = World::new();
        let a = world
            .spawn((Health { current: 3, max: 10 }, agent()))
            .id();
        let b = world.spawn(Health { current: 10, max: 10 }).id();
        world.spawn(());

        let first = save_world(&world, &registry).unwrap();
        let second = save_world(&world, &registry).unwrap();
        assert_eq!(first, second, "identical worlds serialize to identical bytes");
        assert!(!first.is_empty(), "the document is not empty");

        let mut target = World::new();
        let report = load_world(&mut target, &registry, &first).unwrap();
        assert_eq!(report.entities.len(), 2, "the component-less entity is skipped");
        assert_eq!(report.components, 3);
        assert!(report.entity(u64::MAX).is_none());

        let loaded_a = report.entity(a.to_bits()).unwrap();
        assert_eq!(
            target.get::<Health>(loaded_a).unwrap(),
            &Health { current: 3, max: 10 }
        );
        assert_eq!(target.get::<Agent>(loaded_a).unwrap(), &agent());
        let loaded_b = report.entity(b.to_bits()).unwrap();
        assert_eq!(
            target.get::<Health>(loaded_b).unwrap(),
            &Health { current: 10, max: 10 }
        );
        assert!(target.get::<Agent>(loaded_b).is_none());
    }

    #[test]
    fn loading_an_unregistered_component_is_a_typed_error() {
        let registry = registry();
        let mut world = World::new();
        world.spawn(Health { current: 1, max: 2 });
        let text = save_world(&world, &registry).unwrap();
        let tampered = text.replace("game::Health", "game::Unknown");

        let mut target = World::new();
        let error = load_world(&mut target, &registry, &tampered).unwrap_err();
        assert!(
            matches!(error, ReflectError::UnknownType { ref name } if name == "game::Unknown"),
            "got {error:?}"
        );
        assert!(
            entities(&target, &registry).is_empty(),
            "an unregistered name spawns nothing"
        );
    }

    #[test]
    fn inspector_lists_registered_components_by_name() {
        let registry = registry();
        let mut world = World::new();
        let a = world
            .spawn((Health { current: 3, max: 10 }, agent()))
            .id();
        let b = world.spawn(Health { current: 1, max: 1 }).id();

        assert_eq!(entities(&world, &registry), vec![a, b]);

        let views = inspect(&world, &registry, a);
        let names: Vec<&str> = views.iter().map(|view| view.name.as_str()).collect();
        assert_eq!(names, vec!["game::Agent", "game::Health"]);
        assert_eq!(views[0].value["name"], json!("scout"));
        assert_eq!(views[0].value["stance"], json!({ "Walk": { "speed": 1.5 } }));
        assert_eq!(views[0].value["path"], json!([[0, 0], [4, 7]]));

        assert!(has(&world, &registry, a, "game::Agent"));
        assert!(!has(&world, &registry, b, "game::Agent"));
        assert!(!has(&world, &registry, a, "game::Missing"));
    }

    #[test]
    fn set_replaces_a_value_and_reports_typed_errors() {
        let registry = registry();
        let mut world = World::new();
        let entity = world.spawn(Health { current: 1, max: 5 }).id();

        set(
            &mut world,
            &registry,
            entity,
            "game::Health",
            json!({ "current": 4, "max": 9 }),
        )
        .unwrap();
        assert_eq!(
            world.get::<Health>(entity).unwrap(),
            &Health { current: 4, max: 9 }
        );

        let error = set(
            &mut world,
            &registry,
            entity,
            "game::Health",
            json!({ "current": "many", "max": 9 }),
        )
        .unwrap_err();
        assert!(matches!(error, ReflectError::BadValue { .. }), "got {error:?}");

        let error = set(&mut world, &registry, entity, "game::Nope", serde_json::Value::Null)
            .unwrap_err();
        assert!(
            matches!(error, ReflectError::NotRegistered { ref name } if name == "game::Nope"),
            "got {error:?}"
        );
    }

    #[test]
    fn version_mismatch_is_a_typed_error() {
        let registry = registry();
        let mut world = World::new();
        let error = load_world(&mut world, &registry, "(format: 99, entities: [])").unwrap_err();
        assert!(
            matches!(
                error,
                ReflectError::Version {
                    found: 99,
                    expected: WORLD_FORMAT
                }
            ),
            "got {error:?}"
        );
    }
}
