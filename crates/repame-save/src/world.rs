//! The sim bridge: sim state as an ordinary part of a save.
//!
//! Sim state does not need a hand-written DTO. Register the components a game
//! wants to keep with [`repame_reflect::Registry`], capture them with
//! [`WorldDoc::capture`], and the RON world document travels inside the save
//! payload verbatim; [`WorldDoc::restore`] spawns the entities back and returns
//! the [`LoadReport`] that maps the document's ids onto the new ones.
//!
//! ```no_run
//! use repame_reflect::Registry;
//! use repame_save::{SaveFile, WorldDoc, AppId};
//! use repame_sim::Sim;
//! use serde::{Deserialize, Serialize};
//!
//! const GAME: AppId = AppId::new("com.mlm-games.gravity-riot");
//!
//! #[derive(Default, Serialize, Deserialize)]
//! struct SaveData {
//!     best_score: i32,
//!     world: WorldDoc,
//! }
//!
//! let registry = Registry::new();
//! let mut sim = Sim::with_default_step();
//! let save: SaveFile = SaveFile::new(GAME, "save.ron", 1);
//!
//! let mut data: SaveData = save.load_or_default().expect("readable save");
//! data.world.restore_sim(&mut sim, &registry).expect("world document");
//!
//! data.world = WorldDoc::capture_sim(&sim, &registry).expect("world document");
//! save.save(&data).expect("writable data dir");
//! ```

use repame_reflect::{LoadReport, ReflectError, Registry, load_world, save_world};
use repame_sim::Sim;
use repame_sim::bevy_ecs::prelude::World;
use serde::{Deserialize, Serialize};

/// `repame-reflect`'s world document, carried verbatim.
///
/// It is a plain section of a save payload, so a game that keeps both its own
/// fields and sim state writes one document:
///
/// ```ron
/// (
///     best_score: 1200,
///     world: "(\n    format: 1,\n    entities: [],\n)",
/// )
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WorldDoc(String);

impl WorldDoc {
    /// Capture every registered component of `world`.
    ///
    /// A component that fails to serialize is an error, not a dropped field,
    /// and entities carrying no registered component are omitted.
    pub fn capture(world: &World, registry: &Registry) -> Result<Self, ReflectError> {
        save_world(world, registry).map(Self)
    }

    /// Capture every registered component of a [`Sim`]'s world.
    pub fn capture_sim(sim: &Sim, registry: &Registry) -> Result<Self, ReflectError> {
        Self::capture(&sim.world, registry)
    }

    /// Spawn the document's entities into `world`.
    ///
    /// An empty document means "this save has no sim section yet" and restores
    /// nothing, rather than failing on a document that was never written. A
    /// document naming a component the registry does not know is
    /// [`ReflectError::UnknownType`](repame_reflect::ReflectError) before
    /// anything is spawned, so a save from a build with an extra component is
    /// refused whole.
    pub fn restore(
        &self,
        world: &mut World,
        registry: &Registry,
    ) -> Result<LoadReport, ReflectError> {
        if self.is_empty() {
            return Ok(LoadReport::default());
        }
        load_world(world, registry, &self.0)
    }

    /// Spawn the document's entities into a [`Sim`]'s world.
    pub fn restore_sim(
        &self,
        sim: &mut Sim,
        registry: &Registry,
    ) -> Result<LoadReport, ReflectError> {
        self.restore(&mut sim.world, registry)
    }

    /// Wrap a document that already exists, e.g. one imported from a build's
    /// export.
    pub fn from_text(text: impl Into<String>) -> Self {
        Self(text.into())
    }

    /// The RON world document.
    pub fn text(&self) -> &str {
        &self.0
    }

    /// Whether nothing was captured: the empty section a fresh save carries.
    pub fn is_empty(&self) -> bool {
        self.0.trim().is_empty()
    }
}

impl From<String> for WorldDoc {
    fn from(text: String) -> Self {
        Self(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // The `Component` derive writes `bevy_ecs::` paths, and this crate reaches
    // the ECS through `repame-sim`'s re-export rather than a direct dependency.
    use repame_sim::bevy_ecs;
    use repame_sim::bevy_ecs::prelude::Component;

    #[derive(Component, Clone, Debug, PartialEq, Serialize, Deserialize)]
    struct Score(u32);

    fn registry() -> Registry {
        let mut registry = Registry::new();
        registry
            .register::<Score>("game::Score")
            .expect("registers");
        registry
    }

    #[test]
    fn sim_state_round_trips_without_a_dto() {
        let registry = registry();
        let mut sim = Sim::with_default_step();
        sim.world.spawn(Score(7));

        let doc = WorldDoc::capture_sim(&sim, &registry).expect("captures");
        assert!(!doc.is_empty());

        let mut target = Sim::with_default_step();
        let report = doc.restore_sim(&mut target, &registry).expect("restores");
        assert_eq!(report.components, 1);
        let entity = *report.entities.values().next().expect("one entity");
        assert_eq!(target.world.get::<Score>(entity), Some(&Score(7)));
    }

    #[test]
    fn an_empty_document_restores_nothing() {
        let registry = registry();
        let mut world = World::new();
        let report = WorldDoc::default()
            .restore(&mut world, &registry)
            .expect("nothing");
        assert_eq!(report.components, 0);
        assert!(report.entities.is_empty());
        assert!(WorldDoc::from_text("").is_empty());
        assert!(WorldDoc::from_text("   \n").is_empty());
    }

    #[test]
    fn a_world_document_is_a_plain_payload_field() {
        let registry = registry();
        let mut sim = Sim::with_default_step();
        sim.world.spawn(Score(1));
        let doc = WorldDoc::capture_sim(&sim, &registry).expect("captures");

        #[derive(Debug, Default, Serialize, Deserialize, PartialEq)]
        struct SaveData {
            best_score: i32,
            world: WorldDoc,
        }

        let data = SaveData {
            best_score: 1200,
            world: doc,
        };
        let document = crate::SaveDoc::new(1, &data).expect("serializes");
        let parsed = crate::SaveDoc::parse(&document.to_text().expect("writes")).expect("parses");
        assert_eq!(parsed.body_as::<SaveData>().expect("payload"), data);
    }
}
