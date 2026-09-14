//! Sim-side game feel: CPU particles, trauma shake, flash, floaters, transitions.
//! Steps on integer 100 Hz ticks. Deterministic and headless-testable.
//! See `effect` for the spawner model and `particles` for 2D output.

pub mod chroma;
pub mod decals;
pub mod effect;
pub mod flash;
pub mod numbers;
pub mod particles;
pub mod particles3d;
pub mod transitions;
pub mod trauma;

pub use chroma::Chroma;

pub use decals::{
    Decal, DecalDef, blob_shadow, decal_groups, spawn_blob_shadow, spawn_decal, step_decals,
};
pub use effect::{EaseKind, EffectDef, Gradient, Jittered, SpawnerDef};
pub use flash::Flash;
pub use numbers::{DamageNumber, spawn_number, step_numbers};
pub use particles::{
    Particle, Spawner, burst, particle_sprites, particle_sprites_with_white, step_particles,
    tick_spawners,
};
pub use particles3d::{
    Particle3, Spawner3, burst3, particle_groups, spawn_particle3, step_particles3, tick_spawners3,
};
pub use transitions::{TransitionFx, TransitionVisual, VORTEX_CUSTOM_ID};
pub use trauma::Trauma;

use repame_sim::bevy_ecs::prelude::*;

/// Register all fx resources on a `Sim` world. Call once at boot.
pub fn init_resources(world: &mut World) {
    world.init_resource::<Trauma>();
    world.init_resource::<Flash>();
    world.init_resource::<TransitionFx>();
    world.init_resource::<Chroma>();
}
