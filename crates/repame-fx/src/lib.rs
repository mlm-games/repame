//! Sim-side game feel: CPU particles, trauma shake, flash, floaters, transitions.
//! Timebase: seconds. Every stepper takes `dt_secs: f32` (one fixed sim
//! step's [`SimTime::delta_secs`](repame_sim::SimTime), or a frame delta for
//! ungated effects) and all stored ages/lives are seconds. Deterministic
//! and headless-testable.
//!
//! Legacy `ticks: i32` wrappers assume 100 Hz quanta (`ticks / 100.0` s) and
//! exist for 100 Hz games only; new code must use the `_secs` variants.
//! See `effect` for the spawner model and `particles` for 2D output.
//!
//! Clock ownership: the library owns no clock. Step transitions, trauma,
//! flash, and chroma on an *ungated* clock every frame (wall or raw fixed
//! dt) and read `alpha`/`blocking` from them; gate only gameplay systems
//! and input on [`TransitionFx::blocking`]. Stepping a transition only
//! inside the gated schedule deadlocks it (no ticks run while blocking, so
//! it never unblocks). `repame-shell`'s [`SimDriver`](repame_shell::SimDriver)
//! encodes this split.
//!
//! ```ignore
//! // Per frame, Hydraulic-style wiring:
//! fx_transition.step_secs(raw_dt); // always, even while blocking
//! fx_trauma.decay_secs(raw_dt);
//! let blocked = flow.paused || fx_transition.blocking();
//! if (!blocked) { sim.tick(); } else { clear_edges(); }
//! ```

pub mod chroma;
pub mod decals;
pub mod driver;
pub mod effect;
pub mod flash;
pub mod numbers;
pub mod particles;
pub mod particles3d;
pub mod transitions;
pub mod trauma;

pub use chroma::Chroma;
pub use driver::{FxTick, SimDriver, accumulate_ticks, secs_to_ticks_100hz, ticks_to_secs_100hz};

pub use decals::{
    Decal, DecalDef, blob_shadow, decal_groups, spawn_blob_shadow, spawn_decal, step_decals,
    step_decals_secs,
};
pub use effect::{EaseKind, EffectDef, Gradient, Jittered, SpawnerDef};
pub use flash::Flash;
pub use numbers::{
    DamageNumber, spawn_number, spawn_number_secs, step_numbers, step_numbers_secs,
};
pub use particles::{
    Particle, Spawner, burst, particle_sprites, particle_sprites_with_white, step_particles,
    step_particles_secs, tick_spawners, tick_spawners_secs,
};
pub use particles3d::{
    Particle3, Spawner3, burst3, particle_groups, spawn_particle3, step_particles3,
    step_particles3_secs, tick_spawners3, tick_spawners3_secs,
};
pub use transitions::{
    TransitionFx, TransitionVisual, VORTEX_CUSTOM_ID, secs_to_ticks as transition_secs_to_ticks,
    ticks_to_secs as transition_ticks_to_secs,
};
pub use trauma::Trauma;

use repame_sim::bevy_ecs::prelude::*;

/// Register all fx resources on a `Sim` world. Call once at boot.
pub fn init_resources(world: &mut World) {
    world.init_resource::<Trauma>();
    world.init_resource::<Flash>();
    world.init_resource::<TransitionFx>();
    world.init_resource::<Chroma>();
}
