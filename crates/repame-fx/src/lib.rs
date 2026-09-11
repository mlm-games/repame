//! Sim-side game feel for all repame games: CPU particles, trauma
//! shake, fullscreen flash, damage floaters, and state transitions.
//!
//! Design notes (all bevy-bound or GPU-only, so referenced, not used as the shell is different and this is mainly for dogfooding repose with (translated or my) bevy games that already work):
//! - `bevy_hanabi`: init/update/render modifier chain, spawner settings,
//!   runtime effect properties, GPU spawn events. Here that becomes
//!   [`EffectDef`] (init/update/render) + [`Spawner`] runtime state, all
//!   fixed-tick CPU, canvas sprites out.
//! - `bevy_enoki`: RON effect assets, spawner/state/instance split,
//!   CPU-sim + instancing that works on wasm/mobile. `EffectDef` derives
//!   serde so `.fx.ron` files work later; rendering stays instancing-ready
//!   plain [`repame_sprite::SpriteInstance`] lists.
//! - `bevy_particle_systems`: `JitteredValue` (here [`Jittered`]) and
//!   color-over-lifetime gradients (here [`Gradient`]).
//! - Bevy's `2d_screen_shake` example: trauma 0..1, decay, squared
//!   response, Perlin-noise offsets, shake applied to rendering only.
//! - Easing curves come from `easer` (Penner), noise from `noise`
//!   (both bevy-free, MIT/Apache).
//!
//! Everything steps on integer 100 Hz ticks like the sim: deterministic,
//! headless-testable, identical on desktop/web/mobile.

pub mod chroma;
pub mod effect;
pub mod flash;
pub mod numbers;
pub mod particles;
pub mod transitions;
pub mod trauma;

pub use chroma::Chroma;

pub use effect::{EaseKind, EffectDef, Gradient, Jittered, SpawnerDef};
pub use flash::Flash;
pub use numbers::{DamageNumber, spawn_number, step_numbers};
pub use particles::{Particle, Spawner, burst, particle_sprites, particle_sprites_with_white, step_particles, tick_spawners};
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
