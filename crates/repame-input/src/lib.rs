//! Physical inputs (keys, mouse buttons, pad buttons, axes) bound to
//! game actions. (using leafwing `InputMap`/`ActionState` as ref., flattened for
//! fixed-tick CPU sims on raw `bevy_ecs` + `repose` input types.)

use std::collections::HashMap;
use std::hash::Hash;

use repame_sim::bevy_ecs::prelude::*;
use repose_core::input::GamepadAxis;

mod binding;
mod map;
mod state;

pub use binding::Binding;
pub use map::ActionMap;
pub use state::ActionState;

/// Action key bound. Games use their own enums (must be `Clone + Eq +
/// `Hash`); a `&'static str` works for quick wiring.
pub trait ActionLike: Clone + Eq + Hash + Send + Sync + 'static {}

impl<T: Clone + Eq + Hash + Send + Sync + 'static> ActionLike for T {}

/// Insert an [`ActionState`] built from `map` into the sim world.
pub fn init_state<A: ActionLike + std::fmt::Debug>(
    sim: &mut repame_sim::Sim,
    map: ActionMap<A>,
) {
    sim.world.insert_resource(ActionState::new(map));
}

/// Fixed-step edge rollover: register first in the sim schedule so
/// `just_pressed`/`just_released` live exactly one tick.
pub fn begin_tick_system<A: ActionLike + std::fmt::Debug>(
    mut state: ResMut<ActionState<A>>,
) {
    state.begin_tick();
}

/// Readout of one polled axis pair (left stick by default).
pub fn stick_pair(state_axes: &HashMap<GamepadAxis, f32>) -> (f32, f32) {
    let x = state_axes.get(&GamepadAxis::LeftStickX).copied().unwrap_or(0.0);
    let y = state_axes.get(&GamepadAxis::LeftStickY).copied().unwrap_or(0.0);
    (x, y)
}

pub use repose_core::input::{GamepadAxis as Axis, GamepadButton as PadButton};
pub use repose_core::shortcuts::KeyChord as Chord;
pub use repose_core::input::PointerButton as MouseButton;
