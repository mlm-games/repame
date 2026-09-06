//! Live vector actors: `.ren` rigs played through `renamite-player`.
//!
//! Actors animate via timelines + state machines (walk/attack/die),
//! driven by the game through host overrides and machine inputs, and
//! painted through the repose Canvas bridge. This replaces the bake-to-
//! sprite pipeline (a bevy-shell workaround) with live playback; `bake`
//! stays as the perf fallback and conformance reference.
//!
//! Repose types cross the renamite boundary here (`View`, `RenderContext`
//! inside player-ui), which proves the version alignment: the workspace
//! `[patch.crates-io]` forces a single repose source graph-wide.

use std::cell::RefCell;
use std::rc::Rc;

pub use renamite_player::{Player, PlayerError};
pub use renamite_player_ui::{PlayerHost, PlayerHostRef};
use repose_core::View;

/// Host a `.ren` rig from source text. The returned handle owns engine +
/// tessellator + playback state; the game ticks it per frame and mounts
/// [`actors_view`] to paint it.
pub fn host_from_str(source: &str) -> Result<PlayerHostRef, PlayerError> {
    Ok(Rc::new(RefCell::new(PlayerHost::from_ren_str(source)?)))
}

/// Paint a hosted rig as a Repose view. Machine inputs (`set_bool`,
/// triggers) and host overrides are applied by game code before this;
/// full per-actor wiring lands with the rozvp pilot.
pub fn actors_view(_host: &PlayerHostRef) -> View {
    todo!("actor view wiring lands with the rozvp pilot")
}
