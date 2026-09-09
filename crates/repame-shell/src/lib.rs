//! App wiring: Repose platform runner + sim stepping + viewport mount.
//!
//! A game provides a root view and a [`ShellHooks`] implementation; the
//! shell owns the frame loop glue (desktop runner, fixed-step sim advance,
//! gamepad polling). Per-game UI stays in the game crate as Repose views.

use std::collections::HashSet;
use web_time::Duration;

use anyhow::Result;
pub use repame_sim::{Sim, SimTime};
pub use repame_sprite::{Camera2d, FrameInput, PickEvent, SpriteInstance};
pub use repose_core::input::{GamepadEvent, GamepadId};
use repose_platform::gamepad::{GamepadBackend, create_backend};

/// Per-frame hooks the game implements. Snapshot production (`frame_input`)
/// is the only required render coupling: plain data out, no renderer types.
pub trait ShellHooks {
    /// Build this frame's viewport snapshot from sim + UI state.
    fn frame_input(&mut self, sim: &Sim) -> FrameInput;
    /// Fixed-step systems run inside [`Sim`]; this is the per-frame
    /// variable-rate hook (tweens, audio triggers, autosave timers).
    fn on_frame(&mut self, _sim: &mut Sim, _dt: Duration) {}
}

/// Desktop entry point. Mounts `root` on the repose-platform runner with
/// the given window title/size. The root closure owns stepping (sim,
/// rigs) and must call `request_frame()` for continuous frames; see
/// the rozvp pilot runner for the reference wiring.
#[cfg(all(not(target_os = "android"), not(target_arch = "wasm32")))]
pub fn run_desktop(
    title: &str,
    size: (u32, u32),
    root: impl FnMut(
        &mut repose_core::runtime::Scheduler,
        &repose_core::RenderContext,
    ) -> repose_core::View
    + 'static,
) -> Result<()> {
    let config = repose_app::AppConfig {
        window_title: title.to_string(),
        window_size: size,
        ..Default::default()
    };
    repose_platform::run_desktop_app_with_config(root, config)?;
    Ok(())
}

/// Web entry point. Mounts `root` on the repose-platform web runner with
/// default options (prevent-default on, default canvas).
#[cfg(target_arch = "wasm32")]
pub fn run_web(
    root: impl FnMut(
        &mut repose_core::runtime::Scheduler,
        &repose_core::RenderContext,
    ) -> repose_core::View
    + 'static,
) -> Result<(), wasm_bindgen::JsValue> {
    let mut options = repose_platform::web::WebOptions::new(None);
    options.set_prevent_default(true);
    repose_platform::web::run_web_app(root, options)
}

/// Android entry point. Mounts `root` on the repose-platform android
/// runner with default options. Logging + window insets stay app-side.
#[cfg(target_os = "android")]
pub fn run_android(
    app: winit::platform::android::activity::AndroidApp,
    root: impl FnMut(
        &mut repose_core::runtime::Scheduler,
        &repose_core::RenderContext,
    ) -> repose_core::View
    + 'static,
) -> Result<()> {
    repose_platform::android::run_android_app_with_options(
        app,
        root,
        repose_app::AndroidOptions::default(),
    )
}

/// Gamepad polling unified on the shared `repose-platform` backend: one
/// button layout, one deadzone, one mapping policy for repose apps and
/// repame games alike. Events feed straight into
/// `ReposeRuntime::handle_gamepad` (UI nav) and `rt.gamepads` (gameplay).
pub struct GamepadPoller {
    backend: Option<Box<dyn GamepadBackend>>,
    connected: HashSet<u32>,
}

impl GamepadPoller {
    pub fn new() -> Self {
        Self {
            backend: create_backend().map(|b| Box::new(b) as Box<dyn GamepadBackend>),
            connected: HashSet::new(),
        }
    }

    /// Drain hardware events since the last call. Tracks connection state
    /// so [`GamepadPoller::connected_ids`] works even for pads connected
    /// before startup: gilrs only emits `Connected` on its own schedule,
    /// so any button/axis event from an unknown id implicitly marks that
    /// pad connected (first input at the latest).
    pub fn poll(&mut self) -> Vec<GamepadEvent> {
        let Some(backend) = &mut self.backend else {
            return Vec::new();
        };
        let events = backend.poll();
        for ev in &events {
            match ev {
                GamepadEvent::Connected { id, .. } => {
                    self.connected.insert(id.0);
                }
                GamepadEvent::Disconnected { id } => {
                    self.connected.remove(&id.0);
                }
                GamepadEvent::Button { id, .. } | GamepadEvent::Axis { id, .. } => {
                    self.connected.insert(id.0);
                }
            }
        }
        events
    }

    pub fn connected_ids(&self) -> Vec<GamepadId> {
        self.connected.iter().copied().map(GamepadId).collect()
    }

    pub fn connected_count(&self) -> usize {
        self.connected.len()
    }
}

impl Default for GamepadPoller {
    fn default() -> Self {
        Self::new()
    }
}
