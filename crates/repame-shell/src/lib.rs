//! App wiring: Repose platform runners plus sim and gamepad helpers.
/// Games provide the root view; `run_*` runners mount only, stepping
/// stays game-side via [`Sim::step`].
use std::collections::HashSet;
use web_time::Duration;

use anyhow::Result;
pub use repame_sim::{Sim, SimTime};
pub use repame_sprite::{Camera2d, FrameInput, PickEvent, SpriteInstance};
pub use repose_core::input::{GamepadEvent, GamepadId};
use repose_platform::gamepad::{GamepadBackend, create_backend};

/// Per-frame hooks the game implements.
pub trait ShellHooks {
    /// Build this frame's viewport snapshot from sim and UI state.
    fn frame_input(&mut self, sim: &Sim) -> FrameInput;
    /// Variable-rate hook (tweens, audio, autosave timers).
    fn on_frame(&mut self, _sim: &mut Sim, _dt: Duration) {}
}

/// Desktop entry point. Mounts `root` with title and size.
/// The root closure owns stepping and requests frames for continuity.
/// The closure receives the live [`Scheduler`] each frame, whose
/// `held_keys` / `window_focused` / `mouse_*` fields are the single
/// polled hardware snapshot (GML `keyboard_check` /
/// `mouse_check_button` parity).
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

/// Web entry point. Mounts `root` with default options.
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

/// Android entry point. Mounts `root` with default options.
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

/// Gamepad polling on the shared platform backend. Events feed
/// `ReposeRuntime::handle_gamepad` (UI nav) and `rt.gamepads` (gameplay).
pub struct GamepadPoller {
    backend: Option<Box<dyn GamepadBackend>>,
    connected: HashSet<u32>,
}

impl GamepadPoller {
    pub fn new() -> Self {
        Self {
            backend: create_backend().map(|b| Box::new(b) as Box<dyn GamepadBackend>),
            // Boot-plugged pads report on first poll; no press needed.
            connected: HashSet::new(),
        }
    }

    /// Drain hardware events since the last call. Tracks connection state
    /// so `connected_ids` stays live.
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
        let mut ids: Vec<GamepadId> = self.connected.iter().copied().map(GamepadId).collect();
        ids.sort_by_key(|id| id.0);
        ids
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
