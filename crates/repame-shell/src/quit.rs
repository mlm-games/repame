//! Quitting: the one thing a game cannot do the same way on every platform.
//!
//! `std::process::exit` is `unreachable` on `wasm32-unknown-unknown`: it traps
//! the instance and strands every winit `RefCell` guard, so the next event
//! panics with "RefCell already borrowed". A browser has no window to close
//! and no exit to call, so a quit there latches instead: the shell keeps
//! servicing frames while the app stops stepping and paints [`Quit::parked`],
//! leaving the tab to be closed.

use repose_core::View;
use repose_core::prelude::Modifier;
use repose_ui::ZStack;

/// A quit request, latched so a browser shell can park instead of exiting.
#[derive(Default)]
pub struct Quit {
    latched: bool,
}

impl Quit {
    /// Ask to quit. Does not return off the web, where the process exits
    /// here.
    pub fn request(&mut self) {
        self.latched = true;
        #[cfg(not(target_arch = "wasm32"))]
        std::process::exit(0);
    }

    /// Whether the quit is latched, and the app is parked.
    pub fn latched(&self) -> bool {
        self.latched
    }

    /// The root a parked shell paints: empty, so nothing is drawn.
    pub fn parked() -> View {
        ZStack(Modifier::new().fill_max_size())
    }
}