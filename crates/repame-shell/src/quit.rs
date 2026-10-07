//! Quitting: the one thing a game cannot do the same way on every platform.
//!
//! `std::process::exit` is `unreachable` on `wasm32-unknown-unknown`: it traps
//! the instance and strands every winit `RefCell` guard, so the next event
//! panics with "RefCell already borrowed". A browser has no window to close
//! and no exit to call, so a quit there latches instead: the shell keeps
//! servicing frames while the app stops stepping and paints [`Quit::parked`],
//! leaving the tab to be closed. A parked root still says so on screen, and
//! the game should cut its audio (`repame_audio::Audio::silence`) so a
//! looping track does not play on over that message.

use repose_core::View;
use repose_core::prelude::Modifier;
use repose_ui::{Center, Text, ViewExt};

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

    /// The root a parked shell paints: one centered line saying the game
    /// closed, so a blank tab is not mistaken for a hung frame.
    pub fn parked() -> View {
        Center(Modifier::new().fill_max_size()).child(Text("The game has been closed."))
    }
}
