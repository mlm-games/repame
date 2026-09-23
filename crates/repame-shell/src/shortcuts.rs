use std::cell::RefCell;
use std::rc::Rc;

use repose_core::input::{Key, Modifiers};
use repose_core::shortcuts::{self, Action};

pub fn game_shortcut_map(
    pause: &'static str,
    restart: &'static str,
    confirm: &'static str,
) -> shortcuts::ShortcutMap {
    shortcuts::ShortcutMap::new()
        .bind(Key::Escape, Modifiers::default(), action_for(pause))
        .bind(Key::Character('r'), Modifiers::default(), action_for(restart))
        .bind(Key::Enter, Modifiers::default(), action_for(confirm))
}

fn action_for(name: &'static str) -> Action {
    Action::Custom(name.into())
}

#[derive(Default)]
pub struct ShortcutEdges {
    pub pause: bool,
    pub restart: bool,
    pub confirm: bool,
}

pub type SharedEdges = Rc<RefCell<ShortcutEdges>>;

pub fn shared_edges() -> SharedEdges {
    Rc::new(RefCell::new(ShortcutEdges::default()))
}

pub fn shortcut_handler(
    edges: &SharedEdges,
    pause: &'static str,
    restart: &'static str,
    confirm: &'static str,
) -> shortcuts::Handler {
    let inner = edges.clone();
    Rc::new(move |action| {
        let mut e = inner.borrow_mut();
        match action {
            Action::Custom(key) if key.as_ref() == pause => {
                e.pause = true;
                true
            }
            Action::Custom(key) if key.as_ref() == restart => {
                e.restart = true;
                true
            }
            Action::Custom(key) if key.as_ref() == confirm => {
                e.confirm = true;
                true
            }
            _ => false,
        }
    })
}

/// Compose the game map + edges handler once in the root view
/// (mount-once under the hood: safe to call every frame on desktop,
/// web, and Android-with-keyboard alike). One call per game; no
/// per-key duplication at call sites.
pub fn install_game_shortcuts(
    edges: &SharedEdges,
    pause: &'static str,
    restart: &'static str,
    confirm: &'static str,
) {
    let _ = shortcuts::InstallShortcutMap(game_shortcut_map(pause, restart, confirm));
    let _ = shortcuts::InstallShortcutHandler(shortcut_handler(edges, pause, restart, confirm));
}

/// Overwrite the process-global default shortcut map (tests and headless
/// compose without a runner use this path).
pub fn install_map(map: shortcuts::ShortcutMap) {
    shortcuts::set_default_map(map);
}

/// Install only the action handler into long-lived `edges`
/// (tests and headless compose without a runner use this path).
pub fn install_handler_into(edges: &SharedEdges, pause: &'static str, restart: &'static str, confirm: &'static str) {
    shortcuts::set(Some(shortcut_handler(edges, pause, restart, confirm)));
}

pub fn take(edges: &SharedEdges) -> (bool, bool, bool) {
    let mut e = edges.borrow_mut();
    (
        std::mem::take(&mut e.pause),
        std::mem::take(&mut e.restart),
        std::mem::take(&mut e.confirm),
    )
}
