//! Persistent keybinds.
//!
//! Bindings were hand-wired into each game's save. Here they are one document in
//! `repame-input`'s own persisted shape: a list of [`KeymapRow`], one per action,
//! with both sides encoded by `encode_keymap_entry` and read back by
//! `decode_keymap_entry`. There is no second binding format: what a game writes
//! is what `repame-input` parses, and a name it does not know is reported, not
//! dropped silently.
//!
//! An empty document is not an error: it means "the game's own defaults apply",
//! which is exactly what a first run wants.

use std::path::{Path, PathBuf};

use game_utils::storage::{FsStorage, Storage};
use repame_input::{Keymap, KeymapEntry, KeymapRow, decode_keymap_entry, encode_keymap_entry};
use serde::{Deserialize, Serialize};

use crate::error::SaveError;
use crate::identity::AppId;
use crate::store::SaveFile;

/// The keybind document inside the app's data directory.
pub const KEYBINDS_FILE: &str = "input.ron";

/// The schema this crate writes for [`KEYBINDS_FILE`].
pub const KEYBINDS_SCHEMA: u32 = 1;

/// One row for an action, both sides encoded the way `repame-input` writes them.
pub fn keybind_row(
    action: impl Into<String>,
    keyboard: &KeymapEntry,
    gamepad: &KeymapEntry,
) -> KeymapRow {
    KeymapRow {
        action: action.into(),
        keyboard: encode_keymap_entry(keyboard),
        gamepad: encode_keymap_entry(gamepad),
    }
}

/// The persisted bindings, in file order.
///
/// ```rust
/// use repame_input::{Keymap, KeymapEntry};
/// use repame_save::{Keybinds, keybind_row};
///
/// let mut keymap: Keymap<&str> = Keymap::new();
/// keymap.set_keyboard("jump", KeymapEntry::None);
/// let binds = Keybinds::capture(&keymap, &["jump"], |action| action.to_string());
/// assert_eq!(binds.rows()[0].keyboard, "");
///
/// let loaded = Keybinds::from_rows(vec![keybind_row(
///     "jump",
///     &KeymapEntry::None,
///     &KeymapEntry::None,
/// )]);
/// assert_eq!(loaded.rows().len(), 1);
/// ```
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Keybinds {
    rows: Vec<KeymapRow>,
}

impl Keybinds {
    /// No bindings: the game's defaults apply.
    pub fn new() -> Self {
        Self::default()
    }

    /// Wrap rows that were built elsewhere.
    pub fn from_rows(rows: Vec<KeymapRow>) -> Self {
        Self { rows }
    }

    /// The rows, in file order.
    pub fn rows(&self) -> &[KeymapRow] {
        &self.rows
    }

    /// The rows, consumed.
    pub fn into_rows(self) -> Vec<KeymapRow> {
        self.rows
    }

    /// Whether nothing is bound.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Snapshot one row per action of `actions`, in that order.
    ///
    /// The action list is explicit rather than taken from the map's keys so the
    /// document has one row per rebindable action, in a stable order, including
    /// the ones the player has not touched yet. `name` writes an action the way
    /// [`resolve`](Keybinds::apply) will read it back.
    pub fn capture<A, N>(keymap: &Keymap<A>, actions: &[A], name: N) -> Self
    where
        A: Clone + Eq + std::hash::Hash,
        N: Fn(&A) -> String,
    {
        let rows = actions
            .iter()
            .map(|action| {
                keybind_row(
                    name(action),
                    &keymap.keyboard(action),
                    &keymap.gamepad(action),
                )
            })
            .collect();
        Self { rows }
    }

    /// Apply the rows to `keymap`.
    ///
    /// Names are resolved by the game, so its action enum decides what an action
    /// string means. A row whose name `resolve` rejects is skipped; call
    /// [`Keybinds::unknown`] to report those instead of letting a renamed action
    /// disappear quietly.
    pub fn apply<A, R>(&self, keymap: &mut Keymap<A>, resolve: R)
    where
        A: Clone + Eq + std::hash::Hash,
        R: Fn(&str) -> Option<A>,
    {
        for row in &self.rows {
            let Some(action) = resolve(&row.action) else {
                continue;
            };
            keymap.set_keyboard(action.clone(), decode_keymap_entry(&row.keyboard));
            keymap.set_gamepad(action, decode_keymap_entry(&row.gamepad));
        }
    }

    /// A keymap built from the rows, resolving names like [`Keybinds::apply`].
    pub fn to_keymap<A, R>(&self, resolve: R) -> Keymap<A>
    where
        A: Clone + Eq + std::hash::Hash,
        R: Fn(&str) -> Option<A>,
    {
        let mut keymap = Keymap::new();
        self.apply(&mut keymap, resolve);
        keymap
    }

    /// The row names that `known` does not list: actions that were renamed or
    /// removed since the document was written.
    pub fn unknown<'a>(&'a self, known: &[&str]) -> Vec<&'a str> {
        self.rows
            .iter()
            .map(|row| row.action.as_str())
            .filter(|action| !known.contains(action))
            .collect()
    }
}

/// The keybind document of one app.
#[derive(Clone, Debug)]
pub struct KeybindStore<S: Storage = FsStorage> {
    file: SaveFile<S>,
}

impl KeybindStore<FsStorage> {
    /// This app's keybind document.
    pub fn new(app: AppId) -> Self {
        Self::from_file(SaveFile::new(app, KEYBINDS_FILE, KEYBINDS_SCHEMA))
    }
}

impl<S: Storage> KeybindStore<S> {
    /// A keybind document in an explicit file, e.g. an in-memory store in a test.
    pub fn from_file(file: SaveFile<S>) -> Self {
        Self { file }
    }

    /// The document underneath.
    pub fn file(&self) -> &SaveFile<S> {
        &self.file
    }

    /// The path the bindings are read from and written to.
    pub fn path(&self) -> PathBuf {
        self.file.path()
    }

    /// Load the bindings: an empty table when there is no document yet.
    ///
    /// Apply the result over the game's defaults, so a row the document does not
    /// mention keeps its default binding.
    pub fn load(&self) -> Result<Keybinds, SaveError> {
        self.file.load_or_default::<Keybinds>()
    }

    /// Write the bindings through the atomic store.
    pub fn save(&self, binds: &Keybinds) -> Result<(), SaveError> {
        self.file.save(binds)
    }

    /// Delete the document, and with it the `.bak`/`temp_` siblings: the next
    /// load is the game's defaults again. This is "reset bindings".
    pub fn reset(&self) {
        self.file.delete();
    }

    /// The directory the document lives in.
    pub fn dir(&self) -> &Path {
        self.file.dir()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_utils::storage::MemoryStorage;
    use repame_input::{MouseButton, PadButton};

    const ACTIONS: [&str; 2] = ["jump", "fire"];

    fn resolve(name: &str) -> Option<&'static str> {
        ACTIONS.iter().copied().find(|action| *action == name)
    }

    fn keymap() -> Keymap<&'static str> {
        let mut keymap = Keymap::new();
        keymap.set_keyboard("jump", KeymapEntry::Mouse(MouseButton::Primary));
        keymap.set_gamepad("jump", KeymapEntry::Pad(PadButton::South));
        keymap.set_keyboard("fire", KeymapEntry::Mouse(MouseButton::Secondary));
        keymap.set_gamepad("fire", KeymapEntry::Pad(PadButton::East));
        keymap
    }

    fn store() -> KeybindStore<MemoryStorage> {
        KeybindStore::from_file(SaveFile::in_dir_with_storage(
            "keybinds",
            KEYBINDS_FILE,
            KEYBINDS_SCHEMA,
            MemoryStorage::new(),
        ))
    }

    #[test]
    fn captured_rows_carry_both_sides() {
        let binds = Keybinds::capture(&keymap(), &ACTIONS, |action| action.to_string());
        assert_eq!(binds.rows().len(), 2);
        assert_eq!(binds.rows()[0].action, "jump");
        assert_eq!(binds.rows()[0].keyboard, "MouseLeft");
        assert_eq!(binds.rows()[0].gamepad, "PadSouth");
        assert_eq!(binds.rows()[1].action, "fire");
        assert_eq!(binds.rows()[1].gamepad, "PadEast");
    }

    #[test]
    fn bindings_round_trip_through_the_store() {
        let store = store();
        assert!(store.load().expect("first run").is_empty());

        let binds = Keybinds::capture(&keymap(), &ACTIONS, |action| action.to_string());
        store.save(&binds).expect("writes");
        let loaded = store.load().expect("reads");
        assert_eq!(loaded, binds);

        let restored = loaded.to_keymap(resolve);
        let original = keymap();
        for action in ACTIONS {
            assert_eq!(
                restored.keyboard(&action),
                original.keyboard(&action),
                "keyboard {action}"
            );
            assert_eq!(
                restored.gamepad(&action),
                original.gamepad(&action),
                "gamepad {action}"
            );
        }
    }

    #[test]
    fn an_unknown_action_is_reported_not_dropped() {
        let binds = Keybinds::from_rows(vec![
            keybind_row(
                "jump",
                &KeymapEntry::Mouse(MouseButton::Primary),
                &KeymapEntry::None,
            ),
            keybind_row("dash", &KeymapEntry::None, &KeymapEntry::None),
        ]);
        assert_eq!(binds.unknown(&ACTIONS), vec!["dash"]);

        let mut applied: Keymap<&'static str> = Keymap::new();
        binds.apply(&mut applied, resolve);
        let mut names: Vec<&str> = applied.actions().copied().collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names, vec!["jump"], "the unknown row binds nothing");

        let store = store();
        store.save(&binds).expect("writes");
        assert_eq!(
            store.load().expect("reads").unknown(&ACTIONS),
            vec!["dash"],
            "the unknown row survives the round trip so the player can see it"
        );
    }

    #[test]
    fn reset_returns_the_game_defaults() {
        let store = store();
        store
            .save(&Keybinds::capture(&keymap(), &ACTIONS, |action| {
                action.to_string()
            }))
            .expect("writes");
        store.reset();
        assert!(store.load().expect("reads").is_empty());
    }

    #[test]
    fn a_hand_written_document_loads() {
        // The file a game writes is a RON list of the rows `repame-input` names.
        let file = SaveFile::in_dir_with_storage(
            "keybinds_hand_written",
            KEYBINDS_FILE,
            KEYBINDS_SCHEMA,
            MemoryStorage::new(),
        );
        let doc = crate::SaveDoc::from_body(
            KEYBINDS_SCHEMA,
            r#"[
    (action: "jump", keyboard: "Space", gamepad: "PadSouth"),
]"#,
        );
        file.save_doc(&doc).expect("writes");
        let binds = KeybindStore::from_file(file).load().expect("reads");
        assert_eq!(binds.rows().len(), 1);
        assert_eq!(binds.rows()[0].keyboard, "Space");
        assert_eq!(
            decode_keymap_entry(&binds.rows()[0].gamepad),
            KeymapEntry::Pad(PadButton::South)
        );
    }
}
