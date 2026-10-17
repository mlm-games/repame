//! Player settings: one persisted document per app, shared by the shell, the
//! mixer, and the accessibility options.
//!
//! Games used to hand-write `settings.json`, or mix `reduced_motion` into the
//! save, or keep them only in memory. Here they are one document, written
//! through the same atomic store as a save, versioned by the same envelope.
//!
//! # Keys
//!
//! The document is keyed `section.field`, and the keys of the file are the field
//! paths of these structs. They are the stable names an options screen binds to:
//!
//! | key | type | default |
//! |---|---|---|
//! | `audio.master` | `f32` gain | `1.0` |
//! | `audio.music` | `f32` gain | `1.0` |
//! | `audio.sfx` | `f32` gain | `1.0` |
//! | `audio.ui` | `f32` gain | `1.0` |
//! | `audio.muted` | `bool` | `false` |
//! | `display.fullscreen` | `bool` | `false` |
//! | `display.vsync` | `bool` | `true` |
//! | `display.ui_scale` | `f32` | `1.0` |
//! | `accessibility.reduced_motion` | `bool` | `false` |
//! | `accessibility.screen_shake` | `f32` (`0..=1`) | `1.0` |
//! | `accessibility.high_contrast` | `bool` | `false` |
//! | `accessibility.captions` | `bool` | `false` |
//! | `accessibility.text_scale` | `f32` | `1.0` |
//!
//! The document itself lives at `<data dir>/settings.ron`, where the data
//! directory is the app's own ([`AppId`](crate::AppId)), and on wasm under the
//! same `<qualifier>-<org>-<app>` prefix in `ropfs`/localStorage. Settings a
//! game keeps that are not here belong in the game's own document
//! ([`SaveFile`](crate::SaveFile)), not in a side file.
//!
//! Every field is `#[serde(default)]`, so a file written by an older build
//! loads with the new fields at their defaults. A file from a *newer* build is
//! refused by the envelope ([`SaveError::ForwardVersion`](crate::SaveError)),
//! never half-read.

use std::path::{Path, PathBuf};

use game_utils::storage::{FsStorage, Storage};
use serde::{Deserialize, Serialize};

use crate::error::SaveError;
use crate::identity::AppId;
use crate::store::SaveFile;

/// The settings document inside the app's data directory.
pub const SETTINGS_FILE: &str = "settings.ron";

/// The schema this crate writes for [`SETTINGS_FILE`].
///
/// Settings migrate here, in this crate, not in every game: bump this and add
/// the step to [`SettingsStore::new`] when a field's meaning changes, and no
/// game changes at all.
pub const SETTINGS_SCHEMA: u32 = 1;

/// The mixer's linear gains and the master mute.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioSettings {
    /// `audio.master`: master gain, `0.0..=1.0`.
    pub master: f32,
    /// `audio.music`: music gain, `0.0..=1.0`.
    pub music: f32,
    /// `audio.sfx`: effects gain, `0.0..=1.0`.
    pub sfx: f32,
    /// `audio.ui`: interface gain, `0.0..=1.0`.
    pub ui: f32,
    /// `audio.muted`: mute everything, keeping the gains.
    pub muted: bool,
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self {
            master: 1.0,
            music: 1.0,
            sfx: 1.0,
            ui: 1.0,
            muted: false,
        }
    }
}

/// What the window shows and how large the interface is.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DisplaySettings {
    /// `display.fullscreen`.
    pub fullscreen: bool,
    /// `display.vsync`.
    pub vsync: bool,
    /// `display.ui_scale`: interface scale (`1.0` is authored size).
    pub ui_scale: f32,
}

impl Default for DisplaySettings {
    fn default() -> Self {
        Self {
            fullscreen: false,
            vsync: true,
            ui_scale: 1.0,
        }
    }
}

/// Comfort flags: motion, contrast, text size.
///
/// These are first-class fields rather than a key/value bag because games read
/// them on every spawned effect: `reduced_motion` is already a player-facing
/// option in two games on this stack.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AccessibilitySettings {
    /// `accessibility.reduced_motion`: cut transitions, slides, and camera
    /// movement down to their smallest visible step.
    pub reduced_motion: bool,
    /// `accessibility.screen_shake`: shake strength scale, `0.0..=1.0`.
    pub screen_shake: f32,
    /// `accessibility.high_contrast`.
    pub high_contrast: bool,
    /// `accessibility.captions`: show text for sounds that matter to play.
    pub captions: bool,
    /// `accessibility.text_scale`: text size scale (`1.0` is authored size).
    pub text_scale: f32,
}

impl Default for AccessibilitySettings {
    fn default() -> Self {
        Self {
            reduced_motion: false,
            screen_shake: 1.0,
            high_contrast: false,
            captions: false,
            text_scale: 1.0,
        }
    }
}

/// Everything the settings document holds.
///
/// A game-specific preference is not a field here: keep it in the game's own
/// document so this one stays the shell's contract.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// `audio.*`.
    pub audio: AudioSettings,
    /// `display.*`.
    pub display: DisplaySettings,
    /// `accessibility.*`.
    pub accessibility: AccessibilitySettings,
}

/// The settings document of one app.
///
/// ```no_run
/// use repame_save::{AppId, SettingsStore};
///
/// const GAME: AppId = AppId::new("com.mlm-games.pocket-wonders");
///
/// let store = SettingsStore::new(GAME);
/// let mut settings = store.load().expect("readable settings");
/// settings.accessibility.reduced_motion = true;
/// store.save(&settings).expect("writable data dir");
/// ```
#[derive(Clone, Debug)]
pub struct SettingsStore<S: Storage = FsStorage> {
    file: SaveFile<S>,
}

impl SettingsStore<FsStorage> {
    /// This app's settings document.
    pub fn new(app: AppId) -> Self {
        Self::from_file(SaveFile::new(app, SETTINGS_FILE, SETTINGS_SCHEMA))
    }
}

impl<S: Storage> SettingsStore<S> {
    /// A settings document in an explicit file, e.g. an in-memory store in a
    /// test or a document next to a portable build.
    pub fn from_file(file: SaveFile<S>) -> Self {
        Self { file }
    }

    /// The document underneath.
    pub fn file(&self) -> &SaveFile<S> {
        &self.file
    }

    /// The path the settings are read from and written to.
    pub fn path(&self) -> PathBuf {
        self.file.path()
    }

    /// Load the settings.
    ///
    /// No file yet is [`Settings::default`]; a file that does not fit, or one
    /// from a newer build, is an error the caller can report instead of
    /// silently replacing the player's settings with defaults.
    pub fn load(&self) -> Result<Settings, SaveError> {
        self.file.load_or_default::<Settings>()
    }

    /// Write the settings through the atomic store (temp file + rename, with the
    /// previous document kept as a `.bak`).
    pub fn save(&self, settings: &Settings) -> Result<(), SaveError> {
        self.file.save(settings)
    }

    /// Delete the settings, and with them the `.bak`/`temp_` siblings, so the
    /// next load is the defaults again. This is "restore defaults".
    pub fn reset(&self) {
        self.file.delete();
    }

    /// The file the document is written to.
    pub fn dir(&self) -> &Path {
        self.file.dir()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_utils::storage::MemoryStorage;

    fn store() -> SettingsStore<MemoryStorage> {
        SettingsStore::from_file(SaveFile::in_dir_with_storage(
            "settings",
            SETTINGS_FILE,
            SETTINGS_SCHEMA,
            MemoryStorage::new(),
        ))
    }

    #[test]
    fn defaults_are_usable_gains_and_open_accessibility() {
        let settings = Settings::default();
        assert_eq!(settings.audio.master, 1.0);
        assert_eq!(settings.audio.music, 1.0);
        assert!(!settings.audio.muted);
        assert!(settings.display.vsync);
        assert_eq!(settings.display.ui_scale, 1.0);
        assert!(!settings.accessibility.reduced_motion);
        assert_eq!(settings.accessibility.screen_shake, 1.0);
        assert_eq!(settings.accessibility.text_scale, 1.0);
    }

    #[test]
    fn accessibility_flags_round_trip_through_the_store() {
        let store = store();
        assert_eq!(store.load().expect("first run"), Settings::default());

        let mut settings = store.load().expect("first run");
        settings.accessibility.reduced_motion = true;
        settings.accessibility.high_contrast = true;
        settings.accessibility.screen_shake = 0.0;
        settings.accessibility.text_scale = 1.25;
        settings.audio.muted = true;
        store.save(&settings).expect("writes");

        assert_eq!(store.load().expect("reads"), settings);
    }

    #[test]
    fn a_partial_document_keeps_the_defaults_it_does_not_mention() {
        let file = SaveFile::in_dir_with_storage(
            "settings_partial",
            SETTINGS_FILE,
            SETTINGS_SCHEMA,
            MemoryStorage::new(),
        );
        let doc = crate::SaveDoc::from_body(SETTINGS_SCHEMA, "(display: (fullscreen: true,),)");
        file.save_doc(&doc).expect("writes");
        let settings = SettingsStore::from_file(file).load().expect("reads");
        assert!(settings.display.fullscreen);
        assert!(
            settings.display.vsync,
            "unmentioned fields keep their default"
        );
        assert_eq!(settings.audio.master, 1.0);
    }

    #[test]
    fn reset_deletes_the_document() {
        let store = store();
        store.save(&Settings::default()).expect("writes");
        assert!(store.file().exists());
        store.reset();
        assert!(!store.file().exists());
        assert_eq!(store.load().expect("first run again"), Settings::default());
    }
}
