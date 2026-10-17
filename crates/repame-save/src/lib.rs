//! One app identity, one versioned save envelope, and one place for settings,
//! keybinds, and sim state.
//!
//! Games on this stack each carried a `save.rs` of their own: the same manager,
//! the same store, the same version stamp, copied per game. Six of them passed
//! the same data-directory identity, so they shared a directory and overwrote
//! each other's files. Schema migration was reinvented per game, and a save from
//! a newer build was warned about and then parsed anyway. This crate is the
//! shared answer to all of it.
//!
//! # What is here
//!
//! - [`AppId`]: the game's reverse-DNS identity, declared once as a `const`.
//!   It resolves the per-platform data directory through `game-utils`, so no game
//!   hardcodes a shared `"WebGame"` again.
//! - [`SaveDoc`] / [`SaveFile`]: a versioned envelope around the game's own
//!   document, with a [`MigrationChain`] and an explicit refusal of a file
//!   stamped newer than the binary.
//! - [`Settings`] / [`SettingsStore`]: persisted preferences with first-class
//!   accessibility flags.
//! - [`Keybinds`] / [`KeybindStore`]: bindings in `repame-input`'s existing
//!   [`KeymapRow`](repame_input::KeymapRow) shape, encoded by its
//!   `encode_keymap_entry`/`decode_keymap_entry`.
//! - [`WorldDoc`]: the sim half of a save, joining `repame-reflect`'s
//!   `save_world`/`load_world` so sim state needs no hand-written DTO.
//!
//! # What is reused
//!
//! Nothing here is a second implementation of anything `game-utils` already
//! ships:
//!
//! - `game_utils::save_store::SaveStore` (re-exported as [`SaveStore`]) does
//!   every byte of the file work: the temp file plus atomic rename, the `.bak`
//!   rotation, the corruption quarantine, and the recovery from the last good
//!   backup. This crate supplies the document validator and the envelope rules.
//! - `game_utils::storage::{FsStorage, Storage}` (re-exported as [`FsStorage`],
//!   [`Storage`]) is the backend, so the same code runs on the filesystem, on
//!   wasm through `ropfs`/localStorage, on Android through the runtime dir the
//!   game sets, and on an in-memory backend in tests ([`MemoryStorage`]).
//! - `game_utils::save::SaveManager` (re-exported as [`SaveManager`]) resolves
//!   the platform data directory behind [`AppId::data_dir`], and
//!   [`AppId::manager`] hands one out so code that already speaks `SaveManager`
//!   keeps working.
//! - `game_utils::save::Versioned` (re-exported as [`Versioned`]) keeps working
//!   through [`SaveFile::load_versioned`] and [`SaveFile::save_versioned`] for
//!   payloads that stamp their own version field. New code uses
//!   [`MigrationChain`], which rewrites the payload before the type is known.
//!
//! # The envelope
//!
//! A save, a settings document, and a keybind document are all the same file
//! shape:
//!
//! ```ron
//! (
//!     schema: 3,
//!     body: r#"(
//!     best_score: 1200,
//!     muted: false,
//! )"#,
//! )
//! ```
//!
//! `schema` is the number this binary knows. On load:
//!
//! - an older stamp runs the [`MigrationChain`] one step per version, with no
//!   version skipped ([`SaveError::MissingMigration`] names the gap);
//! - an equal stamp is used as it stands;
//! - a newer stamp is [`SaveError::ForwardVersion`]. The file is refused whole,
//!   because parsing it with an older type would drop the fields the newer build
//!   wrote, and the next save would make that loss permanent.
//!
//! A corrupt or truncated file is quarantined and the previous `.bak` is adopted
//! and written back, so a player loses one save's worth of progress at worst;
//! the [`Loaded`] status says that it happened.
//!
//! Adopting a save file a game already shipped is the game's own step: read the
//! old format with the code that knows it, build a [`SaveDoc`] at the current
//! schema, and write it through [`SaveFile::save_doc`]. The store never guesses
//! at a foreign file.
//!
//! # Key namespace
//!
//! Every document lives under the app's own data directory, so a path is
//! `<qualifier>-<org>-<app>` on wasm and `<data dir>/<file>` everywhere else:
//!
//! | document | file | schema |
//! |---|---|---|
//! | settings | [`SETTINGS_FILE`] | [`SETTINGS_SCHEMA`] |
//! | keybinds | [`KEYBINDS_FILE`] | [`KEYBINDS_SCHEMA`] |
//! | game save | the game's choice, e.g. `save.ron` or `slot1.ron` | the game's |
//!
//! Inside the settings document, keys are `section.field`; the sections and
//! their keys are listed on [`Settings`], [`AudioSettings`],
//! [`DisplaySettings`], and [`AccessibilitySettings`].
//!
//! # Example
//!
//! ```no_run
//! use repame_input::Keymap;
//! use repame_reflect::Registry;
//! use repame_save::{
//!     AppId, KeybindStore, Keybinds, MigrationChain, SaveFile, SettingsStore, WorldDoc,
//! };
//! use repame_sim::Sim;
//! use serde::{Deserialize, Serialize};
//!
//! const GAME: AppId = AppId::new("com.mlm-games.gravity-riot");
//! const SCHEMA: u32 = 2;
//!
//! #[derive(Default, Serialize, Deserialize)]
//! struct SaveData {
//!     best_score: i32,
//!     world: WorldDoc,
//! }
//!
//! #[derive(Deserialize)]
//! struct SaveV1 {
//!     best_score: i32,
//! }
//!
//! #[derive(Serialize)]
//! struct SaveV2 {
//!     best_score: i32,
//!     tickets: i32,
//! }
//!
//! fn add_tickets(_from: u32, body: &str) -> Result<String, repame_save::MigrationError> {
//!     repame_save::rewrite::<SaveV1, SaveV2>(body, |old| SaveV2 {
//!         best_score: old.best_score,
//!         tickets: 0,
//!     })
//! }
//!
//! let registry = Registry::new();
//! let mut sim = Sim::with_default_step();
//!
//! let save: SaveFile = SaveFile::new(GAME, "save.ron", SCHEMA)
//!     .with_migrations(MigrationChain::new().step(1, add_tickets));
//! let mut data: SaveData = save.load_or_default().expect("readable save");
//! data.world.restore_sim(&mut sim, &registry).expect("world document");
//! data.world = WorldDoc::capture_sim(&sim, &registry).expect("world document");
//! save.save(&data).expect("writable data dir");
//!
//! let settings = SettingsStore::new(GAME);
//! let mut prefs = settings.load().expect("readable settings");
//! prefs.accessibility.reduced_motion = true;
//! settings.save(&prefs).expect("writable data dir");
//!
//! let bindings = KeybindStore::new(GAME);
//! let binds: Keybinds = bindings.load().expect("readable keybinds");
//! let keymap: Keymap<&str> = binds.to_keymap(|name| (name == "jump").then_some("jump"));
//! let _ = keymap;
//! ```

mod doc;
mod error;
mod identity;
mod keybinds;
mod migration;
mod settings;
mod store;
mod world;

pub use doc::SaveDoc;
pub use error::{DocError, MigrationError, SaveError};
pub use identity::AppId;
pub use keybinds::{KEYBINDS_FILE, KEYBINDS_SCHEMA, KeybindStore, Keybinds, keybind_row};
pub use migration::{MigrationChain, MigrationFn, rewrite};
pub use settings::{
    AccessibilitySettings, AudioSettings, DisplaySettings, SETTINGS_FILE, SETTINGS_SCHEMA,
    Settings, SettingsStore,
};
pub use store::{Loaded, SaveFile};
pub use world::WorldDoc;

/// The platform data-directory resolver, re-exported from `game-utils`.
pub use game_utils::save::{SaveManager, Versioned};
/// The crash-safe store this crate writes through, re-exported from
/// `game-utils` so one import wires the whole save path.
pub use game_utils::save_store::{LoadStatus, SaveStore};
/// The storage backends, re-exported from `game-utils` (`MemoryStorage` is the
/// in-memory one tests use).
pub use game_utils::storage::{FsStorage, MemoryStorage, Storage};
