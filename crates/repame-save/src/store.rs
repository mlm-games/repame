//! The file layer: a document in a crash-safe store.
//!
//! [`SaveFile`] binds one filename under one [`AppId`]'s data directory to one
//! schema and one [`MigrationChain`]. The bytes are not handled here: every
//! write and read goes through `game_utils::save_store::SaveStore`, which
//! already does the temp-file + atomic-rename write, the `.bak` rotation, the
//! corruption quarantine, and the recovery from the last good backup. This
//! module supplies the RON document validator and the envelope rules.

use std::path::{Path, PathBuf};

use game_utils::save::Versioned;
use game_utils::save_store::{LoadStatus, SaveStore};
use game_utils::storage::{FsStorage, Storage};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::doc::SaveDoc;
use crate::error::SaveError;
use crate::identity::AppId;
use crate::migration::MigrationChain;

/// What one load found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Loaded<T> {
    /// The migrated document, when the file (or a backup) held one.
    pub data: Option<T>,
    /// What the store found: `Ok`, `Missing`, `Corrupt` when the file was
    /// replaced by a backup, or `Unreadable`.
    pub status: LoadStatus,
    /// The `.bak`/`temp_` file the bytes actually came from, when recovery
    /// happened. Worth logging: the player's newest save was broken.
    pub recovered_from: Option<PathBuf>,
}

/// One versioned save file.
///
/// ```no_run
/// use repame_save::{AppId, MigrationChain, SaveFile, SaveDoc};
/// use serde::{Deserialize, Serialize};
///
/// const GAME: AppId = AppId::new("com.mlm-games.kickback");
///
/// #[derive(Default, Serialize, Deserialize)]
/// struct SaveData {
///     best_score: i32,
///     muted: bool,
/// }
///
/// let save: SaveFile = SaveFile::new(GAME, "save.ron", 1);
/// let data: SaveData = save.load_or_default().expect("readable save");
/// save.save(&data).expect("writable data dir");
/// ```
#[derive(Clone, Debug)]
pub struct SaveFile<S: Storage = FsStorage> {
    schema: u32,
    migrations: MigrationChain,
    store: SaveStore<S>,
}

impl SaveFile<FsStorage> {
    /// The app's data directory plus `file`.
    pub fn new(app: AppId, file: impl Into<String>, schema: u32) -> Self {
        let file = file.into();
        Self::in_dir_with_storage(app.data_dir(), file, schema, FsStorage)
    }

    /// An explicit directory, for tests, tools, and saves exported next to
    /// something else. Games use [`SaveFile::new`].
    pub fn in_dir(dir: impl Into<PathBuf>, file: impl Into<String>, schema: u32) -> Self {
        Self::in_dir_with_storage(dir, file, schema, FsStorage)
    }
}

impl<S: Storage> SaveFile<S> {
    /// Build a file in `dir` on an explicit backend.
    ///
    /// The backend is the whole platform story: `FsStorage` is the filesystem
    /// on native and `ropfs` on wasm, [`MemoryStorage`](game_utils::storage::MemoryStorage)
    /// is an in-memory save for tests.
    pub fn in_dir_with_storage(
        dir: impl Into<PathBuf>,
        file: impl Into<String>,
        schema: u32,
        storage: S,
    ) -> Self {
        let store =
            SaveStore::new_with_storage(dir, file, storage).with_validator(SaveDoc::is_document);
        Self {
            schema,
            migrations: MigrationChain::new(),
            store,
        }
    }

    /// Move the same file and schema onto another backend, e.g. an in-memory
    /// store for a test that must not touch the player's save.
    pub fn with_storage<S2: Storage>(self, storage: S2) -> SaveFile<S2> {
        let store = SaveStore::new_with_storage(
            self.store.dir.clone(),
            self.store.file_name.clone(),
            storage,
        )
        .with_validator(SaveDoc::is_document);
        SaveFile {
            schema: self.schema,
            migrations: self.migrations,
            store,
        }
    }

    /// Register the schema migrations this file applies on load.
    #[must_use]
    pub fn with_migrations(mut self, migrations: MigrationChain) -> Self {
        self.migrations = migrations;
        self
    }

    /// The schema this file writes and migrates up to.
    pub fn schema(&self) -> u32 {
        self.schema
    }

    /// The registered migrations.
    pub fn migrations(&self) -> &MigrationChain {
        &self.migrations
    }

    /// The directory the file lives in.
    pub fn dir(&self) -> &Path {
        &self.store.dir
    }

    /// The filename inside [`SaveFile::dir`].
    pub fn file(&self) -> &str {
        &self.store.file_name
    }

    /// The full path of the save.
    pub fn path(&self) -> PathBuf {
        self.store.path()
    }

    /// Whether a save file exists. The `.bak`/`temp_` siblings do not count.
    pub fn exists(&self) -> bool {
        self.store.exists()
    }

    /// Delete the save together with its `.bak`/`temp_` siblings, so a load
    /// cannot resurrect it.
    pub fn delete(&self) {
        self.store.delete();
    }

    /// Load and migrate the document, with the store's verdict.
    ///
    /// `Ok(None)` means no file yet. A corrupt file that no backup can replace
    /// is [`SaveError::Corrupt`], and an unreadable one is
    /// [`SaveError::Unreadable`]: neither is an empty save.
    pub fn load_with_status(&self) -> Result<Loaded<SaveDoc>, SaveError> {
        let path = self.path();
        let result = self.store.load(&SaveDoc::is_document, &[]);
        let Some(bytes) = result.data else {
            return match result.status {
                LoadStatus::Unreadable => Err(SaveError::Unreadable { path }),
                LoadStatus::Corrupt => Err(SaveError::Corrupt { path }),
                LoadStatus::Ok | LoadStatus::Missing => Ok(Loaded {
                    data: None,
                    status: result.status,
                    recovered_from: result.recovered_from,
                }),
            };
        };
        let Ok(text) = std::str::from_utf8(&bytes) else {
            // The validator parsed these bytes, so this cannot happen unless the
            // store's validator was replaced; either way it is not a document.
            return Err(SaveError::Corrupt { path });
        };
        let mut doc = SaveDoc::parse(text).map_err(|e| SaveError::from((path.clone(), e)))?;
        // `migrate` is also the forward-version refusal: a file stamped newer
        // than this binary is never partially parsed.
        doc.migrate(&self.migrations, self.schema)
            .map_err(|e| SaveError::from((path.clone(), e)))?;
        Ok(Loaded {
            data: Some(doc),
            status: result.status,
            recovered_from: result.recovered_from,
        })
    }

    /// The migrated document, or `None` when there is no file yet.
    pub fn load_doc(&self) -> Result<Option<SaveDoc>, SaveError> {
        Ok(self.load_with_status()?.data)
    }

    /// The typed payload, or `None` when there is no file yet.
    pub fn load<T: DeserializeOwned>(&self) -> Result<Option<T>, SaveError> {
        let path = self.path();
        match self.load_doc()? {
            None => Ok(None),
            Some(doc) => doc
                .body_as::<T>()
                .map(Some)
                .map_err(|e| SaveError::from((path, e))),
        }
    }

    /// The typed payload, or [`Default`] when there is no file yet.
    ///
    /// A missing file is a first run; a file that does not fit is an error, so a
    /// broken save cannot look like a brand new one.
    pub fn load_or_default<T: DeserializeOwned + Default>(&self) -> Result<T, SaveError> {
        Ok(self.load::<T>()?.unwrap_or_default())
    }

    /// Save a typed payload at this file's schema.
    pub fn save<T: Serialize>(&self, payload: &T) -> Result<(), SaveError> {
        let path = self.path();
        let doc = SaveDoc::new(self.schema, payload).map_err(|e| SaveError::from((path, e)))?;
        self.save_doc(&doc)
    }

    /// Save a payload that stamps its own version field (`game_utils::save::Versioned`),
    /// setting that field to this file's schema first.
    pub fn save_versioned<T: Serialize + Versioned>(
        &self,
        payload: &mut T,
    ) -> Result<(), SaveError> {
        payload.set_version(self.schema);
        self.save(&*payload)
    }

    /// Load a payload that stamps its own version field, running its `migrate`
    /// when the payload claims an older version than the document.
    ///
    /// Prefer a [`MigrationChain`] for new code: it rewrites the payload text
    /// before the type is known, so a step can reshape fields that the current
    /// type no longer has. This exists for the shape the games already carry.
    pub fn load_versioned<T: DeserializeOwned + Default + Versioned>(
        &self,
    ) -> Result<T, SaveError> {
        let path = self.path();
        let Some(mut payload) = self.load::<T>()? else {
            return Ok(T::default());
        };
        let claimed = payload.version();
        if claimed > self.schema {
            return Err(SaveError::ForwardVersion {
                path,
                found: claimed,
                current: self.schema,
            });
        }
        if claimed < self.schema {
            payload.migrate(claimed, self.schema);
        }
        payload.set_version(self.schema);
        Ok(payload)
    }

    /// Write a document as it stands.
    ///
    /// A document stamped newer than this file is refused, so a downgraded
    /// binary cannot overwrite a newer save with an older schema and drop the
    /// fields it does not know about.
    pub fn save_doc(&self, doc: &SaveDoc) -> Result<(), SaveError> {
        let path = self.path();
        if doc.schema() > self.schema {
            return Err(SaveError::ForwardVersion {
                path,
                found: doc.schema(),
                current: self.schema,
            });
        }
        let text = doc
            .to_text()
            .map_err(|e| SaveError::from((path.clone(), e)))?;
        self.store
            .write(text.as_bytes())
            .map_err(|message| SaveError::Write { path, message })
    }

    /// The backend this file reads and writes through.
    pub fn storage(&self) -> &S {
        self.store.storage()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    struct SaveData {
        best_score: i32,
        muted: bool,
    }

    #[derive(Serialize, Deserialize)]
    struct SaveDataV1 {
        best_score: i32,
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("repame_save_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    fn data() -> SaveData {
        SaveData {
            best_score: 1200,
            muted: true,
        }
    }

    #[test]
    fn save_then_load_round_trips() {
        let dir = scratch("round_trip");
        let file: SaveFile = SaveFile::in_dir(&dir, "save.ron", 1);
        assert!(!file.exists());
        assert_eq!(file.load::<SaveData>().expect("no file"), None);
        file.save(&data()).expect("writes");
        assert!(file.exists());
        assert_eq!(file.load::<SaveData>().expect("reads"), Some(data()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_write_is_atomic_and_leaves_no_temp_file() {
        let dir = scratch("atomic");
        let file: SaveFile = SaveFile::in_dir(&dir, "save.ron", 1);
        file.save(&data()).expect("writes");
        let names: Vec<String> = std::fs::read_dir(&dir)
            .expect("dir")
            .map(|entry| {
                entry
                    .expect("entry")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        assert_eq!(names, vec!["save.ron".to_string()], "no temp_ file is left");
        assert!(SaveDoc::is_document(
            &std::fs::read(file.path()).expect("bytes")
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_corrupt_file_is_recovered_from_the_backup() {
        let dir = scratch("recovery");
        let file: SaveFile = SaveFile::in_dir(&dir, "save.ron", 1);
        file.save(&data()).expect("first write");
        let mut second = data();
        second.best_score = 1300;
        file.save(&second).expect("second write rotates a .bak");
        std::fs::write(file.path(), "truncated (schema:").expect("corrupt it on purpose");

        let loaded = file.load_with_status().expect("recovers");
        assert_eq!(loaded.status, LoadStatus::Corrupt);
        assert!(loaded.recovered_from.is_some());
        assert_eq!(
            loaded
                .data
                .expect("document")
                .body_as::<SaveData>()
                .expect("payload"),
            data(),
            "the backup holds the previous save"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_file_from_a_newer_build_is_refused_not_partially_parsed() {
        let dir = scratch("forward");
        let file: SaveFile = SaveFile::in_dir(&dir, "save.ron", 1);
        let newer = SaveDoc::new(2, &data()).expect("serializes");
        std::fs::write(file.path(), newer.to_text().expect("writes")).expect("write");

        let error = file.load::<SaveData>().expect_err("newer schema");
        assert!(
            matches!(
                error,
                SaveError::ForwardVersion {
                    found: 2,
                    current: 1,
                    ..
                }
            ),
            "{error}"
        );
        assert_eq!(error.path(), file.path());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_newer_document_cannot_be_written_over_an_older_file() {
        let dir = scratch("write_forward");
        let file: SaveFile = SaveFile::in_dir(&dir, "save.ron", 1);
        let newer = SaveDoc::new(4, &data()).expect("serializes");
        let error = file.save_doc(&newer).expect_err("refuses to stamp up");
        assert!(
            matches!(
                error,
                SaveError::ForwardVersion {
                    found: 4,
                    current: 1,
                    ..
                }
            ),
            "{error}"
        );
        assert!(!file.exists(), "nothing was written");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_applies_the_migration_chain() {
        fn add_muted(_from: u32, body: &str) -> Result<String, crate::MigrationError> {
            crate::migration::rewrite::<SaveDataV1, SaveData>(body, |old| SaveData {
                best_score: old.best_score,
                muted: false,
            })
        }

        let dir = scratch("migrate");
        let old: SaveFile = SaveFile::in_dir(&dir, "save.ron", 1);
        old.save(&SaveDataV1 { best_score: 40 }).expect("writes");

        let file: SaveFile = SaveFile::in_dir(&dir, "save.ron", 2)
            .with_migrations(MigrationChain::new().step(1, add_muted));
        assert_eq!(
            file.load::<SaveData>().expect("reads"),
            Some(SaveData {
                best_score: 40,
                muted: false,
            })
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_file_is_a_first_run_and_a_broken_one_is_not() {
        let dir = scratch("defaults");
        let file: SaveFile = SaveFile::in_dir(&dir, "save.ron", 1);
        assert_eq!(
            file.load_or_default::<SaveData>().expect("first run"),
            SaveData::default()
        );

        std::fs::write(file.path(), "not a save at all").expect("write");
        let error = file.load_or_default::<SaveData>().expect_err("broken file");
        assert!(matches!(error, SaveError::Corrupt { .. }), "{error}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
