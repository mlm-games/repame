//! Dev-mode source watching.
//!
//! [`AssetWatcher`] polls the content root over the crate's [`Storage`]
//! abstraction (no dependency needed, and it works on the in-memory store in
//! tests) and reports added, modified and removed source files as a diff of
//! cheap `size + mtime` stamps. Feed the events to
//! [`HotReloader`](crate::HotReloader) to re-import just those sources.
//!
//! Stamps are second-resolution, the coarse limit of the portable `Storage`
//! API: two writes inside the same second that keep the file size identical are
//! invisible. [`HotReloader::reimport_all`](crate::HotReloader::reimport_all)
//! is the escape hatch when a watcher is asked to force a refresh.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::error::ContentError;
use game_utils::storage::{FsStorage, Storage};

/// A cheap change stamp for one source file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FileStamp {
    pub size: u64,
    pub modified_secs: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WatchEventKind {
    Added,
    Modified,
    Removed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WatchEvent {
    pub path: PathBuf,
    pub kind: WatchEventKind,
}

/// Polling watcher over a content root.
pub struct AssetWatcher<S: Storage = FsStorage> {
    root: PathBuf,
    storage: S,
    snapshot: BTreeMap<PathBuf, FileStamp>,
}

impl AssetWatcher<FsStorage> {
    /// Watch `root`, taking the initial snapshot so the first poll reports only
    /// changes made after construction.
    pub fn new(root: impl Into<PathBuf>) -> Result<Self, ContentError> {
        Self::new_with_storage(root, FsStorage)
    }
}

impl<S: Storage> AssetWatcher<S> {
    pub fn new_with_storage(root: impl Into<PathBuf>, storage: S) -> Result<Self, ContentError> {
        let root = root.into();
        if !storage.is_dir(&root) {
            return Err(ContentError::new(&root, "watch root is not a directory"));
        }
        let snapshot = scan(&root, &storage)?;
        Ok(Self {
            root,
            storage,
            snapshot,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The stamps recorded by the last poll.
    pub fn snapshot(&self) -> &BTreeMap<PathBuf, FileStamp> {
        &self.snapshot
    }

    /// Rescan and return the events since the previous poll, then record them.
    pub fn poll(&mut self) -> Result<Vec<WatchEvent>, ContentError> {
        let current = scan(&self.root, &self.storage)?;
        let events = diff_snapshots(&self.snapshot, &current);
        self.snapshot = current;
        Ok(events)
    }
}

/// Compare two snapshots into `Added`/`Modified`/`Removed` events, ordered by path.
pub fn diff_snapshots(
    previous: &BTreeMap<PathBuf, FileStamp>,
    current: &BTreeMap<PathBuf, FileStamp>,
) -> Vec<WatchEvent> {
    let paths: BTreeSet<&PathBuf> = previous.keys().chain(current.keys()).collect();
    let mut events = Vec::new();
    for path in paths {
        let (before, after) = (previous.get(path), current.get(path));
        let kind = match (before, after) {
            (None, Some(_)) => WatchEventKind::Added,
            (Some(_), None) => WatchEventKind::Removed,
            (Some(before), Some(after)) if before != after => WatchEventKind::Modified,
            _ => continue,
        };
        events.push(WatchEvent {
            path: path.clone(),
            kind,
        });
    }
    events
}

fn scan<S: Storage>(
    root: &Path,
    storage: &S,
) -> Result<BTreeMap<PathBuf, FileStamp>, ContentError> {
    let mut snapshot = BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(directory) = stack.pop() {
        let entries = storage
            .read_dir(&directory)
            .map_err(|error| ContentError::new(&directory, format!("watch directory: {error}")))?;
        for entry in entries {
            let Some(name) = entry.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if name.starts_with('.') {
                continue;
            }
            if storage.is_dir(&entry) {
                stack.push(entry);
            } else if storage.is_file(&entry) {
                snapshot.insert(
                    entry.clone(),
                    FileStamp {
                        size: storage.metadata_len(&entry).unwrap_or(0),
                        modified_secs: storage.mtime_secs(&entry).unwrap_or(0),
                    },
                );
            }
        }
    }
    Ok(snapshot)
}
