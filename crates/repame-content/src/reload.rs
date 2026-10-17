//! Dev-mode hot reload: watch the content root, re-import only what changed,
//! and publish a queue of swapped artifacts.
//!
//! [`HotReloader`] pairs an [`AssetWatcher`] with the project's
//! [`ImportCache`]. [`HotReloader::poll`] is meant to be called once per frame
//! in development; it re-imports the changed sources and their dependents and
//! returns the re-imported artifacts as [`ArtifactSwap`]s. The game applies the
//! queue at a simulation or view boundary, so a reload never mutates state
//! mid-step and never needs a restart.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use crate::error::ContentError;
use crate::import::{AssetKind, ImportCache, ImportedAsset};
use crate::project::{AssetSource, Project};
use crate::watch::{AssetWatcher, WatchEvent};
use game_utils::storage::{FsStorage, Storage};

/// One re-imported artifact, ready to be applied at a boundary.
#[derive(Clone, Debug)]
pub struct ArtifactSwap {
    pub id: String,
    pub kind: AssetKind,
    pub source: PathBuf,
    pub artifact: PathBuf,
    pub bytes: Vec<u8>,
    pub metadata: BTreeMap<String, ron::Value>,
}

impl ArtifactSwap {
    fn from_import(imported: ImportedAsset, source: PathBuf) -> Self {
        Self {
            id: imported.id,
            kind: imported.kind,
            source,
            artifact: imported.artifact,
            bytes: imported.bytes,
            metadata: imported.metadata,
        }
    }
}

/// What one poll produced.
#[derive(Debug, Default)]
pub struct ReloadOutcome {
    /// Raw file events since the previous poll.
    pub events: Vec<WatchEvent>,
    /// True when `project.ron` changed and the project was reloaded.
    pub manifest_changed: bool,
    /// Ids of scenes whose source changed.
    pub scenes: Vec<String>,
    /// Re-imported artifacts to apply at the next boundary.
    pub swaps: Vec<ArtifactSwap>,
    /// Re-import failures. Reported, never silently dropped; a failed source
    /// keeps its previous artifact until it imports again.
    pub failures: Vec<ContentError>,
}

impl ReloadOutcome {
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
            && !self.manifest_changed
            && self.scenes.is_empty()
            && self.swaps.is_empty()
            && self.failures.is_empty()
    }

    pub fn swap(&self, id: &str) -> Option<&ArtifactSwap> {
        self.swaps.iter().find(|swap| swap.id == id)
    }
}

/// Watches a project's content root and re-imports changed sources.
pub struct HotReloader<S: Storage = FsStorage> {
    root: PathBuf,
    project: Project,
    cache: ImportCache<S>,
    watcher: AssetWatcher<S>,
}

impl HotReloader<FsStorage> {
    pub fn new(project: Project) -> Result<Self, ContentError> {
        Self::with_storage(project, FsStorage)
    }
}

impl<S: Storage> HotReloader<S> {
    pub fn with_storage(project: Project, storage: S) -> Result<Self, ContentError> {
        let root = project.root().to_path_buf();
        let cache =
            ImportCache::new_with_storage(root.join(".repame").join("imports"), storage.clone())?;
        let watcher = AssetWatcher::new_with_storage(&root, storage)?;
        Ok(Self {
            root,
            project,
            cache,
            watcher,
        })
    }

    pub fn project(&self) -> &Project {
        &self.project
    }

    pub fn watcher(&self) -> &AssetWatcher<S> {
        &self.watcher
    }

    pub fn cache(&self) -> &ImportCache<S> {
        &self.cache
    }

    /// Rescan the content root and re-import every changed source plus the
    /// assets that depend on it.
    pub fn poll(&mut self) -> Result<ReloadOutcome, ContentError> {
        let events = self.watcher.poll()?;
        let mut outcome = ReloadOutcome {
            events: events.clone(),
            ..ReloadOutcome::default()
        };
        if events.is_empty() {
            return Ok(outcome);
        }

        if events
            .iter()
            .any(|event| event.path == self.project.project_path())
        {
            match Project::load(&self.root) {
                Ok(project) => {
                    self.project = project;
                    outcome.manifest_changed = true;
                }
                Err(error) => outcome.failures.push(error),
            }
        }

        let sources = self.project.asset_sources()?;
        let by_id: BTreeMap<&str, &AssetSource> = sources
            .iter()
            .map(|source| (source.id.as_str(), source))
            .collect();
        let changed = affected_assets(&sources, &events);

        for id in self.project.asset_import_order() {
            if !changed.contains(&id) {
                continue;
            }
            let Some(source) = by_id.get(id.as_str()).copied() else {
                continue;
            };
            let dependencies = dependency_fingerprints(source, &by_id);
            match self.cache.import(source, &dependencies) {
                Ok(imported) => {
                    outcome
                        .swaps
                        .push(ArtifactSwap::from_import(imported, source.path.clone()));
                }
                Err(error) => outcome.failures.push(error),
            }
        }

        outcome.scenes = changed_scenes(&self.project, &events);
        Ok(outcome)
    }

    /// Re-import every asset regardless of stamps, for an explicit refresh.
    pub fn reimport_all(&mut self) -> Result<ReloadOutcome, ContentError> {
        let sources = self.project.asset_sources()?;
        let by_id: BTreeMap<&str, &AssetSource> = sources
            .iter()
            .map(|source| (source.id.as_str(), source))
            .collect();
        let mut outcome = ReloadOutcome::default();
        for id in self.project.asset_import_order() {
            let Some(source) = by_id.get(id.as_str()).copied() else {
                continue;
            };
            let dependencies = dependency_fingerprints(source, &by_id);
            match self.cache.import(source, &dependencies) {
                Ok(imported) => outcome
                    .swaps
                    .push(ArtifactSwap::from_import(imported, source.path.clone())),
                Err(error) => outcome.failures.push(error),
            }
        }
        Ok(outcome)
    }
}

fn dependency_fingerprints(
    source: &AssetSource,
    by_id: &BTreeMap<&str, &AssetSource>,
) -> BTreeMap<String, u64> {
    source
        .dependencies
        .iter()
        .filter_map(|dependency| {
            by_id
                .get(dependency.as_str())
                .map(|entry| (dependency.clone(), entry.fingerprint))
        })
        .collect()
}

/// Changed asset ids plus everything that transitively depends on them.
fn affected_assets(sources: &[AssetSource], events: &[WatchEvent]) -> BTreeSet<String> {
    let mut affected: BTreeSet<String> = sources
        .iter()
        .filter(|source| events.iter().any(|event| event.path == source.path))
        .map(|source| source.id.clone())
        .collect();
    let mut grew = true;
    while grew {
        grew = false;
        for source in sources {
            if affected.contains(&source.id) {
                continue;
            }
            if source
                .dependencies
                .iter()
                .any(|dependency| affected.contains(dependency))
            {
                affected.insert(source.id.clone());
                grew = true;
            }
        }
    }
    affected
}

fn changed_scenes(project: &Project, events: &[WatchEvent]) -> Vec<String> {
    let mut scenes = BTreeSet::new();
    let mut ids: Vec<String> = project
        .manifest()
        .scenes
        .iter()
        .map(|entry| entry.id.clone())
        .collect();
    ids.sort();
    for id in ids {
        let Some(path) = project.scene_path(&id) else {
            continue;
        };
        if events.iter().any(|event| event.path == path) {
            scenes.insert(id);
        }
    }
    scenes.into_iter().collect()
}
