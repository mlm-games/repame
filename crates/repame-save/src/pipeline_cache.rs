//! Driver pipeline-cache blobs, stored next to the app's other data.
//!
//! A cache blob is opaque bytes from `wgpu::PipelineCache::get_data`. The key
//! should be stable per device (for example `wgpu::util::pipeline_cache_key`),
//! so a blob from an incompatible adapter is never fed into a new cache.

use std::io;
use std::path::PathBuf;

use game_utils::storage::{FsStorage, Storage};

use crate::identity::AppId;

pub const CACHE_DIR: &str = "pipeline-cache";

#[derive(Clone, Debug)]
pub struct PipelineCacheStore<S: Storage = FsStorage> {
    dir: PathBuf,
    storage: S,
}

impl PipelineCacheStore<FsStorage> {
    pub fn new(app: AppId) -> Self {
        Self::in_dir(app.data_dir(), FsStorage)
    }
}

impl<S: Storage> PipelineCacheStore<S> {
    pub fn in_dir(dir: impl Into<PathBuf>, storage: S) -> Self {
        Self {
            dir: dir.into().join(CACHE_DIR),
            storage,
        }
    }

    fn path(&self, key: &str) -> PathBuf {
        self.dir.join(format!("{key}.bin"))
    }

    pub fn load(&self, key: &str) -> Option<Vec<u8>> {
        self.storage.read(&self.path(key)).ok().flatten()
    }

    pub fn save(&self, key: &str, bytes: &[u8]) -> io::Result<()> {
        let path = self.path(key);
        self.storage.create_dir_all(&self.dir)?;
        let temp = PathBuf::from(format!("{}.tmp", path.display()));
        self.storage.write(&temp, bytes)?;
        match self.storage.rename(&temp, &path) {
            Ok(()) => Ok(()),
            Err(err) => {
                self.storage.remove_file(&temp).ok();
                Err(err)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_utils::storage::MemoryStorage;

    #[test]
    fn blobs_round_trip_and_missing_keys_read_as_none() {
        let store = PipelineCacheStore::in_dir("/tmp/repame-cache-test", MemoryStorage::new());
        assert_eq!(store.load("abc"), None);
        store.save("abc", &[1, 2, 3, 4]).expect("writes");
        assert_eq!(store.load("abc"), Some(vec![1, 2, 3, 4]));
    }
}
