//! Chunk geometry cache: dirty-tracked groups in front of the batch.
//! Game rebuilds a chunk when its generation changes.
//! Validation in [`validate_group`] is shared with batch and viewport.

use std::collections::HashMap;

use super::mesh::MeshGroup;

/// One cached chunk. `tri_count` avoids re-walk for stats.
#[derive(Clone, Debug, Default)]
pub struct ChunkEntry {
    /// Generation that produced these groups.
    pub generation: u64,
    /// Validated groups, index-checked at insert.
    pub groups: Vec<MeshGroup>,
    /// Cached `groups.iter().map(tri_count).sum()`.
    pub tri_count: usize,
}

impl ChunkEntry {
    /// Validate and store `groups` under `generation`. Drops bad groups.
    /// Returns groups kept.
    pub fn insert(&mut self, generation: u64, groups: &[MeshGroup]) -> usize {
        self.generation = generation;
        self.groups.clear();
        self.tri_count = 0;
        for group in groups {
            if !validate_group(group) {
                continue;
            }
            self.tri_count += group.tri_count();
            self.groups.push(group.clone());
        }
        self.groups.len()
    }

    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }
}

/// Shared group validator for cache, viewport, and batch.
/// Checks lengths and index range. Degenerate tris cull at batch time.
/// Returns false for malformed groups. Logs a warning, does not panic.
///
/// Page bounds stay batch-side: cache is desc-agnostic.
pub fn validate_group(group: &MeshGroup) -> bool {
    if group.is_empty() {
        return false;
    }
    if group.positions.len() != group.colors.len() {
        log::warn!(
            "chunk_cache: dropping group ({} positions vs {} colors)",
            group.positions.len(),
            group.colors.len()
        );
        return false;
    }
    if !group.normals.is_empty() && group.normals.len() != group.positions.len() {
        log::warn!(
            "chunk_cache: dropping group ({} positions vs {} normals)",
            group.positions.len(),
            group.normals.len()
        );
        return false;
    }
    if !group.uvs.is_empty() && group.uvs.len() != group.positions.len() {
        log::warn!(
            "chunk_cache: dropping group ({} positions vs {} uvs)",
            group.positions.len(),
            group.uvs.len()
        );
        return false;
    }
    if group
        .indices
        .iter()
        .any(|i| (*i as usize) >= group.positions.len())
    {
        log::warn!("chunk_cache: dropping group (index out of range)");
        return false;
    }
    true
}

/// Validated group plus owner chunk key. Viewport copies per frame.
#[derive(Clone, Copy, Debug)]
pub struct ChunkDraw<'a> {
    /// Chunk key that owns this group.
    pub chunk: [i32; 3],
    /// The cached group.
    pub group: &'a MeshGroup,
}

/// Chunk store keyed by chunk position. Keys are opaque here.
#[derive(Clone, Debug, Default)]
pub struct ChunkCache {
    entries: HashMap<[i32; 3], ChunkEntry>,
}

impl ChunkCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Store `groups` under `chunk` at `generation`. Returns groups kept.
    pub fn store(&mut self, chunk: [i32; 3], generation: u64, groups: &[MeshGroup]) -> usize {
        let mut entry = ChunkEntry::default();
        let kept = entry.insert(generation, groups);
        if entry.is_empty() {
            self.entries.remove(&chunk);
        } else {
            self.entries.insert(chunk, entry);
        }
        kept
    }

    /// Drop one chunk (unloaded region, cleared level).
    pub fn remove(&mut self, chunk: &[i32; 3]) -> bool {
        self.entries.remove(chunk).is_some()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Stored generation for `chunk`, if any.
    pub fn generation(&self, chunk: &[i32; 3]) -> Option<u64> {
        self.entries.get(chunk).map(|e| e.generation)
    }

    /// True when `generation` is newer than stored, or nothing stored.
    pub fn is_stale(&self, chunk: &[i32; 3], generation: u64) -> bool {
        self.generation(chunk).is_none_or(|g| generation > g)
    }

    /// All cached draws, chunk-sorted for stable submission order.
    pub fn draws(&self) -> Vec<ChunkDraw<'_>> {
        let mut keys: Vec<[i32; 3]> = self.entries.keys().copied().collect();
        keys.sort();
        let mut out = Vec::new();
        for key in keys {
            if let Some(entry) = self.entries.get(&key) {
                for group in &entry.groups {
                    out.push(ChunkDraw { chunk: key, group });
                }
            }
        }
        out
    }

    pub fn chunk_count(&self) -> usize {
        self.entries.len()
    }

    pub fn tri_count(&self) -> usize {
        self.entries.values().map(|e| e.tri_count).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad_group(tint: [f32; 3]) -> MeshGroup {
        let mut g = MeshGroup {
            depth_test: true,
            ..Default::default()
        };
        g.push_quad(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            tint,
        );
        g
    }

    #[test]
    fn store_and_draws_are_chunk_sorted() {
        let mut cache = ChunkCache::new();
        cache.store([1, 0, 0], 1, &[quad_group([1.0, 0.0, 0.0])]);
        cache.store([0, 0, 0], 1, &[quad_group([0.0, 1.0, 0.0])]);
        assert_eq!(cache.chunk_count(), 2);
        assert_eq!(cache.tri_count(), 4);
        let draws = cache.draws();
        assert_eq!(draws.len(), 2);
        assert_eq!(draws[0].chunk, [0, 0, 0]);
        assert_eq!(draws[1].chunk, [1, 0, 0]);
        assert_eq!(cache.generation(&[0, 0, 0]), Some(1));
    }

    #[test]
    fn newer_generation_replaces_older() {
        let mut cache = ChunkCache::new();
        cache.store([0, 0, 0], 1, &[quad_group([1.0, 0.0, 0.0])]);
        assert!(cache.is_stale(&[0, 0, 0], 2));
        assert!(!cache.is_stale(&[0, 0, 0], 1));
        cache.store(
            [0, 0, 0],
            2,
            &[quad_group([0.0, 0.0, 1.0]), quad_group([0.0, 0.0, 1.0])],
        );
        assert_eq!(cache.generation(&[0, 0, 0]), Some(2));
        assert_eq!(cache.draws().len(), 2);
        assert_eq!(cache.tri_count(), 4);
    }

    #[test]
    fn empty_store_evicts() {
        let mut cache = ChunkCache::new();
        cache.store([0, 0, 0], 1, &[quad_group([1.0, 0.0, 0.0])]);
        assert_eq!(cache.store([0, 0, 0], 2, &[]), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.generation(&[0, 0, 0]), None);
        assert!(!cache.remove(&[0, 0, 0]));
    }

    #[test]
    fn malformed_groups_drop_at_store() {
        let mut cache = ChunkCache::new();
        let bad = MeshGroup {
            positions: vec![[0.0, 0.0, 0.0]],
            colors: vec![],
            indices: vec![0, 0, 0],
            depth_test: true,
            ..Default::default()
        };
        assert_eq!(
            cache.store([0, 0, 0], 1, &[bad, quad_group([1.0, 1.0, 1.0])]),
            1
        );
        assert_eq!(cache.tri_count(), 2);
    }

    #[test]
    fn stale_builds_drop_before_store() {
        // Two background builds for one chunk, older finishing last.
        let mut cache = ChunkCache::new();
        cache.store([3, 0, 1], 5, &[quad_group([1.0, 0.0, 0.0])]);
        assert!(
            !cache.is_stale(&[3, 0, 1], 4),
            "older build must be dropped"
        );
        assert!(!cache.is_stale(&[3, 0, 1], 5), "same build must be dropped");
        assert!(cache.is_stale(&[3, 0, 1], 6), "newer build stores");
    }
}
