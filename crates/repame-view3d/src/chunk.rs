//! Persistent chunk geometry: dirty-tracked static groups behind the
//! per-frame snapshot.
//!
//! The per-frame path (`SceneBatch::push_group`) rebuilds + re-uploads every
//! vertex every frame. That is correct for dynamic scenes (agents, markers,
//! previews) and stays the path for them. Chunked/voxel worlds (rustbox
//! gate) instead keep one [`ChunkEntry`] per chunk: the game rebuilds a
//! chunk's [`MeshGroup`]s only when its `generation` changes, and
//! [`ChunkCache`] turns them into stable [`ChunkDraw`]s the viewport appends
//! to its snapshot. Same GPU path, same validators — persistence lives
//! entirely on the CPU side, in front of the batch.
//!
//! Validation lives here ([`validate_group`]) so the cache, the viewport's
//! [`Frame3d::extend_chunks`](super::viewport::Frame3d::extend_chunks), and
//! the batch share one implementation instead of three copies drifting.
//!
//! Generations mirror the rustbox `MeshGenerations` contract: the game owns a
//! `u64` per chunk key, bumps it on edit, and the cache rebuilds entries
//! whose generation moved. Stale background builds (older generation than
//! the stored one) are dropped by the caller comparing generations — this
//! module never guesses, it only stores what the game hands it.

use std::collections::HashMap;

use super::mesh::MeshGroup;

/// One cached chunk: its validated draw groups plus the generation that
/// produced them. `groups` are stored post-clone (the caller keeps owning
/// its source); `tri_count` is cached so HUDs/stats skip re-walking.
#[derive(Clone, Debug, Default)]
pub struct ChunkEntry {
    /// Generation that produced these groups (game-owned counter).
    pub generation: u64,
    /// Validated, ready-to-push groups (index-checked at insert).
    pub groups: Vec<MeshGroup>,
    /// Cached `groups.iter().map(tri_count).sum()`.
    pub tri_count: usize,
}

impl ChunkEntry {
    /// Validate `groups` the way [`SceneBatch`](super::render::SceneBatch)
    /// does (matching lengths, in-range indices) and store them under
    /// `generation`. Malformed groups are dropped with a warning — the same
    /// never-panic contract, applied at cache time so bad data never sits
    /// in the cache. Returns the number of groups kept.
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

/// Shared group validator: the single implementation behind
/// [`ChunkEntry::insert`], [`Frame3d::extend_chunks`](super::viewport::Frame3d::extend_chunks),
/// and [`SceneBatch::push_group`](super::render::SceneBatch::push_group).
/// Matching attribute lengths, in-range indices; degenerate triangles are
/// culled at batch time (not here — the cache stores source geometry, and
/// picking runs its own per-triangle degeneracy test).
/// Returns `false` for malformed groups (logged with a warning), never panics.
///
/// Page bounds (`texture_page < layers`) stay batch-side: the cache is
/// desc-agnostic and the viewport resolves the effective desc per frame.
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

/// A validated group plus the chunk key that owns it, ready to append to a
/// [`Frame3d`](super::viewport::Frame3d). Borrowed: the viewport copies
/// what it needs per frame (the snapshot stays the owned path).
#[derive(Clone, Copy, Debug)]
pub struct ChunkDraw<'a> {
    /// Chunk key that owns this group.
    pub chunk: [i32; 3],
    /// The cached group.
    pub group: &'a MeshGroup,
}

/// Dirty-tracked chunk store. Keys are chunk positions (16^3 convention,
/// but any `[i32; 3]` key works — this module never interprets them).
#[derive(Clone, Debug, Default)]
pub struct ChunkCache {
    entries: HashMap<[i32; 3], ChunkEntry>,
}

impl ChunkCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Store `groups` under `chunk` at `generation`, replacing any older
    /// entry. Call only when the game bumped the generation (or for the
    /// initial build). Returns groups kept.
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

    /// Stored generation for `chunk`, if any. Background builders compare
    /// their build generation against this before calling `store`: a newer
    /// stored generation means the build is stale and must be dropped.
    pub fn generation(&self, chunk: &[i32; 3]) -> Option<u64> {
        self.entries.get(chunk).map(|e| e.generation)
    }

    /// True when `generation` is newer than what's stored (or nothing is
    /// stored): the chunk needs a rebuild/store.
    pub fn is_stale(&self, chunk: &[i32; 3], generation: u64) -> bool {
        self.generation(chunk).is_none_or(|g| generation > g)
    }

    /// All cached draws, chunk-sorted for deterministic submission order.
    /// The viewport appends these to its per-frame groups alongside dynamic
    /// content; depth-tested-first flattening still applies at batch time.
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
    fn malformed_groups_never_enter_the_cache() {
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
    fn stale_builds_are_detectable_before_store() {
        // The rustbox race: two background builds for one chunk, the older
        // finishing last. The caller checks `is_stale` (or `generation`)
        // before storing, so the newer result survives.
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
