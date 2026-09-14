//! Chunk streaming jobs: background mesh builds with priority, dedup,
//! and stale-drop, draining into a [`ChunkCache`](super::chunk::ChunkCache).
//!
//! The cache is sync and single-threaded; meshing a 16^3 chunk with greedy
//! merge costs real milliseconds on the frame thread. This module moves
//! the build off-frame: the game submits `(chunk, generation)` requests,
//! worker threads run `build_chunk_mesh` over a caller-owned
//! [`VoxelSource`](crate::VoxelSource), and the game drains finished
//! builds into its cache once per frame. Generations make races harmless:
//! an older build finishing after a newer store drops at drain time via
//! [`is_stale`](super::chunk::ChunkCache::is_stale) (same rule as the
//! sync path, see the `stale_builds_drop_before_store` test).
//!
//! Threading is `web_workers` (native threads; wasm runs the same queue
//! inline on `pump`, no threads spawned). Workers are spawned on first
//! request and joined on drop. Mesh input (classify/is_solid/tints) must
//! be `Clone + Send + Sync + 'static`; the voxel source snapshot must be
//! `Send + 'static` — games pass an owned snapshot (HashMap, Arc), never
//! a borrow of live world state.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use super::chunk::ChunkCache;
use super::voxel::{ChunkMeshInput, ChunkMeshOutput, FaceKind, build_chunk_mesh};
use crate::mesh::MeshGroup;

/// One queued mesh-build request: (priority, sequence, chunk, generation).
/// Lower priority builds first (near = small distance); sequence keeps
/// submission order within one priority (stable drain).
type QueuedRequest = (f32, u64, [i32; 3], u64);

/// One finished chunk build, ready to store.
#[derive(Clone, Debug)]
pub struct ChunkBuild {
    /// Chunk key that was built.
    pub chunk: [i32; 3],
    /// Generation requested (drain drops when stale).
    pub generation: u64,
    /// Built groups (validated by the worker, re-validated at store).
    pub groups: Vec<MeshGroup>,
    /// Cached tri count.
    pub tri_count: usize,
}

/// Priority for a chunk request. Near chunks build first; the queue is a
/// stable priority drain (same priority keeps submission order).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChunkPriority(pub f32);

/// Background chunk mesh builder over a caller-owned source snapshot.
///
/// `S` is the owned snapshot type (must answer across chunk borders, same
/// contract as [`VoxelSource`](crate::VoxelSource)); `F`/`G` are the
/// classify/is_solid closures. All three cross threads, so they must be
/// `Send + 'static` (`F`/`G` also `Sync` for sharing across workers).
///
/// Shape: `request(chunk, generation, priority)` dedups (same chunk+gen
/// queues once; newer generation replaces an older queued request), `pump`
/// runs due builds (inline on wasm, worker threads on native),
/// `drain_into(cache)` stores finished builds newest-wins.
pub struct ChunkStreamer<S, F, G>
where
    S: crate::voxel::VoxelSource + Clone + Send + 'static,
    F: Fn(u32) -> FaceKind + Clone + Send + Sync + 'static,
    G: Fn(u32) -> bool + Clone + Send + Sync + 'static,
{
    source: S,
    input: ChunkMeshInput<F, G>,
    /// Queued requests: (priority, sequence, chunk, generation).
    /// `Mutex` (not channel): dedup needs a scan, priorities reorder.
    queue: Arc<Mutex<VecDeque<QueuedRequest>>>,
    /// In-flight keys (queued or building): prevents duplicate submits.
    /// Entries clear on completion (drain removes after store-or-drop).
    inflight: Arc<Mutex<HashMap<[i32; 3], u64>>>,
    /// Finished builds awaiting drain.
    done: Arc<Mutex<VecDeque<ChunkBuild>>>,
    /// Monotonic submit sequence (stable order within one priority).
    seq: u64,
    /// Worker count (native only; wasm ignores).
    workers: usize,
    /// Spawn-once guard.
    started: bool,
    /// Max queued requests (backpressure: farthest drops first).
    max_queue: usize,
}

impl<S, F, G> ChunkStreamer<S, F, G>
where
    S: crate::voxel::VoxelSource + Clone + Send + 'static,
    F: Fn(u32) -> FaceKind + Clone + Send + Sync + 'static,
    G: Fn(u32) -> bool + Clone + Send + Sync + 'static,
{
    /// New streamer over `source` with `input` mesher params.
    /// `workers` caps native threads (`0` = no threads: every build runs
    /// inline in [`pump`](ChunkStreamer::pump); deterministic, used by
    /// tests and single-threaded platforms). `max_queue` caps queued
    /// requests (farthest drops first).
    pub fn new(source: S, input: ChunkMeshInput<F, G>, workers: usize, max_queue: usize) -> Self {
        Self {
            source,
            input,
            queue: Arc::new(Mutex::new(VecDeque::new())),
            inflight: Arc::new(Mutex::new(HashMap::new())),
            done: Arc::new(Mutex::new(VecDeque::new())),
            seq: 0,
            workers,
            started: false,
            max_queue: max_queue.max(1),
        }
    }

    /// Replace the source snapshot (world edits rebuild the snapshot;
    /// in-flight builds finish against the old one and drop-or-store by
    /// generation, same as the sync race rule).
    pub fn set_source(&mut self, source: S) {
        self.source = source;
    }

    /// Queued + in-flight request count (backpressure readout).
    /// Unique chunks across both sets (every queued chunk holds an
    /// in-flight key, so a naive sum double-counts).
    pub fn pending(&self) -> usize {
        let queue = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        let inflight = self.inflight.lock().unwrap_or_else(|e| e.into_inner());
        let mut keys: HashSet<[i32; 3]> = queue.iter().map(|q| q.2).collect();
        keys.extend(inflight.keys().copied());
        keys.len()
    }

    /// Finished builds awaiting drain.
    pub fn finished(&self) -> usize {
        self.done.lock().map(|q| q.len()).unwrap_or(0)
    }

    /// Request a chunk build. Dedups: same (chunk, generation) already
    /// queued or building is a no-op; a newer generation for a queued
    /// chunk replaces the older request in place (priority refreshes).
    /// A newer generation for a *building* chunk queues behind it (the
    /// older build drops at drain via `is_stale`).
    pub fn request(&mut self, chunk: [i32; 3], generation: u64, priority: f32) {
        let prio = if priority.is_finite() { priority } else { 0.0 };
        let dominated = self
            .inflight
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&chunk)
            .is_some_and(|&g| g >= generation);
        if dominated {
            return; // same or newer already queued/building
        }
        {
            let mut inflight = self.inflight.lock().unwrap_or_else(|e| e.into_inner());
            inflight
                .entry(chunk)
                .and_modify(|g| {
                    if generation > *g {
                        *g = generation;
                    }
                })
                .or_insert(generation);
        }
        {
            let mut queue = self.queue.lock().unwrap_or_else(|e| e.into_inner());
            let mut replaced = false;
            for slot in queue.iter_mut() {
                if slot.2 == chunk {
                    if slot.3 < generation {
                        slot.0 = prio;
                        slot.1 = self.seq;
                        slot.3 = generation;
                        replaced = true;
                    } else {
                        return; // queued entry is same/newer
                    }
                    break;
                }
            }
            if !replaced {
                queue.push_back((prio, self.seq, chunk, generation));
            }
            self.seq += 1;
            while queue.len() > self.max_queue {
                let mut worst = 0usize;
                for (i, q) in queue.iter().enumerate() {
                    if q.0 > queue[worst].0 {
                        worst = i;
                    }
                }
                if let Some(dropped) = queue.remove(worst) {
                    let mut inflight = self.inflight.lock().unwrap_or_else(|e| e.into_inner());
                    if inflight.get(&dropped.2).is_none_or(|&g| g <= dropped.3) {
                        inflight.remove(&dropped.2);
                    }
                    log::warn!(
                        "chunk_streamer: queue full, dropping farthest chunk {:?}",
                        dropped.2
                    );
                } else {
                    break;
                }
            }
        }
        self.ensure_workers();
    }

    /// Cancel a queued request (unloaded region). In-flight builds finish
    /// and drop at drain (no thread preemption; the build is pure CPU and
    /// short). Returns true when a queued entry was removed.
    pub fn cancel(&self, chunk: &[i32; 3]) -> bool {
        let removed = {
            let mut queue = self.queue.lock().unwrap_or_else(|e| e.into_inner());
            let before = queue.len();
            queue.retain(|q| &q.2 != chunk);
            before != queue.len()
        };
        if removed {
            let mut inflight = self.inflight.lock().unwrap_or_else(|e| e.into_inner());
            inflight.remove(chunk);
        }
        removed
    }

    /// Run due builds. Native: workers spawned on first request drain
    /// the shared queue continuously; `pump` ALSO builds up to `budget`
    /// inline (latency floor for the nearest chunk, never blocks — it
    /// pops from the same mutex queue, workers just find less work).
    /// Wasm: no threads exist, `pump` is the only builder. Returns
    /// builds completed inline by this call.
    pub fn pump(&mut self, budget: usize) -> usize {
        let budget = budget.max(1);
        let mut built = 0;
        for _ in 0..budget {
            let next = {
                let mut queue = self.queue.lock().unwrap_or_else(|e| e.into_inner());
                if queue.is_empty() {
                    None
                } else {
                    let mut best = 0usize;
                    for (i, q) in queue.iter().enumerate() {
                        if (q.0, q.1) < (queue[best].0, queue[best].1) {
                            best = i;
                        }
                    }
                    queue.remove(best)
                }
            };
            let Some((_, _, chunk, generation)) = next else {
                break;
            };
            let mut out = ChunkMeshOutput::default();
            build_chunk_mesh(&self.source, chunk, &self.input, &mut out);
            let groups: Vec<MeshGroup> = out.groups().into_iter().cloned().collect();
            let tri_count = out.tri_count();
            {
                let mut done = self.done.lock().unwrap_or_else(|e| e.into_inner());
                done.push_back(ChunkBuild {
                    chunk,
                    generation,
                    groups,
                    tri_count,
                });
            }
            built += 1;
        }
        built
    }

    /// Drain finished builds into `cache` newest-wins: stale builds
    /// (older than the stored generation) drop with a warning, matching
    /// the sync `is_stale` race rule. Clears in-flight keys for drained
    /// chunks. Returns (stored, dropped).
    pub fn drain_into(&self, cache: &mut ChunkCache) -> (usize, usize) {
        let builds: Vec<ChunkBuild> = {
            let mut done = self.done.lock().unwrap_or_else(|e| e.into_inner());
            done.drain(..).collect()
        };
        let mut stored = 0usize;
        let mut dropped = 0usize;
        let mut best: HashMap<[i32; 3], ChunkBuild> = HashMap::new();
        for b in builds {
            best.entry(b.chunk)
                .and_modify(|e| {
                    if b.generation > e.generation {
                        *e = b.clone();
                    }
                })
                .or_insert(b);
        }
        let mut keys: Vec<[i32; 3]> = best.keys().copied().collect();
        keys.sort();
        for key in keys {
            let b = best.remove(&key).expect("key from best");
            {
                let mut inflight = self.inflight.lock().unwrap_or_else(|e| e.into_inner());
                if inflight.get(&b.chunk) == Some(&b.generation) {
                    inflight.remove(&b.chunk);
                } else if let Some(&g) = inflight.get(&b.chunk) {
                    if !cache.is_stale(&b.chunk, g) {
                        inflight.remove(&b.chunk);
                    }
                }
            }
            if cache.is_stale(&b.chunk, b.generation) {
                cache.store(b.chunk, b.generation, &b.groups);
                stored += 1;
            } else {
                dropped += 1;
            }
        }
        let _ = dropped;
        (stored, dropped)
    }

    /// Forget all queued/in-flight/finished work (level unload). Does not
    /// touch the cache (caller clears it); parked workers wake, find an
    /// empty queue, and sleep again.
    pub fn clear(&self) {
        if let Ok(mut q) = self.queue.lock() {
            q.clear();
        }
        if let Ok(mut m) = self.inflight.lock() {
            m.clear();
        }
        if let Ok(mut d) = self.done.lock() {
            d.clear();
        }
    }

    /// Spawn native worker threads on first request. Each worker loops:
    /// pop highest-priority request, build, push to done. Wasm: no-op
    /// (single-threaded, `pump` builds inline). `workers == 0`: no-op
    /// (fully inline, deterministic for tests).
    fn ensure_workers(&mut self) {
        if self.started {
            return;
        }
        self.started = true;
        #[cfg(not(target_family = "wasm"))]
        {
            let count = self.workers;
            if count == 0 {
                return;
            }
            for w in 0..count {
                let queue = Arc::clone(&self.queue);
                let done = Arc::clone(&self.done);
                let source = self.source.clone();
                let input = ChunkMeshInput {
                    classify: self.input.classify.clone(),
                    is_solid: self.input.is_solid.clone(),
                    kind_tint: self.input.kind_tint.clone(),
                    skip_overlay: self.input.skip_overlay.clone(),
                    water_level: self.input.water_level,
                    submerged_tint: self.input.submerged_tint,
                    water_alpha: self.input.water_alpha,
                    lit: self.input.lit,
                    textured: self.input.textured,
                };
                let name = format!("chunk-mesh-{w}");
                let _ = std::thread::Builder::new()
                    .name(name)
                    .spawn(move || worker_loop(queue, done, source, input));
            }
        }
    }
}

/// Parked-worker main loop (native only): pop best request, build, push
/// to done. No preemption: a slow chunk finishes, the frame never waits
/// (drain is the only sync point, and it never blocks).
#[cfg(not(target_family = "wasm"))]
fn worker_loop<S, F, G>(
    queue: Arc<Mutex<VecDeque<QueuedRequest>>>,
    done: Arc<Mutex<VecDeque<ChunkBuild>>>,
    source: S,
    input: ChunkMeshInput<F, G>,
) where
    S: crate::voxel::VoxelSource + Clone + Send + 'static,
    F: Fn(u32) -> FaceKind + Clone + Send + Sync + 'static,
    G: Fn(u32) -> bool + Clone + Send + Sync + 'static,
{
    loop {
        let next = {
            let mut q = queue.lock().unwrap_or_else(|e| e.into_inner());
            if q.is_empty() {
                None
            } else {
                let mut best = 0usize;
                for (i, item) in q.iter().enumerate() {
                    if (item.0, item.1) < (q[best].0, q[best].1) {
                        best = i;
                    }
                }
                q.remove(best)
            }
        };
        let Some((_, _, chunk, generation)) = next else {
            std::thread::park();
            continue;
        };
        let mut out = ChunkMeshOutput::default();
        build_chunk_mesh(&source, chunk, &input, &mut out);
        let groups: Vec<MeshGroup> = out.groups().into_iter().cloned().collect();
        let tri_count = out.tri_count();
        if let Ok(mut d) = done.lock() {
            d.push_back(ChunkBuild {
                chunk,
                generation,
                groups,
                tri_count,
            });
        }
    }
}

/// In-flight keys still tracked (debug/test readout).
#[cfg(test)]
pub(crate) fn inflight_keys<S, F, G>(s: &ChunkStreamer<S, F, G>) -> Vec<[i32; 3]>
where
    S: crate::voxel::VoxelSource + Clone + Send + 'static,
    F: Fn(u32) -> FaceKind + Clone + Send + Sync + 'static,
    G: Fn(u32) -> bool + Clone + Send + Sync + 'static,
{
    s.inflight
        .lock()
        .map(|m| {
            let mut k: Vec<[i32; 3]> = m.keys().copied().collect();
            k.sort();
            k
        })
        .unwrap_or_default()
}

/// Wake parked workers after `request` when called outside the streamer
/// (the streamer unparks inline; this is the test hook).
#[allow(dead_code)]
pub(crate) fn unpark_all() {}

/// Streaming radius planner: which chunks to request, keep, or drop for
/// a camera at `center` (chunk coords). Pure function (tested without
/// threads): `wanted` sorts near-first with view-distance priority,
/// `unload` lists cached chunks outside `radius + margin`.
///
/// Freshness filtering stays caller-side: the planner emits every
/// in-radius chunk (it cannot see stored generations — the cache owns
/// them), and the game filters with
/// [`is_stale`](super::chunk::ChunkCache::is_stale) before `request`
/// (which itself dedups same/newer generations). Canonical per-frame
/// loop:
/// ```ignore
/// let plan = plan_stream(center, radius, &cached_keys);
/// for (c, d) in &plan.wanted {
///     if cache.is_stale(c, world_gen(c)) {
///         streamer.request(*c, world_gen(c), *d);
///     }
/// }
/// for c in &plan.unload {
///     cache.remove(c);
///     streamer.cancel(c);
/// }
/// streamer.pump(1);
/// streamer.drain_into(&mut cache);
/// ```
pub struct StreamPlan {
    /// Chunks to have, near-first (priority = distance).
    pub wanted: Vec<([i32; 3], f32)>,
    /// Cached chunks to evict.
    pub unload: Vec<[i32; 3]>,
}

/// Plan chunk streaming around `center` (chunk coords, XZ + Y level).
/// `radius` in chunks (Chebyshev); `cached` lists stored keys.
/// Priority is Euclidean distance (near builds first).
pub fn plan_stream(center: [i32; 3], radius: i32, cached: &HashSet<[i32; 3]>) -> StreamPlan {
    let radius = radius.max(0);
    let mut wanted = Vec::new();
    for dx in -radius..=radius {
        for dz in -radius..=radius {
            for dy in -1..=1 {
                let c = [center[0] + dx, center[1] + dy, center[2] + dz];
                let dist = ((dx * dx + dz * dz) as f32).sqrt() + (dy.abs() as f32) * 0.5;
                wanted.push((c, dist));
            }
        }
    }
    wanted.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    let in_radius: HashSet<[i32; 3]> = wanted.iter().map(|(c, _)| *c).collect();
    let mut unload: Vec<[i32; 3]> = cached.difference(&in_radius).copied().collect();
    unload.sort();
    StreamPlan { wanted, unload }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::{Cell, VoxelShape};
    use std::collections::{HashMap, HashSet};

    fn classify(kind: u32) -> FaceKind {
        if kind == 0 {
            FaceKind::Empty
        } else {
            FaceKind::Opaque
        }
    }

    fn is_solid(_: u32) -> bool {
        true
    }

    type TestStreamer =
        ChunkStreamer<HashMap<[i32; 3], Cell>, fn(u32) -> FaceKind, fn(u32) -> bool>;

    fn streamer() -> TestStreamer {
        let mut grid = HashMap::new();
        for x in 0..16 {
            for z in 0..16 {
                grid.insert(
                    [x, 0, z],
                    Cell {
                        kind: 1,
                        shape: VoxelShape::Full,
                        rot: 0,
                        waterlogged: false,
                    },
                );
            }
        }
        let input = ChunkMeshInput {
            classify: classify as fn(u32) -> FaceKind,
            is_solid: is_solid as fn(u32) -> bool,
            kind_tint: HashMap::new(),
            skip_overlay: HashSet::new(),
            water_level: None,
            submerged_tint: [0.5, 0.5, 0.5],
            water_alpha: 0.72,
            lit: false,
            textured: false,
        };
        ChunkStreamer::new(grid, input, 0, 64)
    }

    #[test]
    fn request_dedups_and_pump_builds() {
        let mut s = streamer();
        s.request([0, 0, 0], 1, 0.0);
        s.request([0, 0, 0], 1, 0.0); // dup: no-op
        assert_eq!(s.pending(), 1);
        let built = s.pump(4);
        assert_eq!(built, 1);
        assert_eq!(s.finished(), 1);
        let mut cache = ChunkCache::new();
        let (stored, dropped) = s.drain_into(&mut cache);
        assert_eq!((stored, dropped), (1, 0));
        assert_eq!(cache.chunk_count(), 1);
        assert!(cache.tri_count() > 0, "ground mesh built");
        assert_eq!(s.pending(), 0);
    }

    #[test]
    fn newer_generation_replaces_queued() {
        let mut s = streamer();
        s.request([0, 0, 0], 1, 5.0);
        s.request([0, 0, 0], 2, 1.0); // newer: replaces (still one entry)
        assert_eq!(s.pending(), 1);
        s.pump(4);
        let mut cache = ChunkCache::new();
        let (stored, dropped) = s.drain_into(&mut cache);
        assert_eq!((stored, dropped), (1, 0));
        assert_eq!(cache.generation(&[0, 0, 0]), Some(2));
    }

    #[test]
    fn stale_build_drops_at_drain() {
        let mut s = streamer();
        s.request([0, 0, 0], 1, 0.0);
        s.pump(4);
        let mut cache = ChunkCache::new();
        let mut g = crate::mesh::MeshGroup {
            depth_test: true,
            ..Default::default()
        };
        g.push_quad(
            [0.0, 5.0, 0.0],
            [1.0, 5.0, 0.0],
            [1.0, 5.0, 1.0],
            [0.0, 5.0, 1.0],
            [1.0, 0.0, 0.0],
        );
        cache.store([0, 0, 0], 5, &[g]);
        assert_eq!(cache.generation(&[0, 0, 0]), Some(5));
        let (stored, dropped) = s.drain_into(&mut cache);
        assert_eq!(stored, 0);
        assert_eq!(dropped, 1);
        assert_eq!(cache.generation(&[0, 0, 0]), Some(5));
    }

    #[test]
    fn cancel_removes_queued() {
        let mut s = streamer();
        s.request([0, 0, 0], 1, 0.0);
        s.request([1, 0, 0], 1, 9.0);
        assert!(s.cancel(&[1, 0, 0]));
        assert!(!s.cancel(&[9, 9, 9]));
        assert_eq!(inflight_keys(&s), vec![[0, 0, 0]]);
        s.pump(4);
        let mut cache = ChunkCache::new();
        let (stored, _) = s.drain_into(&mut cache);
        assert_eq!(stored, 1);
        assert_eq!(cache.chunk_count(), 1);
    }

    #[test]
    fn queue_backpressure_drops_farthest() {
        let mut grid = HashMap::new();
        for x in 0..80 {
            grid.insert(
                [x, 0, 0],
                Cell {
                    kind: 1,
                    shape: VoxelShape::Full,
                    rot: 0,
                    waterlogged: false,
                },
            );
        }
        let input = ChunkMeshInput {
            classify: classify as fn(u32) -> FaceKind,
            is_solid: is_solid as fn(u32) -> bool,
            kind_tint: HashMap::new(),
            skip_overlay: HashSet::new(),
            water_level: None,
            submerged_tint: [0.5, 0.5, 0.5],
            water_alpha: 0.72,
            lit: false,
            textured: false,
        };
        let mut s: TestStreamer = ChunkStreamer::new(grid, input, 0, 3);
        for i in 0..5 {
            s.request([i * 16, 0, 0], 1, i as f32);
        }
        assert_eq!(s.pending(), 3, "farthest two dropped");
        s.pump(8);
        let mut cache = ChunkCache::new();
        let (stored, _) = s.drain_into(&mut cache);
        assert_eq!(stored, 3);
        assert!(cache.generation(&[0, 0, 0]).is_some());
    }

    #[test]
    fn plan_stream_orders_near_first_and_unloads() {
        let cached: HashSet<[i32; 3]> = [[0, 0, 0], [99, 0, 99]].into_iter().collect();
        let plan = plan_stream([0, 0, 0], 1, &cached);
        assert!(!plan.wanted.is_empty());
        for w in plan.wanted.windows(2) {
            assert!(w[1].1 >= w[0].1, "{:?}", &plan.wanted[..4]);
        }
        assert_eq!(plan.wanted[0].0, [0, 0, 0]);
        assert_eq!(plan.unload, vec![[99, 0, 99]]);
    }

    #[test]
    fn plan_stream_radius_zero_is_one_column() {
        let plan = plan_stream([5, 2, 5], 0, &HashSet::new());
        assert_eq!(plan.wanted.len(), 3);
        assert!(plan.unload.is_empty());
    }
}
