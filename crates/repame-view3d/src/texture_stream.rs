//! Streamed texture mip residency: budget, desired levels, eviction, I/O.
//!
//! A streamed texture is a mip chain ([`TextureChain`], one page per texture)
//! whose levels arrive one at a time through the same job pattern as chunk
//! streaming ([`super::stream`]): priority, dedup, generations, and stale-drop
//! at drain time. The level a texture *wants* comes from its projected
//! on-screen size ([`project_bounds`] over the camera and the group bounds),
//! with hysteresis so a texture sitting on a threshold does not thrash
//! ([`select_mip_level`]); a byte budget coarsens the longest-unrequested
//! texture first and never drops a chain's smallest (coarsest) level.
//!
//! The design follows Godot's `modules/texture_streaming`: a request mip
//! derived from how many texels one screen pixel covers, a memory budget that
//! raises the least important textures one mip at a time inside a deadband
//! (Godot's ±5%), per-texture feedback smoothed so a jittering camera does not
//! reload every frame, and a reload queued only when the resident mip differs
//! from the fitted one. Godot reads its request mip from shader uv feedback;
//! this crate has no feedback pass, so the projected AABB stands in for it.
//!
//! # Policy
//!
//! * **Desired level** — `ratio = level0_width / projected_px` is the texture's
//!   texels per screen pixel. Level `n` is wanted while
//!   `2^n <= ratio < 2^(n+1)`: the finest level whose texels are still at
//!   least one screen pixel wide.
//! * **Hysteresis** — a level only changes when the ratio leaves the band
//!   `[2^n * (1 - h), 2^(n+1) * (1 + h))` around the current level, so drifting
//!   across a threshold keeps the level. The band is symmetric: coarsening
//!   needs the upper edge, refining the lower one, and neither re-fires while
//!   the ratio oscillates inside it.
//! * **Budget** — the plan charges each texture's wanted range
//!   `wanted..coarsest`. Above `budget * (1 + slack)` it coarsens one texture
//!   at a time — the longest-unrequested first (`frame - last_seen`), ties
//!   broken by the largest wanted footprint, then the lowest page — until the
//!   charge is back at `budget * (1 - slack)`. A chain's smallest level is
//!   never dropped, so a texture stays sampleable once its coarsest level has
//!   arrived; when even minimum quality does not fit, the policy holds and
//!   logs once instead of shrinking any further.
//! * **Over-quality** — a resident level finer than the wanted range (the
//!   policy coarsened after it arrived, or it arrived late) is dropped first,
//!   so the promised residency and the budget's charge always agree. Godot's
//!   fit does the same by reloading a reduced texture from its fitted mip.
//!
//! # Never a fabricated sample
//!
//! [`TextureResidency::resolve_level`] answers with the finest *resident*
//! level. `None` means nothing is resident at all: the caller draws no texture
//! for that page (its tint), never an invented level and never an unwritten
//! mip. A level that has not arrived yet resolves to the finest resident level
//! and logs once per missing episode; a level whose load failed is remembered
//! and not requested again until the page is re-declared. Texels that do not
//! match the chain's declared extent and byte length are refused, so nothing
//! partial reaches the GPU.
//!
//! # Render thread
//!
//! Loading runs off-thread; everything that touches render state is explicit.
//! [`MipStreamer::drain_into`] stores finished levels during the frame and
//! [`TextureResidency::take_uploads`] hands them to
//! [`SceneBatch::upload_mip`](super::render::SceneBatch::upload_mip), which
//! writes each level into its own mip of the batch's array texture on the
//! render thread. Generation checks make a late arrival harmless: it drops
//! exactly like a stale chunk build. Texture contents do not survive a batch
//! rebuild, so the game calls [`TextureResidency::invalidate`] for the pages
//! the rebuild dropped.
//!
//! A game drawing through [`Viewport3d`](super::viewport::Viewport3d) stages
//! the same drain on its [`Frame3d`](super::viewport::Frame3d) instead of a
//! [`SceneBatch`](super::render::SceneBatch):
//! [`extend_mip_uploads`](super::viewport::Frame3d::extend_mip_uploads) takes
//! the `take_uploads` vec and
//! [`publish_page_level`](super::viewport::Frame3d::publish_page_level) takes
//! each [`resolve_level`](TextureResidency::resolve_level) answer; `prepare`
//! applies both to the batch unchanged, so the sampling contract below holds
//! on either path.
//!
//! ```ignore
//! // game thread, once per frame, per visible group:
//! let Some((min, max)) = group_bounds(group) else { continue };
//! let Some(size) = project_bounds(view_proj, viewport_px, min.into(), max.into()) else { continue };
//! residency.touch(group.texture_page, size.max_px(), frame);
//! // publish the level the shaders sample for this page (None = untextured)
//! batch.set_page_level(group.texture_page, residency.resolve_level(group.texture_page));
//!
//! let plan = residency.plan(frame);
//! for request in &plan.requests {
//!     streamer.request(*request);
//! }
//! for eviction in &plan.evictions {
//!     log::debug!("dropping page {} level {}", eviction.page, eviction.level);
//! }
//! streamer.pump(1);
//! // render thread, after `SceneBatch::finish`:
//! let (stored, dropped) = streamer.drain_into(&mut residency);
//! for upload in residency.take_uploads() {
//!     batch.upload_mip(upload);
//! }
//! // a drain can change a page's finest resident level; republish it on the
//! // next frame's touch (the loop above does), or immediately for pages that
//! // are not touched every frame.
//! ```
//!
//! # Sampling contract
//!
//! [`TextureResidency::resolve_level`] is the level to publish with
//! [`SceneBatch::set_page_level`](super::render::SceneBatch::set_page_level):
//! the finest level actually resident, or `None` when nothing is resident.
//! Every textured site samples exactly that level — the explicit sites and
//! the material programs' implicit-LOD samples alike read it, never a finer
//! (unwritten) mip — and a page with nothing resident draws untextured (its
//! tint, the identity texel in a material program) instead of a fabricated
//! sample, so a page stays drawable while its levels stream in. A partially
//! resident page therefore samples the finest level it *has*, not an implicit
//! blend whose finer levels never arrived. A page the
//! game never publishes stays at level 0, the fully resident page, which is
//! the historical behavior; residency can be published for the first
//! [`MAX_RESIDENT_PAGES`](super::render::MAX_RESIDENT_PAGES) pages (256),
//! which covers every material page (`u8`) a draw can name. The wanted range
//! is `wanted..=coarsest`, which is what the manager loads and keeps; the
//! finest level can only be coarser than `wanted` while levels are missing.
//!
//! # Presentation only
//!
//! Residency is a rendering decision: it must not feed sim state, must not
//! change the order in which jobs finish, and must not influence determinism.
//! A game's snapshot does not carry residency — rebuild it from the camera and
//! the budget after a load, and never serialize it.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use glam::{Mat4, Vec4};

use super::render::MipUpload;

/// Levels a streamed chain may hold. 16 covers a 32k texture (dimensions halve
/// to 1x1) and keeps every residency bitmask inside one `u32`.
pub const MAX_MIP_LEVELS: u32 = 16;

/// Residency bit for one level; 0 for a level outside the bitmask.
fn level_bit(level: u32) -> u32 {
    if level < 32 { 1u32 << level } else { 0 }
}

/// Why a chain was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamError {
    /// A chain needs at least one level.
    EmptyChain,
    /// More levels than [`MAX_MIP_LEVELS`].
    TooManyLevels(usize),
    /// A level has a zero extent.
    ZeroExtent(u32),
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyChain => write!(f, "texture chain has no levels"),
            Self::TooManyLevels(n) => {
                write!(f, "{n} mip levels past the {MAX_MIP_LEVELS} level cap")
            }
            Self::ZeroExtent(level) => write!(f, "mip level {level} has a zero extent"),
        }
    }
}

impl std::error::Error for StreamError {}

/// One level of a streamed texture, in decoded texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MipLevelDesc {
    pub width: u32,
    pub height: u32,
    /// Decoded texel bytes at this level (`width * height * 4` for RGBA8).
    pub bytes: u64,
}

/// Declared mip chain of one streamed texture.
///
/// `page` is the identity in the residency manager and the array layer the
/// levels upload into ([`SceneBatch::upload_mip`](super::render::SceneBatch::upload_mip)),
/// matching [`SceneUpload::page`](super::render::SceneUpload). Level 0 is the
/// finest (full size); the last entry is the smallest level, which eviction
/// never drops. A chain describes a `BatchDesc::layer_size` page when its
/// level 0 is the page edge ([`TextureChain::new`] with `layer_size`), and the
/// page must declare at least as many `mips` as the chain has levels, so every
/// streamed level is a level the texture actually has.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextureChain {
    pub page: u32,
    /// Level 0 first, every level half the previous one.
    pub levels: Vec<MipLevelDesc>,
}

impl TextureChain {
    /// Standard chain for a `width x height` RGBA8 texture: each level is half
    /// the previous one, ending at 1x1. Level extents match the container
    /// parsers' `>> level` arithmetic, so a chain built from a real texture's
    /// level table lines up byte for byte.
    pub fn new(page: u32, width: u32, height: u32) -> Result<Self, StreamError> {
        if width == 0 || height == 0 {
            return Err(StreamError::ZeroExtent(0));
        }
        let count = 32 - width.max(height).leading_zeros();
        if count > MAX_MIP_LEVELS {
            return Err(StreamError::TooManyLevels(count as usize));
        }
        let levels = (0..count)
            .map(|level| {
                let w = (width >> level).max(1);
                let h = (height >> level).max(1);
                MipLevelDesc {
                    width: w,
                    height: h,
                    bytes: u64::from(w) * u64::from(h) * 4,
                }
            })
            .collect();
        Ok(Self { page, levels })
    }

    /// Chain from explicit level extents (a container's level table), level 0
    /// first and RGBA8 byte counts.
    pub fn from_dimensions(
        page: u32,
        levels: impl IntoIterator<Item = (u32, u32)>,
    ) -> Result<Self, StreamError> {
        let mut out = Vec::new();
        for (index, (width, height)) in levels.into_iter().enumerate() {
            if width == 0 || height == 0 {
                return Err(StreamError::ZeroExtent(index as u32));
            }
            out.push(MipLevelDesc {
                width,
                height,
                bytes: u64::from(width) * u64::from(height) * 4,
            });
            if out.len() > MAX_MIP_LEVELS as usize {
                return Err(StreamError::TooManyLevels(out.len()));
            }
        }
        if out.is_empty() {
            return Err(StreamError::EmptyChain);
        }
        Ok(Self { page, levels: out })
    }

    pub fn level_count(&self) -> u32 {
        self.levels.len() as u32
    }

    /// The smallest level (the last one), which eviction never drops.
    pub fn coarsest(&self) -> u32 {
        self.level_count() - 1
    }

    pub fn level(&self, level: u32) -> Option<&MipLevelDesc> {
        self.levels.get(level as usize)
    }

    pub fn bytes_at(&self, level: u32) -> Option<u64> {
        self.level(level).map(|desc| desc.bytes)
    }

    /// Bytes of levels `level..=coarsest`.
    pub fn bytes_from(&self, level: u32) -> u64 {
        self.levels
            .iter()
            .skip(level as usize)
            .map(|desc| desc.bytes)
            .sum()
    }

    pub fn bytes(&self) -> u64 {
        self.levels.iter().map(|desc| desc.bytes).sum()
    }

    /// Level this chain wants for a projected on-screen size in pixels, biased
    /// by `previous` through [`select_mip_level`].
    pub fn desired_level(&self, projected_px: f32, previous: Option<u32>, hysteresis: f32) -> u32 {
        let width = self.levels.first().map(|desc| desc.width).unwrap_or(1);
        desired_level(width, projected_px, previous, hysteresis, self.coarsest())
    }
}

/// `2^n` as an `f32`. `n` never exceeds [`MAX_MIP_LEVELS`].
fn pow2(n: u32) -> f32 {
    (1u64 << n.min(31)) as f32
}

/// Pure level policy: `ratio` is level-0 texels per projected screen pixel
/// (`level0_width / projected_px`).
///
/// Without `previous`, the level is `floor(log2(ratio))` clamped to
/// `max_level`. With `previous`, a change must clear the hysteresis band
/// around the boundary it crosses — coarsening at or above
/// `2^(previous+1) * (1 + h)`, refining below `2^previous * (1 - h)` — so a
/// ratio oscillating around a threshold keeps its level (see the module docs).
/// A non-finite ratio keeps the previous level (or level 0); a zero ratio asks
/// for the finest level. With `hysteresis = 0` the rule is exact except that
/// reaching the coarsening boundary and standing exactly on it is a change.
pub fn select_mip_level(
    ratio: f32,
    previous: Option<u32>,
    hysteresis: f32,
    max_level: u32,
) -> u32 {
    let max_level = max_level.min(MAX_MIP_LEVELS - 1);
    if !ratio.is_finite() {
        return previous.unwrap_or(0).min(max_level);
    }
    let target = if ratio > 0.0 {
        (ratio.log2().floor()).clamp(0.0, max_level as f32) as u32
    } else {
        0
    };
    let hysteresis = if hysteresis.is_finite() {
        hysteresis.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let Some(previous) = previous.filter(|level| *level <= max_level) else {
        return target;
    };
    if target == previous {
        return previous;
    }
    if target > previous {
        if ratio >= pow2(previous + 1) * (1.0 + hysteresis) {
            target
        } else {
            previous
        }
    } else if ratio < pow2(previous) * (1.0 - hysteresis) {
        target
    } else {
        previous
    }
}

/// The level a `level0_width` chain wants for a projected on-screen size in
/// pixels, biased by `previous` through [`select_mip_level`]. See
/// [`TextureChain::desired_level`] for the chain-bound version.
pub fn desired_level(
    level0_width: u32,
    projected_px: f32,
    previous: Option<u32>,
    hysteresis: f32,
    max_level: u32,
) -> u32 {
    let width = level0_width.max(1) as f32;
    let projected = if projected_px.is_finite() {
        projected_px.max(1.0)
    } else {
        1.0
    };
    select_mip_level(width / projected, previous, hysteresis, max_level)
}

/// Projected extent of a world-space AABB, in physical pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectedSize {
    pub width_px: f32,
    pub height_px: f32,
}

impl ProjectedSize {
    /// Larger screen extent: the size the mip policy keys on.
    pub fn max_px(self) -> f32 {
        self.width_px.max(self.height_px)
    }
}

/// Project the eight corners of `min..max` through `view_proj` and return the
/// AABB's on-screen extent in pixels.
///
/// `None` when any corner is at or behind the eye, the viewport is degenerate,
/// or a coordinate is not finite: the caller keeps the level it had instead of
/// guessing, so a bound straddling the camera never fabricates a size (and
/// therefore never fabricates a level).
pub fn project_bounds(
    view_proj: Mat4,
    viewport_px: [f32; 2],
    min: [f32; 3],
    max: [f32; 3],
) -> Option<ProjectedSize> {
    if !(viewport_px[0] > 0.0) || !(viewport_px[1] > 0.0) {
        return None;
    }
    let (mut x0, mut x1, mut y0, mut y1) = (
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
    );
    for corner in [
        [min[0], min[1], min[2]],
        [min[0], min[1], max[2]],
        [min[0], max[1], min[2]],
        [min[0], max[1], max[2]],
        [max[0], min[1], min[2]],
        [max[0], min[1], max[2]],
        [max[0], max[1], min[2]],
        [max[0], max[1], max[2]],
    ] {
        let clip = view_proj * Vec4::new(corner[0], corner[1], corner[2], 1.0);
        if !(clip.w > 0.0) {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        let x = (ndc.x * 0.5 + 0.5) * viewport_px[0];
        let y = (0.5 - ndc.y * 0.5) * viewport_px[1];
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        x0 = x0.min(x);
        x1 = x1.max(x);
        y0 = y0.min(y);
        y1 = y1.max(y);
    }
    Some(ProjectedSize {
        width_px: (x1 - x0).max(0.0),
        height_px: (y1 - y0).max(0.0),
    })
}

/// Priority for a level request. Larger loads first — a texture covering more
/// screen pixels buys more detail per byte; equal priorities keep submission
/// order, so the drain is stable ([`ChunkPriority`](crate::ChunkPriority) is
/// the same idea with the near-first sign).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MipPriority(pub f32);

/// One level load for the [`MipStreamer`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MipRequest {
    pub page: u32,
    pub level: u32,
    /// Generation the residency issued; arrivals carry it back.
    pub generation: u64,
    pub priority: MipPriority,
}

/// One resident level the policy dropped.
///
/// Eviction is a residency decision, not a GPU operation: wgpu has no "unwrite
/// a mip", so the texels stay in the texture until something overwrites them.
/// After an eviction the manager no longer promises that level, so
/// [`TextureResidency::resolve_level`] will not report it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MipEviction {
    pub page: u32,
    pub level: u32,
    pub bytes: u64,
}

/// What one [`TextureResidency::plan`] wants the frame to do.
#[derive(Clone, Debug, Default)]
pub struct TextureStreamPlan {
    /// Loads, highest priority first.
    pub requests: Vec<MipRequest>,
    /// Resident levels the policy dropped, in the order it chose them.
    pub evictions: Vec<MipEviction>,
}

/// Policy knobs for [`TextureResidency`].
#[derive(Clone, Copy, Debug)]
pub struct StreamTuning {
    /// Fractional band a level boundary must be crossed by before the level
    /// changes ([`select_mip_level`]).
    pub hysteresis: f32,
    /// Budget deadband: reduce above `budget * (1 + slack)`, stop at
    /// `budget * (1 - slack)` (Godot's ±5% fit).
    pub budget_slack: f32,
    /// Frames an unanswered request may stay pending before it is re-issued.
    pub pending_timeout: u64,
}

impl Default for StreamTuning {
    fn default() -> Self {
        Self {
            hysteresis: 0.25,
            budget_slack: 0.05,
            pending_timeout: 120,
        }
    }
}

/// One level's decoded texels, as a loader returns them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MipPayload {
    pub width: u32,
    pub height: u32,
    /// Tight texels, `width * height * 4` bytes for RGBA8.
    pub texels: Vec<u8>,
}

/// One finished load, ready to store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MipArrival {
    pub page: u32,
    pub level: u32,
    /// Generation the request carried. An older generation than the newest
    /// request for the level drops, exactly like a stale chunk build.
    pub generation: u64,
    /// `None` = the loader could not produce this level (unsupported format,
    /// missing source, decode failure); remembered so it is not retried every
    /// frame.
    pub payload: Option<MipPayload>,
}

/// Residency readout for stats overlays.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TextureStreamStats {
    pub textures: usize,
    /// Resident levels across every page.
    pub resident_levels: usize,
    /// Bytes those levels hold.
    pub resident_bytes: u64,
    pub budget_bytes: u64,
    /// Requested, not yet arrived.
    pub pending_levels: usize,
    /// Arrived levels waiting for [`TextureResidency::take_uploads`].
    pub queued_uploads: usize,
}

/// Request bookkeeping for one level: the newest generation ever requested
/// (stale arrivals drop against it) and the request still awaiting an answer.
#[derive(Clone, Copy, Debug, Default)]
struct LevelState {
    newest: u64,
    /// Generation and frame of the pending request, once issued.
    pending: Option<(u64, u64)>,
}

struct Resident {
    chain: TextureChain,
    /// Bit per resident level (`1 << level`).
    resident: u32,
    /// Level -> request bookkeeping. Only levels ever requested get an entry.
    requested: BTreeMap<u32, LevelState>,
    /// Levels the loader could not produce; not requested again.
    failed: BTreeSet<u32>,
    /// Size policy result from the last [`TextureResidency::touch`].
    policy: u32,
    /// Finest level the budget allows after the last plan.
    wanted: u32,
    /// Projected size from the last touch, the request priority.
    projected_px: f32,
    /// Frame of the last touch (eviction prefers the longest unrequested).
    last_seen: u64,
    /// Bytes of resident levels.
    bytes: u64,
    /// Logged the missing-level warning for the current episode.
    warned: bool,
}

impl Resident {
    fn new(chain: TextureChain) -> Self {
        let coarsest = chain.coarsest();
        Self {
            chain,
            resident: 0,
            requested: BTreeMap::new(),
            failed: BTreeSet::new(),
            // Start at minimum quality, like Godot's smoothed mip: the first
            // touch refines freely, and only a budget fit coarsens.
            policy: coarsest,
            wanted: coarsest,
            projected_px: 0.0,
            last_seen: 0,
            bytes: 0,
            warned: false,
        }
    }

    fn finest_resident(&self) -> Option<u32> {
        (0..self.chain.level_count()).find(|level| self.resident & level_bit(*level) != 0)
    }

    fn wanted_mask(&self) -> u32 {
        let mut mask = 0;
        for level in self.wanted..self.chain.level_count() {
            mask |= level_bit(level);
        }
        mask
    }

    fn wanted_resident(&self) -> bool {
        let mask = self.wanted_mask();
        self.resident & mask == mask
    }

    fn drop_level(&mut self, level: u32) -> Option<MipEviction> {
        let bit = level_bit(level);
        if bit == 0 || self.resident & bit == 0 {
            return None;
        }
        self.resident &= !bit;
        let bytes = self.chain.bytes_at(level).unwrap_or(0);
        self.bytes = self.bytes.saturating_sub(bytes);
        Some(MipEviction {
            page: self.chain.page,
            level,
            bytes,
        })
    }
}

/// Mip residency for streamed textures: desired levels, a byte budget, and
/// eviction. Pure state — no threads, no device, no clock of its own.
///
/// The game declares each page's chain, touches the visible ones with their
/// projected size, calls [`plan`](Self::plan) once per frame, feeds the
/// requests to a [`MipStreamer`], and stores the arrivals it drains back. See
/// the module docs for the canonical loop and the sampling contract.
pub struct TextureResidency {
    tuning: StreamTuning,
    budget_bytes: u64,
    textures: BTreeMap<u32, Resident>,
    uploads: Vec<MipUpload>,
    generations: u64,
    budget_warned: bool,
}

impl TextureResidency {
    /// New manager with a byte budget and the policy knobs.
    pub fn new(budget_bytes: u64, tuning: StreamTuning) -> Self {
        Self {
            tuning,
            budget_bytes,
            textures: BTreeMap::new(),
            uploads: Vec::new(),
            generations: 0,
            budget_warned: false,
        }
    }

    pub fn tuning(&self) -> StreamTuning {
        self.tuning
    }

    pub fn set_tuning(&mut self, tuning: StreamTuning) {
        self.tuning = tuning;
    }

    pub fn budget(&self) -> u64 {
        self.budget_bytes
    }

    pub fn set_budget(&mut self, budget_bytes: u64) {
        self.budget_bytes = budget_bytes;
    }

    /// Declare a page's mip chain. Re-declaring the same chain is a no-op;
    /// declaring a different one replaces the page's state (its residency is
    /// dropped, and the game re-uploads the levels it still holds).
    pub fn declare(&mut self, chain: TextureChain) -> Result<(), StreamError> {
        if chain.level_count() == 0 {
            return Err(StreamError::EmptyChain);
        }
        if chain.level_count() > MAX_MIP_LEVELS {
            return Err(StreamError::TooManyLevels(chain.level_count() as usize));
        }
        let page = chain.page;
        match self.textures.get(&page) {
            Some(existing) if existing.chain == chain => Ok(()),
            _ => {
                self.textures.insert(page, Resident::new(chain));
                Ok(())
            }
        }
    }

    /// Forget a page (unloaded texture).
    pub fn remove(&mut self, page: u32) -> bool {
        self.textures.remove(&page).is_some()
    }

    /// Forget every page. Arrivals for removed pages log and drop.
    pub fn clear(&mut self) {
        self.textures.clear();
    }

    pub fn chain(&self, page: u32) -> Option<&TextureChain> {
        self.textures.get(&page).map(|resident| &resident.chain)
    }

    /// The size policy's level for `page` (before any budget fit).
    pub fn policy_level(&self, page: u32) -> Option<u32> {
        self.textures.get(&page).map(|resident| resident.policy)
    }

    /// The finest level the budget last allowed for `page`.
    pub fn wanted_level(&self, page: u32) -> Option<u32> {
        self.textures.get(&page).map(|resident| resident.wanted)
    }

    pub fn is_resident(&self, page: u32, level: u32) -> bool {
        let bit = level_bit(level);
        bit != 0
            && self
                .textures
                .get(&page)
                .is_some_and(|resident| resident.resident & bit != 0)
    }

    /// Finest resident level (most detail available), if any.
    pub fn finest_resident(&self, page: u32) -> Option<u32> {
        self.textures.get(&page).and_then(Resident::finest_resident)
    }

    pub fn resident_bytes(&self) -> u64 {
        self.textures.values().map(|resident| resident.bytes).sum()
    }

    /// Record that `page` was visible at `projected_px` this frame and update
    /// its desired level through the hysteresis band. `None` for an undeclared
    /// page: declare it first, then touch.
    pub fn touch(&mut self, page: u32, projected_px: f32, frame: u64) -> Option<u32> {
        let Some(resident) = self.textures.get_mut(&page) else {
            return None;
        };
        resident.last_seen = frame;
        if projected_px.is_finite() {
            resident.projected_px = projected_px.max(0.0);
        }
        let hysteresis = self.tuning.hysteresis;
        let level = resident.chain.desired_level(
            resident.projected_px,
            Some(resident.policy),
            hysteresis,
        );
        resident.policy = level;
        Some(level)
    }

    /// Plan this frame: evictions first (over-quality levels, then budget
    /// coarsening), then the loads still missing.
    ///
    /// Mutates residency state (evicted levels stop being promised) and returns
    /// what it did, so a caller can log or test the decision. Feed
    /// [`TextureStreamPlan::requests`] to [`MipStreamer::request`] unchanged:
    /// the generations are what makes late arrivals harmless.
    pub fn plan(&mut self, frame: u64) -> TextureStreamPlan {
        let pages: Vec<u32> = self.textures.keys().copied().collect();
        let slack = if self.tuning.budget_slack.is_finite() {
            f64::from(self.tuning.budget_slack).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let mut wanted: BTreeMap<u32, u32> = BTreeMap::new();
        let mut assigned = 0u64;
        for page in &pages {
            if let Some(resident) = self.textures.get(page) {
                wanted.insert(*page, resident.policy);
                assigned = assigned.saturating_add(resident.chain.bytes_from(resident.policy));
            }
        }

        let mut evictions = Vec::new();
        // Over-quality first: a resident level finer than the wanted range is
        // pure waste (Godot's highest-priority reduction candidate).
        for page in &pages {
            let start = wanted.get(page).copied().unwrap_or(0);
            let Some(resident) = self.textures.get_mut(page) else {
                continue;
            };
            for level in 0..start {
                if let Some(eviction) = resident.drop_level(level) {
                    evictions.push(eviction);
                }
            }
        }

        // Budget fit inside the deadband, one level at a time.
        let budget = self.budget_bytes;
        let high = budget.saturating_add((budget as f64 * slack) as u64);
        let low = budget.saturating_sub((budget as f64 * slack) as u64);
        if assigned > high {
            loop {
                if assigned <= low {
                    break;
                }
                let Some(victim) = pick_victim(&self.textures, &wanted, frame) else {
                    if !self.budget_warned {
                        self.budget_warned = true;
                        log::warn!(
                            "texture_stream: budget {budget} bytes cannot be met at minimum quality ({assigned} bytes); holding the smallest levels"
                        );
                    }
                    break;
                };
                let level = wanted.get(&victim).copied().unwrap_or(0);
                let Some(resident) = self.textures.get_mut(&victim) else {
                    break;
                };
                wanted.insert(victim, level + 1);
                let freed = resident.chain.bytes_at(level).unwrap_or(0);
                assigned = assigned.saturating_sub(freed);
                if let Some(eviction) = resident.drop_level(level) {
                    evictions.push(eviction);
                }
            }
        } else {
            self.budget_warned = false;
        }

        let timeout = self.tuning.pending_timeout;
        let mut requests = Vec::new();
        for page in &pages {
            let Some(resident) = self.textures.get_mut(page) else {
                continue;
            };
            resident.wanted = wanted.get(page).copied().unwrap_or(resident.policy);
            // Requests that never arrived (backpressure, cancelled job) are
            // re-issued after the timeout instead of stranding the level.
            for state in resident.requested.values_mut() {
                if let Some((_, requested_frame)) = state.pending
                    && frame.saturating_sub(requested_frame) > timeout
                {
                    state.pending = None;
                }
            }
            let priority = MipPriority(resident.projected_px);
            for level in resident.wanted..resident.chain.level_count() {
                if resident.resident & level_bit(level) != 0 || resident.failed.contains(&level) {
                    continue;
                }
                let state = resident.requested.entry(level).or_default();
                if state.pending.is_some() {
                    continue;
                }
                self.generations += 1;
                state.newest = self.generations;
                state.pending = Some((self.generations, frame));
                requests.push(MipRequest {
                    page: *page,
                    level,
                    generation: self.generations,
                    priority,
                });
            }
        }
        requests.sort_by(|a, b| {
            b.priority
                .0
                .partial_cmp(&a.priority.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.page.cmp(&b.page))
                .then_with(|| a.level.cmp(&b.level))
        });
        TextureStreamPlan {
            requests,
            evictions,
        }
    }

    /// Store one arrived level. `false` when the arrival is stale, for an
    /// undeclared page or an unknown level, or mis-sized — every case logged
    /// and dropped whole, never stored partially.
    pub fn store(&mut self, arrival: MipArrival) -> bool {
        let Some(resident) = self.textures.get_mut(&arrival.page) else {
            log::warn!(
                "texture_stream: dropping arrival for undeclared page {}",
                arrival.page
            );
            return false;
        };
        let level = arrival.level;
        let Some(desc) = resident.chain.level(level) else {
            log::warn!(
                "texture_stream: dropping arrival for page {} level {level}: the chain has {} levels",
                arrival.page,
                resident.chain.level_count()
            );
            return false;
        };
        if resident
            .requested
            .get(&level)
            .is_some_and(|state| arrival.generation < state.newest)
        {
            return false;
        }
        if let Some(state) = resident.requested.get_mut(&level) {
            state.pending = None;
        }
        let Some(payload) = arrival.payload else {
            resident.failed.insert(level);
            log::warn!(
                "texture_stream: page {} level {level} could not be loaded; not retrying until the page is re-declared",
                arrival.page
            );
            return false;
        };
        if payload.width != desc.width
            || payload.height != desc.height
            || payload.texels.len() as u64 != desc.bytes
        {
            log::warn!(
                "texture_stream: dropping mis-sized arrival for page {} level {level}: got {}x{} {} bytes, chain declares {}x{} {} bytes",
                arrival.page,
                payload.width,
                payload.height,
                payload.texels.len(),
                desc.width,
                desc.height,
                desc.bytes
            );
            return false;
        }
        if resident.resident & level_bit(level) == 0 {
            resident.resident |= level_bit(level);
            resident.bytes = resident.bytes.saturating_add(desc.bytes);
        }
        self.uploads.push(MipUpload {
            page: arrival.page,
            level,
            width: payload.width,
            height: payload.height,
            texels: payload.texels,
        });
        true
    }

    /// Take the arrived levels for the render thread to upload
    /// ([`SceneBatch::upload_mip`](super::render::SceneBatch::upload_mip)).
    ///
    /// Call once per frame on the render thread: the texels leave the manager
    /// here (the GPU keeps them after the upload), while the level stays marked
    /// resident.
    pub fn take_uploads(&mut self) -> Vec<MipUpload> {
        std::mem::take(&mut self.uploads)
    }

    /// Drop a page's residency without touching its policy or pending loads.
    ///
    /// The batch recreates its texture array when a pipeline/desc rebuild
    /// happens, which discards every level; call this for each affected page so
    /// the manager stops promising levels the GPU no longer holds. Pending
    /// arrivals still store, so an in-flight level is not lost.
    pub fn invalidate(&mut self, page: u32) -> bool {
        match self.textures.get_mut(&page) {
            Some(resident) => {
                resident.resident = 0;
                resident.bytes = 0;
                resident.warned = false;
                true
            }
            None => false,
        }
    }

    /// [`invalidate`](Self::invalidate) every page, for a full resource rebuild.
    pub fn invalidate_all(&mut self) {
        for resident in self.textures.values_mut() {
            resident.resident = 0;
            resident.bytes = 0;
            resident.warned = false;
        }
    }

    /// The level to publish for `page` with
    /// [`SceneBatch::set_page_level`](super::render::SceneBatch::set_page_level):
    /// the finest resident level, or `None` when nothing is resident (the
    /// shaders draw the page untextured — never an invented level). Logs once
    /// per missing episode; see the module docs.
    pub fn resolve_level(&mut self, page: u32) -> Option<u32> {
        let Some(resident) = self.textures.get_mut(&page) else {
            return None;
        };
        let level = resident.finest_resident();
        if resident.wanted_resident() {
            resident.warned = false;
        } else if !resident.warned {
            resident.warned = true;
            let wanted = resident.wanted;
            match level {
                Some(level) => log::warn!(
                    "texture_stream: page {page} wants level {wanted} but samples the finest resident level {level}"
                ),
                None => log::warn!(
                    "texture_stream: page {page} has no resident level (wanted {wanted}); draw untextured until one arrives"
                ),
            }
        }
        level
    }

    pub fn stats(&self) -> TextureStreamStats {
        TextureStreamStats {
            textures: self.textures.len(),
            resident_levels: self
                .textures
                .values()
                .map(|resident| resident.resident.count_ones() as usize)
                .sum(),
            resident_bytes: self.resident_bytes(),
            budget_bytes: self.budget_bytes,
            pending_levels: self
                .textures
                .values()
                .map(|resident| {
                    resident
                        .requested
                        .values()
                        .filter(|state| state.pending.is_some())
                        .count()
                })
                .sum(),
            queued_uploads: self.uploads.len(),
        }
    }
}

/// The budget's victim: longest unrequested, then largest wanted footprint,
/// then lowest page. A texture at its coarsest level is never a victim, so a
/// chain's smallest level always stays.
fn pick_victim(
    textures: &BTreeMap<u32, Resident>,
    wanted: &BTreeMap<u32, u32>,
    frame: u64,
) -> Option<u32> {
    let mut best: Option<(u32, u64, u64)> = None;
    for (page, resident) in textures {
        let level = wanted.get(page).copied().unwrap_or(resident.policy);
        if level >= resident.chain.coarsest() {
            continue;
        }
        let inactivity = frame.saturating_sub(resident.last_seen);
        let footprint = resident.chain.bytes_from(level);
        let better = match best {
            None => true,
            Some((best_page, best_inactivity, best_footprint)) => {
                inactivity > best_inactivity
                    || (inactivity == best_inactivity && footprint > best_footprint)
                    || (inactivity == best_inactivity
                        && footprint == best_footprint
                        && *page < best_page)
            }
        };
        if better {
            best = Some((*page, inactivity, footprint));
        }
    }
    best.map(|(page, _, _)| page)
}

/// Decode one level's texels off-thread. `None` = the level could not be
/// produced; see [`MipArrival::payload`].
///
/// The loader runs on worker threads, so it owns its inputs and must not touch
/// the world, the ECS, or anything else that is not `Send`. It only reads
/// bytes and decodes them: it must not influence sim state or determinism.
pub struct MipStreamer<L>
where
    L: Fn(u32, u32) -> Option<MipPayload> + Clone + Send + Sync + 'static,
{
    loader: L,
    /// Queued requests: (priority, sequence, page, level, generation).
    /// `Mutex` (not a channel): dedup needs a scan and priorities reorder.
    queue: Arc<(Mutex<VecDeque<(f32, u64, u32, u32, u64)>>, Condvar)>,
    /// In-flight keys (queued or building), so duplicates never submit twice.
    inflight: Arc<Mutex<HashMap<(u32, u32), u64>>>,
    /// Finished loads awaiting drain.
    done: Arc<Mutex<VecDeque<MipArrival>>>,
    /// Monotonic submit sequence (stable order within one priority).
    seq: u64,
    /// Worker count (native only; wasm and `0` run inline in `pump`).
    workers: usize,
    handles: Vec<std::thread::JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    /// Max queued requests (backpressure: the smallest priority drops first).
    max_queue: usize,
}

impl<L> MipStreamer<L>
where
    L: Fn(u32, u32) -> Option<MipPayload> + Clone + Send + Sync + 'static,
{
    /// New streamer over `loader`. `workers` caps native threads (`0` = no
    /// threads: every level loads inline in [`pump`](MipStreamer::pump), which
    /// is deterministic and what tests use). `max_queue` caps queued requests.
    pub fn new(loader: L, workers: usize, max_queue: usize) -> Self {
        Self {
            loader,
            queue: Arc::new((Mutex::new(VecDeque::new()), Condvar::new())),
            inflight: Arc::new(Mutex::new(HashMap::new())),
            done: Arc::new(Mutex::new(VecDeque::new())),
            seq: 0,
            workers,
            handles: Vec::new(),
            shutdown: Arc::new(AtomicBool::new(false)),
            max_queue: max_queue.max(1),
        }
    }

    /// Queue a level load. Dedups on `(page, level)`: the same or a newer
    /// generation already queued or building is a no-op, a newer generation
    /// replaces a queued entry in place (its priority refreshes). A newer
    /// generation for a *building* level queues behind it; the older build
    /// drops at drain via the generation check.
    pub fn request(&mut self, request: MipRequest) {
        let priority = if request.priority.0.is_finite() {
            request.priority.0
        } else {
            0.0
        };
        let key = (request.page, request.level);
        let dominated = self
            .inflight
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .is_some_and(|&generation| generation >= request.generation);
        if dominated {
            return;
        }
        {
            let mut inflight = self
                .inflight
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            inflight
                .entry(key)
                .and_modify(|generation| {
                    if request.generation > *generation {
                        *generation = request.generation;
                    }
                })
                .or_insert(request.generation);
        }
        {
            let mut queue = self
                .queue
                .0
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let mut replaced = false;
            for slot in queue.iter_mut() {
                if (slot.2, slot.3) == key {
                    if slot.4 < request.generation {
                        slot.0 = priority;
                        slot.1 = self.seq;
                        slot.4 = request.generation;
                        replaced = true;
                    } else {
                        return; // queued entry is same or newer
                    }
                    break;
                }
            }
            if !replaced {
                queue.push_back((
                    priority,
                    self.seq,
                    request.page,
                    request.level,
                    request.generation,
                ));
            }
            self.seq += 1;
            while queue.len() > self.max_queue {
                let mut worst = 0usize;
                for (index, item) in queue.iter().enumerate() {
                    if item.0 < queue[worst].0 {
                        worst = index;
                    }
                }
                if let Some(dropped) = queue.remove(worst) {
                    let mut inflight = self
                        .inflight
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    if inflight.get(&(dropped.2, dropped.3)).is_none_or(|&generation| {
                        generation <= dropped.4
                    }) {
                        inflight.remove(&(dropped.2, dropped.3));
                    }
                    log::warn!(
                        "mip_streamer: queue full, dropping page {} level {}",
                        dropped.2,
                        dropped.3
                    );
                } else {
                    break;
                }
            }
        }
        self.ensure_workers();
        self.queue.1.notify_one();
    }

    /// Cancel a queued level. A level already being decoded finishes and drops
    /// at drain (no preemption; the load is pure I/O and a short decode).
    pub fn cancel(&self, page: u32, level: u32) -> bool {
        let removed = {
            let mut queue = self
                .queue
                .0
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let before = queue.len();
            queue.retain(|item| (item.2, item.3) != (page, level));
            before != queue.len()
        };
        if removed {
            let mut inflight = self
                .inflight
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            inflight.remove(&(page, level));
        }
        removed
    }

    /// Queued + building level count (backpressure readout).
    pub fn pending(&self) -> usize {
        self.inflight
            .lock()
            .map(|map| map.len())
            .unwrap_or(0)
    }

    /// Finished loads awaiting drain.
    pub fn finished(&self) -> usize {
        self.done.lock().map(|queue| queue.len()).unwrap_or(0)
    }

    /// Run due loads. Native workers drain the shared queue continuously;
    /// `pump` additionally loads up to `budget` levels inline (a latency floor
    /// for the highest priority request, never blocking). Wasm and `workers ==
    /// 0` build only here. Returns loads completed inline by this call.
    pub fn pump(&mut self, budget: usize) -> usize {
        let mut loaded = 0;
        for _ in 0..budget.max(1) {
            let next = {
                let mut queue = self
                    .queue
                    .0
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if queue.is_empty() {
                    None
                } else {
                    let mut best = 0usize;
                    for (index, item) in queue.iter().enumerate() {
                        if (item.0, std::cmp::Reverse(item.1))
                            > (queue[best].0, std::cmp::Reverse(queue[best].1))
                        {
                            best = index;
                        }
                    }
                    queue.remove(best)
                }
            };
            let Some((_, _, page, level, generation)) = next else {
                break;
            };
            let payload = (self.loader)(page, level);
            if let Ok(mut done) = self.done.lock() {
                done.push_back(MipArrival {
                    page,
                    level,
                    generation,
                    payload,
                });
            }
            loaded += 1;
        }
        loaded
    }

    /// Drain finished loads into `residency` newest-first per level: an older
    /// generation for a level drops, exactly like a stale chunk build. Clears
    /// in-flight keys for drained levels. Returns `(stored, dropped)`.
    pub fn drain_into(&self, residency: &mut TextureResidency) -> (usize, usize) {
        let arrivals: Vec<MipArrival> = {
            let mut done = self
                .done
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            done.drain(..).collect()
        };
        let mut best: HashMap<(u32, u32), MipArrival> = HashMap::new();
        for arrival in arrivals {
            best.entry((arrival.page, arrival.level))
                .and_modify(|existing| {
                    if arrival.generation > existing.generation {
                        *existing = arrival.clone();
                    }
                })
                .or_insert(arrival);
        }
        let mut keys: Vec<(u32, u32)> = best.keys().copied().collect();
        keys.sort();
        let mut stored = 0usize;
        let mut dropped = 0usize;
        for key in keys {
            let Some(arrival) = best.remove(&key) else {
                continue;
            };
            {
                let mut inflight = self
                    .inflight
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                match inflight.get(&key) {
                    Some(&generation) if generation > arrival.generation => {}
                    Some(_) => {
                        inflight.remove(&key);
                    }
                    None => {}
                }
            }
            if residency.store(arrival) {
                stored += 1;
            } else {
                dropped += 1;
            }
        }
        (stored, dropped)
    }

    /// Forget all queued, in-flight and finished work (level unload). Does not
    /// touch the residency (the caller clears or invalidates it).
    pub fn clear(&self) {
        if let Ok(mut queue) = self.queue.0.lock() {
            queue.clear();
        }
        if let Ok(mut inflight) = self.inflight.lock() {
            inflight.clear();
        }
        if let Ok(mut done) = self.done.lock() {
            done.clear();
        }
        self.queue.1.notify_all();
    }

    /// Spawn native worker threads on first request. Each worker loops: pop the
    /// highest-priority request, load it, push the arrival. Wasm: no-op (`pump`
    /// loads inline); `workers == 0`: no-op (fully inline, deterministic).
    fn ensure_workers(&mut self) {
        if !self.handles.is_empty() {
            return;
        }
        #[cfg(not(target_family = "wasm"))]
        {
            let count = self.workers;
            if count == 0 {
                return;
            }
            for worker in 0..count {
                let queue = Arc::clone(&self.queue);
                let done = Arc::clone(&self.done);
                let shutdown = Arc::clone(&self.shutdown);
                let loader = self.loader.clone();
                let name = format!("mip-load-{worker}");
                match std::thread::Builder::new()
                    .name(name)
                    .spawn(move || mip_worker_loop(loader, queue, done, shutdown))
                {
                    Ok(handle) => self.handles.push(handle),
                    Err(error) => {
                        log::warn!("mip_streamer: worker spawn failed ({error}), inline fallback");
                        break;
                    }
                }
            }
        }
    }
}

impl<L> Drop for MipStreamer<L>
where
    L: Fn(u32, u32) -> Option<MipPayload> + Clone + Send + Sync + 'static,
{
    /// Signal shutdown, wake parked workers, and join them: no detached thread
    /// outlives the streamer that owns its loader.
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        self.queue.1.notify_all();
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

/// Parked-worker loop (native only): pop the highest-priority request, load,
/// push the arrival. Workers sleep on the queue condvar so `request`, `clear`
/// and drop wake them promptly; shutdown exits instead of sleeping forever.
#[cfg(not(target_family = "wasm"))]
fn mip_worker_loop<L>(
    loader: L,
    queue: Arc<(Mutex<VecDeque<(f32, u64, u32, u32, u64)>>, Condvar)>,
    done: Arc<Mutex<VecDeque<MipArrival>>>,
    shutdown: Arc<AtomicBool>,
) where
    L: Fn(u32, u32) -> Option<MipPayload> + Send + 'static,
{
    loop {
        let next = {
            let (lock, condvar) = &*queue;
            let mut queue = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            while queue.is_empty() && !shutdown.load(Ordering::SeqCst) {
                queue = condvar.wait(queue).unwrap_or_else(|poisoned| poisoned.into_inner());
            }
            if shutdown.load(Ordering::SeqCst) {
                return;
            }
            let mut best = 0usize;
            for (index, item) in queue.iter().enumerate() {
                if (item.0, std::cmp::Reverse(item.1))
                    > (queue[best].0, std::cmp::Reverse(queue[best].1))
                {
                    best = index;
                }
            }
            queue.remove(best)
        };
        let Some((_, _, page, level, generation)) = next else {
            continue;
        };
        let payload = loader(page, level);
        if let Ok(mut done) = done.lock() {
            done.push_back(MipArrival {
                page,
                level,
                generation,
                payload,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tuning() -> StreamTuning {
        StreamTuning {
            hysteresis: 0.25,
            budget_slack: 0.0,
            pending_timeout: 4,
        }
    }

    fn chain(page: u32, edge: u32) -> TextureChain {
        TextureChain::new(page, edge, edge).expect("chain builds")
    }

    fn payload(chain: &TextureChain, level: u32) -> MipPayload {
        let desc = chain.level(level).expect("level exists");
        MipPayload {
            width: desc.width,
            height: desc.height,
            texels: vec![0x5A; desc.bytes as usize],
        }
    }

    fn arrival(chain: &TextureChain, level: u32, generation: u64) -> MipArrival {
        MipArrival {
            page: chain.page,
            level,
            generation,
            payload: Some(payload(chain, level)),
        }
    }

    #[test]
    fn desired_levels_follow_the_projected_size() {
        let texture = chain(0, 1024);
        assert_eq!(texture.level_count(), 11);
        assert_eq!(texture.coarsest(), 10);
        assert_eq!(texture.bytes_at(0), Some(1024 * 1024 * 4));
        assert_eq!(texture.bytes_from(0), texture.bytes());
        assert_eq!(texture.bytes_from(10), 4);
        assert_eq!(chain(1, 8).bytes(), 256 + 64 + 16 + 4);
        // One texel per screen pixel or better keeps level 0; every doubling
        // of the screen size coarsens one level.
        assert_eq!(texture.desired_level(1024.0, None, 0.25), 0);
        assert_eq!(texture.desired_level(512.0, None, 0.25), 1);
        assert_eq!(texture.desired_level(400.0, None, 0.25), 1);
        assert_eq!(texture.desired_level(256.0, None, 0.25), 2);
        assert_eq!(texture.desired_level(128.0, None, 0.25), 3);
        assert_eq!(texture.desired_level(8.0, None, 0.25), 7);
        assert_eq!(texture.desired_level(1.0, None, 0.25), 10);
        assert_eq!(texture.desired_level(0.5, None, 0.25), 10);
        assert_eq!(desired_level(1024, 400.0, None, 0.25, 10), 1);
        // Non-finite input keeps the previous level instead of guessing.
        assert_eq!(select_mip_level(f32::NAN, Some(4), 0.25, 10), 4);
    }

    #[test]
    fn hysteresis_holds_a_level_across_a_threshold() {
        let hysteresis = 0.25;
        let max = 8;
        // Level 2 covers ratio 4..8. The dead band runs 3..10 around it.
        assert_eq!(select_mip_level(7.9, Some(2), hysteresis, max), 2);
        assert_eq!(select_mip_level(8.0, Some(2), hysteresis, max), 2);
        assert_eq!(select_mip_level(9.9, Some(2), hysteresis, max), 2);
        assert_eq!(select_mip_level(10.1, Some(2), hysteresis, max), 3);
        assert_eq!(select_mip_level(3.1, Some(2), hysteresis, max), 2);
        assert_eq!(select_mip_level(2.9, Some(2), hysteresis, max), 1);
        // Coming back the other way does not refine inside the same band.
        assert_eq!(select_mip_level(8.0, Some(3), hysteresis, max), 3);
        assert_eq!(select_mip_level(6.1, Some(3), hysteresis, max), 3);
        assert_eq!(select_mip_level(5.9, Some(3), hysteresis, max), 2);
        // A ratio oscillating across the boundary never changes the level.
        let mut level = 2;
        for step in 0..64 {
            let ratio = if step % 2 == 0 { 7.999 } else { 8.001 };
            level = select_mip_level(ratio, Some(level), hysteresis, max);
            assert_eq!(level, 2, "flip-flopped at step {step}");
        }
        // Outside the band it jumps straight to the ideal level, so a fresh
        // texture (which starts at minimum quality) refines in one touch.
        assert_eq!(select_mip_level(64.0, Some(0), hysteresis, max), 6);
        assert_eq!(select_mip_level(1.0, Some(8), hysteresis, max), 0);
        // Zero hysteresis is the exact rule.
        assert_eq!(select_mip_level(8.0, Some(2), 0.0, max), 3);
        assert_eq!(select_mip_level(7.9, Some(3), 0.0, max), 2);
    }

    #[test]
    fn projected_bounds_track_the_viewport_and_reject_behind_the_camera() {
        let viewport = [100.0, 200.0];
        let ortho = glam::camera::rh::proj::directx::orthographic(-2.0, 2.0, -2.0, 2.0, 0.1, 10.0);
        let size = project_bounds(ortho, viewport, [-0.5, -0.5, -1.0], [0.5, 0.5, -1.0])
            .expect("in view");
        assert!((size.width_px - 25.0).abs() < 1e-3, "{size:?}");
        assert!((size.height_px - 50.0).abs() < 1e-3, "{size:?}");
        assert_eq!(size.max_px(), size.height_px);
        assert_eq!(project_bounds(ortho, [0.0, 200.0], [-1.0; 3], [1.0; 3]), None);

        let perspective = glam::camera::rh::proj::directx::perspective(
            std::f32::consts::FRAC_PI_4,
            1.0,
            0.1,
            100.0,
        );
        assert!(
            project_bounds(perspective, viewport, [-1.0, -1.0, 2.0], [1.0, 1.0, 4.0]).is_none(),
            "a bound behind the eye has no projected size"
        );
        assert!(
            project_bounds(perspective, viewport, [-1.0, -1.0, -4.0], [1.0, 1.0, -2.0]).is_some(),
            "a bound in front projects"
        );
    }

    #[test]
    fn unarrived_levels_sample_the_finest_resident() {
        let mut residency = TextureResidency::new(u64::MAX, tuning());
        let chain = chain(3, 16);
        residency.declare(chain.clone()).expect("declare");
        assert_eq!(
            residency.resolve_level(3),
            None,
            "nothing resident: draw untextured, never a fabricated level"
        );

        // The coarsest level arrives first: that is what samples, not level 0.
        assert!(residency.store(arrival(&chain, 4, 1)));
        assert_eq!(residency.resolve_level(3), Some(4));
        assert_eq!(residency.finest_resident(3), Some(4));

        // A finer arrival improves the sample.
        assert!(residency.store(arrival(&chain, 2, 1)));
        assert_eq!(residency.resolve_level(3), Some(2));
        assert_eq!(residency.finest_resident(3), Some(2));
        assert_eq!(residency.resident_bytes(), 4 * 4 * 4 + 4);

        // A single-level texture has nothing finer to wait for.
        let single = TextureChain::new(4, 1, 1).expect("single level");
        assert_eq!(single.level_count(), 1);
        residency.declare(single.clone()).expect("declare");
        assert_eq!(residency.resolve_level(4), None);
        assert!(residency.store(arrival(&single, 0, 1)));
        assert_eq!(residency.resolve_level(4), Some(0));
    }

    #[test]
    fn budget_eviction_prefers_the_longest_unrequested() {
        let mut residency = TextureResidency::new(400, tuning());
        for page in 0..3u32 {
            let page_chain = chain(page, 8);
            residency.declare(page_chain.clone()).expect("declare");
            // Every page is 100 px on screen and wants level 0; page 0 was
            // touched longest ago, page 2 most recently.
            residency.touch(page, 100.0, 10 + u64::from(page) * 2);
            assert_eq!(residency.policy_level(page), Some(0));
            assert!(residency.store(arrival(&page_chain, 0, 1)));
        }
        assert!(residency.store(arrival(&chain(0, 8), 1, 1)));
        assert_eq!(residency.resident_bytes(), 256 * 3 + 64);

        let plan = residency.plan(20);
        assert_eq!(
            plan.evictions,
            vec![
                MipEviction {
                    page: 0,
                    level: 0,
                    bytes: 256,
                },
                MipEviction {
                    page: 0,
                    level: 1,
                    bytes: 64,
                },
                MipEviction {
                    page: 1,
                    level: 0,
                    bytes: 256,
                },
            ],
            "longest-unrequested first, and a chain's smallest level (3) never leaves"
        );
        assert_eq!(residency.wanted_level(0), Some(3));
        assert_eq!(residency.wanted_level(1), Some(2));
        assert_eq!(residency.wanted_level(2), Some(0));
        assert_eq!(residency.resident_bytes(), 256, "only page 2 level 0 is left");

        // The loads left to issue cover each texture's wanted range, in
        // priority (all equal here), then page, then level order.
        let levels: Vec<(u32, u32)> = plan
            .requests
            .iter()
            .map(|request| (request.page, request.level))
            .collect();
        assert_eq!(
            levels,
            vec![(0, 3), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
        );
        assert!(plan.requests.iter().all(|request| request.priority.0 == 100.0));
        assert!(plan.requests.iter().all(|request| request.generation > 0));
    }

    #[test]
    fn a_tight_budget_holds_minimum_quality() {
        let mut residency = TextureResidency::new(1, tuning());
        let page_chain = chain(7, 8);
        residency.declare(page_chain.clone()).expect("declare");
        residency.touch(7, 100.0, 1);
        assert!(residency.store(arrival(&page_chain, 0, 1)));

        let plan = residency.plan(2);
        assert_eq!(
            plan.evictions,
            vec![MipEviction {
                page: 7,
                level: 0,
                bytes: 256,
            }]
        );
        assert_eq!(residency.wanted_level(7), Some(3));
        assert_eq!(
            plan.requests.iter().map(|request| request.level).collect::<Vec<_>>(),
            vec![3],
            "only the smallest level is still wanted"
        );

        // Once the smallest level is resident, further pressure keeps it: the
        // sample stays valid instead of emptying the page.
        assert!(residency.store(arrival(&page_chain, 3, 1)));
        let _ = residency.plan(3);
        assert!(residency.is_resident(7, 3));
        assert_eq!(residency.resolve_level(7), Some(3));
        assert_eq!(residency.stats().resident_bytes, 4);
    }

    #[test]
    fn stale_generations_drop_and_pending_requests_expire() {
        let mut residency = TextureResidency::new(u64::MAX, tuning());
        let page_chain = chain(1, 8);
        residency.declare(page_chain.clone()).expect("declare");

        // Untouched, the page wants its smallest level only.
        let plan = residency.plan(0);
        assert_eq!(plan.requests.len(), 1);
        let request = plan.requests[0];
        assert_eq!(request.level, 3);
        assert!(
            !residency.store(MipArrival {
                page: 1,
                level: 3,
                generation: request.generation - 1,
                payload: Some(payload(&page_chain, 3)),
            }),
            "an older generation than the newest request drops"
        );
        assert!(residency.store(MipArrival {
            page: 1,
            level: 3,
            generation: request.generation,
            payload: Some(payload(&page_chain, 3)),
        }));

        // A second page never gets answered: after the timeout the request is
        // re-issued with a newer generation instead of stranding the level.
        let mut residency = TextureResidency::new(u64::MAX, tuning());
        residency.declare(page_chain.clone()).expect("declare");
        let first = residency.plan(0);
        assert!(residency.plan(2).requests.is_empty(), "still pending");
        let reissued = residency.plan(6);
        assert_eq!(reissued.requests.len(), 1);
        assert!(reissued.requests[0].generation > first.requests[0].generation);
    }

    #[test]
    fn failed_levels_are_not_retried() {
        let mut residency = TextureResidency::new(u64::MAX, tuning());
        let page_chain = chain(2, 4);
        residency.declare(page_chain.clone()).expect("declare");
        let request = residency.plan(0).requests[0];
        assert!(!residency.store(MipArrival {
            page: 2,
            level: request.level,
            generation: request.generation,
            payload: None,
        }));
        assert!(
            residency.plan(1000).requests.is_empty(),
            "an unsupported level is remembered, not re-requested every frame"
        );
    }

    #[test]
    fn invalidating_a_page_clears_residency_and_reloads_it() {
        let mut residency = TextureResidency::new(u64::MAX, tuning());
        let page_chain = chain(5, 4);
        residency.declare(page_chain.clone()).expect("declare");
        assert!(residency.store(arrival(&page_chain, 2, 1)));
        assert_eq!(residency.resolve_level(5), Some(2));
        assert!(residency.invalidate(5));
        assert_eq!(residency.finest_resident(5), None);
        assert_eq!(residency.resolve_level(5), None);
        assert_eq!(residency.resident_bytes(), 0);
        let plan = residency.plan(0);
        assert_eq!(plan.requests.len(), 1);
        assert_eq!(plan.requests[0].level, 2, "the resident level reloads");
    }

    #[test]
    fn mis_sized_and_undeclared_arrivals_never_store() {
        let mut residency = TextureResidency::new(u64::MAX, tuning());
        let page_chain = chain(6, 4);
        residency.declare(page_chain.clone()).expect("declare");
        assert!(!residency.store(MipArrival {
            page: 6,
            level: 0,
            generation: 1,
            payload: Some(MipPayload {
                width: 4,
                height: 4,
                texels: vec![0; 8],
            }),
        }));
        assert!(!residency.store(MipArrival {
            page: 6,
            level: 9,
            generation: 1,
            payload: Some(MipPayload {
                width: 1,
                height: 1,
                texels: vec![0; 4],
            }),
        }));
        assert!(
            !residency.store(arrival(&chain(9, 4), 0, 1)),
            "an arrival for an undeclared page drops"
        );
        assert_eq!(residency.resident_bytes(), 0);
        assert!(residency.take_uploads().is_empty());    }

    #[test]
    fn declaring_a_bad_chain_is_a_typed_error() {
        assert_eq!(TextureChain::new(0, 0, 4), Err(StreamError::ZeroExtent(0)));
        assert_eq!(
            TextureChain::from_dimensions(0, Vec::<(u32, u32)>::new()),
            Err(StreamError::EmptyChain)
        );
        assert_eq!(
            TextureChain::from_dimensions(0, [(4, 4), (0, 2)]),
            Err(StreamError::ZeroExtent(1))
        );
        let mut residency = TextureResidency::new(0, tuning());
        let mut oversized = chain(0, 4);
        for _ in 0..MAX_MIP_LEVELS {
            let last = *oversized.levels.last().expect("levels");
            oversized.levels.push(MipLevelDesc {
                width: last.width,
                height: last.height,
                bytes: last.bytes,
            });
        }
        assert!(matches!(
            residency.declare(oversized),
            Err(StreamError::TooManyLevels(_))
        ));
    }

    #[test]
    fn streamer_loads_levels_and_drains_them_into_residency() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let record = Arc::clone(&calls);
        let chain_for_loader = chain(0, 4);
        let loader = move |page: u32, level: u32| {
            if let Ok(mut calls) = record.lock() {
                calls.push((page, level));
            }
            let Some(desc) = chain_for_loader.level(level) else {
                return None;
            };
            Some(MipPayload {
                width: desc.width,
                height: desc.height,
                texels: vec![0x11; desc.bytes as usize],
            })
        };
        let mut streamer = MipStreamer::new(loader, 0, 8);
        let mut residency = TextureResidency::new(u64::MAX, tuning());
        let page_chain = chain(0, 4);
        residency.declare(page_chain.clone()).expect("declare");
        // 4 px on screen: the texture wants level 0.
        assert_eq!(residency.touch(0, 4.0, 0), Some(0));

        let plan = residency.plan(0);
        assert_eq!(plan.requests.len(), 3);
        for request in &plan.requests {
            streamer.request(*request);
        }
        assert_eq!(streamer.pending(), 3);
        assert_eq!(streamer.pump(8), 3);
        assert_eq!(streamer.finished(), 3);
        assert_eq!(streamer.drain_into(&mut residency), (3, 0));
        assert_eq!(streamer.pending(), 0);
        assert_eq!(residency.resolve_level(0), Some(0));
        assert_eq!(residency.resident_bytes(), 64 + 16 + 4);
        let uploads = residency.take_uploads();
        assert_eq!(uploads.len(), 3);
        assert!(uploads.iter().all(|upload| upload.page == 0));
        assert_eq!(
            calls.lock().map(|calls| calls.len()).unwrap_or(0),
            3,
            "each level loaded once"
        );

        // The plan is empty once everything wanted is resident.
        assert!(residency.plan(1).requests.is_empty());
        // Duplicate submits dedup on (page, level).
        streamer.request(MipRequest {
            page: 0,
            level: 0,
            generation: 99,
            priority: MipPriority(1.0),
        });
        assert_eq!(streamer.pending(), 1);
    }
}
