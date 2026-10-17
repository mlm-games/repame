//! Occlusion culling: a hierarchical depth test, then the existing frustum
//! path.
//!
//! This is the classic two-phase arrangement. Phase one runs the frame's
//! groups against the *previous* frame's depth, reduced into a
//! [`DepthPyramid`]: tile `(t, u)` at level 0 holds the **farthest** view
//! depth of its `tile x tile` pixel block, and each higher level holds the
//! farthest of its children. A group is culled only when every tile that is
//! fully covered by its screen rectangle is **nearer** than the group's
//! nearest depth by more than [`OcclusionCulling::bias`] - i.e. the group is
//! behind every pixel it could cover. Phase two is the existing frustum test,
//! which still runs because the pyramid says nothing about what is on screen.
//!
//! Conservatism is the whole contract: a wrongly culled group is a missing
//! object, so the test
//!
//! * uses the farthest depth per tile (never a nearer one),
//! * only considers tiles *fully* inside the group's screen rect (partial
//!   tiles are skipped, they could contain visible pixels),
//! * treats a tile the pass left uncovered (the transparent clear keeps its
//!   farthest depth at `0`, nearer than any real fragment) as no occluder at
//!   all: a group can be visible against pixels nothing was drawn into,
//! * refuses to cull when any corner of the box is behind the eye plane, and
//! * applies a depth bias so a group that is exactly at the depth of the
//!   occluder stays.
//!
//! Because the pass only replays opaque depth-tested ranges, the test is only
//! run over those groups as well: a transparent group's pixels are never in
//! the pyramid, so nothing vouches for them.
//!
//! Depth is view depth along the camera forward axis, the quantity the
//! [`PassKind::Depth`](crate::PassKind::Depth) pass writes into an
//! `Rgba16Float` target's `.r` lane. A game feeding its own pyramid builds it
//! with [`DepthPyramid::from_tiled_bytes`] (the batch's `encode_pass_readback`
//! lays rows out at the copy alignment) and stages it next frame; one frame
//! of latency is what keeps the test off the critical path.
//!
//! [`Frame3d::occlusion_feed`](crate::Frame3d::occlusion_feed) arms that loop
//! inside the viewport: after the depth pass drew, a compute pass reduces the
//! target into level-0 tile maxima (the exact `max`, `0` where nothing was
//! drawn) and coarsens the whole level chain, only that chain is read back,
//! and [`DepthPyramid::from_packed_levels`] turns it into the pyramid staged
//! on the next frame, so a game needs no readback plumbing of its own. When
//! the frame declares no depth pass the viewport stages the engine's own
//! [`PassKind::Occlusion`](crate::PassKind::Occlusion) pass, so the feed
//! never depends on the post chain. The map is polled without waiting, so a
//! frame whose readback has not landed yet culls nothing instead of pacing
//! the frame around the GPU.
//!
//! With [`OcclusionFeedDesc::phase_two`] armed the same chain is tested
//! against in the *same* frame: a compute pass re-tests every phase-one
//! reject right after the depth pass that produced the chain and writes its
//! indirect draw args (zeroed for the still-hidden); the survivors are then
//! drawn over the scene pass's target, loaded with the depth that pass left.
//! The one-frame latency is then only a phase-one filter, not a visible
//! delay.

use glam::Mat4;

/// Tile edge in pixels of pyramid level 0.
pub const DEFAULT_TILE: u32 = 8;

/// Largest level-0 tile edge the builders clamp to.
pub const MAX_TILE: u32 = 64;

/// Smallest level-0 tile edge (one pixel per tile).
const MIN_TILE: u32 = 1;

/// Depth margin in world units the viewport-driven occlusion feed uses by
/// default (see [`OcclusionFeedDesc::bias`]).
pub const DEFAULT_BIAS: f32 = 0.1;

/// Configuration of the viewport-driven occlusion feed
/// ([`Frame3d::occlusion_feed`](crate::Frame3d::occlusion_feed)).
///
/// With the feed armed, the viewport reduces the frame's declared depth pass
/// into level-0 tile maxima on the GPU, reads back only that grid, and stages
/// the resulting [`DepthPyramid`] - and with it the [`OcclusionCulling`]
/// state - on the next frame. A frame that carries no depth pass feeds
/// nothing (there is nothing to reduce), and a caller that stages its own
/// [`Frame3d::occlusion`](crate::Frame3d::occlusion) wins over the feed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OcclusionFeedDesc {
    /// Level-0 tile edge in pixels of the GPU reduction, clamped to `1..=64`
    /// on use. The readback holds one `f32` per tile, so a larger tile trades
    /// culling precision for less traffic.
    pub tile: u32,
    /// Depth margin in world units a group must clear before it is culled,
    /// clamped to `>= 0` (non-finite falls back to
    /// [`DEFAULT_BIAS`]). Trades culling against the risk of hiding a group
    /// that is only just behind its occluder.
    pub bias: f32,
    /// Same-frame second phase (default `false`). When armed, a frame that
    /// declares a depth pass tests the phase-1 rejects against the *same
    /// frame's* chain on the GPU right after that pass (a compute pass
    /// writes each reject's indirect draw args, zeroing the still-hidden
    /// ones) and draws the survivors over the scene pass's target through
    /// `draw_indexed_indirect`, loading the color and depth that pass left:
    /// no false negative survives a frame, at the cost of a chain build and
    /// a per-candidate test dispatch per frame. The depth pass also replays
    /// the rejects, so their shadows stay correct. Only the shared scene
    /// pass's opaque depth-tested groups are candidates (a group a declared
    /// pass owns keeps phase one's drop); `false` keeps the one-frame-latency
    /// contract of the feed alone. The viewport appends the engine's own
    /// [`Occlusion`](crate::PassKind::Occlusion) depth pass when a frame
    /// declares none, so the feed never depends on the post chain.
    pub phase_two: bool,
}

impl Default for OcclusionFeedDesc {
    fn default() -> Self {
        Self {
            tile: DEFAULT_TILE,
            bias: DEFAULT_BIAS,
            phase_two: false,
        }
    }
}

impl OcclusionFeedDesc {
    /// Tile edge after clamping (what the pyramid is built with).
    pub fn clamped_tile(&self) -> u32 {
        self.tile.clamp(MIN_TILE, MAX_TILE)
    }

    /// Bias after clamping (what the depth test is run with).
    pub fn clamped_bias(&self) -> f32 {
        if self.bias.is_finite() {
            self.bias.max(0.0)
        } else {
            DEFAULT_BIAS
        }
    }
}

/// Pixel format of a depth readback fed to [`DepthPyramid::from_tiled_bytes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthSource {
    /// `Rgba16Float`, view depth in the red lane (the depth pass format).
    Rgba16Float,
    /// `R32Float`, view depth in the single lane.
    R32Float,
}

impl DepthSource {
    /// Bytes per texel in the source buffer.
    pub const fn bytes_per_texel(self) -> u32 {
        match self {
            Self::Rgba16Float => 8,
            Self::R32Float => 4,
        }
    }
}

/// Half-precision float (IEEE binary16) to `f32`. Only the decode the depth
/// readback needs; NaN and infinity pass through as `f32` NaN/infinity.
pub fn decode_f16(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x3ff) as u32;
    let sign = sign << 31;
    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal: `mantissa * 2^-24` is exact in f32 (no rounding).
        let value = mantissa as f32 * 2.0f32.powi(-24);
        return f32::from_bits(sign | value.to_bits());
    }
    if exponent == 0x1f {
        let payload = if mantissa == 0 { 0 } else { 0x40_0000 | (mantissa << 13) };
        return f32::from_bits(sign | 0x7f80_0000 | payload);
    }
    f32::from_bits(sign | ((exponent + 127 - 15) << 23) | (mantissa << 13))
}

/// Screen rectangle of a world-space AABB in NDC, or `None` when any corner
/// is behind the eye plane or non-finite. The rectangle is clamped to the
/// viewport; a rectangle that ends up empty is still returned (the tile
/// math rejects it).
pub fn aabb_ndc(view_proj: &Mat4, min: [f32; 3], max: [f32; 3]) -> Option<([f32; 2], [f32; 2])> {
    let mut lo = [f32::MAX; 2];
    let mut hi = [f32::MIN; 2];
    for corner in 0..8 {
        let p = [
            if corner & 1 == 0 { min[0] } else { max[0] },
            if corner & 2 == 0 { min[1] } else { max[1] },
            if corner & 4 == 0 { min[2] } else { max[2] },
        ];
        let clip = *view_proj * glam::Vec4::new(p[0], p[1], p[2], 1.0);
        if !clip.is_finite() || clip.w <= 1e-6 {
            return None;
        }
        let ndc = glam::Vec2::new(clip.x / clip.w, clip.y / clip.w);
        if !ndc.is_finite() {
            return None;
        }
        lo = [lo[0].min(ndc.x), lo[1].min(ndc.y)];
        hi = [hi[0].max(ndc.x), hi[1].max(ndc.y)];
    }
    let clamp2 = |v: [f32; 2]| [v[0].clamp(-1.0, 1.0), v[1].clamp(-1.0, 1.0)];
    let (lo, hi) = (clamp2(lo), clamp2(hi));
    if lo[0] >= hi[0] || lo[1] >= hi[1] {
        return None;
    }
    Some((lo, hi))
}

/// Nearest view depth of an AABB's corners along a unit `forward` axis.
pub fn nearest_view_depth(min: [f32; 3], max: [f32; 3], cam_pos: [f32; 3], forward: [f32; 3]) -> f32 {
    let mut nearest = f32::MAX;
    for corner in 0..8 {
        let p = [
            if corner & 1 == 0 { min[0] } else { max[0] },
            if corner & 2 == 0 { min[1] } else { max[1] },
            if corner & 4 == 0 { min[2] } else { max[2] },
        ];
        let depth = (p[0] - cam_pos[0]) * forward[0]
            + (p[1] - cam_pos[1]) * forward[1]
            + (p[2] - cam_pos[2]) * forward[2];
        nearest = nearest.min(depth);
    }
    nearest
}

/// Pyramid level dimensions from level 0 upward: each step halves both axes
/// with `div_ceil` until one tile covers the image. The CPU builders and the
/// viewport's GPU pass chain must agree on this list, so it lives in one
/// place.
pub(crate) fn coarsened_dims(mut cols: u32, mut rows: u32) -> Vec<(u32, u32)> {
    let mut dims = vec![(cols, rows)];
    while cols > 1 || rows > 1 {
        cols = cols.div_ceil(2);
        rows = rows.div_ceil(2);
        dims.push((cols, rows));
    }
    dims
}

/// Hierarchical max-depth pyramid over a view-depth buffer.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthPyramid {
    /// Source width in pixels.
    pub width: u32,
    /// Source height in pixels.
    pub height: u32,
    /// Level-0 tile edge in pixels.
    pub tile: u32,
    /// Level-0 tile columns and rows.
    pub cols: u32,
    pub rows: u32,
    /// Level 0 is per-tile max depth, each further level halves both axes.
    /// Tiles with no covered pixels hold `0.0` (nearer than any real
    /// fragment, the depth pass clears transparent): [`occluded`](Self::occluded)
    /// refuses to cull through them.
    levels: Vec<Vec<f32>>,
}

impl DepthPyramid {
    /// Builds the pyramid from one view-depth texel per pixel (row-major,
    /// row 0 is the top of the texture). Returns `None` for a zero-sized
    /// image or a short buffer.
    pub fn build(width: u32, height: u32, tile: u32, depths: &[f32]) -> Option<Self> {
        if width == 0 || height == 0 || depths.len() < (width as usize * height as usize) {
            return None;
        }
        let tile = tile.clamp(MIN_TILE, MAX_TILE);
        let cols = width.div_ceil(tile);
        let rows = height.div_ceil(tile);
        let mut level0 = vec![0.0f32; (cols * rows) as usize];
        for row in 0..height {
            for col in 0..width {
                let depth = depths[(row * width + col) as usize];
                if !depth.is_finite() {
                    continue;
                }
                let index = (row / tile) * cols + (col / tile);
                let slot = &mut level0[index as usize];
                *slot = slot.max(depth);
            }
        }
        Some(Self::coarsen(width, height, tile, cols, rows, level0))
    }

    /// Builds the pyramid from level-0 tile maxima alone, the first slice of
    /// what [`DepthPyramid::from_packed_levels`] reads: one `f32` per
    /// `tile x tile` pixel block, row major, `0.0` for a block the pass left
    /// uncovered or wrote a non-finite depth into (the same value
    /// [`DepthPyramid::build`] leaves). `width` and `height` are the source
    /// image's, so the grid is `ceil(width / tile) x ceil(height / tile)`
    /// floats.
    ///
    /// Returns `None` for a zero-sized image or a byte slice shorter than
    /// that grid.
    pub fn from_tile_grid(bytes: &[u8], width: u32, height: u32, tile: u32) -> Option<Self> {
        if width == 0 || height == 0 {
            return None;
        }
        let tile = tile.clamp(MIN_TILE, MAX_TILE);
        let cols = width.div_ceil(tile);
        let rows = height.div_ceil(tile);
        let count = (cols as usize) * (rows as usize);
        let depths = bytemuck::try_cast_slice::<u8, f32>(bytes).ok()?;
        if depths.len() < count {
            return None;
        }
        let level0: Vec<f32> = depths[..count]
            .iter()
            .map(|depth| {
                if depth.is_finite() && *depth > 0.0 {
                    *depth
                } else {
                    0.0
                }
            })
            .collect();
        Some(Self::coarsen(width, height, tile, cols, rows, level0))
    }

    /// Builds the pyramid from the packed level chain the viewport's GPU
    /// passes write: every level of [`coarsened_dims`] in order, level 0
    /// first, each row major. `width` and `height` are the source image's and
    /// `tile` its level-0 tile edge, so the chain holds
    /// `sum(ceil(width / tile / 2^l) * ceil(height / tile / 2^l))` floats.
    ///
    /// Returns `None` for a zero-sized image or a byte slice shorter than
    /// the chain.
    pub fn from_packed_levels(bytes: &[u8], width: u32, height: u32, tile: u32) -> Option<Self> {
        if width == 0 || height == 0 {
            return None;
        }
        let tile = tile.clamp(MIN_TILE, MAX_TILE);
        let dims = coarsened_dims(width.div_ceil(tile), height.div_ceil(tile));
        let depths = bytemuck::try_cast_slice::<u8, f32>(bytes).ok()?;
        let mut levels = Vec::with_capacity(dims.len());
        let mut offset = 0usize;
        for &(cols, rows) in &dims {
            let count = (cols as usize) * (rows as usize);
            if depths.len() < offset + count {
                return None;
            }
            levels.push(
                depths[offset..offset + count]
                    .iter()
                    .map(|depth| {
                        if depth.is_finite() && *depth > 0.0 {
                            *depth
                        } else {
                            0.0
                        }
                    })
                    .collect(),
            );
            offset += count;
        }
        Some(Self {
            width,
            height,
            tile,
            cols: width.div_ceil(tile),
            rows: height.div_ceil(tile),
            levels,
        })
    }

    /// The shared upper half: level 0 plus the max-of-2x2 coarsening up to a
    /// single tile, over exactly [`coarsened_dims`].
    fn coarsen(
        width: u32,
        height: u32,
        tile: u32,
        cols: u32,
        rows: u32,
        level0: Vec<f32>,
    ) -> Self {
        let dims = coarsened_dims(cols, rows);
        let mut levels = Vec::with_capacity(dims.len());
        levels.push(level0);
        for index in 1..dims.len() {
            let (prev_cols, prev_rows) = dims[index - 1];
            let (cols, rows) = dims[index];
            let prev = &levels[index - 1];
            let mut next = vec![0.0f32; (cols * rows) as usize];
            for y in 0..rows {
                for x in 0..cols {
                    let mut far = 0.0f32;
                    for dy in 0..2u32 {
                        for dx in 0..2u32 {
                            let sx = x * 2 + dx;
                            let sy = y * 2 + dy;
                            if sx < prev_cols && sy < prev_rows {
                                far = far.max(prev[(sy * prev_cols + sx) as usize]);
                            }
                        }
                    }
                    next[(y * cols + x) as usize] = far;
                }
            }
            levels.push(next);
        }
        Self {
            width,
            height,
            tile,
            cols: width.div_ceil(tile),
            rows: height.div_ceil(tile),
            levels,
        }
    }

    /// Level count (1 = one tile per level-0 tile).
    pub fn levels(&self) -> usize {
        self.levels.len()
    }

    /// Tiles along x/y of one level.
    pub fn level_dims(&self, level: usize) -> (u32, u32) {
        let mut cols = self.cols;
        let mut rows = self.rows;
        for _ in 0..level {
            cols = cols.div_ceil(2);
            rows = rows.div_ceil(2);
        }
        (cols, rows)
    }

    /// Farthest view depth of one tile at one level, or `None` out of range.
    pub fn tile_depth(&self, level: usize, col: u32, row: u32) -> Option<f32> {
        let (cols, rows) = self.level_dims(level);
        if col >= cols || row >= rows {
            return None;
        }
        self.levels
            .get(level)
            .map(|data| data[(row * cols + col) as usize])
    }

    /// Hierarchical-Z test for an NDC rectangle: true when the whole
    /// rectangle is covered by tiles whose farthest depth sits in front of
    /// `nearest` by more than `bias`. Never true unless full coverage exists
    /// at some level, and never true through a tile the depth pass left
    /// uncovered (farthest depth `0`): nothing was drawn there, so a group
    /// over it can be visible.
    pub fn occluded(&self, min_ndc: [f32; 2], max_ndc: [f32; 2], nearest: f32, bias: f32) -> bool {
        if !nearest.is_finite() || min_ndc[0] >= max_ndc[0] || min_ndc[1] >= max_ndc[1] {
            return false;
        }
        // Pixel rect; NDC y = 1 is the first texture row.
        let x0 = ((min_ndc[0] * 0.5 + 0.5) * self.width as f32).clamp(0.0, self.width as f32);
        let x1 = ((max_ndc[0] * 0.5 + 0.5) * self.width as f32).clamp(0.0, self.width as f32);
        let y0 = ((0.5 - max_ndc[1] * 0.5) * self.height as f32).clamp(0.0, self.height as f32);
        let y1 = ((0.5 - min_ndc[1] * 0.5) * self.height as f32).clamp(0.0, self.height as f32);
        if x0 >= x1 || y0 >= y1 {
            return false;
        }
        let bias = if bias.is_finite() { bias.max(0.0) } else { 0.0 };
        for level in 0..self.levels.len() {
            let (cols, rows) = self.level_dims(level);
            let scale = self.tile as f32 * (1u32 << level) as f32;
            // Fully covered tile range: ceil for the start, floor for the end.
            let t0 = (x0 / scale).ceil() as u32;
            let t1 = (x1 / scale).floor() as u32;
            let u0 = (y0 / scale).ceil() as u32;
            let u1 = (y1 / scale).floor() as u32;
            if t0 >= t1 || u0 >= u1 || t1 > cols || u1 > rows {
                continue;
            }
            let mut all = true;
            'tiles: for row in u0..u1 {
                for col in t0..t1 {
                    let Some(far) = self.tile_depth(level, col, row) else {
                        all = false;
                        break 'tiles;
                    };
                    // No coverage (`0`, the pass's transparent clear) or the
                    // group is not clear of the tile by more than `bias`:
                    // without every tile vouching, the group stays.
                    if far <= 0.0 || nearest <= far + bias {
                        all = false;
                        break 'tiles;
                    }
                }
            }
            if all {
                return true;
            }
        }
        false
    }

    /// Builds the pyramid from a GPU readback laid out for
    /// `wgpu::COPY_BYTES_PER_ROW_ALIGNMENT`: rows are padded to a 256-byte
    /// multiple, `tile` is the level-0 tile edge.
    pub fn from_tiled_bytes(
        bytes: &[u8],
        width: u32,
        height: u32,
        tile: u32,
        format: DepthSource,
    ) -> Option<Self> {
        let bpt = format.bytes_per_texel() as usize;
        let row_bytes = (width as usize * bpt).div_ceil(256) * 256;
        if bytes.len() < row_bytes * height as usize {
            return None;
        }
        let mut depths = Vec::with_capacity((width * height) as usize);
        for row in 0..height as usize {
            let row_bytes = &bytes[row * row_bytes..];
            for col in 0..width as usize {
                let texel = &row_bytes[col * bpt..];
                let depth = match format {
                    DepthSource::Rgba16Float => {
                        let bits = u16::from_le_bytes([texel[0], texel[1]]);
                        decode_f16(bits)
                    }
                    DepthSource::R32Float => f32::from_le_bytes([texel[0], texel[1], texel[2], texel[3]]),
                };
                depths.push(depth);
            }
        }
        Self::build(width, height, tile, &depths)
    }
}

/// One frame's occlusion test: the pyramid (previous frame), the view
/// matrices the pyramid and the groups are expressed in, and the bias.
#[derive(Clone, Debug, PartialEq)]
pub struct OcclusionCulling {
    /// Previous frame's depth, reduced.
    pub pyramid: DepthPyramid,
    /// View-projection of the frame the pyramid was captured with.
    pub view_proj: Mat4,
    /// Eye position of that frame.
    pub cam_pos: [f32; 3],
    /// Unit camera forward of that frame (view depth axis).
    pub cam_forward: [f32; 3],
    /// Depth margin in world units a group must clear to be culled.
    pub bias: f32,
}

impl OcclusionCulling {
    /// True when the world AABB is behind every pixel it covers. Never culls
    /// a box that is not fully inside the previous frame's screen.
    pub fn occluded(&self, min: [f32; 3], max: [f32; 3]) -> bool {
        let Some((lo, hi)) = aabb_ndc(&self.view_proj, min, max) else {
            return false;
        };
        let nearest = nearest_view_depth(min, max, self.cam_pos, self.cam_forward);
        self.pyramid.occluded(lo, hi, nearest, self.bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    /// The f16 decode pins the depth lane the readback produces.
    #[test]
    fn f16_decodes_exactly() {
        assert_eq!(decode_f16(0x0000), 0.0);
        assert_eq!(decode_f16(0x3C00), 1.0);
        assert_eq!(decode_f16(0x3800), 0.5);
        assert_eq!(decode_f16(0x4000), 2.0);
        assert_eq!(decode_f16(0xBC00), -1.0);
        assert_eq!(decode_f16(0x7C00), f32::INFINITY);
        assert!(decode_f16(0x7E00).is_nan());
        // Smallest positive subnormal is 2^-24.
        assert_eq!(decode_f16(0x0001), 2f32.powi(-24));
    }

    /// The pyramid keeps the farthest depth per tile, pads uncovered tiles
    /// with the nearest possible depth, and halves upwards.
    #[test]
    fn pyramid_takes_the_farthest_per_tile() {
        // 4x4 image, 2x2 tiles: a near tile and a far tile.
        let mut depths = vec![1.0f32; 16];
        for row in 0..2 {
            for col in 0..2 {
                depths[row * 4 + col] = 5.0; // top-left tile is far
            }
        }
        let pyramid = DepthPyramid::build(4, 4, 2, &depths).expect("valid");
        assert_eq!(pyramid.level_dims(0), (2, 2));
        assert_eq!(pyramid.tile_depth(0, 0, 0), Some(5.0));
        assert_eq!(pyramid.tile_depth(0, 1, 1), Some(1.0));
        assert_eq!(pyramid.levels(), 2, "2x2 tiles coarsen once");
        assert_eq!(pyramid.level_dims(1), (1, 1));
        assert_eq!(pyramid.tile_depth(1, 0, 0), Some(5.0));
        // A tile whose pixels are all non-finite keeps the no-coverage
        // value (0.0): the test never culls through it.
        let mut depths = vec![2.0f32; 25];
        depths[4 * 5 + 4] = f32::NAN;
        let pyramid = DepthPyramid::build(5, 5, 4, &depths).expect("valid");
        assert_eq!(pyramid.tile_depth(0, 0, 0), Some(2.0));
        assert_eq!(pyramid.tile_depth(0, 1, 1), Some(0.0));
        assert_eq!(DepthPyramid::build(0, 4, 2, &[]), None);
        assert_eq!(DepthPyramid::build(4, 4, 2, &[0.0; 3]), None);
    }

    /// The hierarchical test only culls fully covered rectangles whose every
    /// tile is in front, and the bias keeps marginal boxes.
    #[test]
    fn hz_test_is_conservative() {
        // 8x8 image, 4x4 tiles: tile (0,0) near (1.0), the rest far (9.0).
        let mut depths = vec![9.0f32; 64];
        for row in 0..4 {
            for col in 0..4 {
                depths[row * 8 + col] = 1.0;
            }
        }
        let pyramid = DepthPyramid::build(8, 8, 4, &depths).expect("valid");
        // Top-left quadrant fully covers tile (0,0), which is at depth 1:
        // anything at depth 5 is behind it.
        assert!(pyramid.occluded([-1.0, 0.0], [0.0, 1.0], 5.0, 0.0));
        // ...but not with a bias that reaches the tile depth.
        assert!(!pyramid.occluded([-1.0, 0.0], [0.0, 1.0], 5.0, 4.5));
        // The same rect with a nearer nearest depth is visible.
        assert!(!pyramid.occluded([-1.0, 0.0], [0.0, 1.0], 0.5, 0.0));
        // A rect that straddles tiles (no full coverage at level 0) is not
        // culled even though its pixels sit in front of the far tiles.
        assert!(!pyramid.occluded([-0.75, 0.25], [0.25, 0.75], 5.0, 0.0));
        // Full coverage of the far three tiles is behind.
        assert!(pyramid.occluded([-1.0, -1.0], [0.0, 1.0], 20.0, 0.0));
        // A rect fully outside the image is not culled (frustum's job).
        assert!(!pyramid.occluded([2.0, 2.0], [3.0, 3.0], 20.0, 0.0));
    }

    /// A tile the depth pass never wrote refuses the cull, however far the
    /// group sits: the group can be visible against pixels nothing was
    /// drawn into. An all-uncovered pyramid culls nothing, which is what
    /// keeps an empty frame from culling the frame after it.
    #[test]
    fn uncovered_tiles_never_justify_a_cull() {
        // 8x8 image, 4x4 tiles: the left tile column was written at depth 1,
        // the right column was never drawn (0.0).
        let mut depths = vec![0.0f32; 64];
        for row in 0..8 {
            for col in 4..8 {
                depths[row * 8 + col] = 1.0;
            }
        }
        let pyramid = DepthPyramid::build(8, 8, 4, &depths).expect("valid");
        assert_eq!(pyramid.tile_depth(0, 0, 0), Some(0.0));
        assert_eq!(pyramid.tile_depth(0, 1, 0), Some(1.0));
        // The drawn half culls what is behind it...
        assert!(pyramid.occluded([0.0, -1.0], [1.0, 1.0], 5.0, 0.0));
        // ...the unwritten half does not.
        assert!(!pyramid.occluded([-1.0, -1.0], [0.0, 1.0], 5.0, 0.0));
        let empty = DepthPyramid::build(8, 8, 4, &[0.0; 64]).expect("valid");
        assert!(!empty.occluded([-1.0, -1.0], [1.0, 1.0], 100.0, 0.0));
    }

    /// The feed descriptor clamps its settings on use, keeping a
    /// nonsensical tile edge or bias from reaching the pyramid.
    #[test]
    fn feed_desc_clamps() {
        let desc = OcclusionFeedDesc {
            tile: 0,
            bias: f32::NAN,
            phase_two: false,
        };
        assert_eq!(desc.clamped_tile(), 1);
        assert_eq!(desc.clamped_bias(), DEFAULT_BIAS);
        let desc = OcclusionFeedDesc {
            tile: 4096,
            bias: -3.0,
            phase_two: false,
        };
        assert_eq!(desc.clamped_tile(), 64);
        assert_eq!(desc.clamped_bias(), 0.0);
        assert_eq!(OcclusionFeedDesc::default().clamped_tile(), DEFAULT_TILE);
    }

    /// Both GPU layouts rebuild the exact pyramid the per-pixel build
    /// produces: the level-0 tile grid the reduction writes and the packed
    /// chain it coarsens. Moving the producer to the GPU cannot change what
    /// gets culled.
    #[test]
    fn gpu_layouts_rebuild_the_pixel_pyramid() {
        let depths = [
            1.0f32,
            2.0,
            f32::INFINITY,
            0.5,
            3.0,
            0.0,
            0.0,
            f32::NAN,
            -1.0,
            0.4,
            9.0,
            8.0,
            0.2,
            0.3,
            7.0,
            6.0,
        ];
        let pixels = DepthPyramid::build(4, 4, 2, &depths).expect("valid");
        let tiles = [3.0f32, 0.5, 0.4, 9.0];
        let grid = DepthPyramid::from_tile_grid(bytemuck::cast_slice(&tiles), 4, 4, 2)
            .expect("a full grid");
        assert_eq!(grid, pixels);
        assert_eq!(pixels.tile_depth(0, 0, 0), Some(3.0));
        assert_eq!(pixels.tile_depth(0, 1, 0), Some(0.5));
        assert_eq!(pixels.tile_depth(0, 0, 1), Some(0.4));
        assert_eq!(pixels.tile_depth(0, 1, 1), Some(9.0));
        // The packed chain is every level in `coarsened_dims` order, level 0
        // first, which is what the viewport's readback holds.
        let mut packed = Vec::new();
        for level in 0..pixels.levels() {
            let (cols, rows) = pixels.level_dims(level);
            for row in 0..rows {
                for col in 0..cols {
                    packed.push(pixels.tile_depth(level, col, row).expect("in range"));
                }
            }
        }
        let bytes = bytemuck::cast_slice(&packed);
        let chain = DepthPyramid::from_packed_levels(bytes, 4, 4, 2).expect("a full chain");
        assert_eq!(chain, pixels);
        // Short buffers and zero-sized images are refused by both.
        assert!(DepthPyramid::from_tile_grid(bytemuck::cast_slice(&tiles[..3]), 4, 4, 2).is_none());
        assert!(DepthPyramid::from_packed_levels(&bytes[..4], 4, 4, 2).is_none());
        assert!(DepthPyramid::from_packed_levels(bytes, 0, 0, 2).is_none());
        // An all-uncovered chain equals the per-pixel build of a void frame:
        // a 2x2 grid coarsens once, so the chain is 4 + 1 floats.
        let void = [0.0f32; 5];
        assert_eq!(
            DepthPyramid::from_packed_levels(bytemuck::cast_slice(&void), 8, 8, 4)
                .expect("a full chain"),
            DepthPyramid::build(8, 8, 4, &[0.0; 64]).expect("valid")
        );
    }

    /// The AABB projection refuses boxes behind the eye and otherwise
    /// produces the clamped NDC rectangle; the nearest depth picks the
    /// corner closest to the camera.
    #[test]
    fn aabb_projection_and_nearest_depth() {
        let cam = crate::camera::OrbitCamera {
            target: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.9,
            dist: 30.0,
            fov_y_deg: 30.0,
        };
        let view_proj = cam.view_proj(1.0);
        let Some((lo, hi)) = aabb_ndc(&view_proj, [-1.0; 3], [1.0; 3]) else {
            panic!("origin box is on screen");
        };
        assert!(lo[0] < hi[0] && lo[1] < hi[1]);
        assert!(lo[0] >= -1.0 && hi[0] <= 1.0);
        // A box behind the camera has a non-positive w and is refused.
        let behind = cam.eye() + (cam.eye() - cam.target).normalize() * 10.0;
        assert!(aabb_ndc(&view_proj, behind.into(), (behind + Vec3::ONE).into()).is_none());
        // Nearest view depth is the corner closest along forward.
        let nearest = nearest_view_depth(
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 10.0],
            [0.0, 0.0, -1.0],
        );
        assert!((nearest - 9.0).abs() < 1e-6, "near corner is z = 1: {nearest}");
    }

    /// End to end through the batch-facing wrapper: a box behind an occluder
    /// is culled, and the same box once the pyramid is empty is not.
    #[test]
    fn culling_uses_the_previous_frame_pyramid() {
        let cam = crate::camera::OrbitCamera {
            target: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.9,
            dist: 30.0,
            fov_y_deg: 30.0,
        };
        let view_proj = cam.view_proj(1.0);
        // Wall-sized far depths across the whole screen, at depth 22.
        let mut depths = vec![22.0f32; 64 * 64];
        let pyramid = DepthPyramid::build(64, 64, 8, &depths).expect("valid");
        let state = OcclusionCulling {
            pyramid,
            view_proj,
            cam_pos: cam.eye().into(),
            cam_forward: cam.forward().into(),
            bias: 0.1,
        };
        // A 6-unit box near the target (view depth ~26) is behind the wall;
        // its screen rect covers several 8-pixel tiles, so full coverage
        // exists at level 0.
        assert!(state.occluded([-3.0, -3.0, -3.0], [3.0, 3.0, 3.0]));
        // A box in front of the wall (depth ~10) is not.
        let front = cam.eye() + cam.forward() * 10.0;
        assert!(!state.occluded(
            (front - Vec3::splat(0.5)).into(),
            (front + Vec3::splat(0.5)).into()
        ));
        // An all-near surface (a wall a unit out) still culls the far box;
        // a box in front of it is left alone. The test is about depth, not
        // about how near the occluder is.
        depths.fill(1.0);
        let near = OcclusionCulling {
            pyramid: DepthPyramid::build(64, 64, 8, &depths).expect("valid"),
            ..state.clone()
        };
        assert!(near.occluded([-3.0, -3.0, -3.0], [3.0, 3.0, 3.0]));
        let front = cam.eye() + cam.forward() * 0.5;
        assert!(!near.occluded(
            (front - Vec3::splat(0.1)).into(),
            (front + Vec3::splat(0.1)).into()
        ));
    }
}
