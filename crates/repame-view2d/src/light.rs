//! 2D lighting data, validation, and the per-frame lighting plan.
//!
//! This is everything about lights that does not need a GPU: the snapshot
//! types a game fills in, the occluder validation, the shadow-wedge geometry,
//! and [`plan_lighting`], the deterministic CPU plan the GPU pass in
//! [`lightpass`](crate::lightpass) uploads. Keeping it pure means the same
//! code decides the pixels and answers the tests, and that a game can inspect
//! exactly what a frame will do before it draws.
//!
//! # Units
//!
//! * `position`, `radius`, `half_extent`, `shadow_softness`, and occluder
//!   points are world units in the same y-down space as
//!   [`SpriteInstance::center`](crate::SpriteInstance).
//! * `LightStop::at` is a normalized distance over `0..=1` of the light's
//!   `radius` (so a stop list is unit-free).
//! * `color` and `energy` are linear, unbounded multipliers; a light's
//!   contribution is `color * energy * ramp`.
//! * `LightStop::color` is `(rgb, a)`: rgb scales the light colour and `a`
//!   scales how much of the light reaches the pixel, matching the deleted
//!   `light2d.rs` convention.
//!
//! # Shading model
//!
//! A point light contributes, per fragment, `lit_rgb = color * ramp.rgb` and
//! `lit_a = energy * ramp.a * shadow`, where `ramp` is the
//! [`LightFalloff`] evaluated at `distance / radius` (`0` at the centre, `1`
//! at the rim) or sampled from the falloff texture in light-local uv. The
//! per-light [`LightBlend`] then decides what `lit` does to the accumulator
//! (Godot `light_blend_compute`):
//!
//! * [`LightBlend::Add`]: `acc.rgb += lit_rgb * lit_a`
//! * [`LightBlend::Sub`]: `acc.rgb -= lit_rgb * lit_a`
//! * [`LightBlend::Mix`]: `acc.rgb = mix(acc.rgb, lit_rgb, lit_a)`;
//!   `acc.a *= 1 - lit_a`
//!
//! The accumulator is `(rgb = light, a = item multiplier)`, which is what
//! makes all three modes expressible with fixed-function blend states in one
//! render target. Items consume it as
//! `final.rgb = item.rgb * modulate * acc.a + acc.rgb`.
//!
//! # Masking
//!
//! [`Light2d::light_mask`] and [`Occluder2d::light_mask`] are Godot
//! `item_cull_mask` / `occluder_light_mask` bits. A light is accumulated into
//! one accumulator layer per bit it sets ([`MAX_LIGHT_LAYERS`] layers,
//! bits `0..8`), and an item is lit by it exactly when the item's mask shares
//! that bit: the item reads the layer of its lowest set bit
//! ([`item_layer`]), which is the layer every light with that bit was
//! accumulated into. Items whose mask is `0` (or whose bits are all outside
//! the layer range) read the trailing identity layer and render unlit.
//!
//! # Not supported
//!
//! Each of these is a deliberate boundary, not an oversight:
//!
//! * No directional-light shadows, and no `shadow_color`: a shadowed fragment
//!   simply loses the light ([`ShadowFilter`] softens the edge with an angular
//!   penumbra of [`Light2d::shadow_softness`] world units). The global
//!   [`AmbientLight`] is the only unpositioned light and never shadows.
//! * No normal maps, no specular, and no per-item light culling: the
//!   accumulator is one light map covering the visible world rect, consumed by
//!   the sprite shader once per fragment, so a fragment cannot ask *which*
//!   light reached it.
//! * No SDF occluders and no occluder cull modes: every edge of a valid
//!   polygon shadows (see [`Occluder2d`]), which is what makes winding
//!   irrelevant.
//! * The accumulator is 8-bit linear per channel: a [`LightBlend::Sub`] light
//!   saturates at black, and a sum of additive lights saturates at white.
//! * Lights do not light the canvas path, only batches that name a light
//!   source (see [`SpriteBatch::set_light_source`](crate::SpriteBatch::set_light_source)).
//! * World coordinates far from the origin lose ramp precision, because the
//!   shader interpolates world positions as `f32`: keep the play area within
//!   roughly a million units of the origin.
//!
//! # Determinism and budget
//!
//! The plan walks the input slices in order and never consults a hash map, so
//! the same snapshot always yields the same plan, the same layer assignment,
//! and the same accumulation order (which matters for `Sub`/`Mix`). Nothing
//! consults a clock or a random number generator, the WGSL is one embedded
//! source compiled for every profile, and `stops_at` is the same expression
//! the shader evaluates, so a light looks the same in a dev and a packaged
//! run. Every limit in [`LightBudget`] is enforced by dropping entries in that
//! order and recording a typed [`LightIssue`] for each drop; nothing is
//! truncated silently. Occluders that are invalid are rejected once per frame,
//! again with a typed reason, instead of producing garbage shadows.

use glam::{Mat4, Vec2, Vec3};

/// Stops accepted in one [`LightFalloff::Stops`] ramp.
pub const MAX_FALLOFF_STOPS: usize = 8;

/// Accumulator layers a light can be masked into (bits `0..8`).
///
/// A light with a bit outside this range lights nothing on that bit; the plan
/// reports it as [`LightIssueKind::MaskBeyondLayerRange`].
pub const MAX_LIGHT_LAYERS: u32 = 8;

/// Points accepted in one [`Occluder2d`] before it is rejected.
pub const MAX_OCCLUDER_POINTS: usize = 256;

/// Tap channels one light's shadow mask holds: one per
/// [`ShadowFilter`] tap. A filter needing more taps than this cannot be
/// represented, which the const assertion below rules out.
pub const MAX_MASK_TAPS: usize = 4;

const _: () = assert!(MAX_MASK_TAPS >= 4, "Pcf4 needs one mask channel per tap");

/// Polygons up to this many points get the `O(edges^2)` self-intersection
/// check; larger ones keep the fast checks only.
pub const SELF_INTERSECT_CHECK_POINTS: usize = 64;

/// One stop of a light's radial ramp, matching the deleted `light2d.rs`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LightStop {
    /// Normalized distance from the light centre in `0..=1`.
    pub at: f32,
    /// Ramp value: rgb scales the light colour, a scales the ramp.
    pub color: [f32; 4],
}

/// How a light's value varies over its footprint.
#[derive(Clone, Debug, PartialEq)]
pub enum LightFalloff {
    /// A radial ramp over `0..=1` of the light's `radius`, interpolated
    /// linearly between ascending stops (`at` strictly ascending, all in
    /// `0..=1`). `t` below the first stop is the first colour, `t` above the
    /// last is the last colour. If the last stop is not zero the light still
    /// ends at the rim: fragments past `radius` are discarded.
    Stops(Vec<LightStop>),
    /// A falloff texture (Godot `PointLight2D.texture`) sampled in
    /// light-local uv: the texture covers `position + offset +- half_extent`,
    /// `rgb` scales the light colour and `a` scales the ramp, and fragments
    /// outside the texture footprint are discarded. `slot` selects an upload
    /// queued on the pass ([`LightTextureUpload`](crate::LightTextureUpload));
    /// a slot with no texture paints nothing.
    Texture {
        slot: u32,
        half_extent: Vec2,
        offset: Vec2,
    },
}

impl LightFalloff {
    /// The deleted `light2d.rs` default ramp: white at the centre, half at
    /// 0.5, black at the rim.
    pub fn stops(stops: impl Into<Vec<LightStop>>) -> Self {
        Self::Stops(stops.into())
    }

    /// Footprint half extent in world units: `radius` for stops, the
    /// texture's own half extent for a texture.
    pub fn half_extent(&self, radius: f32) -> Vec2 {
        match self {
            Self::Stops(_) => Vec2::splat(radius),
            Self::Texture { half_extent, .. } => *half_extent,
        }
    }
}

impl Default for LightFalloff {
    fn default() -> Self {
        Self::Stops(vec![
            LightStop {
                at: 0.0,
                color: [1.0, 1.0, 1.0, 1.0],
            },
            LightStop {
                at: 0.5,
                color: [0.5, 0.5, 0.5, 1.0],
            },
            LightStop {
                at: 1.0,
                color: [0.0, 0.0, 0.0, 1.0],
            },
        ])
    }
}

/// How a light combines with what is already lit (Godot `Light2D.BlendMode`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LightBlend {
    /// `acc += lit` (Godot default).
    #[default]
    Add,
    /// `acc -= lit`; the 8-bit accumulator saturates at zero.
    Sub,
    /// `acc = mix(acc, lit_rgb, lit_a)`, the accumulator multiplier included.
    Mix,
}

/// Shadow edge filtering (Godot `Light2D.ShadowFilter`).
///
/// The filter is angular: the silhouette is re-derived for
/// [`ShadowFilter::tap_count`] rotations around the light, spaced over the
/// penumbra and each weighted `1 / taps`, so the penumbra width in world units
/// is [`Light2d::shadow_softness`] at every occluder distance instead of a
/// screen-space blur.
///
/// Within one tap the edges of a solid occluder overlap, and the shadow there
/// is their *union* (a point is shadowed when the ray to the light crosses any
/// edge); only the taps themselves sum, as the average over the rotations. The
/// mask pass in [`lightpass`](crate::lightpass) implements exactly that: one
/// channel per tap, combined with max blending, summed by the light shader.
///
/// Each edge spaces its taps from its own midpoint's distance to the light,
/// not from one rotation for the whole occluder, so on a solid body two
/// silhouette edges' bands overlap and the union can reach full cover a few
/// pixels sooner on the shadow side than a lone edge would. The penumbra's mid
/// level still sits on the silhouette: the closed-polygon golden measures it
/// 1.5 px from the geometric edge, against a 3 px tolerance.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ShadowFilter {
    /// One hard edge.
    #[default]
    None,
    /// Four taps: a stepped penumbra, the cheapest soft edge.
    Pcf4,
}

impl ShadowFilter {
    /// Number of geometry copies drawn per silhouette edge.
    pub fn tap_count(self) -> u32 {
        match self {
            Self::None => 1,
            Self::Pcf4 => 4,
        }
    }

    /// Weight of one tap: taps sum to 1, so a fully covered fragment is fully
    /// shadowed.
    pub fn tap_weight(self) -> f32 {
        1.0 / self.tap_count() as f32
    }
}

/// A point light (Godot `PointLight2D`) in world space.
///
/// See the module docs for the shading contract. A light with a non-finite
/// position, a non-positive `radius`/`half_extent`, an empty/descending stop
/// list, or an all-zero `light_mask` is dropped by [`plan_lighting`] with a
/// typed issue rather than shaded half-way.
#[derive(Clone, Debug, PartialEq)]
pub struct Light2d {
    /// World-space light centre.
    pub position: Vec2,
    /// World-unit radius: the ramp's `1.0`, the shadow extrusion's bound, and
    /// the discard radius for [`LightFalloff::Stops`] (ignored when the
    /// falloff is a texture, whose footprint sets its own extent, but still
    /// used for nothing else).
    pub radius: f32,
    /// Peak light colour (linear rgb).
    pub color: [f32; 3],
    /// Brightness multiplier applied to the ramped colour.
    pub energy: f32,
    /// Radial ramp or falloff texture.
    pub falloff: LightFalloff,
    /// How the light combines with the lit result.
    pub blend: LightBlend,
    /// Cast shadows from the occluders whose mask it shares.
    pub shadows: bool,
    /// Shadow edge filtering.
    pub shadow_filter: ShadowFilter,
    /// Penumbra width in world units for [`ShadowFilter::Pcf4`]; ignored when
    /// the filter is [`ShadowFilter::None`].
    pub shadow_softness: f32,
    /// Layer bits this light lights and shadows cast against `0` = nothing.
    pub light_mask: u32,
}

impl Default for Light2d {
    fn default() -> Self {
        Self {
            position: Vec2::ZERO,
            radius: 128.0,
            color: [1.0, 1.0, 1.0],
            energy: 1.0,
            falloff: LightFalloff::default(),
            blend: LightBlend::Add,
            shadows: false,
            shadow_filter: ShadowFilter::None,
            shadow_softness: 4.0,
            light_mask: 1,
        }
    }
}

impl Light2d {
    /// A white additive light with the default ramp.
    pub fn point(position: Vec2, radius: f32) -> Self {
        Self {
            position,
            radius,
            ..Default::default()
        }
    }

    /// Ramp value at normalized distance `t` for a stops falloff, or `None`
    /// when the falloff is a texture (nothing to evaluate on the CPU).
    pub fn falloff_at(&self, t: f32) -> Option<[f32; 4]> {
        match &self.falloff {
            LightFalloff::Stops(stops) => Some(stops_at(stops, t)),
            LightFalloff::Texture { .. } => None,
        }
    }

    /// Light-local uv of a world point for a texture falloff: `(0,0)` is the
    /// texture's top-left, `(1,1)` its bottom-right, outside `0..=1` is off
    /// the light. Meaningless (but finite) for a stops falloff.
    pub fn texture_uv(&self, world: Vec2) -> Vec2 {
        match &self.falloff {
            LightFalloff::Texture {
                half_extent,
                offset,
                ..
            } => texture_uv(self.position, *offset, *half_extent, world),
            LightFalloff::Stops(_) => Vec2::splat(0.5),
        }
    }
}

/// Global light with no position (Godot `DirectionalLight2D` for flat 2D
/// items): `color * energy` is added to every lit layer it masks into, before
/// any light is accumulated.
///
/// It is additive, exactly like a covering [`LightBlend::Add`] light, so it
/// cannot darken a scene: darken items with their tint or a batch's
/// [`light_modulate`](crate::SpriteBatch::set_light_modulate) instead. It has
/// no shadows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmbientLight {
    /// Linear rgb.
    pub color: [f32; 3],
    /// Brightness multiplier.
    pub energy: f32,
    /// Layer bits this ambient reaches. Bits also define layers, exactly like
    /// a light's mask does.
    pub light_mask: u32,
}

impl Default for AmbientLight {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0],
            energy: 0.0,
            light_mask: 1,
        }
    }
}

/// A shadow-casting polygon (Godot `LightOccluder2D` + `OccluderPolygon2D`).
///
/// Every edge extrudes a shadow wedge away from the light, which gives *solid*
/// occluders: a point is shadowed exactly when the segment from the light to
/// the point crosses the polygon, regardless of the polygon's winding. That
/// is why there is no cull mode here; requiring a consistent winding is the
/// fragile part of a classic shadow volume, and dropping it means a game can
/// hand the engine any ordering of points from a level editor.
///
/// [`ValidatedOccluder::validate`] rejects polygons that would make the wedges
/// meaningless (too few points, non-finite, zero area, self-intersecting) and
/// the plan reports each rejection once per frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Occluder2d {
    /// World-space corners in order.
    pub points: Vec<Vec2>,
    /// Whether the last point connects back to the first. `false` is an open
    /// polyline: a thin wall casts a shadow but has no interior.
    pub closed: bool,
    /// Layer bits this occluder shadows (`light.light_mask & light_mask`).
    pub light_mask: u32,
}

impl Default for Occluder2d {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            closed: true,
            light_mask: 1,
        }
    }
}

impl Occluder2d {
    /// Occluder spanning the two corners (any corner order).
    pub fn rect(min: Vec2, max: Vec2) -> Self {
        let min = min.min(max);
        let max = min.max(max);
        Self {
            points: vec![
                min,
                Vec2::new(max.x, min.y),
                max,
                Vec2::new(min.x, max.y),
            ],
            closed: true,
            light_mask: 1,
        }
    }

    /// One level cell: `[col, row]` index at `cell_size` world units.
    pub fn cell(cell: [i32; 2], cell_size: f32) -> Self {
        let min = Vec2::new(cell[0] as f32 * cell_size, cell[1] as f32 * cell_size);
        Self::rect(min, min + Vec2::splat(cell_size))
    }

    /// A closed polygon through `points`.
    pub fn polygon(points: impl Into<Vec<Vec2>>) -> Self {
        Self {
            points: points.into(),
            closed: true,
            light_mask: 1,
        }
    }

    /// Inclusive point-in-polygon test (fills the deleted `light2d.rs`
    /// helper). Open polylines and invalid polygons report `false`.
    pub fn contains(&self, p: Vec2) -> bool {
        if !self.closed || self.points.len() < 3 {
            return false;
        }
        let mut inside = false;
        let n = self.points.len();
        for i in 0..n {
            let a = self.points[i];
            let b = self.points[(i + 1) % n];
            if (a.y > p.y) != (b.y > p.y) {
                let t = (p.y - a.y) / (b.y - a.y);
                if p.x < a.x + t * (b.x - a.x) {
                    inside = !inside;
                }
            }
        }
        inside
    }

    /// [`Self::rect`] with an explicit layer mask.
    pub fn rect_masked(min: Vec2, max: Vec2, light_mask: u32) -> Self {
        Self {
            light_mask,
            ..Self::rect(min, max)
        }
    }

    /// World-space bounds as `[min_x, min_y, max_x, max_y]`.
    pub fn aabb(&self) -> [f32; 4] {
        aabb_of(&self.points)
    }

    /// Check what [`plan_lighting`] checks, without a plan.
    pub fn validate(&self) -> Result<(), OccluderRejectReason> {
        ValidatedOccluder::validate(self).map(|_| ())
    }
}

/// Why an occluder was dropped for the frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OccluderRejectReason {
    /// Closed needs 3 points, open needs 2.
    TooFewPoints,
    /// More than [`MAX_OCCLUDER_POINTS`] points.
    TooManyPoints,
    /// A NaN or infinite point.
    NonFinitePoint,
    /// A closed polygon with zero area (collinear or duplicate points).
    DegenerateArea,
    /// The polygon crosses itself, which makes the silhouette ill-defined.
    SelfIntersecting,
    /// `light_mask` is `0`: it shadows no light, so it is not a caster.
    ZeroMask,
}

/// An occluder that passed validation, with its cached bounds.
#[derive(Clone, Debug, PartialEq)]
pub struct ValidatedOccluder {
    points: Vec<Vec2>,
    closed: bool,
    light_mask: u32,
    aabb: [f32; 4],
}

impl ValidatedOccluder {
    /// Validate a snapshot occluder.
    pub fn validate(occluder: &Occluder2d) -> Result<Self, OccluderRejectReason> {
        let n = occluder.points.len();
        if n > MAX_OCCLUDER_POINTS {
            return Err(OccluderRejectReason::TooManyPoints);
        }
        let min_points = if occluder.closed { 3 } else { 2 };
        if n < min_points {
            return Err(OccluderRejectReason::TooFewPoints);
        }
        if occluder.points.iter().any(|p| !p.is_finite()) {
            return Err(OccluderRejectReason::NonFinitePoint);
        }
        if occluder.light_mask == 0 {
            // Nothing is shadowed by it, so it is not a caster at all.
            return Err(OccluderRejectReason::ZeroMask);
        }
        let aabb = aabb_of(&occluder.points);
        if occluder.closed {
            // Crossings are checked before the area: the shoelace sum of a
            // bowtie cancels to zero (its lobes wind opposite ways), so a
            // degenerate-area test first would report a self-crossing polygon
            // as collinear. Collinear and duplicate-point polygons cast no
            // proper crossings, so they still fall through to the area test.
            if n <= SELF_INTERSECT_CHECK_POINTS && self_intersects(&occluder.points) {
                return Err(OccluderRejectReason::SelfIntersecting);
            }
            let width = aabb[2] - aabb[0];
            let height = aabb[3] - aabb[1];
            let scale = width.max(height);
            let area = signed_area(&occluder.points).abs();
            if !(area > scale * scale * 1e-6) {
                return Err(OccluderRejectReason::DegenerateArea);
            }
        }
        Ok(Self {
            points: occluder.points.clone(),
            closed: occluder.closed,
            light_mask: occluder.light_mask,
            aabb,
        })
    }

    /// Corners in world space.
    pub fn points(&self) -> &[Vec2] {
        &self.points
    }

    /// Whether the last corner connects back to the first.
    pub fn closed(&self) -> bool {
        self.closed
    }

    /// Layer bits this occluder shadows.
    pub fn light_mask(&self) -> u32 {
        self.light_mask
    }

    /// Bounds as `[min_x, min_y, max_x, max_y]`.
    pub fn aabb(&self) -> [f32; 4] {
        self.aabb
    }
}

/// One triangle-fan quad of a light's shadow, in world space, drawn as
/// `[near_a, near_b, far_b, far_a]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowWedge {
    /// Corners: two on the occluder edge, two extruded away from the light.
    pub quad: [Vec2; 4],
    /// This tap's share of the shadow: taps sum to `1.0`.
    pub weight: f32,
    /// Which tap this quad belongs to, `0..taps-1` in the order
    /// [`shadow_wedges`] emitted them.
    pub tap: u32,
}

impl ShadowWedge {
    /// The quad's contribution to the shadow mask, one entry per mask channel:
    /// `weight` in [`Self::tap`]'s channel and zero in the others.
    ///
    /// Overlapping quads of one tap max-blend into their shared channel, so a
    /// solid occluder's edges union per tap (a point is shadowed when the ray
    /// to the light crosses any edge), and the light shader sums the channels,
    /// which is the taps' average.
    pub fn mask_weights(&self) -> [f32; MAX_MASK_TAPS] {
        let mut out = [0.0; MAX_MASK_TAPS];
        if let Some(slot) = out.get_mut(self.tap as usize) {
            *slot = self.weight;
        }
        out
    }
}

/// How many occluders a light's shadow pass consumed and left behind.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShadowCasters {
    /// Occluders whose mask and bounds qualified.
    pub used: usize,
    /// Qualified occluders dropped by `max_casters`.
    pub dropped: usize,
}

/// Per-frame limits, all enforced in input order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightBudget {
    /// Lights shaded per frame; extra lights are reported and skipped.
    pub max_lights: usize,
    /// Layers the accumulator may have (`1..=`[`MAX_LIGHT_LAYERS`]; each light
    /// costs one pass per layer it lights).
    pub max_layers: u32,
    /// Occluders considered per light.
    pub max_shadow_casters: usize,
    /// Shadow vertices (`6` per wedge) across the frame; the light that would
    /// exceed it is shaded without shadows.
    pub max_shadow_vertices: usize,
}

impl Default for LightBudget {
    fn default() -> Self {
        Self {
            max_lights: 16,
            max_layers: MAX_LIGHT_LAYERS,
            max_shadow_casters: 64,
            max_shadow_vertices: 12_288,
        }
    }
}

/// A light that made it into the frame, in the order it is accumulated.
#[derive(Clone, Debug, PartialEq)]
pub struct PlannedLight {
    /// Index into the input slice the plan was built from.
    pub source: usize,
    /// Layer bits this light is accumulated into (a subset of its mask).
    pub layers: u32,
    /// World-space centre.
    pub center: Vec2,
    /// Footprint half extent (see [`LightFalloff::half_extent`]).
    pub half_extent: Vec2,
    /// Distance from the centre to the farthest footprint corner.
    pub extent: f32,
    /// Peak colour.
    pub color: [f32; 3],
    /// Brightness multiplier.
    pub energy: f32,
    /// Blend into the accumulator.
    pub blend: LightBlend,
    /// Validated ramp stops (`stop_count` of `MAX_FALLOFF_STOPS` are live).
    pub stops: [LightStop; MAX_FALLOFF_STOPS],
    /// Live stop count; `0` means the falloff is a texture.
    pub stop_count: u32,
    /// Falloff texture slot, used when `stop_count == 0`.
    pub texture_slot: u32,
    /// Falloff texture offset from the light centre (texture mode).
    pub texture_offset: Vec2,
    /// Falloff texture half extent (texture mode; `1.0` square otherwise).
    pub texture_half_extent: Vec2,
    /// Whether shadows were built (a light with no casters is unshadowed).
    pub shadowed: bool,
    /// Shadow edge filter used for the wedges.
    pub shadow_filter: ShadowFilter,
    /// World-space shadow wedges, `shadow_filter.tap_count()` per silhouette
    /// edge.
    pub wedges: Vec<ShadowWedge>,
}

impl PlannedLight {
    /// Vertices the wedges need (`6` per quad, triangle list).
    pub fn wedge_vertex_count(&self) -> usize {
        self.wedges.len() * 6
    }

    /// Layer indices this light is accumulated into, ascending.
    pub fn layer_indices(&self) -> impl Iterator<Item = u32> + '_ {
        (0..MAX_LIGHT_LAYERS).filter(move |bit| self.layers & (1 << bit) != 0)
    }
}

/// Why a light was dropped or altered for the frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LightIssueKind {
    /// `light_mask` was `0`: it lights nothing.
    ZeroMask,
    /// Mask bits at or above [`MAX_LIGHT_LAYERS`] were dropped.
    MaskBeyondLayerRange {
        /// The bits that could not be used.
        dropped: u32,
    },
    /// A position that is not finite.
    NonFinitePosition,
    /// A colour, energy, or texture offset that is not finite.
    NonFiniteValue,
    /// The ambient light's colour or energy is not finite; it is treated as
    /// black for the frame. Reported with `source == usize::MAX`.
    NonFiniteAmbient,
    /// `radius` (stops) or `half_extent` (texture) is not a positive, finite
    /// extent, so the light has no footprint.
    ZeroExtent,
    /// A stops falloff with no stops.
    StopsEmpty,
    /// More stops than [`MAX_FALLOFF_STOPS`].
    TooManyStops {
        /// Stops supplied.
        count: usize,
        /// Stops allowed.
        max: usize,
    },
    /// A stop's `at` was not finite.
    StopNotFinite,
    /// A stop's `at` was outside `0..=1`.
    StopOutOfRange {
        /// The offending `at`.
        at: f32,
    },
    /// A stop's `at` did not ascend.
    StopsNotAscending {
        /// The offending `at`.
        at: f32,
        /// The previous stop's `at`.
        previous: f32,
    },
    /// More lights than [`LightBudget::max_lights`].
    OverLightBudget {
        /// The budget.
        max: usize,
    },
    /// Qualified occluders dropped by [`LightBudget::max_shadow_casters`].
    ShadowCastersExhausted {
        /// Occluders used.
        used: usize,
        /// Occluders dropped.
        dropped: usize,
    },
    /// The light's wedges would exceed [`LightBudget::max_shadow_vertices`];
    /// the light is accumulated without shadows.
    ShadowVertexBudget {
        /// Vertices the light needed.
        needed: usize,
        /// The budget.
        max: usize,
    },
}

/// A typed reason one light was altered or dropped this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightIssue {
    /// Index into the input slice the plan was built from, or
    /// [`AMBIENT_SOURCE`] when the issue is the ambient light's.
    pub source: usize,
    /// What happened.
    pub kind: LightIssueKind,
}

/// The [`LightIssue::source`] an ambient-light issue carries: the ambient is
/// not one of the indexed lights.
pub const AMBIENT_SOURCE: usize = usize::MAX;

/// An occluder rejected for the frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OccluderReject {
    /// Index into the input slice the plan was built from.
    pub index: usize,
    /// Why it was rejected.
    pub reason: OccluderRejectReason,
}

/// Everything the GPU pass needs for one frame, decided on the CPU.
#[derive(Clone, Debug, PartialEq)]
pub struct LightingPlan {
    /// Visible world rect the accumulator covers: `[x, y, w, h]` with `x/y`
    /// the top-left (the accumulator's uv origin).
    pub world_rect: [f32; 4],
    /// Accumulator layers, including the trailing identity layer items with
    /// no layer read. At least `2`, because a one-layer array texture is
    /// unsafe to sample on GL/ANGLE (see
    /// [`atlas_shape`](crate::batch::atlas_shape)).
    pub layers: u32,
    /// `layers - 1`: the highest index an item may sample.
    pub layer_clamp: u32,
    /// Lights in accumulation order.
    pub lights: Vec<PlannedLight>,
    /// The ambient light, resolved per layer by [`Self::ambient_rgb`].
    pub ambient: AmbientLight,
    /// Every light the plan altered or dropped, in input order.
    pub issues: Vec<LightIssue>,
    /// Occluders rejected before any light was planned.
    pub occluder_rejects: Vec<OccluderReject>,
    /// Lights skipped because their footprint misses the visible rect. Culling
    /// is not a budget drop, so these are counted, not reported.
    pub culled: usize,
}

impl LightingPlan {
    /// Vertices all lights' wedges need.
    pub fn wedge_vertices(&self) -> usize {
        self.lights.iter().map(PlannedLight::wedge_vertex_count).sum()
    }

    /// Ambient rgb added to `layer` before any light: `color * energy` when
    /// the ambient's mask reaches the layer. The trailing identity layer is
    /// always `[0, 0, 0]`, so unlit items stay exactly unlit.
    pub fn ambient_rgb(&self, layer: u32) -> [f32; 3] {
        if layer + 1 >= self.layers || self.ambient.light_mask & (1u32 << layer) == 0 {
            return [0.0; 3];
        }
        [
            self.ambient.color[0] * self.ambient.energy,
            self.ambient.color[1] * self.ambient.energy,
            self.ambient.color[2] * self.ambient.energy,
        ]
    }
}

/// Ramp value at normalized distance `t` for a stops list (clamped, linearly
/// interpolated, ascending stops assumed: [`plan_lighting`] rejects the rest).
pub fn stops_at(stops: &[LightStop], t: f32) -> [f32; 4] {
    let Some(first) = stops.first() else {
        return [0.0; 4];
    };
    let t = if t.is_finite() { t.clamp(0.0, 1.0) } else { 1.0 };
    let mut ramp = first.color;
    let mut previous = 0.0f32;
    for stop in &stops[1..] {
        // `max(...)` rather than a branch so this is character for character
        // what `shaders/light_disc.wgsl::stops_ramp` computes.
        let span = (stop.at - previous).max(1e-6);
        let w = ((t - previous) / span).clamp(0.0, 1.0);
        ramp = [
            ramp[0] + (stop.color[0] - ramp[0]) * w,
            ramp[1] + (stop.color[1] - ramp[1]) * w,
            ramp[2] + (stop.color[2] - ramp[2]) * w,
            ramp[3] + (stop.color[3] - ramp[3]) * w,
        ];
        previous = stop.at;
    }
    ramp
}

/// Light-local uv of `world` for a texture falloff centred on `center +
/// offset` and covering `+-half_extent`. Exact `0..=1` inside the footprint.
pub fn texture_uv(center: Vec2, offset: Vec2, half_extent: Vec2, world: Vec2) -> Vec2 {
    let half = Vec2::new(
        if half_extent.x.abs() > 1e-6 {
            half_extent.x
        } else {
            1.0
        },
        if half_extent.y.abs() > 1e-6 {
            half_extent.y
        } else {
            1.0
        },
    );
    (world - center - offset) / (half * 2.0) + Vec2::splat(0.5)
}

/// Layer an item with `light_mask` reads: its lowest set bit, or
/// `MAX_LIGHT_LAYERS` (the identity layer) when no bit is in range.
pub fn item_layer(light_mask: u32) -> u32 {
    light_mask.trailing_zeros().min(MAX_LIGHT_LAYERS)
}

/// The visible world rect of a y-down orthographic `view_proj`, or `None` when
/// the matrix is not invertible or maps the view to a degenerate area.
pub fn visible_world_rect(view_proj: Mat4) -> Option<[f32; 4]> {
    if !view_proj.is_finite() {
        return None;
    }
    let inverse = view_proj.inverse();
    if !inverse.is_finite() || inverse.determinant() == 0.0 {
        return None;
    }
    let mut min = Vec2::splat(f32::INFINITY);
    let mut max = Vec2::splat(f32::NEG_INFINITY);
    for corner in [[-1.0f32, -1.0f32], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]] {
        let p = inverse.project_point3(Vec3::new(corner[0], corner[1], 0.0));
        if !p.is_finite() {
            return None;
        }
        let p = p.truncate();
        min = min.min(p);
        max = max.max(p);
    }
    let size = max - min;
    if !(size.x > 1e-3 && size.y > 1e-3) {
        return None;
    }
    Some([min.x, min.y, size.x, size.y])
}

/// Build every wedge a light's shadows need, in world space.
///
/// Each edge of each qualifying occluder contributes one quad per tap. The
/// quad's far corners are pushed to `1.5 * extent + 2 * softness` away from
/// the light (and never closer than 1.05x their own distance), which covers
/// the footprint of the light plus the filter's angular spread. A corner
/// exactly at the light keeps its position, so a light sitting on an occluder
/// vertex produces a degenerate quad instead of a NaN.
///
/// Wedges come out in occluder order, then edge order, then tap order, and
/// each carries its [`ShadowWedge::tap`] so the mask pass can put it in the
/// right channel whatever order a game's occluders arrive in.
///
/// Occluders are taken in slice order up to `max_casters`; `out` is appended
/// to.
#[allow(clippy::too_many_arguments)]
pub fn shadow_wedges(
    center: Vec2,
    extent: f32,
    filter: ShadowFilter,
    softness: f32,
    light_mask: u32,
    light_aabb: [f32; 4],
    occluders: &[ValidatedOccluder],
    max_casters: usize,
    out: &mut Vec<ShadowWedge>,
) -> ShadowCasters {
    let mut casters = ShadowCasters::default();
    if !(extent.is_finite() && extent > 0.0) {
        return casters;
    }
    let softness = if softness.is_finite() {
        softness.max(0.0)
    } else {
        0.0
    };
    let taps = filter.tap_count();
    let weight = filter.tap_weight();
    debug_assert!(
        taps as usize <= MAX_MASK_TAPS,
        "a filter with more taps than mask channels would lose its penumbra"
    );
    let reach = extent * 1.5 + softness * 2.0;
    for occluder in occluders {
        if occluder.light_mask & light_mask == 0 || !aabb_overlap(occluder.aabb, light_aabb) {
            continue;
        }
        if casters.used >= max_casters {
            casters.dropped += 1;
            continue;
        }
        casters.used += 1;
        let points = &occluder.points;
        let n = points.len();
        let edges = if occluder.closed { n } else { n - 1 };
        for edge in 0..edges {
            let a = points[edge];
            let b = points[(edge + 1) % n];
            let mid = (a + b) * 0.5;
            let delta = if taps > 1 && softness > 0.0 {
                (softness / (mid - center).length().max(1e-3)).min(0.5)
            } else {
                0.0
            };
            for tap in 0..taps {
                let offset = if taps == 1 {
                    0.0
                } else {
                    TAP_OFFSETS[tap as usize % TAP_OFFSETS.len()]
                };
                let angle = offset * delta;
                let ar = rotate_about(center, a, angle);
                let br = rotate_about(center, b, angle);
                let af = extrude(center, ar, reach);
                let bf = extrude(center, br, reach);
                out.push(ShadowWedge {
                    quad: [ar, br, bf, af],
                    weight,
                    tap,
                });
            }
        }
    }
    casters
}

/// Tap offsets as fractions of the penumbra's angular half width, ascending
/// and symmetric; fixed so a frame is reproducible.
const TAP_OFFSETS: [f32; 4] = [-0.75, -0.25, 0.25, 0.75];

/// Plan one frame: validate, cull, budget, and build the shadow geometry.
///
/// Pure and deterministic: input order in, same output out, no hash iteration
/// and no time or randomness. Drops and rejections are returned as typed
/// entries in [`LightingPlan::issues`] / [`LightingPlan::occluder_rejects`]
/// and never truncate quietly.
pub fn plan_lighting(
    lights: &[Light2d],
    occluders: &[Occluder2d],
    ambient: &AmbientLight,
    world_rect: [f32; 4],
    budget: &LightBudget,
) -> LightingPlan {
    let mut plan = LightingPlan {
        world_rect,
        layers: 2,
        layer_clamp: 1,
        lights: Vec::new(),
        ambient: *ambient,
        issues: Vec::new(),
        occluder_rejects: Vec::new(),
        culled: 0,
    };
    if !ambient.color.iter().all(|c| c.is_finite()) || !ambient.energy.is_finite() {
        // A NaN would reach the layer clears, so the frame falls back to black
        // and reports it.
        plan.ambient = AmbientLight {
            color: [0.0; 3],
            ..*ambient
        };
        plan.issues.push(LightIssue {
            source: AMBIENT_SOURCE,
            kind: LightIssueKind::NonFiniteAmbient,
        });
    }
    let mut validated: Vec<ValidatedOccluder> = Vec::with_capacity(occluders.len());
    for (index, occluder) in occluders.iter().enumerate() {
        match ValidatedOccluder::validate(occluder) {
            Ok(valid) => validated.push(valid),
            Err(reason) => plan.occluder_rejects.push(OccluderReject { index, reason }),
        }
    }

    let max_layers = budget.max_layers.clamp(1, MAX_LIGHT_LAYERS);
    let layer_bits = if max_layers >= 32 {
        u32::MAX
    } else {
        (1u32 << max_layers) - 1
    };
    let mut used_layers = if ambient.light_mask & layer_bits != 0 {
        (32 - (ambient.light_mask & layer_bits).leading_zeros()).max(1)
    } else {
        0
    };
    let view_bounds = [
        world_rect[0],
        world_rect[1],
        world_rect[0] + world_rect[2],
        world_rect[1] + world_rect[3],
    ];
    let mut wedge_vertices = 0usize;

    for (source, light) in lights.iter().enumerate() {
        let issue = |plan: &mut LightingPlan, kind: LightIssueKind| {
            plan.issues.push(LightIssue { source, kind });
        };
        if plan.lights.len() >= budget.max_lights {
            issue(
                &mut plan,
                LightIssueKind::OverLightBudget {
                    max: budget.max_lights,
                },
            );
            continue;
        }
        if light.light_mask == 0 {
            issue(&mut plan, LightIssueKind::ZeroMask);
            continue;
        }
        if !light.position.is_finite() {
            issue(&mut plan, LightIssueKind::NonFinitePosition);
            continue;
        }
        if !light.color.iter().all(|c| c.is_finite()) || !light.energy.is_finite() {
            issue(&mut plan, LightIssueKind::NonFiniteValue);
            continue;
        }
        let dropped = light.light_mask & !layer_bits;
        if dropped != 0 {
            issue(&mut plan, LightIssueKind::MaskBeyondLayerRange { dropped });
        }
        let layers = light.light_mask & layer_bits;
        if layers == 0 {
            issue(&mut plan, LightIssueKind::ZeroMask);
            continue;
        }
        let mut stops = [LightStop::default(); MAX_FALLOFF_STOPS];
        let mut stop_count = 0u32;
        let mut texture_slot = 0u32;
        let mut texture_offset = Vec2::ZERO;
        let mut texture_half_extent = Vec2::ONE;
        match &light.falloff {
            LightFalloff::Stops(list) => {
                if list.is_empty() {
                    issue(&mut plan, LightIssueKind::StopsEmpty);
                    continue;
                }
                if list.len() > MAX_FALLOFF_STOPS {
                    issue(
                        &mut plan,
                        LightIssueKind::TooManyStops {
                            count: list.len(),
                            max: MAX_FALLOFF_STOPS,
                        },
                    );
                    continue;
                }
                let mut ok = true;
                for (i, stop) in list.iter().enumerate() {
                    if !stop.at.is_finite() || stop.color.iter().any(|c| !c.is_finite()) {
                        issue(&mut plan, LightIssueKind::StopNotFinite);
                        ok = false;
                        break;
                    }
                    if !(0.0..=1.0).contains(&stop.at) {
                        issue(&mut plan, LightIssueKind::StopOutOfRange { at: stop.at });
                        ok = false;
                        break;
                    }
                    if i > 0 && stop.at <= list[i - 1].at {
                        issue(
                            &mut plan,
                            LightIssueKind::StopsNotAscending {
                                at: stop.at,
                                previous: list[i - 1].at,
                            },
                        );
                        ok = false;
                        break;
                    }
                    stops[i] = *stop;
                }
                if !ok {
                    continue;
                }
                stop_count = list.len() as u32;
            }
            LightFalloff::Texture {
                slot,
                half_extent,
                offset,
            } => {
                texture_slot = *slot;
                texture_offset = *offset;
                texture_half_extent = *half_extent;
            }
        }
        let half_extent = light.falloff.half_extent(light.radius);
        if !half_extent.is_finite()
            || !(half_extent.x > 0.0 && half_extent.y > 0.0)
            || !light.radius.is_finite()
        {
            issue(&mut plan, LightIssueKind::ZeroExtent);
            continue;
        }
        if !texture_offset.is_finite() {
            issue(&mut plan, LightIssueKind::NonFiniteValue);
            continue;
        }
        let extent = half_extent.length();
        let aabb = [
            light.position.x - half_extent.x,
            light.position.y - half_extent.y,
            light.position.x + half_extent.x,
            light.position.y + half_extent.y,
        ];
        if !aabb_overlap(aabb, view_bounds) {
            plan.culled += 1;
            continue;
        }

        let mut wedges = Vec::new();
        let mut shadowed = false;
        if light.shadows && light.shadow_filter.tap_count() > 0 {
            let casters = shadow_wedges(
                light.position,
                extent,
                light.shadow_filter,
                light.shadow_softness,
                light.light_mask,
                aabb,
                &validated,
                budget.max_shadow_casters,
                &mut wedges,
            );
            if casters.dropped > 0 {
                issue(
                    &mut plan,
                    LightIssueKind::ShadowCastersExhausted {
                        used: casters.used,
                        dropped: casters.dropped,
                    },
                );
            }
            let needed = wedges.len() * 6;
            if needed > 0 && wedge_vertices + needed > budget.max_shadow_vertices {
                wedges.clear();
                issue(
                    &mut plan,
                    LightIssueKind::ShadowVertexBudget {
                        needed,
                        max: budget.max_shadow_vertices,
                    },
                );
            } else {
                wedge_vertices += needed;
                shadowed = !wedges.is_empty();
            }
        }

        used_layers = used_layers.max(32 - layers.leading_zeros());
        plan.lights.push(PlannedLight {
            source,
            layers,
            center: light.position,
            half_extent,
            extent,
            color: light.color,
            energy: light.energy,
            blend: light.blend,
            stops,
            stop_count,
            texture_slot,
            texture_offset,
            texture_half_extent,
            shadowed,
            shadow_filter: light.shadow_filter,
            wedges,
        });
    }

    plan.layers = (used_layers + 1).max(2);
    plan.layer_clamp = plan.layers - 1;
    plan
}

fn aabb_of(points: &[Vec2]) -> [f32; 4] {
    let mut aabb = [
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    for p in points {
        aabb[0] = aabb[0].min(p.x);
        aabb[1] = aabb[1].min(p.y);
        aabb[2] = aabb[2].max(p.x);
        aabb[3] = aabb[3].max(p.y);
    }
    aabb
}

/// Whether two axis-aligned boxes overlap. Both boxes are corners, not
/// corner-plus-size: `[min_x, min_y, max_x, max_y]`, the same convention as
/// [`Occluder2d::aabb`], a light's footprint in [`plan_lighting`], and its
/// visible world rect.
fn aabb_overlap(a_min_max: [f32; 4], b_min_max: [f32; 4]) -> bool {
    a_min_max[0] <= b_min_max[2]
        && a_min_max[2] >= b_min_max[0]
        && a_min_max[1] <= b_min_max[3]
        && a_min_max[3] >= b_min_max[1]
}

fn signed_area(points: &[Vec2]) -> f32 {
    let n = points.len();
    let mut area = 0.0;
    for i in 0..n {
        let a = points[i];
        let b = points[(i + 1) % n];
        area += a.x * b.y - b.x * a.y;
    }
    area * 0.5
}

/// Proper crossings only: two edges that merely touch at a point are a pinch,
/// not a self-intersection, and the wedge geometry survives a pinch.
fn self_intersects(points: &[Vec2]) -> bool {
    let n = points.len();
    for i in 0..n {
        let a0 = points[i];
        let a1 = points[(i + 1) % n];
        for j in (i + 1)..n {
            if j == i || (j + 1) % n == i || j == (i + 1) % n {
                continue;
            }
            let b0 = points[j];
            let b1 = points[(j + 1) % n];
            if segments_cross(a0, a1, b0, b1) {
                return true;
            }
        }
    }
    false
}

fn segments_cross(a0: Vec2, a1: Vec2, b0: Vec2, b1: Vec2) -> bool {
    let d1 = cross(b1 - b0, a0 - b0);
    let d2 = cross(b1 - b0, a1 - b0);
    let d3 = cross(a1 - a0, b0 - a0);
    let d4 = cross(a1 - a0, b1 - a0);
    d1 * d2 < 0.0 && d3 * d4 < 0.0
}

fn cross(a: Vec2, b: Vec2) -> f32 {
    a.x * b.y - a.y * b.x
}

fn rotate_about(center: Vec2, p: Vec2, angle: f32) -> Vec2 {
    if angle == 0.0 {
        return p;
    }
    let (s, c) = angle.sin_cos();
    let d = p - center;
    center + Vec2::new(d.x * c - d.y * s, d.x * s + d.y * c)
}

/// Push `p` away from `c` to `reach` (never closer than its own distance).
/// A point exactly at the centre is returned unchanged: the caller's quad
/// degenerates to a triangle, which still covers the right angle.
fn extrude(c: Vec2, p: Vec2, reach: f32) -> Vec2 {
    let d = p - c;
    let len = d.length();
    if len > 1e-6 {
        c + d / len * reach.max(len * 1.05)
    } else {
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stops3() -> Vec<LightStop> {
        vec![
            LightStop {
                at: 0.0,
                color: [1.0, 1.0, 1.0, 1.0],
            },
            LightStop {
                at: 0.5,
                color: [0.5, 0.5, 0.5, 1.0],
            },
            LightStop {
                at: 1.0,
                color: [0.0, 0.0, 0.0, 1.0],
            },
        ]
    }

    #[test]
    fn stops_ramp_interpolates_at_known_distances() {
        let ramp = stops3();
        assert_eq!(stops_at(&ramp, 0.0), [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(stops_at(&ramp, 0.5), [0.5, 0.5, 0.5, 1.0]);
        assert_eq!(stops_at(&ramp, 1.0), [0.0, 0.0, 0.0, 1.0]);
        // Mid-segment: halfway between stop 0 and stop 1.
        let mid = stops_at(&ramp, 0.25);
        assert!((mid[0] - 0.75).abs() < 1e-6, "got {mid:?}");
        // Clamped outside the ramp; NaN cannot escape.
        assert_eq!(stops_at(&ramp, -1.0), [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(stops_at(&ramp, 4.0), [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(stops_at(&ramp, f32::NAN), [0.0, 0.0, 0.0, 1.0]);
        // A single stop is a flat light.
        let flat = [LightStop {
            at: 0.3,
            color: [0.25, 0.25, 0.25, 0.5],
        }];
        assert_eq!(stops_at(&flat, 0.0), [0.25, 0.25, 0.25, 0.5]);
        assert_eq!(stops_at(&flat, 1.0), [0.25, 0.25, 0.25, 0.5]);
        // No stops at all is zero light, not a panic.
        assert_eq!(stops_at(&[], 0.5), [0.0; 4]);
    }

    #[test]
    fn radius_ramp_matches_stops_at_known_distances() {
        let light = Light2d {
            position: Vec2::ZERO,
            radius: 100.0,
            falloff: LightFalloff::stops(stops3()),
            ..Default::default()
        };
        let at = |d: f32| light.falloff_at(d / light.radius).expect("stops ramp");
        assert_eq!(at(0.0), [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(at(50.0), [0.5, 0.5, 0.5, 1.0]);
        assert_eq!(at(100.0), [0.0, 0.0, 0.0, 1.0]);
        assert!((at(25.0)[0] - 0.75).abs() < 1e-6);
        // Past the rim the ramp keeps its last value: the GPU discards there.
        assert_eq!(at(250.0), [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn texture_falloff_maps_world_to_local_uv() {
        let light = Light2d {
            position: Vec2::new(100.0, 100.0),
            radius: 64.0,
            falloff: LightFalloff::Texture {
                slot: 0,
                half_extent: Vec2::new(64.0, 32.0),
                offset: Vec2::new(8.0, -4.0),
            },
            ..Default::default()
        };
        assert_eq!(light.falloff_at(0.5), None, "texture ramps stay on the GPU");
        let uv = |x: f32, y: f32| light.texture_uv(Vec2::new(x, y));
        let center = Vec2::new(108.0, 96.0);
        assert!((uv(center.x, center.y) - Vec2::splat(0.5)).length() < 1e-6);
        assert!((uv(center.x - 64.0, center.y) - Vec2::new(0.0, 0.5)).length() < 1e-6);
        assert!((uv(center.x, center.y + 32.0) - Vec2::new(0.5, 1.0)).length() < 1e-6);
        // Outside the footprint stays outside, so the shader can discard.
        assert!(uv(center.x + 100.0, center.y).x > 1.0);
        // Zero extents cannot produce NaN.
        let degenerate = Light2d {
            falloff: LightFalloff::Texture {
                slot: 0,
                half_extent: Vec2::ZERO,
                offset: Vec2::ZERO,
            },
            ..Default::default()
        };
        assert!(degenerate.texture_uv(Vec2::new(3.0, 4.0)).is_finite());
    }

    #[test]
    fn item_layers_follow_the_lowest_shared_bit() {
        assert_eq!(item_layer(1), 0);
        assert_eq!(item_layer(0b010), 1);
        assert_eq!(item_layer(0b1001), 0);
        assert_eq!(item_layer(0b1000_0000), 7);
        // Nothing in range reads the identity layer.
        assert_eq!(item_layer(0), MAX_LIGHT_LAYERS);
        assert_eq!(item_layer(1 << 9), MAX_LIGHT_LAYERS);
    }

    #[test]
    fn mask_filtering_selects_lit_layers() {
        let dim = AmbientLight {
            color: [1.0, 1.0, 1.0],
            energy: 0.25,
            light_mask: 0b101,
        };
        let lights = vec![
            Light2d {
                position: Vec2::new(0.0, 0.0),
                light_mask: 0b011,
                ..Default::default()
            },
            Light2d {
                position: Vec2::new(0.0, 0.0),
                light_mask: 0b100,
                ..Default::default()
            },
        ];
        let plan = plan_lighting(
            &lights,
            &[],
            &dim,
            [-100.0, -100.0, 200.0, 200.0],
            &LightBudget::default(),
        );
        assert_eq!(plan.lights.len(), 2);
        assert_eq!(plan.lights[0].layer_indices().collect::<Vec<_>>(), [0, 1]);
        assert_eq!(plan.lights[1].layer_indices().collect::<Vec<_>>(), [2]);
        // Layers up to the highest bit, plus the identity layer.
        assert_eq!((plan.layers, plan.layer_clamp), (4, 3));
        // An item on bit 1 sees light 0; an item on bit 2 sees light 1 only.
        assert_eq!(item_layer(0b010), 1);
        assert_eq!(plan.lights[0].layers & (1 << item_layer(0b010)), 1 << 1);
        assert_eq!(plan.lights[1].layers & (1 << item_layer(0b010)), 0);
        assert_eq!(plan.lights[1].layers & (1 << item_layer(0b100)), 1 << 2);
        // Ambient mask 0b101: layers 0 and 2, not 1, and never the identity.
        assert_eq!(plan.ambient_rgb(0), [0.25, 0.25, 0.25]);
        assert_eq!(plan.ambient_rgb(1), [0.0, 0.0, 0.0]);
        assert_eq!(plan.ambient_rgb(2), [0.25, 0.25, 0.25]);
        assert_eq!(plan.ambient_rgb(3), [0.0, 0.0, 0.0]);
        assert_eq!(plan.ambient_rgb(9), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn zero_mask_lights_and_out_of_range_bits_are_reported() {
        let lights = vec![
            Light2d {
                light_mask: 0,
                ..Default::default()
            },
            Light2d {
                light_mask: 0x100,
                ..Default::default()
            },
            Light2d {
                light_mask: 0x101,
                ..Default::default()
            },
        ];        let plan = plan_lighting(
            &lights,
            &[],
            &AmbientLight::default(),
            [-10.0, -10.0, 20.0, 20.0],
            &LightBudget::default(),
        );
        assert_eq!(plan.lights.len(), 1);
        assert_eq!(plan.lights[0].source, 2);
        assert_eq!(plan.lights[0].layers, 1);
        assert_eq!(
            plan.issues,
            [
                LightIssue {
                    source: 0,
                    kind: LightIssueKind::ZeroMask
                },
                LightIssue {
                    source: 1,
                    kind: LightIssueKind::MaskBeyondLayerRange { dropped: 0x100 }
                },
                LightIssue {
                    source: 1,
                    kind: LightIssueKind::ZeroMask
                },
                LightIssue {
                    source: 2,
                    kind: LightIssueKind::MaskBeyondLayerRange { dropped: 0x100 }
                },
            ]
        );
    }

    #[test]
    fn degenerate_lights_are_dropped_with_typed_issues() {
        let budget = LightBudget::default();
        let world = [-1e4, -1e4, 2e4, 2e4];
        let ambient = AmbientLight::default();
        let cases: Vec<(Light2d, LightIssueKind)> = vec![
            (
                Light2d {
                    radius: 0.0,
                    ..Default::default()
                },
                LightIssueKind::ZeroExtent,
            ),
            (
                Light2d {
                    radius: f32::NAN,
                    ..Default::default()
                },
                LightIssueKind::ZeroExtent,
            ),
            (
                Light2d {
                    position: Vec2::new(f32::INFINITY, 0.0),
                    ..Default::default()
                },
                LightIssueKind::NonFinitePosition,
            ),
            (
                Light2d {
                    color: [f32::NAN, 1.0, 1.0],
                    ..Default::default()
                },
                LightIssueKind::NonFiniteValue,
            ),
            (
                Light2d {
                    energy: f32::INFINITY,
                    ..Default::default()
                },
                LightIssueKind::NonFiniteValue,
            ),
            (
                Light2d {
                    falloff: LightFalloff::Texture {
                        slot: 0,
                        half_extent: Vec2::splat(4.0),
                        offset: Vec2::new(f32::NAN, 0.0),
                    },
                    ..Default::default()
                },
                LightIssueKind::NonFiniteValue,
            ),
            (
                Light2d {
                    falloff: LightFalloff::stops(Vec::new()),
                    ..Default::default()
                },
                LightIssueKind::StopsEmpty,
            ),
            (
                Light2d {
                    falloff: LightFalloff::stops(vec![LightStop::default(); 9]),
                    ..Default::default()
                },
                LightIssueKind::TooManyStops { count: 9, max: 8 },
            ),
            (
                Light2d {
                    falloff: LightFalloff::stops(vec![
                        LightStop {
                            at: 0.0,
                            color: [1.0; 4],
                        },
                        LightStop {
                            at: 0.2,
                            color: [1.0; 4],
                        },
                        LightStop {
                            at: 0.1,
                            color: [1.0; 4],
                        },
                    ]),
                    ..Default::default()
                },
                LightIssueKind::StopsNotAscending {
                    at: 0.1,
                    previous: 0.2,
                },
            ),
            (
                Light2d {
                    falloff: LightFalloff::stops(vec![
                        LightStop {
                            at: 0.0,
                            color: [1.0; 4],
                        },
                        LightStop {
                            at: 1.5,
                            color: [1.0; 4],
                        },
                    ]),
                    ..Default::default()
                },
                LightIssueKind::StopOutOfRange { at: 1.5 },
            ),
        ];
        for (light, expected) in cases {
            let plan = plan_lighting(&[light], &[], &ambient, world, &budget);
            assert!(plan.lights.is_empty(), "{expected:?} must not be shaded");
            assert_eq!(plan.issues.len(), 1, "{expected:?}");
            assert_eq!(plan.issues[0].kind, expected);
        }
    }

    #[test]
    fn non_finite_ambient_falls_back_to_black_and_reports_it() {
        let dim = AmbientLight {
            color: [f32::NAN, 1.0, 1.0],
            energy: 1.0,
            light_mask: 1,
        };
        let plan = plan_lighting(
            &[Light2d::point(Vec2::ZERO, 8.0)],
            &[],
            &dim,
            [-10.0, -10.0, 20.0, 20.0],
            &LightBudget::default(),
        );
        assert_eq!(plan.ambient_rgb(0), [0.0, 0.0, 0.0], "a NaN never clears");
        assert_eq!(
            plan.issues,
            [LightIssue {
                source: AMBIENT_SOURCE,
                kind: LightIssueKind::NonFiniteAmbient
            }]
        );
        assert_eq!(plan.lights.len(), 1, "the light itself is unaffected");
    }

    #[test]
    fn occluder_validation_rejects_degenerate_shapes() {
        assert_eq!(
            Occluder2d::polygon(Vec::new()).validate(),
            Err(OccluderRejectReason::TooFewPoints)
        );
        assert_eq!(
            Occluder2d::polygon(vec![Vec2::ZERO, Vec2::X]).validate(),
            Err(OccluderRejectReason::TooFewPoints)
        );
        assert_eq!(
            Occluder2d::polygon(vec![Vec2::ZERO, Vec2::X, Vec2::new(f32::NAN, 0.0)]).validate(),
            Err(OccluderRejectReason::NonFinitePoint)
        );
        // Collinear corners: no area to cast from.
        assert_eq!(
            Occluder2d::polygon(vec![Vec2::ZERO, Vec2::new(1.0, 0.0), Vec2::new(2.0, 0.0)]).validate(),
            Err(OccluderRejectReason::DegenerateArea)
        );
        assert_eq!(
            Occluder2d::polygon(vec![Vec2::ZERO, Vec2::ZERO, Vec2::ZERO]).validate(),
            Err(OccluderRejectReason::DegenerateArea)
        );
        // A bowtie crosses itself.
        assert_eq!(
            Occluder2d::polygon(vec![
                Vec2::ZERO,
                Vec2::new(2.0, 0.0),
                Vec2::new(0.0, 2.0),
                Vec2::new(2.0, 2.0),
            ])
            .validate(),
            Err(OccluderRejectReason::SelfIntersecting)
        );
        assert_eq!(
            Occluder2d::polygon(vec![Vec2::ZERO; MAX_OCCLUDER_POINTS + 1]).validate(),
            Err(OccluderRejectReason::TooManyPoints)
        );
        // A mask of zero shadows nothing, so it is not a caster.
        assert_eq!(
            Occluder2d::rect_masked(Vec2::ZERO, Vec2::splat(4.0), 0).validate(),
            Err(OccluderRejectReason::ZeroMask)
        );
        // Valid shapes pass, and keep their mask and bounds.
        let valid = ValidatedOccluder::validate(&Occluder2d::rect_masked(
            Vec2::new(-1.0, -2.0),
            Vec2::new(3.0, 4.0),
            0b101,
        ))
        .expect("rect is valid");
        assert_eq!(valid.light_mask(), 0b101);
        assert_eq!(valid.aabb(), [-1.0, -2.0, 3.0, 4.0]);
        assert!(valid.closed());
        let open = ValidatedOccluder::validate(&Occluder2d {
            points: vec![Vec2::ZERO, Vec2::new(10.0, 0.0)],
            closed: false,
            light_mask: 1,
        });
        assert!(open.is_ok(), "an open polyline is a thin wall, not a reject");
        let pinched = Occluder2d::polygon(vec![
            Vec2::ZERO,
            Vec2::new(4.0, 0.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(4.0, 4.0),
            Vec2::new(0.0, 4.0),
            Vec2::new(2.0, 2.0),
        ]);
        assert!(
            pinched.validate().is_ok(),
            "touching at a point is not a self-crossing"
        );
        // The rejected occluders land in the plan's report, once.
        let plan = plan_lighting(
            &[],
            &[
                Occluder2d::polygon(Vec::new()),
                Occluder2d::rect(Vec2::ZERO, Vec2::splat(1.0)),
            ],
            &AmbientLight::default(),
            [-10.0, -10.0, 20.0, 20.0],
            &LightBudget::default(),
        );
        assert_eq!(
            plan.occluder_rejects,
            [OccluderReject {
                index: 0,
                reason: OccluderRejectReason::TooFewPoints
            }]
        );
    }

    #[test]
    fn over_budget_lights_drop_in_input_order() {
        let lights: Vec<Light2d> = (0..5)
            .map(|i| Light2d {
                position: Vec2::new(i as f32 * 4.0, 0.0),
                radius: 8.0,
                ..Default::default()
            })
            .collect();
        let budget = LightBudget {
            max_lights: 2,
            ..Default::default()
        };
        let world = [-100.0, -100.0, 200.0, 200.0];
        let first = plan_lighting(
            &lights,
            &[],
            &AmbientLight::default(),
            world,
            &budget,
        );
        let again = plan_lighting(
            &lights,
            &[],
            &AmbientLight::default(),
            world,
            &budget,
        );
        assert_eq!(first, again, "the plan is deterministic");
        assert_eq!(
            first
                .lights
                .iter()
                .map(|light| light.source)
                .collect::<Vec<_>>(),
            [0, 1]
        );
        assert_eq!(
            first
                .issues
                .iter()
                .map(|issue| (issue.source, issue.kind))
                .collect::<Vec<_>>(),
            [
                (2, LightIssueKind::OverLightBudget { max: 2 }),
                (3, LightIssueKind::OverLightBudget { max: 2 }),
                (4, LightIssueKind::OverLightBudget { max: 2 }),
            ]
        );
        // Culling is counted, not reported as a budget drop.
        let offscreen = LightBudget::default();
        let plan = plan_lighting(
            &[Light2d::point(Vec2::new(10_000.0, 10_000.0), 4.0)],
            &[],
            &AmbientLight::default(),
            [-10.0, -10.0, 20.0, 20.0],
            &offscreen,
        );
        assert!(plan.lights.is_empty() && plan.issues.is_empty() && plan.culled == 1);
    }

    #[test]
    fn light_on_an_occluder_vertex_keeps_finite_wedges() {
        // The light sits exactly on the occluder's corner (10, 5): two edges
        // start there, and neither may produce a NaN direction.
        let light = Light2d {
            position: Vec2::new(10.0, 5.0),
            radius: 10.0,
            shadows: true,
            ..Default::default()
        };
        let occluder = Occluder2d::rect(Vec2::new(10.0, -5.0), Vec2::new(20.0, 5.0));
        let valid = ValidatedOccluder::validate(&occluder).expect("rect is valid");
        let mut wedges = Vec::new();
        let casters = shadow_wedges(
            light.position,
            light.radius,
            ShadowFilter::None,
            light.shadow_softness,
            light.light_mask,
            occluder.aabb(),
            std::slice::from_ref(&valid),
            64,
            &mut wedges,
        );
        assert_eq!((casters.used, casters.dropped), (1, 0));
        assert_eq!(wedges.len(), 4, "one wedge per edge");
        for wedge in &wedges {
            assert!(
                wedge.quad.iter().all(|p| p.is_finite()),
                "no NaN from a light exactly on a vertex: {:?}",
                wedge.quad
            );
        }
        // The two edges that meet at the light keep that corner pinned: the
        // quad degenerates to a triangle instead of extruding a NaN.
        let corner = light.position;
        let touching = wedges
            .iter()
            .filter(|w| w.quad.iter().filter(|p| **p == corner).count() == 2)
            .count();
        assert_eq!(touching, 2, "the two edges at the vertex stay finite");
        // The far corners still extrude, so the shadow behind the wall is
        // covered rather than collapsing onto the light.
        assert!(wedges.iter().any(|w| w.quad[1].x > 20.0 || w.quad[2].x > 20.0));
        // With a soft filter the taps also stay finite and sum to one.
        let mut soft = Vec::new();
        shadow_wedges(
            light.position,
            light.radius,
            ShadowFilter::Pcf4,
            4.0,
            light.light_mask,
            occluder.aabb(),
            std::slice::from_ref(&valid),
            64,
            &mut soft,
        );
        assert_eq!(soft.len(), 16, "four taps per edge");
        for (index, wedge) in soft.iter().enumerate() {
            assert!(wedge.quad.iter().all(|p| p.is_finite()));
            assert!((wedge.weight - 0.25).abs() < 1e-6);
            // Taps cycle inside each edge's run, and every wedge names its own
            // mask channel with its weight in it.
            assert_eq!(wedge.tap, index as u32 % 4);
            let mut want = [0.0; MAX_MASK_TAPS];
            want[wedge.tap as usize] = 0.25;
            assert_eq!(wedge.mask_weights(), want);
        }
    }

    #[test]
    fn wedges_cover_the_shadow_behind_a_wall_and_stop_at_the_light() {
        // Light left of a wall; the shadow is the wall's projection away from
        // the light, and nothing between light and wall is shadowed.
        let light = Light2d {
            position: Vec2::new(-20.0, 0.0),
            radius: 100.0,
            shadows: true,
            ..Default::default()
        };
        let wall = Occluder2d::rect(Vec2::new(0.0, -10.0), Vec2::new(10.0, 10.0));
        let valid = ValidatedOccluder::validate(&wall).expect("rect is valid");
        let mut wedges = Vec::new();
        shadow_wedges(
            light.position,
            light.radius,
            ShadowFilter::None,
            0.0,
            light.light_mask,
            wall.aabb(),
            std::slice::from_ref(&valid),
            64,
            &mut wedges,
        );
        let covered = |p: Vec2| {
            wedges
                .iter()
                .any(|w| point_in_quad(p, w.quad[0], w.quad[1], w.quad[2], w.quad[3]))
        };
        assert!(covered(Vec2::new(20.0, 0.0)), "behind the wall");
        assert!(covered(Vec2::new(5.0, 0.0)), "inside the wall is solid");
        assert!(!covered(Vec2::new(-1.0, 0.0)), "in front of the wall stays lit");
        // Culling by mask: a wall on another layer casts nothing here.
        let mut masked = Vec::new();
        let other = ValidatedOccluder::validate(&Occluder2d::rect_masked(
            Vec2::new(0.0, -10.0),
            Vec2::new(10.0, 10.0),
            0b010,
        ))
        .expect("rect is valid");
        let casters = shadow_wedges(
            light.position,
            light.radius,
            ShadowFilter::None,
            0.0,
            light.light_mask,
            wall.aabb(),
            std::slice::from_ref(&other),
            64,
            &mut masked,
        );
        assert_eq!((casters.used, masked.len()), (0, 0));
        // The per-light caster budget is applied in slice order.
        let mut drops = Vec::new();
        let many: Vec<ValidatedOccluder> = (0..4)
            .map(|i| {
                ValidatedOccluder::validate(&Occluder2d::rect(
                    Vec2::new(i as f32 - 1.0, -1.0),
                    Vec2::new(i as f32, 1.0),
                ))
                .expect("valid")
            })
            .collect();
        let casters = shadow_wedges(
            light.position,
            light.radius,
            ShadowFilter::None,
            0.0,
            light.light_mask,
            [-4.0, -4.0, 8.0, 8.0],
            &many,
            2,
            &mut drops,
        );
        assert_eq!((casters.used, casters.dropped), (2, 2));
        assert_eq!(drops.len(), 2 * 4);
    }

    #[test]
    fn shadow_vertex_budget_unshadows_loudly() {
        let mut lights = Vec::new();
        for i in 0..2 {
            let mut light = Light2d::point(Vec2::new(i as f32 * 4.0, 0.0), 20.0);
            light.shadows = true;
            lights.push(light);
        }
        let budget = LightBudget {
            max_shadow_vertices: 32,
            ..Default::default()
        };
        let plan = plan_lighting(
            &lights,
            &[Occluder2d::rect(Vec2::new(1.0, -2.0), Vec2::new(2.0, 2.0))],
            &AmbientLight::default(),
            [-100.0, -100.0, 200.0, 200.0],
            &budget,
        );
        assert_eq!(plan.lights.len(), 2);
        // The first light's 24 vertices fit; the second's do not.
        assert!(plan.lights[0].shadowed);
        assert!(!plan.lights[1].shadowed);
        assert_eq!(
            plan.issues,
            [LightIssue {
                source: 1,
                kind: LightIssueKind::ShadowVertexBudget {
                    needed: 24,
                    max: 32
                }
            }]
        );
    }

    /// Both boxes of `aabb_overlap` are corners, not corner-plus-size. Reading
    /// the second one as a size makes a box whose min is large positive overlap
    /// one it is nowhere near, and skips one whose min is negative.
    #[test]
    fn aabb_overlap_reads_min_max_corners() {
        assert!(!aabb_overlap([70.0, 70.0, 80.0, 80.0], [50.0, 50.0, 60.0, 60.0]));
        assert!(aabb_overlap([0.0, 0.0, 10.0, 10.0], [-100.0, -100.0, 50.0, 50.0]));
        assert!(aabb_overlap([0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]));
        assert!(
            aabb_overlap([0.0, 0.0, 10.0, 10.0], [10.0, 10.0, 20.0, 20.0]),
            "touching corners count as overlap"
        );
        assert!(!aabb_overlap([0.0, 0.0, 10.0, 10.0], [11.0, 0.0, 20.0, 10.0]));
        assert!(!aabb_overlap([0.0, 0.0, 10.0, 10.0], [0.0, 11.0, 10.0, 20.0]));
    }

    /// The scene that caught the bounds convention: a light at the origin with
    /// its caster to the `+x` side (where `2 * light.position` is zero, so a
    /// size read skipped every caster) and a light inside a panned view (whose
    /// min is negative, so a size read of the view culled it).
    #[test]
    fn casters_and_lights_are_bounds_tested_against_corners() {
        let light = Light2d {
            position: Vec2::ZERO,
            radius: 24.0,
            shadows: true,
            shadow_filter: ShadowFilter::None,
            ..Default::default()
        };
        let occluder = Occluder2d::rect(Vec2::new(8.0, -4.0), Vec2::new(16.0, 4.0));
        let plan = plan_lighting(
            std::slice::from_ref(&light),
            std::slice::from_ref(&occluder),
            &AmbientLight::default(),
            [-64.0, -64.0, 128.0, 128.0],
            &LightBudget::default(),
        );
        assert_eq!(plan.culled, 0);
        assert_eq!(plan.lights.len(), 1);
        let planned = &plan.lights[0];
        assert!(planned.shadowed, "the caster past the light was skipped");
        assert_eq!(planned.wedges.len(), 4, "one wedge per rect edge, not none");
        // The point behind the caster is in the shadow, so the wedges that
        // survived are the real ones.
        let behind = Vec2::new(30.0, 0.0);
        assert!(
            planned.wedges.iter().any(|w| point_in_quad(
                behind,
                w.quad[0],
                w.quad[1],
                w.quad[2],
                w.quad[3]
            )),
            "no wedge covers the point behind the caster"
        );

        // The same convention culls lights against the view: this one sits
        // inside a camera panned to negative coordinates.
        let panned = plan_lighting(
            &[Light2d::point(Vec2::new(-350.0, -275.0), 10.0)],
            &[],
            &AmbientLight::default(),
            [-400.0, -300.0, 100.0, 50.0],
            &LightBudget::default(),
        );
        assert_eq!(panned.culled, 0, "a light inside the view was culled");
        assert_eq!(panned.lights.len(), 1);
    }

    #[test]
    fn visible_world_rect_inverts_a_y_down_camera() {
        let camera = crate::screen_camera([800.0, 600.0]);
        let rect = visible_world_rect(camera).expect("an ortho camera has a rect");
        assert!((rect[0]).abs() < 1e-3 && (rect[1]).abs() < 1e-3);
        assert!((rect[2] - 800.0).abs() < 1e-3 && (rect[3] - 600.0).abs() < 1e-3);
        assert_eq!(visible_world_rect(Mat4::ZERO), None);
        assert_eq!(visible_world_rect(Mat4::from_scale(Vec3::ZERO)), None);
    }

    #[test]
    fn occluder_contains_matches_solid_polygons() {
        let wall = Occluder2d::rect(Vec2::new(2.0, -2.0), Vec2::new(5.0, 2.0));
        assert!(wall.contains(Vec2::new(3.0, 0.0)));
        assert!(!wall.contains(Vec2::new(1.0, 0.0)));
        assert!(!Occluder2d::polygon(vec![Vec2::ZERO, Vec2::X]).contains(Vec2::ZERO));
        // A cell is one cell wide and aligned to the grid.
        let cell = Occluder2d::cell([-1, 2], 8.0);
        assert_eq!(cell.aabb(), [-8.0, 16.0, 0.0, 24.0]);
    }

    /// Point-in-convex-quad for the wedge coverage assertions above.
    fn point_in_quad(p: Vec2, a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> bool {
        let side = |p: Vec2, q: Vec2, r: Vec2| cross(q - p, r - p);
        let s1 = side(a, b, p) >= 0.0;
        let s2 = side(b, c, p) >= 0.0;
        let s3 = side(c, d, p) >= 0.0;
        let s4 = side(d, a, p) >= 0.0;
        (s1 && s2 && s3 && s4) || (!s1 && !s2 && !s3 && !s4)
    }
}
