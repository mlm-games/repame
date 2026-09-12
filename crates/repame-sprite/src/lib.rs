//! 2D sprite viewport: snapshot in, pixels out.
//!
//! The UI builds a [`FrameInput`] per frame (plain data, cheap to rebuild
//! during composition) and mounts [`Viewport2d`] as a Repose view, which
//! draws the snapshot through a dp-space contain-fit and reports pointer
//! picks back in world coords. Unit discipline lives here, once:
//! games work purely in world/dp units and never touch physical px.
//! Camera state lives in Repose signals, never in the renderer.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use glam::{Mat4, Vec2, Vec3};
use repose_canvas::{Canvas, DrawScope, Embedded};
use repose_core::locals::effective_density_scale;
use repose_core::{Color, Dp, Modifier, Px, Rect, View};
use repose_render_wgpu::{Callback, CallbackResources, ScreenDescriptor, WgpuCallback};
use repose_ui::Box as UiBox;
use repose_ui::ViewExt;

pub mod batch;
pub use batch::{
    AtlasUpload, BatchDesc, SpriteBatch, TextureFilter, draw_batch, draw_batch_with_id, frame_uv,
    instance_rows, screen_camera, sprite_aabb,
};

pub mod fullscreen;
pub use fullscreen::{FullscreenDesc, FullscreenPass, FullscreenTexture};

pub mod post;

/// 2D orthographic camera. Owned by the UI (signals), copied into the
/// snapshot per frame.
///
/// The camera's look point is [`effective_center`](Camera2d::effective_center)
/// (`center + offset`); `zoom` and `units_per_pixel` scale the view
/// uniformly. All three viewport consumers (canvas drawing, the GPU batch
/// matrix, and pointer picks) derive from these fields through one shared
/// transform, so they can never disagree.
#[derive(Clone, Copy, Debug)]
pub struct Camera2d {
    /// World-space point the camera follows (the follow target, room
    /// center, or player position). Defaults to `(0, 0)`.
    ///
    /// Scroll limits (see [`apply_limits`]) clamp this field; put
    /// momentary displacement such as trauma shake in [`offset`](Camera2d::offset)
    /// instead so it can push past the limits.
    pub center: Vec2,
    /// Momentary displacement added to [`center`](Camera2d::center).
    /// Defaults to `(0, 0)`.
    ///
    /// Useful for looking around or camera shake animations: applied after
    /// limits, so a shake impulse still moves the view even when the follow
    /// target is pinned at a scroll edge. Decays back to zero under
    /// game-side trauma handling.
    pub offset: Vec2,
    /// World units per screen pixel at zoom 1. Defaults to `1.0`.
    ///
    /// Combined with the viewport size to build the orthographic
    /// projection: a smaller value shows less of the world (larger
    /// sprites). Together with `zoom`, the visible world width is
    /// `viewport_dp * units_per_pixel / zoom`. Non-positive values fall
    /// back to `1.0` so a bad snapshot can never divide by zero.
    pub units_per_pixel: f32,
    /// Magnification. Defaults to `1.0`.
    ///
    /// Higher values zoom in: `2.0` shows half the world width on each
    /// axis (a quarter of the area); `0.5` shows twice as much. Applied on
    /// canvas and GPU alike. Non-positive values fall back to `1.0`.
    pub zoom: f32,
    /// Roll in radians around the look point (trauma shake). Defaults to `0.0`.
    ///
    /// Applied after look/zoom on canvas, GPU, picks, and actor surfaces,
    /// so all consumers stay glued. Positive is clockwise in y-down space.
    pub roll: f32,
}

impl Default for Camera2d {
    fn default() -> Self {
        Self {
            center: Vec2::ZERO,
            offset: Vec2::ZERO,
            units_per_pixel: 1.0,
            zoom: 1.0,
            roll: 0.0,
        }
    }
}

/// Scroll limits in world units. Disabled by default.
///
/// The clamp applies to [`center`](Camera2d::center) only: pass the follow
/// target through [`apply_limits`] before writing it into the camera, and
/// keep momentary displacement in [`offset`](Camera2d::offset) so shake
/// bypasses the clamp by design.
#[derive(Clone, Copy, Debug)]
pub struct CameraLimits {
    /// Smallest visible center `x`. Defaults to `0.0`.
    pub left: f32,
    /// Smallest visible center `y`. Defaults to `0.0`.
    pub top: f32,
    /// Largest visible center `x`. Defaults to `0.0`.
    pub right: f32,
    /// Largest visible center `y`. Defaults to `0.0`.
    pub bottom: f32,
    /// Master switch. Defaults to `false` (no clamping).
    pub enabled: bool,
}

impl Default for CameraLimits {
    fn default() -> Self {
        Self {
            left: 0.0,
            top: 0.0,
            right: 0.0,
            bottom: 0.0,
            enabled: false,
        }
    }
}

/// Clamp a follow target into scroll limits.
///
/// Returns `center` unchanged when limits are disabled or when a pair is
/// inverted (`left > right` is normalized, never a trap). Each axis clamps
/// independently, so a corner target slides along the clamped edge.
///
/// ```rust
/// use repame_sprite::{CameraLimits, apply_limits};
///
/// let limits = CameraLimits { left: 100.0, top: 100.0, right: 700.0, bottom: 500.0, enabled: true };
/// assert_eq!(apply_limits([50.0, 300.0], limits), [100.0, 300.0]);
/// assert_eq!(apply_limits([400.0, 300.0], limits), [400.0, 300.0]);
/// ```
pub fn apply_limits(center: [f32; 2], limits: CameraLimits) -> [f32; 2] {
    if !limits.enabled {
        return center;
    }
    [
        center[0].clamp(limits.left.min(limits.right), limits.left.max(limits.right)),
        center[1].clamp(limits.top.min(limits.bottom), limits.top.max(limits.bottom)),
    ]
}

/// Ease the camera toward its follow target.
///
/// Moves `current` toward `target` with exponential smoothing at `speed`
/// world units per second: fast when far away, settling gently without
/// overshooting. Large `speed` values approach a snap; the motion is
/// frame-rate independent for a fixed `dt`.
///
/// - `speed <= 0` (or non-finite) snaps directly to `target`.
/// - `dt <= 0` holds `current` (a paused frame never moves the camera).
///
/// ```rust
/// use repame_sprite::smooth_toward;
///
/// let p = smooth_toward([0.0, 0.0], [100.0, 0.0], 5.0, 0.016);
/// assert!(p[0] > 0.0 && p[0] < 100.0);
/// ```
pub fn smooth_toward(current: [f32; 2], target: [f32; 2], speed: f32, dt: f32) -> [f32; 2] {
    if dt <= 0.0 {
        return current;
    }
    if speed <= 0.0 || !speed.is_finite() {
        return target;
    }
    let t = 1.0 - (-speed * dt).exp();
    [
        current[0] + (target[0] - current[0]) * t,
        current[1] + (target[1] - current[1]) * t,
    ]
}

impl Camera2d {
    /// Effective look point: `center + offset`.
    ///
    /// All framing (canvas drawing, the GPU batch matrix, pointer picks,
    /// actor surfaces) uses this value, so shake and look-around applied
    /// through `offset` move every consumer together.
    pub fn effective_center(&self) -> [f32; 2] {
        [self.center.x + self.offset.x, self.center.y + self.offset.y]
    }

    /// Framing matrix used by GPU + picks. Prefer this over raw ortho.
    /// Includes [`roll`](Camera2d::roll) around the look point.
    pub fn fit_matrix(&self, canvas_dp: [f32; 2], world_size: [f32; 2]) -> Mat4 {
        let fit = effective_fit(canvas_dp, world_size, self);
        fit_view_proj_with_roll(
            canvas_dp,
            world_size,
            self.effective_center(),
            fit,
            self.roll,
        )
    }

    /// World under a dp-space cursor (canvas coords, density already divided out).
    /// Includes [`roll`](Camera2d::roll).
    pub fn dp_to_world_pt(&self, canvas_dp: [f32; 2], world_size: [f32; 2], dp: [f32; 2]) -> Vec2 {
        let fit = effective_fit(canvas_dp, world_size, self);
        let w = dp_to_world_with_roll(dp, world_size, self.effective_center(), fit, self.roll);
        Vec2::new(w[0], w[1])
    }

    /// Raw ortho ignoring contain-fit letterbox. Kept for HUD overlays in
    /// screen space; viewport code must use [`fit_matrix`](Camera2d::fit_matrix).
    pub fn view_proj_screen(&self, viewport_px: [f32; 2]) -> Mat4 {
        let vp_w = viewport_px[0].max(1.0);
        let vp_h = viewport_px[1].max(1.0);
        let zoom = sane_positive(self.zoom);
        let upp = sane_positive(self.units_per_pixel);
        let w = vp_w * upp / zoom;
        let h = vp_h * upp / zoom;
        let proj = glam::camera::rh::proj::directx::orthographic(
            -w / 2.0,
            w / 2.0,
            h / 2.0,
            -h / 2.0,
            -1000.0,
            1000.0,
        );
        let view = Mat4::from_translation(Vec3::new(
            -(self.center.x + self.offset.x),
            -(self.center.y + self.offset.y),
            0.0,
        ));
        proj * view
    }

    /// Legacy y-down view-projection: world +y points screen-down.
    ///
    /// Diverges from the shared framing contract: this frames from
    /// `viewport * units_per_pixel / zoom` and ignores
    /// [`FrameInput::world_size`], so it disagrees with canvas, GPU, and
    /// picks whenever a contain-fit letterboxes. Kept for guards/tests;
    /// viewport code must use [`fit_view_proj`] instead.
    #[deprecated(
        since = "0.1.8",
        note = "ignores world_size; use fit_view_proj for the shared canvas/GPU/pick framing contract"
    )]
    pub fn view_proj(&self, viewport_px: [f32; 2]) -> Mat4 {
        self.view_proj_screen(viewport_px)
    }

    /// Legacy world-space position under a viewport-pixel cursor.
    ///
    /// Inverts [`view_proj`](Camera2d::view_proj), so it shares that
    /// function's divergence from the contain-fit contract. Pointer picks
    /// must go through [`pick_world`] over the painted [`FrameGeom`].
    #[deprecated(
        since = "0.1.8",
        note = "ignores world_size; use pick_world over the painted FrameGeom instead"
    )]
    #[allow(deprecated)]
    pub fn screen_to_world(&self, viewport_px: [f32; 2], px: [f32; 2]) -> Vec2 {
        if viewport_px[0] <= 0.0 || viewport_px[1] <= 0.0 {
            let [cx, cy] = self.effective_center();
            return Vec2::new(cx, cy);
        }
        let ndc = Vec2::new(
            (px[0] / viewport_px[0]) * 2.0 - 1.0,
            1.0 - (px[1] / viewport_px[1]) * 2.0,
        );
        let inv = self.view_proj(viewport_px).inverse();
        let world = inv.project_point3(ndc.extend(0.0));
        Vec2::new(world.x, world.y)
    }
}

/// Blend mode per sprite. Alpha is the default; Additive draws with
/// `SrcAlpha + One` (GML `bm_add` parity, second pipeline, split draw
/// ranges) so the tint alpha scales the glow.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SpriteBlend {
    #[default]
    Alpha,
    Additive,
    Multiply,
}

/// One batched sprite. `uv` is in atlas texels normalized to 0..1.
#[derive(Clone, Copy, Debug)]
pub struct SpriteInstance {
    /// World-space center, rotation radians, size in world units.
    pub center: Vec2,
    pub rotation: f32,
    pub size: Vec2,
    /// Normalized anchor (origin): `[0.5, 0.5]` centers the quad on
    /// `center` (legacy default), `[0, 0]` pins the top-left corner -
    /// bevy `Anchor` semantics, y-down.
    pub anchor: Vec2,
    /// Mirror around the anchor axes (left-walking hordes, etc.).
    /// Solid canvas fills are flip-invariant; the GPU batch mirrors
    /// geometry (UVs untouched).
    pub flip_x: bool,
    pub flip_y: bool,
    pub uv_min: Vec2,
    pub uv_max: Vec2,
    /// RGBA tint, same byte semantics as the legacy canvas path
    /// (`(c * 255) as u8` per channel).
    pub color: [f32; 4],
    /// Atlas page index for multi-texture batches.
    pub page: u32,
    /// Stable draw key; higher draws on top. Default 0. The batch
    /// stable-sorts by `z` before upload (Bevy `z` semantics).
    pub z: f32,
    pub blend: SpriteBlend,
}

impl Default for SpriteInstance {
    fn default() -> Self {
        Self {
            center: Vec2::ZERO,
            rotation: 0.0,
            size: Vec2::ONE,
            anchor: Vec2::new(0.5, 0.5),
            flip_x: false,
            flip_y: false,
            uv_min: Vec2::ZERO,
            uv_max: Vec2::ONE,
            color: [1.0, 1.0, 1.0, 1.0],
            page: 0,
            z: 0.0,
            blend: SpriteBlend::Alpha,
        }
    }
}

/// World-anchored text floater (damage numbers, `+25`).
#[derive(Clone, Debug, Default)]
pub struct WorldText {
    pub text: String,
    /// World-space anchor, same convention as the legacy canvas text path.
    pub pos: Vec2,
    /// RGBA, same byte semantics as [`SpriteInstance::color`].
    pub color: [f32; 4],
    /// Font size in world units.
    pub size: f32,
}

/// Everything the viewport draws this frame. Plain data, snapshot per frame.
///
/// Framing contract (single source of truth): canvas, GPU, and picks all
/// derive from `world_size` + `cam` through [`effective_fit`] /
/// [`fit_view_proj`] / [`world_to_dp`]. `cam.effective_center()` is the
/// look point (rest + `offset` shake included); `cam.zoom` /
/// `cam.units_per_pixel` scale the fit uniformly on every backend.
#[derive(Clone, Default, Debug)]
pub struct FrameInput {
    pub cam: Camera2d,
    /// World-space extent the contain-fit maps into the viewport.
    pub world_size: [f32; 2],
    /// Cold-start viewport size in **dp** (physical px / density), used for
    /// the GPU camera until the first painted [`FrameGeom`] arrives.
    /// Named for its units: divide physical px by density before storing
    /// here, or the first frame's fit is off by the density factor on
    /// HiDPI screens. (Contrast [`FrameGeom::viewport_px`], which is
    /// genuinely physical px.)
    pub viewport_dp: [f32; 2],
    pub sprites: Vec<SpriteInstance>,
    /// World-anchored text, drawn after the sprite pass on the canvas
    /// viewport. The GPU viewport ignores this field: compose texts as
    /// sibling views (same pattern as `overlay_color` consumers that need
    /// typography) so GPU and canvas agree by construction.
    pub texts: Vec<WorldText>,
    /// Full-viewport fill under everything (letterbox included).
    pub background: Option<[f32; 4]>,
    /// Optional fullscreen tint/color-grading hook (e.g. FOW dimming,
    /// damage flash). Applied after the sprite pass on canvas *and* GPU
    /// (after the chroma composite when `chroma > 0.0`), with the same
    /// byte semantics as the canvas path (`(c * 255) as u8` per channel,
    /// alpha-blended over the scene).
    pub overlay_color: Option<[f32; 4]>,
    /// Chromatic aberration amount (bevy `chromatic_intensity` units;
    /// NT pulses land at 0.04..0.7). GPU viewports render the scene
    /// offscreen and composite it back with an RGB split; `0.0` keeps
    /// the zero-cost direct path. Canvas viewports ignore it (sampling
    /// FX need pixels, and the canvas path is vector commands).
    pub chroma: f32,
}

/// UI-facing pointer events from the viewport.
///
/// `Click` fires on pointer **up** when the press started inside and the
/// pointer moved less than the slop threshold (drag-off cancels, real UI
/// button feel). `Press` is the down-edge for games that need it.
#[derive(Clone, Debug)]
pub enum PickEvent {
    Press {
        world: Vec2,
        screen: [f32; 2],
    },
    Click {
        world: Vec2,
        screen: [f32; 2],
    },
    Hover {
        world: Vec2,
    },
    /// Touch/pen contact began: pointer id + window-physical px
    /// (y-down, `origin + position`). Mouse never emits these; taps
    /// still emit `Click` too (button/UI parity).
    TouchDown {
        id: u64,
        screen: [f32; 2],
    },
    /// Touch/pen contact moved (same coordinate space as `TouchDown`).
    TouchMove {
        id: u64,
        screen: [f32; 2],
    },
    /// Touch/pen contact ended (up / leave).
    TouchUp {
        id: u64,
    },
}

/// Touch/pen pointers drive game touch zones; mouse stays on the
/// click/hover path.
fn is_touch(ev: &repose_core::input::PointerEvent) -> bool {
    matches!(
        ev.kind,
        repose_core::input::PointerKind::Touch | repose_core::input::PointerKind::Pen
    )
}

/// Window-physical px (y-down) for touch-zone sampling.
fn screen_of(ev: &repose_core::input::PointerEvent) -> [f32; 2] {
    let p = ev.position_in_window();
    [p.x, p.y]
}

/// Clamp helper: finite positive values pass through, everything else
/// (zero, negative, NaN, infinite) falls back to 1.0 so framing math can
/// never div-by-zero.
fn sane_positive(v: f32) -> f32 {
    if v.is_finite() && v > 1e-6 { v } else { 1.0 }
}

/// Contain-fit of a `world` extent into a dp-space canvas: uniform scale
/// plus centering offsets, all dp. Pure, so games and tests can pin it.
/// No silent clamp: `world_to_dp` and `dp_to_world` are exact inverses
/// for any positive scale, so picks never drift from drawn content.
pub fn contain_fit(canvas_dp: [f32; 2], world: [f32; 2]) -> (f32, f32, f32) {
    if canvas_dp[0] <= 0.0 || canvas_dp[1] <= 0.0 || world[0] <= 0.0 || world[1] <= 0.0 {
        return (1.0, 0.0, 0.0);
    }
    let s = (canvas_dp[0] / world[0]).min(canvas_dp[1] / world[1]);
    if !s.is_finite() || s <= 1e-6 {
        return (1.0, 0.0, 0.0);
    }
    (
        s,
        (canvas_dp[0] - world[0] * s) * 0.5,
        (canvas_dp[1] - world[1] * s) * 0.5,
    )
}

/// Effective fit for one frame: base contain-fit scaled by
/// `zoom / units_per_pixel`, re-centered. Default camera (`zoom = 1`,
/// `units_per_pixel = 1`) is exactly [`contain_fit`]; zoom 2 doubles
/// on-screen size on canvas and GPU alike.
pub fn effective_fit(canvas_dp: [f32; 2], world: [f32; 2], cam: &Camera2d) -> (f32, f32, f32) {
    let (base, _, _) = contain_fit(canvas_dp, world);
    let zoom = sane_positive(cam.zoom);
    let upp = sane_positive(cam.units_per_pixel);
    let s = base * zoom / upp;
    if !s.is_finite() || s <= 1e-6 {
        return (1.0, 0.0, 0.0);
    }
    (
        s,
        (canvas_dp[0] - world[0] * s) * 0.5,
        (canvas_dp[1] - world[1] * s) * 0.5,
    )
}

/// GPU camera for the shared framing contract: the exact matrix form of
/// [`world_to_dp`] over `canvas_dp` (dp = physical px / density).
/// Canvas draws via `world_to_dp`, GPU draws via this matrix, picks invert
/// via `dp_to_world`. One transform, three consumers.
pub fn fit_view_proj(
    canvas_dp: [f32; 2],
    world_size: [f32; 2],
    cam_center: [f32; 2],
    fit: (f32, f32, f32),
) -> Mat4 {
    fit_view_proj_with_roll(canvas_dp, world_size, cam_center, fit, 0.0)
}

/// [`fit_view_proj`] plus camera roll around the look point.
/// Roll rotates world points around `cam_center` before the contain-fit
/// mapping; `0.0` is exactly [`fit_view_proj`].
pub fn fit_view_proj_with_roll(
    canvas_dp: [f32; 2],
    world_size: [f32; 2],
    cam_center: [f32; 2],
    fit: (f32, f32, f32),
    roll: f32,
) -> Mat4 {
    let w = canvas_dp[0].max(1.0);
    let h = canvas_dp[1].max(1.0);
    let (s, ox, oy) = fit;
    let s = if s.is_finite() && s > 1e-6 { s } else { 1.0 };
    let lx = cam_center[0] - world_size[0] * 0.5;
    let ly = cam_center[1] - world_size[1] * 0.5;
    let sx = 2.0 * s / w;
    let sy = -2.0 * s / h;
    let tx = 2.0 * (ox - lx * s) / w - 1.0;
    let ty = 1.0 - 2.0 * (oy - ly * s) / h;
    let base = Mat4::from_cols(
        glam::Vec4::new(sx, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, sy, 0.0, 0.0),
        glam::Vec4::new(0.0, 0.0, 0.001, 0.0),
        glam::Vec4::new(tx, ty, 0.0, 1.0),
    );
    if !roll.is_finite() || roll.abs() < 1e-7 {
        return base;
    }
    let (c, r) = (roll.cos(), roll.sin());
    let rot = Mat4::from_cols(
        glam::Vec4::new(c, r, 0.0, 0.0),
        glam::Vec4::new(-r, c, 0.0, 0.0),
        glam::Vec4::new(0.0, 0.0, 1.0, 0.0),
        glam::Vec4::new(
            cam_center[0] - c * cam_center[0] + r * cam_center[1],
            cam_center[1] - r * cam_center[0] - c * cam_center[1],
            0.0,
            1.0,
        ),
    );
    base * rot
}

/// World point -> dp canvas point through the fit. `cam_center` shifts the
/// look point (trauma shake included): at the default center
/// (`world / 2`) this is exactly `offset + world * scale`.
pub fn world_to_dp(
    world: [f32; 2],
    world_size: [f32; 2],
    cam_center: [f32; 2],
    fit: (f32, f32, f32),
) -> [f32; 2] {
    world_to_dp_with_roll(world, world_size, cam_center, fit, 0.0)
}

/// [`world_to_dp`] plus camera roll around `cam_center`.
pub fn world_to_dp_with_roll(
    world: [f32; 2],
    world_size: [f32; 2],
    cam_center: [f32; 2],
    fit: (f32, f32, f32),
    roll: f32,
) -> [f32; 2] {
    let (s, ox, oy) = fit;
    let [wx, wy] = rotate_about(world, cam_center, roll);
    [
        ox + (wx - (cam_center[0] - world_size[0] * 0.5)) * s,
        oy + (wy - (cam_center[1] - world_size[1] * 0.5)) * s,
    ]
}

/// Inverse of [`world_to_dp`]: dp canvas point -> world point.
/// Guards degenerate scales (returns the look point instead of Inf).
pub fn dp_to_world(
    dp: [f32; 2],
    world_size: [f32; 2],
    cam_center: [f32; 2],
    fit: (f32, f32, f32),
) -> [f32; 2] {
    dp_to_world_with_roll(dp, world_size, cam_center, fit, 0.0)
}

/// Inverse of [`world_to_dp_with_roll`].
pub fn dp_to_world_with_roll(
    dp: [f32; 2],
    world_size: [f32; 2],
    cam_center: [f32; 2],
    fit: (f32, f32, f32),
    roll: f32,
) -> [f32; 2] {
    let (s, ox, oy) = fit;
    if !s.is_finite() || s.abs() < 1e-6 {
        return cam_center;
    }
    let base = [
        (dp[0] - ox) / s + (cam_center[0] - world_size[0] * 0.5),
        (dp[1] - oy) / s + (cam_center[1] - world_size[1] * 0.5),
    ];
    rotate_about(base, cam_center, -roll)
}

fn rotate_about(p: [f32; 2], center: [f32; 2], roll: f32) -> [f32; 2] {
    if !roll.is_finite() || roll.abs() < 1e-7 {
        return p;
    }
    let (c, r) = (roll.cos(), roll.sin());
    let dx = p[0] - center[0];
    let dy = p[1] - center[1];
    [center[0] + c * dx - r * dy, center[1] + r * dx + c * dy]
}

/// Viewport-local physical-px press -> world point through the painted
/// [`FrameGeom`]. Takes the region-local position (`PointerEvent.position`,
/// already relative to the viewport origin), so HUD chrome, docking, or a
/// non-fullscreen viewport never shift picks.
pub fn pick_world(local_px: [f32; 2], geom: FrameGeom, world_size: [f32; 2]) -> [f32; 2] {
    let d = if geom.density.is_finite() && geom.density > 1e-6 {
        geom.density
    } else {
        1.0
    };
    dp_to_world_with_roll(
        [local_px[0] / d, local_px[1] / d],
        world_size,
        geom.pivot,
        geom.fit,
        geom.roll,
    )
}

/// One painted frame's geometry, shared between [`Viewport2d`] /
/// [`Viewport2dGpu`] (writers) and world-anchored siblings like
/// [`ActorFrame`] (readers). Everything needed to map either direction,
/// so board sprites, picks, and actor surfaces stay glued - including
/// under camera shake.
///
/// Timing: the GPU viewport publishes size/fit/look at layout (ahead of
/// `prepare`, so resizes apply the same frame) and re-publishes at paint;
/// canvas publishes at draw. Composition-time readers ([`ActorFrame`]
/// offsets) still see the previous frame's geometry — math is exact,
/// placement trails the board by one frame under camera motion.
/// Event-time readers (picks) always see the latest publish.
#[derive(Clone, Copy, Debug)]
pub struct FrameGeom {
    /// Effective dp fit of the world extent: `(scale, off_x, off_y)` from
    /// [`effective_fit`] (contain-fit scaled by zoom/units_per_pixel).
    pub fit: (f32, f32, f32),
    /// Look-point shift in world units: `cam.effective_center() - world / 2`
    /// (rest + offset shake included; zero when the camera is default).
    pub look: [f32; 2],
    /// Physical px per dp at paint time.
    pub density: f32,
    /// Painted viewport size in physical px (drives the GPU camera).
    pub viewport_px: [f32; 2],
    /// Camera roll in radians around the look point.
    pub roll: f32,
    /// Camera look point in world coords (`cam.effective_center()` at
    /// publish time). Roll pivot for [`surface_dp`] and picks.
    pub pivot: [f32; 2],
}

impl Default for FrameGeom {
    fn default() -> Self {
        Self {
            fit: (1.0, 0.0, 0.0),
            look: [0.0, 0.0],
            density: 1.0,
            viewport_px: [1.0, 1.0],
            roll: 0.0,
            pivot: [0.0, 0.0],
        }
    }
}

/// Shared paint-time geometry. Canvas views use this on the UI thread;
/// GPU callbacks need `Send + Sync`, so the inner cell is mutex-backed.
#[derive(Clone, Default)]
pub struct GeomHandle(Arc<Mutex<FrameGeom>>);

impl GeomHandle {
    pub fn new() -> Self {
        Self(Arc::new(Mutex::new(FrameGeom::default())))
    }

    pub fn get(&self) -> FrameGeom {
        self.0.lock().map(|g| *g).unwrap_or_default()
    }

    pub fn set(&self, g: FrameGeom) {
        if let Ok(mut slot) = self.0.lock() {
            *slot = g;
        }
    }

    pub fn arc(&self) -> Arc<Mutex<FrameGeom>> {
        self.0.clone()
    }
}

impl From<FrameGeom> for GeomHandle {
    fn from(g: FrameGeom) -> Self {
        Self(Arc::new(Mutex::new(g)))
    }
}

impl From<std::sync::Arc<std::sync::Mutex<FrameGeom>>> for GeomHandle {
    /// Migrate GPU call sites that still hold the raw mutex handle.
    fn from(arc: std::sync::Arc<std::sync::Mutex<FrameGeom>>) -> Self {
        Self(arc)
    }
}

impl From<std::rc::Rc<std::cell::Cell<FrameGeom>>> for GeomHandle {
    /// Migrate canvas call sites that still hold the old cell handle:
    /// snapshots the current value into the shared handle.
    fn from(rc: std::rc::Rc<std::cell::Cell<FrameGeom>>) -> Self {
        Self(Arc::new(Mutex::new(rc.get())))
    }
}

/// Press slop in physical px: pointer-up farther than this from
/// pointer-down cancels the `Click` (drag-off-cancel).
pub const CLICK_SLOP_PX: f32 = 12.0;

fn click_within_slop(a: [f32; 2], b: [f32; 2]) -> bool {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy <= CLICK_SLOP_PX * CLICK_SLOP_PX
}

/// World-anchored surface rect in dp: `([off_x, off_y], [w, h])` for a
/// `size` box centered on `center`. Pure; `ActorFrame` is its view form.
/// The center rotates with [`FrameGeom::roll`] about [`FrameGeom::pivot`];
/// size stays axis-aligned.
pub fn surface_dp(center: [f32; 2], size: [f32; 2], geom: FrameGeom) -> ([f32; 2], [f32; 2]) {
    let (s, ox, oy) = geom.fit;
    let r = rotate_about(center, geom.pivot, geom.roll);
    (
        [
            ox + (r[0] - size[0] * 0.5 - geom.look[0]) * s,
            oy + (r[1] - size[1] * 0.5 - geom.look[1]) * s,
        ],
        [size[0] * s, size[1] * s],
    )
}

fn rgba8(c: [f32; 4]) -> Color {
    Color::from_rgba(
        (c[0].clamp(0.0, 1.0) * 255.0) as u8,
        (c[1].clamp(0.0, 1.0) * 255.0) as u8,
        (c[2].clamp(0.0, 1.0) * 255.0) as u8,
        (c[3].clamp(0.0, 1.0) * 255.0) as u8,
    )
}

/// 2D viewport view. Owns nothing render-side; gesture handling and camera
/// state live in the [`FrameInput`] snapshot (same split as resims
/// `Viewport3d`).
///
/// Layout contract: fills its parent. Draws `background`, then sprites,
/// then world texts, then the fullscreen tint - all through the shared
/// [`effective_fit`] framing of [`FrameInput::world_size`]. Pointer presses
/// (`Click`) and cursor moves (`Hover`) are reported via `on_event` in
/// world coords (viewport-local, so a non-zero viewport origin never
/// shifts picks). Each paint publishes a
/// [`FrameGeom`] snapshot to `geom_out` for dp-space siblings
/// ([`ActorFrame`]); the viewport, picks, and actor surfaces therefore
/// share one transform by construction.
///
/// Canvas note: the canvas path draws debug solids (axis-aligned bounding
/// boxes of the snapshot quads). Rotated sprites are exact on the GPU
/// viewport; on canvas they draw as their AABB with a debug assert.
#[allow(non_snake_case)] // Repose view convention (cf. resims `Viewport3d`).
pub fn Viewport2d(
    input: FrameInput,
    geom_out: GeomHandle,
    on_event: impl Fn(PickEvent) + 'static,
) -> View {
    let input = Rc::new(input);
    let world_size = input.world_size;
    let pick_geom = geom_out.clone();
    let move_geom = geom_out.clone();
    let release_geom = geom_out.clone();
    let draw_input = input.clone();
    let draw_geom = geom_out.clone();
    let on_event = Rc::new(on_event);
    let on_down = on_event.clone();
    let on_touch_down = on_event.clone();
    let on_move = on_event.clone();
    let on_touch_move = on_event.clone();
    let on_up = on_event.clone();
    let on_up_click = on_event.clone();
    let on_leave = on_event;
    let press: Rc<Cell<Option<[f32; 2]>>> = Rc::new(Cell::new(None));
    let press_down = press.clone();
    let press_up = press.clone();
    let press_leave = press.clone();

    let modifier = Modifier::new()
        .fill_max_size()
        .on_pointer_down(move |ev: repose_core::input::PointerEvent| {
            let p = ev.position;
            let g = pick_geom.get();
            let world = pick_world([p.x, p.y], g, world_size);
            press_down.set(Some([p.x, p.y]));
            on_down(PickEvent::Press {
                world: Vec2::new(world[0], world[1]),
                screen: [p.x, p.y],
            });
            if is_touch(&ev) {
                on_touch_down(PickEvent::TouchDown {
                    id: ev.id.0,
                    screen: screen_of(&ev),
                });
            }
        })
        .on_pointer_move(move |ev: repose_core::input::PointerEvent| {
            let p = ev.position;
            let g = move_geom.get();
            let world = pick_world([p.x, p.y], g, world_size);
            on_move(PickEvent::Hover {
                world: Vec2::new(world[0], world[1]),
            });
            if is_touch(&ev) {
                on_touch_move(PickEvent::TouchMove {
                    id: ev.id.0,
                    screen: screen_of(&ev),
                });
            }
        })
        .on_pointer_up(move |ev: repose_core::input::PointerEvent| {
            let p = ev.position;
            if let Some(start) = press_up.take()
                && click_within_slop(start, [p.x, p.y])
            {
                let g = release_geom.get();
                let world = pick_world([p.x, p.y], g, world_size);
                on_up_click(PickEvent::Click {
                    world: Vec2::new(world[0], world[1]),
                    screen: [p.x, p.y],
                });
            }
            if is_touch(&ev) {
                on_up(PickEvent::TouchUp { id: ev.id.0 });
            }
        })
        .on_pointer_leave(move |ev: repose_core::input::PointerEvent| {
            press_leave.set(None);
            if is_touch(&ev) {
                on_leave(PickEvent::TouchUp { id: ev.id.0 });
            }
        });
    Canvas(modifier, move |scope: &mut DrawScope| {
        let d = effective_density_scale();
        let cam = draw_input.cam;
        let cam_center = cam.effective_center();
        let fit = effective_fit(
            [scope.size.width / d, scope.size.height / d],
            draw_input.world_size,
            &cam,
        );
        draw_geom.set(FrameGeom {
            fit,
            look: [
                cam_center[0] - draw_input.world_size[0] * 0.5,
                cam_center[1] - draw_input.world_size[1] * 0.5,
            ],
            density: d,
            viewport_px: [scope.size.width, scope.size.height],
            roll: cam.roll,
            pivot: cam_center,
        });
        let project = |wx: f32, wy: f32| -> [f32; 2] {
            let [dx, dy] = world_to_dp_with_roll(
                [wx, wy],
                draw_input.world_size,
                cam_center,
                (fit.0, fit.1, fit.2),
                cam.roll,
            );
            [dx * d, dy * d]
        };
        if let Some(bg) = draw_input.background {
            scope.draw_rect(
                Rect {
                    x: 0.0,
                    y: 0.0,
                    w: scope.size.width,
                    h: scope.size.height,
                },
                rgba8(bg),
                Px(0.0),
            );
        }
        for spr in draw_input.sprites.iter() {
            if spr.rotation.abs() > 1e-4 {
                debug_assert!(
                    false,
                    "canvas viewport draws rotated sprites as AABB; use GPU viewport for exact rotation"
                );
            }
            let (row0, row1) = batch::instance_rows(
                [spr.center.x, spr.center.y],
                [spr.size.x, spr.size.y],
                spr.rotation,
                [spr.anchor.x, spr.anchor.y],
                spr.flip_x,
                spr.flip_y,
            );
            let mut min_x = f32::INFINITY;
            let mut min_y = f32::INFINITY;
            let mut max_x = f32::NEG_INFINITY;
            let mut max_y = f32::NEG_INFINITY;
            for corner in [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]] {
                let wx = row0[0] * corner[0] + row0[1] * corner[1] + row0[3];
                let wy = row1[0] * corner[0] + row1[1] * corner[1] + row1[3];
                let [px, py] = project(wx, wy);
                min_x = min_x.min(px);
                min_y = min_y.min(py);
                max_x = max_x.max(px);
                max_y = max_y.max(py);
            }
            scope.draw_rect(
                Rect {
                    x: min_x,
                    y: min_y,
                    w: (max_x - min_x).max(0.0),
                    h: (max_y - min_y).max(0.0),
                },
                rgba8(spr.color),
                Px(0.0),
            );
        }
        for t in draw_input.texts.iter() {
            let [tx, ty] = project(t.pos.x, t.pos.y);
            scope.draw_text(
                t.text.clone(),
                repose_core::Vec2 { x: tx, y: ty },
                rgba8(t.color),
                Px(t.size * fit.0 * d),
            );
        }
        if let Some(tint) = draw_input.overlay_color {
            scope.draw_rect(
                Rect {
                    x: 0.0,
                    y: 0.0,
                    w: scope.size.width,
                    h: scope.size.height,
                },
                rgba8(tint),
                Px(0.0),
            );
        }
    })
}

/// GPU twin of [`Viewport2d`]: same snapshot in, same picks out, but
/// sprites draw as textured atlas quads through [`SpriteBatch`].
///
/// Framing is identical to the canvas path: the batch camera is
/// [`fit_view_proj`] over the density-corrected viewport (physical px /
/// density, same dp the canvas path draws in). `background` paints first
/// as a uniform-only fullscreen fill and `overlay_color` paints last the
/// same way, so GPU and canvas agree on every `FrameInput` field except
/// world texts, which stay a canvas-view feature for now: GPU consumers
/// compose those as sibling views.
///
/// Frame-timing: layout publishes the viewport size (plus fit/look from
/// the snapshot camera) through `on_size_changed` ahead of `prepare`, so
/// the batch camera, the chroma target size, and picks all share one
/// fresh geometry — resizes take effect the same frame. `paint`
/// re-publishes authoritatively from its callback info.
///
/// Layout contract: fills its parent. The payload is rebuilt from the
/// snapshot every frame (like the canvas draw closure): `prepare`
/// rebuilds the batch, and camera from the last painted viewport (cold
/// start falls back to [`FrameInput::viewport_dp`]), and `paint`
/// records the fresh viewport into `geom_out` and issues the shared
/// draw calls. Atlas uploads ride along per frame; games drain their
/// atlas queue once per frame, so each upload applies exactly once.
/// Geometry crosses the render thread behind a mutex (`WgpuCallback`
/// payloads must be `Send + Sync`), picks read it back on the UI side.
/// `FrameInput::chroma` above `0.0` renders the scene offscreen and
/// composites it back with an RGB split (see `post`); `0.0` draws the
/// batch straight into the main pass.
#[allow(non_snake_case)]
pub fn Viewport2dGpu(
    input: FrameInput,
    geom_out: GeomHandle,
    uploads: Vec<AtlasUpload>,
    desc: BatchDesc,
    on_event: impl Fn(PickEvent) + 'static,
) -> View {
    Viewport2dGpuWithId(input, geom_out, uploads, desc, "viewport2d.main", on_event)
}

/// GPU viewport with an explicit batch id so a second viewport/minimap can
/// coexist (each id owns its pipeline/instances in `CallbackResources`,
/// including its bg/overlay/composite targets).
#[allow(non_snake_case)]
pub fn Viewport2dGpuWithId(
    input: FrameInput,
    geom_out: GeomHandle,
    uploads: Vec<AtlasUpload>,
    desc: BatchDesc,
    batch_id: impl Into<String>,
    on_event: impl Fn(PickEvent) + 'static,
) -> View {
    let batch_id: String = batch_id.into();
    let bg_id = format!("{batch_id}.background");
    let overlay_id = format!("{batch_id}.overlay");
    let input = Arc::new(input);
    let world_size = input.world_size;
    let pick_geom = geom_out.clone();
    let move_geom = geom_out.clone();
    let release_geom = geom_out.clone();
    let size_geom = geom_out.clone();
    let size_input = input.clone();
    let geom_for_payload = geom_out.clone();
    let on_event = Arc::new(on_event);
    let on_down = on_event.clone();
    let on_touch_down = on_event.clone();
    let on_move = on_event.clone();
    let on_touch_move = on_event.clone();
    let on_up = on_event.clone();
    let on_up_click = on_event.clone();
    let on_leave = on_event;
    let press: Arc<Mutex<Option<[f32; 2]>>> = Arc::new(Mutex::new(None));
    let press_down = press.clone();
    let press_up = press.clone();
    let press_leave = press.clone();
    let payload = GpuViewport {
        input: input.clone(),
        geom: geom_for_payload.arc(),
        batch_id,
        uploads: Arc::from(uploads.into_boxed_slice()),
        desc,
        bg: FullscreenPass::new(
            bg_id,
            fullscreen::SOLID_WGSL,
            FullscreenDesc {
                texture_slots: 0,
                filter: TextureFilter::Nearest,
            },
        ),
        overlay: FullscreenPass::new(
            overlay_id,
            fullscreen::SOLID_WGSL,
            FullscreenDesc {
                texture_slots: 0,
                filter: TextureFilter::Nearest,
            },
        ),
    };
    let modifier = Modifier::new()
        .fill_max_size()
        .on_size_changed(move |dp: repose_core::Vec2| {
            let d = effective_density_scale();
            let d = if d.is_finite() && d > 1e-6 { d } else { 1.0 };
            let fit = effective_fit([dp.x, dp.y], size_input.world_size, &size_input.cam);
            let center = size_input.cam.effective_center();
            size_geom.set(FrameGeom {
                viewport_px: [dp.x * d, dp.y * d],
                density: d,
                fit,
                look: [
                    center[0] - size_input.world_size[0] * 0.5,
                    center[1] - size_input.world_size[1] * 0.5,
                ],
                roll: size_input.cam.roll,
                pivot: center,
            });
        })
        .on_pointer_down(move |ev: repose_core::input::PointerEvent| {
            let p = ev.position;
            let g = pick_geom.get();
            let world = pick_world([p.x, p.y], g, world_size);
            if let Ok(mut slot) = press_down.lock() {
                *slot = Some([p.x, p.y]);
            }
            on_down(PickEvent::Press {
                world: Vec2::new(world[0], world[1]),
                screen: [p.x, p.y],
            });
            if is_touch(&ev) {
                on_touch_down(PickEvent::TouchDown {
                    id: ev.id.0,
                    screen: screen_of(&ev),
                });
            }
        })
        .on_pointer_move(move |ev: repose_core::input::PointerEvent| {
            let p = ev.position;
            let g = move_geom.get();
            let world = pick_world([p.x, p.y], g, world_size);
            on_move(PickEvent::Hover {
                world: Vec2::new(world[0], world[1]),
            });
            if is_touch(&ev) {
                on_touch_move(PickEvent::TouchMove {
                    id: ev.id.0,
                    screen: screen_of(&ev),
                });
            }
        })
        .on_pointer_up(move |ev: repose_core::input::PointerEvent| {
            let p = ev.position;
            let start = press_up.lock().ok().and_then(|mut s| s.take());
            if let Some(start) = start
                && click_within_slop(start, [p.x, p.y])
            {
                let g = release_geom.get();
                let world = pick_world([p.x, p.y], g, world_size);
                on_up_click(PickEvent::Click {
                    world: Vec2::new(world[0], world[1]),
                    screen: [p.x, p.y],
                });
            }
            if is_touch(&ev) {
                on_up(PickEvent::TouchUp { id: ev.id.0 });
            }
        })
        .on_pointer_leave(move |ev: repose_core::input::PointerEvent| {
            if let Ok(mut s) = press_leave.lock() {
                *s = None;
            }
            if is_touch(&ev) {
                on_leave(PickEvent::TouchUp { id: ev.id.0 });
            }
        });
    Embedded(modifier, Callback::new(payload))
}

struct GpuViewport {
    input: Arc<FrameInput>,
    geom: Arc<Mutex<FrameGeom>>,
    batch_id: String,
    uploads: Arc<[AtlasUpload]>,
    desc: BatchDesc,
    bg: FullscreenPass,
    overlay: FullscreenPass,
}

impl WgpuCallback for GpuViewport {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        screen: &ScreenDescriptor,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let (vp_phys, density) = self
            .geom
            .lock()
            .map(|g| {
                let d = if g.density.is_finite() && g.density > 1e-6 {
                    g.density
                } else {
                    1.0
                };
                (g.viewport_px, d)
            })
            .unwrap_or(([1.0, 1.0], 1.0));
        let dp = if vp_phys[0] > 1.0 && vp_phys[1] > 1.0 {
            [vp_phys[0] / density, vp_phys[1] / density]
        } else {
            self.input.viewport_dp
        };
        let mut batch = SpriteBatch::with_id(self.batch_id.clone(), self.desc);
        batch.set_camera(self.input.cam.fit_matrix(dp, self.input.world_size));
        for s in &self.input.sprites {
            batch.push_sprite(s);
        }
        batch.extend_uploads(self.uploads.iter().cloned());
        batch.prepare(device, queue, encoder, screen, resources);
        let composite = post::use_composite(self.input.chroma);
        // Background first (uniform-only fill, same color the canvas
        // path fills under the batch). Skipped when the snapshot has
        if let Some(bg) = self.input.background
            && !composite
        {
            let words = [bg[0], bg[1], bg[2], bg[3]];
            self.bg.prepare_with(
                device,
                queue,
                screen,
                resources,
                bytemuck::cast_slice(&words),
                &[],
            );
        }
        if let Some(ov) = self.input.overlay_color {
            let words = [ov[0], ov[1], ov[2], ov[3]];
            self.overlay.prepare_with(
                device,
                queue,
                screen,
                resources,
                bytemuck::cast_slice(&words),
                &[],
            );
        }
        if post::use_composite(self.input.chroma) {
            let w = vp_phys[0].max(1.0) as u32;
            let h = vp_phys[1].max(1.0) as u32;
            post::prepare_composite(
                device,
                queue,
                encoder,
                screen,
                resources,
                self.batch_id.as_str(),
                w,
                h,
                self.input.chroma,
                self.input.background,
            );
        }
        Vec::new()
    }

    fn paint(
        &self,
        info: repose_core::PaintCallbackInfo,
        rpass: &mut wgpu::RenderPass<'static>,
        resources: &CallbackResources,
    ) {
        let d = if info.pixels_per_point.is_finite() && info.pixels_per_point > 1e-6 {
            info.pixels_per_point
        } else {
            1.0
        };
        let vp = [info.viewport.w, info.viewport.h];
        let fit = effective_fit(
            [vp[0] / d, vp[1] / d],
            self.input.world_size,
            &self.input.cam,
        );
        let cam = self.input.cam.effective_center();
        let roll = self.input.cam.roll;
        if let Ok(mut g) = self.geom.lock() {
            *g = FrameGeom {
                fit,
                look: [
                    cam[0] - self.input.world_size[0] * 0.5,
                    cam[1] - self.input.world_size[1] * 0.5,
                ],
                density: d,
                viewport_px: vp,
                roll,
                pivot: cam,
            };
        }
        if post::use_composite(self.input.chroma) {
            post::paint_composite(self.batch_id.as_str(), rpass, resources);
        } else {
            if self.input.background.is_some() {
                self.bg.paint(info, rpass, resources);
            }
            draw_batch_with_id(self.batch_id.as_str(), rpass, resources);
        }
        if self.input.overlay_color.is_some() {
            self.overlay.paint(info, rpass, resources);
        }
    }
}

/// Stack GPU sprites under canvas texts/tint using the same FrameInput.
///
/// Until a glyph atlas exists, GPU games compose world texts as a canvas
/// overlay sibling: this helper mounts `Viewport2dGpu` plus a transparent
/// canvas pass that only draws `texts`/`overlay_color` with the same geom,
/// so GPU games aren't silently missing damage numbers.
#[allow(non_snake_case)]
pub fn Viewport2dGpuWithHud(
    input: FrameInput,
    geom: GeomHandle,
    uploads: Vec<AtlasUpload>,
    desc: BatchDesc,
    on_event: impl Fn(PickEvent) + 'static,
) -> View {
    use repose_ui::ViewExt as _;
    let texts = input.texts.clone();
    let overlay = input.overlay_color;
    let cam = input.cam;
    let world_size = input.world_size;
    let hud_input = std::rc::Rc::new((cam, world_size, texts, overlay));
    let hud_geom = geom.clone();
    let hud = Canvas(
        Modifier::new().fill_max_size().hit_passthrough(),
        move |scope: &mut DrawScope| {
            let (cam, world_size, texts, overlay) =
                (hud_input.0, hud_input.1, &hud_input.2, hud_input.3);
            let d = effective_density_scale();
            let cam_center = cam.effective_center();
            let fit = effective_fit(
                [scope.size.width / d, scope.size.height / d],
                world_size,
                &cam,
            );
            hud_geom.set(FrameGeom {
                fit,
                look: [
                    cam_center[0] - world_size[0] * 0.5,
                    cam_center[1] - world_size[1] * 0.5,
                ],
                density: d,
                viewport_px: [scope.size.width, scope.size.height],
                roll: cam.roll,
                pivot: cam_center,
            });
            let project = |wx: f32, wy: f32| -> [f32; 2] {
                let [dx, dy] = world_to_dp_with_roll(
                    [wx, wy],
                    world_size,
                    cam_center,
                    (fit.0, fit.1, fit.2),
                    cam.roll,
                );
                [dx * d, dy * d]
            };
            for t in texts.iter() {
                let [tx, ty] = project(t.pos.x, t.pos.y);
                scope.draw_text(
                    t.text.clone(),
                    repose_core::Vec2 { x: tx, y: ty },
                    rgba8(t.color),
                    Px(t.size * fit.0 * d),
                );
            }
            if let Some(tint) = overlay {
                scope.draw_rect(
                    Rect {
                        x: 0.0,
                        y: 0.0,
                        w: scope.size.width,
                        h: scope.size.height,
                    },
                    rgba8(tint),
                    Px(0.0),
                );
            }
        },
    );
    repose_ui::ZStack(Modifier::new().fill_max_size())
        .child(Viewport2dGpu(input, geom, uploads, desc, on_event))
        .child(hud)
}
/// A world-anchored surface: `child` (painting in its own local coords)
/// is boxed to a `size` rect centered on a world `center`, positioned
/// through the viewport's [`FrameGeom`]. Siblings of [`Viewport2d`] -
/// e.g. live actor surfaces - track sim positions without game-side unit
/// math, glued to sprites and picks even under camera shake. `mirror_x`
/// flips around the surface center (left-walking hordes reusing
/// right-facing art).
///
/// Timing: offsets derive from the last painted [`FrameGeom`], so the
/// surface trails the board by one frame while the camera moves
/// (see [`FrameGeom`]). Steady-state placement is exact.
#[allow(non_snake_case)]
pub fn ActorFrame(
    center: [f32; 2],
    size: [f32; 2],
    geom: GeomHandle,
    mirror_x: bool,
    child: View,
) -> View {
    let ([ox, oy], [w, h]) = surface_dp(center, size, geom.get());
    // `.absolute()` is load-bearing: without it taffy ignores the offsets
    // and every surface stacks at the same fixed spot.
    let mut modifier = Modifier::new().size(Dp(w), Dp(h)).absolute().offset(
        Some(Dp(ox)),
        Some(Dp(oy)),
        None,
        None,
    );
    if mirror_x {
        modifier = modifier.scale2(-1.0, 1.0);
    }
    UiBox(modifier).child(child)
}

#[cfg(test)]
mod tests {
    use super::*;
    use repose_core::locals::{Density, set_density_default};

    #[test]
    fn contain_fit_roundtrip_at_density_125() {
        // Production eDP-1 runs at 1.25: px canvas -> dp fit -> world ->
        // back must be identity or clicks miss their visuals.
        set_density_default(Density { scale: 1.25 });
        let d = effective_density_scale();
        // 1250x750 px canvas, 800x600 world: dp fit s=1, ox=100, oy=0.
        let fit = contain_fit([1250.0 / d, 750.0 / d], [800.0, 600.0]);
        let cam = [400.0, 300.0];
        // Draw: world (400,300) -> dp -> px must hit the press point.
        let [dx, dy] = world_to_dp([400.0, 300.0], [800.0, 600.0], cam, fit);
        // Pick: window px -> dp -> world must map back exactly.
        let [wx, wy] = dp_to_world([625.0 / d, 375.0 / d], [800.0, 600.0], cam, fit);
        set_density_default(Density { scale: 1.0 });
        assert!(
            (fit.0 - 1.0).abs() < 1e-4 && (fit.1 - 100.0).abs() < 1e-4 && fit.2.abs() < 1e-4,
            "dp fit, got {fit:?}"
        );
        assert!(
            (dx * d - 625.0).abs() < 1e-3 && (dy * d - 375.0).abs() < 1e-3,
            "draw forward, got ({dx},{dy})"
        );
        assert!(
            (wx - 400.0).abs() < 1e-3 && (wy - 300.0).abs() < 1e-3,
            "pick inverse, got ({wx},{wy})"
        );
        // Camera shift (trauma shake) moves draw and pick together.
        let shifted = world_to_dp([400.0, 300.0], [800.0, 600.0], [410.0, 290.0], fit);
        let back = dp_to_world(shifted, [800.0, 600.0], [410.0, 290.0], fit);
        assert!(
            (back[0] - 400.0).abs() < 1e-4 && (back[1] - 300.0).abs() < 1e-4,
            "shifted round trip, got {back:?}"
        );
    }

    #[test]
    fn gpu_viewport_honors_cam_and_background() {
        // End-to-end GPU proof for the viewport contract: the snapshot
        // camera frames world sprites (cam center lands on the screen
        // center) and `background` fills every uncovered pixel. Skips
        // gracefully where no GPU exists.
        use repose_core::{Color, Rect, Scene, SceneNode};
        use repose_render_wgpu::{Callback, offscreen::OffscreenRenderer};

        let mut renderer = match OffscreenRenderer::new_blocking(256, 256, 1) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP gpu viewport test (no GPU): {e}");
                return;
            }
        };
        // Page 0: solid magenta 2x2.
        let page = [255, 0, 255, 255].repeat(4);
        let uploads = vec![AtlasUpload {
            page: 0,
            x: 0,
            y: 0,
            w: 2,
            h: 2,
            rgba: page,
        }];
        let cam = Camera2d {
            center: Vec2::new(64.0, 64.0),
            offset: Vec2::ZERO,
            units_per_pixel: 0.5,
            zoom: 1.0,
            roll: 0.0,
        };
        let input = FrameInput {
            cam,
            world_size: [128.0, 128.0],
            viewport_dp: [256.0, 256.0],
            sprites: vec![SpriteInstance {
                center: Vec2::new(64.0, 64.0),
                size: Vec2::new(32.0, 32.0),
                uv_min: Vec2::new(0.0, 0.0),
                uv_max: Vec2::new(1.0, 1.0),
                color: [1.0, 1.0, 1.0, 1.0],
                ..Default::default()
            }],
            texts: Vec::new(),
            background: Some([0.0, 0.0, 1.0, 1.0]),
            overlay_color: None,
            chroma: 0.0,
        };
        let geom = Arc::new(Mutex::new(FrameGeom {
            fit: (1.0, 0.0, 0.0),
            look: [0.0, 0.0],
            density: 1.0,
            viewport_px: [256.0, 256.0],
            roll: 0.0,
            pivot: [64.0, 64.0],
        }));
        let payload = GpuViewport {
            input: Arc::new(input.clone()),
            geom,
            batch_id: "test.viewport2d.main".to_string(),
            uploads: Arc::from(uploads.into_boxed_slice()),
            desc: BatchDesc {
                layer_size: 2,
                layers: 1,
                filter: TextureFilter::Nearest,
            },
            bg: FullscreenPass::new(
                "test.viewport2d.background",
                fullscreen::SOLID_WGSL,
                FullscreenDesc {
                    texture_slots: 0,
                    filter: TextureFilter::Nearest,
                },
            ),
            overlay: FullscreenPass::new(
                "test.viewport2d.overlay",
                fullscreen::SOLID_WGSL,
                FullscreenDesc {
                    texture_slots: 0,
                    filter: TextureFilter::Nearest,
                },
            ),
        };
        let scene = Scene {
            clear_color: Color::from_rgba(0, 0, 0, 255),
            nodes: vec![SceneNode::Callback {
                rect: Rect {
                    x: 0.0,
                    y: 0.0,
                    w: 256.0,
                    h: 256.0,
                },
                payload: Callback::new(payload),
            }],
        };
        let px = renderer
            .render_rgba(&scene, Some([0.0, 0.0, 0.0, 1.0]))
            .expect("offscreen render");
        let at = |x: u32, y: u32| -> [u8; 4] {
            let i = ((y * 256 + x) * 4) as usize;
            [px[i], px[i + 1], px[i + 2], px[i + 3]]
        };
        // Corners are outside the centered quad: snapshot background.
        assert_eq!(at(8, 8), [0, 0, 255, 255], "background fill");
        assert_eq!(at(247, 247), [0, 0, 255, 255], "background fill");
        assert_eq!(at(128, 128), [255, 0, 255, 255], "cam-centered quad");
        let overlay_input = FrameInput {
            overlay_color: Some([1.0, 0.0, 0.0, 1.0]),
            ..input
        };
        let overlay_payload = GpuViewport {
            input: Arc::new(overlay_input),
            geom: Arc::new(Mutex::new(FrameGeom {
                fit: (1.0, 0.0, 0.0),
                look: [0.0, 0.0],
                density: 1.0,
                viewport_px: [256.0, 256.0],
                roll: 0.0,
                pivot: [64.0, 64.0],
            })),
            batch_id: "test.viewport2d.overlay".to_string(),
            uploads: Arc::from(
                vec![AtlasUpload {
                    page: 0,
                    x: 0,
                    y: 0,
                    w: 2,
                    h: 2,
                    rgba: [255, 0, 255, 255].repeat(4),
                }]
                .into_boxed_slice(),
            ),
            desc: BatchDesc {
                layer_size: 2,
                layers: 1,
                filter: TextureFilter::Nearest,
            },
            bg: FullscreenPass::new(
                "test.viewport2d.background",
                fullscreen::SOLID_WGSL,
                FullscreenDesc {
                    texture_slots: 0,
                    filter: TextureFilter::Nearest,
                },
            ),
            overlay: FullscreenPass::new(
                "test.viewport2d.overlay",
                fullscreen::SOLID_WGSL,
                FullscreenDesc {
                    texture_slots: 0,
                    filter: TextureFilter::Nearest,
                },
            ),
        };
        let scene = Scene {
            clear_color: Color::from_rgba(0, 0, 0, 255),
            nodes: vec![SceneNode::Callback {
                rect: Rect {
                    x: 0.0,
                    y: 0.0,
                    w: 256.0,
                    h: 256.0,
                },
                payload: Callback::new(overlay_payload),
            }],
        };
        let px = renderer
            .render_rgba(&scene, Some([0.0, 0.0, 0.0, 1.0]))
            .expect("offscreen render");
        let at = |x: u32, y: u32| -> [u8; 4] {
            let i = ((y * 256 + x) * 4) as usize;
            [px[i], px[i + 1], px[i + 2], px[i + 3]]
        };
        assert_eq!(at(8, 8), [255, 0, 0, 255], "overlay covers background");
        assert_eq!(at(128, 128), [255, 0, 0, 255], "overlay covers sprite");
    }

    #[test]
    fn surface_dp_matches_board_point_under_shake() {
        // Actor surfaces and board sprites share FrameGeom: the surface
        // center must land exactly on the board-space point, including
        // with a shifted look point (trauma shake).
        let geom = FrameGeom {
            fit: (1.0, 100.0, 0.0),
            look: [0.0, 0.0],
            density: 1.25,
            viewport_px: [1250.0, 750.0],
            roll: 0.0,
            pivot: [400.0, 300.0],
        };
        let ([ox, oy], [w, h]) = surface_dp([400.0, 300.0], [64.0, 80.0], geom);
        assert!(
            (ox - 468.0).abs() < 1e-4 && (oy - 260.0).abs() < 1e-4,
            "offset, got ({ox},{oy})"
        );
        assert!(
            (w - 64.0).abs() < 1e-4 && (h - 80.0).abs() < 1e-4,
            "size, got ({w},{h})"
        );
        // Surface center == board point for the same world pos.
        let board = world_to_dp([400.0, 300.0], [800.0, 600.0], [400.0, 300.0], geom.fit);
        assert!(
            (ox + w * 0.5 - board[0]).abs() < 1e-4 && (oy + h * 0.5 - board[1]).abs() < 1e-4,
            "surface glued to board point"
        );
        let shaken = FrameGeom {
            look: [10.0, -10.0],
            ..geom
        };
        let ([sx, sy], _) = surface_dp([400.0, 300.0], [64.0, 80.0], shaken);
        let sboard = world_to_dp([400.0, 300.0], [800.0, 600.0], [410.0, 290.0], shaken.fit);
        assert!(
            (sx + w * 0.5 - sboard[0]).abs() < 1e-4 && (sy + h * 0.5 - sboard[1]).abs() < 1e-4,
            "surface glued under shake"
        );
    }

    #[test]
    fn contain_fit_has_no_silent_clamp() {
        for (canvas, world) in [
            ([8000.0, 6000.0], [80.0, 60.0]),
            ([80.0, 60.0], [8000.0, 6000.0]),
            ([1000.0, 750.0], [800.0, 600.0]),
        ] {
            let fit = contain_fit(canvas, world);
            let cam = [world[0] * 0.5, world[1] * 0.5];
            let dp = world_to_dp([cam[0], cam[1]], world, cam, fit);
            let back = dp_to_world(dp, world, cam, fit);
            assert!(
                (back[0] - cam[0]).abs() < 1e-3 && (back[1] - cam[1]).abs() < 1e-3,
                "round trip for {canvas:?}/{world:?}, fit {fit:?} gave {back:?}"
            );
        }
        assert_eq!(contain_fit([0.0, 600.0], [800.0, 600.0]), (1.0, 0.0, 0.0));
        assert_eq!(
            dp_to_world(
                [10.0, 10.0],
                [800.0, 600.0],
                [400.0, 300.0],
                (0.0, 0.0, 0.0)
            ),
            [400.0, 300.0]
        );
    }

    #[test]
    #[allow(deprecated)]
    fn camera_guards_never_div_by_zero() {
        let bad = Camera2d {
            center: Vec2::ZERO,
            offset: Vec2::ZERO,
            units_per_pixel: 0.0,
            zoom: 0.0,
            roll: 0.0,
        };
        let m = bad.view_proj([0.0, 0.0]);
        assert!(m.is_finite(), "view_proj must stay finite, got {m:?}");
        assert_eq!(bad.screen_to_world([0.0, 0.0], [5.0, 5.0]), Vec2::ZERO);
    }

    #[test]
    fn fit_matrix_canvas_gpu_picks_agree() {
        let cam = Camera2d {
            center: Vec2::new(410.0, 290.0),
            offset: Vec2::ZERO,
            units_per_pixel: 1.0,
            zoom: 2.0,
            roll: 0.0,
        };
        let world_size = [800.0, 600.0];
        let density = 1.25;
        let vp_phys = [1250.0, 750.0];
        let canvas_dp = [vp_phys[0] / density, vp_phys[1] / density];
        let cam_center = cam.effective_center();
        let fit = effective_fit(canvas_dp, world_size, &cam);
        assert!((fit.0 - 2.0).abs() < 1e-4, "effective fit, got {fit:?}");
        let vp = cam.fit_matrix(canvas_dp, world_size);
        assert!(vp.is_finite());
        for world in [[400.0, 300.0], [410.0, 290.0], [0.0, 0.0], [800.0, 600.0]] {
            let [dx, dy] = world_to_dp(world, world_size, cam_center, fit);
            let px = [dx * density, dy * density];
            let ndc = vp.project_point3(glam::Vec3::new(world[0], world[1], 0.0));
            let gpu_px = [
                (ndc.x + 1.0) * 0.5 * vp_phys[0],
                (1.0 - ndc.y) * 0.5 * vp_phys[1],
            ];
            assert!(
                (px[0] - gpu_px[0]).abs() < 1e-2 && (px[1] - gpu_px[1]).abs() < 1e-2,
                "canvas/GPU disagree for {world:?}: canvas {px:?} vs gpu {gpu_px:?}"
            );
            let back =
                cam.dp_to_world_pt(canvas_dp, world_size, [px[0] / density, px[1] / density]);
            assert!(
                (back.x - world[0]).abs() < 1e-3 && (back.y - world[1]).abs() < 1e-3,
                "pick round trip for {world:?} gave {back:?}"
            );
        }
    }

    #[test]
    fn letterbox_pick_matches_draw() {
        // canvas 1600x900 dp, world 800x800 → horizontal letterbox
        let cam = Camera2d {
            center: Vec2::new(400.0, 400.0),
            ..Default::default()
        };
        let fit = effective_fit([1600.0, 900.0], [800.0, 800.0], &cam);
        let dp = world_to_dp([400.0, 400.0], [800.0, 800.0], cam.effective_center(), fit);
        let back = dp_to_world(dp, [800.0, 800.0], cam.effective_center(), fit);
        assert!((back[0] - 400.0).abs() < 1e-3 && (back[1] - 400.0).abs() < 1e-3);
        // fit_matrix agrees with the free function.
        let m = cam.fit_matrix([1600.0, 900.0], [800.0, 800.0]);
        let free = fit_view_proj([1600.0, 900.0], [800.0, 800.0], cam.effective_center(), fit);
        let dm = m - free;
        let max = dm
            .to_cols_array()
            .iter()
            .fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(max < 1e-5);
    }

    #[test]
    fn picks_ignore_viewport_origin() {
        let geom = FrameGeom {
            fit: (1.0, 100.0, 0.0),
            look: [0.0, 0.0],
            density: 1.25,
            viewport_px: [1250.0, 750.0],
            roll: 0.0,
            pivot: [400.0, 300.0],
        };
        let world_size = [800.0, 600.0];
        let a = pick_world([125.0, 75.0], geom, world_size);
        let b = pick_world([125.0, 75.0], geom, world_size);
        assert_eq!(a, b);
        let [wx, wy] = pick_world([125.0, 75.0], geom, world_size);
        assert!(
            (wx - 0.0).abs() < 1e-4 && (wy - 60.0).abs() < 1e-4,
            "local pick, got ({wx},{wy})"
        );
    }

    #[test]
    fn offset_shifts_look_and_draw_together() {
        // Shake rides the offset channel: the same world point draws and
        // picks with the offset applied.
        let world_size = [800.0, 600.0];
        let canvas_dp = [1000.0, 600.0];
        let mut cam = Camera2d {
            center: Vec2::new(400.0, 300.0),
            offset: Vec2::new(10.0, -10.0),
            units_per_pixel: 1.0,
            zoom: 1.0,
            roll: 0.0,
        };
        assert_eq!(cam.effective_center(), [410.0, 290.0]);
        let fit = effective_fit(canvas_dp, world_size, &cam);
        let dp = world_to_dp([400.0, 300.0], world_size, cam.effective_center(), fit);
        let back = dp_to_world(dp, world_size, cam.effective_center(), fit);
        assert!((back[0] - 400.0).abs() < 1e-4 && (back[1] - 300.0).abs() < 1e-4);
        cam.offset = Vec2::ZERO;
        let dp2 = world_to_dp([400.0, 300.0], world_size, cam.effective_center(), fit);
        // Offset 10 world units at scale 1 moves the drawing by 10 dp.
        assert!((dp[0] - dp2[0] + 10.0).abs() < 1e-4);
        assert!((dp[1] - dp2[1] - 10.0).abs() < 1e-4);
    }

    #[test]
    fn limits_clamp_center_not_offset() {
        let limits = CameraLimits {
            left: 100.0,
            top: 100.0,
            right: 700.0,
            bottom: 500.0,
            enabled: true,
        };
        assert_eq!(apply_limits([50.0, 300.0], limits), [100.0, 300.0]);
        assert_eq!(apply_limits([400.0, 900.0], limits), [400.0, 500.0]);
        assert_eq!(apply_limits([400.0, 300.0], limits), [400.0, 300.0]);
        assert_eq!(
            apply_limits(
                [50.0, 50.0],
                CameraLimits {
                    enabled: false,
                    ..limits
                }
            ),
            [50.0, 50.0]
        );
    }

    #[test]
    fn smoothing_converges_without_overshoot() {
        let target = [100.0, 0.0];
        let p1 = smooth_toward([0.0, 0.0], target, 5.0, 0.016);
        assert!(p1[0] > 0.0 && p1[0] < 100.0, "eases forward, got {p1:?}");
        let mut p = [0.0, 0.0];
        for _ in 0..600 {
            p = smooth_toward(p, target, 5.0, 0.016);
        }
        assert!((p[0] - 100.0).abs() < 1e-2, "settles, got {p:?}");
        assert_eq!(smooth_toward([1.0, 2.0], target, 0.0, 0.016), target);
        assert_eq!(smooth_toward([1.0, 2.0], target, 5.0, 0.0), [1.0, 2.0]);
    }

    #[test]
    fn touch_gate_and_screen_space() {
        use repose_core::Vec2 as RVec2;
        use repose_core::input::{
            Modifiers, PointerButton, PointerEvent, PointerEventKind, PointerId, PointerKind,
        };
        let ev_of = |kind: PointerKind| {
            let mut ev = PointerEvent::new(
                PointerId(3),
                kind,
                PointerEventKind::Down(PointerButton::Primary),
                RVec2 { x: 10.0, y: 20.0 },
                1.0,
                Modifiers::default(),
            );
            ev.origin = RVec2 { x: 5.0, y: 7.0 };
            ev
        };
        // Touch + pen forward to touch zones; mouse stays on clicks.
        assert!(is_touch(&ev_of(PointerKind::Touch)));
        assert!(is_touch(&ev_of(PointerKind::Pen)));
        assert!(!is_touch(&ev_of(PointerKind::Mouse)));
        // Touch zones sample window-physical px (origin + position).
        assert_eq!(screen_of(&ev_of(PointerKind::Touch)), [15.0, 27.0]);
    }

    #[test]
    fn click_slop_cancels_drags() {
        assert!(click_within_slop([100.0, 100.0], [105.0, 105.0]));
        assert!(!click_within_slop(
            [100.0, 100.0],
            [100.0 + CLICK_SLOP_PX + 1.0, 100.0]
        ));
    }

    #[test]
    fn roll_roundtrips_and_moves_pixels() {
        let cam = Camera2d {
            center: Vec2::new(400.0, 300.0),
            offset: Vec2::ZERO,
            units_per_pixel: 1.0,
            zoom: 1.0,
            roll: std::f32::consts::FRAC_PI_2,
        };
        let world = [800.0, 600.0];
        let dp = [1000.0, 600.0];
        let fit = effective_fit(dp, world, &cam);
        for p in [[400.0, 300.0], [500.0, 300.0], [0.0, 0.0]] {
            let fwd = world_to_dp_with_roll(p, world, cam.effective_center(), fit, cam.roll);
            let back = dp_to_world_with_roll(fwd, world, cam.effective_center(), fit, cam.roll);
            assert!(
                (back[0] - p[0]).abs() < 1e-3 && (back[1] - p[1]).abs() < 1e-3,
                "roll roundtrip for {p:?} gave {back:?}"
            );
        }
        let plain = world_to_dp([500.0, 300.0], world, cam.effective_center(), fit);
        let rolled =
            world_to_dp_with_roll([500.0, 300.0], world, cam.effective_center(), fit, cam.roll);
        assert!(
            (plain[0] - rolled[0]).abs() + (plain[1] - rolled[1]).abs() > 1.0,
            "roll must move pixels, plain {plain:?} rolled {rolled:?}"
        );
        let m0 = fit_view_proj_with_roll(dp, world, cam.effective_center(), fit, 0.0);
        let legacy = fit_view_proj(dp, world, cam.effective_center(), fit);
        assert_eq!(m0, legacy);
    }

    #[test]
    fn surface_tracks_roll_about_pivot() {
        let geom = FrameGeom {
            fit: (1.0, 0.0, 0.0),
            look: [0.0, 0.0],
            density: 1.0,
            viewport_px: [800.0, 600.0],
            roll: std::f32::consts::FRAC_PI_2,
            pivot: [400.0, 300.0],
        };
        let ([ox, oy], _) = surface_dp([500.0, 300.0], [10.0, 10.0], geom);
        assert!(
            (ox - 395.0).abs() < 1e-3 && (oy - 395.0).abs() < 1e-3,
            "rolled surface, got ({ox},{oy})"
        );
    }

    #[test]
    fn geom_handle_compat_wraps_legacy_handles() {
        use std::cell::Cell;
        use std::rc::Rc;
        let rc = Rc::new(Cell::new(FrameGeom {
            fit: (2.0, 1.0, 2.0),
            ..Default::default()
        }));
        let h: GeomHandle = rc.into();
        assert_eq!(h.get().fit, (2.0, 1.0, 2.0));
        let arc = Arc::new(Mutex::new(FrameGeom::default()));
        let h2: GeomHandle = arc.into();
        assert_eq!(h2.get().density, 1.0);
    }
}
