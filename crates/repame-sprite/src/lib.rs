//! 2D sprite viewport: snapshot in, pixels out.
//!
//! The UI builds a [`FrameInput`] per frame (plain data, rebuilt
//! during composition) and mounts [`Viewport2d`] as a Repose view, which
//! draws the snapshot through a dp-space contain-fit and reports pointer
//! picks back in world coords. Unit discipline lives here, once:
//! games work in world/dp units and do not touch physical px.
//! Camera state lives in Repose signals, not in the renderer.

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
/// snapshot per frame. `effective_center` is `center + offset`.
/// Canvas, GPU batch, and picks derive from these fields together.
#[derive(Clone, Copy, Debug)]
pub struct Camera2d {
    /// Follow target in world units. Defaults to `(0, 0)`.
    /// Scroll limits clamp this field; shake goes in `offset`.
    pub center: Vec2,
    /// Momentary displacement added to `center`. Defaults to `(0, 0)`.
    /// Applied after scroll limits; game code decays it toward zero.
    pub offset: Vec2,
    /// World units per screen pixel at zoom 1. Defaults to `1.0`.
    /// Visible width is `viewport_dp * units_per_pixel / zoom`.
    /// Non-positive values fall back to `1.0` (guards division).
    pub units_per_pixel: f32,
    /// Magnification. Defaults to `1.0`; higher zooms in.
    /// Non-positive values fall back to `1.0`.
    pub zoom: f32,
    /// Roll in radians about the look point. Defaults to `0.0`.
    /// Applied on canvas, GPU, picks, and actor surfaces alike.
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

/// Scroll limits and follow smoothing live in `game_utils_repame::feel`.
/// They run game-side when producing the snapshot; this crate only
/// frames the snapshot contents.

impl Camera2d {
    /// Effective look point: `center + offset`.
    /// Canvas, GPU batch, picks, and actor surfaces all use this value.
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

    /// World point under a dp-space cursor (density divided out).
    /// Includes roll.
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
    /// Frames from `viewport * units_per_pixel / zoom` and ignores
    /// `world_size`, so it disagrees with canvas, GPU, and picks
    /// under letterbox. Viewport code must use [`fit_view_proj`].
    #[deprecated(
        since = "0.1.8",
        note = "ignores world_size; use fit_view_proj for the shared canvas/GPU/pick framing contract"
    )]
    pub fn view_proj(&self, viewport_px: [f32; 2]) -> Mat4 {
        self.view_proj_screen(viewport_px)
    }

    /// Legacy position under a viewport-pixel cursor.
    /// Inverts `view_proj` with the same letterbox divergence.
    /// Picks must use [`pick_world`] over the painted [`FrameGeom`].
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

/// Blend mode per sprite. Additive uses `SrcAlpha + One`.
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
    /// Normalized anchor: `[0.5, 0.5]` centers the quad on `center`,
    /// `[0, 0]` pins the top-left corner. Y-down.
    pub anchor: Vec2,
    /// Mirror about the anchor axes. Canvas fills are flip-invariant;
    /// the GPU batch mirrors geometry (UVs untouched).
    pub flip_x: bool,
    pub flip_y: bool,
    pub uv_min: Vec2,
    pub uv_max: Vec2,
    /// RGBA tint, `(c * 255) as u8` per channel.
    pub color: [f32; 4],
    /// Atlas page index for multi-texture batches.
    pub page: u32,
    /// Draw key; higher draws on top. Default 0. Stable-sorted by `z`.
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
    /// World-space anchor.
    pub pos: Vec2,
    /// RGBA, `(c * 255) as u8` per channel.
    pub color: [f32; 4],
    /// Font size in world units.
    pub size: f32,
}

/// Everything the viewport draws this frame. Plain snapshot data.
///
/// Framing contract: canvas, GPU, and picks all derive from `world_size`
/// + `cam` through [`effective_fit`] / [`fit_view_proj`] / [`world_to_dp`].
#[derive(Clone, Default, Debug)]
pub struct FrameInput {
    pub cam: Camera2d,
    /// World-space extent the contain-fit maps into the viewport.
    pub world_size: [f32; 2],
    /// Cold-start viewport size in dp (physical px / density), used for
    /// the GPU camera until the first painted [`FrameGeom`] arrives.
    /// Divide physical px by density before storing here.
    pub viewport_dp: [f32; 2],
    pub sprites: Vec<SpriteInstance>,
    /// World-anchored text, drawn after the sprite pass on canvas.
    /// GPU viewports ignore this field; compose texts as sibling views.
    pub texts: Vec<WorldText>,
    /// Full-viewport fill under everything (letterbox included).
    pub background: Option<[f32; 4]>,
    /// Optional fullscreen tint, applied after the sprite pass on
    /// canvas and GPU (`(c * 255) as u8` per channel, alpha-blended).
    pub overlay_color: Option<[f32; 4]>,
    /// Chromatic aberration amount. GPU renders offscreen and composites
    /// back with an RGB split; `0.0` draws the batch into the main pass.
    /// Canvas ignores it (vector path has no pixels to sample).
    pub chroma: f32,
}

/// UI-facing pointer events from the viewport.
/// `Click` fires on pointer up when the press started inside and the
/// pointer stayed within slop. `Press` is the down-edge.
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
        /// Window-physical px (y-down), same space as `Press.screen`.
        /// Games stage this raw and unproject through the live camera
        /// each frame (screen-anchored aim); the `world` point is baked
        /// through the camera at event time and goes stale as the
        /// camera moves.
        screen: [f32; 2],
    },
    /// Touch/pen contact began: pointer id + window-physical px.
    /// Mouse does not emit these; taps still emit `Click` too.
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

/// Touch/pen pointers drive game touch zones;
/// mouse stays on the click/hover path.
/// All `screen` fields below are window-physical px (y-down)
/// (`position_in_window`): region-local `position` would offset picks
/// by the region origin on any non-fullscreen viewport.
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

/// Clamp helper: finite positive values pass through, the rest fall
/// back to 1.0 so framing math holds a valid divisor.
fn sane_positive(v: f32) -> f32 {
    if v.is_finite() && v > 1e-6 { v } else { 1.0 }
}

/// Contain-fit of a `world` extent into a dp-space canvas: uniform
/// scale plus centering offsets. `world_to_dp` / `dp_to_world` invert
/// each other for any positive scale.
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
/// `zoom / units_per_pixel`, re-centered. Default camera matches
/// [`contain_fit`]; zoom 2 doubles on-screen size on both backends.
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

/// GPU camera for the shared framing contract: matrix form of
/// [`world_to_dp`] over `canvas_dp` (dp = physical px / density).
pub fn fit_view_proj(
    canvas_dp: [f32; 2],
    world_size: [f32; 2],
    cam_center: [f32; 2],
    fit: (f32, f32, f32),
) -> Mat4 {
    fit_view_proj_with_roll(canvas_dp, world_size, cam_center, fit, 0.0)
}

/// [`fit_view_proj`] plus camera roll about the look point.
/// `0.0` matches [`fit_view_proj`].
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

/// World point to dp canvas point through the fit. At the default
/// center (`world / 2`) this is `offset + world * scale`.
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

/// Inverse of [`world_to_dp`]. Degenerate scales return the look point.
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

/// Viewport-local physical-px press to world point through the painted
/// [`FrameGeom`]. Takes the region-local position, so HUD chrome or a
/// non-fullscreen viewport does not shift picks.
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

/// One painted frame's geometry, shared between viewports (writers)
/// and world-anchored siblings such as [`ActorFrame`] (readers).
/// Maps either direction so board sprites, picks, and actor
/// surfaces track together, including under camera shake.
///
/// Timing: layout publishes ahead of `prepare` and paint re-publishes;
/// composition-time readers see the previous frame's geometry, so
/// placement trails the board by one frame under camera motion.
/// Event-time readers (picks) see the latest publish.
#[derive(Clone, Copy, Debug)]
pub struct FrameGeom {
    /// Effective dp fit of the world extent: `(scale, off_x, off_y)` from
    /// [`effective_fit`] (contain-fit scaled by zoom/units_per_pixel).
    pub fit: (f32, f32, f32),
    /// Look-point shift in world units: `cam.effective_center() - world / 2`.
    /// Zero when the camera is default.
    pub look: [f32; 2],
    /// Physical px per dp at paint time.
    pub density: f32,
    /// Painted viewport size in physical px (drives the GPU camera).
    pub viewport_px: [f32; 2],
    /// Camera roll in radians around the look point.
    pub roll: f32,
    /// Camera look point in world coords at publish time.
    /// Roll pivot for [`surface_dp`] and picks.
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
/// GPU callbacks hold it behind a mutex (`Send + Sync`).
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
    /// Wrap GPU call sites that hold the raw mutex handle.
    fn from(arc: std::sync::Arc<std::sync::Mutex<FrameGeom>>) -> Self {
        Self(arc)
    }
}

impl From<std::rc::Rc<std::cell::Cell<FrameGeom>>> for GeomHandle {
    /// Wrap canvas call sites holding the old cell handle by snapshotting
    /// the current value into the shared handle.
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

/// World-anchored surface rect in dp: `([off_x, off_y], [w, h])`.
/// `ActorFrame` is its view form. The center rotates with roll about
/// the pivot; size stays axis-aligned.
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

/// 2D viewport view. Gesture handling and camera state live in the
/// [`FrameInput`] snapshot. Fills its parent; draws `background`, then
/// sprites, then world texts, then the fullscreen tint. Each paint
/// publishes [`FrameGeom`] to `geom_out` for dp-space siblings.
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
            let w = ev.position_in_window();
            on_down(PickEvent::Press {
                world: Vec2::new(world[0], world[1]),
                screen: [w.x, w.y],
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
            let w = ev.position_in_window();
            on_move(PickEvent::Hover {
                world: Vec2::new(world[0], world[1]),
                screen: [w.x, w.y],
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
                let w = ev.position_in_window();
                on_up_click(PickEvent::Click {
                    world: Vec2::new(world[0], world[1]),
                    screen: [w.x, w.y],
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
            // Rotation goes through draw_rect_rotated: unrotated quad,
            // rotated in px about the projected center. The rect origin
            // is pivot based (not projected twice) and the angle is
            // roll + rotation. Matches batch::instance_rows.
            let px_per_world = fit.0 * d;
            let [px, py] = project(spr.center.x, spr.center.y);
            let ax = if spr.flip_x {
                1.0 - spr.anchor.x
            } else {
                spr.anchor.x
            };
            let ay = if spr.flip_y {
                1.0 - spr.anchor.y
            } else {
                spr.anchor.y
            };
            scope.draw_rect_rotated(
                Rect {
                    x: px - ax * spr.size.x * px_per_world,
                    y: py - ay * spr.size.y * px_per_world,
                    w: (spr.size.x * px_per_world).max(0.0),
                    h: (spr.size.y * px_per_world).max(0.0),
                },
                rgba8(spr.color),
                Px(0.0),
                cam.roll + spr.rotation,
                repose_core::Vec2 { x: px, y: py },
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

/// GPU twin of [`Viewport2d`]: same snapshot in, same picks out, with
/// sprites as textured atlas quads through [`SpriteBatch`]. Fills its
/// parent. `background` paints first and `overlay_color` last, matching
/// canvas except world texts, which stay canvas-side as sibling views.
///
/// Timing: layout publishes viewport size ahead of `prepare`; paint
/// re-publishes from callback info. `prepare` rebuilds the batch and
/// camera; atlas uploads apply once per frame. `chroma` above `0.0`
/// renders offscreen and composites back (see `post`).
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

/// GPU viewport with explicit batch id so a second viewport or minimap
/// can coexist (each id owns its pipeline and instances).
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
            let w = ev.position_in_window();
            on_down(PickEvent::Press {
                world: Vec2::new(world[0], world[1]),
                screen: [w.x, w.y],
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
            let w = ev.position_in_window();
            on_move(PickEvent::Hover {
                world: Vec2::new(world[0], world[1]),
                screen: [w.x, w.y],
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
                let w = ev.position_in_window();
                on_up_click(PickEvent::Click {
                    world: Vec2::new(world[0], world[1]),
                    screen: [w.x, w.y],
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
        // Background fill under the batch. Skipped when the snapshot has
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

/// Stack GPU sprites under canvas texts/tint from one [`FrameInput`].
/// Mounts `Viewport2dGpu` plus a transparent canvas pass for `texts` /
/// `overlay_color`, so GPU games keep damage numbers and tints.
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
/// World-anchored surface: `child` is boxed to a `size` rect centered
/// on a world `center`, positioned through [`FrameGeom`]. `mirror_x`
/// flips about the surface center. Offsets trail the board by one
/// frame under camera motion; steady state matches.
#[allow(non_snake_case)]
pub fn ActorFrame(
    center: [f32; 2],
    size: [f32; 2],
    geom: GeomHandle,
    mirror_x: bool,
    child: View,
) -> View {
    let ([ox, oy], [w, h]) = surface_dp(center, size, geom.get());
    // `.absolute()` is load-bearing: without it all surfaces stack
    // at one fixed spot.
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
        // 1.25 density: px canvas -> dp fit -> world -> back is identity.
        set_density_default(Density { scale: 1.25 });
        let d = effective_density_scale();
        // 1250x750 px canvas, 800x600 world: dp fit s=1, ox=100, oy=0.
        let fit = contain_fit([1250.0 / d, 750.0 / d], [800.0, 600.0]);
        let cam = [400.0, 300.0];
        // Draw: world (400,300) -> dp -> px lands on the press point.
        let [dx, dy] = world_to_dp([400.0, 300.0], [800.0, 600.0], cam, fit);
        // Pick: window px -> dp -> world maps back.
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
        // Camera shift moves draw and pick together.
        let shifted = world_to_dp([400.0, 300.0], [800.0, 600.0], [410.0, 290.0], fit);
        let back = dp_to_world(shifted, [800.0, 600.0], [410.0, 290.0], fit);
        assert!(
            (back[0] - 400.0).abs() < 1e-4 && (back[1] - 300.0).abs() < 1e-4,
            "shifted round trip, got {back:?}"
        );
    }

    #[test]
    fn gpu_viewport_honors_cam_and_background() {
        // Snapshot camera frames world sprites; `background` fills pixels
        // the quad does not cover. Skips where no GPU exists.
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
        // Actor surfaces share FrameGeom with board sprites: the surface
        // center lands on the board point, including with a shifted look.
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
    fn canvas_rotated_rect_matches_instance_rows() {
        // Canvas draws the mirror-adjusted quad plus a px-space rotation;
        // GPU draws `instance_rows` corners. Both land on the same
        // px corners (compared as sets) across combos.
        let cam = Camera2d {
            center: Vec2::new(400.0, 300.0),
            offset: Vec2::ZERO,
            units_per_pixel: 1.0,
            zoom: 1.0,
            roll: 0.3,
        };
        let world_size = [800.0, 600.0];
        let canvas_dp = [800.0, 600.0];
        let fit = effective_fit(canvas_dp, world_size, &cam);
        let center = cam.effective_center();
        let project = |wx: f32, wy: f32| -> [f32; 2] {
            world_to_dp_with_roll([wx, wy], world_size, center, fit, cam.roll)
        };
        let k = fit.0;
        for (rotation, flip_x, flip_y, anchor) in [
            (0.7f32, false, false, [0.5, 0.5]),
            (0.7, true, false, [0.5, 0.5]),
            (0.7, false, true, [0.5, 0.5]),
            (0.7, true, true, [0.5, 0.5]),
            (-1.2, true, false, [0.0, 0.0]),
            (2.1, false, true, [1.0, 0.25]),
            (0.0, true, false, [0.5, 0.5]),
        ] {
            let size = [64.0f32, 40.0];
            let ctr = [400.0f32, 300.0];
            // GPU reference: instance_rows corners through the projection.
            let (row0, row1) = batch::instance_rows(ctr, size, rotation, anchor, flip_x, flip_y);
            let mut expect = [[0.0f32; 2]; 4];
            for (i, corner) in [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
                .iter()
                .enumerate()
            {
                let wx = row0[0] * corner[0] + row0[1] * corner[1] + row0[3];
                let wy = row1[0] * corner[0] + row1[1] * corner[1] + row1[3];
                expect[i] = project(wx, wy);
            }
            // Canvas: mirror-adjusted rect about the pivot, rotated by
            // roll + rotation. Mirroring is captured by the rect as a
            // set, so the angle does not negate.
            let ax = if flip_x { 1.0 - anchor[0] } else { anchor[0] };
            let ay = if flip_y { 1.0 - anchor[1] } else { anchor[1] };
            let [px, py] = project(ctr[0], ctr[1]);
            let angle = cam.roll + rotation;
            let (c, s) = (angle.cos(), angle.sin());
            let (ox, oy) = (px - ax * size[0] * k, py - ay * size[1] * k);
            let w = size[0] * k;
            let h = size[1] * k;
            for (i, corner) in [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]].iter().enumerate() {
                let rx = px + (ox + corner[0] - px) * c - (oy + corner[1] - py) * s;
                let ry = py + (ox + corner[0] - px) * s + (oy + corner[1] - py) * c;
                let mut best = f32::INFINITY;
                for e in expect {
                    best = best.min((rx - e[0]).hypot(ry - e[1]));
                }
                assert!(
                    best < 1e-2,
                    "rot={rotation} flip=({flip_x},{flip_y}) anchor={anchor:?}: corner {i} ({rx},{ry}) matches none of {expect:?}"
                );
            }
        }
    }

    #[test]
    #[allow(deprecated)]
    fn camera_guards_div_by_zero() {
        let bad = Camera2d {
            center: Vec2::ZERO,
            offset: Vec2::ZERO,
            units_per_pixel: 0.0,
            zoom: 0.0,
            roll: 0.0,
        };
        let m = bad.view_proj([0.0, 0.0]);
        assert!(m.is_finite(), "view_proj stays finite, got {m:?}");
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
        // canvas 1600x900 dp, world 800x800 -> horizontal letterbox
        let cam = Camera2d {
            center: Vec2::new(400.0, 400.0),
            ..Default::default()
        };
        let fit = effective_fit([1600.0, 900.0], [800.0, 800.0], &cam);
        let dp = world_to_dp([400.0, 400.0], [800.0, 800.0], cam.effective_center(), fit);
        let back = dp_to_world(dp, [800.0, 800.0], cam.effective_center(), fit);
        assert!((back[0] - 400.0).abs() < 1e-3 && (back[1] - 400.0).abs() < 1e-3);
        // Method and plain function share the same matrix.
        let m = cam.fit_matrix([1600.0, 900.0], [800.0, 800.0]);
        let plain = fit_view_proj([1600.0, 900.0], [800.0, 800.0], cam.effective_center(), fit);
        let dm = m - plain;
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
