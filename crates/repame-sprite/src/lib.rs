//! Instanced 2D sprite batch: atlas quads in, pixels out.
//!
//! The batch draws [`SpriteInstance`] rows from an atlas through a dp-space
//! [`FitMode`] camera. Placement, z-order and picks live in `repame-actors`;
//! this crate owns the GPU batch, the atlas types, the camera math and the
//! pick event types. Unit discipline lives here, once: games work in
//! world/dp units and do not touch physical px.

use glam::{Mat4, Vec2, Vec3};
use repose_core::Color;

pub mod batch;
pub use batch::{
    AtlasUpload, BatchDesc, BatchFrame, SpriteBatch, TextureFilter, draw_batch, draw_batch_with_id,
    frame_uv, instance_rows, screen_camera, sprite_aabb,
};

pub mod fullscreen;
pub use fullscreen::{
    FullscreenDesc, FullscreenPass, FullscreenTexture, FullscreenTextureRef,
    FullscreenTextureUpload,
};

/// How the world rect is framed when the canvas aspect differs from
/// `world_size`. Named like Godot's `Window.stretch/aspect.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FitMode {
    #[default]
    Keep,
    /// Same scale, world rect anchored at the canvas
    /// origin, so the roomier axis reveals world past `world_size` and
    /// the roomier side of the canvas is not wasted on bars.
    Expand,
}

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
    /// Aspect-fit policy when the canvas and `world_size` disagree.
    /// Defaults to [`FitMode::Keep`].
    pub fit: FitMode,
}

impl Default for Camera2d {
    fn default() -> Self {
        Self {
            center: Vec2::ZERO,
            offset: Vec2::ZERO,
            units_per_pixel: 1.0,
            zoom: 1.0,
            roll: 0.0,
            fit: FitMode::Keep,
        }
    }
}

/// Scroll limits and follow smoothing live in `game_utils_repame::feel`.
/// They run game-side when producing the snapshot; this crate only
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
/// Sorted into ranges (alpha, then multiply, then additive): push order
/// across blend modes is NOT preserved, see the batch module docs.
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
    /// Draw key within one blend range; higher draws on top. Default 0.
    /// Stable-sorted by `z` inside the range, but blend ranges draw
    /// alpha-then-multiply-then-additive regardless of `z` (see
    /// [`SpriteBlend`]).
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
    /// Font family name to resolve, or `None` (default) for the stack the
    /// text engine picks itself. The name must already be known to the
    /// engine; an unresolved name falls back to the default stack.
    pub font_family: Option<&'static str>,
}

pub enum PickEvent {
    Press {
        world: Vec2,
        screen: [f32; 2],
        button: repose_core::input::PointerButton,
    },
    Click {
        world: Vec2,
        screen: [f32; 2],
        button: repose_core::input::PointerButton,
    },
    /// Pointer released, always. [`PickEvent::Click`] is gated behind
    /// [`click_within_slop_dp`], so a drag that ends far from where it started
    /// emits no click at all; a game that drags something has no other way to
    /// learn that the drag ended. Emitted before [`PickEvent::Click`], and
    /// whether or not a click follows.
    Release {
        world: Vec2,
        screen: [f32; 2],
        button: repose_core::input::PointerButton,
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
    TouchDown { id: u64, screen: [f32; 2] },
    /// Touch/pen contact moved (same coordinate space as `TouchDown`).
    TouchMove { id: u64, screen: [f32; 2] },
    /// Touch/pen contact ended (up / leave).
    TouchUp { id: u64 },
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
///
/// The scale is [`FitMode`]-independent — both modes scale uniformly, so
/// neither can crop the world rect. Only the offsets differ: `Keep`
/// centers the world box (bars on the roomier axis), `Expand` anchors it
/// at the origin so the roomier axis shows world past `world_size`.
pub fn effective_fit(canvas_dp: [f32; 2], world: [f32; 2], cam: &Camera2d) -> (f32, f32, f32) {
    let (base, _, _) = contain_fit(canvas_dp, world);
    let zoom = sane_positive(cam.zoom);
    let upp = sane_positive(cam.units_per_pixel);
    let s = base * zoom / upp;
    if !s.is_finite() || s <= 1e-6 {
        return (1.0, 0.0, 0.0);
    }
    let (ox, oy) = match cam.fit {
        FitMode::Keep => (
            (canvas_dp[0] - world[0] * s) * 0.5,
            (canvas_dp[1] - world[1] * s) * 0.5,
        ),
        FitMode::Expand => (0.0, 0.0),
    };
    (s, ox, oy)
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

/// Press slop in dp: pointer-up farther than this from pointer-down
/// cancels the `Click` (drag-off-cancel). Density-independent: compared
/// after dividing physical-px deltas by the painted density, so a tap
/// cancels identically on density 1 and density 3.
pub const CLICK_SLOP_DP: f32 = 12.0;

/// Dp-space slop check: physical-px points divided by `density` (falling
/// back to 1.0) before comparing against [`CLICK_SLOP_DP`].
pub fn click_within_slop_dp(a: [f32; 2], b: [f32; 2], density: f32) -> bool {
    let d = if density.is_finite() && density > 1e-6 {
        density
    } else {
        1.0
    };
    let dx = (a[0] - b[0]) / d;
    let dy = (a[1] - b[1]) / d;
    dx * dx + dy * dy <= CLICK_SLOP_DP * CLICK_SLOP_DP
}

pub fn rgba8(c: [f32; 4]) -> Color {
    Color::from_rgba(
        (c[0].clamp(0.0, 1.0) * 255.0) as u8,
        (c[1].clamp(0.0, 1.0) * 255.0) as u8,
        (c[2].clamp(0.0, 1.0) * 255.0) as u8,
        (c[3].clamp(0.0, 1.0) * 255.0) as u8,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use repose_core::locals::{Density, effective_density_scale, set_density_default};

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
    fn single_layer_atlas_draws_tinted_sprites() {
        // A one-layer atlas must be a plain D2 texture, not a one-element
        // D2Array: GL/ANGLE assumes D2 for it and then samples the array view
        // as black, which silently erases every sprite with no validation
        // error. Guards the shape choice for the layout, the view and the
        // shader together, since they only agree as a set.
        assert_eq!(
            crate::batch::atlas_shape(1),
            (wgpu::TextureViewDimension::D2, false)
        );
        assert_eq!(
            crate::batch::atlas_shape(4),
            (wgpu::TextureViewDimension::D2Array, true)
        );
        assert!(
            crate::batch::sprite_shader(1).contains("texture_2d<")
                && !crate::batch::sprite_shader(1).contains("texture_2d_array"),
            "single-layer shader samples a 2d atlas"
        );
        assert!(
            crate::batch::sprite_shader(4).contains("texture_2d_array"),
            "multi-layer shader samples an array atlas"
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

    /// `Keep` is the pre-`FitMode` framing: centered world box, bars on
    /// the roomier axis, exactly `world` visible.
    #[test]
    fn fit_mode_keep_is_contain_fit() {
        for (canvas, world) in [
            ([1600.0, 900.0], [800.0, 800.0]),
            ([900.0, 1600.0], [800.0, 800.0]),
            ([1000.0, 1000.0], [800.0, 600.0]),
        ] {
            let cam = Camera2d {
                center: Vec2::new(world[0] * 0.5, world[1] * 0.5),
                fit: FitMode::Keep,
                ..Default::default()
            };
            let fit = effective_fit(canvas, world, &cam);
            assert_eq!(fit, contain_fit(canvas, world), "{canvas:?}/{world:?}");
            let (s, ox, oy) = fit;
            // World box lands centered and no larger than the canvas.
            assert!((ox - (canvas[0] - world[0] * s) * 0.5).abs() < 1e-4);
            assert!((oy - (canvas[1] - world[1] * s) * 0.5).abs() < 1e-4);
            assert!(world[0] * s <= canvas[0] + 1e-4 && world[1] * s <= canvas[1] + 1e-4);
        }
    }

    /// `Expand` keeps the same uniform scale but anchors the world box at
    /// the canvas origin, so the roomier axis shows world past `world` and
    /// nothing is ever cropped.
    #[test]
    fn fit_mode_expand_reveals_more_and_never_crops() {
        let canvas = [1600.0, 900.0];
        let world = [800.0, 800.0];
        let cam = Camera2d {
            center: Vec2::new(world[0] * 0.5, world[1] * 0.5),
            fit: FitMode::Expand,
            ..Default::default()
        };
        let (s, ox, oy) = effective_fit(canvas, world, &cam);
        let keep = effective_fit(
            canvas,
            world,
            &Camera2d {
                fit: FitMode::Keep,
                ..cam
            },
        );
        assert_eq!(s, keep.0, "expand must not change the uniform scale");
        assert_eq!((ox, oy), (0.0, 0.0), "expand anchors at the origin");
        // Never crops: the whole world box still lands inside the canvas.
        assert!(world[0] * s <= canvas[0] + 1e-4 && world[1] * s <= canvas[1] + 1e-4);
        // Reveals more on the roomier axis, while the constraining axis
        // still shows exactly `world` (its scale is unchanged).
        let visible = [canvas[0] / s, canvas[1] / s];
        let exact = [
            (visible[0] - world[0]).abs() < 1e-3,
            (visible[1] - world[1]).abs() < 1e-3,
        ];
        assert!(exact[0] ^ exact[1], "one axis exact: {visible:?}");
        if exact[0] {
            assert!(visible[1] > world[1], "y reveals: {visible:?}");
        } else {
            assert!(visible[0] > world[0], "x reveals: {visible:?}");
        }
        // Picks invert the expanded framing like any other.
        let dp = world_to_dp([100.0, 700.0], world, cam.effective_center(), (s, ox, oy));
        let back = dp_to_world(dp, world, cam.effective_center(), (s, ox, oy));
        assert!((back[0] - 100.0).abs() < 1e-3 && (back[1] - 700.0).abs() < 1e-3);
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
            fit: FitMode::Keep,
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
            fit: FitMode::Keep,
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
            fit: FitMode::Keep,
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
            fit: FitMode::Keep,
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
    fn click_slop_cancels_drags() {
        assert!(click_within_slop_dp([100.0, 100.0], [105.0, 105.0], 1.0));
        assert!(click_within_slop_dp([100.0, 100.0], [105.0, 105.0], 1.0));
        assert!(!click_within_slop_dp(
            [100.0, 100.0],
            [100.0 + CLICK_SLOP_DP + 1.0, 100.0],
            1.0
        ));
    }

    #[test]
    fn click_slop_is_density_independent() {
        // 12 dp at density 3 is 36 px: same tap cancels on both densities.
        assert!(click_within_slop_dp([0.0, 0.0], [36.0, 0.0], 3.0));
        assert!(!click_within_slop_dp([0.0, 0.0], [37.0, 0.0], 3.0));
        assert!(click_within_slop_dp([0.0, 0.0], [12.0, 0.0], 1.0));
        assert!(!click_within_slop_dp([0.0, 0.0], [13.0, 0.0], 1.0));
    }

    #[test]
    fn roll_roundtrips_and_moves_pixels() {
        let cam = Camera2d {
            center: Vec2::new(400.0, 300.0),
            offset: Vec2::ZERO,
            units_per_pixel: 1.0,
            zoom: 1.0,
            roll: std::f32::consts::FRAC_PI_2,
            fit: FitMode::Keep,
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
    fn px_round_trips_through_fit() {
        let cam = Camera2d {
            center: Vec2::new(213.0, 120.0),
            offset: Vec2::ZERO,
            units_per_pixel: 1.0 / 3.0,
            zoom: 1.0,
            roll: 0.0,
            fit: FitMode::Keep,
        };
        let world = Vec2::new(100.0, 60.0);
        let viewport_dp = [1024.0, 600.0];
        let world_size = [1024.0, 600.0];
        let (s, ox, oy) = effective_fit(viewport_dp, world_size, &cam);
        let [dx, dy] = world_to_dp(
            [world.x, world.y],
            world_size,
            cam.effective_center(),
            (s, ox, oy),
        );
        let px = Vec2::new(dx * 1.25, dy * 1.25);
        let back = unproject_px(px, viewport_dp, world_size, 1.25, &cam);
        assert!(
            (back - world).length() < 0.01,
            "round trip, got {back:?} for {world:?}"
        );
    }
}

/// One decoded + magnified hardware-cursor frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CursorFrame {
    pub rgba: Vec<u8>,
    pub size: [u16; 2],
    pub hotspot: [u16; 2],
    pub key: (i32, [u32; 3], u64, u32),
}

pub fn cursor_frame(
    strip: &[u8],
    strip_w: u32,
    strip_h: u32,
    frames: u32,
    frame: i32,
    cell: (u32, u32),
    origin: (f32, f32),
    tint: [f32; 4],
    mag: u32,
    strip_hash: u64,
) -> Option<CursorFrame> {
    let frames = frames.max(1) as i32;
    let frame = frame.clamp(0, frames - 1);
    let (cw, ch) = (cell.0.max(1), cell.1.max(1));
    if strip_w < cw * (frame as u32 + 1) || strip_h < ch {
        return None;
    }
    let mag = mag.clamp(1, 8);
    let tint_u8 = [
        (tint[0].clamp(0.0, 1.0) * 255.0) as u32,
        (tint[1].clamp(0.0, 1.0) * 255.0) as u32,
        (tint[2].clamp(0.0, 1.0) * 255.0) as u32,
    ];
    let key = (frame, tint_u8, strip_hash, mag);
    let fx = (frame as u32 * cw) as usize;
    let (dw, dh) = (cw * mag, ch * mag);
    if dw > 2048 || dh > 2048 {
        return None;
    }
    let mut px = Vec::with_capacity((dw * dh * 4) as usize);
    for row in 0..ch {
        let start = ((row * strip_w) as usize + fx) * 4;
        let end = start + (cw as usize) * 4;
        let cell = strip.get(start..end)?;
        let mut mag_row = Vec::with_capacity((cw * mag * 4) as usize);
        for pix in cell.chunks_exact(4) {
            let t = [
                (pix[0] as u32 * tint_u8[0] / 255) as u8,
                (pix[1] as u32 * tint_u8[1] / 255) as u8,
                (pix[2] as u32 * tint_u8[2] / 255) as u8,
                pix[3],
            ];
            for _ in 0..mag {
                mag_row.extend_from_slice(&t);
            }
        }
        for _ in 0..mag {
            px.extend_from_slice(&mag_row);
        }
    }
    let (w16, h16) = (dw.min(2048) as u16, dh.min(2048) as u16);
    Some(CursorFrame {
        rgba: px,
        size: [w16, h16],
        hotspot: [
            (origin.0 * mag as f32)
                .round()
                .clamp(0.0, w16.saturating_sub(1) as f32) as u16,
            (origin.1 * mag as f32)
                .round()
                .clamp(0.0, h16.saturating_sub(1) as f32) as u16,
        ],
        key,
    })
}

/// Unproject one window-physical px point through this frame's camera.
pub fn unproject_px(
    px: Vec2,
    viewport_dp: [f32; 2],
    world_size: [f32; 2],
    density: f32,
    cam: &Camera2d,
) -> Vec2 {
    let d = if density.is_finite() && density > 1e-6 {
        density
    } else {
        1.0
    };
    let dp = [px.x / d, px.y / d];
    cam.dp_to_world_pt(viewport_dp, world_size, dp)
}
