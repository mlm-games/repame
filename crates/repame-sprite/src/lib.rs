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
use repose_core::{Color, Modifier, Rect, View};
use repose_render_wgpu::{Callback, CallbackResources, ScreenDescriptor, WgpuCallback};
use repose_ui::Box as UiBox;
use repose_ui::ViewExt;

pub mod batch;
use batch::draw_batch;
pub use batch::{AtlasUpload, BatchDesc, SpriteBatch, TextureFilter, instance_rows, screen_camera};

pub mod fullscreen;
pub use fullscreen::{FullscreenDesc, FullscreenPass, FullscreenTexture};

pub mod post;

/// 2D orthographic camera. Owned by the UI (Repose signal), copied into the
/// snapshot per frame.
#[derive(Clone, Copy, Debug)]
pub struct Camera2d {
    /// World-space center the camera looks at.
    pub center: Vec2,
    /// World units per screen pixel at zoom 1. Combined with viewport size
    /// to build the ortho projection.
    pub units_per_pixel: f32,
    pub zoom: f32,
}

impl Default for Camera2d {
    fn default() -> Self {
        Self {
            center: Vec2::ZERO,
            units_per_pixel: 1.0,
            zoom: 1.0,
        }
    }
}

impl Camera2d {
    /// Y-down view-projection (Godot/canvas convention: world +y points
    /// screen-down). Guards degenerate inputs (zero viewport/zoom) so a
    /// bad snapshot can never div-by-zero; callers driving viewports
    /// should prefer [`fit_view_proj`] (contain-fit + look), which is the
    /// single framing contract canvas, GPU, and picks share.
    pub fn view_proj(&self, viewport_px: [f32; 2]) -> Mat4 {
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
        let view = Mat4::from_translation(Vec3::new(-self.center.x, -self.center.y, 0.0));
        proj * view
    }

    /// World-space position under a viewport-pixel cursor position.
    /// Guards zero viewports (returns the look point instead of NaN).
    pub fn screen_to_world(&self, viewport_px: [f32; 2], px: [f32; 2]) -> Vec2 {
        if viewport_px[0] <= 0.0 || viewport_px[1] <= 0.0 {
            return self.center;
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
/// [`fit_view_proj`] / [`world_to_dp`]. `cam.center` is the look point
/// (rest/shake included); `cam.zoom` / `cam.units_per_pixel` scale the
/// fit uniformly on every backend.
#[derive(Clone, Default, Debug)]
pub struct FrameInput {
    pub cam: Camera2d,
    /// World-space extent the contain-fit maps into the viewport.
    pub world_size: [f32; 2],
    pub viewport_px: [f32; 2],
    pub sprites: Vec<SpriteInstance>,
    /// World-anchored text, drawn after the sprite pass.
    pub texts: Vec<WorldText>,
    /// Full-viewport fill under everything (letterbox included).
    pub background: Option<[f32; 4]>,
    /// Optional fullscreen tint/color-grading hook (e.g. FOW dimming,
    /// damage flash). Applied after the sprite pass.
    pub overlay_color: Option<[f32; 4]>,
    /// Chromatic aberration amount (bevy `chromatic_intensity` units;
    /// NT pulses land at 0.04..0.7). GPU viewports render the scene
    /// offscreen and composite it back with an RGB split; `0.0` keeps
    /// the zero-cost direct path. Canvas viewports ignore it (sampling
    /// FX need pixels, and the canvas path is vector commands).
    pub chroma: f32,
}

/// UI-facing pointer events from the viewport.
#[derive(Clone, Debug)]
pub enum PickEvent {
    Click { world: Vec2, screen: [f32; 2] },
    Hover { world: Vec2 },
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
/// via `dp_to_world` — one transform, three consumers.
pub fn fit_view_proj(
    canvas_dp: [f32; 2],
    world_size: [f32; 2],
    cam_center: [f32; 2],
    fit: (f32, f32, f32),
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
    Mat4::from_cols(
        glam::Vec4::new(sx, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, sy, 0.0, 0.0),
        glam::Vec4::new(0.0, 0.0, 0.001, 0.0),
        glam::Vec4::new(tx, ty, 0.0, 1.0),
    )
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
    let (s, ox, oy) = fit;
    [
        ox + (world[0] - (cam_center[0] - world_size[0] * 0.5)) * s,
        oy + (world[1] - (cam_center[1] - world_size[1] * 0.5)) * s,
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
    let (s, ox, oy) = fit;
    if !s.is_finite() || s.abs() < 1e-6 {
        return cam_center;
    }
    [
        (dp[0] - ox) / s + (cam_center[0] - world_size[0] * 0.5),
        (dp[1] - oy) / s + (cam_center[1] - world_size[1] * 0.5),
    ]
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
    dp_to_world(
        [local_px[0] / d, local_px[1] / d],
        world_size,
        [
            geom.look[0] + world_size[0] * 0.5,
            geom.look[1] + world_size[1] * 0.5,
        ],
        geom.fit,
    )
}

/// One painted frame's geometry, shared between [`Viewport2d`] /
/// [`Viewport2dGpu`] (writers) and world-anchored siblings like
/// [`ActorFrame`] (readers). Everything needed to map either direction,
/// so board sprites, picks, and actor surfaces stay glued - including
/// under camera shake.
#[derive(Clone, Copy, Debug)]
pub struct FrameGeom {
    /// Effective dp fit of the world extent: `(scale, off_x, off_y)` from
    /// [`effective_fit`] (contain-fit scaled by zoom/units_per_pixel).
    pub fit: (f32, f32, f32),
    /// Look-point shift in world units: `cam.center - world / 2`
    /// (trauma shake included; zero when the camera is default).
    pub look: [f32; 2],
    /// Physical px per dp at paint time.
    pub density: f32,
    /// Painted viewport size in physical px (drives the GPU camera).
    pub viewport_px: [f32; 2],
}

impl Default for FrameGeom {
    fn default() -> Self {
        Self {
            fit: (1.0, 0.0, 0.0),
            look: [0.0, 0.0],
            density: 1.0,
            viewport_px: [1.0, 1.0],
        }
    }
}

/// World-anchored surface rect in dp: `([off_x, off_y], [w, h])` for a
/// `size` box centered on `center`. Pure; `ActorFrame` is its view form.
pub fn surface_dp(center: [f32; 2], size: [f32; 2], geom: FrameGeom) -> ([f32; 2], [f32; 2]) {
    let (s, ox, oy) = geom.fit;
    (
        [
            ox + (center[0] - size[0] * 0.5 - geom.look[0]) * s,
            oy + (center[1] - size[1] * 0.5 - geom.look[1]) * s,
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
/// are reported via `on_event` in world coords (viewport-local, so a
/// non-zero viewport origin never shifts picks). Each paint publishes a
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
    geom_out: Rc<Cell<FrameGeom>>,
    on_event: impl Fn(PickEvent) + 'static,
) -> View {
    let input = Rc::new(input);
    let world_size = input.world_size;
    let pick_geom = geom_out.clone();
    let draw_input = input.clone();
    let draw_geom = geom_out.clone();

    let modifier = Modifier::new().fill_max_size().on_pointer_down(
        move |ev: repose_core::input::PointerEvent| {
            let p = ev.position;
            let g = pick_geom.get();
            let world = pick_world([p.x, p.y], g, world_size);
            on_event(PickEvent::Click {
                world: Vec2::new(world[0], world[1]),
                screen: [p.x, p.y],
            });
        },
    );
    Canvas(modifier, move |scope: &mut DrawScope| {
        let d = effective_density_scale();
        let cam = draw_input.cam;
        let cam_center = [cam.center.x, cam.center.y];
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
        });
        let project = |wx: f32, wy: f32| -> [f32; 2] {
            let [dx, dy] = world_to_dp(
                [wx, wy],
                draw_input.world_size,
                cam_center,
                (fit.0, fit.1, fit.2),
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
                0.0,
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
                0.0,
            );
        }
        for t in draw_input.texts.iter() {
            let [tx, ty] = project(t.pos.x, t.pos.y);
            scope.draw_text(
                t.text.clone(),
                repose_core::Vec2 { x: tx, y: ty },
                rgba8(t.color),
                t.size * fit.0 * d,
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
                0.0,
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
/// as a uniform-only fullscreen fill so GPU and canvas agree on every
/// `FrameInput` field except world texts and tint, which stay canvas-view
/// features for now: GPU consumers compose those as sibling views.
///
/// Layout contract: fills its parent. The payload is rebuilt from the
/// snapshot every frame (like the canvas draw closure): `prepare`
/// rebuilds the batch, and camera from the last painted viewport (cold
/// start falls back to [`FrameInput::viewport_px`]), and `paint`
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
    geom_out: Arc<Mutex<FrameGeom>>,
    uploads: Vec<AtlasUpload>,
    desc: BatchDesc,
    on_event: impl Fn(PickEvent) + 'static,
) -> View {
    let input = Arc::new(input);
    let world_size = input.world_size;
    let pick_geom = geom_out.clone();
    let payload = GpuViewport {
        input: input.clone(),
        geom: geom_out,
        uploads,
        desc,
        bg: FullscreenPass::new(
            "viewport2d.background",
            fullscreen::SOLID_WGSL,
            FullscreenDesc {
                texture_slots: 0,
                filter: TextureFilter::Nearest,
            },
        ),
    };
    let modifier = Modifier::new().fill_max_size().on_pointer_down(
        move |ev: repose_core::input::PointerEvent| {
            let p = ev.position;
            let Ok(g) = pick_geom.lock() else {
                return;
            };
            let world = pick_world([p.x, p.y], *g, world_size);
            on_event(PickEvent::Click {
                world: Vec2::new(world[0], world[1]),
                screen: [p.x, p.y],
            });
        },
    );
    Embedded(modifier, Callback::new(payload))
}

struct GpuViewport {
    input: Arc<FrameInput>,
    geom: Arc<Mutex<FrameGeom>>,
    uploads: Vec<AtlasUpload>,
    desc: BatchDesc,
    bg: FullscreenPass,
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
            self.input.viewport_px
        };
        let fit = effective_fit(dp, self.input.world_size, &self.input.cam);
        let cam_center = [self.input.cam.center.x, self.input.cam.center.y];
        let mut batch = SpriteBatch::new(self.desc);
        batch.set_camera(fit_view_proj(dp, self.input.world_size, cam_center, fit));
        for s in &self.input.sprites {
            batch.push_sprite(s);
        }
        batch.extend_uploads(self.uploads.clone());
        batch.prepare(device, queue, encoder, screen, resources);
        // Background first (uniform-only fill, same color the canvas
        // path fills under the batch). Skipped when the snapshot has
        // none so transparent scenes keep compositing.
        if let Some(bg) = self.input.background {
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
        if post::use_composite(self.input.chroma) {
            let w = vp_phys[0].max(1.0) as u32;
            let h = vp_phys[1].max(1.0) as u32;
            post::prepare_composite(
                device,
                queue,
                encoder,
                screen,
                resources,
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
        let cam = [self.input.cam.center.x, self.input.cam.center.y];
        if let Ok(mut g) = self.geom.lock() {
            *g = FrameGeom {
                fit,
                look: [
                    cam[0] - self.input.world_size[0] * 0.5,
                    cam[1] - self.input.world_size[1] * 0.5,
                ],
                density: d,
                viewport_px: vp,
            };
        }
        if post::use_composite(self.input.chroma) {
            post::paint_composite(rpass, resources);
        } else {
            // Snapshot background first (opaque uniform fill), then the
            // sprite batch on top: same order as the canvas path.
            if self.input.background.is_some() {
                self.bg.paint(info, rpass, resources);
            }
            draw_batch(rpass, resources);
        }
    }
}

/// A world-anchored surface: `child` (painting in its own local coords)/// is boxed to a `size` rect centered on a world `center`, positioned
/// through the viewport's [`FrameGeom`]. Siblings of [`Viewport2d`] -
/// e.g. live actor surfaces - track sim positions without game-side unit
/// math, glued to sprites and picks even under camera shake. `mirror_x`
/// flips around the surface center (left-walking hordes reusing
/// right-facing art).
#[allow(non_snake_case)]
pub fn ActorFrame(
    center: [f32; 2],
    size: [f32; 2],
    geom: Rc<Cell<FrameGeom>>,
    mirror_x: bool,
    child: View,
) -> View {
    let ([ox, oy], [w, h]) = surface_dp(center, size, geom.get());
    // `.absolute()` is load-bearing: without it taffy ignores the offsets
    // and every surface stacks at the same fixed spot.
    let mut modifier = Modifier::new()
        .size(w, h)
        .absolute()
        .offset(Some(ox), Some(oy), None, None);
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
            units_per_pixel: 0.5,
            zoom: 1.0,
        };
        let input = FrameInput {
            cam,
            world_size: [128.0, 128.0],
            viewport_px: [256.0, 256.0],
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
        }));
        let payload = GpuViewport {
            input: Arc::new(input),
            geom,
            uploads,
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
        // Screen center carries the cam-centered sprite (magenta page).
        assert_eq!(at(128, 128), [255, 0, 255, 255], "cam-centered quad");
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
    fn camera_guards_never_div_by_zero() {
        let bad = Camera2d {
            center: Vec2::ZERO,
            units_per_pixel: 0.0,
            zoom: 0.0,
        };
        let m = bad.view_proj([0.0, 0.0]);
        assert!(m.is_finite(), "view_proj must stay finite, got {m:?}");
        assert_eq!(bad.screen_to_world([0.0, 0.0], [5.0, 5.0]), Vec2::ZERO);
    }

    #[test]
    fn fit_matrix_canvas_gpu_picks_agree() {
        let cam = Camera2d {
            center: Vec2::new(410.0, 290.0),
            units_per_pixel: 1.0,
            zoom: 2.0,
        };
        let world_size = [800.0, 600.0];
        let density = 1.25;
        let vp_phys = [1250.0, 750.0];
        let canvas_dp = [vp_phys[0] / density, vp_phys[1] / density];
        let cam_center = [cam.center.x, cam.center.y];
        let fit = effective_fit(canvas_dp, world_size, &cam);
        assert!((fit.0 - 2.0).abs() < 1e-4, "effective fit, got {fit:?}");
        let vp = fit_view_proj(canvas_dp, world_size, cam_center, fit);
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
            let back = dp_to_world(
                [px[0] / density, px[1] / density],
                world_size,
                cam_center,
                fit,
            );
            assert!(
                (back[0] - world[0]).abs() < 1e-3 && (back[1] - world[1]).abs() < 1e-3,
                "pick round trip for {world:?} gave {back:?}"
            );
        }
    }

    #[test]
    fn picks_ignore_viewport_origin() {
        let geom = FrameGeom {
            fit: (1.0, 100.0, 0.0),
            look: [0.0, 0.0],
            density: 1.25,
            viewport_px: [1250.0, 750.0],
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
}
