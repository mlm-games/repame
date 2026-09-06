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

use glam::{Mat4, Vec2, Vec3};
use repose_canvas::{Canvas, DrawScope};
use repose_core::locals::effective_density_scale;
use repose_core::{Color, Modifier, Rect, View};

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
    pub fn view_proj(&self, viewport_px: [f32; 2]) -> Mat4 {
        let w = viewport_px[0] * self.units_per_pixel / self.zoom;
        let h = viewport_px[1] * self.units_per_pixel / self.zoom;
        // Right-handed, 0..1 depth: matches wgpu NDC.
        let proj = glam::camera::rh::proj::directx::orthographic(
            -w / 2.0,
            w / 2.0,
            -h / 2.0,
            h / 2.0,
            -1000.0,
            1000.0,
        );
        let view = Mat4::from_translation(Vec3::new(-self.center.x, -self.center.y, 0.0));
        proj * view
    }

    /// World-space position under a viewport-pixel cursor position.
    pub fn screen_to_world(&self, viewport_px: [f32; 2], px: [f32; 2]) -> Vec2 {
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
#[derive(Clone, Copy, Debug, Default)]
pub struct SpriteInstance {
    /// World-space center, rotation radians, size in world units.
    pub center: Vec2,
    pub rotation: f32,
    pub size: Vec2,
    pub uv_min: Vec2,
    pub uv_max: Vec2,
    /// RGBA tint, same byte semantics as the legacy canvas path
    /// (`(c * 255) as u8` per channel).
    pub color: [f32; 4],
    /// Atlas page index for multi-texture batches.
    pub page: u32,
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
#[derive(Clone, Default, Debug)]
pub struct FrameInput {
    pub cam: Camera2d,
    /// World-space extent the contain-fit maps into the viewport.
    /// `Viewport2d` uses `cam.center` (rest/shake included) for the look
    /// point; `viewport_px` / `units_per_pixel` / `zoom` are reserved for
    /// the future wgpu backend and ignored by the canvas renderer.
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
}

/// UI-facing pointer events from the viewport.
#[derive(Clone, Debug)]
pub enum PickEvent {
    Click { world: Vec2, screen: [f32; 2] },
    Hover { world: Vec2 },
}

/// Contain-fit of a `world` extent into a dp-space canvas: uniform scale
/// plus centering offsets, all dp. Pure, so games and tests can pin it.
pub fn contain_fit(canvas_dp: [f32; 2], world: [f32; 2]) -> (f32, f32, f32) {
    if canvas_dp[0] <= 0.0 || canvas_dp[1] <= 0.0 || world[0] <= 0.0 || world[1] <= 0.0 {
        return (1.0, 0.0, 0.0);
    }
    let s = (canvas_dp[0] / world[0]).min(canvas_dp[1] / world[1]);
    let s = s.clamp(0.1, 8.0);
    (
        s,
        (canvas_dp[0] - world[0] * s) * 0.5,
        (canvas_dp[1] - world[1] * s) * 0.5,
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
pub fn dp_to_world(
    dp: [f32; 2],
    world_size: [f32; 2],
    cam_center: [f32; 2],
    fit: (f32, f32, f32),
) -> [f32; 2] {
    let (s, ox, oy) = fit;
    [
        (dp[0] - ox) / s + (cam_center[0] - world_size[0] * 0.5),
        (dp[1] - oy) / s + (cam_center[1] - world_size[1] * 0.5),
    ]
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
/// then world texts, then the fullscreen tint — all through the dp
/// contain-fit of [`FrameInput::world_size`]. Pointer presses are reported
/// via `on_event` in world coords. The dp fit is published to `fit_out`
/// for dp-space siblings (actor surfaces); rig layout and click mapping
/// therefore share one transform by construction.
#[allow(non_snake_case)] // Repose view convention (cf. resims `Viewport3d`).
pub fn Viewport2d(
    input: FrameInput,
    fit_out: Rc<Cell<(f32, f32, f32)>>,
    on_event: impl Fn(PickEvent) + 'static,
) -> View {
    let input = Rc::new(input);
    let world_size = input.world_size;
    let cam_center = [input.cam.center.x, input.cam.center.y];
    // Last painted frame: fit (dp), density. Events map through these so
    // picks agree with what's on screen by construction.
    let painted: Rc<Cell<((f32, f32, f32), f32)>> = Rc::new(Cell::new(((1.0, 0.0, 0.0), 1.0)));
    let pick_state = painted.clone();
    let draw_input = input.clone();

    let modifier = Modifier::new().fill_max_size().on_pointer_down(
        move |ev: repose_core::input::PointerEvent| {
            // Region-local px -> window px (robust to a non-zero viewport
            // origin) -> dp -> world through the painted fit.
            let w = ev.position_in_window();
            let ((s, ox, oy), d) = pick_state.get();
            let dp = [w.x / d, w.y / d];
            let world = dp_to_world(dp, world_size, cam_center, (s, ox, oy));
            on_event(PickEvent::Click {
                world: Vec2::new(world[0], world[1]),
                screen: [w.x, w.y],
            });
        },
    );
    Canvas(modifier, move |scope: &mut DrawScope| {
        // DrawScope size is physical px (layout/paint run in px); the fit
        // is dp (1 world unit == 1 dp). Convert down, publish dp, and
        // scale back up once for drawing.
        let d = effective_density_scale();
        let fit = contain_fit(
            [scope.size.width / d, scope.size.height / d],
            draw_input.world_size,
        );
        fit_out.set(fit);
        painted.set((fit, d));
        let s = fit.0 * d;
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
            let [tx, ty] = project(
                spr.center.x - spr.size.x * 0.5,
                spr.center.y - spr.size.y * 0.5,
            );
            scope.draw_rect(
                Rect {
                    x: tx,
                    y: ty,
                    w: spr.size.x * s,
                    h: spr.size.y * s,
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
                t.size * s,
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
}
