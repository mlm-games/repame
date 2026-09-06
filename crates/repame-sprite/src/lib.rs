//! 2D sprite viewport: snapshot in, pixels out.
//!
//! The UI builds a [`FrameInput`] per frame (plain data, cheap to rebuild
//! during composition) and mounts [`Viewport2d`] as a Repose view. The
//! renderer consumes the snapshot inside a `repose_render_wgpu::Callback`.
//! Camera state lives in Repose signals, never in the renderer.

use glam::{Mat4, Vec2, Vec3};
use repose_core::View;

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
    /// Linear-space RGBA tint.
    pub color: [f32; 4],
    /// Atlas page index for multi-texture batches.
    pub page: u32,
}

/// Everything the viewport draws this frame. Plain data, snapshot per frame.
#[derive(Clone, Default, Debug)]
pub struct FrameInput {
    pub cam: Camera2d,
    pub viewport_px: [f32; 2],
    pub sprites: Vec<SpriteInstance>,
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

/// 2D viewport view. Owns nothing render-side; gesture handling and camera
/// state live in Repose signals in the calling crate (same split as
/// resims `Viewport3d`). Full renderer implementation lands with the rozvp
/// pilot; this scaffold fixes the snapshot boundary first.
#[allow(non_snake_case)] // Repose view convention (cf. resims `Viewport3d`).
pub fn Viewport2d(input: FrameInput, _on_event: impl Fn(PickEvent) + 'static) -> View {
    let _ = input;
    todo!("viewport renderer lands with the rozvp pilot")
}
