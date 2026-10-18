//! Sprite data: one row per drawn quad, its blend, and its tint.
//!
//! [`SpriteInstance`] is the input to the instanced batch in [`batch`](crate::batch)
//! and the output type of the sim-side particle steppers, so both ends of the
//! snapshot speak it.

use glam::Vec2;
use repose_core::Color;

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

pub fn rgba8(c: [f32; 4]) -> Color {
    Color::from_rgba(
        (c[0].clamp(0.0, 1.0) * 255.0) as u8,
        (c[1].clamp(0.0, 1.0) * 255.0) as u8,
        (c[2].clamp(0.0, 1.0) * 255.0) as u8,
        (c[3].clamp(0.0, 1.0) * 255.0) as u8,
    )
}
