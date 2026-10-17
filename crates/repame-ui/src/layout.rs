//! Design-space placement: the viewport scale every game recomputes, and the
//! anchor/offset rectangle they all place panels with.
//!
//! Two scales come out of one viewport: `fit` (the whole design rectangle is
//! visible, so nothing is cropped) and `cover` (the design rectangle covers the
//! viewport, so nothing is letterboxed). Games use `fit` for HUD and menus and
//! `cover` for overlays that must reach past the design box.
//!
//! [`anchor_rect`] is the games' `rect(vp, s, anchor, offset)`: the anchor is
//! four viewport fractions (left, top, right, bottom) and the offset adds four
//! design-unit distances to those edges before the right/bottom edges are
//! clamped into the viewport. The left edge is not clamped, matching the copies,
//! so an off-screen panel stays off-screen.

use repose_core::prelude::{Color, Modifier, View};
use repose_core::Dp;
use repose_ui::{Box as RBox, Spacer, ViewExt as _};

/// The design box the template line draws at (960 x 540, 16:9).
pub const DESIGN: [f32; 2] = [960.0, 540.0];

/// Why a viewport or design box cannot produce a scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleError {
    /// A side is NaN or infinite. Viewport size is read from the platform, so
    /// this means a divide went wrong upstream.
    NotFinite,
    /// A side is zero or negative: the window has no area yet, or the caller
    /// swapped an argument.
    NotPositive,
}

impl std::fmt::Display for ScaleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScaleError::NotFinite => f.write_str("viewport or design side is not finite"),
            ScaleError::NotPositive => f.write_str("viewport or design side is not positive"),
        }
    }
}

impl std::error::Error for ScaleError {}

/// The two viewport scales, computed once per frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scale {
    fit: f32,
    cover: f32,
}

impl Scale {
    /// Scales for `viewport` (design units on screen) and `design` (the box
    /// the game was drawn in).
    pub fn new(viewport: [f32; 2], design: [f32; 2]) -> Result<Self, ScaleError> {
        for side in viewport.into_iter().chain(design) {
            if !side.is_finite() {
                return Err(ScaleError::NotFinite);
            }
            if side <= 0.0 {
                return Err(ScaleError::NotPositive);
            }
        }
        let fit = (viewport[0] / design[0]).min(viewport[1] / design[1]);
        let cover = (viewport[0] / design[0]).max(viewport[1] / design[1]);
        Ok(Self { fit, cover })
    }

    /// The design box scaled to the design box itself: a neutral fallback for a
    /// viewport that has not been measured yet.
    pub const fn unit() -> Self {
        Self {
            fit: 1.0,
            cover: 1.0,
        }
    }

    /// [`new`](Self::new) for a frame loop: a viewport that is not measured yet
    /// (the first frames, a minimized window) reads as [`unit`](Self::unit)
    /// instead of failing the frame.
    pub fn fit_or_unit(viewport: [f32; 2], design: [f32; 2]) -> Self {
        Self::new(viewport, design).unwrap_or_else(|_| Self::unit())
    }

    /// Everything fits: no cropping.
    pub const fn fit(&self) -> f32 {
        self.fit
    }

    /// Everything is covered: no letterboxing.
    pub const fn cover(&self) -> f32 {
        self.cover
    }

    /// Scale a design-unit measurement for the `fit` scale.
    pub fn fit_units(&self, units: f32) -> f32 {
        units * self.fit
    }

    /// Scale a design-unit measurement for the `cover` scale.
    pub fn cover_units(&self, units: f32) -> f32 {
        units * self.cover
    }
}

impl Default for Scale {
    fn default() -> Self {
        Self::unit()
    }
}

/// The four design-space edges [`anchor_rect`] produces.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnchorEdges {
    pub left: f32,
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
}

/// Resolve an anchor and offset against a viewport.
///
/// `anchor` is `[left, top, right, bottom]` as fractions of the viewport;
/// `offset` is `[left, top, right, bottom]` in design units, scaled by `scale`.
/// Only the right and bottom edges are clamped into the viewport.
pub fn anchor_edges(
    viewport: [f32; 2],
    scale: f32,
    anchor: [f32; 4],
    offset: [f32; 4],
) -> AnchorEdges {
    let left = anchor[0] * viewport[0] + offset[0] * scale;
    let top = anchor[1] * viewport[1] + offset[1] * scale;
    let right = anchor[2] * viewport[0] + offset[2] * scale;
    let bottom = anchor[3] * viewport[1] + offset[3] * scale;
    AnchorEdges {
        left,
        top,
        right: (viewport[0] - right).max(0.0),
        bottom: (viewport[1] - bottom).max(0.0),
    }
}

/// An absolutely positioned, full-bleed helper box: the games' `rect(...)`.
///
/// The returned modifier carries no paint or input behaviour, so the caller
/// adds a background, a border, a click handler or `hit_passthrough()`.
pub fn anchor_rect(viewport: [f32; 2], scale: f32, anchor: [f32; 4], offset: [f32; 4]) -> Modifier {
    let edges = anchor_edges(viewport, scale, anchor, offset);
    Modifier::new()
        .absolute()
        .offset_left(Dp(edges.left))
        .offset_top(Dp(edges.top))
        .offset_right(Dp(edges.right))
        .offset_bottom(Dp(edges.bottom))
}

/// A filled circle at a design-space point, with a design-unit radius.
pub fn disc(viewport: [f32; 2], scale: f32, at: [f32; 2], radius: f32, color: Color) -> View {
    RBox(
        anchor_rect(
            viewport,
            scale,
            [0.0; 4],
            [
                at[0] - radius,
                at[1] - radius,
                at[0] + radius,
                at[1] + radius,
            ],
        )
        .background(color)
        .clip_rounded(Dp(radius * scale))
        .hit_passthrough(),
    )
    .child(Spacer())
}

/// An unfilled circle: the virtual stick's ring.
pub fn ring(
    viewport: [f32; 2],
    scale: f32,
    at: [f32; 2],
    radius: f32,
    width: f32,
    color: Color,
) -> View {
    RBox(
        anchor_rect(
            viewport,
            scale,
            [0.0; 4],
            [
                at[0] - radius,
                at[1] - radius,
                at[0] + radius,
                at[1] + radius,
            ],
        )
        .border(Dp(width * scale), color, Dp(radius * scale))
        .background(Color::TRANSPARENT)
        .hit_passthrough(),
    )
    .child(Spacer())
}

/// Pointer position in physical pixels to design units.
///
/// The platform reports pointers in physical pixels and layout in density
/// pixels; this is the division the games do before comparing a pointer against
/// design coordinates. `scale` is the frame's [`Scale::fit`].
pub fn design_units(pixels: repose_core::Vec2, scale: f32) -> glam::Vec2 {
    let density = repose_core::locals::effective_density_scale().max(1e-6);
    let factor = (density * scale.max(1e-6)).max(1e-6);
    glam::Vec2::new(pixels.x, pixels.y) / factor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_is_the_smaller_ratio_and_cover_the_larger() {
        let scale = Scale::new([1280.0, 720.0], DESIGN).expect("16:9 viewport scales");
        assert!((scale.fit() - 1280.0 / 960.0).abs() < 1e-6);
        assert!((scale.cover() - 1280.0 / 960.0).abs() < 1e-6);

        let tall = Scale::new([600.0, 900.0], DESIGN).expect("4:3-ish viewport scales");
        assert!((tall.fit() - 600.0 / 960.0).abs() < 1e-6);
        assert!((tall.cover() - 900.0 / 540.0).abs() < 1e-6);
        assert!(tall.cover() > tall.fit());
    }

    #[test]
    fn unit_scales_are_neutral() {
        let unit = Scale::unit();
        assert_eq!(unit.fit_units(46.0), 46.0);
        assert_eq!(unit.cover_units(46.0), 46.0);
        assert_eq!(Scale::fit_or_unit([0.0, 0.0], DESIGN), Scale::unit());
        assert_eq!(
            Scale::fit_or_unit([960.0, 540.0], DESIGN),
            Scale::new([960.0, 540.0], DESIGN).expect("scales")
        );
    }

    #[test]
    fn a_zero_sized_viewport_is_rejected_until_it_is_measured() {
        assert_eq!(Scale::new([0.0, 720.0], DESIGN), Err(ScaleError::NotPositive));
        assert_eq!(Scale::new([1280.0, -1.0], DESIGN), Err(ScaleError::NotPositive));
        assert_eq!(Scale::new([1280.0, 720.0], [960.0, 0.0]), Err(ScaleError::NotPositive));
    }

    #[test]
    fn a_non_finite_viewport_is_rejected() {
        assert_eq!(Scale::new([f32::NAN, 720.0], DESIGN), Err(ScaleError::NotFinite));
        assert_eq!(
            Scale::new([1280.0, 720.0], [f32::INFINITY, 540.0]),
            Err(ScaleError::NotFinite)
        );
    }

    #[test]
    fn anchor_edges_follow_the_viewport_fractions_and_offsets() {
        let edges = anchor_edges([1000.0, 500.0], 2.0, [0.5, 0.5, 0.5, 0.5], [-100.0, -50.0, 100.0, 50.0]);
        assert_eq!(edges.left, 500.0 - 200.0);
        assert_eq!(edges.top, 250.0 - 100.0);
        assert_eq!(edges.right, 1000.0 - 700.0);
        assert_eq!(edges.bottom, 500.0 - 350.0);
    }

    #[test]
    fn right_and_bottom_edges_clamp_into_the_viewport_left_does_not() {
        let edges = anchor_edges([1000.0, 500.0], 1.0, [0.0, 0.0, 2.0, 2.0], [-40.0, 0.0, 0.0, 0.0]);
        assert_eq!(edges.left, -40.0, "an off-screen left edge stays off-screen");
        assert_eq!(edges.right, 0.0);
        assert_eq!(edges.bottom, 0.0);
    }

    #[test]
    fn a_full_bleed_anchor_covers_the_viewport() {
        let edges = anchor_edges([1000.0, 500.0], 3.0, [0.0, 0.0, 1.0, 1.0], [0.0; 4]);
        assert_eq!((edges.left, edges.top), (0.0, 0.0));
        assert_eq!((edges.right, edges.bottom), (0.0, 0.0));
    }
}
