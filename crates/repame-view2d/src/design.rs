//! Design-space conversion: a design box onto a dp canvas. This is the scale
//! games hand-roll as `grid_scale` / `ui_scale` / `world_scale`, the
//! fill-and-crop [`cover_fit`] beside the existing contain fit, and the
//! `field_size` view of how much design space a canvas shows.
//!
//! Units: the canvas is dp (physical px divided by density, the space
//! [`contain_fit`](crate::contain_fit) and
//! [`effective_fit`](crate::effective_fit) already work in) and design units
//! are the game's authored grid (`[960, 540]` in the template). A
//! [`DesignMapping`] reports dp per design unit: multiply a design
//! measurement by [`DesignMapping::scale`] to draw it. Design coordinates run
//! from the design box's top-left, y down, matching rig coordinates and the
//! default centered camera, so [`DesignMapping::design_to_dp`] equals
//! [`world_to_dp`](crate::world_to_dp) with `cam_center == design / 2`.
//!
//! Policies: [`DesignFit::Contain`] keeps the whole design box on screen and
//! bars the roomier axis, [`DesignFit::Cover`] fills the canvas and crops the
//! overflow. Both scales are uniform, so neither can change the aspect.
//! [`DesignMapping::fit`] gives the `(scale, offset_x, offset_y)` tuple the
//! framing functions take, so one mapping feeds canvas, GPU batch and picks
//! unchanged: hand it to
//! [`fit_view_proj_with_roll`](crate::fit_view_proj_with_roll),
//! [`world_to_dp`](crate::world_to_dp) and [`dp_to_world`](crate::dp_to_world).
//! This adds no camera mode: [`Camera2d::fit_matrix`](crate::Camera2d::fit_matrix)
//! keeps its contain contract and [`FitMode`](crate::FitMode) its anchoring,
//! and cover framing is the same functions called with the cover tuple.
//!
//! Rounding: the scale is never rounded, and neither is a single design
//! coordinate. One scale and one offset pair per frame, one multiply per
//! conversion, so two elements that share a design-space edge share the
//! converted edge exactly, and dp is density-independent, so the result is
//! identical at every density. Content that must land on whole dp is snapped
//! once, by [`DesignMapping::snap_box`], which rounds both corners and derives
//! the extent from them; rounding a position and a size independently
//! (`round(x) + round(w)`) is what opens the 1 dp seam, and repeating that
//! seam across a grid or a text line is the half-pixel cascade.

use crate::contain_fit;

/// Which ratio maps the design box onto the canvas. The anchoring question
/// ("where does the box sit once the scale is picked") stays with
/// [`FitMode`](crate::FitMode); this is only the ratio.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DesignFit {
    /// The smaller ratio: the whole design box is visible, bars on the roomier
    /// axis. The games' `grid_scale` / `ui_scale`, and
    /// [`contain_fit`](crate::contain_fit).
    #[default]
    Contain,
    /// The larger ratio: the canvas is filled and the overflowing axis is
    /// cropped. The games' `world_scale` and `cover_box`, and [`cover_fit`].
    Cover,
}

/// Cover-fit of a `design` extent onto a dp canvas: uniform scale plus
/// centering offsets, the fill-and-crop sibling of
/// [`contain_fit`](crate::contain_fit). The design box is scaled up until it
/// covers the canvas, so the offsets are zero on the tight axis and negative
/// on the cropped one. `design` is the box `contain_fit` calls `world`;
/// degenerate sides read as `(1.0, 0.0, 0.0)` like `contain_fit`.
pub fn cover_fit(canvas_dp: [f32; 2], design: [f32; 2]) -> (f32, f32, f32) {
    if canvas_dp[0] <= 0.0 || canvas_dp[1] <= 0.0 || design[0] <= 0.0 || design[1] <= 0.0 {
        return (1.0, 0.0, 0.0);
    }
    let s = (canvas_dp[0] / design[0]).max(canvas_dp[1] / design[1]);
    if !s.is_finite() || s <= 1e-6 {
        return (1.0, 0.0, 0.0);
    }
    (
        s,
        (canvas_dp[0] - design[0] * s) * 0.5,
        (canvas_dp[1] - design[1] * s) * 0.5,
    )
}

/// A design box mapped onto a dp canvas under one [`DesignFit`] policy: the
/// scale, the offsets it implies, the inverse, and the snapping rule. Build it
/// per frame, or on resize, from the measured canvas and the game's design
/// size.
///
/// Degenerate sides (`<= 0.0`: the state before the first measure) fall back
/// exactly like [`contain_fit`] / [`cover_fit`]: scale `1.0`, offsets `0.0`,
/// so the mapping is the identity on design units and `visible` reports that
/// canvas over `1.0`. Sides so far apart that the ratio underflows past the
/// shared `1e-6` guard read the same way. A non-finite side is not repaired:
/// it surfaces in the offsets rather than being replaced by an invented
/// number.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DesignMapping {
    /// Design box in design units.
    pub design: [f32; 2],
    /// Measured canvas in dp.
    pub canvas_dp: [f32; 2],
    /// Which ratio wins.
    pub policy: DesignFit,
}

impl DesignMapping {
    /// Contain, the design box fully visible. Canvas first, matching
    /// [`contain_fit`](crate::contain_fit).
    pub fn contain(canvas_dp: [f32; 2], design: [f32; 2]) -> Self {
        Self {
            design,
            canvas_dp,
            policy: DesignFit::Contain,
        }
    }

    /// Cover, the canvas fully filled and the overflow cropped. Canvas first,
    /// matching [`cover_fit`].
    pub fn cover(canvas_dp: [f32; 2], design: [f32; 2]) -> Self {
        Self {
            design,
            canvas_dp,
            policy: DesignFit::Cover,
        }
    }

    /// dp per design unit for this policy: the games' `grid_scale` /
    /// `ui_scale` under [`DesignFit::Contain`], `world_scale` under
    /// [`DesignFit::Cover`].
    pub fn scale(&self) -> f32 {
        match self.policy {
            DesignFit::Contain => contain_fit(self.canvas_dp, self.design).0,
            DesignFit::Cover => cover_fit(self.canvas_dp, self.design).0,
        }
    }

    /// The `(scale, offset_x, offset_y)` tuple the framing functions take. With
    /// the default centered camera the design origin lands at the offsets, so
    /// this tuple feeds [`fit_view_proj_with_roll`](crate::fit_view_proj_with_roll),
    /// [`world_to_dp`](crate::world_to_dp) and picks alike; under
    /// [`DesignFit::Cover`] the offsets are negative on the cropped axis.
    pub fn fit(&self) -> (f32, f32, f32) {
        match self.policy {
            DesignFit::Contain => contain_fit(self.canvas_dp, self.design),
            DesignFit::Cover => cover_fit(self.canvas_dp, self.design),
        }
    }

    /// The design box scaled to dp at [`scale`](Self::scale): the games'
    /// `cover_box` under [`DesignFit::Cover`], whose top-left sits at the
    /// offsets [`fit`](Self::fit) reports.
    pub fn box_dp(&self) -> [f32; 2] {
        let s = self.scale();
        [self.design[0] * s, self.design[1] * s]
    }

    /// Design units the canvas shows, the games' `field_size`. For a measured
    /// canvas under [`DesignFit::Contain`] this is the design box or more (the
    /// roomier axis reveals past it); under [`DesignFit::Cover`] it is the
    /// design sub-rect that survives cropping, so content that must stay
    /// visible belongs inside it.
    pub fn visible(&self) -> [f32; 2] {
        let s = self.scale();
        [self.canvas_dp[0] / s, self.canvas_dp[1] / s]
    }

    /// Design units to dp through the fit: `offset + design * scale`, no
    /// rounding.
    pub fn design_to_dp(&self, design: [f32; 2]) -> [f32; 2] {
        let (s, ox, oy) = self.fit();
        [ox + design[0] * s, oy + design[1] * s]
    }

    /// Inverse of [`design_to_dp`](Self::design_to_dp). Exact up to float
    /// error for any scale the fits can report (they never report zero).
    pub fn dp_to_design(&self, dp: [f32; 2]) -> [f32; 2] {
        let (s, ox, oy) = self.fit();
        [(dp[0] - ox) / s, (dp[1] - oy) / s]
    }

    /// Snap a design box onto whole dp: round the converted top-left and
    /// bottom-right corners, then derive the dp extent from the two snapped
    /// corners. Two boxes that share a design-space edge therefore share the
    /// snapped edge exactly, and the extent can never disagree with its own
    /// corners (`round(x) + round(w)` can, by up to a dp per seam). Opt in:
    /// this is for art that must land on whole dp, not for moving content.
    pub fn snap_box(&self, at: [f32; 2], size: [f32; 2]) -> ([f32; 2], [f32; 2]) {
        let a = self.design_to_dp(at);
        let b = self.design_to_dp([at[0] + size[0], at[1] + size[1]]);
        let (ax, ay) = (a[0].round(), a[1].round());
        let (bx, by) = (b[0].round(), b[1].round());
        ([ax, ay], [bx - ax, by - ay])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `[960, 540]`, the template line.
    const DESIGN: [f32; 2] = [960.0, 540.0];

    /// Both policies keep the design aspect at a non-integer ratio; contain
    /// keeps the box on screen, cover makes it cover the canvas.
    #[test]
    fn contain_and_cover_preserve_aspect_at_a_non_integer_ratio() {
        let canvas = [1000.0, 747.0];
        let contain = DesignMapping::contain(canvas, DESIGN);
        let cover = DesignMapping::cover(canvas, DESIGN);
        for m in [contain, cover] {
            let scaled = m.box_dp();
            let aspect = scaled[0] / scaled[1];
            assert!(
                (aspect - DESIGN[0] / DESIGN[1]).abs() < 1e-4,
                "{:?} changed the aspect: {scaled:?}",
                m.policy
            );
        }
        let fitted = contain.box_dp();
        assert!(
            fitted[0] <= canvas[0] + 1e-4 && fitted[1] <= canvas[1] + 1e-4,
            "contain must not crop: {fitted:?}"
        );
        assert_eq!(contain.fit(), contain_fit(canvas, DESIGN));
        let filled = cover.box_dp();
        assert!(
            filled[0] >= canvas[0] - 1e-4 && filled[1] >= canvas[1] - 1e-4,
            "cover must not letterbox: {filled:?}"
        );
        assert!(cover.scale() > contain.scale(), "cover scales up");
        assert_eq!(cover.fit(), cover_fit(canvas, DESIGN));
    }

    /// The four games' `world_scale` and `cover_box`, item for item.
    #[test]
    fn cover_matches_the_hand_rolled_cover_box() {
        let aspect = DESIGN[0] / DESIGN[1];
        for canvas in [
            [1280.0, 720.0],
            [390.0, 844.0],
            [2560.0, 720.0],
            [1000.0, 747.0],
        ] {
            let hand_scale = (canvas[0] / DESIGN[0]).max(canvas[1] / DESIGN[1]);
            let hand_box = [
                canvas[0].max(canvas[1] * aspect),
                canvas[1].max(canvas[0] / aspect),
            ];
            let m = DesignMapping::cover(canvas, DESIGN);
            assert!(
                (m.scale() - hand_scale).abs() < 1e-6,
                "{canvas:?} world_scale"
            );
            let scaled = m.box_dp();
            assert!(
                (scaled[0] - hand_box[0]).abs() < 1e-3 && (scaled[1] - hand_box[1]).abs() < 1e-3,
                "{canvas:?}: {scaled:?} vs cover_box {hand_box:?}"
            );
            let (_, ox, oy) = m.fit();
            assert!((ox - (canvas[0] - hand_box[0]) * 0.5).abs() < 1e-3);
            assert!((oy - (canvas[1] - hand_box[1]) * 0.5).abs() < 1e-3);
            let (safe_x, safe_y) = (m.visible()[0], m.visible()[1]);
            assert!(
                safe_x <= DESIGN[0] + 1e-3 && safe_y <= DESIGN[1] + 1e-3,
                "cover shows at most the design box: {safe_x} {safe_y}"
            );
            assert!(
                (safe_x - DESIGN[0]).abs() < 1e-3 || (safe_y - DESIGN[1]).abs() < 1e-3,
                "the tight axis survives exactly: {safe_x} {safe_y}"
            );
        }
    }

    /// Known design/window sizes: the scale is the games' ratio, `visible` is
    /// their `field_size`, and design and dp round-trip through each other.
    #[test]
    fn scale_and_visible_round_trip_at_known_sizes() {
        for (canvas, scale) in [
            ([1280.0, 720.0], 4.0 / 3.0),
            ([960.0, 540.0], 1.0),
            ([800.0, 601.0], 800.0 / 960.0),
        ] {
            let m = DesignMapping::contain(canvas, DESIGN);
            assert!((m.scale() - scale).abs() < 1e-6, "{canvas:?}");
            assert_eq!(
                m.scale(),
                (canvas[0] / DESIGN[0]).min(canvas[1] / DESIGN[1]),
                "grid_scale / ui_scale"
            );
            let visible = m.visible();
            assert!((visible[0] - canvas[0] / m.scale()).abs() < 1e-3);
            assert!((visible[1] - canvas[1] / m.scale()).abs() < 1e-3);
            let padded_axis = if (visible[0] - DESIGN[0]).abs() > (visible[1] - DESIGN[1]).abs() {
                1
            } else {
                0
            };
            assert!(
                (visible[padded_axis] - DESIGN[padded_axis]).abs() < 1e-3,
                "the constraining axis shows exactly the design box: {visible:?}"
            );
            for design in [
                [0.0, 0.0],
                DESIGN,
                [DESIGN[0] * 0.5, DESIGN[1] * 0.5],
                [123.5, 45.25],
            ] {
                let back = m.dp_to_design(m.design_to_dp(design));
                assert!(
                    (back[0] - design[0]).abs() < 1e-3 && (back[1] - design[1]).abs() < 1e-3,
                    "{canvas:?} {design:?} came back {back:?}"
                );
            }
            let (_, ox, oy) = m.fit();
            assert_eq!(m.design_to_dp([0.0, 0.0]), [ox, oy]);
        }
    }

    /// The rounding rule: conversions stay fractional, snapping keeps shared
    /// edges shared, and an extent never rounds away from its corners.
    #[test]
    fn snap_box_keeps_shared_edges_on_whole_dp() {
        let m = DesignMapping::contain([1280.0, 720.0], DESIGN);
        let exact = m.design_to_dp([100.0, 0.0]);
        assert!(
            exact[0].fract().abs() > 1e-3,
            "an unrounded conversion stays fractional: {exact:?}"
        );
        let (left, left_size) = m.snap_box([0.0, 0.0], [100.0, 100.0]);
        let (right, right_size) = m.snap_box([100.0, 0.0], [100.0, 100.0]);
        assert_eq!(
            left[0] + left_size[0],
            right[0],
            "the shared edge snaps to one dp value"
        );
        assert_eq!(left_size[1], right_size[1]);
        for snapped in [left_size[0], right_size[0], left_size[1]] {
            let ideal = 100.0 * m.scale();
            assert!(
                (snapped - ideal).abs() <= 1.0,
                "a snapped extent stays within a dp of the exact one: {snapped} vs {ideal}"
            );
        }
        assert_eq!((left, left_size), ([0.0, 0.0], [133.0, 133.0]));
        // A fractional letterbox offset is absorbed into the snapped corner.
        let odd = DesignMapping::contain([1000.0, 601.0], DESIGN);
        let (odd_at, odd_size) = odd.snap_box([0.0, 0.0], [64.0, 64.0]);
        assert_eq!(odd_at, [odd_at[0].round(), odd_at[1].round()]);
        assert!(odd_size[0] > 0.0 && odd_size[1] > 0.0);
    }

    /// Degenerate canvases read as the identity, and a non-finite side is
    /// carried rather than replaced.
    #[test]
    fn degenerate_canvas_reads_as_the_identity_on_design_units() {
        for canvas in [[0.0, 0.0], [-1.0, 720.0], [DESIGN[0], 0.0]] {
            for m in [
                DesignMapping::contain(canvas, DESIGN),
                DesignMapping::cover(canvas, DESIGN),
            ] {
                assert_eq!(m.scale(), 1.0, "{canvas:?} {:?}", m.policy);
                assert_eq!(m.fit(), (1.0, 0.0, 0.0));
                assert_eq!(m.box_dp(), DESIGN);
                assert_eq!(m.design_to_dp([10.0, 20.0]), [10.0, 20.0]);
                assert_eq!(m.dp_to_design([10.0, 20.0]), [10.0, 20.0]);
            }
        }
        assert_eq!(contain_fit([0.0, 0.0], DESIGN), (1.0, 0.0, 0.0));
        assert_eq!(cover_fit([0.0, 0.0], DESIGN), (1.0, 0.0, 0.0));
        assert_eq!(cover_fit(DESIGN, DESIGN), contain_fit(DESIGN, DESIGN));
        // A ratio below the shared 1e-6 guard reads the same way.
        assert_eq!(DesignMapping::contain([1e-4, 1e-4], DESIGN).scale(), 1.0);
    }

    /// The cover tuple needs no camera mode: the existing framing functions
    /// take it as they take the contain tuple, and `design_to_dp` is exactly
    /// their draw path.
    #[test]
    fn the_cover_tuple_drives_the_existing_framing_functions() {
        let canvas = [1000.0, 747.0];
        let m = DesignMapping::cover(canvas, DESIGN);
        let (s, ox, oy) = m.fit();
        assert!(
            s > 0.0 && ox < 0.0 && oy.abs() < 1e-3,
            "cropped on x: {ox} {oy}"
        );
        let center = [DESIGN[0] * 0.5, DESIGN[1] * 0.5];
        let matrix = crate::fit_view_proj_with_roll(canvas, DESIGN, center, m.fit(), 0.0);
        for design in [[0.0, 0.0], DESIGN, center, [1200.0, 300.0]] {
            let [dx, dy] = crate::world_to_dp(design, DESIGN, center, m.fit());
            assert_eq!(
                m.design_to_dp(design),
                [dx, dy],
                "one conversion for canvas and GPU"
            );
            let ndc = matrix.project_point3(glam::Vec3::new(design[0], design[1], 0.0));
            let gpu = [
                (ndc.x + 1.0) * 0.5 * canvas[0],
                (1.0 - ndc.y) * 0.5 * canvas[1],
            ];
            assert!(
                (dx - gpu[0]).abs() < 1e-2 && (dy - gpu[1]).abs() < 1e-2,
                "{design:?}: draw ({dx},{dy}) vs gpu {gpu:?}"
            );
            let back = crate::dp_to_world([dx, dy], DESIGN, center, m.fit());
            assert!(
                (back[0] - design[0]).abs() < 1e-2 && (back[1] - design[1]).abs() < 1e-2,
                "pick inverse for {design:?}: {back:?}"
            );
        }
    }

    /// `contain_fit` / `cover_fit` guard the scale, not the caller: a
    /// non-finite canvas side loses its ratio to `min`/`max` and surfaces in
    /// the offsets.
    #[test]
    fn a_non_finite_canvas_side_is_not_repaired() {
        for m in [
            DesignMapping::contain([f32::NAN, 720.0], DESIGN),
            DesignMapping::cover([f32::NAN, 720.0], DESIGN),
        ] {
            let (s, ox, oy) = m.fit();
            assert!(s.is_finite() && s > 0.0, "{:?} scale {s}", m.policy);
            assert!(ox.is_nan(), "{:?} carries the NaN, got {ox}", m.policy);
            assert!(oy.abs() < 1e-3, "{:?} oy {oy}", m.policy);
            assert!(m.visible()[0].is_nan());
        }
    }
}
