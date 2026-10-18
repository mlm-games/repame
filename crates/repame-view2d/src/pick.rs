//! Pointer picks in world space: the event vocabulary and the click rule.
//!
//! Reuse this vocabulary for any surface that reports picks; the emitters
//! (`actors_view_world_with_events`, a game's own sprite viewport) and the
//! consumers then agree on what the pointer did.

use glam::Vec2;

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
