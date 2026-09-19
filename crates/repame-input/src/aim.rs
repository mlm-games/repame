//! Screen-anchored pointer staging: raw window-physical px in, live
//! world point out.

use glam::Vec2;

/// Staged pointer click: world position (aim-at-click) plus canvas-dp
/// position (camera-independent menu hit-testing).
#[derive(Clone, Copy, Debug)]
pub struct StagedClick {
    pub world: Vec2,
    pub dp: [f32; 2],
}

/// Raw window-physical px from two sources sharing one order clock:
/// the root `on_pointer_move` (fires almost exclusively while a button
/// is held.
#[derive(Clone, Copy, Debug, Default)]
pub struct AimTracker {
    hover: Option<Vec2>,
    hover_px: Option<Vec2>,
    cursor_px: Option<Vec2>,
    cursor_seq: u64,
    hover_seq: u64,
    hover_px_seq: u64,
    input_seq: u64,
}

impl AimTracker {
    /// Frames a staged px stays live without a refresh.
    pub const STALE_AFTER: u64 = 30;

    /// Stage the cursor's window-physical px position (root
    /// `on_pointer_move`, y-down). Stored raw — unprojected through
    /// the live camera each frame by the caller.
    pub fn cursor_move(&mut self, phys_px: Vec2) {
        self.input_seq += 1;
        self.cursor_seq = self.input_seq;
        self.cursor_px = Some(phys_px);
    }

    /// Stage one viewport hover: the baked world point AND the raw
    /// window-physical `screen` px.
    pub fn stage_hover(&mut self, world: Vec2, screen: [f32; 2]) {
        self.input_seq += 1;
        self.hover_seq = self.input_seq;
        self.hover = Some(world);
        self.hover_px_seq = self.input_seq;
        self.hover_px = Some(Vec2::new(screen[0], screen[1]));
    }

    /// Fresh staged px in priority order (hover px, root px), or `None`
    /// when every source is stale/absent and the caller must fall back
    /// to the baked hover point.
    pub fn live_px(&self) -> Option<Vec2> {
        if self.hover_px_seq > 0
            && self.input_seq.wrapping_sub(self.hover_px_seq) <= Self::STALE_AFTER
        {
            return self.hover_px;
        }
        if self.cursor_seq > 0
            && self.input_seq.wrapping_sub(self.cursor_seq) <= Self::STALE_AFTER
        {
            return self.cursor_px;
        }
        None
    }

    /// Last-resort fallback: the baked hover world point (touch/pen
    /// never stage any px).
    pub fn baked(&self) -> Option<Vec2> {
        self.hover
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_px_wins_stale_px_yields_none() {
        let mut t = AimTracker::default();
        assert_eq!(t.live_px(), None);
        t.stage_hover(Vec2::new(10.0, 10.0), [0.0, 0.0]);
        t.cursor_move(Vec2::new(100.0, 100.0));
        assert_eq!(t.live_px(), Some(Vec2::new(0.0, 0.0)));
        for _ in 0..40 {
            t.stage_hover(Vec2::new(10.0, 10.0), [50.0, 50.0]);
        }
        assert_eq!(t.live_px(), Some(Vec2::new(50.0, 50.0)));
        assert_eq!(t.baked(), Some(Vec2::new(10.0, 10.0)));
    }
}
