//! Fullscreen flash: one-shot color overlay with an eased fade.
//! The game maps [`Flash::rgba`] onto `FrameInput.overlay_color`.

use super::effect::EaseKind;

/// White hit-flash, red damage vignette, gold pickup glow: all the
/// same struct, different color and duration.
#[derive(Clone, Copy, Debug, Default)]
pub struct Flash {
    pub color: [f32; 4],
    /// Ticks left / total (100 Hz).
    pub remaining_ticks: i32,
    pub total_ticks: i32,
    pub ease: EaseKind,
}

impl Flash {
    pub fn trigger(&mut self, color: [f32; 4], duration_ticks: i32) {
        self.color = color;
        self.remaining_ticks = duration_ticks.max(1);
        self.total_ticks = duration_ticks.max(1);
    }

    pub fn tick(&mut self, ticks: i32) {
        self.remaining_ticks = (self.remaining_ticks - ticks).max(0);
    }

    pub fn active(&self) -> bool {
        self.remaining_ticks > 0
    }

    /// Current RGBA: full color at trigger, eased to transparent.
    /// `None` when idle so the game can skip the overlay write.
    pub fn rgba(&self) -> Option<[f32; 4]> {
        if !self.active() {
            return None;
        }
        let t = self.remaining_ticks as f32 / self.total_ticks.max(1) as f32;
        let mut c = self.color;
        c[3] *= 1.0 - self.ease.apply(1.0 - t);
        Some(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flash_fades_to_transparent() {
        let mut f = Flash::default();
        assert_eq!(f.rgba(), None);
        f.trigger([1.0, 1.0, 1.0, 0.8], 10);
        let start = f.rgba().unwrap();
        assert!((start[3] - 0.8).abs() < 1e-4);
        f.tick(5);
        assert!(f.rgba().unwrap()[3] < 0.8);
        f.tick(5);
        assert_eq!(f.rgba(), None);
    }
}
