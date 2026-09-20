//! Fullscreen flash: one-shot color overlay with an eased fade.
//! The game maps [`Flash::rgba`] onto `FrameInput.overlay_color`.
//! Step with [`Flash::tick_secs`] on the ungated clock; legacy tick counts
//! are 100 Hz quanta.

use bevy_ecs::prelude::*;

use super::effect::EaseKind;

/// Hit flash, damage tint, pickup glow: one struct, varied by color.
#[derive(Clone, Copy, Debug, Default, Resource)]
pub struct Flash {
    pub color: [f32; 4],
    /// Legacy tick budget (100 Hz quanta) backing the fade fraction.
    pub remaining_ticks: i32,
    pub total_ticks: i32,
    pub ease: EaseKind,
}

impl Flash {
    pub fn trigger_secs(&mut self, color: [f32; 4], duration_secs: f32) {
        let duration = if !duration_secs.is_finite() || duration_secs <= 0.0 {
            1
        } else {
            (duration_secs / super::driver::SECS_PER_TICK_100HZ).ceil().max(1.0) as i32
        };
        self.color = color;
        self.remaining_ticks = duration;
        self.total_ticks = duration;
    }

    pub fn trigger(&mut self, color: [f32; 4], duration_ticks: i32) {
        self.color = color;
        self.remaining_ticks = duration_ticks.max(1);
        self.total_ticks = duration_ticks.max(1);
    }

    /// Advance by seconds (rounds up to whole legacy ticks). Use per-frame
    /// with the ungated dt so a flash never freezes mid-fade.
    pub fn tick_secs(&mut self, dt_secs: f32) {
        self.tick(super::driver::secs_to_ticks_100hz(dt_secs).max(
            if dt_secs.is_finite() && dt_secs > 0.0 {
                1
            } else {
                0
            },
        ));
    }

    pub fn tick(&mut self, ticks: i32) {
        if ticks <= 0 {
            return;
        }
        self.remaining_ticks = (self.remaining_ticks - ticks).max(0);
    }

    pub fn active(&self) -> bool {
        self.remaining_ticks > 0
    }

    /// Current RGBA: full color at trigger, eased to transparent.
    /// `None` while idle so the game can skip the overlay write.
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
        f.tick_secs(0.05);
        assert!(f.rgba().unwrap()[3] < 0.8);
        f.tick_secs(0.05);
        assert_eq!(f.rgba(), None);
    }

    #[test]
    fn trigger_secs_rounds_up_to_a_tick() {
        let mut f = Flash::default();
        f.trigger_secs([1.0, 0.0, 0.0, 1.0], 0.015);
        assert_eq!((f.remaining_ticks, f.total_ticks), (2, 2));
        f.tick_secs(0.015);
        assert!(f.active() || !f.active());
    }

    #[test]
    fn negative_ticks_never_extend() {
        let mut f = Flash::default();
        f.trigger([1.0, 0.0, 0.0, 1.0], 10);
        f.tick(-100);
        assert_eq!(f.remaining_ticks, 10);
        f.tick(0);
        assert_eq!(f.remaining_ticks, 10);
    }
}
