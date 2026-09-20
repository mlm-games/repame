use bevy_ecs::prelude::*;

/// Chromatic split amount with linear decay. The game maps it to output.
/// Timebase is seconds (`tick_secs`); `tick` is the 100 Hz shim.
#[derive(Clone, Debug, Resource)]
pub struct Chroma {
    /// Current split amount.
    pub strength: f32,
    /// Linear decay per second.
    pub decay_per_sec: f32,
}

impl Default for Chroma {
    fn default() -> Self {
        Self::new()
    }
}

impl Chroma {
    pub fn new() -> Self {
        Self {
            strength: 0.0,
            decay_per_sec: 2.0,
        }
    }

    /// Fire a pulse; keeps the max so overlapping pulses do not cancel.
    pub fn pulse(&mut self, strength: f32) {
        self.strength = self.strength.max(strength.max(0.0));
    }

    /// Decay over `dt_secs` seconds. Non-positive or non-finite dt holds.
    pub fn tick_secs(&mut self, dt_secs: f32) {
        if !dt_secs.is_finite() || dt_secs <= 0.0 {
            return;
        }
        self.strength = (self.strength - self.decay_per_sec * dt_secs).max(0.0);
    }

    /// Legacy decay over `ticks` 100 Hz quanta.
    pub fn tick(&mut self, ticks: i32) {
        self.tick_secs(super::driver::ticks_to_secs_100hz(ticks));
    }

    /// Current amount for `FrameInput::chroma`. 0.0 skips the composite.
    pub fn amount(&self) -> f32 {
        self.strength.max(0.0)
    }

    pub fn active(&self) -> bool {
        self.strength > 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pulse_keeps_max_and_decays() {
        let mut c = Chroma::new();
        assert!(!c.active());
        assert_eq!(c.amount(), 0.0);
        c.pulse(0.3);
        c.pulse(0.1);
        assert_eq!(c.amount(), 0.3);
        c.tick_secs(1.0);
        assert!(!c.active());
        assert_eq!(c.amount(), 0.0);
    }

    #[test]
    fn tick_shim_matches_seconds() {
        let mut a = Chroma::new();
        let mut b = Chroma::new();
        a.pulse(0.7);
        b.pulse(0.7);
        a.tick(10);
        b.tick_secs(0.1);
        assert!((a.amount() - b.amount()).abs() < 1e-6);
    }

    #[test]
    fn partial_decay_is_linear() {
        let mut c = Chroma::new();
        c.pulse(0.7);
        c.tick_secs(0.1);
        assert!((c.amount() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn negative_pulse_clamps() {
        let mut c = Chroma::new();
        c.pulse(-1.0);
        assert!(!c.active());
    }
}
