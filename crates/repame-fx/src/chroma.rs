//! Chromatic aberration pulse: fullscreen RGB-split amount with linear
//! decay. Mirrors `game-utils-bevy`'s `ChromaticAberration` (`pulse`
//! keeps the max, 2.0/s decay, shader `offset = intensity * 0.02`) —
//! that ran on Bevy's generalized post-process API (Bevy itself never
//! shipped a chroma effect), same split as here: state in fx, sampling
//! composite in `repame-sprite`'s `post`.
//!
//! Plain data: the game maps [`Chroma::amount`] onto
//! `repame_sprite::FrameInput::chroma` each frame (canvas viewports
//! ignore it — sampling FX need the GPU viewport). NT pulses land at
//! 0.04 (pickups) .. 0.7 (throne kills).

/// Fullscreen RGB-split state. Own it as a resource (see
/// [`init`](crate::init_resources)) or a plain field — stepping is a
/// method call either way.
#[derive(Clone, Debug)]
pub struct Chroma {
    /// Current split amount (bevy `chromatic_intensity` units).
    pub strength: f32,
    /// Linear decay per second (bevy `chromatic_decay`, 2.0).
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

    /// Fire a pulse; keeps the max so overlapping hits never cancel.
    /// (NT: 0.04 pickups, 0.08–0.3 hits, 0.4–0.7 kills/throne.)
    pub fn pulse(&mut self, strength: f32) {
        self.strength = self.strength.max(strength.max(0.0));
    }

    /// Decay over `ticks` (100 Hz). No-op at 0.
    pub fn tick(&mut self, ticks: i32) {
        if ticks <= 0 {
            return;
        }
        self.strength = (self.strength - self.decay_per_sec * ticks as f32 / 100.0).max(0.0);
    }

    /// Current amount for `FrameInput::chroma`. `0.0` takes the
    /// zero-cost direct path (no offscreen composite).
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
        c.tick(100);
        assert!(!c.active());
        assert_eq!(c.amount(), 0.0);
    }

    #[test]
    fn partial_decay_is_linear() {
        let mut c = Chroma::new();
        c.pulse(0.7);
        c.tick(10);
        assert!((c.amount() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn negative_pulse_clamps() {
        let mut c = Chroma::new();
        c.pulse(-1.0);
        assert!(!c.active());
    }
}
