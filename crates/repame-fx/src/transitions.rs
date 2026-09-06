//! State transitions: fade cover/hold/uncover with input blocking.
//! Mirrors the retired `Transition` numbers exactly (Fade at speed 2.5
//! = 0.4 s cover + 0.4 s uncover at 100 Hz); the game paints [`Self::alpha`]
//! as a black rect and gates its tick driver on [`Self::blocking`].

/// Ticks per half (0.4 s at 100 Hz).
pub const HALF_TICKS: i32 = 40;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Phase {
    #[default]
    Idle,
    Cover,
    Uncover,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TransitionFx {
    phase: Phase,
    t: i32,
}

impl TransitionFx {
    pub fn new() -> Self {
        Self::default()
    }

    /// Start a cover->uncover cycle (state already swapped underneath,
    /// like `begin_to_state`).
    pub fn begin(&mut self) {
        self.phase = Phase::Cover;
        self.t = 0;
    }

    pub fn blocking(&self) -> bool {
        self.phase != Phase::Idle
    }

    /// Black overlay alpha 0..1 (1 = fully covered).
    pub fn alpha(&self) -> f32 {
        match self.phase {
            Phase::Idle => 0.0,
            Phase::Cover => (self.t as f32 / HALF_TICKS as f32).clamp(0.0, 1.0),
            Phase::Uncover => 1.0 - (self.t as f32 / HALF_TICKS as f32).clamp(0.0, 1.0),
        }
    }

    pub fn step(&mut self, ticks: i32) {
        if ticks <= 0 {
            return;
        }
        match self.phase {
            Phase::Idle => {}
            Phase::Cover => {
                self.t += ticks;
                if self.t >= HALF_TICKS {
                    self.phase = Phase::Uncover;
                    self.t = 0;
                }
            }
            Phase::Uncover => {
                self.t += ticks;
                if self.t >= HALF_TICKS {
                    self.phase = Phase::Idle;
                    self.t = 0;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_cycle_blocks_then_releases() {
        let mut fx = TransitionFx::new();
        assert!(!fx.blocking());
        assert_eq!(fx.alpha(), 0.0);
        fx.begin();
        assert!(fx.blocking());
        fx.step(20);
        assert!((fx.alpha() - 0.5).abs() < 1e-5);
        fx.step(20);
        assert!(fx.blocking());
        assert_eq!(fx.alpha(), 1.0);
        fx.step(40);
        assert!(!fx.blocking());
        assert_eq!(fx.alpha(), 0.0);
    }
}
