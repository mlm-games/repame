//! State transitions: fade cover, hold, uncover with input blocking.
//! Fade runs 0.4 s cover plus 0.4 s uncover at 100 Hz.
//! Custom visuals drive their own pass from cover amount.

/// Ticks per half (0.4 s at 100 Hz).
pub const HALF_TICKS: i32 = 40;
/// Custom id convention for spiral-vortex wipes.
pub const VORTEX_CUSTOM_ID: u8 = 1;

use bevy_ecs::prelude::*;

/// What the transition looks like. Timing and blocking match
/// across variants; only the renderer interprets the kind.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TransitionVisual {
    #[default]
    Fade,
    /// Opaque custom visual; this crate only drives timing and cover.
    /// The consumer renders it from cover amount.
    Custom(u8),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Phase {
    #[default]
    Idle,
    Cover,
    Uncover,
}

#[derive(Clone, Copy, Debug, Default, Resource)]
pub struct TransitionFx {
    phase: Phase,
    t: i32,
    kind: TransitionVisual,
    /// Kind used by [`Self::begin`] unless the consumer overrides it.
    pub default_kind: TransitionVisual,
}

impl TransitionFx {
    pub fn new() -> Self {
        Self::default()
    }

    /// Start a cover and uncover cycle. Uses [`Self::default_kind`].
    pub fn begin(&mut self) {
        self.begin_with(self.default_kind);
    }

    /// Start a cover and uncover cycle with an explicit visual.
    pub fn begin_with(&mut self, kind: TransitionVisual) {
        self.phase = Phase::Cover;
        self.t = 0;
        self.kind = kind;
    }

    /// Start a custom-visual cycle by id.
    pub fn begin_custom(&mut self, id: u8) {
        self.begin_with(TransitionVisual::Custom(id));
    }

    /// Start a vortex wipe using the `Custom(1)` convention.
    pub fn begin_vortex(&mut self) {
        self.begin_custom(VORTEX_CUSTOM_ID);
    }

    /// Current visual kind.
    pub fn kind(&self) -> TransitionVisual {
        self.kind
    }

    /// True when the active visual is `Custom(id)`.
    pub fn is_custom(&self, id: u8) -> bool {
        matches!(self.kind, TransitionVisual::Custom(v) if v == id)
    }

    /// True when the active visual is the vortex convention.
    pub fn is_vortex(&self) -> bool {
        self.is_custom(VORTEX_CUSTOM_ID)
    }

    pub fn blocking(&self) -> bool {
        self.phase != Phase::Idle
    }

    /// Black overlay alpha 0..1 (1 is covered). Custom visuals can use
    /// [`Self::cover_amount`] as the effect intensity instead.
    pub fn alpha(&self) -> f32 {
        match self.phase {
            Phase::Idle => 0.0,
            Phase::Cover => (self.t as f32 / HALF_TICKS as f32).clamp(0.0, 1.0),
            Phase::Uncover => 1.0 - (self.t as f32 / HALF_TICKS as f32).clamp(0.0, 1.0),
        }
    }

    /// 0.0 is uncovered, 1.0 is covered. Custom passes read this value.
    pub fn cover_amount(&self) -> f32 {
        self.alpha()
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

    #[test]
    fn custom_vortex_kind_drives_cover_amount() {
        let mut fx = TransitionFx::new();
        fx.begin_vortex();
        assert!(fx.is_vortex());
        assert!(fx.is_custom(VORTEX_CUSTOM_ID));
        assert_eq!(fx.kind(), TransitionVisual::Custom(VORTEX_CUSTOM_ID));
        fx.step(20);
        assert!((fx.cover_amount() - 0.5).abs() < 1e-5);
        let mut fx2 = TransitionFx::new();
        fx2.default_kind = TransitionVisual::Custom(7);
        fx2.begin();
        assert!(fx2.is_custom(7));
        assert!(!fx2.is_vortex());
    }
}
