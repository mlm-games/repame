//! State transitions: fade cover and uncover with input blocking.
//!
//! Timebase is **seconds**: [`HALF_SECS`] per half, [`TransitionFx::step_secs`]
//! advancing on the caller's clock. `step`/`HALF_TICKS` are 100 Hz shims for
//! legacy games.
//!
//! Clock rule: step the transition on the *ungated* clock (before the
//! `blocking` gate) every frame. Gating the fx step on `blocking` deadlocks:
//! nothing advances the transition, so it never unblocks. Gate gameplay
//! systems and input edges on [`TransitionFx::blocking`], never the fx step.

/// Seconds per transition half.
pub const HALF_SECS: f32 = 0.4;
/// Legacy ticks per half (0.4 s at 100 Hz).
pub const HALF_TICKS: i32 = 40;
/// Custom id convention for spiral-vortex wipes.
pub const VORTEX_CUSTOM_ID: u8 = 1;
/// Legacy 100 Hz quantum helpers (one mapping: `ticks / 100.0` s).
pub fn ticks_to_secs(ticks: i32) -> f32 {
    super::driver::ticks_to_secs_100hz(ticks)
}

/// Legacy seconds-to-quanta helper.
pub fn secs_to_ticks(dt_secs: f32) -> i32 {
    super::driver::secs_to_ticks_100hz(dt_secs)
}

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
    t_secs: f32,
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
        self.t_secs = 0.0;
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
            Phase::Cover => (self.t_secs / HALF_SECS).clamp(0.0, 1.0),
            Phase::Uncover => 1.0 - (self.t_secs / HALF_SECS).clamp(0.0, 1.0),
        }
    }

    /// 0.0 is uncovered, 1.0 is covered. Custom passes read this value.
    pub fn cover_amount(&self) -> f32 {
        self.alpha()
    }

    /// Advance on the ungated clock. Non-positive or non-finite dt holds.
    pub fn step_secs(&mut self, dt_secs: f32) {
        if !dt_secs.is_finite() || dt_secs <= 0.0 {
            return;
        }
        match self.phase {
            Phase::Idle => {}
            Phase::Cover => {
                self.t_secs += dt_secs;
                if self.t_secs >= HALF_SECS {
                    self.phase = Phase::Uncover;
                    self.t_secs = 0.0;
                }
            }
            Phase::Uncover => {
                self.t_secs += dt_secs;
                if self.t_secs >= HALF_SECS {
                    self.phase = Phase::Idle;
                    self.t_secs = 0.0;
                }
            }
        }
    }

    /// Legacy 100 Hz shim: `step_secs(ticks / 100)`.
    pub fn step(&mut self, ticks: i32) {
        self.step_secs(super::driver::ticks_to_secs_100hz(ticks));
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
        fx.step_secs(0.2);
        assert!((fx.alpha() - 0.5).abs() < 1e-5);
        fx.step_secs(0.2);
        assert!(fx.blocking());
        assert_eq!(fx.alpha(), 1.0);
        fx.step_secs(0.4);
        assert!(!fx.blocking());
        assert_eq!(fx.alpha(), 0.0);
    }

    #[test]
    fn tick_shim_matches_seconds() {
        let mut a = TransitionFx::new();
        let mut b = TransitionFx::new();
        a.begin();
        b.begin();
        a.step(20);
        b.step_secs(0.2);
        assert!((a.alpha() - b.alpha()).abs() < 1e-6);
    }

    #[test]
    fn non_positive_dt_holds() {
        let mut fx = TransitionFx::new();
        fx.begin();
        fx.step_secs(0.0);
        fx.step_secs(f32::NAN);
        assert!((fx.alpha() - 0.0).abs() < 1e-6);
        assert!(fx.blocking());
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
