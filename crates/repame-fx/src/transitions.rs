//! State transitions: fade cover/hold/uncover with input blocking.
//! Mirrors the retired `Transition` numbers exactly (Fade at speed 2.5
//! = 0.4 s cover + 0.4 s uncover at 100 Hz); the game paints
//! [`TransitionFx::alpha`] as a black rect and gates its tick driver on
//! [`TransitionFx::blocking`].
//!
//! Custom visuals (spiral vortex, circle wipes, …) attach the Godot/Bevy
//! way: the game picks a [`TransitionVisual::Custom`] id, drives its own
//! fullscreen pass (e.g. `repame-sprite::FullscreenPass`) from
//! [`TransitionFx::cover_amount`], and gates input on [`TransitionFx::blocking`].
//! `Custom(1)` is the spiral-vortex convention.

/// Ticks per half (0.4 s at 100 Hz).
pub const HALF_TICKS: i32 = 40;
/// Custom id convention for spiral-vortex wipes.
pub const VORTEX_CUSTOM_ID: u8 = 1;

/// What the transition looks like. Timing/alpha/blocking are identical
/// for every variant; only the game's renderer interprets the kind.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TransitionVisual {
    #[default]
    Fade,
    /// Opaque custom visual — this crate only drives timing/cover amount.
    /// The consumer decides how to render it (e.g. a spiral-vortex shader).
    Custom(u8),
}

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
    kind: TransitionVisual,
    /// Kind used by [`Self::begin`] unless the consumer overrides it.
    /// Set once (e.g. at startup) to pick a house style.
    pub default_kind: TransitionVisual,
}

impl TransitionFx {
    pub fn new() -> Self {
        Self::default()
    }

    /// Start a cover->uncover cycle (state already swapped underneath,
    /// like `begin_to_state`). Uses [`Self::default_kind`].
    pub fn begin(&mut self) {
        self.begin_with(self.default_kind);
    }

    /// Start a cover->uncover cycle with an explicit visual.
    pub fn begin_with(&mut self, kind: TransitionVisual) {
        self.phase = Phase::Cover;
        self.t = 0;
        self.kind = kind;
    }

    /// Start a custom-visual cycle (spiral vortex, circle wipe, …).
    pub fn begin_custom(&mut self, id: u8) {
        self.begin_with(TransitionVisual::Custom(id));
    }

    /// Convenience for vortex/spiral wipes — uses `Custom(1)` convention.
    /// This crate does not render it; check [`Self::is_vortex`] and draw
    /// the vortex pass with intensity from [`Self::cover_amount`].
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

    /// True when the active visual is the vortex convention
    /// (`Custom(1)` — see [`Self::begin_vortex`]).
    pub fn is_vortex(&self) -> bool {
        self.is_custom(VORTEX_CUSTOM_ID)
    }

    pub fn blocking(&self) -> bool {
        self.phase != Phase::Idle
    }

    /// Black overlay alpha 0..1 (1 = fully covered). For custom visuals
    /// this is a fallback dim; prefer [`Self::cover_amount`] as the
    /// effect intensity.
    pub fn alpha(&self) -> f32 {
        match self.phase {
            Phase::Idle => 0.0,
            Phase::Cover => (self.t as f32 / HALF_TICKS as f32).clamp(0.0, 1.0),
            Phase::Uncover => 1.0 - (self.t as f32 / HALF_TICKS as f32).clamp(0.0, 1.0),
        }
    }

    /// 0.0 = uncovered, 1.0 = fully covered. Drive custom fullscreen
    /// effects (e.g. vortex density/radius) from this; fades can paint
    /// it directly as a black rect (same values as [`Self::alpha`]).
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
