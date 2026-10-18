//! One eased value: `from` to `to` over a duration along a shared
//! [`EaseKind`], one-shot or repeating. This is the "entrance" / "burst alpha"
//! helper games hand-roll. The curve comes from `repose_core::animation` and
//! the clock from [`StepTimer`], so neither is duplicated here.
//!
//! Timebase: seconds, like the rest of the crate. Step on the ungated clock
//! with the fixed step's `delta_secs`; a non-finite or non-positive `dt`
//! holds the value.
//!
//! Endpoints are exact, not curve-evaluated: at `t = 0` the value is `from`
//! and at `t = duration_secs` it is `to`, for every [`EaseKind`], including an
//! overshooting one like `BackOut` (which passes `to` in the middle and still
//! lands on it exactly). A repeating value restarts at `from` every period.
//!
//! Not every hand-rolled curve is one of these. The games' `burst_alpha`
//! holds at full until its fade start and then fades along `1 - u * u`, which
//! is `QuadOut` sampled on reversed time (`QuadOut(1 - u) == 1 - u * u`). The
//! driver runs time forward only, so that two-phase hold stays game-side, or
//! the game drives just its fade segment.

use repame_sim::{StepTimer, TimerMode};

use super::effect::EaseKind;

/// A value eased from `from` to `to` over `duration_secs`. Own it wherever the
/// value lives (a spawn animation, a HUD pop, a fade): it is plain state, not
/// a resource. Rebuild it, or [`reset`](Self::reset) it, to replay the same
/// move; `from` and `to` stay public so a retarget starts from the value on
/// screen.
#[derive(Clone, Copy, Debug)]
pub struct EasedValue {
    /// Value at `t = 0`.
    pub from: f32,
    /// Value at `t = duration_secs`.
    pub to: f32,
    /// Curve shape between the endpoints.
    pub ease: EaseKind,
    /// Elapsed time, duration, and one-shot/looping state. Private: step it
    /// through [`step_secs`](Self::step_secs) and sample with
    /// [`value_at`](Self::value_at), so time never runs backwards.
    timer: StepTimer,
}

impl EasedValue {
    /// A one-shot value: it holds at `to` once it arrives.
    pub fn new(from: f32, to: f32, duration_secs: f32, ease: EaseKind) -> Self {
        Self {
            from,
            to,
            ease,
            timer: StepTimer::from_seconds(duration_secs, TimerMode::Once),
        }
    }

    /// A looping value: it restarts at `from` every period, so
    /// [`finished`](Self::finished) is never true.
    pub fn repeating(from: f32, to: f32, duration_secs: f32, ease: EaseKind) -> Self {
        Self {
            timer: StepTimer::from_seconds(duration_secs, TimerMode::Repeating),
            ..Self::new(from, to, duration_secs, ease)
        }
    }

    /// Advance `dt_secs` seconds.
    pub fn step_secs(&mut self, dt_secs: f32) {
        if !dt_secs.is_finite() || dt_secs <= 0.0 {
            return;
        }
        self.timer.tick(dt_secs);
    }

    /// Current value.
    pub fn value(&self) -> f32 {
        self.value_of(self.timer.fraction())
    }

    /// Value `t_secs` seconds after the start, without stepping: a pure sample
    /// for code that derives time from the sim clock and never wants the
    /// driver to hold time of its own. Past the end a one-shot holds at `to`;
    /// a repeating value wraps, so exactly one period reads as `from` again.
    /// A non-finite sample reads as the start.
    pub fn value_at(&self, t_secs: f32) -> f32 {
        if !t_secs.is_finite() {
            return self.from;
        }
        let mut probe = self.timer;
        probe.reset();
        probe.tick(t_secs);
        self.value_of(probe.fraction())
    }

    /// The duration copied at construction.
    pub fn duration_secs(&self) -> f32 {
        self.timer.duration()
    }

    /// 0..1 progress through the current period (`1.0` once a one-shot is
    /// done; `1.0` from the start for a value with no duration).
    pub fn progress(&self) -> f32 {
        self.timer.fraction()
    }

    /// A one-shot that has arrived, or a value with no duration, which reads
    /// as already arrived. A repeating value never finishes.
    pub fn finished(&self) -> bool {
        self.timer.finished()
    }

    /// Replay from `t = 0`.
    pub fn reset(&mut self) {
        self.timer.reset();
    }

    /// Exact endpoints; the curve only shapes the middle.
    fn value_of(&self, fraction: f32) -> f32 {
        if fraction <= 0.0 {
            return self.from;
        }
        if fraction >= 1.0 {
            return self.to;
        }
        self.from + (self.to - self.from) * self.ease.apply(fraction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Endpoints land exactly on `from` and `to` for every curve, however the
    /// curve itself evaluates at 0 and 1.
    #[test]
    fn endpoints_are_exact_for_every_ease() {
        for ease in [
            EaseKind::Linear,
            EaseKind::QuadOut,
            EaseKind::CubicOut,
            EaseKind::BackOut,
        ] {
            let mut v = EasedValue::new(-3.25, 7.5, 0.4, ease);
            assert_eq!(v.value(), -3.25);
            assert_eq!(v.value_at(0.0), -3.25);
            assert_eq!(v.value_at(0.4), 7.5);
            assert_eq!(v.value_at(-1.0), -3.25, "before the start");
            assert_eq!(v.value_at(9.0), 7.5, "past the end");
            v.step_secs(0.4);
            assert_eq!(v.value(), 7.5);
            assert_eq!(v.progress(), 1.0);
            assert!(v.finished());
            v.step_secs(0.4);
            assert_eq!(v.value(), 7.5, "a finished one-shot holds");
        }
    }

    /// The curve shapes the middle: `BackOut` overshoots past `to` and still
    /// ends on it; `QuadOut` leads a linear blend.
    #[test]
    fn the_curve_shapes_the_middle() {
        let back = EasedValue::new(0.0, 1.0, 0.5, EaseKind::BackOut);
        assert!(back.value_at(0.25) > 1.0, "back-out overshoots");
        assert_eq!(back.value_at(0.5), 1.0);
        let quad = EasedValue::new(0.0, 1.0, 0.5, EaseKind::QuadOut);
        assert!((quad.value_at(0.25) - 0.75).abs() < 1e-6);
        assert!(quad.value_at(0.25) > 0.5, "out curves lead");
    }

    /// The four games' `entrance()`: `C1 = 1.70158` back-out over normalized
    /// age. This is the same curve the shared `EaseKind` carries.
    #[test]
    fn back_out_matches_the_hand_rolled_entrance() {
        let hand_rolled = |t: f32| {
            const C1: f32 = 1.70158;
            const C3: f32 = C1 + 1.0;
            let u = t - 1.0;
            1.0 + C3 * u * u * u + C1 * u * u
        };
        let v = EasedValue::new(0.0, 1.0, 1.0, EaseKind::BackOut);
        for step in 0..=10 {
            let t = step as f32 / 10.0;
            assert!(
                (v.value_at(t) - hand_rolled(t)).abs() < 1e-6,
                "t = {t}: {} vs {}",
                v.value_at(t),
                hand_rolled(t)
            );
        }
    }

    /// A repeating value wraps on its period, keeps its phase across a whole
    /// period's step, and never finishes.
    #[test]
    fn repeating_wraps_on_the_period() {
        let mut v = EasedValue::repeating(1.0, 0.0, 0.25, EaseKind::Linear);
        assert_eq!(v.value(), 1.0);
        v.step_secs(0.1);
        assert!((v.value() - 0.6).abs() < 1e-5);
        assert_eq!(v.value_at(0.1), v.value(), "sampling agrees with stepping");
        v.step_secs(0.25);
        assert!((v.value() - 0.6).abs() < 1e-5, "phase survives a period");
        assert!(!v.finished());
        let mut w = EasedValue::repeating(1.0, 0.0, 0.25, EaseKind::Linear);
        w.step_secs(0.25);
        assert_eq!(w.value(), 1.0, "one period reads as the start again");
        assert!(!w.finished());
    }

    /// A duration of zero (or none at all) is an instant move, and dt guards
    /// keep time from moving on bad input.
    #[test]
    fn degenerate_durations_and_dt_do_not_move() {
        for v in [
            EasedValue::new(0.0, 2.5, 0.0, EaseKind::BackOut),
            EasedValue::new(0.0, 2.5, -1.0, EaseKind::BackOut),
            EasedValue::new(0.0, 2.5, f32::NAN, EaseKind::BackOut),
        ] {
            assert_eq!(v.value(), 2.5, "an instant value reads as arrived");
            assert!(v.finished());
        }
        let mut held = EasedValue::new(0.0, 1.0, 1.0, EaseKind::Linear);
        held.step_secs(-0.5);
        held.step_secs(f32::NAN);
        held.step_secs(0.0);
        assert_eq!(held.value(), 0.0, "time never runs backwards");
        held.step_secs(0.5);
        assert!((held.value() - 0.5).abs() < 1e-6);
        held.reset();
        assert_eq!((held.value(), held.progress()), (0.0, 0.0));
    }
}
