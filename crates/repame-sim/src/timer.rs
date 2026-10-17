//! Fixed-step countdown timer: the alarm pattern step-based games run on.
//! Counts down in sim steps, fires exactly on the landing step, re-arms for
//! repeats; a disarmed timer never fires.
//!
//! Timebase is seconds like [`SimTime`](crate::SimTime): construct from
//! seconds (`15.0 / 30.0` for 15 steps at 30 Hz), tick with the step's
//! `delta_secs`. [`StepTimer::just_finished`] is true only on the completing
//! step, because branching on that single-step edge is the whole point.

use bevy_ecs::prelude::*;
use serde::{Deserialize, Serialize};

/// Once (clamp at duration) vs repeating (wrap elapsed) expiry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimerMode {
    Once,
    Repeating,
}

/// Countdown/up timer over plain seconds, ticked once per fixed step.
#[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct StepTimer {
    elapsed: f32,
    dur: f32,
    mode: TimerMode,
    just_finished: bool,
}

impl Default for StepTimer {
    /// Disarmed (finished from birth): the "no timer" state.
    fn default() -> Self {
        Self::disarmed()
    }
}

impl StepTimer {
    pub fn from_seconds(dur: f32, mode: TimerMode) -> Self {
        Self {
            elapsed: 0.0,
            dur: dur.max(0.0),
            mode,
            just_finished: false,
        }
    }

    /// Disarmed timer (finished from birth).
    pub fn disarmed() -> Self {
        Self {
            elapsed: 0.0,
            dur: 0.0,
            mode: TimerMode::Once,
            just_finished: false,
        }
    }

    /// Advance by `dt` (read `just_finished()` / `finished()` afterwards; a
    /// finished `Once` never re-fires, so a repeat comes from re-arming).
    pub fn tick(&mut self, dt: f32) {
        self.just_finished = false;
        if self.dur <= 0.0 {
            return;
        }
        match self.mode {
            TimerMode::Once => {
                let was = self.finished();
                self.elapsed = (self.elapsed + dt).min(self.dur);
                self.just_finished = !was && self.finished();
            }
            TimerMode::Repeating => {
                self.elapsed += dt;
                if self.elapsed >= self.dur {
                    self.elapsed %= self.dur;
                    self.just_finished = true;
                }
            }
        }
    }

    /// True only on the step the timer completed (cleared next step).
    pub fn just_finished(&self) -> bool {
        self.just_finished
    }

    /// Latching finished state (stays true for `Once`).
    pub fn finished(&self) -> bool {
        self.elapsed >= self.dur
    }

    /// Alias for call sites that read `is_finished`.
    pub fn is_finished(&self) -> bool {
        self.finished()
    }

    pub fn duration(&self) -> f32 {
        self.dur
    }

    pub fn elapsed_secs(&self) -> f32 {
        self.elapsed
    }

    pub fn remaining_secs(&self) -> f32 {
        (self.dur - self.elapsed).max(0.0)
    }

    /// 0..1 progress (1.0 for zero-duration timers).
    pub fn fraction(&self) -> f32 {
        if self.dur <= 0.0 {
            return 1.0;
        }
        (self.elapsed / self.dur).clamp(0.0, 1.0)
    }

    /// Remaining seconds for countdown users.
    pub fn remaining(&self) -> f32 {
        self.remaining_secs()
    }

    /// Re-arm to full duration.
    pub fn reset(&mut self) {
        self.elapsed = 0.0;
        self.just_finished = false;
    }

    /// Re-arm with a new duration (elapsed preserved).
    pub fn set_duration(&mut self, dur: f32) {
        self.dur = dur.max(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 30.0;

    #[test]
    fn once_fires_exactly_once_and_stays_finished() {
        let mut t = StepTimer::from_seconds(3.0 * DT, TimerMode::Once);
        let mut fires = 0;
        for _ in 0..2 {
            t.tick(DT);
            assert!(!t.just_finished());
        }
        for _ in 0..7 {
            t.tick(DT);
            fires += t.just_finished() as u32;
        }
        assert_eq!(fires, 1);
        assert!(t.finished());
    }

    #[test]
    fn disarmed_and_zero_duration_never_fire() {
        let mut d = StepTimer::disarmed();
        d.tick(DT);
        assert!(!d.just_finished());
        assert!(d.finished());
        assert_eq!(d.fraction(), 1.0);

        let mut z = StepTimer::from_seconds(0.0, TimerMode::Once);
        for _ in 0..10 {
            z.tick(DT);
            assert!(!z.just_finished());
        }
    }

    #[test]
    fn repeating_waits_the_period_then_fires() {
        let mut t = StepTimer::from_seconds(4.0 * DT, TimerMode::Repeating);
        let mut fires = 0;
        for i in 0..6 {
            t.tick(DT);
            if i < 3 {
                assert!(!t.just_finished());
            }
            fires += t.just_finished() as u32;
        }
        assert_eq!(fires, 1);
    }

    #[test]
    fn fraction_and_remaining_track_progress() {
        let mut t = StepTimer::from_seconds(4.0 * DT, TimerMode::Once);
        t.tick(DT);
        assert!((t.fraction() - 0.25).abs() < 1e-6);
        assert!((t.remaining_secs() - 3.0 * DT).abs() < 1e-6);
        t.tick(10.0 * DT);
        assert_eq!(t.fraction(), 1.0);
        assert_eq!(t.remaining_secs(), 0.0);
    }

    #[test]
    fn reset_rearms() {
        let mut t = StepTimer::from_seconds(2.0 * DT, TimerMode::Once);
        for _ in 0..4 {
            t.tick(DT);
        }
        assert!(t.finished());
        t.reset();
        assert!(!t.finished());
        let mut fires = 0;
        for _ in 0..4 {
            t.tick(DT);
            fires += t.just_finished() as u32;
        }
        assert_eq!(fires, 1);
    }
}
