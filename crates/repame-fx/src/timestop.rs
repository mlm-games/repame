//! Virtual-time dips: hit-stop and slow-motion.
//!
//! Both resources describe a *requested* virtual-time scale; the game reads
//! [`HitStop::scale`] / [`SlowMotion::scale`] and multiplies its gated sim
//! delta. Recovery runs on the real fixed clock -- unaffected by the dip
//! itself -- so a stop can never wedge the clock. Step them from the ungated
//! schedule: the gameplay that consumes the scale is what pauses.
//!
//! Hit-stop arbitration while a dip is live: a weaker or equal trigger is
//! ignored, a stronger one replaces it, so the strongest dip wins and the
//! clock eases back to 1.0 exactly once.

use bevy_ecs::prelude::*;
use repame_sim::{StepTimer, TimerMode};

/// Hit-stop dip state.
#[derive(Resource, Clone, Debug)]
pub struct HitStop {
    /// Whether a dip is currently active.
    pub active: bool,
    /// Virtual-time speed factor while recovering.
    pub scale: f32,
    /// Recovery timer (real time, unaffected by the dip itself).
    pub recover: StepTimer,
    /// Initial dip scale applied immediately on trigger.
    pub start_scale: f32,
}

impl Default for HitStop {
    fn default() -> Self {
        Self {
            active: false,
            scale: 1.0,
            recover: StepTimer::disarmed(),
            start_scale: 1.0,
        }
    }
}

impl HitStop {
    /// Dip virtual time to `scale` for `recover_secs` real seconds.
    pub fn trigger(&mut self, scale: f32, recover_secs: f32) {
        let scale = scale.clamp(0.01, 1.0);
        if self.active && scale >= self.scale {
            return;
        }
        self.start_scale = scale;
        self.scale = self.start_scale;
        self.recover = StepTimer::from_seconds(recover_secs.max(0.0), TimerMode::Once);
        self.active = true;
    }

    pub fn cancel(&mut self) {
        self.active = false;
        self.scale = 1.0;
    }

    /// Advance recovery on the real fixed clock; eases `scale` back to 1.
    pub fn step_secs(&mut self, dt_secs: f32) {
        if !self.active {
            return;
        }
        self.recover.tick(dt_secs);
        if self.recover.finished() {
            self.active = false;
            self.scale = 1.0;
        } else {
            let f = self.recover.fraction().clamp(0.0, 1.0);
            self.scale = self.start_scale + (1.0 - self.start_scale) * f;
        }
    }
}

/// Slow-motion state.
#[derive(Resource, Clone, Debug)]
pub struct SlowMotion {
    pub active: bool,
    pub scale: f32,
    pub timer: StepTimer,
}

impl Default for SlowMotion {
    fn default() -> Self {
        Self {
            active: false,
            scale: 1.0,
            timer: StepTimer::disarmed(),
        }
    }
}

/// Engage slow motion at `scale` for `duration_secs`.
pub fn slow_motion(slow_mo: &mut SlowMotion, scale: f32, duration_secs: f32) {
    slow_mo.scale = scale.clamp(0.01, 1.0);
    slow_mo.timer = StepTimer::from_seconds(duration_secs.max(0.0), TimerMode::Once);
    slow_mo.active = true;
}

impl SlowMotion {
    /// Advance the hold on the real fixed clock; clears back to 1.0 at the
    /// end.
    pub fn step_secs(&mut self, dt_secs: f32) {
        if !self.active {
            return;
        }
        self.timer.tick(dt_secs);
        if self.timer.finished() {
            self.active = false;
            self.scale = 1.0;
        }
    }
}

/// Advance both dips on the real fixed clock (`dt_secs`); call from the
/// ungated schedule so recovery keeps running while gameplay is paused.
pub fn step_hitstop_slowmo_secs(
    hs: Option<&mut HitStop>,
    sm: Option<&mut SlowMotion>,
    dt_secs: f32,
) {
    if let Some(hs) = hs {
        hs.step_secs(dt_secs);
    }
    if let Some(sm) = sm {
        sm.step_secs(dt_secs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 30.0;

    #[test]
    fn hitstop_keeps_the_strongest_dip() {
        let mut hs = HitStop::default();
        hs.trigger(0.2, 1.0);
        assert!(hs.active);
        assert!((hs.scale - 0.2).abs() < 1e-6);
        // Weaker (higher scale) trigger is ignored while the dip is live.
        hs.trigger(0.5, 1.0);
        assert!((hs.scale - 0.2).abs() < 1e-6);
        // Stronger trigger replaces it.
        hs.trigger(0.05, 1.0);
        assert!((hs.scale - 0.05).abs() < 1e-6);
    }

    #[test]
    fn hitstop_eases_out_and_clears() {
        let mut hs = HitStop::default();
        hs.trigger(0.0, 4.0 * DT);
        hs.step_secs(DT);
        assert!(hs.active);
        assert!(hs.scale > 0.0 && hs.scale < 1.0, "easing: {}", hs.scale);
        for _ in 0..4 {
            hs.step_secs(DT);
        }
        assert!(!hs.active);
        assert_eq!(hs.scale, 1.0);
    }

    #[test]
    fn hitstop_cancel_restores_full_speed() {
        let mut hs = HitStop::default();
        hs.trigger(0.1, 10.0);
        hs.cancel();
        assert!(!hs.active);
        assert_eq!(hs.scale, 1.0);
    }

    #[test]
    fn slow_motion_holds_then_expires() {
        let mut sm = SlowMotion::default();
        slow_motion(&mut sm, 0.4, 3.0 * DT);
        assert!(sm.active);
        assert!((sm.scale - 0.4).abs() < 1e-6);
        for _ in 0..4 {
            sm.step_secs(DT);
        }
        assert!(!sm.active);
        assert_eq!(sm.scale, 1.0);
    }

    #[test]
    fn scale_clamps_to_a_usable_range() {
        let mut sm = SlowMotion::default();
        slow_motion(&mut sm, 4.0, 1.0);
        assert_eq!(sm.scale, 1.0);
        slow_motion(&mut sm, -1.0, 1.0);
        assert_eq!(sm.scale, 0.01);
    }

    #[test]
    fn combined_step_skips_missing_resources() {
        let mut hs = HitStop::default();
        hs.trigger(0.0, 2.0 * DT);
        for _ in 0..3 {
            step_hitstop_slowmo_secs(Some(&mut hs), None, DT);
        }
        assert!(!hs.active);
    }
}
