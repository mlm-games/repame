//! Fx timebase and clock split: one place for the seconds/ticks mapping.
//!
//! The canonical stepping unit across this crate is **seconds** (`dt_secs`,
//! ages and lives stored as seconds). The `*_ticks` wrappers exist for 100 Hz
//! games and convert with [`ticks_to_secs_100hz`]; do not invent a second
//! mapping.
//!
//! [`SimDriver`] encodes the transition-gating split: fx steps on the raw
//! (ungated) dt every frame so a transition always unblocks, while sim
//! gameplay ticks only run when the game is not blocked. Games that drive
//! `Sim::tick` by hand should mirror this ordering instead of re-deriving it.

/// Seconds per 100 Hz quantum. Only conversion in this crate.
pub const SECS_PER_TICK_100HZ: f32 = 0.01;

/// 100 Hz quanta to seconds. Non-positive input clamps to 0.
pub fn ticks_to_secs_100hz(ticks: i32) -> f32 {
    (ticks.max(0) as f32) * SECS_PER_TICK_100HZ
}

/// Seconds to whole 100 Hz quanta (floored, never negative).
/// `0.01f32` is `0.0099999997...`, so the raw quotient sits a hair *above*
/// the integer (0.4 s -> 40.0000015): the epsilon only matters for values
/// just below a boundary from genuine float error, and cannot overshoot a
/// true fractional value (e.g. 0.015 s -> 1.5 stays 1).
pub fn secs_to_ticks_100hz(dt_secs: f32) -> i32 {
    if !dt_secs.is_finite() || dt_secs <= 0.0 {
        return 0;
    }
    (dt_secs / SECS_PER_TICK_100HZ + 1e-4).floor() as i32
}

/// Carry fractional 100 Hz quanta across frames: add `dt_secs`, take the
/// whole count, keep the remainder. Returns whole ticks to run.
///
/// Precision: the carry is `f32` and quanta (`0.01`) are not exactly
/// representable, so the remainder drifts ~1 quantum per several minutes
/// of small-dt accumulation (a sub-frame phase error only: whole-tick
/// counts stay exact, and `step_*_secs` paths never touch this). Games
/// that need zero long-run drift should accumulate in `f64` or integer
/// nanoseconds game-side (like `Sim::step`) and convert once per frame.
pub fn accumulate_ticks(carry_secs: &mut f32, dt_secs: f32) -> i32 {
    if !dt_secs.is_finite() || dt_secs <= 0.0 {
        return 0;
    }
    *carry_secs += dt_secs;
    let whole = secs_to_ticks_100hz(*carry_secs);
    *carry_secs -= whole as f32 * SECS_PER_TICK_100HZ;
    if *carry_secs < 0.0 {
        *carry_secs = 0.0;
    }
    whole
}

/// Outcome of one gated frame: how many gameplay ticks ran and whether fx
/// advanced (fx always advances, even while blocked).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FxTick {
    pub ran: u32,
    pub blocked: bool,
}

/// Gate helper: step ungated fx first, then tick the sim only when open.
///
/// `step_fx` runs unconditionally (transitions, trauma decay, flash), `blocked`
/// decides whether `tick_sim` runs, `clear_edges` runs when it does not so a
/// paused frame never double-fires an edge on resume. Stateless: callers keep
/// their own accumulator.
pub struct SimDriver;

impl SimDriver {
    pub fn frame(
        mut step_fx: impl FnMut(),
        blocked: bool,
        mut tick_sim: impl FnMut(),
        mut clear_edges: impl FnMut(),
    ) -> FxTick {
        step_fx();
        if blocked {
            clear_edges();
            FxTick {
                ran: 0,
                blocked: true,
            }
        } else {
            tick_sim();
            FxTick {
                ran: 1,
                blocked: false,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_round_trip_seconds() {
        assert!((ticks_to_secs_100hz(40) - 0.4).abs() < 1e-6);
        assert_eq!(secs_to_ticks_100hz(0.4), 40);
        assert_eq!(secs_to_ticks_100hz(-1.0), 0);
        assert_eq!(ticks_to_secs_100hz(-5), 0.0);
    }

    #[test]
    fn carry_keeps_fraction_for_next_frame() {
        let mut carry = 0.0;
        assert_eq!(accumulate_ticks(&mut carry, 0.015), 1);
        assert!((carry - 0.005).abs() < 1e-6, "keeps 5 ms, got {carry}");
        assert_eq!(accumulate_ticks(&mut carry, 0.005), 1);
        assert!(carry.abs() < 1e-6, "drains, got {carry}");
    }

    #[test]
    fn blocked_frame_steps_fx_and_rolls_edges() {
        let mut fx = 0;
        let mut sim = 0;
        let mut edges = 0;
        let out = SimDriver::frame(|| fx += 1, true, || sim += 1, || edges += 1);
        assert_eq!((fx, sim, edges), (1, 0, 1));
        assert_eq!(out, FxTick { ran: 0, blocked: true });
        let out = SimDriver::frame(|| fx += 1, false, || sim += 1, || edges += 1);
        assert_eq!((fx, sim, edges), (2, 1, 1));
        assert_eq!(out, FxTick { ran: 1, blocked: false });
    }
}
