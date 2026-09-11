//! Headless sim: a `bevy_ecs` world stepped without a renderer.
//!
//! The game owns a [`Sim`], advances it with fixed-timestep [`Sim::step`],
//! and pulls a snapshot out each frame for the viewport (see `repame-sprite`).
//! Systems are plain functions registered on the schedule; no `bevy_app`,
//! no window, no renderer. The sim stays portable to full Bevy later.

use web_time::Duration;

use bevy_ecs::prelude::*;
use bevy_ecs::schedule::{IntoScheduleConfigs, Schedule};
use bevy_ecs::system::ScheduleSystem;

pub use bevy_ecs;

/// Fixed-timestep simulation state.
///
/// The game owns a `Sim`, registers plain systems on its schedule, feeds it
/// wall-clock time every frame with [`Sim::step`], and pulls a snapshot out
/// for the viewport. There is no window, no renderer, and no app object.
pub struct Sim {
    /// Entity/component store. Game crates add their own components here.
    pub world: World,
    schedule: Schedule,
    accumulator: Duration,
    /// Fixed step size. Defaults to 60 Hz (see [`Sim::with_default_step`]).
    ///
    /// Each [`Sim::tick`] advances [`SimTime`] by exactly this amount, so
    /// gameplay stays deterministic regardless of frame rate. Change it
    /// before the first [`Sim::step`] call; changing it mid-run rescales
    /// future ticks but leaves already-simulated time untouched.
    pub step: Duration,
    /// Maximum fixed steps per [`Sim::step`] call. Defaults to `8`.
    ///
    /// Guards against the spiral of death: after a long hitch (app resume,
    /// debugger pause, slow device), the sim runs at most this many catch-up
    /// ticks and drops the excess wall time instead of scheduling dozens of
    /// ticks that make the next frame even longer. The dropped time is gone,
    /// not carried over. The sim slows down instead of freezing.
    ///
    /// **Note:** clamped to a minimum of `1` per call, so the sim always
    /// makes progress while wall time keeps arriving.
    pub max_steps: u32,
}

/// Sim time resource, advanced once per fixed step.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct SimTime {
    /// Total simulated seconds.
    pub elapsed_secs: f64,
    /// Fixed delta of the current step.
    pub delta_secs: f32,
}

impl Sim {
    pub fn new(step: Duration) -> Self {
        let mut world = World::new();
        world.init_resource::<SimTime>();
        Self {
            world,
            schedule: Schedule::default(),
            accumulator: Duration::ZERO,
            step,
            max_steps: 8,
        }
    }

    /// 60 Hz fixed-step sim.
    pub fn with_default_step() -> Self {
        Self::new(Duration::from_secs_f64(1.0 / 60.0))
    }

    /// Register a system into the per-step schedule.
    pub fn add_system<M>(&mut self, system: impl IntoSystem<(), (), M>) -> &mut Self {
        self.schedule.add_systems(system);
        self
    }

    /// Register systems with a guaranteed execution order.
    ///
    /// A bare `Schedule` does not preserve `add_system` insertion order for
    /// systems with conflicting accesses, so multi-step pipelines (economy,
    /// combat) must register as one `(a, b, c).chain()` tuple through here
    /// instead of separate `add_system` calls.
    pub fn add_chained_systems<M>(
        &mut self,
        systems: impl IntoScheduleConfigs<ScheduleSystem, M>,
    ) -> &mut Self {
        self.schedule.add_systems(systems);
        self
    }

    /// Advance the sim by `dt` wall time, running zero or more fixed steps.
    /// Returns the number of steps run.
    ///
    /// Wall time accumulates across calls: a `dt` smaller than [`Sim::step`]
    /// banks its fraction for later, and a large `dt` runs several ticks at
    /// once. At most [`Sim::max_steps`] ticks run per call; any time still
    /// banked beyond that is dropped (see [`Sim::max_steps`]).
    ///
    /// Typical frame pump:
    ///
    /// ```ignore
    /// let steps = sim.step(frame_dt);
    /// // `steps` ticks ran; build the viewport snapshot from `sim.world`.
    /// ```
    pub fn step(&mut self, dt: Duration) -> u32 {
        if self.step.is_zero() {
            return 0;
        }
        self.accumulator += dt;
        let mut ran = 0;
        while self.accumulator >= self.step && ran < self.max_steps.max(1) {
            self.accumulator -= self.step;
            self.tick();
            ran += 1;
        }
        if self.accumulator >= self.step {
            // On hitch: drop only *whole* extra steps, keep fraction for
            // alpha continuity.
            let step_ns = self.step.as_nanos();
            if step_ns > 0 {
                let acc_ns = self.accumulator.as_nanos() % step_ns;
                self.accumulator = Duration::from_nanos(acc_ns as u64);
            } else {
                self.accumulator = Duration::ZERO;
            }
        }
        ran
    }

    /// Unconsumed fractional time carried to the next frame.
    ///
    /// Always less than one [`Sim::step`] (a full step would have ticked).
    /// Returns [`Duration::ZERO`] right after a step boundary; after a
    /// hitch that hit the [`Sim::max_steps`] cap only the sub-step
    /// fraction is kept (whole extra steps are dropped, alpha stays
    /// continuous).
    pub fn leftover(&self) -> Duration {
        self.accumulator
    }

    /// Blend factor between the last ticked state and the next one, from
    /// `0.0` (just ticked) to `1.0` (a full step banked, tick imminent).
    ///
    /// Computed as `accumulator / step`, clamped to `0..1`; returns `0.0`
    /// for a degenerate (zero) step size. The sim itself never interpolates.
    /// Snapshots read the last ticked state, but games that keep the
    /// previous snapshot can blend toward the current one with this factor
    /// for smooth rendering.
    ///
    /// **Note:** after a capped hitch (see [`Sim::max_steps`]) only whole
    /// steps are dropped, so `alpha` keeps the leftover fraction.
    pub fn alpha(&self) -> f32 {
        let step = self.step.as_secs_f64();
        if step <= 0.0 {
            return 0.0;
        }
        (self.accumulator.as_secs_f64() / step).clamp(0.0, 1.0) as f32
    }

    /// Run the schedule exactly once, advancing sim time by one step.
    /// Tick-model games (integer logic steps) drive this directly instead
    /// of the wall-clock [`Sim::step`] accumulator.
    pub fn tick(&mut self) {
        {
            let mut time = self.world.resource_mut::<SimTime>();
            time.delta_secs = self.step.as_secs_f32();
            time.elapsed_secs += self.step.as_secs_f64();
        }
        self.schedule.run(&mut self.world);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_steps_accumulate() {
        let mut sim = Sim::with_default_step();
        let ran = sim.step(Duration::from_nanos(50_000_001));
        assert_eq!(ran, 3);
        let time = sim.world.resource::<SimTime>();
        assert!((time.elapsed_secs - 3.0 / 60.0).abs() < 1e-9);
    }

    #[test]
    fn zero_step_never_ticks() {
        let mut sim = Sim::new(Duration::ZERO);
        let ran = sim.step(Duration::from_secs(10));
        assert_eq!(ran, 0);
        let time = sim.world.resource::<SimTime>();
        assert_eq!(time.elapsed_secs, 0.0);
    }

    #[test]
    fn spiral_of_death_is_capped() {
        let mut sim = Sim::with_default_step();
        sim.max_steps = 4;
        let ran = sim.step(Duration::from_secs(10));
        assert_eq!(ran, 4);
        assert!(sim.leftover() < sim.step, "keeps only fraction, got {:?}", sim.leftover());
    }

    #[test]
    fn alpha_is_interpolation_fraction() {
        let mut sim = Sim::new(Duration::from_millis(16));
        sim.step(Duration::from_millis(8));
        assert!((sim.alpha() - 0.5).abs() < 1e-6, "got {}", sim.alpha());
        sim.step(Duration::from_millis(8));
        assert!(sim.alpha() < 1e-6, "step boundary, got {}", sim.alpha());
    }

    #[test]
    fn chained_systems_run_in_order() {
        use std::sync::{Arc, Mutex};
        let log: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
        let (a_log, b_log, c_log) = (log.clone(), log.clone(), log.clone());
        let mut sim = Sim::with_default_step();
        sim.add_chained_systems(
            (
                move || a_log.lock().unwrap().push("a"),
                move || b_log.lock().unwrap().push("b"),
                move || c_log.lock().unwrap().push("c"),
            )
                .chain(),
        );
        sim.tick();
        assert_eq!(*log.lock().unwrap(), vec!["a", "b", "c"]);
    }
}
