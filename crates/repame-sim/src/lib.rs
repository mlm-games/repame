//! Headless sim: `bevy_ecs` world stepped without a renderer.
//! Game owns `Sim`, feeds wall time via `Sim::step`,
//! then reads a snapshot for the viewport.
//!
//! Determinism stance: `bevy_ecs` query iteration follows entity allocation
//! history, so a replay with an identical spawn history iterates
//! identically. The sim does NOT sort systems' iteration for you: any
//! gameplay outcome that depends on iteration order (damage application,
//! projectile-vs-multiple-enemies resolution) must sort explicitly in the
//! affected system, or replays diverge the moment allocation history does.
//! Cross-platform float equality (notably `Perlin`/`sin`/`powf` in fx) is
//! additionally *not* guaranteed between native libm and wasm; same-binary
//! replay is the supported claim, cross-target equality needs a harness
//! (run one trace twice, hash the world per tick, diff).

use web_time::Duration;

use bevy_ecs::prelude::*;
use bevy_ecs::schedule::{IntoScheduleConfigs, Schedule};
use bevy_ecs::system::ScheduleSystem;

pub use bevy_ecs;

/// Fixed-step sim state. Game registers systems,
/// feeds wall time with `step`, reads `world` for the snapshot.
pub struct Sim {
    /// Entity/component store. Game crates add their own components here.
    pub world: World,
    schedule: Schedule,
    accumulator: Duration,
    /// Fixed step size. Default 60 Hz.
    ///
    /// Each tick adds this to `SimTime`. Set before first `step`.
    pub step: Duration,
    /// Max ticks per `step` call. Default `8`.
    ///
    /// Caps catch-up after hitches. Excess wall time is dropped.
    /// Clamped to a minimum of `1` so the sim keeps moving.
    pub max_steps: u32,
}

/// Sim time, advanced once per fixed step.
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

    /// Register systems with a fixed execution order.
    ///
    /// Use one `(a, b, c).chain()` tuple for pipelines that need order.
    pub fn add_chained_systems<M>(
        &mut self,
        systems: impl IntoScheduleConfigs<ScheduleSystem, M>,
    ) -> &mut Self {
        self.schedule.add_systems(systems);
        self
    }

    /// Advance by `dt` wall time. Returns ticks run.
    ///
    /// Small `dt` banks for later. Large `dt` runs up to `max_steps`
    /// ticks, then drops leftover whole steps. Example:
    ///
    /// ```ignore
    /// let steps = sim.step(frame_dt);
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
            // Hitch: keep only the sub-step fraction for alpha continuity.
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

    /// Unconsumed time carried to the next frame. Below one step.
    pub fn leftover(&self) -> Duration {
        self.accumulator
    }

    /// Blend factor `accumulator / step`, clamped to `0..1`.
    /// Zero step returns `0.0`. For render blending between snapshots: keep
    /// the previous snapshot and lerp toward the current one by this alpha
    /// (NT motion at 30 Hz sim on 144 Hz display needs it; without it every
    /// fast mover judders on 4-5 identical frames then jumps).
    pub fn alpha(&self) -> f32 {
        let step = self.step.as_secs_f64();
        if step <= 0.0 {
            return 0.0;
        }
        (self.accumulator.as_secs_f64() / step).clamp(0.0, 1.0) as f32
    }

    /// Run the schedule once, advancing sim time by one step.
    /// Tick-model games drive this directly instead of `step`.
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
        assert!(
            sim.leftover() < sim.step,
            "keeps only fraction, got {:?}",
            sim.leftover()
        );
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
