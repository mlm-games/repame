//! Headless sim: a `bevy_ecs` world stepped without a renderer.
//!
//! The game owns a [`Sim`], advances it with fixed-timestep [`Sim::step`],
//! and pulls a snapshot out each frame for the viewport (see `repame-sprite`).
//! Systems are plain functions registered on the schedule; no `bevy_app`,
//! no window, no renderer. The sim stays portable to full Bevy later.

use std::time::Duration;

use bevy_ecs::prelude::*;
use bevy_ecs::schedule::Schedule;

pub use bevy_ecs;

/// Fixed-timestep simulation state.
pub struct Sim {
    /// Entity/component store. Game crates add their own components here.
    pub world: World,
    schedule: Schedule,
    accumulator: Duration,
    /// Fixed step size. Defaults to 60 Hz.
    pub step: Duration,
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

    /// Advance the sim by `dt` wall time, running zero or more fixed steps.
    /// Returns the number of steps run.
    pub fn step(&mut self, dt: Duration) -> u32 {
        self.accumulator += dt;
        let mut ran = 0;
        while self.accumulator >= self.step {
            self.accumulator -= self.step;
            {
                let mut time = self.world.resource_mut::<SimTime>();
                time.delta_secs = self.step.as_secs_f32();
                time.elapsed_secs += self.step.as_secs_f64();
            }
            self.schedule.run(&mut self.world);
            ran += 1;
        }
        ran
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
}
