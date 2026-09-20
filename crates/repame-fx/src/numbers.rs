//! Damage floaters: rising text rendered through the canvas text path.
//! Timebase is seconds (`step_numbers_secs`); `step_numbers` is the 100 Hz shim.

use repame_sim::bevy_ecs::component::{Mutable, StorageType};
use repame_sim::bevy_ecs::prelude::*;

/// One floating label. Rises `rise_pps` while alive, then despawns.
#[derive(Clone, Debug)]
pub struct DamageNumber {
    pub text: String,
    pub x: f32,
    pub y: f32,
    pub age_secs: f32,
    pub life_secs: f32,
    pub rise_pps: f32,
    pub color: [f32; 4],
    /// Legacy 100 Hz mirrors; new code reads `age_secs`/`life_secs`.
    pub age_ticks: i32,
    pub life_ticks: i32,
}

impl Component for DamageNumber {
    const STORAGE_TYPE: StorageType = StorageType::Table;
    type Mutability = Mutable;
}

/// Spawn a floater. Returns the entity for game tagging.
/// Lifetime is 0.8 s; the tick shim writes the legacy mirrors.
/// Spawn a floater with an explicit lifetime. Returns the entity.
pub fn spawn_number_secs(
    commands: &mut Commands,
    x: f32,
    y: f32,
    text: impl Into<String>,
    color: [f32; 4],
    life_secs: f32,
) -> Entity {
    let life_secs = if life_secs.is_finite() {
        life_secs.max(1.0 / 100.0)
    } else {
        0.8
    };
    commands
        .spawn((DamageNumber {
            text: text.into(),
            x,
            y,
            age_secs: 0.0,
            life_secs,
            age_ticks: 0,
            life_ticks: (life_secs / super::driver::SECS_PER_TICK_100HZ).ceil().max(1.0) as i32,
            rise_pps: 40.0,
            color,
        },))
        .id()
}

/// Spawn a floater. Returns the entity for game tagging.
pub fn spawn_number(
    commands: &mut Commands,
    x: f32,
    y: f32,
    text: impl Into<String>,
    color: [f32; 4],
) -> Entity {
    spawn_number_secs(commands, x, y, text, color, 0.8)
}

/// Rise and despawn the spent.
pub fn step_numbers_secs(
    commands: &mut Commands,
    numbers: &mut Query<(Entity, &mut DamageNumber)>,
    dt_secs: f32,
) {
    if !dt_secs.is_finite() || dt_secs <= 0.0 {
        return;
    }
    for (e, mut n) in numbers {
        n.age_secs += dt_secs;
        if n.age_secs + 1e-6 >= n.life_secs {
            commands.entity(e).try_despawn();
            continue;
        }
        n.y -= n.rise_pps * dt_secs;
        n.age_ticks = (n.age_secs / super::driver::SECS_PER_TICK_100HZ).floor() as i32;
    }
}

/// Legacy 100 Hz wrapper over [`step_numbers_secs`].
pub fn step_numbers(
    commands: &mut Commands,
    numbers: &mut Query<(Entity, &mut DamageNumber)>,
    ticks: i32,
) {
    step_numbers_secs(
        commands,
        numbers,
        super::driver::ticks_to_secs_100hz(ticks),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use repame_sim::bevy_ecs::system::RunSystemOnce;

    fn spawn(world: &mut World, x: f32, y: f32) {
        let _ = world.run_system_once(move |mut cmds: Commands| {
            spawn_number(&mut cmds, x, y, "+25", [1.0, 1.0, 0.0, 1.0]);
        });
    }

    fn step(world: &mut World, ticks: i32) {
        let _ = world.run_system_once(
            move |mut q: Query<(Entity, &mut DamageNumber)>, mut cmds: Commands| {
                step_numbers(&mut cmds, &mut q, ticks);
            },
        );
    }

    #[test]
    fn numbers_rise_and_expire() {
        let mut world = World::new();
        spawn(&mut world, 5.0, 10.0);
        assert_eq!(world.query::<&DamageNumber>().iter(&world).count(), 1);
        step(&mut world, 40);
        let mut q = world.query::<&DamageNumber>();
        let n = q.iter(&world).next().unwrap();
        assert!(n.y < 10.0);
        assert!((n.age_secs - 0.4).abs() < 1e-6);
        step(&mut world, 40);
        assert_eq!(world.query::<&DamageNumber>().iter(&world).count(), 0);
    }

    #[test]
    fn secs_stepper_matches_tick_shim() {
        let mut a = World::new();
        let mut b = World::new();
        spawn(&mut a, 0.0, 10.0);
        spawn(&mut b, 0.0, 10.0);
        step(&mut a, 20);
        let _ = b.run_system_once(
            move |mut q: Query<(Entity, &mut DamageNumber)>, mut cmds: Commands| {
                step_numbers_secs(&mut cmds, &mut q, 0.2);
            },
        );
        let ya = a.query::<&DamageNumber>().iter(&a).next().unwrap().y;
        let yb = b.query::<&DamageNumber>().iter(&b).next().unwrap().y;
        assert!((ya - yb).abs() < 1e-6, "tick shim matches, {ya} vs {yb}");
    }
}
