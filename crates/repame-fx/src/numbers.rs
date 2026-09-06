//! Damage floaters: rising text pops (`+25`, `1800`) rendered by
//! the game through its canvas text path. Plain data + step fn.

use repame_sim::bevy_ecs::component::{Mutable, StorageType};
use repame_sim::bevy_ecs::prelude::*;

/// One floating label. Rises `rise_pps` while alive, then despawns.
#[derive(Clone, Debug)]
pub struct DamageNumber {
    pub text: String,
    pub x: f32,
    pub y: f32,
    pub age_ticks: i32,
    pub life_ticks: i32,
    pub rise_pps: f32,
    pub color: [f32; 4],
}

impl Component for DamageNumber {
    const STORAGE_TYPE: StorageType = StorageType::Table;
    type Mutability = Mutable;
}

/// Spawn a floater. Returns the entity for game tagging.
pub fn spawn_number(
    commands: &mut Commands,
    x: f32,
    y: f32,
    text: impl Into<String>,
    color: [f32; 4],
) -> Entity {
    commands
        .spawn((DamageNumber {
            text: text.into(),
            x,
            y,
            age_ticks: 0,
            life_ticks: 80,
            rise_pps: 40.0,
            color,
        },))
        .id()
}

/// Rise and despawn the spent.
pub fn step_numbers(
    commands: &mut Commands,
    numbers: &mut Query<(Entity, &mut DamageNumber)>,
    ticks: i32,
) {
    if ticks <= 0 {
        return;
    }
    let dt = ticks as f32 / 100.0;
    for (e, mut n) in numbers {
        n.age_ticks += ticks;
        if n.age_ticks >= n.life_ticks {
            commands.entity(e).try_despawn();
            continue;
        }
        n.y -= n.rise_pps * dt;
    }
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
        step(&mut world, 40);
        assert_eq!(world.query::<&DamageNumber>().iter(&world).count(), 0);
    }
}
