//! CPU particles on fixed 100 Hz ticks: spawners plus one-shot
//! bursts, integrated with gravity/drag, rendered as tinted rects
//! through [`SpriteInstance`]. (hanabi's spawner + modifier chain and
//! enoki's CPU-sim/instancing model, flattened for canvas sprites.)
//!
//! Step fns take whole ticks (`i32`), not a game clock type, so any
//! fixed-step game can drive them.

use glam::Vec2;
use rand::{Rng, RngExt};
use repame_sim::bevy_ecs::component::{Mutable, StorageType};
use repame_sim::bevy_ecs::prelude::*;
use repame_sprite::SpriteInstance;

use super::effect::EffectDef;

/// One live particle. Age/life in 100 Hz ticks; deterministic.
#[derive(Clone, Debug)]
pub struct Particle {
    pub pos: [f32; 2],
    pub vel: [f32; 2],
    pub age_ticks: i32,
    pub life_ticks: i32,
    pub size_px: f32,
    pub gravity_pps2: f32,
    pub drag_per_sec: f32,
    pub gradient: super::effect::Gradient,
    pub ease: super::effect::EaseKind,
}

impl Component for Particle {
    const STORAGE_TYPE: StorageType = StorageType::Table;
    type Mutability = Mutable;
}

/// Continuous emitter: rate accumulates into whole particles.
/// Cap is enforced globally by [`tick_spawners`] (`max_total`).
#[derive(Clone, Debug)]
pub struct Spawner {
    pub def: EffectDef,
    pub x: f32,
    pub y: f32,
    pub acc: f32,
}

impl Component for Spawner {
    const STORAGE_TYPE: StorageType = StorageType::Table;
    type Mutability = Mutable;
}

/// Spawn one particle from a def at `(x, y)`. Returns the entity so
/// the game can tag it (cleanup groups, layers).
pub fn spawn_particle(
    commands: &mut Commands,
    x: f32,
    y: f32,
    def: &EffectDef,
    rng: &mut impl Rng,
) -> Entity {
    let angle = rng.random_range(0.0..std::f32::consts::TAU);
    let speed = def.speed_pps.sample(rng);
    commands
        .spawn((Particle {
            pos: [x, y],
            vel: [angle.cos() * speed, angle.sin() * speed],
            age_ticks: 0,
            life_ticks: def.lifetime_ticks.sample(rng).max(1.0) as i32,
            size_px: def.size_px.sample(rng).max(1.0),
            gravity_pps2: def.gravity_pps2,
            drag_per_sec: def.drag_per_sec,
            gradient: def.gradient.clone(),
            ease: def.ease,
        },))
        .id()
}

/// One-shot burst of `count` particles (pea puffs, explosions).
pub fn burst(
    commands: &mut Commands,
    x: f32,
    y: f32,
    def: &EffectDef,
    count: usize,
    rng: &mut impl Rng,
) {
    for _ in 0..count {
        spawn_particle(commands, x, y, def, rng);
    }
}

/// Advance emitter clocks; spawn whole particles at the def rate.
/// `ticks` scales accumulation (0 pauses). `max_total` caps the
/// global live count so runaway emitters can't flood the world.
pub fn tick_spawners(
    commands: &mut Commands,
    spawners: &mut Query<(Entity, &mut Spawner)>,
    live: usize,
    max_total: usize,
    ticks: i32,
    rng: &mut impl Rng,
) {
    if ticks <= 0 {
        return;
    }
    let mut spawned = 0usize;
    for (_, mut spawner) in spawners {
        if spawner.def.spawner.rate_per_sec <= 0.0 {
            continue;
        }
        spawner.acc += spawner.def.spawner.rate_per_sec * ticks as f32 / 100.0;
        while spawner.acc >= 1.0 {
            spawner.acc -= 1.0;
            if live + spawned >= max_total {
                spawner.acc = 0.0;
                break;
            }
            let (x, y) = (spawner.x, spawner.y);
            let def = spawner.def.clone();
            spawn_particle(commands, x, y, &def, rng);
            spawned += 1;
        }
    }
}

/// Integrate motion, age, and despawn the spent. Gravity pulls +y
/// (canvas y-down); drag is exponential per second.
pub fn step_particles(
    commands: &mut Commands,
    particles: &mut Query<(Entity, &mut Particle)>,
    ticks: i32,
) {
    if ticks <= 0 {
        return;
    }
    let dt = ticks as f32 / 100.0;
    for (e, mut p) in particles {
        p.age_ticks += ticks;
        if p.age_ticks >= p.life_ticks {
            commands.entity(e).try_despawn();
            continue;
        }
        let drag = (1.0 - p.drag_per_sec * dt).max(0.0);
        p.vel[0] *= drag;
        p.vel[1] = p.vel[1] * drag + p.gravity_pps2 * dt;
        p.pos[0] += p.vel[0] * dt;
        p.pos[1] += p.vel[1] * dt;
    }
}

/// Map live particles to sprite instances: gradient color at life
/// fraction, size shrinking along the ease curve. Takes an iterator
/// so both systems (`query.iter()`) and tests can feed it.
pub fn particle_sprites<'a>(particles: impl Iterator<Item = &'a Particle>) -> Vec<SpriteInstance> {
    particles
        .map(|p| {
            let t = p.age_ticks as f32 / p.life_ticks.max(1) as f32;
            let shrink = 1.0 - p.ease.apply(t);
            let s = (p.size_px * shrink).max(0.5);
            SpriteInstance {
                center: Vec2::new(p.pos[0], p.pos[1]),
                rotation: 0.0,
                size: Vec2::new(s, s),
                uv_min: Vec2::ZERO,
                uv_max: Vec2::ONE,
                color: p.gradient.sample(t),
                page: 0,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effect::{Gradient, Jittered, SpawnerDef};
    use rand::SeedableRng;
    use repame_sim::bevy_ecs::system::RunSystemOnce;

    fn rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(7)
    }

    fn def() -> EffectDef {
        EffectDef {
            spawner: SpawnerDef {
                rate_per_sec: 100.0,
                max_alive: 8,
            },
            speed_pps: Jittered::exact(0.0),
            lifetime_ticks: Jittered::exact(50.0),
            size_px: Jittered::exact(4.0),
            gravity_pps2: 0.0,
            drag_per_sec: 0.0,
            gradient: Gradient::solid([1.0, 0.0, 0.0, 1.0]),
            ease: crate::effect::EaseKind::Linear,
        }
    }

    fn step(world: &mut World, ticks: i32) {
        let _ = world.run_system_once(
            move |mut q: Query<(Entity, &mut Particle)>, mut cmds: Commands| {
                step_particles(&mut cmds, &mut q, ticks);
            },
        );
    }

    fn burst_n(world: &mut World, x: f32, y: f32, d: &EffectDef, n: usize) {
        let mut rng = rng();
        let d = d.clone();
        let _ = world.run_system_once(move |mut cmds: Commands| {
            burst(&mut cmds, x, y, &d, n, &mut rng);
        });
    }

    fn tick(world: &mut World, live: usize, max_total: usize, ticks: i32) {
        let mut rng = rng();
        let _ = world.run_system_once(
            move |mut q: Query<(Entity, &mut Spawner)>, mut cmds: Commands| {
                tick_spawners(&mut cmds, &mut q, live, max_total, ticks, &mut rng);
            },
        );
    }

    #[test]
    fn burst_lifetime_and_despawn() {
        let mut world = World::new();
        let d = def();
        burst_n(&mut world, 10.0, 20.0, &d, 3);
        assert_eq!(world.query::<&Particle>().iter(&world).count(), 3);
        step(&mut world, 49);
        assert_eq!(world.query::<&Particle>().iter(&world).count(), 3);
        step(&mut world, 1);
        assert_eq!(world.query::<&Particle>().iter(&world).count(), 0);
    }

    #[test]
    fn gravity_pulls_down_canvas_y() {
        let mut world = World::new();
        let mut d = def();
        d.gravity_pps2 = 200.0;
        d.lifetime_ticks = Jittered::exact(100.0);
        burst_n(&mut world, 0.0, 0.0, &d, 1);
        step(&mut world, 10);
        let mut q = world.query::<&Particle>();
        let p = q.iter(&world).next().unwrap();
        assert!(p.pos[1] > 0.0, "falls toward +y, y = {}", p.pos[1]);
        assert!(p.vel[1] > 0.0);
    }

    #[test]
    fn spawner_rate_accumulates_whole_particles() {
        let mut world = World::new();
        let d = def();
        world.spawn((Spawner {
            def: d,
            x: 0.0,
            y: 0.0,
            acc: 0.0,
        },));
        // 100/s == exactly 1 per 100 Hz tick.
        for _ in 0..5 {
            let live = world.query::<&Particle>().iter(&world).count();
            tick(&mut world, live, 64, 1);
        }
        assert_eq!(world.query::<&Particle>().iter(&world).count(), 5);
    }

    #[test]
    fn spawner_cap_holds() {
        let mut world = World::new();
        let d = def();
        world.spawn((Spawner {
            def: d,
            x: 0.0,
            y: 0.0,
            acc: 0.0,
        },));
        let live = world.query::<&Particle>().iter(&world).count();
        tick(&mut world, live, 2, 10);
        assert!(world.query::<&Particle>().iter(&world).count() <= 2);
    }

    #[test]
    fn sprites_shrink_with_age() {
        let mut world = World::new();
        let d = def();
        burst_n(&mut world, 0.0, 0.0, &d, 1);
        let first = particle_sprites(world.query::<&Particle>().iter(&world));
        assert_eq!(first.len(), 1);
        assert!((first[0].size.x - 4.0).abs() < 1e-4);
        step(&mut world, 25);
        let second = particle_sprites(world.query::<&Particle>().iter(&world));
        assert!(second[0].size.x < 4.0);
    }
}
