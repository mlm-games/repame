//! CPU particles on fixed 100 Hz ticks: spawners plus one-shot
//! bursts, integrated with gravity/drag, rendered as tinted rects
//! through [`SpriteInstance`]. (hanabi's spawner + modifier chain and
//! enoki's CPU-sim/instancing model, flattened for canvas sprites.)
//!
//! Step fns take whole ticks (`i32`), not a game clock type, so any
//! fixed-step game can drive them.

use std::collections::HashMap;

use glam::Vec2;
use rand::{Rng, RngExt};
use repame_sim::bevy_ecs::component::{Mutable, StorageType};
use repame_sim::bevy_ecs::prelude::*;
use repame_sprite::SpriteInstance;

use super::effect::EffectDef;

/// One live particle. Age/life in 100 Hz ticks; deterministic.
/// `spawner` tags the emitter for per-spawner caps (see
/// [`tick_spawners`]); `None` for untracked one-shot bursts.
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
    pub spawner: Option<Entity>,
    /// Atlas page + sub-rect sampled for this particle, copied from the
    /// def at spawn (textured sparks/puffs vs solid tinted rects).
    pub page: u32,
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
}

impl Component for Particle {
    const STORAGE_TYPE: StorageType = StorageType::Table;
    type Mutability = Mutable;
}

/// Continuous emitter: rate accumulates into whole particles.
/// Both caps are enforced by [`tick_spawners`]: the global `max_total`
/// and the per-spawner [`SpawnerDef::max_alive`](super::effect::SpawnerDef::max_alive).
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
/// the game can tag it (cleanup groups, layers). `spawner` tags the
/// emitter for per-spawner caps; pass `None` for untracked bursts.
pub fn spawn_particle(
    commands: &mut Commands,
    x: f32,
    y: f32,
    def: &EffectDef,
    spawner: Option<Entity>,
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
            spawner,
            page: def.page,
            uv_min: def.uv_min,
            uv_max: def.uv_max,
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
        spawn_particle(commands, x, y, def, None, rng);
    }
}

/// Advance emitter clocks; spawn whole particles at the def rate.
/// `ticks` scales accumulation (0 pauses). Two caps: the global
/// `max_total` on live count (runaway emitters can't flood the world)
/// and each def's `spawner.max_alive` per spawner (a lone emitter can
/// never exceed its own budget). Hitting either cap zeroes that
/// spawner's accumulator, so emission resumes clean instead of
/// burst-catching-up.
pub fn tick_spawners(
    commands: &mut Commands,
    spawners: &mut Query<(Entity, &mut Spawner)>,
    particles: &Query<&Particle>,
    live: usize,
    max_total: usize,
    ticks: i32,
    rng: &mut impl Rng,
) {
    if ticks <= 0 {
        return;
    }
    let mut per_spawner: HashMap<Entity, usize> = HashMap::new();
    for p in particles {
        if let Some(e) = p.spawner {
            *per_spawner.entry(e).or_default() += 1;
        }
    }
    let mut spawned = 0usize;
    for (entity, mut spawner) in spawners {
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
            let alive = per_spawner.get(&entity).copied().unwrap_or(0);
            if alive >= spawner.def.spawner.max_alive {
                spawner.acc = 0.0;
                break;
            }
            let (x, y) = (spawner.x, spawner.y);
            let def = spawner.def.clone();
            spawn_particle(commands, x, y, &def, Some(entity), rng);
            spawned += 1;
            *per_spawner.entry(entity).or_default() += 1;
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
        let drag = (-p.drag_per_sec * dt).exp();
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
                anchor: Vec2::new(0.5, 0.5),
                flip_x: false,
                flip_y: false,
                uv_min: Vec2::new(p.uv_min[0], p.uv_min[1]),
                uv_max: Vec2::new(p.uv_max[0], p.uv_max[1]),
                color: p.gradient.sample(t),
                page: p.page,
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
            page: 0,
            uv_min: [0.0, 0.0],
            uv_max: [1.0, 1.0],
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
            move |mut q: Query<(Entity, &mut Spawner)>,
                  pq: Query<&Particle>,
                  mut cmds: Commands| {
                tick_spawners(&mut cmds, &mut q, &pq, live, max_total, ticks, &mut rng);
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
    fn per_spawner_max_alive_holds() {
        let mut world = World::new();
        let mut d = def();
        d.spawner.max_alive = 2;
        world.spawn((Spawner {
            def: d,
            x: 0.0,
            y: 0.0,
            acc: 0.0,
        },));
        for _ in 0..5 {
            let live = world.query::<&Particle>().iter(&world).count();
            tick(&mut world, live, 64, 10);
        }
        assert_eq!(world.query::<&Particle>().iter(&world).count(), 2);
        step(&mut world, 50);
        assert_eq!(world.query::<&Particle>().iter(&world).count(), 0);
        let live = world.query::<&Particle>().iter(&world).count();
        tick(&mut world, live, 64, 10);
        assert_eq!(world.query::<&Particle>().iter(&world).count(), 2);
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

    #[test]
    fn high_drag_over_many_ticks_decays_without_popping() {
        let mut world = World::new();
        let mut d = def();
        d.speed_pps = Jittered::exact(100.0);
        d.lifetime_ticks = Jittered::exact(100.0);
        d.drag_per_sec = 15.0;
        burst_n(&mut world, 0.0, 0.0, &d, 1);
        step(&mut world, 8);
        let mut q = world.query::<&Particle>();
        let p = q.iter(&world).next().unwrap();
        let speed = (p.vel[0] * p.vel[0] + p.vel[1] * p.vel[1]).sqrt();
        let expect = 100.0 * (-15.0 * 0.08f32).exp();
        assert!(
            (speed - expect).abs() < 1e-3,
            "exponential decay, got {speed}"
        );
        assert!(speed > 1.0, "never pops to a stop, got {speed}");
    }

    #[test]
    fn sprites_carry_def_uv_and_page() {
        let mut world = World::new();
        let mut d = def();
        d.page = 2;
        d.uv_min = [0.25, 0.5];
        d.uv_max = [0.5, 0.75];
        burst_n(&mut world, 0.0, 0.0, &d, 1);
        let sprites = particle_sprites(world.query::<&Particle>().iter(&world));
        assert_eq!(sprites.len(), 1);
        assert_eq!(sprites[0].page, 2);
        assert_eq!(sprites[0].uv_min, Vec2::new(0.25, 0.5));
        assert_eq!(sprites[0].uv_max, Vec2::new(0.5, 0.75));
    }
}
