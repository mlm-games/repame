//! Sim-side 3D particles on fixed 100 Hz ticks: spawners plus one-shot
//! bursts, integrated with gravity/drag, rendered as camera-facing
//! billboard quads ([`MeshGroup`]) through `repame-view3d`.
//!
//! Same design as the 2D [`super::particles`] module (spawner rate
//! accumulation, per-spawner + global caps, deterministic seeded RNG,
//! gradient-over-life, ease-curve shrink), lifted to world space:
//!
//! - `pos`/`vel` are world `[f32; 3]` (Y-up). Gravity pulls **−Y**
//!   (world down) — note the sign flip vs the 2D module, whose `+y` is
//!   canvas-down.
//! - [`EffectDef`](super::effect::EffectDef) is reused verbatim
//!   (`SpawnerDef`, [`Jittered`](super::effect::Jittered),
//!   [`Gradient`](super::effect::Gradient),
//!   [`EaseKind`](super::effect::EaseKind)): one asset shape for 2D and 3D.
//!   `size_px` reads as **world units** here (documented at the call site,
//!   not converted — there is no dpi in world space).
//! - Rendering is one [`MeshGroup`] per live particle (unlit textured
//!   quad, billboarded around the camera eye): per-particle alpha rides
//!   the group's `alpha`, so fading particles sort back-to-front in the
//!   transparent pass instead of sharing one flat fade.
//!
//! Step fns take whole ticks (`i32`), not a game clock type, so any
//! fixed-step game can drive them.

use bevy_ecs::prelude::*;
use glam::Vec3;
use rand::{Rng, RngExt};
use repame_sim::bevy_ecs::component::{Mutable, StorageType};
use repame_view3d::MeshGroup;

use super::effect::EffectDef;

/// One live 3D particle. Age/life in 100 Hz ticks; deterministic.
/// `spawner` tags the emitter for per-spawner caps (see
/// [`tick_spawners3`]); `None` for untracked one-shot bursts.
#[derive(Clone, Debug)]
pub struct Particle3 {
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub age_ticks: i32,
    pub life_ticks: i32,
    /// Billboard edge length in world units (before the ease shrink).
    pub size_world: f32,
    /// Gravity strength in world units/s², applied toward −Y.
    pub gravity_pps2: f32,
    pub drag_per_sec: f32,
    pub gradient: super::effect::Gradient,
    pub ease: super::effect::EaseKind,
    pub spawner: Option<Entity>,
    /// Texture array page + sub-rect sampled for this particle, copied
    /// from the def at spawn (textured sparks/puffs vs solid tinted quads).
    pub page: u32,
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
}

impl Component for Particle3 {
    const STORAGE_TYPE: StorageType = StorageType::Table;
    type Mutability = Mutable;
}

/// Continuous emitter: rate accumulates into whole particles.
#[derive(Clone, Debug)]
pub struct Spawner3 {
    pub def: EffectDef,
    /// World-space emitter origin.
    pub pos: [f32; 3],
    pub acc: f32,
}

impl Component for Spawner3 {
    const STORAGE_TYPE: StorageType = StorageType::Table;
    type Mutability = Mutable;
}

/// Spawn one particle from a def at `pos`. Initial velocity is uniform on
/// the sphere (3D analog of the 2D disc sample). Returns the entity so
/// the game can tag it. `spawner` tags the emitter for per-spawner caps;
/// pass `None` for untracked bursts.
pub fn spawn_particle3(
    commands: &mut Commands,
    pos: [f32; 3],
    def: &EffectDef,
    spawner: Option<Entity>,
    rng: &mut impl Rng,
) -> Entity {
    // Uniform sphere: z uniform in −1..1, azimuth uniform in 0..TAU.
    let z: f32 = rng.random_range(-1.0..1.0);
    let theta: f32 = rng.random_range(0.0..std::f32::consts::TAU);
    let r = (1.0 - z * z).max(0.0).sqrt();
    let dir = [r * theta.cos(), z, r * theta.sin()];
    let speed = def.speed_pps.sample(rng);
    commands
        .spawn((Particle3 {
            pos,
            vel: [dir[0] * speed, dir[1] * speed, dir[2] * speed],
            age_ticks: 0,
            life_ticks: def.lifetime_ticks.sample(rng).max(1.0) as i32,
            size_world: def.size_px.sample(rng).max(0.001),
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

/// One-shot burst of `count` particles (explosions, impact puffs).
pub fn burst3(
    commands: &mut Commands,
    pos: [f32; 3],
    def: &EffectDef,
    count: usize,
    rng: &mut impl Rng,
) {
    for _ in 0..count {
        spawn_particle3(commands, pos, def, None, rng);
    }
}

/// Advance emitter clocks; spawn whole particles at the def rate.
/// Same two caps as the 2D path: the global `max_total` on live count
/// and each def's `spawner.max_alive` per spawner. Hitting either cap
/// zeroes that spawner's accumulator, so emission resumes clean instead
/// of burst-catching-up. `ticks <= 0` pauses.
pub fn tick_spawners3(
    commands: &mut Commands,
    spawners: &mut Query<(Entity, &mut Spawner3)>,
    particles: &Query<&Particle3>,
    live: usize,
    max_total: usize,
    ticks: i32,
    rng: &mut impl Rng,
) {
    if ticks <= 0 {
        return;
    }
    let mut per_spawner: std::collections::HashMap<Entity, usize> =
        std::collections::HashMap::new();
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
            let pos = spawner.pos;
            let def = spawner.def.clone();
            spawn_particle3(commands, pos, &def, Some(entity), rng);
            spawned += 1;
            *per_spawner.entry(entity).or_default() += 1;
        }
    }
}

/// Integrate motion, age, and despawn the spent. Gravity pulls −Y
/// (world down); drag is exponential per second.
pub fn step_particles3(
    commands: &mut Commands,
    particles: &mut Query<(Entity, &mut Particle3)>,
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
        p.vel[1] = p.vel[1] * drag - p.gravity_pps2 * dt;
        p.vel[2] *= drag;
        p.pos[0] += p.vel[0] * dt;
        p.pos[1] += p.vel[1] * dt;
        p.pos[2] += p.vel[2] * dt;
    }
}

/// Camera basis for spherical billboards: `right = Y × fwd`, `up = fwd ×
/// right`, where `fwd` runs from the particle toward the eye. Top-down
/// degenerate (eye directly overhead): `right` falls back to +X. The
/// basis is right-handed with `right × up == fwd`, so quads wound
/// `(a, b, c) + (a, c, d)` with `a = c − r − u` face the camera and
/// survive backface culling.
fn billboard_basis(eye: Vec3, center: Vec3) -> (Vec3, Vec3) {
    let fwd = (eye - center).normalize_or_zero();
    let mut right = Vec3::Y.cross(fwd);
    if right.length_squared() < 1e-12 {
        right = Vec3::X;
    } else {
        right = right.normalize();
    }
    let up = fwd.cross(right).normalize_or(Vec3::Y);
    (right, up)
}

/// Map live particles to one billboard [`MeshGroup`] each: gradient color
/// at life fraction, size shrinking along the ease curve, per-particle
/// alpha on the group (fading particles land in the transparent pass and
/// sort back-to-front; fully opaque ones stay in the opaque pass).
/// Unlit textured quads: tint × texel, no scene-light dependence (sparks
/// pop in the dark). Takes an iterator so both systems (`query.iter()`)
/// and tests can feed it.
pub fn particle_groups<'a>(
    particles: impl Iterator<Item = &'a Particle3>,
    eye: Vec3,
) -> Vec<MeshGroup> {
    particles
        .map(|p| {
            let t = p.age_ticks as f32 / p.life_ticks.max(1) as f32;
            let shrink = 1.0 - p.ease.apply(t);
            let s = (p.size_world * shrink).max(1e-4);
            let hs = s * 0.5;
            let rgba = p.gradient.sample(t);
            let center = Vec3::from(p.pos);
            let (right, up) = billboard_basis(eye, center);
            let a = center - right * hs - up * hs;
            let b = center + right * hs - up * hs;
            let c = center + right * hs + up * hs;
            let d = center - right * hs + up * hs;
            let tint = [rgba[0], rgba[1], rgba[2]];
            let uvs = [
                [p.uv_min[0], p.uv_min[1]],
                [p.uv_max[0], p.uv_min[1]],
                [p.uv_max[0], p.uv_max[1]],
                [p.uv_min[0], p.uv_max[1]],
            ];
            let mut g = MeshGroup {
                texture_page: p.page,
                depth_test: true,
                transparent: rgba[3] < 0.999,
                alpha: rgba[3].clamp(0.0, 1.0),
                ..Default::default()
            };
            g.push_quad_textured(a.into(), b.into(), c.into(), d.into(), tint, uvs);
            g
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
            size_px: Jittered::exact(2.0),
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
            move |mut q: Query<(Entity, &mut Particle3)>, mut cmds: Commands| {
                step_particles3(&mut cmds, &mut q, ticks);
            },
        );
    }

    fn burst_n(world: &mut World, pos: [f32; 3], d: &EffectDef, n: usize) {
        let mut rng = rng();
        let d = d.clone();
        let _ = world.run_system_once(move |mut cmds: Commands| {
            burst3(&mut cmds, pos, &d, n, &mut rng);
        });
    }

    fn tick(world: &mut World, live: usize, max_total: usize, ticks: i32) {
        let mut rng = rng();
        let _ = world.run_system_once(
            move |mut q: Query<(Entity, &mut Spawner3)>,
                  pq: Query<&Particle3>,
                  mut cmds: Commands| {
                tick_spawners3(&mut cmds, &mut q, &pq, live, max_total, ticks, &mut rng);
            },
        );
    }

    #[test]
    fn burst_lifetime_and_despawn() {
        let mut world = World::new();
        let d = def();
        burst_n(&mut world, [10.0, 20.0, 30.0], &d, 3);
        assert_eq!(world.query::<&Particle3>().iter(&world).count(), 3);
        step(&mut world, 49);
        assert_eq!(world.query::<&Particle3>().iter(&world).count(), 3);
        step(&mut world, 1);
        assert_eq!(world.query::<&Particle3>().iter(&world).count(), 0);
    }

    #[test]
    fn gravity_pulls_down_world_minus_y() {
        let mut world = World::new();
        let mut d = def();
        d.gravity_pps2 = 200.0;
        d.lifetime_ticks = Jittered::exact(100.0);
        burst_n(&mut world, [0.0, 0.0, 0.0], &d, 1);
        step(&mut world, 10);
        let mut q = world.query::<&Particle3>();
        let p = q.iter(&world).next().unwrap();
        assert!(p.pos[1] < 0.0, "falls toward −Y, y = {}", p.pos[1]);
        assert!(p.vel[1] < 0.0);
    }

    #[test]
    fn initial_velocities_cover_the_sphere() {
        // Zero jitter on speed would hide direction bugs; fixed speed +
        // many samples must spread over all octants.
        let mut world = World::new();
        let mut d = def();
        d.speed_pps = Jittered::exact(10.0);
        d.lifetime_ticks = Jittered::exact(100.0);
        burst_n(&mut world, [0.0, 0.0, 0.0], &d, 64);
        let mut q = world.query::<&Particle3>();
        let mut signs = std::collections::HashSet::new();
        for p in q.iter(&world) {
            let speed = (p.vel[0] * p.vel[0] + p.vel[1] * p.vel[1] + p.vel[2] * p.vel[2]).sqrt();
            assert!((speed - 10.0).abs() < 1e-4, "fixed speed: {speed}");
            signs.insert((p.vel[0] > 0.0, p.vel[1] > 0.0, p.vel[2] > 0.0));
        }
        assert!(
            signs.len() >= 6,
            "sphere sample reaches most octants: {}",
            signs.len()
        );
    }

    #[test]
    fn spawner_caps_hold() {
        let mut world = World::new();
        let mut d = def();
        d.spawner.max_alive = 2;
        world.spawn((Spawner3 {
            def: d,
            pos: [0.0, 0.0, 0.0],
            acc: 0.0,
        },));
        for _ in 0..5 {
            let live = world.query::<&Particle3>().iter(&world).count();
            tick(&mut world, live, 64, 10);
        }
        assert_eq!(world.query::<&Particle3>().iter(&world).count(), 2);
    }

    #[test]
    fn billboard_faces_the_eye() {
        // Eye on +Z: quad must sit in the z = 0 plane, 2x2, CCW from +Z
        // (front face toward the camera, survives backface culling).
        let mut world = World::new();
        burst_n(&mut world, [0.0, 0.0, 0.0], &def(), 1);
        let groups = particle_groups(
            world.query::<&Particle3>().iter(&world),
            Vec3::new(0.0, 0.0, 10.0),
        );
        assert_eq!(groups.len(), 1);
        let g = &groups[0];
        assert_eq!(g.tri_count(), 2);
        assert!(g.positions.iter().all(|p| p[2].abs() < 1e-5));
        let xs: Vec<f32> = g.positions.iter().map(|p| p[0]).collect();
        let ys: Vec<f32> = g.positions.iter().map(|p| p[1]).collect();
        assert!((xs.iter().cloned().fold(f32::INFINITY, f32::min) + 1.0).abs() < 1e-5);
        assert!((xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max) - 1.0).abs() < 1e-5);
        assert!((ys.iter().cloned().fold(f32::INFINITY, f32::min) + 1.0).abs() < 1e-5);
        assert!((ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max) - 1.0).abs() < 1e-5);
        // Winding: first triangle normal faces +Z (toward the eye).
        let a = Vec3::from(g.positions[g.indices[0] as usize]);
        let b = Vec3::from(g.positions[g.indices[1] as usize]);
        let c = Vec3::from(g.positions[g.indices[2] as usize]);
        let n = (b - a).cross(c - a).normalize();
        assert!((n - Vec3::Z).length() < 1e-5, "faces the eye: {n:?}");
        // Solid red: opaque pass, full alpha, unlit textured quad.
        assert!(!g.transparent);
        assert_eq!((g.alpha, g.uvs.len()), (1.0, g.positions.len()));
        assert!(g.normals.is_empty());
    }

    #[test]
    fn fading_particles_go_transparent_and_shrink() {
        let mut world = World::new();
        let mut d = def();
        d.gradient = Gradient::fade_out([0.0, 1.0, 0.0, 1.0]);
        burst_n(&mut world, [0.0, 5.0, 0.0], &d, 1);
        let eye = Vec3::new(10.0, 5.0, 0.0);
        let fresh = particle_groups(world.query::<&Particle3>().iter(&world), eye);
        assert!(!fresh[0].transparent, "alpha 1 starts opaque");
        step(&mut world, 25);
        let old = particle_groups(world.query::<&Particle3>().iter(&world), eye);
        assert!(old[0].transparent, "faded alpha takes the blend pass");
        assert!(old[0].alpha < 1.0);
        assert!(
            old[0].tri_count() == 2 && {
                let e0 = edge_len(&fresh[0]);
                let e1 = edge_len(&old[0]);
                e1 < e0
            },
            "ease shrink narrows the quad"
        );
    }

    fn edge_len(g: &MeshGroup) -> f32 {
        let a = Vec3::from(g.positions[0]);
        let b = Vec3::from(g.positions[1]);
        (b - a).length()
    }

    #[test]
    fn particle_groups_flatten_in_the_batch() {
        // The batch contract holds for emitted groups: validation passes,
        // opaque + transparent ranges both flatten.
        use repame_view3d::SceneBatch;
        let mut world = World::new();
        let mut d = def();
        d.gradient = Gradient::fade_out([0.0, 1.0, 0.0, 1.0]);
        burst_n(&mut world, [0.0, 0.0, 0.0], &d, 2);
        step(&mut world, 25);
        let eye = Vec3::new(0.0, 0.0, 10.0);
        let groups = particle_groups(world.query::<&Particle3>().iter(&world), eye);
        let mut batch = SceneBatch::with_id("test.particles3");
        batch.set_camera(glam::Mat4::IDENTITY);
        for g in &groups {
            batch.push_group(g);
        }
        batch.finish();
        assert_eq!(batch.len_tris(), 4);
        assert_eq!(batch.culled(), 0, "identity camera disables culling");
    }
}
