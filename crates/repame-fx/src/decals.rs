//! Ground decals: flat quads above the terrain that fade over a fixed life.
//! One transparent [`MeshGroup`] per decal; ages and lives are seconds
//! (`step_decals_secs`), `*_ticks` fields and fns are 100 Hz shims.
//! Quads sit 0.02 above the plane to avoid z-fighting.

use bevy_ecs::prelude::*;
use repame_sim::bevy_ecs::component::{Mutable, StorageType};
use repame_view3d::MeshGroup;

/// One live decal. Age and life in seconds; legacy tick mirrors kept.
#[derive(Clone, Debug)]
pub struct Decal {
    pub pos: [f32; 3],
    /// Quad edge length in world units, before the grow curve.
    pub size_world: f32,
    /// Final size as a multiple of spawn size (1.0 means fixed size).
    pub grow: f32,
    /// Fade length in seconds: alpha runs 1 to 0 over the last
    /// `fade_secs` of life. 0 holds full alpha, then vanishes.
    pub fade_secs: f32,
    pub age_secs: f32,
    pub life_secs: f32,
    /// Yaw about +Y in radians.
    pub yaw: f32,
    pub tint: [f32; 3],
    /// Texture page and sub-rect for the decal sprite.
    pub page: u32,
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    /// Legacy 100 Hz mirrors; new code reads the `_secs` fields.
    pub fade_ticks: i32,
    pub age_ticks: i32,
    pub life_ticks: i32,
}

impl Component for Decal {
    const STORAGE_TYPE: StorageType = StorageType::Table;
    type Mutability = Mutable;
}

/// Spawn parameters for [`spawn_decal`].
#[derive(Clone, Copy, Debug)]
pub struct DecalDef {
    /// Quad center (lift is added on top at spawn).
    pub pos: [f32; 3],
    pub size_world: f32,
    pub tint: [f32; 3],
    /// Life in seconds. Accepts the legacy `life_ticks` name (100 Hz
    /// quanta) at construction via [`DecalDef::with_life_ticks`]; struct
    /// literals use seconds.
    pub life_secs: f32,
    /// Fade length in seconds.
    pub fade_secs: f32,
    pub page: u32,
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    pub yaw: f32,
    pub lift: f32,
    pub grow: f32,
    /// Legacy 100 Hz mirrors; constructors keep them in sync.
    pub life_ticks: i32,
    pub fade_ticks: i32,
}

fn secs_of_ticks(ticks: i32) -> f32 {
    super::driver::ticks_to_secs_100hz(ticks)
}

impl DecalDef {
    /// Legacy constructor from 100 Hz quanta.
    pub fn with_life_ticks(mut self, life_ticks: i32, fade_ticks: i32) -> Self {
        self.life_secs = secs_of_ticks(life_ticks.max(1));
        self.fade_secs = secs_of_ticks(fade_ticks.max(0));
        self.life_ticks = life_ticks.max(1);
        self.fade_ticks = fade_ticks.max(0);
        self
    }

    /// Legacy 100 Hz views.
    pub fn life_ticks_view(&self) -> i32 {
        self.life_ticks
    }
}

impl Default for DecalDef {
    fn default() -> Self {
        Self {
            pos: [0.0, 0.0, 0.0],
            size_world: 1.0,
            tint: [0.0, 0.0, 0.0],
            life_secs: 1.0,
            fade_secs: 0.3,
            life_ticks: 100,
            fade_ticks: 30,
            page: 0,
            uv_min: [0.0, 0.0],
            uv_max: [1.0, 1.0],
            yaw: 0.0,
            lift: 0.02,
            grow: 1.0,
        }
    }
}

/// Spawn one decal from a [`DecalDef`]. Returns the entity.
/// `life_secs = f32::INFINITY` is persistent: full alpha until despawned.
pub fn spawn_decal(commands: &mut Commands, def: DecalDef) -> Entity {
    let persistent = !def.life_secs.is_finite();
    commands
        .spawn((Decal {
            pos: [def.pos[0], def.pos[1] + def.lift, def.pos[2]],
            size_world: def.size_world.max(1e-4),
            grow: def.grow,
            fade_secs: if def.fade_secs.is_finite() {
                def.fade_secs.max(0.0)
            } else {
                0.0
            },
            age_secs: 0.0,
            life_secs: def.life_secs,
            fade_ticks: def.fade_ticks.max(0),
            age_ticks: 0,
            life_ticks: if persistent {
                i32::MAX
            } else {
                def.life_ticks.max(1)
            },
            yaw: def.yaw,
            tint: def.tint,
            page: def.page,
            uv_min: def.uv_min,
            uv_max: def.uv_max,
        },))
        .id()
}

/// Blob shadow: a dark disc pinned under an agent.
/// The game writes agent XZ into `pos` each frame.
pub fn blob_shadow(pos: [f32; 3], radius_world: f32) -> Decal {
    Decal {
        pos: [pos[0], pos[1] + 0.02, pos[2]],
        size_world: (radius_world * 2.0).max(1e-4),
        grow: 1.0,
        fade_secs: 0.0,
        age_secs: 0.0,
        life_secs: f32::INFINITY,
        fade_ticks: 0,
        age_ticks: 0,
        life_ticks: i32::MAX,
        yaw: 0.0,
        tint: [0.0, 0.0, 0.0],
        page: 0,
        uv_min: [0.0, 0.0],
        uv_max: [1.0, 1.0],
    }
}

/// Spawn a [`blob_shadow`] directly. Persistent until despawned.
pub fn spawn_blob_shadow(commands: &mut Commands, pos: [f32; 3], radius_world: f32) -> Entity {
    let decal = blob_shadow(pos, radius_world);
    commands.spawn((decal,)).id()
}

/// Age decals and despawn the spent.
pub fn step_decals_secs(
    commands: &mut Commands,
    decals: &mut Query<(Entity, &mut Decal)>,
    dt_secs: f32,
) {
    if !dt_secs.is_finite() || dt_secs <= 0.0 {
        return;
    }
    for (e, mut d) in decals {
        if !d.life_secs.is_finite() {
            continue;
        }
        d.age_secs += dt_secs;
        d.age_ticks = (d.age_secs / super::driver::SECS_PER_TICK_100HZ).floor() as i32;
        if d.age_secs + 1e-6 >= d.life_secs {
            commands.entity(e).try_despawn();
        }
    }
}

/// Legacy 100 Hz wrapper over [`step_decals_secs`].
pub fn step_decals(commands: &mut Commands, decals: &mut Query<(Entity, &mut Decal)>, ticks: i32) {
    step_decals_secs(commands, decals, super::driver::ticks_to_secs_100hz(ticks));
}

/// Alpha at decal age: full until the fade window, then linear to 0.
fn decal_alpha(d: &Decal) -> f32 {
    if !d.life_secs.is_finite() {
        return 1.0;
    }
    if d.fade_secs <= 0.0 || d.life_secs <= d.fade_secs {
        return if d.age_secs < d.life_secs { 1.0 } else { 0.0 };
    }
    let fade_start = d.life_secs - d.fade_secs;
    if d.age_secs <= fade_start {
        1.0
    } else {
        (1.0 - (d.age_secs - fade_start) / d.fade_secs).clamp(0.0, 1.0)
    }
}

/// One [`MeshGroup`] per live decal: horizontal yaw-rotated quad.
/// Size grows toward `size_world * grow` over the first quarter of life.
pub fn decal_groups<'a>(decals: impl Iterator<Item = &'a Decal>) -> Vec<MeshGroup> {
    decals
        .map(|d| {
            let t = if d.life_secs.is_finite() {
                (d.age_secs / d.life_secs.max(1e-6)).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let grow_t = (t / 0.25).clamp(0.0, 1.0);
            let s = d.size_world * (1.0 + (d.grow - 1.0) * grow_t);
            let hs = s * 0.5;
            let (sy, cy) = d.yaw.sin_cos();
            let rot = |lx: f32, lz: f32| {
                [
                    d.pos[0] + lx * cy + lz * sy,
                    d.pos[1],
                    d.pos[2] - lx * sy + lz * cy,
                ]
            };
            let uvs = [
                [d.uv_min[0], d.uv_min[1]],
                [d.uv_max[0], d.uv_min[1]],
                [d.uv_max[0], d.uv_max[1]],
                [d.uv_min[0], d.uv_max[1]],
            ];
            let mut g = MeshGroup {
                texture_page: d.page,
                depth_test: true,
                transparent: true,
                alpha: decal_alpha(d),
                ..Default::default()
            };
            // CCW from above, so the face normal is +Y and backface culling
            // keeps the quad for top-down cameras.
            g.push_quad_textured(
                rot(-hs, hs),
                rot(hs, hs),
                rot(hs, -hs),
                rot(-hs, -hs),
                d.tint,
                uvs,
            );
            g
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use repame_sim::bevy_ecs::system::RunSystemOnce;

    fn step_secs(world: &mut World, dt: f32) {
        let _ = world.run_system_once(
            move |mut q: Query<(Entity, &mut Decal)>, mut cmds: Commands| {
                step_decals_secs(&mut cmds, &mut q, dt);
            },
        );
    }

    fn step(world: &mut World, ticks: i32) {
        step_secs(world, super::super::driver::ticks_to_secs_100hz(ticks));
    }

    #[test]
    fn decal_lies_flat_and_fades() {
        let mut world = World::new();
        let _ = world.run_system_once(move |mut cmds: Commands| {
            spawn_decal(
                &mut cmds,
                DecalDef {
                    pos: [4.0, 0.0, 6.0],
                    size_world: 2.0,
                    tint: [1.0, 0.0, 0.0],
                    life_secs: 1.0,
                    fade_secs: 0.4,
                    life_ticks: 100,
                    fade_ticks: 40,
                    page: 1,
                    yaw: 0.0,
                    lift: 0.02,
                    ..DecalDef::default()
                },
            );
        });
        assert_eq!(world.query::<&Decal>().iter(&world).count(), 1);
        let groups: Vec<MeshGroup> = decal_groups(world.query::<&Decal>().iter(&world));
        assert_eq!(groups.len(), 1);
        let g = &groups[0];
        // Flat at the lifted height, 2x2, full alpha, transparent pass.
        assert!(g.positions.iter().all(|p| (p[1] - 0.02).abs() < 1e-6));
        assert!(g.transparent && g.depth_test);
        assert_eq!((g.alpha, g.texture_page), (1.0, 1));
        assert_eq!(g.tri_count(), 2);
        // Up-facing winding, kept by backface culling from above.
        let a = glam::Vec3::from(g.positions[g.indices[0] as usize]);
        let b = glam::Vec3::from(g.positions[g.indices[1] as usize]);
        let c = glam::Vec3::from(g.positions[g.indices[2] as usize]);
        let n = (b - a).cross(c - a).normalize();
        assert!((n - glam::Vec3::Y).length() < 1e-5, "{n:?}");
        // Fade window: last 40 ticks linear to 0, then despawn.
        step(&mut world, 60);
        let groups: Vec<MeshGroup> = decal_groups(world.query::<&Decal>().iter(&world));
        assert_eq!(groups[0].alpha, 1.0, "fade starts at tick 60");
        step(&mut world, 20);
        let groups: Vec<MeshGroup> = decal_groups(world.query::<&Decal>().iter(&world));
        assert!(
            (groups[0].alpha - 0.5).abs() < 1e-5,
            "half faded: {}",
            groups[0].alpha
        );
        step(&mut world, 20);
        assert_eq!(world.query::<&Decal>().iter(&world).count(), 0);
    }

    #[test]
    fn decal_groups_flatten_in_the_batch() {
        use repame_view3d::SceneBatch;
        let mut world = World::new();
        let _ = world.run_system_once(move |mut cmds: Commands| {
            spawn_decal(
                &mut cmds,
                DecalDef {
                    size_world: 3.0,
                    tint: [0.2, 0.2, 0.2],
                    life_secs: 0.5,
                    fade_secs: 0.1,
                    life_ticks: 50,
                    fade_ticks: 10,
                    yaw: 0.7,
                    ..DecalDef::default()
                },
            );
        });
        let groups: Vec<MeshGroup> = decal_groups(world.query::<&Decal>().iter(&world));
        let mut batch = SceneBatch::with_id("test.decals");
        batch.set_camera(glam::Mat4::IDENTITY);
        for g in &groups {
            batch.push_group(g);
        }
        batch.finish();
        assert_eq!(batch.len_tris(), 2);
    }

    #[test]
    fn blob_shadow_grounds_and_follows() {
        let mut world = World::new();
        let e = world
            .run_system_once(|mut cmds: Commands| {
                spawn_blob_shadow(&mut cmds, [3.0, 0.0, 4.0], 0.5)
            })
            .expect("spawn system runs");
        let d = world.get::<Decal>(e).expect("shadow spawned");
        assert_eq!((d.pos[0], d.pos[2]), (3.0, 4.0));
        assert!((d.pos[1] - 0.02).abs() < 1e-6, "lifted: {}", d.pos[1]);
        assert_eq!(d.life_ticks, i32::MAX, "persistent until despawned");
        assert_eq!(d.tint, [0.0, 0.0, 0.0]);
        // Game follow: agent moves, shadow XZ tracks, group follows.
        world.get_mut::<Decal>(e).unwrap().pos[0] = 7.0;
        world.get_mut::<Decal>(e).unwrap().pos[2] = -2.0;
        let groups: Vec<MeshGroup> = decal_groups(world.query::<&Decal>().iter(&world));
        let cx: f32 = groups[0].positions.iter().map(|p| p[0]).sum::<f32>()
            / groups[0].positions.len() as f32;
        let cz: f32 = groups[0].positions.iter().map(|p| p[2]).sum::<f32>()
            / groups[0].positions.len() as f32;
        assert!(
            (cx - 7.0).abs() < 1e-5 && (cz + 2.0).abs() < 1e-5,
            "({cx}, {cz})"
        );
        assert_eq!(groups[0].alpha, 1.0, "no fade on a persistent shadow");
        step(&mut world, 10_000);
        assert_eq!(world.query::<&Decal>().iter(&world).count(), 1);
    }
}
