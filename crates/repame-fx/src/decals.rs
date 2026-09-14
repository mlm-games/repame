//! Ground decals: flat quads pinned just above the terrain that fade over
//! a fixed life (scorch marks, blood splats, selection rings, footprints).
//!
//! A decal is one transparent [`MeshGroup`]: a horizontal quad at
//! `y = ground_y + lift`, unlit textured (tint × texel), `depth_test = true`
//! so walls occlude it and it occludes nothing (no depth writes in the
//! transparent pass). Fading rides the group's `alpha`, so decals sort
//! back-to-front with particles in the same pass instead of needing their
//! own. Sim-side state ([`Decal`]) steps on the same 100 Hz ticks as
//! [`super::particles3d`]; rendering is [`decal_groups`], one group per
//! live decal.
//!
//! `lift` exists because depth precision at range makes coplanar quads
//! flicker: 0.02 world units is invisible at game scale and keeps the quad
//! above the ground plane it marks. Not a polygon-offset — same lift every
//! frame, deterministic.

use bevy_ecs::prelude::*;
use repame_sim::bevy_ecs::component::{Mutable, StorageType};
use repame_view3d::MeshGroup;

/// One live decal. Age/life in 100 Hz ticks; deterministic.
#[derive(Clone, Debug)]
pub struct Decal {
    /// Quad center in world space.
    pub pos: [f32; 3],
    /// Quad edge length in world units (before the grow curve).
    pub size_world: f32,
    /// Growth from spawn size to final size over life (1.0 = fixed size).
    /// Scorch marks pop slightly larger than their spawn flash.
    pub grow: f32,
    /// Ticks to full fade (alpha 1 → 0 linearly over the last `fade_ticks`
    /// of life; `0` = hold full alpha then vanish at expiry).
    pub fade_ticks: i32,
    pub age_ticks: i32,
    pub life_ticks: i32,
    /// Yaw about +Y in radians (randomized at spawn for variety).
    pub yaw: f32,
    pub tint: [f32; 3],
    /// Texture array page + sub-rect (splat sprite vs solid ring tint).
    pub page: u32,
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
}

impl Component for Decal {
    const STORAGE_TYPE: StorageType = StorageType::Table;
    type Mutability = Mutable;
}

/// Spawn spec for [`spawn_decal`] (11 params collapse into one struct so
/// call sites stay readable and clippy stays quiet).
#[derive(Clone, Copy, Debug)]
pub struct DecalDef {
    /// Quad center in world space (lift applied on top at spawn).
    pub pos: [f32; 3],
    pub size_world: f32,
    pub tint: [f32; 3],
    pub life_ticks: i32,
    pub fade_ticks: i32,
    pub page: u32,
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    /// Yaw about +Y in radians (randomized at spawn for variety).
    pub yaw: f32,
    /// Height above the ground plane (0.02 unless z-fighting says otherwise).
    pub lift: f32,
    /// Growth from spawn size to final size over life (1.0 = fixed).
    pub grow: f32,
}

impl Default for DecalDef {
    fn default() -> Self {
        Self {
            pos: [0.0, 0.0, 0.0],
            size_world: 1.0,
            tint: [0.0, 0.0, 0.0],
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
///
/// Pass `life_ticks = i32::MAX` for a persistent decal (selection rings,
/// blob shadows): it holds full alpha until explicitly despawned.
pub fn spawn_decal(commands: &mut Commands, def: DecalDef) -> Entity {
    commands
        .spawn((Decal {
            pos: [def.pos[0], def.pos[1] + def.lift, def.pos[2]],
            size_world: def.size_world.max(1e-4),
            grow: def.grow,
            fade_ticks: def.fade_ticks.max(0),
            age_ticks: 0,
            life_ticks: def.life_ticks.max(1),
            yaw: def.yaw,
            tint: def.tint,
            page: def.page,
            uv_min: def.uv_min,
            uv_max: def.uv_max,
        },))
        .id()
}

/// A blob shadow: one dark ellipse pinned under an agent, following it
/// every frame. Not a light, not a depth texture — the 90% solution that
/// ships in every game in `external/` history that grounds its characters
/// (neither rustbox nor resims enables shadow maps: rustbox sets
/// `shadow_maps_enabled: false`, resims bakes shade per face).
///
/// The game owns the follow: each frame it writes the agent's XZ into the
/// decal's `pos` (and optionally scales `size_world` by height for the
/// shrink-with-altitude read). This module only builds the decal spec —
/// a soft dark disc, full alpha, persistent life — so the follow loop is
/// three lines at the call site. Despawn explicitly when the agent dies.
///
/// Tint `[0, 0, 0]` with a radial texture (dark center, transparent edge)
/// reads as a contact shadow; without a texture page the solid disc still
/// grounds, but the edge is hard.
pub fn blob_shadow(pos: [f32; 3], radius_world: f32) -> Decal {
    Decal {
        pos: [pos[0], pos[1] + 0.02, pos[2]],
        size_world: (radius_world * 2.0).max(1e-4),
        grow: 1.0,
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

/// Spawn a [`blob_shadow`] directly (persistent until despawned).
pub fn spawn_blob_shadow(commands: &mut Commands, pos: [f32; 3], radius_world: f32) -> Entity {
    let decal = blob_shadow(pos, radius_world);
    commands.spawn((decal,)).id()
}

/// Age decals and despawn the spent. `ticks <= 0` pauses.
pub fn step_decals(commands: &mut Commands, decals: &mut Query<(Entity, &mut Decal)>, ticks: i32) {
    if ticks <= 0 {
        return;
    }
    for (e, mut d) in decals {
        d.age_ticks += ticks;
        if d.age_ticks >= d.life_ticks {
            commands.entity(e).try_despawn();
        }
    }
}

/// Alpha at the decal's age: full until the fade window, then linear to 0.
fn decal_alpha(d: &Decal) -> f32 {
    if d.fade_ticks <= 0 || d.life_ticks <= d.fade_ticks {
        return if d.age_ticks < d.life_ticks { 1.0 } else { 0.0 };
    }
    let fade_start = d.life_ticks - d.fade_ticks;
    if d.age_ticks <= fade_start {
        1.0
    } else {
        (1.0 - (d.age_ticks - fade_start) as f32 / d.fade_ticks as f32).clamp(0.0, 1.0)
    }
}

/// Map live decals to one [`MeshGroup`] each: horizontal yaw-rotated quad,
/// unlit textured, transparent with per-decal alpha (back-to-front with
/// particles in the same pass). Size grows `size_world → size_world * grow`
/// over the first quarter of life (spawn pop), then holds.
pub fn decal_groups<'a>(decals: impl Iterator<Item = &'a Decal>) -> Vec<MeshGroup> {
    decals
        .map(|d| {
            let t = d.age_ticks as f32 / d.life_ticks.max(1) as f32;
            let grow_t = (t / 0.25).clamp(0.0, 1.0);
            let s = d.size_world * (1.0 + (d.grow - 1.0) * grow_t);
            let hs = s * 0.5;
            let (sy, cy) = d.yaw.sin_cos();
            // Yaw about +Y: local (x, z) → world (x * cy + z * sy, −x * sy + z * cy).
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
            // Corner order (a, b, c, d) is CCW-from-above: a=(−hs,+hs),
            // b=(+hs,+hs), c=(+hs,−hs), d=(−hs,−hs), so the face normal is
            // +Y and backface culling keeps it for above-looking-down
            // cameras (same convention as the renderer's up-quad fixtures).
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

    fn step(world: &mut World, ticks: i32) {
        let _ = world.run_system_once(
            move |mut q: Query<(Entity, &mut Decal)>, mut cmds: Commands| {
                step_decals(&mut cmds, &mut q, ticks);
            },
        );
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
        // Winding: up-facing (CCW from above) so backface culling keeps it
        // for the above-looking-down camera.
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
        // The game follow: agent moves, shadow XZ tracks, group follows.
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
        // Persistent shadows never age out.
        step(&mut world, 10_000);
        assert_eq!(world.query::<&Decal>().iter(&world).count(), 1);
    }
}
