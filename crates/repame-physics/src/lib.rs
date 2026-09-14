//! Sim-side character/voxel physics: one AABB mover over one query.

use bevy_ecs::prelude::{Component, Query, Res, Resource};
use bevy_ecs::schedule::IntoScheduleConfigs;
use glam::Vec3;
use repame_sim::{Sim, SimTime};

/// Feet position + half extents (AABB min corner + half sizes): `pos` is the
/// min corner so standing on top of cell `y` means `pos.y == y + 1`.
#[derive(Clone, Copy, Debug)]
pub struct Body {
    pub pos: Vec3,
    pub half_extents: Vec3,
    pub vel: Vec3,
    pub on_ground: bool,
}

impl Body {
    pub fn new(pos: Vec3, half_extents: Vec3) -> Self {
        Self {
            pos,
            half_extents,
            vel: Vec3::ZERO,
            on_ground: false,
        }
    }

    pub fn aabb(&self) -> (Vec3, Vec3) {
        let min = self.pos;
        let max = Vec3::new(
            self.pos.x + self.half_extents.x * 2.0,
            self.pos.y + self.half_extents.y * 2.0,
            self.pos.z + self.half_extents.z * 2.0,
        );
        (min, max)
    }

    pub fn center(&self) -> Vec3 {
        self.pos + self.half_extents
    }
}

/// Last-resolve contact state: what the last [`move_and_collide`] touched,
/// plus the ground/platform velocity underfoot (conveyor ride without
/// stale-velocity hacks).
#[derive(Clone, Debug, Default)]
pub struct PhysicsState {
    pub on_ground: bool,
    pub on_ceiling: bool,
    pub on_wall: Option<[f32; 3]>,
    pub in_fluid: bool,
    pub ground_vel: [f32; 3],
}

/// Collision shapes: boxes plus ramp wedges. Mesh colliders stay a rotated
/// AABB (see [`rotated_box_aabb`]) — explicit instead of mover divergence.
#[derive(Clone, Debug)]
pub enum Collider {
    Box {
        half_extents: [f32; 3],
    },
    /// Ramp rising toward local +X (mirrors block `Slope` shapes).
    Wedge {
        half_extents: [f32; 3],
        rot: u8,
    },
}

/// World solidity query shared by every mover (the single truth that
/// replaces per-system solid tables). Pulse/on-off already resolved by the
/// caller: `is_solid` answers for movement *now*.
pub trait VoxelQuery {
    /// Is `cell` solid for movement?
    fn is_solid(&self, cell: [i32; 3]) -> bool;
    /// Top surface height of `cell` at world (`wx`, `wz`), if any material.
    /// Flat cells return `(cell[1] + 1)`; ramps interpolate; `None` = air.
    fn surface_top(&self, cell: [i32; 3], wx: f32, wz: f32) -> Option<f32>;
    /// Ground/platform velocity at `cell` (conveyors, drift plates).
    fn ground_velocity(&self, cell: [i32; 3]) -> [f32; 3] {
        let _ = cell;
        [0.0, 0.0, 0.0]
    }
}

#[derive(Clone, Debug, Default)]
pub struct MoveResult {
    pub hit_x: bool,
    pub hit_y: bool,
    pub hit_z: bool,
    pub grounded: bool,
    pub ground_vel: [f32; 3],
}

/// Skin offset: bodies rest `SKIN` off surfaces so repeated resolves don't
/// jitter. Small enough to stay invisible at game scale.
pub const SKIN: f32 = 0.001;

/// Segmented axis-separated resolve: integrate in slices so fast bodies
/// can't tunnel through 1-cell walls. `delta` is the full-step displacement
/// (`vel * dt`, caller-owned integration); contact state lands in `state`.
pub fn move_and_collide<Q: VoxelQuery>(
    query: &Q,
    body: &mut Body,
    delta: Vec3,
    state: &mut PhysicsState,
) -> MoveResult {
    let mut result = MoveResult::default();
    let travel = (delta.x.abs() + delta.y.abs() + delta.z.abs()).max(0.0);
    let min_he = body
        .half_extents
        .x
        .min(body.half_extents.y)
        .min(body.half_extents.z)
        .max(0.05);
    let step_len = (min_he * 0.85).clamp(0.05, 0.3);
    let steps = ((travel / step_len).ceil() as usize).clamp(1, 100);

    state.on_ground = false;
    state.on_ceiling = false;
    state.on_wall = None;
    state.ground_vel = [0.0, 0.0, 0.0];

    for _ in 0..steps {
        let step = Vec3::new(
            delta.x / steps as f32,
            delta.y / steps as f32,
            delta.z / steps as f32,
        );
        resolve_axis(query, body, 0, step.x, &mut result, state);
        resolve_axis(query, body, 1, step.y, &mut result, state);
        resolve_axis(query, body, 2, step.z, &mut result, state);
    }

    body.on_ground = result.grounded;
    result
}

fn resolve_axis<Q: VoxelQuery>(
    query: &Q,
    body: &mut Body,
    axis: usize,
    amount: f32,
    result: &mut MoveResult,
    state: &mut PhysicsState,
) {
    if amount == 0.0 {
        return;
    }
    let pos = match axis {
        0 => &mut body.pos.x,
        1 => &mut body.pos.y,
        _ => &mut body.pos.z,
    };
    *pos += amount;

    let (min, max) = body.aabb();
    let x0 = (min.x + SKIN).floor() as i32;
    let x1 = (max.x - SKIN).floor() as i32;
    let y0 = (min.y + SKIN).floor() as i32;
    let y1 = (max.y - SKIN).floor() as i32;
    let z0 = (min.z + SKIN).floor() as i32;
    let z1 = (max.z - SKIN).floor() as i32;

    // Scan direction-aware: resolving against the first overlapping cell in
    // ascending order regardless of direction corrects a -X/-Z move against
    // the far face (leaving penetration) and lands a fall on the lowest
    // stacked top instead of the highest.
    let xs: Vec<i32> = if axis == 0 && amount < 0.0 {
        (x0..=x1).rev().collect()
    } else {
        (x0..=x1).collect()
    };
    let ys: Vec<i32> = if axis == 1 && amount < 0.0 {
        (y0..=y1).rev().collect()
    } else {
        (y0..=y1).collect()
    };
    let zs: Vec<i32> = if axis == 2 && amount < 0.0 {
        (z0..=z1).rev().collect()
    } else {
        (z0..=z1).collect()
    };
    for cx in &xs {
        for cy in &ys {
            for cz in &zs {
                let (cx, cy, cz) = (*cx, *cy, *cz);
                if !query.is_solid([cx, cy, cz]) {
                    continue;
                }
                match axis {
                    0 => {
                        if amount > 0.0 {
                            body.pos.x = cx as f32 - body.half_extents.x * 2.0 - SKIN;
                        } else {
                            body.pos.x = (cx + 1) as f32 + SKIN;
                        }
                        body.vel.x = 0.0;
                        result.hit_x = true;
                        state.on_wall = Some([-amount.signum(), 0.0, 0.0]);
                    }
                    1 => {
                        if amount > 0.0 {
                            body.pos.y = cy as f32 - body.half_extents.y * 2.0 - SKIN;
                            state.on_ceiling = true;
                        } else {
                            let wx = body.pos.x + body.half_extents.x;
                            let wz = body.pos.z + body.half_extents.z;
                            let top = query
                                .surface_top([cx, cy, cz], wx, wz)
                                .unwrap_or((cy + 1) as f32);
                            body.pos.y = top + SKIN;
                            result.grounded = true;
                            state.on_ground = true;
                            state.ground_vel = query.ground_velocity([cx, cy, cz]);
                            result.ground_vel = state.ground_vel;
                        }
                        body.vel.y = 0.0;
                        result.hit_y = true;
                    }
                    _ => {
                        if amount > 0.0 {
                            body.pos.z = cz as f32 - body.half_extents.z * 2.0 - SKIN;
                        } else {
                            body.pos.z = (cz + 1) as f32 + SKIN;
                        }
                        body.vel.z = 0.0;
                        result.hit_z = true;
                        state.on_wall = Some([0.0, 0.0, -amount.signum()]);
                    }
                }
                return;
            }
        }
    }
}

/// Entity-entity pushback: separate overlapping AABBs along the
/// min-penetration axis before the terrain pass. Pure function so crates
/// *and* the player share it.
pub fn pushback(a: &mut Body, b: &mut Body) {
    let (amin, amax) = a.aabb();
    let (bmin, bmax) = b.aabb();
    let ox = (amax.x.min(bmax.x) - amin.x.max(bmin.x)).max(0.0);
    let oy = (amax.y.min(bmax.y) - amin.y.max(bmin.y)).max(0.0);
    let oz = (amax.z.min(bmax.z) - amin.z.max(bmin.z)).max(0.0);
    if ox == 0.0 || oy == 0.0 || oz == 0.0 {
        return;
    }
    if ox <= oy && ox <= oz {
        let push = ox / 2.0 + SKIN;
        if a.pos.x < b.pos.x {
            a.pos.x -= push;
            b.pos.x += push;
        } else {
            a.pos.x += push;
            b.pos.x -= push;
        }
    } else if oy <= ox && oy <= oz {
        let push = oy / 2.0 + SKIN;
        if a.pos.y < b.pos.y {
            a.pos.y -= push;
            b.pos.y += push;
        } else {
            a.pos.y += push;
            b.pos.y -= push;
        }
    } else {
        let push = oz / 2.0 + SKIN;
        if a.pos.z < b.pos.z {
            a.pos.z -= push;
            b.pos.z += push;
        } else {
            a.pos.z += push;
            b.pos.z -= push;
        }
    }
}

/// Rotated box AABB (single implementation both movers share).
pub fn rotated_box_aabb(half: [f32; 3], rot: u8) -> [f32; 3] {
    match rot % 4 {
        1 | 3 => [half[2], half[1], half[0]],
        _ => half,
    }
}

/// Flat level query over a caller-owned solid predicate + uniform top.
/// Adapter for tests and flat-ground games; voxel levels implement
/// [`VoxelQuery`] directly.
#[derive(Clone, Debug, Default, Resource)]
pub struct FlatQuery {
    /// Solid floor top at y = `floor_top` over the whole XZ plane.
    pub floor_top: f32,
}

impl VoxelQuery for FlatQuery {
    fn is_solid(&self, cell: [i32; 3]) -> bool {
        (cell[1] + 1) as f32 <= self.floor_top
    }

    fn surface_top(&self, cell: [i32; 3], _wx: f32, _wz: f32) -> Option<f32> {
        self.is_solid(cell).then_some(self.floor_top)
    }
}

/// Sim component: a body stepped once per fixed step by [`step_bodies`].
/// Gravity/controls stay game-side (games write `vel` before the step);
/// this module integrates `pos += vel * dt` then collides.
#[derive(Component, Clone, Debug)]
pub struct PhysBody {
    pub body: Body,
    pub state: PhysicsState,
}

impl Default for PhysBody {
    fn default() -> Self {
        Self {
            body: Body::new(Vec3::ZERO, Vec3::splat(0.3)),
            state: PhysicsState::default(),
        }
    }
}

/// Renderer-agnostic transform snapshot. Games copy this into their viewport
/// frame (same pattern as the vehicle integration: translation + rotation,
/// no renderer types here).
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct BodyTransform {
    pub translation: Vec3,
    pub yaw_rad: f32,
}

/// Fixed-step integration + collide for every [`PhysBody`], against the
/// [`VoxelQuery`] resource `Q`. Register as
/// `sim.add_system(step_bodies::<MyQuery>)` chained before
/// [`sync_body_transforms`].
pub fn step_bodies<Q: VoxelQuery + Resource>(
    time: Res<SimTime>,
    query: Res<Q>,
    mut bodies: Query<&mut PhysBody>,
) {
    let dt = time.delta_secs.max(0.0);
    for mut b in &mut bodies {
        let PhysBody { body, state } = &mut *b;
        move_and_collide(&*query, body, body.vel * dt, state);
    }
}

/// Copy body positions into [`BodyTransform`] snapshots (translation
/// tracks the body center; yaw stays game-owned).
pub fn sync_body_transforms(mut query: Query<(&PhysBody, &mut BodyTransform)>) {
    for (b, mut t) in &mut query {
        t.translation = b.body.center();
    }
}

/// Register stepping + sync chained (every step runs before every sync).
pub fn register_physics_systems<Q: VoxelQuery + Resource>(sim: &mut Sim) {
    sim.add_chained_systems((step_bodies::<Q>, sync_body_transforms).chain());
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::schedule::IntoScheduleConfigs;
    use std::collections::HashSet;

    struct Floor {
        solids: HashSet<[i32; 3]>,
    }

    impl VoxelQuery for Floor {
        fn is_solid(&self, cell: [i32; 3]) -> bool {
            self.solids.contains(&cell)
        }
        fn surface_top(&self, cell: [i32; 3], _wx: f32, _wz: f32) -> Option<f32> {
            self.solids.contains(&cell).then_some((cell[1] + 1) as f32)
        }
    }

    fn floor() -> Floor {
        let mut solids = HashSet::new();
        for x in -4..=4 {
            for z in -4..=4 {
                solids.insert([x, 0, z]);
            }
        }
        Floor { solids }
    }

    #[test]
    fn falls_and_grounds() {
        let q = floor();
        let mut body = Body::new(Vec3::new(0.0, 3.0, 0.0), Vec3::new(0.3, 0.9, 0.3));
        let mut state = PhysicsState::default();
        for _ in 0..120 {
            let d = Vec3::new(0.0, -10.0 * (1.0 / 60.0), 0.0);
            move_and_collide(&q, &mut body, d, &mut state);
            if state.on_ground {
                break;
            }
        }
        assert!(state.on_ground);
        assert!((body.pos.y - 1.0).abs() < 0.05, "y={}", body.pos.y);
    }

    #[test]
    fn wall_blocks() {
        struct Wall(Floor);
        impl VoxelQuery for Wall {
            fn is_solid(&self, cell: [i32; 3]) -> bool {
                cell == [1, 1, 0] || self.0.is_solid(cell)
            }
            fn surface_top(&self, cell: [i32; 3], wx: f32, wz: f32) -> Option<f32> {
                self.0.surface_top(cell, wx, wz)
            }
        }
        let q = Wall(floor());
        let mut body = Body::new(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.3, 0.9, 0.3));
        body.on_ground = true;
        let mut state = PhysicsState::default();
        move_and_collide(&q, &mut body, Vec3::new(5.0, 0.0, 0.0), &mut state);
        assert!(body.pos.x + 0.6 < 1.0 + 0.01, "x={}", body.pos.x);
    }

    #[test]
    fn pushback_separates() {
        let mut a = Body::new(Vec3::ZERO, Vec3::splat(0.5));
        let mut b = Body::new(Vec3::new(0.5, 0.0, 0.0), Vec3::splat(0.5));
        pushback(&mut a, &mut b);
        let (amin, amax) = a.aabb();
        let (bmin, bmax) = b.aabb();
        assert!(amax.x <= bmin.x + 0.01 || bmax.x <= amin.x + 0.01);
    }

    #[test]
    fn rotated_aabb_swaps_xz_on_quarter_turns() {
        assert_eq!(rotated_box_aabb([1.0, 2.0, 3.0], 0), [1.0, 2.0, 3.0]);
        assert_eq!(rotated_box_aabb([1.0, 2.0, 3.0], 1), [3.0, 2.0, 1.0]);
        assert_eq!(rotated_box_aabb([1.0, 2.0, 3.0], 3), [3.0, 2.0, 1.0]);
    }

    #[test]
    fn sim_steps_bodies_with_gravity() {
        let mut sim = Sim::with_default_step();
        sim.world.insert_resource(FlatQuery { floor_top: 1.0 });
        sim.world.spawn((
            PhysBody {
                body: Body {
                    pos: Vec3::new(0.0, 5.0, 0.0),
                    half_extents: Vec3::new(0.3, 0.9, 0.3),
                    vel: Vec3::ZERO,
                    on_ground: false,
                },
                state: PhysicsState::default(),
            },
            BodyTransform::default(),
        ));
        // Game-side gravity applied as a chained pre-system (the documented
        // pattern: games own forces, this crate owns integration+collide).
        fn gravity(mut q: Query<&mut PhysBody>, time: Res<SimTime>) {
            for mut b in &mut q {
                if !b.body.on_ground {
                    b.body.vel.y -= 20.0 * time.delta_secs;
                }
            }
        }
        sim.add_chained_systems((gravity, step_bodies::<FlatQuery>, sync_body_transforms).chain());
        for _ in 0..240 {
            sim.tick();
        }
        let mut q = sim.world.query::<(&PhysBody, &BodyTransform)>();
        let (b, t) = q.single(&sim.world).expect("one body");
        assert!(b.body.on_ground, "grounded after fall");
        assert!(
            (b.body.pos.y - 1.0).abs() < 0.05,
            "feet rest on floor top: {}",
            b.body.pos.y
        );
        assert!(
            (t.translation.y - (1.0 + 0.9 + SKIN)).abs() < 0.05,
            "snapshot tracks center: {}",
            t.translation.y
        );
    }
}
