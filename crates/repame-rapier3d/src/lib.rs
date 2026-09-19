use std::collections::HashMap;

use bevy_ecs::lifecycle::RemovedComponents;
use bevy_ecs::prelude::{Component, Entity, Query, Res, ResMut, Resource};
use bevy_ecs::schedule::IntoScheduleConfigs;
use glam::{Quat, Vec3};
use rapier3d::dynamics::{
    CCDSolver, ImpulseJointSet, IntegrationParameters, IslandManager, JointAxis, MultibodyJointSet,
    RigidBodyBuilder, RigidBodyHandle, RigidBodySet,
};
use rapier3d::geometry::{BroadPhaseBvh, ColliderSet, InteractionGroups, NarrowPhase};
use rapier3d::pipeline::PhysicsPipeline;
use repame_sim::{Sim, SimTime};

pub use repame_sim;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BodyKind {
    #[default]
    Dynamic,
    Fixed,
    Kinematic,
}

#[derive(Component, Clone, Debug)]
pub struct RapierBody3d {
    pub kind: BodyKind,
    pub spawn_pos: Vec3,
    pub spawn_rot: Quat,
    pub half_extents: Vec3,
    pub mass: f32,
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub memberships: u32,
    pub filter: u32,
    pub lock_rotations: [bool; 3],
    pub gravity_scale: f32,
}

impl Default for RapierBody3d {
    fn default() -> Self {
        Self {
            kind: BodyKind::Dynamic,
            spawn_pos: Vec3::ZERO,
            spawn_rot: Quat::IDENTITY,
            half_extents: Vec3::splat(0.5),
            mass: 1.0,
            linear_damping: 0.0,
            angular_damping: 0.0,
            memberships: 0xFFFF_FFFF,
            filter: 0xFFFF_FFFF,
            lock_rotations: [false; 3],
            gravity_scale: 1.0,
        }
    }
}

#[derive(Component, Clone, Debug)]
pub struct RapierSpherical {
    pub parent: Entity,
    pub child: Entity,
    pub anchor_a: Vec3,
    pub anchor_b: Vec3,
    pub rest_angles: [f32; 3],
    pub stiffness: f32,
    pub damping: f32,
    pub limits: [[f32; 2]; 3],
}

#[derive(Component, Clone, Debug)]
pub struct RapierFixed3d {
    pub parent: Entity,
    pub child: Entity,
    pub anchor_a: Vec3,
    pub anchor_b: Vec3,
}

#[derive(Component, Clone, Debug)]
pub struct RapierPrismatic3d {
    pub parent: Entity,
    pub child: Entity,
    pub anchor_a: Vec3,
    pub anchor_b: Vec3,
    pub axis: Vec3,
    pub limits: [f32; 2],
}

#[derive(Component, Clone, Debug)]
pub struct RapierRope3d {
    pub parent: Entity,
    pub child: Entity,
    pub anchor_a: Vec3,
    pub anchor_b: Vec3,
    pub max_length: f32,
}

#[derive(Component, Clone, Debug)]
pub struct RapierSpring3d {
    pub parent: Entity,
    pub child: Entity,
    pub anchor_a: Vec3,
    pub anchor_b: Vec3,
    pub stiffness: f32,
    pub damping: f32,
    pub rest_length: f32,
}

#[derive(Component, Clone, Copy, Debug, Default)]
pub struct ImpulseRequest3d {
    pub linear: Vec3,
    pub torque: Vec3,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RayHit3d {
    pub distance: f32,
    pub point: Vec3,
    pub normal: Vec3,
    pub surface: u32,
}

#[derive(Component, Clone, Copy, Debug, Default)]
pub struct BodySnapshot3d {
    pub pos: Vec3,
    pub rot: Quat,
    pub linvel: Vec3,
    pub angvel: Vec3,
}

#[derive(Resource)]
pub struct RapierWorld3d {
    pub gravity: Vec3,
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    pipeline: PhysicsPipeline,
    islands: IslandManager,
    broad_phase: BroadPhaseBvh,
    narrow_phase: NarrowPhase,
    ccd: CCDSolver,
    params: IntegrationParameters,
    entity_to_body: HashMap<Entity, RigidBodyHandle>,
    entity_to_joint: HashMap<Entity, rapier3d::dynamics::ImpulseJointHandle>,
}

impl RapierWorld3d {
    pub fn new(gravity: Vec3, length_unit: f32) -> Self {
        let params = IntegrationParameters {
            length_unit: length_unit.max(1e-6),
            ..Default::default()
        };
        Self {
            gravity,
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            pipeline: PhysicsPipeline::new(),
            islands: IslandManager::new(),
            broad_phase: BroadPhaseBvh::new(),
            narrow_phase: NarrowPhase::new(),
            ccd: CCDSolver::new(),
            params,
            entity_to_body: HashMap::new(),
            entity_to_joint: HashMap::new(),
        }
    }

    pub fn body_count(&self) -> usize {
        self.entity_to_body.len()
    }

    pub fn joint_count(&self) -> usize {
        self.entity_to_joint.len()
    }

    pub fn body_handle(&self, entity: Entity) -> Option<RigidBodyHandle> {
        self.entity_to_body.get(&entity).copied()
    }

    pub fn cast_ray(
        &self,
        origin: Vec3,
        dir: Vec3,
        max_dist: f32,
        solid: bool,
    ) -> Option<RayHit3d> {
        if max_dist <= 0.0 || dir.length_squared() < 1e-8 {
            return None;
        }
        let ray = rapier3d::geometry::Ray::new(to_vec(origin), to_vec(dir));
        let pipeline = self.broad_phase.as_query_pipeline(
            self.narrow_phase.query_dispatcher(),
            &self.bodies,
            &self.colliders,
            rapier3d::pipeline::QueryFilter::default(),
        );
        let (handle, hit) = pipeline.cast_ray_and_get_normal(&ray, max_dist.max(0.0), solid)?;
        let collider = self.colliders.get(handle)?;
        Some(RayHit3d {
            distance: hit.time_of_impact,
            point: origin + dir.normalize_or_zero() * hit.time_of_impact,
            normal: Vec3::new(hit.normal.x, hit.normal.y, hit.normal.z).normalize_or_zero(),
            surface: collider.user_data as u32,
        })
    }

    fn step_once(&mut self, dt: f32) {
        self.params.dt = dt.max(1e-6);
        let gravity = rapier3d::math::Vector::new(self.gravity.x, self.gravity.y, self.gravity.z);
        self.pipeline.step(
            gravity,
            &self.params,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd,
            &(),
            &(),
        );
    }
}

fn group_bits(raw: u32, fallback_all: bool) -> rapier3d::geometry::Group {
    let bits = if raw == 0 && fallback_all {
        0xFFFF_FFFF
    } else {
        raw
    };
    rapier3d::geometry::Group::from_bits(bits).unwrap_or(rapier3d::geometry::Group::ALL)
}

fn to_vec(p: Vec3) -> rapier3d::math::Vector {
    rapier3d::math::Vector::new(p.x, p.y, p.z)
}

fn to_quat(q: Quat) -> rapier3d::math::Rotation {
    q.normalize()
}

fn from_quat(r: &rapier3d::math::Rotation) -> Quat {
    r.normalize()
}

fn spawn_bodies(mut world: ResMut<RapierWorld3d>, mut bodies: Query<(Entity, &RapierBody3d)>) {
    for (entity, spec) in &mut bodies {
        if world.entity_to_body.contains_key(&entity) {
            continue;
        }
        let hx = spec.half_extents.x.max(1e-4);
        let hy = spec.half_extents.y.max(1e-4);
        let hz = spec.half_extents.z.max(1e-4);
        let mut builder = match spec.kind {
            BodyKind::Dynamic => RigidBodyBuilder::dynamic(),
            BodyKind::Fixed => RigidBodyBuilder::fixed(),
            BodyKind::Kinematic => RigidBodyBuilder::kinematic_position_based(),
        };
        builder = builder
            .translation(to_vec(spec.spawn_pos))
            .pose(rapier3d::math::Pose::from_parts(
                to_vec(spec.spawn_pos),
                to_quat(spec.spawn_rot),
            ))
            .linear_damping(spec.linear_damping.max(0.0))
            .angular_damping(spec.angular_damping.max(0.0))
            .gravity_scale(spec.gravity_scale)
            .enabled_rotations(
                !spec.lock_rotations[0],
                !spec.lock_rotations[1],
                !spec.lock_rotations[2],
            )
            .can_sleep(false);
        let handle = world.bodies.insert(builder);
        {
            let RapierWorld3d {
                bodies, colliders, ..
            } = &mut *world;
            if let Some(body) = bodies.get_mut(handle) {
                if spec.kind == BodyKind::Dynamic && spec.mass > 0.0 {
                    body.set_additional_mass(spec.mass.max(1e-4), true);
                }
                let groups = InteractionGroups::new(
                    group_bits(spec.memberships, true),
                    group_bits(spec.filter, true),
                    rapier3d::geometry::InteractionTestMode::default(),
                );
                let collider = rapier3d::geometry::ColliderBuilder::cuboid(hx, hy, hz)
                    .collision_groups(groups)
                    .build();
                colliders.insert_with_parent(collider, handle, bodies);
            }
        }
        world.entity_to_body.insert(entity, handle);
    }
}

fn joint_handles(
    world: &RapierWorld3d,
    parent: Entity,
    child: Entity,
) -> Option<(
    rapier3d::dynamics::RigidBodyHandle,
    rapier3d::dynamics::RigidBodyHandle,
)> {
    Some((
        *world.entity_to_body.get(&parent)?,
        *world.entity_to_body.get(&child)?,
    ))
}

fn track_joint(
    world: &mut RapierWorld3d,
    entity: Entity,
    parent_handle: rapier3d::dynamics::RigidBodyHandle,
    child_handle: rapier3d::dynamics::RigidBodyHandle,
    joint: impl Into<rapier3d::dynamics::GenericJoint>,
) {
    let handle = world
        .impulse_joints
        .insert(parent_handle, child_handle, joint, true);
    world.entity_to_joint.insert(entity, handle);
}

fn spawn_spherical_joints(
    mut world: ResMut<RapierWorld3d>,
    mut joints: Query<(Entity, &RapierSpherical)>,
) {
    for (entity, spec) in &mut joints {
        if world.entity_to_joint.contains_key(&entity) {
            continue;
        }
        let Some((parent_handle, child_handle)) = joint_handles(&world, spec.parent, spec.child)
        else {
            continue;
        };
        let mut builder = rapier3d::dynamics::SphericalJointBuilder::new()
            .local_anchor1(to_vec(spec.anchor_a))
            .local_anchor2(to_vec(spec.anchor_b));
        for (axis, rest, limits) in [
            (JointAxis::AngX, spec.rest_angles[0], spec.limits[0]),
            (JointAxis::AngY, spec.rest_angles[1], spec.limits[1]),
            (JointAxis::AngZ, spec.rest_angles[2], spec.limits[2]),
        ] {
            builder =
                builder.motor_position(axis, rest, spec.stiffness.max(0.0), spec.damping.max(0.0));
            builder = builder.limits(axis, limits);
        }
        track_joint(
            &mut world,
            entity,
            parent_handle,
            child_handle,
            builder.build(),
        );
    }
}

fn spawn_fixed_joints(
    mut world: ResMut<RapierWorld3d>,
    mut joints: Query<(Entity, &RapierFixed3d)>,
) {
    for (entity, spec) in &mut joints {
        if world.entity_to_joint.contains_key(&entity) {
            continue;
        }
        let Some((parent_handle, child_handle)) = joint_handles(&world, spec.parent, spec.child)
        else {
            continue;
        };
        let builder = rapier3d::dynamics::FixedJointBuilder::new()
            .local_anchor1(to_vec(spec.anchor_a))
            .local_anchor2(to_vec(spec.anchor_b))
            .build();
        track_joint(&mut world, entity, parent_handle, child_handle, builder);
    }
}

fn spawn_prismatic_joints(
    mut world: ResMut<RapierWorld3d>,
    mut joints: Query<(Entity, &RapierPrismatic3d)>,
) {
    for (entity, spec) in &mut joints {
        if world.entity_to_joint.contains_key(&entity) {
            continue;
        }
        let Some((parent_handle, child_handle)) = joint_handles(&world, spec.parent, spec.child)
        else {
            continue;
        };
        let axis = spec.axis.normalize_or_zero();
        let builder = rapier3d::dynamics::PrismaticJointBuilder::new(to_vec(axis))
            .local_anchor1(to_vec(spec.anchor_a))
            .local_anchor2(to_vec(spec.anchor_b))
            .limits(spec.limits)
            .build();
        track_joint(&mut world, entity, parent_handle, child_handle, builder);
    }
}

fn spawn_rope_joints(mut world: ResMut<RapierWorld3d>, mut joints: Query<(Entity, &RapierRope3d)>) {
    for (entity, spec) in &mut joints {
        if world.entity_to_joint.contains_key(&entity) {
            continue;
        }
        let Some((parent_handle, child_handle)) = joint_handles(&world, spec.parent, spec.child)
        else {
            continue;
        };
        let builder = rapier3d::dynamics::RopeJointBuilder::new(spec.max_length.max(1e-4))
            .local_anchor1(to_vec(spec.anchor_a))
            .local_anchor2(to_vec(spec.anchor_b))
            .build();
        track_joint(&mut world, entity, parent_handle, child_handle, builder);
    }
}

fn spawn_spring_joints(
    mut world: ResMut<RapierWorld3d>,
    mut joints: Query<(Entity, &RapierSpring3d)>,
) {
    for (entity, spec) in &mut joints {
        if world.entity_to_joint.contains_key(&entity) {
            continue;
        }
        let Some((parent_handle, child_handle)) = joint_handles(&world, spec.parent, spec.child)
        else {
            continue;
        };
        let builder = rapier3d::dynamics::SpringJointBuilder::new(
            spec.rest_length.max(0.0),
            spec.stiffness.max(0.0),
            spec.damping.max(0.0),
        )
        .local_anchor1(to_vec(spec.anchor_a))
        .local_anchor2(to_vec(spec.anchor_b))
        .build();
        track_joint(&mut world, entity, parent_handle, child_handle, builder);
    }
}

fn apply_impulse_requests(
    mut world: ResMut<RapierWorld3d>,
    mut requests: Query<(Entity, &mut ImpulseRequest3d)>,
) {
    for (entity, mut request) in &mut requests {
        let Some(&handle) = world.entity_to_body.get(&entity) else {
            continue;
        };
        if let Some(body) = world.bodies.get_mut(handle) {
            if request.linear != Vec3::ZERO {
                body.apply_impulse(to_vec(request.linear), true);
            }
            if request.torque != Vec3::ZERO {
                body.apply_torque_impulse(to_vec(request.torque), true);
            }
        }
        request.linear = Vec3::ZERO;
        request.torque = Vec3::ZERO;
    }
}

fn step_rapier(mut world: ResMut<RapierWorld3d>, time: ResMut<SimTime>) {
    world.step_once(time.delta_secs);
}

fn sync_snapshots(world: Res<RapierWorld3d>, mut snapshots: Query<(Entity, &mut BodySnapshot3d)>) {
    for (entity, mut snapshot) in &mut snapshots {
        let Some(&handle) = world.entity_to_body.get(&entity) else {
            continue;
        };
        let Some(body) = world.bodies.get(handle) else {
            continue;
        };
        let translation = body.translation();
        snapshot.pos = Vec3::new(translation.x, translation.y, translation.z);
        snapshot.rot = from_quat(body.rotation());
        let linvel = body.linvel();
        snapshot.linvel = Vec3::new(linvel.x, linvel.y, linvel.z);
        let angvel = body.angvel();
        snapshot.angvel = Vec3::new(angvel.x, angvel.y, angvel.z);
    }
}

fn despawn_cleanup(
    mut world: ResMut<RapierWorld3d>,
    mut removed_bodies: RemovedComponents<RapierBody3d>,
    mut removed_spherical: RemovedComponents<RapierSpherical>,
    mut removed_fixed: RemovedComponents<RapierFixed3d>,
    mut removed_prismatic: RemovedComponents<RapierPrismatic3d>,
    mut removed_rope: RemovedComponents<RapierRope3d>,
    mut removed_spring: RemovedComponents<RapierSpring3d>,
) {
    for entity in removed_bodies.read() {
        if let Some(handle) = world.entity_to_body.remove(&entity) {
            let RapierWorld3d {
                bodies,
                islands,
                colliders,
                impulse_joints,
                multibody_joints,
                ..
            } = &mut *world;
            bodies.remove(
                handle,
                islands,
                colliders,
                impulse_joints,
                multibody_joints,
                true,
            );
        }
    }
    for entity in removed_spherical
        .read()
        .chain(removed_fixed.read())
        .chain(removed_prismatic.read())
        .chain(removed_rope.read())
        .chain(removed_spring.read())
    {
        if let Some(handle) = world.entity_to_joint.remove(&entity) {
            world.impulse_joints.remove(handle, true);
        }
    }
}

pub fn init_world(world: &mut bevy_ecs::prelude::World, gravity: Vec3, length_unit: f32) {
    world.insert_resource(RapierWorld3d::new(gravity, length_unit));
}

pub fn register_rapier3d_systems(sim: &mut Sim) {
    sim.add_chained_systems(
        (
            spawn_bodies,
            spawn_spherical_joints,
            spawn_fixed_joints,
            spawn_prismatic_joints,
            spawn_rope_joints,
            spawn_spring_joints,
            apply_impulse_requests,
            step_rapier,
            sync_snapshots,
            despawn_cleanup,
        )
            .chain(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sim_with_world() -> Sim {
        let mut sim = Sim::with_default_step();
        sim.world
            .insert_resource(RapierWorld3d::new(Vec3::new(0.0, -9.81, 0.0), 1.0));
        register_rapier3d_systems(&mut sim);
        sim
    }

    #[test]
    fn raycast_hits_fixed_floor_with_surface_id() {
        let mut sim = sim_with_world();
        sim.world.spawn(RapierBody3d {
            kind: BodyKind::Fixed,
            spawn_pos: Vec3::new(0.0, -0.5, 0.0),
            half_extents: Vec3::new(10.0, 0.5, 10.0),
            ..Default::default()
        });
        sim.tick();
        sim.tick();
        let world = sim.world.resource::<RapierWorld3d>();
        let hit = world
            .cast_ray(Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y, 20.0, true)
            .expect("ray hits the floor");
        assert!((hit.distance - 5.0).abs() < 0.05);
        assert!((hit.normal - Vec3::Y).length() < 1e-3);
        assert!((hit.point.y - 0.0).abs() < 0.05);
        assert!(
            world
                .cast_ray(Vec3::new(0.0, 5.0, 0.0), Vec3::Y, 20.0, true)
                .is_none()
        );
        assert!(
            world
                .cast_ray(Vec3::new(0.0, 5.0, 0.0), Vec3::ZERO, 20.0, true)
                .is_none()
        );
    }

    #[test]
    fn pause_means_identical_snapshots() {
        let mut sim = sim_with_world();
        let entity = sim
            .world
            .spawn((
                RapierBody3d {
                    spawn_pos: Vec3::new(0.0, 10.0, 0.0),
                    half_extents: Vec3::splat(0.5),
                    ..Default::default()
                },
                BodySnapshot3d::default(),
            ))
            .id();
        for _ in 0..10 {
            sim.tick();
        }
        let before = *sim.world.get::<BodySnapshot3d>(entity).unwrap();
        let bodies_before = sim.world.resource::<RapierWorld3d>().body_count();
        for _ in 0..60 {
            let after = *sim.world.get::<BodySnapshot3d>(entity).unwrap();
            assert_eq!(after.pos, before.pos);
            assert_eq!(after.linvel, before.linvel);
            assert_eq!(after.angvel, before.angvel);
        }
        assert_eq!(
            sim.world.resource::<RapierWorld3d>().body_count(),
            bodies_before
        );
    }

    #[test]
    fn puppet_pair_spawns_two_bodies_one_spherical_joint() {
        let mut sim = sim_with_world();
        let torso = sim
            .world
            .spawn((
                RapierBody3d {
                    spawn_pos: Vec3::ZERO,
                    half_extents: Vec3::new(0.3, 0.5, 0.2),
                    ..Default::default()
                },
                BodySnapshot3d::default(),
            ))
            .id();
        let head = sim
            .world
            .spawn((
                RapierBody3d {
                    spawn_pos: Vec3::new(0.0, 0.8, 0.0),
                    half_extents: Vec3::splat(0.2),
                    ..Default::default()
                },
                BodySnapshot3d::default(),
            ))
            .id();
        sim.world.spawn(RapierSpherical {
            parent: torso,
            child: head,
            anchor_a: Vec3::new(0.0, 0.5, 0.0),
            anchor_b: Vec3::new(0.0, -0.2, 0.0),
            rest_angles: [0.0; 3],
            stiffness: 10.0,
            damping: 2.0,
            limits: [[-0.78, 0.78]; 3],
        });
        sim.tick();
        sim.tick();
        let world = sim.world.resource::<RapierWorld3d>();
        assert_eq!(world.body_count(), 2);
        assert_eq!(world.joint_count(), 1);
    }

    #[test]
    fn knockback_impulse_moves_body_and_settles() {
        let mut sim = sim_with_world();
        let entity = sim
            .world
            .spawn((
                RapierBody3d {
                    spawn_pos: Vec3::ZERO,
                    half_extents: Vec3::splat(0.5),
                    gravity_scale: 0.0,
                    linear_damping: 1.2,
                    ..Default::default()
                },
                ImpulseRequest3d::default(),
                BodySnapshot3d::default(),
            ))
            .id();
        sim.tick();
        sim.world
            .get_mut::<ImpulseRequest3d>(entity)
            .unwrap()
            .linear = Vec3::new(50.0, 0.0, 0.0);
        sim.tick();
        let pushed = *sim.world.get::<BodySnapshot3d>(entity).unwrap();
        assert!(pushed.pos.x > 0.0);
        assert!(pushed.linvel.x > 0.0);
        let request = sim.world.get::<ImpulseRequest3d>(entity).unwrap();
        assert_eq!(request.linear, Vec3::ZERO);
        assert_eq!(request.torque, Vec3::ZERO);
        for _ in 0..240 {
            sim.tick();
        }
        let settled = *sim.world.get::<BodySnapshot3d>(entity).unwrap();
        assert!(settled.linvel.x.abs() < pushed.linvel.x.abs());
    }

    #[test]
    fn falling_body_grounds_on_fixed_floor() {
        let mut sim = sim_with_world();
        let entity = sim
            .world
            .spawn((
                RapierBody3d {
                    spawn_pos: Vec3::new(0.0, 5.0, 0.0),
                    half_extents: Vec3::new(0.3, 0.9, 0.3),
                    ..Default::default()
                },
                BodySnapshot3d::default(),
            ))
            .id();
        sim.world.spawn(RapierBody3d {
            kind: BodyKind::Fixed,
            spawn_pos: Vec3::new(0.0, -0.5, 0.0),
            half_extents: Vec3::new(10.0, 0.5, 10.0),
            ..Default::default()
        });
        for _ in 0..240 {
            sim.tick();
        }
        let snapshot = *sim.world.get::<BodySnapshot3d>(entity).unwrap();
        assert!((snapshot.pos.y - 0.9).abs() < 0.3);
        assert!(snapshot.linvel.length() < 1.0);
    }

    #[test]
    fn joint_removal_separates_pair() {
        let mut sim = sim_with_world();
        let torso = sim
            .world
            .spawn((
                RapierBody3d {
                    spawn_pos: Vec3::new(0.0, 5.0, 0.0),
                    half_extents: Vec3::new(0.3, 0.5, 0.2),
                    ..Default::default()
                },
                BodySnapshot3d::default(),
            ))
            .id();
        let held = sim
            .world
            .spawn((
                RapierBody3d {
                    spawn_pos: Vec3::new(0.0, 4.0, 0.0),
                    half_extents: Vec3::splat(0.2),
                    memberships: 0x0002,
                    filter: 0x0001,
                    ..Default::default()
                },
                BodySnapshot3d::default(),
            ))
            .id();
        let joint_entity = sim
            .world
            .spawn(RapierSpherical {
                parent: torso,
                child: held,
                anchor_a: Vec3::new(0.0, -0.4, 0.0),
                anchor_b: Vec3::new(0.0, 0.2, 0.0),
                rest_angles: [0.0; 3],
                stiffness: 15.0,
                damping: 4.0,
                limits: [[-0.1, 0.1]; 3],
            })
            .id();
        sim.world.spawn(RapierBody3d {
            kind: BodyKind::Fixed,
            spawn_pos: Vec3::new(0.0, -0.5, 0.0),
            half_extents: Vec3::new(10.0, 0.5, 10.0),
            ..Default::default()
        });
        for _ in 0..60 {
            sim.tick();
        }
        let before = *sim.world.get::<BodySnapshot3d>(held).unwrap();
        let torso_before = *sim.world.get::<BodySnapshot3d>(torso).unwrap();
        sim.world
            .entity_mut(joint_entity)
            .remove::<RapierSpherical>();
        sim.world.entity_mut(torso).insert(ImpulseRequest3d {
            linear: Vec3::new(200.0, 0.0, 0.0),
            torque: Vec3::new(0.0, 0.0, 50.0),
        });
        sim.world.entity_mut(held).insert(ImpulseRequest3d {
            linear: Vec3::new(-200.0, 0.0, 0.0),
            torque: Vec3::new(0.0, 0.0, -50.0),
        });
        for _ in 0..240 {
            sim.tick();
        }
        assert_eq!(sim.world.resource::<RapierWorld3d>().joint_count(), 0);
        let after = *sim.world.get::<BodySnapshot3d>(held).unwrap();
        let torso_after = *sim.world.get::<BodySnapshot3d>(torso).unwrap();
        assert!((after.pos - before.pos).length() > 0.5);
        assert!((torso_after.pos - torso_before.pos).length() > 0.5);
    }
}
