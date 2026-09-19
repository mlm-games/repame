use std::collections::HashMap;

use bevy_ecs::lifecycle::RemovedComponents;
use bevy_ecs::prelude::{Component, Entity, Query, Res, ResMut, Resource};
use bevy_ecs::schedule::IntoScheduleConfigs;
use glam::Vec2;
use rapier2d::dynamics::{
    CCDSolver, ImpulseJointSet, IntegrationParameters, IslandManager, MultibodyJointSet,
    RigidBodyBuilder, RigidBodyHandle, RigidBodySet,
};
use rapier2d::geometry::{BroadPhaseBvh, ColliderSet, InteractionGroups, NarrowPhase};
use rapier2d::pipeline::PhysicsPipeline;
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
pub struct RapierBody2d {
    pub kind: BodyKind,
    pub spawn_pos: Vec2,
    pub spawn_angle: f32,
    pub half_extents: Vec2,
    pub mass: f32,
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub memberships: u32,
    pub filter: u32,
    pub lock_rotation: bool,
    pub gravity_scale: f32,
}

impl Default for RapierBody2d {
    fn default() -> Self {
        Self {
            kind: BodyKind::Dynamic,
            spawn_pos: Vec2::ZERO,
            spawn_angle: 0.0,
            half_extents: Vec2::splat(0.5),
            mass: 1.0,
            linear_damping: 0.0,
            angular_damping: 0.0,
            memberships: 0xFFFF_FFFF,
            filter: 0xFFFF_FFFF,
            lock_rotation: false,
            gravity_scale: 1.0,
        }
    }
}

#[derive(Component, Clone, Debug)]
pub struct RapierRevolute {
    pub parent: Entity,
    pub child: Entity,
    pub anchor_a: Vec2,
    pub anchor_b: Vec2,
    pub rest_angle: f32,
    pub stiffness: f32,
    pub damping: f32,
    pub limits: [f32; 2],
}

#[derive(Component, Clone, Debug)]
pub struct RapierFixed {
    pub parent: Entity,
    pub child: Entity,
    pub anchor_a: Vec2,
    pub anchor_b: Vec2,
}

#[derive(Component, Clone, Debug)]
pub struct RapierPrismatic {
    pub parent: Entity,
    pub child: Entity,
    pub anchor_a: Vec2,
    pub anchor_b: Vec2,
    pub axis: Vec2,
    pub limits: [f32; 2],
}

#[derive(Component, Clone, Debug)]
pub struct RapierRope {
    pub parent: Entity,
    pub child: Entity,
    pub anchor_a: Vec2,
    pub anchor_b: Vec2,
    pub max_length: f32,
}

#[derive(Component, Clone, Debug)]
pub struct RapierSpring {
    pub parent: Entity,
    pub child: Entity,
    pub anchor_a: Vec2,
    pub anchor_b: Vec2,
    pub stiffness: f32,
    pub damping: f32,
    pub rest_length: f32,
}

#[derive(Component, Clone, Copy, Debug, Default)]
pub struct ImpulseRequest {
    pub linear: Vec2,
    pub torque: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RayHit2d {
    pub distance: f32,
    pub point: Vec2,
    pub normal: Vec2,
    pub surface: u32,
}

#[derive(Component, Clone, Copy, Debug, Default)]
pub struct BodySnapshot2d {
    pub pos: Vec2,
    pub angle: f32,
    pub linvel: Vec2,
    pub angvel: f32,
}

#[derive(Resource)]
pub struct RapierWorld2d {
    pub gravity: Vec2,
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
    entity_to_joint: HashMap<Entity, rapier2d::dynamics::ImpulseJointHandle>,
}

impl RapierWorld2d {
    pub fn new(gravity: Vec2, length_unit: f32) -> Self {
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
        origin: Vec2,
        dir: Vec2,
        max_dist: f32,
        solid: bool,
    ) -> Option<RayHit2d> {
        if max_dist <= 0.0 || dir.length_squared() < 1e-8 {
            return None;
        }
        let ray = rapier2d::geometry::Ray::new(
            rapier2d::math::Vector::new(origin.x, origin.y),
            rapier2d::math::Vector::new(dir.x, dir.y),
        );
        let pipeline = self.broad_phase.as_query_pipeline(
            self.narrow_phase.query_dispatcher(),
            &self.bodies,
            &self.colliders,
            rapier2d::pipeline::QueryFilter::default(),
        );
        let (handle, hit) = pipeline.cast_ray_and_get_normal(&ray, max_dist.max(0.0), solid)?;
        let collider = self.colliders.get(handle)?;
        let normal = Vec2::new(hit.normal.x, hit.normal.y).normalize_or_zero();
        Some(RayHit2d {
            distance: hit.time_of_impact,
            point: origin + dir.normalize_or_zero() * hit.time_of_impact,
            normal,
            surface: collider.user_data as u32,
        })
    }

    fn step_once(&mut self, dt: f32) {
        self.params.dt = dt.max(1e-6);
        let gravity = rapier2d::math::Vector::new(self.gravity.x, self.gravity.y);
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

fn group_bits(raw: u32, fallback_all: bool) -> rapier2d::geometry::Group {
    let bits = if raw == 0 && fallback_all {
        0xFFFF_FFFF
    } else {
        raw
    };
    rapier2d::geometry::Group::from_bits(bits).unwrap_or(rapier2d::geometry::Group::ALL)
}

fn spawn_bodies(mut world: ResMut<RapierWorld2d>, mut bodies: Query<(Entity, &RapierBody2d)>) {
    for (entity, spec) in &mut bodies {
        if world.entity_to_body.contains_key(&entity) {
            continue;
        }
        let hx = spec.half_extents.x.max(1e-4);
        let hy = spec.half_extents.y.max(1e-4);
        let mut builder = match spec.kind {
            BodyKind::Dynamic => RigidBodyBuilder::dynamic(),
            BodyKind::Fixed => RigidBodyBuilder::fixed(),
            BodyKind::Kinematic => RigidBodyBuilder::kinematic_position_based(),
        };
        builder = builder
            .translation(rapier2d::math::Vector::new(
                spec.spawn_pos.x,
                spec.spawn_pos.y,
            ))
            .rotation(spec.spawn_angle)
            .linear_damping(spec.linear_damping.max(0.0))
            .angular_damping(spec.angular_damping.max(0.0))
            .gravity_scale(spec.gravity_scale)
            .can_sleep(false);
        if spec.lock_rotation {
            builder = builder.lock_rotations();
        }
        let handle = world.bodies.insert(builder);
        {
            let RapierWorld2d {
                bodies, colliders, ..
            } = &mut *world;
            if let Some(body) = bodies.get_mut(handle) {
                if spec.kind == BodyKind::Dynamic && spec.mass > 0.0 {
                    body.set_additional_mass(spec.mass.max(1e-4), true);
                }
                let groups = InteractionGroups::new(
                    group_bits(spec.memberships, true),
                    group_bits(spec.filter, true),
                    rapier2d::geometry::InteractionTestMode::default(),
                );
                let collider = rapier2d::geometry::ColliderBuilder::cuboid(hx, hy)
                    .collision_groups(groups)
                    .build();
                colliders.insert_with_parent(collider, handle, bodies);
            }
        }
        world.entity_to_body.insert(entity, handle);
    }
}

fn joint_handles(
    world: &RapierWorld2d,
    parent: Entity,
    child: Entity,
) -> Option<(
    rapier2d::dynamics::RigidBodyHandle,
    rapier2d::dynamics::RigidBodyHandle,
)> {
    Some((
        *world.entity_to_body.get(&parent)?,
        *world.entity_to_body.get(&child)?,
    ))
}

fn track_joint(
    world: &mut RapierWorld2d,
    entity: Entity,
    parent_handle: rapier2d::dynamics::RigidBodyHandle,
    child_handle: rapier2d::dynamics::RigidBodyHandle,
    joint: impl Into<rapier2d::dynamics::GenericJoint>,
) {
    let handle = world
        .impulse_joints
        .insert(parent_handle, child_handle, joint, true);
    world.entity_to_joint.insert(entity, handle);
}

fn spawn_joints(mut world: ResMut<RapierWorld2d>, mut joints: Query<(Entity, &RapierRevolute)>) {
    for (entity, spec) in &mut joints {
        if world.entity_to_joint.contains_key(&entity) {
            continue;
        }
        let Some((parent_handle, child_handle)) = joint_handles(&world, spec.parent, spec.child)
        else {
            continue;
        };
        let builder = rapier2d::dynamics::RevoluteJointBuilder::new()
            .local_anchor1(rapier2d::math::Vector::new(
                spec.anchor_a.x,
                spec.anchor_a.y,
            ))
            .local_anchor2(rapier2d::math::Vector::new(
                spec.anchor_b.x,
                spec.anchor_b.y,
            ))
            .motor_position(
                spec.rest_angle,
                spec.stiffness.max(0.0),
                spec.damping.max(0.0),
            )
            .limits(spec.limits)
            .build();
        track_joint(&mut world, entity, parent_handle, child_handle, builder);
    }
}

fn spawn_fixed_joints(mut world: ResMut<RapierWorld2d>, mut joints: Query<(Entity, &RapierFixed)>) {
    for (entity, spec) in &mut joints {
        if world.entity_to_joint.contains_key(&entity) {
            continue;
        }
        let Some((parent_handle, child_handle)) = joint_handles(&world, spec.parent, spec.child)
        else {
            continue;
        };
        let builder = rapier2d::dynamics::FixedJointBuilder::new()
            .local_anchor1(rapier2d::math::Vector::new(
                spec.anchor_a.x,
                spec.anchor_a.y,
            ))
            .local_anchor2(rapier2d::math::Vector::new(
                spec.anchor_b.x,
                spec.anchor_b.y,
            ))
            .build();
        track_joint(&mut world, entity, parent_handle, child_handle, builder);
    }
}

fn spawn_prismatic_joints(
    mut world: ResMut<RapierWorld2d>,
    mut joints: Query<(Entity, &RapierPrismatic)>,
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
        let builder = rapier2d::dynamics::PrismaticJointBuilder::new(rapier2d::math::Vector::new(
            axis.x, axis.y,
        ))
        .local_anchor1(rapier2d::math::Vector::new(
            spec.anchor_a.x,
            spec.anchor_a.y,
        ))
        .local_anchor2(rapier2d::math::Vector::new(
            spec.anchor_b.x,
            spec.anchor_b.y,
        ))
        .limits(spec.limits)
        .build();
        track_joint(&mut world, entity, parent_handle, child_handle, builder);
    }
}

fn spawn_rope_joints(mut world: ResMut<RapierWorld2d>, mut joints: Query<(Entity, &RapierRope)>) {
    for (entity, spec) in &mut joints {
        if world.entity_to_joint.contains_key(&entity) {
            continue;
        }
        let Some((parent_handle, child_handle)) = joint_handles(&world, spec.parent, spec.child)
        else {
            continue;
        };
        let builder = rapier2d::dynamics::RopeJointBuilder::new(spec.max_length.max(1e-4))
            .local_anchor1(rapier2d::math::Vector::new(
                spec.anchor_a.x,
                spec.anchor_a.y,
            ))
            .local_anchor2(rapier2d::math::Vector::new(
                spec.anchor_b.x,
                spec.anchor_b.y,
            ))
            .build();
        track_joint(&mut world, entity, parent_handle, child_handle, builder);
    }
}

fn spawn_spring_joints(
    mut world: ResMut<RapierWorld2d>,
    mut joints: Query<(Entity, &RapierSpring)>,
) {
    for (entity, spec) in &mut joints {
        if world.entity_to_joint.contains_key(&entity) {
            continue;
        }
        let Some((parent_handle, child_handle)) = joint_handles(&world, spec.parent, spec.child)
        else {
            continue;
        };
        let builder = rapier2d::dynamics::SpringJointBuilder::new(
            spec.rest_length.max(0.0),
            spec.stiffness.max(0.0),
            spec.damping.max(0.0),
        )
        .local_anchor1(rapier2d::math::Vector::new(
            spec.anchor_a.x,
            spec.anchor_a.y,
        ))
        .local_anchor2(rapier2d::math::Vector::new(
            spec.anchor_b.x,
            spec.anchor_b.y,
        ))
        .build();
        track_joint(&mut world, entity, parent_handle, child_handle, builder);
    }
}

fn apply_impulse_requests(
    mut world: ResMut<RapierWorld2d>,
    mut requests: Query<(Entity, &mut ImpulseRequest)>,
) {
    for (entity, mut request) in &mut requests {
        let Some(&handle) = world.entity_to_body.get(&entity) else {
            continue;
        };
        if let Some(body) = world.bodies.get_mut(handle) {
            if request.linear != Vec2::ZERO {
                body.apply_impulse(
                    rapier2d::math::Vector::new(request.linear.x, request.linear.y),
                    true,
                );
            }
            if request.torque != 0.0 {
                body.apply_torque_impulse(request.torque, true);
            }
        }
        request.linear = Vec2::ZERO;
        request.torque = 0.0;
    }
}

fn step_rapier(mut world: ResMut<RapierWorld2d>, time: ResMut<SimTime>) {
    world.step_once(time.delta_secs);
}

fn sync_snapshots(world: Res<RapierWorld2d>, mut snapshots: Query<(Entity, &mut BodySnapshot2d)>) {
    for (entity, mut snapshot) in &mut snapshots {
        let Some(&handle) = world.entity_to_body.get(&entity) else {
            continue;
        };
        let Some(body) = world.bodies.get(handle) else {
            continue;
        };
        let translation = body.translation();
        snapshot.pos = Vec2::new(translation.x, translation.y);
        snapshot.angle = body.rotation().angle();
        let linvel = body.linvel();
        snapshot.linvel = Vec2::new(linvel.x, linvel.y);
        snapshot.angvel = body.angvel();
    }
}

fn despawn_cleanup(
    mut world: ResMut<RapierWorld2d>,
    mut removed_bodies: RemovedComponents<RapierBody2d>,
    mut removed_revolute: RemovedComponents<RapierRevolute>,
    mut removed_fixed: RemovedComponents<RapierFixed>,
    mut removed_prismatic: RemovedComponents<RapierPrismatic>,
    mut removed_rope: RemovedComponents<RapierRope>,
    mut removed_spring: RemovedComponents<RapierSpring>,
) {
    for entity in removed_bodies.read() {
        if let Some(handle) = world.entity_to_body.remove(&entity) {
            let RapierWorld2d {
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
    for entity in removed_revolute
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

pub fn init_world(world: &mut bevy_ecs::prelude::World, gravity: Vec2, length_unit: f32) {
    world.insert_resource(RapierWorld2d::new(gravity, length_unit));
}

pub fn register_rapier2d_systems(sim: &mut Sim) {
    sim.add_chained_systems(
        (
            spawn_bodies,
            spawn_joints,
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
            .insert_resource(RapierWorld2d::new(Vec2::new(0.0, -980.0), 20.0));
        register_rapier2d_systems(&mut sim);
        sim
    }

    #[test]
    fn raycast_hits_fixed_floor_with_surface_id() {
        let mut sim = sim_with_world();
        sim.world.spawn(RapierBody2d {
            kind: BodyKind::Fixed,
            spawn_pos: Vec2::new(0.0, -170.0),
            half_extents: Vec2::new(1200.0, 100.0),
            ..Default::default()
        });
        sim.tick();
        sim.tick();
        let world = sim.world.resource::<RapierWorld2d>();
        let hit = world
            .cast_ray(Vec2::new(0.0, 100.0), Vec2::new(0.0, -1.0), 500.0, true)
            .expect("ray hits the floor");
        assert!((hit.distance - 170.0).abs() < 1.0);
        assert!((hit.normal - Vec2::new(0.0, 1.0)).length() < 1e-3);
        assert!((hit.point.y - -70.0).abs() < 1.0);
        assert!(
            world
                .cast_ray(Vec2::new(0.0, 100.0), Vec2::new(0.0, 1.0), 500.0, true)
                .is_none()
        );
        assert!(
            world
                .cast_ray(Vec2::new(0.0, 100.0), Vec2::ZERO, 500.0, true)
                .is_none()
        );
        assert!(
            world
                .cast_ray(Vec2::new(0.0, 100.0), Vec2::new(0.0, -1.0), 0.0, true)
                .is_none()
        );
    }

    #[test]
    fn pause_means_identical_snapshots() {
        let mut sim = sim_with_world();
        let entity = sim
            .world
            .spawn((
                RapierBody2d {
                    spawn_pos: Vec2::new(0.0, 100.0),
                    half_extents: Vec2::new(15.0, 26.0),
                    ..Default::default()
                },
                BodySnapshot2d::default(),
            ))
            .id();
        for _ in 0..10 {
            sim.tick();
        }
        let before = *sim.world.get::<BodySnapshot2d>(entity).unwrap();
        let bodies_before = sim.world.resource::<RapierWorld2d>().body_count();
        let before_pos = before.pos;
        for _ in 0..60 {
            let after = *sim.world.get::<BodySnapshot2d>(entity).unwrap();
            assert_eq!(after.pos, before_pos);
            assert_eq!(after.angle, before.angle);
            assert_eq!(after.linvel, before.linvel);
            assert_eq!(after.angvel, before.angvel);
        }
        assert_eq!(
            sim.world.resource::<RapierWorld2d>().body_count(),
            bodies_before
        );
    }

    #[test]
    fn warrior_puppet_spawns_eight_bodies_seven_joints() {
        let mut sim = sim_with_world();
        let limbs = [
            (Vec2::new(0.0, 0.0), Vec2::new(15.0, 26.0)),
            (Vec2::new(0.0, 28.0), Vec2::new(12.0, 12.0)),
            (Vec2::new(-19.0, 15.0), Vec2::new(6.5, 14.0)),
            (Vec2::new(19.0, 15.0), Vec2::new(6.5, 14.0)),
            (Vec2::new(-30.0, -8.0), Vec2::new(6.5, 19.0)),
            (Vec2::new(30.0, -8.0), Vec2::new(6.5, 19.0)),
            (Vec2::new(-8.0, -41.0), Vec2::new(6.5, 17.5)),
            (Vec2::new(8.0, -41.0), Vec2::new(6.5, 17.5)),
        ];
        let mut entities = Vec::new();
        for (pos, half) in limbs {
            entities.push(
                sim.world
                    .spawn((
                        RapierBody2d {
                            spawn_pos: pos,
                            half_extents: half,
                            linear_damping: 1.2,
                            angular_damping: 2.5,
                            mass: 1.0,
                            ..Default::default()
                        },
                        BodySnapshot2d::default(),
                    ))
                    .id(),
            );
        }
        let torso = entities[0];
        let joints = [
            (1, Vec2::new(0.0, 24.0), Vec2::new(0.0, -9.0)),
            (2, Vec2::new(-14.0, 12.0), Vec2::new(0.0, 13.0)),
            (3, Vec2::new(14.0, 12.0), Vec2::new(0.0, 13.0)),
            (4, Vec2::new(0.0, -13.0), Vec2::new(0.0, 15.0)),
            (5, Vec2::new(0.0, -13.0), Vec2::new(0.0, 15.0)),
            (6, Vec2::new(-8.0, -21.0), Vec2::new(0.0, 15.0)),
            (7, Vec2::new(8.0, -21.0), Vec2::new(0.0, 15.0)),
        ];
        for (child, anchor_a, anchor_b) in joints {
            sim.world.spawn(RapierRevolute {
                parent: torso,
                child: entities[child],
                anchor_a,
                anchor_b,
                rest_angle: 0.0,
                stiffness: 10.0,
                damping: 6.0,
                limits: [-0.78, 0.78],
            });
        }
        sim.tick();
        sim.tick();
        let world = sim.world.resource::<RapierWorld2d>();
        assert_eq!(world.body_count(), 8);
        assert_eq!(world.joint_count(), 7);
    }

    #[test]
    fn knockback_impulse_moves_torso_and_settles() {
        let mut sim = sim_with_world();
        let entity = sim
            .world
            .spawn((
                RapierBody2d {
                    spawn_pos: Vec2::ZERO,
                    half_extents: Vec2::new(15.0, 26.0),
                    gravity_scale: 0.0,
                    linear_damping: 1.2,
                    ..Default::default()
                },
                ImpulseRequest::default(),
                BodySnapshot2d::default(),
            ))
            .id();
        sim.tick();
        sim.world.get_mut::<ImpulseRequest>(entity).unwrap().linear = Vec2::new(5000.0, 0.0);
        sim.tick();
        let pushed = *sim.world.get::<BodySnapshot2d>(entity).unwrap();
        assert!(pushed.pos.x > 0.0);
        assert!(pushed.linvel.x > 0.0);
        let request = sim.world.get::<ImpulseRequest>(entity).unwrap();
        assert_eq!(request.linear, Vec2::ZERO);
        assert_eq!(request.torque, 0.0);
        for _ in 0..240 {
            sim.tick();
        }
        let settled = *sim.world.get::<BodySnapshot2d>(entity).unwrap();
        assert!(settled.linvel.x.abs() < pushed.linvel.x.abs());
    }

    #[test]
    fn death_joint_removal_flops_beyond_hip_limit() {
        let mut sim = sim_with_world();
        let torso = sim
            .world
            .spawn((
                RapierBody2d {
                    spawn_pos: Vec2::new(0.0, 100.0),
                    half_extents: Vec2::new(15.0, 26.0),
                    ..Default::default()
                },
                BodySnapshot2d::default(),
            ))
            .id();
        let leg = sim
            .world
            .spawn((
                RapierBody2d {
                    spawn_pos: Vec2::new(-8.0, 30.0),
                    half_extents: Vec2::new(6.5, 17.5),
                    memberships: 0x0002,
                    filter: 0x0001,
                    ..Default::default()
                },
                BodySnapshot2d::default(),
            ))
            .id();
        let joint_entity = sim
            .world
            .spawn(RapierRevolute {
                parent: torso,
                child: leg,
                anchor_a: Vec2::new(-8.0, -21.0),
                anchor_b: Vec2::new(0.0, 15.0),
                rest_angle: 0.0,
                stiffness: 15.0,
                damping: 10.0,
                limits: [-0.035, 0.035],
            })
            .id();
        sim.world.spawn(RapierBody2d {
            kind: BodyKind::Fixed,
            spawn_pos: Vec2::new(0.0, -170.0),
            half_extents: Vec2::new(1200.0, 100.0),
            ..Default::default()
        });
        for _ in 0..120 {
            sim.tick();
        }
        let held = *sim.world.get::<BodySnapshot2d>(leg).unwrap();
        let torso_snap = *sim.world.get::<BodySnapshot2d>(torso).unwrap();
        let torso_held = torso_snap;
        sim.world
            .entity_mut(joint_entity)
            .remove::<RapierRevolute>();
        sim.world.entity_mut(torso).insert(ImpulseRequest {
            linear: Vec2::new(9000.0, 4000.0),
            torque: 400.0,
        });
        sim.world.entity_mut(leg).insert(ImpulseRequest {
            linear: Vec2::new(-6000.0, 1000.0),
            torque: -300.0,
        });
        for _ in 0..240 {
            sim.tick();
        }
        assert_eq!(sim.world.resource::<RapierWorld2d>().joint_count(), 0);
        let flopped = *sim.world.get::<BodySnapshot2d>(leg).unwrap();
        let torso_after = *sim.world.get::<BodySnapshot2d>(torso).unwrap();
        let torso_fell = (torso_after.pos.y - torso_held.pos.y).abs();
        let leg_moved = (flopped.pos - held.pos).length();
        assert!(torso_fell > 1.0);
        assert!(leg_moved > 1.0);
    }

    #[test]
    fn puppet_motor_holds_stand_height() {
        let mut sim = sim_with_world();
        let entity = sim
            .world
            .spawn((
                RapierBody2d {
                    spawn_pos: Vec2::new(0.0, -102.0),
                    half_extents: Vec2::new(15.0, 26.0),
                    linear_damping: 1.2,
                    angular_damping: 2.5,
                    ..Default::default()
                },
                ImpulseRequest::default(),
                BodySnapshot2d::default(),
            ))
            .id();
        sim.world.spawn(RapierBody2d {
            kind: BodyKind::Fixed,
            spawn_pos: Vec2::new(0.0, -260.0),
            half_extents: Vec2::new(1200.0, 100.0),
            ..Default::default()
        });
        for _ in 0..240 {
            let snapshot = *sim.world.get::<BodySnapshot2d>(entity).unwrap();
            let y_error = -102.0 - snapshot.pos.y;
            let downward = snapshot.linvel.y.min(0.0);
            let impulse = Vec2::Y * (y_error * 5.0 - downward * 1.8) * (1.0 / 60.0);
            sim.world.get_mut::<ImpulseRequest>(entity).unwrap().linear += impulse;
            let torque = -snapshot.angle * 10.0 - snapshot.angvel * 2.4;
            sim.world.get_mut::<ImpulseRequest>(entity).unwrap().torque += torque * (1.0 / 60.0);
            sim.tick();
        }
        let snapshot = *sim.world.get::<BodySnapshot2d>(entity).unwrap();
        assert!((snapshot.pos.y - -102.0).abs() < 60.0);
        assert!(snapshot.linvel.length() < 600.0);
        assert!(snapshot.angvel.abs() < 8.0);
    }
}
