//! Aim modifier: rotate one joint about a fixed axis so its forward axis
//! tracks the target, the swing/twist split of a look-at.

use std::collections::HashMap;

use glam::{Quat, Vec3};
use repame_view3d::{JointPose, Skeleton};

use crate::chain::{self, JointLimits};
use crate::error::IkError;
use crate::look_at::ModifierOutcome;
use crate::pose;

/// Aim settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aim {
    /// Unit forward axis in the joint's local space. Defaults to `+Y`.
    pub forward: Vec3,
    /// Unit rotation axis in the joint's local space. The joint only turns
    /// about this axis, like a knee bending on one hinge. Defaults to `+X`.
    pub axis: Vec3,
    /// Maximum rotation in radians about `axis`. [`f32::INFINITY`] always
    /// reaches.
    pub max_angle: f32,
}

impl Default for Aim {
    fn default() -> Self {
        Self {
            forward: Vec3::Y,
            axis: Vec3::X,
            max_angle: f32::INFINITY,
        }
    }
}

/// Rotate `joint` about `axis` (in its local space) so the projection of
/// `forward` onto the plane perpendicular to `axis` points at the world
/// `target`, clamped to `max_angle`.
///
/// This is the twist-style aim: the joint cannot leave the plane normal to
/// `axis`, only spin within it. Writes the joint rotation into `locals` in
/// place and reports the signed angle applied.
pub fn aim(
    skeleton: &Skeleton,
    locals: &mut HashMap<usize, JointPose>,
    joint: usize,
    target: Vec3,
    config: &Aim,
) -> Result<ModifierOutcome, IkError> {
    let world = pose::world_matrix(skeleton, locals, joint).ok_or(IkError::UnknownJoint(joint))?;
    let origin = world.transform_point3(Vec3::ZERO);
    let rotation = pose::world_rotation(&world);
    let forward = (rotation * config.forward).normalize_or_zero();
    let axis = (rotation * config.axis).normalize_or_zero();
    if forward == Vec3::ZERO || axis == Vec3::ZERO {
        return Ok(ModifierOutcome {
            angle: 0.0,
            reached: false,
        });
    }
    let to_target = pose::direction(origin, target, forward);
    let flat_forward = forward - axis * forward.dot(axis);
    let flat_target = to_target - axis * to_target.dot(axis);
    if flat_forward.length_squared() <= 1e-12 || flat_target.length_squared() <= 1e-12 {
        return Ok(ModifierOutcome {
            angle: 0.0,
            reached: false,
        });
    }
    let flat_forward = flat_forward.normalize();
    let flat_target = flat_target.normalize();
    let sign = if flat_forward.cross(flat_target).dot(axis) < 0.0 {
        -1.0
    } else {
        1.0
    };
    let angle = flat_forward.dot(flat_target).clamp(-1.0, 1.0).acos() * sign;
    let applied = if angle.abs() > config.max_angle {
        config.max_angle * angle.signum()
    } else {
        angle
    };
    let delta = Quat::from_axis_angle(axis, applied);
    chain::rotate_joint(skeleton, locals, joint, config.axis, delta, JointLimits::FREE)?;
    Ok(ModifierOutcome {
        angle,
        reached: angle.abs() <= config.max_angle,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Mat4;

    fn bone() -> Skeleton {
        Skeleton {
            parents: HashMap::from([(0, None)]),
            bind_local: HashMap::from([(0, Mat4::IDENTITY)]),
        }
    }

    #[test]
    fn aim_keeps_the_rotation_on_its_axis() {
        let skeleton = bone();
        let mut locals = HashMap::new();
        // Forward +Y, hinge +X: turning about +X swings +Y toward +Z, so a
        // target in the YZ plane is reachable while its X offset is ignored.
        let target = Vec3::new(0.0, 1.0, 1.0);
        let outcome = aim(&skeleton, &mut locals, 0, target, &Aim::default()).expect("aims");
        assert!(outcome.reached);
        assert!(
            (outcome.angle - std::f32::consts::FRAC_PI_4).abs() < 1e-4,
            "angle {}",
            outcome.angle
        );
        let forward =
            pose::world_rotation(&pose::world_matrix(&skeleton, &locals, 0).unwrap()) * Vec3::Y;
        assert!(
            forward.normalize().dot(Vec3::new(0.0, 1.0, 1.0).normalize()) > 0.999,
            "forward {forward:?}"
        );
    }

    #[test]
    fn aim_leaves_an_axis_parallel_target_alone() {
        let skeleton = bone();
        let mut locals = HashMap::new();
        // Target along the hinge axis: the plane projection is empty.
        let outcome = aim(
            &skeleton,
            &mut locals,
            0,
            Vec3::new(5.0, 0.0, 0.0),
            &Aim::default(),
        )
        .expect("aims");
        assert!(!outcome.reached);
        assert!(locals.is_empty() || locals[&0].rotation == [0.0, 0.0, 0.0, 1.0]);
    }
}
