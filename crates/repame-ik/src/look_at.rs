//! Look-at modifier: aim one joint's forward axis at a world target.

use std::collections::HashMap;

use glam::{Quat, Vec3};
use repame_view3d::{JointPose, Skeleton};

use crate::chain::{self, JointLimits};
use crate::error::IkError;
use crate::pose;

/// Look-at settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LookAt {
    /// Unit forward axis in the joint's local space. Defaults to `+Y`, the
    /// value used by [`Aim`](crate::Aim) too.
    pub forward: Vec3,
    /// Maximum deviation in radians from the target before the aim is
    /// clamped. [`f32::INFINITY`] always reaches.
    pub max_angle: f32,
}

impl Default for LookAt {
    fn default() -> Self {
        Self {
            forward: Vec3::Y,
            max_angle: f32::INFINITY,
        }
    }
}

/// How far a modifier got.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModifierOutcome {
    /// Angle in radians between the forward axis and the target before
    /// clamping.
    pub angle: f32,
    /// True when `angle` was within the configured limit.
    pub reached: bool,
}

/// Rotate `joint` so its `forward` axis points at the world-space `target`,
/// clamped to `max_angle`. Writes the joint rotation into `locals` in place.
pub fn look_at(
    skeleton: &Skeleton,
    locals: &mut HashMap<usize, JointPose>,
    joint: usize,
    target: Vec3,
    config: &LookAt,
) -> Result<ModifierOutcome, IkError> {
    let world = pose::world_matrix(skeleton, locals, joint).ok_or(IkError::UnknownJoint(joint))?;
    let origin = world.transform_point3(Vec3::ZERO);
    let rotation = pose::world_rotation(&world);
    let forward = (rotation * config.forward).normalize_or_zero();
    if forward == Vec3::ZERO {
        return Ok(ModifierOutcome {
            angle: 0.0,
            reached: false,
        });
    }
    let to_target = pose::direction(origin, target, forward);
    if to_target == Vec3::ZERO {
        return Ok(ModifierOutcome {
            angle: 0.0,
            reached: false,
        });
    }
    let angle = forward.dot(to_target).clamp(-1.0, 1.0).acos();
    let delta = if angle > config.max_angle {
        let axis = forward.cross(to_target);
        if axis.length_squared() > 1e-12 {
            Quat::from_axis_angle(axis.normalize(), config.max_angle)
        } else {
            Quat::IDENTITY
        }
    } else {
        pose::rotation_between(forward, to_target)
    };
    chain::rotate_joint(skeleton, locals, joint, Vec3::Y, delta, JointLimits::FREE)?;
    Ok(ModifierOutcome {
        angle,
        reached: angle <= config.max_angle,
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
    fn look_at_points_the_forward_axis_at_the_target() {
        let skeleton = bone();
        let mut locals = HashMap::new();
        let target = Vec3::new(1.0, 0.0, 0.0);
        let outcome = look_at(&skeleton, &mut locals, 0, target, &LookAt::default()).expect("aims");
        assert!(outcome.reached);
        let forward = pose::world_rotation(&pose::world_matrix(&skeleton, &locals, 0).unwrap())
            * Vec3::Y;
        assert!(forward.normalize().dot(Vec3::X) > 0.999, "forward {forward:?}");
    }

    #[test]
    fn look_at_clamps_past_the_limit() {
        let skeleton = bone();
        let mut locals = HashMap::new();
        let config = LookAt {
            forward: Vec3::Y,
            max_angle: 0.2,
        };
        let outcome =
            look_at(&skeleton, &mut locals, 0, Vec3::new(1.0, 0.0, 0.0), &config).expect("aims");
        assert!(!outcome.reached);
        assert!((outcome.angle - std::f32::consts::FRAC_PI_2).abs() < 1e-4);
        let forward = pose::world_rotation(&pose::world_matrix(&skeleton, &locals, 0).unwrap())
            * Vec3::Y;
        let angle = forward.normalize().dot(Vec3::Y).clamp(-1.0, 1.0).acos();
        assert!((angle - 0.2).abs() < 1e-3, "clamped to {angle}");
    }
}
