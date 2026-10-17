//! Two-bone analytic IK for the shoulder/elbow/knee class of limbs.
//!
//! Two segments of fixed length `L1` (root to middle) and `L2` (middle to
//! end) with an analytic triangle solution: the root is fixed, the end is
//! placed on the target when reachable, and the middle joint lands on the
//! pole side. When the target is farther than `L1 + L2` the limb straightens
//! toward it; when it is closer than `|L1 - L2|` the target is pushed out to
//! the nearer reachable sphere.

use std::collections::HashMap;

use glam::Vec3;
use repame_view3d::{JointPose, Skeleton};

use crate::chain::{self, JointLimits};
use crate::error::IkError;
use crate::iterative::IkResult;
use crate::pose;

/// Solve a two-bone limb `root -> middle -> end` so the end reaches
/// `target`, with the middle joint bent toward `pole` (a world-space point
/// the elbow/knee leans to). Rotations are written into `locals` in place.
///
/// `pole` must not lie exactly on the root-to-target line; a degenerate
/// pole falls back to the bind elbow plane.
pub fn solve_two_bone(
    skeleton: &Skeleton,
    locals: &mut HashMap<usize, JointPose>,
    root: usize,
    middle: usize,
    end: usize,
    target: Vec3,
    pole: Vec3,
) -> Result<IkResult, IkError> {
    if !chain::is_descendant(skeleton, root, middle) || !chain::is_descendant(skeleton, middle, end)
    {
        return Err(IkError::BrokenChain {
            parent: root,
            child: middle,
        });
    }

    let root_world = pose::rest_matrix(skeleton, root).ok_or(IkError::UnknownJoint(root))?;
    let middle_world = pose::rest_matrix(skeleton, middle).ok_or(IkError::UnknownJoint(middle))?;
    let end_world = pose::rest_matrix(skeleton, end).ok_or(IkError::UnknownJoint(end))?;
    let l1 = (middle_world.transform_point3(Vec3::ZERO)
        - root_world.transform_point3(Vec3::ZERO))
    .length();
    let l2 = (end_world.transform_point3(Vec3::ZERO)
        - middle_world.transform_point3(Vec3::ZERO))
    .length();

    let root_world = pose::world_matrix(skeleton, locals, root).ok_or(IkError::UnknownJoint(root))?;
    let root_pos = root_world.transform_point3(Vec3::ZERO);
    let to_target = target - root_pos;
    let distance = to_target.length();
    if distance < 1e-6 || l1 + l2 < 1e-6 {
        return Ok(IkResult {
            error: distance,
            iterations: 0,
            reached: false,
        });
    }
    let u = to_target / distance;

    let middle_pos;
    let end_pos;
    if distance >= l1 + l2 {
        // Unreachable: straighten toward the target.
        middle_pos = root_pos + u * l1;
        end_pos = root_pos + u * (l1 + l2);
    } else {
        let clamped = distance.max((l1 - l2).abs());
        let reachable = root_pos + u * clamped;
        let a = (clamped * clamped + l1 * l1 - l2 * l2) / (2.0 * clamped);
        let h = (l1 * l1 - a * a).max(0.0).sqrt();
        let base = root_pos + u * a;
        let mut plane = pole - root_pos;
        plane -= u * plane.dot(u);
        if plane.length_squared() <= 1e-12 {
            plane = bind_elbow_normal(skeleton, root, middle, end, u);
        }
        let plane = plane.normalize_or_zero();
        middle_pos = base + plane * h;
        end_pos = reachable;
    }

    aim_joint(skeleton, locals, root, middle, middle_pos)?;
    aim_joint(skeleton, locals, middle, end, end_pos)?;

    let error = (pose::world_matrix(skeleton, locals, end)
        .ok_or(IkError::UnknownJoint(end))?
        .transform_point3(Vec3::ZERO)
        - target)
        .length();
    Ok(IkResult {
        error,
        iterations: 1,
        reached: error <= 1e-3,
    })
}

/// Rotate `joint` about its origin so the direction to `toward` matches the
/// direction to `destination`.
fn aim_joint(
    skeleton: &Skeleton,
    locals: &mut HashMap<usize, JointPose>,
    joint: usize,
    toward: usize,
    destination: Vec3,
) -> Result<(), IkError> {
    let world = pose::world_matrix(skeleton, locals, joint).ok_or(IkError::UnknownJoint(joint))?;
    let origin = world.transform_point3(Vec3::ZERO);
    let toward = pose::world_matrix(skeleton, locals, toward)
        .ok_or(IkError::UnknownJoint(toward))?
        .transform_point3(Vec3::ZERO);
    let from = toward - origin;
    let to = destination - origin;
    if from.length_squared() <= 1e-12 || to.length_squared() <= 1e-12 {
        return Ok(());
    }
    let delta = pose::rotation_between(from, to);
    chain::rotate_joint(skeleton, locals, joint, Vec3::Y, delta, JointLimits::FREE)
}

/// Bind-pose normal of the limb plane, used when the pole is degenerate.
fn bind_elbow_normal(
    skeleton: &Skeleton,
    root: usize,
    middle: usize,
    end: usize,
    _u: Vec3,
) -> Vec3 {
    let outer = pose::rest_matrix(skeleton, root).map(|m| m.transform_point3(Vec3::ZERO));
    let inner = pose::rest_matrix(skeleton, middle).map(|m| m.transform_point3(Vec3::ZERO));
    let tip = pose::rest_matrix(skeleton, end).map(|m| m.transform_point3(Vec3::ZERO));
    match (outer, inner, tip) {
        (Some(a), Some(b), Some(c)) => {
            let normal = (b - a).cross(c - b);
            if normal.length_squared() > 1e-12 {
                normal.normalize()
            } else {
                Vec3::X
            }
        }
        _ => Vec3::X,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Mat4;

    fn limb_skeleton() -> Skeleton {
        Skeleton {
            parents: HashMap::from([(0, None), (1, Some(0)), (2, Some(1))]),
            bind_local: HashMap::from([
                (0, Mat4::IDENTITY),
                (1, Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0))),
                (2, Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0))),
            ]),
        }
    }

    #[test]
    fn two_bone_is_exact_for_an_analytic_case() {
        let skeleton = limb_skeleton();
        let mut locals = HashMap::new();
        let target = Vec3::new(1.0, 1.0, 0.0);
        let pole = Vec3::new(1.0, 0.0, 0.0);
        let result = solve_two_bone(&skeleton, &mut locals, 0, 1, 2, target, pole).expect("solves");
        assert!(result.reached, "error {}", result.error);
        let end = pose::world_matrix(&skeleton, &locals, 2)
            .unwrap()
            .transform_point3(Vec3::ZERO);
        let middle = pose::world_matrix(&skeleton, &locals, 1)
            .unwrap()
            .transform_point3(Vec3::ZERO);
        assert!(
            (end - target).length() < 1e-4,
            "end {end:?} should land on the target"
        );
        assert!(
            (middle - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-4,
            "elbow {middle:?} should sit on the pole side"
        );
    }

    #[test]
    fn two_bone_straightens_when_unreachable() {
        let skeleton = limb_skeleton();
        let mut locals = HashMap::new();
        let target = Vec3::new(0.0, 10.0, 0.0);
        let result =
            solve_two_bone(&skeleton, &mut locals, 0, 1, 2, target, Vec3::new(1.0, 0.0, 0.0))
                .expect("solves");
        assert!(!result.reached);
        let end = pose::world_matrix(&skeleton, &locals, 2)
            .unwrap()
            .transform_point3(Vec3::ZERO);
        assert!((end - Vec3::new(0.0, 2.0, 0.0)).length() < 1e-4, "end {end:?}");
    }
}
