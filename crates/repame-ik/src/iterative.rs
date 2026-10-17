//! Iterative solvers for an arbitrary joint chain: FABRIK and CCD.
//!
//! Both walk the chain in place, honouring [`JointLimits`]. When the target
//! is farther from the root than the chain is long the solvers straighten
//! the chain toward it (FABRIK lays every joint on the root-to-target line,
//! CCD aims each joint at the target in turn); the reported
//! [`IkResult::error`] is then the leftover gap and `reached` is false.

use std::collections::HashMap;

use glam::Vec3;
use repame_view3d::{JointPose, Skeleton};

use crate::chain::{self, JointChain, JointLimits};
use crate::error::IkError;
use crate::pose;

/// Outcome of an iterative solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IkResult {
    /// Distance between the effector and the target after solving.
    pub error: f32,
    /// Iterations actually run (0 when the chain was already straight).
    pub iterations: u32,
    /// True when [`error`](IkResult::error) fell within the tolerance.
    pub reached: bool,
}

/// FABRIK (forward and backward reaching IK) settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fabrik {
    /// Maximum forward-and-backward passes.
    pub iterations: u32,
    /// Effector distance in world units that counts as reached.
    pub tolerance: f32,
}

impl Default for Fabrik {
    fn default() -> Self {
        Self {
            iterations: 16,
            tolerance: 1e-4,
        }
    }
}

impl Fabrik {
    /// Solve `chain` toward `target`, writing joint rotations into `locals`
    /// in place. `limits` is aligned to the chain's joints; missing entries
    /// are unlimited.
    pub fn solve(
        &self,
        skeleton: &Skeleton,
        locals: &mut HashMap<usize, JointPose>,
        chain: &JointChain,
        target: Vec3,
        limits: &[JointLimits],
    ) -> Result<IkResult, IkError> {
        let mut points = chain.points(skeleton, locals)?;
        let count = chain.joints().len();
        let root = points[0];
        let total = chain.total_length();
        let mut iterations = 0u32;

        if (target - root).length() >= total {
            let fallback = pose::direction(root, points[count], Vec3::Y);
            let dir = pose::direction(root, target, fallback);
            let mut cursor = root;
            for i in 0..count {
                cursor += dir * chain.lengths()[i];
                points[i + 1] = cursor;
            }
        } else {
            for _ in 0..self.iterations {
                iterations += 1;
                points[count] = target;
                for i in (0..count).rev() {
                    let fallback = pose::direction(points[i + 1], root, Vec3::Y);
                    let dir = pose::direction(points[i + 1], points[i], fallback);
                    points[i] = points[i + 1] + dir * chain.lengths()[i];
                }
                points[0] = root;
                for i in 0..count {
                    let fallback = pose::direction(points[i], target, Vec3::Y);
                    let dir = pose::direction(points[i], points[i + 1], fallback);
                    points[i + 1] = points[i] + dir * chain.lengths()[i];
                }
                if (points[count] - target).length() <= self.tolerance {
                    break;
                }
            }
        }

        chain::apply_point_targets(skeleton, locals, chain, &points, limits)?;
        let error = (chain.effector(skeleton, locals)? - target).length();
        Ok(IkResult {
            error,
            iterations,
            reached: error <= self.tolerance,
        })
    }
}

/// CCD (cyclic coordinate descent) settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ccd {
    /// Maximum sweeps from the last joint to the root.
    pub iterations: u32,
    /// Effector distance in world units that counts as reached.
    pub tolerance: f32,
}

impl Default for Ccd {
    fn default() -> Self {
        Self {
            iterations: 24,
            tolerance: 1e-4,
        }
    }
}

impl Ccd {
    /// Solve `chain` toward `target`, writing joint rotations into `locals`
    /// in place. `limits` is aligned to the chain's joints.
    pub fn solve(
        &self,
        skeleton: &Skeleton,
        locals: &mut HashMap<usize, JointPose>,
        chain: &JointChain,
        target: Vec3,
        limits: &[JointLimits],
    ) -> Result<IkResult, IkError> {
        let count = chain.joints().len();
        let mut iterations = 0u32;
        let mut error = (chain.effector(skeleton, locals)? - target).length();
        while iterations < self.iterations && error > self.tolerance {
            iterations += 1;
            for i in (0..count).rev() {
                let effector = chain.effector(skeleton, locals)?;
                let joint = chain.joints()[i];
                let world =
                    pose::world_matrix(skeleton, locals, joint).ok_or(IkError::UnknownJoint(joint))?;
                let origin = world.transform_point3(Vec3::ZERO);
                let from = effector - origin;
                let to = target - origin;
                if from.length_squared() <= 1e-12 || to.length_squared() <= 1e-12 {
                    continue;
                }
                let delta = pose::rotation_between(from, to);
                chain::rotate_joint(
                    skeleton,
                    locals,
                    joint,
                    chain.axes()[i],
                    delta,
                    chain::limit_at(limits, i),
                )?;
            }
            error = (chain.effector(skeleton, locals)? - target).length();
        }
        Ok(IkResult {
            error,
            iterations,
            reached: error <= self.tolerance,
        })
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

    fn straight_limb() -> (Skeleton, JointChain) {
        let skeleton = limb_skeleton();
        let chain = JointChain::new(&skeleton, vec![0, 1, 2]).expect("valid chain");
        (skeleton, chain)
    }

    #[test]
    fn fabrik_reaches_a_known_target() {
        let (skeleton, chain) = straight_limb();
        let mut locals = HashMap::new();
        let result = Fabrik::default()
            .solve(
                &skeleton,
                &mut locals,
                &chain,
                Vec3::new(1.0, 1.0, 0.0),
                &[],
            )
            .expect("solves");
        assert!(result.reached, "error {}", result.error);
        assert!(result.error < 1e-3, "error {}", result.error);
    }

    #[test]
    fn ccd_reaches_a_known_target() {
        let (skeleton, chain) = straight_limb();
        let mut locals = HashMap::new();
        let result = Ccd::default()
            .solve(
                &skeleton,
                &mut locals,
                &chain,
                Vec3::new(1.0, 1.0, 0.0),
                &[],
            )
            .expect("solves");
        assert!(result.reached, "error {}", result.error);
        assert!(result.error < 1e-3, "error {}", result.error);
    }

    #[test]
    fn joint_limits_are_respected() {
        let (skeleton, chain) = straight_limb();
        let mut locals = HashMap::new();
        let limits = [JointLimits::cone(0.3), JointLimits::FREE, JointLimits::FREE];
        Fabrik::default()
            .solve(&skeleton, &mut locals, &chain, Vec3::new(10.0, 0.0, 0.0), &limits)
            .expect("solves");
        let points = chain.points(&skeleton, &locals).expect("points");
        let bone = (points[1] - points[0]).normalize();
        let angle = bone.dot(Vec3::Y).clamp(-1.0, 1.0).acos();
        assert!(angle <= 0.3 + 1e-4, "swing {angle} exceeds the cone");
    }

    #[test]
    fn unreachable_target_extends_toward_it() {
        let (skeleton, chain) = straight_limb();
        let mut locals = HashMap::new();
        let target = Vec3::new(0.0, 10.0, 0.0);
        let result = Fabrik::default()
            .solve(&skeleton, &mut locals, &chain, target, &[])
            .expect("solves");
        assert!(!result.reached);
        let effector = chain.effector(&skeleton, &locals).expect("effector");
        assert!(
            (effector - Vec3::new(0.0, 2.0, 0.0)).length() < 1e-3,
            "effector {effector:?} should reach the chain length toward the target"
        );
        let dir = (effector - Vec3::ZERO).normalize();
        assert!(dir.dot(Vec3::Y) > 0.999);
    }
}
