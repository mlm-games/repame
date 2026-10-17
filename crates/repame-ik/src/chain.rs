//! Joint chains: an ancestor-to-descendant path through a [`Skeleton`] with
//! bind-derived segment lengths, a virtual tip, and per-joint limits.

use std::collections::HashMap;

use glam::{Mat4, Quat, Vec3};
use repame_view3d::{JointPose, Skeleton};

use crate::error::IkError;
use crate::pose;

/// Per-joint rotation limits, measured from the joint's bind rotation.
///
/// `max_swing` bounds the cone half-angle of the rotation axis; `max_twist`
/// bounds the rotation about the bone axis. Either may be
/// [`f32::INFINITY`] for no limit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointLimits {
    /// Cone half-angle in radians. [`f32::INFINITY`] disables the swing limit.
    pub max_swing: f32,
    /// Twist in radians about the bone axis. [`f32::INFINITY`] disables it.
    pub max_twist: f32,
}

impl JointLimits {
    /// No limit on either axis.
    pub const FREE: Self = Self {
        max_swing: f32::INFINITY,
        max_twist: f32::INFINITY,
    };

    /// A swing cone with unlimited twist.
    pub const fn cone(max_swing: f32) -> Self {
        Self {
            max_swing,
            max_twist: f32::INFINITY,
        }
    }

    /// True when neither axis is limited.
    pub fn is_free(&self) -> bool {
        self.max_swing.is_infinite() && self.max_twist.is_infinite()
    }
}

impl Default for JointLimits {
    fn default() -> Self {
        Self::FREE
    }
}

/// A path of joints `joints[0] ..= joints[n]` where each joint is an
/// ancestor of the next, plus a virtual tip `end_offset` past the last
/// joint. The effector is the last joint's origin when `end_offset` is
/// zero (the default), or the tip under the last joint's rotation when it
/// is not.
///
/// Segment lengths come from the bind pose, so the chain is rigid under
/// rotation: a solver moves the joints' rotations, never their offsets.
#[derive(Clone, Debug)]
pub struct JointChain {
    joints: Vec<usize>,
    lengths: Vec<f32>,
    axes: Vec<Vec3>,
    end_offset: Vec3,
}

impl JointChain {
    /// Build a chain from an ancestor path. Fails on an empty path, an
    /// unknown joint, or a pair that is not ancestor and descendant.
    pub fn new(skeleton: &Skeleton, joints: Vec<usize>) -> Result<Self, IkError> {
        if joints.is_empty() {
            return Err(IkError::EmptyChain);
        }
        for joint in &joints {
            if !skeleton.parents.contains_key(joint) {
                return Err(IkError::UnknownJoint(*joint));
            }
        }
        for pair in joints.windows(2) {
            if !is_descendant(skeleton, pair[0], pair[1]) {
                return Err(IkError::BrokenChain {
                    parent: pair[0],
                    child: pair[1],
                });
            }
        }
        let mut lengths = Vec::with_capacity(joints.len());
        let mut axes = Vec::with_capacity(joints.len());
        for i in 0..joints.len() - 1 {
            let a = pose::rest_matrix(skeleton, joints[i]).unwrap_or(Mat4::IDENTITY);
            let b = pose::rest_matrix(skeleton, joints[i + 1]).unwrap_or(Mat4::IDENTITY);
            let offset = a.inverse().transform_point3(b.transform_point3(Vec3::ZERO));
            lengths.push(offset.length());
            axes.push(offset.normalize_or_zero());
        }
        lengths.push(0.0);
        axes.push(Vec3::ZERO);
        Ok(Self {
            joints,
            lengths,
            axes,
            end_offset: Vec3::ZERO,
        })
    }

    /// Extend the chain past the last joint by a local-space offset, so the
    /// effector becomes that tip instead of the last joint's origin.
    pub fn with_end_offset(mut self, offset: Vec3) -> Self {
        let last = self.joints.len() - 1;
        self.end_offset = offset;
        self.lengths[last] = offset.length();
        self.axes[last] = offset.normalize_or_zero();
        self
    }

    /// Joint indices, root first.
    pub fn joints(&self) -> &[usize] {
        &self.joints
    }

    /// Segment length per joint (the last entry is the tip length, 0 without
    /// an end offset).
    pub fn lengths(&self) -> &[f32] {
        &self.lengths
    }

    /// Unit bone axis per joint in that joint's bind-local space.
    pub fn axes(&self) -> &[Vec3] {
        &self.axes
    }

    /// Local offset from the last joint to the virtual tip.
    pub fn end_offset(&self) -> Vec3 {
        self.end_offset
    }

    /// Number of joints (points minus one).
    pub fn segment_count(&self) -> usize {
        self.joints.len()
    }

    /// Sum of every segment length.
    pub fn total_length(&self) -> f32 {
        self.lengths.iter().sum()
    }

    /// Current world points: every joint origin followed by the effector.
    pub fn points(
        &self,
        skeleton: &Skeleton,
        locals: &HashMap<usize, JointPose>,
    ) -> Result<Vec<Vec3>, IkError> {
        let mut points = Vec::with_capacity(self.joints.len() + 1);
        for joint in &self.joints {
            let world =
                pose::world_matrix(skeleton, locals, *joint).ok_or(IkError::UnknownJoint(*joint))?;
            points.push(world.transform_point3(Vec3::ZERO));
        }
        points.push(self.effector(skeleton, locals)?);
        Ok(points)
    }

    /// Current world position of the effector.
    pub fn effector(
        &self,
        skeleton: &Skeleton,
        locals: &HashMap<usize, JointPose>,
    ) -> Result<Vec3, IkError> {
        let last = *self.joints.last().ok_or(IkError::EmptyChain)?;
        let world = pose::world_matrix(skeleton, locals, last).ok_or(IkError::UnknownJoint(last))?;
        Ok(world.transform_point3(self.end_offset))
    }

    /// Bind-pose world points, the rigid reference the chain keeps.
    pub fn rest_points(&self, skeleton: &Skeleton) -> Vec<Vec3> {
        let mut points: Vec<Vec3> = self
            .joints
            .iter()
            .map(|joint| {
                pose::rest_matrix(skeleton, *joint)
                    .map(|m| m.transform_point3(Vec3::ZERO))
                    .unwrap_or(Vec3::ZERO)
            })
            .collect();
        let tip = self
            .joints
            .last()
            .and_then(|joint| pose::rest_matrix(skeleton, *joint))
            .map(|m| m.transform_point3(self.end_offset))
            .unwrap_or(Vec3::ZERO);
        points.push(tip);
        points
    }
}

/// True when `node` is `ancestor` or a descendant of it.
pub(crate) fn is_descendant(skeleton: &Skeleton, ancestor: usize, node: usize) -> bool {
    let limit = skeleton.node_count() + 1;
    let mut current = Some(node);
    let mut steps = 0;
    while let Some(n) = current {
        if n == ancestor {
            return true;
        }
        current = skeleton.parents.get(&n).copied().flatten();
        steps += 1;
        if steps > limit {
            return false;
        }
    }
    false
}

/// Limit for chain position `index`, free when the slice is short.
pub(crate) fn limit_at(limits: &[JointLimits], index: usize) -> JointLimits {
    limits.get(index).copied().unwrap_or_default()
}

/// Rotate `joint` by `delta_world` (about its own origin), then clamp the
/// new local rotation to `limits` measured from the bind rotation.
pub(crate) fn rotate_joint(
    skeleton: &Skeleton,
    locals: &mut HashMap<usize, JointPose>,
    joint: usize,
    axis_local: Vec3,
    delta_world: Quat,
    limits: JointLimits,
) -> Result<(), IkError> {
    let world = pose::world_matrix(skeleton, locals, joint).ok_or(IkError::UnknownJoint(joint))?;
    let parent_world = match skeleton.parents.get(&joint).copied().flatten() {
        Some(parent) => {
            pose::world_matrix(skeleton, locals, parent).ok_or(IkError::UnknownJoint(parent))?
        }
        None => Mat4::IDENTITY,
    };
    let new_world = delta_world * pose::world_rotation(&world);
    let parent_rotation = pose::world_rotation(&parent_world);
    let mut local = parent_rotation.inverse() * new_world;
    if !limits.is_free() {
        let bind_local = pose::bind_rotation(skeleton, joint);
        let delta = bind_local.inverse() * local;
        local = bind_local * clamp_swing_twist(delta, axis_local, limits);
    }
    pose::set_local_rotation(skeleton, locals, joint, local);
    Ok(())
}

/// Clamp a joint-local rotation delta to a swing cone and a twist about
/// `axis` (decomposed at `axis`, which must be the bone axis).
pub(crate) fn clamp_swing_twist(delta: Quat, axis: Vec3, limits: JointLimits) -> Quat {
    if limits.is_free() {
        return delta;
    }
    let axis = axis.normalize_or_zero();
    if axis == Vec3::ZERO {
        return delta;
    }
    let vector = Vec3::new(delta.x, delta.y, delta.z);
    let projected = axis * vector.dot(axis);
    let twist = Quat::from_xyzw(projected.x, projected.y, projected.z, delta.w);
    let twist = if twist.length_squared() > 1e-12 {
        twist.normalize()
    } else {
        Quat::IDENTITY
    };
    let mut swing = delta * twist.conjugate();
    if swing.w < 0.0 {
        swing = Quat::from_xyzw(-swing.x, -swing.y, -swing.z, -swing.w);
    }
    let swing_angle = 2.0 * swing.w.clamp(-1.0, 1.0).acos();
    let swing_axis = Vec3::new(swing.x, swing.y, swing.z);
    let swing = if swing_angle > limits.max_swing && swing_axis.length_squared() > 1e-12 {
        Quat::from_axis_angle(swing_axis.normalize(), limits.max_swing)
    } else {
        swing
    };
    let twist_angle = 2.0 * projected.dot(axis).atan2(delta.w);
    let twist = if twist_angle.abs() > limits.max_twist {
        Quat::from_axis_angle(axis, limits.max_twist * twist_angle.signum())
    } else {
        twist
    };
    (swing * twist).normalize()
}

/// Apply a set of desired world points (`points[0]` the fixed root,
/// `points[n]` the effector) to `chain` by writing each joint's rotation so
/// its outgoing segment points along the desired direction. Runs in place
/// on `locals`, in root-to-tip order so every step sees the updated
/// ancestors.
pub(crate) fn apply_point_targets(
    skeleton: &Skeleton,
    locals: &mut HashMap<usize, JointPose>,
    chain: &JointChain,
    points: &[Vec3],
    limits: &[JointLimits],
) -> Result<(), IkError> {
    let count = chain.joints.len();
    if points.len() < count + 1 {
        return Ok(());
    }
    for i in 0..count {
        let joint = chain.joints[i];
        let world = pose::world_matrix(skeleton, locals, joint).ok_or(IkError::UnknownJoint(joint))?;
        let origin = world.transform_point3(Vec3::ZERO);
        let outgoing = if i + 1 < count {
            let child = chain.joints[i + 1];
            let child_world =
                pose::world_matrix(skeleton, locals, child).ok_or(IkError::UnknownJoint(child))?;
            child_world.transform_point3(Vec3::ZERO)
        } else {
            world.transform_point3(chain.end_offset)
        };
        let from = outgoing - origin;
        let to = points[i + 1] - origin;
        if from.length_squared() <= 1e-12 || to.length_squared() <= 1e-12 {
            continue;
        }
        let delta = pose::rotation_between(from, to);
        rotate_joint(
            skeleton,
            locals,
            joint,
            chain.axes[i],
            delta,
            limit_at(limits, i),
        )?;
    }
    Ok(())
}
