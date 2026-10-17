//! Pose queries and rotations over [`Skeleton`], the shared basis for every
//! solver and modifier in this crate.
//!
//! All functions read a [`Skeleton`] (parent links plus bind locals) and the
//! animated local poses the game keeps in a
//! `HashMap<usize, JointPose>`. Nothing here owns pose state: the map is
//! the pose, exactly as [`Skeleton::world_matrices`] expects.

use std::collections::HashMap;

use glam::{Mat4, Quat, Vec3};
use repame_view3d::{JointPose, Skeleton};

/// Bind local matrix for `node` (identity when the skeleton omits it).
pub fn bind_matrix(skeleton: &Skeleton, node: usize) -> Mat4 {
    skeleton
        .bind_local
        .get(&node)
        .copied()
        .unwrap_or(Mat4::IDENTITY)
}

/// Bind local TRS for `node` as a [`JointPose`].
pub fn bind_pose(skeleton: &Skeleton, node: usize) -> JointPose {
    let (scale, rotation, translation) = bind_matrix(skeleton, node).to_scale_rotation_translation();
    JointPose {
        translation: translation.to_array(),
        rotation: rotation.to_array(),
        scale: scale.to_array(),
    }
}

/// Bind local rotation for `node`, normalized.
pub fn bind_rotation(skeleton: &Skeleton, node: usize) -> Quat {
    world_rotation(&bind_matrix(skeleton, node))
}

/// The local matrix used for `node`: the animated pose when `locals` has it,
/// otherwise the bind local.
pub fn local_matrix(skeleton: &Skeleton, locals: &HashMap<usize, JointPose>, node: usize) -> Mat4 {
    locals
        .get(&node)
        .map(|pose| pose.matrix())
        .unwrap_or_else(|| bind_matrix(skeleton, node))
}

/// World matrix for `node`, composing local poses down from the scene root.
/// Returns `None` when `node` is not in the skeleton or the parent links
/// contain a cycle.
pub fn world_matrix(
    skeleton: &Skeleton,
    locals: &HashMap<usize, JointPose>,
    node: usize,
) -> Option<Mat4> {
    world_from(skeleton, |n| local_matrix(skeleton, locals, n), node)
}

/// Bind-pose world matrix for `node`, ignoring the animated poses.
pub fn rest_matrix(skeleton: &Skeleton, node: usize) -> Option<Mat4> {
    world_from(skeleton, |n| bind_matrix(skeleton, n), node)
}

fn world_from(
    skeleton: &Skeleton,
    local: impl Fn(usize) -> Mat4,
    node: usize,
) -> Option<Mat4> {
    if !skeleton.parents.contains_key(&node) {
        return None;
    }
    let limit = skeleton.node_count() + 1;
    let mut locals = Vec::new();
    let mut current = Some(node);
    while let Some(n) = current {
        locals.push(local(n));
        current = skeleton.parents.get(&n).copied().flatten();
        if locals.len() > limit {
            return None;
        }
    }
    let mut world = Mat4::IDENTITY;
    for m in locals.iter().rev() {
        world *= *m;
    }
    Some(world)
}

/// World rotation of a world matrix, normalized.
pub fn world_rotation(m: &Mat4) -> Quat {
    Quat::from_mat4(m).normalize()
}

/// Write the local rotation of `node`, seeding the entry from the bind TRS
/// when the node is not animated yet.
pub fn set_local_rotation(
    skeleton: &Skeleton,
    locals: &mut HashMap<usize, JointPose>,
    node: usize,
    rotation: Quat,
) {
    let pose = locals
        .entry(node)
        .or_insert_with(|| bind_pose(skeleton, node));
    pose.rotation = rotation.normalize().to_array();
}

/// Rotation taking unit vector `from` onto unit vector `to`, robust for the
/// parallel and antiparallel cases.
pub fn rotation_between(from: Vec3, to: Vec3) -> Quat {
    let from = from.normalize_or_zero();
    let to = to.normalize_or_zero();
    let dot = from.dot(to).clamp(-1.0, 1.0);
    if dot > 1.0 - 1e-6 {
        return Quat::IDENTITY;
    }
    if dot < -1.0 + 1e-6 {
        return Quat::from_axis_angle(from.any_orthonormal_vector(), std::f32::consts::PI);
    }
    Quat::from_axis_angle(from.cross(to).normalize(), dot.acos())
}

/// Unit direction from `a` to `b`, falling back to `fallback` when the two
/// points coincide (zero when the fallback is zero too).
pub fn direction(a: Vec3, b: Vec3, fallback: Vec3) -> Vec3 {
    let d = b - a;
    if d.length_squared() > 1e-12 {
        d.normalize()
    } else {
        fallback.normalize_or_zero()
    }
}
