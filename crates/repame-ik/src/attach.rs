//! Bone attachment: pin a world transform or point to a joint, the
//! skeleton-space half of Godot's `BoneAttachment3D`.
//!
//! For rigid *meshes* the viewport already has
//! [`repame_view3d::attach_to_joint`], which bakes a [`MeshGroup`] into a
//! joint's space. This module covers the other direction: resolving a
//! marker, muzzle, or prop transform from a bone each frame, so the game can
//! place any object (or spawn a particle) at it.

use std::collections::HashMap;

use glam::{Mat4, Vec3};
use repame_view3d::{JointPose, Skeleton};

use crate::error::IkError;
use crate::pose;

/// A transform fixed to a joint: `world = world_matrix(bone) * offset`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoneAttachment {
    /// Joint node index the attachment rides.
    pub bone: usize,
    /// Attachment transform in the joint's local space (identity puts it at
    /// the joint's origin and orientation).
    pub offset: Mat4,
}

impl BoneAttachment {
    /// Attach at a joint with a local offset transform.
    pub fn new(bone: usize, offset: Mat4) -> Self {
        Self { bone, offset }
    }

    /// Attach at the joint's origin, no local rotation.
    pub fn at(bone: usize) -> Self {
        Self::new(bone, Mat4::IDENTITY)
    }

    /// Attach at a local translation from the joint's origin.
    pub fn translated(bone: usize, translation: Vec3) -> Self {
        Self::new(bone, Mat4::from_translation(translation))
    }

    /// World matrix of the attachment under the current pose.
    pub fn world_matrix(
        &self,
        skeleton: &Skeleton,
        locals: &HashMap<usize, JointPose>,
    ) -> Result<Mat4, IkError> {
        let bone = pose::world_matrix(skeleton, locals, self.bone)
            .ok_or(IkError::UnknownJoint(self.bone))?;
        Ok(bone * self.offset)
    }

    /// World position of a point given in the attachment's local space.
    pub fn point(
        &self,
        skeleton: &Skeleton,
        locals: &HashMap<usize, JointPose>,
        local_point: Vec3,
    ) -> Result<Vec3, IkError> {
        Ok(self
            .world_matrix(skeleton, locals)?
            .transform_point3(local_point))
    }

    /// World direction of a vector given in the attachment's local space
    /// (rotated, not translated).
    pub fn direction(
        &self,
        skeleton: &Skeleton,
        locals: &HashMap<usize, JointPose>,
        local_direction: Vec3,
    ) -> Result<Vec3, IkError> {
        Ok(self
            .world_matrix(skeleton, locals)?
            .transform_vector3(local_direction))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    fn skeleton() -> Skeleton {
        Skeleton {
            parents: HashMap::from([(0, None), (1, Some(0))]),
            bind_local: HashMap::from([
                (0, Mat4::IDENTITY),
                (1, Mat4::from_translation(Vec3::new(0.0, 2.0, 0.0))),
            ]),
        }
    }

    #[test]
    fn attachment_follows_the_joint_pose() {
        let skeleton = skeleton();
        let mut locals = HashMap::new();
        // Rotate the root a quarter turn about Z and move it up.
        let root = JointPose {
            translation: [0.0, 1.0, 0.0],
            rotation: Quat::from_rotation_z(std::f32::consts::FRAC_PI_2).to_array(),
            ..Default::default()
        };
        locals.insert(0, root);

        let muzzle = BoneAttachment::translated(1, Vec3::new(1.0, 0.0, 0.0));
        let point = muzzle.point(&skeleton, &locals, Vec3::ZERO).expect("resolves");
        assert!(point.distance(Vec3::new(-2.0, 2.0, 0.0)) < 1e-4, "{point:?}");
        let forward = muzzle
            .direction(&skeleton, &locals, Vec3::Y)
            .expect("resolves");
        assert!(forward.distance(Vec3::new(-1.0, 0.0, 0.0)) < 1e-4, "{forward:?}");
    }
}
