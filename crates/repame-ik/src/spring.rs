//! Spring-bone modifier: a verlet chain that lags the animated pose, with
//! stiffness, damping, and gravity.

use std::collections::HashMap;

use glam::Vec3;
use repame_view3d::{JointPose, Skeleton};

use crate::chain::{self, JointChain, JointLimits};
use crate::error::IkError;
use crate::pose;

/// Spring-bone settings, applied to every joint of the chain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpringBoneSettings {
    /// Pull toward the animated bone direction, per second.
    pub stiffness: f32,
    /// Velocity damping in `0..=1`: `0` keeps the full previous velocity,
    /// `1` discards it (the chain settles without overshoot).
    pub damping: f32,
    /// World-space acceleration in units per second squared.
    pub gravity: Vec3,
}

impl Default for SpringBoneSettings {
    fn default() -> Self {
        Self {
            stiffness: 0.5,
            damping: 0.0,
            gravity: Vec3::ZERO,
        }
    }
}

/// A verlet spring chain over a joint path. Each joint carries one tail
/// point; [`step`](SpringBoneChain::step) integrates the tails and writes
/// the joints' rotations back into the caller's pose map.
///
/// The driven direction of joint `i` is its bind direction under the
/// current parent pose, so the spring's own rotation never becomes its own
/// rest, and the chain has a stable pendulum target.
#[derive(Clone, Debug)]
pub struct SpringBoneChain {
    chain: JointChain,
    tails: Vec<Vec3>,
    previous: Vec<Vec3>,
}

impl SpringBoneChain {
    /// Build a chain over an ancestor path (see [`JointChain::new`]).
    pub fn new(skeleton: &Skeleton, joints: Vec<usize>) -> Result<Self, IkError> {
        let chain = JointChain::new(skeleton, joints)?;
        let count = chain.joints().len();
        Ok(Self {
            chain,
            tails: vec![Vec3::ZERO; count],
            previous: vec![Vec3::ZERO; count],
        })
    }

    /// Extend the chain past the last joint (hair/tail tip), see
    /// [`JointChain::with_end_offset`].
    pub fn with_end_offset(mut self, offset: Vec3) -> Self {
        self.chain = self.chain.with_end_offset(offset);
        self
    }

    /// The underlying chain.
    pub fn chain(&self) -> &JointChain {
        &self.chain
    }

    /// Current world tail per joint.
    pub fn tails(&self) -> &[Vec3] {
        &self.tails
    }

    /// Overwrite one joint's tail while keeping the previous tail: an
    /// external impulse. Call [`reseed`](SpringBoneChain::reseed) first for
    /// a clean start.
    pub fn set_tail(&mut self, index: usize, world: Vec3) {
        if index < self.tails.len() {
            self.tails[index] = world;
        }
    }

    /// Drop all velocity and pin every tail to the current bind-under-parent
    /// pose. Call after teleporting the skeleton.
    pub fn reseed(
        &mut self,
        skeleton: &Skeleton,
        locals: &HashMap<usize, JointPose>,
    ) -> Result<(), IkError> {
        for i in 0..self.chain.joints().len() {
            let tail = self.rest_tail(skeleton, locals, i)?;
            self.tails[i] = tail;
            self.previous[i] = tail;
        }
        Ok(())
    }

    /// Advance the chain by `dt` seconds. Integrates every tail, keeps each
    /// segment rigid, and writes joint rotations into `locals` in place.
    pub fn step(
        &mut self,
        skeleton: &Skeleton,
        locals: &mut HashMap<usize, JointPose>,
        settings: &SpringBoneSettings,
        dt: f32,
    ) -> Result<(), IkError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Ok(());
        }
        let damping = settings.damping.clamp(0.0, 1.0);
        let count = self.chain.joints().len();
        for i in 0..count {
            let length = self.chain.lengths()[i];
            if length <= 1e-8 {
                continue;
            }
            let joint = self.chain.joints()[i];
            let world =
                pose::world_matrix(skeleton, locals, joint).ok_or(IkError::UnknownJoint(joint))?;
            let origin = world.transform_point3(Vec3::ZERO);
            let rest_dir = self.rest_direction(skeleton, locals, i)?;
            if rest_dir == Vec3::ZERO {
                continue;
            }

            let current = self.tails[i];
            let velocity = (current - self.previous[i]) * (1.0 - damping);
            let pull = rest_dir * (settings.stiffness * dt) + settings.gravity * dt;
            let mut next = current + velocity + pull;
            let offset = next - origin;
            next = if offset.length_squared() > 1e-12 {
                origin + offset.normalize() * length
            } else {
                origin + rest_dir * length
            };

            self.previous[i] = current;
            self.tails[i] = next;

            let to = next - origin;
            if to.length_squared() > 1e-12 {
                let delta = pose::rotation_between(rest_dir, to);
                chain::rotate_joint(skeleton, locals, joint, self.chain.axes()[i], delta, JointLimits::FREE)?;
            }
        }
        Ok(())
    }

    fn rest_tail(
        &self,
        skeleton: &Skeleton,
        locals: &HashMap<usize, JointPose>,
        index: usize,
    ) -> Result<Vec3, IkError> {
        let joint = self.chain.joints()[index];
        let world = pose::world_matrix(skeleton, locals, joint).ok_or(IkError::UnknownJoint(joint))?;
        let offset = self.chain.axes()[index] * self.chain.lengths()[index];
        Ok(world.transform_point3(offset))
    }

    /// The joint's bind direction under the current parent pose: where the
    /// bone would point if this joint kept its bind local rotation.
    fn rest_direction(
        &self,
        skeleton: &Skeleton,
        locals: &HashMap<usize, JointPose>,
        index: usize,
    ) -> Result<Vec3, IkError> {
        let joint = self.chain.joints()[index];
        let parent_world = match skeleton.parents.get(&joint).copied().flatten() {
            Some(parent) => {
                pose::world_matrix(skeleton, locals, parent).ok_or(IkError::UnknownJoint(parent))?
            }
            None => glam::Mat4::IDENTITY,
        };
        let rest_world = parent_world * pose::bind_matrix(skeleton, joint);
        Ok((pose::world_rotation(&rest_world) * self.chain.axes()[index]).normalize_or_zero())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Mat4;

    fn hanging_chain() -> (Skeleton, SpringBoneChain) {
        let skeleton = Skeleton {
            parents: HashMap::from([(0, None), (1, Some(0))]),
            bind_local: HashMap::from([
                (0, Mat4::IDENTITY),
                (1, Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0))),
            ]),
        };
        let chain = SpringBoneChain::new(&skeleton, vec![0, 1]).expect("valid chain");
        (skeleton, chain)
    }

    fn deviation(tail: Vec3) -> f32 {
        tail.dot(Vec3::Y).clamp(-1.0, 1.0).acos()
    }

    #[test]
    fn spring_bone_oscillates_without_damping() {
        let (skeleton, mut chain) = hanging_chain();
        let mut locals = HashMap::new();
        chain.reseed(&skeleton, &locals).expect("seeds");
        // A tail displaced from rest is an impulse: the undamped chain must
        // swing across the rest direction instead of settling onto it.
        chain.set_tail(0, Vec3::new(0.5, 0.866_025_4, 0.0));
        let settings = SpringBoneSettings {
            stiffness: 1.0,
            damping: 0.0,
            gravity: Vec3::ZERO,
        };
        let mut signs = Vec::new();
        for _ in 0..200 {
            chain
                .step(&skeleton, &mut locals, &settings, 0.05)
                .expect("steps");
            let tail = chain.tails()[0];
            assert!(
                (tail.length() - 1.0).abs() < 1e-3,
                "rigid, got {}",
                tail.length()
            );
            signs.push(tail.x.is_sign_positive());
        }
        assert!(
            signs.windows(2).any(|w| w[0] != w[1]),
            "an undamped spring should swing across its rest direction"
        );
    }

    #[test]
    fn spring_bone_settles_when_fully_damped() {
        let (skeleton, mut chain) = hanging_chain();
        let mut locals = HashMap::new();
        chain.reseed(&skeleton, &locals).expect("seeds");
        chain.set_tail(0, Vec3::new(0.5, 0.866_025_4, 0.0));
        let settings = SpringBoneSettings {
            stiffness: 1.0,
            damping: 1.0,
            gravity: Vec3::ZERO,
        };
        let mut last = deviation(chain.tails()[0]);
        for _ in 0..120 {
            chain
                .step(&skeleton, &mut locals, &settings, 0.05)
                .expect("steps");
            let now = deviation(chain.tails()[0]);
            assert!(
                now <= last + 1e-4,
                "fully damped spring must not overshoot: {last} -> {now}"
            );
            last = now;
        }
        assert!(last < 1e-2, "settles at rest, deviation {last}");
    }
}
