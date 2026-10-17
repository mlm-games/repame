//! Inverse kinematics and skeleton modifiers over [`repame_view3d`]
//! skeletons.
//!
//! Everything works in place on the engine's own types: a
//! [`Skeleton`](repame_view3d::Skeleton) (parent links plus bind locals)
//! together with the animated local poses the game already keeps in a
//! `HashMap<usize, JointPose>`. No parallel skeleton format is introduced.
//! A solver reads that map, writes joint *rotations* back into it, and the
//! caller re-bakes the mesh exactly as it does for plain animation
//! ([`Skeleton::world_matrices`], [`SkinnedMesh::pose`], or a
//! [`SkinnedDraw`]).
//!
//! ```ignore
//! use repame_ik::{Fabrik, JointChain, solve_two_bone};
//!
//! let chain = JointChain::new(&skeleton, vec![shoulder, elbow, wrist])?;
//! Fabrik::default().solve(&skeleton, &mut locals, &chain, hand_target, &limits)?;
//! solve_two_bone(&skeleton, &mut locals, shoulder, elbow, wrist, grip, pole)?;
//! let mats = skeleton.world_matrices(&locals);
//! ```
//!
//! [`SkinnedMesh::pose`]: repame_view3d::SkinnedMesh::pose
//! [`SkinnedDraw`]: repame_view3d::SkinnedDraw
//!
//! # Solvers and modifiers
//!
//! - [`Fabrik`] and [`Ccd`]: iterative solvers for an arbitrary
//!   ancestor-to-descendant [`JointChain`], with per-joint [`JointLimits`],
//!   an iteration cap, and a tolerance. An unreachable target straightens
//!   the chain toward it.
//! - [`solve_two_bone`]: the analytic shoulder/elbow/knee limb with a pole.
//! - [`look_at`] and [`aim`]: point one joint's forward axis at a target,
//!   the second constrained to a single rotation axis.
//! - [`SpringBoneChain`]: a verlet chain with stiffness, damping, and
//!   gravity for hair and tails.
//! - [`BoneAttachment`]: resolve a transform or point fixed to a joint.

pub mod aim;
pub mod attach;
pub mod chain;
pub mod error;
pub mod iterative;
pub mod look_at;
pub mod pose;
pub mod spring;
pub mod two_bone;

pub use aim::{Aim, aim};
pub use attach::BoneAttachment;
pub use chain::{JointChain, JointLimits};
pub use error::IkError;
pub use iterative::{Ccd, Fabrik, IkResult};
pub use look_at::{LookAt, ModifierOutcome, look_at};
pub use spring::{SpringBoneChain, SpringBoneSettings};
pub use two_bone::solve_two_bone;
