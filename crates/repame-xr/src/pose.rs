//! Head and hand poses: rigid transforms, tracking state, and the snapshot type.
//!
//! A pose is a placement in a reference frame: position plus unit-quaternion
//! orientation. Repame follows the XR convention (right-handed, `+y` up, `-z`
//! forward), so an orientation coming off the runtime drops straight into
//! [`glam::Quat::from_xyzw`] with no axis fixups.
//!
//! [`TrackedPose::pose`] is an `Option` on purpose. When a runtime cannot track
//! a role it reports that in its own flags; the only faithful translation is "no
//! pose", so the abstraction carries `None` rather than a stale or identity
//! transform.

use glam::{Mat4, Quat, Vec3};

use crate::{InputPath, UserPath};

/// Which tracked device a pose belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TrackedRole {
    /// The headset itself.
    Head,
    /// One controller or tracked hand.
    Hand(Hand),
}

impl TrackedRole {
    /// Every role, in a stable order.
    pub const ALL: [TrackedRole; 3] = [
        TrackedRole::Head,
        TrackedRole::Hand(Hand::Left),
        TrackedRole::Hand(Hand::Right),
    ];

    /// Top-level user path this role lives under, in `/user/...` form.
    pub fn user_path(self) -> &'static str {
        match self {
            TrackedRole::Head => "/user/head",
            TrackedRole::Hand(Hand::Left) => "/user/hand/left",
            TrackedRole::Hand(Hand::Right) => "/user/hand/right",
        }
    }

    /// Index into a two-element per-hand array, or `None` for the head.
    pub fn hand_index(self) -> Option<usize> {
        match self {
            TrackedRole::Head => None,
            TrackedRole::Hand(hand) => Some(hand.index()),
        }
    }
}

/// Left or right hand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Hand {
    Left,
    Right,
}

impl Hand {
    /// Both hands, left first.
    pub const ALL: [Hand; 2] = [Hand::Left, Hand::Right];

    /// Position in a per-hand array.
    pub fn index(self) -> usize {
        match self {
            Hand::Left => 0,
            Hand::Right => 1,
        }
    }

    /// `"left"` or `"right"`, the OpenXR user-path segment.
    pub fn name(self) -> &'static str {
        match self {
            Hand::Left => "left",
            Hand::Right => "right",
        }
    }
}

/// Which pose the runtime reports for a hand: where it points, or where it is
/// held.
///
/// A runtime exposes both on every hand device, and they are not
/// interchangeable, so which one an app acts on is a choice rather than a
/// fallback. Aim is the default because it is the one a pointer, a ray or a
/// cursor wants.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum HandPose {
    /// Where the hand or controller points.
    #[default]
    Aim,
    /// Where the hand or controller is held.
    Grip,
}

impl HandPose {
    /// Every source, aim first.
    pub const ALL: [HandPose; 2] = [HandPose::Aim, HandPose::Grip];

    /// The OpenXR input this source reads: `"aim"` or `"grip"`.
    pub fn input(self) -> &'static str {
        match self {
            HandPose::Aim => "aim",
            HandPose::Grip => "grip",
        }
    }

    /// The `/user/hand/{left,right}/input/{aim,grip}/pose` path this source is
    /// read from, spelled the way this crate spells binding paths. This is what
    /// a backend binds its pose actions against.
    pub fn input_path(self, hand: Hand) -> InputPath {
        InputPath::new(
            UserPath::Hand(hand),
            self.input(),
            Some("pose".to_string()),
        )
    }
}

/// Rigid placement in a reference frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    /// Where the frame sits.
    pub position: Vec3,
    /// Unit quaternion taking frame-local axes into the reference frame.
    pub orientation: Quat,
}

impl Default for Pose {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Pose {
    /// Origin, unrotated.
    pub const IDENTITY: Pose = Pose {
        position: Vec3::ZERO,
        orientation: Quat::IDENTITY,
    };

    /// A pose with `orientation` normalized. A zero quaternion becomes the
    /// identity; non-finite input is passed through unchanged so that
    /// [`Pose::is_finite`] still rejects it.
    pub fn new(position: Vec3, orientation: Quat) -> Self {
        let len_sq = orientation.length_squared();
        Self {
            position,
            orientation: if len_sq.is_finite() && len_sq > 0.0 {
                orientation / len_sq.sqrt()
            } else {
                orientation
            },
        }
    }

    /// From runtime components: `[x, y, z]` position and `[x, y, z, w]`
    /// orientation, both in meters and as the runtime reported them.
    pub fn from_xyzw(position: [f32; 3], orientation: [f32; 4]) -> Self {
        Self::new(
            Vec3::from_array(position),
            Quat::from_xyzw(
                orientation[0],
                orientation[1],
                orientation[2],
                orientation[3],
            ),
        )
    }

    /// True when every component is finite, so the pose can be trusted.
    pub fn is_finite(&self) -> bool {
        self.position.is_finite() && self.orientation.is_finite()
    }

    /// Frame-to-reference transform.
    pub fn transform(&self) -> Mat4 {
        Mat4::from_rotation_translation(self.orientation, self.position)
    }

    /// Reference-to-frame transform, i.e. the view matrix of a camera placed by
    /// this pose.
    pub fn view_matrix(&self) -> Mat4 {
        self.transform().inverse()
    }

    /// This pose expressed in `parent`'s frame.
    pub fn relative_to(&self, parent: &Pose) -> Pose {
        let local = parent.view_matrix() * self.transform();
        let (_, rotation, translation) = local.to_scale_rotation_translation();
        Pose::new(translation, rotation)
    }

    /// The inverse placement.
    pub fn inverse(&self) -> Pose {
        let orientation = self.orientation.conjugate();
        Pose::new(orientation * -self.position, orientation)
    }

    /// Move a reference-frame point into this frame.
    pub fn to_local_point(&self, point: Vec3) -> Vec3 {
        self.orientation.conjugate() * (point - self.position)
    }

    /// Move a frame-local point into the reference frame.
    pub fn to_world_point(&self, point: Vec3) -> Vec3 {
        self.orientation * point + self.position
    }

    /// Where this frame looks, in the reference frame.
    pub fn forward(&self) -> Vec3 {
        self.orientation * Vec3::NEG_Z
    }
}

/// How trustworthy the runtime's answer for a tracked role is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoseState {
    /// Position and orientation both came from tracking.
    Tracked,
    /// The runtime is inferring this pose from another source, not measuring it.
    Emulated,
    /// No pose was available. [`TrackedPose::pose`] is `None`.
    Untracked,
}

/// One role's pose plus how it was obtained.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackedPose {
    /// `None` exactly when no pose could be reported. Never a stand-in.
    pub pose: Option<Pose>,
    /// Where [`TrackedPose::pose`] came from.
    pub state: PoseState,
}

impl TrackedPose {
    /// No pose.
    pub const UNTRACKED: TrackedPose = TrackedPose {
        pose: None,
        state: PoseState::Untracked,
    };

    /// A tracked pose.
    pub fn tracked(pose: Pose) -> Self {
        Self {
            pose: Some(pose),
            state: PoseState::Tracked,
        }
    }

    /// Build from runtime flags: `valid` must already combine position and
    /// orientation validity.
    pub fn from_valid_flags(pose: Pose, valid: bool) -> Self {
        if valid {
            Self::tracked(pose)
        } else {
            Self::UNTRACKED
        }
    }

    /// Build from a runtime's validity and tracking flags: `valid` says the
    /// runtime has a pose, `tracked` says it measured it rather than inferring
    /// it. An invalid answer is [`TrackedPose::UNTRACKED`], never a stand-in;
    /// a valid but inferred pose is reported as [`PoseState::Emulated`].
    pub fn from_tracking_flags(pose: Pose, valid: bool, tracked: bool) -> Self {
        match (valid, tracked) {
            (false, _) => Self::UNTRACKED,
            (true, true) => Self::tracked(pose),
            (true, false) => Self {
                pose: Some(pose),
                state: PoseState::Emulated,
            },
        }
    }

    /// The pose, or an error naming `what` when there is none.
    pub fn required(&self, what: &'static str) -> Result<Pose, crate::XrError> {
        self.pose.ok_or(crate::XrError::Untracked { what })
    }
}

impl Default for TrackedPose {
    /// Untracked. A role the runtime did not report is not a zero pose.
    fn default() -> Self {
        Self::UNTRACKED
    }
}

/// The poses a frame asked for. Roles that were not requested, and roles the
/// runtime could not track, are both [`TrackedPose::UNTRACKED`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Poses {
    head: TrackedPose,
    hands: [TrackedPose; 2],
}

impl Poses {
    /// An empty snapshot.
    pub fn new() -> Self {
        Self {
            head: TrackedPose::UNTRACKED,
            hands: [TrackedPose::UNTRACKED; 2],
        }
    }

    /// Head pose.
    pub fn head(&self) -> TrackedPose {
        self.head
    }

    /// One hand's pose.
    pub fn hand(&self, hand: Hand) -> TrackedPose {
        self.hands[hand.index()]
    }

    /// Any role's pose.
    pub fn get(&self, role: TrackedRole) -> TrackedPose {
        match role {
            TrackedRole::Head => self.head,
            TrackedRole::Hand(hand) => self.hand(hand),
        }
    }

    /// Record a role's pose.
    pub fn set(&mut self, role: TrackedRole, pose: TrackedPose) {
        match role {
            TrackedRole::Head => self.head = pose,
            TrackedRole::Hand(hand) => self.hands[hand.index()] = pose,
        }
    }

    /// The head pose, or an error when the runtime is not tracking it.
    pub fn head_pose(&self) -> Result<Pose, crate::XrError> {
        self.head.required("head")
    }

    /// One hand's pose, or an error when the runtime is not tracking it.
    pub fn hand_pose(&self, hand: Hand) -> Result<Pose, crate::XrError> {
        self.hand(hand).required(match hand {
            Hand::Left => "left hand",
            Hand::Right => "right hand",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::vec3;

    fn close(a: Mat4, b: Mat4) -> bool {
        (a - b).to_cols_array().iter().all(|x| x.abs() < 1e-5)
    }

    fn sample() -> Pose {
        Pose::new(
            vec3(0.25, 1.6, -0.75),
            Quat::from_rotation_y(0.6) * Quat::from_rotation_x(-0.3),
        )
    }

    #[test]
    fn view_matrix_inverts_the_frame_transform() {
        let pose = sample();
        assert!(close(pose.transform() * pose.view_matrix(), Mat4::IDENTITY));
        assert!(Pose::IDENTITY.view_matrix() == Mat4::IDENTITY);

        // A camera placed by the pose looks down its own -z.
        let point = pose.to_world_point(vec3(0.0, 0.0, -2.0));
        let eye = pose.view_matrix().transform_point3(point);
        assert!((eye - vec3(0.0, 0.0, -2.0)).length() < 1e-5);
    }

    #[test]
    fn inverse_and_relative_agree_with_the_matrices() {
        let pose = sample();
        let inverse = pose.inverse();
        assert!(close(inverse.transform(), pose.view_matrix()));
        assert!(close(pose.transform() * inverse.transform(), Mat4::IDENTITY));

        // Expressing a pose in its own frame is the identity.
        let relative = pose.relative_to(&pose);
        assert!(relative.position.length() < 1e-5);
        assert!((relative.orientation.x).abs() < 1e-5);
        assert!((relative.orientation.y).abs() < 1e-5);
        assert!((relative.orientation.z).abs() < 1e-5);

        // child = parent * local, for an arbitrary parent.
        let parent = Pose::new(vec3(-1.0, 0.5, 2.0), Quat::from_rotation_z(0.9));
        let local = pose.relative_to(&parent);
        assert!(close(parent.transform() * local.transform(), pose.transform()));
    }

    #[test]
    fn normalization_and_finiteness() {
        let scaled = Pose::new(Vec3::ZERO, Quat::from_xyzw(0.0, 0.0, 0.0, 4.0));
        assert!((scaled.orientation.length() - 1.0).abs() < 1e-6);
        assert!(scaled.is_finite());
        assert!(!Pose::new(vec3(f32::NAN, 0.0, 0.0), Quat::IDENTITY).is_finite());
        assert!(
            !Pose::new(
                Vec3::ZERO,
                Quat::from_xyzw(0.0, f32::INFINITY, 0.0, 1.0)
            )
            .is_finite()
        );

        let zero = Pose::new(Vec3::ZERO, Quat::from_xyzw(0.0, 0.0, 0.0, 0.0));
        assert_eq!(zero.orientation, Quat::IDENTITY);

        let (position, orientation) = (sample().position, sample().orientation);
        let restored = Pose::from_xyzw(
            position.to_array(),
            [orientation.x, orientation.y, orientation.z, orientation.w],
        );
        assert!(close(restored.transform(), sample().transform()));

        assert!(Pose::IDENTITY.forward().dot(Vec3::NEG_Z) > 0.999);
    }

    #[test]
    fn hand_pose_sources_spell_the_input_paths_they_read() {
        assert_eq!(HandPose::default(), HandPose::Aim);
        assert_eq!(HandPose::ALL, [HandPose::Aim, HandPose::Grip]);
        assert_eq!(
            HandPose::Aim.input_path(Hand::Left).as_path(),
            "/user/hand/left/input/aim/pose"
        );
        assert_eq!(
            HandPose::Grip.input_path(Hand::Right).as_path(),
            "/user/hand/right/input/grip/pose"
        );

        let path = HandPose::Grip.input_path(Hand::Left);
        assert_eq!(InputPath::parse(&path.as_path()).unwrap(), path);
        assert_eq!(UserPath::Hand(Hand::Left), path.user);
        assert_eq!(path.input, HandPose::Grip.input());
        assert_eq!(path.component.as_deref(), Some("pose"));
    }

    #[test]
    fn untracked_roles_are_absent_not_faked() {
        let mut poses = Poses::new();
        assert_eq!(poses.head().state, PoseState::Untracked);
        assert!(poses.head().pose.is_none());
        assert!(poses.head_pose().is_err());

        poses.set(TrackedRole::Hand(Hand::Right), TrackedPose::tracked(sample()));
        assert!(poses.hand_pose(Hand::Right).is_ok());
        assert!(poses.hand_pose(Hand::Left).is_err());
        assert_eq!(
            poses.get(TrackedRole::Hand(Hand::Right)).state,
            PoseState::Tracked
        );
        assert!(TrackedPose::from_valid_flags(sample(), false).pose.is_none());
        assert!(TrackedPose::from_valid_flags(sample(), true).pose.is_some());

        assert_eq!(
            TrackedPose::from_tracking_flags(sample(), true, false).state,
            PoseState::Emulated
        );
        assert!(
            TrackedPose::from_tracking_flags(sample(), true, false)
                .pose
                .is_some()
        );
        assert!(
            TrackedPose::from_tracking_flags(sample(), false, true)
                .pose
                .is_none()
        );
        assert_eq!(
            TrackedPose::from_tracking_flags(sample(), true, true).state,
            PoseState::Tracked
        );
    }
}
