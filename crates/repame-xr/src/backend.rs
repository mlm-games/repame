//! Which runtime this build talks to.
//!
//! The runtime-agnostic core is everything in the sibling modules: session
//! state, reference spaces, stereo view math, the action model and the
//! swapchain handoff are pure and need no runtime at all. A backend adds only
//! the calls that touch one, and this module picks it:
//!
//! - The OpenXR backend (feature `openxr`, every target but wasm32) dlopens the
//!   OpenXR loader at run time and creates a real session against the render
//!   layer's Vulkan device. A machine with no loader, no runtime or no headset
//!   answers [`XrError::Unavailable`] from [`Backend::probe`] and
//!   [`Backend::open`], which is a real answer: run flat.
//! - The WebXR backend (feature `webxr`, wasm32 only) drives the browser's own
//!   WebXR implementation. It needs the render layer's WebGL2 context, so it is
//!   opened with [`crate::XrSession::open_webgl`] rather than from a
//!   [`SessionConfig`] alone.
//! - A build with neither feature, or a WebXR build opened without a context,
//!   has no runtime behind it: every call answers a typed error and the
//!   reported state stays [`SessionState::Idle`]. That is what tells a game to
//!   run flat, and no backend ever fabricates a pose or an image to avoid it.

use crate::action::{ActionSet, ActionState, InteractionProfile, UserPath, find_action};
use crate::error::{XrError, XrResult};
use crate::pose::{HandPose, Poses};
use crate::session::{SessionConfig, SessionState};
use crate::space::ReferenceSpace;
use crate::swapchain::{FrameError, FrameOutcome, StereoSubmit};

/// Backend name reported when no runtime backend is compiled in.
pub const NONE: &str = "none";

const NO_BACKEND: &str = "this build has no XR runtime backend";

/// The runtime seam.
///
/// `probe` tries each backend this build compiled in, in order: WebXR first on
/// wasm32, then OpenXR. `open` hands the session to the first that answers, and
/// [`Backend::None`] carries the runtime-less arm every other build falls back
/// to.
pub(crate) enum Backend {
    /// No runtime behind this build.
    None(NoneBackend),
    /// A real OpenXR runtime, loaded at run time.
    #[cfg(all(feature = "openxr", not(target_arch = "wasm32")))]
    OpenXr(Box<crate::openxr::OpenXrBackend>),
    /// The browser's WebXR runtime.
    #[cfg(all(target_arch = "wasm32", feature = "webxr"))]
    WebXr(Box<crate::webxr::WebXrBackend>),
}

impl Backend {
    /// Probe for a usable runtime without creating a session.
    pub(crate) fn probe() -> XrResult<()> {
        #[cfg(all(feature = "openxr", not(target_arch = "wasm32")))]
        return crate::openxr::probe();
        #[cfg(all(target_arch = "wasm32", feature = "webxr"))]
        return crate::webxr::probe();
        #[cfg(not(any(
            all(feature = "openxr", not(target_arch = "wasm32")),
            all(target_arch = "wasm32", feature = "webxr")
        )))]
        Err(no_backend())
    }

    /// Create a session from configuration on the first backend that answers.
    ///
    /// The WebXR arm cannot answer here: a WebXR session is created against the
    /// render layer's WebGL2 context, which a [`SessionConfig`] cannot carry.
    /// [`crate::XrSession::open_webgl`] is that backend's door.
    pub(crate) fn open(_config: &SessionConfig) -> XrResult<Self> {
        #[cfg(all(feature = "openxr", not(target_arch = "wasm32")))]
        return crate::openxr::OpenXrBackend::open(_config)
            .map(|backend| Backend::OpenXr(Box::new(backend)));
        #[cfg(all(target_arch = "wasm32", feature = "webxr"))]
        return Err(crate::webxr::needs_context_error());
        #[cfg(not(any(
            all(feature = "openxr", not(target_arch = "wasm32")),
            all(target_arch = "wasm32", feature = "webxr")
        )))]
        Err(no_backend())
    }

    /// A backend with no runtime behind it. Every runtime call reports the
    /// typed error and the reported state stays [`SessionState::Idle`].
    pub(crate) fn unattached() -> Self {
        Backend::None(NoneBackend)
    }

    /// Name of the backend that would answer.
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Backend::None(_) => NONE,
            #[cfg(all(feature = "openxr", not(target_arch = "wasm32")))]
            Backend::OpenXr(_) => crate::openxr::NAME,
            #[cfg(all(target_arch = "wasm32", feature = "webxr"))]
            Backend::WebXr(_) => crate::webxr::NAME,
        }
    }

    /// Runtime-reported session state.
    pub(crate) fn state(&self) -> SessionState {
        match self {
            Backend::None(backend) => backend.state(),
            #[cfg(all(feature = "openxr", not(target_arch = "wasm32")))]
            Backend::OpenXr(backend) => backend.state(),
            #[cfg(all(target_arch = "wasm32", feature = "webxr"))]
            Backend::WebXr(backend) => backend.state(),
        }
    }

    /// Ask the runtime to begin the session.
    pub(crate) fn begin(&mut self) -> XrResult<()> {
        match self {
            Backend::None(backend) => backend.begin(),
            #[cfg(all(feature = "openxr", not(target_arch = "wasm32")))]
            Backend::OpenXr(backend) => backend.begin(),
            #[cfg(all(target_arch = "wasm32", feature = "webxr"))]
            Backend::WebXr(backend) => backend.begin(),
        }
    }

    /// Ask the runtime to end the session.
    pub(crate) fn end(&mut self) -> XrResult<()> {
        match self {
            Backend::None(backend) => backend.end(),
            #[cfg(all(feature = "openxr", not(target_arch = "wasm32")))]
            Backend::OpenXr(backend) => backend.end(),
            #[cfg(all(target_arch = "wasm32", feature = "webxr"))]
            Backend::WebXr(backend) => backend.end(),
        }
    }

    /// Latest poses for the roles the session tracks.
    pub(crate) fn poses(&self) -> Poses {
        match self {
            Backend::None(backend) => backend.poses(),
            #[cfg(all(feature = "openxr", not(target_arch = "wasm32")))]
            Backend::OpenXr(backend) => backend.poses(),
            #[cfg(all(target_arch = "wasm32", feature = "webxr"))]
            Backend::WebXr(backend) => backend.poses(),
        }
    }

    /// Install the session's action sets.
    pub(crate) fn set_action_sets(&mut self, sets: Vec<ActionSet>) -> XrResult<()> {
        match self {
            Backend::None(backend) => backend.set_action_sets(sets),
            #[cfg(all(feature = "openxr", not(target_arch = "wasm32")))]
            Backend::OpenXr(backend) => backend.set_action_sets(sets),
            #[cfg(all(target_arch = "wasm32", feature = "webxr"))]
            Backend::WebXr(backend) => backend.set_action_sets(sets),
        }
    }

    /// Choose which of the two hand poses is read.
    pub(crate) fn set_hand_pose_source(&mut self, pose: HandPose) {
        match self {
            Backend::None(backend) => backend.set_hand_pose_source(pose),
            #[cfg(all(feature = "openxr", not(target_arch = "wasm32")))]
            Backend::OpenXr(backend) => backend.set_hand_pose_source(pose),
            #[cfg(all(target_arch = "wasm32", feature = "webxr"))]
            Backend::WebXr(backend) => backend.set_hand_pose_source(pose),
        }
    }

    /// Add profiles whose bindings the runtime registers.
    pub(crate) fn set_fallback_profiles(&mut self, profiles: Vec<InteractionProfile>) {
        match self {
            Backend::None(backend) => backend.set_fallback_profiles(profiles),
            #[cfg(all(feature = "openxr", not(target_arch = "wasm32")))]
            Backend::OpenXr(backend) => backend.set_fallback_profiles(profiles),
            #[cfg(all(target_arch = "wasm32", feature = "webxr"))]
            Backend::WebXr(backend) => backend.set_fallback_profiles(profiles),
        }
    }

    /// States produced by the last sync.
    pub(crate) fn action_states(&self) -> &[ActionState] {
        match self {
            Backend::None(backend) => backend.action_states(),
            #[cfg(all(feature = "openxr", not(target_arch = "wasm32")))]
            Backend::OpenXr(backend) => backend.action_states(),
            #[cfg(all(target_arch = "wasm32", feature = "webxr"))]
            Backend::WebXr(backend) => backend.action_states(),
        }
    }

    /// One action's state by set and action name.
    pub(crate) fn action(&self, set: &str, action: &str) -> XrResult<&ActionState> {
        match self {
            Backend::None(backend) => backend.action(set, action),
            #[cfg(all(feature = "openxr", not(target_arch = "wasm32")))]
            Backend::OpenXr(backend) => backend.action(set, action),
            #[cfg(all(target_arch = "wasm32", feature = "webxr"))]
            Backend::WebXr(backend) => backend.action(set, action),
        }
    }

    /// The interaction profile the runtime reports for a device.
    pub(crate) fn interaction_profile(&self, user: &UserPath) -> Option<InteractionProfile> {
        match self {
            Backend::None(backend) => backend.interaction_profile(user),
            #[cfg(all(feature = "openxr", not(target_arch = "wasm32")))]
            Backend::OpenXr(backend) => backend.interaction_profile(user),
            #[cfg(all(target_arch = "wasm32", feature = "webxr"))]
            Backend::WebXr(backend) => backend.interaction_profile(user),
        }
    }

    /// Drive one frame, handing each eye's images to `sink`.
    pub(crate) fn frame<S: StereoSubmit>(
        &mut self,
        space: ReferenceSpace,
        sink: &mut S,
    ) -> Result<FrameOutcome, FrameError<S::Error>> {
        match self {
            Backend::None(backend) => backend.frame(space, sink),
            #[cfg(all(feature = "openxr", not(target_arch = "wasm32")))]
            Backend::OpenXr(backend) => backend.frame(space, sink),
            #[cfg(all(target_arch = "wasm32", feature = "webxr"))]
            Backend::WebXr(backend) => backend.frame(space, sink),
        }
    }
}

/// The typed answer of a build with no runtime behind it.
pub(crate) fn no_backend() -> XrError {
    XrError::Unavailable {
        backend: NONE,
        detail: NO_BACKEND.to_string(),
    }
}

/// The runtime-less arm.
pub(crate) struct NoneBackend;

impl NoneBackend {
    fn state(&self) -> SessionState {
        SessionState::Idle
    }

    fn begin(&self) -> XrResult<()> {
        Err(no_backend())
    }

    fn end(&self) -> XrResult<()> {
        Err(no_backend())
    }

    fn poses(&self) -> Poses {
        Poses::new()
    }

    /// Installing sets is a runtime call: with no runtime there is nothing to
    /// attach them to, so it is the same typed error every other runtime call
    /// gives.
    fn set_action_sets(&mut self, _sets: Vec<ActionSet>) -> XrResult<()> {
        Err(no_backend())
    }

    /// A preference about which hand pose to read. Nothing here ever reports a
    /// hand, so there is nothing for it to affect; it is accepted so that an
    /// app can drive one code path with or without a runtime.
    fn set_hand_pose_source(&mut self, _pose: HandPose) {}

    /// Likewise for profiles: this backend registers no bindings, so there is
    /// no binding UI to add them to.
    fn set_fallback_profiles(&mut self, _profiles: Vec<InteractionProfile>) {}

    fn action_states(&self) -> &[ActionState] {
        &[]
    }

    fn action(&self, set: &str, action: &str) -> XrResult<&ActionState> {
        find_action(&[], set, action)
    }

    fn interaction_profile(&self, _user: &UserPath) -> Option<InteractionProfile> {
        None
    }

    fn frame<S: StereoSubmit>(
        &self,
        _space: ReferenceSpace,
        _sink: &mut S,
    ) -> Result<FrameOutcome, FrameError<S::Error>> {
        Err(FrameError::Xr(no_backend()))
    }
}
