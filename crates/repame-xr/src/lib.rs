//! XR: session lifecycle, reference spaces, stereo views, action sets.
//!
//! The crate is split so the parts a game can use today are exactly the parts
//! that do not need a runtime:
//!
//! - [`SessionState`] is the runtime's own state machine, including which
//!   call is legal in which state ([`SessionState::can_begin`],
//!   [`SessionState::should_submit_frames`]).
//! - [`ReferenceSpace`], [`Pose`]/[`Poses`] and [`ViewSetup`] are the tracking
//!   side. A pose is either tracked or explicitly untracked
//!   ([`TrackedPose::UNTRACKED`]), never a fabricated identity.
//! - [`ActionSet`], [`Action`], [`ActionValue`] and [`parse_binding_path`]
//!   model inputs and binding paths. [`XrSession::set_action_sets`] hands them
//!   to the runtime, [`XrSession::frame`] syncs them once per frame, and
//!   [`XrSession::action_states`] reports what the runtime said. An action the
//!   runtime has no input for is inactive at its type's zero value;
//!   [`combine`] is the rule that folds one action's per-device readings into
//!   that single answer, and [`gamepad`] is how a WebXR input source's gamepad
//!   carries a binding the runtime has no interaction profile for.
//! - [`StereoSubmit`] is what a render layer implements, so a backend can hand
//!   it each eye's images for the duration of a frame.
//!
//! [`XrSession`] is the runtime seam. Two real backends sit behind it:
//!
//! - **OpenXR** (feature `openxr`, every target but wasm32) dlopens the OpenXR
//!   loader, creates an instance and a session against the render layer's Vulkan
//!   device, and drives the runtime's frame loop.
//! - **WebXR** (feature `webxr`, wasm32 only) opens a session against the render
//!   layer's WebGL2 context and drives the runtime's animation frames.
//!
//! With no backend compiled in — or a runtime that is not there —
//! [`XrSession::probe`] answers [`XrError::Unavailable`], which is what a game
//! checks to decide to run flat. [`XrSession::open`] answers the same when there
//! is no runtime, and something more specific when a runtime answered but cannot
//! carry this build's session (a missing graphics binding, say). A session
//! object with no runtime behind it is available from [`XrSession::unattached`]
//! so a game can wire one code path regardless.
//!
//! No backend fabricates anything: an eye the runtime is not tracking has no
//! pose and no view matrix, a frame the runtime does not want is reported as
//! not rendered, an action the runtime is not driving reads as inactive at its
//! type's zero value rather than as a stale one, a hand the runtime is not
//! tracking has no pose, and a runtime that is not there is a typed error
//! rather than a stand-in session.

mod backend;

#[cfg(all(feature = "openxr", not(target_arch = "wasm32")))]
mod openxr;

#[cfg(all(target_arch = "wasm32", feature = "webxr"))]
mod webxr;

pub mod action;
pub mod error;
pub mod gamepad;
pub mod pose;
pub mod session;
pub mod space;
pub mod stereo;
pub mod swapchain;

pub use action::{
    Action, ActionKind, ActionReading, ActionSet, ActionState, ActionValue, Binding, InputPath,
    InteractionProfile, ParsedBindingPath, UserPath, combine, find_action, parse_binding_path,
};
pub use backend::NONE as NO_BACKEND_NAME;
pub use error::{PathError, XrError};
pub use gamepad::{GamepadButton, GamepadElement, element_reading, gamepad_element};
pub use pose::{Hand, HandPose, Pose, PoseState, Poses, TrackedPose, TrackedRole};
pub use session::{SessionConfig, SessionState};
pub use space::ReferenceSpace;
pub use stereo::{
    BlendMode, Eye, Fov, PixelFormat, ViewConfig, ViewConfiguration, ViewSetup, projection_from_fov,
};
#[cfg(all(target_arch = "wasm32", feature = "webxr"))]
pub use webxr::XrFrame;
pub use swapchain::{
    EyeImage, FrameError, FrameOutcome, GraphicsApi, GraphicsBinding, StereoFrame, StereoSubmit,
    SwapchainFormat, SwapchainImageHandle, VulkanBinding,
};

/// Result of a runtime call.
pub type XrResult<T> = Result<T, XrError>;

/// A session with a runtime behind it.
///
/// Construct with [`XrSession::open`] when a runtime may be present, or with
/// [`XrSession::unattached`] when the game wants a session object it can hold
/// and drive regardless of whether XR is there.
///
/// The session is deliberately not a renderer: [`XrSession::frame`] hands the
/// runtime's images to a [`StereoSubmit`] implementation and returns only what
/// the runtime said. Drawing stays in the render layer.
///
/// Action sets are installed once with [`XrSession::set_action_sets`]; from
/// then on [`XrSession::frame`] syncs them automatically and
/// [`XrSession::action_states`] reports what the last sync produced. Input is
/// only ever reported as the runtime reported it: an action nothing is driving
/// is inactive at its type's zero value, and a hand the runtime is not tracking
/// has no pose.
pub struct XrSession {
    config: SessionConfig,
    space: ReferenceSpace,
    backend: backend::Backend,
}

impl XrSession {
    /// Probe for a usable runtime without creating a session.
    ///
    /// This is the check a game makes at startup: `Ok` means XR can be
    /// offered, [`XrError::Unavailable`] means run flat.
    pub fn probe() -> XrResult<()> {
        backend::Backend::probe()
    }

    /// Open a session on the first backend that answers.
    pub fn open(config: SessionConfig) -> XrResult<Self> {
        let backend = backend::Backend::open(&config)?;
        Ok(Self {
            config,
            space: ReferenceSpace::default(),
            backend,
        })
    }

    /// A session object with no runtime behind it.
    ///
    /// Every runtime call answers [`XrError::Unavailable`] and the reported
    /// state stays [`SessionState::Idle`], so callers get the typed answer
    /// instead of an option to unwrap.
    pub fn unattached(config: SessionConfig) -> Self {
        Self {
            config,
            space: ReferenceSpace::default(),
            backend: backend::Backend::unattached(),
        }
    }

    /// The configuration this session was created with.
    pub fn config(&self) -> &SessionConfig {
        &self.config
    }

    /// Name of the backend that would answer (`"none"` when runtime-less).
    pub fn backend(&self) -> &'static str {
        self.backend.name()
    }

    /// Runtime-reported session state.
    pub fn state(&self) -> SessionState {
        self.backend.state()
    }

    /// The reference space poses are expressed in.
    pub fn space(&self) -> ReferenceSpace {
        self.space
    }

    /// Change the reference space. Takes effect on the next frame.
    pub fn set_space(&mut self, space: ReferenceSpace) {
        self.space = space;
    }

    /// Ask the runtime to begin the session.
    ///
    /// Legal in [`SessionState::Ready`] and nothing else; a backend reports
    /// [`XrError::WrongState`] rather than letting the runtime refuse.
    pub fn begin(&mut self) -> XrResult<()> {
        self.backend.begin()
    }

    /// Ask the runtime to end the session.
    ///
    /// A runtime that has to stop asynchronously answers
    /// [`XrError::NotReady`] until it reaches the stopping state, so the call
    /// is made again on the next tick.
    pub fn end(&mut self) -> XrResult<()> {
        self.backend.end()
    }

    /// Latest poses for the roles this session tracks.
    ///
    /// The snapshot is the one the most recent frame located, so it is empty
    /// before the first frame; no backend invents a pose to fill it.
    ///
    /// Hands are reported once action sets are installed
    /// ([`XrSession::set_action_sets`]), because that is what gives a backend
    /// the pose actions it reads them through; which of the two hand poses the
    /// runtime reports is [`XrSession::set_hand_pose_source`]'s choice. A hand
    /// the runtime is not tracking stays [`TrackedPose::UNTRACKED`], never a
    /// stand-in pose.
    pub fn poses(&self) -> Poses {
        self.backend.poses()
    }

    /// Install the action sets this session binds against.
    ///
    /// Once per session: a second call answers
    /// [`XrError::ActionSetsInstalled`] rather than merging two installs, and
    /// the sets are attached to the runtime here. An empty iterator installs
    /// nothing an app can read ([`XrSession::action_states`] stays empty) but
    /// still counts as installed, which is what lets [`XrSession::poses`]
    /// report hand poses on a session with no gameplay actions of its own.
    ///
    /// The backend registers a binding for every profile the sets declare, and
    /// creates the hand pose actions it reads hands through. Nothing is
    /// fabricated on the way: an action whose input the runtime does not have
    /// right now is reported inactive.
    pub fn set_action_sets(&mut self, sets: impl IntoIterator<Item = ActionSet>) -> XrResult<()> {
        self.backend.set_action_sets(sets.into_iter().collect())
    }

    /// Which hand pose the runtime reports, [`HandPose::Aim`] by default.
    ///
    /// Call it before the first frame; a change takes effect at the next sync.
    /// Both poses stay bound at the runtime, and this picks which of them is
    /// read, so switching cannot fail and needs no rebinding.
    pub fn set_hand_pose_source(&mut self, pose: HandPose) {
        self.backend.set_hand_pose_source(pose);
    }

    /// Add interaction profiles whose bindings the runtime registers, beyond
    /// the ones the sets declare.
    ///
    /// This is for a runtime's binding UI: the more profiles it has bindings
    /// for, the more layouts it can offer the user. A profile with no bindings
    /// anywhere is registered with the hand pose bindings alone.
    ///
    /// Call it before [`XrSession::set_action_sets`], because attaching the
    /// sets at the runtime is what makes them immutable there; a profile added
    /// afterwards is only suggested where a runtime allows that.
    ///
    /// The WebXR backend has no such registration: the browser decides what a
    /// controller exposes and there is no profile to bind against, so it
    /// records nothing and this is only meaningful on OpenXR.
    pub fn set_fallback_profiles(
        &mut self,
        profiles: impl IntoIterator<Item = InteractionProfile>,
    ) {
        self.backend
            .set_fallback_profiles(profiles.into_iter().collect());
    }

    /// States produced by the last sync, one entry per declared action.
    ///
    /// Empty before the first frame, and for the rest of a session that
    /// installed no sets. An action the runtime is not driving is `active:
    /// false` at its type's zero value (`false`, `0.0`, or the zero vector):
    /// the spec leaves an inactive action's state undefined, so a stale or
    /// garbage reading is never passed on. One action bound on several devices
    /// is still one entry, combined by [`combine`]'s rule.
    pub fn action_states(&self) -> &[ActionState] {
        self.backend.action_states()
    }

    /// One action's state by set and action name, or
    /// [`XrError::UnknownAction`] when the last sync produced none for it: an
    /// unknown name, or a sync that has not happened yet.
    pub fn action(&self, set: &str, action: &str) -> XrResult<&ActionState> {
        self.backend.action(set, action)
    }

    /// The interaction profile the runtime reports for a device, if any.
    ///
    /// The OpenXR backend asks the runtime for the device's profile as of the
    /// call; the WebXR backend reports the profile of the input source that
    /// hand had in the last frame. A runtime that reports none (nothing
    /// connected, a device without a profile, or, on OpenXR, a session that has
    /// not installed its action sets yet) answers `None`, and no profile is
    /// inferred from binding paths.
    ///
    /// WebXR's input sources name their profiles the WebXR way
    /// (`"oculus-touch-v3"`), not as OpenXR interaction profile paths, and
    /// there is no faithful mapping between the two, so a WebXR hand answers
    /// `None` unless its runtime reports an `/interaction_profiles/...` path.
    pub fn interaction_profile(&self, user: &UserPath) -> Option<InteractionProfile> {
        self.backend.interaction_profile(user)
    }

    /// Drive one frame, handing each eye's images to `sink`.
    ///
    /// Returns what the runtime said about the frame, or the render layer's
    /// own error wrapped in [`FrameError::Sink`]. A frame the session is not
    /// presenting comes back with `rendered: false` rather than an error, and
    /// a runtime that wants a frame but will not show it comes back with
    /// `should_render: false`.
    ///
    /// Once action sets are installed, this is also where they are synced:
    /// the states [`XrSession::action_states`] reports are the ones the last
    /// frame produced.
    pub fn frame<S: StereoSubmit>(
        &mut self,
        sink: &mut S,
    ) -> Result<FrameOutcome, FrameError<S::Error>> {
        self.backend.frame(self.space, sink)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "webxr"))]
impl XrSession {
    /// Open a session against the WebGL2 context the render layer draws with.
    ///
    /// WebXR cannot create a session synchronously: this starts the runtime's
    /// `requestSession` and the session reaches [`SessionState::Ready`] when the
    /// runtime has answered, with the layer and the reference spaces set up. A
    /// runtime that refuses later surfaces as a typed error from the next call
    /// that returns one.
    ///
    /// The reference spaces this crate can ask for are requested up front, so
    /// [`XrSession::set_space`] needs no async step of its own.
    pub fn open_webgl(
        config: SessionConfig,
        context: web_sys::WebGl2RenderingContext,
    ) -> XrResult<Self> {
        let backend = webxr::WebXrBackend::open(&config, context)?;
        Ok(Self {
            config,
            space: ReferenceSpace::default(),
            backend: backend::Backend::WebXr(Box::new(backend)),
        })
    }

    /// Hand the crate the body of a WebXR animation frame.
    ///
    /// WebXR delivers an `XRFrame` only inside the runtime's animation frame
    /// callback, and the layer's framebuffer may only be drawn into there, so
    /// the callback is where a frame has to be drawn:
    ///
    /// ```ignore
    /// let session = Rc::new(RefCell::new(XrSession::open_webgl(config, context)?));
    /// let renderer = Rc::new(RefCell::new(sink));
    /// session.borrow_mut().set_animation_frame_callback({
    ///     // A weak handle: this callback lives in the session, so holding the
    ///     // session itself strongly here would keep it alive for good.
    ///     let session = Rc::downgrade(&session);
    ///     move |time_ms, frame| {
    ///         let Some(session) = session.upgrade() else { return };
    ///         let mut session = session.borrow_mut();
    ///         let outcome = session
    ///             .frame_with_xr_frame(time_ms, &frame, &mut *renderer.borrow_mut());
    ///         if let Err(error) = outcome {
    ///             log::error!("webxr: {error}");
    ///         }
    ///     }
    /// })?;
    /// session.borrow_mut().begin()?;
    /// ```
    ///
    /// The crate registers the callback with the runtime, asks for the next
    /// frame after every call and stops on [`XrSession::end`] or when the session
    /// is dropped, so the loop does not have to be re-armed by hand.
    pub fn set_animation_frame_callback<F>(&mut self, callback: F) -> XrResult<()>
    where
        F: FnMut(f64, XrFrame) + 'static,
    {
        match &mut self.backend {
            backend::Backend::WebXr(backend) => backend.set_frame_callback(callback),
            _ => Err(XrError::Unsupported {
                backend: self.backend.name(),
                detail: "an animation frame callback; this session has no WebXR runtime".to_string(),
            }),
        }
    }

    /// Drive one frame from the animation frame the runtime handed to
    /// [`XrSession::set_animation_frame_callback`].
    ///
    /// [`XrSession::frame`] answers [`XrError::NotReady`] on WebXR: there is no
    /// frame to drive outside the callback the runtime delivered one in. Input
    /// is read here, from the session's input sources, and that is what makes
    /// [`XrSession::action_states`] and the hand poses current.
    pub fn frame_with_xr_frame<S: StereoSubmit>(
        &mut self,
        time_ms: f64,
        frame: &XrFrame,
        sink: &mut S,
    ) -> Result<FrameOutcome, FrameError<S::Error>> {
        match &mut self.backend {
            backend::Backend::WebXr(backend) => {
                backend.frame_with_xr_frame(self.space, time_ms, frame, sink)
            }
            _ => Err(FrameError::Xr(XrError::Unsupported {
                backend: self.backend.name(),
                detail: "a WebXR frame; this session has no WebXR runtime".to_string(),
            })),
        }
    }

    /// The framebuffer the runtime composites this session's layer into.
    ///
    /// WebGL addresses resources as objects rather than as integer names, so a
    /// render layer takes the framebuffer from here; the
    /// [`EyeImage::handle`] it is handed alongside carries an opaque token.
    pub fn webgl_framebuffer(&self) -> Option<web_sys::WebGlFramebuffer> {
        match &self.backend {
            backend::Backend::WebXr(backend) => backend.framebuffer(),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SessionConfig {
        SessionConfig::default()
    }

    #[test]
    fn probe_reports_unavailable_on_a_runtime_less_build() {
        let error = XrSession::probe().expect_err("no runtime backend is compiled in");
        assert!(error.is_unavailable(), "got {error:?}");
    }

    #[test]
    fn open_reports_unavailable_rather_than_a_half_session() {
        let error = XrSession::open(config())
            .err()
            .expect("no runtime backend is compiled in");
        assert!(error.is_unavailable(), "got {error:?}");
    }

    #[test]
    fn an_unattached_session_answers_every_runtime_call_with_a_typed_error() {
        let mut session = XrSession::unattached(config());
        assert_eq!(session.state(), SessionState::Idle);
        assert_eq!(session.backend(), NO_BACKEND_NAME);
        assert!(session.begin().unwrap_err().is_unavailable());
        assert!(session.end().unwrap_err().is_unavailable());

        struct Sink;
        impl StereoSubmit for Sink {
            type Error = core::convert::Infallible;
            fn submit_eye(
                &mut self,
                _image: &EyeImage,
                _view: &ViewSetup,
            ) -> Result<(), Self::Error> {
                Ok(())
            }
        }
        let mut sink = Sink;
        let error = session.frame(&mut sink).expect_err("no runtime");
        assert!(matches!(error, FrameError::Xr(e) if e.is_unavailable()));
    }

    #[test]
    fn the_reference_space_is_settable_and_reported() {
        let mut session = XrSession::unattached(config());
        assert_eq!(session.space(), ReferenceSpace::View);
        session.set_space(ReferenceSpace::Stage);
        assert_eq!(session.space(), ReferenceSpace::Stage);
    }

    #[test]
    fn an_unattached_session_tracks_nothing_rather_than_faking_a_pose() {
        let session = XrSession::unattached(config());
        assert!(session.poses().head().pose.is_none());
    }

    #[test]
    fn an_unattached_session_has_no_actions_and_says_so() {
        let mut session = XrSession::unattached(config());
        assert!(session.action_states().is_empty());
        assert!(matches!(
            session.action("gameplay", "jump"),
            Err(XrError::UnknownAction { .. })
        ));

        // Installing sets is a runtime call, so it is the typed error rather
        // than a local half-install.
        let sets = [ActionSet::new("gameplay", "Gameplay", 0)];
        assert!(session.set_action_sets(sets).unwrap_err().is_unavailable());

        // Preferences are local, and a session that tracks nothing still
        // accepts them without pretending they took effect.
        session.set_hand_pose_source(HandPose::Grip);
        session.set_fallback_profiles([InteractionProfile::new("oculus", "touch_controller")]);
        assert!(session.action_states().is_empty());
        assert!(session.interaction_profile(&UserPath::Hand(Hand::Left)).is_none());
    }
}
