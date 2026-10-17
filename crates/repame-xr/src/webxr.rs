//! The WebXR backend: the browser's own runtime, wasm32 only.
//!
//! WebXR is shaped differently from OpenXR in four ways, and each one shows up
//! in the design:
//!
//! - **Everything is asynchronous.** A session comes from `requestSession`, a
//!   `Promise`, and so do reference spaces. [`crate::XrSession::open_webgl`]
//!   starts the request and the session reaches [`SessionState::Ready`] when the
//!   runtime has answered; a refusal is a typed error from the next call that
//!   returns one.
//! - **A frame only exists inside the runtime's callback.** `XRFrame` is invalid
//!   outside the animation frame callback that delivered it and the runtime's
//!   framebuffer may only be drawn into there, so the crate owns the callback and
//!   calls the app's body from inside it
//!   ([`crate::XrSession::set_animation_frame_callback`],
//!   [`crate::XrSession::frame_with_xr_frame`]).
//! - **The runtime composites its own framebuffer.** There is no swapchain to
//!   acquire: the render layer binds the `XRWebGLLayer` framebuffer, which WebGL
//!   addresses as an object rather than as an integer name, so
//!   [`crate::XrSession::webgl_framebuffer`] hands it over and
//!   [`EyeImage::handle`] carries an opaque token beside it.
//! - **There is no session state machine of its own.** WebXR sessions are either
//!   `visible`, `visible-blurred` or `hidden`; those become
//!   [`SessionState::Focused`], [`SessionState::Visible`] and
//!   [`SessionState::Synchronized`], and `begin`/`end` are this crate's own
//!   boundary (the runtime's `end` event is what makes the session
//!   [`SessionState::Idle`]).
//!
//! WebXR hands over a finished projection matrix per eye in the OpenGL `[-1, 1]`
//! depth convention, which [`crate::stereo`] deliberately does not forward to
//! the renderer: the fov is recovered from it with
//! [`crate::Fov::from_gl_projection`] and the crate's own `[0, 1]`-depth
//! projection is rebuilt from that, exactly as it is for OpenXR. A matrix in the
//! wrong depth range is the classic half-depth-range stereo bug.
//!
//! web-sys' own WebXR interfaces sit behind `--cfg=web_sys_unstable_apis`, a
//! flag a crate cannot set from its own manifest, so the interface is declared
//! here with `wasm-bindgen` — the same generator web-sys uses — and only the
//! stable WebGL2 types are taken from web-sys.
//!
//! Input comes from the session's input sources, read once per frame. WebXR
//! has no interaction profiles to bind actions against, so an action's own
//! binding paths say what feeds it: each hand's input source is read through
//! [`crate::gamepad`], which maps only the controls the `xr-standard` gamepad
//! mapping defines. An input source with another mapping, a gamepad shorter
//! than the control an action names, a hand without a gamepad, or a path the
//! standard mapping does not carry leaves the actions bound to it inactive,
//! and nothing here guesses an index for them. An action bound on both hands is
//! read once per hand and combined the same way the OpenXR backend combines
//! per-device readings.
//!
//! Hand poses come from the same input sources: `targetRaySpace` for
//! [`HandPose::Aim`] and `gripSpace` for [`HandPose::Grip`], whichever
//! [`crate::XrSession::set_hand_pose_source`] asked for. A hand has a pose
//! only when the runtime reports one for that space this frame and says it is
//! not merely emulated; nothing here fabricates one.
//!
//! An app's interaction profiles are not registered anywhere: the browser, not
//! the app, decides what its controllers expose, and a session that reports
//! them as WebXR profile ids (`"oculus-touch-v3"`) names no OpenXR profile to
//! report back.

use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

use glam::Mat4;
use wasm_bindgen::JsCast;
use wasm_bindgen::JsValue;
use wasm_bindgen::closure::Closure;
use web_sys::{WebGl2RenderingContext, WebGlFramebuffer};

use crate::action::{
    ActionKind, ActionReading, ActionSet, ActionState, InteractionProfile, UserPath, combine,
};
use crate::error::{XrError, XrResult};
use crate::gamepad::{GamepadButton, GamepadElement, element_reading, gamepad_element};
use crate::pose::{Hand, HandPose, Pose, Poses, TrackedPose, TrackedRole};
use crate::session::{SessionConfig, SessionState};
use crate::space::ReferenceSpace;
use crate::stereo::{BlendMode, Eye, Fov, PixelFormat, ViewConfig, ViewConfiguration, ViewSetup};
use crate::swapchain::{
    EyeImage, FrameError, FrameOutcome, GraphicsApi, StereoFrame, StereoSubmit,
    SwapchainImageHandle,
};

pub use js::XrFrame;

/// Backend name reported for WebXR.
pub(crate) const NAME: &str = "webxr";

/// Token an [`EyeImage`] carries for the layer's one framebuffer. WebGL names
/// its resources as objects rather than as integers, so the render layer takes
/// the framebuffer itself from [`WebXrBackend::framebuffer`]; this token only
/// identifies it, and is the same for every eye.
const FRAMEBUFFER_TOKEN: u64 = 1;

/// WebGL framebuffers are single-sample: antialiasing in WebGL is a property of
/// the drawing buffer, not of the framebuffer a render pass targets.
const FRAMEBUFFER_SAMPLES: u32 = 1;

/// Probe for a browser that has WebXR at all.
///
/// Whether a device is attached is only answerable through
/// `XRSystem.isSessionSupported`, a promise this synchronous call cannot await;
/// the answer arrives from [`WebXrBackend::open`]'s request instead.
pub(crate) fn probe() -> XrResult<()> {
    if xr_system().is_none() {
        return Err(XrError::Unavailable {
            backend: NAME,
            detail: "no WebXR in this browser (navigator.xr is missing)".to_string(),
        });
    }
    Ok(())
}

/// The error [`crate::XrSession::open`] answers with on a WebXR build.
pub(crate) fn needs_context_error() -> XrError {
    XrError::Unsupported {
        backend: NAME,
        detail: "opening from configuration alone; a WebXR session is created against the render \
                 layer's WebGL2 context, so use XrSession::open_webgl"
            .to_string(),
    }
}

/// A real WebXR session, and everything one of its frames needs.
pub(crate) struct WebXrBackend {
    shared: Rc<RefCell<Shared>>,
    view_configuration: ViewConfiguration,
    blend: BlendMode,
    near: f32,
    far: f32,
    /// Pixel format of the layer's framebuffer: 8-bit RGBA, which the XR
    /// compositor treats as sRGB-encoded (WebXR's opaque framebuffer).
    format: PixelFormat,
    poses: Cell<Poses>,
    display_time_ns: Cell<i64>,
    state: Cell<SessionState>,
    /// The app's action sets, compiled into what each frame reads.
    installed: Option<Planned>,
    /// Which of the two hand poses a frame reads.
    hand_pose: HandPose,
    /// States the last frame's sync produced, in declared-action order.
    states: Vec<ActionState>,
    /// The interaction profiles the input sources of the last frame named, per
    /// hand. Empty for a runtime that names none the way OpenXR does.
    profiles: Vec<(Hand, InteractionProfile)>,
}

impl WebXrBackend {
    /// Start a session against the render layer's context.
    ///
    /// Everything after this is asynchronous: the return value is a session that
    /// is not running yet, and it reports [`SessionState::Idle`] until the
    /// runtime has answered.
    pub(crate) fn open(config: &SessionConfig, context: WebGl2RenderingContext) -> XrResult<Self> {
        if let Some(binding) = config.graphics {
            return Err(XrError::Unsupported {
                backend: NAME,
                detail: format!(
                    "a {} graphics binding; a WebXR session is created against the render layer's \
                     WebGL2 context",
                    binding.api().name()
                ),
            });
        }
        let system = xr_system().ok_or_else(|| XrError::Unavailable {
            backend: NAME,
            detail: "no WebXR in this browser (navigator.xr is missing)".to_string(),
        })?;
        let mode = mode_of(config.view_configuration);
        let blend = blend_of(mode);
        if !config.blend_modes.contains(&blend) {
            log::warn!(
                "webxr: a {mode} session composites with {blend:?}, which is not in \
                 SessionConfig::blend_modes; the session mode decides this, not the app"
            );
        }
        let shared = Rc::new(RefCell::new(Shared::new(context, layer_alpha(mode))));
        request_session(&system, mode, Rc::downgrade(&shared));
        Ok(Self {
            shared,
            view_configuration: config.view_configuration,
            blend,
            near: config.near,
            far: config.far,
            format: PixelFormat::Rgba8UnormSrgb,
            poses: Cell::new(Poses::new()),
            display_time_ns: Cell::new(0),
            state: Cell::new(SessionState::Idle),
            installed: None,
            hand_pose: HandPose::default(),
            states: Vec::new(),
            profiles: Vec::new(),
        })
    }

    /// Runtime-reported session state.
    pub(crate) fn state(&self) -> SessionState {
        self.refresh();
        self.state.get()
    }

    /// Start the runtime's animation frame loop. There is nothing for the loop
    /// to call without a frame body, so one has to be set first.
    pub(crate) fn begin(&mut self) -> XrResult<()> {
        self.refresh();
        self.check_error()?;
        let state = self.state.get();
        if !state.can_begin() {
            return Err(XrError::WrongState {
                expected: "Ready",
                actual: state,
            });
        }
        if self.shared.borrow().frame_callback.is_none() {
            return Err(XrError::NotReady {
                what: "an animation frame callback",
            });
        }
        self.install_loop();
        self.shared.borrow_mut().presenting = true;
        request_next_frame(&self.shared);
        self.refresh();
        Ok(())
    }

    /// Ask the runtime to end the session, and stop the frame loop.
    ///
    /// `XRSession.end()` answers asynchronously; the runtime's `end` event is
    /// what moves the session to [`SessionState::Idle`], and until it arrives the
    /// state is [`SessionState::Stopping`], which stops frames. Ending before the
    /// session request has resolved ends the session as soon as it arrives.
    pub(crate) fn end(&mut self) -> XrResult<()> {
        self.refresh();
        self.check_error()?;
        let session = {
            let mut shared = self.shared.borrow_mut();
            if shared.ending || shared.ended {
                // Already asked for, or already over.
                return Ok(());
            }
            shared.presenting = false;
            shared.ending = true;
            shared.cancel_frame();
            shared.session.clone()
        };
        if let Some(session) = session {
            catch_log(session.end(), "ending the session");
        }
        self.state.set(SessionState::Stopping);
        Ok(())
    }

    /// Poses of the most recent frame. The head comes from the runtime's viewer
    /// pose; hands come from the input sources once action sets are installed.
    pub(crate) fn poses(&self) -> Poses {
        self.poses.get()
    }

    /// Install the app's action sets: compile their bindings into what each
    /// frame reads off the input sources.
    ///
    /// Once per session; a second call answers
    /// [`XrError::ActionSetsInstalled`]. There is nothing runtime-side to
    /// attach, since a WebXR session has input sources rather than action
    /// sets, so this is this crate's own read plan, and the frame loop is what
    /// runs it.
    pub(crate) fn set_action_sets(&mut self, sets: Vec<ActionSet>) -> XrResult<()> {
        if self.installed.is_some() {
            return Err(XrError::ActionSetsInstalled);
        }
        self.installed = Some(Planned {
            entries: plan(&sets),
        });
        self.states.clear();
        Ok(())
    }

    /// Which hand pose a frame reads. Both spaces are there for every
    /// controller, so this only picks one.
    pub(crate) fn set_hand_pose_source(&mut self, pose: HandPose) {
        self.hand_pose = pose;
    }

    /// WebXR has no interaction profiles to bind against and no binding UI to
    /// register them with: the browser decides what a controller exposes, and
    /// this backend reads whatever its input sources have. The call is
    /// accepted, and there is nothing it could register, since the profiles
    /// that matter to OpenXR's binding UI have no WebXR counterpart.
    pub(crate) fn set_fallback_profiles(&mut self, _profiles: Vec<InteractionProfile>) {}

    /// States the last frame's sync produced, in declared-action order.
    pub(crate) fn action_states(&self) -> &[ActionState] {
        &self.states
    }

    /// One action's state by set and action name.
    pub(crate) fn action(&self, set: &str, action: &str) -> XrResult<&ActionState> {
        crate::find_action(&self.states, set, action)
    }

    /// The profile the input source of a hand named in the last frame, when it
    /// named one the way OpenXR does. WebXR profile ids are not interaction
    /// profile paths, and no mapping between the two is invented here.
    pub(crate) fn interaction_profile(&self, user: &UserPath) -> Option<InteractionProfile> {
        let UserPath::Hand(hand) = user else {
            return None;
        };
        self.profiles
            .iter()
            .find(|(reported, _)| reported == hand)
            .map(|(_, profile)| profile.clone())
    }

    /// The framebuffer the runtime composites this session's layer into.
    pub(crate) fn framebuffer(&self) -> Option<WebGlFramebuffer> {
        self.shared.borrow().framebuffer.clone()
    }

    /// Store the body the animation frame loop calls.
    pub(crate) fn set_frame_callback<F>(&mut self, callback: F) -> XrResult<()>
    where
        F: FnMut(f64, XrFrame) + 'static,
    {
        self.shared.borrow_mut().frame_callback = Some(Box::new(callback));
        Ok(())
    }

    /// There is never a frame to drive outside the runtime's callback: WebXR
    /// hands an `XRFrame` only to the animation frame callback, and the layer's
    /// framebuffer may only be drawn into there.
    /// [`crate::XrSession::frame_with_xr_frame`] is the WebXR way into a frame.
    pub(crate) fn frame<S: StereoSubmit>(
        &mut self,
        _space: ReferenceSpace,
        _sink: &mut S,
    ) -> Result<FrameOutcome, FrameError<S::Error>> {
        Err(FrameError::Xr(XrError::NotReady {
            what: "a WebXR animation frame",
        }))
    }

    /// Drive one frame from the animation frame the runtime handed to the
    /// callback.
    pub(crate) fn frame_with_xr_frame<S: StereoSubmit>(
        &mut self,
        space: ReferenceSpace,
        time_ms: f64,
        frame: &XrFrame,
        sink: &mut S,
    ) -> Result<FrameOutcome, FrameError<S::Error>> {
        self.refresh();
        self.check_error()?;
        let state = self.state.get();
        if !state.should_submit_frames() {
            log::debug!("webxr: no frame to render in state {state:?}");
            return Ok(FrameOutcome {
                rendered: false,
                should_render: false,
                display_time_ns: self.display_time_ns.get(),
                views: 0,
            });
        }
        let display_time_ns = nanos_of(time_ms);
        self.display_time_ns.set(display_time_ns);

        let (reference, layer, size) = {
            let shared = self.shared.borrow();
            let Some(reference) = shared.spaces[space_index(space)].clone() else {
                let reason = match &shared.missing_space {
                    Some((missing, reason)) if *missing == space => reason.clone(),
                    _ => "the runtime has not answered for it".to_string(),
                };
                return Err(FrameError::Xr(XrError::Unsupported {
                    backend: NAME,
                    detail: format!("the {} reference space: {reason}", space.webxr_name()),
                }));
            };
            let Some(layer) = shared.layer.clone() else {
                return Err(FrameError::Xr(XrError::NotReady { what: "a WebXR layer" }));
            };
            let size = [layer.framebuffer_width(), layer.framebuffer_height()];
            (reference, layer, size)
        };

        // Input is read for every frame, whether or not the runtime has a
        // viewer pose: an action's state belongs to the frame it was read in,
        // and an untracked head says nothing about the hands.
        self.sync_inputs(frame, &reference);

        let viewer = match viewer_pose(frame, &reference) {
            Ok(viewer) => viewer,
            Err(error) => {
                self.clear_poses();
                return Err(FrameError::Xr(error));
            }
        };
        match &viewer {
            Some(viewer) => {
                let emulated = viewer.emulated_position();
                let head = pose_of(&viewer.transform());
                self.set_pose(
                    TrackedRole::Head,
                    TrackedPose::from_tracking_flags(head, true, !emulated),
                );
            }
            None => {
                log::debug!("webxr: the runtime has no viewer pose for this frame");
                self.set_pose(TrackedRole::Head, TrackedPose::UNTRACKED);
            }
        }
        let Some(viewer) = viewer else {
            // WebXR reports no viewer pose when the device is not tracked this
            // frame; there is nothing to draw and nothing is invented in its
            // place.
            return Ok(FrameOutcome {
                rendered: false,
                should_render: false,
                display_time_ns,
                views: 0,
            });
        };

        let views = viewer.views();
        let mut setups = Vec::with_capacity(views.length() as usize);
        let mut images = Vec::with_capacity(views.length() as usize);
        for index in 0..views.length() {
            let Some(view) = views
                .get_checked(index)
                .and_then(|value| value.dyn_into::<js::XrView>().ok())
            else {
                continue;
            };
            let eye = eye_of(&view.eye(), index as usize, self.view_configuration);
            let projection = projection_of(&view).map_err(FrameError::Xr)?;
            let fov = Fov::from_gl_projection(projection).ok_or(FrameError::Xr(
                XrError::InvalidData {
                    what: "view projection matrix",
                },
            ))?;
            let viewport = layer
                .get_viewport(&view)
                .ok_or(FrameError::Xr(XrError::InvalidData {
                    what: "view viewport",
                }))?;
            let viewport = [
                viewport.x().max(0) as u32,
                viewport.y().max(0) as u32,
                viewport.width().max(0) as u32,
                viewport.height().max(0) as u32,
            ];
            let mut setup = ViewSetup::from_pose_fov(
                eye,
                Some(pose_of(&view.transform())),
                fov,
                self.near,
                self.far,
            );
            setup.viewport = Some(viewport);
            setups.push(setup);
            images.push(EyeImage {
                eye,
                size,
                format: self.format,
                sample_count: FRAMEBUFFER_SAMPLES,
                viewport,
                handle: SwapchainImageHandle {
                    api: GraphicsApi::WebGl,
                    handle: FRAMEBUFFER_TOKEN,
                    array_index: index,
                },
                swapchain: FRAMEBUFFER_TOKEN,
                image_index: 0,
            });
        }
        let expected = self.view_configuration.view_count();
        if setups.len() != expected {
            return Err(FrameError::Xr(XrError::InvalidData {
                what: "view count",
            }));
        }
        if expected > 1 && setups[0].eye == setups[1].eye {
            // Two views for one eye would draw the same eye twice; the runtime
            // does not get to name them both the same.
            return Err(FrameError::Xr(XrError::InvalidData {
                what: "view eyes",
            }));
        }

        let config = ViewConfig {
            views: setups,
            recommended_size: size,
            near: self.near,
            far: self.far,
            blend_mode: self.blend,
        };
        StereoFrame {
            config: &config,
            images: &images,
        }
        .submit(sink)
        .map_err(FrameError::Sink)?;

        Ok(FrameOutcome {
            rendered: true,
            should_render: true,
            display_time_ns,
            views: config.views.len(),
        })
    }

    /// Where the session stands, read from the runtime and the asynchronous
    /// steps that are still in flight.
    fn refresh(&self) {
        let mut state = SessionState::Idle;
        {
            let shared = self.shared.borrow();
            if shared.error.is_none()
                && !shared.ended
                && shared.session.is_some()
                && shared.layer.is_some()
            {
                state = if shared.ending {
                    SessionState::Stopping
                } else if shared.presenting {
                    visibility_state(shared.session.as_ref(), self.state.get())
                } else {
                    SessionState::Ready
                };
            }
        }
        self.state.set(state);
    }

    /// The first failure of an asynchronous step, reported by every later call.
    fn check_error(&self) -> XrResult<()> {
        match &self.shared.borrow().error {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }

    /// Register the loop's own callback, which the runtime re-enters once per
    /// animation frame.
    fn install_loop(&self) {
        let mut shared = self.shared.borrow_mut();
        if shared.loop_callback.is_some() {
            return;
        }
        let weak = Rc::downgrade(&self.shared);
        let callback = Closure::wrap_assert_unwind_safe(
            Box::new(move |time: f64, frame: JsValue| animation_frame(&weak, time, frame))
                as Box<dyn FnMut(f64, JsValue)>,
        );
        shared.loop_callback = Some(callback);
    }

    /// Read the session's input sources for this frame: the states the action
    /// sets produce, the hand poses, and the profiles the runtime reports.
    ///
    /// Input sources are only valid inside the frame callback they were handed
    /// to, so everything is read into owned values here and nothing JS-shaped
    /// is kept.
    fn sync_inputs(&mut self, frame: &XrFrame, reference: &js::XrReferenceSpace) {
        let session = self.shared.borrow().session.clone();
        let (sources, profiles) = match &session {
            Some(session) => read_sources(session, frame, reference),
            None => (Vec::new(), Vec::new()),
        };
        self.profiles = profiles;

        let mut states = Vec::new();
        if let Some(installed) = &self.installed {
            states.reserve(installed.entries.len());
            for entry in &installed.entries {
                states.push(entry_state(entry, &sources));
            }
            for hand in Hand::ALL {
                let pose = match sources.iter().find(|source| source.hand == hand) {
                    Some(source) => match self.hand_pose {
                        HandPose::Aim => source.aim,
                        HandPose::Grip => source.grip,
                    },
                    None => TrackedPose::UNTRACKED,
                };
                self.set_pose(TrackedRole::Hand(hand), pose);
            }
        }
        self.states = states;
    }

    /// Record one role's pose, leaving the others as the frame left them.
    fn set_pose(&self, role: TrackedRole, pose: TrackedPose) {
        let mut poses = self.poses.get();
        poses.set(role, pose);
        self.poses.set(poses);
    }

    /// Forget every pose: a frame the runtime could not locate must not leave
    /// the previous ones looking current.
    fn clear_poses(&self) {
        self.poses.set(Poses::new());
    }
}

impl Drop for WebXrBackend {
    fn drop(&mut self) {
        let mut shared = self.shared.borrow_mut();
        shared.presenting = false;
        shared.cancel_frame();
        if let Some(session) = shared.session.as_ref() {
            session.set_onend(None);
            // Dropping the handle must not leave the user in a live immersive
            // session: end it, the way dropping an OpenXR session destroys it.
            if !shared.ended && !shared.ending {
                catch_log(session.end(), "ending a dropped session");
            }
        }
        shared.loop_callback = None;
        shared.frame_callback = None;
    }
}

/// The app's action sets, compiled into what each frame reads.
struct Planned {
    /// One entry per declared action, in declaration order.
    entries: Vec<PlanEntry>,
}

/// One declared action and where each hand's reading of it comes from.
struct PlanEntry {
    set: String,
    action: String,
    kind: ActionKind,
    /// One entry per hand and control the action's bindings name, in binding
    /// declaration order. A binding the xr-standard mapping does not carry
    /// contributes nothing.
    targets: Vec<(Hand, GamepadElement)>,
}

/// One input source, read for one frame.
struct InputSource {
    hand: Hand,
    /// The source's gamepad, when it has one this crate can read.
    gamepad: Option<GamepadReading>,
    /// Where the source points this frame.
    aim: TrackedPose,
    /// Where the source is held this frame.
    grip: TrackedPose,
}

/// A gamepad's controls, read for one frame.
struct GamepadReading {
    buttons: Vec<GamepadButton>,
    axes: Vec<f32>,
}

/// Everything the runtime's callbacks and the backend both touch.
///
/// The callbacks hold a [`Weak`] to this, so a promise that settles after the
/// session is gone finds nothing to do instead of aborting on a freed closure.
struct Shared {
    /// The render layer's context, until the layer is created.
    context: Option<WebGl2RenderingContext>,
    /// Layer options, built once at open.
    layer_init: JsValue,
    /// The runtime's session, once the request resolved.
    session: Option<js::XrSession>,
    /// The layer the session composites.
    layer: Option<js::XrWebGlLayer>,
    /// The framebuffer the layer composites into.
    framebuffer: Option<WebGlFramebuffer>,
    /// Reference spaces, by [`space_index`].
    spaces: [Option<js::XrReferenceSpace>; 3],
    /// A reference space the runtime refused, with its reason.
    missing_space: Option<(ReferenceSpace, String)>,
    /// The first failure of an asynchronous step.
    error: Option<XrError>,
    /// The app's animation frame body.
    frame_callback: Option<Box<dyn FnMut(f64, js::XrFrame)>>,
    /// The loop's callback, kept so it can be re-registered and cancelled.
    loop_callback: Option<Closure<dyn FnMut(f64, JsValue)>>,
    /// The animation frame waiting to be delivered.
    animation_frame: Option<u32>,
    /// The app asked for the session to be shown.
    presenting: bool,
    /// The app asked for the session to end.
    ending: bool,
    /// The runtime ended the session.
    ended: bool,
}

impl Shared {
    fn new(context: WebGl2RenderingContext, alpha: bool) -> Self {
        Self {
            layer_init: layer_init(alpha),
            context: Some(context),
            session: None,
            layer: None,
            framebuffer: None,
            spaces: [None, None, None],
            missing_space: None,
            error: None,
            frame_callback: None,
            loop_callback: None,
            animation_frame: None,
            presenting: false,
            ending: false,
            ended: false,
        }
    }

    /// Stop the pending animation frame, if there is one.
    fn cancel_frame(&mut self) {
        if let (Some(handle), Some(session)) = (self.animation_frame.take(), self.session.as_ref())
        {
            session.cancel_animation_frame(handle);
        }
    }

    fn fail(&mut self, error: XrError) {
        if self.error.is_none() {
            log::warn!("webxr: {error}");
            self.error = Some(error);
        }
    }
}

/// Ask the runtime for a session, and set everything up when it answers.
fn request_session(system: &js::XrSystem, mode: &'static str, shared: Weak<RefCell<Shared>>) {
    let options = session_options();
    let promise = system.request_session(mode, options.as_ref());
    then2(
        &promise,
        {
            let shared = shared.clone();
            move |value| on_session(&shared, value)
        },
        move |error| on_async_failure(&shared, "requestSession", &error),
    );
}

/// The session arrived: create the layer over the render layer's context, set it
/// as the base layer, and ask for the reference spaces.
fn on_session(shared: &Weak<RefCell<Shared>>, value: JsValue) {
    let Some(shared) = shared.upgrade() else {
        return;
    };
    let Ok(session) = value.dyn_into::<js::XrSession>() else {
        shared.borrow_mut().fail(XrError::InvalidData {
            what: "WebXR session",
        });
        return;
    };
    let (context, init) = {
        let state = shared.borrow();
        (state.context.clone(), state.layer_init.clone())
    };
    let Some(context) = context else {
        shared.borrow_mut().fail(XrError::InvalidData {
            what: "WebXR layer context",
        });
        return;
    };
    let layer = match js::XrWebGlLayer::new(&session, context.as_ref(), &init) {
        Ok(layer) => layer,
        Err(error) => {
            shared.borrow_mut().fail(XrError::Unsupported {
                backend: NAME,
                detail: format!(
                    "a WebGL2 layer over this context (is it an XR compatible WebGL2 context?): {}",
                    error_text(&error)
                ),
            });
            return;
        }
    };
    let framebuffer = layer.framebuffer();
    let render_state = render_state(&layer);
    session.update_render_state(render_state.as_ref());

    // The runtime's own end, however it happens, is the one state change this
    // backend has to hear about.
    let onend = Closure::wrap_assert_unwind_safe({
        let shared = Rc::downgrade(&shared);
        Box::new(move |_event: JsValue| on_end(&shared)) as Box<dyn FnMut(JsValue)>
    });
    session.set_onend(Some(onend.as_ref().unchecked_ref()));
    onend.forget();

    for space in ReferenceSpace::ALL {
        request_reference_space(&session, space, Rc::downgrade(&shared));
    }

    let mut state = shared.borrow_mut();
    state.framebuffer = framebuffer;
    state.layer = Some(layer);
    state.context = None;
    state.session = Some(session.clone());
    let ended_early = state.ending;
    drop(state);
    if ended_early {
        // The app asked for the session to end before the runtime had answered.
        catch_log(session.end(), "ending the session");
    }
}

/// The runtime refused an asynchronous step.
fn on_async_failure(shared: &Weak<RefCell<Shared>>, what: &'static str, error: &JsValue) {
    let Some(shared) = shared.upgrade() else {
        return;
    };
    shared.borrow_mut().fail(XrError::Runtime {
        backend: NAME,
        code: js_error_code(error),
        detail: format!("{what}: {}", error_text(error)),
    });
}

/// The runtime ended the session, whether the app asked or the user did.
fn on_end(shared: &Weak<RefCell<Shared>>) {
    let Some(shared) = shared.upgrade() else {
        return;
    };
    log::info!("webxr: the runtime ended the session");
    let mut state = shared.borrow_mut();
    state.ended = true;
    state.ending = false;
    state.presenting = false;
    state.animation_frame = None;
    state.loop_callback = None;
    state.frame_callback = None;
}

/// Ask for one reference space and remember it, or remember why it is missing.
fn request_reference_space(
    session: &js::XrSession,
    space: ReferenceSpace,
    shared: Weak<RefCell<Shared>>,
) {
    let promise = session.request_reference_space(space.webxr_name());
    then2(
        &promise,
        {
            let shared = shared.clone();
            move |value| match value.dyn_into::<js::XrReferenceSpace>() {
                Ok(reference) => {
                    if let Some(shared) = shared.upgrade() {
                        shared.borrow_mut().spaces[space_index(space)] = Some(reference);
                    }
                }
                Err(value) => on_reference_space_failure(&shared, space, &value),
            }
        },
        move |error| on_reference_space_failure(&shared, space, &error),
    );
}

/// A refused reference space is not fatal: only the app that asks to be placed
/// in it hears about it.
fn on_reference_space_failure(shared: &Weak<RefCell<Shared>>, space: ReferenceSpace, error: &JsValue) {
    let Some(shared) = shared.upgrade() else {
        return;
    };
    shared.borrow_mut().missing_space = Some((space, error_text(error)));
}

/// The runtime's animation frame: call the app's body, then ask for the next
/// one.
fn animation_frame(shared: &Weak<RefCell<Shared>>, time: f64, frame: JsValue) {
    let Some(shared) = shared.upgrade() else {
        return;
    };
    // The body is taken out while it runs: it drives the session, which borrows
    // the same state.
    let Some(mut callback) = shared.borrow_mut().frame_callback.take() else {
        log::warn!("webxr: an animation frame arrived with no callback to call");
        return;
    };
    callback(time, frame.unchecked_into::<XrFrame>());
    shared.borrow_mut().frame_callback = Some(callback);
    request_next_frame(&shared);
}

/// Re-register the loop's callback, unless the session stopped.
fn request_next_frame(shared: &Rc<RefCell<Shared>>) {
    let state = shared.borrow();
    if !state.presenting || state.ended {
        return;
    }
    let (Some(session), Some(callback)) = (state.session.as_ref(), state.loop_callback.as_ref())
    else {
        return;
    };
    let function: &js_sys::Function = callback.as_ref().unchecked_ref();
    let handle = session.request_animation_frame(function);
    drop(state);
    shared.borrow_mut().animation_frame = Some(handle);
}

/// Map the runtime's visibility onto this crate's states: a session that is
/// shown and receiving input is `Focused`, one that is shown without input is
/// `Visible`, and one the user cannot see is running but not visible.
fn visibility_state(session: Option<&js::XrSession>, previous: SessionState) -> SessionState {
    let Some(session) = session else {
        return SessionState::Ready;
    };
    match session.visibility_state().as_str() {
        "visible" => SessionState::Focused,
        "visible-blurred" => SessionState::Visible,
        "hidden" => SessionState::Synchronized,
        other => {
            log::warn!("webxr: unknown visibility state {other}; keeping {previous:?}");
            previous
        }
    }
}

/// The viewer pose for the app's reference space, if the runtime has one.
fn viewer_pose(
    frame: &XrFrame,
    reference: &js::XrReferenceSpace,
) -> XrResult<Option<js::XrViewerPose>> {
    frame.get_viewer_pose(reference).map_err(|error| XrError::Runtime {
        backend: NAME,
        code: js_error_code(&error),
        detail: format!("getViewerPose: {}", error_text(&error)),
    })
}

/// A pose from the runtime's rigid transform.
fn pose_of(transform: &js::XrRigidTransform) -> Pose {
    let position = transform.position();
    let orientation = transform.orientation();
    Pose::from_xyzw(
        [
            position.x() as f32,
            position.y() as f32,
            position.z() as f32,
        ],
        [
            orientation.x() as f32,
            orientation.y() as f32,
            orientation.z() as f32,
            orientation.w() as f32,
        ],
    )
}

/// Compile the declared bindings into what each frame reads.
///
/// WebXR routes nothing through interaction profiles: a binding's own path says
/// which input on which hand feeds an action, and that is what is read off the
/// input source of that hand. A binding the xr-standard mapping does not carry
/// contributes no target, so the action it feeds stays inactive rather than
/// reading a guessed control.
fn plan(sets: &[ActionSet]) -> Vec<PlanEntry> {
    let mut entries = Vec::new();
    for set in sets {
        for action in set.actions() {
            let mut targets: Vec<(Hand, GamepadElement)> = Vec::new();
            for binding in set
                .bindings()
                .iter()
                .filter(|binding| binding.action == action.name)
            {
                let UserPath::Hand(hand) = binding.input.user else {
                    continue;
                };
                let Some(element) = gamepad_element(action.kind, &binding.input) else {
                    continue;
                };
                if !targets.contains(&(hand, element)) {
                    targets.push((hand, element));
                }
            }
            entries.push(PlanEntry {
                set: set.name.clone(),
                action: action.name.clone(),
                kind: action.kind,
                targets,
            });
        }
    }
    entries
}

/// One planned action's state: one reading per hand and control it is bound on,
/// combined by [`combine`]'s rule in declaration order. A hand that is not
/// there, or a control this frame's gamepad does not carry, contributes no
/// reading.
fn entry_state(entry: &PlanEntry, sources: &[InputSource]) -> ActionState {
    let readings: Vec<ActionReading> = entry
        .targets
        .iter()
        .filter_map(|(hand, element)| {
            let source = sources.iter().find(|source| source.hand == *hand)?;
            let gamepad = source.gamepad.as_ref()?;
            let value = element_reading(&gamepad.buttons, &gamepad.axes, *element)?;
            Some(ActionReading::reported(entry.kind, true, value))
        })
        .collect();
    let reading = combine(entry.kind, &readings);
    ActionState {
        set: entry.set.clone(),
        action: entry.action.clone(),
        kind: entry.kind,
        value: reading.value,
        active: reading.active,
    }
}

/// Read every input source the session has for this frame, with the profiles
/// they name.
fn read_sources(
    session: &js::XrSession,
    frame: &XrFrame,
    reference: &js::XrReferenceSpace,
) -> (Vec<InputSource>, Vec<(Hand, InteractionProfile)>) {
    let sources = session.input_sources();
    let mut readings: Vec<InputSource> = Vec::new();
    let mut profiles: Vec<(Hand, InteractionProfile)> = Vec::new();
    for index in 0..sources.length() {
        let Ok(value) = js_sys::Reflect::get(sources.as_ref(), &JsValue::from_f64(f64::from(index)))
        else {
            continue;
        };
        let Ok(source) = value.dyn_into::<js::XrInputSource>() else {
            continue;
        };
        let hand = match source.handedness().as_str() {
            "left" => Hand::Left,
            "right" => Hand::Right,
            other => {
                // A source without a hand is not a device this crate can
                // address: action paths name hands, and `/user/gamepad` has no
                // WebXR counterpart to be sure of.
                log::debug!(
                    "webxr: an input source reports {other:?} handedness; only left and right \
                     hands are read"
                );
                continue;
            }
        };
        // The first source a hand is listed with is the one it is read from;
        // a runtime listing two devices for one hand does not get to mix them.
        if readings.iter().any(|reading| reading.hand == hand) {
            continue;
        }
        if let Some(profile) = source_profile(&source) {
            profiles.push((hand, profile));
        }
        readings.push(InputSource {
            hand,
            gamepad: read_gamepad(&source),
            aim: source_pose(frame, &source.target_ray_space(), reference),
            grip: source
                .grip_space()
                .map(|space| source_pose(frame, &space, reference))
                .unwrap_or(TrackedPose::UNTRACKED),
        });
    }
    (readings, profiles)
}

/// The input source's gamepad, read for this frame, or `None` when it has none
/// or does not report the `xr-standard` mapping, the only mapping whose button
/// and axis indices this crate knows. Guessing another mapping's indices would
/// invent readings.
fn read_gamepad(source: &js::XrInputSource) -> Option<GamepadReading> {
    let gamepad = source.gamepad()?;
    let mapping = gamepad.mapping();
    if mapping != "xr-standard" {
        log::debug!(
            "webxr: an input source reports a {mapping:?} gamepad; only xr-standard controls are \
             read"
        );
        return None;
    }
    let buttons = gamepad.buttons();
    let mut readings = Vec::with_capacity(buttons.length() as usize);
    for index in 0..buttons.length() {
        let Ok(button) = buttons.get(index).dyn_into::<js::GamepadButton>() else {
            log::warn!("webxr: a gamepad button is not a GamepadButton; not reading this gamepad");
            return None;
        };
        readings.push(GamepadButton {
            pressed: button.pressed(),
            value: button.value() as f32,
        });
    }
    let axes = gamepad.axes();
    let mut values = Vec::with_capacity(axes.length() as usize);
    for index in 0..axes.length() {
        // A non-numeric axis reaches the action as a non-finite reading, which
        // is reported as inactive rather than passed on.
        values.push(axes.get(index).as_f64().unwrap_or(f64::NAN) as f32);
    }
    Some(GamepadReading {
        buttons: readings,
        axes: values,
    })
}

/// A pose the runtime reports for an input source's space, or
/// [`TrackedPose::UNTRACKED`] when it has none this frame.
fn source_pose(
    frame: &XrFrame,
    space: &js::XrSpace,
    reference: &js::XrReferenceSpace,
) -> TrackedPose {
    match frame.get_pose(space, reference) {
        Ok(Some(pose)) => {
            let emulated = pose.emulated_position();
            TrackedPose::from_tracking_flags(pose_of(&pose.transform()), true, !emulated)
        }
        Ok(None) => TrackedPose::UNTRACKED,
        Err(error) => {
            log::warn!(
                "webxr: reading an input source pose failed: {}",
                error_text(&error)
            );
            TrackedPose::UNTRACKED
        }
    }
}

/// The interaction profile an input source names, if it names one the way
/// OpenXR does. WebXR profile ids (`"oculus-touch-v3"`) are not interaction
/// profile paths, and no mapping between the two is invented here.
fn source_profile(source: &js::XrInputSource) -> Option<InteractionProfile> {
    let profiles = source.profiles();
    for index in 0..profiles.length() {
        let Some(text) = profiles.get(index).as_string() else {
            continue;
        };
        match InteractionProfile::parse(&text) {
            Ok(profile) => return Some(profile),
            Err(_) => log::debug!(
                "webxr: an input source reports the profile id {text:?}, which is not an \
                 interaction profile path"
            ),
        }
    }
    None
}

/// The column-major projection matrix WebXR reports for a view.
fn projection_of(view: &js::XrView) -> XrResult<Mat4> {
    let values = view.projection_matrix().to_vec();
    let values: [f32; 16] = values.try_into().map_err(|_| XrError::InvalidData {
        what: "view projection matrix",
    })?;
    Ok(Mat4::from_cols_array(&values))
}

/// Which eye a view belongs to. `"none"` is what a runtime reports for a mono
/// configuration, which this crate's configurations still call the left eye.
fn eye_of(name: &str, index: usize, configuration: ViewConfiguration) -> Eye {
    match name {
        "left" => Eye::Left,
        "right" => Eye::Right,
        _ => configuration.eyes().get(index).copied().unwrap_or(Eye::Left),
    }
}

/// WebXR's animation frame clock is milliseconds; frames report nanoseconds.
fn nanos_of(time_ms: f64) -> i64 {
    (time_ms * 1_000_000.0) as i64
}

/// The session mode a view configuration asks for.
fn mode_of(configuration: ViewConfiguration) -> &'static str {
    match configuration {
        ViewConfiguration::PrimaryStereo => "immersive-vr",
        ViewConfiguration::PrimaryMono => "immersive-ar",
    }
}

/// How a session mode composites: `immersive-ar` is laid over the camera image,
/// `immersive-vr` replaces it.
fn blend_of(mode: &str) -> BlendMode {
    if mode == "immersive-ar" {
        BlendMode::AlphaBlend
    } else {
        BlendMode::Opaque
    }
}

/// An RGBA framebuffer is what the runtime composites; `alpha` is what decides
/// whether it also blends over the camera image, and the XR compositor reads the
/// pixels as sRGB either way.
fn layer_alpha(mode: &str) -> bool {
    blend_of(mode) == BlendMode::AlphaBlend
}

/// Which slot a reference space lives in.
fn space_index(space: ReferenceSpace) -> usize {
    match space {
        ReferenceSpace::View => 0,
        ReferenceSpace::Local => 1,
        ReferenceSpace::Stage => 2,
    }
}

/// `navigator.xr`, if this browser has one.
fn xr_system() -> Option<js::XrSystem> {
    let global = js_sys::global();
    let navigator = js_sys::Reflect::get(global.as_ref(), &JsValue::from_str("navigator")).ok()?;
    let xr = js_sys::Reflect::get(&navigator, &JsValue::from_str("xr")).ok()?;
    xr.dyn_into().ok()
}

/// `{ optionalFeatures: ["local-floor"] }`.
///
/// `viewer` and `local` are defaults of every immersive session; `local-floor`
/// is not, and `requestReferenceSpace("local-floor")` is refused without it.
fn session_options() -> js_sys::Object {
    let options = js_sys::Object::new();
    let features = js_sys::Array::new();
    features.push(&JsValue::from_str(ReferenceSpace::Stage.webxr_name()));
    set_property(options.as_ref(), "optionalFeatures", features.as_ref());
    options
}

/// `{ baseLayer: layer }`.
fn render_state(layer: &js::XrWebGlLayer) -> js_sys::Object {
    let state = js_sys::Object::new();
    set_property(state.as_ref(), "baseLayer", layer.as_ref());
    state
}

/// Layer options. `alpha` follows the session mode; the rest are the runtime's
/// defaults, spelled out because they decide what the framebuffer is.
fn layer_init(alpha: bool) -> JsValue {
    let init = js_sys::Object::new();
    let alpha_value = if alpha {
        JsValue::TRUE
    } else {
        JsValue::FALSE
    };
    set_property(init.as_ref(), "alpha", &alpha_value);
    set_property(init.as_ref(), "antialias", &JsValue::TRUE);
    set_property(init.as_ref(), "depth", &JsValue::TRUE);
    set_property(init.as_ref(), "stencil", &JsValue::FALSE);
    init.into()
}

/// Set a property of a JavaScript options object. This cannot fail for an object
/// this crate just made, so a refusal is logged rather than carried around.
fn set_property(object: &JsValue, key: &str, value: &JsValue) {
    match js_sys::Reflect::set(object, &JsValue::from_str(key), value) {
        Ok(true) => {}
        Ok(false) => log::warn!("webxr: {key} could not be set on a runtime options object"),
        Err(error) => log::warn!("webxr: setting {key} failed: {}", error_text(&error)),
    }
}

/// Attach both promise handlers.
///
/// They are forgotten rather than kept: the runtime may invoke one after the
/// session is gone, and a closure that was dropped aborts when JavaScript calls
/// it. Both hold a [`Weak`] to the session state, so nothing keeps a dead
/// session alive on their account.
fn then2<F, R>(promise: &js_sys::Promise, resolve: F, reject: R)
where
    F: FnMut(JsValue) + 'static,
    R: FnMut(JsValue) + 'static,
{
    let resolve = Closure::wrap_assert_unwind_safe(Box::new(resolve) as Box<dyn FnMut(JsValue)>);
    let reject = Closure::wrap_assert_unwind_safe(Box::new(reject) as Box<dyn FnMut(JsValue)>);
    let _next = promise.then2(&resolve, &reject);
    resolve.forget();
    reject.forget();
}

/// Log a promise's failure instead of hiding it.
fn catch_log(promise: js_sys::Promise, what: &'static str) {
    let reject = Closure::wrap_assert_unwind_safe(Box::new(move |error: JsValue| {
        log::warn!("webxr: {what} failed: {}", error_text(&error));
    }) as Box<dyn FnMut(JsValue)>);
    let _next = promise.catch(&reject);
    reject.forget();
}

/// The runtime's own words for a JavaScript exception.
fn error_text(error: &JsValue) -> String {
    if let Some(exception) = error.dyn_ref::<js_sys::Error>() {
        return String::from(exception.message());
    }
    match error.as_string() {
        Some(text) => text,
        None => format!("{error:?}"),
    }
}

/// The legacy numeric code of a DOM exception, when it has one.
fn js_error_code(error: &JsValue) -> i64 {
    js_sys::Reflect::get(error, &JsValue::from_str("code"))
        .ok()
        .and_then(|code| code.as_f64())
        .map(|code| code as i64)
        .unwrap_or(0)
}

/// The WebXR interfaces this backend uses.
///
/// Every one of these is a WebXR interface web-sys gates behind
/// `--cfg=web_sys_unstable_apis`, so they are declared here instead; the
/// attribute set is the one web-sys itself is generated with.
mod js {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(extends = "::js_sys::Object", js_name = "XRSystem", typescript_type = "XRSystem")]
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub type XrSystem;

        #[wasm_bindgen(method, js_class = "XRSystem", js_name = "requestSession")]
        pub fn request_session(this: &XrSystem, mode: &str, options: &JsValue) -> js_sys::Promise;

        #[wasm_bindgen(extends = "::js_sys::Object", js_name = "XRSession", typescript_type = "XRSession")]
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub type XrSession;

        #[wasm_bindgen(method, getter, js_class = "XRSession", js_name = "visibilityState")]
        pub fn visibility_state(this: &XrSession) -> String;

        #[wasm_bindgen(method, getter, js_class = "XRSession", js_name = "inputSources")]
        pub fn input_sources(this: &XrSession) -> XrInputSourceArray;

        #[wasm_bindgen(method, js_class = "XRSession", js_name = "end")]
        pub fn end(this: &XrSession) -> js_sys::Promise;

        #[wasm_bindgen(method, setter, js_class = "XRSession", js_name = "onend")]
        pub fn set_onend(this: &XrSession, value: Option<&js_sys::Function>);

        #[wasm_bindgen(method, js_class = "XRSession", js_name = "updateRenderState")]
        pub fn update_render_state(this: &XrSession, state: &JsValue);

        #[wasm_bindgen(method, js_class = "XRSession", js_name = "requestReferenceSpace")]
        pub fn request_reference_space(this: &XrSession, kind: &str) -> js_sys::Promise;

        #[wasm_bindgen(method, js_class = "XRSession", js_name = "requestAnimationFrame")]
        pub fn request_animation_frame(this: &XrSession, callback: &js_sys::Function) -> u32;

        #[wasm_bindgen(method, js_class = "XRSession", js_name = "cancelAnimationFrame")]
        pub fn cancel_animation_frame(this: &XrSession, handle: u32);

        #[wasm_bindgen(extends = "::js_sys::Object", js_name = "XRWebGLLayer", typescript_type = "XRWebGLLayer")]
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub type XrWebGlLayer;

        #[wasm_bindgen(catch, constructor, js_class = "XRWebGLLayer")]
        pub fn new(
            session: &XrSession,
            context: &JsValue,
            init: &JsValue,
        ) -> Result<XrWebGlLayer, JsValue>;

        #[wasm_bindgen(method, getter, js_class = "XRWebGLLayer", js_name = "framebuffer")]
        pub fn framebuffer(this: &XrWebGlLayer) -> Option<web_sys::WebGlFramebuffer>;

        #[wasm_bindgen(method, getter, js_class = "XRWebGLLayer", js_name = "framebufferWidth")]
        pub fn framebuffer_width(this: &XrWebGlLayer) -> u32;

        #[wasm_bindgen(method, getter, js_class = "XRWebGLLayer", js_name = "framebufferHeight")]
        pub fn framebuffer_height(this: &XrWebGlLayer) -> u32;

        #[wasm_bindgen(method, js_class = "XRWebGLLayer", js_name = "getViewport")]
        pub fn get_viewport(this: &XrWebGlLayer, view: &XrView) -> Option<XrViewport>;

        #[wasm_bindgen(extends = "::js_sys::Object", js_name = "XRSpace", typescript_type = "XRSpace")]
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub type XrSpace;

        #[wasm_bindgen(extends = "XrSpace", js_name = "XRReferenceSpace", typescript_type = "XRReferenceSpace")]
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub type XrReferenceSpace;

        #[wasm_bindgen(extends = "::js_sys::Object", js_name = "XRFrame", typescript_type = "XRFrame")]
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub type XrFrame;

        #[wasm_bindgen(catch, method, js_class = "XRFrame", js_name = "getViewerPose")]
        pub fn get_viewer_pose(
            this: &XrFrame,
            space: &XrSpace,
        ) -> Result<Option<XrViewerPose>, JsValue>;

        #[wasm_bindgen(catch, method, js_class = "XRFrame", js_name = "getPose")]
        pub fn get_pose(
            this: &XrFrame,
            space: &XrSpace,
            base: &XrSpace,
        ) -> Result<Option<XrPose>, JsValue>;

        #[wasm_bindgen(extends = "::js_sys::Object", js_name = "XRPose", typescript_type = "XRPose")]
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub type XrPose;

        #[wasm_bindgen(method, getter, js_class = "XRPose", js_name = "transform")]
        pub fn transform(this: &XrPose) -> XrRigidTransform;

        #[wasm_bindgen(method, getter, js_class = "XRPose", js_name = "emulatedPosition")]
        pub fn emulated_position(this: &XrPose) -> bool;

        #[wasm_bindgen(extends = "XrPose", js_name = "XRViewerPose", typescript_type = "XRViewerPose")]
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub type XrViewerPose;

        #[wasm_bindgen(method, getter, js_class = "XRViewerPose", js_name = "views")]
        pub fn views(this: &XrViewerPose) -> js_sys::Array;

        #[wasm_bindgen(extends = "::js_sys::Object", js_name = "XRView", typescript_type = "XRView")]
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub type XrView;

        #[wasm_bindgen(method, getter, js_class = "XRView", js_name = "eye")]
        pub fn eye(this: &XrView) -> String;

        #[wasm_bindgen(method, getter, js_class = "XRView", js_name = "projectionMatrix")]
        pub fn projection_matrix(this: &XrView) -> js_sys::Float32Array;

        #[wasm_bindgen(method, getter, js_class = "XRView", js_name = "transform")]
        pub fn transform(this: &XrView) -> XrRigidTransform;

        #[wasm_bindgen(extends = "::js_sys::Object", js_name = "XRViewport", typescript_type = "XRViewport")]
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub type XrViewport;

        #[wasm_bindgen(method, getter, js_class = "XRViewport", js_name = "x")]
        pub fn x(this: &XrViewport) -> i32;

        #[wasm_bindgen(method, getter, js_class = "XRViewport", js_name = "y")]
        pub fn y(this: &XrViewport) -> i32;

        #[wasm_bindgen(method, getter, js_class = "XRViewport", js_name = "width")]
        pub fn width(this: &XrViewport) -> i32;

        #[wasm_bindgen(method, getter, js_class = "XRViewport", js_name = "height")]
        pub fn height(this: &XrViewport) -> i32;

        #[wasm_bindgen(extends = "::js_sys::Object", js_name = "XRRigidTransform", typescript_type = "XRRigidTransform")]
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub type XrRigidTransform;

        #[wasm_bindgen(method, getter, js_class = "XRRigidTransform", js_name = "position")]
        pub fn position(this: &XrRigidTransform) -> web_sys::DomPointReadOnly;

        #[wasm_bindgen(method, getter, js_class = "XRRigidTransform", js_name = "orientation")]
        pub fn orientation(this: &XrRigidTransform) -> web_sys::DomPointReadOnly;

        // `XRInputSourceArray` is array-like rather than an `Array`: it has a
        // length and an indexed getter, so a source is read through
        // `Reflect::get` by index.

        #[wasm_bindgen(extends = "::js_sys::Object", js_name = "XRInputSourceArray", typescript_type = "XRInputSourceArray")]
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub type XrInputSourceArray;

        #[wasm_bindgen(method, getter, js_class = "XRInputSourceArray", js_name = "length")]
        pub fn length(this: &XrInputSourceArray) -> u32;

        #[wasm_bindgen(extends = "::js_sys::Object", js_name = "XRInputSource", typescript_type = "XRInputSource")]
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub type XrInputSource;

        #[wasm_bindgen(method, getter, js_class = "XRInputSource", js_name = "handedness")]
        pub fn handedness(this: &XrInputSource) -> String;

        #[wasm_bindgen(method, getter, js_class = "XRInputSource", js_name = "targetRaySpace")]
        pub fn target_ray_space(this: &XrInputSource) -> XrSpace;

        #[wasm_bindgen(method, getter, js_class = "XRInputSource", js_name = "gripSpace")]
        pub fn grip_space(this: &XrInputSource) -> Option<XrSpace>;

        #[wasm_bindgen(method, getter, js_class = "XRInputSource", js_name = "gamepad")]
        pub fn gamepad(this: &XrInputSource) -> Option<Gamepad>;

        #[wasm_bindgen(method, getter, js_class = "XRInputSource", js_name = "profiles")]
        pub fn profiles(this: &XrInputSource) -> js_sys::Array;

        // The Gamepad API itself is stable, but its types are declared here
        // too, so that this module is the one place the WebXR surface is
        // spelled.

        #[wasm_bindgen(extends = "::js_sys::Object", js_name = "Gamepad", typescript_type = "Gamepad")]
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub type Gamepad;

        #[wasm_bindgen(method, getter, js_class = "Gamepad", js_name = "mapping")]
        pub fn mapping(this: &Gamepad) -> String;

        #[wasm_bindgen(method, getter, js_class = "Gamepad", js_name = "buttons")]
        pub fn buttons(this: &Gamepad) -> js_sys::Array;

        #[wasm_bindgen(method, getter, js_class = "Gamepad", js_name = "axes")]
        pub fn axes(this: &Gamepad) -> js_sys::Array;

        #[wasm_bindgen(extends = "::js_sys::Object", js_name = "GamepadButton", typescript_type = "GamepadButton")]
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub type GamepadButton;

        #[wasm_bindgen(method, getter, js_class = "GamepadButton", js_name = "pressed")]
        pub fn pressed(this: &GamepadButton) -> bool;

        #[wasm_bindgen(method, getter, js_class = "GamepadButton", js_name = "value")]
        pub fn value(this: &GamepadButton) -> f64;
    }
}
