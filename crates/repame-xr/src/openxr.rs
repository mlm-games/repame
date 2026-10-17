//! The OpenXR backend: a real runtime, loaded at run time.
//!
//! The loader is dlopened ([`Entry::load`]), so a machine without an OpenXR
//! runtime produces a typed [`XrError::Unavailable`] from [`probe`] and
//! [`OpenXrBackend::open`] instead of a link failure, and a game can decide to
//! run flat.
//!
//! Design, and where it differs from the obvious:
//!
//! - **The session renders with the app's device.** OpenXR creates a session
//!   *against* the Vulkan instance and device the render layer already owns
//!   (`XR_KHR_vulkan_enable2`), so [`SessionConfig::graphics`] must carry them.
//!   A runtime that needs Vulkan extensions the app has not enabled refuses
//!   session creation; that refusal is passed on as [`XrError::Runtime`] with
//!   the runtime's own code and text rather than shrugged off, because with an
//!   existing instance there is nothing this crate can enable on the app's
//!   behalf.
//! - **Nothing is fabricated.** A view the runtime's flags call invalid becomes
//!   a [`ViewSetup`] with no pose and no view matrix, and
//!   [`StereoFrame::submit`] skips that eye. A failed `locate` never falls back
//!   to the last pose.
//! - **One swapchain, one layer per eye.** The swapchain has one array layer per
//!   view, so both eyes share the image and [`EyeImage::handle`] differs only in
//!   [`SwapchainImageHandle::array_index`], which is what the renderer imports.
//! - **State comes from events.** [`OpenXrBackend::state`] drains `xrPollEvent`
//!   and maps the runtime's session states one-for-one; a state this crate does
//!   not model keeps the last known one and is logged.
//!
//! Actions are attached once ([`OpenXrBackend::set_action_sets`]) and synced
//! once per frame after the frame has begun, so an action's reading always
//! belongs to the frame it was read in. The bindings registered with the
//! runtime are the ones the sets declare, suggested before the attach: the spec
//! freezes a set when it is attached, and that suggestion is what the runtime's
//! own binding UI reads. A profile change is reported, and re-suggested where
//! the runtime allows it. Hand poses come from pose actions this backend
//! creates for itself, one per hand and per pose source, spelled from
//! [`crate::HandPose::input_path`]; a hand the runtime is not tracking has no
//! pose rather than a stand-in one.

use std::cell::Cell;
use std::ffi::c_void;

use ::openxr::sys::{self, Extent2Di, Fovf, Handle as _, Offset2Di, Posef, Rect2Di};
use ::openxr::{
    Action as XrAction, ActionSet as XrActionSet, ActiveActionSet, ApplicationInfo,
    Binding as XrBinding, CompositionLayerProjection, CompositionLayerProjectionView, Duration,
    Entry, EntryError, EnvironmentBlendMode, Event, EventDataBuffer, ExtensionSet, FormFactor,
    FrameStream, FrameWaiter, Instance, Path as XrPath, Session, Space, Swapchain,
    SwapchainCreateFlags, SwapchainCreateInfo, SwapchainSubImage, SwapchainUsageFlags, Time,
    Vector2f, ViewConfigurationType, ViewStateFlags, Vulkan,
};

use glam::Vec2;

use crate::action::{
    ActionKind, ActionReading, ActionSet, ActionState, ActionValue, InteractionProfile, UserPath,
    combine,
};
use crate::error::{XrError, XrResult};
use crate::pose::{Hand, HandPose, Pose, Poses, TrackedPose, TrackedRole};
use crate::session::{SessionConfig, SessionState};
use crate::space::ReferenceSpace;
use crate::stereo::{BlendMode, Eye, Fov, PixelFormat, ViewConfig, ViewConfiguration, ViewSetup};
use crate::swapchain::{
    EyeImage, FrameError, FrameOutcome, GraphicsApi, GraphicsBinding, StereoFrame, StereoSubmit,
    SwapchainImageHandle,
};

/// Backend name reported for OpenXR.
pub(crate) const NAME: &str = "openxr";

/// Application name used by [`probe`], which has no [`SessionConfig`] to take one
/// from.
const PROBE_APP_NAME: &str = "repame";

/// The OpenXR API version this crate asks for. `1.0` is what every runtime
/// answers; asking for a newer one is the runtime's to refuse.
const API_VERSION: (u16, u16, u32) = (1, 0, 0);

/// The action set this backend keeps its own hand pose actions in. Action set
/// names are unique across an instance, so an app's set cannot use this one and
/// the one set does not share a name with the app's.
const POSE_SET_NAME: &str = "repame_poses";

/// Localized name of that set, for the runtime's binding UI.
const POSE_SET_LOCALIZED_NAME: &str = "Tracked hand poses";

// The Vulkan formats this crate can describe, at their spec-frozen values.
// `openxr` hands `VkFormat` over as a bare `u32`, so the names live here.
const VK_FORMAT_R8G8B8A8_UNORM: u32 = 37;
const VK_FORMAT_R8G8B8A8_SRGB: u32 = 43;
const VK_FORMAT_B8G8R8A8_UNORM: u32 = 44;
const VK_FORMAT_B8G8R8A8_SRGB: u32 = 50;
const VK_FORMAT_R16G16B16A16_SFLOAT: u32 = 97;

/// Swapchain formats in the order this crate prefers them: an sRGB target, as
/// the engine's colour pipeline expects, then 16-bit float, then linear 8-bit.
const FORMAT_PREFERENCE: [PixelFormat; 5] = [
    PixelFormat::Bgra8UnormSrgb,
    PixelFormat::Rgba8UnormSrgb,
    PixelFormat::Rgba16Float,
    PixelFormat::Rgba8Unorm,
    PixelFormat::Bgra8Unorm,
];

/// Probe for a loader, a runtime, an HMD and the Vulkan extension this backend
/// renders through, without creating a session.
pub(crate) fn probe() -> XrResult<()> {
    let entry = load_entry()?;
    require_vulkan(&entry)?;
    let app_info = app_info(PROBE_APP_NAME, (0, 0, 0), PROBE_APP_NAME, (0, 0, 0));
    let instance = entry
        .create_instance(&app_info, &vulkan_extensions(), &[], &())
        .map_err(|code| XrError::Unavailable {
            backend: NAME,
            detail: format!("xrCreateInstance found no runtime: {code}"),
        })?;
    let system = instance
        .system(FormFactor::HEAD_MOUNTED_DISPLAY)
        .map_err(|code| XrError::Unavailable {
            backend: NAME,
            detail: format!("no head-mounted display: {code}"),
        })?;
    log::debug!("openxr: probe found system {}", system.into_raw());
    Ok(())
}

/// A real OpenXR session, and everything one frame of it needs.
pub(crate) struct OpenXrBackend {
    instance: Instance,
    session: Session<Vulkan>,
    waiter: FrameWaiter,
    stream: FrameStream<Vulkan>,
    swapchain: SwapchainState,
    eyes: Vec<EyeState>,
    /// `XR_REFERENCE_SPACE_TYPE_VIEW`, the space whose pose in the app's space is
    /// the head pose.
    view_space: Space,
    /// The app's reference space, recreated when
    /// [`crate::XrSession::set_space`] changes it.
    world_space: Space,
    world_space_kind: ReferenceSpace,
    view_configuration: ViewConfigurationType,
    blend_mode: EnvironmentBlendMode,
    blend: BlendMode,
    format: PixelFormat,
    sample_count: u32,
    near: f32,
    far: f32,
    state: Cell<SessionState>,
    poses: Cell<Poses>,
    display_time_ns: Cell<i64>,
    /// The app's action sets as the runtime holds them, once installed.
    installed: Option<Installed>,
    /// Profiles whose bindings the app asked for on top of the declared ones.
    fallbacks: Vec<InteractionProfile>,
    /// Which of the two hand poses a sync reads.
    hand_pose: HandPose,
    /// States the last sync produced, in declared-action order.
    states: Vec<ActionState>,
}

impl OpenXrBackend {
    /// Create an instance, a system, a session and everything a frame needs.
    pub(crate) fn open(config: &SessionConfig) -> XrResult<Self> {
        let binding = match config.graphics {
            Some(GraphicsBinding::Vulkan(binding)) => binding,
            None => {
                return Err(XrError::Unsupported {
                    backend: NAME,
                    detail: "a session without the render layer's Vulkan handles; set \
                             SessionConfig::graphics"
                        .to_string(),
                });
            }
        };
        check_names(config)?;
        let entry = load_entry()?;
        require_vulkan(&entry)?;
        let app_info = app_info(
            &config.app_name,
            config.app_version,
            &config.engine_name,
            config.engine_version,
        );
        let instance = entry
            .create_instance(&app_info, &vulkan_extensions(), &[], &())
            .map_err(|code| XrError::Unavailable {
                backend: NAME,
                detail: format!("xrCreateInstance found no runtime: {code}"),
            })?;
        let system = instance
            .system(FormFactor::HEAD_MOUNTED_DISPLAY)
            .map_err(|code| XrError::Unavailable {
                backend: NAME,
                detail: format!("no head-mounted display: {code}"),
            })?;
        match instance.system_properties(system) {
            Ok(properties) => log::info!(
                "openxr: runtime {} (vendor {:#x}), system {}",
                properties.system_name,
                properties.vendor_id,
                properties.system_id.into_raw()
            ),
            Err(code) => log::debug!(
                "openxr: the runtime would not describe its system: {}",
                error_text(&instance, code)
            ),
        }

        let view_configuration = view_configuration_type(config.view_configuration);
        let offered = instance
            .enumerate_view_configurations(system)
            .map_err(|code| runtime_error(&instance, code))?;
        if !offered.contains(&view_configuration) {
            return Err(XrError::Unsupported {
                backend: NAME,
                detail: format!(
                    "the {} view configuration; the runtime offers {offered:?}",
                    view_configuration_name(config.view_configuration)
                ),
            });
        }
        let views = instance
            .enumerate_view_configuration_views(system, view_configuration)
            .map_err(|code| runtime_error(&instance, code))?;
        if views.len() != config.view_configuration.view_count() {
            return Err(XrError::InvalidData {
                what: "view configuration view count",
            });
        }
        let blend_mode = choose_blend_mode(
            &instance
                .enumerate_environment_blend_modes(system, view_configuration)
                .map_err(|code| runtime_error(&instance, code))?,
            &config.blend_modes,
        )
        .ok_or_else(|| XrError::Unsupported {
            backend: NAME,
            detail: "an environment blend mode from SessionConfig::blend_modes".to_string(),
        })?;

        // The runtime's own answer about the Vulkan versions it accepts, and the
        // proof that the extension really is usable. The app's device stays the
        // app's: what the runtime needs from it shows up as a refusal below.
        let requirements = instance
            .graphics_requirements::<Vulkan>(system)
            .map_err(|code| runtime_error(&instance, code))?;
        log::debug!(
            "openxr: vulkan {}..={}",
            requirements.min_api_version_supported,
            requirements.max_api_version_supported
        );

        let create_info = ::openxr::vulkan::SessionCreateInfo {
            instance: binding.instance as *const c_void,
            physical_device: binding.physical_device as *const c_void,
            device: binding.device as *const c_void,
            queue_family_index: binding.queue_family_index,
            queue_index: binding.queue_index,
        };
        let (session, waiter, stream) =
            unsafe { instance.create_session::<Vulkan>(system, &create_info) }.map_err(|code| {
                XrError::Runtime {
                    backend: NAME,
                    code: code.into_raw() as i64,
                    detail: format!(
                        "xrCreateSession refused the render layer's Vulkan handles, which the \
                         runtime will do when they do not carry the extensions it asked for: {code}"
                    ),
                }
            })?;

        let mut eyes = Vec::with_capacity(views.len());
        for (index, view) in views.iter().enumerate() {
            if view.recommended_image_rect_width == 0 || view.recommended_image_rect_height == 0 {
                return Err(XrError::InvalidData {
                    what: "recommended view size",
                });
            }
            eyes.push(EyeState {
                eye: config.view_configuration.eyes()[index],
                rect: [
                    0,
                    0,
                    view.recommended_image_rect_width,
                    view.recommended_image_rect_height,
                ],
            });
        }
        let size = [
            eyes.iter().map(|eye| eye.rect[2]).max().unwrap_or(1),
            eyes.iter().map(|eye| eye.rect[3]).max().unwrap_or(1),
        ];
        let sample_count = swapchain_sample_count(&views);

        let offered = session
            .enumerate_swapchain_formats()
            .map_err(|code| runtime_error(&instance, code))?;
        let raw_format = choose_format(&offered).ok_or_else(|| XrError::Unsupported {
            backend: NAME,
            detail: format!(
                "a swapchain format this crate can import; the runtime offers {offered:?}"
            ),
        })?;
        let format = vk_pixel_format(raw_format).ok_or(XrError::InvalidData {
            what: "swapchain format",
        })?;
        let swapchain = session
            .create_swapchain(&SwapchainCreateInfo {
                create_flags: SwapchainCreateFlags::EMPTY,
                usage_flags: SwapchainUsageFlags::COLOR_ATTACHMENT | SwapchainUsageFlags::SAMPLED,
                format: raw_format,
                sample_count,
                width: size[0],
                height: size[1],
                face_count: 1,
                array_size: eyes.len() as u32,
                mip_count: 1,
            })
            .map_err(|code| runtime_error(&instance, code))?;
        let images = swapchain
            .enumerate_images()
            .map_err(|code| runtime_error(&instance, code))?;
        let handle = swapchain.as_raw().into_raw();
        let view_space = session
            .create_reference_space(::openxr::ReferenceSpaceType::VIEW, Posef::IDENTITY)
            .map_err(|code| runtime_error(&instance, code))?;
        let world_space = session
            .create_reference_space(
                reference_space_type(ReferenceSpace::default()),
                Posef::IDENTITY,
            )
            .map_err(|code| runtime_error(&instance, code))?;

        log::info!(
            "openxr: session for {} eye(s), {}x{} {format:?}, {sample_count} sample(s), blend \
             {blend_mode:?}",
            eyes.len(),
            size[0],
            size[1],
        );
        Ok(Self {
            instance,
            session,
            waiter,
            stream,
            swapchain: SwapchainState {
                swapchain,
                images,
                handle,
                size,
            },
            eyes,
            view_space,
            world_space,
            world_space_kind: ReferenceSpace::default(),
            view_configuration,
            blend_mode,
            blend: blend_of(blend_mode),
            format,
            sample_count,
            near: config.near,
            far: config.far,
            state: Cell::new(SessionState::Idle),
            poses: Cell::new(Poses::new()),
            display_time_ns: Cell::new(0),
            installed: None,
            fallbacks: Vec::new(),
            hand_pose: HandPose::default(),
            states: Vec::new(),
        })
    }

    /// Runtime-reported session state.
    pub(crate) fn state(&self) -> SessionState {
        self.poll();
        self.state.get()
    }

    /// Ask the runtime to begin the session. Legal in `Ready` only; the runtime
    /// refuses anything else, and so does this.
    pub(crate) fn begin(&mut self) -> XrResult<()> {
        let state = self.state();
        if !state.can_begin() {
            return Err(XrError::WrongState {
                expected: "Ready",
                actual: state,
            });
        }
        match self.session.begin(self.view_configuration) {
            Ok(code) => {
                if code != sys::Result::SUCCESS {
                    log::warn!("openxr: xrBeginSession answered {code}");
                }
                Ok(())
            }
            Err(code) => Err(runtime_error(&self.instance, code)),
        }
    }

    /// Ask the runtime to end the session.
    ///
    /// `xrEndSession` is legal in `Stopping` only, so a running session is asked
    /// to stop first and answers [`XrError::NotReady`] until the runtime reaches
    /// that state.
    pub(crate) fn end(&mut self) -> XrResult<()> {
        let state = self.state();
        match state {
            SessionState::Stopping => match self.session.end() {
                Ok(code) => {
                    if code != sys::Result::SUCCESS {
                        log::warn!("openxr: xrEndSession answered {code}");
                    }
                    // xrEndSession moves the session to idle; the runtime's own
                    // event will confirm it.
                    self.state.set(SessionState::Idle);
                    Ok(())
                }
                Err(code) => Err(runtime_error(&self.instance, code)),
            },
            state if state.can_end() => match self.session.request_exit() {
                Ok(()) => Err(XrError::NotReady { what: "session stop" }),
                Err(code) => Err(runtime_error(&self.instance, code)),
            },
            state => Err(XrError::WrongState {
                expected: "Stopping",
                actual: state,
            }),
        }
    }

    /// Poses of the most recent frame. The head comes from the runtime's view
    /// space; hands come from the pose actions, once action sets are installed.
    pub(crate) fn poses(&self) -> Poses {
        self.poses.get()
    }

    /// Install the app's action sets: create the runtime's own sets and
    /// actions, register their bindings, attach them, and create the hand pose
    /// actions hands are read through.
    ///
    /// Once per session: a second call is refused here rather than merged, and
    /// the runtime refuses to attach a second time on its own.
    pub(crate) fn set_action_sets(&mut self, sets: Vec<ActionSet>) -> XrResult<()> {
        if self.installed.is_some() {
            return Err(XrError::ActionSetsInstalled);
        }
        if sets.iter().any(|set| set.name == POSE_SET_NAME) {
            return Err(XrError::Unsupported {
                backend: NAME,
                detail: format!(
                    "an action set named {POSE_SET_NAME}; this backend keeps its own hand \
                     pose actions there"
                ),
            });
        }

        let mut installed = self.create_actions(sets)?;
        // Suggesting bindings is only legal while the sets are unattached:
        // attaching them is what makes them immutable at the runtime. Every
        // profile the app declares goes in here, which is what the runtime's
        // binding UI reads.
        for profile in self.installed_profiles(&installed) {
            self.register_profile(&installed, &profile, Registration::Declared);
        }
        // Attaching is what commits the sets: it is the first call the runtime
        // sees them through, and it makes them immutable. Everything above is
        // building blocks it is handed here.
        let mut all_sets: Vec<&XrActionSet> = installed.runtime_sets.iter().collect();
        all_sets.push(&installed.poses.set);
        self.session
            .attach_action_sets(&all_sets)
            .map_err(|code| runtime_error(&self.instance, code))?;
        // Spaces for the pose actions. A runtime that refuses one leaves that
        // hand untracked and the rest of the session working.
        self.create_pose_spaces(&mut installed);
        self.states.clear();
        self.installed = Some(installed);
        Ok(())
    }

    /// Which hand pose a sync reads. Both are bound already, so this only
    /// decides which of the two spaces is located.
    pub(crate) fn set_hand_pose_source(&mut self, pose: HandPose) {
        self.hand_pose = pose;
    }

    /// Register bindings for profiles the app named beyond the declared ones.
    ///
    /// A profile added before the sets are installed is part of the install; one
    /// added afterwards can only be suggested where the runtime allows it, since
    /// attaching the sets is what makes them immutable.
    pub(crate) fn set_fallback_profiles(&mut self, profiles: Vec<InteractionProfile>) {
        for profile in profiles {
            if !self.fallbacks.contains(&profile) {
                self.fallbacks.push(profile);
            }
        }
        let Some(installed) = self.installed.as_ref() else {
            // Remembered; registering them is part of installing the sets.
            return;
        };
        for profile in &self.fallbacks {
            self.register_profile(installed, profile, Registration::BestEffort);
        }
    }

    /// States the last sync produced, in declared-action order.
    pub(crate) fn action_states(&self) -> &[ActionState] {
        &self.states
    }

    /// One action's state by set and action name.
    pub(crate) fn action(&self, set: &str, action: &str) -> XrResult<&ActionState> {
        crate::find_action(&self.states, set, action)
    }

    /// What the runtime says is driving a device right now, asked of the
    /// runtime rather than remembered from a binding.
    pub(crate) fn interaction_profile(&self, user: &UserPath) -> Option<InteractionProfile> {
        self.query_profile(user)
    }

    /// Drive one frame: wait, begin, locate the views, hand the runtime's images
    /// to `sink`, release them, end the frame.
    pub(crate) fn frame<S: StereoSubmit>(
        &mut self,
        space: ReferenceSpace,
        sink: &mut S,
    ) -> Result<FrameOutcome, FrameError<S::Error>> {
        self.poll();
        let state = self.state.get();
        if !state.should_submit_frames() {
            log::debug!("openxr: no frame to render in state {state:?}");
            return Ok(FrameOutcome {
                rendered: false,
                should_render: false,
                display_time_ns: self.display_time_ns.get(),
                views: 0,
            });
        }
        self.set_world_space(space)?;

        let frame_state = self
            .waiter
            .wait()
            .map_err(|code| FrameError::Xr(runtime_error(&self.instance, code)))?;
        let display_time = frame_state.predicted_display_time;
        self.display_time_ns.set(display_time.as_nanos());
        self.capture_head_pose(display_time);
        self.stream
            .begin()
            .map_err(|code| FrameError::Xr(runtime_error(&self.instance, code)))?;
        // Input is synced for every frame the runtime hands out, including the
        // ones it does not want rendered: a sync belongs to the frame, not to
        // the draw.
        self.sync_actions(display_time);

        if !frame_state.should_render {
            // The runtime wants the frame, but nothing of it is shown: close it
            // without a layer so the user keeps the last accepted image.
            let closed = self.stream.end(display_time, self.blend_mode, &[]);
            closed.map_err(|code| FrameError::Xr(runtime_error(&self.instance, code)))?;
            return Ok(FrameOutcome {
                rendered: false,
                should_render: false,
                display_time_ns: display_time.as_nanos(),
                views: 0,
            });
        }
        self.render(display_time, sink)
    }

    /// The rendering half of a frame, with the frame already begun: every path
    /// out of here ends it.
    fn render<S: StereoSubmit>(
        &mut self,
        display_time: Time,
        sink: &mut S,
    ) -> Result<FrameOutcome, FrameError<S::Error>> {
        let (flags, views) = match self.session.locate_views(
            self.view_configuration,
            display_time,
            &self.world_space,
        ) {
            Ok(located) => located,
            Err(code) => {
                self.close_frame(display_time);
                return Err(FrameError::Xr(runtime_error(&self.instance, code)));
            }
        };
        if views.len() != self.eyes.len() {
            self.close_frame(display_time);
            return Err(FrameError::Xr(XrError::InvalidData {
                what: "located view count",
            }));
        }

        let mut setups = Vec::with_capacity(views.len());
        for (view, eye) in views.iter().zip(&self.eyes) {
            let mut setup = ViewSetup::from_pose_fov(
                eye.eye,
                view_pose(view, flags).pose,
                fov_of(view.fov),
                self.near,
                self.far,
            );
            setup.viewport = Some(eye.rect);
            setups.push(setup);
        }
        let config = ViewConfig {
            views: setups,
            recommended_size: self.swapchain.size,
            near: self.near,
            far: self.far,
            blend_mode: self.blend,
        };

        let image_index = match self.swapchain.swapchain.acquire_image() {
            Ok(index) => index,
            Err(code) => {
                self.close_frame(display_time);
                return Err(FrameError::Xr(runtime_error(&self.instance, code)));
            }
        };
        if let Err(code) = self.swapchain.swapchain.wait_image(Duration::INFINITE) {
            // openxr's wrapper releases an image only once it was waited for, and
            // every way this can fail ends the session anyway.
            self.close_frame(display_time);
            return Err(FrameError::Xr(runtime_error(&self.instance, code)));
        }
        let Some(raw_image) = self.swapchain.images.get(image_index as usize).copied() else {
            if let Err(code) = self.swapchain.swapchain.release_image() {
                log::error!(
                    "openxr: releasing a bad image also failed: {}",
                    error_text(&self.instance, code)
                );
            }
            self.close_frame(display_time);
            return Err(FrameError::Xr(XrError::InvalidData {
                what: "swapchain image index",
            }));
        };

        let images: Vec<EyeImage> = self
            .eyes
            .iter()
            .map(|eye| EyeImage {
                eye: eye.eye,
                size: self.swapchain.size,
                format: self.format,
                sample_count: self.sample_count,
                viewport: eye.rect,
                handle: SwapchainImageHandle {
                    api: GraphicsApi::Vulkan,
                    handle: raw_image,
                    array_index: eye.eye.index() as u32,
                },
                swapchain: self.swapchain.handle,
                image_index,
            })
            .collect();
        let submitted = StereoFrame {
            config: &config,
            images: &images,
        }
        .submit(sink);

        if let Err(code) = self.swapchain.swapchain.release_image() {
            self.close_frame(display_time);
            return Err(FrameError::Xr(runtime_error(&self.instance, code)));
        }
        if let Err(error) = submitted {
            // The render layer drew nothing usable: end the frame without a
            // layer so the user keeps the last accepted image, and report the
            // layer's own error untouched.
            if let Err(code) = self.stream.end(display_time, self.blend_mode, &[]) {
                log::error!(
                    "openxr: closing a failed frame also failed: {}",
                    error_text(&self.instance, code)
                );
            }
            return Err(FrameError::Sink(error));
        }

        let layer_views: Vec<CompositionLayerProjectionView<'_, Vulkan>> = views
            .iter()
            .zip(&self.eyes)
            .map(|(view, eye)| {
                CompositionLayerProjectionView::new()
                    .pose(view.pose)
                    .fov(view.fov)
                    .sub_image(
                        SwapchainSubImage::new()
                            .swapchain(&self.swapchain.swapchain)
                            .image_array_index(eye.eye.index() as u32)
                            .image_rect(Rect2Di {
                                offset: Offset2Di { x: 0, y: 0 },
                                extent: Extent2Di {
                                    width: eye.rect[2] as i32,
                                    height: eye.rect[3] as i32,
                                },
                            }),
                    )
            })
            .collect();
        let layer = CompositionLayerProjection::new()
            .space(&self.world_space)
            .views(&layer_views);
        let ended = self.stream.end(display_time, self.blend_mode, &[&layer]);
        ended.map_err(|code| FrameError::Xr(runtime_error(&self.instance, code)))?;

        Ok(FrameOutcome {
            rendered: true,
            should_render: true,
            display_time_ns: display_time.as_nanos(),
            views: views.len(),
        })
    }

    /// End a frame that a failed runtime call left open, so the runtime is not
    /// left waiting for an `xrEndFrame` that will never come.
    fn close_frame(&mut self, display_time: Time) {
        if let Err(code) = self.stream.end(display_time, self.blend_mode, &[]) {
            log::error!(
                "openxr: closing a failed frame also failed: {}",
                error_text(&self.instance, code)
            );
        }
    }

    /// Create the reference space the app asked for, when it is not the current
    /// one.
    fn set_world_space(&mut self, space: ReferenceSpace) -> XrResult<()> {
        if self.world_space_kind == space {
            return Ok(());
        }
        let created = self
            .session
            .create_reference_space(reference_space_type(space), Posef::IDENTITY)
            .map_err(|code| runtime_error(&self.instance, code))?;
        self.world_space = created;
        self.world_space_kind = space;
        Ok(())
    }

    /// Record the head pose for [`OpenXrBackend::poses`], or leave it untracked
    /// when the runtime is not tracking it. A failed locate clears the snapshot
    /// rather than leaving the previous pose to look current.
    fn capture_head_pose(&self, display_time: Time) {
        let mut poses = self.poses.get();
        match self.view_space.locate(&self.world_space, display_time) {
            Ok(location) => {
                poses.set(
                    TrackedRole::Head,
                    location_pose(posef_pose(&location.pose), location.location_flags),
                );
            }
            Err(code) => {
                log::debug!(
                    "openxr: locating the view space failed: {}",
                    error_text(&self.instance, code)
                );
                poses.set(TrackedRole::Head, TrackedPose::UNTRACKED);
            }
        }
        self.poses.set(poses);
    }

    /// Sync the installed actions, read what they are worth, and locate the
    /// hands through the pose actions the sync just made current.
    ///
    /// One sync per frame, and the only place action states come from: an
    /// action's state is undefined until the frame it was synced in, so a
    /// reading is never taken anywhere else.
    fn sync_actions(&mut self, display_time: Time) {
        let Some(installed) = self.installed.as_ref() else {
            return;
        };
        let active: Vec<ActiveActionSet<'_>> = installed
            .runtime_sets
            .iter()
            .map(ActiveActionSet::new)
            .chain(std::iter::once(ActiveActionSet::new(&installed.poses.set)))
            .collect();
        if let Err(code) = self.session.sync_actions(&active) {
            log::warn!(
                "openxr: syncing actions failed: {}",
                error_text(&self.instance, code)
            );
            // A failed sync has nothing to say about any action, so every
            // action reads as not driven rather than as whatever it last was.
            for state in &mut self.states {
                state.value = ActionReading::inactive(state.kind).value;
                state.active = false;
            }
            let mut poses = self.poses.get();
            for hand in Hand::ALL {
                poses.set(TrackedRole::Hand(hand), TrackedPose::UNTRACKED);
            }
            self.poses.set(poses);
            return;
        }

        let mut states = Vec::with_capacity(installed.actions.len());
        for slot in &installed.actions {
            states.push(self.read_action(slot));
        }
        let hands = self.locate_hands(installed, display_time);
        self.states = states;
        let mut poses = self.poses.get();
        for hand in Hand::ALL {
            poses.set(TrackedRole::Hand(hand), hands[hand.index()]);
        }
        self.poses.set(poses);
    }

    /// Read one action: one reading per device its bindings name, combined in
    /// declaration order by [`combine`]'s rule. An action with no bindings has
    /// no device to ask and reads inactive.
    fn read_action(&self, slot: &ActionSlot) -> ActionState {
        let reading = if slot.subactions.is_empty() {
            ActionReading::inactive(slot.kind)
        } else {
            let readings: Vec<ActionReading> = slot
                .subactions
                .iter()
                .map(|(_, path)| self.read_subaction(slot, *path))
                .collect();
            combine(slot.kind, &readings)
        };
        ActionState {
            set: slot.set.clone(),
            action: slot.action.clone(),
            kind: slot.kind,
            value: reading.value,
            active: reading.active,
        }
    }

    /// One device's reading of one action. A device the runtime will not answer
    /// for reads inactive, at the type's zero value.
    fn read_subaction(&self, slot: &ActionSlot, path: XrPath) -> ActionReading {
        let read = match &slot.runtime {
            RuntimeAction::Boolean(action) => action
                .state(&self.session, path)
                .map(|state| (state.is_active, ActionValue::Bool(state.current_state))),
            RuntimeAction::Float(action) => action
                .state(&self.session, path)
                .map(|state| (state.is_active, ActionValue::Float(state.current_state))),
            RuntimeAction::Vector2(action) => action.state(&self.session, path).map(|state| {
                (
                    state.is_active,
                    ActionValue::Vector2(Vec2::new(state.current_state.x, state.current_state.y)),
                )
            }),
        };
        match read {
            Ok((active, value)) => ActionReading::reported(slot.kind, active, value),
            Err(code) => {
                log::warn!(
                    "openxr: reading action {} failed: {}",
                    slot.action,
                    error_text(&self.instance, code)
                );
                ActionReading::inactive(slot.kind)
            }
        }
    }

    /// Locate both hands through the pose actions for the configured source. A
    /// hand the runtime is not tracking is [`TrackedPose::UNTRACKED`], never a
    /// stand-in.
    fn locate_hands(&self, installed: &Installed, display_time: Time) -> [TrackedPose; 2] {
        let mut poses = [TrackedPose::UNTRACKED; 2];
        for hand in Hand::ALL {
            let slot = installed.poses.hands[hand.index()].get(self.hand_pose);
            let active = match slot.action.is_active(&self.session, slot.path) {
                Ok(active) => active,
                Err(code) => {
                    log::warn!(
                        "openxr: reading whether the {} hand pose is active failed: {}",
                        hand.name(),
                        error_text(&self.instance, code)
                    );
                    false
                }
            };
            if !active {
                continue;
            }
            let Some(space) = &slot.space else {
                // Creating the space failed when the sets were installed; the
                // hand stays untracked, which was already reported then.
                continue;
            };
            match space.locate(&self.world_space, display_time) {
                Ok(location) => {
                    poses[hand.index()] =
                        location_pose(posef_pose(&location.pose), location.location_flags);
                }
                Err(code) => log::debug!(
                    "openxr: locating the {} hand failed: {}",
                    hand.name(),
                    error_text(&self.instance, code)
                ),
            }
        }
        poses
    }

    /// Create the runtime's action sets, one runtime action per declared
    /// action, and the internal pose actions.
    fn create_actions(&self, sets: Vec<ActionSet>) -> XrResult<Installed> {
        let mut runtime_sets = Vec::with_capacity(sets.len());
        for set in &sets {
            let created = self
                .instance
                .create_action_set(&set.name, &set.localized_name, set.priority)
                .map_err(|code| runtime_error(&self.instance, code))?;
            runtime_sets.push(created);
        }

        let mut actions = Vec::new();
        for (index, set) in sets.iter().enumerate() {
            for action in set.actions() {
                let subactions = self.subactions(set, &action.name)?;
                let paths: Vec<XrPath> = subactions.iter().map(|(_, path)| *path).collect();
                let runtime_set = &runtime_sets[index];
                let runtime = match action.kind {
                    ActionKind::Boolean => RuntimeAction::Boolean(
                        runtime_set
                            .create_action(&action.name, &action.localized_name, &paths)
                            .map_err(|code| runtime_error(&self.instance, code))?,
                    ),
                    ActionKind::Float => RuntimeAction::Float(
                        runtime_set
                            .create_action(&action.name, &action.localized_name, &paths)
                            .map_err(|code| runtime_error(&self.instance, code))?,
                    ),
                    ActionKind::Vector2 => RuntimeAction::Vector2(
                        runtime_set
                            .create_action(&action.name, &action.localized_name, &paths)
                            .map_err(|code| runtime_error(&self.instance, code))?,
                    ),
                };
                actions.push(ActionSlot {
                    set: set.name.clone(),
                    action: action.name.clone(),
                    kind: action.kind,
                    runtime,
                    subactions,
                });
            }
        }

        Ok(Installed {
            devices: reported_devices(&sets),
            sets,
            runtime_sets,
            actions,
            poses: self.create_pose_actions()?,
        })
    }

    /// The device paths one action is bound on, as runtime paths.
    fn subactions(&self, set: &ActionSet, action: &str) -> XrResult<Vec<(UserPath, XrPath)>> {
        let mut subactions = Vec::new();
        for user in set.subaction_paths(action) {
            let text = format!("/user/{user}");
            let path = self
                .instance
                .string_to_path(&text)
                .map_err(|code| runtime_error(&self.instance, code))?;
            subactions.push((user.clone(), path));
        }
        Ok(subactions)
    }

    /// The internal action set, its four pose actions, and the subaction path
    /// each is declared with.
    fn create_pose_actions(&self) -> XrResult<PoseActions> {
        let set = self
            .instance
            .create_action_set(POSE_SET_NAME, POSE_SET_LOCALIZED_NAME, 0)
            .map_err(|code| runtime_error(&self.instance, code))?;
        let left = self.create_hand_poses(&set, Hand::Left)?;
        let right = self.create_hand_poses(&set, Hand::Right)?;
        Ok(PoseActions {
            set,
            hands: [left, right],
        })
    }

    fn create_hand_poses(&self, set: &XrActionSet, hand: Hand) -> XrResult<HandPoses> {
        Ok(HandPoses {
            aim: self.create_pose_action(set, hand, HandPose::Aim)?,
            grip: self.create_pose_action(set, hand, HandPose::Grip)?,
        })
    }

    fn create_pose_action(
        &self,
        set: &XrActionSet,
        hand: Hand,
        pose: HandPose,
    ) -> XrResult<PoseSlot> {
        let path = self
            .instance
            .string_to_path(&pose.input_path(hand).as_path())
            .map_err(|code| runtime_error(&self.instance, code))?;
        let name = format!("{}_{}", hand.name(), pose.input());
        let localized = format!("{} hand {} pose", hand.name(), pose.input());
        let action = set
            .create_action::<Posef>(&name, &localized, &[path])
            .map_err(|code| runtime_error(&self.instance, code))?;
        Ok(PoseSlot {
            action,
            path,
            space: None,
        })
    }

    /// Create the action spaces the pose actions are located through. Created
    /// after the attach, so every space belongs to an action the session can
    /// drive; a refusal leaves that hand untracked instead of failing the whole
    /// install.
    fn create_pose_spaces(&self, installed: &mut Installed) {
        for hand in Hand::ALL {
            for pose in HandPose::ALL {
                let slot = installed.poses.hands[hand.index()].get_mut(pose);
                let created = slot
                    .action
                    .create_space(&self.session, slot.path, Posef::IDENTITY);
                match created {
                    Ok(space) => slot.space = Some(space),
                    Err(code) => log::warn!(
                        "openxr: creating the {} hand {} space failed: {}",
                        hand.name(),
                        pose.input(),
                        error_text(&self.instance, code)
                    ),
                }
            }
        }
    }

    /// The profiles to register bindings for: every profile the sets declare,
    /// plus the app's fallbacks to that list, with
    /// [`InteractionProfile::COMMON`] when the app named none of its own.
    ///
    /// The runtime's own answer about a device is not part of this, and cannot
    /// be: it only reports a profile once the sets are attached, and the spec
    /// forbids suggesting bindings for an attached set. Nothing is missing
    /// either: a runtime reports only profiles the app suggested bindings for,
    /// or an emulation of one, so everything it can report is registered here,
    /// before the attach that freezes it.
    fn installed_profiles(&self, installed: &Installed) -> Vec<InteractionProfile> {
        let mut profiles: Vec<InteractionProfile> = Vec::new();
        for set in &installed.sets {
            for profile in set.profiles() {
                push_profile(&mut profiles, profile.clone());
            }
        }
        for profile in &self.fallbacks {
            push_profile(&mut profiles, profile.clone());
        }
        if profiles.is_empty() {
            for path in InteractionProfile::COMMON {
                match InteractionProfile::parse(path) {
                    Ok(profile) => profiles.push(profile),
                    Err(error) => log::warn!("openxr: {path} does not parse: {error}"),
                }
            }
        }
        profiles
    }

    /// Suggest the bindings one profile carries: everything the sets declare
    /// against it, then the pose bindings, each group in its own runtime call so
    /// a profile that carries only one of the two poses still gets that one.
    fn register_profile(
        &self,
        installed: &Installed,
        profile: &InteractionProfile,
        registration: Registration,
    ) {
        let text = profile.as_path();
        let Ok(profile_path) = self.instance.string_to_path(&text) else {
            log::warn!("openxr: the runtime refuses the interaction profile path {text}");
            return;
        };

        let mut bindings: Vec<XrBinding<'_>> = Vec::new();
        for set in &installed.sets {
            for (action, input) in set.bindings_for(&text) {
                let Some(slot) = installed
                    .actions
                    .iter()
                    .find(|slot| slot.set == set.name && slot.action == action.name)
                else {
                    continue;
                };
                let binding_text = input.as_path();
                match self.instance.string_to_path(&binding_text) {
                    Ok(path) => bindings.push(binding_of(&slot.runtime, path)),
                    Err(code) => log::warn!(
                        "openxr: the runtime refuses the binding path {binding_text}: {}",
                        error_text(&self.instance, code)
                    ),
                }
            }
        }
        if !bindings.is_empty() {
            match self
                .instance
                .suggest_interaction_profile_bindings(profile_path, &bindings)
            {
                Ok(()) => log::debug!(
                    "openxr: suggested {} binding(s) for {text}",
                    bindings.len()
                ),
                Err(code) => registration.refused(
                    &format!("{text}'s {} binding(s)", bindings.len()),
                    code,
                    &error_text(&self.instance, code),
                ),
            }
        }

        for pose in HandPose::ALL {
            let mut bindings: Vec<XrBinding<'_>> = Vec::new();
            for hand in Hand::ALL {
                let slot = installed.poses.hands[hand.index()].get(pose);
                let binding_text = pose.input_path(hand).as_path();
                match self.instance.string_to_path(&binding_text) {
                    Ok(path) => bindings.push(XrBinding::new(&slot.action, path)),
                    Err(code) => log::warn!(
                        "openxr: the runtime refuses the binding path {binding_text}: {}",
                        error_text(&self.instance, code)
                    ),
                }
            }
            match self
                .instance
                .suggest_interaction_profile_bindings(profile_path, &bindings)
            {
                Ok(()) => log::debug!(
                    "openxr: suggested {} {}-pose binding(s) for {text}",
                    bindings.len(),
                    pose.input()
                ),
                // A profile without one of the poses is possible, and so is a
                // runtime that takes neither; either way the hands bound here
                // read as untracked rather than as anything invented.
                Err(code) => log::debug!(
                    "openxr: the runtime refuses the {}-pose bindings for {text}: {}",
                    pose.input(),
                    error_text(&self.instance, code)
                ),
            }
        }
    }

    /// The runtime says a device's interaction profile changed: report what it
    /// now is, and suggest the bindings for it.
    ///
    /// Re-suggesting is asked for here rather than assumed away: a conformant
    /// runtime refuses it with `XR_ERROR_ACTIONSETS_ALREADY_ATTACHED`, because
    /// the attach that made the sets usable also made them immutable, which is
    /// why every declared profile was registered before it. A runtime that does
    /// take the call gets the bindings its binding UI needs for the profile it
    /// just reported; either way the runtime's answer is logged, not swallowed.
    fn profile_changed(&self) {
        let Some(installed) = self.installed.as_ref() else {
            return;
        };
        for device in &installed.devices {
            match self.query_profile(device) {
                Some(profile) => {
                    log::info!("openxr: /user/{device} is now a {profile}");
                    self.register_profile(installed, &profile, Registration::BestEffort);
                }
                None => log::debug!("openxr: /user/{device} has no interaction profile now"),
            }
        }
    }

    /// Ask the runtime which interaction profile a device is driving, or
    /// `None` when it reports none.
    fn query_profile(&self, user: &UserPath) -> Option<InteractionProfile> {
        let text = format!("/user/{user}");
        let path = match self.instance.string_to_path(&text) {
            Ok(path) => path,
            Err(code) => {
                log::warn!(
                    "openxr: the runtime refuses the user path {text}: {}",
                    error_text(&self.instance, code)
                );
                return None;
            }
        };
        let profile = match self.session.current_interaction_profile(path) {
            Ok(profile) => profile,
            Err(code) => {
                log::debug!(
                    "openxr: the runtime would not report the profile of {text}: {}",
                    error_text(&self.instance, code)
                );
                return None;
            }
        };
        if profile == XrPath::NULL {
            return None;
        }
        let reported = match self.instance.path_to_string(profile) {
            Ok(text) => text,
            Err(code) => {
                log::warn!(
                    "openxr: the runtime would not name an interaction profile: {}",
                    error_text(&self.instance, code)
                );
                return None;
            }
        };
        match InteractionProfile::parse(&reported) {
            Ok(profile) => Some(profile),
            Err(error) => {
                log::warn!("openxr: the runtime reported `{reported}` as a profile: {error}");
                None
            }
        }
    }

    /// Drain the runtime's events into the reported state.
    fn poll(&self) {
        let mut buffer = EventDataBuffer::new();
        loop {
            match self.instance.poll_event(&mut buffer) {
                Ok(Some(event)) => self.apply_event(&event),
                Ok(None) => return,
                Err(code) => {
                    log::warn!(
                        "openxr: polling events failed: {}",
                        error_text(&self.instance, code)
                    );
                    return;
                }
            }
        }
    }

    /// One event. A state this crate does not model keeps the last known one, so
    /// the session never claims to be in a state the runtime did not report.
    fn apply_event(&self, event: &Event<'_>) {
        match event {
            Event::SessionStateChanged(changed) => {
                if changed.session() != self.session.as_raw() {
                    return;
                }
                match session_state(changed.state()) {
                    Some(state) => {
                        log::debug!("openxr: session state {state:?}");
                        self.state.set(state);
                    }
                    None => log::warn!(
                        "openxr: runtime reported session state {:?}, which this crate does not \
                         model; keeping {:?}",
                        changed.state(),
                        self.state.get()
                    ),
                }
            }
            Event::InstanceLossPending(_) => {
                log::warn!("openxr: the runtime is going away; reporting the session as exiting");
                self.state.set(SessionState::Exiting);
            }
            Event::EventsLost(lost) => log::warn!(
                "openxr: the runtime dropped {} events",
                lost.lost_event_count()
            ),
            Event::InteractionProfileChanged(changed) => {
                if changed.session() != self.session.as_raw() {
                    return;
                }
                self.profile_changed();
            }
            // Everything else (visibility mask changes, performance hints) is
            // outside what this crate models.
            _ => {}
        }
    }
}

/// The one swapchain a session renders with, and the images it handed out.
struct SwapchainState {
    swapchain: Swapchain<Vulkan>,
    /// `VkImage` handles, enumerated once: they do not change over a session's
    /// life, so a frame only acquires an index.
    images: Vec<u64>,
    /// The raw `XrSwapchain` value, the identity a renderer keys its imported
    /// resources on.
    handle: u64,
    /// Extent of every array layer.
    size: [u32; 2],
}

/// One eye's slice of the shared swapchain.
struct EyeState {
    eye: Eye,
    /// `[x, y, width, height]` of this eye's image rect.
    rect: [u32; 4],
}

/// Everything the app's action sets became at the runtime.
struct Installed {
    /// The declared sets, kept because a profile change re-registers the
    /// bindings they declare.
    sets: Vec<ActionSet>,
    /// The runtime's own sets, in the same order, for `sync_actions`.
    runtime_sets: Vec<XrActionSet>,
    /// One entry per declared action, in declaration order.
    actions: Vec<ActionSlot>,
    /// The internal pose actions hands are read through.
    poses: PoseActions,
    /// Devices whose interaction profile the runtime reports.
    devices: Vec<UserPath>,
}

/// One declared action and how a sync reads it.
struct ActionSlot {
    /// Owning set name, as the app declared it.
    set: String,
    /// Action name, as the app declared it.
    action: String,
    kind: ActionKind,
    runtime: RuntimeAction,
    /// One entry per device the action is bound on, in binding declaration
    /// order.
    subactions: Vec<(UserPath, XrPath)>,
}

/// A runtime action of one of the three kinds this crate models.
enum RuntimeAction {
    Boolean(XrAction<bool>),
    Float(XrAction<f32>),
    Vector2(XrAction<Vector2f>),
}

/// The internal action set the hand pose actions live in.
struct PoseActions {
    set: XrActionSet,
    /// One entry per hand, in [`Hand::ALL`] order.
    hands: [HandPoses; 2],
}

/// How a refused binding suggestion is reported.
///
/// Installing the sets is where the app's own declarations go in, and a runtime
/// refusing one of them is worth a warning. A suggestion made after the attach
/// can only be best effort: the spec makes an attached set immutable and
/// requires the runtime to refuse new bindings for it, so nothing is wrong with
/// the app when that refusal comes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Registration {
    /// Bindings the app declared, registered before the sets are attached.
    Declared,
    /// Bindings suggested after the sets were attached, where a runtime allows
    /// it.
    BestEffort,
}

impl Registration {
    /// Log a runtime's refusal at the level this registration calls for.
    fn refused(self, what: &str, code: sys::Result, detail: &str) {
        match self {
            Registration::Declared => {
                log::warn!("openxr: the runtime refused {what}: {detail}");
            }
            Registration::BestEffort
                if code == sys::Result::ERROR_ACTIONSETS_ALREADY_ATTACHED =>
            {
                log::debug!(
                    "openxr: {what} was not re-registered: action sets are immutable once \
                     attached, so the bindings suggested before the attach are the ones in use"
                );
            }
            Registration::BestEffort => {
                log::debug!("openxr: the runtime refused {what}: {detail}");
            }
        }
    }
}

/// One hand's two pose actions: the one it points with and the one it is held
/// with.
struct HandPoses {
    aim: PoseSlot,
    grip: PoseSlot,
}

impl HandPoses {
    fn get(&self, pose: HandPose) -> &PoseSlot {
        match pose {
            HandPose::Aim => &self.aim,
            HandPose::Grip => &self.grip,
        }
    }

    fn get_mut(&mut self, pose: HandPose) -> &mut PoseSlot {
        match pose {
            HandPose::Aim => &mut self.aim,
            HandPose::Grip => &mut self.grip,
        }
    }
}

/// A pose action, the subaction path declared for it, and the space created
/// from it once the set carrying it is attached.
struct PoseSlot {
    action: XrAction<Posef>,
    /// The one subaction path the action was created with.
    path: XrPath,
    /// `None` when the runtime refused to create the space; the hand then stays
    /// untracked.
    space: Option<Space>,
}

/// Load the OpenXR loader, or say why it is not there.
fn load_entry() -> XrResult<Entry> {
    match unsafe { Entry::load(&()) } {
        Ok(entry) => Ok(entry),
        Err(EntryError::Load(error)) => {
            log::debug!("openxr: loading the loader failed: {error}");
            Err(XrError::Unavailable {
                backend: NAME,
                detail: "no OpenXR loader (libopenxr_loader) on this machine".to_string(),
            })
        }
        Err(EntryError::Xr(code)) => Err(XrError::Runtime {
            backend: NAME,
            code: code.into_raw() as i64,
            detail: format!("the OpenXR loader refused to initialize: {code}"),
        }),
    }
}

/// The instance extensions this backend renders through.
fn vulkan_extensions() -> ExtensionSet {
    let mut extensions = ExtensionSet::default();
    extensions.khr_vulkan_enable2 = true;
    extensions
}

/// Refuse a runtime that cannot render with the app's Vulkan device before
/// creating an instance that would only fail later.
fn require_vulkan(entry: &Entry) -> XrResult<()> {
    let extensions = entry.enumerate_extensions().map_err(|code| XrError::Unavailable {
        backend: NAME,
        detail: format!("the OpenXR loader could not list extensions: {code}"),
    })?;
    if !extensions.khr_vulkan_enable2 {
        return Err(XrError::Unsupported {
            backend: NAME,
            detail: "XR_KHR_vulkan_enable2, which this backend renders through".to_string(),
        });
    }
    Ok(())
}

/// Application info, with the versions packed the way the runtime reads them.
fn app_info<'a>(
    app_name: &'a str,
    app_version: (u16, u16, u32),
    engine_name: &'a str,
    engine_version: (u16, u16, u32),
) -> ApplicationInfo<'a> {
    ApplicationInfo {
        application_name: app_name,
        application_version: SessionConfig::packed_version(app_version),
        engine_name,
        engine_version: SessionConfig::packed_version(engine_version),
        api_version: ::openxr::Version::new(API_VERSION.0, API_VERSION.1, API_VERSION.2),
    }
}

/// Names are copied into fixed-size runtime buffers, so an over-long one is
/// refused rather than truncated or panicked on.
fn check_names(config: &SessionConfig) -> XrResult<()> {
    if config.app_name.len() >= sys::MAX_APPLICATION_NAME_SIZE {
        return Err(XrError::Unsupported {
            backend: NAME,
            detail: format!(
                "an application name of {} bytes or more",
                sys::MAX_APPLICATION_NAME_SIZE
            ),
        });
    }
    if config.engine_name.len() >= sys::MAX_ENGINE_NAME_SIZE {
        return Err(XrError::Unsupported {
            backend: NAME,
            detail: format!(
                "an engine name of {} bytes or more",
                sys::MAX_ENGINE_NAME_SIZE
            ),
        });
    }
    Ok(())
}

/// The runtime's own words for a result code, or the code itself when it cannot
/// be asked.
fn error_text(instance: &Instance, code: sys::Result) -> String {
    instance
        .result_to_string(code)
        .unwrap_or_else(|_| code.to_string())
}

/// A refusal from a runtime that answered.
fn runtime_error(instance: &Instance, code: sys::Result) -> XrError {
    XrError::Runtime {
        backend: NAME,
        code: code.into_raw() as i64,
        detail: error_text(instance, code),
    }
}

/// Map the runtime's session state onto this crate's. `None` is a state this
/// crate does not model, including `XR_SESSION_STATE_UNKNOWN`.
fn session_state(state: sys::SessionState) -> Option<SessionState> {
    use sys::SessionState as Raw;
    match state {
        state if state == Raw::IDLE => Some(SessionState::Idle),
        state if state == Raw::READY => Some(SessionState::Ready),
        state if state == Raw::SYNCHRONIZED => Some(SessionState::Synchronized),
        state if state == Raw::VISIBLE => Some(SessionState::Visible),
        state if state == Raw::FOCUSED => Some(SessionState::Focused),
        state if state == Raw::STOPPING => Some(SessionState::Stopping),
        state if state == Raw::LOSS_PENDING => Some(SessionState::Lost),
        state if state == Raw::EXITING => Some(SessionState::Exiting),
        _ => None,
    }
}

/// A view's pose plus the flags the runtime reported for the whole view list: a
/// pose is usable only when both components are valid, and is only *measured*
/// when both tracked bits are set. Anything else is emulated or absent.
fn view_pose(view: &::openxr::View, flags: ViewStateFlags) -> TrackedPose {
    let pose = posef_pose(&view.pose);
    let valid = flags.contains(ViewStateFlags::ORIENTATION_VALID | ViewStateFlags::POSITION_VALID);
    let tracked = valid
        && flags.contains(ViewStateFlags::ORIENTATION_TRACKED | ViewStateFlags::POSITION_TRACKED);
    TrackedPose::from_tracking_flags(pose, valid, tracked)
}

/// This crate's pose for the runtime's.
fn posef_pose(pose: &Posef) -> Pose {
    Pose::from_xyzw(
        [pose.position.x, pose.position.y, pose.position.z],
        [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
    )
}

/// The runtime's suggested binding for one of this crate's actions.
fn binding_of<'a>(action: &'a RuntimeAction, path: XrPath) -> XrBinding<'a> {
    match action {
        RuntimeAction::Boolean(action) => XrBinding::new(action, path),
        RuntimeAction::Float(action) => XrBinding::new(action, path),
        RuntimeAction::Vector2(action) => XrBinding::new(action, path),
    }
}

/// Add a profile to a list unless it is already there, so a profile the app
/// declared twice is registered once.
fn push_profile(profiles: &mut Vec<InteractionProfile>, profile: InteractionProfile) {
    if !profiles.contains(&profile) {
        profiles.push(profile);
    }
}

/// Every device whose interaction profile matters: the ones the bindings name,
/// and both hands, whose pose actions are bound wherever anything is
/// registered.
fn reported_devices(sets: &[ActionSet]) -> Vec<UserPath> {
    let mut devices: Vec<UserPath> = Vec::new();
    for set in sets {
        for binding in set.bindings() {
            if !devices.contains(&binding.input.user) {
                devices.push(binding.input.user.clone());
            }
        }
    }
    for hand in Hand::ALL {
        let user = UserPath::Hand(hand);
        if !devices.contains(&user) {
            devices.push(user);
        }
    }
    devices
}

/// The same mapping for a space location, whose flags mean the same thing.
fn location_pose(pose: Pose, flags: sys::SpaceLocationFlags) -> TrackedPose {
    let valid = flags
        .contains(sys::SpaceLocationFlags::ORIENTATION_VALID | sys::SpaceLocationFlags::POSITION_VALID);
    let tracked = valid
        && flags.contains(
            sys::SpaceLocationFlags::ORIENTATION_TRACKED
                | sys::SpaceLocationFlags::POSITION_TRACKED,
        );
    TrackedPose::from_tracking_flags(pose, valid, tracked)
}

/// The runtime's asymmetric fov, as this crate models it.
fn fov_of(fov: Fovf) -> Fov {
    Fov {
        angle_left: fov.angle_left,
        angle_right: fov.angle_right,
        angle_up: fov.angle_up,
        angle_down: fov.angle_down,
    }
}

/// This crate's pixel format for a Vulkan format, or `None` when a renderer
/// could not import it.
fn vk_pixel_format(format: u32) -> Option<PixelFormat> {
    match format {
        VK_FORMAT_R8G8B8A8_UNORM => Some(PixelFormat::Rgba8Unorm),
        VK_FORMAT_R8G8B8A8_SRGB => Some(PixelFormat::Rgba8UnormSrgb),
        VK_FORMAT_B8G8R8A8_UNORM => Some(PixelFormat::Bgra8Unorm),
        VK_FORMAT_B8G8R8A8_SRGB => Some(PixelFormat::Bgra8UnormSrgb),
        VK_FORMAT_R16G16B16A16_SFLOAT => Some(PixelFormat::Rgba16Float),
        _ => None,
    }
}

/// Pick the swapchain format: this crate's preference order, intersected with
/// what the runtime offers.
fn choose_format(offered: &[u32]) -> Option<u32> {
    FORMAT_PREFERENCE.iter().find_map(|wanted| {
        offered
            .iter()
            .copied()
            .find(|format| vk_pixel_format(*format) == Some(*wanted))
    })
}

/// The sample count to create the swapchain with: the runtime's recommendation,
/// clamped to what it allows and never zero.
fn swapchain_sample_count(views: &[::openxr::ViewConfigurationView]) -> u32 {
    views.iter().fold(1, |count, view| {
        let wanted = view.recommended_swapchain_sample_count.max(1);
        let allowed = view.max_swapchain_sample_count.max(1);
        count.max(wanted.min(allowed))
    })
}

/// The first blend mode the app accepts that the runtime offers.
fn choose_blend_mode(
    offered: &[EnvironmentBlendMode],
    accepted: &[BlendMode],
) -> Option<EnvironmentBlendMode> {
    accepted.iter().find_map(|mode| {
        let wanted = environment_blend_mode(*mode);
        offered.iter().copied().find(|mode| *mode == wanted)
    })
}

/// This crate's blend mode for the runtime's.
fn blend_of(mode: EnvironmentBlendMode) -> BlendMode {
    if mode == EnvironmentBlendMode::ADDITIVE {
        BlendMode::Additive
    } else if mode == EnvironmentBlendMode::ALPHA_BLEND {
        BlendMode::AlphaBlend
    } else {
        BlendMode::Opaque
    }
}

/// The runtime's blend mode for this crate's.
fn environment_blend_mode(mode: BlendMode) -> EnvironmentBlendMode {
    match mode {
        BlendMode::Opaque => EnvironmentBlendMode::OPAQUE,
        BlendMode::Additive => EnvironmentBlendMode::ADDITIVE,
        BlendMode::AlphaBlend => EnvironmentBlendMode::ALPHA_BLEND,
    }
}

/// Views this crate presents, in the runtime's own enumeration.
fn view_configuration_type(configuration: ViewConfiguration) -> ViewConfigurationType {
    match configuration {
        ViewConfiguration::PrimaryStereo => ViewConfigurationType::PRIMARY_STEREO,
        ViewConfiguration::PrimaryMono => ViewConfigurationType::PRIMARY_MONO,
    }
}

/// A name for the view configuration, for error text.
fn view_configuration_name(configuration: ViewConfiguration) -> &'static str {
    match configuration {
        ViewConfiguration::PrimaryStereo => "primary stereo",
        ViewConfiguration::PrimaryMono => "primary mono",
    }
}

/// The reference space the app asked for.
fn reference_space_type(space: ReferenceSpace) -> ::openxr::ReferenceSpaceType {
    match space {
        ReferenceSpace::View => ::openxr::ReferenceSpaceType::VIEW,
        ReferenceSpace::Local => ::openxr::ReferenceSpaceType::LOCAL,
        ReferenceSpace::Stage => ::openxr::ReferenceSpaceType::STAGE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pose::PoseState;

    fn view() -> ::openxr::View {
        ::openxr::View {
            pose: Posef::IDENTITY,
            fov: Fovf {
                angle_left: -0.5,
                angle_right: 0.5,
                angle_up: 0.5,
                angle_down: -0.5,
            },
        }
    }

    #[test]
    fn view_flags_map_to_tracked_emulated_and_untracked() {
        let view = view();
        let valid = ViewStateFlags::ORIENTATION_VALID | ViewStateFlags::POSITION_VALID;
        let tracked = ViewStateFlags::ORIENTATION_TRACKED | ViewStateFlags::POSITION_TRACKED;

        let untracked = view_pose(&view, ViewStateFlags::EMPTY);
        assert_eq!(untracked.state, PoseState::Untracked);
        assert!(untracked.pose.is_none());

        let measured = view_pose(&view, valid | tracked);
        assert_eq!(measured.state, PoseState::Tracked);
        assert!(measured.pose.is_some());

        let inferred = view_pose(&view, valid | ViewStateFlags::ORIENTATION_TRACKED);
        assert_eq!(inferred.state, PoseState::Emulated);
        assert!(inferred.pose.is_some());

        // A tracked bit without a valid one is not a usable pose, and half of a
        // pose is not a pose.
        assert_eq!(
            view_pose(&view, ViewStateFlags::ORIENTATION_TRACKED).state,
            PoseState::Untracked
        );
        let partial = view_pose(&view, ViewStateFlags::ORIENTATION_VALID);
        assert_eq!(partial.state, PoseState::Untracked);
        assert!(partial.pose.is_none());
    }

    #[test]
    fn vk_formats_map_and_are_preferred_in_order() {
        assert_eq!(
            vk_pixel_format(VK_FORMAT_R8G8B8A8_UNORM),
            Some(PixelFormat::Rgba8Unorm)
        );
        assert_eq!(
            vk_pixel_format(VK_FORMAT_R8G8B8A8_SRGB),
            Some(PixelFormat::Rgba8UnormSrgb)
        );
        assert_eq!(
            vk_pixel_format(VK_FORMAT_B8G8R8A8_UNORM),
            Some(PixelFormat::Bgra8Unorm)
        );
        assert_eq!(
            vk_pixel_format(VK_FORMAT_B8G8R8A8_SRGB),
            Some(PixelFormat::Bgra8UnormSrgb)
        );
        assert_eq!(
            vk_pixel_format(VK_FORMAT_R16G16B16A16_SFLOAT),
            Some(PixelFormat::Rgba16Float)
        );
        assert_eq!(vk_pixel_format(0), None);
        assert_eq!(vk_pixel_format(100), None);

        assert_eq!(
            choose_format(&[
                VK_FORMAT_R8G8B8A8_UNORM,
                VK_FORMAT_R8G8B8A8_SRGB,
                VK_FORMAT_B8G8R8A8_SRGB
            ]),
            Some(VK_FORMAT_B8G8R8A8_SRGB)
        );
        assert_eq!(
            choose_format(&[VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM]),
            Some(VK_FORMAT_R16G16B16A16_SFLOAT)
        );
        assert_eq!(choose_format(&[1, 2, 3]), None);
        assert_eq!(choose_format(&[]), None);
    }

    #[test]
    fn openxr_session_states_map_one_for_one() {
        use sys::SessionState as Raw;
        assert_eq!(session_state(Raw::IDLE), Some(SessionState::Idle));
        assert_eq!(session_state(Raw::READY), Some(SessionState::Ready));
        assert_eq!(
            session_state(Raw::SYNCHRONIZED),
            Some(SessionState::Synchronized)
        );
        assert_eq!(session_state(Raw::VISIBLE), Some(SessionState::Visible));
        assert_eq!(session_state(Raw::FOCUSED), Some(SessionState::Focused));
        assert_eq!(session_state(Raw::STOPPING), Some(SessionState::Stopping));
        assert_eq!(session_state(Raw::LOSS_PENDING), Some(SessionState::Lost));
        assert_eq!(session_state(Raw::EXITING), Some(SessionState::Exiting));
        assert_eq!(session_state(Raw::UNKNOWN), None);
        assert_eq!(session_state(Raw::from_raw(-1)), None);
    }
}
