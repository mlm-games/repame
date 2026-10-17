//! Session state and configuration.
//!
//! The state enum mirrors `XrSessionState` in the OpenXR spec, so a backend maps
//! its runtime's states across one-for-one and the ordering carries meaning:
//! a session walks `Idle` -> `Ready` -> `Synchronized` -> `Visible` -> `Focused`
//! and back. Rendering is only legal in `Visible` and `Focused`.

/// Where a session is in its lifecycle.
///
/// Stops are included that a game rarely drives itself: `Stopping` while the
/// runtime is pausing the session, `Lost` when tracking was lost, and `Exiting`
/// when the runtime wants the app to shut down.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SessionState {
    /// No session has been created, or it has been fully closed.
    #[default]
    Idle,
    /// The runtime is ready for the app to begin the session.
    Ready,
    /// The session is running but the runtime does not consider it visible yet.
    Synchronized,
    /// Frames submitted are shown to the user.
    Visible,
    /// Visible and receiving input focus.
    Focused,
    /// The runtime asked to pause; the app stops submitting frames.
    Stopping,
    /// The session is no longer tracking, e.g. the headset was removed.
    Lost,
    /// The runtime is shutting the app's XR session down.
    Exiting,
}

impl SessionState {
    /// True when rendering and submitting a frame is legal.
    pub fn is_running(self) -> bool {
        matches!(self, SessionState::Visible | SessionState::Focused)
    }

    /// True while the runtime expects frames: running, plus synchronized, where
    /// a frame may be submitted but nothing is shown yet.
    pub fn should_submit_frames(self) -> bool {
        matches!(
            self,
            SessionState::Synchronized | SessionState::Visible | SessionState::Focused
        )
    }

    /// True when `begin` is the right call next.
    pub fn can_begin(self) -> bool {
        matches!(self, SessionState::Ready)
    }

    /// True when `end` is the right call next.
    pub fn can_end(self) -> bool {
        matches!(
            self,
            SessionState::Synchronized | SessionState::Visible | SessionState::Focused
        )
    }

    /// True when the session can no longer be driven and must be recreated.
    pub fn is_terminal(self) -> bool {
        matches!(self, SessionState::Idle | SessionState::Exiting)
    }
}

/// What a session needs at creation: names, versions, and the shape of the views
/// the app wants.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionConfig {
    /// Application name reported to the runtime.
    pub app_name: String,
    /// `(major, minor, patch)` application version.
    pub app_version: (u16, u16, u32),
    /// Engine name reported to the runtime.
    pub engine_name: String,
    /// `(major, minor, patch)` engine version.
    pub engine_version: (u16, u16, u32),
    /// Views to present.
    pub view_configuration: crate::ViewConfiguration,
    /// Compositing modes the app accepts, best first. A runtime picks the first
    /// one it supports.
    pub blend_modes: Vec<crate::BlendMode>,
    /// Graphics device the runtime renders with, when the render layer has one
    /// to hand over.
    ///
    /// An OpenXR session is created against handles the app already owns, so a
    /// build with the `openxr` feature needs this to be set before
    /// [`crate::XrSession::open`] can create a session; XR cannot pick a device
    /// on the render layer's behalf. A backend that takes its binding some
    /// other way (WebXR takes the WebGL2 context) leaves this `None`.
    pub graphics: Option<crate::GraphicsBinding>,
    /// Near plane of the projection handed to the renderer, in meters.
    pub near: f32,
    /// Far plane of that projection, in meters. A value at or below
    /// [`SessionConfig::near`] places the far plane at infinity.
    pub far: f32,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            app_name: "repame".to_string(),
            app_version: (0, 0, 0),
            engine_name: "repame".to_string(),
            engine_version: (0, 0, 0),
            view_configuration: crate::ViewConfiguration::PrimaryStereo,
            blend_modes: vec![crate::BlendMode::Opaque],
            graphics: None,
            near: 0.05,
            far: 100.0,
        }
    }
}

impl SessionConfig {
    /// Version packed into a `u32` the way Vulkan and the runtime's informational
    /// version fields do it: 10 bits major, 10 bits minor, 12 bits patch.
    pub fn packed_version(version: (u16, u16, u32)) -> u32 {
        ((version.0 as u32 & 0x3FF) << 22)
            | ((version.1 as u32 & 0x3FF) << 12)
            | (version.2 & 0xFFF)
    }
}
