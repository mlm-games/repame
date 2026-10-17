//! The single error type every XR call returns.

use std::fmt;

use crate::{ActionKind, SessionState};

/// Why a binding path could not be parsed. Carried inside [`XrError::BadPath`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathError {
    /// The path was empty.
    Empty,
    /// The path did not start with `/`.
    NoLeadingSlash,
    /// The path ended with `/`, or held an empty segment.
    EmptySegment,
    /// A `/user/...` path was expected and this segment was not `user`.
    NotAUserPath,
    /// The user path named no device repame knows.
    UnknownUser,
    /// An interaction profile path had no vendor or no profile name.
    IncompleteProfile,
    /// A path that has to name a controller layout carries no profile prefix.
    MissingProfile,
    /// `input` was missing between the user path and the input name.
    MissingInput,
    /// The input name or component name was empty.
    MissingName,
}

impl fmt::Display for PathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            PathError::Empty => "path is empty",
            PathError::NoLeadingSlash => "path does not start with `/`",
            PathError::EmptySegment => "path has an empty segment",
            PathError::NotAUserPath => "path does not name a `/user/...` source",
            PathError::UnknownUser => "path names an unknown user device",
            PathError::IncompleteProfile => "interaction profile needs a vendor and a name",
            PathError::MissingProfile => "binding path needs an interaction profile prefix",
            PathError::MissingInput => "path has no `input` component",
            PathError::MissingName => "path component has no name",
        };
        f.write_str(text)
    }
}

/// Everything that can go wrong at the XR runtime boundary.
///
/// The variants are deliberately about *where* the failure happened, because
/// that is what a caller has to act on: [`XrError::Unavailable`] means there is
/// no runtime to talk to and the app should run flat, while a
/// [`XrError::Runtime`] code means a runtime answered and refused.
#[derive(Clone, Debug, PartialEq)]
pub enum XrError {
    /// No XR runtime is reachable for this backend, or the backend compiled into
    /// this build cannot do XR at all. Every runtime call returns this when no
    /// headset stack is installed, which is the normal desktop case.
    Unavailable {
        /// Backend that answered.
        backend: &'static str,
        /// Why, in the runtime's or the loader's own words.
        detail: String,
    },
    /// The backend wants an asynchronous step the caller has not completed yet,
    /// e.g. a WebXR frame has not been handed in.
    NotReady {
        /// Which call or resource is waiting.
        what: &'static str,
    },
    /// A call that needs a session was made with none created.
    NoSession,
    /// The session is in the wrong state for this call.
    WrongState {
        /// State the called method requires.
        expected: &'static str,
        /// State the session is actually in.
        actual: SessionState,
    },
    /// The runtime answered and refused. `code` is the backend's own result
    /// code, kept verbatim so it can be looked up in that runtime's table.
    Runtime {
        /// Backend that answered.
        backend: &'static str,
        /// Negative runtime result code.
        code: i64,
        /// Text the backend derived for the code.
        detail: String,
    },
    /// The runtime does not support something the call needs.
    Unsupported {
        /// Backend that answered.
        backend: &'static str,
        /// What is missing.
        detail: String,
    },
    /// The runtime could not track the requested device.
    Untracked {
        /// Device name, e.g. `"left hand"`.
        what: &'static str,
    },
    /// The runtime returned data this crate refuses to hand on: non-finite
    /// components, a zero-size image, an empty view list.
    InvalidData {
        /// What was wrong with it.
        what: &'static str,
    },
    /// A binding path or interaction profile path was malformed.
    BadPath {
        /// The offending text.
        path: String,
        /// What was wrong.
        reason: PathError,
    },
    /// No action set carries that action.
    UnknownAction {
        /// Action set looked up.
        set: String,
        /// Action looked up.
        action: String,
    },
    /// An action was read as a type it was not declared as.
    TypeMismatch {
        /// Action read.
        action: String,
        /// Type the caller asked for.
        expected: ActionKind,
        /// Type the action was declared with.
        actual: ActionKind,
    },
    /// Action sets were installed on a session that already has them. A
    /// session binds its sets once; a second install would be a silent merge,
    /// so it is refused instead.
    ActionSetsInstalled,
}

impl XrError {
    /// True when the error means "there is no XR stack here", the one case an
    /// app that supports both flat and XR screens handles by carrying on flat.
    pub fn is_unavailable(&self) -> bool {
        matches!(self, XrError::Unavailable { .. })
    }

    /// Short name of the variant, for logs and tests.
    pub fn kind(&self) -> &'static str {
        match self {
            XrError::Unavailable { .. } => "unavailable",
            XrError::NotReady { .. } => "not_ready",
            XrError::NoSession => "no_session",
            XrError::WrongState { .. } => "wrong_state",
            XrError::Runtime { .. } => "runtime",
            XrError::Unsupported { .. } => "unsupported",
            XrError::Untracked { .. } => "untracked",
            XrError::InvalidData { .. } => "invalid_data",
            XrError::BadPath { .. } => "bad_path",
            XrError::UnknownAction { .. } => "unknown_action",
            XrError::TypeMismatch { .. } => "type_mismatch",
            XrError::ActionSetsInstalled => "action_sets_installed",
        }
    }
}

impl fmt::Display for XrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            XrError::Unavailable { backend, detail } => {
                write!(f, "xr {backend} unavailable: {detail}")
            }
            XrError::NotReady { what } => write!(f, "xr {what} is not ready yet"),
            XrError::NoSession => f.write_str("xr call needs a session"),
            XrError::WrongState { expected, actual } => {
                write!(f, "xr session is {actual:?}, expected {expected}")
            }
            XrError::Runtime {
                backend,
                code,
                detail,
            } => write!(f, "xr {backend} returned {code}: {detail}"),
            XrError::Unsupported { backend, detail } => {
                write!(f, "xr {backend} does not support {detail}")
            }
            XrError::Untracked { what } => write!(f, "xr {what} is not tracked"),
            XrError::InvalidData { what } => write!(f, "xr runtime returned bad {what}"),
            XrError::BadPath { path, reason } => {
                write!(f, "xr binding path `{path}`: {reason}")
            }
            XrError::UnknownAction { set, action } => {
                write!(f, "xr action `{action}` is not in set `{set}`")
            }
            XrError::TypeMismatch {
                action,
                expected,
                actual,
            } => write!(
                f,
                "xr action `{action}` is {actual:?}, read as {expected:?}"
            ),
            XrError::ActionSetsInstalled => {
                f.write_str("xr action sets are already installed on this session")
            }
        }
    }
}

impl std::error::Error for XrError {}

/// Result alias for this crate.
pub type XrResult<T> = Result<T, XrError>;
