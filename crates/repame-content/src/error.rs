//! Typed content errors.
//!
//! Every failure carries the path it happened on and a coarse [`ContentErrorKind`],
//! so callers can branch on "this is deliberately not implemented" without
//! string matching. Repame never approximates a format it does not implement:
//! it returns [`ContentErrorKind::Unsupported`].

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

/// Coarse classification of a [`ContentError`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContentErrorKind {
    /// The underlying filesystem or stream failed.
    Io,
    /// The named asset, manifest or archive is not there. Distinct from [`Io`]
    /// so a caller can substitute a placeholder for an absent asset without
    /// matching on message text.
    ///
    /// [`Io`]: ContentErrorKind::Io
    Missing,
    /// The bytes were read but did not match the expected structure.
    Format,
    /// The structure is understood, but the format or feature is deliberately
    /// not implemented. Never produced instead of a lossy approximation.
    Unsupported,
    /// Validation or any other failure.
    Other,
}

#[derive(Debug)]
pub struct ContentError {
    path: PathBuf,
    message: String,
    kind: ContentErrorKind,
}

impl ContentError {
    pub fn new(path: impl Into<PathBuf>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
            kind: ContentErrorKind::Other,
        }
    }

    /// An error with an explicit [`ContentErrorKind`].
    pub fn of(
        path: impl Into<PathBuf>,
        kind: ContentErrorKind,
        message: impl Into<String>,
    ) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
            kind,
        }
    }

    /// A format or feature that is recognised but deliberately not implemented.
    pub fn unsupported(path: impl Into<PathBuf>, message: impl Into<String>) -> Self {
        Self::of(
            path,
            ContentErrorKind::Unsupported,
            format!("unsupported: {}", message.into()),
        )
    }

    /// The named file, asset, manifest or archive is not there. `what` names
    /// the kind of thing that is missing; the path is the caller's key, so a
    /// missing asset reports its content path and not an absolute location.
    pub fn missing(path: impl Into<PathBuf>, what: impl Into<String>) -> Self {
        Self::of(
            path,
            ContentErrorKind::Missing,
            format!("{} is missing", what.into()),
        )
    }

    pub fn is_missing(&self) -> bool {
        self.kind == ContentErrorKind::Missing
    }

    /// A malformed structure.
    pub fn format(path: impl Into<PathBuf>, message: impl Into<String>) -> Self {
        Self::of(path, ContentErrorKind::Format, message)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn kind(&self) -> ContentErrorKind {
        self.kind
    }

    pub fn is_unsupported(&self) -> bool {
        self.kind == ContentErrorKind::Unsupported
    }
}

impl fmt::Display for ContentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path.display(), self.message)
    }
}

impl std::error::Error for ContentError {}

pub(crate) fn io_error(path: &Path, action: impl Into<String>, error: io::Error) -> ContentError {
    ContentError::of(path, ContentErrorKind::Io, format!("{}: {error}", action.into()))
}

pub(crate) fn ron_error(path: &Path, error: impl fmt::Display) -> ContentError {
    ContentError::format(path, format!("invalid RON: {error}"))
}
