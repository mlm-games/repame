//! Typed failures for the whole save path.
//!
//! Nothing on the path is dropped, defaulted, or logged and forgotten: a file
//! stamped with a schema this binary does not know is a
//! [`DocError::ForwardVersion`] (surfaced by the file layer as
//! [`SaveError::ForwardVersion`]), never a partial parse, and a payload that
//! does not fit the requested type is a [`DocError::Payload`], never a silently
//! default save.
//!
//! [`DocError`] is the document layer, which knows no paths because it can be
//! used on a string that never had a file behind it. [`SaveError`] is the same
//! failures plus the path they happened on, so a game can log one line that says
//! which file broke.

use std::fmt;
use std::path::{Path, PathBuf};

use repame_reflect::ReflectError;

/// Why one migration step refused to run.
#[derive(Debug)]
pub enum MigrationError {
    /// The payload is not the shape the step expects; usually a parse failure,
    /// which [`From<ron::error::SpannedError>`](MigrationError) converts.
    Shape {
        /// What was wrong.
        message: String,
    },
    /// The step refused deliberately: an old value it cannot represent in the
    /// new schema. Refusing is an error, not a reason to guess.
    Refused {
        /// What the step would not do.
        message: String,
    },
    /// The reflected world document did not load (see
    /// [`repame_reflect::ReflectError`]).
    Reflect(ReflectError),
}

impl MigrationError {
    /// A payload this step does not understand.
    pub fn shape(message: impl Into<String>) -> Self {
        Self::Shape {
            message: message.into(),
        }
    }

    /// A deliberate refusal.
    pub fn refused(message: impl Into<String>) -> Self {
        Self::Refused {
            message: message.into(),
        }
    }
}

impl From<ron::error::SpannedError> for MigrationError {
    fn from(source: ron::error::SpannedError) -> Self {
        Self::shape(source.to_string())
    }
}

impl From<ron::Error> for MigrationError {
    fn from(source: ron::Error) -> Self {
        Self::shape(source.to_string())
    }
}

impl From<ReflectError> for MigrationError {
    fn from(source: ReflectError) -> Self {
        Self::Reflect(source)
    }
}

impl fmt::Display for MigrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape { message } => write!(f, "payload shape: {message}"),
            Self::Refused { message } => write!(f, "refused: {message}"),
            Self::Reflect(source) => write!(f, "world document: {source}"),
        }
    }
}

impl std::error::Error for MigrationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Reflect(source) => Some(source),
            _ => None,
        }
    }
}

/// Why a document failed, without a path.
#[derive(Debug)]
pub enum DocError {
    /// The text is not a save document: not `(schema: N, body: "...")`.
    Parse(ron::error::SpannedError),
    /// The payload text did not deserialize into the type asked for.
    Payload(ron::error::SpannedError),
    /// The payload could not be serialized.
    Encode(ron::Error),
    /// The document is stamped with a newer schema than this binary writes.
    ///
    /// The document is refused whole. A file from a newer build is never
    /// partially parsed, because dropping the fields this binary does not know
    /// about would destroy them on the next save.
    ForwardVersion {
        /// The schema in the document.
        found: u32,
        /// The schema this binary writes.
        current: u32,
    },
    /// No migration step is registered for a version bump on the way to the
    /// current schema.
    MissingMigration {
        /// The version the chain has no step for.
        from: u32,
        /// The version being migrated to.
        to: u32,
    },
    /// A migration step ran and refused.
    Migration {
        /// The version the failing step migrates from.
        from: u32,
        /// What the step reported.
        source: MigrationError,
    },
}

impl fmt::Display for DocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(source) => write!(f, "not a save document: {source}"),
            Self::Payload(source) => write!(f, "payload: {source}"),
            Self::Encode(source) => write!(f, "payload could not be written: {source}"),
            Self::ForwardVersion { found, current } => write!(
                f,
                "schema {found} is newer than {current}; refusing to parse it"
            ),
            Self::MissingMigration { from, to } => {
                write!(f, "no migration step from schema {from} to {to}")
            }
            Self::Migration { from, source } => {
                write!(f, "migration from schema {from}: {source}")
            }
        }
    }
}

impl std::error::Error for DocError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Parse(source) | Self::Payload(source) => Some(source),
            Self::Encode(source) => Some(source),
            Self::Migration { source, .. } => Some(source),
            Self::ForwardVersion { .. } | Self::MissingMigration { .. } => None,
        }
    }
}

/// Why a save file could not be loaded or written.
#[derive(Debug)]
pub enum SaveError {
    /// The file is there but could not be read (a lock, a device error), so
    /// nothing was parsed and no backup was adopted: a file that may still be
    /// intact is never overwritten.
    Unreadable {
        /// The file.
        path: PathBuf,
    },
    /// The file is not a save document and no `.bak`/`temp_` fallback parsed.
    Corrupt {
        /// The file.
        path: PathBuf,
    },
    /// The bytes did not reach the file.
    Write {
        /// The file that was being written.
        path: PathBuf,
        /// What the store reported.
        message: String,
    },
    /// The document text did not parse.
    Parse {
        /// The file.
        path: PathBuf,
        /// The RON failure, with its position.
        source: ron::error::SpannedError,
    },
    /// The payload did not deserialize into the type asked for.
    Payload {
        /// The file.
        path: PathBuf,
        /// The RON failure, with its position.
        source: ron::error::SpannedError,
    },
    /// The payload did not serialize.
    Encode {
        /// The file that was going to hold it.
        path: PathBuf,
        /// The RON failure.
        source: ron::Error,
    },
    /// The file is stamped with a newer schema than this binary writes, or a
    /// caller asked to write one. Refused whole, never partially parsed.
    ForwardVersion {
        /// The file.
        path: PathBuf,
        /// The schema found (or offered).
        found: u32,
        /// The schema this file writes.
        current: u32,
    },
    /// No migration step is registered for a version bump on the way to the
    /// current schema.
    MissingMigration {
        /// The file.
        path: PathBuf,
        /// The version the chain has no step for.
        from: u32,
        /// The version being migrated to.
        to: u32,
    },
    /// A migration step ran and refused.
    Migration {
        /// The file.
        path: PathBuf,
        /// The version the failing step migrates from.
        from: u32,
        /// What the step reported.
        source: MigrationError,
    },
}

impl SaveError {
    /// The file the failure is about.
    pub fn path(&self) -> &Path {
        match self {
            Self::Unreadable { path }
            | Self::Corrupt { path }
            | Self::Write { path, .. }
            | Self::Parse { path, .. }
            | Self::Payload { path, .. }
            | Self::Encode { path, .. }
            | Self::ForwardVersion { path, .. }
            | Self::MissingMigration { path, .. }
            | Self::Migration { path, .. } => path,
        }
    }
}

impl From<(PathBuf, DocError)> for SaveError {
    /// Attach the file a document-layer failure happened on.
    fn from((path, source): (PathBuf, DocError)) -> Self {
        match source {
            DocError::Parse(source) => Self::Parse { path, source },
            DocError::Payload(source) => Self::Payload { path, source },
            DocError::Encode(source) => Self::Encode { path, source },
            DocError::ForwardVersion { found, current } => Self::ForwardVersion {
                path,
                found,
                current,
            },
            DocError::MissingMigration { from, to } => Self::MissingMigration { path, from, to },
            DocError::Migration { from, source } => Self::Migration { path, from, source },
        }
    }
}

impl fmt::Display for SaveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unreadable { path } => write!(f, "save {}: unreadable", path.display()),
            Self::Corrupt { path } => {
                write!(
                    f,
                    "save {}: not a save document and no backup",
                    path.display()
                )
            }
            Self::Write { path, message } => {
                write!(f, "save {}: write failed: {message}", path.display())
            }
            Self::Parse { path, source } => {
                write!(f, "save {}: not a save document: {source}", path.display())
            }
            Self::Payload { path, source } => {
                write!(f, "save {}: payload: {source}", path.display())
            }
            Self::Encode { path, source } => {
                write!(
                    f,
                    "save {}: payload could not be written: {source}",
                    path.display()
                )
            }
            Self::ForwardVersion {
                path,
                found,
                current,
            } => write!(
                f,
                "save {}: schema {found} is newer than {current}; refusing to parse it",
                path.display()
            ),
            Self::MissingMigration { path, from, to } => write!(
                f,
                "save {}: no migration step from schema {from} to {to}",
                path.display()
            ),
            Self::Migration { path, from, source } => write!(
                f,
                "save {}: migration from schema {from}: {source}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for SaveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Parse { source, .. } | Self::Payload { source, .. } => Some(source),
            Self::Encode { source, .. } => Some(source),
            Self::Migration { source, .. } => Some(source),
            Self::Unreadable { .. }
            | Self::Corrupt { .. }
            | Self::Write { .. }
            | Self::ForwardVersion { .. }
            | Self::MissingMigration { .. } => None,
        }
    }
}
