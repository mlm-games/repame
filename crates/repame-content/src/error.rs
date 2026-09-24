use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct ContentError {
    path: PathBuf,
    message: String,
}

impl ContentError {
    pub fn new(path: impl Into<PathBuf>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ContentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path.display(), self.message)
    }
}

impl std::error::Error for ContentError {}

pub(crate) fn io_error(path: &Path, action: impl Into<String>, error: io::Error) -> ContentError {
    ContentError::new(path, format!("{}: {error}", action.into()))
}

pub(crate) fn ron_error(path: &Path, error: impl fmt::Display) -> ContentError {
    ContentError::new(path, format!("invalid RON: {error}"))
}
