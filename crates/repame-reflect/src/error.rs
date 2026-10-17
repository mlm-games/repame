//! Typed reflection errors.
//!
//! Every failure is a variant: a component type or name that is not in the
//! registry, a value that does not deserialize into its component, a snapshot
//! whose format version is not the one this crate writes, or a document whose
//! RON did not parse. Nothing is skipped, defaulted, or logged and forgotten on
//! the save/load path; a component named in a document but absent from the
//! registry is [`ReflectError::UnknownType`], never a silent drop.

use std::fmt;

use bevy_ecs::prelude::Entity;

/// Everything that can go wrong while registering, snapshotting, or inspecting.
#[derive(Debug)]
pub enum ReflectError {
    /// A component name in a document is not registered.
    UnknownType {
        /// The unregistered name.
        name: String,
    },
    /// A component type or name was registered twice.
    AlreadyRegistered {
        /// The name it was already registered under.
        name: String,
    },
    /// A name was used that the registry does not know.
    NotRegistered {
        /// The unknown name.
        name: String,
    },
    /// The entity does not exist, or does not carry the named component.
    MissingComponent {
        /// The entity the operation addressed.
        entity: Entity,
        /// The component name.
        name: String,
    },
    /// A component name was empty, padded, or held control characters.
    BadName {
        /// The rejected name.
        name: String,
    },
    /// A component value did not deserialize into its type.
    BadValue {
        /// The component the value was meant for.
        name: String,
        /// The underlying deserialization failure.
        source: serde_json::Error,
    },
    /// A snapshot document did not parse as RON.
    Format {
        /// The underlying RON failure.
        source: ron::Error,
    },
    /// A snapshot carried a format version this crate does not write.
    Version {
        /// The version found in the document.
        found: u32,
        /// The version this crate writes.
        expected: u32,
    },
}

impl fmt::Display for ReflectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownType { name } => write!(f, "reflect unknown component type `{name}`"),
            Self::AlreadyRegistered { name } => {
                write!(f, "reflect component `{name}` is already registered")
            }
            Self::NotRegistered { name } => write!(f, "reflect component `{name}` is not registered"),
            Self::MissingComponent { entity, name } => {
                write!(f, "reflect entity {entity:?} has no component `{name}`")
            }
            Self::BadName { name } => write!(f, "reflect invalid component name `{name}`"),
            Self::BadValue { name, source } => {
                write!(f, "reflect value for component `{name}`: {source}")
            }
            Self::Format { source } => write!(f, "reflect snapshot ron: {source}"),
            Self::Version { found, expected } => {
                write!(f, "reflect snapshot format {found}; expected {expected}")
            }
        }
    }
}

impl std::error::Error for ReflectError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::BadValue { source, .. } => Some(source),
            Self::Format { source } => Some(source),
            _ => None,
        }
    }
}

pub(crate) fn bad_value(name: &str, source: serde_json::Error) -> ReflectError {
    ReflectError::BadValue {
        name: name.to_string(),
        source,
    }
}
