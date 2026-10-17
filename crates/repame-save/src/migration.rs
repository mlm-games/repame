//! Schema migrations: one registered step per version bump.
//!
//! A step is a plain function pointer: it is handed the version it migrates
//! *from* and the payload text, and returns the payload text of the next schema.
//! Text, not a value tree, because a payload may hold an enum and a
//! `ron::Value` cannot represent one: rewriting the text keeps the payload's
//! own document intact.
//!
//! Most steps are a typed hop: parse the old shape, build the new one. [`rewrite`]
//! does exactly that, so a step is one line:
//!
//! ```rust
//! use repame_save::{MigrationChain, MigrationError, rewrite};
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Deserialize)]
//! struct V1 { best: i32 }
//!
//! #[derive(Serialize)]
//! struct V2 { best: i32, tickets: i32 }
//!
//! fn add_tickets(_from: u32, body: &str) -> Result<String, MigrationError> {
//!     rewrite::<V1, V2>(body, |old| V2 { best: old.best, tickets: 0 })
//! }
//!
//! let chain = MigrationChain::new().step(1, add_tickets);
//! assert!(chain.has(1));
//! ```

use std::collections::BTreeMap;
use std::fmt;

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::error::{DocError, MigrationError};

/// One migration step: rewrite the payload of a document stamped `from` into the
/// payload of schema `from + 1`.
///
/// The second argument is the payload document (the `body` of
/// [`SaveDoc`](crate::SaveDoc)), not the whole save file: the schema stamp stays
/// the envelope's business.
pub type MigrationFn = fn(from: u32, body: &str) -> Result<String, MigrationError>;

/// The registered steps, keyed by the schema they migrate from.
///
/// Build it once at boot. A chain is only allowed to have gaps if the game never
/// meets them: migrating a file needs a step for **every** version between the
/// file's stamp and the current schema, and a missing one is
/// [`DocError::MissingMigration`], not a skipped version.
///
/// A second [`MigrationChain::step`] for the same version replaces the first, so
/// build the chain in one place.
#[derive(Clone, Default)]
pub struct MigrationChain {
    steps: BTreeMap<u32, MigrationFn>,
}

impl fmt::Debug for MigrationChain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MigrationChain")
            .field("from_versions", &self.steps.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl MigrationChain {
    /// An empty chain: every migration is missing until one is registered.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            steps: BTreeMap::new(),
        }
    }

    /// Register the step that migrates schema `from` to `from + 1`.
    ///
    /// A second step for the same `from` replaces the first.
    #[must_use]
    pub fn step(mut self, from: u32, apply: MigrationFn) -> Self {
        self.steps.insert(from, apply);
        self
    }

    /// Whether a step from `from` is registered.
    pub fn has(&self, from: u32) -> bool {
        self.steps.contains_key(&from)
    }

    /// How many steps are registered.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether no step is registered.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Run every step from `from` up to `to` against `body`.
    ///
    /// The body is only replaced once all the steps succeeded, so a refusal
    /// leaves the caller's document exactly as it was.
    pub(crate) fn apply(&self, from: u32, to: u32, body: &mut String) -> Result<(), DocError> {
        let mut text = body.clone();
        for version in from..to {
            let Some(step) = self.steps.get(&version) else {
                return Err(DocError::MissingMigration { from: version, to });
            };
            text = step(version, &text).map_err(|source| DocError::Migration {
                from: version,
                source,
            })?;
        }
        *body = text;
        Ok(())
    }
}

/// Rewrite a payload through a typed hop: deserialize it as `Old`, serialize the
/// value `reshape` builds.
///
/// This is the body of a [`MigrationFn`] for the common case where a version
/// bump changes fields and the new shape is built from the old one.
pub fn rewrite<Old, New>(
    body: &str,
    reshape: impl FnOnce(Old) -> New,
) -> Result<String, MigrationError>
where
    Old: DeserializeOwned,
    New: Serialize,
{
    let old: Old = ron::from_str(body)?;
    let new = reshape(old);
    ron::ser::to_string_pretty(&new, Default::default()).map_err(MigrationError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_two(_from: u32, body: &str) -> Result<String, MigrationError> {
        rewrite::<u32, u32>(body, |value| value * 2)
    }

    fn refuses(_from: u32, _body: &str) -> Result<String, MigrationError> {
        Err(MigrationError::refused("nothing to migrate"))
    }

    #[test]
    fn steps_apply_in_order_until_the_target() {
        let chain = MigrationChain::new().step(1, to_two).step(2, to_two);
        let mut body = String::from("3");
        chain.apply(1, 3, &mut body).expect("both steps run");
        assert_eq!(body, "12");
        assert_eq!(chain.len(), 2);
        assert!(chain.has(1) && chain.has(2) && !chain.has(3));
    }

    #[test]
    fn a_gap_is_a_typed_error_and_leaves_the_body_alone() {
        let chain = MigrationChain::new().step(1, to_two);
        let mut body = String::from("3");
        let error = chain.apply(1, 3, &mut body).expect_err("no step from 2");
        assert!(
            matches!(error, DocError::MissingMigration { from: 2, to: 3 }),
            "{error}"
        );
        assert_eq!(body, "3", "a failed chain does not touch the payload");
    }

    #[test]
    fn a_step_that_refuses_stops_the_chain_and_names_its_version() {
        let chain = MigrationChain::new().step(1, to_two).step(2, refuses);
        let mut body = String::from("3");
        let error = chain.apply(1, 3, &mut body).expect_err("step 2 refuses");
        assert!(
            matches!(error, DocError::Migration { from: 2, .. }),
            "{error}"
        );
        assert_eq!(body, "3");
    }

    #[test]
    fn a_same_version_apply_is_a_no_op() {
        let chain = MigrationChain::new();
        let mut body = String::from("3");
        chain.apply(4, 4, &mut body).expect("nothing to do");
        assert_eq!(body, "3");
        assert!(chain.is_empty());
        assert_eq!(format!("{chain:?}"), "MigrationChain { from_versions: [] }");
    }
}
