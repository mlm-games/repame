//! The versioned save envelope.
//!
//! A save file is one document:
//!
//! ```ron
//! (
//!     schema: 3,
//!     body: r#"(
//!     best: 1200,
//!     muted: false,
//! )"#,
//! )
//! ```
//!
//! `schema` is the number the *binary* knows; `body` is the game's own document,
//! carried verbatim. The payload travels as text on purpose: a save type with an
//! enum in it cannot be represented by a `ron::Value` tree, so a tree would have
//! to reject or mangle real saves, and the payload's own structure stays readable
//! between the `r#"` and `"#` markers.
//!
//! Rules, in the order [`SaveDoc::migrate`] applies them:
//!
//! - an older stamp runs the [`MigrationChain`] from that stamp to the current
//!   one, one step per version, with no version skipped;
//! - an equal stamp is used as-is;
//! - a newer stamp is [`DocError::ForwardVersion`]. The file is refused whole,
//!   because parsing it with an older type would drop fields the newer build
//!   wrote, and the next save would make that loss permanent.

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::error::DocError;
use crate::migration::MigrationChain;

/// The document as it is written on disk.
#[derive(serde::Serialize, serde::Deserialize)]
struct Wire {
    schema: u32,
    #[serde(default)]
    body: String,
}

/// A parsed save document: the schema stamp plus the payload text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SaveDoc {
    schema: u32,
    body: String,
}

impl SaveDoc {
    /// Build a document from a typed payload.
    pub fn new<T: Serialize>(schema: u32, payload: &T) -> Result<Self, DocError> {
        let config = ron::ser::PrettyConfig::new();
        let body = ron::ser::to_string_pretty(payload, config).map_err(DocError::Encode)?;
        Ok(Self { schema, body })
    }

    /// Parse a document.
    pub fn parse(text: &str) -> Result<Self, DocError> {
        let wire: Wire = ron::from_str(text).map_err(DocError::Parse)?;
        Ok(Self {
            schema: wire.schema,
            body: wire.body,
        })
    }

    /// Build a document from an existing payload text: a tool, an import, or a
    /// hand-written file that already holds a payload document.
    ///
    /// The text is not checked here; [`SaveDoc::body_as`] and
    /// [`SaveDoc::migrate`] report what is wrong with it.
    pub fn from_body(schema: u32, body: impl Into<String>) -> Self {
        Self {
            schema,
            body: body.into(),
        }
    }

    /// Whether these bytes are a save document: UTF-8 RON carrying a `schema`
    /// stamp.
    ///
    /// This is the integrity probe the store runs before it parses anything, so
    /// a truncated or foreign file is quarantined and a backup is adopted
    /// instead. It says nothing about the schema number: a file from a newer
    /// build is intact and is refused by [`SaveDoc::migrate`], not by this.
    pub fn is_document(bytes: &[u8]) -> bool {
        std::str::from_utf8(bytes).is_ok_and(|text| ron::from_str::<Wire>(text).is_ok())
    }

    /// The schema this document is stamped with.
    pub fn schema(&self) -> u32 {
        self.schema
    }

    /// Restamp the document without touching the payload.
    pub fn set_schema(&mut self, schema: u32) {
        self.schema = schema;
    }

    /// The payload document.
    pub fn body(&self) -> &str {
        &self.body
    }

    /// Replace the payload document.
    pub fn set_body(&mut self, body: impl Into<String>) {
        self.body = body.into();
    }

    /// Deserialize the payload.
    pub fn body_as<T: DeserializeOwned>(&self) -> Result<T, DocError> {
        ron::from_str(&self.body).map_err(DocError::Payload)
    }

    /// Bring the document up to `current`: run the chain from the stamp, then
    /// restamp.
    ///
    /// A document already at `current` is left alone, and one *newer* than
    /// `current` is [`DocError::ForwardVersion`]. A failed chain leaves the
    /// document untouched.
    pub fn migrate(&mut self, chain: &MigrationChain, current: u32) -> Result<(), DocError> {
        if self.schema > current {
            return Err(DocError::ForwardVersion {
                found: self.schema,
                current,
            });
        }
        if self.schema == current {
            return Ok(());
        }
        let mut body = self.body.clone();
        chain.apply(self.schema, current, &mut body)?;
        self.body = body;
        self.schema = current;
        Ok(())
    }

    /// The document text to write.
    ///
    /// The payload is written unescaped so the file stays editable by hand and
    /// readable in a diff.
    pub fn to_text(&self) -> Result<String, DocError> {
        let wire = Wire {
            schema: self.schema,
            body: self.body.clone(),
        };
        let pretty = ron::ser::PrettyConfig::new().escape_strings(false);
        ron::ser::to_string_pretty(&wire, pretty).map_err(DocError::Encode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::MigrationError;

    #[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
    struct Payload {
        best: i32,
        /// An enum field: a `ron::Value` intermediate could not hold this, which
        /// is why the body is text.
        stance: Stance,
    }

    #[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
    enum Stance {
        Idle,
        Walk { speed: f32 },
    }

    #[derive(Serialize, serde::Deserialize)]
    struct PayloadV1 {
        best: i32,
    }

    #[derive(Serialize)]
    struct PayloadV2 {
        best: i32,
        stance: Stance,
    }

    fn payload() -> Payload {
        Payload {
            best: 12,
            stance: Stance::Walk { speed: 1.5 },
        }
    }

    #[test]
    fn envelope_round_trips_with_an_enum_in_the_payload() {
        let doc = SaveDoc::new(3, &payload()).expect("serializes");
        let text = doc.to_text().expect("writes");
        assert!(text.contains("body:"), "{text}");
        assert!(SaveDoc::is_document(text.as_bytes()));
        let parsed = SaveDoc::parse(&text).expect("parses");
        assert_eq!(parsed.schema(), 3);
        assert_eq!(parsed, doc);
        assert_eq!(parsed.body_as::<Payload>().expect("payload"), payload());
    }

    #[test]
    fn a_whole_document_round_trips_through_its_own_text() {
        let doc = SaveDoc::new(1, &7u32).expect("serializes");
        let text = doc.to_text().expect("writes");
        assert_eq!(SaveDoc::parse(&text).expect("parses"), doc);
    }

    #[test]
    fn anything_else_is_not_a_document() {
        assert!(!SaveDoc::is_document(b""));
        assert!(!SaveDoc::is_document(b"{\"best\": 1}"));
        assert!(!SaveDoc::is_document(b"(body: \"\",)"));
        assert!(SaveDoc::is_document(b"(schema: 1,)"));
    }

    #[test]
    fn migration_chain_brings_an_old_document_up() {
        fn add_stance(_from: u32, body: &str) -> Result<String, MigrationError> {
            crate::migration::rewrite::<PayloadV1, PayloadV2>(body, |old| PayloadV2 {
                best: old.best,
                stance: Stance::Idle,
            })
        }

        let chain = MigrationChain::new().step(1, add_stance);
        let old = SaveDoc::new(1, &PayloadV1 { best: 5 }).expect("serializes");
        let mut doc = SaveDoc::parse(&old.to_text().expect("writes")).expect("parses");
        doc.migrate(&chain, 2).expect("migrates");
        assert_eq!(doc.schema(), 2);
        assert_eq!(
            doc.body_as::<Payload>().expect("payload"),
            Payload {
                best: 5,
                stance: Stance::Idle,
            }
        );
    }

    #[test]
    fn a_missing_step_is_typed_and_leaves_the_document_alone() {
        let mut doc = SaveDoc::new(1, &PayloadV1 { best: 5 }).expect("serializes");
        let before = doc.clone();
        let error = doc
            .migrate(&MigrationChain::new(), 3)
            .expect_err("no steps registered");
        assert!(
            matches!(error, DocError::MissingMigration { from: 1, to: 3 }),
            "{error}"
        );
        assert_eq!(doc, before);
    }

    #[test]
    fn a_newer_document_is_refused_whole() {
        let mut doc = SaveDoc::new(5, &payload()).expect("serializes");
        let before = doc.clone();
        let error = doc.migrate(&MigrationChain::new(), 3).expect_err("newer");
        assert!(
            matches!(
                error,
                DocError::ForwardVersion {
                    found: 5,
                    current: 3
                }
            ),
            "{error}"
        );
        assert_eq!(doc, before);
    }

    #[test]
    fn a_document_at_the_current_schema_is_untouched() {
        let mut doc = SaveDoc::new(3, &payload()).expect("serializes");
        let before = doc.clone();
        doc.migrate(&MigrationChain::new(), 3)
            .expect("nothing to do");
        assert_eq!(doc, before);
    }

    #[test]
    fn a_payload_that_does_not_fit_is_an_error_not_a_default() {
        let doc = SaveDoc::new(1, &PayloadV1 { best: 5 }).expect("serializes");
        let error = doc.body_as::<Payload>().expect_err("stance is missing");
        assert!(matches!(error, DocError::Payload(_)), "{error}");
    }

    #[test]
    fn unknown_top_level_keys_are_tolerated() {
        let text = "(schema: 1, body: \"3\", extra: 9,)";
        let doc = SaveDoc::parse(text).expect("unknown top-level keys are tolerated");
        assert_eq!(doc.schema(), 1);
        assert_eq!(doc.body_as::<u32>().expect("payload"), 3);
    }
}
