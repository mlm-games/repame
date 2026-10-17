use std::fmt;

/// Everything that can go wrong while loading or using a catalog.
///
/// The parser variants carry the source line, so a bad catalog is diagnosable
/// without the caller re-reading the file. The lookup variants are returned by
/// [`Translator::tr`](crate::Translator::tr) and friends; every one of them is
/// also mirrored into the translator's missing-key log where that applies, so a
/// missing string is never mistaken for a hit.
#[derive(Debug)]
pub enum L10nError {
    /// A RON catalog did not parse.
    Ron(ron::error::SpannedError),
    /// A catalog parsed but its content was inconsistent (unknown plural
    /// category, a plural entry without `other`).
    Catalog {
        /// What was wrong.
        message: String,
    },
    /// A locale id had no valid `language[_Script][_Country][@variant]` shape.
    InvalidLocale {
        /// The offending input.
        input: String,
    },
    /// A `.po` catalog was malformed.
    Po {
        /// 1-based source line the parser stopped on.
        line: usize,
        /// What was wrong.
        message: String,
    },
    /// A `.csv` catalog was malformed.
    Csv {
        /// 1-based record number the parser stopped on.
        line: usize,
        /// What was wrong.
        message: String,
    },
    /// No catalog in the fallback chain held the key.
    MissingKey {
        /// The lookup key.
        key: String,
        /// The requested locale at the time of the lookup.
        locale: String,
    },
    /// The key resolved to a plural-only message but was read as singular.
    NotSingular {
        /// The lookup key.
        key: String,
        /// The locale the key was found in.
        locale: String,
    },
    /// A `{name}` placeholder had no matching argument.
    MissingArgument {
        /// The lookup key.
        key: String,
        /// The placeholder name, without braces.
        name: String,
    },
    /// A `{` opened a placeholder that was never closed.
    UnbalancedPlaceholder {
        /// The lookup key.
        key: String,
    },
}

impl fmt::Display for L10nError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ron(e) => write!(f, "l10n ron catalog: {e}"),
            Self::Catalog { message } => write!(f, "l10n catalog: {message}"),
            Self::InvalidLocale { input } => write!(f, "l10n invalid locale `{input}`"),
            Self::Po { line, message } => write!(f, "l10n po catalog line {line}: {message}"),
            Self::Csv { line, message } => write!(f, "l10n csv catalog record {line}: {message}"),
            Self::MissingKey { key, locale } => {
                write!(f, "l10n missing key `{key}` for locale `{locale}`")
            }
            Self::NotSingular { key, locale } => {
                write!(
                    f,
                    "l10n key `{key}` in locale `{locale}` has no singular form"
                )
            }
            Self::MissingArgument { key, name } => {
                write!(f, "l10n key `{key}` needs argument `{name}`")
            }
            Self::UnbalancedPlaceholder { key } => {
                write!(f, "l10n key `{key}` has an unclosed placeholder")
            }
        }
    }
}

impl std::error::Error for L10nError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Ron(e) => Some(e),
            _ => None,
        }
    }
}
