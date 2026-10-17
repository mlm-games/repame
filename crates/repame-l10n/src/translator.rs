//! The runtime registry: catalogs, the active locale, fallback lookup, and the
//! missing-key log.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::catalog::{Catalog, Message};
use crate::error::L10nError;
use crate::format::{Args, substitute};
use crate::locale::{language_of, resolve_locale, standardize_locale};

/// One key that was looked up and not found, with how many times.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MissingKey {
    /// The requested locale at the time of each lookup.
    pub locale: String,
    /// The lookup key.
    pub key: String,
    /// Number of lookups that missed this key in this locale.
    pub count: u64,
}

/// A deduplicated, sorted record of every missing lookup.
///
/// Fed automatically by [`Translator::tr`] and [`Translator::trn`]; read it
/// with [`Translator::missing`] or [`Translator::missing_entries`] to drive a
/// coverage report or a failing test in CI. Keys are counted per
/// `(locale, key)`, and iteration is sorted, so a report built from it is
/// deterministic.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MissingLog {
    entries: BTreeMap<(String, String), u64>,
}

impl MissingLog {
    /// Record one miss.
    pub fn record(&mut self, locale: &str, key: &str) {
        *self
            .entries
            .entry((locale.to_string(), key.to_string()))
            .or_insert(0) += 1;
    }

    /// Every recorded key, sorted by locale then key.
    pub fn entries(&self) -> Vec<MissingKey> {
        self.entries
            .iter()
            .map(|((locale, key), count)| MissingKey {
                locale: locale.clone(),
                key: key.clone(),
                count: *count,
            })
            .collect()
    }

    /// Miss count for one key in one locale, or `0`.
    pub fn count(&self, locale: &str, key: &str) -> u64 {
        self.entries
            .get(&(locale.to_string(), key.to_string()))
            .copied()
            .unwrap_or(0)
    }

    /// Number of distinct missing keys.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing has been recorded.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Drop every record.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

/// Holds the registered catalogs, the active locale, and the missing-key log.
///
/// Construct with the default locale, [`Translator::add_catalog`] once per
/// locale, and [`Translator::set_locale`] (or
/// [`Translator::set_locale_from`] with a caller-supplied preference list) to
/// switch at runtime. Lookups are shared-safe: `tr`/`trn` take `&self` and the
/// missing-key log uses interior mutability.
#[derive(Debug)]
pub struct Translator {
    default_locale: String,
    locale: String,
    catalogs: BTreeMap<String, Catalog>,
    missing: Mutex<MissingLog>,
}

impl Default for Translator {
    fn default() -> Self {
        Self::new("en")
    }
}

impl Translator {
    /// A translator with no catalogs, defaulting to `default_locale`.
    ///
    /// The locale is standardized when possible and kept verbatim otherwise,
    /// so an odd id still round-trips through [`Translator::locale`].
    pub fn new(default_locale: impl Into<String>) -> Self {
        let default_locale = default_locale.into();
        let locale = standardize_locale(&default_locale)
            .unwrap_or_else(|| default_locale.trim().to_string());
        Self {
            default_locale: locale.clone(),
            locale,
            catalogs: BTreeMap::new(),
            missing: Mutex::new(MissingLog::default()),
        }
    }

    /// Register a catalog, replacing any catalog with the same locale.
    /// Returns the replaced catalog, if any.
    pub fn add_catalog(&mut self, catalog: Catalog) -> Option<Catalog> {
        let locale = catalog.locale().to_string();
        self.catalogs.insert(locale, catalog)
    }

    /// The default locale.
    pub fn default_locale(&self) -> &str {
        &self.default_locale
    }

    /// Change the default locale; the active locale is unchanged.
    pub fn set_default_locale(&mut self, locale: impl Into<String>) {
        let locale = locale.into();
        self.default_locale =
            standardize_locale(&locale).unwrap_or_else(|| locale.trim().to_string());
    }

    /// The requested locale.
    pub fn locale(&self) -> &str {
        &self.locale
    }

    /// Switch the requested locale. Any id is accepted; an unregistered one
    /// simply falls through the fallback chain at lookup time.
    pub fn set_locale(&mut self, locale: impl AsRef<str>) {
        let locale = locale.as_ref();
        self.locale = standardize_locale(locale).unwrap_or_else(|| locale.trim().to_string());
    }

    /// Registered locale ids, sorted.
    pub fn locales(&self) -> Vec<&str> {
        self.catalogs.keys().map(String::as_str).collect()
    }

    /// A registered catalog by locale id.
    pub fn catalog(&self, locale: &str) -> Option<&Catalog> {
        self.catalogs.get(locale)
    }

    /// Number of registered catalogs.
    pub fn len(&self) -> usize {
        self.catalogs.len()
    }

    /// Whether no catalog is registered.
    pub fn is_empty(&self) -> bool {
        self.catalogs.is_empty()
    }

    /// The registered locales a lookup will try, in order.
    ///
    /// The order is the requested locale, then its language subtag, then the
    /// default locale, duplicates and unregistered ids dropped. Lookup uses the
    /// same order, so this is a diagnostic view of it.
    pub fn fallback_chain(&self) -> Vec<&str> {
        let requested = self.locale.as_str();
        let language = language_of(requested);
        let default = self.default_locale.as_str();
        let mut chain: Vec<&str> = Vec::with_capacity(3);
        for candidate in [requested, language, default] {
            if self.catalogs.contains_key(candidate) && !chain.contains(&candidate) {
                chain.push(candidate);
            }
        }
        chain
    }

    /// The best registered locale for an ordered preference list.
    ///
    /// For each preference, an exact registered match wins, then the first
    /// registered locale with the same language; if nothing matches, the
    /// default locale is used (its exact id, then its language). No operating
    /// system, environment, or process locale is ever consulted.
    pub fn detect_locale(&self, preferences: &[&str]) -> String {
        let available: Vec<&str> = self.locales();
        resolve_locale(&available, preferences, &self.default_locale)
    }

    /// Resolve `preferences` and switch to the result, returning it.
    pub fn set_locale_from(&mut self, preferences: &[&str]) -> String {
        let locale = self.detect_locale(preferences);
        self.set_locale(&locale);
        locale
    }

    /// Whether the key resolves in the current fallback chain.
    pub fn has_key(&self, key: &str) -> bool {
        self.lookup(key).is_some()
    }

    /// Translate a singular key, substituting `{placeholder}`s from `args`.
    ///
    /// A miss returns [`L10nError::MissingKey`] and records it; the key is
    /// never echoed back as if it were a translation. A key that exists only
    /// as a plural returns [`L10nError::NotSingular`]. The result borrows the
    /// catalog unless a placeholder had to be filled.
    pub fn tr<'a>(&'a self, key: &str, args: &Args<'_>) -> Result<Cow<'a, str>, L10nError> {
        match self.lookup(key) {
            Some((locale, message)) => match message.singular() {
                Some(text) => substitute(text, args, None, key),
                None => Err(L10nError::NotSingular {
                    key: key.to_string(),
                    locale: locale.to_string(),
                }),
            },
            None => {
                self.record_missing(key);
                Err(L10nError::MissingKey {
                    key: key.to_string(),
                    locale: self.locale.clone(),
                })
            }
        }
    }

    /// Translate a plural key for count `n`.
    ///
    /// The category is chosen from the language of the catalog the key was
    /// found in (see [`crate::plural_category`]). `{n}` is available to the
    /// template unless `args` binds a named argument `n`, which takes
    /// precedence. A key with only a singular form uses it, with `n` still
    /// available. A miss returns [`L10nError::MissingKey`] and records it.
    pub fn trn<'a>(
        &'a self,
        key: &str,
        n: u64,
        args: &Args<'_>,
    ) -> Result<Cow<'a, str>, L10nError> {
        match self.lookup(key) {
            Some((locale, message)) => {
                let language = language_of(locale);
                let text = message
                    .plural()
                    .and_then(|plural| plural.select(language, n))
                    .or_else(|| message.singular());
                match text {
                    Some(text) => substitute(text, args, Some(n), key),
                    None => {
                        self.record_missing(key);
                        Err(L10nError::MissingKey {
                            key: key.to_string(),
                            locale: self.locale.clone(),
                        })
                    }
                }
            }
            None => {
                self.record_missing(key);
                Err(L10nError::MissingKey {
                    key: key.to_string(),
                    locale: self.locale.clone(),
                })
            }
        }
    }

    /// Snapshot of the missing-key log.
    pub fn missing(&self) -> MissingLog {
        self.missing
            .lock()
            .map(|log| log.clone())
            .unwrap_or_default()
    }

    /// Snapshot of the missing-key log, sorted.
    pub fn missing_entries(&self) -> Vec<MissingKey> {
        self.missing().entries()
    }

    /// Number of distinct missing keys.
    pub fn missing_count(&self) -> usize {
        self.missing().len()
    }

    /// Drop every missing-key record.
    pub fn clear_missing(&self) {
        if let Ok(mut log) = self.missing.lock() {
            log.clear();
        }
    }

    fn lookup(&self, key: &str) -> Option<(&str, &Message)> {
        let requested = self.locale.as_str();
        let language = language_of(requested);
        let default = self.default_locale.as_str();
        if let Some(hit) = self.lookup_in(requested, key) {
            return Some(hit);
        }
        if language != requested
            && let Some(hit) = self.lookup_in(language, key)
        {
            return Some(hit);
        }
        if default != requested
            && default != language
            && let Some(hit) = self.lookup_in(default, key)
        {
            return Some(hit);
        }
        None
    }

    fn lookup_in<'a>(&'a self, locale: &'a str, key: &str) -> Option<(&'a str, &'a Message)> {
        self.catalogs
            .get(locale)
            .and_then(|catalog| catalog.get(key).map(|message| (locale, message)))
    }

    fn record_missing(&self, key: &str) {
        if let Ok(mut log) = self.missing.lock() {
            log.record(&self.locale, key);
        }
    }
}
