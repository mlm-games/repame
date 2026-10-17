//! Translation catalogs: the in-memory map, the RON format, and the Godot-style
//! `.po` and `.csv` parsers.
//!
//! A catalog is keyed by message id. A context folds into the key with
//! [`context_key`], so contextual `.po`/`.csv` entries keep both halves. A key
//! carries a singular form, a plural form, or both.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::error::L10nError;
use crate::format::c_unescape;
use crate::locale::{context_key, standardize_locale};
use crate::plural::{PluralCategory, plural_category, plural_index};

/// The plural forms of one message.
///
/// RON catalogs name their forms by [`PluralCategory`] ([`Plural::ByCategory`]);
/// `.po` and `.csv` catalogs list them positionally ([`Plural::Forms`]),
/// because their order comes from the file's own plural rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Plural {
    /// Forms named by CLDR category.
    ByCategory(BTreeMap<PluralCategory, String>),
    /// Forms in gettext index order.
    Forms(Vec<String>),
}

impl Plural {
    /// Number of stored forms.
    pub fn len(&self) -> usize {
        match self {
            Self::ByCategory(forms) => forms.len(),
            Self::Forms(forms) => forms.len(),
        }
    }

    /// Whether no form is stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Select the form for `n` in `language`.
    ///
    /// Returns `None` when the selected form is empty, which marks the entry
    /// untranslated so a lookup reports a miss instead of an empty hit. A
    /// category with no stored form falls back to `other`; a gettext index past
    /// the end of the list clamps to the last form.
    pub fn select(&self, language: &str, n: u64) -> Option<&str> {
        let text = match self {
            Self::ByCategory(forms) => forms
                .get(&plural_category(language, n))
                .or_else(|| forms.get(&PluralCategory::Other))?
                .as_str(),
            Self::Forms(forms) => {
                if forms.is_empty() {
                    return None;
                }
                let index = plural_index(language, plural_category(language, n));
                forms.get(index).or_else(|| forms.last())?.as_str()
            }
        };
        if text.is_empty() { None } else { Some(text) }
    }
}

/// One catalog entry: a singular string, a plural set, or both.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Message {
    singular: Option<String>,
    plural: Option<Plural>,
}

impl Message {
    /// The singular form, if the entry has one.
    pub fn singular(&self) -> Option<&str> {
        self.singular.as_deref()
    }

    /// The plural forms, if the entry has any.
    pub fn plural(&self) -> Option<&Plural> {
        self.plural.as_ref()
    }

    /// Add or replace the singular form. An empty string clears it, matching
    /// the `.po` convention where an empty translation is untranslated.
    pub fn set_singular(&mut self, text: impl Into<String>) {
        let text = text.into();
        self.singular = if text.is_empty() { None } else { Some(text) };
    }

    /// Add or replace the plural forms.
    pub fn set_plural(&mut self, forms: Plural) {
        self.plural = if forms.is_empty() { None } else { Some(forms) };
    }
}

/// Field separator for [`Catalog::from_csv`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CsvDelimiter {
    /// `,`
    #[default]
    Comma,
    /// `;`
    Semicolon,
    /// A tab.
    Tab,
}

impl CsvDelimiter {
    fn as_char(self) -> char {
        match self {
            Self::Comma => ',',
            Self::Semicolon => ';',
            Self::Tab => '\t',
        }
    }
}

/// A set of translated messages for one locale.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Catalog {
    locale: String,
    messages: BTreeMap<String, Message>,
}

impl Catalog {
    /// An empty catalog for `locale`, standardized on the way in.
    pub fn new(locale: &str) -> Result<Self, L10nError> {
        let locale = standardize_locale(locale).ok_or_else(|| L10nError::InvalidLocale {
            input: locale.to_string(),
        })?;
        Ok(Self {
            locale,
            messages: BTreeMap::new(),
        })
    }

    /// The standardized locale this catalog translates.
    pub fn locale(&self) -> &str {
        &self.locale
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Whether the catalog has no entries.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Entry ids, sorted.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.messages.keys().map(String::as_str)
    }

    /// Entry by key.
    pub fn get(&self, key: &str) -> Option<&Message> {
        self.messages.get(key)
    }

    /// Insert or update a singular entry. An empty `text` clears the singular
    /// form; an entry left with no form at all is removed.
    pub fn insert_singular(&mut self, key: impl Into<String>, text: impl Into<String>) {
        let key = key.into();
        let message = self.messages.entry(key.clone()).or_default();
        message.set_singular(text);
        if message.singular.is_none() && message.plural.is_none() {
            self.messages.remove(&key);
        }
    }

    /// Insert or update a plural entry from named categories.
    pub fn insert_plural_categories(
        &mut self,
        key: impl Into<String>,
        forms: BTreeMap<PluralCategory, String>,
    ) {
        let key = key.into();
        let message = self.messages.entry(key.clone()).or_default();
        message.set_plural(Plural::ByCategory(forms));
        if message.singular.is_none() && message.plural.is_none() {
            self.messages.remove(&key);
        }
    }

    /// Insert or update a plural entry from gettext-ordered forms.
    pub fn insert_plural_forms(&mut self, key: impl Into<String>, forms: Vec<String>) {
        let key = key.into();
        let message = self.messages.entry(key.clone()).or_default();
        message.set_plural(Plural::Forms(forms));
        if message.singular.is_none() && message.plural.is_none() {
            self.messages.remove(&key);
        }
    }

    /// Parse a RON catalog.
    ///
    /// ```ron
    /// (
    ///     locale: "en",
    ///     messages: {
    ///         "menu.play": "Play",
    ///         "greet": "Hello {name}!",
    ///     },
    ///     plurals: {
    ///         "apples": {
    ///             "one": "{n} apple",
    ///             "other": "{n} apples",
    ///         },
    ///     },
    /// )
    /// ```
    ///
    /// `messages` and `plurals` may each be omitted. Every plural entry must
    /// name at least `other`, so no entry can fail to resolve a form later.
    pub fn from_ron(text: &str) -> Result<Self, L10nError> {
        let raw: RawCatalog = ron::from_str(text).map_err(L10nError::Ron)?;
        let mut catalog = Self::new(&raw.locale)?;
        for (key, text) in raw.messages {
            catalog.insert_singular(key, text);
        }
        for (key, forms) in raw.plurals {
            let mut named = BTreeMap::new();
            for (name, text) in forms {
                let category = PluralCategory::from_name(&name).ok_or_else(|| {
                    L10nError::Catalog {
                        message: format!("plural category `{name}` in key `{key}` is unknown"),
                    }
                })?;
                named.insert(category, text);
            }
            if !named.contains_key(&PluralCategory::Other) {
                return Err(L10nError::Catalog {
                    message: format!("plural key `{key}` has no `other` form"),
                });
            }
            catalog.insert_plural_categories(key, named);
        }
        Ok(catalog)
    }

    /// Parse a Godot-style text `.po` catalog, taking the locale from the
    /// `Language:`/`X-Language:` header.
    pub fn from_po(text: &str) -> Result<Self, L10nError> {
        let parsed = parse_po(text)?;
        let locale = match &parsed.locale {
            Some(locale) => locale.clone(),
            None => {
                return Err(L10nError::Po {
                    line: 1,
                    message: "header has no `Language:` or `X-Language:` field".to_string(),
                });
            }
        };
        parsed.into_catalog(&locale)
    }

    /// Parse a `.po` catalog using `locale` instead of the header field.
    pub fn from_po_with_locale(text: &str, locale: &str) -> Result<Self, L10nError> {
        let parsed = parse_po(text)?;
        parsed.into_catalog(locale)
    }

    /// Parse a Godot-style `.csv` catalog.
    ///
    /// The first row is a header: column 0 is the key column, `?context` and
    /// `?plural` columns are special, columns whose name starts with `_` are
    /// metadata, and every other column names a locale. Plural rows use the
    /// `?plural` column as the plural id and collect one form per following
    /// row. A `?pluralrule` row is accepted and ignored: categories come from
    /// the built-in rules, not from an evaluated expression. Returns one
    /// catalog per locale column, in column order.
    pub fn from_csv(text: &str, delimiter: CsvDelimiter) -> Result<Vec<Self>, L10nError> {
        let records = parse_csv_records(text, delimiter.as_char());
        let Some(header) = records.first() else {
            return Ok(Vec::new());
        };

        let mut columns: Vec<(usize, Self)> = Vec::new();
        let mut context_column: Option<usize> = None;
        let mut plural_column: Option<usize> = None;
        for (index, field) in header.iter().enumerate().skip(1) {
            if field.starts_with('_') {
                continue;
            }
            let lowered = field.to_ascii_lowercase();
            if lowered == "?context" {
                context_column = Some(index);
                continue;
            }
            if lowered == "?plural" {
                plural_column = Some(index);
                continue;
            }
            let catalog = Self::new(field)?;
            columns.push((index, catalog));
        }

        let mut reading_plural = false;
        let mut plural_id = String::new();
        let mut plural_context: Option<String> = None;
        let mut plural_forms: BTreeMap<usize, Vec<String>> = BTreeMap::new();

        for (record_index, fields) in records.iter().enumerate().skip(1) {
            let line = record_index + 1;
            if fields.len() == 1 && fields[0].is_empty() {
                continue;
            }
            let key = fields.first().cloned().unwrap_or_default();
            if key.eq_ignore_ascii_case("?pluralrule") {
                continue;
            }
            let context = column_value(fields, context_column);
            let id_plural = column_value(fields, plural_column);

            if reading_plural
                && (!key.is_empty() || !context.is_empty() || !id_plural.is_empty())
            {
                commit_plural(
                    &mut columns,
                    &mut plural_forms,
                    &plural_id,
                    plural_context.as_deref(),
                );
                reading_plural = false;
            }
            if !reading_plural && !id_plural.is_empty() {
                if key.is_empty() {
                    return Err(L10nError::Csv {
                        line,
                        message: "a plural row has no key".to_string(),
                    });
                }
                reading_plural = true;
                plural_id = key.clone();
                plural_context = if context.is_empty() {
                    None
                } else {
                    Some(context.clone())
                };
            }

            for (column, catalog) in columns.iter_mut() {
                let value = match fields.get(*column) {
                    Some(value) if !value.is_empty() => c_unescape(value),
                    _ => continue,
                };
                if reading_plural {
                    plural_forms.entry(*column).or_default().push(value);
                } else if !key.is_empty() {
                    catalog.insert_singular(context_key(&context, &key), value);
                }
            }
        }
        if reading_plural {
            commit_plural(
                &mut columns,
                &mut plural_forms,
                &plural_id,
                plural_context.as_deref(),
            );
        }

        Ok(columns.into_iter().map(|(_, catalog)| catalog).collect())
    }
}

fn column_value(fields: &[String], column: Option<usize>) -> String {
    column
        .and_then(|column| fields.get(column))
        .cloned()
        .unwrap_or_default()
}

fn commit_plural(
    columns: &mut [(usize, Catalog)],
    forms: &mut BTreeMap<usize, Vec<String>>,
    id: &str,
    context: Option<&str>,
) {
    let key = context_key(context.unwrap_or(""), id);
    for (column, catalog) in columns.iter_mut() {
        if let Some(values) = forms.remove(column) {
            catalog.insert_plural_forms(key.clone(), values);
        }
    }
}

#[derive(Deserialize)]
struct RawCatalog {
    locale: String,
    #[serde(default)]
    messages: BTreeMap<String, String>,
    #[serde(default)]
    plurals: BTreeMap<String, BTreeMap<String, String>>,
}

/// Parsed `.po` content before it becomes a [`Catalog`].
struct ParsedPo {
    locale: Option<String>,
    header_plural_count: Option<usize>,
    entries: Vec<PoEntry>,
}

enum PoEntry {
    Singular {
        context: Option<String>,
        id: String,
        text: String,
    },
    Plural {
        context: Option<String>,
        id: String,
        forms: Vec<String>,
    },
}

impl ParsedPo {
    fn into_catalog(self, locale: &str) -> Result<Catalog, L10nError> {
        let count = self.header_plural_count.unwrap_or(0);
        let mut catalog = Catalog::new(locale)?;
        for entry in self.entries {
            match entry {
                PoEntry::Singular { context, id, text } => {
                    catalog.insert_singular(context_key(context.as_deref().unwrap_or(""), &id), text);
                }
                PoEntry::Plural { context, id, forms } => {
                    if count == 0 {
                        return Err(L10nError::Po {
                            line: 1,
                            message: format!(
                                "plural key `{id}` needs `Plural-Forms: nplurals=N;` in the header"
                            ),
                        });
                    }
                    if forms.len() != count {
                        return Err(L10nError::Po {
                            line: 1,
                            message: format!(
                                "plural key `{id}` has {} form(s), header declares {count}",
                                forms.len()
                            ),
                        });
                    }
                    catalog
                        .insert_plural_forms(context_key(context.as_deref().unwrap_or(""), &id), forms);
                }
            }
        }
        Ok(catalog)
    }
}

#[derive(Default)]
struct PoBuilder {
    context: Option<String>,
    id: String,
    id_plural: Option<String>,
    text: Option<String>,
    forms: Vec<Option<String>>,
    fuzzy: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PoField {
    None,
    Context,
    Id,
    IdPlural,
    Text,
    Form(usize),
}

fn parse_po(text: &str) -> Result<ParsedPo, L10nError> {
    let mut parsed = ParsedPo {
        locale: None,
        header_plural_count: None,
        entries: Vec::new(),
    };
    let mut builder = PoBuilder::default();
    let mut has_entry = false;
    let mut field = PoField::None;

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim();
        if line.is_empty() {
            if has_entry {
                flush(&mut parsed, std::mem::take(&mut builder), line_number)?;
                has_entry = false;
            }
            field = PoField::None;
            continue;
        }
        if line.starts_with('#') {
            if line.contains("fuzzy") {
                builder.fuzzy = true;
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("msgctxt") {
            if has_entry && (!builder.id.is_empty() || builder.text.is_some()) {
                flush(&mut parsed, std::mem::take(&mut builder), line_number)?;
            }
            builder.context = Some(String::new());
            field = PoField::Context;
            append_value(&mut builder, field, rest, line_number)?;
        } else if let Some(rest) = line.strip_prefix("msgid_plural") {
            builder.id_plural = Some(String::new());
            field = PoField::IdPlural;
            append_value(&mut builder, field, rest, line_number)?;
        } else if let Some(rest) = line.strip_prefix("msgid") {
            if has_entry && (!builder.id.is_empty() || builder.text.is_some()) {
                flush(&mut parsed, std::mem::take(&mut builder), line_number)?;
            }
            builder.id.clear();
            field = PoField::Id;
            append_value(&mut builder, field, rest, line_number)?;
        } else if let Some(rest) = line.strip_prefix("msgstr[") {
            let (form, rest) = parse_form_index(rest, line_number)?;
            if form >= builder.forms.len() {
                builder.forms.resize(form + 1, None);
            }
            field = PoField::Form(form);
            append_value(&mut builder, field, rest, line_number)?;
        } else if let Some(rest) = line.strip_prefix("msgstr") {
            builder.text = Some(String::new());
            field = PoField::Text;
            append_value(&mut builder, field, rest, line_number)?;
        } else if line.starts_with('"') {
            append_value(&mut builder, field, line, line_number)?;
        } else {
            return Err(L10nError::Po {
                line: line_number,
                message: format!("unexpected line `{line}`"),
            });
        }
        has_entry = true;
    }
    if has_entry {
        let line_number = text.lines().count() + 1;
        flush(&mut parsed, builder, line_number)?;
    }
    Ok(parsed)
}

fn append_value(
    builder: &mut PoBuilder,
    field: PoField,
    rest: &str,
    line_number: usize,
) -> Result<(), L10nError> {
    let rest = rest.trim_start();
    if rest.is_empty() {
        return Ok(());
    }
    if !rest.starts_with('"') {
        return Err(L10nError::Po {
            line: line_number,
            message: format!("expected a quoted string but found `{rest}`"),
        });
    }
    let value = unquote(rest, line_number)?;
    match field {
        PoField::None => {}
        PoField::Context => {
            if let Some(context) = builder.context.as_mut() {
                context.push_str(&value);
            }
        }
        PoField::Id => builder.id.push_str(&value),
        PoField::IdPlural => {
            if let Some(id) = builder.id_plural.as_mut() {
                id.push_str(&value);
            }
        }
        PoField::Text => {
            if let Some(text) = builder.text.as_mut() {
                text.push_str(&value);
            }
        }
        PoField::Form(index) => {
            builder.forms[index].get_or_insert_with(String::new).push_str(&value);
        }
    }
    Ok(())
}

fn parse_form_index(rest: &str, line_number: usize) -> Result<(usize, &str), L10nError> {
    let close = rest.find(']').ok_or_else(|| L10nError::Po {
        line: line_number,
        message: "`msgstr[` has no closing `]`".to_string(),
    })?;
    let form = rest[..close].trim().parse::<usize>().map_err(|_| L10nError::Po {
        line: line_number,
        message: format!("invalid plural form index `{}`", &rest[..close]),
    })?;
    Ok((form, &rest[close + 1..]))
}

fn flush(parsed: &mut ParsedPo, builder: PoBuilder, line_number: usize) -> Result<(), L10nError> {
    if builder.fuzzy {
        return Ok(());
    }
    let context = builder.context.filter(|context| !context.is_empty());
    if builder.id.is_empty() && builder.id_plural.is_none() {
        if let Some(header) = builder.text {
            parsed.locale = header_value(&header, "Language")
                .or_else(|| header_value(&header, "X-Language"));
            parsed.header_plural_count = parse_nplurals(&header);
        }
        return Ok(());
    }
    if let Some(_id_plural) = builder.id_plural {
        let mut forms = Vec::with_capacity(builder.forms.len());
        for (index, form) in builder.forms.into_iter().enumerate() {
            let Some(form) = form else {
                return Err(L10nError::Po {
                    line: line_number,
                    message: format!("plural key `{}` is missing form {index}", builder.id),
                });
            };
            forms.push(form);
        }
        parsed.entries.push(PoEntry::Plural {
            context,
            id: builder.id,
            forms,
        });
        return Ok(());
    }
    if let Some(text) = builder.text
        && !text.is_empty()
    {
        parsed.entries.push(PoEntry::Singular {
            context,
            id: builder.id,
            text,
        });
    }
    Ok(())
}

fn unquote(line: &str, line_number: usize) -> Result<String, L10nError> {
    let rest = &line[1..];
    let bytes = rest.as_bytes();
    let mut end = None;
    let mut escaped = false;
    for (index, &byte) in bytes.iter().enumerate() {
        if escaped {
            escaped = false;
            continue;
        }
        if byte == b'\\' {
            escaped = true;
            continue;
        }
        if byte == b'"' {
            end = Some(index);
            break;
        }
    }
    let end = end.ok_or_else(|| L10nError::Po {
        line: line_number,
        message: "quoted string has no closing quote".to_string(),
    })?;
    Ok(c_unescape(&rest[..end]))
}

fn header_value(header: &str, key: &str) -> Option<String> {
    for line in header.lines() {
        if let Some((name, value)) = line.split_once(':')
            && name.trim().eq_ignore_ascii_case(key)
        {
            return Some(value.trim().to_string());
        }
    }
    None
}

fn parse_nplurals(header: &str) -> Option<usize> {
    for line in header.lines() {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if !name.trim().eq_ignore_ascii_case("Plural-Forms") {
            continue;
        }
        for part in value.split(';') {
            if let Some((key, number)) = part.split_once('=')
                && key.trim().eq_ignore_ascii_case("nplurals")
            {
                return number.trim().parse::<usize>().ok();
            }
        }
    }
    None
}

/// Split CSV text into records of fields. Quotes group delimiters and
/// newlines; `""` is a literal quote.
fn parse_csv_records(text: &str, delimiter: char) -> Vec<Vec<String>> {
    let mut records: Vec<Vec<String>> = Vec::new();
    let mut record: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if quoted {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    field.push('"');
                    chars.next();
                } else {
                    quoted = false;
                }
            } else {
                field.push(c);
            }
        } else if c == '"' {
            quoted = true;
        } else if c == delimiter {
            record.push(std::mem::take(&mut field));
        } else if c == '\n' {
            record.push(std::mem::take(&mut field));
            records.push(std::mem::take(&mut record));
        } else if c == '\r' {
            if chars.peek() != Some(&'\n') {
                record.push(std::mem::take(&mut field));
                records.push(std::mem::take(&mut record));
            }
        } else {
            field.push(c);
        }
    }
    if !field.is_empty() || !record.is_empty() {
        record.push(field);
        records.push(record);
    }
    records
}
