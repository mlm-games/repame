//! Locale ids, standardization, and preference resolution.
//!
//! Locale detection is entirely caller-driven: the crate never reads the
//! operating system, environment, or process locale. A caller supplies an
//! ordered preference list (typically from a platform layer, a settings file,
//! or a command line) and [`resolve_locale`] picks the first entry the registry
//! can serve.

/// The separator gettext uses between a context and a message id.
pub const CONTEXT_SEPARATOR: char = '\u{4}';

/// Normalize a locale id to `language[_Script][_Country][@variant]`.
///
/// Mirrors Godot's `TranslationServer::standardize_locale`: `-` becomes `_`,
/// the language is lowercased, a four-letter second element is title-cased as a
/// script, a two-letter element is uppercased as a country, and `@extra` is
/// lowercased and kept. Returns `None` when the language subtag is missing or
/// not alphabetic.
pub fn standardize_locale(input: &str) -> Option<String> {
    let unified: String = input
        .trim()
        .chars()
        .map(|c| if c == '-' { '_' } else { c })
        .collect();
    let (base, variant) = match unified.split_once('@') {
        Some((base, variant)) => (base, Some(variant)),
        None => (unified.as_str(), None),
    };
    let mut elements = base.split('_');
    let language = elements.next().unwrap_or("");
    if language.is_empty() || !language.bytes().all(|b| b.is_ascii_alphabetic()) {
        return None;
    }

    let mut out = language.to_ascii_lowercase();
    let mut script: Option<String> = None;
    let mut country: Option<String> = None;
    for (idx, element) in elements.filter(|e| !e.is_empty()).enumerate() {
        if idx == 0 && element.len() == 4 && element.bytes().all(|b| b.is_ascii_alphabetic()) {
            let mut chars = element.chars();
            let mut title = String::new();
            if let Some(first) = chars.next() {
                title.extend(first.to_uppercase());
                title.push_str(&chars.as_str().to_ascii_lowercase());
            }
            script = Some(title);
        } else if element.len() == 2 && element.bytes().all(|b| b.is_ascii_alphabetic()) {
            country = Some(element.to_ascii_uppercase());
        }
    }
    if let Some(script) = script {
        out.push('_');
        out.push_str(&script);
    }
    if let Some(country) = country {
        out.push('_');
        out.push_str(&country);
    }
    if let Some(variant) = variant {
        let variant = variant.trim();
        if !variant.is_empty() && variant.bytes().all(|b| b.is_ascii_alphanumeric()) {
            out.push('@');
            out.push_str(&variant.to_ascii_lowercase());
        }
    }
    Some(out)
}

/// The language subtag of a locale id: everything before the first `_` or `@`.
pub fn language_of(locale: &str) -> &str {
    let end = locale.find(['_', '@']).unwrap_or(locale.len());
    &locale[..end]
}

/// Build the catalog key for a contextual message.
///
/// A message with no context keeps its bare id. A contextual message becomes
/// `context\u{4}id`, the separator the gettext MO format uses (and Godot's
/// `get_message_list` emits), so both halves survive round-tripping.
pub fn context_key(context: &str, id: &str) -> String {
    if context.is_empty() {
        id.to_string()
    } else {
        format!("{context}{CONTEXT_SEPARATOR}{id}")
    }
}

/// Pick the best locale for an ordered preference list.
///
/// Resolution order: for each preference in order, an exact registered match
/// wins; otherwise the first registered locale with the same language. If no
/// preference matches, the default is used, again preferring its exact
/// registered id and then its language. When nothing is registered at all, the
/// standardized default is returned unchanged, so the caller still has a
/// stable id. `available` order breaks ties, so callers pass a sorted slice.
pub fn resolve_locale(available: &[&str], preferences: &[&str], default: &str) -> String {
    let find_language = |language: &str| {
        available
            .iter()
            .find(|candidate| language_of(candidate) == language)
            .map(|candidate| (*candidate).to_string())
    };
    for preference in preferences {
        let Some(std) = standardize_locale(preference) else {
            continue;
        };
        if available.contains(&std.as_str()) {
            return std;
        }
        if let Some(hit) = find_language(language_of(&std)) {
            return hit;
        }
    }
    let default = standardize_locale(default).unwrap_or_else(|| default.to_string());
    if available.contains(&default.as_str()) {
        return default;
    }
    if let Some(hit) = find_language(language_of(&default)) {
        return hit;
    }
    default
}
