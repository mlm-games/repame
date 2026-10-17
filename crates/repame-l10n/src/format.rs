//! `{placeholder}` substitution and the C string escapes shared with `.po`.
//!
//! A placeholder is `{name}`, where `name` is either an all-digit positional
//! index into [`Args::positional`] or a named argument from [`Args::named`].
//! `{{` and `}}` are literal braces. Substitution borrows the template when it
//! has no placeholders to fill, so the common path allocates nothing.

use std::borrow::Cow;

use crate::error::L10nError;

/// Positional and named values for one lookup.
///
/// Build one with [`Args::positional`] for `{0}`, `{1}`, ... placeholders,
/// [`Args::named`] for `{count}` style placeholders, or both.
#[derive(Clone, Debug, Default)]
pub struct Args<'a> {
    positional: Vec<&'a str>,
    named: Vec<(&'a str, &'a str)>,
}

impl<'a> Args<'a> {
    /// No arguments.
    pub fn new() -> Self {
        Self::default()
    }

    /// Positional values, addressed as `{0}`, `{1}`, ...
    pub fn positional(values: &[&'a str]) -> Self {
        Self {
            positional: values.to_vec(),
            named: Vec::new(),
        }
    }

    /// Add a named value, addressed as `{name}`.
    pub fn named(mut self, name: &'a str, value: &'a str) -> Self {
        self.named.push((name, value));
        self
    }

    /// Append positional values to an existing set.
    pub fn and_positional(mut self, values: &[&'a str]) -> Self {
        self.positional.extend_from_slice(values);
        self
    }

    /// Number of positional and named values together.
    pub fn len(&self) -> usize {
        self.positional.len() + self.named.len()
    }

    /// Whether there are no values.
    pub fn is_empty(&self) -> bool {
        self.positional.is_empty() && self.named.is_empty()
    }

    fn named_value(&self, name: &str) -> Option<&'a str> {
        self.named
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| *value)
    }
}

/// Replace `{placeholder}`s in `template`.
///
/// `count` is exposed to the template as `{n}` unless `args` already binds a
/// named argument `n`, which takes precedence. Returns a borrowed slice when
/// nothing had to be replaced. `key` is only used to annotate errors.
pub(crate) fn substitute<'a>(
    template: &'a str,
    args: &Args<'_>,
    count: Option<u64>,
    key: &str,
) -> Result<Cow<'a, str>, L10nError> {
    let bytes = template.as_bytes();
    let mut out: Option<String> = None;
    let mut copied = 0usize;
    let mut i = 0usize;
    let mut count_buf: Option<String> = None;

    while i < bytes.len() {
        match bytes[i] {
            b'{' if bytes.get(i + 1) == Some(&b'{') => {
                let out = out.get_or_insert_with(String::new);
                out.push_str(&template[copied..i]);
                out.push('{');
                copied = i + 2;
                i += 2;
            }
            b'}' if bytes.get(i + 1) == Some(&b'}') => {
                let out = out.get_or_insert_with(String::new);
                out.push_str(&template[copied..i]);
                out.push('}');
                copied = i + 2;
                i += 2;
            }
            b'{' => {
                let close = match template[i + 1..].find('}') {
                    Some(offset) => i + 1 + offset,
                    None => {
                        return Err(L10nError::UnbalancedPlaceholder { key: key.to_string() });
                    }
                };
                let name = &template[i + 1..close];
                let value: &str = if !name.is_empty() && name.bytes().all(|b| b.is_ascii_digit())
                {
                    match name.parse::<usize>().ok().and_then(|idx| args.positional.get(idx)) {
                        Some(value) => value,
                        None => {
                            return Err(L10nError::MissingArgument {
                                key: key.to_string(),
                                name: name.to_string(),
                            });
                        }
                    }
                } else if let Some(value) = args.named_value(name) {
                    value
                } else if name == "n" && count.is_some() {
                    if count_buf.is_none() {
                        count_buf = Some(count.unwrap_or(0).to_string());
                    }
                    count_buf.as_deref().unwrap_or("")
                } else {
                    return Err(L10nError::MissingArgument {
                        key: key.to_string(),
                        name: name.to_string(),
                    });
                };
                let out = out.get_or_insert_with(String::new);
                out.push_str(&template[copied..i]);
                out.push_str(value);
                copied = close + 1;
                i = close + 1;
            }
            _ => i += 1,
        }
    }

    match out {
        Some(mut out) => {
            out.push_str(&template[copied..]);
            Ok(Cow::Owned(out))
        }
        None => Ok(Cow::Borrowed(template)),
    }
}

/// Decode C escapes (`\n`, `\t`, `\"`, `\\`, `\uXXXX`, ...) in a `.po`/`.csv`
/// string body. An unknown escape keeps its backslash, so no input is lost.
pub(crate) fn c_unescape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('a') => out.push('\u{7}'),
            Some('b') => out.push('\u{8}'),
            Some('f') => out.push('\u{c}'),
            Some('v') => out.push('\u{b}'),
            Some('\\') => out.push('\\'),
            Some('"') => out.push('"'),
            Some('\'') => out.push('\''),
            Some('u') => push_hex(&mut out, &mut chars, 4),
            Some('U') => push_hex(&mut out, &mut chars, 6),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

fn push_hex(out: &mut String, chars: &mut std::str::Chars<'_>, max: usize) {
    let mut digits = String::new();
    while digits.len() < max {
        let mut peek = chars.clone();
        match peek.next() {
            Some(c) if c.is_ascii_hexdigit() => {
                digits.push(c);
                chars.next();
            }
            _ => break,
        }
    }
    match u32::from_str_radix(&digits, 16).ok().and_then(char::from_u32) {
        Some(c) => out.push(c),
        None => {
            out.push('\\');
            out.push(if max == 4 { 'u' } else { 'U' });
            out.push_str(&digits);
        }
    }
}
