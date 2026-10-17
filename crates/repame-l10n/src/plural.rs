//! CLDR plural category selection, plus the gettext form index used by `.po`
//! and `.csv` catalogs.
//!
//! The category is chosen by built-in per-language rules, never at runtime by
//! an evaluated expression. The gettext index is a separate, fixed table
//! because a `.po` file orders its forms by its own `plural=` expression, and
//! that order is not always the CLDR category order (Russian and Polish, for
//! instance, collapse `many` and `other` into one gettext form).

use std::fmt;

/// One CLDR plural category.
///
/// Derived ordering is the canonical CLDR order
/// (`zero < one < two < few < many < other`), which is also the order of
/// [`PluralCategory::ALL`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PluralCategory {
    /// Explicit zero form. Used by languages without a `one` for zero.
    Zero,
    /// Singular form.
    One,
    /// Dual form.
    Two,
    /// Paucal form.
    Few,
    /// Greater plural form.
    Many,
    /// Catch-all form. Every catalog must provide it.
    Other,
}

impl PluralCategory {
    /// Every category, in canonical CLDR order.
    pub const ALL: [PluralCategory; 6] = [
        PluralCategory::Zero,
        PluralCategory::One,
        PluralCategory::Two,
        PluralCategory::Few,
        PluralCategory::Many,
        PluralCategory::Other,
    ];

    /// Parse a category name such as `"few"`, case-insensitively.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "zero" => Some(Self::Zero),
            "one" => Some(Self::One),
            "two" => Some(Self::Two),
            "few" => Some(Self::Few),
            "many" => Some(Self::Many),
            "other" => Some(Self::Other),
            _ => None,
        }
    }

    /// The lowercase category name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Zero => "zero",
            Self::One => "one",
            Self::Two => "two",
            Self::Few => "few",
            Self::Many => "many",
            Self::Other => "other",
        }
    }
}

impl fmt::Display for PluralCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Categories a language selects, in canonical order.
///
/// Covers `en`, `fr`, `de`, `es`, `ru`, `pl`, `ja`, `zh`; any other language
/// falls back to the English `[one, other]` set. `ja` and `zh` have a single
/// form, so their only category is `other`.
pub fn plural_categories(language: &str) -> &'static [PluralCategory] {
    match language {
        "ru" | "pl" => &[
            PluralCategory::One,
            PluralCategory::Few,
            PluralCategory::Many,
            PluralCategory::Other,
        ],
        "ja" | "zh" => &[PluralCategory::Other],
        _ => &[PluralCategory::One, PluralCategory::Other],
    }
}

/// The CLDR category for `n` in `language`.
///
/// `n` is treated as its magnitude, which matches CLDR, where the category
/// operands are absolute; callers with negative counts should pass the
/// magnitude. `ja` and `zh` always return [`PluralCategory::Other`]. An
/// unsupported language uses the English rule (`n == 1` is
/// [`PluralCategory::One`]).
pub fn plural_category(language: &str, n: u64) -> PluralCategory {
    match language {
        "fr" => {
            if n == 0 || n == 1 {
                PluralCategory::One
            } else {
                PluralCategory::Other
            }
        }
        "ru" => {
            let m10 = n % 10;
            let m100 = n % 100;
            if m10 == 1 && m100 != 11 {
                PluralCategory::One
            } else if (2..=4).contains(&m10) && !(12..=14).contains(&m100) {
                PluralCategory::Few
            } else if m10 == 0 || (5..=9).contains(&m10) || (11..=14).contains(&m100) {
                PluralCategory::Many
            } else {
                PluralCategory::Other
            }
        }
        "pl" => {
            let m10 = n % 10;
            let m100 = n % 100;
            if n == 1 {
                PluralCategory::One
            } else if (2..=4).contains(&m10) && !(12..=14).contains(&m100) {
                PluralCategory::Few
            } else if m10 <= 1 || (5..=9).contains(&m10) || (12..=14).contains(&m100) {
                PluralCategory::Many
            } else {
                PluralCategory::Other
            }
        }
        "ja" | "zh" => PluralCategory::Other,
        _ => {
            if n == 1 {
                PluralCategory::One
            } else {
                PluralCategory::Other
            }
        }
    }
}

/// The gettext form index for `category` in `language`.
///
/// Indexes follow the plural rules Godot ships (and the gettext rules they are
/// taken from): English-like languages use `one -> 0`, `other -> 1`; Russian
/// and Polish use `one -> 0`, `few -> 1`, `many -> 2`, `other -> 2`; Japanese
/// and Chinese use `other -> 0`. A category the language does not use maps to
/// its `other` index, so the result is always a valid form position.
pub fn plural_index(language: &str, category: PluralCategory) -> usize {
    let cats = plural_categories(language);
    let pos = |c: PluralCategory| cats.iter().position(|&x| x == c);
    pos(category)
        .or_else(|| pos(PluralCategory::Other))
        .unwrap_or(0)
}
