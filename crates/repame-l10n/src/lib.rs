//! Runtime localization for repame games.
//!
//! A [`Catalog`] maps message ids to strings for one locale. Catalogs come
//! from a RON document, from a Godot-style text `.po` file, or from a
//! Godot-style `.csv` file. A [`Translator`] registers them, holds the active
//! locale and its fallback chain, and answers [`Translator::tr`] (singular)
//! and [`Translator::trn`] (plural) lookups with `{placeholder}` substitution.
//!
//! # Design
//!
//! - **Catalog.** Entries live in a `BTreeMap`, so keys, plural categories and
//!   every derived listing are sorted. A message carries a singular form, a
//!   plural set, or both. A context folds into the key with
//!   [`context_key`], the gettext MO separator, so a contextual `.po`/`.csv`
//!   entry keeps its context instead of losing it.
//! - **Plurals.** [`plural_category`] selects a CLDR category with built-in
//!   rules for `en`, `fr`, `de`, `es`, `ru`, `pl`, `ja` and `zh` (any other
//!   language uses the English rule). `.po`/`.csv` forms are positional, so
//!   [`plural_index`] maps the category onto the gettext form order the file
//!   used; RON catalogs name their forms by category directly. The two are
//!   consistent for the supported languages.
//! - **Locale.** Locales are explicit. Nothing reads the operating system,
//!   the environment, or the process locale. [`resolve_locale`] and
//!   [`Translator::detect_locale`] take a caller-supplied ordered preference
//!   list and pick the first exact match, then the first same-language match,
//!   then the default.
//! - **Fallback at lookup.** A lookup tries the requested locale, then its
//!   language subtag, then the default locale ([`Translator::fallback_chain`]
//!   is that order, registered entries only). The first catalog holding the
//!   key wins.
//! - **Missing keys.** A miss is never the key echoed back and never an empty
//!   string. It returns [`L10nError::MissingKey`] and is recorded in a
//!   collectable [`MissingLog`], deduplicated per `(locale, key)` with a
//!   count, so a test or a coverage report can assert zero misses. An empty
//!   `.po`/`.csv` translation is treated as untranslated for the same reason.
//!
//! # Determinism
//!
//! The string returned by `tr`/`trn` is a pure function of the registered
//! catalogs, the active locale, the key, the count, and the arguments. There
//! is no time, randomness, thread-order, or OS input on the lookup path:
//! locale detection uses only the preference list the caller hands in, the
//! fallback order is fixed, and plural selection is integer arithmetic. The
//! missing-key log is a diagnostics side channel (`Mutex`-guarded interior
//! mutability, written only on a miss) and never affects a returned value; its
//! snapshot is sorted, so a report drawn from it is stable run to run.
//!
//! The lookup path allocates nothing on a hit with no arguments (it borrows
//! the catalog string) and takes no lock; allocation and locking happen only
//! when a placeholder must be filled or a miss recorded.

mod catalog;
mod error;
mod format;
mod locale;
mod plural;
mod translator;

pub use catalog::{Catalog, CsvDelimiter, Message, Plural};
pub use error::L10nError;
pub use format::Args;
pub use locale::{CONTEXT_SEPARATOR, context_key, language_of, resolve_locale, standardize_locale};
pub use plural::{PluralCategory, plural_categories, plural_category, plural_index};
pub use translator::{MissingKey, MissingLog, Translator};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::substitute;

    #[test]
    fn plural_categories_match_cldr() {
        use PluralCategory::{Few, Many, One, Other};
        let cases: &[(&str, u64, PluralCategory)] = &[
            ("en", 0, Other),
            ("en", 1, One),
            ("en", 2, Other),
            ("de", 1, One),
            ("de", 2, Other),
            ("es", 1, One),
            ("es", 0, Other),
            ("fr", 0, One),
            ("fr", 1, One),
            ("fr", 2, Other),
            ("ru", 0, Many),
            ("ru", 1, One),
            ("ru", 2, Few),
            ("ru", 4, Few),
            ("ru", 5, Many),
            ("ru", 11, Many),
            ("ru", 12, Many),
            ("ru", 21, One),
            ("ru", 22, Few),
            ("ru", 25, Many),
            ("ru", 100, Many),
            ("ru", 101, One),
            ("pl", 0, Many),
            ("pl", 1, One),
            ("pl", 2, Few),
            ("pl", 4, Few),
            ("pl", 5, Many),
            ("pl", 12, Many),
            ("pl", 22, Few),
            ("pl", 25, Many),
            ("pl", 101, Many),
            ("ja", 0, Other),
            ("ja", 2, Other),
            ("zh", 5, Other),
        ];
        for &(language, n, expected) in cases {
            assert_eq!(
                plural_category(language, n),
                expected,
                "plural_category({language}, {n})"
            );
        }
        assert_eq!(plural_categories("ja"), &[PluralCategory::Other]);
        assert_eq!(plural_index("ru", PluralCategory::Many), 2);
        assert_eq!(plural_index("pl", PluralCategory::Many), 2);
        assert_eq!(plural_index("en", PluralCategory::Other), 1);
        assert_eq!(plural_index("ja", PluralCategory::Other), 0);
    }

    #[test]
    fn arguments_substitute_positionally_and_by_name() {
        let positional = Args::positional(&["Ana", "3"]);
        assert_eq!(
            substitute("Hello {0}, you have {1}", &positional, None, "k").unwrap(),
            "Hello Ana, you have 3"
        );
        let named = Args::new().named("count", "3").named("name", "Bo");
        assert_eq!(
            substitute("{count} items for {name}", &named, None, "k").unwrap(),
            "3 items for Bo"
        );
        assert_eq!(substitute("{{literal}} {0}", &positional, None, "k").unwrap(), "{literal} Ana");
        assert_eq!(
            substitute("{n} apples", &Args::new(), Some(3), "k").unwrap(),
            "3 apples"
        );
        let override_n = Args::new().named("n", "9");
        assert_eq!(
            substitute("{n} apples", &override_n, Some(3), "k").unwrap(),
            "9 apples"
        );
        assert!(matches!(
            substitute("{0}", &Args::new(), None, "k"),
            Err(L10nError::MissingArgument { .. })
        ));
    }

    #[test]
    fn po_catalog_parses_locale_context_and_plural_forms() {
        let source = "\
msgid \"\"
msgstr \"\"
\"Language: ru\\n\"
\"Plural-Forms: nplurals=3; plural=(n%10==1 && n%100!=11 ? 0 : 2);\\n\"

#: menu
msgid \"menu.play\"
msgstr \"Играть\"

#: untranslated
msgid \"menu.quit\"
msgstr \"\"

msgctxt \"item\"
msgid \"apple\"
msgid_plural \"apples\"
msgstr[0] \"{n} яблоко\"
msgstr[1] \"{n} яблока\"
msgstr[2] \"{n} яблок\"
";
        let catalog = Catalog::from_po(source).expect("parses");
        assert_eq!(catalog.locale(), "ru");
        assert_eq!(catalog.get("menu.play").unwrap().singular(), Some("Играть"));
        assert!(catalog.get("menu.quit").is_none(), "empty translation is untranslated");
        let apple = catalog.get(&context_key("item", "apple")).unwrap();
        let plural = apple.plural().unwrap();
        assert_eq!(plural.select("ru", 1), Some("{n} яблоко"));
        assert_eq!(plural.select("ru", 2), Some("{n} яблока"));
        assert_eq!(plural.select("ru", 5), Some("{n} яблок"));
    }

    #[test]
    fn csv_catalog_parses_locales_and_plural_rows() {
        let source = "\
keys,?plural,en,de
GOOD_MORNING,,\"Good Morning\",\"Guten Morgen\"
GOOD_EVENING,,\"Good Evening\",
apple,apples,\"{n} apple\",
,,\"{n} apples\",
";
        let catalogs = Catalog::from_csv(source, CsvDelimiter::Comma).expect("parses");
        let locales: Vec<&str> = catalogs.iter().map(|c| c.locale()).collect();
        assert_eq!(locales, vec!["en", "de"]);
        assert_eq!(
            catalogs[0].get("GOOD_MORNING").unwrap().singular(),
            Some("Good Morning")
        );
        assert!(catalogs[1].get("GOOD_EVENING").is_none(), "empty cell is untranslated");
        let apple = catalogs[0].get("apple").unwrap().plural().unwrap();
        assert_eq!(apple.select("en", 1), Some("{n} apple"));
        assert_eq!(apple.select("en", 2), Some("{n} apples"));
    }

    #[test]
    fn fallback_chain_is_requested_then_language_then_default() {
        let mut translator = Translator::new("en");
        let mut en = Catalog::new("en").unwrap();
        en.insert_singular("both", "en both");
        en.insert_singular("only_en", "EN");
        let mut en_us = Catalog::new("en_US").unwrap();
        en_us.insert_singular("both", "us both");
        en_us.insert_singular("only_us", "US");
        translator.add_catalog(en);
        translator.add_catalog(en_us);

        translator.set_locale("en_US");
        assert_eq!(translator.fallback_chain(), vec!["en_US", "en"]);
        assert_eq!(translator.tr("both", &Args::new()).unwrap(), "us both");
        assert_eq!(translator.tr("only_en", &Args::new()).unwrap(), "EN");
        assert_eq!(translator.tr("only_us", &Args::new()).unwrap(), "US");

        // An unregistered country still reaches its language.
        translator.set_locale("en_GB");
        assert_eq!(translator.fallback_chain(), vec!["en"]);
        assert_eq!(translator.tr("both", &Args::new()).unwrap(), "en both");

        // A locale with no catalog and no matching language only reaches the
        // default; en_US is not in the chain.
        translator.set_locale("fr");
        assert_eq!(translator.fallback_chain(), vec!["en"]);
        assert_eq!(translator.tr("only_en", &Args::new()).unwrap(), "EN");
        assert!(matches!(
            translator.tr("only_us", &Args::new()),
            Err(L10nError::MissingKey { .. })
        ));
    }

    #[test]
    fn locale_detection_uses_the_preference_list() {
        let mut translator = Translator::new("en");
        translator.add_catalog(Catalog::new("en").unwrap());
        translator.add_catalog(Catalog::new("en_US").unwrap());
        translator.add_catalog(Catalog::new("ru").unwrap());
        assert_eq!(translator.detect_locale(&["fr", "en_US"]), "en_US");
        assert_eq!(translator.detect_locale(&["en_GB"]), "en");
        assert_eq!(translator.detect_locale(&["ru_RU"]), "ru");
        assert_eq!(translator.detect_locale(&[]), "en");
        assert_eq!(translator.set_locale_from(&["ru_RU"]), "ru");
        assert_eq!(translator.locale(), "ru");
    }

    #[test]
    fn trn_selects_the_plural_form_from_named_categories() {
        let ron = r#"
            (
                locale: "en",
                messages: { "greet": "Hello {name}!" },
                plurals: { "apples": { "one": "{n} apple", "other": "{n} apples" } },
            )
        "#;
        let mut translator = Translator::new("en");
        translator.add_catalog(Catalog::from_ron(ron).expect("parses"));
        let greet = Args::new().named("name", "Ada");
        assert_eq!(translator.tr("greet", &greet).unwrap(), "Hello Ada!");
        assert_eq!(translator.trn("apples", 1, &Args::new()).unwrap(), "1 apple");
        assert_eq!(translator.trn("apples", 4, &Args::new()).unwrap(), "4 apples");
    }

    #[test]
    fn missing_keys_are_typed_and_logged_once_per_locale_key() {
        let mut translator = Translator::new("en");
        let mut en = Catalog::new("en").unwrap();
        en.insert_singular("present", "Here");
        translator.add_catalog(en);

        assert_eq!(translator.tr("present", &Args::new()).unwrap(), "Here");
        for _ in 0..2 {
            assert!(matches!(
                translator.tr("absent", &Args::new()),
                Err(L10nError::MissingKey { ref key, ref locale }) if key == "absent" && locale == "en"
            ));
        }
        translator.set_locale("ru");
        let _ = translator.trn("absent_plural", 3, &Args::new());

        let entries = translator.missing_entries();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].locale, "en");
        assert_eq!(entries[0].key, "absent");
        assert_eq!(entries[0].count, 2);
        assert_eq!(entries[1].locale, "ru");
        assert_eq!(entries[1].key, "absent_plural");
        assert_eq!(translator.missing().count("en", "absent"), 2);

        translator.clear_missing();
        assert!(translator.missing().is_empty());
    }

    #[test]
    fn singular_lookup_rejects_a_plural_only_key() {
        let mut translator = Translator::new("en");
        let mut en = Catalog::new("en").unwrap();
        en.insert_plural_forms("apples", vec!["{n} apple".into(), "{n} apples".into()]);
        translator.add_catalog(en);
        assert!(matches!(
            translator.tr("apples", &Args::new()),
            Err(L10nError::NotSingular { .. })
        ));
    }
}
