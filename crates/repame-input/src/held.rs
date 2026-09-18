//! Held-level repair against the platform snapshot.
//! Events own press edges; the polled set only repairs missed releases
//! (release outside the window, swallowed key-up) and missed presses
//! (down while unfocused). Never synthesizes edges, so repaired levels
//! cannot double-fire gameplay pulses or shortcut actions.

use std::collections::HashSet;
use std::hash::Hash;

/// Reconcile an event-staged held set against polled hardware truth.
///
/// - `staged`: the game's level set, mutated in place.
/// - `polled`: physical names currently down (e.g. `Scheduler::held_keys`).
/// - `code_for`: polled name -> staged code (`None` = unread key).
/// - `names_for`: staged code -> driving physical names;
///   `None` means "no physical source" (leave the level alone).
///   Takes `&'static` name tables (`&'static [&'static str]`); games
///   with owned names normalize to that shape at the call site.
///
/// Levels only: inserts polled-down codes, drops codes whose names are
/// all up. Callers stage edges exclusively from events.
pub fn reconcile_held<C>(staged: &mut HashSet<C>, polled: &HashSet<String>, code_for: impl Fn(&str) -> Option<C>, names_for: impl Fn(&C) -> Option<&'static [&'static str]>)
where
    C: Eq + Hash + Clone,
{
    for name in polled.iter() {
        if let Some(code) = code_for(name) {
            staged.insert(code);
        }
    }
    staged.retain(|code| {
        names_for(code).is_none_or(|names| names.iter().any(|n| polled.iter().any(|k| k == n)))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    enum Code {
        W,
        Space,
    }

    fn code_for(name: &str) -> Option<Code> {
        Some(match name {
            "KeyW" => Code::W,
            "Space" => Code::Space,
            _ => return None,
        })
    }

    fn names(code: &Code) -> Option<&'static [&'static str]> {
        Some(match code {
            Code::W => &["KeyW"],
            Code::Space => &["Space"],
        })
    }

    #[test]
    fn drops_stuck_levels_and_repairs_missed_presses() {
        let mut staged: HashSet<Code> = [Code::W].into_iter().collect();
        let polled: HashSet<String> = ["Space".to_string()].into_iter().collect();
        reconcile_held(&mut staged, &polled, code_for, names);
        assert!(!staged.contains(&Code::W));
        assert!(staged.contains(&Code::Space));
    }

    #[test]
    fn unknown_codes_are_left_alone() {
        let mut staged: HashSet<Code> = [Code::W].into_iter().collect();
        let polled: HashSet<String> = ["KeyX".to_string()].into_iter().collect();
        reconcile_held(
            &mut staged,
            &polled,
            |_: &str| None::<Code>,
            |_: &Code| None::<&'static [&'static str]>,
        );
        assert!(staged.contains(&Code::W));
    }
}
