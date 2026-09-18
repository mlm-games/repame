//! Held-level repair against the platform snapshot.
//! Events own press edges; the polled set only repairs missed releases
//! (release outside the window, swallowed key-up) and missed presses
//! (down while unfocused). Never synthesizes edges, so repaired levels
//! cannot double-fire gameplay pulses or shortcut actions.

use std::collections::HashSet;
use std::hash::Hash;

use repose_core::input::PhysicalKey;

/// Reconcile an event-staged held set against polled hardware truth.
///
/// - `staged`: the game's level set, mutated in place.
/// - `polled`: physical keys currently down (`Scheduler::held_keys`).
/// - `code_for`: polled key -> staged code (`None` = unread key).
/// - `names_for`: staged code -> driving physical keys;
///   `None` means "no physical source" (leave the level alone).
///
/// Levels only: inserts polled-down codes, drops codes whose keys are
/// all up. Callers stage edges exclusively from events.
pub fn reconcile_held<C>(staged: &mut HashSet<C>, polled: &HashSet<PhysicalKey>, code_for: impl Fn(PhysicalKey) -> Option<C>, names_for: impl Fn(&C) -> Option<&'static [PhysicalKey]>)
where
    C: Eq + Hash + Clone,
{
    for key in polled.iter() {
        if let Some(code) = code_for(*key) {
            staged.insert(code);
        }
    }
    staged.retain(|code| {
        names_for(code).is_none_or(|keys| keys.iter().any(|k| polled.contains(k)))
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

    fn code_for(key: PhysicalKey) -> Option<Code> {
        Some(match key {
            PhysicalKey::KeyW => Code::W,
            PhysicalKey::Space => Code::Space,
            _ => return None,
        })
    }

    fn names(code: &Code) -> Option<&'static [PhysicalKey]> {
        Some(match code {
            Code::W => &[PhysicalKey::KeyW],
            Code::Space => &[PhysicalKey::Space],
        })
    }

    #[test]
    fn drops_stuck_levels_and_repairs_missed_presses() {
        let mut staged: HashSet<Code> = [Code::W].into_iter().collect();
        let polled: HashSet<PhysicalKey> = [PhysicalKey::Space].into_iter().collect();
        reconcile_held(&mut staged, &polled, code_for, names);
        assert!(!staged.contains(&Code::W));
        assert!(staged.contains(&Code::Space));
    }

    #[test]
    fn unknown_codes_are_left_alone() {
        let mut staged: HashSet<Code> = [Code::W].into_iter().collect();
        let polled: HashSet<PhysicalKey> = [PhysicalKey::KeyX].into_iter().collect();
        reconcile_held(
            &mut staged,
            &polled,
            |_: PhysicalKey| None::<Code>,
            |_: &Code| None::<&'static [PhysicalKey]>,
        );
        assert!(staged.contains(&Code::W));
    }
}
