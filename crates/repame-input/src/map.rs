//! Action map: bindings grouped per action, gated by contexts.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use super::binding::Binding;

/// Named set of actions, e.g. `"gameplay"` vs `"menu"`. When a map has
/// no contexts every action is live; otherwise only actions in an
/// active context fire. Mirrors repose `InstallShortcutMap` scopes,
/// but for sim-side game actions instead of UI shortcuts.
#[derive(Clone, Debug, Default)]
pub struct ActionMap<A> {
    bindings: HashMap<A, Vec<Binding>>,
    contexts: HashMap<String, HashSet<A>>,
}

impl<A: Clone + Eq + Hash> ActionMap<A> {
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
            contexts: HashMap::new(),
        }
    }

    /// Bind one more physical source to an action (many-to-one).
    pub fn bind(&mut self, action: A, binding: Binding) -> &mut Self {
        self.bindings.entry(action).or_default().push(binding);
        self
    }

    /// Put an action in a named context (member of several allowed).
    pub fn in_context(&mut self, context: &str, action: A) -> &mut Self {
        self.contexts
            .entry(context.to_string())
            .or_default()
            .insert(action);
        self
    }

    /// All bindings driving `action` (empty if unbound).
    pub fn bindings_for(&self, action: &A) -> &[Binding] {
        self.bindings.get(action).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Actions holding at least one binding.
    pub fn actions(&self) -> impl Iterator<Item = &A> {
        self.bindings.keys()
    }

    /// True when the map defines no contexts at all.
    pub fn has_contexts(&self) -> bool {
        !self.contexts.is_empty()
    }

    /// True when `action` may fire under `active` contexts. Actions in
    /// no context are always live (shared buttons like pause).
    pub fn live_in(&self, action: &A, active: &HashSet<String>) -> bool {
        if !self.has_contexts() {
            return true;
        }
        if !self.contexts.values().any(|set| set.contains(action)) {
            return true;
        }
        self.contexts
            .iter()
            .any(|(name, set)| active.contains(name) && set.contains(action))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use repose_core::input::{Key, Modifiers};

    fn chord() -> Binding {
        Binding::Key(repose_core::shortcuts::KeyChord::new(
            Key::Space,
            Modifiers::default(),
        ))
    }

    #[test]
    fn unbound_action_has_no_bindings() {
        let map: ActionMap<&str> = ActionMap::new();
        assert!(map.bindings_for(&"jump").is_empty());
    }

    #[test]
    fn context_gating() {
        let mut map = ActionMap::new();
        map.bind("jump", chord());
        map.bind("pause", chord());
        map.in_context("gameplay", "jump");
        let active: HashSet<String> = ["gameplay".to_string()].into_iter().collect();
        assert!(map.live_in(&"jump", &active));
        assert!(map.live_in(&"pause", &active));
        let menu: HashSet<String> = ["menu".to_string()].into_iter().collect();
        assert!(!map.live_in(&"jump", &menu));
        assert!(map.live_in(&"pause", &menu));
    }
}
