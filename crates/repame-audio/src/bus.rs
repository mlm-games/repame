//! [`BusTree`]: named buses with parent routing, send routing, effect chains,
//! and sidechain keys.
//!
//! The four legacy [`AudioChannel`] buses (`master`, `music`, `sfx`, `ui`)
//! are the first nodes, so a game that never touches the tree gets exactly
//! the flat mixer it had before. Extra buses are added by name and route to
//! a parent; a bus may also send into another bus and may drive a
//! compressor's sidechain key.
//!
//! Routing is a DAG. Three edge kinds constrain the audio-thread render
//! order: a child renders before its parent, a send source before its
//! target, and a sidechain key before the bus that keys on it. Every
//! mutation validates the whole edge set and rejects a cycle with
//! [`BusError::Cycle`]; a rejected mutation leaves the tree untouched.

use std::collections::HashMap;
use std::fmt;

use crate::AudioChannel;
use crate::effects::EffectSpec;

/// Stable index of a bus inside one [`BusTree`].
///
/// Ids come from [`BusTree::new`] (the four built-ins), [`BusId::channel`],
/// or [`BusTree::add_bus`]. An id is only valid for the tree it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BusId(usize);

/// The master bus, always index `0` and always the tree root.
pub const MASTER: BusId = BusId(0);

impl BusId {
    /// Master bus (index `0`).
    pub const fn master() -> Self {
        MASTER
    }

    /// Built-in bus behind an [`AudioChannel`].
    pub const fn channel(channel: AudioChannel) -> Self {
        match channel {
            AudioChannel::Master => BusId(0),
            AudioChannel::Music => BusId(1),
            AudioChannel::Sfx => BusId(2),
            AudioChannel::Ui => BusId(3),
        }
    }

    /// Raw node index (stable inside one tree).
    pub const fn index(self) -> usize {
        self.0
    }
}

impl From<AudioChannel> for BusId {
    fn from(channel: AudioChannel) -> Self {
        Self::channel(channel)
    }
}

/// Rejected bus-tree mutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BusError {
    /// No bus goes by that name.
    UnknownBus(String),
    /// A bus with that name already exists.
    DuplicateBus(String),
    /// The name is empty or already used by a built-in bus.
    InvalidName(String),
    /// Routing the bus would close a loop in the render graph.
    Cycle(String),
}

impl fmt::Display for BusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownBus(name) => write!(f, "unknown audio bus `{name}`"),
            Self::DuplicateBus(name) => write!(f, "audio bus `{name}` already exists"),
            Self::InvalidName(name) => write!(f, "invalid audio bus name `{name}`"),
            Self::Cycle(name) => {
                write!(f, "audio bus `{name}` would close a routing cycle")
            }
        }
    }
}

impl std::error::Error for BusError {}

/// A gain-staged side route from one bus into another.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BusSend {
    pub to: BusId,
    pub gain: f32,
}

/// One bus: name, parent, fader, effect chain, sends.
#[derive(Debug, Clone, PartialEq)]
pub struct BusDef {
    pub(crate) name: String,
    pub(crate) parent: Option<BusId>,
    pub(crate) gain: f32,
    pub(crate) effects: Vec<EffectSpec>,
    pub(crate) sends: Vec<BusSend>,
}

/// Named bus tree shared by the game thread and the audio thread.
///
/// The audio thread keeps its own DSP mirror (see `mixer`); this type is the
/// authored description. Send it with [`crate::Audio::sync_buses`].
#[derive(Debug, Clone, PartialEq)]
pub struct BusTree {
    buses: Vec<BusDef>,
    names: HashMap<String, usize>,
}

impl Default for BusTree {
    fn default() -> Self {
        Self::new()
    }
}

impl BusTree {
    /// Flat default mixer: `music` (0.8), `sfx`, and `ui` all under `master`.
    pub fn new() -> Self {
        let buses = vec![
            BusDef {
                name: "master".to_string(),
                parent: None,
                gain: 1.0,
                effects: Vec::new(),
                sends: Vec::new(),
            },
            BusDef {
                name: "music".to_string(),
                parent: Some(MASTER),
                gain: 0.8,
                effects: Vec::new(),
                sends: Vec::new(),
            },
            BusDef {
                name: "sfx".to_string(),
                parent: Some(MASTER),
                gain: 1.0,
                effects: Vec::new(),
                sends: Vec::new(),
            },
            BusDef {
                name: "ui".to_string(),
                parent: Some(MASTER),
                gain: 1.0,
                effects: Vec::new(),
                sends: Vec::new(),
            },
        ];
        let mut names = HashMap::new();
        for (i, bus) in buses.iter().enumerate() {
            names.insert(bus.name.clone(), i);
        }
        Self { buses, names }
    }

    /// Number of buses (built-ins included).
    pub fn len(&self) -> usize {
        self.buses.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buses.is_empty()
    }

    /// Look up a bus by name.
    pub fn id(&self, name: &str) -> Option<BusId> {
        self.names.get(name).copied().map(BusId)
    }

    /// Every bus as `(id, def)` in index order.
    pub fn iter(&self) -> impl Iterator<Item = (BusId, &BusDef)> {
        self.buses.iter().enumerate().map(|(i, b)| (BusId(i), b))
    }

    fn def(&self, bus: BusId) -> Option<&BusDef> {
        self.buses.get(bus.0)
    }

    fn def_mut(&mut self, bus: BusId) -> Option<&mut BusDef> {
        self.buses.get_mut(bus.0)
    }

    fn name_of(&self, bus: BusId) -> String {
        self.def(bus)
            .map(|b| b.name.clone())
            .unwrap_or_else(|| format!("#{}", bus.0))
    }

    /// Bus name.
    pub fn name(&self, bus: BusId) -> Option<&str> {
        self.def(bus).map(|b| b.name.as_str())
    }

    /// Parent bus, or `None` for the root.
    pub fn parent(&self, bus: BusId) -> Option<BusId> {
        self.def(bus).and_then(|b| b.parent)
    }

    /// Bus fader (linear).
    pub fn gain(&self, bus: BusId) -> f32 {
        self.def(bus).map(|b| b.gain).unwrap_or(0.0)
    }

    /// Effect chain in processing order.
    pub fn effects(&self, bus: BusId) -> &[EffectSpec] {
        self.def(bus).map(|b| b.effects.as_slice()).unwrap_or(&[])
    }

    /// Side routes out of this bus.
    pub fn sends(&self, bus: BusId) -> &[BusSend] {
        self.def(bus).map(|b| b.sends.as_slice()).unwrap_or(&[])
    }

    /// Effective linear gain from this bus to the master output, faders of
    /// every bus on the way included (effects excluded).
    pub fn gain_to_master(&self, bus: BusId) -> f32 {
        let mut gain = 1.0;
        let mut cursor = Some(bus);
        while let Some(id) = cursor {
            gain *= self.gain(id);
            cursor = self.parent(id);
        }
        gain
    }

    /// Direct children of a bus.
    pub fn children(&self, bus: BusId) -> Vec<BusId> {
        self.buses
            .iter()
            .enumerate()
            .filter(|(_, b)| b.parent == Some(bus))
            .map(|(i, _)| BusId(i))
            .collect()
    }

    /// Remove a bus. Its children re-parent to its parent, and any send or
    /// sidechain reference to it is dropped.
    pub fn remove_bus(&mut self, bus: BusId) -> Result<(), BusError> {
        if bus == MASTER {
            return Err(BusError::Cycle(self.name_of(bus)));
        }
        if self.def(bus).is_none() {
            return Err(BusError::UnknownBus(self.name_of(bus)));
        }
        let parent = self.parent(bus);
        let removed = bus.0;
        self.buses.remove(removed);
        for b in &mut self.buses {
            if b.parent == Some(bus) {
                b.parent = parent;
            }
            b.sends.retain(|s| s.to != bus);
            for e in &mut b.effects {
                if let EffectSpec::Compressor { sidechain, .. } = e
                    && *sidechain == Some(bus)
                {
                    *sidechain = None;
                }
            }
        }
        // Indices above `removed` shifted down.
        let shift = |id: BusId| if id.0 > removed { BusId(id.0 - 1) } else { id };
        for b in &mut self.buses {
            b.parent = b.parent.map(shift);
            for s in &mut b.sends {
                s.to = shift(s.to);
            }
            for e in &mut b.effects {
                if let EffectSpec::Compressor { sidechain, .. } = e {
                    *sidechain = sidechain.map(shift);
                }
            }
        }
        self.reindex_names();
        Ok(())
    }

    fn reindex_names(&mut self) {
        self.names.clear();
        for (i, bus) in self.buses.iter().enumerate() {
            self.names.insert(bus.name.clone(), i);
        }
    }

    /// Add a bus under `parent`. Fails on a duplicate name, an unknown
    /// parent, or a routing cycle.
    pub fn add_bus(&mut self, name: &str, parent: BusId) -> Result<BusId, BusError> {
        let name = name.trim();
        if name.is_empty() {
            return Err(BusError::InvalidName(name.to_string()));
        }
        if self.names.contains_key(name) {
            return Err(BusError::DuplicateBus(name.to_string()));
        }
        if self.def(parent).is_none() {
            return Err(BusError::UnknownBus(format!("#{}", parent.0)));
        }
        let id = BusId(self.buses.len());
        self.buses.push(BusDef {
            name: name.to_string(),
            parent: Some(parent),
            gain: 1.0,
            effects: Vec::new(),
            sends: Vec::new(),
        });
        if self.acyclic() {
            self.names.insert(name.to_string(), id.0);
            Ok(id)
        } else {
            self.buses.pop();
            Err(BusError::Cycle(name.to_string()))
        }
    }

    /// Route a bus to a new parent (`None` only for the root). Rejects a
    /// parent that would close a cycle.
    pub fn set_parent(&mut self, bus: BusId, parent: Option<BusId>) -> Result<(), BusError> {
        let name = self.name_of(bus);
        let Some(def) = self.def(bus) else {
            return Err(BusError::UnknownBus(name));
        };
        if bus == MASTER {
            if parent.is_some() {
                return Err(BusError::Cycle(name));
            }
            return Ok(());
        }
        if let Some(p) = parent {
            if self.def(p).is_none() {
                return Err(BusError::UnknownBus(format!("#{}", p.0)));
            }
            if p == bus {
                return Err(BusError::Cycle(name));
            }
        }
        let previous = def.parent;
        self.buses[bus.0].parent = parent;
        if self.acyclic() {
            Ok(())
        } else {
            self.buses[bus.0].parent = previous;
            Err(BusError::Cycle(name))
        }
    }

    /// Set a bus fader (clamped to `>= 0`).
    pub fn set_gain(&mut self, bus: BusId, gain: f32) -> Result<(), BusError> {
        let name = self.name_of(bus);
        let Some(def) = self.def_mut(bus) else {
            return Err(BusError::UnknownBus(name));
        };
        def.gain = gain.max(0.0);
        Ok(())
    }

    /// Replace a bus effect chain.
    pub fn set_effects(
        &mut self,
        bus: BusId,
        effects: Vec<EffectSpec>,
    ) -> Result<(), BusError> {
        let name = self.name_of(bus);
        if self.def(bus).is_none() {
            return Err(BusError::UnknownBus(name));
        }
        for e in &effects {
            self.check_effect_refs(e)?;
        }
        let previous = std::mem::replace(&mut self.buses[bus.0].effects, effects);
        if self.acyclic() {
            Ok(())
        } else {
            self.buses[bus.0].effects = previous;
            Err(BusError::Cycle(name))
        }
    }

    /// Append one effect to a bus chain.
    pub fn push_effect(&mut self, bus: BusId, effect: EffectSpec) -> Result<(), BusError> {
        self.check_effect_refs(&effect)?;
        let name = self.name_of(bus);
        let Some(def) = self.def_mut(bus) else {
            return Err(BusError::UnknownBus(name));
        };
        def.effects.push(effect);
        if self.acyclic() {
            Ok(())
        } else {
            self.buses[bus.0].effects.pop();
            Err(BusError::Cycle(name))
        }
    }

    fn check_effect_refs(&self, effect: &EffectSpec) -> Result<(), BusError> {
        if let EffectSpec::Compressor {
            sidechain: Some(key),
            ..
        } = effect
            && self.def(*key).is_none()
        {
            return Err(BusError::UnknownBus(format!("#{}", key.0)));
        }
        Ok(())
    }

    /// Add a side route. Repeated calls with the same target replace the
    /// gain instead of stacking.
    pub fn add_send(&mut self, from: BusId, to: BusId, gain: f32) -> Result<(), BusError> {
        if self.def(from).is_none() {
            return Err(BusError::UnknownBus(format!("#{}", from.0)));
        }
        if self.def(to).is_none() {
            return Err(BusError::UnknownBus(format!("#{}", to.0)));
        }
        let previous = self.buses[from.0].sends.clone();
        match self.buses[from.0].sends.iter_mut().find(|s| s.to == to) {
            Some(s) => s.gain = gain.max(0.0),
            None => self.buses[from.0].sends.push(BusSend {
                to,
                gain: gain.max(0.0),
            }),
        }
        if self.acyclic() {
            Ok(())
        } else {
            self.buses[from.0].sends = previous;
            Err(BusError::Cycle(self.name_of(from)))
        }
    }

    /// Drop every side route out of a bus.
    pub fn clear_sends(&mut self, bus: BusId) -> Result<(), BusError> {
        let name = self.name_of(bus);
        let Some(def) = self.def_mut(bus) else {
            return Err(BusError::UnknownBus(name));
        };
        def.sends.clear();
        Ok(())
    }

    /// Full structural check: known references and an acyclic render graph.
    pub fn validate(&self) -> Result<(), BusError> {
        for (id, def) in self.iter() {
            if let Some(parent) = def.parent
                && self.def(parent).is_none()
            {
                return Err(BusError::UnknownBus(format!("#{}", parent.0)));
            }
            for s in &def.sends {
                if self.def(s.to).is_none() {
                    return Err(BusError::UnknownBus(format!("#{}", s.to.0)));
                }
            }
            for e in &def.effects {
                self.check_effect_refs(e)?;
            }
            if id == MASTER && def.parent.is_some() {
                return Err(BusError::Cycle(def.name.clone()));
            }
        }
        if !self.acyclic() {
            let name = self
                .iter()
                .find(|(_, d)| d.parent.is_some())
                .map(|(_, d)| d.name.clone())
                .unwrap_or_default();
            return Err(BusError::Cycle(name));
        }
        Ok(())
    }

    /// Render order: every bus after everything it depends on. Children come
    /// before parents, send sources before targets, sidechain keys before the
    /// buses they key. Ties break on index, so the order is deterministic.
    pub fn render_order(&self) -> Vec<BusId> {
        self.topological().unwrap_or_else(|_| vec![MASTER])
    }

    fn topological(&self) -> Result<Vec<BusId>, BusError> {
        let n = self.buses.len();
        let edges = self.edges();
        let mut indegree = vec![0usize; n];
        let mut adjacency = vec![Vec::new(); n];
        for &(u, v) in &edges {
            adjacency[u].push(v);
            indegree[v] += 1;
        }
        let mut order = Vec::with_capacity(n);
        let mut ready: Vec<usize> = (0..n).filter(|i| indegree[*i] == 0).collect();
        while let Some(pos) = ready
            .iter()
            .enumerate()
            .min_by_key(|(_, i)| **i)
            .map(|(p, _)| p)
        {
            let u = ready.swap_remove(pos);
            order.push(BusId(u));
            for &v in &adjacency[u] {
                indegree[v] -= 1;
                if indegree[v] == 0 {
                    ready.push(v);
                }
            }
        }
        if order.len() == n {
            Ok(order)
        } else {
            Err(BusError::Cycle(String::new()))
        }
    }

    fn edges(&self) -> Vec<(usize, usize)> {
        let mut edges = Vec::new();
        for (i, def) in self.buses.iter().enumerate() {
            if let Some(parent) = def.parent {
                edges.push((i, parent.0));
            }
            for s in &def.sends {
                edges.push((i, s.to.0));
            }
            for e in &def.effects {
                if let EffectSpec::Compressor {
                    sidechain: Some(key),
                    ..
                } = e
                {
                    edges.push((key.0, i));
                }
            }
        }
        edges
    }

    fn acyclic(&self) -> bool {
        self.topological().is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::FilterMode;

    #[test]
    fn default_tree_mirrors_the_flat_mixer() {
        let tree = BusTree::new();
        assert_eq!(tree.len(), 4);
        assert_eq!(tree.id("master"), Some(MASTER));
        assert_eq!(tree.id("music"), Some(BusId::channel(AudioChannel::Music)));
        assert_eq!(tree.parent(BusId::channel(AudioChannel::Sfx)), Some(MASTER));
        assert!((tree.gain(BusId::channel(AudioChannel::Music)) - 0.8).abs() < 1e-6);
        assert!((tree.gain_to_master(BusId::channel(AudioChannel::Sfx)) - 1.0).abs() < 1e-6);
        tree.validate().expect("default tree is valid");
    }

    #[test]
    fn cycles_are_rejected_and_leave_the_tree_intact() {
        let mut tree = BusTree::new();
        let music = BusId::channel(AudioChannel::Music);
        let bus = tree.add_bus("music_sub", music).expect("adds");
        assert_eq!(tree.parent(bus), Some(music));
        // Music may not become a child of its own child.
        let err = tree.set_parent(music, Some(bus)).unwrap_err();
        assert!(matches!(err, BusError::Cycle(_)), "{err:?}");
        assert_eq!(tree.parent(music), Some(MASTER));
        assert_eq!(tree.len(), 5);

        // A send that closes the loop is refused too: `music` feeding its own
        // child would demand music renders first, but the child already
        // renders first as music's child.
        let err = tree.add_send(music, bus, 0.5).unwrap_err();
        assert!(matches!(err, BusError::Cycle(_)));
        assert!(tree.sends(music).is_empty());
        // The same pair the other way is no cycle: a child sending into its
        // parent repeats the parent edge, so the order is unchanged.
        tree.add_send(bus, music, 0.5)
            .expect("forward send is fine");
        tree.validate().expect("still valid");
    }

    #[test]
    fn sidechain_key_feeds_render_order() {
        let mut tree = BusTree::new();
        let sfx = BusId::channel(AudioChannel::Sfx);
        let music = BusId::channel(AudioChannel::Music);
        tree.push_effect(
            music,
            EffectSpec::Compressor {
                threshold_db: -30.0,
                ratio: 4.0,
                attack_ms: 5.0,
                release_ms: 100.0,
                makeup_db: 0.0,
                mix: 1.0,
                sidechain: Some(sfx),
            },
        )
        .expect("key exists");
        let order = tree.render_order();
        let key = order.iter().position(|b| *b == sfx).unwrap();
        let target = order.iter().position(|b| *b == music).unwrap();
        assert!(key < target, "key must render first: {order:?}");
        // A key downstream of the bus it keys is a cycle: `sfx` sends into
        // `music`, so music cannot render before it.
        let mut tree = BusTree::new();
        let mut spec = EffectSpec::compressor(-24.0, 4.0);
        if let EffectSpec::Compressor { sidechain, .. } = &mut spec {
            *sidechain = Some(music);
        }
        tree.add_send(sfx, music, 0.5).expect("send");
        let err = tree.push_effect(sfx, spec).unwrap_err();
        assert!(matches!(err, BusError::Cycle(_)), "{err:?}");
        assert!(tree.effects(sfx).is_empty(), "rejected effect is not kept");
    }

    /// Removing a bus re-parents its children onto its parent, drops every
    /// send and sidechain reference to it, and compacts the ids above it so
    /// the surviving names still resolve.
    #[test]
    fn removing_a_bus_relinks_children_and_drops_references() {
        let mut tree = BusTree::new();
        let sfx = BusId::channel(AudioChannel::Sfx);
        let sub = tree.add_bus("sfx_weapons", sfx).expect("adds");
        tree.add_bus("sfx_weapons_dry", sub).expect("adds");
        let aux = tree.add_bus("aux", MASTER).expect("adds");
        let ui = BusId::channel(AudioChannel::Ui);
        // Both reference kinds point *at* the bus that goes away.
        tree.add_send(ui, sub, 0.25).expect("send");
        tree.push_effect(aux, EffectSpec::sidechain_duck(sub, -30.0, 4.0))
            .expect("effect");
        tree.push_effect(
            ui,
            EffectSpec::Filter {
                mode: FilterMode::LowPass,
                cutoff_hz: 2000.0,
                resonance: 0.5,
                stages: 1,
            },
        )
        .expect("effect");
        tree.remove_bus(sub).expect("removable");
        assert_eq!(tree.len(), 6);
        assert!(tree.id("sfx_weapons").is_none());
        // The child re-parented onto `sfx` and shifted down; `aux` was above
        // the hole, so it shifted too and still resolves by name.
        let dry = tree.id("sfx_weapons_dry").expect("child survives");
        let aux = tree.id("aux").expect("bus above the hole survives");
        assert_eq!(dry.index(), 4);
        assert_eq!(tree.parent(dry), Some(sfx));
        assert_eq!(aux.index(), 5);
        assert_eq!(tree.parent(aux), Some(MASTER));
        // `ui` sits below the hole, so its id still names its bus.
        assert_eq!(tree.name(ui), Some("ui"));
        // The send and the sidechain key dropped; the effects themselves stay.
        assert!(tree.sends(ui).is_empty());
        assert_eq!(tree.effects(ui).len(), 1);
        assert_eq!(tree.effects(aux).len(), 1);
        let key = match &tree.effects(aux)[0] {
            EffectSpec::Compressor { sidechain, .. } => *sidechain,
            other => panic!("expected a compressor, got {other:?}"),
        };
        assert_eq!(key, None, "sidechain reference must drop");
        tree.validate().expect("valid after removal");
    }

    #[test]
    fn names_and_gains_are_typed_errors_not_panics() {
        let mut tree = BusTree::new();
        assert!(matches!(
            tree.add_bus("music", MASTER),
            Err(BusError::DuplicateBus(_))
        ));
        assert!(matches!(
            tree.add_bus("  ", MASTER),
            Err(BusError::InvalidName(_))
        ));
        let missing = BusId::master();
        assert!(tree.set_gain(missing, 0.5).is_ok());
        let mut orphan = BusTree::new();
        let id = orphan.add_bus("x", MASTER).expect("adds");
        orphan.remove_bus(id).expect("removes");
        assert!(matches!(
            orphan.set_gain(id, 0.5),
            Err(BusError::UnknownBus(_))
        ));
        assert!(matches!(
            orphan.add_bus("y", id),
            Err(BusError::UnknownBus(_))
        ));
    }
}
