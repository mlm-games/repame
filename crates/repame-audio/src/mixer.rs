//! Audio-thread mirror of a [`crate::BusTree`]: per-bus mix buffers, effect
//! instances, and the routing pass that folds every bus into the master.
//!
//! One render pass per block segment:
//! 1. voices are summed into their bus's mix buffer (the engine does this),
//! 2. each bus is visited in [`BusTree::render_order`] — fader first, then the
//!    effect chain, then a snapshot for sidechain keys,
//! 3. the bus is added to its parent and to each send target,
//! 4. the master buffer becomes the output block.
//!
//! Because the render order is topological over parent, send, and sidechain
//! edges, a sidechain key bus is always complete before the compressor that
//! listens to it runs. Buffers are reused between blocks; nothing in the pass
//! allocates once the block size has settled.

use crate::bus::BusTree;
use crate::effects::EffectDsp;

#[derive(Default)]
struct RtNode {
    parent: Option<usize>,
    gain: f32,
    effects: Vec<EffectDsp>,
    sends: Vec<(usize, f32)>,
}

/// DSP mirror of the authored bus tree.
pub(crate) struct BusGraph {
    nodes: Vec<RtNode>,
    order: Vec<usize>,
    rate: u32,
    buffers: Vec<Vec<f32>>,
    keys: Vec<Vec<f32>>,
}

impl BusGraph {
    /// Empty graph; call [`BusGraph::sync`] with a tree.
    pub(crate) fn new(rate: u32) -> Self {
        Self {
            nodes: Vec::new(),
            order: Vec::new(),
            rate,
            buffers: Vec::new(),
            keys: Vec::new(),
        }
    }

    /// Number of buses mirrored.
    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Rebuild from an authored tree, keeping the DSP state of effects that
    /// kept the same position and kind (so a fader move does not flush a
    /// reverb tail).
    pub(crate) fn sync(&mut self, tree: &BusTree) {
        let n = tree.len();
        let rate = self.rate;
        self.nodes.truncate(n);
        self.buffers.truncate(n);
        self.keys.truncate(n);
        // Grow to the tree's bus count before indexing: a first sync against
        // a tree with more buses than mirrored nodes used to panic.
        if self.nodes.len() < n {
            self.nodes.resize_with(n, RtNode::default);
            self.buffers.resize_with(n, Vec::new);
            self.keys.resize_with(n, Vec::new);
        }
        for (i, (_, def)) in tree.iter().enumerate() {
            let mut old = std::mem::take(&mut self.nodes[i].effects).into_iter();
            let mut effects = Vec::with_capacity(def.effects.len());
            for spec in &def.effects {
                let mut reused = old.next();
                let dsp = match reused.as_mut() {
                    Some(dsp) if dsp.same_kind(spec) => {
                        dsp.set_params(spec);
                        dsp.set_rate(rate);
                        reused.take().expect("checked above")
                    }
                    _ => EffectDsp::new(spec, rate),
                };
                effects.push(dsp);
            }
            self.nodes[i] = RtNode {
                parent: def.parent.map(|p| p.index()),
                gain: def.gain,
                effects,
                sends: def.sends.iter().map(|s| (s.to.index(), s.gain)).collect(),
            };
        }
        for _ in self.nodes.len()..n {
            self.nodes.push(RtNode {
                parent: None,
                gain: 1.0,
                effects: Vec::new(),
                sends: Vec::new(),
            });
        }
        for _ in self.buffers.len()..n {
            self.buffers.push(Vec::new());
            self.keys.push(Vec::new());
        }
        self.order = tree.render_order().iter().map(|b| b.index()).collect();
    }

    /// Live fader move (no rebuild, no DSP flush).
    pub(crate) fn set_gain(&mut self, bus: usize, gain: f32) {
        if let Some(node) = self.nodes.get_mut(bus) {
            node.gain = gain.max(0.0);
        }
    }

    /// True when `bus` is `root` or one of its descendants. Used by
    /// `StopBus`, so stopping a bus stops everything routed under it.
    pub(crate) fn descends_from(&self, bus: usize, root: usize) -> bool {
        let mut cursor = Some(bus);
        let mut steps = 0;
        while let Some(i) = cursor {
            if i == root {
                return true;
            }
            cursor = self.nodes.get(i).and_then(|n| n.parent);
            steps += 1;
            if steps > self.nodes.len() {
                break;
            }
        }
        false
    }

    /// Re-configure every effect for a new device rate.
    pub(crate) fn prepare(&mut self, rate: u32) {
        if self.rate == rate {
            return;
        }
        self.rate = rate;
        for node in &mut self.nodes {
            for e in &mut node.effects {
                e.set_rate(rate);
            }
        }
    }

    /// Zero the mix buffers for a `frames`-long segment.
    pub(crate) fn begin(&mut self, frames: usize) {
        let len = frames * 2;
        for buf in &mut self.buffers {
            if buf.len() < len {
                buf.resize(len, 0.0);
            } else {
                buf[..len].fill(0.0);
            }
        }
    }

    /// Mix buffer for one bus: interleaved stereo, `frames * 2` long. An
    /// out-of-range bus (stale id after a tree edit) folds into the master.
    pub(crate) fn bus_buffer(&mut self, bus: usize, frames: usize) -> &mut [f32] {
        let index = if bus < self.buffers.len() { bus } else { 0 };
        let len = frames * 2;
        let buf = &mut self.buffers[index];
        if buf.len() < len {
            buf.resize(len, 0.0);
        }
        &mut buf[..len]
    }

    /// Run effects and routing, then write the master mix into `out`.
    /// `out` is overwritten, not accumulated.
    pub(crate) fn process(&mut self, frames: usize, out: &mut [f32], out_channels: usize) {
        let len = frames * 2;
        if self.nodes.is_empty() || self.buffers.is_empty() {
            out.fill(0.0);
            return;
        }
        let Self {
            nodes,
            buffers,
            keys,
            order,
            ..
        } = self;
        for &i in order.iter() {
            if i >= nodes.len() || i >= buffers.len() {
                continue;
            }
            let gain = nodes[i].gain;
            if gain != 1.0 {
                for v in &mut buffers[i][..len] {
                    *v *= gain;
                }
            }
            let effects = &mut nodes[i].effects;
            for e in effects.iter_mut() {
                let key = e
                    .sidechain_key()
                    .filter(|k| *k < keys.len())
                    .map(|k| &keys[k][..len.min(keys[k].len())]);
                e.process(&mut buffers[i], frames, key);
            }
            if keys[i].len() < len {
                keys[i].resize(len, 0.0);
            }
            keys[i][..len].copy_from_slice(&buffers[i][..len]);
            if let Some(parent) = nodes[i].parent {
                add_scaled(buffers, i, parent, 1.0, len);
            }
            for (target, send_gain) in nodes[i].sends.clone() {
                add_scaled(buffers, i, target, send_gain, len);
            }
        }
        write_output(buffers, out, frames, out_channels);
    }
}

/// `dst += src * scale` across two mix buffers without aliasing.
fn add_scaled(buffers: &mut [Vec<f32>], from: usize, to: usize, scale: f32, len: usize) {
    if from == to || to >= buffers.len() || scale == 0.0 {
        return;
    }
    if from < to {
        let (lo, hi) = buffers.split_at_mut(to);
        let src = &lo[from][..len];
        let dst = &mut hi[0][..len];
        for (d, s) in dst.iter_mut().zip(src) {
            *d += *s * scale;
        }
    } else {
        let (lo, hi) = buffers.split_at_mut(from);
        let dst = &mut lo[to][..len];
        let src = &hi[0][..len];
        for (d, s) in dst.iter_mut().zip(src) {
            *d += *s * scale;
        }
    }
}

/// Master mix buffer to the device layout (stereo passthrough or mono sum).
fn write_output(buffers: &[Vec<f32>], out: &mut [f32], frames: usize, out_channels: usize) {
    let master = &buffers[0];
    let len = frames * 2;
    if master.len() < len {
        out.fill(0.0);
        return;
    }
    if out_channels >= 2 {
        let n = (frames * out_channels).min(out.len());
        for i in 0..n {
            out[i] = master[i];
        }
    } else {
        for i in 0..frames.min(out.len()) {
            out[i] = (master[i * 2] + master[i * 2 + 1]) * 0.5;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioChannel;
    use crate::bus::{BusId, MASTER};
    use crate::effects::EffectSpec;

    fn graph(tree: &BusTree, rate: u32) -> BusGraph {
        let mut g = BusGraph::new(rate);
        g.sync(tree);
        g
    }

    fn fill(g: &mut BusGraph, bus: usize, frames: usize, l: f32, r: f32) {
        let buf = g.bus_buffer(bus, frames);
        for frame in buf.chunks_mut(2) {
            frame[0] = l;
            frame[1] = r;
        }
    }

    fn render(g: &mut BusGraph, frames: usize) -> Vec<f32> {
        let mut out = vec![0.0; frames * 2];
        g.process(frames, &mut out, 2);
        out
    }

    /// Golden gain staging: a fixed `1.0` fed into a three-level chain must
    /// leave the master at the exact product of the faders.
    #[test]
    fn gain_staging_through_the_tree_is_exact() {
        let mut tree = BusTree::new();
        let sfx = BusId::channel(AudioChannel::Sfx);
        let sub = tree.add_bus("sfx_weapons", sfx).expect("adds");
        tree.set_gain(sfx, 0.5).expect("gain");
        tree.set_gain(sub, 0.5).expect("gain");
        tree.set_gain(MASTER, 0.8).expect("gain");
        let mut g = graph(&tree, 48000);
        g.begin(16);
        fill(&mut g, sub.index(), 16, 1.0, 1.0);
        let out = render(&mut g, 16);
        // 1.0 * 0.5 * 0.5 * 0.8 = 0.2
        for v in out {
            assert!((v - 0.2).abs() < 1e-6, "v = {v}");
        }
    }

    /// A send adds a second, independently gain-staged copy of the source.
    #[test]
    fn sends_add_a_parallel_copy() {
        let mut tree = BusTree::new();
        let sfx = BusId::channel(AudioChannel::Sfx);
        let ui = BusId::channel(AudioChannel::Ui);
        tree.set_gain(MASTER, 1.0).expect("gain");
        tree.set_gain(sfx, 1.0).expect("gain");
        tree.set_gain(ui, 0.25).expect("gain");
        tree.add_send(sfx, ui, 0.5).expect("send");
        let mut g = graph(&tree, 48000);
        g.begin(8);
        fill(&mut g, sfx.index(), 8, 1.0, 1.0);
        let out = render(&mut g, 8);
        // Direct 1.0 + ui(0.5 * 0.25) = 1.125
        for v in out {
            assert!((v - 1.125).abs() < 1e-6, "v = {v}");
        }
    }

    /// Golden ducking: a loud `sfx` key drives a compressor on `music`, and
    /// the gain reduction is pinned at the first sample and after 4800
    /// samples (values from the documented Godot compressor recursion).
    #[test]
    fn sidechain_ducking_gain_reduction_is_pinned() {
        let mut tree = BusTree::new();
        let music = BusId::channel(AudioChannel::Music);
        let sfx = BusId::channel(AudioChannel::Sfx);
        tree.set_gain(MASTER, 1.0).expect("gain");
        tree.set_gain(music, 1.0).expect("gain");
        tree.set_gain(sfx, 1.0).expect("gain");
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
        .expect("key is routable");
        let mut g = graph(&tree, 48000);
        let frames = 4800;
        g.begin(frames);
        fill(&mut g, sfx.index(), frames, 1.0, 1.0);
        fill(&mut g, music.index(), frames, 1.0, 1.0);
        let out = render(&mut g, frames);
        // `sfx` passes at unity into the same master mix, so the music duck
        // gain is the output minus that copy: 0.9778312 on the first sample,
        // 0.0045547 once the attack has settled.
        assert!(
            (out[0] - 1.0 - 0.9778312).abs() < 1e-5,
            "sample 0 = {}",
            out[0]
        );
        assert!(
            (out[(frames - 1) * 2] - 1.0 - 0.0045547).abs() < 1e-5,
            "settled = {}",
            out[(frames - 1) * 2]
        );
        // Sfx itself passes untouched: with the music bus silent and the same
        // compressor still keyed on sfx, every output sample is sfx alone.
        let mut quiet = graph(&tree, 48000);
        quiet.begin(frames);
        fill(&mut quiet, sfx.index(), frames, 1.0, 1.0);
        for v in render(&mut quiet, frames) {
            assert!((v - 1.0).abs() < 1e-6, "key bus is ducked: {v}");
        }
    }

    /// Without a key the compressor must not touch the signal.
    #[test]
    fn compressor_without_sidechain_is_flat_below_threshold() {
        let mut tree = BusTree::new();
        let music = BusId::channel(AudioChannel::Music);
        tree.set_gain(music, 1.0).expect("gain");
        tree.push_effect(music, EffectSpec::compressor(-6.0, 4.0))
            .expect("effect");
        let mut g = graph(&tree, 48000);
        g.begin(64);
        fill(&mut g, music.index(), 64, 0.1, 0.1);
        let out = render(&mut g, 64);
        for v in out {
            assert!((v - 0.1).abs() < 1e-6, "v = {v}");
        }
    }

    /// Re-syncing a fader keeps effect state; re-syncing a non-empty chain
    /// keeps the tail alive rather than resetting it.
    #[test]
    fn resync_keeps_effect_state() {
        let mut tree = BusTree::new();
        let ui = BusId::channel(AudioChannel::Ui);
        tree.set_gain(MASTER, 1.0).expect("gain");
        tree.push_effect(ui, EffectSpec::delay(10.0, 0.5, 1.0))
            .expect("effect");
        let mut g = graph(&tree, 48000);
        g.begin(480);
        fill(&mut g, ui.index(), 480, 1.0, 1.0);
        let _ = render(&mut g, 480);
        tree.set_gain(ui, 0.5).expect("gain");
        g.sync(&tree);
        g.begin(480);
        fill(&mut g, ui.index(), 480, 0.0, 0.0);
        let tail = render(&mut g, 480);
        assert!(
            tail.iter().any(|v| v.abs() > 1e-6),
            "delay tail must survive a fader move"
        );
    }

    #[test]
    fn mono_output_downs_mixes() {
        let tree = BusTree::new();
        let mut g = graph(&tree, 48000);
        let sfx = BusId::channel(AudioChannel::Sfx).index();
        g.begin(4);
        fill(&mut g, sfx, 4, 1.0, 0.0);
        let mut out = vec![0.0; 4];
        g.process(4, &mut out, 1);
        for v in out {
            assert!((v - 0.5).abs() < 1e-6, "v = {v}");
        }
    }
}
