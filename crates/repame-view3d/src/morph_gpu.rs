//! GPU morph-target packing: per-draw slots, weights and vertex deltas.
//!
//! A single storage buffer holds fixed slot/weight regions followed by a
//! runtime-sized delta array. The vertex stage indexes a slot by the draw's
//! skin base, so no extra vertex attributes are needed.

use bytemuck::{Pod, Zeroable};

pub const MAX_MORPH_DRAWS: usize = 64;
pub const MAX_MORPH_TARGETS: usize = 16;
pub const WEIGHT_SLOTS: usize = MAX_MORPH_DRAWS * 4;
pub const MAX_MORPH_DELTAS: usize = 65536;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MorphSlot {
    pub a: [u32; 4],
    pub b: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct MorphDelta {
    pub pos: [f32; 4],
    pub nrm: [f32; 4],
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MorphDraw {
    pub weights: Vec<f32>,
    pub deltas: Vec<MorphDelta>,
    pub vertex_count: u32,
}

#[derive(Clone, Debug, Default)]
pub struct MorphStore {
    pub slots: Vec<MorphSlot>,
    pub weights: Vec<f32>,
    pub deltas: Vec<MorphDelta>,
}

impl MorphStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, draw: &MorphDraw, vertex_base: u32) -> Option<u32> {
        let slot = self.slots.len();
        if slot >= MAX_MORPH_DRAWS
            || draw.weights.len() > MAX_MORPH_TARGETS
            || draw.deltas.len() > MAX_MORPH_DELTAS.saturating_sub(self.deltas.len())
        {
            return None;
        }
        let weight_base = self.weights.len() as u32;
        let delta_base = self.deltas.len() as u32;
        self.slots.push(MorphSlot {
            a: [
                delta_base,
                vertex_base,
                draw.vertex_count,
                draw.weights.len() as u32,
            ],
            b: [weight_base, 0, 0, 0],
        });
        self.weights.extend_from_slice(&draw.weights);
        while self.weights.len() % 4 != 0 {
            self.weights.push(0.0);
        }
        self.deltas.extend_from_slice(&draw.deltas);
        Some(slot as u32)
    }

    pub fn bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let mut slots = vec![MorphSlot::zeroed(); MAX_MORPH_DRAWS];
        slots[..self.slots.len()].copy_from_slice(&self.slots);
        out.extend_from_slice(bytemuck::cast_slice(&slots));
        let mut weights = vec![[0.0f32; 4]; WEIGHT_SLOTS];
        for (chunk, value) in weights.iter_mut().zip(self.weights.chunks(4)) {
            for (slot, weight) in chunk.iter_mut().zip(value) {
                *slot = *weight;
            }
        }
        out.extend_from_slice(bytemuck::cast_slice(&weights));
        out.extend_from_slice(bytemuck::cast_slice(&self.deltas));
        out
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn clear(&mut self) {
        self.slots.clear();
        self.weights.clear();
        self.deltas.clear();
    }
}

pub const MORPH_STRUCT_WGSL: &str = "\
struct MorphSlot {
    a: vec4<u32>,
    b: vec4<u32>,
};
struct MorphDelta {
    pos: vec4<f32>,
    nrm: vec4<f32>,
};
struct MorphData {
    slots: array<MorphSlot, 64>,
    weights: array<vec4<f32>, 256>,
    deltas: array<MorphDelta>,
};
";

pub const MORPH_BINDING_WGSL: &str =
    "@group(0) @binding(2) var<storage, read> morph_data: MorphData;\n";

pub const MORPH_POS_WGSL: &str = "\
        let morph_index = u32(skin_base + 0.5) / 256u;
        if (morph_index < 64u) {
            let slot = morph_data.slots[morph_index];
            let target_count = slot.a.w;
            let vertex_count = slot.a.z;
            let local = vertex_index - slot.a.y;
            if (target_count > 0u && target_count <= 16u && local < vertex_count) {
                var delta_pos = vec3<f32>(0.0);
                for (var k: u32 = 0u; k < target_count; k = k + 1u) {
                    let w = morph_data.weights[slot.b.x / 4u + k / 4u][k % 4u];
                    let d = morph_data.deltas[slot.a.x + k * vertex_count + local];
                    delta_pos = delta_pos + d.pos.xyz * w;
                }
                morph_pos = pos + delta_pos;
            }
        }
";

pub const MORPH_POS_FN_WGSL: &str = "\
fn morph_pose(pos: vec3<f32>, skin_base: f32, vertex_index: u32, skin_mix: f32) -> vec3<f32> {
    if (skin_mix <= 0.5) { return pos; }
    let morph_index = u32(skin_base + 0.5) / 256u;
    var morph_pos = pos;
    if (morph_index < 64u) {
        let slot = morph_data.slots[morph_index];
        let target_count = slot.a.w;
        let vertex_count = slot.a.z;
        let local = vertex_index - slot.a.y;
        if (target_count > 0u && target_count <= 16u && local < vertex_count) {
            var delta_pos = vec3<f32>(0.0);
            for (var k: u32 = 0u; k < target_count; k = k + 1u) {
                let w = morph_data.weights[slot.b.x / 4u + k / 4u][k % 4u];
                let d = morph_data.deltas[slot.a.x + k * vertex_count + local];
                delta_pos = delta_pos + d.pos.xyz * w;
            }
            morph_pos = pos + delta_pos;
        }
    }
    return morph_pos;
}
";

pub const MORPH_POS_FN_PASSTHROUGH_WGSL: &str = "\
fn morph_pose(pos: vec3<f32>, skin_base: f32, vertex_index: u32, skin_mix: f32) -> vec3<f32> {
    return pos;
}
";

pub const MORPH_BODY_WGSL: &str = "\
        let morph_index = u32(skin_base + 0.5) / 256u;
        if (morph_index < 64u) {
            let slot = morph_data.slots[morph_index];
            let target_count = slot.a.w;
            let vertex_count = slot.a.z;
            let local = vertex_index - slot.a.y;
            if (target_count > 0u && target_count <= 16u && local < vertex_count) {
                var delta_pos = vec3<f32>(0.0);
                var delta_nrm = vec3<f32>(0.0);
                for (var k: u32 = 0u; k < target_count; k = k + 1u) {
                    let w = morph_data.weights[slot.b.x / 4u + k / 4u][k % 4u];
                    let d = morph_data.deltas[slot.a.x + k * vertex_count + local];
                    delta_pos = delta_pos + d.pos.xyz * w;
                    delta_nrm = delta_nrm + d.nrm.xyz * w;
                }
                skinned_pos = pos + delta_pos;
                skinned_nrm = normal + delta_nrm;
                skinned_nrm = select(
                    skinned_nrm,
                    normalize(skinned_nrm),
                    dot(skinned_nrm, skinned_nrm) > 1e-10,
                );
            }
        }
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_weights_and_deltas_pack_into_one_buffer() {
        let mut store = MorphStore::new();
        let draw = MorphDraw {
            weights: vec![0.25, 0.5],
            deltas: vec![
                MorphDelta {
                    pos: [1.0, 0.0, 0.0, 0.0],
                    nrm: [0.0, 1.0, 0.0, 0.0],
                },
                MorphDelta {
                    pos: [0.0, 1.0, 0.0, 0.0],
                    nrm: [1.0, 0.0, 0.0, 0.0],
                },
            ],
            vertex_count: 1,
        };
        assert_eq!(store.add(&draw, 12), Some(0));
        let bytes = store.bytes();
        assert_eq!(
            bytes.len(),
            MAX_MORPH_DRAWS * size_of::<MorphSlot>()
                + WEIGHT_SLOTS * 16
                + 2 * size_of::<MorphDelta>()
        );
        let slots: &[MorphSlot] = bytemuck::cast_slice(&bytes[..MAX_MORPH_DRAWS * 32]);
        assert_eq!(slots[0].a, [0, 12, 1, 2]);
        let second = MorphDraw {
            weights: vec![1.0],
            deltas: vec![MorphDelta::zeroed()],
            vertex_count: 1,
        };
        assert_eq!(store.add(&second, 20), Some(1));
        let padded = store.bytes();
        let slots: &[MorphSlot] = bytemuck::cast_slice(&padded[..MAX_MORPH_DRAWS * 32]);
        assert_eq!(slots[1].b[0], 4, "weight bases stay vec4-aligned");
    }
}
