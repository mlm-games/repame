//! Bridges [`ShadingModel`] onto [`MeshGroup`], the batch and the GPU path.
//!
//! A group that carries a shading model routes through the material path:
//! the batch picks the lowered WGSL variant keyed by the model's cache key,
//! binds its constant registers, and the fragment stage evaluates the same
//! combinator chain [`crate::tev::evaluate`] defines. Groups without a model
//! keep the legacy PBR-lite path untouched, so existing games see no change.

use std::collections::HashMap;

use crate::material::{KColors, ShadingError, ShadingModel};
use crate::mesh::MeshGroup;
use crate::stats::FrameStats;
use crate::tev::wgsl_program;

/// Uniform block a material binds: constant colors, ambient and fog.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadingUniform {
    pub kcolors: [[f32; 4]; 4],
    pub ambient: f32,
    pub ao: f32,
    pub fog: [f32; 4],
    pub fog_range: [f32; 2],
    pub light_count: u32,
    pub _pad: u32,
}

impl ShadingUniform {
    /// Packs a model plus its resolved light state for upload.
    pub fn pack(model: &ShadingModel, light_colors: &[[f32; 3]], fog_enabled: bool) -> Self {
        let mut kcolors = [[0.0_f32; 4]; 4];
        for (slot, entry) in model.kcolors.iter().enumerate() {
            kcolors[slot] = *entry;
        }
        Self {
            kcolors,
            ambient: model.ambient as f32,
            ao: model.ao,
            fog: [
                model.material_fog.color[0],
                model.material_fog.color[1],
                model.material_fog.color[2],
                if fog_enabled { 1.0 } else { 0.0 },
            ],
            fog_range: [model.material_fog.near, model.material_fog.far],
            light_count: light_colors.len() as u32,
            _pad: 0,
        }
    }
}

/// Cache of lowered WGSL variants, keyed by [`ShadingModel::cache_key`].
#[derive(Default)]
pub struct ShaderCache {
    variants: HashMap<u64, String>,
}

impl ShaderCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Lowered WGSL for a model, compiled at most once per distinct program.
    pub fn get(&mut self, model: &ShadingModel) -> Result<&str, ShadingError> {
        model.validate()?;
        let key = model.cache_key();
        let source = self
            .variants
            .entry(key)
            .or_insert_with(|| wgsl_program(model));
        Ok(source.as_str())
    }

    pub fn len(&self) -> usize {
        self.variants.len()
    }

    pub fn is_empty(&self) -> bool {
        self.variants.is_empty()
    }

    pub fn clear(&mut self) {
        self.variants.clear();
    }
}

/// A mesh group plus the material it shades with.
#[derive(Clone, Debug)]
pub struct ShadedGroup {
    pub group: MeshGroup,
    pub model: ShadingModel,
    /// Texture units sampled by the program, in declaration order.
    pub texture_pages: Vec<u32>,
}

impl ShadedGroup {
    /// Wraps a group, taking the pages each stage samples.
    pub fn new(group: MeshGroup, model: ShadingModel) -> Result<Self, ShadingError> {
        model.validate()?;
        let texture_pages = model
            .stages
            .iter()
            .map(|stage| stage.tex_unit as u32)
            .collect();
        Ok(Self {
            group,
            model,
            texture_pages,
        })
    }

    /// Triangles the group will submit.
    pub fn tri_count(&self) -> usize {
        self.group.tri_count()
    }

    pub fn is_empty(&self) -> bool {
        self.group.is_empty()
    }

    /// Uniform block for the batch to bind.
    pub fn uniform(&self, light_colors: &[[f32; 3]]) -> ShadingUniform {
        ShadingUniform::pack(&self.model, light_colors, self.model.material_fog.enabled)
    }
}

/// A frame's shaded groups, with the per-frame stats they contribute.
#[derive(Default)]
pub struct ShadedBatch {
    pub groups: Vec<ShadedGroup>,
    pub stats: FrameStats,
}

impl ShadedBatch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a group, skipping empty ones and counting what it costs.
    pub fn push(&mut self, shaded: ShadedGroup) {
        if shaded.is_empty() {
            self.stats.culled_groups += 1;
            return;
        }
        self.stats.draw_calls += 1;
        self.stats.triangles += shaded.tri_count() as u32;
        self.stats.vertices += shaded.group.positions.len() as u32;
        self.stats.indices += shaded.group.indices.len() as u32;
        self.groups.push(shaded);
    }

    pub fn len(&self) -> usize {
        self.groups.len()
    }

    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }
}

/// Constant colors a model reads, for uniform packing.
pub fn kcolor_view(model: &ShadingModel) -> &KColors {
    &model.kcolors
}
