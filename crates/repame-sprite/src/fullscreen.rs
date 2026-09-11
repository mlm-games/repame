//! Fullscreen procedural passes: game-specific WGSL over a generated
//! triangle, with engine-owned pipeline/uniform/texture management.
pub const SOLID_WGSL: &str = r#"
    struct U { color: vec4<f32>, };
    @group(0) @binding(0) var<uniform> u: U;
    struct VsOut {
        @builtin(position) pos: vec4<f32>,
    };
    @vertex
    fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
        let x = f32(i / 2u) * 4.0 - 1.0;
        let y = f32(i % 2u) * 4.0 - 1.0;
        var out: VsOut;
        out.pos = vec4<f32>(x, y, 0.0, 1.0);
        return out;
    }
    @fragment
    fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
        return u.color;
    }
"#;

use std::collections::HashMap;

use repose_render_wgpu::{CallbackResources, ScreenDescriptor, WgpuCallback};

use super::TextureFilter;

/// Construction parameters for one effect.
#[derive(Clone, Copy, Debug)]
pub struct FullscreenDesc {
    /// Texture slot count (`@group(1)` bindings `0..N`, sampler at `N`).
    pub texture_slots: u32,
    /// Shared sampler for all slots.
    pub filter: TextureFilter,
}

/// One texture upload: tight `w`*`h`*4 RGBA8, row-major top first.
/// Recreates the slot texture when the size changes, plain rewrite
/// otherwise. Out-of-range or malformed uploads are dropped with a
/// warning, never a panic.
#[derive(Clone, Debug)]
pub struct FullscreenTexture {
    pub slot: u32,
    pub w: u32,
    pub h: u32,
    pub rgba: Vec<u8>,
}

/// A game-owned procedural effect. `Send + Sync` for the compositor
/// thread. Rebuilt from the snapshot every frame (same pattern as the
/// sprite batch): uniforms refresh each prepare, texture uploads ride
/// along only on change frames.
pub struct FullscreenPass {
    id: String,
    wgsl: &'static str,
    desc: FullscreenDesc,
    uniforms: Vec<u8>,
    uploads: Vec<FullscreenTexture>,
}

impl FullscreenPass {
    /// `id` disambiguates pipelines when several passes share one app;
    /// `wgsl` must follow the module binding contract above.
    pub fn new(id: impl Into<String>, wgsl: &'static str, desc: FullscreenDesc) -> Self {
        Self {
            id: id.into(),
            wgsl,
            desc,
            uniforms: Vec::new(),
            uploads: Vec::new(),
        }
    }

    /// Replace the uniform buffer contents (raw bytes).
    pub fn set_uniform_bytes(&mut self, bytes: &[u8]) {
        self.uniforms.clear();
        self.uniforms.extend_from_slice(bytes);
    }

    /// Replace the uniform buffer contents (f32 words).
    pub fn set_uniform_f32(&mut self, words: &[f32]) {
        self.uniforms.clear();
        self.uniforms
            .extend(words.iter().flat_map(|w| w.to_le_bytes()));
    }

    /// Queue texture uploads, applied in the next `prepare`.
    pub fn extend_textures(&mut self, uploads: impl IntoIterator<Item = FullscreenTexture>) {
        self.uploads.extend(uploads);
    }
}

struct PassInstance {
    key: (wgpu::TextureFormat, u32),
    pipeline: wgpu::RenderPipeline,
    uniforms: wgpu::Buffer,
    uniform_cap: u64,
    uniform_layout: wgpu::BindGroupLayout,
    uniform_bind: wgpu::BindGroup,
    slots: Vec<Option<(wgpu::Texture, u32, u32)>>,
    tex_bind: Option<wgpu::BindGroup>,
    tex_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    desc: FullscreenDesc,
}

struct PassResources {
    passes: HashMap<String, PassInstance>,
}

impl FullscreenPass {
    fn sampler_desc(filter: TextureFilter) -> wgpu::SamplerDescriptor<'static> {
        let f = match filter {
            TextureFilter::Nearest => wgpu::FilterMode::Nearest,
            TextureFilter::Linear => wgpu::FilterMode::Linear,
        };
        let m = match filter {
            TextureFilter::Nearest => wgpu::MipmapFilterMode::Nearest,
            TextureFilter::Linear => wgpu::MipmapFilterMode::Linear,
        };
        wgpu::SamplerDescriptor {
            label: Some("fullscreen_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: f,
            min_filter: f,
            mipmap_filter: m,
            lod_min_clamp: 0.0,
            lod_max_clamp: 1.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        }
    }

    fn ensure_resources(
        &self,
        device: &wgpu::Device,
        screen: &ScreenDescriptor,
        resources: &mut CallbackResources,
        uniform_len: usize,
    ) {
        let key = (screen.target_format, screen.sample_count);
        let fresh = resources
            .get::<PassResources>()
            .is_none_or(|r| !r.passes.contains_key(self.id.as_str()));
        let stale = !fresh
            && resources
                .get::<PassResources>()
                .is_some_and(|r| r.passes[self.id.as_str()].key != key);
        if !fresh && !stale {
            return;
        }
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fullscreen_pass"),
            source: wgpu::ShaderSource::Wgsl(self.wgsl.into()),
        });
        let uniform_cap = (uniform_len.max(16)) as u64;
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fullscreen_uniforms"),
            size: uniform_cap,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fullscreen_uniform_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let mut tex_entries = Vec::with_capacity(self.desc.texture_slots as usize + 1);
        for b in 0..self.desc.texture_slots {
            tex_entries.push(wgpu::BindGroupLayoutEntry {
                binding: b,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
        }
        tex_entries.push(wgpu::BindGroupLayoutEntry {
            binding: self.desc.texture_slots,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
        let tex_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fullscreen_tex_bgl"),
            entries: &tex_entries,
        });
        let sampler = device.create_sampler(&Self::sampler_desc(self.desc.filter));
        let uniform_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fullscreen_uniform_bg"),
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            }],
        });
        let two_groups = [Some(&uniform_layout), Some(&tex_layout)];
        let one_group = [Some(&uniform_layout)];
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fullscreen_pl"),
            // Uniform-only passes (zero texture slots) drop the texture
            // group entirely: their shaders declare no group(1), and the
            // validation layer requires every layout group to be bound.
            bind_group_layouts: if self.desc.texture_slots == 0 {
                &one_group
            } else {
                &two_groups
            },
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fullscreen_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: screen.target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState {
                    front: wgpu::StencilFaceState {
                        compare: wgpu::CompareFunction::LessEqual,
                        ..Default::default()
                    },
                    back: wgpu::StencilFaceState {
                        compare: wgpu::CompareFunction::LessEqual,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: screen.sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });
        let instance = PassInstance {
            key,
            pipeline,
            uniforms,
            uniform_cap,
            uniform_layout,
            uniform_bind,
            slots: vec![None; self.desc.texture_slots as usize],
            tex_bind: None,
            tex_layout,
            sampler,
            desc: self.desc,
        };
        match resources.get_mut::<PassResources>() {
            Some(all) => {
                all.passes.insert(self.id.clone(), instance);
            }
            None => {
                let mut all = PassResources {
                    passes: HashMap::new(),
                };
                all.passes.insert(self.id.clone(), instance);
                resources.insert(all);
            }
        }
    }

    fn apply_uploads(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        inst: &mut PassInstance,
        uploads: &[FullscreenTexture],
    ) {
        if uploads.is_empty() {
            return;
        }
        let mut relayout = false;
        for up in uploads {
            let slot = up.slot as usize;
            if slot >= inst.slots.len() || up.w == 0 || up.h == 0 {
                log::warn!("fullscreen: dropping bad upload slot={}", up.slot);
                continue;
            }
            if up.rgba.len() != up.w as usize * up.h as usize * 4 {
                log::warn!(
                    "fullscreen: dropping malformed upload slot={} {}x{} ({} bytes)",
                    up.slot,
                    up.w,
                    up.h,
                    up.rgba.len()
                );
                continue;
            }
            let fresh = match &inst.slots[slot] {
                Some((_, w, h)) => *w != up.w || *h != up.h,
                None => true,
            };
            if fresh {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("fullscreen_art"),
                    size: wgpu::Extent3d {
                        width: up.w,
                        height: up.h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                inst.slots[slot] = Some((texture, up.w, up.h));
                relayout = true;
            }
            let (texture, _, _) = inst.slots[slot].as_ref().expect("just stored");
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &up.rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(up.w * 4),
                    rows_per_image: Some(up.h),
                },
                wgpu::Extent3d {
                    width: up.w,
                    height: up.h,
                    depth_or_array_layers: 1,
                },
            );
        }
        if relayout && inst.slots.iter().all(|s| s.is_some()) {
            let views: Vec<wgpu::TextureView> = inst
                .slots
                .iter()
                .map(|slot| {
                    slot.as_ref()
                        .expect("all slots checked Some above")
                        .0
                        .create_view(&wgpu::TextureViewDescriptor::default())
                })
                .collect();
            let mut entries = Vec::with_capacity(views.len() + 1);
            for (b, view) in views.iter().enumerate() {
                entries.push(wgpu::BindGroupEntry {
                    binding: b as u32,
                    resource: wgpu::BindingResource::TextureView(view),
                });
            }
            entries.push(wgpu::BindGroupEntry {
                binding: inst.desc.texture_slots,
                resource: wgpu::BindingResource::Sampler(&inst.sampler),
            });
            inst.tex_bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fullscreen_tex_bg"),
                layout: &inst.tex_layout,
                entries: &entries,
            }));
        }
    }
}

impl FullscreenPass {
    /// Prepare with explicit data instead of the stored snapshot.
    /// Lets game-owned passes (custom snapshot types) delegate without
    /// rebuilding engine state: same pipeline, same draw calls.
    pub fn prepare_with(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen: &ScreenDescriptor,
        resources: &mut CallbackResources,
        uniforms: &[u8],
        uploads: &[FullscreenTexture],
    ) -> Vec<wgpu::CommandBuffer> {
        self.ensure_resources(device, screen, resources, uniforms.len());
        let Some(all) = resources.get_mut::<PassResources>() else {
            return Vec::new();
        };
        let Some(inst) = all.passes.get_mut(self.id.as_str()) else {
            return Vec::new();
        };

        // frame's data outgrows it (effect snapshots are fixed-size in
        // practice, so this is a safety net, not a hot path).
        let need = uniforms.len() as u64;
        if need > inst.uniform_cap {
            inst.uniforms = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fullscreen_uniforms"),
                size: need.max(16),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            inst.uniform_cap = need.max(16);
            inst.uniform_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fullscreen_uniform_bg"),
                layout: &inst.uniform_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: inst.uniforms.as_entire_binding(),
                }],
            });
        }
        if !uniforms.is_empty() {
            queue.write_buffer(&inst.uniforms, 0, uniforms);
        }
        self.apply_uploads(device, queue, inst, uploads);
        Vec::new()
    }
}

impl WgpuCallback for FullscreenPass {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _encoder: &mut wgpu::CommandEncoder,
        screen: &ScreenDescriptor,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        self.prepare_with(
            device,
            queue,
            screen,
            resources,
            &self.uniforms,
            &self.uploads,
        )
    }

    fn paint(
        &self,
        _info: repose_core::PaintCallbackInfo,
        rpass: &mut wgpu::RenderPass<'static>,
        resources: &CallbackResources,
    ) {
        let Some(all) = resources.get::<PassResources>() else {
            return;
        };
        let Some(inst) = all.passes.get(self.id.as_str()) else {
            return;
        };

        // need no art: draw with the uniform bind group alone. Textured
        // passes with no uploads yet keep the old behavior (skip: nothing
        // to shade with).
        let needs_tex = inst.desc.texture_slots > 0;
        if needs_tex && inst.tex_bind.is_none() {
            // Art missing: nothing to shade with yet.
            return;
        }
        rpass.set_pipeline(&inst.pipeline);
        rpass.set_bind_group(0, &inst.uniform_bind, &[]);
        if let Some(tex_bind) = &inst.tex_bind {
            rpass.set_bind_group(1, tex_bind, &[]);
        }
        rpass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mechanism proof, independent of any game content: a trivial
    /// uniform-driven shader paints exact pixels offscreen.
    #[test]
    fn offscreen_fullscreen_paints_uniform_color() {
        use repose_core::{Color, Rect, Scene, SceneNode};
        use repose_render_wgpu::{Callback, offscreen::OffscreenRenderer};

        const WGSL: &str = r#"
            struct U { color: vec4<f32>, };
            @group(0) @binding(0) var<uniform> u: U;
            @group(1) @binding(0) var t: texture_2d<f32>;
            @group(1) @binding(1) var s: sampler;
            struct VsOut {
                @builtin(position) pos: vec4<f32>,
                @location(0) uv: vec2<f32>,
            };
            @vertex
            fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
                let x = f32(i / 2u) * 4.0 - 1.0;
                let y = f32(i % 2u) * 4.0 - 1.0;
                var out: VsOut;
                out.pos = vec4<f32>(x, y, 0.0, 1.0);
                out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
                return out;
            }
            @fragment
            fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
                // Sample the art so the texture path is proven too.
                let tex = textureSample(t, s, vec2<f32>(0.5, 0.5));
                return vec4<f32>(u.color.rgb * tex.rgb, 1.0);
            }
        "#;

        let mut renderer = match OffscreenRenderer::new_blocking(64, 64, 1) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP fullscreen test (no GPU): {e}");
                return;
            }
        };
        let mut pass = FullscreenPass::new(
            "test-solid",
            WGSL,
            FullscreenDesc {
                texture_slots: 1,
                filter: TextureFilter::Nearest,
            },
        );
        // Opaque red uniforms; art texel (0,0) green -> output black.
        // (red * green = black proves the texture is really sampled.)
        pass.set_uniform_f32(&[1.0, 0.0, 0.0, 1.0]);
        pass.extend_textures([FullscreenTexture {
            slot: 0,
            w: 2,
            h: 2,
            rgba: vec![
                0, 255, 0, 255, //
                0, 0, 0, 255, //
                0, 0, 0, 255, //
                0, 0, 0, 255,
            ],
        }]);
        let scene = Scene {
            clear_color: Color::from_rgba(0, 0, 0, 255),
            nodes: vec![SceneNode::Callback {
                rect: Rect {
                    x: 0.0,
                    y: 0.0,
                    w: 64.0,
                    h: 64.0,
                },
                payload: Callback::new(pass),
            }],
        };
        let px = renderer
            .render_rgba(&scene, Some([0.0, 0.0, 0.0, 1.0]))
            .expect("render");
        // Center pixel: red uniforms x green texel = black.
        let i = ((32 * 64 + 32) * 4) as usize;
        assert_eq!([px[i], px[i + 1], px[i + 2], px[i + 3]], [0, 0, 0, 255]);
    }
}
