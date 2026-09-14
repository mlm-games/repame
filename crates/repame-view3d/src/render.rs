/// Viewport-owned offscreen scene + own depth buffer. The shared UI pass
/// this viewport paints into carries no depth ops (`depth_ops: None`), so
/// real depth writes are rejected there by validation. Like the 2D
/// `post::prepare_composite` path, the scene renders into a viewport-owned
/// target (with its own depth texture) during `prepare`, then a graded
/// fullscreen triangle composites back in `paint`.
///
/// Stencil stays `Always` + `LessEqual` (the UI contract) everywhere, so
/// clips keep working while depth stays viewport-local.
const BLIT_WGSL: &str = r#"
@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var scene_smp: sampler;
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
    return textureSample(scene_tex, scene_smp, in.uv);
}
"#;

// Flat-shaded 3D pass: world-space pos+color through a view-projection uniform.
const SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(@location(0) pos: vec3<f32>, @location(1) color: vec3<f32>) -> VsOut {
    var out: VsOut;
    out.pos = camera.view_proj * vec4<f32>(pos, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"#;

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use repose_render_wgpu::{CallbackResources, ScreenDescriptor};

use super::mesh::MeshGroup;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vert {
    pos: [f32; 3],
    color: [f32; 3],
}

/// One validated group awaiting [`SceneBatch::finish`].
struct Pending {
    positions: Vec<[f32; 3]>,
    colors: Vec<[f32; 3]>,
    indices: Vec<u32>,
    depth_test: bool,
}

/// One draw call's layer: index range + whether depth testing applies.
/// Groups flatten depth-tested-first, so tested geometry occludes and
/// untested overlays (ground decals, gizmos) always draw.
#[derive(Clone, Copy)]
struct DrawRange {
    index_start: u32,
    index_end: u32,
    depth_test: bool,
}

/// Per-frame snapshot batch. `Send + Sync` so it can cross into the
/// compositor thread via [`repose_render_wgpu::Callback`].
///
/// Each `id` owns its pipelines in `CallbackResources` (the
/// `repame-sprite` `SpriteBatch` pattern): viewports coexist. Rebuilding on
/// format/sample change drops nothing game-side — the snapshot rebuilds
/// the buffers from the next frame's groups anyway.
pub struct SceneBatch {
    id: String,
    camera: [[f32; 4]; 4],
    pending: Vec<Pending>,
    verts: Vec<Vert>,
    indices: Vec<u32>,
    ranges: Vec<DrawRange>,
}

impl SceneBatch {
    pub fn with_id(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            camera: Mat4::IDENTITY.to_cols_array_2d(),
            pending: Vec::new(),
            verts: Vec::new(),
            indices: Vec::new(),
            ranges: Vec::new(),
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn set_camera(&mut self, view_proj: Mat4) {
        self.camera = view_proj.to_cols_array_2d();
    }

    pub fn clear(&mut self) {
        self.pending.clear();
        self.verts.clear();
        self.indices.clear();
        self.ranges.clear();
    }

    pub fn len_tris(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Append one group. Malformed groups (index out of range, or
    /// position/color length mismatch) are dropped with a warning — never
    /// a panic, never partial draws.
    pub fn push_group(&mut self, group: &MeshGroup) {
        if group.is_empty() {
            return;
        }
        if group.positions.len() != group.colors.len() {
            log::warn!(
                "scene_batch[{}]: dropping group ({} positions vs {} colors)",
                self.id,
                group.positions.len(),
                group.colors.len()
            );
            return;
        }
        if group
            .indices
            .iter()
            .any(|i| (*i as usize) >= group.positions.len())
        {
            log::warn!(
                "scene_batch[{}]: dropping group (index out of range)",
                self.id
            );
            return;
        }
        self.pending.push(Pending {
            positions: group.positions.clone(),
            colors: group.colors.clone(),
            indices: group.indices.clone(),
            depth_test: group.depth_test,
        });
    }

    /// Flatten pending groups into the draw buffers, depth-tested first.
    /// Stable: submission order decides ties, so overlay order stays
    /// deterministic. Split from [`push_group`] so the viewport payload,
    /// which rebuilds the batch per frame, shares the path.
    pub fn finish(&mut self) {
        self.verts.clear();
        self.indices.clear();
        self.ranges.clear();
        self.pending.sort_by_key(|g| !g.depth_test);
        for g in self.pending.drain(..) {
            let base = self.verts.len() as u32;
            self.verts.extend(
                g.positions
                    .iter()
                    .zip(g.colors.iter())
                    .map(|(p, c)| Vert { pos: *p, color: *c }),
            );
            let start = self.indices.len() as u32;
            self.indices.extend(g.indices.iter().map(|i| i + base));
            self.ranges.push(DrawRange {
                index_start: start,
                index_end: self.indices.len() as u32,
                depth_test: g.depth_test,
            });
        }
    }

    pub(crate) fn ensure_resources(
        &self,
        device: &wgpu::Device,
        screen: &ScreenDescriptor,
        resources: &mut CallbackResources,
        w: u32,
        h: u32,
    ) {
        let w = w.max(1);
        let h = h.max(1);
        let key = (screen.target_format, screen.sample_count, w, h);
        let fresh = resources
            .get::<SceneResources>()
            .is_none_or(|all| !all.batches.contains_key(self.id.as_str()));
        let stale = !fresh
            && resources
                .get::<SceneResources>()
                .is_some_and(|all| all.batches[self.id.as_str()].key != key);
        if !fresh && !stale {
            return;
        }
        if stale {
            log::warn!(
                "scene_batch[{}]: rebuilding pipeline (format/sample changed)",
                self.id
            );
        }
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("repame_view3d"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let camera = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("repame_view3d_camera"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cam_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("repame_view3d_cam_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let cam_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("repame_view3d_cam_bg"),
            layout: &cam_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera.as_entire_binding(),
            }],
        });
        let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("repame_view3d_pl"),
            bind_group_layouts: &[Some(&cam_layout)],
            immediate_size: 0,
        });
        let buffers = [Some(wgpu::VertexBufferLayout {
            array_stride: size_of::<Vert>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 12,
                    shader_location: 1,
                },
            ],
        })];
        let mk = |label: &'static str, depth_test: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipe_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: screen.target_format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                // Real depth: tested ranges occlude correctly; untested
                // ranges still run inside the shared UI pass (Always) so
                // the stencil contract the UI relies on is untouched.
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24PlusStencil8,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(if depth_test {
                        wgpu::CompareFunction::Less
                    } else {
                        wgpu::CompareFunction::Always
                    }),
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
            })
        };
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("repame_view3d_blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
        });
        let (scene_tex, scene_view, depth_tex, depth_view, blit_pipeline, blit_bind) =
            make_target(device, screen, &blit_shader, w, h);
        let entry = SceneEntry {
            key,
            pipeline_depth: mk("repame_view3d_depth", true),
            pipeline_flat: mk("repame_view3d_flat", false),
            verts: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("repame_view3d_verts"),
                size: 64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            vert_cap: 0,
            indices: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("repame_view3d_indices"),
                size: 64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            index_cap: 0,
            camera,
            cam_bind,
            scene: scene_tex,
            scene_view,
            depth: depth_tex,
            depth_view,
            blit_pipeline,
            blit_bind,
            clear: [0.0, 0.0, 0.0, 1.0],
            camera_mat: self.camera,
            last_ranges: Vec::new(),
        };
        match resources.get_mut::<SceneResources>() {
            Some(all) => {
                all.batches.insert(self.id.clone(), entry);
            }
            None => {
                let mut all = SceneResources {
                    batches: HashMap::new(),
                };
                all.batches.insert(self.id.clone(), entry);
                resources.insert(all);
            }
        }
    }

    /// Upload camera + geometry after [`finish`]. No-op when the id has no
    /// prepared entry (call [`ensure_resources`](Self::ensure_resources)
    /// first — [`prepare_scene_with_id`] + the viewport do).
    pub(crate) fn upload(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resources: &mut CallbackResources,
    ) {
        let Some(all) = resources.get_mut::<SceneResources>() else {
            return;
        };
        let Some(res) = all.batches.get_mut(self.id.as_str()) else {
            return;
        };
        if self.verts.len() > res.vert_cap {
            let cap = self.verts.len().next_power_of_two().max(256);
            res.verts = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("repame_view3d_verts"),
                size: (cap * size_of::<Vert>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            res.vert_cap = cap;
        }
        if self.indices.len() > res.index_cap {
            let cap = self.indices.len().next_power_of_two().max(256);
            res.indices = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("repame_view3d_indices"),
                size: (cap * size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            res.index_cap = cap;
        }
        queue.write_buffer(
            &res.camera,
            0,
            bytemuck::cast_slice(&[CameraUniform {
                view_proj: self.camera,
            }]),
        );
        res.camera_mat = self.camera;
        if !self.verts.is_empty() {
            queue.write_buffer(&res.verts, 0, bytemuck::cast_slice(&self.verts));
        }
        if !self.indices.is_empty() {
            queue.write_buffer(&res.indices, 0, bytemuck::cast_slice(&self.indices));
        }
        res.last_ranges = self.ranges.clone();
    }

    /// Guard path shared by tests: run validation + flatten without a GPU.
    #[cfg(test)]
    pub(crate) fn entry_ranges(&self) -> Vec<(u32, u32, bool)> {
        self.ranges
            .iter()
            .map(|r| (r.index_start, r.index_end, r.depth_test))
            .collect()
    }
}

/// Viewport-owned offscreen scene + own depth buffer. MSAA is off here
/// (sample count 1): the shared UI pass resolves its own MSAA around the
/// callback, while the scene target stays a plain sampled texture.
#[allow(clippy::too_many_arguments)]
fn make_target(
    device: &wgpu::Device,
    screen: &ScreenDescriptor,
    blit_shader: &wgpu::ShaderModule,
    w: u32,
    h: u32,
) -> (
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::RenderPipeline,
    wgpu::BindGroup,
) {
    let scene = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("repame_view3d_scene"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: screen.target_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let scene_view = scene.create_view(&wgpu::TextureViewDescriptor::default());
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("repame_view3d_depth_tex"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth24PlusStencil8,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("repame_view3d_blit_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("repame_view3d_blit_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        lod_min_clamp: 0.0,
        lod_max_clamp: 1.0,
        compare: None,
        anisotropy_clamp: 1,
        border_color: None,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("repame_view3d_blit_bg"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&scene_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });
    let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("repame_view3d_blit_pl"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("repame_view3d_blit"),
        layout: Some(&pipe_layout),
        vertex: wgpu::VertexState {
            module: blit_shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: blit_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: screen.target_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        // Matches the main UI pass (which always carries depth): depth ops
        // disabled, so the blit never disturbs UI depth/stencil.
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
    (scene, scene_view, depth, depth_view, pipeline, bind)
}

struct SceneEntry {
    key: (wgpu::TextureFormat, u32, u32, u32),
    pipeline_depth: wgpu::RenderPipeline,
    pipeline_flat: wgpu::RenderPipeline,
    verts: wgpu::Buffer,
    vert_cap: usize,
    indices: wgpu::Buffer,
    index_cap: usize,
    camera: wgpu::Buffer,
    cam_bind: wgpu::BindGroup,
    // Owned offscreen target: the shared UI pass this paints into has no
    // depth ops, so the scene renders here (own depth) during `prepare`
    // and composites back in `paint`. Views borrow the textures below.
    #[allow(dead_code)]
    scene: wgpu::Texture,
    scene_view: wgpu::TextureView,
    #[allow(dead_code)]
    depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
    blit_pipeline: wgpu::RenderPipeline,
    blit_bind: wgpu::BindGroup,
    clear: [f32; 4],
    camera_mat: [[f32; 4]; 4],
    last_ranges: Vec<DrawRange>,
}

impl SceneEntry {
    fn clear_camera(&self) -> [CameraUniform; 1] {
        [CameraUniform {
            view_proj: self.camera_mat,
        }]
    }
}

struct SceneResources {
    batches: HashMap<String, SceneEntry>,
}

/// Render the prepared batch for one id into its own offscreen target
/// (with real depth), then composite the target into the main pass.
/// Shared by the viewport payload and the offscreen proof test.
///
/// `w`/`h` size the viewport-owned target; call from `prepare` (owns the
/// encoder) with the matching [`paint_scene_with_id`] in `paint`.
#[allow(clippy::too_many_arguments)] // extends `WgpuCallback::prepare` by (id, w, h, clear)
pub fn prepare_scene_with_id(
    id: &str,
    _device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    screen: &ScreenDescriptor,
    resources: &mut CallbackResources,
    w: u32,
    h: u32,
    clear: [f32; 4],
) {
    let Some(all) = resources.get_mut::<SceneResources>() else {
        return;
    };
    let Some(res) = all.batches.get_mut(id) else {
        return;
    };
    res.clear = clear;
    queue.write_buffer(&res.camera, 0, bytemuck::cast_slice(&res.clear_camera()));
    // Copy the small range list out so the mutable scene pass can coexist
    // with the borrow (a handful of entries; no per-tri cost).
    let ranges = res.last_ranges.clone();
    let has_content = !ranges.is_empty();
    if has_content {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("repame_view3d_scene"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &res.scene_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: clear[0] as f64,
                        g: clear[1] as f64,
                        b: clear[2] as f64,
                        a: clear[3] as f64,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &res.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0),
                    store: wgpu::StoreOp::Store,
                }),
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_viewport(0.0, 0.0, w.max(1) as f32, h.max(1) as f32, 0.0, 1.0);
        pass.set_bind_group(0, &res.cam_bind, &[]);
        pass.set_vertex_buffer(0, res.verts.slice(..));
        pass.set_index_buffer(res.indices.slice(..), wgpu::IndexFormat::Uint32);
        for r in &ranges {
            pass.set_pipeline(if r.depth_test {
                &res.pipeline_depth
            } else {
                &res.pipeline_flat
            });
            pass.draw_indexed(r.index_start..r.index_end, 0, 0..1);
        }
    }
    let _ = queue;
    let _ = screen;
}

/// Draw the offscreen scene target into the main pass. The renderer has
/// already set the viewport to the callback rect, which matches the
/// offscreen texture 1:1 (both come from the painted frame geometry).
pub fn paint_scene_with_id(
    id: &str,
    rpass: &mut wgpu::RenderPass<'_>,
    resources: &CallbackResources,
) {
    let Some(all) = resources.get::<SceneResources>() else {
        return;
    };
    let Some(res) = all.batches.get(id) else {
        return;
    };
    rpass.set_pipeline(&res.blit_pipeline);
    rpass.set_bind_group(0, &res.blit_bind, &[]);
    rpass.draw(0..3, 0..1);
}

#[cfg(test)]
mod tests {
    use super::super::mesh::MeshGroup;
    use super::*;
    fn solid_box() -> MeshGroup {
        let mut g = MeshGroup {
            depth_test: true,
            ..Default::default()
        };
        g.push_tri(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        );
        g
    }

    #[test]
    fn valid_group_flattens() {
        let mut batch = SceneBatch::with_id("test.valid");
        batch.set_camera(Mat4::IDENTITY);
        batch.push_group(&solid_box());
        batch.finish();
        assert_eq!(batch.len_tris(), 1);
        assert!(!batch.is_empty());
        assert_eq!(batch.entry_ranges(), vec![(0, 3, true)]);
    }

    #[test]
    fn malformed_groups_are_dropped() {
        let mut batch = SceneBatch::with_id("test.malformed");
        // Position/color length mismatch.
        batch.push_group(&MeshGroup {
            positions: vec![[0.0, 0.0, 0.0]],
            colors: vec![],
            indices: vec![0, 0, 0],
            depth_test: true,
        });
        // Index out of range.
        batch.push_group(&MeshGroup {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            colors: vec![[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            indices: vec![0, 1, 9],
            depth_test: true,
        });
        batch.push_group(&solid_box());
        batch.finish();
        assert_eq!(batch.len_tris(), 1, "only the valid group draws");
    }

    #[test]
    fn depth_tested_groups_draw_first() {
        let mut batch = SceneBatch::with_id("test.order");
        let mut flat = solid_box();
        flat.depth_test = false;
        batch.push_group(&flat);
        batch.push_group(&solid_box());
        batch.finish();
        assert_eq!(
            batch.entry_ranges(),
            vec![(0, 3, true), (3, 6, false)],
            "tested range first regardless of push order"
        );
    }

    /// End-to-end GPU proof: two screen-filling quads at different depths,
    /// far pushed first (painter-wrong order). Real depth testing must show
    /// the NEAR color; painter sorting would show the far color. Skips
    /// gracefully where no GPU exists.
    #[test]
    fn offscreen_depth_orders_by_distance_not_push_order() {
        use repose_core::{Color, Rect, Scene, SceneNode};
        use repose_render_wgpu::{Callback, WgpuCallback, offscreen::OffscreenRenderer};

        use super::super::camera::OrbitCamera;

        /// Viewport-shaped probe: batch in, offscreen scene + blit out
        /// (same calls `GpuViewport3d::prepare`/`paint` make).
        struct Direct {
            cam: OrbitCamera,
            far: MeshGroup,
            near: MeshGroup,
        }

        impl WgpuCallback for Direct {
            fn prepare(
                &self,
                device: &wgpu::Device,
                queue: &wgpu::Queue,
                encoder: &mut wgpu::CommandEncoder,
                screen: &repose_render_wgpu::ScreenDescriptor,
                resources: &mut repose_render_wgpu::CallbackResources,
            ) -> Vec<wgpu::CommandBuffer> {
                let mut batch = SceneBatch::with_id("test.depth");
                batch.set_camera(self.cam.view_proj(1.0));
                batch.push_group(&self.far);
                batch.push_group(&self.near);
                batch.finish();
                batch.ensure_resources(device, screen, resources, 64, 64);
                batch.upload(device, queue, resources);
                prepare_scene_with_id(
                    "test.depth",
                    device,
                    queue,
                    encoder,
                    screen,
                    resources,
                    64,
                    64,
                    [0.0, 0.0, 0.0, 1.0],
                );
                Vec::new()
            }

            fn paint(
                &self,
                _info: repose_core::PaintCallbackInfo,
                rpass: &mut wgpu::RenderPass<'static>,
                resources: &repose_render_wgpu::CallbackResources,
            ) {
                paint_scene_with_id("test.depth", rpass, resources);
            }
        }

        let mut renderer = match OffscreenRenderer::new_blocking(64, 64, 1) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP depth test (no GPU): {e}");
                return;
            }
        };
        // Tilted orbit: depth is view distance, no straight-down
        // `look_at` degeneracy (up-vector parallel to view dir).
        let cam = OrbitCamera {
            target: glam::Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.9,
            dist: 30.0,
            fov_y_deg: 30.0,
        };
        // Far quad (red) pushed FIRST, near quad (green) pushed second:
        // depth must still resolve to green at the center pixel.
        // Winding is CCW-from-above (+Y normal): backface culling keeps
        // upward faces for the above-looking-down camera.
        let mut far = MeshGroup {
            depth_test: true,
            ..Default::default()
        };
        far.push_quad(
            [-5.0, -5.0, 5.0],
            [5.0, -5.0, 5.0],
            [5.0, -5.0, -5.0],
            [-5.0, -5.0, -5.0],
            [1.0, 0.0, 0.0],
        );
        let mut near = MeshGroup {
            depth_test: true,
            ..Default::default()
        };
        near.push_quad(
            [-5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0],
            [5.0, 5.0, -5.0],
            [-5.0, 5.0, -5.0],
            [0.0, 1.0, 0.0],
        );
        let scene = Scene {
            clear_color: Color::from_rgba(0, 0, 0, 255),
            nodes: vec![SceneNode::Callback {
                rect: Rect {
                    x: 0.0,
                    y: 0.0,
                    w: 64.0,
                    h: 64.0,
                },
                payload: Callback::new(Direct { cam, far, near }),
            }],
        };
        let px = renderer
            .render_rgba(&scene, Some([0.0, 0.0, 0.0, 1.0]))
            .expect("offscreen render");
        let at = |x: u32, y: u32| -> [u8; 4] {
            let i = ((y * 64 + x) * 4) as usize;
            [px[i], px[i + 1], px[i + 2], px[i + 3]]
        };
        // Pure primaries are sRGB fixed points: exact asserts.
        assert_eq!(at(32, 32), [0, 255, 0, 255], "near quad wins by depth");
    }
}
