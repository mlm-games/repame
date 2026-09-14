//! Flat-shaded + single-light 3D scene batch: snapshot in, depth-tested pixels out.
//!
//! [`SceneBatch`] owns the per-frame mesh snapshot (validate → flatten →
//! upload) plus its depth-tested/flat pipelines.
//!
//! The offscreen scene target + depth buffer + blit live in
//! [`DepthComposite`](repose_render_wgpu::DepthComposite): the shared UI
//! pass carries no depth ops, so depth-tested content renders into a
//! viewport-owned target during `prepare` and composites back in `paint`.

//! Flat-shaded 3D pass: world-space pos+color through a view-projection uniform.
//! Lit groups add per-vertex normals and sample the frame light from the
//! same uniform block (ambient + one directional, linear space).
const SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    light_dir: vec3<f32>,
    ambient: f32,
    light_color: vec3<f32>,
    diffuse: f32,
};

@group(0) @binding(0) var<uniform> camera: Camera;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) lit_flag: f32,
};

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) lit: f32,
) -> VsOut {
    var out: VsOut;
    out.pos = camera.view_proj * vec4<f32>(pos, 1.0);
    out.color = color;
    out.normal = normal;
    out.lit_flag = lit;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // `light_dir` points from the surface toward the light (Godot
    // DirectionalLight3D convention). Flat groups carry lit_flag 0 and
    // pass their baked color through untouched.
    let n = normalize(in.normal);
    let ndl = max(dot(n, camera.light_dir), 0.0);
    let lit = in.color * (camera.ambient + camera.light_color * (camera.diffuse * ndl));
    return vec4<f32>(mix(in.color, lit, in.lit_flag), 1.0);
}
"#;

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use repose_render_wgpu::{CallbackResources, DepthComposite, ScreenDescriptor};

use super::mesh::MeshGroup;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    light_dir: [f32; 3],
    ambient: f32,
    light_color: [f32; 3],
    diffuse: f32,
}

const _: () = assert!(size_of::<CameraUniform>() == 96);

/// One directional light + ambient for the frame, linear space.
///
/// The direction points from the surface toward the light (Godot
/// `DirectionalLight3D` convention). `shade_for_dir` producers keep
/// working: their baked colors ride the unlit path untouched.
#[derive(Clone, Copy, Debug)]
pub struct SceneLight {
    /// Unit vector from the surface toward the light (normalized on use).
    pub direction: [f32; 3],
    /// Light color (linear RGB).
    pub color: [f32; 3],
    /// Diffuse strength.
    pub diffuse: f32,
    /// Ambient floor (linear RGB added to every lit fragment).
    pub ambient: [f32; 3],
}

impl Default for SceneLight {
    fn default() -> Self {
        Self {
            direction: [0.3, 1.0, 0.4],
            color: [1.0, 1.0, 1.0],
            diffuse: 0.9,
            ambient: [0.35, 0.35, 0.38],
        }
    }
}

/// One vertex: position + albedo + normal + lit flag in a single
/// layout, so flat and lit groups share one buffer and one pipeline.
/// Flat vertices carry a dummy up-normal and `lit = 0.0` (their baked
/// color passes through untouched); lit vertices carry the true normal
/// and `lit = 1.0`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vert {
    pos: [f32; 3],
    color: [f32; 3],
    normal: [f32; 3],
    lit: f32,
}

/// One validated group awaiting [`SceneBatch::finish`].
struct Pending {
    positions: Vec<[f32; 3]>,
    colors: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
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
    camera: CameraUniform,
    pending: Vec<Pending>,
    verts: Vec<Vert>,
    indices: Vec<u32>,
    ranges: Vec<DrawRange>,
}

impl SceneBatch {
    pub fn with_id(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            camera: CameraUniform {
                view_proj: Mat4::IDENTITY.to_cols_array_2d(),
                light_dir: [0.0, 1.0, 0.0],
                ambient: 0.35,
                light_color: [1.0, 1.0, 1.0],
                diffuse: 0.9,
            },
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
        self.camera.view_proj = view_proj.to_cols_array_2d();
    }

    /// Frame light for lit groups. Flat groups ignore it entirely.
    pub fn set_light(&mut self, light: SceneLight) {
        let d = glam::Vec3::from(light.direction);
        let d = if d.length_squared() > 1e-8 {
            d.normalize()
        } else {
            glam::Vec3::Y
        };
        self.camera.light_dir = d.into();
        // Ambient is a single scalar in the uniform: use luminance so the
        // producer's tint keeps working without a per-channel path.
        self.camera.ambient =
            0.2126 * light.ambient[0] + 0.7152 * light.ambient[1] + 0.0722 * light.ambient[2];
        self.camera.light_color = light.color;
        self.camera.diffuse = light.diffuse.max(0.0);
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
    /// position/color length mismatch, or partial normals) are dropped
    /// with a warning — never a panic, never partial draws.
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
        if !group.normals.is_empty() && group.normals.len() != group.positions.len() {
            log::warn!(
                "scene_batch[{}]: dropping group ({} positions vs {} normals)",
                self.id,
                group.positions.len(),
                group.normals.len()
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
            normals: group.normals.clone(),
            indices: group.indices.clone(),
            depth_test: group.depth_test,
        });
    }

    /// Flatten pending groups into the draw buffers, depth-tested first.
    /// Stable: submission order decides ties, so overlay order stays
    /// deterministic. Split from [`push_group`] so the viewport payload,
    /// which rebuilds the batch per frame, shares the path.
    ///
    /// Flat vertices (no normals) carry a dummy up-normal and `lit = 0.0`
    /// so the shader passes their baked color through untouched.
    pub fn finish(&mut self) {
        self.verts.clear();
        self.indices.clear();
        self.ranges.clear();
        self.pending.sort_by_key(|g| !g.depth_test);
        for g in self.pending.drain(..) {
            let base = self.verts.len() as u32;
            let lit = !g.normals.is_empty();
            if lit {
                self.verts.extend(
                    g.positions
                        .iter()
                        .zip(g.colors.iter())
                        .zip(g.normals.iter())
                        .map(|((p, c), n)| Vert {
                            pos: *p,
                            color: *c,
                            normal: *n,
                            lit: 1.0,
                        }),
                );
            } else {
                self.verts
                    .extend(g.positions.iter().zip(g.colors.iter()).map(|(p, c)| Vert {
                        pos: *p,
                        color: *c,
                        normal: [0.0, 1.0, 0.0],
                        lit: 0.0,
                    }));
            }
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
    ) {
        let key = (screen.target_format, screen.sample_count);
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
            size: size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cam_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("repame_view3d_cam_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 24,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 36,
                    shader_location: 3,
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
        queue.write_buffer(&res.camera, 0, bytemuck::cast_slice(&[self.camera]));
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

    /// Test-only vertex readout after [`finish`]: (normal, lit flag).
    #[cfg(test)]
    pub(crate) fn test_vert_lighting(&self) -> Vec<([f32; 3], f32)> {
        self.verts.iter().map(|v| (v.normal, v.lit)).collect()
    }
}

struct SceneEntry {
    key: (wgpu::TextureFormat, u32),
    pipeline_depth: wgpu::RenderPipeline,
    pipeline_flat: wgpu::RenderPipeline,
    verts: wgpu::Buffer,
    vert_cap: usize,
    indices: wgpu::Buffer,
    index_cap: usize,
    camera: wgpu::Buffer,
    cam_bind: wgpu::BindGroup,
    camera_mat: CameraUniform,
    last_ranges: Vec<DrawRange>,
}

struct SceneResources {
    batches: HashMap<String, SceneEntry>,
}

/// Draw the prepared batch for one id: scene into the shared
/// [`DepthComposite`] offscreen target (real depth), composited back in
/// `paint`. Shared by the viewport payload and the offscreen proof test.
///
/// Call from `prepare` (owns the encoder) with the matching
/// [`paint_scene_with_id`] in `paint`. `w`/`h` size the viewport-owned
/// target; `clear` is the scene clear color (linear 0..1 RGBA).
#[allow(clippy::too_many_arguments)] // extends `WgpuCallback::prepare` by (id, w, h, clear)
pub fn prepare_scene_with_id(
    id: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    screen: &ScreenDescriptor,
    resources: &mut CallbackResources,
    w: u32,
    h: u32,
    clear: [f32; 4],
) {
    DepthComposite::get(resources).ensure(device, screen, id, w, h);
    // Snapshot the draw state into owned bind groups: pipelines and buffers
    // are shared resources, so clone the (cheap) handles and end the borrow
    // before beginning the mutable scene pass.
    struct Snapshot {
        cam_bind: wgpu::BindGroup,
        verts: wgpu::Buffer,
        indices: wgpu::Buffer,
        pipeline_depth: wgpu::RenderPipeline,
        pipeline_flat: wgpu::RenderPipeline,
        ranges: Vec<DrawRange>,
    }
    let snapshot: Option<Snapshot> = {
        let Some(all) = resources.get_mut::<SceneResources>() else {
            return;
        };
        let Some(res) = all.batches.get_mut(id) else {
            return;
        };
        queue.write_buffer(&res.camera, 0, bytemuck::cast_slice(&[res.camera_mat]));
        if res.last_ranges.is_empty() {
            return;
        }
        Some(Snapshot {
            cam_bind: res.cam_bind.clone(),
            verts: res.verts.clone(),
            indices: res.indices.clone(),
            pipeline_depth: res.pipeline_depth.clone(),
            pipeline_flat: res.pipeline_flat.clone(),
            ranges: res.last_ranges.clone(),
        })
    };
    let Some(snap) = snapshot else { return };
    let composite = DepthComposite::get(resources);
    let Some(mut pass) = composite.begin_scene(id, encoder, clear) else {
        return;
    };
    pass.set_bind_group(0, &snap.cam_bind, &[]);
    pass.set_vertex_buffer(0, snap.verts.slice(..));
    pass.set_index_buffer(snap.indices.slice(..), wgpu::IndexFormat::Uint32);
    for r in &snap.ranges {
        pass.set_pipeline(if r.depth_test {
            &snap.pipeline_depth
        } else {
            &snap.pipeline_flat
        });
        pass.draw_indexed(r.index_start..r.index_end, 0, 0..1);
    }
}

/// Composite the offscreen scene for `id` into the main pass. The renderer
/// has already set the viewport to the callback rect, which matches the
/// offscreen texture 1:1 (both come from the painted frame geometry).
/// No-op when `id` has no target (call [`prepare_scene_with_id`] first).
pub fn paint_scene_with_id(
    id: &str,
    rpass: &mut wgpu::RenderPass<'_>,
    resources: &CallbackResources,
) {
    let Some(composite) = resources.get::<DepthComposite>() else {
        return;
    };
    composite.blit(id, rpass);
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
        // Flat vertices pass through: dummy up-normal, lit 0.
        assert_eq!(batch.test_vert_lighting(), vec![([0.0, 1.0, 0.0], 0.0); 3]);
    }

    #[test]
    fn lit_group_flattens_with_normals() {
        let mut batch = SceneBatch::with_id("test.lit");
        batch.set_camera(Mat4::IDENTITY);
        let mut g = MeshGroup {
            depth_test: true,
            ..Default::default()
        };
        g.push_quad_lit(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
        );
        batch.push_group(&g);
        batch.finish();
        assert_eq!(batch.len_tris(), 2);
        let got = batch.test_vert_lighting();
        assert_eq!(got.len(), 6);
        assert!(got.iter().all(|(n, l)| *n == [0.0, 1.0, 0.0] && *l == 1.0));
    }

    #[test]
    fn set_light_normalizes_and_clamps() {
        let mut batch = SceneBatch::with_id("test.light");
        batch.set_light(SceneLight {
            direction: [0.0, 0.0, 0.0],
            diffuse: -2.0,
            ..SceneLight::default()
        });
        let d = glam::Vec3::from(batch.camera.light_dir);
        assert!((d.length() - 1.0).abs() < 1e-6, "degenerate dir falls back");
        assert_eq!(batch.camera.diffuse, 0.0, "negative diffuse clamps");
    }

    #[test]
    fn malformed_groups_are_dropped() {
        let mut batch = SceneBatch::with_id("test.malformed");
        // Position/color length mismatch.
        batch.push_group(&MeshGroup {
            positions: vec![[0.0, 0.0, 0.0]],
            colors: vec![],
            normals: vec![],
            indices: vec![0, 0, 0],
            depth_test: true,
        });
        // Index out of range.
        batch.push_group(&MeshGroup {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            colors: vec![[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            normals: vec![],
            indices: vec![0, 1, 9],
            depth_test: true,
        });
        // Partial normals.
        batch.push_group(&MeshGroup {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            colors: vec![[1.0, 0.0, 0.0]; 3],
            normals: vec![[0.0, 1.0, 0.0]],
            indices: vec![0, 1, 2],
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
                batch.ensure_resources(device, screen, resources);
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

    /// End-to-end GPU proof for the lit path: one up-facing white quad
    /// under a straight-down light reads back full white; the same quad
    /// facing away from the light reads back ambient only. Flat quads are
    /// unaffected by the light (baked color passes through). Skips
    /// gracefully where no GPU exists.
    #[test]
    fn offscreen_lit_shades_by_normal() {
        use repose_core::{Color, Rect, Scene, SceneNode};
        use repose_render_wgpu::{Callback, WgpuCallback, offscreen::OffscreenRenderer};

        use super::super::camera::OrbitCamera;

        struct Lit {
            cam: OrbitCamera,
            group: MeshGroup,
            light: SceneLight,
        }

        impl WgpuCallback for Lit {
            fn prepare(
                &self,
                device: &wgpu::Device,
                queue: &wgpu::Queue,
                encoder: &mut wgpu::CommandEncoder,
                screen: &repose_render_wgpu::ScreenDescriptor,
                resources: &mut repose_render_wgpu::CallbackResources,
            ) -> Vec<wgpu::CommandBuffer> {
                let mut batch = SceneBatch::with_id("test.lit");
                batch.set_camera(self.cam.view_proj(1.0));
                batch.set_light(self.light);
                batch.push_group(&self.group);
                batch.finish();
                batch.ensure_resources(device, screen, resources);
                batch.upload(device, queue, resources);
                prepare_scene_with_id(
                    "test.lit",
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
                paint_scene_with_id("test.lit", rpass, resources);
            }
        }

        fn render_case(group: MeshGroup, light: SceneLight) -> Option<[u8; 4]> {
            let mut renderer = match OffscreenRenderer::new_blocking(64, 64, 1) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP lit test (no GPU): {e}");
                    return None;
                }
            };
            let cam = OrbitCamera {
                target: glam::Vec3::ZERO,
                yaw: 0.0,
                pitch: 0.9,
                dist: 30.0,
                fov_y_deg: 30.0,
            };
            let scene = Scene {
                clear_color: Color::from_rgba(0, 0, 0, 255),
                nodes: vec![SceneNode::Callback {
                    rect: Rect {
                        x: 0.0,
                        y: 0.0,
                        w: 64.0,
                        h: 64.0,
                    },
                    payload: Callback::new(Lit { cam, group, light }),
                }],
            };
            let px = renderer
                .render_rgba(&scene, Some([0.0, 0.0, 0.0, 1.0]))
                .expect("offscreen render");
            let i = ((32 * 64 + 32) * 4) as usize;
            Some([px[i], px[i + 1], px[i + 2], px[i + 3]])
        }

        // Screen-filling up-facing quad (CCW-from-above, like the depth test).
        fn up_quad(lit: bool) -> MeshGroup {
            let mut g = MeshGroup {
                depth_test: true,
                ..Default::default()
            };
            if lit {
                g.push_quad_lit(
                    [-5.0, 5.0, 5.0],
                    [5.0, 5.0, 5.0],
                    [5.0, 5.0, -5.0],
                    [-5.0, 5.0, -5.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 1.0, 0.0],
                );
            } else {
                g.push_quad(
                    [-5.0, 5.0, 5.0],
                    [5.0, 5.0, 5.0],
                    [5.0, 5.0, -5.0],
                    [-5.0, 5.0, -5.0],
                    [0.5, 0.5, 0.5],
                );
            }
            g
        }

        let face_light = SceneLight {
            direction: [0.0, 1.0, 0.0],
            color: [1.0, 1.0, 1.0],
            diffuse: 1.0,
            ambient: [0.0, 0.0, 0.0],
        };
        let Some(full) = render_case(up_quad(true), face_light) else {
            return;
        };
        assert_eq!(full, [255, 255, 255, 255], "face-on light = full white");

        let back_light = SceneLight {
            direction: [0.0, -1.0, 0.0],
            ..face_light
        };
        let Some(dark) = render_case(up_quad(true), back_light) else {
            return;
        };
        assert_eq!(dark, [0, 0, 0, 255], "back light + no ambient = black");

        // Flat path ignores the light: baked gray passes through. The
        // offscreen target is sRGB, so linear 0.5 reads back as 188.
        let Some(flat) = render_case(up_quad(false), back_light) else {
            return;
        };
        assert_eq!(flat, [188, 188, 188, 255], "flat ignores light");
    }
}
