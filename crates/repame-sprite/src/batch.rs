//! GPU sprite batch: instanced textured quads as a [`WgpuCallback`].
//!
//! First-class engine piece: the batch owns its atlas texture array and
//! instance buffer, feeds from [`AtlasUpload`]s (built from a
//! `repame-atlas` drain), and draws through a caller-supplied camera
//! matrix. Per-frame usage is snapshot-style.
//!
//! ```ignore
//! let mut batch = SpriteBatch::new(BatchDesc::default());
//! batch.set_camera(screen_camera([1920.0, 1080.0]));
//! for s in &input.sprites {
//!     batch.push_sprite(s);
//! }
//! Embedded(Modifier::new().fill_max_size(), Callback::new(batch))
//! ```
//!
//! NOTE: World space is y-down (canvas/snapshot convention). Instance rows map
//! quad corners to world coords; the camera uniform maps world to clip.
//! Draw order is input order (no depth test), exactly like the canvas
//! path. One batch per app for now: [`CallbackResources`] is keyed by
//! type, so two live batches would share pipelines (multi-viewport
//! support adds explicit ids later).

use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use repose_render_wgpu::{CallbackResources, ScreenDescriptor, WgpuCallback};
use wgpu::util::DeviceExt;

use super::SpriteInstance;

/// Texture sampling for atlas layers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TextureFilter {
    /// Pixel-art crisp (nt default).
    #[default]
    Nearest,
    Linear,
}

/// Batch construction parameters. Pages map 1:1 onto array layers.
#[derive(Clone, Copy, Debug)]
pub struct BatchDesc {
    /// Square atlas layer edge in pixels.
    pub layer_size: u32,
    /// Array layer count (= max atlas pages).
    pub layers: u32,
    pub filter: TextureFilter,
}

impl Default for BatchDesc {
    fn default() -> Self {
        Self {
            layer_size: 2048,
            layers: 4,
            filter: TextureFilter::Nearest,
        }
    }
}

/// One pending atlas upload: blit `rgba` (tight `w`*`h`*4 bytes) into
/// `page` at (`x`, `y`). Built from a `repame-atlas` drain by the game.
#[derive(Clone, Debug)]
pub struct AtlasUpload {
    pub page: u32,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    pub rgba: Vec<u8>,
}

impl AtlasUpload {
    /// Bridge an atlas placement to a GPU upload. `rgba` must hold the
    /// sprite's pixels row-major, `w`*`h`*4 bytes; mismatches are dropped
    /// with a warning at `prepare` time, never a panic.
    pub fn from_write(write: &repame_atlas::AtlasWrite, rgba: Vec<u8>) -> Self {
        Self {
            page: write.page,
            x: write.x,
            y: write.y,
            w: write.w,
            h: write.h,
            rgba,
        }
    }
}

/// Pure transform rows mapping quad corners (-0.5..0.5) to world coords,
/// honoring size, rotation, anchor, and mirror. Matches the canvas path
/// pixel-for-pixel at the default anchor: corner (-0.5,-0.5) lands on
/// `center - size / 2`.
pub fn instance_rows(
    center: [f32; 2],
    size: [f32; 2],
    rotation: f32,
    anchor: [f32; 2],
    flip_x: bool,
    flip_y: bool,
) -> ([f32; 4], [f32; 4]) {
    let (c, s) = (rotation.cos(), rotation.sin());
    let sx = if flip_x { -size[0] } else { size[0] };
    let sy = if flip_y { -size[1] } else { size[1] };
    // Normalized offset of the anchor from the quad middle: the anchor
    // point itself must land on `center`.
    let (ax, ay) = (0.5 - anchor[0], 0.5 - anchor[1]);
    (
        [c * sx, -s * sy, 0.0, center[0] + c * sx * ax - s * sy * ay],
        [s * sx, c * sy, 0.0, center[1] + s * sx * ax + c * sy * ay],
    )
}

/// UV rect for one cell of a sprite-sheet grid: `hframes` columns by
/// `vframes` rows, `frame` counted row-major from the top-left.
///
/// Returns `(uv_min, uv_max)` normalized to `0..1`, ready for
/// [`SpriteInstance`](super::SpriteInstance) `uv_min`/`uv_max`. The grid
/// covers the whole texture: cell `(col, row)` spans
/// `col/hframes..(col+1)/hframes` by `row/vframes..(row+1)/vframes`.
///
/// - Out-of-range `frame` values clamp to the last cell instead of
///   wrapping; advance the frame index with your animation clock.
/// - Degenerate grids (`hframes` or `vframes` of `0`) yield the full
///   texture, matching a single-frame sprite.
///
/// ```rust
/// use repame_sprite::{frame_uv, sprite_aabb};
///
/// // 4x2 sheet: frame 5 is column 1, row 1.
/// let (mn, mx) = frame_uv(4, 2, 5);
/// assert_eq!(mn, [0.25, 0.5]);
/// assert_eq!(mx, [0.5, 1.0]);
/// ```
pub fn frame_uv(hframes: u32, vframes: u32, frame: u32) -> ([f32; 2], [f32; 2]) {
    let hf = hframes.max(1);
    let vf = vframes.max(1);
    let f = frame.min(hf * vf - 1);
    let col = f % hf;
    let row = f / hf;
    (
        [col as f32 / hf as f32, row as f32 / vf as f32],
        [(col + 1) as f32 / hf as f32, (row + 1) as f32 / vf as f32],
    )
}

/// World-space axis-aligned bounds of a sprite quad.
///
/// Returns `([min_x, min_y], [max_x, max_y])` through the same
/// [`instance_rows`] math as the GPU batch, so the box matches the drawn
/// quad: exact for unrotated sprites, the outer box for rotated ones.
/// `flip_x`/`flip_y` do not change the box (mirroring preserves extents).
///
/// Useful for click hit-testing and layout: a press at world point `p`
/// hits the sprite when `min <= p <= max`.
///
/// ```rust
/// use repame_sprite::{frame_uv, sprite_aabb};
///
/// let (mn, mx) = sprite_aabb([400.0, 300.0], [64.0, 80.0], 0.0, [0.5, 0.5], false, false);
/// assert_eq!(mn, [368.0, 260.0]);
/// assert_eq!(mx, [432.0, 340.0]);
/// ```
pub fn sprite_aabb(
    center: [f32; 2],
    size: [f32; 2],
    rotation: f32,
    anchor: [f32; 2],
    flip_x: bool,
    flip_y: bool,
) -> ([f32; 2], [f32; 2]) {
    let (row0, row1) = instance_rows(center, size, rotation, anchor, flip_x, flip_y);
    let mut min = [f32::INFINITY, f32::INFINITY];
    let mut max = [f32::NEG_INFINITY, f32::NEG_INFINITY];
    for corner in [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]] {
        let x = row0[0] * corner[0] + row0[1] * corner[1] + row0[3];
        let y = row1[0] * corner[0] + row1[1] * corner[1] + row1[3];
        min[0] = min[0].min(x);
        min[1] = min[1].min(y);
        max[0] = max[0].max(x);
        max[1] = max[1].max(y);
    }
    (min, max)
}

/// Y-down orthographic camera: world (0,0) is the top-left of the
/// viewport, matching canvas orientation by construction (same matrix
/// `Camera2d::view_proj` builds).
pub fn screen_camera(viewport_px: [f32; 2]) -> Mat4 {
    // Right-handed, 0..1 depth: matches wgpu NDC.
    glam::camera::rh::proj::directx::orthographic(
        0.0,
        viewport_px[0].max(1.0),
        viewport_px[1].max(1.0),
        0.0,
        -1000.0,
        1000.0,
    )
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BatchInstance {
    row0: [f32; 4],
    row1: [f32; 4],
    uv_min: [f32; 2],
    uv_max: [f32; 2],
    tint: [f32; 4],
    page: f32,
}

// Locked to `shaders/sprite.wgsl::Instance` (row0@0, row1@16,
// uv_min@32, uv_max@40, tint@48, page@64; stride 68). If this fires,
// update the WGSL offsets and the vertex buffer layout below together.
const _: () = assert!(size_of::<BatchInstance>() == 68);

/// Per-frame snapshot batch. `Send + Sync` so it can cross into the
/// compositor thread via [`repose_render_wgpu::Callback`].
pub struct SpriteBatch {
    desc: BatchDesc,
    camera: [[f32; 4]; 4],
    instances: Vec<BatchInstance>,
    uploads: Vec<AtlasUpload>,
}

impl SpriteBatch {
    pub fn new(desc: BatchDesc) -> Self {
        Self {
            desc,
            camera: Mat4::IDENTITY.to_cols_array_2d(),
            instances: Vec::new(),
            uploads: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.instances.clear();
        self.uploads.clear();
    }

    pub fn len(&self) -> usize {
        self.instances.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// View-projection matrix, y-down screen convention (see
    /// [`screen_camera`]).
    pub fn set_camera(&mut self, view_proj: Mat4) {
        self.camera = view_proj.to_cols_array_2d();
    }

    /// Queue atlas uploads, applied in the next `prepare`.
    pub fn upload(&mut self, upload: AtlasUpload) {
        self.uploads.push(upload);
    }

    /// Queue several uploads at once (per-frame atlas drains).
    pub fn extend_uploads(&mut self, uploads: impl IntoIterator<Item = AtlasUpload>) {
        self.uploads.extend(uploads);
    }

    /// Push one textured quad.
    #[allow(clippy::too_many_arguments)]
    pub fn push(
        &mut self,
        center: [f32; 2],
        size: [f32; 2],
        rotation: f32,
        anchor: [f32; 2],
        flip_x: bool,
        flip_y: bool,
        uv_min: [f32; 2],
        uv_max: [f32; 2],
        tint: [f32; 4],
        page: u32,
    ) {
        let (row0, row1) = instance_rows(center, size, rotation, anchor, flip_x, flip_y);
        self.instances.push(BatchInstance {
            row0,
            row1,
            uv_min,
            uv_max,
            tint,
            page: page as f32,
        });
    }

    /// Bridge a snapshot sprite into the batch.
    pub fn push_sprite(&mut self, s: &SpriteInstance) {
        self.push(
            [s.center.x, s.center.y],
            [s.size.x, s.size.y],
            s.rotation,
            [s.anchor.x, s.anchor.y],
            s.flip_x,
            s.flip_y,
            [s.uv_min.x, s.uv_min.y],
            [s.uv_max.x, s.uv_max.y],
            s.color,
            s.page,
        );
    }
}

struct BatchResources {
    key: (wgpu::TextureFormat, u32, u32, u32, TextureFilter),
    pipeline: wgpu::RenderPipeline,
    corners: wgpu::Buffer,
    instances: wgpu::Buffer,
    instance_cap: usize,
    /// Instance count of the last prepared batch; `paint` draws this.
    /// Single live batch per app: `CallbackResources` is keyed by type,
    /// so two viewports (or a viewport + minimap) sharing one app would
    /// overwrite each other's count. Supported today: one GPU sprite
    /// consumer per app; multi-viewport needs per-id resources (like
    /// `FullscreenPass`'s id map).
    last_count: u32,
    camera: wgpu::Buffer,
    cam_bind: wgpu::BindGroup,
    tex_bind: wgpu::BindGroup,
    texture: wgpu::Texture,
}

const CORNERS: &[f32] = &[
    -0.5, -0.5, //
    0.5, -0.5, //
    0.5, 0.5, //
    -0.5, -0.5, //
    0.5, 0.5, //
    -0.5, 0.5, //
];

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
        label: Some("sprite_batch_sampler"),
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

impl SpriteBatch {
    fn ensure_resources(
        &self,
        device: &wgpu::Device,
        screen: &ScreenDescriptor,
        resources: &mut CallbackResources,
    ) {
        let key = (
            screen.target_format,
            screen.sample_count,
            self.desc.layer_size,
            self.desc.layers,
            self.desc.filter,
        );
        let rebuild = resources
            .get::<BatchResources>()
            .is_none_or(|r| r.key != key);
        if !rebuild {
            return;
        }
        if resources.get::<BatchResources>().is_some() {
            log::warn!(
                "sprite_batch: rebuilding pipeline/texture (format/sample/desc changed); atlas contents dropped, re-upload required"
            );
        }
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sprite_batch"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sprite.wgsl").into()),
        });
        let corners = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sprite_batch_corners"),
            contents: bytemuck::cast_slice(CORNERS),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let camera = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sprite_batch_camera"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sprite_batch_instances"),
            size: (self.instances.len().max(1) * size_of::<BatchInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sprite_batch_atlas"),
            size: wgpu::Extent3d {
                width: self.desc.layer_size,
                height: self.desc.layer_size,
                depth_or_array_layers: self.desc.layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let sampler = device.create_sampler(&sampler_desc(self.desc.filter));
        let cam_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sprite_batch_cam_bgl"),
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
        let tex_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sprite_batch_tex_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
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
        let cam_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sprite_batch_cam_bg"),
            layout: &cam_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera.as_entire_binding(),
            }],
        });
        let tex_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sprite_batch_tex_bg"),
            layout: &tex_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sprite_batch_pl"),
            bind_group_layouts: &[Some(&cam_layout), Some(&tex_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sprite_batch_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    Some(wgpu::VertexBufferLayout {
                        array_stride: 8,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        }],
                    }),
                    Some(wgpu::VertexBufferLayout {
                        array_stride: size_of::<BatchInstance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 0,
                                shader_location: 2,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 16,
                                shader_location: 3,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 32,
                                shader_location: 4,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 40,
                                shader_location: 5,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 48,
                                shader_location: 6,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32,
                                offset: 64,
                                shader_location: 7,
                            },
                        ],
                    }),
                ],
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
            // Same etiquette as the showcase embedded view: the compositor
            // owns a Depth24PlusStencil8 buffer; we never write depth and
            // pass the stencil test the frame leaves behind.
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
        resources.insert(BatchResources {
            key,
            pipeline,
            corners,
            instances,
            instance_cap: self.instances.len().max(1),
            last_count: 0,
            camera,
            cam_bind,
            tex_bind,
            texture,
        });
    }

    /// Record this batch's instance count after uploading. Split out so
    /// sibling payloads (e.g. viewport views rebuilding the batch per
    /// frame) can share one prepared pipeline.
    fn finish_prepare(&self, resources: &mut CallbackResources) {
        if let Some(res) = resources.get_mut::<BatchResources>() {
            res.last_count = self.instances.len() as u32;
        }
    }
}

/// Draw the prepared batch. Shared by [`SpriteBatch`] and viewport
/// payloads so all GPU consumers issue identical draw calls.
pub(crate) fn draw_batch(rpass: &mut wgpu::RenderPass<'_>, resources: &CallbackResources) {
    let Some(res) = resources.get::<BatchResources>() else {
        return;
    };
    if res.last_count == 0 {
        return;
    }
    rpass.set_pipeline(&res.pipeline);
    rpass.set_bind_group(0, &res.cam_bind, &[]);
    rpass.set_bind_group(1, &res.tex_bind, &[]);
    rpass.set_vertex_buffer(0, res.corners.slice(..));
    rpass.set_vertex_buffer(1, res.instances.slice(..));
    rpass.draw(0..6, 0..res.last_count);
}

impl WgpuCallback for SpriteBatch {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _encoder: &mut wgpu::CommandEncoder,
        screen: &ScreenDescriptor,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        self.ensure_resources(device, screen, resources);
        let Some(res) = resources.get_mut::<BatchResources>() else {
            return Vec::new();
        };
        // Grow the instance buffer when the batch outgrows it.
        if self.instances.len() > res.instance_cap {
            res.instances = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sprite_batch_instances"),
                size: (self.instances.len().max(1) * size_of::<BatchInstance>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            res.instance_cap = self.instances.len().max(1);
        }
        queue.write_buffer(
            &res.camera,
            0,
            bytemuck::cast_slice(&[CameraUniform {
                view_proj: self.camera,
            }]),
        );
        if !self.instances.is_empty() {
            queue.write_buffer(&res.instances, 0, bytemuck::cast_slice(&self.instances));
        }
        // Apply pending atlas uploads straight into array layers.
        for up in &self.uploads {
            let expected = up.w as usize * up.h as usize * 4;
            if up.page >= self.desc.layers
                || up.x + up.w > self.desc.layer_size
                || up.y + up.h > self.desc.layer_size
                || up.rgba.len() != expected
            {
                log::warn!(
                    "sprite_batch: dropping out-of-range upload page={} {}x{}+{}+{} ({} bytes)",
                    up.page,
                    up.w,
                    up.h,
                    up.x,
                    up.y,
                    up.rgba.len()
                );
                continue;
            }
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &res.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: up.x,
                        y: up.y,
                        z: up.page,
                    },
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
        self.finish_prepare(resources);
        Vec::new()
    }

    fn paint(
        &self,
        _info: repose_core::PaintCallbackInfo,
        rpass: &mut wgpu::RenderPass<'static>,
        resources: &CallbackResources,
    ) {
        draw_batch(rpass, resources);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply(rows: ([f32; 4], [f32; 4]), corner: [f32; 2]) -> [f32; 2] {
        let (r0, r1) = rows;
        [
            r0[0] * corner[0] + r0[1] * corner[1] + r0[3],
            r1[0] * corner[0] + r1[1] * corner[1] + r1[3],
        ]
    }

    #[test]
    fn default_rows_match_canvas_top_left() {
        let rows = instance_rows([400.0, 300.0], [64.0, 80.0], 0.0, [0.5, 0.5], false, false);
        let tl = apply(rows, [-0.5, -0.5]);
        let br = apply(rows, [0.5, 0.5]);
        assert_eq!(tl, [368.0, 260.0]);
        assert_eq!(br, [432.0, 340.0]);
    }

    #[test]
    fn frame_uv_indexes_row_major() {
        // 4x2 sheet: frame 5 is column 1, row 1.
        let (mn, mx) = frame_uv(4, 2, 5);
        assert_eq!(mn, [0.25, 0.5]);
        assert_eq!(mx, [0.5, 1.0]);
        // First cell starts at the origin; clamping holds the last cell.
        assert_eq!(frame_uv(4, 2, 0).0, [0.0, 0.0]);
        assert_eq!(frame_uv(4, 2, 99), frame_uv(4, 2, 7));
        assert_eq!(frame_uv(0, 0, 3), ([0.0, 0.0], [1.0, 1.0]));
    }

    #[test]
    fn sprite_aabb_matches_unrotated_quad() {
        let (mn, mx) = sprite_aabb([400.0, 300.0], [64.0, 80.0], 0.0, [0.5, 0.5], false, false);
        assert_eq!(mn, [368.0, 260.0]);
        assert_eq!(mx, [432.0, 340.0]);
        // A quarter turn swaps the footprint: 64x80 becomes 80x64.
        let (mn, mx) = sprite_aabb(
            [0.0, 0.0],
            [64.0, 80.0],
            std::f32::consts::FRAC_PI_2,
            [0.5, 0.5],
            false,
            false,
        );
        assert!((mn[0] + 40.0).abs() < 1e-4 && (mn[1] + 32.0).abs() < 1e-4);
        assert!((mx[0] - 40.0).abs() < 1e-4 && (mx[1] - 32.0).abs() < 1e-4);
    }

    #[test]
    fn quarter_turn_rotates_footprint() {
        let rows = instance_rows(
            [0.0, 0.0],
            [64.0, 80.0],
            std::f32::consts::FRAC_PI_2,
            [0.5, 0.5],
            false,
            false,
        );
        // Local top-left (-32,-40) rotates to (40,-32) in y-down space.
        let p = apply(rows, [-0.5, -0.5]);
        assert!(
            (p[0] - 40.0).abs() < 1e-4 && (p[1] + 32.0).abs() < 1e-4,
            "got {p:?}"
        );
    }

    #[test]
    fn top_left_anchor_pins_corner() {
        let rows = instance_rows([100.0, 100.0], [64.0, 80.0], 0.0, [0.0, 0.0], false, false);
        let tl = apply(rows, [-0.5, -0.5]);
        assert_eq!(tl, [100.0, 100.0]);
    }

    #[test]
    fn flip_x_mirrors_around_anchor() {
        let plain = instance_rows([0.0, 0.0], [64.0, 8.0], 0.0, [0.5, 0.5], false, false);
        let flipped = instance_rows([0.0, 0.0], [64.0, 8.0], 0.0, [0.5, 0.5], true, false);
        let a = apply(plain, [-0.5, 0.0]);
        let b = apply(flipped, [-0.5, 0.0]);
        assert_eq!(a, [-32.0, 0.0]);
        assert_eq!(b, [32.0, 0.0]);
    }

    #[test]
    fn screen_camera_is_y_down() {
        let cam = screen_camera([800.0, 600.0]);
        let top_left = cam.project_point3(glam::Vec3::new(0.0, 0.0, 0.0));
        let bottom_right = cam.project_point3(glam::Vec3::new(800.0, 600.0, 0.0));
        let center = cam.project_point3(glam::Vec3::new(400.0, 300.0, 0.0));
        // NDC y-up: screen top is +1, screen bottom is -1.
        assert!((top_left.x + 1.0).abs() < 1e-5 && (top_left.y - 1.0).abs() < 1e-5);
        assert!((bottom_right.x - 1.0).abs() < 1e-5 && (bottom_right.y + 1.0).abs() < 1e-5);
        assert!(center.x.abs() < 1e-5 && center.y.abs() < 1e-5);
    }

    #[test]
    fn push_sprite_bridges_snapshot_fields() {
        let mut batch = SpriteBatch::new(BatchDesc::default());
        batch.push_sprite(&SpriteInstance {
            center: glam::Vec2::new(10.0, 20.0),
            size: glam::Vec2::new(4.0, 6.0),
            uv_min: glam::Vec2::new(0.25, 0.5),
            uv_max: glam::Vec2::new(0.5, 0.75),
            color: [1.0, 0.5, 0.25, 0.8],
            page: 3,
            ..Default::default()
        });
        assert_eq!(batch.len(), 1);
        let inst = batch.instances[0];
        assert_eq!(inst.uv_min, [0.25, 0.5]);
        assert_eq!(inst.uv_max, [0.5, 0.75]);
        assert_eq!(inst.tint, [1.0, 0.5, 0.25, 0.8]);
        assert_eq!(inst.page, 3.0);
        // Default anchor centers: top-left at (8, 17).
        let p = apply((inst.row0, inst.row1), [-0.5, -0.5]);
        assert_eq!(p, [8.0, 17.0]);
    }

    /// End-to-end GPU proof: upload atlas pages, draw textured quads
    /// offscreen, and assert exact pixels (positions, colors, page,
    /// mirror). sRGB primaries are transfer-function fixed points, so
    /// asserts are exact. Skips gracefully where no GPU exists.
    #[test]
    fn offscreen_batch_draws_oriented_textured_quads() {
        use repose_core::{Color, Rect, Scene, SceneNode};
        use repose_render_wgpu::{Callback, offscreen::OffscreenRenderer};

        let mut renderer = match OffscreenRenderer::new_blocking(256, 256, 1) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP offscreen batch test (no GPU): {e}");
                return;
            }
        };
        // Page 0 (2x2, row-major top first): R G / B W.
        let page0 = vec![
            255, 0, 0, 255, //
            0, 255, 0, 255, //
            0, 0, 255, 255, //
            255, 255, 255, 255,
        ];
        // Page 1: solid magenta.
        let page1 = [255, 0, 255, 255].repeat(4);
        let mut batch = SpriteBatch::new(BatchDesc {
            layer_size: 2,
            layers: 2,
            filter: TextureFilter::Nearest,
        });
        batch.upload(AtlasUpload {
            page: 0,
            x: 0,
            y: 0,
            w: 2,
            h: 2,
            rgba: page0,
        });
        batch.upload(AtlasUpload {
            page: 1,
            x: 0,
            y: 0,
            w: 2,
            h: 2,
            rgba: page1,
        });
        batch.set_camera(screen_camera([256.0, 256.0]));
        // Big quad over the middle sampling page 0 fully.
        batch.push(
            [128.0, 128.0],
            [128.0, 128.0],
            0.0,
            [0.5, 0.5],
            false,
            false,
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            0,
        );
        // Small quad bottom-right sampling page 1.
        batch.push(
            [224.0, 224.0],
            [32.0, 32.0],
            0.0,
            [0.5, 0.5],
            false,
            false,
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            1,
        );
        let scene = Scene {
            clear_color: Color::from_rgba(0, 0, 0, 255),
            nodes: vec![SceneNode::Callback {
                rect: Rect {
                    x: 0.0,
                    y: 0.0,
                    w: 256.0,
                    h: 256.0,
                },
                payload: Callback::new(batch),
            }],
        };
        let px = renderer
            .render_rgba(&scene, Some([0.0, 0.0, 0.0, 1.0]))
            .expect("offscreen render");
        let at = |x: u32, y: u32| -> [u8; 4] {
            let i = ((y * 256 + x) * 4) as usize;
            [px[i], px[i + 1], px[i + 2], px[i + 3]]
        };
        // Orientation: uv (0,0) is the PNG top-left.
        assert_eq!(at(64, 64), [255, 0, 0, 255], "top-left red");
        assert_eq!(at(191, 64), [0, 255, 0, 255], "top-right green");
        assert_eq!(at(64, 191), [0, 0, 255, 255], "bottom-left blue");
        assert_eq!(at(191, 191), [255, 255, 255, 255], "bottom-right white");
        assert_eq!(at(224, 224), [255, 0, 255, 255], "page 1 magenta");
        assert_eq!(at(8, 8), [0, 0, 0, 255], "background clear");

        // Mirror check: same quad flipped samples the opposite column.
        let mut flipped = SpriteBatch::new(BatchDesc {
            layer_size: 2,
            layers: 2,
            filter: TextureFilter::Nearest,
        });
        flipped.set_camera(screen_camera([256.0, 256.0]));
        flipped.push(
            [128.0, 128.0],
            [128.0, 128.0],
            0.0,
            [0.5, 0.5],
            true,
            false,
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            0,
        );
        let scene = Scene {
            clear_color: Color::from_rgba(0, 0, 0, 255),
            nodes: vec![SceneNode::Callback {
                rect: Rect {
                    x: 0.0,
                    y: 0.0,
                    w: 256.0,
                    h: 256.0,
                },
                payload: Callback::new(flipped),
            }],
        };
        let px = renderer
            .render_rgba(&scene, Some([0.0, 0.0, 0.0, 1.0]))
            .expect("offscreen render");
        let at = |x: u32, y: u32| -> [u8; 4] {
            let i = ((y * 256 + x) * 4) as usize;
            [px[i], px[i + 1], px[i + 2], px[i + 3]]
        };
        assert_eq!(at(64, 64), [0, 255, 0, 255], "flipped top-left green");
        assert_eq!(at(191, 191), [0, 0, 255, 255], "flipped bottom-right blue");
    }
}
