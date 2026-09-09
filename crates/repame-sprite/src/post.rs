//! Post-process composite for the GPU viewport: scene renders into an
//! offscreen texture, then a fullscreen pass grades it back to the
//! screen. (Bevy `post_process` analog: NT's `screen_effects.wgsl`
//! sampled the view target; in-pass callbacks cannot, so the viewport
//! owns the intermediate.)
//!
//! Today the composite is chromatic aberration (`offset = amount * 0.02`
//! horizontal RGB split, ported from `game-utils-bevy`'s
//! `screen_effects.wgsl` — which itself ran on Bevy's generalized
//! post-process API, not a Bevy-built-in effect); the mechanism
//! (offscreen target + fullscreen triangle + uniform words) is shared
//! by future grades. Amount `0.0` skips the composite entirely: the batch draws
//! straight into the main pass, exactly the old path.

use repose_render_wgpu::{CallbackResources, ScreenDescriptor};

use super::batch::draw_batch;

/// Fullscreen triangle + scene sampler. Uniforms: `amount` followed by
/// three scalar pads (16 bytes total; a `vec3` pad would align the
/// struct to 32).
const CHROMA_WGSL: &str = r#"
struct U {
    amount: f32,
    _p0: f32,
    _p1: f32,
    _p2: f32,
};
@group(0) @binding(0) var<uniform> u: U;
@group(1) @binding(0) var scene_tex: texture_2d<f32>;
@group(1) @binding(1) var scene_smp: sampler;
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
    var color = textureSample(scene_tex, scene_smp, in.uv);
    let a = u.amount;
    if (a > 0.0) {
        let offset = a * 0.02;
        let r = textureSample(scene_tex, scene_smp, in.uv + vec2<f32>(offset, 0.0)).r;
        let b = textureSample(scene_tex, scene_smp, in.uv - vec2<f32>(offset, 0.0)).b;
        color = vec4<f32>(r, color.g, b, color.a);
    }
    return color;
}
"#;

/// True when the composite path is needed. `0.0` keeps the zero-cost
/// direct path (batch straight into the main pass).
pub(crate) fn use_composite(amount: f32) -> bool {
    amount > 0.0
}

struct PostTargets {
    key: (wgpu::TextureFormat, u32, u32, u32),
    // Owned textures outlive their views below (views borrow them).
    #[allow(dead_code)]
    scene: wgpu::Texture,
    scene_view: wgpu::TextureView,
    #[allow(dead_code)]
    resolve: wgpu::Texture,
    resolve_view: wgpu::TextureView,
    #[allow(dead_code)]
    depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    uniform_bind: wgpu::BindGroup,
    tex_bind: wgpu::BindGroup,
}

struct PostResources {
    targets: Option<PostTargets>,
}

fn ensure_targets(
    device: &wgpu::Device,
    screen: &ScreenDescriptor,
    resources: &mut CallbackResources,
    w: u32,
    h: u32,
) {
    let key = (screen.target_format, screen.sample_count, w, h);
    let fresh = resources
        .get::<PostResources>()
        .is_none_or(|r| r.targets.as_ref().is_none_or(|t| t.key != key));
    if !fresh {
        return;
    }
    let samples = screen.sample_count.max(1);
    // Multisampled scene resolves into the sampled texture; at 1x the
    // scene texture is sampled directly.
    let scene = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("post_scene"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: samples,
        dimension: wgpu::TextureDimension::D2,
        format: screen.target_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | if samples > 1 {
                wgpu::TextureUsages::empty()
            } else {
                wgpu::TextureUsages::TEXTURE_BINDING
            },
        view_formats: &[],
    });
    let resolve = if samples > 1 {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("post_resolve"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: screen.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    } else {
        // Placeholder; never bound (scene_view is sampled instead).
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("post_resolve_unused"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: screen.target_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    };
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("post_depth"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: samples,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth24PlusStencil8,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let scene_view = scene.create_view(&wgpu::TextureViewDescriptor::default());
    let resolve_view = resolve.create_view(&wgpu::TextureViewDescriptor::default());
    let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("post_chroma"),
        source: wgpu::ShaderSource::Wgsl(CHROMA_WGSL.into()),
    });
    let uniform = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("post_uniforms"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("post_uniform_bgl"),
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
    let tex_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("post_tex_bgl"),
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
        label: Some("post_sampler"),
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
    let uniform_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("post_uniform_bg"),
        layout: &uniform_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform.as_entire_binding(),
        }],
    });
    let sampled_view = if samples > 1 { &resolve_view } else { &scene_view };
    let tex_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("post_tex_bg"),
        layout: &tex_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(sampled_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });
    let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("post_pl"),
        bind_group_layouts: &[Some(&uniform_layout), Some(&tex_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("post_chroma"),
        layout: Some(&pipe_layout),
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
        // Matches the main UI pass (which always carries depth): depth
        // ops disabled, like `FullscreenPass`.
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: wgpu::StencilState::default(),
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
    let targets = PostTargets {
        key,
        scene,
        scene_view,
        resolve,
        resolve_view,
        depth,
        depth_view,
        pipeline,
        uniform,
        uniform_bind,
        tex_bind,
    };
    match resources.get_mut::<PostResources>() {
        Some(all) => {
            all.targets = Some(targets);
        }
        None => {
            resources.insert(PostResources {
                targets: Some(targets),
            });
        }
    }
}

/// Render the prepared batch into the offscreen scene texture and stage
/// the chroma uniforms. Call from `prepare` (owns the encoder); the
/// matching [`paint_composite`] runs in `paint`.
#[allow(clippy::too_many_arguments)] // extends `WgpuCallback::prepare` by (w, h, amount)
pub(crate) fn prepare_composite(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    screen: &ScreenDescriptor,
    resources: &mut CallbackResources,
    w: u32,
    h: u32,
    amount: f32,
) {
    ensure_targets(device, screen, resources, w, h);
    let Some(all) = resources.get::<PostResources>() else {
        return;
    };
    let Some(t) = all.targets.as_ref() else {
        return;
    };
    queue.write_buffer(&t.uniform, 0, bytemuck::cast_slice(&[amount, 0.0, 0.0, 0.0]));
    let (color_view, resolve_target) = if screen.sample_count.max(1) > 1 {
        (&t.scene_view, Some(&t.resolve_view))
    } else {
        (&t.scene_view, None)
    };
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("post_scene"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: color_view,
            resolve_target,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 0.0,
                }),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &t.depth_view,
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
    pass.set_viewport(0.0, 0.0, w as f32, h as f32, 0.0, 1.0);
    draw_batch(&mut pass, resources);
}

/// Draw the graded fullscreen triangle into the main pass. The renderer
/// has already set the viewport to the callback rect, which matches the
/// offscreen texture 1:1 (both come from the painted frame geometry).
pub(crate) fn paint_composite(
    rpass: &mut wgpu::RenderPass<'static>,
    resources: &CallbackResources,
) {
    let Some(all) = resources.get::<PostResources>() else {
        return;
    };
    let Some(t) = all.targets.as_ref() else {
        return;
    };
    rpass.set_pipeline(&t.pipeline);
    rpass.set_bind_group(0, &t.uniform_bind, &[]);
    rpass.set_bind_group(1, &t.tex_bind, &[]);
    rpass.draw(0..3, 0..1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AtlasUpload, BatchDesc, SpriteBatch, TextureFilter, screen_camera};
    use glam::Vec2;
    use repose_core::{Color, Rect, Scene, SceneNode};
    use repose_render_wgpu::{Callback, WgpuCallback, offscreen::OffscreenRenderer};

    use crate::SpriteInstance;

    /// Minimal viewport-shaped payload: batch in, composite out.
    struct Probe {
        uploads: Vec<AtlasUpload>,
        amount: f32,
    }

    impl WgpuCallback for Probe {
        fn prepare(
            &self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            encoder: &mut wgpu::CommandEncoder,
            screen: &ScreenDescriptor,
            resources: &mut CallbackResources,
        ) -> Vec<wgpu::CommandBuffer> {
            let mut batch = SpriteBatch::new(BatchDesc {
                layer_size: 8,
                layers: 1,
                filter: TextureFilter::Nearest,
            });
            // Rebuilt per frame through the public API (same calls
            // `GpuViewport::prepare` makes from its snapshot).
            for s in Probe::debug_sprites() {
                batch.push_sprite(&s);
            }
            batch.set_camera(screen_camera([64.0, 64.0]));
            batch.extend_uploads(self.uploads.clone());
            batch.prepare(device, queue, encoder, screen, resources);
            if use_composite(self.amount) {
                prepare_composite(device, queue, encoder, screen, resources, 64, 64, self.amount);
            }
            Vec::new()
        }

        fn paint(
            &self,
            _info: repose_core::PaintCallbackInfo,
            rpass: &mut wgpu::RenderPass<'static>,
            resources: &CallbackResources,
        ) {
            if use_composite(self.amount) {
                paint_composite(rpass, resources);
            } else {
                draw_batch(rpass, resources);
            }
        }
    }

    impl Probe {
        /// Left half red, right half blue, full-bleed 64x64.
        fn debug_sprites() -> [SpriteInstance; 2] {
            let quad = |cx: f32, color: [f32; 4]| SpriteInstance {
                center: Vec2::new(cx, 32.0),
                rotation: 0.0,
                size: Vec2::new(32.0, 64.0),
                anchor: Vec2::new(0.5, 0.5),
                flip_x: false,
                flip_y: false,
                uv_min: Vec2::ZERO,
                uv_max: Vec2::new(0.249, 0.249),
                color,
                page: 0,
            };
            [quad(16.0, [1.0, 0.0, 0.0, 1.0]), quad(48.0, [0.0, 0.0, 1.0, 1.0])]
        }
    }

    /// White 2x2 atlas upload so sprites sample solid texels.
    fn white_upload() -> AtlasUpload {
        AtlasUpload {
            page: 0,
            x: 0,
            y: 0,
            w: 2,
            h: 2,
            rgba: vec![255; 2 * 2 * 4],
        }
    }

    fn render(amount: f32) -> Option<Vec<u8>> {
        let mut renderer = match OffscreenRenderer::new_blocking(64, 64, 1) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP post test (no GPU): {e}");
                return None;
            }
        };
        let probe = Probe {
            uploads: vec![white_upload()],
            amount,
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
                payload: Callback::new(probe),
            }],
        };
        Some(
            renderer
                .render_rgba(&scene, Some([0.0, 0.0, 0.0, 1.0]))
                .expect("render"),
        )
    }

    fn px(buf: &[u8], x: u32, y: u32) -> [u8; 4] {
        let i = ((y * 64 + x) * 4) as usize;
        [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]
    }

    #[test]
    fn composite_passthrough_matches_direct() {
        let Some(a) = render(0.0) else { return };
        // Deep inside each half: exact primary colors.
        assert_eq!(px(&a, 24, 32), [255, 0, 0, 255]);
        assert_eq!(px(&a, 40, 32), [0, 0, 255, 255]);
    }

    #[test]
    fn chroma_splits_boundary_pixels() {
        let (Some(plain), Some(chroma)) = (render(0.0), render(0.7)) else {
            return;
        };
        // Deep inside the red half the shift samples red-on-red: identical.
        assert_eq!(px(&plain, 24, 32), px(&chroma, 24, 32));
        // At the boundary the red channel reaches across into blue:
        // must differ (shift = 0.7 * 0.02 * 64px ~= 0.9px).
        assert_ne!(px(&plain, 31, 32)[0], px(&chroma, 31, 32)[0]);
    }
}
