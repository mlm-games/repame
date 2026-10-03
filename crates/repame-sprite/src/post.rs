//! Post-process composite for the GPU viewport: scene renders into an
//! offscreen texture, then a fullscreen pass grades it back to the
//! screen. (Bevy `post_process` analog: NT's `screen_effects.wgsl`
//! sampled the view target; in-pass callbacks cannot, so the viewport
//! owns the intermediate.)
//!
//! The composite grades chromatic aberration (`offset = amount * 0.02`
//! horizontal RGB split, ported from `game-utils-bevy`'s
//! `screen_effects.wgsl`) and the [`Light2d`](crate::Light2d) point
//! lighting of [`FrameInput`](crate::FrameInput).

use std::collections::HashMap;

use repose_render_wgpu::{CallbackRenderPass, CallbackResources, ScreenDescriptor};

use super::batch::draw_batch_with_id;
use super::light2d::{MAX_LIGHTS, MAX_TOTAL_SHADOW_BINS};
use crate::{FrameInput, Light2d};

/// Fullscreen triangle + scene sampler. Uniforms: chroma head, the
/// lighting payload, four [`Light2d`] entries and the packed angular
/// shadow maps. Layout is locked to [`pack_composite_uniform`].
const COMPOSITE_WGSL: &str = r#"
// Uniform layout locked to `pack_composite_uniform` (const-asserted on
// the Rust side): 80-byte header, 4 lights x 96 bytes, then 2048 vec4
// bins holding two `[distance, hit]` bins each (a uniform array stride
// must be a multiple of 16, so bins cannot be `array<vec2<f32>>`).
struct Light {
    pos: vec4<f32>,
    color: vec4<f32>,
    ctrl: vec4<f32>,
    s0: vec4<f32>,
    s1: vec4<f32>,
    s2: vec4<f32>,
};
struct U {
    amount: f32,
    _p0: f32,
    _p1: f32,
    _p2: f32,
    ambient: vec4<f32>,
    fit: vec4<f32>,
    frame: vec4<f32>,
    view: vec4<f32>,
    lights: array<Light, 4>,
    bins: array<vec4<f32>, 2048>,
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
// Framebuffer pixel -> world through the same fit/roll/density the
// batch camera was built from.
fn pixel_to_world(p: vec2<f32>) -> vec2<f32> {
    let dp = p / max(u.view.x, 1e-6);
    let s = max(u.fit.x, 1e-6);
    let base = vec2<f32>(
        (dp.x - u.fit.y) / s + (u.frame.z - u.frame.x * 0.5),
        (dp.y - u.fit.z) / s + (u.frame.w - u.frame.y * 0.5)
    );
    let roll = u.fit.w;
    if (abs(roll) < 1e-7) {
        return base;
    }
    let c = cos(roll);
    let sn = sin(roll);
    let v = base - u.frame.zw;
    return u.frame.zw + vec2<f32>(c * v.x + sn * v.y, -sn * v.x + c * v.y);
}
// Analytic radial ramp between the three stops (rgb premultiplied by
// the stop alpha on upload).
fn ramp_at(l: Light, t: f32) -> vec3<f32> {
    if (t <= l.s0.x) {
        return l.s0.yzw;
    }
    if (t >= l.s2.x) {
        return l.s2.yzw;
    }
    if (t <= l.s1.x) {
        return mix(l.s0.yzw, l.s1.yzw, (t - l.s0.x) / max(l.s1.x - l.s0.x, 1e-6));
    }
    return mix(l.s1.yzw, l.s2.yzw, (t - l.s1.x) / max(l.s2.x - l.s1.x, 1e-6));
}
// The fragment tests its distance against the interpolated min-hit
// distance of its bin: before the first occluder it stays lit, past it
// it falls dark. Averaging that test over N taps of `ctrl.x` half-width
// (linear interpolation across bins) is the PCF penumbra, whose
// world-space width grows with distance from the light.
fn shadow_at(idx: u32, d: vec2<f32>) -> f32 {
    let l = u.lights[idx];
    let n = u32(l.ctrl.y);
    if (n == 0u) {
        return 1.0;
    }
    let nf = f32(n);
    let fd = length(d) / max(l.pos.z, 1e-6);
    let x0 = atan2(d.y, d.x) * 0.159154943091 * nf - 0.5;
    let taps = i32(floor(max(l.ctrl.x, 0.0)));
    let base = u32(l.ctrl.z);
    var sum = 0.0;
    for (var k: i32 = -taps; k <= taps; k = k + 1) {
        var x = x0 + f32(k);
        x = x - floor(x / nf) * nf;
        let i0 = u32(floor(x)) % n;
        let i1 = (i0 + 1u) % n;
        let v0 = u.bins[base + (i0 >> 1u)];
        let v1 = u.bins[base + (i1 >> 1u)];
        let a = select(v0.zw, v0.xy, (i0 & 1u) == 0u);
        let b = select(v1.zw, v1.xy, (i1 & 1u) == 0u);
        sum += select(0.0, 1.0, fd < mix(a, b, fract(x)).x);
    }
    return sum / f32(2 * taps + 1);
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
    let count = u32(u.view.y);
    if (count > 0u) {
        let world = pixel_to_world(in.pos.xy);
        var lit = u.ambient.xyz;
        for (var i: u32 = 0u; i < count; i = i + 1u) {
            let l = u.lights[i];
            let d = world - l.pos.xy;
            let t = length(d) / max(l.pos.z, 1e-6);
            var sh = 1.0;
            if (l.color.w > 0.5 && t < 1.0) {
                sh = shadow_at(i, d);
            }
            lit += l.color.xyz * l.pos.w * ramp_at(l, t) * sh;
        }
        color = vec4<f32>(color.rgb * clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0)), color.a);
    }
    return color;
}
"#;

/// Bytes before the `lights` array: chroma head + ambient + fit +
/// frame + view, one `vec4` each after the 16-byte chroma head.
const HEADER_BYTES: usize = 80;
/// Per-light uniform entry (`Light` in WGSL).
const LIGHT_BYTES: usize = 96;
/// Bins packed per `vec4` (two `[distance, hit]` bins).
const BINS_PER_VEC: usize = 2;
/// Fixed uniform size of the composite, pinned to `U` in the WGSL.
const UNIFORM_BYTES: usize =
    HEADER_BYTES + MAX_LIGHTS * LIGHT_BYTES + (MAX_TOTAL_SHADOW_BINS / BINS_PER_VEC) * 16;
/// WebGPU's guaranteed `maxUniformBufferBindingSize`.
const UNIFORM_BUDGET: usize = 64 * 1024;
const _: () = assert!(UNIFORM_BYTES <= UNIFORM_BUDGET);
const _: () = assert!(UNIFORM_BYTES == 33232);

/// True when the composite path is needed: chromatic aberration or any
/// point light. Both off keeps the zero-cost direct path (batch
/// straight into the main pass).
pub(crate) fn needs_post(input: &FrameInput) -> bool {
    input.chroma > 0.0 || !input.lights.is_empty()
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
    uniform_cap: u64,
    uniform_layout: wgpu::BindGroupLayout,
    uniform_bind: wgpu::BindGroup,
    tex_bind: wgpu::BindGroup,
}

struct PostResources {
    targets: HashMap<String, PostTargets>,
}

fn ensure_targets(
    device: &wgpu::Device,
    screen: &ScreenDescriptor,
    resources: &mut CallbackResources,
    id: &str,
    w: u32,
    h: u32,
) {
    let key = (screen.target_format, screen.sample_count, w, h);
    let fresh = resources
        .get::<PostResources>()
        .is_none_or(|r| r.targets.get(id).is_none_or(|t| t.key != key));
    if !fresh {
        return;
    }
    let samples = screen.sample_count.max(1);
    // Multisampled scene resolves into the sampled texture; at 1x the
    // scene texture is sampled directly.
    let scene = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("post_scene"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
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
        })
    } else {
        // Placeholder; never bound (scene_view is sampled instead).
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("post_resolve_unused"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
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
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
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
        label: Some("post_composite"),
        source: wgpu::ShaderSource::Wgsl(COMPOSITE_WGSL.into()),
    });
    let uniform = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("post_uniforms"),
        size: UNIFORM_BYTES as u64,
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
    let sampled_view = if samples > 1 {
        &resolve_view
    } else {
        &scene_view
    };
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
        label: Some("post_composite"),
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
        uniform_cap: UNIFORM_BYTES as u64,
        uniform_layout,
        uniform_bind,
        tex_bind,
    };
    match resources.get_mut::<PostResources>() {
        Some(all) => {
            all.targets.insert(id.to_string(), targets);
        }
        None => {
            let mut targets_map = HashMap::new();
            targets_map.insert(id.to_string(), targets);
            resources.insert(PostResources {
                targets: targets_map,
            });
        }
    }
}

/// Everything the offscreen composite grades this frame. Built from the
/// snapshot inside [`GpuViewport`](crate::GpuViewport) — never from the
/// one-frame-stale [`FrameGeom`](crate::FrameGeom) — so light math,
/// sprite projection and the letterbox clip all derive from the same
/// fit/roll/density.
pub(crate) struct CompositeDesc<'a> {
    pub batch_id: &'a str,
    pub amount: f32,
    pub background: Option<[f32; 4]>,
    /// [`FrameInput::ambient`], or a black floor when unset.
    pub ambient: [f32; 3],
    /// At most the first [`MAX_LIGHTS`] lights of the snapshot.
    pub lights: &'a [Light2d],
    /// Per-light angular maps, parallel to `lights`; empty = no map.
    pub shadow_maps: &'a [Vec<[f32; 2]>],
    /// `effective_fit` of the same dp canvas the batch camera used.
    pub fit: (f32, f32, f32),
    pub roll: f32,
    pub world: [f32; 2],
    pub center: [f32; 2],
    pub density: f32,
    /// Contain-fit world box (physical px) clipping the batch inside the
    /// offscreen texture.
    pub scissor: Option<(u32, u32, u32, u32)>,
}

/// Softness cap: 17 taps per light per pixel is already generous.
const MAX_SOFTNESS: f32 = 8.0;

fn put_f32(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// Serialize `desc` into the `U` layout of [`COMPOSITE_WGSL`]: header,
/// four light slots, then the bins the frame actually uses. Light `i`
/// reads its bins from vec4 `ctrl.z` onward, two bins per vec4.
fn pack_composite_uniform(desc: &CompositeDesc<'_>, out: &mut Vec<u8>) {
    out.clear();
    let n = desc.lights.len().min(MAX_LIGHTS);
    for v in [
        desc.amount,
        0.0,
        0.0,
        0.0,
        desc.ambient[0],
        desc.ambient[1],
        desc.ambient[2],
        0.0,
        desc.fit.0,
        desc.fit.1,
        desc.fit.2,
        desc.roll,
        desc.world[0],
        desc.world[1],
        desc.center[0],
        desc.center[1],
        desc.density,
        n as f32,
        0.0,
        0.0,
    ] {
        put_f32(out, v);
    }
    let cap_vecs = MAX_TOTAL_SHADOW_BINS / BINS_PER_VEC;
    let mut starts = Vec::with_capacity(n);
    let mut used: Vec<&[[f32; 2]]> = Vec::with_capacity(n);
    let mut first = 0usize;
    for i in 0..n {
        let map = desc
            .shadow_maps
            .get(i)
            .map_or(&[] as &[[f32; 2]], |m| m.as_slice());
        let vecs = map.len().div_ceil(BINS_PER_VEC);
        starts.push(first);
        if first + vecs <= cap_vecs {
            first += vecs;
            used.push(map);
        } else {
            log::warn!(
                "post: dropping the shadow map of light {i} ({} bins overflow the {cap_vecs}-vec4 uniform)",
                map.len()
            );
            used.push(&[]);
        }
    }
    for i in 0..MAX_LIGHTS {
        if i >= n {
            out.resize(out.len() + LIGHT_BYTES, 0);
            continue;
        }
        let l = &desc.lights[i];
        let bins = used[i].len();
        for v in [
            l.position.x,
            l.position.y,
            l.radius,
            l.energy,
            l.color[0],
            l.color[1],
            l.color[2],
            if bins > 0 { 1.0 } else { 0.0 },
            l.shadow_softness.clamp(0.0, MAX_SOFTNESS),
            bins as f32,
            starts[i] as f32,
            0.0,
        ] {
            put_f32(out, v);
        }
        for s in &l.falloff {
            for v in [
                s.at,
                s.color[0] * s.color[3],
                s.color[1] * s.color[3],
                s.color[2] * s.color[3],
            ] {
                put_f32(out, v);
            }
        }
    }
    for map in &used {
        for pair in map.chunks(BINS_PER_VEC) {
            put_f32(out, pair[0][0]);
            put_f32(out, pair[0][1]);
            put_f32(out, pair.get(1).map_or(0.0, |p| p[0]));
            put_f32(out, pair.get(1).map_or(0.0, |p| p[1]));
        }
    }
}

/// Render the prepared batch into the offscreen scene texture and stage
/// the composite uniforms. Call from `prepare` (owns the encoder); the
/// matching [`paint_composite`] runs in `paint`. `background` sets the
/// offscreen clear color so the composite preserves the snapshot clear
/// (`None` clears to transparent black, as before).
///
/// The batch is clipped to `desc.scissor` here: the fitted-box
/// letterbox clip only exists on the direct path's `paint`, and without
/// it sprites would bleed over the letterbox bars whenever the
/// composite is on.
#[allow(clippy::too_many_arguments)] // extends `WgpuCallback::prepare` by (w, h, desc)
pub(crate) fn prepare_composite(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    screen: &ScreenDescriptor,
    resources: &mut CallbackResources,
    w: u32,
    h: u32,
    desc: &CompositeDesc<'_>,
) {
    ensure_targets(device, screen, resources, desc.batch_id, w, h);
    let mut bytes = Vec::new();
    pack_composite_uniform(desc, &mut bytes);
    if let Some(all) = resources.get_mut::<PostResources>()
        && let Some(t) = all.targets.get_mut(desc.batch_id)
    {
        // Same growth safety net as `FullscreenPass::prepare_with`: the
        // payload only outgrows the buffer if the uniform layout grew.
        let need = (bytes.len() as u64).max(16);
        if need > t.uniform_cap {
            t.uniform = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("post_uniforms"),
                size: need,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            t.uniform_cap = need;
            t.uniform_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("post_uniform_bg"),
                layout: &t.uniform_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: t.uniform.as_entire_binding(),
                }],
            });
        }
        queue.write_buffer(&t.uniform, 0, &bytes);
    }
    let Some(all) = resources.get::<PostResources>() else {
        return;
    };
    let Some(t) = all.targets.get(desc.batch_id) else {
        return;
    };
    let (color_view, resolve_target) = if screen.sample_count.max(1) > 1 {
        (&t.scene_view, Some(&t.resolve_view))
    } else {
        (&t.scene_view, None)
    };
    let [cr, cg, cb, ca] = desc.background.unwrap_or([0.0, 0.0, 0.0, 0.0]);
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("post_scene"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: color_view,
            resolve_target,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: cr as f64,
                    g: cg as f64,
                    b: cb as f64,
                    a: ca as f64,
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
    if let Some(sc) = desc.scissor {
        pass.set_scissor_rect(sc.0, sc.1, sc.2, sc.3);
    }
    draw_batch_with_id(desc.batch_id, &mut pass, resources);
    if desc.scissor.is_some() {
        pass.set_scissor_rect(0, 0, w, h);
    }
}

/// Draw the graded fullscreen triangle into the main pass. The renderer
/// has already set the viewport to the callback rect, which matches the
/// offscreen texture 1:1 (both come from the painted frame geometry).
pub(crate) fn paint_composite(
    id: &str,
    rpass: &mut CallbackRenderPass<'_, '_>,
    resources: &CallbackResources,
) {
    let Some(all) = resources.get::<PostResources>() else {
        return;
    };
    let Some(t) = all.targets.get(id) else {
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
    use crate::batch::draw_batch_with_id_callback;
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
                ..Default::default()
            });
            // Rebuilt per frame through the public API (same calls
            // `GpuViewport::prepare` makes from its snapshot).
            for s in Probe::debug_sprites() {
                batch.push_sprite(&s);
            }
            batch.set_camera(screen_camera([64.0, 64.0]));
            batch.extend_uploads(self.uploads.clone());
            batch.prepare(device, queue, encoder, screen, resources);
            let input = self.input();
            if needs_post(&input) {
                prepare_composite(
                    device,
                    queue,
                    encoder,
                    screen,
                    resources,
                    64,
                    64,
                    &CompositeDesc {
                        batch_id: batch.id(),
                        amount: self.amount,
                        background: None,
                        ambient: [0.0; 3],
                        lights: &[],
                        shadow_maps: &[],
                        fit: (1.0, 0.0, 0.0),
                        roll: 0.0,
                        world: [64.0, 64.0],
                        center: [32.0, 32.0],
                        density: 1.0,
                        scissor: None,
                    },
                );
            }
            Vec::new()
        }

        fn paint(
            &self,
            _info: repose_core::PaintCallbackInfo,
            rpass: &mut CallbackRenderPass<'_, '_>,
            resources: &CallbackResources,
        ) {
            if needs_post(&self.input()) {
                paint_composite("sprite_batch.default", rpass, resources);
            } else {
                draw_batch_with_id_callback("sprite_batch.default", rpass, resources);
            }
        }
    }

    impl Probe {
        fn input(&self) -> FrameInput {
            FrameInput {
                chroma: self.amount,
                ..Default::default()
            }
        }

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
                ..Default::default()
            };
            [
                quad(16.0, [1.0, 0.0, 0.0, 1.0]),
                quad(48.0, [0.0, 0.0, 1.0, 1.0]),
            ]
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

    #[test]
    fn composite_gate_is_chroma_or_lights() {
        assert!(!needs_post(&FrameInput::default()));
        // Ambient alone stays on the zero-cost direct path.
        assert!(!needs_post(&FrameInput {
            ambient: Some([0.2, 0.2, 0.3]),
            ..Default::default()
        }));
        assert!(needs_post(&FrameInput {
            chroma: 0.1,
            ..Default::default()
        }));
        assert!(needs_post(&FrameInput {
            lights: vec![Light2d::default()],
            ..Default::default()
        }));
    }

    #[test]
    fn composite_uniform_fits_binding_budget() {
        use crate::light2d::resolve_shadow_bins;
        assert_eq!(HEADER_BYTES, 80);
        assert_eq!(LIGHT_BYTES, 96);
        assert_eq!(UNIFORM_BYTES, 33232);
        const { assert!(UNIFORM_BYTES <= UNIFORM_BUDGET) };
        assert!(COMPOSITE_WGSL.contains("array<vec4<f32>, 2048>"));
        // Worst case: four shadowed lights, each wanting the whole budget.
        let lights = [Light2d {
            shadows: true,
            ..Default::default()
        }; MAX_LIGHTS];
        let bins = resolve_shadow_bins(&lights);
        assert_eq!(bins, [1024; MAX_LIGHTS]);
        let maps: Vec<Vec<[f32; 2]>> = bins.iter().map(|b| vec![[0.5, 1.0]; *b as usize]).collect();
        let desc = CompositeDesc {
            batch_id: "sprite_batch.default",
            amount: 0.0,
            background: None,
            ambient: [0.1, 0.1, 0.1],
            lights: &lights,
            shadow_maps: &maps,
            fit: (1.0, 0.0, 0.0),
            roll: 0.0,
            world: [64.0, 64.0],
            center: [32.0, 32.0],
            density: 1.0,
            scissor: None,
        };
        let mut bytes = Vec::new();
        pack_composite_uniform(&desc, &mut bytes);
        assert_eq!(bytes.len(), UNIFORM_BYTES, "packed worst case");
        assert!(bytes.len() <= UNIFORM_BUDGET);
        // Absurd requests shrink to the budget instead of overflowing it,
        // and unshadowed lights cost no bins.
        let hungry = [Light2d {
            shadows: true,
            shadow_bins: u32::MAX,
            ..Default::default()
        }; MAX_LIGHTS];
        assert_eq!(resolve_shadow_bins(&hungry), [1024; MAX_LIGHTS]);
        assert_eq!(
            resolve_shadow_bins(&[Light2d::default(); MAX_LIGHTS]),
            [0; MAX_LIGHTS]
        );
    }
}
