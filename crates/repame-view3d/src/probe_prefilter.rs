//! GGX prefiltering for reflection probe cubes.
//!
//! Mip 0 is the captured radiance. Higher levels hold the aGGX convolution of
//! that cube at increasing roughness, so a fragment can select a mip from its
//! own roughness instead of reading a sharp reflection at every angle.

const PARAM_SLOTS: usize = 64;
const PARAM_STRIDE: u32 = 256;

const PREFILTER_WGSL: &str = r#"
struct Params {
    face: u32,
    roughness: f32,
    samples: u32,
    mode: u32,
};

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var src: texture_cube<f32>;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var<uniform> params: Params;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
    let x = f32((index << 1u) & 2u);
    let y = f32(index & 2u);
    var out: VsOut;
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0);
    return out;
}

fn face_direction(face: u32, uv: vec2<f32>) -> vec3<f32> {
    let u = uv.x;
    let v = uv.y;
    if (face == 0u) { return vec3<f32>(1.0, -v, -u); }
    if (face == 1u) { return vec3<f32>(-1.0, -v, u); }
    if (face == 2u) { return vec3<f32>(u, 1.0, v); }
    if (face == 3u) { return vec3<f32>(u, -1.0, -v); }
    if (face == 4u) { return vec3<f32>(u, -v, 1.0); }
    return vec3<f32>(-u, -v, -1.0);
}

fn radical_inverse(value: u32) -> f32 {
    var bits = value;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xaaaaaaaau) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xccccccccu) >> 2u);
    bits = ((bits & 0x0f0f0f0fu) << 4u) | ((bits & 0xf0f0f0f0u) >> 4u);
    bits = ((bits & 0x00ff00ffu) << 8u) | ((bits & 0xff00ff00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10;
}

fn basis(n: vec3<f32>) -> mat3x3<f32> {
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.999);
    let t = normalize(cross(up, n));
    return mat3x3<f32>(t, cross(n, t), n);
}

fn ggx_direction(n: vec3<f32>, roughness: f32, xi: vec2<f32>) -> vec3<f32> {
    let a = roughness * roughness;
    let phi = 6.2831853 * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let h_local = vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    let h = normalize(basis(n) * h_local);
    return normalize(reflect(-n, h));
}

fn cosine_direction(n: vec3<f32>, xi: vec2<f32>) -> vec3<f32> {
    let phi = 6.2831853 * xi.x;
    let r = sqrt(xi.y);
    let local = vec3<f32>(cos(phi) * r, sin(phi) * r, sqrt(max(1.0 - xi.y, 0.0)));
    return normalize(basis(n) * local);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(face_direction(params.face, in.uv));
    if (params.roughness <= 0.001 && params.mode == 0u) {
        return textureSampleLevel(src, smp, n, 0.0);
    }
    var total = vec3<f32>(0.0);
    var weight = 0.0;
    for (var i: u32 = 0u; i < params.samples; i = i + 1u) {
        let xi = vec2<f32>(radical_inverse(i), radical_inverse(i + 1u));
        let l = select(ggx_direction(n, params.roughness, xi), cosine_direction(n, xi), params.mode == 1u);
        let ndl = max(dot(n, l), 0.0);
        if (ndl > 0.0) {
            let w = select(ndl, 1.0, params.mode == 1u);
            total += textureSampleLevel(src, smp, l, 0.0).rgb * w;
            weight += w;
        }
    }
    return vec4<f32>(total / max(weight, 1e-4), 1.0);
}
"#;

pub fn mip_count(edge: u32) -> u32 {
    edge.max(1).ilog2() + 1
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PrefilterParams {
    face: u32,
    roughness: f32,
    samples: u32,
    mode: u32,
}

pub struct ProbePrefilter {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    params: wgpu::Buffer,
    samples: u32,
}

impl ProbePrefilter {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        cache: Option<&wgpu::PipelineCache>,
    ) -> Self {
        let module = shader_compat::module(
            device,
            "repame_view3d_probe_prefilter",
            PREFILTER_WGSL.to_string(),
        );
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("repame_view3d_probe_prefilter_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("repame_view3d_probe_prefilter_pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("repame_view3d_probe_prefilter"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("repame_view3d_probe_prefilter_smp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("repame_view3d_probe_prefilter_params"),
            size: (PARAM_SLOTS as u32 * PARAM_STRIDE) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            layout,
            sampler,
            params,
            samples: 64,
        }
    }

    pub fn encode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        cube: &wgpu::Texture,
        levels: u32,
    ) {
        if levels <= 1 {
            return;
        }
        let source = cube.create_view(&wgpu::TextureViewDescriptor {
            label: Some("repame_view3d_probe_prefilter_src"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            base_mip_level: 0,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("repame_view3d_probe_prefilter_bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&source),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.params,
                        offset: 0,
                        size: std::num::NonZeroU64::new(size_of::<PrefilterParams>() as u64),
                    }),
                },
            ],
        });
        for level in 1..levels {
            let irradiance = level == levels - 1;
            let roughness = if irradiance {
                0.0
            } else {
                level as f32 / (levels - 2).max(1) as f32
            };
            for face in 0..6u32 {
                let offset = ((level - 1) * 6 + face) * PARAM_STRIDE;
                queue.write_buffer(
                    &self.params,
                    offset as u64,
                    bytemuck::bytes_of(&PrefilterParams {
                        face,
                        roughness,
                        samples: self.samples,
                        mode: u32::from(irradiance),
                    }),
                );
                let target = cube.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("repame_view3d_probe_prefilter_dst"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    base_array_layer: face,
                    array_layer_count: Some(1),
                    ..Default::default()
                });
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("repame_view3d_probe_prefilter"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind, &[offset]);
                pass.draw(0..3, 0..1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mip_counts_cover_the_full_chain() {
        assert_eq!(mip_count(16), 5);
        assert_eq!(mip_count(128), 8);
        assert_eq!(mip_count(0), 1);
    }

    #[test]
    fn prefilter_shader_parses_and_validates() {
        let module = naga::front::wgsl::parse_str(PREFILTER_WGSL).expect("prefilter WGSL parses");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("prefilter WGSL validates");
    }
}

use crate::shader_compat;
