// Instanced textured quads for the repame-sprite batch.
//! Consumed by `batch.rs`: per-vertex corner + per-instance transform
//! rows, uv rect, tint, page. The camera uniform maps y-down world
//! space to clip, `page` selects the atlas array layer per instance.

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) tint: vec4<f32>,
    @location(2) page: f32,
};

struct Instance {
    // Column-major 3x2 world transform packed as two vec4s + extras.
    @location(2) row0: vec4<f32>,
    @location(3) row1: vec4<f32>,
    @location(4) uv_min: vec2<f32>,
    @location(5) uv_max: vec2<f32>,
    @location(6) tint: vec4<f32>,
    @location(7) page: f32,
};

struct Camera {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var atlas: texture_2d_array<f32>;
@group(1) @binding(1) var atlas_sampler: sampler;

@vertex
fn vs_main(@location(0) corner: vec2<f32>, inst: Instance) -> VertexOut {
    let world = vec3<f32>(
        inst.row0.x * corner.x + inst.row0.y * corner.y + inst.row0.w,
        inst.row1.x * corner.x + inst.row1.y * corner.y + inst.row1.w,
        0.0,
    );
    var out: VertexOut;
    out.pos = camera.view_proj * vec4<f32>(world, 1.0);
    out.uv = mix(inst.uv_min, inst.uv_max, corner + vec2<f32>(0.5, 0.5));
    out.tint = inst.tint;
    out.page = inst.page;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let tex = textureSample(atlas, atlas_sampler, in.uv, i32(in.page + 0.5));
    return tex * in.tint;
}
