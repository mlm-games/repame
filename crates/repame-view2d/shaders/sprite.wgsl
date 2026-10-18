// Instanced textured quads for the repame-view2d batch.
//! Consumed by `batch.rs`: per-vertex corner + per-instance transform
//! rows, uv rect, tint, page, and the light layer the instance is lit by.
//! The camera uniform maps y-down world space to clip, `page` selects the
//! atlas array layer per instance, and group(2) carries the light map the
//! instance's layer reads (see `lightpass.rs`).

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) tint: vec4<f32>,
    @location(2) page: f32,
    @location(3) world: vec2<f32>,
    @interpolate(flat) @location(4) layer: u32,
};

struct Instance {
    // Column-major 3x2 world transform packed as two vec4s + extras.
    @location(2) row0: vec4<f32>,
    @location(3) row1: vec4<f32>,
    @location(4) uv_min: vec2<f32>,
    @location(5) uv_max: vec2<f32>,
    @location(6) tint: vec4<f32>,
    @location(7) page: f32,
    @location(8) z: f32,
    @location(9) flags: u32,
    // Half a texel in normalized UV.
    @location(10) texel: f32,
    // Layer bits this instance is lit by; its lowest set bit is the layer.
    @location(11) light_mask: u32,
};

struct Camera {
    view_proj: mat4x4<f32>,
};

struct Lightmap {
    // world * xform.xy + xform.zw = light map uv, (0,0) at its top-left.
    xform: vec4<f32>,
    // rgb: scene modulate, applied to the item colour before the light.
    modulate: vec4<f32>,
    // x: highest layer an instance may sample (the identity layer).
    layer: vec4<u32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var atlas: texture_2d_array<f32>;
@group(1) @binding(1) var atlas_sampler: sampler;
@group(2) @binding(0) var lightmap: texture_2d_array<f32>;
@group(2) @binding(1) var lightmap_sampler: sampler;
@group(2) @binding(2) var<uniform> lightmap_u: Lightmap;

@vertex
fn vs_main(@location(0) corner: vec2<f32>, inst: Instance) -> VertexOut {
    let world = vec3<f32>(
        inst.row0.x * corner.x + inst.row0.y * corner.y + inst.row0.w,
        inst.row1.x * corner.x + inst.row1.y * corner.y + inst.row1.w,
        0.0,
    );
    var out: VertexOut;
    out.pos = camera.view_proj * vec4<f32>(world, 1.0);
    // UV rects are cell BOUNDARIES, so the outer half texel on each side
    // belongs to the neighbouring atlas cell. Sampling it bleeds a
    // hairline of adjacent art along every sprite edge; the fringe is
    // sub-pixel, so it only shows when a given scale lands it on a pixel
    // boundary. Inset to texel centres, never past the midpoint so a
    // one-texel cell cannot invert.
    let span = inst.uv_max - inst.uv_min;
    let inset = min(vec2<f32>(inst.texel), span * 0.5);
    out.uv = mix(inst.uv_min + inset, inst.uv_max - inset, corner + vec2<f32>(0.5, 0.5));
    out.tint = inst.tint;
    out.page = inst.page;
    out.world = world.xy;
    // Godot's rule, resolved to an index: an item is lit by every light whose
    // mask shares its lowest set bit, and lights are accumulated per bit.
    // Zero reads past every real layer and lands on the identity one.
    out.layer = countTrailingZeros(inst.light_mask);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let tex = textureSample(atlas, atlas_sampler, in.uv, i32(in.page + 0.5));
    let light_uv = clamp(
        in.world * lightmap_u.xform.xy + lightmap_u.xform.zw,
        vec2<f32>(0.0),
        vec2<f32>(1.0),
    );
    let layer = min(in.layer, lightmap_u.layer.x);
    // (rgb = light, a = item multiplier): modulate darkens the item, the light
    // is added on top, and the multiplier is 1 outside every light.
    let lit = textureSample(lightmap, lightmap_sampler, light_uv, i32(layer));
    let tinted = tex * in.tint;
    // (rgb = light, a = item multiplier): modulate darkens the item, the light
    // is added on top, and the multiplier is 1 outside every light.
    let color = vec4<f32>(
        tinted.rgb * (lightmap_u.modulate.rgb * lit.a) + lit.rgb,
        tinted.a,
    );
    if (color.a < 0.001) {
        discard;
    }
    return color;
}
