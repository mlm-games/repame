// Light accumulation pass for the repame-view2d lighting subsystem.
//! Consumed by `lightpass.rs`. One quad per (light, layer) in light-map clip
//! space; the fragment evaluates the light's ramp and shadow mask, and the
//! pipeline's blend state applies the light's `LightBlend` to the target, so
//! the target holds `(rgb = accumulated light, a = item multiplier)`. Add and
//! sub read `src.rgb * src.a`; mix reads `src.rgb` weighted by `1 - src.a`
//! for the colour and `src.a` for the multiplier.

struct Light {
    // xy: world-space centre, zw: 1 / footprint half extent.
    center_half: vec4<f32>,
    // rgb: light colour, w: energy.
    color_energy: vec4<f32>,
    // x: ramp mode (0 = stops, 1 = texture), y: live stop count, zw: reserved.
    params: vec4<f32>,
    // xy: shadow mask bounds min (world), zw: 1 / mask size.
    mask_xform: vec4<f32>,
    // xy: falloff texture offset from the centre, zw: 1 / texture half extent,
    // which the fragment halves again to land on `0..1` across the footprint.
    tex: vec4<f32>,
    // First eight stop positions, packed four per vec4.
    stop_at: array<vec4<f32>, 2>,
    // Stop colours, `(rgb, a)`.
    stop_rgba: array<vec4<f32>, 8>,
};

@group(0) @binding(0) var<uniform> light: Light;
@group(1) @binding(0) var mask: texture_2d<f32>;
@group(1) @binding(1) var mask_sampler: sampler;
@group(2) @binding(0) var ramp_texture: texture_2d<f32>;
@group(2) @binding(1) var ramp_sampler: sampler;

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) world: vec2<f32>,
};

@vertex
fn vs_main(
    @location(0) clip: vec2<f32>,
    @location(1) world: vec2<f32>,
    @location(2) weight: f32,
) -> VertexOut {
    var out: VertexOut;
    out.pos = vec4<f32>(clip, 0.0, 1.0);
    out.world = world;
    // Wedge weights are the mask pass's business only.
    _ = weight;
    return out;
}

fn stop_at(i: u32) -> f32 {
    return light.stop_at[i / 4u][i % 4u];
}

// Same ramp as `repame_view2d::light::stops_at`: clamped at both ends,
// linearly interpolated between ascending stops.
fn stops_ramp(t: f32) -> vec4<f32> {
    let count = light.params.y;
    var ramp = light.stop_rgba[0u];
    var previous = 0.0;
    for (var i = 1u; i < 8u; i = i + 1u) {
        if (f32(i) >= count) {
            break;
        }
        let at = stop_at(i);
        let span = max(at - previous, 1e-6);
        let w = clamp((t - previous) / span, 0.0, 1.0);
        ramp = mix(ramp, light.stop_rgba[i], w);
        previous = at;
    }
    return ramp;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let local = (in.world - light.center_half.xy) * light.center_half.zw;
    let distance = length(local);
    // Both samples sit in uniform control flow: `params.x` is a uniform, and
    // out-of-footprint fragments are zeroed below instead of discarded.
    let ramp_uv_raw = (in.world - light.center_half.xy - light.tex.xy) * light.tex.zw * 0.5
        + vec2<f32>(0.5, 0.5);
    let texel = textureSample(ramp_texture, ramp_sampler, clamp(ramp_uv_raw, vec2<f32>(0.0), vec2<f32>(1.0)));
    let mask_uv = clamp(
        (in.world - light.mask_xform.xy) * light.mask_xform.zw,
        vec2<f32>(0.0),
        vec2<f32>(1.0),
    );
    // One mask channel per shadow tap, each holding a union of that tap's
    // wedges (the mask pass max-blends). Summing the channels is the taps'
    // average: the penumbra. An unshadowed light binds a one-texel zero mask.
    let mask_sample = textureSample(mask, mask_sampler, mask_uv);
    let coverage = mask_sample.r + mask_sample.g + mask_sample.b + mask_sample.a;
    let shadow = 1.0 - min(1.0, coverage);

    var ramp = stops_ramp(clamp(distance, 0.0, 1.0));
    if (light.params.x > 0.5) {
        let inside = ramp_uv_raw.x >= 0.0 && ramp_uv_raw.x <= 1.0
            && ramp_uv_raw.y >= 0.0 && ramp_uv_raw.y <= 1.0;
        if (!inside) {
            return vec4<f32>(0.0, 0.0, 0.0, 0.0);
        }
        ramp = texel;
    } else if (distance > 1.0) {
        // Stops lights end at the rim; the quad is only their bounds.
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let alpha = light.color_energy.w * ramp.a * shadow;
    return vec4<f32>(light.color_energy.rgb * ramp.rgb, alpha);
}
