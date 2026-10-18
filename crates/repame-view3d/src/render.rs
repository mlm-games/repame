//! Scene batch: snapshot in, depth-tested pixels out.
//! Owns per-frame snapshot plus depth-tested and flat pipelines.
//! Offscreen target and blit live in `DepthComposite`.
//! Texture array is fed from [`SceneUpload`]s, one page per group.
//! Frustum cull runs per group in `finish`. `depth_test = false` overlays skip cull.
const SCENE_TARGET_SAMPLE_COUNT: u32 = 1;
const SHADER_PREFIX: &str = r#"
fn repame_pbr_neutral(c: vec3<f32>) -> vec3<f32> {
    let start_compression = 0.8 - 0.04;
    let desaturation = 0.15;
    var color = c;
    let x = min(color.r, min(color.g, color.b));
    let offset = select(0.04, x - 6.25 * x * x, x < 0.08);
    color = color - vec3<f32>(offset);
    let peak = max(color.r, max(color.g, color.b));
    if (peak < start_compression) {
        return color;
    }
    let d = 1.0 - start_compression;
    let new_peak = 1.0 - d * d / (peak + d - start_compression);
    color = color * (new_peak / peak);
    let g = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0);
    return mix(color, vec3<f32>(new_peak), g);
}

// No dither here on purpose. These stages write linear radiance into an
// `Rgba8UnormSrgb` target, so the encode is hardware-side at write time, and
// sampling that target back decodes to linear again. A +/-0.5/255 perturbation
// in linear space would be amplified by the sRGB transfer's 12.92 slope near
// black, turning dither into visible speckle. Dither belongs after the
// display encode, which means a framework-owned resolve into a non-sRGB
// target, not a few lines in this shader.
fn repame_tonemap(c: vec3<f32>, exposure: f32, method: f32) -> vec3<f32> {
    let m = i32(round(method));
    if (m == 0) {
        return c;
    }
    if (m == 2) {
        return repame_pbr_neutral(c * exposure);
    }
    return (c * exposure) / (c * (exposure - 1.0) + vec3<f32>(1.0));
}

struct Camera {
    view_proj: mat4x4<f32>,
    light_dir: vec3<f32>,
    _pad0: f32,
    light_color: vec3<f32>,
    diffuse: f32,
    ambient: vec3<f32>,
    _pad1: f32,
    fog: vec4<f32>,
    fog_range: vec4<f32>,
    cam_pos: vec3<f32>,
    _pad2: f32,
    cam_forward: vec3<f32>,
    _pad3: f32,
    shadow_vp: mat4x4<f32>,
    shadow_params: vec4<f32>,
    shadow_texel: vec4<f32>,
    cascade_vp0: mat4x4<f32>,
    cascade_vp1: mat4x4<f32>,
    cascade_vp2: mat4x4<f32>,
    cascade_vp3: mat4x4<f32>,
    cascade_splits: vec4<f32>,
    cascade_params: vec4<f32>,
    point_pos0: vec4<f32>,
    point_pos1: vec4<f32>,
    point_col0: vec4<f32>,
    point_col1: vec4<f32>,
    point_pos2: vec4<f32>,
    point_pos3: vec4<f32>,
    point_col2: vec4<f32>,
    point_col3: vec4<f32>,
    point_pos4: vec4<f32>,
    point_pos5: vec4<f32>,
    point_col4: vec4<f32>,
    point_col5: vec4<f32>,
    point_pos6: vec4<f32>,
    point_pos7: vec4<f32>,
    point_col6: vec4<f32>,
    point_col7: vec4<f32>,
    point_params: vec4<f32>,
    // Reflection probes: (xyz position, w range) and
    // (intensity, ambient, specular, enabled).
    probe_pos: array<vec4<f32>, 4>,
    probe_params: array<vec4<f32>, 4>,
};

struct SkinPalette {
    joints: array<mat4x4<f32>>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> skin_palette: SkinPalette;
struct Material {
    kcolors: array<vec4<f32>, 4>,
    fog: vec4<f32>,
    fog_range: vec2<f32>,
    _pad: vec2<f32>,
    pages: array<vec4<f32>, 2>,
};

fn page_of(unit: u32) -> f32 {
    return material.pages[unit / 4u][unit % 4u];
}

/// Finest mip level resident for `page`: the level a textured sample reads.
/// `-1` means nothing is resident, so the caller draws untextured (its tint)
/// instead of sampling an unwritten mip. A page past the published cap keeps
/// the fully resident default.
fn repame_resident_level(page: f32) -> f32 {
    let slot = u32(max(page, 0.0) + 0.5);
    if (slot >= 256u) {
        return 0.0;
    }
    return page_levels[slot / 4u][slot % 4u];
}

const COMBINE_LO: f32 = -4.0;
const COMBINE_HI: f32 = 4.0;
const COMBINE_UNIT: f32 = 1.0;
const COMBINE_ZERO: f32 = 0.0;
const COMBINE_ONE: f32 = 1.0;

@group(1) @binding(0) var scene_tex: texture_2d_array<f32>;
@group(1) @binding(1) var scene_smp: sampler;
/// Finest mip level resident per page, published with
/// `SceneBatch::set_page_level`: `-1` = nothing resident (draw the tint),
/// `0` = the fully resident page. A textured sample reads the resident level
/// and never a finer (unwritten) one. Fixed 64 `vec4` slots cover the first
/// 256 pages (a uniform buffer cannot hold a runtime-sized array).
@group(1) @binding(2) var<uniform> page_levels: array<vec4<f32>, 64>;
@group(2) @binding(0) var shadow_tex: texture_depth_2d;
@group(2) @binding(1) var shadow_smp: sampler_comparison;
@group(2) @binding(2) var cascade_tex: texture_depth_2d_array;
@group(2) @binding(3) var cascade_smp: sampler_comparison;
@group(2) @binding(4) var point_tex: texture_depth_cube;
@group(2) @binding(5) var point_smp: sampler_comparison;
@group(2) @binding(6) var probe_tex0: texture_cube<f32>;
@group(2) @binding(7) var probe_tex1: texture_cube<f32>;
@group(2) @binding(8) var probe_tex2: texture_cube<f32>;
@group(2) @binding(9) var probe_tex3: texture_cube<f32>;
@group(2) @binding(10) var probe_smp: sampler;
@group(3) @binding(0) var<uniform> material: Material;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) lit_flag: f32,
    @location(3) uv: vec2<f32>,
    @location(4) tex_mix: f32,
    @location(5) page: f32,
    @location(6) alpha: f32,
    @location(7) cutoff: f32,
    @location(8) world_pos: vec3<f32>,
    @location(9) metallic: f32,
    @location(10) roughness: f32,
    @location(11) emissive: vec3<f32>,
    @location(12) view_depth: f32,
};

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) lit: f32,
    @location(4) uv: vec2<f32>,
    @location(5) tex_mix: f32,
    @location(6) page: f32,
    @location(7) alpha: f32,
    @location(8) cutoff: f32,
    @location(9) metallic: f32,
    @location(10) roughness: f32,
    @location(11) emissive: vec3<f32>,
    @location(12) joints: vec4<u32>,
    @location(13) weights: vec4<f32>,
    @location(14) skin_mix: f32,
    @location(15) skin_base: f32,
) -> VsOut {
    var out: VsOut;
    var skinned_pos = pos;
    var skinned_nrm = normal;
    if (skin_mix > 0.5) {
        let wsum = weights.x + weights.y + weights.z + weights.w;
        if (wsum > 1e-8) {
            let base = i32(skin_base + 0.5);
            var acc_pos = vec3<f32>(0.0);
            var acc_nrm = vec3<f32>(0.0);
            for (var k: i32 = 0; k < 4; k = k + 1) {
                var w: f32 = 0.0;
                var slot: i32 = 0;
                if (k == 0) { w = weights.x; slot = i32(joints.x); }
                else if (k == 1) { w = weights.y; slot = i32(joints.y); }
                else if (k == 2) { w = weights.z; slot = i32(joints.z); }
                else { w = weights.w; slot = i32(joints.w); }
                w = w / wsum;
                if (w > 0.0) {
                    let m = skin_palette.joints[base + slot];
                    acc_pos = acc_pos + (m * vec4<f32>(pos, 1.0)).xyz * w;
                    acc_nrm = acc_nrm + (m * vec4<f32>(normal, 0.0)).xyz * w;
                }
            }
            skinned_pos = acc_pos;
            // Renormalise here, not per fragment: the reference normalises the
            // blended normal in the vertex stage, and so does the CPU path
            // (`SkinnedMesh::blend_vertex`). Interpolating unnormalised normals
            // and normalising afterwards gives a different direction, and a
            // singular blend can yield zero, which `normalize` does not define.
            skinned_nrm = select(
                acc_nrm,
                normalize(acc_nrm),
                dot(acc_nrm, acc_nrm) > 1e-10,
            );
        }
    }
    out.pos = camera.view_proj * vec4<f32>(skinned_pos, 1.0);
    out.color = color;
    out.normal = skinned_nrm;
    out.lit_flag = lit;
    out.uv = uv;
    out.tex_mix = tex_mix;
    out.page = page;
    out.alpha = alpha;
    out.cutoff = cutoff;
    out.world_pos = skinned_pos;
    out.metallic = metallic;
    out.roughness = roughness;
    out.emissive = emissive;
    // Depth along the view axis. The euclidean distance from the eye grows
    // as `1/cos` off-axis, so fog and cascade slices read from it would
    // saturate towards the screen edges.
    out.view_depth = dot(skinned_pos - camera.cam_pos, camera.cam_forward);
    return out;
}

fn repame_probe_weight(distance: f32, range: f32) -> f32 {
    if (range <= 0.0) {
        return 0.0;
    }
    let inner = range * 0.5;
    if (distance <= inner) {
        return 1.0;
    }
    if (distance >= range) {
        return 0.0;
    }
    let t = (distance - inner) / (range - inner);
    return 1.0 - t * t * (3.0 - 2.0 * t);
}

fn repame_probe_cube(index: i32, dir: vec3<f32>) -> vec3<f32> {
    if (index == 0) {
        return textureSampleLevel(probe_tex0, probe_smp, dir, 0.0).rgb;
    }
    if (index == 1) {
        return textureSampleLevel(probe_tex1, probe_smp, dir, 0.0).rgb;
    }
    if (index == 2) {
        return textureSampleLevel(probe_tex2, probe_smp, dir, 0.0).rgb;
    }
    return textureSampleLevel(probe_tex3, probe_smp, dir, 0.0).rgb;
}

// Ambient + specular contribution of the frame's reflection probes: the
// cube is sampled along the view reflection and weighted by the probe's
// falloff volume, its two mix weights and the surface's smoothness.
fn repame_probe(
    world_pos: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    base: vec3<f32>,
    metallic: f32,
    roughness: f32,
) -> vec3<f32> {
    var acc = vec3<f32>(0.0);
    let reflection = reflect(-v, n);
    for (var i: i32 = 0; i < 4; i = i + 1) {
        let params = camera.probe_params[i];
        if (params.w < 0.5) {
            continue;
        }
        let spec = camera.probe_pos[i];
        let weight = repame_probe_weight(length(spec.xyz - world_pos), spec.w) * params.x;
        if (weight <= 0.0) {
            continue;
        }
        let cube = repame_probe_cube(i, reflection);
        let fresnel = pow(1.0 - max(dot(n, v), 0.0), 5.0);
        let specular = params.z * mix(0.15, 1.0, fresnel) * (1.0 - clamp(roughness, 0.0, 1.0));
        acc = acc + cube * (base * (1.0 - metallic) * params.y + vec3<f32>(specular)) * weight;
    }
    return acc;
}

"#;

/// Legacy PBR-lite fragment stage. Prepended to [`SHADER_PREFIX`] for the
/// default pipeline and to each material program's module for its own.
const SHADER_LEGACY_FS: &str = r#"
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var base = in.color;
    var alpha = in.alpha;
    if (in.tex_mix > 0.5) {
        // Sample the page's finest resident level: level 0 when the page is
        // fully resident (the historical path, byte for byte), the published
        // level when only coarser mips have arrived, nothing when no level is
        // resident (draw the tint instead of a fabricated or unwritten mip).
        let resident = repame_resident_level(in.page);
        if (resident >= 0.0) {
            let tex = textureSampleLevel(scene_tex, scene_smp, in.uv, i32(in.page + 0.5), resident);
            if (tex.a < 0.001 && in.cutoff < 0.001) {
                discard;
            }
            base = in.color * tex.rgb;
            alpha = in.alpha * tex.a;
        }
    }
    if (alpha < in.cutoff) {
        discard;
    }
    let n = normalize(in.normal);
    let v = normalize(camera.cam_pos - in.world_pos);
    let ndl = max(dot(n, camera.light_dir), 0.0);
    let diffuse = base * (1.0 - in.metallic) * ndl * camera.diffuse;
    let h = normalize(camera.light_dir + v);
    let spec_pow = mix(256.0, 8.0, clamp(in.roughness, 0.0, 1.0));
    let spec = mix(camera.light_color, base, in.metallic)
        * pow(max(dot(n, h), 0.0), spec_pow)
        * (1.0 - in.roughness) * camera.diffuse;
    var shadow: f32 = 1.0;
    if (in.lit_flag > 0.5 && camera.shadow_params.y > 0.5) {
        let light_clip = camera.shadow_vp * vec4<f32>(in.world_pos, 1.0);
        let light_ndc = light_clip.xyz / max(light_clip.w, 1e-6);
        let suv = light_ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
        if (all(suv >= vec2<f32>(0.0)) && all(suv <= vec2<f32>(1.0))) {
            let ref_depth = light_ndc.z - camera.shadow_params.z;
            var lit_count: f32 = 0.0;
            let texel = camera.shadow_texel.xy;
            for (var oy: i32 = -1; oy <= 1; oy = oy + 1) {
                for (var ox: i32 = -1; ox <= 1; ox = ox + 1) {
                    lit_count = lit_count + textureSampleCompareLevel(
                        shadow_tex, shadow_smp, suv + vec2<f32>(f32(ox), f32(oy)) * texel, ref_depth);
                }
            }
            let lit_frac = lit_count / 9.0;
            shadow = mix(1.0, lit_frac, camera.shadow_params.x);
        }
    }
    if (in.lit_flag > 0.5 && camera.cascade_params.y > 0.5) {
        let view_depth = in.view_depth;
        var slice: i32 = 0;
        if (view_depth > camera.cascade_splits.x) { slice = 1; }
        if (view_depth > camera.cascade_splits.y) { slice = 2; }
        if (view_depth > camera.cascade_splits.z) { slice = 3; }
        if (f32(slice) < camera.cascade_params.z) {
            var cvp: mat4x4<f32>;
            if (slice == 0) { cvp = camera.cascade_vp0; }
            else if (slice == 1) { cvp = camera.cascade_vp1; }
            else if (slice == 2) { cvp = camera.cascade_vp2; }
            else { cvp = camera.cascade_vp3; }
            let biased_pos = in.world_pos + n * camera.cascade_params.w;
            let light_clip = cvp * vec4<f32>(biased_pos, 1.0);
            let light_ndc = light_clip.xyz / max(light_clip.w, 1e-6);
            let suv = light_ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
            if (all(suv >= vec2<f32>(0.0)) && all(suv <= vec2<f32>(1.0))) {
                let ref_depth = light_ndc.z - camera.shadow_params.z;
                var lit_count: f32 = 0.0;
                let texel = camera.shadow_texel.xy;
                for (var oy: i32 = -1; oy <= 1; oy = oy + 1) {
                    for (var ox: i32 = -1; ox <= 1; ox = ox + 1) {
                        lit_count = lit_count + textureSampleCompareLevel(
                            cascade_tex, cascade_smp, suv + vec2<f32>(f32(ox), f32(oy)) * texel, slice, ref_depth);
                    }
                }
                let lit_frac = lit_count / 9.0;
                shadow = shadow * mix(1.0, lit_frac, camera.cascade_params.x);
            }
        }
    }
    var point_accum = vec3<f32>(0.0);
    let point_count = i32(camera.point_params.x + 0.5);
    for (var pi: i32 = 0; pi < 8; pi = pi + 1) {
        if (pi >= point_count) { break; }
        var lpos: vec3<f32>;
        var lcol: vec3<f32>;
        if (pi == 0) { lpos = camera.point_pos0.xyz; lcol = camera.point_col0.xyz; }
        else if (pi == 1) { lpos = camera.point_pos1.xyz; lcol = camera.point_col1.xyz; }
        else if (pi == 2) { lpos = camera.point_pos2.xyz; lcol = camera.point_col2.xyz; }
        else if (pi == 3) { lpos = camera.point_pos3.xyz; lcol = camera.point_col3.xyz; }
        else if (pi == 4) { lpos = camera.point_pos4.xyz; lcol = camera.point_col4.xyz; }
        else if (pi == 5) { lpos = camera.point_pos5.xyz; lcol = camera.point_col5.xyz; }
        else if (pi == 6) { lpos = camera.point_pos6.xyz; lcol = camera.point_col6.xyz; }
        else { lpos = camera.point_pos7.xyz; lcol = camera.point_col7.xyz; }
        var lrange: f32;
        var linten: f32;
        if (pi == 0) { lrange = camera.point_pos0.w; linten = camera.point_col0.w; }
        else if (pi == 1) { lrange = camera.point_pos1.w; linten = camera.point_col1.w; }
        else if (pi == 2) { lrange = camera.point_pos2.w; linten = camera.point_col2.w; }
        else if (pi == 3) { lrange = camera.point_pos3.w; linten = camera.point_col3.w; }
        else if (pi == 4) { lrange = camera.point_pos4.w; linten = camera.point_col4.w; }
        else if (pi == 5) { lrange = camera.point_pos5.w; linten = camera.point_col5.w; }
        else if (pi == 6) { lrange = camera.point_pos6.w; linten = camera.point_col6.w; }
        else { lrange = camera.point_pos7.w; linten = camera.point_col7.w; }
        let to_light = lpos - in.world_pos;
        let dist = length(to_light);
        if (dist < lrange && dist > 1e-4) {
            let ldir = to_light / dist;
            let atten = linten / (dist * dist + 1.0);
            let pndl = max(dot(n, ldir), 0.0);
            var pshadow: f32 = 1.0;
            if (camera.point_params.y > 0.5 && in.lit_flag > 0.5) {
                let cube_uv = normalize(-to_light);
                let ref_depth = dist / max(lrange, 1e-6) - camera.point_params.z;
                let lit_s = textureSampleCompareLevel(
                    point_tex, point_smp, cube_uv, ref_depth);
                pshadow = mix(1.0, lit_s, camera.point_params.w);
            }
            let pdiff = base * (1.0 - in.metallic) * pndl * atten;
            let ph = normalize(ldir + v);
            let pspec_pow = mix(256.0, 8.0, clamp(in.roughness, 0.0, 1.0));
            let pspec = mix(lcol, base, in.metallic)
                * pow(max(dot(n, ph), 0.0), pspec_pow)
                * (1.0 - in.roughness) * atten;
            point_accum = point_accum + (pdiff * lcol + pspec) * pshadow * in.lit_flag;
        }
    }
    let probe = repame_probe(in.world_pos, n, v, base, in.metallic, in.roughness) * in.lit_flag;
    let lit = base * camera.ambient + (camera.light_color * diffuse + spec) * shadow + point_accum + in.emissive + probe;
    var rgb = mix(base, lit, in.lit_flag);
    let dist = in.view_depth;
    let fog_t = clamp((dist - camera.fog_range.x) / max(camera.fog_range.y - camera.fog_range.x, 1e-6), 0.0, 1.0) * clamp(camera.fog.x, 0.0, 1.0);
    rgb = mix(rgb, camera.fog.yzw, fog_t * in.lit_flag);
    rgb = repame_tonemap(rgb, camera.fog_range.z, camera.fog_range.w);
    return vec4<f32>(rgb, alpha);
}
"#;

/// Fullscreen pass shader: one triangle over the viewport, sampling the
/// source pass target. The fragment stage applies the pass's tint, so a
/// presenting pass can grade or fade what it composites.
const FULLSCREEN_SHADER: &str = r#"
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_smp: sampler;
@group(0) @binding(2) var<uniform> tint: vec4<f32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
    var out: VsOut;
    let x = f32((index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(index & 2u) * 2.0 - 1.0;
    out.pos = vec4<f32>(x, -y, 0.0, 1.0);
    out.uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let tex = textureSample(src_tex, src_smp, in.uv);
    return vec4<f32>(tex.rgb * tint.rgb, tex.a * tint.a);
}
"#;

/// The GPU-skin blend [`SHADER_PREFIX`]'s `vs_main` carries, verbatim.
/// Spliced out for devices without `VERTEX_STORAGE` (see [`scene_prefix`]);
/// `skinned_pos`/`skinned_nrm` stay as plain copies of the bind position
/// and normal, so every other vertex output is untouched.
const SCENE_SKIN_BODY_WGSL: &str = r#"if (skin_mix > 0.5) {
        let wsum = weights.x + weights.y + weights.z + weights.w;
        if (wsum > 1e-8) {
            let base = i32(skin_base + 0.5);
            var acc_pos = vec3<f32>(0.0);
            var acc_nrm = vec3<f32>(0.0);
            for (var k: i32 = 0; k < 4; k = k + 1) {
                var w: f32 = 0.0;
                var slot: i32 = 0;
                if (k == 0) { w = weights.x; slot = i32(joints.x); }
                else if (k == 1) { w = weights.y; slot = i32(joints.y); }
                else if (k == 2) { w = weights.z; slot = i32(joints.z); }
                else { w = weights.w; slot = i32(joints.w); }
                w = w / wsum;
                if (w > 0.0) {
                    let m = skin_palette.joints[base + slot];
                    acc_pos = acc_pos + (m * vec4<f32>(pos, 1.0)).xyz * w;
                    acc_nrm = acc_nrm + (m * vec4<f32>(normal, 0.0)).xyz * w;
                }
            }
            skinned_pos = acc_pos;
            // Renormalise here, not per fragment: the reference normalises the
            // blended normal in the vertex stage, and so does the CPU path
            // (`SkinnedMesh::blend_vertex`). Interpolating unnormalised normals
            // and normalising afterwards gives a different direction, and a
            // singular blend can yield zero, which `normalize` does not define.
            skinned_nrm = select(
                acc_nrm,
                normalize(acc_nrm),
                dot(acc_nrm, acc_nrm) > 1e-10,
            );
        }
    }"#;

/// The scene vertex prefix for `skinning`.
///
/// The supported form is [`SHADER_PREFIX`] byte for byte. The unsupported
/// form splices out the palette declaration and
/// [`SCENE_SKIN_BODY_WGSL`], leaving a module that reads no storage buffer
/// and therefore matches a camera layout without binding 1.
fn scene_prefix(skinning: GpuSkinning) -> String {
    let mut source = String::from(SHADER_PREFIX);
    if !skinning.is_supported() {
        strip_skin_palette(&mut source);
        let stripped = source.replace(SCENE_SKIN_BODY_WGSL, "");
        source = stripped;
    }
    source
}

/// The default pipeline's module for `skinning`: shared prefix plus the
/// legacy fragment stage. Material programs reuse the same prefix with
/// their own fragment stage, so the vertex path and every binding stay
/// identical.
fn scene_shader(skinning: GpuSkinning) -> String {
    let mut source = scene_prefix(skinning);
    source.push_str(SHADER_LEGACY_FS);
    source
}

/// Depth-only vertex shader for the shadow pass: skinned position
/// through the shadow matrix (bind pose + palette, same blend as
/// the scene pass, so animated casters shadow their posed shape,
/// not their bind pose). Static verts carry skin_mix 0 and skip
/// the palette exactly like the scene shader.
///
/// Locations are depth-pass-local (0..4): the pass uses its own
/// buffer layout below (same offsets, remapped locations), because
/// the scene layout's 17 attributes exceed the device's 16-slot
/// vertex limit.
const SHADOW_DEPTH_SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
};

struct SkinPalette {
    joints: array<mat4x4<f32>>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> skin_palette: SkinPalette;

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1) joints: vec4<u32>,
    @location(2) weights: vec4<f32>,
    @location(3) skin_mix: f32,
    @location(4) skin_base: f32,
) -> @builtin(position) vec4<f32> {
    var p = pos;
    if (skin_mix > 0.5) {
        let wsum = weights.x + weights.y + weights.z + weights.w;
        if (wsum > 1e-8) {
            let base = i32(skin_base + 0.5);
            var acc = vec3<f32>(0.0);
            for (var k: i32 = 0; k < 4; k = k + 1) {
                var w: f32 = 0.0;
                var slot: i32 = 0;
                if (k == 0) { w = weights.x; slot = i32(joints.x); }
                else if (k == 1) { w = weights.y; slot = i32(joints.y); }
                else if (k == 2) { w = weights.z; slot = i32(joints.z); }
                else { w = weights.w; slot = i32(joints.w); }
                w = w / wsum;
                if (w > 0.0) {
                    let m = skin_palette.joints[base + slot];
                    acc = acc + (m * vec4<f32>(pos, 1.0)).xyz * w;
                }
            }
            p = acc;
        }
    }
    return camera.view_proj * vec4<f32>(p, 1.0);
}
"#;

/// The GPU-skin blend [`SHADOW_DEPTH_SHADER`] carries, verbatim; spliced
/// out for devices without `VERTEX_STORAGE`.
const SHADOW_SKIN_BODY_WGSL: &str = r#"if (skin_mix > 0.5) {
        let wsum = weights.x + weights.y + weights.z + weights.w;
        if (wsum > 1e-8) {
            let base = i32(skin_base + 0.5);
            var acc = vec3<f32>(0.0);
            for (var k: i32 = 0; k < 4; k = k + 1) {
                var w: f32 = 0.0;
                var slot: i32 = 0;
                if (k == 0) { w = weights.x; slot = i32(joints.x); }
                else if (k == 1) { w = weights.y; slot = i32(joints.y); }
                else if (k == 2) { w = weights.z; slot = i32(joints.z); }
                else { w = weights.w; slot = i32(joints.w); }
                w = w / wsum;
                if (w > 0.0) {
                    let m = skin_palette.joints[base + slot];
                    acc = acc + (m * vec4<f32>(pos, 1.0)).xyz * w;
                }
            }
            p = acc;
        }
    }"#;

/// [`SHADOW_DEPTH_SHADER`] for `skinning`: the supported form is the
/// constant byte for byte, the unsupported form drops the palette
/// declaration and blend so the module matches a camera layout without
/// binding 1.
fn shadow_depth_shader(skinning: GpuSkinning) -> String {
    let mut source = String::from(SHADOW_DEPTH_SHADER);
    if !skinning.is_supported() {
        strip_skin_palette(&mut source);
        let stripped = source.replace(SHADOW_SKIN_BODY_WGSL, "");
        source = stripped;
    }
    source
}

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use repose_render_wgpu::{
    BlitRenderPass, CallbackRenderPass, CallbackResources, DepthComposite, ScenePassOps,
    ScreenDescriptor,
};

use super::mesh::MeshGroup;
use super::particles::{
    GpuColliders, GpuDrawParams, GpuEmitter, GpuSimState, ParticleFrame, SIMULATE_WORKGROUP,
    pack_colliders, pack_emitter, request_capacity,
};
use super::reflection::ProbeSettings;
use super::postfx::{PostProcess, PostStage, POST_WGSL, depth_velocity_shader};
use super::skin::{GpuSkinning, SkinnedDraw, strip_skin_palette};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    light_dir: [f32; 3],
    _pad0: f32,
    light_color: [f32; 3],
    diffuse: f32,
    ambient: [f32; 3],
    _pad1: f32,
    fog: [f32; 4],
    fog_range: [f32; 4],
    cam_pos: [f32; 3],
    _pad2: f32,
    /// Unit vector the camera looks along. Fog and cascade slices are
    /// measured as depth along it, so it must come from the same snapshot
    /// as [`SceneBatch::set_camera_pos`].
    cam_forward: [f32; 3],
    _pad3: f32,
    shadow_vp: [[f32; 4]; 4],
    /// (strength, enabled flag, bias, unused).
    shadow_params: [f32; 4],
    /// (texel, texel, unused, unused).
    shadow_texel: [f32; 4],
    cascade_vp0: [[f32; 4]; 4],
    cascade_vp1: [[f32; 4]; 4],
    cascade_vp2: [[f32; 4]; 4],
    cascade_vp3: [[f32; 4]; 4],
    /// Split depths (x/y/z = slice 1/2/3 far planes in camera distance;
    /// slice 0 starts at the camera near). w unused.
    cascade_splits: [f32; 4],
    /// (strength, enabled flag, armed count, normal bias).
    cascade_params: [f32; 4],
    point_pos0: [f32; 4],
    point_pos1: [f32; 4],
    point_col0: [f32; 4],
    point_col1: [f32; 4],
    point_pos2: [f32; 4],
    point_pos3: [f32; 4],
    point_col2: [f32; 4],
    point_col3: [f32; 4],
    point_pos4: [f32; 4],
    point_pos5: [f32; 4],
    point_col4: [f32; 4],
    point_col5: [f32; 4],
    point_pos6: [f32; 4],
    point_pos7: [f32; 4],
    point_col6: [f32; 4],
    point_col7: [f32; 4],
    /// (count, shadow enabled, bias, strength).
    point_params: [f32; 4],
    /// `(position.xyz, range)` per reflection probe.
    probe_pos: [[f32; 4]; super::reflection::MAX_PROBES],
    /// `(intensity, ambient, specular, enabled)` per reflection probe.
    probe_params: [[f32; 4]; super::reflection::MAX_PROBES],
}

/// The scene vertex layout, shared by the default and material pipelines.
const VERTEX_ATTRIBUTES: &[wgpu::VertexAttribute] = &[
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
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x2,
        offset: 40,
        shader_location: 4,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32,
        offset: 48,
        shader_location: 5,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32,
        offset: 52,
        shader_location: 6,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32,
        offset: 56,
        shader_location: 7,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32,
        offset: 60,
        shader_location: 8,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32,
        offset: 64,
        shader_location: 9,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32,
        offset: 68,
        shader_location: 10,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 72,
        shader_location: 11,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Uint32x4,
        offset: 84,
        shader_location: 12,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 100,
        shader_location: 13,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32,
        offset: 116,
        shader_location: 14,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32,
        offset: 120,
        shader_location: 15,
    },
];

const _: () = assert!(size_of::<CameraUniform>() == 960);

/// Lowers a material's shading uniform into the GPU block, resolving the
/// texture page each unit samples.
pub fn material_uniform_for(
    uniform: &super::shading::ShadingUniform,
    pages: &[u32],
) -> MaterialUniform {
    let mut slots = [[0.0_f32; 4]; 2];
    for (unit, slot) in slots
        .iter_mut()
        .flat_map(|pair| pair.iter_mut())
        .enumerate()
    {
        *slot = pages.get(unit).copied().unwrap_or(unit as u32) as f32;
    }
    MaterialUniform {
        kcolors: uniform.kcolors,
        fog: uniform.fog,
        fog_range: [uniform.fog_range[0], uniform.fog_range[1]],
        _pad: [0.0; 2],
        pages: slots,
    }
}

/// Frame light rig: one directional sun plus up to
/// [`MAX_POINTS`](crate::MAX_POINTS) point lights. The directional light
/// keeps its legacy role (diffuse + specular + single shadow map); point
/// lights add inverse-square diffuse + specular without shadows unless
/// [`LightRig::point_shadows`] arms the cube pass. Everything is linear
/// space; flat groups ignore the whole rig.
#[derive(Clone, Debug)]
pub struct LightRig {
    /// Directional sun (same semantics as the old `SceneLight`).
    pub sun: SceneLight,
    /// Point lights in submission order. Past `MAX_POINTS` the batch
    /// keeps the brightest (color luminance times intensity) and warns.
    pub points: Vec<super::shadow::PointLight>,
    /// Cascade rig for the sun. `Some` replaces the legacy single map
    /// with fitted slices (same strength/bias semantics, per-slice fit).
    /// `None` (default) keeps the legacy `ShadowDesc` path exactly.
    pub cascades: Option<super::shadow::CascadeDesc>,
    /// Shadow-casting point light index into `points` (`None` = no cube
    /// pass; all points shade unshadowed). Out-of-range disables with a
    /// warning, never a panic.
    pub point_shadows: Option<usize>,
    /// Cube-map bias override (default 0.005). Clamped `0..=0.05`.
    pub point_bias: f32,
    /// Cube shadow strength override (default 1.0). Clamped `0..=1`.
    pub point_strength: f32,
}

impl Default for LightRig {
    fn default() -> Self {
        Self {
            sun: SceneLight::default(),
            points: Vec::new(),
            cascades: None,
            point_shadows: None,
            point_bias: 0.005,
            point_strength: 1.0,
        }
    }
}

/// Frame light, linear space. Direction points toward light.
/// Flat groups ignore it. Fog blends lit frags toward fog color.
/// Exposure 1.0 is identity. Flat groups ignore both.
#[derive(Clone, Copy, Debug)]
pub struct SceneLight {
    /// Unit vector toward light (normalized on use).
    pub direction: [f32; 3],
    /// Light color, linear RGB.
    pub color: [f32; 3],
    /// Diffuse strength.
    pub diffuse: f32,
    /// Ambient floor added to each lit frag.
    pub ambient: [f32; 3],
    /// Fog density 0..1. 0 = off.
    pub fog: f32,
    /// Fog color, linear RGB.
    pub fog_color: [f32; 3],
    /// Fog starts here, world units from camera.
    pub fog_start: f32,
    /// Fog is full past here.
    pub fog_end: f32,
    /// Exposure applied before the tone curve. 1.0 = off. `Reinhard` folds it
    /// into the curve; the other methods scale the radiance by it first.
    pub exposure: f32,
    /// Tone curve applied after fog.
    pub tonemap: TonemapMethod,
}

/// Tone curve for the HDR resolve. `Reinhard` is the historical curve;
/// `KhronosPbrNeutral` desaturates highlights without hue shift, which
/// `Reinhard` does not.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TonemapMethod {
    /// Identity, display-linear. Exposure is not applied.
    None,
    /// Extended Reinhard, `(c*e)/(c*(e-1)+1)`. Hue-shifts highlights.
    Reinhard,
    /// Khronos PBR Neutral: shoulder plus a desaturation toward the new peak.
    /// Takes `exposure` as a pre-scale on radiance, not folded into a curve.
    KhronosPbrNeutral,
}

impl TonemapMethod {
    pub const fn index(self) -> f32 {
        match self {
            Self::None => 0.0,
            Self::Reinhard => 1.0,
            Self::KhronosPbrNeutral => 2.0,
        }
    }
}

impl Default for TonemapMethod {
    fn default() -> Self {
        Self::Reinhard
    }
}

impl Default for SceneLight {
    fn default() -> Self {
        Self {
            direction: [0.3, 1.0, 0.4],
            color: [1.0, 1.0, 1.0],
            diffuse: 0.9,
            ambient: [0.35, 0.35, 0.38],
            fog: 0.0,
            fog_color: [0.5, 0.55, 0.6],
            fog_start: 100.0,
            fog_end: 600.0,
            exposure: 1.0,
            tonemap: TonemapMethod::Reinhard,
        }
    }
}

/// One vertex layout for flat, lit, and textured groups.
/// Flat verts carry dummy up-normal and lit 0. Untextured carry dummy uv and mix 0.
/// Skinned verts carry joints/weights + skin 1 (GPU blends from bind pose);
/// static verts carry zeros + skin 0 (shader skips the palette).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vert {
    pos: [f32; 3],
    color: [f32; 3],
    normal: [f32; 3],
    lit: f32,
    uv: [f32; 2],
    tex_mix: f32,
    page: f32,
    alpha: f32,
    cutoff: f32,
    metallic: f32,
    roughness: f32,
    emissive: [f32; 3],
    joints: [u32; 4],
    weights: [f32; 4],
    skin_mix: f32,
    skin_base: f32,
}

/// Uniform block one lowered material program reads: its constant color
/// registers, its fog window, and the texture pages each unit samples.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MaterialUniform {
    pub kcolors: [[f32; 4]; 4],
    pub fog: [f32; 4],
    pub fog_range: [f32; 2],
    pub _pad: [f32; 2],
    /// Texture page per unit, carried as vec4 pairs because uniform arrays
    /// need a 16-byte stride: unit `u` reads `pages[u / 4][u % 4]`.
    pub pages: [[f32; 4]; 2],
}

const _: () = assert!(size_of::<MaterialUniform>() == 128);
const _: () = assert!(super::material::MAX_TEX_UNITS == 8);

/// GPU state for fullscreen passes.
struct FullscreenState {
    pipeline: wgpu::RenderPipeline,
    overlay_pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    binds: HashMap<String, wgpu::BindGroup>,
    layout: wgpu::BindGroupLayout,
    tints: HashMap<String, wgpu::Buffer>,
    present: Option<(String, wgpu::BindGroup)>,
    present_tint: Option<wgpu::Buffer>,
}

/// One pass as the frame executor sees it: its declaration plus the GPU
/// handles it needs, paired by declaration name.
struct PassEntry {
    spec: PassSpec,
    view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    texture: wgpu::Texture,
    /// The target a resolve pass copies from, when it has one.
    source_texture: Option<wgpu::Texture>,
}

/// A declared render target the batch owns: its texture and view, allocated
/// on first use and reused while the declaration is unchanged.
struct PassTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    /// The batch's pipelines are depth-aware, so every pass target carries a
    /// matching depth buffer rather than a separate no-depth pipeline set.
    /// The texture is held so its view stays valid.
    _depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
    width: u32,
    height: u32,
    format: super::pass::RenderTargetFormat,
    samples: u32,
}

/// Async-map state of a depth readback, written by the map callback and read
/// on the render thread's next poll.
const READBACK_IN_FLIGHT: u8 = 0;
const READBACK_MAPPED: u8 = 1;
const READBACK_FAILED: u8 = 2;

/// One frame's occlusion feed readback in flight: `staging` holds a copy of
/// the chain the passes wrote. `MAP_READ` cannot combine with `STORAGE`
/// without the `MAPPABLE_PRIMARY_BUFFERS` feature, so the passes write the
/// chain buffer and it is copied here.
struct PendingDepthReadback {
    staging: wgpu::Buffer,
    /// Source target size in pixels (the pyramid's dimensions).
    width: u32,
    height: u32,
    /// The camera the pass was drawn with. A pyramid only means something
    /// next to the frame it was captured from, so the state handed to
    /// [`SceneBatch::set_occlusion`] carries both together.
    view_proj: Mat4,
    cam_pos: [f32; 3],
    cam_forward: [f32; 3],
    /// Feed settings the frame was captured with.
    tile: u32,
    bias: f32,
    /// Set by the map callback when the host map lands (or fails).
    map_state: Arc<AtomicU8>,
}

/// Invocations per workgroup of the reduction passes ([`HIZ_WGSL`]).
const HIZ_WORKGROUP: u32 = 64;

/// Pyramid levels the level-parameter buffer holds. The chain halves until
/// one tile covers the image, so even a 32 k-texel-wide target fits.
const MAX_HIZ_LEVELS: usize = 32;

/// One level's parameters for the reduction passes, read through a dynamic
/// uniform offset (`HizPipeline::uniform_stride` bytes per level).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HizLevel {
    /// First `f32` index of this level in the packed chain.
    dst_offset: u32,
    dst_cols: u32,
    dst_rows: u32,
    /// First `f32` index of the level being coarsened (`coarsen` only).
    src_offset: u32,
    src_cols: u32,
    src_rows: u32,
    /// Source image size in pixels (`tile_reduce` only).
    width: u32,
    height: u32,
    /// Level-0 tile edge in pixels (`tile_reduce` only).
    tile: u32,
    /// Invocations per dispatch row, so a grid too wide for
    /// `max_compute_workgroups_per_dimension` spills into y.
    row_stride: u32,
    _pad: [u32; 2],
}

/// Must stay the WGSL `Level`: 12 `u32`s.
const _: () = assert!(std::mem::size_of::<HizLevel>() == 48);

/// Uniform of the phase-two test ([`HIZ_WGSL`]'s `hz_test`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HizTestParams {
    view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    cam_forward: [f32; 4],
    /// Tile edge, bias, source width, source height.
    hz: [f32; 4],
    /// Candidate count, level count, invocations per dispatch row, 0.
    counts: [u32; 4],
}

/// Must stay the WGSL `TestParams`: 128 bytes.
const _: () = assert!(std::mem::size_of::<HizTestParams>() == 128);

/// Dispatch grid for `count` invocations of [`HIZ_WORKGROUP`]: x is capped at
/// `max_compute_workgroups_per_dimension`, the remainder spills into y, and
/// the shader strides by `groups_x * HIZ_WORKGROUP`.
fn dispatch_groups(count: u32) -> (u32, u32) {
    let groups_x = count.div_ceil(HIZ_WORKGROUP).min(65535).max(1);
    let groups_y = count.div_ceil(HIZ_WORKGROUP * groups_x);
    (groups_x, groups_y)
}

/// The reduction: `tile_reduce` writes the level-0 tile maxima from the depth
/// target's `.r` (exact `max`, `0` where nothing finite was drawn), then one
/// `coarsen` dispatch per further level takes the max of each 2x2 block. No
/// atomics, no sampler, no filtering: the same level chain the per-pixel CPU
/// build would produce for the same depths.
const HIZ_WGSL: &str = r#"
struct Level {
    dst_offset: u32,
    dst_cols: u32,
    dst_rows: u32,
    src_offset: u32,
    src_cols: u32,
    src_rows: u32,
    width: u32,
    height: u32,
    tile: u32,
    row_stride: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var depth_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> tiles: array<f32>;
@group(0) @binding(2) var<uniform> level: Level;

@compute @workgroup_size(64)
fn tile_reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.y * level.row_stride + gid.x;
    if (index >= level.dst_cols * level.dst_rows) {
        return;
    }
    let x0 = (index % level.dst_cols) * level.tile;
    let y0 = (index / level.dst_cols) * level.tile;
    var far = 0.0;
    for (var dy = 0u; dy < level.tile; dy = dy + 1u) {
        let y = y0 + dy;
        if (y >= level.height) {
            break;
        }
        for (var dx = 0u; dx < level.tile; dx = dx + 1u) {
            let x = x0 + dx;
            if (x >= level.width) {
                break;
            }
            let depth = textureLoad(depth_tex, vec2<i32>(i32(x), i32(y)), 0).r;
            // Skip non-finite and non-positive depths, the same rule the
            // per-pixel CPU build applies: the upper bound is `f32::MAX`,
            // so only finite values pass (NaN already fails `>`), without
            // `isNan`/`isInf`, which naga's WGSL front end does not take.
            if (depth > far && depth <= 3.402823466e38) {
                far = depth;
            }
        }
    }
    tiles[level.dst_offset + index] = far;
}

@compute @workgroup_size(64)
fn coarsen(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.y * level.row_stride + gid.x;
    if (index >= level.dst_cols * level.dst_rows) {
        return;
    }
    let cx = index % level.dst_cols;
    let cy = index / level.dst_cols;
    var far = 0.0;
    for (var dy = 0u; dy < 2u; dy = dy + 1u) {
        let sy = cy * 2u + dy;
        if (sy >= level.src_rows) {
            break;
        }
        for (var dx = 0u; dx < 2u; dx = dx + 1u) {
            let sx = cx * 2u + dx;
            if (sx >= level.src_cols) {
                break;
            }
            let depth = tiles[level.src_offset + sy * level.src_cols + sx];
            if (depth > far) {
                far = depth;
            }
        }
    }
    tiles[level.dst_offset + index] = far;
}
"#;

/// The phase-two test: one invocation per phase-one reject, writing that
/// reject's indirect draw args - a zeroed index count hides a group that is
/// still occluded this frame. Its own module so the test pipeline needs no
/// bind groups it does not use.
const HIZ_TEST_WGSL: &str = r#"
struct Candidate {
    min: vec3<f32>,
    _pad0: f32,
    max: vec3<f32>,
    _pad1: f32,
    args: vec4<u32>,
};

struct TestParams {
    view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    cam_forward: vec4<f32>,
    hz: vec4<f32>,
    counts: vec4<u32>,
};

@group(0) @binding(0) var<storage, read> hz_chain: array<f32>;
@group(0) @binding(1) var<storage, read> hz_levels: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> hz_candidates: array<Candidate>;
@group(0) @binding(3) var<storage, read_write> hz_args: array<u32>;
@group(0) @binding(4) var<uniform> test: TestParams;

// The CPU `OcclusionCulling::occluded`: a world AABB is hidden only when
// every tile fully covered by its screen rectangle holds a farther depth
// (and a tile nothing was drawn into never culls). Same conservatism, same
// pixel-rect convention as the readback path.
fn hz_occluded(bounds_min: vec3<f32>, bounds_max: vec3<f32>) -> bool {
    var lo = vec2<f32>(1e30, 1e30);
    var hi = vec2<f32>(-1e30, -1e30);
    var nearest = 3.402823466e38;
    for (var corner = 0u; corner < 8u; corner = corner + 1u) {
        let p = vec3<f32>(
            select(bounds_min.x, bounds_max.x, (corner & 1u) != 0u),
            select(bounds_min.y, bounds_max.y, (corner & 2u) != 0u),
            select(bounds_min.z, bounds_max.z, (corner & 4u) != 0u),
        );
        let clip = test.view_proj * vec4<f32>(p, 1.0);
        if (!(clip.w > 1e-6)) {
            return false;
        }
        let ndc = clip.xy / clip.w;
        lo = min(lo, ndc);
        hi = max(hi, ndc);
        nearest = min(nearest, dot(p - test.cam_pos.xyz, test.cam_forward.xyz));
    }
    let lo_clamped = clamp(lo, vec2<f32>(-1.0), vec2<f32>(1.0));
    let hi_clamped = clamp(hi, vec2<f32>(-1.0), vec2<f32>(1.0));
    if (lo_clamped.x >= hi_clamped.x || lo_clamped.y >= hi_clamped.y) {
        return false;
    }
    if (!(nearest < 3.402823466e38)) {
        return false;
    }
    let width = test.hz.z;
    let height = test.hz.w;
    let x0 = clamp((lo_clamped.x * 0.5 + 0.5) * width, 0.0, width);
    let x1 = clamp((hi_clamped.x * 0.5 + 0.5) * width, 0.0, width);
    let y0 = clamp((0.5 - hi_clamped.y * 0.5) * height, 0.0, height);
    let y1 = clamp((0.5 - lo_clamped.y * 0.5) * height, 0.0, height);
    if (x0 >= x1 || y0 >= y1) {
        return false;
    }
    let bias = max(test.hz.y, 0.0);
    for (var level = 0u; level < test.counts.y; level = level + 1u) {
        let level_info = hz_levels[level];
        let scale = test.hz.x * f32(1u << level);
        let t0 = u32(ceil(x0 / scale));
        let t1 = u32(floor(x1 / scale));
        let u0 = u32(ceil(y0 / scale));
        let u1 = u32(floor(y1 / scale));
        if (t0 >= t1 || u0 >= u1 || t1 > level_info.y || u1 > level_info.z) {
            continue;
        }
        var all = true;
        for (var row = u0; row < u1; row = row + 1u) {
            for (var col = t0; col < t1; col = col + 1u) {
                let far = hz_chain[level_info.x + row * level_info.y + col];
                if (!(far > 0.0) || nearest <= far + bias) {
                    all = false;
                    break;
                }
            }
            if (!all) {
                break;
            }
        }
        if (all) {
            return true;
        }
    }
    return false;
}

@compute @workgroup_size(64)
fn hz_test(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.y * test.counts.z + gid.x;
    if (index >= test.counts.x) {
        return;
    }
    let candidate = hz_candidates[index];
    let visible = !hz_occluded(candidate.min, candidate.max);
    let base = index * 5u;
    hz_args[base + 0u] = select(0u, candidate.args.x, visible);
    hz_args[base + 1u] = select(0u, candidate.args.w, visible);
    hz_args[base + 2u] = candidate.args.y;
    hz_args[base + 3u] = candidate.args.z;
    hz_args[base + 4u] = 0u;
}
"#;

/// The feed's reduction resources, built once per batch entry.
struct HizPipeline {
    layout: wgpu::BindGroupLayout,
    tile_reduce: wgpu::ComputePipeline,
    coarsen: wgpu::ComputePipeline,
    /// Per-level parameters, `uniform_stride` bytes apart.
    params: wgpu::Buffer,
    /// Uniform dynamic-offset stride: the device's alignment rounded up from
    /// one [`HizLevel`].
    uniform_stride: u32,
    /// Layout of the phase-two test's bind group (its own pipeline).
    test_layout: wgpu::BindGroupLayout,
    hz_test: wgpu::ComputePipeline,
    /// Per-level table the test reads: offset, columns, rows, spare.
    levels: wgpu::Buffer,
    /// The test's frame parameters.
    test_params: wgpu::Buffer,
}

impl HizPipeline {
    fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("repame_view3d_hiz_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        // `textureLoad` only: no sampler, no filtering.
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<HizLevel>() as u64
                        ),
                    },
                    count: None,
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("repame_view3d_hiz"),
            source: wgpu::ShaderSource::Wgsl(HIZ_WGSL.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("repame_view3d_hiz_pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let make = |label: &'static str, entry: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let tile_reduce = make("repame_view3d_hiz_tile_reduce", "tile_reduce");
        let coarsen = make("repame_view3d_hiz_coarsen", "coarsen");
        let test_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("repame_view3d_hiz_test_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<HizTestParams>() as u64
                        ),
                    },
                    count: None,
                },
            ],
        });
        let test_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("repame_view3d_hiz_test"),
            source: wgpu::ShaderSource::Wgsl(HIZ_TEST_WGSL.into()),
        });
        let test_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("repame_view3d_hiz_test_pl"),
            bind_group_layouts: &[Some(&test_layout)],
            immediate_size: 0,
        });
        let hz_test = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("repame_view3d_hiz_test"),
            layout: Some(&test_pipeline_layout),
            module: &test_module,
            entry_point: Some("hz_test"),
            compilation_options: Default::default(),
            cache: None,
        });
        let levels = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("repame_view3d_hiz_levels"),
            size: MAX_HIZ_LEVELS as u64 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let test_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("repame_view3d_hiz_test_params"),
            size: std::mem::size_of::<HizTestParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_stride = (std::mem::size_of::<HizLevel>() as u32)
            .next_multiple_of(device.limits().min_uniform_buffer_offset_alignment.max(1));
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("repame_view3d_hiz_params"),
            size: MAX_HIZ_LEVELS as u64 * uniform_stride as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            layout,
            tile_reduce,
            coarsen,
            params,
            uniform_stride,
            test_layout,
            hz_test,
            levels,
            test_params,
        }
    }
}

/// Cross-frame state of the viewport-driven occlusion feed for one batch id.
///
/// One readback is in flight at a time. The reduction passes are encoded into
/// the frame's own encoder, which the renderer submits with that frame, and
/// the host map is only *scheduled* for that submission, so the data is
/// never available before the next frame: [`take`](Self::take) polls without
/// waiting and a readback that has not landed yet is left in flight, costing
/// that frame its occlusion culling rather than pacing the frame around the
/// GPU.
///
/// The state lives as long as the batch's GPU entry, so a viewport that
/// stops painting resumes on the pyramid it last captured (the one-frame
/// delay spans the gap, it is not aged by wall-clock time) and reads fresh
/// depth again one painted frame later.
#[derive(Default)]
struct OcclusionFeed {
    pending: Option<PendingDepthReadback>,
    /// Staging buffers parked for reuse, matched by size and capped: one
    /// readback is in flight at a time, so a single slot covers the cycle.
    spare: Vec<wgpu::Buffer>,
    /// The reduction and phase-two pipelines, created on first use.
    hiz: Option<HizPipeline>,
    /// This frame's chain (level maxima, packed), rewritten every frame the
    /// feed runs, so phase two never waits on a readback.
    chain: Option<wgpu::Buffer>,
    /// Levels the last [`OcclusionFeed::encode_chain`] wrote.
    chain_levels: u32,
    /// The mid-frame phase two already built this frame's chain; the
    /// end-of-frame readback then only copies it.
    chain_done: bool,
}

impl OcclusionFeed {
    /// Readback buffers kept parked for reuse. One readback is in flight at
    /// a time, so a single slot covers the take/reuse cycle; the pool is
    /// capped so a frame that stops feeding does not hold buffers it can
    /// never hand out.
    const MAX_SPARE: usize = 1;

    /// Encodes this frame's chain: per-level parameters, the level
    /// dispatches, and the packed level table the phase-two test reads.
    /// `false` when the chain would need more levels than
    /// [`MAX_HIZ_LEVELS`] (unreachable for real targets; a short chain must
    /// never be read back as a whole one).
    #[allow(clippy::too_many_arguments)] // device, encoder, target, feed settings
    fn encode_chain(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        width: u32,
        height: u32,
        tile: u32,
    ) -> bool {
        let tile = tile.clamp(1, super::occlusion::MAX_TILE);
        let cols = width.div_ceil(tile).max(1);
        let rows = height.div_ceil(tile).max(1);
        let dims = super::occlusion::coarsened_dims(cols, rows);
        if dims.len() > MAX_HIZ_LEVELS {
            return false;
        }
        // Offsets into the packed chain, level 0 first.
        let mut offsets = Vec::with_capacity(dims.len());
        let mut total = 0u32;
        for &(cols, rows) in &dims {
            offsets.push(total);
            total += cols * rows;
        }
        let size = total as u64 * std::mem::size_of::<f32>() as u64;
        if self.chain.as_ref().is_none_or(|chain| chain.size() != size) {
            self.chain = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("repame_view3d_occlusion_levels"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
        }
        let (layout, tile_reduce, coarsen, params, uniform_stride, levels) = {
            let hiz = self.hiz.get_or_insert_with(|| HizPipeline::new(device));
            (
                hiz.layout.clone(),
                hiz.tile_reduce.clone(),
                hiz.coarsen.clone(),
                hiz.params.clone(),
                hiz.uniform_stride,
                hiz.levels.clone(),
            )
        };
        let groups: Vec<(u32, u32)> = dims
            .iter()
            .map(|&(cols, rows)| dispatch_groups(cols * rows))
            .collect();
        let mut packed_levels = Vec::with_capacity(dims.len());
        for (index, &(cols, rows)) in dims.iter().enumerate() {
            let (src_offset, src_cols, src_rows) = if index == 0 {
                (0, 0, 0)
            } else {
                let (src_cols, src_rows) = dims[index - 1];
                (offsets[index - 1], src_cols, src_rows)
            };
            queue.write_buffer(
                &params,
                index as u64 * uniform_stride as u64,
                bytemuck::bytes_of(&HizLevel {
                    dst_offset: offsets[index],
                    dst_cols: cols,
                    dst_rows: rows,
                    src_offset,
                    src_cols,
                    src_rows,
                    width,
                    height,
                    tile,
                    row_stride: groups[index].0 * HIZ_WORKGROUP,
                    _pad: [0; 2],
                }),
            );
            packed_levels.push([offsets[index], cols, rows, 0]);
        }
        queue.write_buffer(&levels, 0, bytemuck::cast_slice(&packed_levels));
        let chain = self.chain.as_ref().expect("chain ensured above");
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("repame_view3d_hiz_bg"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(target),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: chain.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    // A window of one level, so the dynamic offset can move
                    // over the rest of the buffer: binding everything would
                    // leave `maximum_dynamic_offset` at zero.
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params,
                        offset: 0,
                        size: wgpu::BufferSize::new(uniform_stride as u64),
                    }),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("repame_view3d_hiz"),
                timestamp_writes: None,
            });
            // In order: each level reads what the dispatch before it wrote.
            for (index, &(groups_x, groups_y)) in groups.iter().enumerate() {
                if index == 0 {
                    pass.set_pipeline(&tile_reduce);
                } else {
                    pass.set_pipeline(&coarsen);
                }
                pass.set_bind_group(0, &bind, &[index as u32 * uniform_stride]);
                pass.dispatch_workgroups(groups_x, groups_y, 1);
            }
        }
        self.chain_levels = dims.len() as u32;
        self.chain_done = true;
        true
    }

    /// Requests this frame's feed: encodes the chain unless the mid-frame
    /// phase two already did, then schedules a readback of it for the next
    /// frame's cull. `true` when the chain is in place (a readback already in
    /// flight only skips this frame's copy); `false` when the chain cannot be
    /// built at all.
    ///
    /// The camera is captured with the chain so the cull projection and the
    /// depth it tests against cannot drift apart.
    #[allow(clippy::too_many_arguments)] // device, encoder, target, camera, feed settings
    fn request(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        width: u32,
        height: u32,
        tile: u32,
        bias: f32,
        view_proj: Mat4,
        cam_pos: [f32; 3],
        cam_forward: [f32; 3],
    ) -> bool {
        if !std::mem::replace(&mut self.chain_done, false)
            && !self.encode_chain(device, queue, encoder, target, width, height, tile)
        {
            return false;
        }
        // One readback in flight; a chain that landed is kept until the next
        // poll takes it.
        if self.pending.is_some() {
            return true;
        }
        let chain = self.chain.as_ref().expect("chain encoded above");
        let size = chain.size();
        let staging = match self.spare.iter().position(|buffer| buffer.size() == size) {
            Some(index) => self.spare.swap_remove(index),
            None => device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("repame_view3d_occlusion_readback"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
        };
        // The chain is copied into the staging buffer the map targets:
        // `MAP_READ` may not be combined with `STORAGE` on this stack, and
        // the copy is on-device.
        encoder.copy_buffer_to_buffer(chain, 0, &staging, 0, size);
        let map_state = Arc::new(AtomicU8::new(READBACK_IN_FLIGHT));
        let callback = map_state.clone();
        // Deferred to submission: mapping now would resolve against the
        // buffer's existing (empty) uses and hand back unwritten memory.
        encoder.map_buffer_on_submit(&staging, wgpu::MapMode::Read, 0..size, move |result| {
            callback.store(
                if result.is_ok() {
                    READBACK_MAPPED
                } else {
                    READBACK_FAILED
                },
                Ordering::Release,
            );
        });
        self.pending = Some(PendingDepthReadback {
            staging,
            width,
            height,
            view_proj,
            cam_pos,
            cam_forward,
            tile,
            bias,
            map_state,
        });
        true
    }

    /// Encodes the phase-two test: one invocation per candidate, writing its
    /// indirect draw args (a zeroed index count hides a group the chain
    /// still occludes). Does nothing without a chain to test against.
    #[allow(clippy::too_many_arguments)] // device, camera, feed settings, the frame's buffers
    fn encode_test(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        camera: &CameraUniform,
        tile: u32,
        bias: f32,
        width: u32,
        height: u32,
        candidates: &wgpu::Buffer,
        candidate_count: u32,
        args: &wgpu::Buffer,
    ) {
        let (Some(chain), Some(hiz)) = (self.chain.as_ref(), self.hiz.as_ref()) else {
            return;
        };
        if candidate_count == 0 || self.chain_levels == 0 {
            return;
        }
        let (groups_x, groups_y) = dispatch_groups(candidate_count);
        queue.write_buffer(
            &hiz.test_params,
            0,
            bytemuck::bytes_of(&HizTestParams {
                view_proj: camera.view_proj,
                cam_pos: [camera.cam_pos[0], camera.cam_pos[1], camera.cam_pos[2], 0.0],
                cam_forward: [
                    camera.cam_forward[0],
                    camera.cam_forward[1],
                    camera.cam_forward[2],
                    0.0,
                ],
                hz: [
                    tile.clamp(1, super::occlusion::MAX_TILE) as f32,
                    bias.max(0.0),
                    width as f32,
                    height as f32,
                ],
                counts: [
                    candidate_count,
                    self.chain_levels,
                    groups_x * HIZ_WORKGROUP,
                    0,
                ],
            }),
        );
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("repame_view3d_hiz_test_bg"),
            layout: &hiz.test_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: chain.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: hiz.levels.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: candidates.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: args.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: hiz.test_params.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("repame_view3d_hiz_test"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&hiz.hz_test);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }

    /// Takes the culling state when the in-flight readback has landed, else
    /// `None`. Never waits: `Poll` only resolves work that already finished,
    /// so a readback still on the GPU is left in flight and the frame culls
    /// nothing.
    fn take(
        &mut self,
        device: &wgpu::Device,
        id: &str,
    ) -> Option<super::occlusion::OcclusionCulling> {
        if self.pending.is_none() {
            return None;
        }
        let _ = device.poll(wgpu::PollType::Poll);
        let state = self
            .pending
            .as_ref()
            .map(|pending| pending.map_state.load(Ordering::Acquire))?;
        match state {
            READBACK_MAPPED => {}
            READBACK_FAILED => {
                self.pending = None;
                log::warn!(
                    "scene_batch[{id}]: occlusion readback map failed; feed restarts next frame"
                );
                return None;
            }
            _ => return None,
        }
        let pending = self.pending.take()?;
        let pyramid = match pending.staging.slice(..).get_mapped_range() {
            Ok(bytes) => super::occlusion::DepthPyramid::from_packed_levels(
                &bytes,
                pending.width,
                pending.height,
                pending.tile,
            ),
            Err(_) => None,
        };
        pending.staging.unmap();
        self.park(pending.staging);
        let Some(pyramid) = pyramid else {
            log::warn!(
                "scene_batch[{id}]: occlusion chain of {}x{} did not build a pyramid",
                pending.width,
                pending.height
            );
            return None;
        };
        Some(super::occlusion::OcclusionCulling {
            pyramid,
            view_proj: pending.view_proj,
            cam_pos: pending.cam_pos,
            cam_forward: pending.cam_forward,
            bias: pending.bias,
        })
    }

    /// Stops the feed, dropping any readback still in flight (its reduction
    /// is wasted: the buffer cannot be reused while its map is pending). Used
    /// when a frame declares no depth pass, and when the caller stages its
    /// own occlusion state.
    fn retire(&mut self, device: &wgpu::Device) {
        self.chain_done = false;
        if self.pending.is_none() {
            return;
        }
        let _ = device.poll(wgpu::PollType::Poll);
        if let Some(pending) = self.pending.take()
            && pending.map_state.load(Ordering::Acquire) == READBACK_MAPPED
        {
            pending.staging.unmap();
            self.park(pending.staging);
        }
    }

    /// Parks a staging buffer for reuse, keeping the pool one size and
    /// capped (a resize drops the old size on the next park).
    fn park(&mut self, staging: wgpu::Buffer) {
        let size = staging.size();
        self.spare.retain(|spare| spare.size() == size);
        if self.spare.len() < Self::MAX_SPARE {
            self.spare.push(staging);
        }
    }
}

/// The declaration of one pass for the frame, as the batch records it.
#[derive(Clone, Debug, PartialEq)]
pub struct PassSpec {
    pub target: super::pass::RenderTarget,
    pub kind: super::pass::PassKind,
    pub source: Option<String>,
    /// 1-based position in the declared list, matching the index a game
    /// passes to [`SceneBatch::push_group_for_pass`].
    pub index: u32,
    /// Composites this target over the scene when the frame is painted.
    pub presents: bool,
    /// Tint and opacity the presenting composite applies.
    pub overlay: [f32; 4],
    /// Screen-space effect a fullscreen pass runs (`None` = tinted blit).
    pub post: Option<PostStage>,
    /// Second target a post stage samples.
    pub aux: Option<String>,
    /// Draws over the target's contents instead of clearing them.
    pub load: bool,
}

/// Uniform a post-process pass binds: the stage parameters, the source
/// texel size, the effect mode, and the camera/light state the ray-marched
/// fog stage reads.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PostUniform {
    params: [f32; 4],
    params2: [f32; 4],
    /// `(1/src_w, 1/src_h, src_w, src_h)`.
    texel: [f32; 4],
    /// `(mode, 0, 0, 0)`.
    stage: [f32; 4],
    /// `(cam_pos.xyz, tan_half_fov)`.
    cam_pos: [f32; 4],
    /// `(cam_forward.xyz, aspect)`.
    cam_forward: [f32; 4],
    /// `(cam_right.xyz, 0)`.
    cam_right: [f32; 4],
    /// `(light_dir.xyz, 0)`, unit.
    light_dir: [f32; 4],
    /// `(light_color.rgb, diffuse)`.
    light_color: [f32; 4],
    /// `(fog_color.rgb, 0)`.
    fog_color: [f32; 4],
}

const _: () = assert!(size_of::<PostUniform>() == 160);

/// GPU state for the post-process stages: one shader, one bind layout, and a
/// pipeline per output format (post targets differ in precision).
struct PostState {
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    /// 1x1 stand-in bound at the aux slot when a stage has no second input.
    dummy_view: wgpu::TextureView,
    /// `_texture` keeps `dummy_view` alive; wgpu views do not own their texture.
    _dummy_texture: wgpu::Texture,
    /// Bind groups for this frame's post passes, keyed by [`PassSpec::index`].
    binds: HashMap<u32, wgpu::BindGroup>,
    /// Parameter buffers written this frame; the bind groups reference them, so
    /// they must outlive the encode.
    params: Vec<wgpu::Buffer>,
}

/// Maps a declared target format onto the GPU format it renders as.
fn pass_format(format: super::pass::RenderTargetFormat) -> wgpu::TextureFormat {
    match format {
        super::pass::RenderTargetFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
        super::pass::RenderTargetFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8UnormSrgb,
        super::pass::RenderTargetFormat::Bgra8Unorm => wgpu::TextureFormat::Bgra8Unorm,
        super::pass::RenderTargetFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
        super::pass::RenderTargetFormat::R32Float => wgpu::TextureFormat::R32Float,
        super::pass::RenderTargetFormat::Depth32Float => wgpu::TextureFormat::Depth32Float,
    }
}

/// Fullscreen post-process pipeline for one target format. Post passes always
/// replace (`REPLACE`), so the presenting composite stays the only blended
/// path.
fn make_post_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    layout: &wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("repame_view3d_post_pl"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("repame_view3d_post"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// GPU state for one lowered material program.
struct MaterialPipelines {
    key: u64,
    depth: wgpu::RenderPipeline,
    flat: wgpu::RenderPipeline,
    transparent: wgpu::RenderPipeline,
    transparent_flat: wgpu::RenderPipeline,
    buffer: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

/// One particle system's GPU buffers plus the bind groups the compute and
/// billboard passes use. CPU-side fields are the deterministic simulation
/// clock; they live here (not on the per-frame [`SceneBatch`]) so a rebuilt
/// batch keeps a running emitter running. The spawn cursor itself is
/// GPU-owned (sub-emitter requests advance it), so it is not mirrored here.
struct ParticleSystemGpu {
    capacity: u32,
    seed: u32,
    /// 1-based pass whose draw list the billboards join (0 = scene target).
    pass: u32,
    /// True when the declared texture page is out of range: draw flat.
    flat: bool,
    params: wgpu::Buffer,
    sim: wgpu::Buffer,
    particles: wgpu::Buffer,
    requests: wgpu::Buffer,
    draw_params: wgpu::Buffer,
    sim_bind: wgpu::BindGroup,
    draw_bind: wgpu::BindGroup,
    /// Fractional spawns carried into the next frame.
    carry: f32,
    /// Frames simulated by this emitter.
    frame: u32,
}

/// GPU state shared by a frame's particle systems: the pipelines, the
/// collision world, and the per-system buffer sets.
struct ParticleGpu {
    /// Advances the spawn cursor (one invocation).
    compute_prepare: wgpu::ComputePipeline,
    /// The per-slot spawn/integrate/collide/sub-emit dispatch.
    compute_simulate: wgpu::ComputePipeline,
    billboard_pipeline: wgpu::RenderPipeline,
    sim_layout: wgpu::BindGroupLayout,
    draw_layout: wgpu::BindGroupLayout,
    colliders: wgpu::Buffer,
    /// 1x1x1 stand-in bound when a frame carries no voxel field.
    _dummy_voxels: wgpu::Texture,
    dummy_voxels_view: wgpu::TextureView,
    voxels: wgpu::Texture,
    /// Identity of the uploaded voxel field: (Arc address, cell count).
    voxels_key: Option<(usize, usize)>,
    /// Dispatch order resolved by [`SceneBatch::set_particles`].
    order: Vec<u32>,
    /// One slot per staged system, index-aligned with the frame's list.
    systems: Vec<Option<ParticleSystemGpu>>,
}

impl ParticleGpu {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat, cam_layout: &wgpu::BindGroupLayout, tex_layout: &wgpu::BindGroupLayout) -> Self {
        let sim_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("repame_view3d_particles"),
            source: wgpu::ShaderSource::Wgsl(super::particles::PARTICLE_COMPUTE_WGSL.into()),
        });
        let storage = |binding: u32, read_only: bool, visibility: wgpu::ShaderStages| {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        };
        let sim_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("repame_view3d_particle_sim_bgl"),
            entries: &[
                storage(0, true, wgpu::ShaderStages::COMPUTE),
                storage(1, false, wgpu::ShaderStages::COMPUTE),
                storage(2, false, wgpu::ShaderStages::COMPUTE),
                storage(3, false, wgpu::ShaderStages::COMPUTE),
                storage(4, false, wgpu::ShaderStages::COMPUTE),
                storage(5, true, wgpu::ShaderStages::COMPUTE),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let draw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("repame_view3d_particle_draw_bgl"),
            entries: &[
                storage(0, true, wgpu::ShaderStages::VERTEX),
                storage(1, true, wgpu::ShaderStages::VERTEX),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let sim_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("repame_view3d_particle_sim_pl"),
            bind_group_layouts: &[Some(&sim_layout)],
            immediate_size: 0,
        });
        let make_compute = |label: &'static str, entry: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&sim_pipeline_layout),
                module: &sim_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let compute_prepare = make_compute("repame_view3d_particle_prepare", "prepare");
        let compute_simulate = make_compute("repame_view3d_particle_simulate", "simulate");
        let billboard_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("repame_view3d_particle_billboard"),
            source: wgpu::ShaderSource::Wgsl(super::particles::PARTICLE_BILLBOARD_WGSL.into()),
        });
        let render_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("repame_view3d_particle_render_pl"),
            bind_group_layouts: &[Some(cam_layout), Some(tex_layout), Some(&draw_layout)],
            immediate_size: 0,
        });
        let billboard_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("repame_view3d_particle_billboard"),
                layout: Some(&render_layout),
                vertex: wgpu::VertexState {
                    module: &billboard_module,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &billboard_module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24PlusStencil8,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: SCENE_TARGET_SAMPLE_COUNT,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            });
        let colliders = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("repame_view3d_particle_colliders"),
            size: size_of::<GpuColliders>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dummy_voxels = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("repame_view3d_particle_voxels_dummy"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_voxels_view = dummy_voxels.create_view(&wgpu::TextureViewDescriptor::default());
        let voxels = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("repame_view3d_particle_voxels"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        Self {
            compute_prepare,
            compute_simulate,
            billboard_pipeline,
            sim_layout,
            draw_layout,
            colliders,
            _dummy_voxels: dummy_voxels,
            dummy_voxels_view,
            voxels,
            voxels_key: None,
            order: Vec::new(),
            systems: Vec::new(),
        }
    }
}

/// One reflection probe's cube render target plus the settings it was
/// captured with (a changed setting re-captures immediately).
struct ProbeState {
    settings: ProbeSettings,
    cube: wgpu::Texture,
    view: wgpu::TextureView,
    depth: wgpu::Texture,
    edge: u32,
    /// One capture camera per cube face (buffers written before submission
    /// apply at its start, so faces must not share one).
    cam_binds: Vec<(wgpu::Buffer, wgpu::BindGroup)>,
    /// Captures this frame (interval elapsed or a setting changed).
    capture: bool,
}

/// GPU state for the frame's reflection probes: the cube targets, the
/// capture camera, and two group-2 bind groups (the scene pass samples the
/// real cubes, the capture passes bind stand-ins so a cube is never both an
/// attachment and a texture in one pass).
struct ProbeGpu {
    probes: Vec<ProbeState>,
    /// Group 2 with stand-ins at the probe slots, used by the capture passes.
    capture_bind: wgpu::BindGroup,
    frames: u64,
}

/// One material program registered on the batch, lowered to WGSL and packed.
#[derive(Clone, Debug, PartialEq)]
pub struct MaterialEntry {
    pub model: super::material::ShadingModel,
    pub cache_key: u64,
    pub uniform: super::shading::ShadingUniform,
}

/// Texture sampling for batch layers (same shape as `repame-view2d`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SceneFilter {
    /// Pixel-art crisp (nt default).
    #[default]
    Nearest,
    Linear,
}

/// Batch construction parameters. Layers map 1:1 onto array layers; one
/// page per mesh group, same as sprite atlas pages.
#[derive(Clone, Copy, Debug)]
pub struct BatchDesc {
    /// Square layer edge in pixels.
    pub layer_size: u32,
    /// Array layer count (= max pages).
    pub layers: u32,
    pub filter: SceneFilter,
    /// Mip levels each page is created with (`1` = the historical single
    /// level). Levels above 0 are written by [`SceneBatch::upload_mip`]; the
    /// scene shaders sample each page's finest resident level
    /// ([`SceneBatch::set_page_level`]), level 0 for a fully resident page.
    /// Clamped to what `layer_size` allows by [`BatchDesc::mip_levels`].
    pub mips: u32,
}

impl BatchDesc {
    /// Mip levels the page array actually gets: `mips` clamped to
    /// `layer_size`'s chain (a 1x1 page has one level).
    pub fn mip_levels(&self) -> u32 {
        let max = 32 - self.layer_size.max(1).leading_zeros();
        self.mips.clamp(1, max)
    }
}

/// Sampler clamp for the highest LOD a batch's page array can hold: the last
/// level [`BatchDesc::mip_levels`] creates. A fixed `1.0` capped every
/// implicitly computed LOD (material programs sample through `textureSample`)
/// at level 1, so streamed levels from 2 up were unreachable by any in-engine
/// shader.
fn sampler_lod_max_clamp(mips: u32) -> f32 {
    mips.saturating_sub(1) as f32
}

/// `vec4` slots the shader's per-page resident-level array holds
/// (`array<vec4<f32>, 64>` in [`SHADER_PREFIX`] and the particle billboard).
/// Must stay in lockstep with both WGSL declarations.
const RESIDENT_PAGE_SLOTS: u32 = 64;

/// Pages the resident-level array covers. Material stage pages are `u8`
/// (`<= 255`), and the legacy and particle paths validate
/// `page < BatchDesc::layers`, so every page a draw can name fits. A page
/// past the cap cannot be published ([`SceneBatch::set_page_level`] rejects
/// it) and samples level 0, the fully resident default.
pub const MAX_RESIDENT_PAGES: u32 = RESIDENT_PAGE_SLOTS * 4;

impl Default for BatchDesc {
    fn default() -> Self {
        Self {
            layer_size: 1024,
            layers: 4,
            filter: SceneFilter::Nearest,
            mips: 1,
        }
    }
}

/// One pending mip-level upload: blit `texels` (tight `width`*`height`*4
/// bytes) into `page`'s mip `level`.
///
/// Same shape as [`SceneUpload`] one level down, and the same contract: the
/// game queues it on the frame thread and it is written in
/// [`SceneBatch::upload_all`] on the render thread. Streamed textures fill
/// these from a residency drain
/// ([`TextureResidency::take_uploads`](crate::texture_stream::TextureResidency::take_uploads)).
#[derive(Clone, Debug)]
pub struct MipUpload {
    pub page: u32,
    /// Mip level, `< BatchDesc::mip_levels`.
    pub level: u32,
    /// Level extent in texels; must equal the page's edge at `level`.
    pub width: u32,
    pub height: u32,
    pub texels: Vec<u8>,
}

/// One pending texture upload: blit `rgba` (tight `w`*`h`*4 bytes) into
/// `page` at (`x`, `y`). Games fill these from their image source (atlas
/// drain, decoded PNG, procedural texel) once per frame; mismatches are
/// dropped with a warning at upload time.
#[derive(Clone, Debug)]
pub struct SceneUpload {
    pub page: u32,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    pub rgba: Vec<u8>,
}

/// Drop zero-area (and NaN) triangles from an index list, preserving the
/// survivors in submission order. Mirrors the picker's
/// [`ray_triangle`] degeneracy test (area^2 <=
/// 1e-12 misses there, so it must not draw here either, content and picks
/// stay glued). Returns an empty vec when fewer than 3 indices remain.
/// Non-multiple-of-3 tails are ignored, matching `pick_ray`'s chunking.
///
/// `is_finite` rides alongside the area test (not folded into one
/// comparison): NaN fails every ordering, so name the check explicitly -
/// a bare `area2 <= 1e-12` reads like it culls NaN but actually lets it
/// through.
fn cull_degenerate(positions: &[[f32; 3]], indices: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(indices.len());
    let (chunks, _) = indices.as_chunks::<3>();
    for tri in chunks {
        let (ia, ib, ic) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let (Some(a), Some(b), Some(c)) = (positions.get(ia), positions.get(ib), positions.get(ic))
        else {
            continue; // caller validates range; double guard
        };
        let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let cross = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let area2 = cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2];
        if !area2.is_finite() || area2 <= 1e-12 {
            continue;
        }
        out.extend_from_slice(tri);
    }
    out
}

/// One GPU-skinned draw awaiting [`SceneBatch::finish`]: bind-pose
/// attributes plus the sampled palette. Flattened into the same vertex
/// buffers as static groups (bind positions/normals, joints/weights per
/// vertex) with the palette appended to the batch palette store; the
/// range records the palette offset so the shader indexes the right
/// joints.
struct SkinnedPending {
    positions: Vec<[f32; 3]>,
    colors: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    joints: Vec<[u16; 4]>,
    weights: Vec<[f32; 4]>,
    palette: Vec<[[f32; 4]; 4]>,
    texture_page: u32,
    material: super::mesh::Material,
    /// World-space AABB center of the bind pose (transparent back-to-front
    /// sort; opaque draws ignore it).
    center: Option<[f32; 3]>,
    indices: Vec<u32>,
    transparent: bool,
    alpha: f32,
    alpha_cutoff: f32,
    depth_test: bool,
    /// Index into the frame's pass list; 0 = the shared scene target.
    pass: u32,
    /// Phase-one reject the same-frame phase two keeps: its range is drawn
    /// through indirect args, not dropped.
    hz_culled: bool,
}

/// One validated group awaiting [`SceneBatch::finish`].
struct Pending {
    positions: Vec<[f32; 3]>,
    colors: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    texture_page: u32,
    material: super::mesh::Material,
    /// World-space AABB center (computed at push time) for back-to-front
    /// transparent sorting. Opaque groups ignore it; `None` (degenerate
    /// bounds - post-validation this is None only for empty input) sorts
    /// nearest.
    center: Option<[f32; 3]>,
    /// World-space AABB (min, max), the occlusion test's box. Same `None`
    /// contract as `center`.
    bounds: Option<([f32; 3], [f32; 3])>,
    indices: Vec<u32>,
    transparent: bool,
    alpha: f32,
    alpha_cutoff: f32,
    depth_test: bool,
    /// Material program slot stamped into every vertex of the group; 0 is
    /// the legacy PBR-lite path.
    material_slot: u32,
    /// Index into the frame's pass list; 0 = the shared scene target.
    pass: u32,
    /// Phase-one reject the same-frame phase two keeps: its range is drawn
    /// through indirect args, not dropped.
    hz_culled: bool,
}

/// One draw call's layer: index range + whether depth testing applies.
/// Groups flatten opaque-first (depth-tested, then flat overlays), then
/// transparent back-to-front, so tested geometry occludes and untested
/// overlays (ground decals, gizmos) always draw. Transparent ranges carry
/// the blend pass; a `depth_test = false` transparent group composes
/// without writing depth (HUD ghosts over the scene).
#[derive(Clone, Copy)]
struct DrawRange {
    index_start: u32,
    index_end: u32,
    depth_test: bool,
    transparent: bool,
    /// Material program that drew this range; 0 = legacy path.
    material: u32,
    /// Index into the frame's pass list; 0 = the shared scene target.
    pass: u32,
    /// When the same-frame phase two keeps a phase-one reject: its slot in
    /// the occlusion args buffer, so the scene pass draws it through
    /// `draw_indexed_indirect`. `None` draws directly.
    hz_culled: Option<u32>,
}

/// One phase-one reject the same-frame phase two hands to the GPU test:
/// its world AABB and the indexed draw the test either keeps or zeroes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HizCandidate {
    min: [f32; 3],
    _pad0: f32,
    max: [f32; 3],
    _pad1: f32,
    /// Index count, first index, base vertex, instance count.
    args: [u32; 4],
}

/// Must stay the WGSL `Candidate`: 48 bytes.
const _: () = assert!(std::mem::size_of::<HizCandidate>() == 48);

/// Per-frame snapshot batch. `Send + Sync` so it can cross into the
/// compositor thread via [`repose_render_wgpu::Callback`].
///
/// Each `id` owns its pipelines + texture in `CallbackResources` (the
/// `repame-view2d` `SpriteBatch` pattern): viewports coexist. Rebuilding
/// on format/sample/desc change drops texture contents (logged), the
/// game re-uploads from its next frame's [`SceneUpload`]s, same as the
/// sprite batch's atlas contract.
pub struct SceneBatch {
    id: String,
    desc: BatchDesc,
    camera: CameraUniform,
    /// Camera world position for transparent back-to-front sorting
    /// ([`SceneBatch::set_camera_pos`]). Defaults to the origin (stable,
    /// documented) until the viewport feeds the real eye per frame.
    camera_pos: [f32; 3],
    /// View-projection matrix for frustum culling ([`SceneBatch::finish`]
    /// extracts the six planes from it). Set alongside the uniform matrix
    /// by [`SceneBatch::set_camera`]; identity disables culling (the
    /// unit tests' default, they assert content preservation directly).
    view_proj: Mat4,
    /// Groups culled by the last [`SceneBatch::finish`] (frustum only  -
    /// malformed/degenerate drops are separate, counted nowhere by
    /// design). HUD/stats readout, reset per `finish`.
    culled: usize,
    pending: Vec<Pending>,
    uploads: Vec<SceneUpload>,
    /// Streamed mip levels queued by [`SceneBatch::upload_mip`], applied in the
    /// same pass as `uploads`.
    mip_uploads: Vec<MipUpload>,
    /// Finest resident mip level per page, published with
    /// [`SceneBatch::set_page_level`]: `-1` = nothing resident, `0` = the
    /// historical fully resident page. Staged per frame into the shader's
    /// `page_levels` array by [`SceneBatch::upload_all`]. Survives
    /// [`SceneBatch::clear`]: residency spans frames, unlike the draw state.
    page_levels: Vec<f32>,
    verts: Vec<Vert>,
    indices: Vec<u32>,
    ranges: Vec<DrawRange>,
    /// GPU-skinned draws submitted this frame (bind geometry + palette).
    /// Flattened after static groups in `finish`: one extra vertex block
    /// plus one `DrawRange` each (skinned ranges sort with the opaque
    /// pass; transparency on skinned draws follows the same back-to-front
    /// path by AABB center of the *bind* pose).
    ///
    /// Only staged when [`SceneBatch::set_skinning`] is supported: on
    /// [`GpuSkinning::Unsupported`] draws drop here (logged once) for the
    /// caller to CPU-pose.
    skinned: Vec<SkinnedPending>,
    /// Joint palette store for the frame: every skinned draw appends its
    /// (padded) palette here; ranges record the base offset. One palette
    /// upload per frame, sized to the frame's joint count.
    palette: Vec<[[f32; 4]; 4]>,
    /// Cascade desc staged by [`set_rig`](SceneBatch::set_rig) (`None` =
    /// legacy single map). Sizes the cascade array texture in
    /// `ensure_resources` and drives the per-slice depth passes.
    cascade_desc: Option<super::shadow::CascadeDesc>,
    /// Point-light index (into the staged rig order) casting cube
    /// shadows. `None` = no cube pass.
    point_caster: Option<usize>,
    /// Staged point lights in rig order (uniform mirrors this; the cube
    /// pass reads the caster entry for position/range/size).
    staged_points: Vec<super::shadow::PointLight>,
    /// Counters for the frame in flight, read after `finish`.
    stats: super::stats::FrameStats,
    /// Pages replaced at runtime by the game's override table; consulted
    /// when uploads are applied, after the queue-time range check.
    overrides: super::texfmt::TextureOverrides,
    /// Material programs registered for this frame; index 0 is the legacy
    /// PBR-lite path, so slot `n` reads `materials[n - 1]`.
    materials: Vec<MaterialEntry>,
    /// Passes declared for this frame, in execution order.
    passes: Vec<PassSpec>,
    /// Particle systems staged with [`SceneBatch::set_particles`], plus the
    /// sub-emitter dispatch order resolved when they were staged.
    particles: Option<ParticleFrame>,
    particle_order: Vec<u32>,
    /// Reflection probes staged with [`SceneBatch::set_probes`].
    probes: Vec<ProbeSettings>,
    /// Occlusion test staged with [`SceneBatch::set_occlusion`].
    occlusion: Option<super::occlusion::OcclusionCulling>,
    /// Feed settings staged with [`SceneBatch::set_occlusion_feed`].
    occlusion_feed: Option<crate::OcclusionFeedDesc>,
    /// [`SceneBatch::finish`] decided this frame re-tests its rejects in
    /// phase two.
    occlusion_phase_two: bool,
    /// This frame's phase-one rejects, when the feed's phase two keeps them.
    occlusion_candidates: Vec<HizCandidate>,
    /// Groups culled by the last [`SceneBatch::finish`] occlusion phase
    /// (hierarchical-Z only; the frustum phase counts into `culled`).
    occluded: usize,
    /// Post-process settings staged with [`SceneBatch::set_post`]; the depth
    /// pass reads `previous_view_proj` from it for camera motion vectors.
    post: Option<PostProcess>,
    /// Device GPU-skinning capability staged with
    /// [`SceneBatch::set_skinning`]. Drives the shader/layout variant and
    /// the push-time drop.
    skinning: GpuSkinning,
}

/// Dispatch order for a frame's systems: a system that sub-emits into
/// another must run first, so the target's request ring is filled within the
/// same frame. A cycle (only possible with a link that was not dropped) is
/// broken at the edge that closes it, with a warning.
fn particle_dispatch_order(frame: &ParticleFrame) -> Vec<u32> {
    let count = frame.systems.len();
    let targets: Vec<Option<u32>> = frame
        .systems
        .iter()
        .map(|system| match system.sub {
            Some(sub) if !matches!(sub.mode, super::particles::SubEmitterMode::Off) => {
                Some(sub.target as u32)
            }
            _ => None,
        })
        .collect();
    let mut parented = vec![false; count];
    for target in targets.iter().flatten() {
        let index = *target as usize;
        if index < count {
            parented[index] = true;
        }
    }
    let mut state = vec![0u8; count];
    let mut order = Vec::with_capacity(count);
    // Sub-emitter roots first: a target declared *before* its parent would
    // otherwise dispatch first and only see the spawn requests a frame late.
    for sweep_parented in [false, true] {
        for root in 0..count {
            if parented[root] != sweep_parented || state[root] != 0 {
                continue;
            }
            let mut stack: Vec<(u32, bool)> = vec![(root as u32, false)];
            while let Some((node, expanded)) = stack.pop() {
                let node_index = node as usize;
                if expanded {
                    state[node_index] = 2;
                    continue;
                }
                if state[node_index] != 0 {
                    if state[node_index] == 1 {
                        log::warn!(
                            "scene_batch: particle sub-emitter cycle at system {node}; link dropped"
                        );
                    }
                    continue;
                }
                state[node_index] = 1;
                order.push(node);
                stack.push((node, true));
                if let Some(target) = targets[node_index] {
                    let target_index = target as usize;
                    if target_index < count
                        && target_index != node_index
                        && state[target_index] != 1
                    {
                        stack.push((target, false));
                    }
                }
            }
        }
    }
    order
}

/// Bind group the particle compute pass reads: emitter parameters, the
/// simulation cursor, the particle storage, the incoming request ring (this
/// system's), the outgoing ring (its sub-emitter target's, or its own when it
/// has none), the frame's collider set and the voxel occupancy texture.
#[allow(clippy::too_many_arguments)] // one binding per shader resource
fn particle_sim_bind(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: &wgpu::Buffer,
    sim: &wgpu::Buffer,
    particles: &wgpu::Buffer,
    incoming: &wgpu::Buffer,
    outgoing: &wgpu::Buffer,
    colliders: &wgpu::Buffer,
    voxels: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("repame_view3d_particle_sim_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: sim.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: particles.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: incoming.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: outgoing.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: colliders.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(voxels),
            },
        ],
    })
}

/// Bind group the billboard draw reads: the particle storage, the emitter
/// parameters (for the size and color curves) and the display parameters.
fn particle_draw_bind(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    particles: &wgpu::Buffer,
    params: &wgpu::Buffer,
    draw_params: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("repame_view3d_particle_draw_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: particles.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: draw_params.as_entire_binding(),
            },
        ],
    })
}

/// Group-2 binding for one reflection-probe cube.
fn probe_cube_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::Cube,
            multisampled: false,
        },
        count: None,
    }
}

/// Camera bind group over `camera`: binding 0 always, binding 1 (the joint
/// palette storage buffer) only when `skin` is present. The entry set must
/// match the camera layout exactly, and the layout omits binding 1 on
/// devices without `VERTEX_STORAGE` (see [`SceneBatch::set_skinning`]).
fn camera_bind_group(
    device: &wgpu::Device,
    label: &'static str,
    layout: &wgpu::BindGroupLayout,
    camera: &wgpu::Buffer,
    skin: Option<&wgpu::Buffer>,
) -> wgpu::BindGroup {
    let mut entries = Vec::with_capacity(2);
    entries.push(wgpu::BindGroupEntry {
        binding: 0,
        resource: camera.as_entire_binding(),
    });
    if let Some(skin) = skin {
        entries.push(wgpu::BindGroupEntry {
            binding: 1,
            resource: skin.as_entire_binding(),
        });
    }
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &entries,
    })
}

/// Group 2 of the scene pipelines: shadow maps plus the reflection-probe
/// cubes. Built once per probe set (a probe cube cannot be sampled while it
/// is a render attachment, so the capture passes use a group with stand-ins
/// while the scene pass uses this one).
#[allow(clippy::too_many_arguments)]
fn make_shadow_bind(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    shadow_view: &wgpu::TextureView,
    shadow_sampler: &wgpu::Sampler,
    cascade_view: &wgpu::TextureView,
    cascade_sampler: &wgpu::Sampler,
    point_view: &wgpu::TextureView,
    point_sampler: &wgpu::Sampler,
    probes: &[&wgpu::TextureView; super::reflection::MAX_PROBES],
    probe_sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    let mut entries = Vec::with_capacity(11);
    entries.push(wgpu::BindGroupEntry {
        binding: 0,
        resource: wgpu::BindingResource::TextureView(shadow_view),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: 1,
        resource: wgpu::BindingResource::Sampler(shadow_sampler),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: 2,
        resource: wgpu::BindingResource::TextureView(cascade_view),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: 3,
        resource: wgpu::BindingResource::Sampler(cascade_sampler),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: 4,
        resource: wgpu::BindingResource::TextureView(point_view),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: 5,
        resource: wgpu::BindingResource::Sampler(point_sampler),
    });
    for (slot, view) in probes.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: 6 + slot as u32,
            resource: wgpu::BindingResource::TextureView(view),
        });
    }
    entries.push(wgpu::BindGroupEntry {
        binding: 10,
        resource: wgpu::BindingResource::Sampler(probe_sampler),
    });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("repame_view3d_shadow_bg"),
        layout,
        entries: &entries,
    })
}

impl SceneBatch {
    pub fn with_id(id: impl Into<String>) -> Self {
        Self::with_desc(id, BatchDesc::default())
    }

    pub fn with_desc(id: impl Into<String>, desc: BatchDesc) -> Self {
        Self {
            id: id.into(),
            desc,
            camera: CameraUniform {
                view_proj: Mat4::IDENTITY.to_cols_array_2d(),
                light_dir: [0.0, 1.0, 0.0],
                _pad0: 0.0,
                light_color: [1.0, 1.0, 1.0],
                diffuse: 0.9,
                ambient: [0.35, 0.35, 0.38],
                _pad1: 0.0,
                fog: [0.0, 0.5, 0.55, 0.6],
                fog_range: [100.0, 600.0, 1.0, 0.0],
                cam_pos: [0.0, 0.0, 0.0],
                _pad2: 0.0,
                cam_forward: [0.0, 0.0, -1.0],
                _pad3: 0.0,
                shadow_vp: Mat4::IDENTITY.to_cols_array_2d(),
                shadow_params: [1.0, 0.0, 0.001, 0.0],
                shadow_texel: [1.0 / 1024.0, 1.0 / 1024.0, 0.0, 0.0],
                cascade_vp0: Mat4::IDENTITY.to_cols_array_2d(),
                cascade_vp1: Mat4::IDENTITY.to_cols_array_2d(),
                cascade_vp2: Mat4::IDENTITY.to_cols_array_2d(),
                cascade_vp3: Mat4::IDENTITY.to_cols_array_2d(),
                cascade_splits: [1e30, 1e30, 1e30, 0.0],
                cascade_params: [1.0, 0.0, 0.0, 0.0],
                point_pos0: [0.0, 0.0, 0.0, 1.0],
                point_pos1: [0.0, 0.0, 0.0, 1.0],
                point_col0: [0.0, 0.0, 0.0, 0.0],
                point_col1: [0.0, 0.0, 0.0, 0.0],
                point_pos2: [0.0, 0.0, 0.0, 1.0],
                point_pos3: [0.0, 0.0, 0.0, 1.0],
                point_col2: [0.0, 0.0, 0.0, 0.0],
                point_col3: [0.0, 0.0, 0.0, 0.0],
                point_pos4: [0.0, 0.0, 0.0, 1.0],
                point_pos5: [0.0, 0.0, 0.0, 1.0],
                point_col4: [0.0, 0.0, 0.0, 0.0],
                point_col5: [0.0, 0.0, 0.0, 0.0],
                point_pos6: [0.0, 0.0, 0.0, 1.0],
                point_pos7: [0.0, 0.0, 0.0, 1.0],
                point_col6: [0.0, 0.0, 0.0, 0.0],
                point_col7: [0.0, 0.0, 0.0, 0.0],
                point_params: [0.0, 0.0, 0.005, 1.0],
                probe_pos: [[0.0; 4]; super::reflection::MAX_PROBES],
                probe_params: [[0.0; 4]; super::reflection::MAX_PROBES],
            },
            camera_pos: [0.0, 0.0, 0.0],
            view_proj: Mat4::IDENTITY,
            culled: 0,
            pending: Vec::new(),
            uploads: Vec::new(),
            mip_uploads: Vec::new(),
            page_levels: vec![0.0; RESIDENT_PAGE_SLOTS as usize * 4],
            stats: super::stats::FrameStats::new(),
            overrides: super::texfmt::TextureOverrides::new(),
            materials: Vec::new(),
            passes: Vec::new(),
            particles: None,
            particle_order: Vec::new(),
            probes: Vec::new(),
            occlusion: None,
            occlusion_feed: None,
            occlusion_phase_two: false,
            occluded: 0,
            occlusion_candidates: Vec::new(),
            post: None,
            skinning: GpuSkinning::Supported,
            verts: Vec::new(),
            indices: Vec::new(),
            ranges: Vec::new(),
            skinned: Vec::new(),
            palette: Vec::new(),
            cascade_desc: None,
            point_caster: None,
            staged_points: Vec::new(),
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn set_camera(&mut self, view_proj: Mat4) {
        self.camera.view_proj = view_proj.to_cols_array_2d();
        self.view_proj = view_proj;
    }

    /// Camera world position for transparent back-to-front sorting (see
    /// [`SceneBatch::finish`]) and the specular/fog view vector. The
    /// viewport feeds `cam.eye()` from the same snapshot as the
    /// view-projection matrix, so sort order, speculars, fog, and pixels
    /// share one camera.
    pub fn set_camera_pos(&mut self, pos: [f32; 3]) {
        self.camera_pos = pos;
        self.camera.cam_pos = pos;
    }

    /// Unit vector the camera looks along, from the same snapshot as
    /// [`SceneBatch::set_camera_pos`]. Fog and cascade splits are depth
    /// along this axis, not range from the eye.
    pub fn set_camera_forward(&mut self, forward: [f32; 3]) {
        self.camera.cam_forward = forward;
    }

    /// Groups culled by the last [`SceneBatch::finish`].
    pub fn culled(&self) -> usize {
        self.culled
    }

    /// Shadow-map configuration for the next `prepare`: light-space depth
    /// pass over opaque depth-tested groups, compared per lit fragment.
    /// `None` (default) disables it and reproduces legacy pixels exactly.
    /// The light direction still comes from [`SceneBatch::set_light`]; the
    /// ortho box follows `shadow_center` (pass the camera target) with
    /// `dist` sizing it (pass the camera distance), same sources the
    /// viewport feeds per frame.
    pub fn set_shadow(
        &mut self,
        desc: Option<super::shadow::ShadowDesc>,
        shadow_center: [f32; 3],
        dist: f32,
    ) {
        match desc {
            Some(d) => {
                let vp = super::shadow::light_view_proj(
                    glam::Vec3::from(shadow_center),
                    glam::Vec3::from(self.camera.light_dir),
                    super::shadow::shadow_extent(dist),
                );
                self.camera.shadow_vp = vp.to_cols_array_2d();
                self.camera.shadow_params = [d.clamped_strength(), 1.0, d.clamped_bias(), 0.0];
                let t = d.texel();
                self.camera.shadow_texel = [t, t, 0.0, 0.0];
            }
            None => {
                self.camera.shadow_params = [1.0, 0.0, 0.001, 0.0];
            }
        }
    }

    /// Whether the shadow pass is armed for the next `prepare`.
    pub fn shadows_enabled(&self) -> bool {
        self.camera.shadow_params[1] > 0.5
    }

    /// Frame light for lit groups. Flat groups ignore it entirely.
    /// Legacy path: prefer [`set_rig`](SceneBatch::set_rig) with a
    /// [`LightRig`] (same sun, plus points/cascades). Calling both
    /// applies in order (last wins for the sun fields).
    pub fn set_light(&mut self, light: SceneLight) {
        let d = glam::Vec3::from(light.direction);
        let d = if d.length_squared() > 1e-8 {
            d.normalize()
        } else {
            glam::Vec3::Y
        };
        self.camera.light_dir = d.into();
        self.camera.ambient = light.ambient;
        self.camera.light_color = light.color;
        self.camera.diffuse = light.diffuse.max(0.0);
        self.camera.fog = [
            light.fog.clamp(0.0, 1.0),
            light.fog_color[0],
            light.fog_color[1],
            light.fog_color[2],
        ];
        self.camera.fog_range = [
            light.fog_start.max(0.0),
            light.fog_end.max(light.fog_start.max(0.0) + 1e-3),
            if light.exposure.is_finite() && light.exposure > 0.0 {
                light.exposure
            } else {
                1.0
            },
            light.tonemap.index(),
        ];
    }

    /// Full frame light rig: sun (same fields as [`set_light`](SceneBatch::set_light))
    /// plus point lights and the cascade selector. `set_light` stays the
    /// legacy shorthand; this is the deliberate path for cascades and
    /// points. Points past [`MAX_POINTS`](crate::MAX_POINTS) keep the
    /// brightest (score = max color channel times clamped intensity) with
    /// a warning; NaN colors score zero so corrupt lights drop first.
    /// `point_shadows` out of range disables the cube pass with a warning.
    /// Cascade matrices fit here from the camera (pass `cam.eye()`,
    /// `cam.view_matrix()`, and the inverse view-proj): the same sources
    /// the viewport feeds per frame, so the fit and the pixels agree.
    pub fn set_rig(
        &mut self,
        rig: &LightRig,
        cam_eye: [f32; 3],
        cam_view: glam::Mat4,
        cam_inv_view_proj: glam::Mat4,
    ) {
        self.set_light(rig.sun);
        let mut points: Vec<super::shadow::PointLight> = rig.points.clone();
        if points.len() > crate::MAX_POINTS {
            points.sort_by(|a, b| {
                b.score()
                    .partial_cmp(&a.score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            points.truncate(crate::MAX_POINTS);
            log::warn!(
                "scene_batch[{}]: keeping brightest {} of {} point lights",
                self.id,
                crate::MAX_POINTS,
                rig.points.len()
            );
        }
        let eye = glam::Vec3::from(cam_eye);
        let mut set = |slot: usize, pos: [f32; 4], col: [f32; 4]| match slot {
            0 => {
                self.camera.point_pos0 = pos;
                self.camera.point_col0 = col;
            }
            1 => {
                self.camera.point_pos1 = pos;
                self.camera.point_col1 = col;
            }
            2 => {
                self.camera.point_pos2 = pos;
                self.camera.point_col2 = col;
            }
            3 => {
                self.camera.point_pos3 = pos;
                self.camera.point_col3 = col;
            }
            4 => {
                self.camera.point_pos4 = pos;
                self.camera.point_col4 = col;
            }
            5 => {
                self.camera.point_pos5 = pos;
                self.camera.point_col5 = col;
            }
            6 => {
                self.camera.point_pos6 = pos;
                self.camera.point_col6 = col;
            }
            _ => {
                self.camera.point_pos7 = pos;
                self.camera.point_col7 = col;
            }
        };
        for slot in 0..crate::MAX_POINTS {
            match points.get(slot) {
                Some(p) => {
                    let inten = if p.intensity.is_finite() {
                        p.intensity.max(0.0)
                    } else {
                        0.0
                    };
                    set(
                        slot,
                        [
                            p.position[0],
                            p.position[1],
                            p.position[2],
                            p.clamped_range(),
                        ],
                        [p.color[0], p.color[1], p.color[2], inten],
                    );
                }
                None => set(slot, [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]),
            }
        }
        let caster_ok = rig.point_shadows.is_some_and(|i| {
            points.get(i).is_some_and(|p| {
                glam::Vec3::from(p.position).is_finite()
                    && p.intensity.is_finite()
                    && p.intensity > 0.0
            })
        });
        if rig.point_shadows.is_some() && !caster_ok {
            log::warn!(
                "scene_batch[{}]: point_shadows {:?} invalid, cube pass off",
                self.id,
                rig.point_shadows
            );
        }
        let bias = if rig.point_bias.is_finite() {
            rig.point_bias.clamp(0.0, 0.05)
        } else {
            0.005
        };
        let strength = if rig.point_strength.is_finite() {
            rig.point_strength.clamp(0.0, 1.0)
        } else {
            1.0
        };
        self.camera.point_params = [points.len() as f32, caster_ok as u8 as f32, bias, strength];
        self.point_caster = if caster_ok { rig.point_shadows } else { None };
        self.staged_points = points;
        match rig.cascades {
            Some(desc) => {
                let count = desc.clamped_count();
                let dir = glam::Vec3::from(self.camera.light_dir);
                let slices = desc.fit_all(
                    eye,
                    cam_view,
                    cam_inv_view_proj,
                    dir,
                    super::camera::NEAR,
                    super::camera::FAR,
                );
                let mut vps = [
                    &mut self.camera.cascade_vp0,
                    &mut self.camera.cascade_vp1,
                    &mut self.camera.cascade_vp2,
                    &mut self.camera.cascade_vp3,
                ];
                let mut splits = [1e30f32, 1e30, 1e30, 0.0];
                for (i, vp) in vps.iter_mut().enumerate() {
                    if i < count {
                        **vp = slices[i].view_proj.to_cols_array_2d();
                        if i < 3 {
                            splits[i] = slices[i].far;
                        }
                    } else {
                        **vp = glam::Mat4::IDENTITY.to_cols_array_2d();
                    }
                }
                self.camera.cascade_splits = splits;
                self.camera.cascade_params = [
                    desc.clamped_strength(),
                    1.0,
                    count as f32,
                    desc.clamped_normal_bias(),
                ];
                let t = desc.texel();
                self.camera.shadow_texel = [t, t, 0.0, 0.0];
                self.cascade_desc = Some(desc);
            }
            None => {
                self.camera.cascade_params = [1.0, 0.0, 0.0, 0.0];
                self.camera.cascade_splits = [1e30, 1e30, 1e30, 0.0];
                self.cascade_desc = None;
            }
        }
    }

    /// Armed cascade count for the next `prepare` (0 = legacy path).
    pub fn cascades_enabled(&self) -> usize {
        if self.camera.cascade_params[1] > 0.5 {
            self.camera.cascade_params[2] as usize
        } else {
            0
        }
    }

    /// Point-light count staged for the next `prepare`.
    pub fn point_count(&self) -> usize {
        self.camera.point_params[0] as usize
    }

    /// Whether the cube shadow pass is armed for the next `prepare`.
    pub fn point_shadows_enabled(&self) -> bool {
        self.camera.point_params[1] > 0.5
    }

    pub fn clear(&mut self) {
        self.pending.clear();
        self.uploads.clear();
        self.mip_uploads.clear();
        self.verts.clear();
        self.indices.clear();
        self.ranges.clear();
        self.skinned.clear();
        self.palette.clear();
    }

    pub fn len_tris(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Queue texture uploads, applied in the next `prepare`.
    pub fn upload(&mut self, upload: SceneUpload) {
        // Guard the same contract `push_group` enforces for pages: an
        // out-of-range upload is dropped at queue time (with a warning) so
        // a caller packing against a smaller desc cannot corrupt the texture.
        if upload.page >= self.desc.layers
            || upload.x + upload.w > self.desc.layer_size
            || upload.y + upload.h > self.desc.layer_size
            || upload.rgba.len() != upload.w as usize * upload.h as usize * 4
        {
            log::warn!(
                "scene_batch[{}]: dropping out-of-range upload page={} {}x{}+{}+{} ({} bytes)",
                self.id,
                upload.page,
                upload.w,
                upload.h,
                upload.x,
                upload.y,
                upload.rgba.len()
            );
            return;
        }
        self.uploads.push(upload);
    }

    /// Queue several uploads at once (per-frame texture drains).
    pub fn extend_uploads(&mut self, uploads: impl IntoIterator<Item = SceneUpload>) {
        for upload in uploads {
            self.upload(upload);
        }
    }

    /// Queue one streamed mip level, applied in the next
    /// [`upload_all`](Self::upload_all) on the render thread (same contract as
    /// [`upload`](Self::upload)).
    ///
    /// The level is written whole, at its own mip: the game never has to move
    /// or rescale texels. An out-of-range page or level, or a payload whose
    /// extent does not match the page's mip edge, drops with a warning — never
    /// a partial or fabricated write. The level the shaders sample is
    /// published separately with [`set_page_level`](Self::set_page_level)
    /// (the residency layer's
    /// [`resolve_level`](crate::texture_stream::TextureResidency::resolve_level)).
    /// Texture contents do not survive a batch rebuild, so re-queue the
    /// resident levels afterwards.
    pub fn upload_mip(&mut self, upload: MipUpload) {
        let mips = self.desc.mip_levels();
        if upload.page >= self.desc.layers || upload.level >= mips {
            log::warn!(
                "scene_batch[{}]: dropping mip upload page={} level={} (pages {}, levels {mips})",
                self.id,
                upload.page,
                upload.level,
                self.desc.layers
            );
            return;
        }
        let edge = (self.desc.layer_size.max(1) >> upload.level).max(1);
        let expected = upload.width as usize * upload.height as usize * 4;
        if upload.width != edge || upload.height != edge || upload.texels.len() != expected {
            log::warn!(
                "scene_batch[{}]: dropping mis-sized mip upload page={} level={} {}x{} ({} bytes); page edge {edge}",
                self.id,
                upload.page,
                upload.level,
                upload.width,
                upload.height,
                upload.texels.len()
            );
            return;
        }
        self.mip_uploads.push(upload);
    }

    /// Queue several mip levels at once (per-frame streamed drains).
    pub fn extend_mip_uploads(&mut self, uploads: impl IntoIterator<Item = MipUpload>) {
        for upload in uploads {
            self.upload_mip(upload);
        }
    }

    /// Publish the finest mip level resident for `page`: exactly the level
    /// every textured draw of that page samples, on the legacy and material
    /// paths alike. Games streaming mips call this from the residency layer's
    /// [`resolve_level`](crate::texture_stream::TextureResidency::resolve_level)
    /// (which returns `None` when nothing is resident).
    ///
    /// * `Some(level)` — sample `level`. The default for every declared page
    ///   is `Some(0)`, the fully resident page, which reproduces the
    ///   historical pixels exactly.
    /// * `None` — nothing is resident: the page draws untextured (its tint,
    ///   and the identity texel in material programs) instead of sampling an
    ///   unwritten or evicted mip.
    ///
    /// A page past the batch's layers, a page past
    /// [`MAX_RESIDENT_PAGES`], or a level past the page's declared chain, is
    /// rejected with a warning and changes nothing (a level that cannot exist
    /// must never be sampled). The published level survives
    /// [`clear`](Self::clear): residency spans frames. The value is uploaded
    /// in [`upload_all`](Self::upload_all), so publishing it before
    /// `upload_all` takes effect for that frame's draws.
    pub fn set_page_level(&mut self, page: u32, level: Option<u32>) {
        if page >= self.desc.layers || page >= MAX_RESIDENT_PAGES {
            log::warn!(
                "scene_batch[{}]: page {page} outside {} layers / {MAX_RESIDENT_PAGES} resident pages, level ignored",
                self.id,
                self.desc.layers
            );
            return;
        }
        let mips = self.desc.mip_levels();
        let level = match level {
            None => -1.0,
            Some(level) if level < mips => level as f32,
            Some(level) => {
                log::warn!(
                    "scene_batch[{}]: page {page} has no mip level {level} (chain {mips}), resident level ignored",
                    self.id,
                );
                return;
            }
        };
        self.page_levels[page as usize] = level;
    }

    /// Replaces a page's pixels for every later upload of it, letting a game
    /// swap a page without re-deriving it. Removal restores queued uploads.
    pub fn set_overrides(&mut self, overrides: super::texfmt::TextureOverrides) {
        self.overrides = overrides;
    }

    /// Counters for the frame last finished.
    pub fn stats(&self) -> super::stats::FrameStats {
        self.stats
    }

    /// Publishes the last frame's counters, then clears them.
    pub fn take_stats(&mut self, sink: &mut dyn super::stats::StatsSink) {
        sink.publish(self.stats);
        self.stats = super::stats::FrameStats::new();
    }

    /// Append one group. Malformed groups (index out of range, or
    /// position/color length mismatch, partial normals/uvs, page past the
    /// batch layers, non-finite alpha) are dropped with a warning, never
    /// a panic, never partial draws. Degenerate triangles (zero area, NaN)
    /// are culled tri-by-tri so one bad triangle can't sink its group: the
    /// group draws with the surviving triangles. Normals and uvs stay
    /// all-or-nothing per group. Shares
    /// [`validate_group`](crate::validate_group) with the chunk
    /// cache so both paths agree on malformed.
    /// Appends one group drawn by the given material program (0 = legacy).
    pub fn push_group_with_material(&mut self, group: &MeshGroup, material: u32) {
        self.push_group_inner(group, material, 0);
    }

    pub fn push_group(&mut self, group: &MeshGroup) {
        self.push_group_inner(group, 0, 0);
    }

    fn push_group_inner(&mut self, group: &MeshGroup, material_slot: u32, pass: u32) {
        if !super::chunk::validate_group(group) {
            return;
        }
        if !group.alpha.is_finite() || !group.alpha_cutoff.is_finite() {
            log::warn!(
                "scene_batch[{}]: dropping group (non-finite alpha/cutoff)",
                self.id,
            );
            return;
        }
        if !group.uvs.is_empty() && group.texture_page >= self.desc.layers {
            log::warn!(
                "scene_batch[{}]: dropping group (page {} >= {} layers)",
                self.id,
                group.texture_page,
                self.desc.layers
            );
            return;
        }
        // AABB now, while positions are borrowed: `finish` sorts
        // transparent groups by its center and the occlusion phase tests it
        // without re-walking vertices.
        let bounds = group.bounds();
        let center = bounds.map(|(min, max)| {
            [
                (min[0] + max[0]) * 0.5,
                (min[1] + max[1]) * 0.5,
                (min[2] + max[2]) * 0.5,
            ]
        });
        self.pending.push(Pending {
            positions: group.positions.clone(),
            colors: group.colors.clone(),
            normals: group.normals.clone(),
            uvs: group.uvs.clone(),
            texture_page: group.texture_page,
            material: group.material,
            center,
            bounds,
            indices: cull_degenerate(&group.positions, &group.indices),
            transparent: group.transparent,
            alpha: group.alpha.clamp(0.0, 1.0),
            alpha_cutoff: group.alpha_cutoff.clamp(0.0, 1.0),
            depth_test: group.depth_test,
            material_slot,
            pass,
            hz_culled: false,
        });
    }

    /// Appends one group's geometry for `pass_index`, so a declared
    /// [`super::pass::RenderPass`] owns its draws instead of the shared
    /// scene target.
    pub fn push_group_for_pass(&mut self, group: &MeshGroup, pass_index: u32) {
        self.push_group_inner(group, 0, pass_index);
    }

    /// [`SceneBatch::push_group_for_pass`] with an explicit material program
    /// slot, so a post-process scene pass keeps its shaded materials.
    pub fn push_group_with_material_for_pass(
        &mut self,
        group: &MeshGroup,
        material: u32,
        pass_index: u32,
    ) {
        self.push_group_inner(group, material, pass_index);
    }

    /// Registers a material program, returning the slot later groups stamp
    /// into their vertices. Repeat registrations of the same program (same
    /// [`ShadingModel::cache_key`]) reuse the slot, so a frame of groups
    /// sharing a material lowers one module and binds one uniform.
    pub fn register_material(&mut self, model: &super::material::ShadingModel) -> u32 {
        if let Some((slot, _)) = self
            .materials
            .iter()
            .enumerate()
            .find(|(_, entry)| entry.cache_key == model.cache_key())
        {
            return slot as u32 + 1;
        }
        let entry = MaterialEntry {
            model: model.clone(),
            cache_key: model.cache_key(),
            uniform: super::shading::ShadingUniform::pack(model, &[], model.material_fog.enabled),
        };
        self.materials.push(entry);
        self.materials.len() as u32
    }

    /// The interleaved vertex layout both the default and the material
    /// pipelines bind: 16 slots, at their fixed struct offsets.
    fn vertex_buffer_layout(&self) -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<Vert>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: VERTEX_ATTRIBUTES,
        }
    }

    /// (Re)builds one pipeline set per material program registered this
    /// frame. Pipelines are keyed by program identity, so a frame reusing the
    /// same programs keeps its modules and bind groups.
    fn sync_material_pipelines(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        res: &mut SceneEntry,
    ) {
        if self.materials.is_empty() {
            res.material_pipelines.clear();
            return;
        }
        let layout = res.pipe_layout.clone();
        let format = res.key.0;
        let rebuild = res.material_pipelines.len() != self.materials.len()
            || res
                .material_pipelines
                .iter()
                .zip(self.materials.iter())
                .any(|(gpu, entry)| gpu.key != entry.cache_key);
        if !rebuild {
            return;
        }
        let mut built = Vec::with_capacity(self.materials.len());
        for entry in &self.materials {
            let fragment = super::shading::material_fragment(&entry.model)
                .expect("registered materials validate");
            let mut source = scene_prefix(res.skin);
            source.push_str(&fragment);
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("repame_view3d_material"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let buffers = [Some(self.vertex_buffer_layout())];
            let pipeline = |depth_test: bool, transparent: bool| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("repame_view3d_material_pipeline"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some("vs_main"),
                        buffers: &buffers,
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format,
                            blend: Some(if transparent {
                                wgpu::BlendState::ALPHA_BLENDING
                            } else {
                                wgpu::BlendState::REPLACE
                            }),
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
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth24PlusStencil8,
                        depth_write_enabled: Some(!transparent),
                        depth_compare: Some(if depth_test {
                            wgpu::CompareFunction::LessEqual
                        } else {
                            wgpu::CompareFunction::Always
                        }),
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState {
                        count: SCENE_TARGET_SAMPLE_COUNT,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    multiview_mask: None,
                    cache: None,
                })
            };
            let mut pages: Vec<u32> = vec![0; super::material::MAX_TEX_UNITS];
            for stage in &entry.model.stages {
                if let Some(slot) = pages.get_mut(usize::from(stage.tex_unit)) {
                    *slot = stage.page as u32;
                }
            }
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("repame_view3d_material_program"),
                size: size_of::<MaterialUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("repame_view3d_material_program_bg"),
                layout: &res.material_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            });
            let uniform = material_uniform_for(&entry.uniform, &pages);
            queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&[uniform]));
            built.push(MaterialPipelines {
                depth: pipeline(true, false),
                flat: pipeline(false, false),
                transparent: pipeline(true, true),
                transparent_flat: pipeline(false, true),
                buffer,
                bind,
                key: entry.cache_key,
            });
        }
        res.material_pipelines = built;
    }

    /// The material block uploaded when a frame registers no program: white
    /// kcolors, no fog, every unit on page 0.
    fn material_uniform(&self) -> MaterialUniform {
        let default = super::material::ShadingModel::default();
        match self.materials.first() {
            Some(entry) => material_uniform_for(&entry.uniform, &[]),
            None => material_uniform_for(
                &super::shading::ShadingUniform::pack(&default, &[], false),
                &[],
            ),
        }
    }

    /// Builds/refreshes the particle GPU state for the staged frame: one
    /// buffer set per live system, the frame's collision world, and the
    /// bind groups. Simulation state the CPU cannot know (the spawn cursor,
    /// which sub-emitter requests advance) stays in the GPU buffers; the
    /// CPU owns only the fractional rate carry and the frame counter, both
    /// kept across frames in `res`.
    fn sync_particles(&self, device: &wgpu::Device, queue: &wgpu::Queue, res: &mut SceneEntry) {
        let Some(frame) = self.particles.as_ref() else {
            if let Some(gpu) = res.particles.as_mut() {
                gpu.order.clear();
                for slot in gpu.systems.iter_mut() {
                    *slot = None;
                }
            }
            return;
        };
        let live = |index: usize| {
            frame
                .systems
                .get(index)
                .is_some_and(|system| system.capacity > 0)
        };
        if !self.particle_order.iter().any(|i| live(*i as usize)) {
            // Nothing to simulate this frame: stop dispatching and drawing the
            // previous frame's systems instead of leaving them running.
            if let Some(gpu) = res.particles.as_mut() {
                gpu.order.clear();
                for slot in gpu.systems.iter_mut() {
                    *slot = None;
                }
            }
            return;
        }
        if res.particles.is_none() {
            res.particles = Some(ParticleGpu::new(
                device,
                res.key.0,
                &res.cam_layout,
                &res.tex_layout,
            ));
        }
        let dt = frame.clamped_dt();
        let mut accepts = vec![false; frame.systems.len()];
        for system in &frame.systems {
            if let Some(sub) = system.sub
                && !matches!(sub.mode, super::particles::SubEmitterMode::Off)
                && live(sub.target)
            {
                accepts[sub.target] = true;
            }
        }
        let gpu = res.particles.as_mut().expect("particle state just built");
        gpu.order = self.particle_order.clone();
        gpu.systems.resize_with(frame.systems.len(), || None);
        queue.write_buffer(
            &gpu.colliders,
            0,
            bytemuck::cast_slice(&[pack_colliders(&frame.colliders)]),
        );
        if let Some(voxels) = &frame.colliders.voxels {
            let key = (std::sync::Arc::as_ptr(voxels) as usize, voxels.cells.len());
            if gpu.voxels_key != Some(key) {
                let extent = wgpu::Extent3d {
                    width: voxels.dims[0],
                    height: voxels.dims[1],
                    depth_or_array_layers: voxels.dims[2],
                };
                if gpu.voxels.size() != extent {
                    gpu.voxels = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("repame_view3d_particle_voxels"),
                        size: extent,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D3,
                        format: wgpu::TextureFormat::R32Uint,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING
                            | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    });
                }
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &gpu.voxels,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    bytemuck::cast_slice(&voxels.cells),
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(voxels.dims[0] * 4),
                        rows_per_image: Some(voxels.dims[1]),
                    },
                    extent,
                );
                gpu.voxels_key = Some(key);
            }
        } else {
            gpu.voxels_key = None;
        }

        let voxels_view = frame
            .colliders
            .voxels
            .as_ref()
            .map(|_| gpu.voxels.create_view(&wgpu::TextureViewDescriptor::default()))
            .unwrap_or_else(|| gpu.dummy_voxels_view.clone());
        let blank = |label: &'static str, size: u64, copy_dst: bool| {
            let mut usage = wgpu::BufferUsages::STORAGE;
            if copy_dst {
                usage |= wgpu::BufferUsages::COPY_DST;
            }
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        for (index, system) in frame.systems.iter().enumerate() {
            if !live(index) {
                gpu.systems[index] = None;
                continue;
            }
            let capacity = system.capacity;
            let request_cap = request_capacity(capacity);
            let stale = gpu.systems[index]
                .as_ref()
                .is_none_or(|state| state.capacity != capacity || state.seed != system.seed);
            if stale {
                let state = gpu.systems[index].take();
                let (carry, frame_no) = match state {
                    Some(old) if old.seed == system.seed => (old.carry, old.frame),
                    _ => (0.0, 0),
                };
                let params = blank(
                    "repame_view3d_particle_params",
                    size_of::<GpuEmitter>() as u64,
                    true,
                );
                let sim = blank(
                    "repame_view3d_particle_sim_state",
                    size_of::<GpuSimState>() as u64,
                    true,
                );
                let particles = blank(
                    "repame_view3d_particle_storage",
                    capacity as u64 * super::particles::PARTICLE_BYTES as u64,
                    false,
                );
                let requests = blank(
                    "repame_view3d_particle_requests",
                    16 + request_cap as u64 * super::particles::SUB_REQUEST_BYTES as u64,
                    false,
                );
                let draw_params = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("repame_view3d_particle_draw"),
                    size: size_of::<GpuDrawParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let sim_bind = particle_sim_bind(
                    device,
                    &gpu.sim_layout,
                    &params,
                    &sim,
                    &particles,
                    &requests,
                    &requests,
                    &gpu.colliders,
                    &voxels_view,
                );
                let draw_bind = particle_draw_bind(
                    device,
                    &gpu.draw_layout,
                    &particles,
                    &params,
                    &draw_params,
                );
                gpu.systems[index] = Some(ParticleSystemGpu {
                    capacity,
                    seed: system.seed,
                    pass: system.pass,
                    params,
                    sim,
                    particles,
                    requests,
                    draw_params,
                    sim_bind,
                    draw_bind,
                    carry,
                    frame: frame_no,
                    flat: false,
                });
            }
            let state = gpu.systems[index].as_mut().expect("just built");
            state.pass = system.pass;
            // A page past the batch's layer count cannot be sampled: draw
            // the particles flat instead of reading a foreign layer.
            if let Some(page) = system.display.page
                && page >= self.desc.layers
            {
                if !state.flat {
                    log::warn!(
                        "scene_batch[{}]: particle system {index} page {page} >= {} layers, drawing flat",
                        self.id,
                        self.desc.layers
                    );
                }
                state.flat = true;
            } else {
                state.flat = false;
            }
            let carry = state.carry + system.rate.max(0.0) * dt;
            let whole = carry.floor();
            state.carry = carry - whole;
            state.frame = state.frame.wrapping_add(1);
            let budget = (whole.max(0.0) as u32).min(capacity);
            let sub_budget = if accepts[index] { request_cap } else { 0 };
            let params = pack_emitter(system, budget, sub_budget, state.frame, dt, request_cap);
            queue.write_buffer(&state.params, 0, bytemuck::cast_slice(&[params]));
            // Only the frame counter is CPU-owned; the spawn cursor in front
            // of it belongs to the GPU (sub-emitter requests advance it).
            queue.write_buffer(&state.sim, 20, bytemuck::cast_slice(&[state.frame]));
            let display = system.display;
            let textured = display.page.is_some() && !state.flat;
            let draw = GpuDrawParams {
                scale_alpha: [
                    display.size_scale,
                    display.alpha,
                    display.page.map_or(0.0, |p| p as f32),
                    if textured { 1.0 } else { 0.0 },
                ],
                tint: [display.tint[0], display.tint[1], display.tint[2], 0.0],
            };
            queue.write_buffer(&state.draw_params, 0, bytemuck::cast_slice(&[draw]));
        }

        // Sub-emitter links bind the target's request ring: those bind groups
        // can only be built once every buffer above exists.
        let targets: Vec<Option<u32>> = frame
            .systems
            .iter()
            .map(|system| {
                system.sub.and_then(|sub| {
                    (!matches!(sub.mode, super::particles::SubEmitterMode::Off)
                        && live(sub.target))
                    .then_some(sub.target as u32)
                })
            })
            .collect();
        let outgoing: Vec<Option<wgpu::Buffer>> = (0..frame.systems.len())
            .map(|index| {
                let own = gpu.systems.get(index)?.as_ref()?.requests.clone();
                Some(match targets[index] {
                    Some(target) => gpu
                        .systems
                        .get(target as usize)
                        .and_then(|state| state.as_ref())
                        .map(|state| state.requests.clone())
                        .unwrap_or(own),
                    None => own,
                })
            })
            .collect();
        for (index, state) in gpu.systems.iter_mut().enumerate() {
            let Some(state) = state.as_mut() else {
                continue;
            };
            let Some(outgoing) = outgoing.get(index).and_then(|buffer| buffer.as_ref()) else {
                continue;
            };
            state.sim_bind = particle_sim_bind(
                device,
                &gpu.sim_layout,
                &state.params,
                &state.sim,
                &state.particles,
                &state.requests,
                outgoing,
                &gpu.colliders,
                &voxels_view,
            );
            state.draw_bind = particle_draw_bind(
                device,
                &gpu.draw_layout,
                &state.particles,
                &state.params,
                &state.draw_params,
            );
        }
    }

    /// Builds/refreshes the reflection-probe cubes for the staged probes and
    /// rebuilds the scene's group-2 bind group over them. A probe captures on
    /// its interval, immediately when one of its settings changed, and never
    /// when it is inert; the cube keeps its last capture between captures.
    fn sync_probes(&self, device: &wgpu::Device, queue: &wgpu::Queue, res: &mut SceneEntry) {
        if self.probes.is_empty() && res.probes.is_none() {
            return;
        }
        let format = res.key.0;
        // Probe cubes are sampled with a filtering sampler, so the scene
        // format must be filterable; `R32Float` is the one in the supported
        // set that is not. Disable probe sampling (and say so) rather than
        // build a cube the shader cannot read.
        if matches!(format, wgpu::TextureFormat::R32Float) {
            if self.probes.iter().any(|probe| probe.is_active()) {
                log::warn!(
                    "scene_batch[{}]: reflection probes need a filterable scene format ({:?}), disabled",
                    self.id,
                    format
                );
            }
            let mut uniform = res.camera_mat;
            uniform.probe_params = [[0.0; 4]; super::reflection::MAX_PROBES];
            queue.write_buffer(&res.camera, 0, bytemuck::cast_slice(&[uniform]));
            return;
        }
        if res.probes.is_none() {
            let dummy_refs: [&wgpu::TextureView; super::reflection::MAX_PROBES] =
                std::array::from_fn(|i| &res.probe_dummy_views[i]);
            let capture_bind = make_shadow_bind(
                device,
                &res.shadow_layout,
                &res.shadow_view,
                &res.shadow_sampler,
                &res.cascade_view,
                &res.cascade_sampler,
                &res.point_view,
                &res.point_sampler,
                &dummy_refs,
                &res.probe_sampler,
            );
            res.probes = Some(ProbeGpu {
                probes: Vec::new(),
                capture_bind,
                frames: 0,
            });
        }
        // A capture camera per face: the camera uniform is the frame's, so
        // each face needs its own buffer for the pass it belongs to.
        let cam_layout = res.cam_layout.clone();
        let skin = res.skin_buffer.clone();
        let probes = self.probes.clone();
        let gpu = res.probes.as_mut().expect("probe state just built");
        gpu.frames = gpu.frames.wrapping_add(1);
        while gpu.probes.len() > probes.len() {
            gpu.probes.pop();
        }
        for (slot, probe) in probes.iter().enumerate() {
            let edge = probe.clamped_resolution();
            let rebuild = gpu
                .probes
                .get(slot)
                .is_none_or(|state| state.edge != edge);
            if rebuild {
                let size = wgpu::Extent3d {
                    width: edge,
                    height: edge,
                    depth_or_array_layers: 6,
                };
                let cube = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("repame_view3d_probe_cube"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let view = cube.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::Cube),
                    ..Default::default()
                });
                let depth = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("repame_view3d_probe_depth"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth24PlusStencil8,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
                let cam_binds = (0..6)
                    .map(|_| {
                        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("repame_view3d_probe_camera"),
                            size: size_of::<CameraUniform>() as u64,
                            usage: wgpu::BufferUsages::UNIFORM
                                | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });
                        let bind = camera_bind_group(
                            device,
                            "repame_view3d_probe_cam_bg",
                            &cam_layout,
                            &buffer,
                            skin.as_ref(),
                        );
                        (buffer, bind)
                    })
                    .collect();
                gpu.probes.push(ProbeState {
                    // An impossible range forces the first frame to capture.
                    settings: ProbeSettings {
                        range: -1.0,
                        ..*probe
                    },
                    cube,
                    view,
                    depth,
                    edge,
                    cam_binds,
                    capture: true,
                });
            }
            let state = &mut gpu.probes[slot];
            let changed = state.settings != *probe;
            let interval = probe.clamped_interval() as u64;
            state.capture = probe.is_active()
                && (changed || gpu.frames % interval == 0);
            state.settings = *probe;
        }
        let views: [&wgpu::TextureView; super::reflection::MAX_PROBES] = std::array::from_fn(|i| {
            gpu.probes
                .get(i)
                .map(|state| &state.view)
                .unwrap_or(&res.probe_dummy_views[i])
        });
        res.shadow_bind = make_shadow_bind(
            device,
            &res.shadow_layout,
            &res.shadow_view,
            &res.shadow_sampler,
            &res.cascade_view,
            &res.cascade_sampler,
            &res.point_view,
            &res.point_sampler,
            &views,
            &res.probe_sampler,
        );
    }

    /// Ensures every declared pass has a target texture, reallocating only
    /// when that declaration changed.
    ///
    /// A [`Render`](super::pass::PassKind::Render) pass must match the batch's
    /// own format: the scene pipelines are built for it once, so a differently
    /// formatted attachment would need its own pipeline set. Fullscreen and
    /// resolve passes may use any format (the post pipelines are built per
    /// format), and a [`Depth`](super::pass::PassKind::Depth) or
    /// [`Occlusion`](super::pass::PassKind::Occlusion) pass must be
    /// `Rgba16Float`, or `R32Float` where the device renders to it (view
    /// depth needs the range). A mismatched pass is reported and skipped.
    fn sync_pass_targets(&self, device: &wgpu::Device, res: &mut SceneEntry) {
        let supported = |spec: &PassSpec| match spec.kind {
            super::pass::PassKind::Depth | super::pass::PassKind::Occlusion => {
                // View depth needs the range. Both formats are guaranteed
                // render attachments (wgpu's core format table), and the
                // chain reads either with `textureLoad`; R32Float is not
                // filterable, so post stages cannot sample it.
                matches!(
                    spec.target.format,
                    super::pass::RenderTargetFormat::Rgba16Float
                        | super::pass::RenderTargetFormat::R32Float
                )
            }
            super::pass::PassKind::Render => pass_format(spec.target.format) == res.key.0,
            super::pass::PassKind::Resolve | super::pass::PassKind::Fullscreen => true,
        };
        for spec in &self.passes {
            if !supported(spec) {
                log::warn!(
                    "scene_batch[{}]: pass '{}' cannot render as {:?} for kind {:?}; pass skipped",
                    self.id,
                    spec.target.name,
                    spec.target.format,
                    spec.kind
                );
                continue;
            }
            let stale = res
                .pass_targets
                .get(&spec.target.name)
                .is_none_or(|target| {
                    target.width != spec.target.width
                        || target.height != spec.target.height
                        || target.format != spec.target.format
                        || target.samples != spec.target.samples
                });
            if !stale {
                continue;
            }
            if let Some(state) = res.fullscreen.as_mut() {
                state.binds.remove(&spec.target.name);
            }
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("repame_view3d_pass_target"),
                size: wgpu::Extent3d {
                    width: spec.target.width.max(1),
                    height: spec.target.height.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: spec.target.samples.max(1),
                dimension: wgpu::TextureDimension::D2,
                format: pass_format(spec.target.format),
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let depth = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("repame_view3d_pass_depth"),
                size: texture.size(),
                mip_level_count: 1,
                sample_count: texture.sample_count(),
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
            res.pass_targets.insert(
                spec.target.name.clone(),
                PassTarget {
                    texture,
                    view,
                    _depth: depth,
                    depth_view,
                    width: spec.target.width,
                    height: spec.target.height,
                    format: spec.target.format,
                    samples: spec.target.samples,
                },
            );
        }
        let live: Vec<&String> = self
            .passes
            .iter()
            .filter(|spec| supported(spec))
            .map(|spec| &spec.target.name)
            .collect();
        res.pass_targets.retain(|name, _| live.contains(&name));
        if let Some(state) = res.fullscreen.as_mut() {
            state.binds.retain(|name, _| live.contains(&name));
        }
    }

    /// Declares the frame's passes, replacing any previous list. Pass
    /// geometry goes in through [`SceneBatch::push_group_for_pass`] with the
    /// 1-based index this list uses (0 is the shared scene target).
    pub fn set_passes(
        &mut self,
        passes: &[super::pass::RenderPass],
    ) -> Result<(), super::pass::PassError> {
        super::pass::validate_passes(passes)?;
        let order = super::pass::pass_order(passes);
        self.passes = order
            .iter()
            .map(|&position| {
                let pass = &passes[position];
                PassSpec {
                    target: pass.target.clone(),
                    kind: pass.kind,
                    source: pass.source.clone(),
                    // The declared position, which pass groups and the post
                    // chain key on; the store order is the schedule.
                    index: position as u32 + 1,
                    presents: pass.presents,
                    overlay: pass.overlay,
                    post: pass.post,
                    aux: pass.aux.clone(),
                    load: pass.load,
                }
            })
            .collect();
        Ok(())
    }

    /// Stages the frame's post-process settings. The depth/velocity pass reads
    /// [`PostProcess::previous_view_proj`] from here; the pass list itself is
    /// declared through [`SceneBatch::set_passes`].
    pub fn set_post(&mut self, post: Option<PostProcess>) {
        self.post = post;
    }

    /// Declares the device's GPU-skinning capability
    /// ([`GpuSkinning`], read from the adapter at device creation).
    ///
    /// On [`GpuSkinning::Unsupported`] the camera layout and every
    /// scene-family shader are built without the joint-palette storage
    /// binding (a device without `VERTEX_STORAGE` cannot construct the
    /// binding at all), and [`push_skinned`](SceneBatch::push_skinned) /
    /// [`push_skinned_for_pass`](SceneBatch::push_skinned_for_pass) drop
    /// their draws with a one-time warning naming the reason: the caller
    /// must CPU-pose them through
    /// [`SkinnedMesh::pose`](crate::SkinnedMesh::pose) instead. Nothing
    /// silently no-ops; the drop is the honest degradation.
    ///
    /// The mode is part of the pipeline/texture cache key, so switching it
    /// rebuilds the batch (texture contents drop, re-upload required).
    /// Defaults to [`GpuSkinning::Supported`], which reproduces the
    /// historical GPU path byte for byte.
    pub fn set_skinning(&mut self, skinning: GpuSkinning) {
        self.skinning = skinning;
    }

    /// Passes declared for this frame, in execution order.
    pub fn passes(&self) -> &[PassSpec] {
        &self.passes
    }

    /// Stages the frame's reflection probes (capped at
    /// [`MAX_PROBES`](crate::MAX_PROBES)). Each staged, active probe is
    /// captured into its own cube on its own interval, and every lit
    /// fragment samples the cubes with the probe's falloff volume. Invalid
    /// settings are dropped with a warning; an empty slice disables probe
    /// sampling without dropping the captured cubes.
    pub fn set_probes(&mut self, probes: &[ProbeSettings]) {
        self.probes.clear();
        for (index, probe) in probes.iter().enumerate() {
            if index >= super::reflection::MAX_PROBES {
                log::warn!(
                    "scene_batch[{}]: dropping reflection probe {index} (cap {})",
                    self.id,
                    super::reflection::MAX_PROBES
                );
                break;
            }
            if let Err(err) = probe.validate() {
                log::warn!(
                    "scene_batch[{}]: dropping reflection probe {index}: {err}",
                    self.id
                );
                continue;
            }
            self.probes.push(*probe);
        }
        for slot in 0..super::reflection::MAX_PROBES {
            let probe = self.probes.get(slot).filter(|probe| probe.is_active());
            self.camera.probe_pos[slot] = match probe {
                Some(probe) => [
                    probe.position[0],
                    probe.position[1],
                    probe.position[2],
                    probe.range.max(0.0),
                ],
                None => [0.0; 4],
            };
            self.camera.probe_params[slot] = match probe {
                Some(probe) => [
                    probe.intensity.max(0.0),
                    probe.ambient.clamp(0.0, 1.0),
                    probe.specular.clamp(0.0, 1.0),
                    1.0,
                ],
                None => [0.0; 4],
            };
        }
    }

    /// Reflection probes staged for this frame.
    pub fn probes(&self) -> &[ProbeSettings] {
        &self.probes
    }

    /// Stages the frame's occlusion test: [`finish`](SceneBatch::finish)
    /// runs it before the frustum phase over the frame's *opaque*
    /// depth-tested groups that carry bounds (the depth pass it reads
    /// replays exactly those). `None` (default) disables it and reproduces
    /// the legacy cull counts exactly. The state must be built with the
    /// *previous* frame's camera, which is what the one-frame delay
    /// provides; a game building its own pyramid from a full-resolution
    /// readback uses
    /// [`DepthPyramid::from_tiled_bytes`](crate::DepthPyramid::from_tiled_bytes),
    /// the viewport's feed uses
    /// [`DepthPyramid::from_packed_levels`](crate::DepthPyramid::from_packed_levels).
    ///
    /// The viewport drives that loop itself when the frame arms
    /// [`Frame3d::occlusion_feed`](crate::Frame3d::occlusion_feed): it takes
    /// the pyramid from the batch's own feed and stages it without calling
    /// this. A state handed in here (a game's own pyramid) wins over the
    /// feed.
    pub fn set_occlusion(&mut self, state: Option<super::occlusion::OcclusionCulling>) {
        self.occlusion = state;
    }

    /// Stages the viewport-driven occlusion feed's settings
    /// ([`crate::OcclusionFeedDesc`]), including whether the same-frame
    /// phase two runs. `None` (default) leaves the feed to the caller-staged
    /// [`SceneBatch::set_occlusion`] state alone.
    pub fn set_occlusion_feed(&mut self, feed: Option<crate::OcclusionFeedDesc>) {
        self.occlusion_feed = feed;
    }

    /// Groups culled by the last [`SceneBatch::finish`]'s hierarchical-Z
    /// phase. The frustum phase keeps its own counter
    /// ([`SceneBatch::culled`]).
    pub fn occluded(&self) -> usize {
        self.occluded
    }

    /// Takes the pyramid from the viewport-driven occlusion feed when the
    /// frame's tile-grid readback has landed, as the state to stage through
    /// [`SceneBatch::set_occlusion`].
    ///
    /// Never blocks: `Poll` resolves only callbacks for work that already
    /// finished, so a readback still on the GPU returns `None` and the frame
    /// culls nothing instead of pacing itself around the readback (the
    /// one-frame-latency contract in [`crate::occlusion`]). `None` as well
    /// when the id has no prepared entry yet, or nothing was requested.
    pub(crate) fn poll_occlusion_feed(
        &self,
        device: &wgpu::Device,
        resources: &mut CallbackResources,
    ) -> Option<super::occlusion::OcclusionCulling> {
        let all = resources.get_mut::<SceneResources>()?;
        let res = all.batches.get_mut(self.id.as_str())?;
        res.occlusion_feed.take(device, self.id.as_str())
    }

    /// Requests this frame's occlusion feed: a compute pass chain reduces the
    /// declared [`Depth`](super::pass::PassKind::Depth) or
    /// [`Occlusion`](super::pass::PassKind::Occlusion) pass's `Rgba16Float`
    /// or `R32Float` target into the packed level chain in `encoder` (which
    /// the caller
    /// submits with the frame, so the chain runs after the pass drew), with
    /// the chain's host map scheduled for that same submission.
    ///
    /// `view_proj`/`cam_pos`/`cam_forward` are the frame's camera: the
    /// pyramid is tested with the camera that captured it, which is what the
    /// one-frame delay pairs. `tile`/`bias` are the feed settings
    /// ([`crate::OcclusionFeedDesc`]'s clamped values). Returns `false` and
    /// goes quiet when the frame drew no opaque depth-tested geometry (the
    /// depth pass would draw nothing), declares no depth pass, the pass has
    /// no target, or the target cannot hold view depth: in every such case
    /// there is nothing fresh to feed.
    #[allow(clippy::too_many_arguments)] // id state + frame camera + feed settings, mirrors the call site
    pub(crate) fn encode_occlusion_readback(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resources: &mut CallbackResources,
        encoder: &mut wgpu::CommandEncoder,
        view_proj: Mat4,
        cam_pos: [f32; 3],
        cam_forward: [f32; 3],
        tile: u32,
        bias: f32,
    ) -> bool {
        let Some(all) = resources.get_mut::<SceneResources>() else {
            return false;
        };
        let Some(res) = all.batches.get_mut(self.id.as_str()) else {
            return false;
        };
        // No opaque depth-tested range means the depth pass would draw
        // nothing, and the frame's `flat` path skips the pass entirely, so
        // its target still holds an older frame. Reducing that would pair an
        // old depth with this frame's camera and could keep a stale cull
        // alive; retiring restarts the feed from a frame that really ran the
        // pass.
        if !self
            .ranges
            .iter()
            .any(|range| range.depth_test && !range.transparent)
        {
            res.occlusion_feed.retire(device);
            return false;
        }
        let spec = self.passes.iter().find(|spec| {
            matches!(spec.kind, super::pass::PassKind::Depth | super::pass::PassKind::Occlusion)
                && res.pass_targets.contains_key(&spec.target.name)
        });
        let target = spec.and_then(|spec| res.pass_targets.get(&spec.target.name));
        let Some(target) = target else {
            res.occlusion_feed.retire(device);
            return false;
        };
        if !matches!(
            target.format,
            super::pass::RenderTargetFormat::Rgba16Float
                | super::pass::RenderTargetFormat::R32Float
        ) || target.samples != 1
        {
            // `sync_pass_targets` only allocates `Rgba16Float` and `R32Float`
            // depth targets; anything else is a pass that cannot hold view
            // depth, and a multisampled target cannot be reduced as a
            // texture.
            res.occlusion_feed.retire(device);
            return false;
        }
        res.occlusion_feed.request(
            device,
            queue,
            encoder,
            &target.view,
            target.width.max(1),
            target.height.max(1),
            tile,
            bias,
            view_proj,
            cam_pos,
            cam_forward,
        )
    }

    /// Drops any occlusion readback still in flight: the feed is off this
    /// frame (the frame declares no depth pass, or the caller staged its own
    /// state), so nothing would consume it. Polls once without waiting.
    pub(crate) fn retire_occlusion_feed(
        &self,
        device: &wgpu::Device,
        resources: &mut CallbackResources,
    ) {
        let Some(all) = resources.get_mut::<SceneResources>() else {
            return;
        };
        let Some(res) = all.batches.get_mut(self.id.as_str()) else {
            return;
        };
        res.occlusion_feed.retire(device);
    }

    /// Stages the frame's particle systems. Each system's billboards draw in
    /// its declared pass (0 = the shared scene target), after that pass's
    /// mesh ranges, depth-tested and alpha-blended like transparent scene
    /// geometry; the simulation runs in a compute pass before the frame's
    /// geometry. `None` (default) skips the whole path.
    ///
    /// Invalid systems are dropped with a warning; sub-emitter links that
    /// cannot be ordered (a cycle) or point out of range are dropped the
    /// same way. The systems' simulation state (spawn cursor, fractional
    /// rate carry, frame counter) survives across frames in the batch's GPU
    /// resources keyed by system index and seed, so a per-frame rebuild of
    /// [`SceneBatch`] does not reset a running emitter.
    pub fn set_particles(&mut self, frame: Option<ParticleFrame>) {
        self.particle_order.clear();
        let Some(mut frame) = frame else {
            self.particles = None;
            return;
        };
        if !frame.dt.is_finite() || frame.dt <= 0.0 {
            log::warn!(
                "scene_batch[{}]: dropping particle frame (dt {})",
                self.id,
                frame.dt
            );
            self.particles = None;
            return;
        }
        for index in 0..frame.systems.len() {
            if let Err(err) = frame.systems[index].validate() {
                log::warn!(
                    "scene_batch[{}]: dropping particle system {index}: {err}",
                    self.id
                );
                frame.systems[index].capacity = 0;
            }
        }
        for index in 0..frame.systems.len() {
            if let Some(sub) = frame.systems[index].sub
                && !matches!(sub.mode, super::particles::SubEmitterMode::Off)
            {
                if sub.target >= frame.systems.len() || sub.target == index {
                    log::warn!(
                        "scene_batch[{}]: dropping particle sub-emitter of system {index} (target {})",
                        self.id,
                        sub.target
                    );
                    frame.systems[index].sub = None;
                }
            }
        }
        self.particle_order = particle_dispatch_order(&frame);
        self.particles = Some(frame);
    }

    /// Particle systems staged for this frame, if any.
    pub fn particles(&self) -> Option<&ParticleFrame> {
        self.particles.as_ref()
    }

    /// Builds (once) the fullscreen pipeline and sampler, and records which
    /// pass composites to the screen. GPU state lives on the entry, not the
    /// per-frame batch.
    fn sync_fullscreen(&self, device: &wgpu::Device, res: &mut SceneEntry) {
        res.presenting = self.passes.iter().find(|spec| spec.presents).cloned();
        let needs = self
            .passes
            .iter()
            .any(|spec| spec.kind == super::pass::PassKind::Fullscreen || spec.presents);
        if !needs {
            res.fullscreen = None;
            return;
        }
        if res.fullscreen.is_some() {
            return;
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("repame_view3d_fullscreen"),
            source: wgpu::ShaderSource::Wgsl(FULLSCREEN_SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("repame_view3d_fullscreen_bgl"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("repame_view3d_fullscreen_pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let make_pipeline = |blend: Option<wgpu::BlendState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("repame_view3d_fullscreen"),
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
                        format: res.key.0,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24PlusStencil8,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline = make_pipeline(None);
        let overlay_pipeline = make_pipeline(Some(wgpu::BlendState::ALPHA_BLENDING));
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("repame_view3d_fullscreen_smp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        res.fullscreen = Some(FullscreenState {
            pipeline,
            overlay_pipeline,
            sampler,
            binds: HashMap::new(),
            layout,
            tints: HashMap::new(),
            present: None,
            present_tint: None,
        });
    }

    /// Builds the fullscreen bind groups this frame's passes need, so the
    /// pass execution and the present composite can share them.
    fn warm_fullscreen_binds(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        res: &mut SceneEntry,
    ) {
        for spec in &self.passes {
            if spec.kind == super::pass::PassKind::Fullscreen
                && let Some(source) = spec.source.as_deref()
            {
                SceneBatch::fullscreen_bind(device, queue, res, source);
            }
        }
        let Some(spec) = res.presenting.clone() else {
            if let Some(state) = res.fullscreen.as_mut() {
                state.present = None;
            }
            return;
        };
        let view = res
            .pass_targets
            .get(&spec.target.name)
            .map(|target| target.view.clone());
        let Some(state) = res.fullscreen.as_mut() else {
            return;
        };
        let Some(view) = view else {
            state.present = None;
            return;
        };
        let buffer = state.present_tint.get_or_insert_with(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("repame_view3d_present_tint"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(&spec.overlay));
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("repame_view3d_present_bg"),
            layout: &state.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&state.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer.as_entire_binding(),
                },
            ],
        });
        state.present = Some((spec.target.name.clone(), bind));
    }

    /// Bind group sampling `source`, cached per target name.
    fn fullscreen_bind(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        res: &mut SceneEntry,
        source: &str,
    ) -> Option<wgpu::BindGroup> {
        let state = res.fullscreen.as_mut()?;
        let existing = state.binds.get(source).cloned();
        if let Some(bind) = existing {
            return Some(bind);
        }
        let view = res.pass_targets.get(source)?.view.clone();
        let buffer = state.tints.entry(source.to_string()).or_insert_with(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("repame_view3d_fullscreen_tint"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[1.0f32; 4]));
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("repame_view3d_fullscreen_bg"),
            layout: &state.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&state.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer.as_entire_binding(),
                },
            ],
        });
        state.binds.insert(source.to_string(), bind.clone());
        Some(bind)
    }

    /// Builds the post-process shader, bind layout and sampler when the frame
    /// declares any post stage, and one pipeline per distinct target format.
    fn sync_post(&self, device: &wgpu::Device, res: &mut SceneEntry) {
        let needs = self.passes.iter().any(|spec| spec.post.is_some());
        if !needs {
            res.post = None;
            return;
        }
        if res.post.is_some() {
            return;
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("repame_view3d_post"),
            source: wgpu::ShaderSource::Wgsl(POST_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("repame_view3d_post_bgl"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("repame_view3d_post_smp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let dummy_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("repame_view3d_post_dummy"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_view = dummy_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut pipelines = HashMap::new();
        for spec in self.passes.iter().filter(|spec| spec.post.is_some()) {
            let format = pass_format(spec.target.format);
            pipelines
                .entry(format)
                .or_insert_with(|| make_post_pipeline(device, &module, &layout, format));
        }
        res.post = Some(PostState {
            layout,
            sampler,
            pipelines,
            dummy_view,
            _dummy_texture: dummy_texture,
            binds: HashMap::new(),
            params: Vec::new(),
        });
    }

    /// Builds this frame's post bind groups: source view, shared sampler, the
    /// stage's parameter uniform, and the aux view (or the 1x1 dummy).
    fn warm_post_binds(&self, device: &wgpu::Device, queue: &wgpu::Queue, res: &mut SceneEntry) {
        let Some(dummy) = res.post.as_ref().map(|post| post.dummy_view.clone()) else {
            return;
        };
        let prepared: Vec<(u32, wgpu::TextureView, [f32; 4], wgpu::TextureView, PostStage)> = self
            .passes
            .iter()
            .filter_map(|spec| {
                let stage = spec.post?;
                let source = spec.source.as_deref()?;
                let src = res.pass_targets.get(source)?;
                let texel = [
                    1.0 / src.width.max(1) as f32,
                    1.0 / src.height.max(1) as f32,
                    src.width as f32,
                    src.height as f32,
                ];
                let aux = spec
                    .aux
                    .as_deref()
                    .and_then(|name| res.pass_targets.get(name))
                    .map(|target| target.view.clone())
                    .unwrap_or_else(|| dummy.clone());
                Some((spec.index, src.view.clone(), texel, aux, stage))
            })
            .collect();
        let Some(post) = res.post.as_mut() else {
            return;
        };
        // Camera state the fog stage reads: the same snapshot the frame
        // uniforms use, with the field of view taken from the projection
        // matrix (pp11 = 1 / tan(half fov y)).
        let view_proj = glam::Mat4::from_cols_array_2d(&self.camera.view_proj);
        let tan_half_fov = 1.0 / view_proj.y_axis.y.max(1e-6);
        let forward = glam::Vec3::from(self.camera.cam_forward);
        let forward = if forward.length_squared() > 1e-8 {
            forward.normalize()
        } else {
            glam::Vec3::NEG_Z
        };
        let mut right = forward.cross(glam::Vec3::Y);
        if right.length_squared() < 1e-8 {
            right = glam::Vec3::X;
        }
        let right = right.normalize();
        let light_dir = glam::Vec3::from(self.camera.light_dir);
        let fog_color = self
            .post
            .and_then(|post| post.fog)
            .map(|fog| [fog.color[0], fog.color[1], fog.color[2], 0.0])
            .unwrap_or([0.0; 4]);
        post.binds.clear();
        post.params.clear();
        for (index, src_view, texel, aux_view, stage) in prepared {
            let uniform = PostUniform {
                params: stage.params,
                params2: stage.params2,
                texel,
                stage: [stage.effect.mode(), 0.0, 0.0, 0.0],
                cam_pos: [
                    self.camera.cam_pos[0],
                    self.camera.cam_pos[1],
                    self.camera.cam_pos[2],
                    tan_half_fov,
                ],
                cam_forward: [
                    forward.x,
                    forward.y,
                    forward.z,
                    texel[2].max(1.0) / texel[3].max(1.0),
                ],
                cam_right: [right.x, right.y, right.z, 0.0],
                light_dir: [light_dir.x, light_dir.y, light_dir.z, 0.0],
                light_color: [
                    self.camera.light_color[0],
                    self.camera.light_color[1],
                    self.camera.light_color[2],
                    self.camera.diffuse,
                ],
                fog_color,
            };
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("repame_view3d_post_params"),
                size: size_of::<PostUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&[uniform]));
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("repame_view3d_post_bg"),
                layout: &post.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&post.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&aux_view),
                    },
                ],
            });
            post.binds.insert(index, bind);
            post.params.push(buffer);
        }
    }

    /// Encodes a copy of a declared pass target into `encoder`, returning
    /// the buffer and its row layout. Games drive this inside their own
    /// frame and map the buffer later; [`SceneBatch::map_pass_readback`]
    /// wraps the blocking path for tools and tests.
    pub fn encode_pass_readback(
        &self,
        device: &wgpu::Device,
        resources: &repose_render_wgpu::CallbackResources,
        name: &str,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Option<(wgpu::Buffer, u32, u32)> {
        let all = resources.get::<SceneResources>()?;
        let entry = all.batches.get(self.id.as_str())?;
        let target = entry.pass_targets.get(name)?;
        if target.format.is_depth() || target.samples != 1 {
            return None;
        }
        let (width, height) = (target.width.max(1), target.height.max(1));
        // Rows must be a multiple of COPY_BYTES_PER_ROW_ALIGNMENT (256); the
        // stride is in texels, so a wide texel pads to fewer texels.
        let texel_bytes = target.format.bytes_per_texel() as usize;
        let padded = (width as usize * texel_bytes).div_ceil(256) * 256;
        let bytes_per_row = padded;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("repame_view3d_pass_readback"),
            size: (bytes_per_row * height as usize) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row as u32),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        Some((buffer, width, height))
    }

    /// Maps a buffer from [`SceneBatch::encode_pass_readback`] and returns
    /// tight RGBA8 rows (4 bytes per texel). Blocks until the queue drains.
    pub fn map_pass_readback(
        &self,
        device: &wgpu::Device,
        buffer: &wgpu::Buffer,
        width: u32,
        height: u32,
    ) -> Option<Vec<u8>> {
        self.map_pass_readback_texels(device, buffer, width, height, 4)
    }

    /// [`SceneBatch::map_pass_readback`] for targets whose texels are not 4
    /// bytes (`bytes_per_texel` from
    /// [`RenderTargetFormat::bytes_per_texel`](crate::RenderTargetFormat::bytes_per_texel)),
    /// e.g. the `Rgba16Float` depth pass feeding
    /// [`DepthPyramid::from_tiled_bytes`](crate::DepthPyramid::from_tiled_bytes).
    /// Rows come back tight (no copy-alignment padding).
    pub fn map_pass_readback_texels(
        &self,
        device: &wgpu::Device,
        buffer: &wgpu::Buffer,
        width: u32,
        height: u32,
        bytes_per_texel: usize,
    ) -> Option<Vec<u8>> {
        let bytes_per_texel = bytes_per_texel.max(1);
        let bytes_per_row =
            (width as usize * bytes_per_texel).div_ceil(256) * 256;
        let (tx, rx) = std::sync::mpsc::channel();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        rx.recv().ok()?.ok()?;
        let mapped = buffer.slice(..).get_mapped_range().ok()?;
        let mut out = Vec::with_capacity(width as usize * height as usize * bytes_per_texel);
        for row in mapped.chunks_exact(bytes_per_row) {
            out.extend_from_slice(&row[..width as usize * bytes_per_texel]);
        }
        drop(mapped);
        buffer.unmap();
        Some(out)
    }

    /// Material programs registered this frame, in slot order (slot 0 is the
    /// legacy path, so `materials[i]` serves slot `i + 1`).
    pub fn materials(&self) -> &[MaterialEntry] {
        &self.materials
    }

    /// Append one GPU-skinned draw: bind-pose geometry plus the sampled
    /// palette. The batch uploads bind verts once and the palette into
    /// the per-frame uniform; the vertex shader blends in hardware.
    ///
    /// Validation mirrors [`push_group`](SceneBatch::push_group) (lengths,
    /// index range, page, alpha) plus skin-specific checks: joints/weights
    /// must match positions when present, joint slots past the mesh's
    /// joint count clamp per vertex (warned once per draw, never a panic),
    /// and meshes past [`MAX_SKIN_JOINTS`](crate::MAX_SKIN_JOINTS) drop
    /// (caller falls back to CPU [`pose`](crate::SkinnedMesh::pose)).
    /// Degenerate triangles cull tri-by-tri like static groups.
    ///
    /// On a device staged as [`GpuSkinning::Unsupported`]
    /// ([`SceneBatch::set_skinning`]) the draw drops here with a one-time
    /// warning naming the missing `VERTEX_STORAGE` flag: there is no
    /// palette binding to read, and the caller CPU-poses instead.
    pub fn push_skinned(&mut self, draw: &SkinnedDraw) {
        self.push_skinned_inner(draw, 0);
    }

    /// [`SceneBatch::push_skinned`] for a declared pass, mirroring
    /// [`SceneBatch::push_group_for_pass`], so a post-process scene pass
    /// owns its skinned draws instead of the shared scene target.
    pub fn push_skinned_for_pass(&mut self, draw: &SkinnedDraw, pass_index: u32) {
        self.push_skinned_inner(draw, pass_index);
    }

    fn push_skinned_inner(&mut self, draw: &SkinnedDraw, pass: u32) {
        // One latch for the process: the viewport rebuilds its batch every
        // frame, so a per-batch latch would log every frame.
        static SKIN_UNSUPPORTED_WARNED: AtomicBool = AtomicBool::new(false);
        if !self.skinning.is_supported() {
            if !SKIN_UNSUPPORTED_WARNED.swap(true, Ordering::Relaxed) {
                log::warn!(
                    "scene_batch[{}]: dropping GPU-skinned draws (device lacks \
                     DownlevelFlags::VERTEX_STORAGE, no storage buffer can bind in the \
                     vertex stage); CPU-pose them via SkinnedMesh::pose",
                    self.id
                );
            }
            return;
        }
        let mesh = &draw.mesh;
        let n = mesh.positions.len();
        if mesh.is_empty() {
            return;
        }
        if mesh.positions.len() != mesh.colors.len() {
            log::warn!(
                "scene_batch[{}]: dropping skinned draw ({} positions vs {} colors)",
                self.id,
                mesh.positions.len(),
                mesh.colors.len()
            );
            return;
        }
        if !mesh.normals.is_empty() && mesh.normals.len() != n {
            log::warn!(
                "scene_batch[{}]: dropping skinned draw (normals mismatch)",
                self.id
            );
            return;
        }
        if !mesh.uvs.is_empty() && mesh.uvs.len() != n {
            log::warn!(
                "scene_batch[{}]: dropping skinned draw (uvs mismatch)",
                self.id
            );
            return;
        }
        if mesh.joints.len() != n || mesh.weights.len() != n {
            log::warn!(
                "scene_batch[{}]: dropping skinned draw (skin weights len {} vs {} verts)",
                self.id,
                mesh.joints.len(),
                n
            );
            return;
        }
        if draw.joint_count > crate::MAX_SKIN_JOINTS {
            log::warn!(
                "scene_batch[{}]: dropping skinned draw ({} joints past the cap)",
                self.id,
                draw.joint_count
            );
            return;
        }
        if mesh.indices.iter().any(|i| (*i as usize) >= n) {
            log::warn!(
                "scene_batch[{}]: dropping skinned draw (index out of range)",
                self.id
            );
            return;
        }
        if !mesh.uvs.is_empty() && mesh.texture_page >= self.desc.layers {
            log::warn!(
                "scene_batch[{}]: dropping skinned draw (page {} >= {} layers)",
                self.id,
                mesh.texture_page,
                self.desc.layers
            );
            return;
        }
        let mut clamped_joints = mesh.joints.clone();
        let cap = draw.joint_count.saturating_sub(1);
        let mut clamped = false;
        for j in clamped_joints.iter_mut() {
            for slot in j.iter_mut() {
                if (*slot as usize) > cap {
                    *slot = cap as u16;
                    clamped = true;
                }
            }
        }
        if clamped {
            log::warn!(
                "scene_batch[{}]: clamping skinned joint slots to {} ({})",
                self.id,
                cap,
                mesh.name
            );
        }
        self.skinned.push(SkinnedPending {
            positions: mesh.positions.clone(),
            colors: mesh.colors.clone(),
            normals: mesh.normals.clone(),
            uvs: mesh.uvs.clone(),
            joints: clamped_joints,
            weights: mesh.weights.clone(),
            palette: draw.palette.iter().map(|m| m.to_cols_array_2d()).collect(),
            texture_page: mesh.texture_page,
            material: mesh.material,
            center: None,
            indices: cull_degenerate(&mesh.positions, &mesh.indices),
            transparent: mesh.transparent,
            alpha: mesh.alpha.clamp(0.0, 1.0),
            alpha_cutoff: mesh.alpha_cutoff.clamp(0.0, 1.0),
            depth_test: mesh.depth_test,
            pass,
            hz_culled: false,
        });
        let pending = self.skinned.last_mut().expect("just pushed");
        pending.center = Self::skinned_bounds(pending).map(|(min, max)| {
            [
                (min[0] + max[0]) * 0.5,
                (min[1] + max[1]) * 0.5,
                (min[2] + max[2]) * 0.5,
            ]
        });
    }

    /// Skinned draws staged (pre-`finish`).
    pub fn skinned_count(&self) -> usize {
        self.skinned.len()
    }

    /// Extract the six frustum planes (world space, normalized) from a
    /// view-projection matrix. Row-major extraction on the row vectors of
    /// the column-major `Mat4`: `left = row3 + row0`, etc. (Gribb/Hartmann).
    ///
    /// Depth-convention note: the camera bakes `OPENGL_TO_WGPU`, so NDC
    /// depth is `[0, 1]` (D3D-style), not OpenGL `[-1, 1]`. The side planes
    /// are convention-free (`|x|,|y| <= w`), but near/far differ: near is
    /// `row2` (`z >= 0`), far is `row3 - row2` (`z <= w`). Using the
    /// OpenGL `row3 + row2` near plane here would accept everything (it
    /// tests `w + z >= 0`, always true post-remap), the tests pin the far
    /// plane with a beyond-FAR slab, which only culls with this form.
    fn frustum_planes(view_proj: &Mat4) -> [[f32; 4]; 6] {
        let m = view_proj.to_cols_array_2d();
        // Rows of the row-major view: row[i] = (m[0][i], m[1][i], ...).
        let row = |i: usize| [m[0][i], m[1][i], m[2][i], m[3][i]];
        let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
        let sub = |a: [f32; 4], b: [f32; 4]| [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]];
        let add = |a: [f32; 4], b: [f32; 4]| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
        let mut planes = [
            add(r3, r0), // left
            sub(r3, r0), // right
            add(r3, r1), // bottom
            sub(r3, r1), // top
            r2,          // near (D3D-style: z >= 0)
            sub(r3, r2), // far (z <= w)
        ];
        for p in &mut planes {
            let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            if len > 1e-12 {
                p[0] /= len;
                p[1] /= len;
                p[2] /= len;
                p[3] /= len;
            }
        }
        planes
    }

    /// Flatten pending groups into the draw buffers: opaque first
    /// (depth-tested, then flat overlays, submission order decides ties),
    /// then transparent back-to-front (camera distance of the AABB center;
    /// groups without bounds sort nearest). Frustum culling drops fully
    /// outside groups before flattening (`depth_test = false` overlays are
    /// never culled, gizmos must draw even off-screen). Split from
    /// [`SceneBatch::push_group`] so the viewport payload, which rebuilds the batch
    /// per frame, shares the path.
    ///
    /// Flat vertices (no normals) carry a dummy up-normal and `lit = 0.0`
    /// so the shader passes their baked color through untouched.
    /// Untextured vertices carry a dummy uv and `tex_mix = 0.0` so the
    /// shader skips the sample. Material rides every vertex but only lit
    /// fragments read it.
    pub fn finish(&mut self) {
        self.verts.clear();
        self.indices.clear();
        self.ranges.clear();
        self.occlusion_candidates.clear();
        self.culled = 0;
        self.occluded = 0;
        self.stats = super::stats::FrameStats::new();
        self.stats.culled_groups = self.culled as u32;
        self.stats.passes = 1 + self.passes.len() as u32;
        let planes = Self::frustum_planes(&self.view_proj);
        let culling = self.view_proj != Mat4::IDENTITY;
        // Same-frame phase two keeps phase-one rejects instead of dropping
        // them, so the encode can re-test them on the GPU against this
        // frame's chain; it needs the depth pass that chain reads, and only
        // the scene's groups become candidates (the shared scene target or a
        // declared scene-color pass, the same notion the scheduler and the
        // phase-two draw use).
        let phase_two = self.occlusion_feed.is_some_and(|feed| feed.phase_two)
            && self.occlusion.is_some()
            && culling
            && self
                .passes
                .iter()
                .any(|spec| matches!(spec.kind, super::pass::PassKind::Depth | super::pass::PassKind::Occlusion));
        let scene_passes: Vec<u32> = self
            .passes
            .iter()
            .filter(|spec| spec.target.semantics == super::pass::TargetSemantics::SceneColor)
            .map(|spec| spec.index)
            .collect();
        self.occlusion_phase_two = phase_two;
        let eye = glam::Vec3::from(self.camera_pos);
        // Stable partition: opaque keep submission order; transparent sort
        // back-to-front by AABB-center distance (descending). `sort_by`
        // (stable) on a pre-partitioned vec keeps both contracts.
        let mut opaque: Vec<Pending> = Vec::with_capacity(self.pending.len());
        let mut transparent: Vec<(Pending, f32)> = Vec::new();
        for mut g in self.pending.drain(..) {
            // Two-phase cull, only when the camera is real and
            // `depth_test = true` (overlays are never culled: gizmos must
            // draw even off-screen). Phase 1 is the hierarchical-Z test
            // against the previous frame's depth; phase 2 is the existing
            // vertex-exact frustum test. Phase 1 only runs over opaque
            // groups: the depth pass replays opaque depth-tested ranges, so
            // nothing vouches for a transparent group's pixels there.
            if culling && g.depth_test {
                if !g.transparent
                    && let Some(occlusion) = self.occlusion.as_ref()
                    && let Some((min, max)) = g.bounds
                    && occlusion.occluded(min, max)
                {
                    self.occluded += 1;
                    if phase_two && (g.pass == 0 || scene_passes.contains(&g.pass)) {
                        g.hz_culled = true;
                    } else {
                        continue;
                    }
                }
                if !g.hz_culled
                    && Self::group_outside(&planes, &g.positions)
                {
                    self.culled += 1;
                    self.stats.culled_groups = self.culled as u32;
                    continue;
                }
            }
            if g.transparent {
                let d = g
                    .center
                    .map(|c| (glam::Vec3::from(c) - eye).length_squared());
                // `None` (degenerate bounds) sorts nearest: it draws last,
                // on top - drawn last when bounds are degenerate.
                transparent.push((g, d.unwrap_or(-1.0)));
            } else {
                opaque.push(g);
            }
        }
        transparent.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        opaque.sort_by_key(|g| !g.depth_test);
        let ordered: Vec<Pending> = opaque
            .into_iter()
            .chain(transparent.into_iter().map(|(g, _)| g))
            .collect();
        for g in ordered {
            if g.indices.len() < 3 {
                continue; // all triangles culled as degenerate: draw nothing
            }
            let base = self.verts.len() as u32;
            let lit = !g.normals.is_empty();
            let textured = !g.uvs.is_empty();
            let page = g.texture_page as f32;
            let mat = g.material;
            self.verts
                .extend(
                    g.positions
                        .iter()
                        .zip(g.colors.iter())
                        .enumerate()
                        .map(|(i, (p, c))| Vert {
                            pos: *p,
                            color: *c,
                            normal: if lit { g.normals[i] } else { [0.0, 1.0, 0.0] },
                            lit: if lit { 1.0 } else { 0.0 },
                            uv: if textured { g.uvs[i] } else { [0.0, 0.0] },
                            tex_mix: if textured { 1.0 } else { 0.0 },
                            page,
                            alpha: g.alpha,
                            cutoff: g.alpha_cutoff,
                            metallic: mat.metallic.clamp(0.0, 1.0),
                            roughness: if mat.roughness.is_finite() {
                                mat.roughness.clamp(0.0, 1.0)
                            } else {
                                1.0
                            },
                            emissive: mat.emissive,
                            joints: [0; 4],
                            weights: [0.0; 4],
                            skin_mix: 0.0,
                            skin_base: 0.0,
                        }),
                );
            let start = self.indices.len() as u32;
            self.indices.extend(g.indices.iter().map(|i| i + base));
            let end = self.indices.len() as u32;
            let hz_culled = match (g.hz_culled, g.bounds) {
                (true, Some((min, max))) => {
                    let index = self.occlusion_candidates.len() as u32;
                    self.occlusion_candidates.push(HizCandidate {
                        min,
                        _pad0: 0.0,
                        max,
                        _pad1: 0.0,
                        args: [end - start, start, 0, 1],
                    });
                    Some(index)
                }
                _ => None,
            };
            self.ranges.push(DrawRange {
                index_start: start,
                index_end: end,
                depth_test: g.depth_test,
                transparent: g.transparent,
                material: g.material_slot,
                pass: g.pass,
                hz_culled,
            });
        }
        self.palette.clear();
        let eye = glam::Vec3::from(self.camera_pos);
        let mut skin_opaque: Vec<SkinnedPending> = Vec::new();
        let mut skin_transparent: Vec<(SkinnedPending, f32)> = Vec::new();
        for mut s in self.skinned.drain(..) {
            if s.indices.len() < 3 {
                continue;
            }
            if culling
                && s.depth_test
                && !s.transparent
                && !s.hz_culled
                && Self::occlusion_rejects(self.occlusion.as_ref(), &s)
            {
                self.occluded += 1;
                if phase_two && (s.pass == 0 || scene_passes.contains(&s.pass)) {
                    s.hz_culled = true;
                } else {
                    continue;
                }
            }
            if culling && s.depth_test && Self::skinned_outside(&planes, &s) {
                self.culled += 1;
                continue;
            }
            if s.transparent {
                let d = s
                    .center
                    .map(|c| (glam::Vec3::from(c) - eye).length_squared());
                skin_transparent.push((s, d.unwrap_or(-1.0)));
            } else {
                skin_opaque.push(s);
            }
        }
        skin_transparent.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for mut s in skin_opaque
            .into_iter()
            .chain(skin_transparent.into_iter().map(|(s, _)| s))
        {
            if s.indices.len() < 3 {
                continue;
            }
            if culling
                && s.depth_test
                && !s.transparent
                && !s.hz_culled
                && Self::occlusion_rejects(self.occlusion.as_ref(), &s)
            {
                self.occluded += 1;
                if phase_two && (s.pass == 0 || scene_passes.contains(&s.pass)) {
                    s.hz_culled = true;
                } else {
                    continue;
                }
            }
            if culling && s.depth_test && Self::skinned_outside(&planes, &s) {
                self.culled += 1;
                continue;
            }
            let base = self.verts.len() as u32;
            let skin_base = self.palette.len() as f32;
            self.palette.extend_from_slice(&s.palette);
            let lit = !s.normals.is_empty();
            let textured = !s.uvs.is_empty();
            let page = s.texture_page as f32;
            let mat = s.material;
            let transparent = s.transparent;
            let depth_test = s.depth_test;
            self.verts
                .extend(
                    s.positions
                        .iter()
                        .zip(s.colors.iter())
                        .enumerate()
                        .map(|(i, (p, c))| Vert {
                            pos: *p,
                            color: *c,
                            normal: if lit { s.normals[i] } else { [0.0, 1.0, 0.0] },
                            lit: if lit { 1.0 } else { 0.0 },
                            uv: if textured { s.uvs[i] } else { [0.0, 0.0] },
                            tex_mix: if textured { 1.0 } else { 0.0 },
                            page,
                            alpha: s.alpha,
                            cutoff: s.alpha_cutoff,
                            metallic: mat.metallic.clamp(0.0, 1.0),
                            roughness: if mat.roughness.is_finite() {
                                mat.roughness.clamp(0.0, 1.0)
                            } else {
                                1.0
                            },
                            emissive: mat.emissive,
                            joints: {
                                let j = s.joints[i];
                                [j[0] as u32, j[1] as u32, j[2] as u32, j[3] as u32]
                            },
                            weights: s.weights[i],
                            skin_mix: 1.0,
                            skin_base,
                        }),
                );
            let start = self.indices.len() as u32;
            self.indices.extend(s.indices.iter().map(|i| i + base));
            let end = self.indices.len() as u32;
            let hz_culled = match s.hz_culled {
                true => Self::skinned_bounds(&s).map(|(min, max)| {
                    let index = self.occlusion_candidates.len() as u32;
                    self.occlusion_candidates.push(HizCandidate {
                        min,
                        _pad0: 0.0,
                        max,
                        _pad1: 0.0,
                        args: [end - start, start, 0, 1],
                    });
                    index
                }),
                false => None,
            };
            self.ranges.push(DrawRange {
                index_start: start,
                index_end: end,
                depth_test,
                transparent,
                material: 0,
                pass: s.pass,
                hz_culled,
            });
        }
    }

    /// True when every vertex of `positions` sits outside one frustum
    /// plane. Vertex-exact (no AABB approximation): costs one walk per
    /// group per frame, only when culling is armed (real camera). Groups
    /// are small in practice (chunked terrain splits by material, agents
    /// are single meshes), and a wrongly-culled group is a missing
    /// object, so test every vertex here.
    fn group_outside(planes: &[[f32; 4]; 6], positions: &[[f32; 3]]) -> bool {
        if positions.is_empty() {
            return false;
        }
        for p in planes {
            let mut all_out = true;
            for v in positions {
                if p[0] * v[0] + p[1] * v[1] + p[2] * v[2] + p[3] >= 0.0 {
                    all_out = false;
                    break;
                }
            }
            if all_out {
                return true;
            }
        }
        false
    }

    /// World-space AABB of a skinned draw. Bind positions are model-local;
    /// the world transform lives only in the palette, so each vertex is
    /// placed by the palette matrices of the joints that actually weight it.
    fn skinned_bounds(s: &SkinnedPending) -> Option<([f32; 3], [f32; 3])> {
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        let mut any = false;
        for (i, p) in s.positions.iter().enumerate() {
            for slot in 0..4 {
                if s.weights.get(i).map_or(true, |w| w[slot] <= 0.0) {
                    continue;
                }
                let Some(matrix) = s
                    .joints
                    .get(i)
                    .and_then(|j| s.palette.get(usize::from(j[slot])))
                else {
                    continue;
                };
                let v = Mat4::from_cols_array_2d(matrix) * glam::Vec4::new(p[0], p[1], p[2], 1.0);
                if !v.is_finite() {
                    continue;
                }
                let v = v.truncate();
                min = [min[0].min(v.x), min[1].min(v.y), min[2].min(v.z)];
                max = [max[0].max(v.x), max[1].max(v.y), max[2].max(v.z)];
                any = true;
            }
        }
        any.then_some((min, max))
    }

    /// Phase-1 reject for a skinned draw: its posed world AABB against the
    /// staged occlusion state. Callers only run it for opaque depth-tested
    /// draws: the depth pass never replays transparent ranges.
    fn occlusion_rejects(
        occlusion: Option<&super::occlusion::OcclusionCulling>,
        s: &SkinnedPending,
    ) -> bool {
        let Some(occlusion) = occlusion else {
            return false;
        };
        match Self::skinned_bounds(s) {
            Some((min, max)) => occlusion.occluded(min, max),
            None => false,
        }
    }

    /// Frustum reject for a skinned draw. Consumes the bounds already
    /// computed for transparent sorting rather than re-deriving them.
    fn skinned_outside(planes: &[[f32; 4]; 6], s: &SkinnedPending) -> bool {
        let Some((min, max)) = Self::skinned_bounds(s) else {
            return false;
        };
        Self::aabb_outside(planes, min, max)
    }

    /// Positive-vertex AABB reject: a box lies fully outside one plane only
    /// when its corner most in that plane's direction is still behind it.
    fn aabb_outside(planes: &[[f32; 4]; 6], min: [f32; 3], max: [f32; 3]) -> bool {
        planes.iter().any(|p| {
            let pick = |n: f32, lo: f32, hi: f32| if n >= 0.0 { hi } else { lo };
            let x = pick(p[0], min[0], max[0]);
            let y = pick(p[1], min[1], max[1]);
            let z = pick(p[2], min[2], max[2]);
            p[0] * x + p[1] * y + p[2] * z + p[3] < 0.0
        })
    }

    /// Shadow-map edge armed for the next `prepare` (0 = disabled).
    /// Tracks the last [`SceneBatch::set_shadow`] `Some` desc; `None`
    /// clears it back to 0.
    fn shadow_tex_size(&self) -> u32 {
        if self.camera.shadow_params[1] > 0.5 {
            let t = self.camera.shadow_texel[0];
            if t.is_finite() && t > 0.0 {
                (1.0 / t).round().clamp(64.0, 4096.0) as u32
            } else {
                1024
            }
        } else {
            0
        }
    }

    pub(crate) fn ensure_resources(
        &self,
        device: &wgpu::Device,
        screen: &ScreenDescriptor,
        resources: &mut CallbackResources,
    ) {
        let shadow_size = self.shadow_tex_size();
        let (cascade_size, cascade_count) = match self.cascade_desc {
            Some(d) => (d.clamped_size(), d.clamped_count() as u32),
            None => (0, 0),
        };
        let cube_size = match self.point_caster {
            Some(i) => self
                .staged_points
                .get(i)
                .map(|p| p.clamped_size())
                .unwrap_or(0),
            None => 0,
        };
        let skin_slots = self
            .palette
            .len()
            .next_power_of_two()
            .max(crate::MAX_SKIN_JOINTS);
        // The skin variant is part of the cache key: a device swap (or a
        // capability change) must never reuse pipelines built for the other
        // mode, even when every other dimension matches.
        let skin_supported = self.skinning.is_supported();
        let key = (
            screen.target_format,
            screen.sample_count,
            self.desc.layer_size,
            self.desc.layers,
            self.desc.filter as u32,
            self.desc.mip_levels(),
            shadow_size,
            cascade_size,
            cascade_count,
            cube_size,
            skin_slots,
            skin_supported,
        );
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
                "scene_batch[{}]: rebuilding pipeline/texture (format/sample/desc/shadow/rig/skin changed); texture contents dropped, re-upload required",
                self.id
            );
        }
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("repame_view3d"),
            source: wgpu::ShaderSource::Wgsl(scene_shader(self.skinning).into()),
        });
        let camera = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("repame_view3d_camera"),
            size: size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (mag, min, mipmap) = match self.desc.filter {
            SceneFilter::Nearest => (
                wgpu::FilterMode::Nearest,
                wgpu::FilterMode::Nearest,
                wgpu::MipmapFilterMode::Nearest,
            ),
            SceneFilter::Linear => (
                wgpu::FilterMode::Linear,
                wgpu::FilterMode::Linear,
                wgpu::MipmapFilterMode::Linear,
            ),
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("repame_view3d_tex"),
            size: wgpu::Extent3d {
                width: self.desc.layer_size,
                height: self.desc.layer_size,
                depth_or_array_layers: self.desc.layers,
            },
            mip_level_count: self.desc.mip_levels(),
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let tex_view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("repame_view3d_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: mag,
            min_filter: min,
            mipmap_filter: mipmap,
            lod_min_clamp: 0.0,
            lod_max_clamp: sampler_lod_max_clamp(self.desc.mip_levels()),
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });
        // Binding 1 carries the joint palette and is visible to the vertex
        // stage, which `wgpu` gates on `DownlevelFlags::VERTEX_STORAGE`. A
        // device without the flag cannot construct this layout at all, so
        // the entry (and the matching shader declaration and bind group
        // entry) only exists when the capability does.
        let mut cam_entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        if skin_supported {
            cam_entries.push(wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        let cam_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("repame_view3d_cam_bgl"),
            entries: &cam_entries,
        });
        let skin_buffer = skin_supported.then(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("repame_view3d_skin"),
                size: (skin_slots * 64) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let cam_bind = camera_bind_group(
            device,
            "repame_view3d_cam_bg",
            &cam_layout,
            &camera,
            skin_buffer.as_ref(),
        );
        let page_levels = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("repame_view3d_page_levels"),
            size: u64::from(RESIDENT_PAGE_SLOTS) * 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tex_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("repame_view3d_tex_bgl"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    // One `vec4` per four pages, matching the WGSL
                    // `array<vec4<f32>, 64>` (fixed size: a uniform buffer
                    // cannot hold a runtime-sized array).
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            u64::from(RESIDENT_PAGE_SLOTS) * 16,
                        ),
                    },
                    count: None,
                },
            ],
        });
        let tex_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("repame_view3d_tex_bg"),
            layout: &tex_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: page_levels.as_entire_binding(),
                },
            ],
        });
        let shadow_edge = shadow_size.max(64);
        let shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("repame_view3d_shadow"),
            size: wgpu::Extent3d {
                width: shadow_edge,
                height: shadow_edge,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("repame_view3d_shadow_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 1.0,
            compare: Some(wgpu::CompareFunction::LessEqual),
            anisotropy_clamp: 1,
            border_color: None,
        });
        let cascade_edge = cascade_size.max(1);
        let cascade_layers = cascade_count.max(1);
        let cascade_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("repame_view3d_cascades"),
            size: wgpu::Extent3d {
                width: cascade_edge,
                height: cascade_edge,
                depth_or_array_layers: cascade_layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let cascade_view = cascade_tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let cube_edge = cube_size.max(1);
        let point_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("repame_view3d_point_cube"),
            size: wgpu::Extent3d {
                width: cube_edge,
                height: cube_edge,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let point_view = point_tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        let mk_comparison_sampler = |label: &'static str| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some(label),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                lod_min_clamp: 0.0,
                lod_max_clamp: 1.0,
                compare: Some(wgpu::CompareFunction::LessEqual),
                anisotropy_clamp: 1,
                border_color: None,
            })
        };
        let cascade_sampler = mk_comparison_sampler("repame_view3d_cascade_sampler");
        let point_sampler = mk_comparison_sampler("repame_view3d_point_sampler");
        let shadow_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("repame_view3d_shadow_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                probe_cube_entry(6),
                probe_cube_entry(7),
                probe_cube_entry(8),
                probe_cube_entry(9),
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        // Probe cubes start as 1x1x1 stand-ins; `sync_probes` rebuilds the
        // group with the frame's real cubes.
        let probe_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("repame_view3d_probe_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 1.0,
            anisotropy_clamp: 1,
            border_color: None,
            compare: None,
        });
        let mut probe_dummies = Vec::with_capacity(super::reflection::MAX_PROBES);
        let mut probe_dummy_views = Vec::with_capacity(super::reflection::MAX_PROBES);
        for _ in 0..super::reflection::MAX_PROBES {
            let dummy = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("repame_view3d_probe_dummy"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 6,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: screen.target_format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            probe_dummy_views.push(dummy.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::Cube),
                ..Default::default()
            }));
            probe_dummies.push(dummy);
        }
        let dummy_refs: [&wgpu::TextureView; super::reflection::MAX_PROBES] =
            std::array::from_fn(|i| &probe_dummy_views[i]);
        let shadow_bind = make_shadow_bind(
            device,
            &shadow_layout,
            &shadow_view,
            &shadow_sampler,
            &cascade_view,
            &cascade_sampler,
            &point_view,
            &point_sampler,
            &dummy_refs,
            &probe_sampler,
        );
        let material_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("repame_view3d_material_bgl"),
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
        let material_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("repame_view3d_material"),
            size: size_of::<MaterialUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let material_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("repame_view3d_material_bg"),
            layout: &material_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: material_buffer.as_entire_binding(),
            }],
        });
        let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("repame_view3d_pl"),
            bind_group_layouts: &[
                Some(&cam_layout),
                Some(&tex_layout),
                Some(&shadow_layout),
                Some(&material_layout),
            ],
            immediate_size: 0,
        });
        let buffers = [Some(self.vertex_buffer_layout())];

        let mk_with = |module: &wgpu::ShaderModule, depth_test: bool, transparent: bool| {
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("repame_view3d_pipeline"),
                layout: Some(&pipe_layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("vs_main"),
                    buffers: &buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: screen.target_format,
                        // Transparent pass blends source alpha over the
                        // scene (ghosts, glass, water); opaque replaces.
                        blend: Some(if transparent {
                            wgpu::BlendState::ALPHA_BLENDING
                        } else {
                            wgpu::BlendState::REPLACE
                        }),
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
                // Transparent ranges never write depth (overlapping ghosts
                // must not occlude each other) but still test it (a ghost
                // behind a wall stays behind the wall).
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24PlusStencil8,
                    depth_write_enabled: Some(!transparent),
                    depth_compare: Some(if depth_test {
                        wgpu::CompareFunction::LessEqual
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
                    count: SCENE_TARGET_SAMPLE_COUNT,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            });
            (pipeline, depth_test)
        };
        let mk = |_label: &'static str, depth_test: bool, transparent: bool| {
            mk_with(&shader, depth_test, transparent).0
        };
        let shadow_depth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("repame_view3d_shadow_depth"),
            source: wgpu::ShaderSource::Wgsl(shadow_depth_shader(self.skinning).into()),
        });
        let shadow_depth_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("repame_view3d_shadow_depth_pl"),
            bind_group_layouts: &[Some(&cam_layout)],
            immediate_size: 0,
        });
        let shadow_camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("repame_view3d_shadow_camera"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shadow_cam_bind = camera_bind_group(
            device,
            "repame_view3d_shadow_cam_bg",
            &cam_layout,
            &shadow_camera_buf,
            skin_buffer.as_ref(),
        );
        // One camera per cascade slice and per cube face: `write_buffer` is
        // applied at the start of the submission, so reusing a single buffer
        // across the loop's passes would give every pass the last write.
        let slice_cam = |label: &'static str| {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind = camera_bind_group(
                device,
                "repame_view3d_slice_cam_bg",
                &cam_layout,
                &buffer,
                skin_buffer.as_ref(),
            );
            (buffer, bind)
        };
        let cascade_cam_binds: Vec<(wgpu::Buffer, wgpu::BindGroup)> = (0..crate::MAX_CASCADES)
            .map(|_| slice_cam("repame_view3d_cascade_camera"))
            .collect();
        let point_cam_binds: Vec<(wgpu::Buffer, wgpu::BindGroup)> = (0..6)
            .map(|_| slice_cam("repame_view3d_point_camera"))
            .collect();
        let shadow_depth_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("repame_view3d_shadow_depth"),
                layout: Some(&shadow_depth_layout),
                vertex: wgpu::VertexState {
                    module: &shadow_depth_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: size_of::<Vert>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Uint32x4,
                                offset: 84,
                                shader_location: 1,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 100,
                                shader_location: 2,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32,
                                offset: 116,
                                shader_location: 3,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32,
                                offset: 120,
                                shader_location: 4,
                            },
                        ],
                    })],
                    compilation_options: Default::default(),
                },
                fragment: None,
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            });
        let dv_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("repame_view3d_dv_bgl"),
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
        let dv_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("repame_view3d_depth_velocity"),
            source: wgpu::ShaderSource::Wgsl(depth_velocity_shader(self.skinning).into()),
        });
        let dv_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("repame_view3d_dv_pl"),
            bind_group_layouts: &[Some(&cam_layout), Some(&dv_layout)],
            immediate_size: 0,
        });
        let depth_velocity_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("repame_view3d_dv_prev"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let depth_velocity_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("repame_view3d_dv_bg"),
            layout: &dv_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: depth_velocity_buffer.as_entire_binding(),
            }],
        });
        // The depth-kind passes write Rgba16Float, so their pipelines are
        // built for that format rather than the batch's, and use the same
        // remapped five-attribute layout the shadow depth pass uses. The
        // occlusion pass runs depth alone (`vs_depth`/`fs_depth`), without the
        // previous-frame matrix or velocity outputs.
        let mk_depth_pipeline = |label: &'static str,
                                 vs_entry: &'static str,
                                 fs_entry: &'static str,
                                 format: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&dv_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &dv_module,
                    entry_point: Some(vs_entry),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: size_of::<Vert>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Uint32x4,
                                offset: 84,
                                shader_location: 1,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 100,
                                shader_location: 2,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32,
                                offset: 116,
                                shader_location: 3,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32,
                                offset: 120,
                                shader_location: 4,
                            },
                        ],
                    })],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &dv_module,
                    entry_point: Some(fs_entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24PlusStencil8,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            })
        };
        // A variant per accepted depth format: the target's format and the
        // pipeline's attachment format must match at draw time.
        let depth_velocity_pipeline = mk_depth_pipeline(
            "repame_view3d_depth_velocity",
            "vs_main",
            "fs_main",
            wgpu::TextureFormat::Rgba16Float,
        );
        let depth_only_pipeline = mk_depth_pipeline(
            "repame_view3d_depth_only",
            "vs_depth",
            "fs_depth",
            wgpu::TextureFormat::Rgba16Float,
        );
        let depth_velocity_pipeline_r32f = mk_depth_pipeline(
            "repame_view3d_depth_velocity_r32f",
            "vs_main",
            "fs_main",
            wgpu::TextureFormat::R32Float,
        );
        let depth_only_pipeline_r32f = mk_depth_pipeline(
            "repame_view3d_depth_only_r32f",
            "vs_depth",
            "fs_depth",
            wgpu::TextureFormat::R32Float,
        );
        let entry = SceneEntry {
            key,
            pipeline_depth: mk("repame_view3d_depth", true, false),
            pipeline_flat: mk("repame_view3d_flat", false, false),
            pipeline_transparent: mk("repame_view3d_transparent", true, true),
            pipeline_transparent_flat: mk("repame_view3d_transparent_flat", false, true),
            shadow_depth_pipeline,
            depth_velocity_pipeline,
            depth_only_pipeline,
            depth_velocity_pipeline_r32f,
            depth_only_pipeline_r32f,
            depth_velocity_buffer,
            depth_velocity_bind,
            occlusion_feed: OcclusionFeed::default(),
            occlusion_feed_desc: None,
            occlusion_candidates: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("repame_view3d_occlusion_candidates"),
                size: std::mem::size_of::<HizCandidate>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            occlusion_args: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("repame_view3d_occlusion_args"),
                size: 20,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
                mapped_at_creation: false,
            }),
            occlusion_capacity: 1,
            occlusion_candidate_count: 0,
            occlusion_phase_two: false,
            post: None,
            material_pipelines: Vec::new(),
            pass_targets: HashMap::new(),
            pass_specs: Vec::new(),
            fullscreen: None,
            presenting: None,
            pipe_layout,
            shadow_cam_bind,
            shadow_camera_buf,
            cascade_cam_binds,
            point_cam_binds,
            shadow_bind,
            shadow_view,
            shadow_edge,
            shadow_layout,
            shadow_sampler,
            cascade_tex,
            cascade_view,
            cascade_edge,
            cascade_sampler,
            point_tex,
            point_view,
            point_edge: cube_edge,
            point_sampler,
            probe_sampler,
            _probe_dummies: probe_dummies,
            probe_dummy_views,
            point_faces: None,
            skin: self.skinning,
            skin_buffer,
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
            cam_layout,
            tex_layout,
            material_buffer,
            material_bind,
            material_layout,
            tex_bind,
            page_levels,
            texture,
            camera_mat: self.camera,
            last_ranges: Vec::new(),
            particles: None,
            probes: None,
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

    /// Upload camera + geometry + texture uploads after [`finish`].
    /// No-op when the id has no prepared entry (call
    /// [`ensure_resources`](Self::ensure_resources) first  -
    /// [`prepare_scene_with_id`] + the viewport do). Out-of-range or
    /// mis-sized uploads are dropped with a warning
    /// (sprite-batch contract).
    pub(crate) fn upload_all(
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
        // The shaders' per-page resident levels: one write covers every
        // textured draw this frame (scene, material and particle paths).
        queue.write_buffer(&res.page_levels, 0, bytemuck::cast_slice(&self.page_levels));
        self.sync_material_pipelines(device, queue, res);
        self.sync_particles(device, queue, res);
        self.sync_probes(device, queue, res);
        self.sync_pass_targets(device, res);
        self.sync_fullscreen(device, res);
        self.sync_post(device, res);
        self.warm_fullscreen_binds(device, queue, res);
        self.warm_post_binds(device, queue, res);
        let prev = self
            .post
            .and_then(|post| post.previous_view_proj)
            .unwrap_or_else(|| Mat4::from_cols_array_2d(&self.camera.view_proj));
        queue.write_buffer(
            &res.depth_velocity_buffer,
            0,
            bytemuck::cast_slice(&[prev.to_cols_array_2d()]),
        );
        res.pass_specs = self.passes.clone();
        let program = self.materials.first().map(|entry| entry.cache_key);
        if program.is_some() {
            queue.write_buffer(
                &res.material_buffer,
                0,
                bytemuck::cast_slice(&[self.material_uniform()]),
            );
        }
        res.camera_mat = self.camera;
        res.point_faces = self
            .point_caster
            .and_then(|i| self.staged_points.get(i))
            .and_then(|p| p.cube_faces());
        if !self.palette.is_empty()
            && let Some(skin) = &res.skin_buffer
        {
            queue.write_buffer(skin, 0, bytemuck::cast_slice(&self.palette));
        }
        if !self.verts.is_empty() {
            queue.write_buffer(&res.verts, 0, bytemuck::cast_slice(&self.verts));
        }
        if !self.indices.is_empty() {
            queue.write_buffer(&res.indices, 0, bytemuck::cast_slice(&self.indices));
        }
        // Texture uploads straight into array layers. `desc` lives on the
        // CPU batch (not the GPU entry) so validation matches what the
        // game packed against, even right after a pipeline rebuild.
        for up in &self.uploads {
            let expected = up.w as usize * up.h as usize * 4;
            if up.page >= self.desc.layers
                || up.x + up.w > self.desc.layer_size
                || up.y + up.h > self.desc.layer_size
                || up.rgba.len() != expected
            {
                log::warn!(
                    "scene_batch[{}]: dropping out-of-range upload page={} {}x{}+{}+{} ({} bytes)",
                    self.id,
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
        // Streamed mip levels, straight into their own level of the page.
        // Validated at queue time against the desc the game packed against;
        // dropped levels never reach the texture.
        for up in &self.mip_uploads {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &res.texture,
                    mip_level: up.level,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: up.page,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &up.texels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(up.width * 4),
                    rows_per_image: Some(up.height),
                },
                wgpu::Extent3d {
                    width: up.width,
                    height: up.height,
                    depth_or_array_layers: 1,
                },
            );
        }
        res.last_ranges = self.ranges.clone();
        res.occlusion_feed_desc = self.occlusion_feed;
        res.occlusion_phase_two = self.occlusion_phase_two;
        let candidates = self.occlusion_candidates.len() as u32;
        res.occlusion_candidate_count = candidates;
        if candidates > res.occlusion_capacity {
            let cap = candidates.next_power_of_two();
            res.occlusion_candidates = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("repame_view3d_occlusion_candidates"),
                size: cap as u64 * std::mem::size_of::<HizCandidate>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            res.occlusion_args = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("repame_view3d_occlusion_args"),
                size: cap as u64 * 20,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
                mapped_at_creation: false,
            });
            res.occlusion_capacity = cap;
        }
        if candidates > 0 {
            queue.write_buffer(
                &res.occlusion_candidates,
                0,
                bytemuck::cast_slice(&self.occlusion_candidates),
            );
        }
    }

    /// Guard path shared by tests: run validation + flatten without a GPU.
    #[cfg(test)]
    pub(crate) fn entry_ranges(&self) -> Vec<(u32, u32, bool, bool)> {
        self.ranges
            .iter()
            .map(|r| (r.index_start, r.index_end, r.depth_test, r.transparent))
            .collect()
    }

    /// Test-only vertex readout after [`finish`]: (normal, lit flag).
    #[cfg(test)]
    pub(crate) fn test_vert_lighting(&self) -> Vec<([f32; 3], f32)> {
        self.verts.iter().map(|v| (v.normal, v.lit)).collect()
    }

    /// Test-only vertex readout after [`finish`]: (metallic, roughness, emissive).
    #[cfg(test)]
    pub(crate) fn test_vert_material(&self) -> Vec<(f32, f32, [f32; 3])> {
        self.verts
            .iter()
            .map(|v| (v.metallic, v.roughness, v.emissive))
            .collect()
    }

    /// Test-only vertex readout after [`finish`]: (uv, tex_mix, page).
    #[cfg(test)]
    pub(crate) fn test_vert_texture(&self) -> Vec<([f32; 2], f32, f32)> {
        self.verts
            .iter()
            .map(|v| (v.uv, v.tex_mix, v.page))
            .collect()
    }

    /// Test-only vertex readout after [`finish`]: (alpha, cutoff).
    #[cfg(test)]
    pub(crate) fn test_vert_alpha(&self) -> Vec<(f32, f32)> {
        self.verts.iter().map(|v| (v.alpha, v.cutoff)).collect()
    }

    /// Test-only resident-level readout: `None` = nothing resident.
    #[cfg(test)]
    pub(crate) fn test_page_level(&self, page: u32) -> Option<u32> {
        match self.page_levels.get(page as usize).copied() {
            Some(level) if level < 0.0 => None,
            Some(level) => Some(level as u32),
            None => None,
        }
    }
}

/// GPU-resource cache key for one batch: every dimension that forces a
/// pipeline or texture rebuild. `SceneEntry::key` and `ensure_resources` must
/// agree on it, so it lives in one place.
type SceneKey = (
    wgpu::TextureFormat,
    u32,   // sample count
    u32,   // layer size
    u32,   // layers
    u32,   // filter
    u32,   // mip levels
    u32,   // shadow map size
    u32,   // cascade map size
    u32,   // cascade count
    u32,   // point shadow size
    usize, // skin slots
    bool,  // GPU skinning supported
);

struct SceneEntry {
    key: SceneKey,
    pipeline_depth: wgpu::RenderPipeline,
    pipeline_flat: wgpu::RenderPipeline,
    pipeline_transparent: wgpu::RenderPipeline,
    pipeline_transparent_flat: wgpu::RenderPipeline,
    shadow_depth_pipeline: wgpu::RenderPipeline,
    /// Depth + camera velocity pipeline for a declared
    /// [`PassKind::Depth`](super::pass::PassKind::Depth) target: the scene
    /// geometry replayed as `(view_depth, velocity.x, velocity.y, 1)`.
    depth_velocity_pipeline: wgpu::RenderPipeline,
    /// Depth alone, for a [`PassKind::Occlusion`](super::pass::PassKind::Occlusion)
    /// target: the scene geometry replayed as `(view_depth, 0, 0, 1)`.
    depth_only_pipeline: wgpu::RenderPipeline,
    /// The `R32Float` target variants of both depth-kind pipelines.
    depth_velocity_pipeline_r32f: wgpu::RenderPipeline,
    depth_only_pipeline_r32f: wgpu::RenderPipeline,
    /// Previous-frame view-projection the velocity pass reads.
    depth_velocity_buffer: wgpu::Buffer,
    depth_velocity_bind: wgpu::BindGroup,
    /// Viewport-driven occlusion feed: the depth pass readback in flight
    /// (see [`OcclusionFeed`]). Cross-frame GPU + host state, so it lives on
    /// the entry rather than the per-frame batch.
    occlusion_feed: OcclusionFeed,
    /// Feed settings the last upload staged.
    occlusion_feed_desc: Option<crate::OcclusionFeedDesc>,
    /// The frame's phase-two candidate records.
    occlusion_candidates: wgpu::Buffer,
    /// The frame's indirect draw args, written by the phase-two test;
    /// `occlusion_capacity` slots each.
    occlusion_args: wgpu::Buffer,
    /// Slots both occlusion buffers hold.
    occlusion_capacity: u32,
    /// Candidates the last upload wrote.
    occlusion_candidate_count: u32,
    /// The staged frame re-tests its phase-one rejects in phase two.
    occlusion_phase_two: bool,
    /// Post-process stage state (shader, pipelines, this frame's bind groups).
    post: Option<PostState>,
    /// One set of pipelines, uniform and bind group per material program
    /// registered this frame, indexed by `DrawRange::material - 1`.
    material_pipelines: Vec<MaterialPipelines>,
    /// Pass targets by declared name, allocated on first use.
    pass_targets: HashMap<String, PassTarget>,
    /// Fullscreen pass state: the pipeline, its sampler, and one bind
    /// group per source texture.
    fullscreen: Option<FullscreenState>,
    /// The pass whose target composites over the scene at paint time.
    presenting: Option<PassSpec>,
    /// Pass declarations the batch staged for this frame.
    pass_specs: Vec<PassSpec>,
    pipe_layout: wgpu::PipelineLayout,
    shadow_cam_bind: wgpu::BindGroup,
    shadow_camera_buf: wgpu::Buffer,
    /// Per-slice cameras for the cascade passes (one buffer each).
    cascade_cam_binds: Vec<(wgpu::Buffer, wgpu::BindGroup)>,
    /// Per-face cameras for the point-light cube passes.
    point_cam_binds: Vec<(wgpu::Buffer, wgpu::BindGroup)>,
    shadow_bind: wgpu::BindGroup,
    shadow_view: wgpu::TextureView,
    shadow_edge: u32,
    shadow_layout: wgpu::BindGroupLayout,
    shadow_sampler: wgpu::Sampler,
    cascade_tex: wgpu::Texture,
    cascade_view: wgpu::TextureView,
    cascade_edge: u32,
    cascade_sampler: wgpu::Sampler,
    point_tex: wgpu::Texture,
    point_view: wgpu::TextureView,
    point_edge: u32,
    point_sampler: wgpu::Sampler,
    /// Sampler and 6-face stand-ins bound at the probe slots when a frame
    /// stages no probe. The textures are held so their views stay valid.
    probe_sampler: wgpu::Sampler,
    _probe_dummies: Vec<wgpu::Texture>,
    probe_dummy_views: Vec<wgpu::TextureView>,
    /// Caster cube faces sampled this frame (six VPs), written by
    /// `upload_all` from the staged points. `None` = cube pass off.
    point_faces: Option<[[[f32; 4]; 4]; 6]>,
    /// GPU-skinning capability this entry's shaders and camera layout were
    /// built for (the mode rides the resource key, so a change rebuilds
    /// the entry).
    skin: GpuSkinning,
    /// Joint-palette storage buffer. `None` on devices without
    /// `VERTEX_STORAGE`: the layout omits binding 1 and skinned draws drop
    /// before any palette exists.
    skin_buffer: Option<wgpu::Buffer>,
    verts: wgpu::Buffer,
    vert_cap: usize,
    indices: wgpu::Buffer,
    index_cap: usize,
    camera: wgpu::Buffer,
    cam_bind: wgpu::BindGroup,
    /// The camera and texture-array layouts, kept so the particle billboard
    /// pipeline can be built from the same bind groups the scene uses.
    cam_layout: wgpu::BindGroupLayout,
    tex_layout: wgpu::BindGroupLayout,
    material_buffer: wgpu::Buffer,
    material_bind: wgpu::BindGroup,
    material_layout: wgpu::BindGroupLayout,
    tex_bind: wgpu::BindGroup,
    /// Per-page resident mip levels the scene and particle shaders sample,
    /// written by [`SceneBatch::upload_all`] from the staged table.
    page_levels: wgpu::Buffer,
    texture: wgpu::Texture,
    camera_mat: CameraUniform,
    last_ranges: Vec<DrawRange>,
    /// Particle pipelines, collision world and per-system buffers, built on
    /// first use and reused while the frame stages particle systems.
    particles: Option<ParticleGpu>,
    /// Reflection-probe cubes and the capture camera, built on first use.
    probes: Option<ProbeGpu>,
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
    if let Err(err) = DepthComposite::get(resources).ensure(device, screen, id, w, h) {
        // Without a target the scene would draw into the previous frame's
        // pixels, which reads as a frozen viewport rather than a failure.
        log::error!("scene_batch[{id}]: offscreen target unavailable: {err:#}");
        return;
    }
    // Snapshot the draw state into owned bind groups: pipelines and buffers
    // shared resources: clone the handles, then end the borrow
    // before beginning the mutable scene pass.
    struct Snapshot {
        cam_bind: wgpu::BindGroup,
        material_bind: wgpu::BindGroup,
        material_pipelines: Vec<MaterialPipelines>,
        passes: Vec<PassEntry>,
        /// Source texture -> fullscreen bind group, built during upload.
        fullscreen_binds: HashMap<String, wgpu::BindGroup>,
        fullscreen_pipeline: Option<wgpu::RenderPipeline>,
        tex_bind: wgpu::BindGroup,
        shadow_bind: wgpu::BindGroup,
        shadow_depth_pipeline: wgpu::RenderPipeline,
        shadow_cam_bind: wgpu::BindGroup,
        shadow_camera_buf: wgpu::Buffer,
        cascade_cam_binds: Vec<(wgpu::Buffer, wgpu::BindGroup)>,
        point_cam_binds: Vec<(wgpu::Buffer, wgpu::BindGroup)>,
        shadow_tex_view: wgpu::TextureView,
        shadow_edge: u32,
        shadow_vp: [[f32; 4]; 4],
        shadows_on: bool,
        cascade_tex: wgpu::Texture,
        cascade_edge: u32,
        cascade_vps: [[[f32; 4]; 4]; 4],
        cascade_count: usize,
        point_tex: wgpu::Texture,
        point_edge: u32,
        point_faces: Option<[[[f32; 4]; 4]; 6]>,
        verts: wgpu::Buffer,
        indices: wgpu::Buffer,
        pipeline_depth: wgpu::RenderPipeline,
        pipeline_flat: wgpu::RenderPipeline,
        pipeline_transparent: wgpu::RenderPipeline,
        pipeline_transparent_flat: wgpu::RenderPipeline,
        depth_velocity_pipeline: wgpu::RenderPipeline,
        depth_only_pipeline: wgpu::RenderPipeline,
        depth_velocity_pipeline_r32f: wgpu::RenderPipeline,
        depth_only_pipeline_r32f: wgpu::RenderPipeline,
        depth_velocity_bind: wgpu::BindGroup,
        post_pipeline: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
        post_binds: HashMap<u32, wgpu::BindGroup>,
        ranges: Vec<DrawRange>,
        particles: Option<ParticleSnapshot>,
        probe_captures: Option<ProbeCaptureSet>,
        /// The frame's settled camera uniform, the base for each probe face.
        camera_mat: CameraUniform,
        /// Phase two re-tested this frame's phase-one rejects on the GPU:
        /// the scene pass draws those ranges through
        /// [`Snapshot::occlusion_args`].
        occlusion_indirect: bool,
        /// Candidates [`Snapshot::occlusion_args`] holds args for.
        occlusion_candidates: u32,
        occlusion_args: wgpu::Buffer,
    }
    // A frame with neither geometry nor particles still has to clear the
    // scene target: `ensure` leaves a matching target untouched, and `paint`
    // blits whatever is in it, so skipping the clear re-presents the last
    // frame.
    let flat = resources
        .get_mut::<SceneResources>()
        .and_then(|all| all.batches.get_mut(id))
        .is_some_and(|res| {
            res.last_ranges.is_empty()
                && res
                    .particles
                    .as_ref()
                    .is_none_or(|gpu| !gpu.systems.iter().any(|state| state.is_some()))
        });
    if flat {
        drop(DepthComposite::get(resources).begin_scene(id, encoder, clear));
        return;
    }

    let snapshot: Option<Snapshot> = {
        let Some(all) = resources.get_mut::<SceneResources>() else {
            return;
        };
        let Some(res) = all.batches.get_mut(id) else {
            return;
        };
        queue.write_buffer(&res.camera, 0, bytemuck::cast_slice(&[res.camera_mat]));
        if res.last_ranges.is_empty()
            && res
                .particles
                .as_ref()
                .is_none_or(|gpu| !gpu.systems.iter().any(|state| state.is_some()))
        {
            return;
        }
        let mat = res.camera_mat;
        // Phase two re-tests this frame's rejects against a chain built from
        // the depth pass; it needs that pass's target in this frame's plan.
        let phase_two_ran = res.occlusion_phase_two
            && res.occlusion_candidate_count > 0
            && res.pass_specs.iter().any(|spec| {
                matches!(spec.kind, super::pass::PassKind::Depth | super::pass::PassKind::Occlusion)
                    && res.pass_targets.contains_key(&spec.target.name)
            });
        let cascade_vps = [
            mat.cascade_vp0,
            mat.cascade_vp1,
            mat.cascade_vp2,
            mat.cascade_vp3,
        ];
        let cascade_count = if mat.cascade_params[1] > 0.5 {
            (mat.cascade_params[2] as usize).min(crate::MAX_CASCADES)
        } else {
            0
        };
        Some(Snapshot {
            cam_bind: res.cam_bind.clone(),
            material_bind: res.material_bind.clone(),
            fullscreen_binds: res
                .fullscreen
                .as_ref()
                .map(|state| state.binds.clone())
                .unwrap_or_default(),
            fullscreen_pipeline: res.fullscreen.as_ref().map(|state| state.pipeline.clone()),
            passes: res
                .pass_specs
                .iter()
                .filter_map(|spec| {
                    res.pass_targets
                        .get(&spec.target.name)
                        .map(|target| PassEntry {
                            spec: spec.clone(),
                            view: target.view.clone(),
                            depth_view: target.depth_view.clone(),
                            texture: target.texture.clone(),
                            source_texture: spec
                                .source
                                .as_ref()
                                .and_then(|name| res.pass_targets.get(name))
                                .map(|target| target.texture.clone()),
                        })
                })
                .collect(),
            material_pipelines: res
                .material_pipelines
                .iter()
                .map(|pipes| MaterialPipelines {
                    key: pipes.key,
                    depth: pipes.depth.clone(),
                    flat: pipes.flat.clone(),
                    transparent: pipes.transparent.clone(),
                    transparent_flat: pipes.transparent_flat.clone(),
                    buffer: pipes.buffer.clone(),
                    bind: pipes.bind.clone(),
                })
                .collect(),
            tex_bind: res.tex_bind.clone(),
            shadow_bind: res.shadow_bind.clone(),
            shadow_depth_pipeline: res.shadow_depth_pipeline.clone(),
            shadow_cam_bind: res.shadow_cam_bind.clone(),
            shadow_camera_buf: res.shadow_camera_buf.clone(),
            cascade_cam_binds: res
                .cascade_cam_binds
                .iter()
                .map(|(buffer, bind)| (buffer.clone(), bind.clone()))
                .collect(),
            point_cam_binds: res
                .point_cam_binds
                .iter()
                .map(|(buffer, bind)| (buffer.clone(), bind.clone()))
                .collect(),
            shadow_tex_view: res.shadow_view.clone(),
            shadow_edge: res.shadow_edge,
            shadow_vp: mat.shadow_vp,
            shadows_on: mat.shadow_params[1] > 0.5,
            cascade_tex: res.cascade_tex.clone(),
            cascade_edge: res.cascade_edge,
            cascade_vps,
            cascade_count,
            point_tex: res.point_tex.clone(),
            point_edge: res.point_edge,
            point_faces: res.point_faces,
            verts: res.verts.clone(),
            indices: res.indices.clone(),
            pipeline_depth: res.pipeline_depth.clone(),
            pipeline_flat: res.pipeline_flat.clone(),
            pipeline_transparent: res.pipeline_transparent.clone(),
            pipeline_transparent_flat: res.pipeline_transparent_flat.clone(),
            depth_velocity_pipeline: res.depth_velocity_pipeline.clone(),
            depth_only_pipeline: res.depth_only_pipeline.clone(),
            depth_velocity_pipeline_r32f: res.depth_velocity_pipeline_r32f.clone(),
            depth_only_pipeline_r32f: res.depth_only_pipeline_r32f.clone(),
            depth_velocity_bind: res.depth_velocity_bind.clone(),
            post_pipeline: res
                .post
                .as_ref()
                .map(|post| post.pipelines.clone())
                .unwrap_or_default(),
            post_binds: res
                .post
                .as_ref()
                .map(|post| post.binds.clone())
                .unwrap_or_default(),
            ranges: res.last_ranges.clone(),
            particles: res.particles.as_ref().map(|gpu| ParticleSnapshot {
                compute_prepare: gpu.compute_prepare.clone(),
                compute_simulate: gpu.compute_simulate.clone(),
                billboard_pipeline: gpu.billboard_pipeline.clone(),
                draws: gpu
                    .order
                    .iter()
                    .filter_map(|index| {
                        let state = gpu.systems.get(*index as usize)?.as_ref()?;
                        Some(ParticleDraw {
                            sim_bind: state.sim_bind.clone(),
                            draw_bind: state.draw_bind.clone(),
                            capacity: state.capacity,
                            pass: state.pass,
                        })
                    })
                    .collect(),
            }),
            camera_mat: res.camera_mat,
            occlusion_indirect: phase_two_ran,
            occlusion_candidates: res.occlusion_candidate_count,
            occlusion_args: res.occlusion_args.clone(),
            probe_captures: res.probes.as_ref().map(|gpu| ProbeCaptureSet {
                capture_bind: gpu.capture_bind.clone(),
                probes: gpu
                    .probes
                    .iter()
                    .filter(|state| state.capture && state.settings.is_active())
                    .map(|state| {
                        let (faces, dirs) = probe_face_matrices(
                            state.settings.position,
                            state.settings.range.max(1.0),
                        );
                        ProbeCapture {
                            cube: state.cube.clone(),
                            depth: state.depth.clone(),
                            edge: state.edge,
                            position: state.settings.position,
                            faces,
                            dirs,
                            cam_binds: state
                                .cam_binds
                                .iter()
                                .map(|(buffer, bind)| (buffer.clone(), bind.clone()))
                                .collect(),
                        }
                    })
                    .collect(),
            }),
        })
    };
    let Some(snap) = snapshot else { return };
    if snap.shadows_on {
        queue.write_buffer(
            &snap.shadow_camera_buf,
            0,
            bytemuck::cast_slice(&snap.shadow_vp),
        );
        let mut spass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("repame_view3d_shadow"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &snap.shadow_tex_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        spass.set_viewport(
            0.0,
            0.0,
            snap.shadow_edge as f32,
            snap.shadow_edge as f32,
            0.0,
            1.0,
        );
        spass.set_pipeline(&snap.shadow_depth_pipeline);
        spass.set_bind_group(0, &snap.shadow_cam_bind, &[]);
        spass.set_vertex_buffer(0, snap.verts.slice(..));
        spass.set_index_buffer(snap.indices.slice(..), wgpu::IndexFormat::Uint32);
        for r in &snap.ranges {
            if r.pass == 0 && !r.transparent && r.depth_test {
                spass.draw_indexed(r.index_start..r.index_end, 0, 0..1);
            }
        }
    }
    for slice in 0..snap.cascade_count {
        let vp = snap
            .cascade_vps
            .get(slice)
            .copied()
            .unwrap_or_else(|| glam::Mat4::IDENTITY.to_cols_array_2d());
        let Some((slice_camera, slice_bind)) = snap.cascade_cam_binds.get(slice) else {
            continue;
        };
        queue.write_buffer(slice_camera, 0, bytemuck::cast_slice(&vp));
        let layer = snap.cascade_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("repame_view3d_cascade_layer"),
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_array_layer: slice as u32,
            array_layer_count: Some(1),
            ..Default::default()
        });
        {
            let mut cpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("repame_view3d_cascade"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &layer,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            cpass.set_viewport(
                0.0,
                0.0,
                snap.cascade_edge as f32,
                snap.cascade_edge as f32,
                0.0,
                1.0,
            );
            cpass.set_pipeline(&snap.shadow_depth_pipeline);
            cpass.set_bind_group(0, slice_bind, &[]);
            cpass.set_vertex_buffer(0, snap.verts.slice(..));
            cpass.set_index_buffer(snap.indices.slice(..), wgpu::IndexFormat::Uint32);
            for r in &snap.ranges {
                if r.pass == 0 && !r.transparent && r.depth_test {
                    cpass.draw_indexed(r.index_start..r.index_end, 0, 0..1);
                }
            }
        }
    }
    if let Some(faces) = snap.point_faces {
        for (face, vp) in faces.iter().enumerate() {
            let Some((face_camera, face_bind)) = snap.point_cam_binds.get(face) else {
                continue;
            };
            queue.write_buffer(face_camera, 0, bytemuck::cast_slice(vp));
            let layer = snap.point_tex.create_view(&wgpu::TextureViewDescriptor {
                label: Some("repame_view3d_point_layer"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: face as u32,
                array_layer_count: Some(1),
                ..Default::default()
            });
            {
                let mut ppass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("repame_view3d_point"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &layer,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                ppass.set_viewport(
                    0.0,
                    0.0,
                    snap.point_edge as f32,
                    snap.point_edge as f32,
                    0.0,
                    1.0,
                );
                ppass.set_pipeline(&snap.shadow_depth_pipeline);
                ppass.set_bind_group(0, face_bind, &[]);
                ppass.set_vertex_buffer(0, snap.verts.slice(..));
                ppass.set_index_buffer(snap.indices.slice(..), wgpu::IndexFormat::Uint32);
                for r in &snap.ranges {
                    if r.pass == 0 && !r.transparent && r.depth_test {
                        ppass.draw_indexed(r.index_start..r.index_end, 0, 0..1);
                    }
                }
            }
        }
    }
    // Reflection probes capture before the scene pass, so the frame's
    // materials sample this frame's cube. Each face is the frame's opaque
    // depth-tested geometry replayed with a camera at the probe point and
    // the probe's own face view-projection; fog, shadows, cascades and the
    // probes themselves are switched off in that camera, so the cube holds
    // lit albedo rather than one camera's post state.
    if let Some(captures) = snap.probe_captures.as_ref() {
        for probe in &captures.probes {
            for face in 0..6 {
                let Some((face_camera, face_bind)) = probe.cam_binds.get(face) else {
                    continue;
                };
                let mut camera = snap.camera_mat;
                camera.view_proj = probe.faces[face];
                camera.cam_pos = probe.position;
                camera.cam_forward = probe.dirs[face];
                camera.fog[0] = 0.0;
                camera.shadow_params[1] = 0.0;
                camera.cascade_params[1] = 0.0;
                camera.point_params[1] = 0.0;
                camera.probe_params = [[0.0; 4]; super::reflection::MAX_PROBES];
                queue.write_buffer(face_camera, 0, bytemuck::cast_slice(&[camera]));
                let layer = probe.cube.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("repame_view3d_probe_face"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: face as u32,
                    array_layer_count: Some(1),
                    ..Default::default()
                });
                let depth_layer = probe.depth.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("repame_view3d_probe_face_depth"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: face as u32,
                    array_layer_count: Some(1),
                    ..Default::default()
                });
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("repame_view3d_probe"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &layer,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_layer,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_viewport(
                    0.0,
                    0.0,
                    probe.edge as f32,
                    probe.edge as f32,
                    0.0,
                    1.0,
                );
                pass.set_bind_group(0, face_bind, &[]);
                pass.set_bind_group(1, &snap.tex_bind, &[]);
                pass.set_bind_group(2, &captures.capture_bind, &[]);
                pass.set_bind_group(3, &snap.material_bind, &[]);
                pass.set_vertex_buffer(0, snap.verts.slice(..));
                pass.set_index_buffer(snap.indices.slice(..), wgpu::IndexFormat::Uint32);
                let mut bound_material: Option<u32> = None;
                for r in snap
                    .ranges
                    .iter()
                    .filter(|r| r.pass == 0 && r.depth_test && !r.transparent)
                {
                    let program = snap
                        .material_pipelines
                        .get(r.material.saturating_sub(1) as usize)
                        .filter(|_| r.material > 0);
                    if bound_material != Some(r.material) {
                        match program {
                            Some(pipes) => pass.set_bind_group(3, &pipes.bind, &[]),
                            None => pass.set_bind_group(3, &snap.material_bind, &[]),
                        }
                        bound_material = Some(r.material);
                    }
                    pass.set_pipeline(match program {
                        Some(pipes) => &pipes.depth,
                        None => &snap.pipeline_depth,
                    });
                    pass.draw_indexed(r.index_start..r.index_end, 0, 0..1);
                }
            }
        }
    }

    // The particle simulation runs once per frame, before any pass that
    // draws its billboards. Systems dispatch in sub-emitter order, so a
    // parent's requests are visible to its target in the same pass.
    if let Some(particles) = snap.particles.as_ref()
        && !particles.draws.is_empty()
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("repame_view3d_particles"),
            timestamp_writes: None,
        });
        for draw in &particles.draws {
            pass.set_bind_group(0, &draw.sim_bind, &[]);
            pass.set_pipeline(&particles.compute_prepare);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&particles.compute_simulate);
            pass.dispatch_workgroups(draw.capacity.div_ceil(SIMULATE_WORKGROUP), 1, 1);
        }
    }
    // One declared pass's depth/velocity replay, shared by the phase-two
    // ordering (ahead of the scene pass) and the declaration-order loop.
    let encode_depth_pass = |encoder: &mut wgpu::CommandEncoder, entry: &PassEntry| {
        // Replay the frame's opaque geometry into an Rgba16Float target:
        // view depth, plus camera velocity for a declared Depth pass (the
        // engine's Occlusion kind runs the depth-only pipeline).
        let view = &entry.view;
        let depth_view = &entry.depth_view;
        let spec = &entry.spec;
        let mut dv = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("repame_view3d_depth_velocity"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: if spec.load {
                        wgpu::LoadOp::Load
                    } else {
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                    },
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: if spec.load {
                        wgpu::LoadOp::Load
                    } else {
                        wgpu::LoadOp::Clear(1.0)
                    },
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        dv.set_viewport(
            0.0,
            0.0,
            spec.target.width as f32,
            spec.target.height as f32,
            0.0,
            1.0,
        );
        let depth_only = entry.spec.kind == super::pass::PassKind::Occlusion;
        let r32f = entry.spec.target.format == super::pass::RenderTargetFormat::R32Float;
        dv.set_pipeline(match (depth_only, r32f) {
            (true, true) => &snap.depth_only_pipeline_r32f,
            (true, false) => &snap.depth_only_pipeline,
            (false, true) => &snap.depth_velocity_pipeline_r32f,
            (false, false) => &snap.depth_velocity_pipeline,
        });
        dv.set_bind_group(0, &snap.cam_bind, &[]);
        dv.set_bind_group(1, &snap.depth_velocity_bind, &[]);
        dv.set_vertex_buffer(0, snap.verts.slice(..));
        dv.set_index_buffer(snap.indices.slice(..), wgpu::IndexFormat::Uint32);
        for r in snap.ranges.iter().filter(|r| r.depth_test && !r.transparent) {
            dv.draw_indexed(r.index_start..r.index_end, 0, 0..1);
        }
    };

    // Phase two: the depth pass runs before the scene pass, so this frame's
    // chain is available to re-test the phase-one rejects; the test compute
    // writes their indirect args before the scene pass draws them.
    let mut occlusion_indirect = snap.occlusion_indirect;
    // Phase two runs once per frame: right after the depth pass the chain
    // reads.
    let mut phase_two_done = false;
    // Draws the re-tested rejects into a pass that loaded the scene color and
    // depth; the test zeroed the still-occluded ones' index counts.
    let draw_phase_two = |pass: &mut wgpu::RenderPass<'_>, snap: &Snapshot| {
        pass.set_bind_group(0, &snap.cam_bind, &[]);
        pass.set_bind_group(1, &snap.tex_bind, &[]);
        pass.set_bind_group(2, &snap.shadow_bind, &[]);
        pass.set_bind_group(3, &snap.material_bind, &[]);
        pass.set_vertex_buffer(0, snap.verts.slice(..));
        pass.set_index_buffer(snap.indices.slice(..), wgpu::IndexFormat::Uint32);
        let mut bound_material: Option<u32> = None;
        for r in snap.ranges.iter().filter(|r| r.hz_culled.is_some()) {
            let program = snap
                .material_pipelines
                .get(r.material.saturating_sub(1) as usize)
                .filter(|_| r.material > 0);
            if bound_material != Some(r.material) {
                match program {
                    Some(pipes) => pass.set_bind_group(3, &pipes.bind, &[]),
                    None => pass.set_bind_group(3, &snap.material_bind, &[]),
                }
                bound_material = Some(r.material);
            }
            pass.set_pipeline(match (r.transparent, r.depth_test) {
                (true, true) => match program {
                    Some(pipes) => &pipes.transparent,
                    None => &snap.pipeline_transparent,
                },
                (true, false) => match program {
                    Some(pipes) => &pipes.transparent_flat,
                    None => &snap.pipeline_transparent_flat,
                },
                (false, true) => match program {
                    Some(pipes) => &pipes.depth,
                    None => &snap.pipeline_depth,
                },
                (false, false) => match program {
                    Some(pipes) => &pipes.flat,
                    None => &snap.pipeline_flat,
                },
            });
            if let Some(index) = r.hz_culled {
                pass.draw_indexed_indirect(&snap.occlusion_args, index as u64 * 20);
            }
        }
    };

    let composite = DepthComposite::get(resources);
    {
        let ops = ScenePassOps {
            clear: Some(clear),
            load_depth: false,
            // The phase-two pass loads the scene depth to z-test against
            // what this pass leaves.
            store_depth: occlusion_indirect,
        };
        let Some(mut pass) = composite.begin_scene_ops(id, encoder, ops) else {
            return;
        };
        pass.set_bind_group(0, &snap.cam_bind, &[]);
        pass.set_bind_group(1, &snap.tex_bind, &[]);
        pass.set_bind_group(2, &snap.shadow_bind, &[]);
        pass.set_bind_group(3, &snap.material_bind, &[]);
        pass.set_vertex_buffer(0, snap.verts.slice(..));
        pass.set_index_buffer(snap.indices.slice(..), wgpu::IndexFormat::Uint32);
        let mut bound_material: Option<u32> = None;
        for r in snap.ranges.iter().filter(|r| r.pass == 0) {
            let program = snap
                .material_pipelines
                .get(r.material.saturating_sub(1) as usize)
                .filter(|_| r.material > 0);
            if bound_material != Some(r.material) {
                match program {
                    Some(pipes) => {
                        pass.set_bind_group(3, &pipes.bind, &[]);
                    }
                    None => {
                        pass.set_bind_group(3, &snap.material_bind, &[]);
                    }
                }
                bound_material = Some(r.material);
            }
            pass.set_pipeline(match (r.transparent, r.depth_test) {
                (true, true) => match program {
                    Some(pipes) => &pipes.transparent,
                    None => &snap.pipeline_transparent,
                },
                (true, false) => match program {
                    Some(pipes) => &pipes.transparent_flat,
                    None => &snap.pipeline_transparent_flat,
                },
                (false, true) => match program {
                    Some(pipes) => &pipes.depth,
                    None => &snap.pipeline_depth,
                },
                (false, false) => match program {
                    Some(pipes) => &pipes.flat,
                    None => &snap.pipeline_flat,
                },
            });
            if r.hz_culled.is_some() {
                // A phase-one reject: stays culled unless the phase-two
                // pass draws it after the depth pass.
                continue;
            }
            pass.draw_indexed(r.index_start..r.index_end, 0, 0..1);
        }
        if let Some(particles) = snap.particles.as_ref() {
            draw_particle_billboards(&mut pass, particles, 0, &snap.cam_bind, &snap.tex_bind);
        }
    }

    // Declared passes each own their attachment, so they run outside the
    // shared scene pass, in declaration order.
    for entry in snap.passes.iter() {
        let (spec, view, depth_view) = (&entry.spec, &entry.view, &entry.depth_view);
        let pass_index = spec.index;
        let target = spec.target.clear;
        if matches!(spec.kind, super::pass::PassKind::Depth | super::pass::PassKind::Occlusion) {
            encode_depth_pass(encoder, entry);
            // Phase two: the chain reads this pass's target, so the GPU test
            // and the second scene pass follow it here.
            if occlusion_indirect && !phase_two_done {
                phase_two_done = true;
                if let Some(all) = resources.get_mut::<SceneResources>()
                    && let Some(res) = all.batches.get_mut(id)
                    && let Some(desc) = res.occlusion_feed_desc
                {
                    let width = spec.target.width.max(1);
                    let height = spec.target.height.max(1);
                    if res.occlusion_feed.encode_chain(
                        device,
                        queue,
                        encoder,
                        view,
                        width,
                        height,
                        desc.clamped_tile(),
                    ) {
                        res.occlusion_feed.encode_test(
                            device,
                            queue,
                            encoder,
                            &res.camera_mat,
                            desc.clamped_tile(),
                            desc.clamped_bias(),
                            width,
                            height,
                            &res.occlusion_candidates,
                            snap.occlusion_candidates,
                            &snap.occlusion_args,
                        );
                    } else {
                        // No chain to test against: the rejects stay culled.
                        occlusion_indirect = false;
                    }
                } else {
                    occlusion_indirect = false;
                }
                if occlusion_indirect {
                    let scene = snap.passes.iter().find(|entry| {
                        entry.spec.kind == super::pass::PassKind::Render
                            && entry.spec.target.semantics
                                == super::pass::TargetSemantics::SceneColor
                    });
                    // Chained frames draw over the declared scene pass's
                    // target, a frame without one over the composite scene.
                    if let Some(scene) = scene {
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("repame_view3d_phase_two"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &scene.view,
                                resolve_target: None,
                                depth_slice: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                view: &scene.depth_view,
                                depth_ops: Some(wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                }),
                                stencil_ops: None,
                            }),
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        });
                        pass.set_viewport(
                            0.0,
                            0.0,
                            scene.spec.target.width as f32,
                            scene.spec.target.height as f32,
                            0.0,
                            1.0,
                        );
                        draw_phase_two(&mut pass, &snap);
                    } else if let Some(mut pass) = DepthComposite::get(resources).begin_scene_ops(
                        id,
                        encoder,
                        ScenePassOps {
                            clear: None,
                            load_depth: true,
                            store_depth: false,
                        },
                    ) {
                        draw_phase_two(&mut pass, &snap);
                    }
                }
            }
            continue;
        }
        if spec.kind == super::pass::PassKind::Fullscreen {
            // A post stage runs the effect shader with the stage's bind group;
            // a plain fullscreen pass keeps the tinted blit.
            let post = spec.post.and_then(|_| {
                let format = pass_format(spec.target.format);
                let pipeline = snap.post_pipeline.get(&format)?;
                let bind = snap.post_binds.get(&spec.index)?;
                Some((pipeline, bind))
            });
            let plain = snap.fullscreen_pipeline.as_ref().and_then(|pipeline| {
                spec.source
                    .as_deref()
                    .and_then(|name| snap.fullscreen_binds.get(name))
                    .map(|bind| (pipeline, bind))
            });
            if let Some((pipeline, bind)) = post.or(plain) {
                let mut full = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("repame_view3d_fullscreen"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: target[0] as f64,
                                g: target[1] as f64,
                                b: target[2] as f64,
                                a: target[3] as f64,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                full.set_pipeline(pipeline);
                full.set_bind_group(0, bind, &[]);
                full.draw(0..3, 0..1);
            }
            continue;
        }
        if spec.kind == super::pass::PassKind::Resolve {
            if let Some(from) = entry.source_texture.as_ref() {
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: from,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &entry.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: spec.target.width.max(1),
                        height: spec.target.height.max(1),
                        depth_or_array_layers: 1,
                    },
                );
            }
            continue;
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("repame_view3d_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: if spec.load {
                        wgpu::LoadOp::Load
                    } else {
                        wgpu::LoadOp::Clear(wgpu::Color {
                            r: target[0] as f64,
                            g: target[1] as f64,
                            b: target[2] as f64,
                            a: target[3] as f64,
                        })
                    },
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_bind_group(0, &snap.cam_bind, &[]);
        pass.set_bind_group(1, &snap.tex_bind, &[]);
        pass.set_bind_group(2, &snap.shadow_bind, &[]);
        pass.set_bind_group(3, &snap.material_bind, &[]);
        pass.set_vertex_buffer(0, snap.verts.slice(..));
        pass.set_index_buffer(snap.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.set_viewport(
            0.0,
            0.0,
            spec.target.width as f32,
            spec.target.height as f32,
            0.0,
            1.0,
        );
        for r in snap.ranges.iter().filter(|r| r.pass == pass_index) {
            if r.hz_culled.is_some() {
                // A phase-one reject: stays culled, or the phase-two pass
                // draws it after the depth pass.
                continue;
            }
            let program = snap
                .material_pipelines
                .get(r.material.saturating_sub(1) as usize)
                .filter(|_| r.material > 0);
            pass.set_pipeline(match (r.transparent, r.depth_test) {
                (true, true) => match program {
                    Some(pipes) => &pipes.transparent,
                    None => &snap.pipeline_transparent,
                },
                (true, false) => match program {
                    Some(pipes) => &pipes.transparent_flat,
                    None => &snap.pipeline_transparent_flat,
                },
                (false, true) => match program {
                    Some(pipes) => &pipes.depth,
                    None => &snap.pipeline_depth,
                },
                (false, false) => match program {
                    Some(pipes) => &pipes.flat,
                    None => &snap.pipeline_flat,
                },
            });
            if r.material > 0
                && let Some(pipes) = program
            {
                pass.set_bind_group(3, &pipes.bind, &[]);
            }
            pass.draw_indexed(r.index_start..r.index_end, 0, 0..1);
        }
        if let Some(particles) = snap.particles.as_ref() {
            draw_particle_billboards(&mut pass, particles, pass_index, &snap.cam_bind, &snap.tex_bind);
        }
    }
}

/// One particle system as the frame executor sees it: the bind groups its
/// compute dispatch and billboard draw use, its slot count, and the pass its
/// billboards join.
struct ParticleDraw {
    sim_bind: wgpu::BindGroup,
    draw_bind: wgpu::BindGroup,
    capacity: u32,
    pass: u32,
}

/// One probe capturing this frame: its cube targets, capture point, and the
/// six face view-projections.
struct ProbeCapture {
    cube: wgpu::Texture,
    depth: wgpu::Texture,
    edge: u32,
    position: [f32; 3],
    faces: [[[f32; 4]; 4]; 6],
    dirs: [[f32; 3]; 6],
    /// Per-face capture cameras: buffer plus its bind group.
    cam_binds: Vec<(wgpu::Buffer, wgpu::BindGroup)>,
}

/// A frame's probe captures plus the capture-pass bindings.
struct ProbeCaptureSet {
    /// Group 2 with stand-ins at the probe slots (a cube cannot be sampled
    /// while it is an attachment).
    capture_bind: wgpu::BindGroup,
    probes: Vec<ProbeCapture>,
}

/// Six 90-degree face view-projections (and their forward axes) for a cube
/// capture at `position` covering `range` world units.
fn probe_face_matrices(position: [f32; 3], range: f32) -> ([[[f32; 4]; 4]; 6], [[f32; 3]; 6]) {
    use glam::camera::rh::proj::opengl::perspective as perspective_gl;
    let proj = crate::camera::OPENGL_TO_WGPU
        * perspective_gl(std::f32::consts::FRAC_PI_2, 1.0, 0.05, range.max(0.1));
    let pos = glam::Vec3::from(position);
    let faces = [
        (glam::Vec3::X, glam::Vec3::NEG_Y),
        (glam::Vec3::NEG_X, glam::Vec3::NEG_Y),
        (glam::Vec3::Y, glam::Vec3::NEG_Z),
        (glam::Vec3::NEG_Y, glam::Vec3::Z),
        (glam::Vec3::Z, glam::Vec3::NEG_Y),
        (glam::Vec3::NEG_Z, glam::Vec3::NEG_Y),
    ];
    let mut vps = [[[0.0f32; 4]; 4]; 6];
    let mut dirs = [[0.0f32; 3]; 6];
    for (i, (forward, up)) in faces.iter().enumerate() {
        let view = glam::camera::rh::view::look_at_mat4(pos, pos + *forward, *up);
        vps[i] = (proj * view).to_cols_array_2d();
        dirs[i] = (*forward).into();
    }
    (vps, dirs)
}

/// A frame's particle work: the pipelines plus one [`ParticleDraw`] per live
/// system, in sub-emitter dispatch order.
struct ParticleSnapshot {
    compute_prepare: wgpu::ComputePipeline,
    compute_simulate: wgpu::ComputePipeline,
    billboard_pipeline: wgpu::RenderPipeline,
    draws: Vec<ParticleDraw>,
}

/// Draws one pass's particle billboards after that pass's mesh ranges: one
/// six-vertex quad per simulation slot, depth-tested against the pass's own
/// depth attachment and alpha-blended. Bindings are group 0 = camera, group 1
/// = the scene texture array (particle pages come from the same uploads),
/// group 2 = the system's particles, emitter parameters and display.
fn draw_particle_billboards(
    pass: &mut wgpu::RenderPass<'_>,
    particles: &ParticleSnapshot,
    pass_index: u32,
    cam_bind: &wgpu::BindGroup,
    tex_bind: &wgpu::BindGroup,
) {
    let mut bound = false;
    for draw in particles.draws.iter().filter(|draw| draw.pass == pass_index) {
        if !bound {
            pass.set_pipeline(&particles.billboard_pipeline);
            pass.set_bind_group(0, cam_bind, &[]);
            pass.set_bind_group(1, tex_bind, &[]);
            bound = true;
        }
        pass.set_bind_group(2, &draw.draw_bind, &[]);
        pass.draw(0..6 * draw.capacity, 0..1);
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
    paint_presenting(id, rpass, resources);
}

pub(crate) fn paint_scene_with_callback(
    id: &str,
    rpass: &mut CallbackRenderPass<'_, '_>,
    resources: &CallbackResources,
) {
    let Some(composite) = resources.get::<DepthComposite>() else {
        return;
    };
    composite.blit(id, rpass);
    paint_presenting(id, rpass, resources);
}

/// Composites the frame's presenting pass over the scene, if it declared one.
///
/// The shared UI pass is the only surface a game sees, so this is where a
/// pass target that asked to be presented lands: one fullscreen triangle
/// over the blitted scene, tinted and alpha-blended per the declaration.
fn paint_presenting<P: BlitRenderPass>(id: &str, rpass: &mut P, resources: &CallbackResources) {
    let Some(all) = resources.get::<SceneResources>() else {
        return;
    };
    let Some(entry) = all.batches.get(id) else {
        return;
    };
    let Some(state) = entry.fullscreen.as_ref() else {
        return;
    };
    let Some((_, bind)) = state.present.as_ref() else {
        return;
    };
    rpass.set_blit_pipeline(&state.overlay_pipeline);
    rpass.set_blit_bind_group(0, bind, &[]);
    rpass.draw_blit_triangle();
}

#[cfg(test)]
mod tests {
    use super::super::mesh::MeshGroup;
    use super::*;

    fn parses(label: &str, src: &str) {
        if let Err(e) = naga::front::wgsl::parse_str(src) {
            panic!("{label} does not parse: {e:?}");
        }
    }

    #[test]
    fn every_generated_scene_shader_parses() {
        parses("SHADER_PREFIX", SHADER_PREFIX);
        parses("scene_shader(supported)", &scene_shader(GpuSkinning::Supported));
        parses(
            "scene_shader(unsupported)",
            &scene_shader(GpuSkinning::Unsupported),
        );
        parses(
            "shadow_depth_shader(unsupported)",
            &shadow_depth_shader(GpuSkinning::Unsupported),
        );
        parses("FULLSCREEN_SHADER", FULLSCREEN_SHADER);
        parses("HIZ_WGSL", HIZ_WGSL);
        parses("HIZ_TEST_WGSL", HIZ_TEST_WGSL);
        // The dispatch stride is built from `HIZ_WORKGROUP` and the
        // pipelines name these entries, so a drifted workgroup size would
        // leave levels unwritten (stale values in a reused buffer) and a
        // renamed entry would fail pipeline creation at runtime.
        assert!(
            HIZ_WGSL.contains(&format!("@workgroup_size({HIZ_WORKGROUP})")),
            "HIZ_WGSL must declare @workgroup_size({HIZ_WORKGROUP})"
        );
        assert!(HIZ_WGSL.contains("fn tile_reduce("), "tile_reduce entry");
        assert!(HIZ_WGSL.contains("fn coarsen("), "coarsen entry");
        assert!(HIZ_TEST_WGSL.contains("fn hz_test("), "hz_test entry");
        let model = crate::shading::material_fragment(&crate::material::ShadingModel::default())
            .expect("default model validates");
        parses("material_fragment", &(String::from(SHADER_PREFIX) + &model));
        parses(
            "material_fragment(unsupported)",
            &(scene_prefix(GpuSkinning::Unsupported) + &model),
        );
    }

    /// The scene-family shader selection: the supported form is the
    /// historical source byte for byte, the unsupported form drops every
    /// reference to the palette binding (struct + storage binding + blend)
    /// so it matches a camera layout without binding 1, and both keep the
    /// entry points and the camera binding 0.
    #[test]
    fn unsupported_shaders_drop_the_palette() {
        let supported = scene_shader(GpuSkinning::Supported);
        assert_eq!(supported, String::from(SHADER_PREFIX) + SHADER_LEGACY_FS);
        assert!(supported.contains("skin_palette.joints[base + slot]"));
        let unsupported = scene_shader(GpuSkinning::Unsupported);
        assert!(!unsupported.contains("skin_palette"), "{unsupported}");
        assert!(!unsupported.contains("SkinPalette"), "{unsupported}");
        assert!(unsupported.contains("fn vs_main("), "entry point stays");
        assert!(unsupported.contains("fn fs_main("), "entry point stays");

        let shadow_supported = shadow_depth_shader(GpuSkinning::Supported);
        assert_eq!(shadow_supported, SHADOW_DEPTH_SHADER);
        assert!(shadow_supported.contains("skin_palette.joints[base + slot]"));
        let shadow_unsupported = shadow_depth_shader(GpuSkinning::Unsupported);
        assert!(!shadow_unsupported.contains("skin_palette"));
        assert!(!shadow_unsupported.contains("SkinPalette"));
        assert!(shadow_unsupported.contains("fn vs_main("), "entry point stays");
    }

    /// Minimal valid GPU-skinned draw for the push-path gate.
    fn skinned_draw() -> SkinnedDraw {
        let mesh = crate::skin::SkinnedMesh {
            name: "gate".into(),
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            colors: vec![[1.0, 1.0, 1.0]; 3],
            indices: vec![0, 1, 2],
            joints: vec![[0, 0, 0, 0]; 3],
            weights: vec![[1.0, 0.0, 0.0, 0.0]; 3],
            ..Default::default()
        };
        SkinnedDraw::from_matrices(&mesh, &[Mat4::IDENTITY]).expect("one joint fits the cap")
    }

    /// The capability gate: unsupported mode drops skinned draws at push
    /// (the caller CPU-poses them, never a silent no-op - the drop logs);
    /// the default mode stages them exactly like before.
    #[test]
    fn unsupported_skinning_drops_pushed_draws() {
        let draw = skinned_draw();
        let mut batch = SceneBatch::with_id("test.skin_gate");
        batch.set_skinning(GpuSkinning::Unsupported);
        batch.push_skinned(&draw);
        assert_eq!(batch.skinned_count(), 0, "unsupported drops at push");
        batch.set_skinning(GpuSkinning::Supported);
        batch.push_skinned(&draw);
        assert_eq!(batch.skinned_count(), 1, "supported stages");
    }

    #[test]
    fn tonemap_curves_differ_on_highlights() {
        let src = scene_shader(GpuSkinning::Supported);
        assert!(src.contains("repame_tonemap("), "both paths must share it");
        assert!(
            src.contains("repame_pbr_neutral("),
            "neutral curve is linked"
        );
        // The legacy path used to inline the Reinhard curve. If that ever
        // comes back alongside the helper, the two can drift apart silently.
        assert!(
            !src.contains("(rgb * e) / (rgb * (e - 1.0)"),
            "Reinhard must go through repame_tonemap, not an inline copy"
        );
        let fragment = crate::shading::material_fragment(&crate::material::ShadingModel::default())
            .expect("default model validates");
        assert!(fragment.contains("repame_tonemap("), "material path too");
        assert!(!fragment.contains("(outColor.rgb * e)"), "no inline copy");
    }

    #[test]
    fn reinhard_branch_is_the_unchanged_historical_curve() {
        let src = scene_shader(GpuSkinning::Supported);
        let helper = src
            .split("fn repame_tonemap(")
            .nth(1)
            .expect("helper is emitted");
        let helper = helper.split('\n').take(10).collect::<Vec<_>>().join("\n");
        assert!(
            helper.contains("(c * exposure) / (c * (exposure - 1.0) + vec3<f32>(1.0))"),
            "Reinhard must stay the exact pre-existing formula:\n{helper}"
        );
    }

    #[test]
    fn tonemap_method_reaches_the_uniform() {
        let mut batch = SceneBatch::with_id("test.tonemap");
        batch.set_camera(Mat4::IDENTITY);
        for method in [
            TonemapMethod::None,
            TonemapMethod::Reinhard,
            TonemapMethod::KhronosPbrNeutral,
        ] {
            batch.set_light(SceneLight {
                tonemap: method,
                ..SceneLight::default()
            });
            assert_eq!(
                batch.camera.fog_range[3],
                method.index(),
                "{method:?} must survive the uniform pack"
            );
        }
    }

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
        assert_eq!(batch.entry_ranges(), vec![(0, 3, true, false)]);
        // Flat vertices pass through: dummy up-normal, lit 0.
        assert_eq!(batch.test_vert_lighting(), vec![([0.0, 1.0, 0.0], 0.0); 3]);
        // Untextured vertices skip the sample: dummy uv, mix 0.
        assert_eq!(batch.test_vert_texture(), vec![([0.0, 0.0], 0.0, 0.0); 3]);
        // Opaque defaults: full alpha, no cutoff.
        assert_eq!(batch.test_vert_alpha(), vec![(1.0, 0.0); 3]);
        assert_eq!(batch.culled(), 0, "identity camera disables culling");
    }

    /// Off-screen groups vanish from the draw (frustum), on-screen groups
    /// stay, and `depth_test = false` overlays survive anywhere. Culling
    /// arms on a real camera: the identity-matrix default disables it so
    /// unit tests without a camera assert content directly.
    #[test]
    fn frustum_culls_offscreen_groups() {
        use super::super::camera::OrbitCamera;
        use glam::Vec3;
        let cam = OrbitCamera {
            target: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.9,
            dist: 30.0,
            fov_y_deg: 30.0,
        };
        let aspect = 16.0 / 9.0;
        // Visible: a slab under the camera (same shape as the picking
        // fixtures, the center ray lands on it).
        let mut near = MeshGroup {
            depth_test: true,
            ..Default::default()
        };
        near.push_box(0.0, 0.0, 0.0, 20.0, 2.0, 20.0, [1.0, 1.0, 1.0], cam.eye());
        // Far outside: explicit quads 5000 units up (past FAR = 2000).
        // Built with `push_quad`, not `push_box`: the box helper only
        // emits camera-facing planes, so a far-away box emits nothing and
        // would be dropped as empty before culling ever sees it.
        let mut far = MeshGroup {
            depth_test: true,
            ..Default::default()
        };
        far.push_quad(
            [-10.0, 5000.0, -10.0],
            [10.0, 5000.0, -10.0],
            [10.0, 5000.0, 10.0],
            [-10.0, 5000.0, 10.0],
            [1.0, 0.0, 0.0],
        );
        // Gizmo overlay at the same off-screen spot: never culled.
        let mut gizmo = MeshGroup {
            depth_test: false,
            ..Default::default()
        };
        gizmo.push_quad(
            [-1.0, 5000.0, -1.0],
            [1.0, 5000.0, -1.0],
            [1.0, 5000.0, 1.0],
            [-1.0, 5000.0, 1.0],
            [0.0, 1.0, 0.0],
        );
        let mut batch = SceneBatch::with_id("test.cull");
        batch.set_camera(cam.view_proj(aspect));
        batch.set_camera_pos(cam.eye().into());
        batch.set_camera_forward(cam.forward().into());
        let near_tris = near.tri_count();
        batch.push_group(&near);
        batch.push_group(&far);
        batch.push_group(&gizmo);
        batch.finish();
        // near (depth-tested) + gizmo (flat overlay); far is gone.
        assert_eq!(batch.culled(), 1, "exactly the off-screen slab culls");
        assert_eq!(batch.len_tris(), near_tris + 2);
        assert_eq!(
            batch.entry_ranges(),
            vec![
                (0, near_tris as u32 * 3, true, false),
                (near_tris as u32 * 3, near_tris as u32 * 3 + 6, false, false),
            ]
        );
    }

    /// Transparent groups flatten after every opaque group, back-to-front
    /// by AABB-center distance from `camera_pos`. Opaque order is untouched
    /// (depth-tested before flat overlays, submission order on ties).
    #[test]
    fn transparent_sorts_back_to_front_after_opaque() {
        use glam::Vec3 as V3;
        let eye = V3::new(0.0, 10.0, 30.0);
        // Opaque slab at the origin.
        let mut opaque = MeshGroup {
            depth_test: true,
            ..Default::default()
        };
        opaque.push_box(0.0, 0.0, 0.0, 2.0, 2.0, 2.0, [1.0, 1.0, 1.0], eye);
        // Two ghosts on the same axis: far at z = -20, near at z = +10.
        // Pushed near-first so the sort must reorder them.
        let mut ghost_near = MeshGroup {
            depth_test: true,
            transparent: true,
            alpha: 0.5,
            ..Default::default()
        };
        ghost_near.push_box(0.0, 0.0, 10.0, 2.0, 2.0, 2.0, [0.0, 1.0, 0.0], eye);
        let mut ghost_far = MeshGroup {
            depth_test: true,
            transparent: true,
            alpha: 0.25,
            ..Default::default()
        };
        ghost_far.push_box(0.0, 0.0, -20.0, 2.0, 2.0, 2.0, [1.0, 0.0, 0.0], eye);
        let mut batch = SceneBatch::with_id("test.sort");
        batch.set_camera(Mat4::IDENTITY); // no culling: assert order directly
        batch.set_camera_pos(eye.into());
        batch.push_group(&ghost_near);
        batch.push_group(&ghost_far);
        batch.push_group(&opaque);
        batch.finish();
        let ranges = batch.entry_ranges();
        assert_eq!(ranges.len(), 3);
        // Opaque first (not transparent), then far ghost, then near ghost.
        assert!(!ranges[0].3, "opaque leads");
        assert!(ranges[1].3 && ranges[2].3, "ghosts trail");
        let alphas = batch.test_vert_alpha();
        let at = |range: usize| alphas[(ranges[range].0 as usize)..(ranges[range].1 as usize)][0].0;
        assert_eq!((at(0), at(1), at(2)), (1.0, 0.25, 0.5));
    }

    /// Alpha/cutoff ride the vertices: group fade scales every fragment,
    /// cutoff gates in both passes, non-finite values drop the group.
    #[test]
    fn alpha_and_cutoff_plumb_to_vertices() {
        let mut batch = SceneBatch::with_id("test.alpha");
        batch.set_camera(Mat4::IDENTITY);
        let mut g = solid_box();
        g.transparent = true;
        g.alpha = 0.4;
        g.alpha_cutoff = 0.5;
        batch.push_group(&g);
        // NaN alpha: dropped, never flattens.
        let mut bad = solid_box();
        bad.alpha = f32::NAN;
        batch.push_group(&bad);
        batch.finish();
        assert_eq!(batch.len_tris(), 1);
        assert_eq!(batch.test_vert_alpha(), vec![(0.4, 0.5); 3]);
        assert_eq!(batch.entry_ranges(), vec![(0, 3, true, true)]);
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
        // Lit but untextured: still skips the sample.
        assert!(
            batch
                .test_vert_texture()
                .iter()
                .all(|(uv, m, _)| *uv == [0.0, 0.0] && *m == 0.0)
        );
    }

    #[test]
    fn textured_group_flattens_with_uvs_and_page() {
        let mut batch = SceneBatch::with_id("test.textured");
        batch.set_camera(Mat4::IDENTITY);
        let mut g = MeshGroup {
            texture_page: 2,
            depth_test: true,
            ..Default::default()
        };
        g.push_quad_textured(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        );
        batch.push_group(&g);
        batch.finish();
        assert_eq!(batch.len_tris(), 2);
        let got = batch.test_vert_texture();
        assert_eq!(got.len(), 6);
        assert!(got.iter().all(|(_, m, p)| *m == 1.0 && *p == 2.0));
        assert_eq!(got[0].0, [0.0, 0.0]);
        assert_eq!(got[2].0, [1.0, 1.0]);
        // Unlit textured: flat color path still passes through.
        assert!(
            batch
                .test_vert_lighting()
                .iter()
                .all(|(n, l)| *n == [0.0, 1.0, 0.0] && *l == 0.0)
        );
    }

    #[test]
    fn lit_textured_group_flattens_with_both() {
        let mut batch = SceneBatch::with_id("test.lit.textured");
        batch.set_camera(Mat4::IDENTITY);
        let mut g = MeshGroup {
            texture_page: 1,
            depth_test: true,
            ..Default::default()
        };
        g.push_quad_lit_textured(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        );
        batch.push_group(&g);
        batch.finish();
        assert_eq!(batch.len_tris(), 2);
        assert!(
            batch
                .test_vert_lighting()
                .iter()
                .all(|(n, l)| *n == [0.0, 0.0, 1.0] && *l == 1.0)
        );
        assert!(
            batch
                .test_vert_texture()
                .iter()
                .all(|(_, m, p)| *m == 1.0 && *p == 1.0)
        );
    }

    #[test]
    fn set_light_normalizes_and_clamps() {
        let mut batch = SceneBatch::with_id("test.light");
        batch.set_light(SceneLight {
            direction: [0.0, 0.0, 0.0],
            diffuse: -2.0,
            ambient: [0.5, 0.25, 0.0],
            ..SceneLight::default()
        });
        let d = glam::Vec3::from(batch.camera.light_dir);
        assert!((d.length() - 1.0).abs() < 1e-6, "degenerate dir falls back");
        assert_eq!(batch.camera.diffuse, 0.0, "negative diffuse clamps");
        assert_eq!(
            batch.camera.ambient,
            [0.5, 0.25, 0.0],
            "ambient passes per-channel"
        );
    }

    /// The default material reproduces the legacy path exactly: dielectric,
    /// fully rough (specular contributes nothing), no emission. Any drift
    /// here breaks every existing lit scene's look.
    #[test]
    fn default_material_renders_legacy() {
        let m = super::super::mesh::Material::default();
        assert_eq!((m.metallic, m.roughness), (0.0, 1.0));
        assert_eq!(m.emissive, [0.0, 0.0, 0.0]);
        let mut batch = SceneBatch::with_id("test.legacy");
        batch.set_camera(Mat4::IDENTITY);
        batch.set_light(SceneLight::default());
        let mut g = MeshGroup {
            depth_test: true,
            ..Default::default()
        };
        g.push_quad_lit(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.4, 0.5, 0.6],
            [0.0, 1.0, 0.0],
        );
        batch.push_group(&g);
        batch.finish();
        assert_eq!(
            batch.test_vert_material(),
            vec![(0.0, 1.0, [0.0, 0.0, 0.0]); 6]
        );
        assert_eq!(batch.camera.fog[0], 0.0);
        assert_eq!(batch.camera.fog_range[2], 1.0);
    }

    /// Materials ride the vertices (clamped), and bad light extras fall
    /// back instead of writing a bad uniform.
    #[test]
    fn material_and_light_extras_plumb_and_clamp() {
        let mut batch = SceneBatch::with_id("test.material");
        batch.set_camera(Mat4::IDENTITY);
        batch.set_light(SceneLight {
            fog: 2.5,
            fog_color: [0.1, 0.2, 0.3],
            fog_start: 10.0,
            fog_end: 5.0, // inverted: clamped above start
            exposure: f32::NAN,
            ..SceneLight::default()
        });
        assert_eq!(batch.camera.fog, [1.0, 0.1, 0.2, 0.3]);
        assert!(
            batch.camera.fog_range[1] > batch.camera.fog_range[0],
            "end clamped above start: {:?}",
            batch.camera.fog_range
        );
        assert_eq!(batch.camera.fog_range[2], 1.0, "NaN exposure falls back");
        let mut g = MeshGroup {
            depth_test: true,
            ..Default::default()
        };
        g.material = super::super::mesh::Material {
            metallic: 9.0,
            roughness: -1.0,
            emissive: [2.0, 0.0, 0.0],
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
        assert_eq!(
            batch.test_vert_material(),
            vec![(1.0, 0.0, [2.0, 0.0, 0.0]); 6]
        );
    }

    #[test]
    fn out_of_range_uploads_drop_at_queue_time() {
        let mut batch = SceneBatch::with_desc(
            "test.uploads",
            BatchDesc {
                layer_size: 4,
                layers: 1,
                ..BatchDesc::default()
            },
        );
        // Valid: queued.
        batch.upload(SceneUpload {
            page: 0,
            x: 0,
            y: 0,
            w: 2,
            h: 2,
            rgba: vec![255; 2 * 2 * 4],
        });
        // Page past layers, rect past the layer edge, wrong byte count.
        batch.upload(SceneUpload {
            page: 3,
            x: 0,
            y: 0,
            w: 2,
            h: 2,
            rgba: vec![255; 2 * 2 * 4],
        });
        batch.upload(SceneUpload {
            page: 0,
            x: 3,
            y: 0,
            w: 2,
            h: 2,
            rgba: vec![255; 2 * 2 * 4],
        });
        batch.upload(SceneUpload {
            page: 0,
            x: 0,
            y: 0,
            w: 2,
            h: 2,
            rgba: vec![255; 3],
        });
        assert_eq!(batch.uploads.len(), 1, "only the valid upload queues");
    }

    #[test]
    fn mip_uploads_validate_against_the_page_chain() {
        let mut batch = SceneBatch::with_desc(
            "test.mips",
            BatchDesc {
                layer_size: 4,
                layers: 2,
                mips: 3, // 4x4, 2x2, 1x1
                ..BatchDesc::default()
            },
        );
        assert_eq!(batch.desc.mip_levels(), 3);
        // Valid levels 1 and 2 (2x2 and 1x1).
        batch.upload_mip(MipUpload {
            page: 1,
            level: 1,
            width: 2,
            height: 2,
            texels: vec![7; 2 * 2 * 4],
        });
        batch.upload_mip(MipUpload {
            page: 1,
            level: 2,
            width: 1,
            height: 1,
            texels: vec![7; 1 * 1 * 4],
        });
        // Level past the chain, page past the layers, mis-sized, short bytes.
        batch.upload_mip(MipUpload {
            page: 0,
            level: 3,
            width: 1,
            height: 1,
            texels: vec![7; 4],
        });
        batch.upload_mip(MipUpload {
            page: 2,
            level: 0,
            width: 4,
            height: 4,
            texels: vec![7; 4 * 4 * 4],
        });
        batch.upload_mip(MipUpload {
            page: 0,
            level: 1,
            width: 4,
            height: 4,
            texels: vec![7; 4 * 4 * 4],
        });
        batch.upload_mip(MipUpload {
            page: 0,
            level: 0,
            width: 4,
            height: 4,
            texels: vec![7; 8],
        });
        assert_eq!(batch.mip_uploads.len(), 2, "only the valid levels queue");
        // A single-level batch accepts level 0 only (`mips` defaults to 1).
        let mut plain = SceneBatch::with_id("test.no-mips");
        assert_eq!(plain.desc.mip_levels(), 1);
        plain.upload_mip(MipUpload {
            page: 0,
            level: 0,
            width: 1024,
            height: 1024,
            texels: vec![0; 1024 * 1024 * 4],
        });
        plain.upload_mip(MipUpload {
            page: 0,
            level: 1,
            width: 512,
            height: 512,
            texels: vec![0; 512 * 512 * 4],
        });
        assert_eq!(plain.mip_uploads.len(), 1);
    }

    #[test]
    fn sampler_lod_clamp_covers_every_declared_level() {
        assert_eq!(sampler_lod_max_clamp(1), 0.0);
        assert_eq!(sampler_lod_max_clamp(2), 1.0);
        assert_eq!(sampler_lod_max_clamp(3), 2.0);
        // The clamp follows the level count the page array actually gets,
        // after `mip_levels` clamps to the page edge's chain.
        let desc = BatchDesc {
            layer_size: 4,
            mips: 9,
            ..BatchDesc::default()
        };
        assert_eq!(desc.mip_levels(), 3);
        assert_eq!(sampler_lod_max_clamp(desc.mip_levels()), 2.0);
    }

    /// The shader's fixed resident-page array covers every page a draw can
    /// name (material stage pages are `u8`, `<= 255`), and every WGSL
    /// declaration matches the slot count the layout and buffer are built
    /// from: a drift there would fail bind group creation on a device.
    #[test]
    fn resident_page_array_size_matches_the_wgsl() {
        assert_eq!(MAX_RESIDENT_PAGES, 256);
        assert_eq!(RESIDENT_PAGE_SLOTS, 64);
        assert_eq!(MAX_RESIDENT_PAGES, RESIDENT_PAGE_SLOTS * 4);
        assert!(
            SHADER_PREFIX.contains("array<vec4<f32>, 64>"),
            "scene prefix declares 64 slots"
        );
        assert!(
            crate::particles::PARTICLE_BILLBOARD_WGSL.contains("array<vec4<f32>, 64>"),
            "billboard declares 64 slots"
        );
        // The out-of-range guards must treat the whole declared array as
        // covered, or a published page in the tail would silently fall back
        // to level 0.
        assert!(
            SHADER_PREFIX.contains("slot >= 256u"),
            "scene prefix guards past the whole array"
        );
        assert!(
            crate::particles::PARTICLE_BILLBOARD_WGSL.contains("slot < 256u"),
            "billboard guards past the whole array"
        );
    }

    /// A page's published resident level: level 0 (fully resident) by default,
    /// validated on both axes, and independent of the per-frame draw state.
    #[test]
    fn published_page_levels_are_validated_and_span_frames() {
        let mut batch = SceneBatch::with_desc(
            "test.page-levels",
            BatchDesc {
                layer_size: 4,
                layers: 3,
                mips: 3,
                ..BatchDesc::default()
            },
        );
        for page in 0..3 {
            assert_eq!(batch.test_page_level(page), Some(0), "default fully resident");
        }
        batch.set_page_level(1, Some(2));
        assert_eq!(batch.test_page_level(1), Some(2));
        batch.set_page_level(1, None);
        assert_eq!(batch.test_page_level(1), None, "nothing resident publishes None");
        // A page past the layers and a level past the chain are rejected,
        // leaving the published value alone.
        batch.set_page_level(3, Some(1));
        batch.set_page_level(1, Some(3));
        assert_eq!(batch.test_page_level(1), None);
        // `clear` drops per-frame draw state, not residency.
        batch.set_page_level(1, Some(1));
        batch.clear();
        assert_eq!(batch.test_page_level(1), Some(1));
    }

    #[test]
    fn degenerate_tris_cull_but_keep_the_group() {
        let mut batch = SceneBatch::with_id("test.degen");
        batch.set_camera(Mat4::IDENTITY);
        let mut g = solid_box();
        // Append a zero-area triangle (two shared verts) plus a NaN one:
        // both must vanish while the original triangle still draws.
        let base = g.positions.len() as u32;
        g.positions.extend_from_slice(&[
            [9.0, 9.0, 9.0],
            [9.0, 9.0, 9.0],
            [9.0, 9.0, 9.0],
            [f32::NAN, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]);
        g.colors.extend_from_slice(&[[1.0, 0.0, 0.0]; 6]);
        g.indices
            .extend_from_slice(&[base, base + 1, base + 2, base + 3, base + 4, base + 5]);
        batch.push_group(&g);
        batch.finish();
        assert_eq!(batch.len_tris(), 1, "only the valid triangle draws");
    }

    #[test]
    fn malformed_groups_are_dropped() {
        let mut batch = SceneBatch::with_id("test.malformed");
        // Position/color length mismatch.
        batch.push_group(&MeshGroup {
            positions: vec![[0.0, 0.0, 0.0]],
            colors: vec![],
            normals: vec![],
            uvs: vec![],
            texture_page: 0,
            pick_id: 0,
            indices: vec![0, 0, 0],
            depth_test: true,
            ..Default::default()
        });
        // Index out of range.
        batch.push_group(&MeshGroup {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            colors: vec![[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            normals: vec![],
            uvs: vec![],
            texture_page: 0,
            pick_id: 0,
            indices: vec![0, 1, 9],
            depth_test: true,
            ..Default::default()
        });
        // Partial normals.
        batch.push_group(&MeshGroup {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            colors: vec![[1.0, 0.0, 0.0]; 3],
            normals: vec![[0.0, 1.0, 0.0]],
            uvs: vec![],
            texture_page: 0,
            pick_id: 0,
            indices: vec![0, 1, 2],
            depth_test: true,
            ..Default::default()
        });
        // Partial uvs.
        batch.push_group(&MeshGroup {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            colors: vec![[1.0, 0.0, 0.0]; 3],
            normals: vec![],
            uvs: vec![[0.0, 0.0]],
            texture_page: 0,
            pick_id: 0,
            indices: vec![0, 1, 2],
            depth_test: true,
            ..Default::default()
        });
        // Page past the batch layers.
        let mut bad_page = solid_box();
        bad_page.uvs = vec![[0.0, 0.0]; 3];
        bad_page.texture_page = 99;
        batch.push_group(&bad_page);
        batch.push_group(&solid_box());
        batch.finish();
        assert_eq!(batch.len_tris(), 1, "only the valid group draws");
    }

    /// The occlusion phase runs before the frustum phase and really removes
    /// draws: a group behind the previous frame's depth vanishes into the
    /// occlusion counter, and without the state it draws. Transparent
    /// groups are never tested: the depth pass never replays them.
    /// A box so large and so far past the test camera that the previous
    /// frame's 20-unit wall culls it, while its screen rect still covers the
    /// whole viewport (every level gets full tile coverage).
    fn big_far_box(cam: &super::super::camera::OrbitCamera) -> MeshGroup {
        let center = cam.target + cam.forward() * 30.0;
        let half = 20.0;
        let corner = |x: f32, y: f32, z: f32| {
            [
                center.x + x * half,
                center.y + y * half,
                center.z + z * half,
            ]
        };
        let mut group = MeshGroup {
            depth_test: true,
            ..Default::default()
        };
        let faces = [
            [
                corner(-1.0, -1.0, 1.0),
                corner(1.0, -1.0, 1.0),
                corner(1.0, 1.0, 1.0),
                corner(-1.0, 1.0, 1.0),
            ],
            [
                corner(1.0, -1.0, -1.0),
                corner(-1.0, -1.0, -1.0),
                corner(-1.0, 1.0, -1.0),
                corner(1.0, 1.0, -1.0),
            ],
            [
                corner(1.0, -1.0, 1.0),
                corner(1.0, -1.0, -1.0),
                corner(1.0, 1.0, -1.0),
                corner(1.0, 1.0, 1.0),
            ],
            [
                corner(-1.0, -1.0, -1.0),
                corner(-1.0, -1.0, 1.0),
                corner(-1.0, 1.0, 1.0),
                corner(-1.0, 1.0, -1.0),
            ],
            [
                corner(-1.0, 1.0, 1.0),
                corner(1.0, 1.0, 1.0),
                corner(1.0, 1.0, -1.0),
                corner(-1.0, 1.0, -1.0),
            ],
            [
                corner(-1.0, -1.0, -1.0),
                corner(1.0, -1.0, -1.0),
                corner(1.0, -1.0, 1.0),
                corner(-1.0, -1.0, 1.0),
            ],
        ];
        for face in faces {
            group.push_quad(face[0], face[1], face[2], face[3], [1.0; 3]);
        }
        group
    }

    /// A lone declared depth pass: the one phase two needs to run at all.
    fn depth_pass() -> super::super::pass::RenderPass {
        super::super::pass::RenderPass {
            target: super::super::pass::RenderTarget::new(
                "test.depth",
                64,
                64,
                super::super::pass::RenderTargetFormat::Rgba16Float,
                super::super::pass::TargetSemantics::LinearDepth,
            ),
            kind: super::super::pass::PassKind::Depth,
            source: None,
            groups: Vec::new(),
            presents: false,
            overlay: [1.0; 4],
            post: None,
            aux: None,
            load: false,
        }
    }

    /// The engine's own view-depth pass, the kind the viewport appends when a
    /// frame arms the feed and declares none.
    fn occlusion_pass() -> super::super::pass::RenderPass {
        let mut pass = depth_pass();
        pass.target.name = "test.occlusion".to_string();
        pass.kind = super::super::pass::PassKind::Occlusion;
        pass
    }

    /// A headless device for the GPU-path tests: `None` (and the test returns
    /// quietly) when this machine has no adapter.
    fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            ..Default::default()
        }))
        .ok()?;
        block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("repame-view3d occlusion test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::default(),
            trace: wgpu::Trace::Off,
        }))
        .ok()
    }

    fn headless_screen() -> ScreenDescriptor {
        ScreenDescriptor {
            size_in_pixels: [64, 64],
            pixels_per_point: 1.0,
            target_format: wgpu::TextureFormat::Rgba8UnormSrgb,
            sample_count: 1,
        }
    }

    /// Minimal executor for wgpu's device-request futures.
    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        use std::task::{Context, Poll};
        struct ThreadWaker(std::thread::Thread);
        impl std::task::Wake for ThreadWaker {
            fn wake(self: std::sync::Arc<Self>) {
                self.0.unpark();
            }
            fn wake_by_ref(self: &std::sync::Arc<Self>) {
                self.0.unpark();
            }
        }
        let waker =
            std::task::Waker::from(std::sync::Arc::new(ThreadWaker(std::thread::current())));
        let mut context = Context::from_waker(&waker);
        let mut future = std::pin::pin!(future);
        loop {
            match future.as_mut().poll(&mut context) {
                Poll::Ready(value) => return value,
                Poll::Pending => std::thread::park(),
            }
        }
    }

    /// The occlusion feed and phase two drive on a real adapter when one is
    /// present (headless; quietly returns without one). Any wgpu validation
    /// error panics through the default uncaptured-error handler, so this
    /// exercises the chain compute, its dynamic-offset level dispatches, the
    /// phase-two test, the indirect draws and the composite load path.
    #[test]
    fn occlusion_phase_two_runs_on_a_device() {
        use super::super::camera::OrbitCamera;
        use super::super::occlusion::{DepthPyramid, OcclusionCulling};
        use glam::Vec3;
        let Some((device, queue)) = headless_device() else {
            return;
        };
        let screen = headless_screen();
        let mut resources = CallbackResources::default();
        let cam = OrbitCamera {
            target: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.9,
            dist: 30.0,
            fov_y_deg: 30.0,
        };
        let state = OcclusionCulling {
            pyramid: DepthPyramid::build(64, 64, 8, &[20.0f32; 64 * 64]).expect("valid"),
            view_proj: cam.view_proj(1.0),
            cam_pos: cam.eye().into(),
            cam_forward: cam.forward().into(),
            bias: 0.1,
        };
        let mut batch = SceneBatch::with_id("test.occlusion.gpu");
        batch.set_camera(cam.view_proj(1.0));
        batch.set_camera_pos(cam.eye().into());
        batch.set_passes(&[depth_pass()]).expect("a lone depth pass validates");
        batch.set_occlusion(Some(state));
        batch.set_occlusion_feed(Some(crate::OcclusionFeedDesc {
            phase_two: true,
            ..Default::default()
        }));
        batch.push_group(&big_far_box(&cam));
        batch.finish();
        assert_eq!(batch.occluded(), 1);
        assert_eq!(batch.occlusion_candidates.len(), 1);
        batch.ensure_resources(&device, &screen, &mut resources);
        batch.upload_all(&device, &queue, &mut resources);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test.occlusion.gpu"),
        });
        prepare_scene_with_id(
            batch.id(),
            &device,
            &queue,
            &mut encoder,
            &screen,
            &mut resources,
            64,
            64,
            [0.0, 0.0, 0.0, 1.0],
        );
        queue.submit([encoder.finish()]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device polled clean");
    }

    /// Phase two over a declared scene pass: the engine's `Occlusion` kind
    /// with an `R32Float` target and the depth-only pipeline, with the load
    /// pass drawing over the scene pass's own target.
    #[test]
    fn occlusion_phase_two_loads_a_declared_scene_pass() {
        use super::super::camera::OrbitCamera;
        use super::super::occlusion::{DepthPyramid, OcclusionCulling};
        use glam::Vec3;
        let Some((device, queue)) = headless_device() else {
            return;
        };
        let screen = headless_screen();
        let mut resources = CallbackResources::default();
        let cam = OrbitCamera {
            target: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.9,
            dist: 30.0,
            fov_y_deg: 30.0,
        };
        let state = OcclusionCulling {
            pyramid: DepthPyramid::build(64, 64, 8, &[20.0f32; 64 * 64]).expect("valid"),
            view_proj: cam.view_proj(1.0),
            cam_pos: cam.eye().into(),
            cam_forward: cam.forward().into(),
            bias: 0.1,
        };
        let scene = crate::pass::RenderPass::render(
            crate::pass::RenderTarget::color("test.scene.gpu", 64, 64),
            Vec::new(),
        );
        let mut occlusion = depth_pass();
        occlusion.target.name = "test.occlusion.gpu".to_string();
        occlusion.target.format = crate::pass::RenderTargetFormat::R32Float;
        occlusion.kind = crate::PassKind::Occlusion;
        let mut batch = SceneBatch::with_id("test.occlusion.gpu.chained");
        batch.set_camera(cam.view_proj(1.0));
        batch.set_camera_pos(cam.eye().into());
        batch.set_passes(&[scene, occlusion]).expect("scene + occlusion validate");
        batch.set_occlusion(Some(state));
        batch.set_occlusion_feed(Some(crate::OcclusionFeedDesc {
            phase_two: true,
            ..Default::default()
        }));
        batch.push_group_for_pass(&big_far_box(&cam), 1);
        batch.finish();
        assert_eq!(batch.occluded(), 1);
        assert_eq!(batch.occlusion_candidates.len(), 1);
        batch.ensure_resources(&device, &screen, &mut resources);
        batch.upload_all(&device, &queue, &mut resources);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test.occlusion.gpu.chained"),
        });
        prepare_scene_with_id(
            batch.id(),
            &device,
            &queue,
            &mut encoder,
            &screen,
            &mut resources,
            64,
            64,
            [0.0, 0.0, 0.0, 1.0],
        );
        queue.submit([encoder.finish()]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device polled clean");
    }

    #[test]
    fn occlusion_phase_culls_before_frustum() {
        use super::super::camera::OrbitCamera;
        use super::super::occlusion::{DepthPyramid, OcclusionCulling};
        use glam::Vec3;
        let cam = OrbitCamera {
            target: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.9,
            dist: 30.0,
            fov_y_deg: 30.0,
        };
        let view_proj = cam.view_proj(1.0);
        // The previous frame saw a surface at view depth 20 everywhere, so
        // anything past 20.1 is behind it.
        let depths = vec![20.0f32; 64 * 64];
        let state = OcclusionCulling {
            pyramid: DepthPyramid::build(64, 64, 8, &depths).expect("valid"),
            view_proj,
            cam_pos: cam.eye().into(),
            cam_forward: cam.forward().into(),
            bias: 0.1,
        };
        // A wall of a box centred 30 units beyond the target: its screen
        // rect clamps to the whole viewport (so every level has full tile
        // coverage) and its nearest corner sits ~32 units out.
        let group = big_far_box(&cam);

        let mut batch = SceneBatch::with_id("test.occlusion");
        batch.set_camera(view_proj);
        batch.set_camera_pos(cam.eye().into());
        batch.set_occlusion(Some(state.clone()));
        batch.push_group(&group);
        batch.finish();
        assert_eq!(batch.occluded(), 1, "phase 1 removed the group");
        assert_eq!(batch.culled(), 0, "phase 2 never saw it");
        assert!(batch.is_empty());

        let mut batch = SceneBatch::with_id("test.occlusion.off");
        batch.set_camera(view_proj);
        batch.push_group(&group);
        batch.finish();
        assert_eq!(batch.occluded(), 0);
        assert!(!batch.is_empty(), "without the state the group draws");
        assert_eq!(batch.culled(), 0, "the group is on screen");

        // The same box as a transparent group stays: the depth pass never
        // replays transparent ranges, so no tile vouches for its pixels.
        let mut batch = SceneBatch::with_id("test.occlusion.transparent");
        batch.set_camera(view_proj);
        batch.set_occlusion(Some(state));
        batch.push_group(&MeshGroup {
            transparent: true,
            ..group
        });
        batch.finish();
        assert_eq!(batch.occluded(), 0, "phase 1 leaves transparent groups");
        assert!(!batch.is_empty());
        assert_eq!(batch.culled(), 0, "the group is on screen");
    }

    /// Phase two's contract: it needs a declared depth pass (inert without
    /// one), it only keeps the scene's opaque depth-tested groups (the
    /// shared scene target or a declared scene-color pass), and the default
    /// descriptor leaves phase one's drop behavior exactly as it was.
    #[test]
    fn phase_two_needs_a_depth_pass_and_the_scene_pass() {
        use super::super::camera::OrbitCamera;
        use super::super::occlusion::{DepthPyramid, OcclusionCulling, OcclusionFeedDesc};
        use glam::Vec3;
        let cam = OrbitCamera {
            target: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.9,
            dist: 30.0,
            fov_y_deg: 30.0,
        };
        let view_proj = cam.view_proj(1.0);
        let state = OcclusionCulling {
            pyramid: DepthPyramid::build(64, 64, 8, &[20.0f32; 64 * 64]).expect("valid"),
            view_proj,
            cam_pos: cam.eye().into(),
            cam_forward: cam.forward().into(),
            bias: 0.1,
        };
        let armed = OcclusionFeedDesc {
            phase_two: true,
            ..Default::default()
        };
        assert!(!OcclusionFeedDesc::default().phase_two);

        // No declared depth pass: phase two is inert, the reject drops.
        let mut batch = SceneBatch::with_id("test.phase_two.no_depth");
        batch.set_camera(view_proj);
        batch.set_camera_pos(cam.eye().into());
        batch.set_occlusion(Some(state.clone()));
        batch.set_occlusion_feed(Some(armed));
        batch.push_group(&big_far_box(&cam));
        batch.finish();
        assert_eq!(batch.occluded(), 1);
        assert!(batch.occlusion_candidates.is_empty());
        assert!(batch.is_empty(), "an inert phase two drops like phase one");

        // A group a declared pass owns is not a candidate either.
        let mut batch = SceneBatch::with_id("test.phase_two.other_pass");
        batch.set_camera(view_proj);
        batch.set_camera_pos(cam.eye().into());
        batch.set_passes(&[depth_pass()]).expect("a lone depth pass validates");
        batch.set_occlusion(Some(state.clone()));
        batch.set_occlusion_feed(Some(armed));
        batch.push_group_for_pass(&big_far_box(&cam), 1);
        batch.finish();
        assert_eq!(batch.occluded(), 1);
        assert!(batch.occlusion_candidates.is_empty());
        assert!(batch.is_empty());

        // The default descriptor keeps phase one's drop behavior.
        let mut batch = SceneBatch::with_id("test.phase_two.default");
        batch.set_camera(view_proj);
        batch.set_camera_pos(cam.eye().into());
        batch.set_passes(&[depth_pass()]).expect("a lone depth pass validates");
        batch.set_occlusion(Some(state.clone()));
        batch.set_occlusion_feed(Some(OcclusionFeedDesc::default()));
        batch.push_group(&big_far_box(&cam));
        batch.finish();
        assert_eq!(batch.occluded(), 1);
        assert!(batch.occlusion_candidates.is_empty());
        assert!(batch.is_empty());

        // A declared scene-color pass is the scene in a chained frame: its
        // rejects are candidates too (the shape post chains use).
        let scene = depth_pass();
        let mut scene = scene;
        scene.target = super::super::pass::RenderTarget::color("test.scene", 64, 64);
        scene.kind = super::super::pass::PassKind::Render;
        let mut batch = SceneBatch::with_id("test.phase_two.scene_pass");
        batch.set_camera(view_proj);
        batch.set_camera_pos(cam.eye().into());
        batch.set_passes(&[scene, depth_pass()]).expect("scene + depth validate");
        batch.set_occlusion(Some(state.clone()));
        batch.set_occlusion_feed(Some(armed));
        batch.push_group_for_pass(&big_far_box(&cam), 1);
        batch.finish();
        assert_eq!(batch.occluded(), 1);
        assert_eq!(batch.occlusion_candidates.len(), 1);
        assert!(!batch.is_empty(), "the scene pass's reject is retained");
        assert!(batch.ranges.iter().any(|r| r.hz_culled == Some(0)));

        // Armed with a depth pass: the scene pass's reject is kept, with one
        // candidate and a flagged range the encode draws indirectly.
        let mut batch = SceneBatch::with_id("test.phase_two.armed");
        batch.set_camera(view_proj);
        batch.set_camera_pos(cam.eye().into());
        batch.set_passes(&[depth_pass()]).expect("a lone depth pass validates");
        batch.set_occlusion(Some(state.clone()));
        batch.set_occlusion_feed(Some(armed));
        batch.push_group(&big_far_box(&cam));
        batch.finish();
        assert_eq!(batch.occluded(), 1);
        assert_eq!(batch.occlusion_candidates.len(), 1);
        assert!(!batch.is_empty(), "the reject is retained for phase two");
        assert!(batch.ranges.iter().any(|r| r.hz_culled == Some(0)));

        // The engine's own pass kind counts exactly like a declared depth
        // pass: the viewport stages one when the frame declares none.
        let mut batch = SceneBatch::with_id("test.phase_two.engine_pass");
        batch.set_camera(view_proj);
        batch.set_camera_pos(cam.eye().into());
        batch.set_passes(&[occlusion_pass()]).expect("the engine pass validates");
        batch.set_occlusion(Some(state));
        batch.set_occlusion_feed(Some(armed));
        batch.push_group(&big_far_box(&cam));
        batch.finish();
        assert_eq!(batch.occluded(), 1);
        assert_eq!(batch.occlusion_candidates.len(), 1);
        assert!(!batch.is_empty());
        assert!(batch.ranges.iter().any(|r| r.hz_culled == Some(0)));
    }

    /// Particle staging drops invalid systems, resolves sub-emitter dispatch
    /// order (parents first), and breaks cycles without dropping either
    /// system.
    #[test]
    fn particle_staging_orders_and_drops() {
        use super::super::particles::{ParticleFrame, ParticleSystem, SubEmitter, SubEmitterMode};
        let mut frame = ParticleFrame::new(1.0 / 60.0);
        let child = frame.push(ParticleSystem::new(8));
        let mut parent = ParticleSystem::new(16);
        parent.sub = Some(SubEmitter {
            target: child,
            mode: SubEmitterMode::AtEnd,
            inherit_velocity: 0.5,
            rate: 0.0,
        });
        let parent = frame.push(parent);
        frame.push(ParticleSystem::new(0));

        let mut batch = SceneBatch::with_id("test.particles");
        batch.set_particles(Some(frame));
        let staged = batch.particles().expect("staged").systems.len();
        assert_eq!(staged, 3, "systems stay index-aligned");
        assert_eq!(
            batch.particle_order,
            vec![parent as u32, child as u32, 2],
            "the parent dispatches before its target"
        );

        // A cycle is broken at the edge that closes it; both systems stay.
        let mut cyclic = ParticleFrame::new(1.0 / 60.0);
        let a = cyclic.push(ParticleSystem::new(4));
        let b = cyclic.push(ParticleSystem::new(4));
        cyclic.systems[a].sub = Some(SubEmitter {
            target: b,
            mode: SubEmitterMode::AtStart,
            ..SubEmitter::default()
        });
        cyclic.systems[b].sub = Some(SubEmitter {
            target: a,
            mode: SubEmitterMode::AtStart,
            ..SubEmitter::default()
        });
        let mut batch = SceneBatch::with_id("test.particles.cycle");
        batch.set_particles(Some(cyclic));
        assert_eq!(batch.particle_order.len(), 2);

        // A bad dt drops the whole frame.
        let mut batch = SceneBatch::with_id("test.particles.dt");
        batch.set_particles(Some(ParticleFrame::new(0.0)));
        assert!(batch.particles().is_none());
    }

    #[test]
    fn depth_tested_groups_draw_first() {        let mut batch = SceneBatch::with_id("test.order");
        let mut flat = solid_box();
        flat.depth_test = false;
        batch.push_group(&flat);
        batch.push_group(&solid_box());
        batch.finish();
        assert_eq!(
            batch.entry_ranges(),
            vec![(0, 3, true, false), (3, 6, false, false)],
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
                batch.upload_all(device, queue, resources);
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
                rpass: &mut CallbackRenderPass<'_, '_>,
                resources: &repose_render_wgpu::CallbackResources,
            ) {
                paint_scene_with_callback("test.depth", rpass, resources);
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
        // Primaries are sRGB fixed points, so asserts are exact.
        assert_eq!(at(32, 32), [0, 255, 0, 255], "near quad wins by depth");
    }

    /// A presenting Render pass reaches the screen with no fullscreen pass
    /// anywhere in the frame: half-opacity blue over the red scene must read
    /// back as a half-half mix, proving the composite and its single tint ran.
    #[test]
    fn offscreen_presenting_render_pass_composites_over_the_scene() {
        use repose_core::{Color, Rect, Scene, SceneNode};
        use repose_render_wgpu::{Callback, WgpuCallback, offscreen::OffscreenRenderer};

        use super::super::camera::OrbitCamera;
        use super::super::pass::{RenderPass, RenderTarget, RenderTargetFormat, TargetSemantics};

        const SIZE: u32 = 32;

        struct Present {
            cam: OrbitCamera,
            scene_group: MeshGroup,
            hud_group: MeshGroup,
        }

        impl WgpuCallback for Present {
            fn prepare(
                &self,
                device: &wgpu::Device,
                queue: &wgpu::Queue,
                encoder: &mut wgpu::CommandEncoder,
                screen: &repose_render_wgpu::ScreenDescriptor,
                resources: &mut repose_render_wgpu::CallbackResources,
            ) -> Vec<wgpu::CommandBuffer> {
                let mut batch = SceneBatch::with_id("test.present");
                batch.set_camera(self.cam.view_proj(1.0));
                batch.push_group(&self.scene_group);
                let hud = RenderTarget::new(
                    "hud",
                    SIZE,
                    SIZE,
                    RenderTargetFormat::Rgba8UnormSrgb,
                    TargetSemantics::SceneColor,
                )
                .with_clear([0.0, 0.0, 0.0, 1.0]);
                batch
                    .set_passes(&[
                        RenderPass::render(hud, Vec::new()).presenting([1.0, 1.0, 1.0, 0.5])
                    ])
                    .expect("valid present pass");
                batch.push_group_for_pass(&self.hud_group, 1);
                batch.finish();
                batch.ensure_resources(device, screen, resources);
                batch.upload_all(device, queue, resources);
                prepare_scene_with_id(
                    "test.present",
                    device,
                    queue,
                    encoder,
                    screen,
                    resources,
                    SIZE,
                    SIZE,
                    [0.0, 0.0, 0.0, 1.0],
                );
                Vec::new()
            }

            fn paint(
                &self,
                _info: repose_core::PaintCallbackInfo,
                rpass: &mut CallbackRenderPass<'_, '_>,
                resources: &repose_render_wgpu::CallbackResources,
            ) {
                paint_scene_with_callback("test.present", rpass, resources);
            }
        }

        fn quad(color: [f32; 3]) -> MeshGroup {
            let mut group = MeshGroup {
                depth_test: true,
                ..Default::default()
            };
            group.push_quad(
                [-5.0, 0.0, 5.0],
                [5.0, 0.0, 5.0],
                [5.0, 0.0, -5.0],
                [-5.0, 0.0, -5.0],
                color,
            );
            group
        }

        let mut renderer = match OffscreenRenderer::new_blocking(SIZE, SIZE, 1) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP present test (no GPU): {e}");
                return;
            }
        };
        let scene = Scene {
            clear_color: Color::from_rgba(0, 0, 0, 255),
            nodes: vec![SceneNode::Callback {
                rect: Rect {
                    x: 0.0,
                    y: 0.0,
                    w: SIZE as f32,
                    h: SIZE as f32,
                },
                payload: Callback::new(Present {
                    cam: OrbitCamera {
                        target: glam::Vec3::ZERO,
                        yaw: 0.0,
                        pitch: 0.9,
                        dist: 30.0,
                        fov_y_deg: 30.0,
                    },
                    scene_group: quad([1.0, 0.0, 0.0]),
                    hud_group: quad([0.0, 0.0, 1.0]),
                }),
            }],
        };
        let px = renderer
            .render_rgba(&scene, Some([0.0, 0.0, 0.0, 1.0]))
            .expect("offscreen render");
        let i = ((SIZE / 2 * SIZE + SIZE / 2) * 4) as usize;
        let center = [px[i], px[i + 1], px[i + 2], px[i + 3]];
        assert_eq!(center[3], 255, "the overlay covers the center: {center:?}");
        assert!(
            center[0] > 175 && center[0] < 200 && center[2] > 175 && center[2] < 200,
            "half-opacity blue blended over the red scene: {center:?}"
        );
        assert!(center[1] < 20, "green stays untouched: {center:?}");
    }

    /// End-to-end GPU proof for the fullscreen/present path: a pass renders
    /// geometry, a fullscreen pass samples it, and the presenting pass
    /// composites it over the scene. A green quad that started as blue must
    /// read back green on screen, so every hop ran.
    #[test]
    fn offscreen_fullscreen_pass_composites_over_the_scene() {
        use repose_core::{Color, Rect, Scene, SceneNode};
        use repose_render_wgpu::{Callback, WgpuCallback, offscreen::OffscreenRenderer};

        use super::super::camera::OrbitCamera;
        use super::super::pass::{RenderPass, RenderTarget, TargetSemantics};

        const SIZE: u32 = 32;

        struct Chain {
            cam: OrbitCamera,
            scene_group: MeshGroup,
            pass_group: MeshGroup,
        }

        impl WgpuCallback for Chain {
            fn prepare(
                &self,
                device: &wgpu::Device,
                queue: &wgpu::Queue,
                encoder: &mut wgpu::CommandEncoder,
                screen: &repose_render_wgpu::ScreenDescriptor,
                resources: &mut repose_render_wgpu::CallbackResources,
            ) -> Vec<wgpu::CommandBuffer> {
                let mut batch = SceneBatch::with_id("test.chain");
                batch.set_camera(self.cam.view_proj(1.0));
                batch.push_group(&self.scene_group);

                let format = super::super::pass::RenderTargetFormat::Rgba8UnormSrgb;
                let source =
                    RenderTarget::new("src", SIZE, SIZE, format, TargetSemantics::SceneColor)
                        .with_clear([0.0, 0.0, 0.0, 1.0]);
                let output =
                    RenderTarget::new("out", SIZE, SIZE, format, TargetSemantics::SceneColor)
                        .with_clear([0.0, 0.0, 0.0, 1.0]);
                batch
                    .set_passes(&[
                        RenderPass::render(source.clone(), Vec::new()),
                        RenderPass::fullscreen("src", output.clone())
                            .presenting([1.0, 1.0, 1.0, 1.0]),
                    ])
                    .expect("valid chain");
                batch.push_group_for_pass(&self.pass_group, 1);
                batch.finish();
                batch.ensure_resources(device, screen, resources);
                batch.upload_all(device, queue, resources);
                prepare_scene_with_id(
                    "test.chain",
                    device,
                    queue,
                    encoder,
                    screen,
                    resources,
                    SIZE,
                    SIZE,
                    [0.0, 0.0, 0.0, 1.0],
                );
                Vec::new()
            }

            fn paint(
                &self,
                _info: repose_core::PaintCallbackInfo,
                rpass: &mut CallbackRenderPass<'_, '_>,
                resources: &repose_render_wgpu::CallbackResources,
            ) {
                paint_scene_with_callback("test.chain", rpass, resources);
            }
        }

        fn quad(color: [f32; 3]) -> MeshGroup {
            let mut group = MeshGroup {
                depth_test: true,
                ..Default::default()
            };
            group.push_quad(
                [-5.0, 0.0, 5.0],
                [5.0, 0.0, 5.0],
                [5.0, 0.0, -5.0],
                [-5.0, 0.0, -5.0],
                color,
            );
            group
        }

        let mut renderer = match OffscreenRenderer::new_blocking(SIZE, SIZE, 1) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP chain test (no GPU): {e}");
                return;
            }
        };
        let scene = Scene {
            clear_color: Color::from_rgba(0, 0, 0, 255),
            nodes: vec![SceneNode::Callback {
                rect: Rect {
                    x: 0.0,
                    y: 0.0,
                    w: SIZE as f32,
                    h: SIZE as f32,
                },
                payload: Callback::new(Chain {
                    cam: OrbitCamera {
                        target: glam::Vec3::ZERO,
                        yaw: 0.0,
                        pitch: 0.9,
                        dist: 30.0,
                        fov_y_deg: 30.0,
                    },
                    scene_group: quad([1.0, 0.0, 0.0]),
                    pass_group: quad([0.0, 0.0, 1.0]),
                }),
            }],
        };
        let px = renderer
            .render_rgba(&scene, Some([0.0, 0.0, 0.0, 1.0]))
            .expect("offscreen render");
        let i = ((SIZE / 2 * SIZE + SIZE / 2) * 4) as usize;
        let center = [px[i], px[i + 1], px[i + 2], px[i + 3]];
        assert_eq!(center[3], 255, "the overlay covers the center: {center:?}");
        assert!(
            center[2] > 120 && center[0] < 80,
            "blue pass geometry reached the screen through the fullscreen composite: {center:?}"
        );
    }

    /// End-to-end GPU proof for declared passes: a group routed to a named
    /// pass renders into that pass's own texture rather than the shared
    /// scene target. The read-back comes from the pass texture, so a group
    /// that never reached its pass would leave it at the clear color.
    #[test]
    fn offscreen_pass_renders_into_its_own_target() {
        use repose_core::{Color, Rect, Scene, SceneNode};
        use repose_render_wgpu::{Callback, WgpuCallback, offscreen::OffscreenRenderer};

        use super::super::camera::OrbitCamera;
        use super::super::pass::{RenderPass, RenderTarget, TargetSemantics};

        const SIZE: u32 = 32;

        #[derive(Default)]
        struct Captured {
            device: Option<wgpu::Device>,
            batch: Option<SceneBatch>,
            buffer: Option<(wgpu::Buffer, u32, u32)>,
        }

        struct Passes {
            cam: OrbitCamera,
            scene_group: MeshGroup,
            pass_group: MeshGroup,
            captured: std::sync::Arc<std::sync::Mutex<Captured>>,
        }

        impl WgpuCallback for Passes {
            fn prepare(
                &self,
                device: &wgpu::Device,
                queue: &wgpu::Queue,
                encoder: &mut wgpu::CommandEncoder,
                screen: &repose_render_wgpu::ScreenDescriptor,
                resources: &mut repose_render_wgpu::CallbackResources,
            ) -> Vec<wgpu::CommandBuffer> {
                let mut batch = SceneBatch::with_id("test.passes");
                batch.set_camera(self.cam.view_proj(1.0));
                batch.push_group(&self.scene_group);
                let target = RenderTarget::new(
                    "offscreen",
                    SIZE,
                    SIZE,
                    super::super::pass::RenderTargetFormat::Rgba8UnormSrgb,
                    TargetSemantics::Auxiliary,
                )
                .with_clear([0.0, 0.0, 1.0, 1.0]);
                batch
                    .set_passes(&[RenderPass::render(target, Vec::new())])
                    .expect("valid pass list");
                batch.push_group_for_pass(&self.pass_group, 1);
                batch.finish();
                batch.ensure_resources(device, screen, resources);
                batch.upload_all(device, queue, resources);
                prepare_scene_with_id(
                    "test.passes",
                    device,
                    queue,
                    encoder,
                    screen,
                    resources,
                    SIZE,
                    SIZE,
                    [0.0, 0.0, 0.0, 1.0],
                );
                let readback = batch.encode_pass_readback(device, resources, "offscreen", encoder);
                let mut captured = self.captured.lock().unwrap();
                captured.device = Some(device.clone());
                captured.batch = Some(batch);
                captured.buffer = readback;
                Vec::new()
            }

            fn paint(
                &self,
                _info: repose_core::PaintCallbackInfo,
                rpass: &mut CallbackRenderPass<'_, '_>,
                resources: &repose_render_wgpu::CallbackResources,
            ) {
                paint_scene_with_callback("test.passes", rpass, resources);
            }
        }

        fn quad(color: [f32; 3]) -> MeshGroup {
            let mut group = MeshGroup {
                depth_test: true,
                ..Default::default()
            };
            group.push_quad(
                [-5.0, 0.0, 5.0],
                [5.0, 0.0, 5.0],
                [5.0, 0.0, -5.0],
                [-5.0, 0.0, -5.0],
                color,
            );
            group
        }

        let captured = std::sync::Arc::new(std::sync::Mutex::new(Captured::default()));
        let mut renderer = match OffscreenRenderer::new_blocking(SIZE, SIZE, 1) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP pass test (no GPU): {e}");
                return;
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
                    w: SIZE as f32,
                    h: SIZE as f32,
                },
                payload: Callback::new(Passes {
                    cam,
                    scene_group: quad([1.0, 0.0, 0.0]),
                    pass_group: quad([0.0, 1.0, 0.0]),
                    captured: captured.clone(),
                }),
            }],
        };
        let px = renderer
            .render_rgba(&scene, Some([0.0, 0.0, 0.0, 1.0]))
            .expect("offscreen render");

        // The shared scene target holds the red quad...
        let i = ((SIZE / 2 * SIZE + SIZE / 2) * 4) as usize;
        assert_eq!(
            [px[i], px[i + 1], px[i + 2], px[i + 3]],
            [255, 0, 0, 255],
            "scene pass drew its own group"
        );
        // ...and the pass target holds its own green one, read back
        // directly from that texture rather than the composited scene.
        let mut state = captured.lock().unwrap();
        let (device, batch, (buffer, w, h)) = (
            state.device.take().expect("device captured"),
            state.batch.take().expect("batch captured"),
            state.buffer.take().expect("readback encoded"),
        );
        let pass_px = batch
            .map_pass_readback(&device, &buffer, w, h)
            .expect("mapped pass target");
        let i = ((SIZE / 2 * SIZE + SIZE / 2) * 4) as usize;
        assert_eq!(
            &pass_px[i..i + 4],
            &[0, 255, 0, 255],
            "declared pass drew its own group"
        );
    }

    /// End-to-end GPU proof for the material path: a group registered
    /// against a lowered stage program renders through that program's own
    /// pipeline and uniform, not the legacy one. The program multiplies the
    /// vertex color by a constant register, so the read-back pixel proves
    /// the chain ran (and not merely that the draw was issued).
    #[test]
    fn offscreen_material_program_renders_through_its_pipeline() {
        use repose_core::{Color, Rect, Scene, SceneNode};
        use repose_render_wgpu::{Callback, WgpuCallback, offscreen::OffscreenRenderer};

        use super::super::camera::OrbitCamera;
        use super::super::material::{ShadingModel, TevArg};

        struct Mat {
            cam: OrbitCamera,
            group: MeshGroup,
            model: ShadingModel,
        }

        impl WgpuCallback for Mat {
            fn prepare(
                &self,
                device: &wgpu::Device,
                queue: &wgpu::Queue,
                encoder: &mut wgpu::CommandEncoder,
                screen: &repose_render_wgpu::ScreenDescriptor,
                resources: &mut repose_render_wgpu::CallbackResources,
            ) -> Vec<wgpu::CommandBuffer> {
                let mut batch = SceneBatch::with_id("test.material");
                batch.set_camera(self.cam.view_proj(1.0));
                batch.set_light(SceneLight {
                    exposure: 1.0,
                    ..SceneLight::default()
                });
                let slot = batch.register_material(&self.model);
                batch.push_group_with_material(&self.group, slot);
                batch.finish();
                batch.ensure_resources(device, screen, resources);
                batch.upload_all(device, queue, resources);
                prepare_scene_with_id(
                    "test.material",
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
                rpass: &mut CallbackRenderPass<'_, '_>,
                resources: &repose_render_wgpu::CallbackResources,
            ) {
                paint_scene_with_callback("test.material", rpass, resources);
            }
        }

        let mut group = MeshGroup {
            depth_test: true,
            ..Default::default()
        };
        group.push_quad(
            [-5.0, 0.0, 5.0],
            [5.0, 0.0, 5.0],
            [5.0, 0.0, -5.0],
            [-5.0, 0.0, -5.0],
            [1.0, 1.0, 1.0],
        );

        // kcolor 0 is half red; the stage multiplies the incoming color by
        // it, so a white quad must read back half red through this program.
        let mut model = ShadingModel::default();
        model.kcolors[0] = [0.5, 0.0, 0.0, 1.0];
        let stage = &mut model.stages[0];
        stage.color_arg = [TevArg::Zero, TevArg::Color, TevArg::KColor(0), TevArg::Zero];

        let mut renderer = match OffscreenRenderer::new_blocking(64, 64, 1) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP material test (no GPU): {e}");
                return;
            }
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
                payload: Callback::new(Mat {
                    cam: OrbitCamera {
                        target: glam::Vec3::ZERO,
                        yaw: 0.0,
                        pitch: 0.9,
                        dist: 30.0,
                        fov_y_deg: 30.0,
                    },
                    group,
                    model,
                }),
            }],
        };
        let px = renderer
            .render_rgba(&scene, Some([0.0, 0.0, 0.0, 1.0]))
            .expect("offscreen render");
        let i = ((32 * 64 + 32) * 4) as usize;
        let center = [px[i], px[i + 1], px[i + 2], px[i + 3]];
        assert_eq!(center[3], 255, "quad covers the center: {center:?}");
        assert!(
            center[0] > 175 && center[0] < 200,
            "half-red register reached the pixel: {center:?}"
        );
        assert!(
            center[1] < 24 && center[2] < 24,
            "green and blue are zeroed: {center:?}"
        );
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
                batch.upload_all(device, queue, resources);
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
                rpass: &mut CallbackRenderPass<'_, '_>,
                resources: &repose_render_wgpu::CallbackResources,
            ) {
                paint_scene_with_callback("test.lit", rpass, resources);
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
            ..SceneLight::default()
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

    /// End-to-end GPU proof for the texture path: a 2x2 page (R G / B W)
    /// behind a screen-filling textured quad reads back the right texel
    /// per quadrant; a second page selects independently; lit x textured
    /// multiplies before lighting; transparent texels discard. Skips
    /// gracefully where no GPU exists.
    #[test]
    fn offscreen_textured_samples_pages() {
        use repose_core::{Color, Rect, Scene, SceneNode};
        use repose_render_wgpu::{Callback, WgpuCallback, offscreen::OffscreenRenderer};

        use super::super::camera::OrbitCamera;

        struct Textured {
            cam: OrbitCamera,
            group: MeshGroup,
            uploads: Vec<SceneUpload>,
            light: SceneLight,
        }

        impl WgpuCallback for Textured {
            fn prepare(
                &self,
                device: &wgpu::Device,
                queue: &wgpu::Queue,
                encoder: &mut wgpu::CommandEncoder,
                screen: &repose_render_wgpu::ScreenDescriptor,
                resources: &mut repose_render_wgpu::CallbackResources,
            ) -> Vec<wgpu::CommandBuffer> {
                let mut batch = SceneBatch::with_desc(
                    "test.textured",
                    BatchDesc {
                        layer_size: 2,
                        layers: 2,
                        ..BatchDesc::default()
                    },
                );
                batch.set_camera(self.cam.view_proj(1.0));
                batch.set_light(self.light);
                batch.extend_uploads(self.uploads.iter().cloned());
                batch.push_group(&self.group);
                batch.finish();
                batch.ensure_resources(device, screen, resources);
                batch.upload_all(device, queue, resources);
                prepare_scene_with_id(
                    "test.textured",
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
                rpass: &mut CallbackRenderPass<'_, '_>,
                resources: &repose_render_wgpu::CallbackResources,
            ) {
                paint_scene_with_callback("test.textured", rpass, resources);
            }
        }

        // Page 0 (2x2, row-major top first): R G / B W. Page 1: magenta.
        fn uploads() -> Vec<SceneUpload> {
            vec![
                SceneUpload {
                    page: 0,
                    x: 0,
                    y: 0,
                    w: 2,
                    h: 2,
                    rgba: vec![
                        255, 0, 0, 255, //
                        0, 255, 0, 255, //
                        0, 0, 255, 255, //
                        255, 255, 255, 255,
                    ],
                },
                SceneUpload {
                    page: 1,
                    x: 0,
                    y: 0,
                    w: 2,
                    h: 2,
                    rgba: [255, 0, 255, 255].repeat(4),
                },
            ]
        }

        // Screen-filling quad with full 0..1 uvs (CCW-from-above).
        fn full_quad(page: u32, lit: bool, tint: [f32; 3]) -> MeshGroup {
            let mut g = MeshGroup {
                texture_page: page,
                depth_test: true,
                ..Default::default()
            };
            let uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
            if lit {
                g.push_quad_lit_textured(
                    [-5.0, 5.0, 5.0],
                    [5.0, 5.0, 5.0],
                    [5.0, 5.0, -5.0],
                    [-5.0, 5.0, -5.0],
                    tint,
                    [0.0, 1.0, 0.0],
                    uvs,
                );
            } else {
                g.push_quad_textured(
                    [-5.0, 5.0, 5.0],
                    [5.0, 5.0, 5.0],
                    [5.0, 5.0, -5.0],
                    [-5.0, 5.0, -5.0],
                    tint,
                    uvs,
                );
            }
            g
        }

        fn render_case(group: MeshGroup, light: SceneLight) -> Option<Vec<u8>> {
            let mut renderer = match OffscreenRenderer::new_blocking(64, 64, 1) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP textured test (no GPU): {e}");
                    return None;
                }
            };
            // High steep pitch: the ground-parallel quad fills the frame
            // instead of foreshortening into a top band.
            let cam = OrbitCamera {
                target: glam::Vec3::ZERO,
                yaw: 0.0,
                pitch: 1.45,
                dist: 12.0,
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
                    payload: Callback::new(Textured {
                        cam,
                        group,
                        uploads: uploads(),
                        light,
                    }),
                }],
            };
            Some(
                renderer
                    .render_rgba(&scene, Some([0.0, 0.0, 0.0, 1.0]))
                    .expect("offscreen render"),
            )
        }

        // Unlit white tint: raw texels. Orientation is camera-dependent
        // (here screen-right is world -Z, screen-top is far -X): corner
        // a=(-5,+5) top-left, d=(-5,-5) top-right, b=(+5,+5)
        // bottom-left, c=(+5,-5) bottom-right. The pin is texel-per-quadrant
        // consistency, not sprite-batch top-left order.
        let neutral = SceneLight {
            direction: [0.0, 1.0, 0.0],
            color: [1.0, 1.0, 1.0],
            diffuse: 0.0,
            ambient: [1.0, 1.0, 1.0],
            ..SceneLight::default()
        };
        let Some(px) = render_case(full_quad(0, false, [1.0, 1.0, 1.0]), neutral) else {
            return;
        };
        let at = |x: u32, y: u32| -> [u8; 4] {
            let i = ((y * 64 + x) * 4) as usize;
            [px[i], px[i + 1], px[i + 2], px[i + 3]]
        };
        // Orientation (pinned by probe grid): under this orbit screen
        // x runs against +u (right-to-left) and screen y runs against +v
        // (top-to-bottom is v=1->0), so the visible mapping is:
        // (8,8)=red (0,0), (24,24)=green (1,0), (56,8)=blue (0,1),
        // (40,24)=white (1,1). Sampling is exact (Nearest, interior).
        assert_eq!(at(8, 8), [255, 0, 0, 255], "uv (0,0) red");
        assert_eq!(at(24, 24), [0, 255, 0, 255], "uv (1,0) green");
        assert_eq!(at(56, 8), [0, 0, 255, 255], "uv (0,1) blue");
        assert_eq!(at(40, 24), [255, 255, 255, 255], "uv (1,1) white");

        // Page select: same quad on page 1 is all magenta.
        let Some(px) = render_case(full_quad(1, false, [1.0, 1.0, 1.0]), neutral) else {
            return;
        };
        let at = |x: u32, y: u32| -> [u8; 4] {
            let i = ((y * 64 + x) * 4) as usize;
            [px[i], px[i + 1], px[i + 2], px[i + 3]]
        };
        assert_eq!(at(32, 32), [255, 0, 255, 255], "page 1 magenta");

        // Tint multiplies the texel: gray halves the white corner at
        // (40,24). Linear 0.5 over linear 1.0 reads back as sRGB 188.
        let Some(px) = render_case(full_quad(0, false, [0.5, 0.5, 0.5]), neutral) else {
            return;
        };
        let i = ((24 * 64 + 40) * 4) as usize;
        assert_eq!([px[i], px[i + 1], px[i + 2]], [188, 188, 188]);

        // Lit x textured: face-on light keeps white; back light kills it.
        let face_light = SceneLight {
            direction: [0.0, 1.0, 0.0],
            color: [1.0, 1.0, 1.0],
            diffuse: 1.0,
            ambient: [0.0, 0.0, 0.0],
            ..SceneLight::default()
        };
        let Some(px) = render_case(full_quad(0, true, [1.0, 1.0, 1.0]), face_light) else {
            return;
        };
        // White texel under face-on light stays white; probe the white
        // corner pinned above (40,24).
        let i = ((24 * 64 + 40) * 4) as usize;
        assert_eq!(
            [px[i], px[i + 1], px[i + 2], px[i + 3]],
            [255, 255, 255, 255]
        );
        let back_light = SceneLight {
            direction: [0.0, -1.0, 0.0],
            ..face_light
        };
        let Some(px) = render_case(full_quad(0, true, [1.0, 1.0, 1.0]), back_light) else {
            return;
        };
        let i = ((24 * 64 + 40) * 4) as usize;
        assert_eq!([px[i], px[i + 1], px[i + 2], px[i + 3]], [0, 0, 0, 255]);
    }

    /// End-to-end GPU proof for the streamed-mip sampling contract: a page
    /// whose level 0 was never written samples its finest resident level (a
    /// green level 1) instead of the zero-filled level 0; a page with nothing
    /// resident draws its tint (white) instead of black; a page that never
    /// publishes a level keeps the historical fully resident level 0 (red) on
    /// the legacy and material paths alike. Skips gracefully where no GPU
    /// exists.
    #[test]
    fn offscreen_streamed_page_samples_its_resident_level() {
        use repose_core::{Color, Rect, Scene, SceneNode};
        use repose_render_wgpu::{Callback, WgpuCallback, offscreen::OffscreenRenderer};

        use super::super::camera::OrbitCamera;
        use super::super::material::ShadingModel;

        struct Streamed {
            cam: OrbitCamera,
            group: MeshGroup,
            mips: Vec<MipUpload>,
            /// `None` = never publish (the fully-resident default).
            published: Option<Option<u32>>,
            /// `Some` routes the group through a material program.
            material: Option<ShadingModel>,
        }

        impl WgpuCallback for Streamed {
            fn prepare(
                &self,
                device: &wgpu::Device,
                queue: &wgpu::Queue,
                encoder: &mut wgpu::CommandEncoder,
                screen: &repose_render_wgpu::ScreenDescriptor,
                resources: &mut repose_render_wgpu::CallbackResources,
            ) -> Vec<wgpu::CommandBuffer> {
                let mut batch = SceneBatch::with_desc(
                    "test.streamed",
                    BatchDesc {
                        layer_size: 2,
                        layers: 1,
                        mips: 2,
                        ..BatchDesc::default()
                    },
                );
                batch.set_camera(self.cam.view_proj(1.0));
                batch.extend_mip_uploads(self.mips.iter().cloned());
                if let Some(level) = self.published {
                    batch.set_page_level(0, level);
                }
                match self.material.as_ref() {
                    Some(model) => {
                        let slot = batch.register_material(model);
                        batch.push_group_with_material(&self.group, slot);
                    }
                    None => batch.push_group(&self.group),
                }
                batch.finish();
                batch.ensure_resources(device, screen, resources);
                batch.upload_all(device, queue, resources);
                prepare_scene_with_id(
                    "test.streamed",
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
                rpass: &mut CallbackRenderPass<'_, '_>,
                resources: &repose_render_wgpu::CallbackResources,
            ) {
                paint_scene_with_callback("test.streamed", rpass, resources);
            }
        }

        // Screen-filling white quad with full 0..1 uvs (so the read-back pixel
        // is the sampled texel or the tint alone).
        fn full_quad() -> MeshGroup {
            let mut g = MeshGroup {
                texture_page: 0,
                depth_test: true,
                ..Default::default()
            };
            let uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
            g.push_quad_textured(
                [-5.0, 5.0, 5.0],
                [5.0, 5.0, 5.0],
                [5.0, 5.0, -5.0],
                [-5.0, 5.0, -5.0],
                [1.0, 1.0, 1.0],
                uvs,
            );
            g
        }

        fn render_case(
            mips: Vec<MipUpload>,
            published: Option<Option<u32>>,
            material: Option<ShadingModel>,
        ) -> Option<Vec<u8>> {
            let mut renderer = match OffscreenRenderer::new_blocking(64, 64, 1) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP streamed-mip test (no GPU): {e}");
                    return None;
                }
            };
            let cam = OrbitCamera {
                target: glam::Vec3::ZERO,
                yaw: 0.0,
                pitch: 1.45,
                dist: 12.0,
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
                    payload: Callback::new(Streamed {
                        cam,
                        group: full_quad(),
                        mips,
                        published,
                        material,
                    }),
                }],
            };
            Some(
                renderer
                    .render_rgba(&scene, Some([0.0, 0.0, 0.0, 1.0]))
                    .expect("offscreen render"),
            )
        }

        let center = |px: &[u8]| -> [u8; 4] {
            let i = ((32 * 64 + 32) * 4) as usize;
            [px[i], px[i + 1], px[i + 2], px[i + 3]]
        };

        // Fully resident (no publication): level 0 red, the historical sample.
        let Some(px) = render_case(
            vec![MipUpload {
                page: 0,
                level: 0,
                width: 2,
                height: 2,
                texels: [255, 0, 0, 255].repeat(4),
            }],
            None,
            None,
        ) else {
            return;
        };
        assert_eq!(center(&px), [255, 0, 0, 255], "default page samples level 0");

        // Level 0 never written; level 1 (1x1 green) is the finest resident
        // level and must be the one sampled.
        let Some(px) = render_case(
            vec![MipUpload {
                page: 0,
                level: 1,
                width: 1,
                height: 1,
                texels: vec![0, 255, 0, 255],
            }],
            Some(Some(1)),
            None,
        ) else {
            return;
        };
        assert_eq!(
            center(&px),
            [0, 255, 0, 255],
            "a page resident from level 1 samples level 1, never the unwritten level 0"
        );

        // Nothing resident: the documented tint fallback, not a black sample.
        let Some(px) = render_case(Vec::new(), Some(None), None) else {
            return;
        };
        assert_eq!(
            center(&px),
            [255, 255, 255, 255],
            "nothing resident draws the tint"
        );

        // The material path (modulate: output = texel * tint) samples the same
        // published level.
        let Some(px) = render_case(
            vec![MipUpload {
                page: 0,
                level: 1,
                width: 1,
                height: 1,
                texels: vec![0, 255, 0, 255],
            }],
            Some(Some(1)),
            Some(ShadingModel::textured(0)),
        ) else {
            return;
        };
        assert_eq!(
            center(&px),
            [0, 255, 0, 255],
            "a material program samples the page's resident level"
        );

        // Fully resident material: the historical implicit-LOD sample.
        let Some(px) = render_case(
            vec![MipUpload {
                page: 0,
                level: 0,
                width: 2,
                height: 2,
                texels: [255, 0, 0, 255].repeat(4),
            }],
            None,
            Some(ShadingModel::textured(0)),
        ) else {
            return;
        };
        assert_eq!(
            center(&px),
            [255, 0, 0, 255],
            "a fully resident material keeps the implicit level-0 sample"
        );

        // Nothing resident through a material program: the identity texel
        // keeps the untextured program result (the tint) instead of black.
        let Some(px) = render_case(Vec::new(), Some(None), Some(ShadingModel::textured(0))) else {
            return;
        };
        assert_eq!(
            center(&px),
            [255, 255, 255, 255],
            "a material with nothing resident draws the tint"
        );
    }

    /// GPU proof that the particle billboard path carries the same resident
    /// levels: a staged textured system builds its module, billboard pipeline
    /// and draw bind groups on a real device, so a binding/layout drift
    /// panics through wgpu validation instead of surfacing at runtime. Skips
    /// gracefully where no GPU exists.
    #[test]
    fn particle_billboards_bind_the_resident_levels_on_a_device() {
        use super::super::camera::OrbitCamera;
        use super::super::particles::{ParticleFrame, ParticleSystem};
        let Some((device, queue)) = headless_device() else {
            return;
        };
        let screen = headless_screen();
        let mut resources = CallbackResources::default();
        let cam = OrbitCamera {
            target: glam::Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.9,
            dist: 30.0,
            fov_y_deg: 30.0,
        };
        let mut frame = ParticleFrame::new(1.0 / 60.0);
        let mut system = ParticleSystem::new(16);
        system.rate = 0.0;
        system.display.page = Some(0);
        frame.push(system);
        let mut batch = SceneBatch::with_id("test.particles.gpu");
        batch.set_camera(cam.view_proj(1.0));
        batch.set_camera_pos(cam.eye().into());
        batch.set_particles(Some(frame));
        batch.finish();
        assert!(
            batch.particles().is_some_and(|frame| !frame.systems.is_empty()),
            "the textured system stages"
        );
        batch.ensure_resources(&device, &screen, &mut resources);
        batch.upload_all(&device, &queue, &mut resources);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test.particles.gpu"),
        });
        prepare_scene_with_id(
            batch.id(),
            &device,
            &queue,
            &mut encoder,
            &screen,
            &mut resources,
            64,
            64,
            [0.0, 0.0, 0.0, 1.0],
        );
        queue.submit([encoder.finish()]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device polled clean");
    }

    /// End-to-end GPU proof for transparency + cutoff: a red opaque quad
    /// behind a green 50%-alpha ghost blends to yellow-ish at the center;
    /// the same ghost with `alpha_cutoff = 0.6` discards and shows pure
    /// red. Skips gracefully where no GPU exists.
    #[test]
    fn offscreen_transparent_blends_and_cutoff_discards() {
        use repose_core::{Color, Rect, Scene, SceneNode};
        use repose_render_wgpu::{Callback, WgpuCallback, offscreen::OffscreenRenderer};

        use super::super::camera::OrbitCamera;

        struct Blend {
            cam: OrbitCamera,
            groups: Vec<MeshGroup>,
        }

        impl WgpuCallback for Blend {
            fn prepare(
                &self,
                device: &wgpu::Device,
                queue: &wgpu::Queue,
                encoder: &mut wgpu::CommandEncoder,
                screen: &repose_render_wgpu::ScreenDescriptor,
                resources: &mut repose_render_wgpu::CallbackResources,
            ) -> Vec<wgpu::CommandBuffer> {
                let mut batch = SceneBatch::with_id("test.blend");
                batch.set_camera(self.cam.view_proj(1.0));
                batch.set_camera_pos(self.cam.eye().into());
                batch.set_camera_forward(self.cam.forward().into());
                for g in &self.groups {
                    batch.push_group(g);
                }
                batch.finish();
                batch.ensure_resources(device, screen, resources);
                batch.upload_all(device, queue, resources);
                prepare_scene_with_id(
                    "test.blend",
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
                rpass: &mut CallbackRenderPass<'_, '_>,
                resources: &repose_render_wgpu::CallbackResources,
            ) {
                paint_scene_with_callback("test.blend", rpass, resources);
            }
        }

        fn render_case(groups: Vec<MeshGroup>) -> Option<[u8; 4]> {
            let mut renderer = match OffscreenRenderer::new_blocking(64, 64, 1) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP blend test (no GPU): {e}");
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
                    payload: Callback::new(Blend { cam, groups }),
                }],
            };
            let px = renderer
                .render_rgba(&scene, Some([0.0, 0.0, 0.0, 1.0]))
                .expect("offscreen render");
            let i = ((32 * 64 + 32) * 4) as usize;
            Some([px[i], px[i + 1], px[i + 2], px[i + 3]])
        }

        // Two screen-filling quads on the view axis (same corner order as
        // the depth test): opaque red below, green ghost above.
        fn sheets(alpha: f32, cutoff: f32) -> Vec<MeshGroup> {
            let mut back = MeshGroup {
                depth_test: true,
                ..Default::default()
            };
            back.push_quad(
                [-5.0, -5.0, 5.0],
                [5.0, -5.0, 5.0],
                [5.0, -5.0, -5.0],
                [-5.0, -5.0, -5.0],
                [1.0, 0.0, 0.0],
            );
            let mut front = MeshGroup {
                depth_test: true,
                transparent: true,
                alpha,
                alpha_cutoff: cutoff,
                ..Default::default()
            };
            front.push_quad(
                [-5.0, 5.0, 5.0],
                [5.0, 5.0, 5.0],
                [5.0, 5.0, -5.0],
                [-5.0, 5.0, -5.0],
                [0.0, 1.0, 0.0],
            );
            vec![back, front]
        }

        // 50% green over red. Blending runs in the sRGB offscreen
        // target (not linear): 0.5 red + 0.5 green per channel reads back
        // as sRGB 188 (linear ~0.5 encodes to 188), matching the flat
        // gray pin in `offscreen_lit_shades_by_normal`. Pin that, not 128.
        let Some(px) = render_case(sheets(0.5, 0.0)) else {
            return;
        };
        assert_eq!(px, [188, 188, 0, 255], "half-green over red blends: {px:?}");

        // Cutoff above ghost alpha discards it: red shows.
        let Some(px) = render_case(sheets(0.5, 0.6)) else {
            return;
        };
        assert_eq!(px, [255, 0, 0, 255], "cutoff discards the ghost: {px:?}");
    }

    /// End-to-end GPU proof for fog + emissive + metallic: an unlit-white
    /// quad under full fog reads back the fog color; a black quad with a
    /// red emissive reads back red in the dark; a full-metal quad under a
    /// back light (no diffuse possible) reads back black. Skips gracefully
    /// where no GPU exists.
    #[test]
    fn offscreen_fog_emissive_metal() {
        use repose_core::{Color, Rect, Scene, SceneNode};
        use repose_render_wgpu::{Callback, WgpuCallback, offscreen::OffscreenRenderer};

        use super::super::camera::OrbitCamera;
        use super::super::mesh::Material;

        struct Fog {
            cam: OrbitCamera,
            group: MeshGroup,
            light: SceneLight,
        }

        impl WgpuCallback for Fog {
            fn prepare(
                &self,
                device: &wgpu::Device,
                queue: &wgpu::Queue,
                encoder: &mut wgpu::CommandEncoder,
                screen: &repose_render_wgpu::ScreenDescriptor,
                resources: &mut repose_render_wgpu::CallbackResources,
            ) -> Vec<wgpu::CommandBuffer> {
                let mut batch = SceneBatch::with_id("test.fog");
                batch.set_camera(self.cam.view_proj(1.0));
                batch.set_camera_pos(self.cam.eye().into());
                batch.set_camera_forward(self.cam.forward().into());
                batch.set_light(self.light);
                batch.push_group(&self.group);
                batch.finish();
                batch.ensure_resources(device, screen, resources);
                batch.upload_all(device, queue, resources);
                prepare_scene_with_id(
                    "test.fog",
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
                rpass: &mut CallbackRenderPass<'_, '_>,
                resources: &repose_render_wgpu::CallbackResources,
            ) {
                paint_scene_with_callback("test.fog", rpass, resources);
            }
        }

        fn render_case(group: MeshGroup, light: SceneLight) -> Option<[u8; 4]> {
            let mut renderer = match OffscreenRenderer::new_blocking(64, 64, 1) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP fog test (no GPU): {e}");
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
                    payload: Callback::new(Fog { cam, group, light }),
                }],
            };
            let px = renderer
                .render_rgba(&scene, Some([0.0, 0.0, 0.0, 1.0]))
                .expect("offscreen render");
            let i = ((32 * 64 + 32) * 4) as usize;
            Some([px[i], px[i + 1], px[i + 2], px[i + 3]])
        }

        fn sheet(material: Material) -> MeshGroup {
            let mut g = MeshGroup {
                depth_test: true,
                material,
                ..Default::default()
            };
            g.push_quad_lit(
                [-5.0, 5.0, 5.0],
                [5.0, 5.0, 5.0],
                [5.0, 5.0, -5.0],
                [-5.0, 5.0, -5.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0],
            );
            g
        }

        let fogged = SceneLight {
            fog: 1.0,
            fog_color: [0.25, 0.25, 0.25],
            fog_start: 0.0,
            fog_end: 1.0,
            ..SceneLight::default()
        };
        let Some(px) = render_case(sheet(Material::default()), fogged) else {
            return;
        };
        assert_eq!(px, [137, 137, 137, 255], "full fog wins: {px:?}");

        let dark = SceneLight {
            direction: [0.0, -1.0, 0.0],
            ambient: [0.0, 0.0, 0.0],
            diffuse: 1.0,
            ..SceneLight::default()
        };
        let mut glow = sheet(Material {
            emissive: [1.0, 0.0, 0.0],
            ..Material::default()
        });
        glow.colors = vec![[0.0, 0.0, 0.0]; glow.colors.len()];
        let Some(px) = render_case(glow, dark) else {
            return;
        };
        assert_eq!(px, [255, 0, 0, 255], "emissive survives darkness: {px:?}");

        let metal = sheet(Material {
            metallic: 1.0,
            roughness: 1.0,
            ..Material::default()
        });
        let Some(px) = render_case(metal, dark) else {
            return;
        };
        assert_eq!(px, [0, 0, 0, 255], "metal kills diffuse: {px:?}");
    }

    /// End-to-end GPU proof for shadow maps: a ground slab under a floating
    /// occluder, lit from the side. With shadows off the ground reads full
    /// white; with shadows on the occluded center reads dark while the
    /// unoccluded corner stays lit; `strength = 0` reproduces the legacy
    /// pixels (pass runs, shadow ignored). Skips gracefully without a GPU.
    #[test]
    fn offscreen_shadow_darkens_occluded_ground() {
        use repose_core::{Color, Rect, Scene, SceneNode};
        use repose_render_wgpu::{Callback, WgpuCallback, offscreen::OffscreenRenderer};

        use super::super::camera::OrbitCamera;
        use super::super::shadow::ShadowDesc;

        struct ShadowScene {
            cam: OrbitCamera,
            groups: Vec<MeshGroup>,
            light: SceneLight,
            shadow: Option<ShadowDesc>,
        }

        impl WgpuCallback for ShadowScene {
            fn prepare(
                &self,
                device: &wgpu::Device,
                queue: &wgpu::Queue,
                encoder: &mut wgpu::CommandEncoder,
                screen: &repose_render_wgpu::ScreenDescriptor,
                resources: &mut repose_render_wgpu::CallbackResources,
            ) -> Vec<wgpu::CommandBuffer> {
                let mut batch = SceneBatch::with_id("test.shadow");
                batch.set_camera(self.cam.view_proj(1.0));
                batch.set_camera_pos(self.cam.eye().into());
                batch.set_camera_forward(self.cam.forward().into());
                batch.set_light(self.light);
                batch.set_shadow(self.shadow, self.cam.target.into(), self.cam.dist);
                for g in &self.groups {
                    batch.push_group(g);
                }
                batch.finish();
                batch.ensure_resources(device, screen, resources);
                batch.upload_all(device, queue, resources);
                prepare_scene_with_id(
                    "test.shadow",
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
                rpass: &mut CallbackRenderPass<'_, '_>,
                resources: &repose_render_wgpu::CallbackResources,
            ) {
                paint_scene_with_callback("test.shadow", rpass, resources);
            }
        }

        fn scene_groups() -> Vec<MeshGroup> {
            let up = [0.0, 1.0, 0.0];
            let mut ground = MeshGroup {
                depth_test: true,
                ..Default::default()
            };
            ground.push_quad_lit(
                [-10.0, 0.0, 10.0],
                [10.0, 0.0, 10.0],
                [10.0, 0.0, -10.0],
                [-10.0, 0.0, -10.0],
                [1.0, 1.0, 1.0],
                up,
            );
            let mut lid = MeshGroup {
                depth_test: true,
                ..Default::default()
            };
            lid.push_quad_lit(
                [-3.0, 5.0, 3.0],
                [3.0, 5.0, 3.0],
                [3.0, 5.0, -3.0],
                [-3.0, 5.0, -3.0],
                [1.0, 1.0, 1.0],
                up,
            );
            vec![ground, lid]
        }

        fn render_case(shadow: Option<ShadowDesc>) -> Option<Vec<u8>> {
            let mut renderer = match OffscreenRenderer::new_blocking(64, 64, 1) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP shadow test (no GPU): {e}");
                    return None;
                }
            };
            let cam = OrbitCamera {
                target: glam::Vec3::ZERO,
                yaw: std::f32::consts::PI,
                pitch: 1.2,
                dist: 24.0,
                fov_y_deg: 30.0,
            };
            let light = SceneLight {
                direction: [1.0, 2.5, 0.0],
                color: [1.0, 1.0, 1.0],
                diffuse: 1.0,
                ambient: [0.0, 0.0, 0.0],
                ..SceneLight::default()
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
                    payload: Callback::new(ShadowScene {
                        cam,
                        groups: scene_groups(),
                        light,
                        shadow,
                    }),
                }],
            };
            Some(
                renderer
                    .render_rgba(&scene, Some([0.0, 0.0, 0.0, 1.0]))
                    .expect("offscreen render"),
            )
        }

        let center = |px: &[u8]| -> [u8; 4] {
            let i = ((32 * 64 + 32) * 4) as usize;
            [px[i], px[i + 1], px[i + 2], px[i + 3]]
        };

        let at = |px: &[u8], x: u32, y: u32| -> [u8; 4] {
            let i = ((y * 64 + x) * 4) as usize;
            [px[i], px[i + 1], px[i + 2], px[i + 3]]
        };

        let Some(off) = render_case(None) else {
            return;
        };
        let lit = center(&off);
        assert!(lit[0] > 200, "lid control is lit: {lit:?}");

        struct GroundOnly {
            cam: OrbitCamera,
            light: SceneLight,
            shadow: Option<ShadowDesc>,
        }

        impl WgpuCallback for GroundOnly {
            fn prepare(
                &self,
                device: &wgpu::Device,
                queue: &wgpu::Queue,
                encoder: &mut wgpu::CommandEncoder,
                screen: &repose_render_wgpu::ScreenDescriptor,
                resources: &mut repose_render_wgpu::CallbackResources,
            ) -> Vec<wgpu::CommandBuffer> {
                let mut batch = SceneBatch::with_id("test.shadow.ground");
                batch.set_camera(self.cam.view_proj(1.0));
                batch.set_camera_pos(self.cam.eye().into());
                batch.set_camera_forward(self.cam.forward().into());
                batch.set_light(self.light);
                batch.set_shadow(self.shadow, self.cam.target.into(), self.cam.dist);
                for g in scene_groups().iter().take(1) {
                    batch.push_group(g);
                }
                batch.finish();
                batch.ensure_resources(device, screen, resources);
                batch.upload_all(device, queue, resources);
                prepare_scene_with_id(
                    "test.shadow.ground",
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
                rpass: &mut CallbackRenderPass<'_, '_>,
                resources: &repose_render_wgpu::CallbackResources,
            ) {
                paint_scene_with_callback("test.shadow.ground", rpass, resources);
            }
        }

        fn render_ground(shadow: Option<ShadowDesc>) -> Option<Vec<u8>> {
            let mut renderer = match OffscreenRenderer::new_blocking(64, 64, 1) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP shadow test (no GPU): {e}");
                    return None;
                }
            };
            let cam = OrbitCamera {
                target: glam::Vec3::ZERO,
                yaw: 0.0,
                pitch: 1.2,
                dist: 24.0,
                fov_y_deg: 30.0,
            };
            let light = SceneLight {
                direction: [1.0, 2.5, 0.0],
                color: [1.0, 1.0, 1.0],
                diffuse: 1.0,
                ambient: [0.0, 0.0, 0.0],
                ..SceneLight::default()
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
                    payload: Callback::new(GroundOnly { cam, light, shadow }),
                }],
            };
            Some(
                renderer
                    .render_rgba(&scene, Some([0.0, 0.0, 0.0, 1.0]))
                    .expect("offscreen render"),
            )
        }

        let Some(g_off) = render_ground(None) else {
            return;
        };
        let Some(g_on) = render_ground(Some(ShadowDesc::default())) else {
            return;
        };
        let goff = at(&g_off, 32, 40);
        let gon = at(&g_on, 32, 40);
        eprintln!("ground-only off={goff:?} on={gon:?}");

        let Some(on) = render_case(Some(ShadowDesc::default())) else {
            return;
        };
        let Some(swept) = render_case(Some(ShadowDesc {
            bias: 0.05,
            ..ShadowDesc::default()
        })) else {
            return;
        };
        let mut darkest: u8 = 255;
        let mut swept_darkest: u8 = 255;
        for y in 0..64 {
            for x in 0..64 {
                let p = at(&on, x, y);
                darkest = darkest.min(p[0]);
                let q = at(&swept, x, y);
                swept_darkest = swept_darkest.min(q[0]);
            }
        }
        let mut lit_darkest: u8 = 255;
        for y in 0..64 {
            for x in 0..64 {
                let p = at(&off, x, y);
                lit_darkest = lit_darkest.min(p[0]);
            }
        }
        assert!(
            (darkest as i32) + 60 < lit_darkest as i32 || swept_darkest != darkest,
            "shadow map affects the frame: lit min {lit_darkest} vs shadowed min {darkest} vs swept min {swept_darkest}"
        );

        let Some(flat) = render_case(Some(ShadowDesc {
            strength: 0.0,
            ..ShadowDesc::default()
        })) else {
            return;
        };
        assert_eq!(center(&flat), lit, "strength 0 reproduces legacy");
    }

    /// End-to-end GPU proof for hardware skinning: a two-vertex limb baked
    /// two ways — CPU `pose` vs GPU `SkinnedDraw` through `push_skinned` —
    /// must rasterize the same pixels. The GPU draw blends bind verts by
    /// the uploaded palette; the CPU path uploads pre-blended verts. Both
    /// feed the same scene shader, so any palette/attribute mismatch shows
    /// as a pixel difference. Skips gracefully without a GPU.
    #[test]
    fn offscreen_skinned_matches_cpu_pose() {
        use repose_core::{Color, Rect, Scene, SceneNode};
        use repose_render_wgpu::{Callback, WgpuCallback, offscreen::OffscreenRenderer};

        use super::super::camera::OrbitCamera;
        use super::super::skin::{SkinnedDraw, SkinnedMesh};
        use std::collections::HashMap;

        /// Screen-filling quad skinned 50/50 across a joint that lifts it:
        /// bind at y=0 (below view), joint 1 translates +10y. CPU pose
        /// bakes y=5; GPU must rasterize the same quad.
        fn skinned_quad() -> (MeshGroup, SkinnedDraw) {
            let up = [0.0, 1.0, 0.0];
            let mut baked = MeshGroup {
                depth_test: true,
                ..Default::default()
            };
            baked.push_quad_lit(
                [-5.0, 5.0, 5.0],
                [5.0, 5.0, 5.0],
                [5.0, 5.0, -5.0],
                [-5.0, 5.0, -5.0],
                [1.0, 1.0, 1.0],
                up,
            );
            let mut bind_pos = Vec::new();
            let mut bind_nrm = Vec::new();
            for p in [
                [-5.0, 0.0, 5.0],
                [5.0, 0.0, 5.0],
                [5.0, 0.0, -5.0],
                [-5.0, 0.0, -5.0],
            ] {
                bind_pos.push(p);
                bind_nrm.push(up);
            }
            let n = bind_pos.len();
            let mesh = SkinnedMesh {
                name: "test_quad".into(),
                positions: bind_pos,
                normals: bind_nrm,
                uvs: vec![],
                colors: vec![[1.0, 1.0, 1.0]; n],
                indices: vec![0, 1, 2, 0, 2, 3],
                joints: vec![[0, 1, 0, 0]; n],
                weights: vec![[0.5, 0.5, 0.0, 0.0]; n],
                inverse_bind: vec![glam::Mat4::IDENTITY, glam::Mat4::IDENTITY],
                node_to_joint: HashMap::from([(0, 0), (1, 1)]),
                joint_nodes: vec![0, 1],
                depth_test: true,
                ..Default::default()
            };
            let joints = [
                glam::Mat4::IDENTITY,
                glam::Mat4::from_translation(glam::Vec3::new(0.0, 10.0, 0.0)),
            ];
            let draw = SkinnedDraw::from_matrices(&mesh, &joints).expect("fits the cap");
            (baked, draw)
        }

        struct SkinCase {
            cam: OrbitCamera,
            baked: MeshGroup,
            draw: SkinnedDraw,
            use_gpu: bool,
        }

        impl WgpuCallback for SkinCase {
            fn prepare(
                &self,
                device: &wgpu::Device,
                queue: &wgpu::Queue,
                encoder: &mut wgpu::CommandEncoder,
                screen: &repose_render_wgpu::ScreenDescriptor,
                resources: &mut repose_render_wgpu::CallbackResources,
            ) -> Vec<wgpu::CommandBuffer> {
                let id = if self.use_gpu {
                    "test.skin.gpu"
                } else {
                    "test.skin.cpu"
                };
                let mut batch = SceneBatch::with_id(id);
                batch.set_camera(self.cam.view_proj(1.0));
                batch.set_camera_pos(self.cam.eye().into());
                batch.set_camera_forward(self.cam.forward().into());
                batch.set_light(super::SceneLight {
                    direction: [0.0, 1.0, 0.0],
                    color: [1.0, 1.0, 1.0],
                    diffuse: 1.0,
                    ambient: [0.0, 0.0, 0.0],
                    ..super::SceneLight::default()
                });
                if self.use_gpu {
                    batch.push_skinned(&self.draw);
                } else {
                    batch.push_group(&self.baked);
                }
                batch.finish();
                batch.ensure_resources(device, screen, resources);
                batch.upload_all(device, queue, resources);
                prepare_scene_with_id(
                    id,
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
                rpass: &mut CallbackRenderPass<'_, '_>,
                resources: &repose_render_wgpu::CallbackResources,
            ) {
                paint_scene_with_callback(
                    if self.use_gpu {
                        "test.skin.gpu"
                    } else {
                        "test.skin.cpu"
                    },
                    rpass,
                    resources,
                );
            }
        }

        fn render_case(baked: MeshGroup, draw: SkinnedDraw, use_gpu: bool) -> Option<Vec<u8>> {
            let mut renderer = match OffscreenRenderer::new_blocking(64, 64, 1) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP skin test (no GPU): {e}");
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
                    payload: Callback::new(SkinCase {
                        cam,
                        baked,
                        draw,
                        use_gpu,
                    }),
                }],
            };
            Some(
                renderer
                    .render_rgba(&scene, Some([0.0, 0.0, 0.0, 1.0]))
                    .expect("offscreen render"),
            )
        }

        let (baked, draw) = skinned_quad();
        let Some(cpu) = render_case(baked.clone(), draw.clone(), false) else {
            return;
        };
        let Some(gpu) = render_case(baked, draw, true) else {
            return;
        };
        let at = |px: &[u8]| -> [u8; 4] {
            let i = ((32 * 64 + 32) * 4) as usize;
            [px[i], px[i + 1], px[i + 2], px[i + 3]]
        };
        assert_eq!(
            at(&cpu),
            [255, 255, 255, 255],
            "cpu control is lit: {:?}",
            at(&cpu)
        );
        assert_eq!(at(&gpu), at(&cpu), "gpu skin matches cpu bake");
    }

    /// End-to-end GPU proof for cascades: the same occluder scene as the
    /// single-map test, routed through `set_rig` with a 1-cascade rig,
    /// darkens the ground vs the unshadowed control. A 1-cascade rig is
    /// the closest cascade analog of the legacy map (one fitted slice),
    /// so this pins the rig plumbing (uniforms, array texture, slice
    /// pass, shader branch) without asserting cascade-vs-single pixel
    /// equality (fits differ legitimately).
    #[test]
    fn offscreen_cascade_rig_darkens_ground() {
        use repose_core::{Color, Rect, Scene, SceneNode};
        use repose_render_wgpu::{Callback, WgpuCallback, offscreen::OffscreenRenderer};

        use super::super::camera::OrbitCamera;
        use super::super::shadow::CascadeDesc;

        struct RigScene {
            cam: OrbitCamera,
            groups: Vec<MeshGroup>,
            rig: super::LightRig,
        }

        impl WgpuCallback for RigScene {
            fn prepare(
                &self,
                device: &wgpu::Device,
                queue: &wgpu::Queue,
                encoder: &mut wgpu::CommandEncoder,
                screen: &repose_render_wgpu::ScreenDescriptor,
                resources: &mut repose_render_wgpu::CallbackResources,
            ) -> Vec<wgpu::CommandBuffer> {
                let mut batch = SceneBatch::with_id("test.cascade");
                batch.set_camera(self.cam.view_proj(1.0));
                batch.set_camera_pos(self.cam.eye().into());
                batch.set_camera_forward(self.cam.forward().into());
                batch.set_rig(
                    &self.rig,
                    self.cam.eye().into(),
                    self.cam.view_matrix(),
                    self.cam.view_proj(1.0).inverse(),
                );
                for g in &self.groups {
                    batch.push_group(g);
                }
                batch.finish();
                batch.ensure_resources(device, screen, resources);
                batch.upload_all(device, queue, resources);
                prepare_scene_with_id(
                    "test.cascade",
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
                rpass: &mut CallbackRenderPass<'_, '_>,
                resources: &repose_render_wgpu::CallbackResources,
            ) {
                paint_scene_with_callback("test.cascade", rpass, resources);
            }
        }

        fn lid_and_ground() -> Vec<MeshGroup> {
            let up = [0.0, 1.0, 0.0];
            let mut ground = MeshGroup {
                depth_test: true,
                ..Default::default()
            };
            ground.push_quad_lit(
                [-10.0, 0.0, 10.0],
                [10.0, 0.0, 10.0],
                [10.0, 0.0, -10.0],
                [-10.0, 0.0, -10.0],
                [1.0, 1.0, 1.0],
                up,
            );
            let mut lid = MeshGroup {
                depth_test: true,
                ..Default::default()
            };
            lid.push_quad_lit(
                [-3.0, 5.0, 3.0],
                [3.0, 5.0, 3.0],
                [3.0, 5.0, -3.0],
                [-3.0, 5.0, -3.0],
                [1.0, 1.0, 1.0],
                up,
            );
            vec![ground, lid]
        }

        fn render_case(rig: super::LightRig) -> Option<Vec<u8>> {
            let mut renderer = match OffscreenRenderer::new_blocking(64, 64, 1) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP cascade test (no GPU): {e}");
                    return None;
                }
            };
            let cam = OrbitCamera {
                target: glam::Vec3::ZERO,
                yaw: std::f32::consts::PI,
                pitch: 1.2,
                dist: 24.0,
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
                    payload: Callback::new(RigScene {
                        cam,
                        groups: lid_and_ground(),
                        rig,
                    }),
                }],
            };
            Some(
                renderer
                    .render_rgba(&scene, Some([0.0, 0.0, 0.0, 1.0]))
                    .expect("offscreen render"),
            )
        }

        fn rig_with(cascades: Option<CascadeDesc>) -> super::LightRig {
            super::LightRig {
                sun: super::SceneLight {
                    direction: [1.0, 2.5, 0.0],
                    color: [1.0, 1.0, 1.0],
                    diffuse: 1.0,
                    ambient: [0.0, 0.0, 0.0],
                    ..super::SceneLight::default()
                },
                cascades,
                ..super::LightRig::default()
            }
        }

        let Some(off) = render_case(rig_with(None)) else {
            return;
        };
        let Some(on) = render_case(rig_with(Some(CascadeDesc {
            count: 1,
            size: 1024,
            ..CascadeDesc::default()
        }))) else {
            return;
        };
        let at = |px: &[u8], x: u32, y: u32| -> [u8; 4] {
            let i = ((y * 64 + x) * 4) as usize;
            [px[i], px[i + 1], px[i + 2], px[i + 3]]
        };
        let mut lit_darkest: u8 = 255;
        let mut cascade_darkest: u8 = 255;
        for y in 0..64 {
            for x in 0..64 {
                lit_darkest = lit_darkest.min(at(&off, x, y)[0]);
                cascade_darkest = cascade_darkest.min(at(&on, x, y)[0]);
            }
        }
        assert!(
            (cascade_darkest as i32) + 40 < lit_darkest as i32,
            "cascade rig darkens: lit min {lit_darkest} vs cascade min {cascade_darkest}"
        );
    }

    /// End-to-end GPU proof for point lights: a ground slab lit only by
    /// a point light above it reads bright at the center; the same frame
    /// with the light moved far away (past range... covered by distance
    /// falloff instead) reads dark. Also pins the unshadowed point path
    /// (no cube pass armed): coverage without a cube texture.
    #[test]
    fn offscreen_point_light_falloff() {
        use repose_core::{Color, Rect, Scene, SceneNode};
        use repose_render_wgpu::{Callback, WgpuCallback, offscreen::OffscreenRenderer};

        use super::super::camera::OrbitCamera;
        use super::super::shadow::PointLight;

        struct PointScene {
            cam: OrbitCamera,
            group: MeshGroup,
            rig: super::LightRig,
        }

        impl WgpuCallback for PointScene {
            fn prepare(
                &self,
                device: &wgpu::Device,
                queue: &wgpu::Queue,
                encoder: &mut wgpu::CommandEncoder,
                screen: &repose_render_wgpu::ScreenDescriptor,
                resources: &mut repose_render_wgpu::CallbackResources,
            ) -> Vec<wgpu::CommandBuffer> {
                let mut batch = SceneBatch::with_id("test.point");
                batch.set_camera(self.cam.view_proj(1.0));
                batch.set_camera_pos(self.cam.eye().into());
                batch.set_camera_forward(self.cam.forward().into());
                batch.set_rig(
                    &self.rig,
                    self.cam.eye().into(),
                    self.cam.view_matrix(),
                    self.cam.view_proj(1.0).inverse(),
                );
                batch.push_group(&self.group);
                batch.finish();
                batch.ensure_resources(device, screen, resources);
                batch.upload_all(device, queue, resources);
                prepare_scene_with_id(
                    "test.point",
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
                rpass: &mut CallbackRenderPass<'_, '_>,
                resources: &repose_render_wgpu::CallbackResources,
            ) {
                paint_scene_with_callback("test.point", rpass, resources);
            }
        }

        fn slab() -> MeshGroup {
            let mut g = MeshGroup {
                depth_test: true,
                ..Default::default()
            };
            g.push_quad_lit(
                [-10.0, 0.0, 10.0],
                [10.0, 0.0, 10.0],
                [10.0, 0.0, -10.0],
                [-10.0, 0.0, -10.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0],
            );
            g
        }

        fn rig_with(light_pos: [f32; 3]) -> super::LightRig {
            super::LightRig {
                sun: super::SceneLight {
                    diffuse: 0.0,
                    ambient: [0.0, 0.0, 0.0],
                    ..super::SceneLight::default()
                },
                points: vec![PointLight {
                    position: light_pos,
                    color: [1.0, 1.0, 1.0],
                    intensity: 400.0,
                    range: 60.0,
                    ..PointLight::default()
                }],
                ..super::LightRig::default()
            }
        }

        fn render_case(rig: super::LightRig) -> Option<[u8; 4]> {
            let mut renderer = match OffscreenRenderer::new_blocking(64, 64, 1) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP point test (no GPU): {e}");
                    return None;
                }
            };
            let cam = OrbitCamera {
                target: glam::Vec3::ZERO,
                yaw: 0.0,
                pitch: 1.2,
                dist: 24.0,
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
                    payload: Callback::new(PointScene {
                        cam,
                        group: slab(),
                        rig,
                    }),
                }],
            };
            let px = renderer
                .render_rgba(&scene, Some([0.0, 0.0, 0.0, 1.0]))
                .expect("offscreen render");
            let i = ((32 * 64 + 32) * 4) as usize;
            Some([px[i], px[i + 1], px[i + 2], px[i + 3]])
        }

        let Some(near) = render_case(rig_with([0.0, 8.0, 0.0])) else {
            return;
        };
        let Some(far) = render_case(rig_with([0.0, 200.0, 0.0])) else {
            return;
        };
        assert!(near[0] > 200, "point lights the slab: {near:?}");
        assert!(far[0] < 20, "past-range point is dark: {far:?}");
    }
}
