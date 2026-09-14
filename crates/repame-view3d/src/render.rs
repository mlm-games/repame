//! Scene batch: snapshot in, depth-tested pixels out.
//! Owns per-frame snapshot plus depth-tested and flat pipelines.
//! Offscreen target and blit live in `DepthComposite`.
//! Texture array is fed from [`SceneUpload`]s, one page per group.
//! Frustum cull runs per group in `finish`. `depth_test = false` overlays skip cull.
const SHADER: &str = r#"
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
};

struct SkinPalette {
    joints: array<mat4x4<f32>, 128>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> skin_palette: SkinPalette;
@group(1) @binding(0) var scene_tex: texture_2d_array<f32>;
@group(1) @binding(1) var scene_smp: sampler;
@group(2) @binding(0) var shadow_tex: texture_depth_2d;
@group(2) @binding(1) var shadow_smp: sampler_comparison;
@group(2) @binding(2) var cascade_tex: texture_depth_2d_array;
@group(2) @binding(3) var cascade_smp: sampler_comparison;
@group(2) @binding(4) var point_tex: texture_depth_cube;
@group(2) @binding(5) var point_smp: sampler_comparison;

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
            skinned_nrm = acc_nrm;
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
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var base = in.color;
    var alpha = in.alpha;
    if (in.tex_mix > 0.5) {
        let tex = textureSample(scene_tex, scene_smp, in.uv, i32(in.page + 0.5));
        if (tex.a < 0.001 && in.cutoff < 0.001) {
            discard;
        }
        base = in.color * tex.rgb;
        alpha = in.alpha * tex.a;
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
                    lit_count = lit_count + textureSampleCompare(
                        shadow_tex, shadow_smp, suv + vec2<f32>(f32(ox), f32(oy)) * texel, ref_depth);
                }
            }
            let lit_frac = lit_count / 9.0;
            shadow = mix(1.0, lit_frac, camera.shadow_params.x);
        }
    }
    if (in.lit_flag > 0.5 && camera.cascade_params.y > 0.5) {
        let view_depth = length(camera.cam_pos - in.world_pos);
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
                        lit_count = lit_count + textureSampleCompare(
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
                let lit_s = textureSampleCompare(
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
    let lit = base * camera.ambient + (camera.light_color * diffuse + spec) * shadow + point_accum + in.emissive;
    var rgb = mix(base, lit, in.lit_flag);
    let dist = length(camera.cam_pos - in.world_pos);
    let fog_t = clamp((dist - camera.fog_range.x) / max(camera.fog_range.y - camera.fog_range.x, 1e-6), 0.0, 1.0) * clamp(camera.fog.x, 0.0, 1.0);
    rgb = mix(rgb, camera.fog.yzw, fog_t * in.lit_flag);
    let e = camera.fog_range.z;
    rgb = (rgb * e) / (rgb * (e - 1.0) + vec3<f32>(1.0));
    return vec4<f32>(rgb, alpha);
}
"#;

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use repose_render_wgpu::{CallbackResources, DepthComposite, ScreenDescriptor};

use super::mesh::MeshGroup;
use super::skin::SkinnedDraw;

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
}

const _: () = assert!(size_of::<CameraUniform>() == 816);

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
    /// Reinhard exposure. 1.0 = off.
    pub exposure: f32,
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

/// Texture sampling for batch layers (same shape as `repame-sprite`).
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
}

impl Default for BatchDesc {
    fn default() -> Self {
        Self {
            layer_size: 1024,
            layers: 4,
            filter: SceneFilter::Nearest,
        }
    }
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
    indices: Vec<u32>,
    transparent: bool,
    alpha: f32,
    alpha_cutoff: f32,
    depth_test: bool,
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
}

/// Per-frame snapshot batch. `Send + Sync` so it can cross into the
/// compositor thread via [`repose_render_wgpu::Callback`].
///
/// Each `id` owns its pipelines + texture in `CallbackResources` (the
/// `repame-sprite` `SpriteBatch` pattern): viewports coexist. Rebuilding
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
    verts: Vec<Vert>,
    indices: Vec<u32>,
    ranges: Vec<DrawRange>,
    /// GPU-skinned draws submitted this frame (bind geometry + palette).
    /// Flattened after static groups in `finish`: one extra vertex block
    /// plus one `DrawRange` each (skinned ranges sort with the opaque
    /// pass; transparency on skinned draws follows the same back-to-front
    /// path by AABB center of the *bind* pose).
    skinned: Vec<SkinnedPending>,
    /// Joint palette store for the frame: every skinned draw appends its
    /// (padded) palette here; ranges record the base offset. One uniform
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
            },
            camera_pos: [0.0, 0.0, 0.0],
            view_proj: Mat4::IDENTITY,
            culled: 0,
            pending: Vec::new(),
            uploads: Vec::new(),
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
            0.0,
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

    /// Append one group. Malformed groups (index out of range, or
    /// position/color length mismatch, partial normals/uvs, page past the
    /// batch layers, non-finite alpha) are dropped with a warning, never
    /// a panic, never partial draws. Degenerate triangles (zero area, NaN)
    /// are culled tri-by-tri so one bad triangle can't sink its group: the
    /// group draws with the surviving triangles. Normals and uvs stay
    /// all-or-nothing per group. Shares
    /// [`validate_group`](crate::validate_group) with the chunk
    /// cache so both paths agree on malformed.
    pub fn push_group(&mut self, group: &MeshGroup) {
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
        // AABB center now, while positions are borrowed: `finish` sorts
        // transparent groups by it without re-walking vertices.
        let center = group.bounds().map(|(min, max)| {
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
            indices: cull_degenerate(&group.positions, &group.indices),
            transparent: group.transparent,
            alpha: group.alpha.clamp(0.0, 1.0),
            alpha_cutoff: group.alpha_cutoff.clamp(0.0, 1.0),
            depth_test: group.depth_test,
        });
    }

    /// Append one GPU-skinned draw: bind-pose geometry plus the sampled
    /// palette. The batch uploads bind verts once and the palette into
    /// the per-frame uniform; the vertex shader blends in hardware.
    /// Validation mirrors [`push_group`](SceneBatch::push_group) (lengths,
    /// index range, page, alpha) plus skin-specific checks: joints/weights
    /// must match positions when present, joint slots past the mesh's
    /// joint count clamp per vertex (warned once per draw, never a panic),
    /// and meshes past [`MAX_SKIN_JOINTS`](crate::MAX_SKIN_JOINTS) drop
    /// (caller falls back to CPU [`pose`](crate::SkinnedMesh::pose)).
    /// Degenerate triangles cull tri-by-tri like static groups.
    pub fn push_skinned(&mut self, draw: &SkinnedDraw) {
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
        let center: Option<[f32; 3]> = {
            let mut it = mesh.positions.iter();
            match it.next() {
                None => None,
                Some(first) => {
                    let mut min = glam::Vec3::from(*first);
                    let mut max = min;
                    for p in it {
                        let v = glam::Vec3::from(*p);
                        min = min.min(v);
                        max = max.max(v);
                    }
                    if min.is_finite() && max.is_finite() {
                        Some([
                            (min.x + max.x) * 0.5,
                            (min.y + max.y) * 0.5,
                            (min.z + max.z) * 0.5,
                        ])
                    } else {
                        None
                    }
                }
            }
        };
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
            center,
            indices: cull_degenerate(&mesh.positions, &mesh.indices),
            transparent: mesh.transparent,
            alpha: mesh.alpha.clamp(0.0, 1.0),
            alpha_cutoff: mesh.alpha_cutoff.clamp(0.0, 1.0),
            depth_test: mesh.depth_test,
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
        self.culled = 0;
        let planes = Self::frustum_planes(&self.view_proj);
        let culling = self.view_proj != Mat4::IDENTITY;
        let eye = glam::Vec3::from(self.camera_pos);
        // Stable partition: opaque keep submission order; transparent sort
        // back-to-front by AABB-center distance (descending). `sort_by`
        // (stable) on a pre-partitioned vec keeps both contracts.
        let mut opaque: Vec<Pending> = Vec::with_capacity(self.pending.len());
        let mut transparent: Vec<(Pending, f32)> = Vec::new();
        for g in self.pending.drain(..) {
            // Frustum culling (vertex-exact, only when the camera is real;
            // `depth_test = false` overlays are never culled, gizmos must
            // draw even off-screen).
            if culling && g.depth_test && Self::group_outside(&planes, &g.positions) {
                self.culled += 1;
                continue;
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
            self.ranges.push(DrawRange {
                index_start: start,
                index_end: self.indices.len() as u32,
                depth_test: g.depth_test,
                transparent: g.transparent,
            });
        }
        self.palette.clear();
        let eye = glam::Vec3::from(self.camera_pos);
        let mut skin_opaque: Vec<SkinnedPending> = Vec::new();
        let mut skin_transparent: Vec<(SkinnedPending, f32)> = Vec::new();
        for s in self.skinned.drain(..) {
            if s.indices.len() < 3 {
                continue;
            }
            if culling && s.depth_test && Self::group_outside(&planes, &s.positions) {
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
        for s in skin_opaque
            .into_iter()
            .chain(skin_transparent.into_iter().map(|(s, _)| s))
        {
            if s.indices.len() < 3 {
                continue;
            }
            if culling && s.depth_test && Self::group_outside(&planes, &s.positions) {
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
            self.ranges.push(DrawRange {
                index_start: start,
                index_end: self.indices.len() as u32,
                depth_test,
                transparent,
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
        let key = (
            screen.target_format,
            screen.sample_count,
            self.desc.layer_size,
            self.desc.layers,
            self.desc.filter as u32,
            shadow_size,
            cascade_size,
            cascade_count,
            cube_size,
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
                "scene_batch[{}]: rebuilding pipeline/texture (format/sample/desc/shadow/rig changed); texture contents dropped, re-upload required",
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
            mip_level_count: 1,
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
            lod_max_clamp: 1.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });
        let cam_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("repame_view3d_cam_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let skin_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("repame_view3d_skin"),
            size: (crate::MAX_SKIN_JOINTS * 64) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cam_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("repame_view3d_cam_bg"),
            layout: &cam_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: skin_buffer.as_entire_binding(),
                },
            ],
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
            ],
        });
        let shadow_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("repame_view3d_shadow_bg"),
            layout: &shadow_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&cascade_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&cascade_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&point_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&point_sampler),
                },
            ],
        });
        let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("repame_view3d_pl"),
            bind_group_layouts: &[Some(&cam_layout), Some(&tex_layout), Some(&shadow_layout)],
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
            ],
        })];
        let mk = |label: &'static str, depth_test: bool, transparent: bool| {
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
    joints: array<mat4x4<f32>, 128>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> skin_palette: SkinPalette;

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

        let shadow_depth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("repame_view3d_shadow_depth"),
            source: wgpu::ShaderSource::Wgsl(SHADOW_DEPTH_SHADER.into()),
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
        let shadow_cam_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("repame_view3d_shadow_cam_bg"),
            layout: &cam_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shadow_camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: skin_buffer.as_entire_binding(),
                },
            ],
        });
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
        let entry = SceneEntry {
            key,
            pipeline_depth: mk("repame_view3d_depth", true, false),
            pipeline_flat: mk("repame_view3d_flat", false, false),
            pipeline_transparent: mk("repame_view3d_transparent", true, true),
            pipeline_transparent_flat: mk("repame_view3d_transparent_flat", false, true),
            shadow_depth_pipeline,
            shadow_cam_bind,
            shadow_camera_buf,
            shadow_bind,
            shadow_view,
            shadow_edge,
            cascade_tex,
            cascade_edge,
            point_tex,
            point_edge: cube_edge,
            point_faces: None,
            skin_buffer,
            skin_cap: 0,
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
            tex_bind,
            texture,
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
        res.camera_mat = self.camera;
        res.point_faces = self
            .point_caster
            .and_then(|i| self.staged_points.get(i))
            .and_then(|p| p.cube_faces());
        if !self.palette.is_empty() {
            queue.write_buffer(&res.skin_buffer, 0, bytemuck::cast_slice(&self.palette));
            res.skin_cap = self.palette.len();
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
        res.last_ranges = self.ranges.clone();
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
}

struct SceneEntry {
    key: (wgpu::TextureFormat, u32, u32, u32, u32, u32, u32, u32, u32),
    pipeline_depth: wgpu::RenderPipeline,
    pipeline_flat: wgpu::RenderPipeline,
    pipeline_transparent: wgpu::RenderPipeline,
    pipeline_transparent_flat: wgpu::RenderPipeline,
    shadow_depth_pipeline: wgpu::RenderPipeline,
    shadow_cam_bind: wgpu::BindGroup,
    shadow_camera_buf: wgpu::Buffer,
    shadow_bind: wgpu::BindGroup,
    shadow_view: wgpu::TextureView,
    shadow_edge: u32,
    cascade_tex: wgpu::Texture,
    cascade_edge: u32,
    point_tex: wgpu::Texture,
    point_edge: u32,
    /// Caster cube faces sampled this frame (six VPs), written by
    /// `upload_all` from the staged points. `None` = cube pass off.
    point_faces: Option<[[[f32; 4]; 4]; 6]>,
    skin_buffer: wgpu::Buffer,
    /// Joint count currently uploaded (matrices). Grows like the vert
    /// buffers; shrinks never (uniform upload is a prefix write).
    skin_cap: usize,
    verts: wgpu::Buffer,
    vert_cap: usize,
    indices: wgpu::Buffer,
    index_cap: usize,
    camera: wgpu::Buffer,
    cam_bind: wgpu::BindGroup,
    tex_bind: wgpu::BindGroup,
    texture: wgpu::Texture,
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
    // shared resources: clone the handles, then end the borrow
    // before beginning the mutable scene pass.
    struct Snapshot {
        cam_bind: wgpu::BindGroup,
        tex_bind: wgpu::BindGroup,
        shadow_bind: wgpu::BindGroup,
        shadow_depth_pipeline: wgpu::RenderPipeline,
        shadow_cam_bind: wgpu::BindGroup,
        shadow_camera_buf: wgpu::Buffer,
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
        let mat = res.camera_mat;
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
            tex_bind: res.tex_bind.clone(),
            shadow_bind: res.shadow_bind.clone(),
            shadow_depth_pipeline: res.shadow_depth_pipeline.clone(),
            shadow_cam_bind: res.shadow_cam_bind.clone(),
            shadow_camera_buf: res.shadow_camera_buf.clone(),
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
            ranges: res.last_ranges.clone(),
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
            if !r.transparent && r.depth_test {
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
        queue.write_buffer(&snap.shadow_camera_buf, 0, bytemuck::cast_slice(&vp));
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
            cpass.set_bind_group(0, &snap.shadow_cam_bind, &[]);
            cpass.set_vertex_buffer(0, snap.verts.slice(..));
            cpass.set_index_buffer(snap.indices.slice(..), wgpu::IndexFormat::Uint32);
            for r in &snap.ranges {
                if !r.transparent && r.depth_test {
                    cpass.draw_indexed(r.index_start..r.index_end, 0, 0..1);
                }
            }
        }
    }
    if let Some(faces) = snap.point_faces {
        for (face, vp) in faces.iter().enumerate() {
            queue.write_buffer(&snap.shadow_camera_buf, 0, bytemuck::cast_slice(vp));
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
                ppass.set_bind_group(0, &snap.shadow_cam_bind, &[]);
                ppass.set_vertex_buffer(0, snap.verts.slice(..));
                ppass.set_index_buffer(snap.indices.slice(..), wgpu::IndexFormat::Uint32);
                for r in &snap.ranges {
                    if !r.transparent && r.depth_test {
                        ppass.draw_indexed(r.index_start..r.index_end, 0, 0..1);
                    }
                }
            }
        }
    }
    let composite = DepthComposite::get(resources);
    let Some(mut pass) = composite.begin_scene(id, encoder, clear) else {
        return;
    };
    pass.set_bind_group(0, &snap.cam_bind, &[]);
    pass.set_bind_group(1, &snap.tex_bind, &[]);
    pass.set_bind_group(2, &snap.shadow_bind, &[]);
    pass.set_vertex_buffer(0, snap.verts.slice(..));
    pass.set_index_buffer(snap.indices.slice(..), wgpu::IndexFormat::Uint32);
    for r in &snap.ranges {
        pass.set_pipeline(match (r.transparent, r.depth_test) {
            (true, true) => &snap.pipeline_transparent,
            (true, false) => &snap.pipeline_transparent_flat,
            (false, true) => &snap.pipeline_depth,
            (false, false) => &snap.pipeline_flat,
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
        // Primaries are sRGB fixed points, so asserts are exact.
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
                rpass: &mut wgpu::RenderPass<'static>,
                resources: &repose_render_wgpu::CallbackResources,
            ) {
                paint_scene_with_id("test.textured", rpass, resources);
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
                rpass: &mut wgpu::RenderPass<'static>,
                resources: &repose_render_wgpu::CallbackResources,
            ) {
                paint_scene_with_id("test.blend", rpass, resources);
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
                rpass: &mut wgpu::RenderPass<'static>,
                resources: &repose_render_wgpu::CallbackResources,
            ) {
                paint_scene_with_id("test.fog", rpass, resources);
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
                rpass: &mut wgpu::RenderPass<'static>,
                resources: &repose_render_wgpu::CallbackResources,
            ) {
                paint_scene_with_id("test.shadow", rpass, resources);
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
                rpass: &mut wgpu::RenderPass<'static>,
                resources: &repose_render_wgpu::CallbackResources,
            ) {
                paint_scene_with_id("test.shadow.ground", rpass, resources);
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
                rpass: &mut wgpu::RenderPass<'static>,
                resources: &repose_render_wgpu::CallbackResources,
            ) {
                paint_scene_with_id(
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
                rpass: &mut wgpu::RenderPass<'static>,
                resources: &repose_render_wgpu::CallbackResources,
            ) {
                paint_scene_with_id("test.cascade", rpass, resources);
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
                rpass: &mut wgpu::RenderPass<'static>,
                resources: &repose_render_wgpu::CallbackResources,
            ) {
                paint_scene_with_id("test.point", rpass, resources);
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
