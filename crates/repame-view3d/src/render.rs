//! Flat-shaded + single-light 3D scene batch: snapshot in, depth-tested pixels out.
//!
//! [`SceneBatch`] owns the per-frame mesh snapshot (validate → flatten →
//! upload) plus its depth-tested/flat pipelines.
//!
//! The offscreen scene target + depth buffer + blit live in
//! [`DepthComposite`](repose_render_wgpu::DepthComposite): the shared UI
//! pass carries no depth ops, so depth-tested content renders into a
//! viewport-owned target during `prepare` and composites back in `paint`.
//!
//! Lit/texture plumbing mirrors `repame-sprite` `SpriteBatch`: the batch
//! owns a texture array fed from [`SceneUpload`]s (games drain their
//! image/atlas source once per frame), and each group carries one page.
//! Groups without uvs sample nothing; groups with uvs multiply the texel
//! into the tint before lighting. `BatchDesc` matches the sprite batch
//! shape (layer count + size + filter) so asset code reads the same.
//!
//! Frustum culling runs at group granularity in [`SceneBatch::finish`]
//! (AABB vs the six view-projection planes, extracted per frame —
//! `groups_culled` reports the count for HUDs). `depth_test = false`
//! overlays (gizmos, decals) are never culled: they must draw even when
//! their bounds sit off-screen.

//! Flat-shaded 3D pass: world-space pos+color through a view-projection uniform.
//! Lit groups add per-vertex normals and sample the frame light from the
//! same uniform block (ambient + one directional, linear space).
//! Textured groups add uvs + a page and sample the batch texture array;
//! the texel multiplies the tint before lighting (untextured vertices
//! carry a dummy uv and page 0 with weight 0, so one pipeline fits all).
const SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    light_dir: vec3<f32>,
    _pad0: f32,
    light_color: vec3<f32>,
    diffuse: f32,
    ambient: vec3<f32>,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var scene_tex: texture_2d_array<f32>;
@group(1) @binding(1) var scene_smp: sampler;

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
) -> VsOut {
    var out: VsOut;
    out.pos = camera.view_proj * vec4<f32>(pos, 1.0);
    out.color = color;
    out.normal = normal;
    out.lit_flag = lit;
    out.uv = uv;
    out.tex_mix = tex_mix;
    out.page = page;
    out.alpha = alpha;
    out.cutoff = cutoff;
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
    let ndl = max(dot(n, camera.light_dir), 0.0);
    let lit = base * (camera.ambient + camera.light_color * (camera.diffuse * ndl));
    let rgb = mix(base, lit, in.lit_flag);
    return vec4<f32>(rgb, alpha);
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
    _pad0: f32,
    light_color: [f32; 3],
    diffuse: f32,
    ambient: [f32; 3],
    _pad1: f32,
}

const _: () = assert!(size_of::<CameraUniform>() == 112);

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

/// One vertex: position + albedo + normal + lit flag + uv/page in a
/// single layout, so flat, lit, and textured groups share one buffer and
/// one pipeline. Flat vertices carry a dummy up-normal and `lit = 0.0`
/// (their baked color passes through untouched); untextured vertices
/// carry a dummy uv and `tex_mix = 0.0` (no sampling, tint only).
/// `alpha` is the group multiplier and `cutoff` the discard threshold
/// (both per-vertex so one buffer serves all three alpha behaviors).
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
    /// Square texture layer edge in pixels.
    pub layer_size: u32,
    /// Array layer count (= max texture pages).
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
/// dropped with a warning at upload time, never a panic.
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
/// [`ray_triangle`](super::pick::ray_triangle) degeneracy test (area² ≤
/// 1e-12 misses there, so it must not draw here either — content and picks
/// stay glued). Returns an empty vec when fewer than 3 indices remain.
/// Non-multiple-of-3 tails are ignored, matching `pick_ray`'s chunking.
///
/// `is_finite` rides alongside the area test (not folded into one
/// comparison): NaN poisons every ordering, so the check must name it —
/// a bare `area2 <= 1e-12` reads like it culls NaN but actually lets it
/// through.
fn cull_degenerate(positions: &[[f32; 3]], indices: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(indices.len());
    let (chunks, _) = indices.as_chunks::<3>();
    for tri in chunks {
        let (ia, ib, ic) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let (Some(a), Some(b), Some(c)) = (positions.get(ia), positions.get(ib), positions.get(ic))
        else {
            continue; // validated in-range by the caller; belt and braces
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

/// One validated group awaiting [`SceneBatch::finish`].
struct Pending {
    positions: Vec<[f32; 3]>,
    colors: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    texture_page: u32,
    /// World-space AABB center (computed at push time) for back-to-front
    /// transparent sorting. Opaque groups ignore it; `None` (degenerate
    /// bounds — never happens post-validation, belt and braces) sorts
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
/// on format/sample/desc change drops texture contents (logged) — the
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
    /// unit tests' default — they assert content preservation directly).
    view_proj: Mat4,
    /// Groups culled by the last [`SceneBatch::finish`] (frustum only —
    /// malformed/degenerate drops are separate, counted nowhere by
    /// design). HUD/stats readout, reset per `finish`.
    culled: usize,
    pending: Vec<Pending>,
    uploads: Vec<SceneUpload>,
    verts: Vec<Vert>,
    indices: Vec<u32>,
    ranges: Vec<DrawRange>,
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
            },
            camera_pos: [0.0, 0.0, 0.0],
            view_proj: Mat4::IDENTITY,
            culled: 0,
            pending: Vec::new(),
            uploads: Vec::new(),
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
        self.view_proj = view_proj;
    }

    /// Camera world position for transparent back-to-front sorting (see
    /// [`SceneBatch::finish`]). The viewport feeds `cam.eye()` from the
    /// same snapshot as the view-projection matrix, so sort order and
    /// pixels share one camera.
    pub fn set_camera_pos(&mut self, pos: [f32; 3]) {
        self.camera_pos = pos;
    }

    /// Groups culled by the last [`SceneBatch::finish`].
    pub fn culled(&self) -> usize {
        self.culled
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
        self.camera.ambient = light.ambient;
        self.camera.light_color = light.color;
        self.camera.diffuse = light.diffuse.max(0.0);
    }

    pub fn clear(&mut self) {
        self.pending.clear();
        self.uploads.clear();
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

    /// Queue texture uploads, applied in the next `prepare`.
    pub fn upload(&mut self, upload: SceneUpload) {
        // Guard the same contract `push_group` enforces for pages: an
        // out-of-range upload is dropped at queue time (with a warning) so
        // a caller packing against a smaller desc can't poison the texture.
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
    /// batch layers, non-finite alpha) are dropped with a warning — never
    /// a panic, never partial draws. Degenerate triangles (zero area, NaN)
    /// are culled tri-by-tri so one bad triangle can't sink its group: the
    /// group draws with the surviving triangles. Normals and uvs stay
    /// all-or-nothing per group. Shares
    /// [`validate_group`](super::chunk::validate_group) with the chunk
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
            center,
            indices: cull_degenerate(&group.positions, &group.indices),
            transparent: group.transparent,
            alpha: group.alpha.clamp(0.0, 1.0),
            alpha_cutoff: group.alpha_cutoff.clamp(0.0, 1.0),
            depth_test: group.depth_test,
        });
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
    /// tests `w + z >= 0`, always true post-remap) — the tests pin the far
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
    /// (depth-tested, then flat overlays — submission order decides ties),
    /// then transparent back-to-front (camera distance of the AABB center;
    /// groups without bounds sort nearest). Frustum culling drops fully
    /// outside groups before flattening (`depth_test = false` overlays are
    /// never culled — gizmos must draw even off-screen). Split from
    /// [`push_group`] so the viewport payload, which rebuilds the batch
    /// per frame, shares the path.
    ///
    /// Flat vertices (no normals) carry a dummy up-normal and `lit = 0.0`
    /// so the shader passes their baked color through untouched.
    /// Untextured vertices carry a dummy uv and `tex_mix = 0.0` so the
    /// shader skips the sample.
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
            // `depth_test = false` overlays are never culled — gizmos must
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
                // on top — visible beats culled when the math gives up.
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
    }

    /// True when every vertex of `positions` sits outside one frustum
    /// plane. Vertex-exact (no AABB approximation): costs one walk per
    /// group per frame, only when culling is armed (real camera). Groups
    /// are small in practice (chunked terrain splits by material, agents
    /// are single meshes) — and a wrongly-culled group is a missing
    /// object, so exactness beats cleverness here.
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

    pub(crate) fn ensure_resources(
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
            self.desc.filter as u32,
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
                "scene_batch[{}]: rebuilding pipeline/texture (format/sample/desc changed); texture contents dropped, re-upload required",
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
        let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("repame_view3d_pl"),
            bind_group_layouts: &[Some(&cam_layout), Some(&tex_layout)],
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
        let entry = SceneEntry {
            key,
            pipeline_depth: mk("repame_view3d_depth", true, false),
            pipeline_flat: mk("repame_view3d_flat", false, false),
            pipeline_transparent: mk("repame_view3d_transparent", true, true),
            pipeline_transparent_flat: mk("repame_view3d_transparent_flat", false, true),
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
    /// [`ensure_resources`](Self::ensure_resources) first —
    /// [`prepare_scene_with_id`] + the viewport do). Out-of-range or
    /// mis-sized uploads are dropped with a warning, never a panic
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
    key: (wgpu::TextureFormat, u32, u32, u32, u32),
    pipeline_depth: wgpu::RenderPipeline,
    pipeline_flat: wgpu::RenderPipeline,
    pipeline_transparent: wgpu::RenderPipeline,
    pipeline_transparent_flat: wgpu::RenderPipeline,
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
    // are shared resources, so clone the (cheap) handles and end the borrow
    // before beginning the mutable scene pass.
    struct Snapshot {
        cam_bind: wgpu::BindGroup,
        tex_bind: wgpu::BindGroup,
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
        Some(Snapshot {
            cam_bind: res.cam_bind.clone(),
            tex_bind: res.tex_bind.clone(),
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
    let composite = DepthComposite::get(resources);
    let Some(mut pass) = composite.begin_scene(id, encoder, clear) else {
        return;
    };
    pass.set_bind_group(0, &snap.cam_bind, &[]);
    pass.set_bind_group(1, &snap.tex_bind, &[]);
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
        // fixtures — the center ray lands on it).
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

        // Cutoff above the ghost's alpha discards it: pure red shows.
        let Some(px) = render_case(sheets(0.5, 0.6)) else {
            return;
        };
        assert_eq!(px, [255, 0, 0, 255], "cutoff discards the ghost: {px:?}");
    }
}
