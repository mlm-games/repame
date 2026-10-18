//! GPU light accumulation: [`Light2dPass`] builds a light map, batches consume
//! it.
//!
//! # Technique
//!
//! Derived from Godot 4's canvas light path
//! (`RendererCanvasRenderRD::light_update_shadow`, `renderer_canvas_render_rd.cpp`,
//! and `light_blend_compute` in `shaders/canvas.glsl`), rearranged for a
//! batched 2D renderer:
//!
//! * Godot renders occluders from each light into a shadow atlas and the item
//!   shader ray-marches that map while it loops the lights. A sprite batch
//!   cannot afford a per-item light loop (that is one light's worth of work
//!   *per sprite*), so this crate accumulates instead: every frame the pass
//!   renders the lights into an offscreen light map and the sprite shader
//!   samples it once per fragment. Cost is per light, never per sprite.
//! * Shadows are built on the CPU as shadow-volume wedges (one quad per
//!   occluder edge, extruded away from the light) and rasterised into a small
//!   per-light mask, which that light's own quad then samples. The mask holds
//!   one channel per filter tap, max-blended so the edges overlapped by one
//!   tap are a union, and the light shader sums the channels, which is the
//!   taps' average. Godot instead draws the occluders into a polar distance
//!   map and marches it; wedges were chosen here because they are exact at any
//!   occlusion *depth* (a wall behind a wall needs no extra storage), cost one
//!   extra pass per shadowed light, and are pure data the CPU tests can check
//!   without a GPU. Geometry is clipped by the rasteriser, so a wedge that
//!   runs off the light's footprint is free.
//! * `LightBlend` is Godot's `light_blend_compute`, expressed as
//!   fixed-function blend states on an `(rgb = light, a = item multiplier)`
//!   accumulator, and the ambient light is the clear value of each layer,
//!   before any light is added.
//! * `light_mask` is Godot's `item_cull_mask`: a light is accumulated once per
//!   layer bit it sets and an item reads the layer of its own lowest bit, so
//!   "the light and the item share a bit" is exactly Godot's rule.
//!
//! # Frame structure
//!
//! `prepare` records everything into the renderer's prepare encoder, which the
//! renderer submits before the main pass, so the light map is ready by the
//! time any batch paints, whatever order the nodes were in. Per frame: one
//! clear pass per layer, then per light one mask pass (shadowed lights only)
//! and one load-and-draw pass per layer bit. The light map is world-space (its
//! uv comes from world coordinates), so a consumer needs no rect arithmetic
//! and a sub-rectangle batch lines up as long as it shares this pass's camera.
//!
//! A frame with no lights and a black ambient renders nothing and publishes
//! nothing, and consumers fall back to the identity map (unlit).
//!
//! # What is verified here
//!
//! Everything CPU-side is: the plan, the wedge geometry, the vertex and
//! uniform packing, and the scissor maths have tests. The WGSL is *not*
//! executed anywhere in this crate's checks (there is no GPU in CI): it is
//! kept to straight-line maths derived from the reference above, and the
//! pipeline state it depends on is asserted by construction (formats, blend
//! factors, one render pass per attachment) rather than by rendering.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2};
use repose_render_wgpu::{CallbackResources, ScreenDescriptor, WgpuCallback};
use wgpu::util::DeviceExt;

use super::TextureFilter;
use super::light::{
    AmbientLight, Light2d, LightBlend, LightBudget, LightingPlan, MAX_MASK_TAPS, Occluder2d,
    plan_lighting, visible_world_rect,
};

/// Falloff texture upload: tight `w`*`h`*4 RGBA8 rows, top first, mirroring
/// [`FullscreenTexture`](crate::FullscreenTexture). The texture's `rgb` scales
/// the light colour and its `a` scales the ramp; a light selects one with
/// [`LightFalloff::Texture::slot`]. Malformed uploads are dropped with a
/// warning.
#[derive(Clone, Debug)]
pub struct LightTextureUpload {
    /// Texture slot a light names.
    pub slot: u32,
    pub w: u32,
    pub h: u32,
    pub rgba: Vec<u8>,
}

/// Construction parameters for one [`Light2dPass`].
#[derive(Clone, Copy, Debug)]
pub struct Light2dDesc {
    /// Light map resolution as a fraction of the render target, clamped to
    /// `0.25..=1.0`. Half resolution is usually indistinguishable and a
    /// quarter of the memory.
    pub lightmap_scale: f32,
    /// Shadow mask edge in pixels, clamped to `64..=2048`. The mask is
    /// stretched over each light's footprint, so a smaller mask softens shadow
    /// edges and never moves a shadow.
    pub mask_size: u32,
    /// Sampling for falloff textures; the shadow mask is always linear.
    pub filter: TextureFilter,
    /// Generation of the queued textures. Uploads are applied when it changes,
    /// so a game that queues its level's textures once pays for them once.
    pub textures_gen: u64,
    /// Per-frame limits, enforced by
    /// [`plan_lighting`](crate::light::plan_lighting) with typed issues.
    pub budget: LightBudget,
}

impl Default for Light2dDesc {
    fn default() -> Self {
        Self {
            lightmap_scale: 0.5,
            mask_size: 512,
            filter: TextureFilter::Linear,
            textures_gen: 0,
            budget: LightBudget::default(),
        }
    }
}

/// The light map one frame produced, as its consumers see it.
///
/// Published in [`CallbackResources`] under the pass id and read back by
/// [`SpriteBatch::set_light_source`](crate::SpriteBatch::set_light_source).
/// `view` is a `texture_2d_array<f32>` of `layer_clamp + 1` layers; the last
/// layer is always cleared to `(0, 0, 0, 1)`, the identity for the consumer's
/// maths, so an item that is lit by nothing can simply be pointed at it.
pub struct LightmapInfo {
    /// Accumulator array texture view (`Rgba8Unorm`, one layer per mask bit).
    pub view: wgpu::TextureView,
    /// World rect the map covers: `[x, y, w, h]`, uv origin at its top-left.
    pub world_rect: [f32; 4],
    /// Highest layer an item may sample (the identity layer).
    pub layer_clamp: u32,
    /// Bumped whenever the underlying texture is recreated. Consumers compare
    /// it to know when a cached bind group is stale; the rect changes every
    /// frame and travels in the consumer's own uniform instead.
    pub generation: u64,
}

/// Read the light map a pass published this frame, if any.
///
/// Consumers call this from the renderer's `finish_prepare` hook, which runs
/// after every callback's `prepare`, so scene order between a pass and its
/// consumers does not matter; the first frame a pass exists consumers fall
/// back to the identity map.
pub fn lookup_lightmap(resources: &CallbackResources, id: &str) -> Option<Arc<LightmapInfo>> {
    resources.get::<LightResources>()?.maps.get(id).cloned()
}

/// Marker whose `TypeId`, together with [`SHARED_CALLBACK_KEY`], puts
/// [`Light2dPass`] and the [`SpriteBatch`](crate::SpriteBatch)es that consume
/// it in one renderer callback resource scope.
///
/// The renderer keys callback resources by
/// `(resource_type_id, resource_key, target)`, and the defaults make each
/// callback type its own scope, so a producer/consumer pair has to report one
/// identity to see each other's resources. Both ends also have to render into
/// the same target, which is part of the documented contract anyway.
pub(crate) struct LightScope;

/// The shared resource key of the sprite batch and the light pass; see
/// [`LightScope`].
pub(crate) const SHARED_CALLBACK_KEY: &str = "repame.sprite_batch.resources";

/// A light pass's CPU snapshot: camera, lights, occluders, ambient light, and
/// the level's falloff textures. Rebuilt every frame exactly like
/// [`SpriteBatch`](crate::SpriteBatch); `Send + Sync` for the compositor
/// thread.
///
/// Like [`FullscreenPass`](crate::FullscreenPass), an id keeps its pipelines,
/// shadow mask, light map and falloff textures alive for the life of the
/// renderer, so a game should reuse a handful of ids (one per viewport) and
/// vary the snapshot, not mint a new id per frame.
///
/// ```ignore
/// // One pass per viewport, then point every batch that shares its camera at
/// // it. Both nodes fill the target; scene order between them does not matter.
/// let mut lights = Light2dPass::with_id("main.lights", Light2dDesc::default());
/// lights.set_camera(camera.fit_matrix(canvas_dp, world_size));
/// lights.set_ambient(AmbientLight { color: DUSK, energy: 1.0, light_mask: 0b1 });
/// lights.push_light(Light2d {
///     position: player.pos,
///     radius: 148.0,
///     falloff: LightFalloff::stops(vec![
///         LightStop { at: 0.0, color: [1.0, 0.94, 0.7, 1.0] },
///         LightStop { at: 0.4, color: [0.95, 0.78, 0.45, 0.5] },
///         LightStop { at: 1.0, color: [0.0, 0.0, 0.0, 0.0] },
///     ]),
///     shadows: true,
///     shadow_filter: ShadowFilter::Pcf4,
///     shadow_softness: 8.0,
///     ..Default::default()
/// });
/// for wall in level.walls {
///     lights.push_occluder(Occluder2d::rect(wall.min, wall.max));
/// }
///
/// let mut batch = SpriteBatch::new(BatchDesc::default());
/// batch.set_camera(camera.fit_matrix(canvas_dp, world_size));
/// batch.set_light_source(lights.id());
/// for sprite in frame.sprites {
///     batch.push_sprite(&sprite);   // `sprite.light_mask` picks its layer
/// }
///
/// let full = Rect { x: 0.0, y: 0.0, w: canvas_dp[0], h: canvas_dp[1] };
/// vec![
///     lights.node(full),
///     BatchFrame::new(batch, uploads.clone()).node(full),
/// ]
/// ```
pub struct Light2dPass {
    id: String,
    desc: Light2dDesc,
    camera: Mat4,
    lights: Vec<Light2d>,
    occluders: Vec<Occluder2d>,
    ambient: AmbientLight,
    textures: Vec<LightTextureUpload>,
}

impl Light2dPass {
    /// Default id (`light2d.default`); use [`Self::with_id`] when a scene has
    /// several viewports.
    pub fn new(desc: Light2dDesc) -> Self {
        Self::with_id("light2d.default", desc)
    }

    pub fn with_id(id: impl Into<String>, desc: Light2dDesc) -> Self {
        Self {
            id: id.into(),
            desc,
            camera: Mat4::IDENTITY,
            lights: Vec::new(),
            occluders: Vec::new(),
            ambient: AmbientLight::default(),
            textures: Vec::new(),
        }
    }

    /// The pass id a batch names in
    /// [`set_light_source`](crate::SpriteBatch::set_light_source).
    pub fn id(&self) -> &str {
        &self.id
    }

    /// View-projection of the world this light map covers. It must be the same
    /// camera the consuming batches use, or the light map lands offset from
    /// what it lights.
    pub fn set_camera(&mut self, view_proj: Mat4) {
        self.camera = view_proj;
    }

    /// Drop last frame's lights and occluders. Textures stay: they are level
    /// data, applied when `textures_gen` changes.
    pub fn clear(&mut self) {
        self.lights.clear();
        self.occluders.clear();
    }

    pub fn push_light(&mut self, light: Light2d) {
        self.lights.push(light);
    }

    pub fn extend_lights(&mut self, lights: impl IntoIterator<Item = Light2d>) {
        self.lights.extend(lights);
    }

    pub fn push_occluder(&mut self, occluder: Occluder2d) {
        self.occluders.push(occluder);
    }

    pub fn extend_occluders(&mut self, occluders: impl IntoIterator<Item = Occluder2d>) {
        self.occluders.extend(occluders);
    }

    /// The global light every lit layer starts from.
    pub fn set_ambient(&mut self, ambient: AmbientLight) {
        self.ambient = ambient;
    }

    /// Queue one falloff texture, applied when `textures_gen` changes.
    pub fn queue_texture(&mut self, upload: LightTextureUpload) {
        self.textures.push(upload);
    }

    pub fn extend_textures(&mut self, uploads: impl IntoIterator<Item = LightTextureUpload>) {
        self.textures.extend(uploads);
    }

    /// The frame's plan, without a GPU: what will be lit, occluded, dropped,
    /// and how many layers the accumulator needs. Pure, so a game (or a test)
    /// can assert on the frame before it draws.
    pub fn plan(&self) -> LightingPlan {
        plan_lighting(
            &self.lights,
            &self.occluders,
            &self.ambient,
            visible_world_rect(self.camera).unwrap_or([0.0, 0.0, 1.0, 1.0]),
            &self.desc.budget,
        )
    }

    /// Scene node for this pass. Its rect is presentation only: the light map
    /// is sized from the render target, so `fill_max_size` (or any non-empty
    /// rect) is fine.
    pub fn node(self, rect: repose_core::Rect) -> repose_core::SceneNode {
        repose_core::SceneNode::Callback {
            rect,
            payload: repose_render_wgpu::Callback::new(self),
        }
    }

    /// Report each distinct problem once per pass id, so a per-frame mistake is
    /// loud without flooding the log.
    fn warn_once(resources: &mut CallbackResources, id: &str, message: String) {
        let all = resources.get_or_insert_with::<LightResources>();
        if all.warned.insert((id.to_string(), message.clone())) {
            log::warn!("light2d[{id}]: {message}");
        }
    }
}

/// World-space vertex shared by the mask and light pipelines: a clip-space
/// position the CPU baked for the pass at hand, the world position of the same
/// corner, and the corner's shadow mask value (`0` on light quads).
///
/// One mask channel per [`ShadowFilter`] tap: a wedge puts its
/// [`ShadowWedge::weight`](crate::light::ShadowWedge::weight) in its own tap's
/// channel and zero in the others, so the mask pass max-blends overlapping
/// wedges of a tap into a union while the light shader sums the taps. See
/// [`shadow_wedges`](crate::light::shadow_wedges).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct LightVertex {
    clip: [f32; 2],
    world: [f32; 2],
    weights: [f32; MAX_MASK_TAPS],
}

/// Matches `shaders/light_disc.wgsl::Light` field for field.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct LightUniform {
    center_half: [f32; 4],
    color_energy: [f32; 4],
    params: [f32; 4],
    mask_xform: [f32; 4],
    tex: [f32; 4],
    stop_at: [[f32; 4]; 2],
    stop_rgba: [[f32; 4]; 8],
}

const _: () = assert!(size_of::<LightUniform>() == 240);
const _: () = assert!(size_of::<LightVertex>() == 32);

struct RampTexture {
    /// Written by uploads; the view lives in the slot's bind group.
    texture: wgpu::Texture,
    size: [u32; 2],
}

struct LightTarget {
    size: [u32; 2],
    layers: u32,
    /// Kept alive while layer views and the published info exist.
    #[allow(dead_code)]
    texture: wgpu::Texture,
    views: Vec<wgpu::TextureView>,
    info: Arc<LightmapInfo>,
}

struct PassEntry {
    pipeline_mask: wgpu::RenderPipeline,
    /// Add, sub, mix (`blend_index`).
    pipelines: [wgpu::RenderPipeline; 3],
    ramp_layout: wgpu::BindGroupLayout,
    vertices: wgpu::Buffer,
    vertex_cap: usize,
    uniforms: wgpu::Buffer,
    uniform_bind: wgpu::BindGroup,
    uniform_stride: u32,
    uniform_slots: usize,
    #[allow(dead_code)]
    mask_texture: wgpu::Texture,
    mask_view: wgpu::TextureView,
    mask_bind: wgpu::BindGroup,
    mask_bind_empty: wgpu::BindGroup,
    ramp_bind_empty: wgpu::BindGroup,
    ramp_binds: HashMap<u32, (RampTexture, wgpu::BindGroup)>,
    applied_textures_gen: Option<u64>,
    sampler: wgpu::Sampler,
    target: Option<LightTarget>,
    generation: u64,
}

struct LightResources {
    passes: HashMap<String, PassEntry>,
    maps: HashMap<String, Arc<LightmapInfo>>,
    warned: HashSet<(String, String)>,
}

impl Default for LightResources {
    fn default() -> Self {
        Self {
            passes: HashMap::new(),
            maps: HashMap::new(),
            warned: HashSet::new(),
        }
    }
}

const ADD: wgpu::BlendComponent = wgpu::BlendComponent {
    src_factor: wgpu::BlendFactor::SrcAlpha,
    dst_factor: wgpu::BlendFactor::One,
    operation: wgpu::BlendOperation::Add,
};

const KEEP_MULTIPLIER: wgpu::BlendComponent = wgpu::BlendComponent {
    src_factor: wgpu::BlendFactor::Zero,
    dst_factor: wgpu::BlendFactor::One,
    operation: wgpu::BlendOperation::Add,
};

/// One [`LightBlend`] as fixed-function state on `(rgb = light, a = item
/// multiplier)`. See the module docs for the derivation.
fn blend_state(blend: LightBlend) -> wgpu::BlendState {
    match blend {
        LightBlend::Add => wgpu::BlendState {
            color: ADD,
            alpha: KEEP_MULTIPLIER,
        },
        // Godot: `color -= light`. The accumulator is 8-bit, so this clamps at
        // black exactly like a per-item subtract on an 8-bit target would.
        LightBlend::Sub => wgpu::BlendState {
            color: wgpu::BlendComponent {
                operation: wgpu::BlendOperation::ReverseSubtract,
                ..ADD
            },
            alpha: KEEP_MULTIPLIER,
        },
        LightBlend::Mix => wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        },
    }
}

fn blend_index(blend: LightBlend) -> usize {
    match blend {
        LightBlend::Add => 0,
        LightBlend::Sub => 1,
        LightBlend::Mix => 2,
    }
}

/// Shadow mask target format: one linear channel per tap (`MAX_MASK_TAPS`).
/// Linear, not sRGB: the tap weights are coverage fractions, not colours.
const MASK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Shadow mask blend: per channel, `max(src, dst)`.
///
/// One channel per tap. Overlapping wedges of the same tap touch the same
/// channel, and the shadow there is their union, not their sum: a solid
/// occluder's own edges overlap, and adding them would darken the lit side of
/// a silhouette. Max blending makes each channel the union of its tap, and the
/// light shader resumes the summing across taps, which is the taps' average.
const MASK_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Max,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Max,
    },
};

fn sampler_desc(filter: TextureFilter, label: &'static str) -> wgpu::SamplerDescriptor<'static> {
    let f = match filter {
        TextureFilter::Nearest => wgpu::FilterMode::Nearest,
        TextureFilter::Linear => wgpu::FilterMode::Linear,
    };
    wgpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: f,
        min_filter: f,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        lod_min_clamp: 0.0,
        lod_max_clamp: 1.0,
        compare: None,
        anisotropy_clamp: 1,
        border_color: None,
    }
}

/// World -> light map clip space (`x` right, `y` up, `[-1, 1]` per axis).
fn lightmap_clip(world: Vec2, rect: [f32; 4]) -> [f32; 2] {
    [
        (world.x - rect[0]) * (2.0 / rect[2]) - 1.0,
        1.0 - (world.y - rect[1]) * (2.0 / rect[3]),
    ]
}

/// World -> shadow mask clip space for a light whose footprint covers
/// `center +- half_extent`.
fn mask_clip(world: Vec2, center: Vec2, half_extent: Vec2) -> [f32; 2] {
    [
        (world.x - (center.x - half_extent.x)) / half_extent.x - 1.0,
        1.0 - (world.y - (center.y - half_extent.y)) / half_extent.y,
    ]
}

/// World bounds -> pixel scissor `[x, y, w, h]` in the light map, clamped to
/// it; `None` when the box misses the map entirely.
fn scissor_px(aabb: [f32; 4], rect: [f32; 4], size: [u32; 2]) -> Option<[u32; 4]> {
    let sx = size[0] as f32 / rect[2];
    let sy = size[1] as f32 / rect[3];
    let x0 = ((aabb[0] - rect[0]) * sx).floor().max(0.0);
    let y0 = ((aabb[1] - rect[1]) * sy).floor().max(0.0);
    let x1 = ((aabb[2] - rect[0]) * sx).ceil().min(size[0] as f32);
    let y1 = ((aabb[3] - rect[1]) * sy).ceil().min(size[1] as f32);
    if !(x1 > x0 && y1 > y0) {
        return None;
    }
    Some([x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32])
}

fn align_up(value: u32, align: u32) -> u32 {
    let align = align.max(1);
    value.div_ceil(align) * align
}

/// Per-light vertex ranges: the light quad and, when shadowed, its wedges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LightRanges {
    quad: (u32, u32),
    mask: Option<(u32, u32)>,
}

impl Light2dPass {
    fn uniform_stride(device: &wgpu::Device) -> u32 {
        align_up(
            size_of::<LightUniform>() as u32,
            device.limits().min_uniform_buffer_offset_alignment,
        )
    }

    fn ensure_resources(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resources: &mut CallbackResources,
    ) {
        let stride = Self::uniform_stride(device);
        let slots = self.desc.budget.max_lights.max(1);
        let stale = resources
            .get::<LightResources>()
            .and_then(|all| all.passes.get(self.id.as_str()))
            .is_some_and(|entry| entry.uniform_stride != stride || entry.uniform_slots != slots);
        if stale {
            log::warn!(
                "light2d[{}]: rebuilding pipelines for a budget/alignment change",
                self.id
            );
            if let Some(all) = resources.get_mut::<LightResources>() {
                all.passes.remove(self.id.as_str());
            }
        }
        let present = resources
            .get::<LightResources>()
            .is_some_and(|all| all.passes.contains_key(self.id.as_str()));
        if present {
            return;
        }
        let all = resources.get_or_insert_with::<LightResources>();
        let generation = all
            .passes
            .values()
            .map(|entry| entry.generation)
            .max()
            .unwrap_or(0)
            + 1;
        let entry = create_entry(self, device, queue, stride, slots, generation);
        all.passes.insert(self.id.clone(), entry);
    }

    /// Create or refresh the falloff textures when `textures_gen` advanced.
    fn apply_textures(&self, device: &wgpu::Device, queue: &wgpu::Queue, entry: &mut PassEntry) {
        if entry.applied_textures_gen == Some(self.desc.textures_gen) {
            return;
        }
        entry.applied_textures_gen = Some(self.desc.textures_gen);
        for upload in &self.textures {
            let valid = upload.w > 0
                && upload.h > 0
                && upload.rgba.len() == upload.w as usize * upload.h as usize * 4;
            if !valid {
                log::warn!(
                    "light2d: dropping malformed falloff texture slot={} {}x{} ({} bytes)",
                    upload.slot,
                    upload.w,
                    upload.h,
                    upload.rgba.len()
                );
                continue;
            }
            let size = [upload.w, upload.h];
            let reuse = entry
                .ramp_binds
                .get(&upload.slot)
                .is_some_and(|(texture, _)| texture.size == size);
            if !reuse {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("light2d_ramp"),
                    size: wgpu::Extent3d {
                        width: upload.w,
                        height: upload.h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                let bind = create_texture_bind(
                    device,
                    &entry.ramp_layout,
                    &view,
                    &entry.sampler,
                    "light2d_ramp_bg",
                );
                entry
                    .ramp_binds
                    .insert(upload.slot, (RampTexture { texture, size }, bind));
            }
            let (ramp, _) = entry
                .ramp_binds
                .get_mut(&upload.slot)
                .expect("inserted above");
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &ramp.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &upload.rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(upload.w * 4),
                    rows_per_image: Some(upload.h),
                },
                wgpu::Extent3d {
                    width: upload.w,
                    height: upload.h,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    fn prepare_frame(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        screen: &ScreenDescriptor,
        resources: &mut CallbackResources,
    ) {
        let Some(world_rect) = visible_world_rect(self.camera) else {
            Self::warn_once(
                resources,
                &self.id,
                "camera does not map a visible world rect; skipping lights this frame".to_string(),
            );
            return;
        };
        let plan = plan_lighting(
            &self.lights,
            &self.occluders,
            &self.ambient,
            world_rect,
            &self.desc.budget,
        );
        for issue in &plan.issues {
            Self::warn_once(
                resources,
                &self.id,
                format!("light {} dropped: {:?}", issue.source, issue.kind),
            );
        }
        for reject in &plan.occluder_rejects {
            Self::warn_once(
                resources,
                &self.id,
                format!(
                    "occluder {} rejected: {:?}",
                    reject.index, reject.reason
                ),
            );
        }

        let scale = if self.desc.lightmap_scale.is_finite() {
            self.desc.lightmap_scale.clamp(0.25, 1.0)
        } else {
            0.5
        };
        let max_dim = device.limits().max_texture_dimension_2d;
        let size = [
            ((screen.size_in_pixels[0] as f32 * scale).round() as u32).clamp(1, max_dim),
            ((screen.size_in_pixels[1] as f32 * scale).round() as u32).clamp(1, max_dim),
        ];

        self.ensure_resources(device, queue, resources);
        let stride = Self::uniform_stride(device) as usize;
        let (vertices, ranges) = build_vertices(&plan, world_rect);
        let (uniforms, light_count) = build_uniforms(&plan, stride);

        let LightResources { passes, maps, .. } = resources
            .get_mut::<LightResources>()
            .expect("ensure_resources inserted them");
        let Some(entry) = passes.get_mut(self.id.as_str()) else {
            return;
        };
        if light_count > entry.uniform_slots {
            log::error!(
                "light2d[{}]: {} lights exceed the {} uniform slots; drawing {}",
                self.id,
                light_count,
                entry.uniform_slots,
                entry.uniform_slots
            );
        }
        self.apply_textures(device, queue, entry);
        Self::ensure_target(entry, device, size, plan.layers, world_rect);
        Self::ensure_vertices(entry, device, vertices.len());
        if !vertices.is_empty() {
            queue.write_buffer(&entry.vertices, 0, bytemuck::cast_slice(&vertices));
        }
        if !uniforms.is_empty() {
            queue.write_buffer(&entry.uniforms, 0, &uniforms);
        }

        let info = entry
            .target
            .as_ref()
            .expect("target exists after ensure_target")
            .info
            .clone();
        maps.insert(self.id.clone(), info);

        let target = entry.target.as_ref().expect("target exists");
        record_frame(
            encoder,
            entry,
            target,
            &plan,
            &ranges,
            &world_rect,
            light_count,
        );
    }

    fn ensure_vertices(entry: &mut PassEntry, device: &wgpu::Device, len: usize) {
        if len <= entry.vertex_cap {
            return;
        }
        let cap = len.next_power_of_two().max(64);
        entry.vertices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("light2d_vertices"),
            size: (cap * size_of::<LightVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        entry.vertex_cap = cap;
    }

    fn ensure_target(
        entry: &mut PassEntry,
        device: &wgpu::Device,
        size: [u32; 2],
        layers: u32,
        world_rect: [f32; 4],
    ) {
        let stale = entry
            .target
            .as_ref()
            .is_none_or(|target| target.size != size || target.layers != layers);
        if stale {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("light2d_map"),
                size: wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: layers,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let views: Vec<wgpu::TextureView> = (0..layers)
                .map(|layer| {
                    texture.create_view(&wgpu::TextureViewDescriptor {
                        label: Some("light2d_map_layer"),
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        base_array_layer: layer,
                        array_layer_count: Some(1),
                        ..Default::default()
                    })
                })
                .collect();
            let array_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("light2d_map"),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });
            entry.generation += 1;
            entry.target = Some(LightTarget {
                size,
                layers,
                texture,
                views,
                info: Arc::new(LightmapInfo {
                    view: array_view,
                    world_rect,
                    layer_clamp: layers - 1,
                    generation: entry.generation,
                }),
            });
        }
        let target = entry.target.as_mut().expect("created above");
        // A fresh `Arc` when the rect moves (the camera) or the layer count
        // changes; consumers key their bind groups on `(id, generation)`, not
        // on this handle, so a new one costs one allocation and no churn.
        if target.info.world_rect != world_rect || target.info.layer_clamp != layers - 1 {
            target.info = Arc::new(LightmapInfo {
                view: target.info.view.clone(),
                world_rect,
                layer_clamp: layers - 1,
                generation: target.info.generation,
            });
        }
    }
}

fn create_entry(
    pass: &Light2dPass,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    uniform_stride: u32,
    uniform_slots: usize,
    generation: u64,
) -> PassEntry {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("light2d_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/light_disc.wgsl").into()),
    });
    let mask_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("light2d_mask_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/light_mask.wgsl").into()),
    });
    let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("light2d_uniform_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let stereo_layout = |label: &'static str| {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
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
        })
    };
    let mask_layout = stereo_layout("light2d_mask_bgl");
    let ramp_layout = stereo_layout("light2d_ramp_bgl");
    let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("light2d_uniforms"),
        size: (uniform_slots as u32 * uniform_stride) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let uniform_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("light2d_uniform_bg"),
        layout: &uniform_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &uniforms,
                offset: 0,
                size: std::num::NonZeroU64::new(size_of::<LightUniform>() as u64),
            }),
        }],
    });
    let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("light2d_vertices"),
        contents: bytemuck::cast_slice(&[LightVertex::zeroed(); 6]),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });
    let sampler = device.create_sampler(&sampler_desc(pass.desc.filter, "light2d_sampler"));
    let mask_size = pass.desc.mask_size.clamp(64, 2048);
    let mask_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("light2d_mask"),
        size: wgpu::Extent3d {
            width: mask_size,
            height: mask_size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: MASK_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let mask_view = mask_texture.create_view(&wgpu::TextureViewDescriptor::default());
    // A one-texel mask of zero is "no shadow at all": every light without
    // shadows binds it, so one shader path covers both cases.
    let mask_empty_view = solid_view(device, queue, [0, 0, 0, 0], MASK_FORMAT);
    // A one-texel transparent black ramp: a texture falloff with no upload
    // paints nothing, like Godot's missing light texture.
    let ramp_empty_view = solid_view(
        device,
        queue,
        [0, 0, 0, 0],
        wgpu::TextureFormat::Rgba8UnormSrgb,
    );
    let mask_bind = create_texture_bind(
        device,
        &mask_layout,
        &mask_view,
        &sampler,
        "light2d_mask_bg",
    );
    let mask_bind_empty = create_texture_bind(
        device,
        &mask_layout,
        &mask_empty_view,
        &sampler,
        "light2d_mask_empty_bg",
    );
    let ramp_bind_empty = create_texture_bind(
        device,
        &ramp_layout,
        &ramp_empty_view,
        &sampler,
        "light2d_ramp_empty_bg",
    );
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("light2d_pl"),
        bind_group_layouts: &[
            Some(&uniform_layout),
            Some(&mask_layout),
            Some(&ramp_layout),
        ],
        immediate_size: 0,
    });
    let mask_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("light2d_mask_pl"),
        // The mask pass needs no bindings at all: the CPU baked its vertices
        // into mask clip space.
        bind_group_layouts: &[],
        immediate_size: 0,
    });
    let vertex_buffers = [Some(wgpu::VertexBufferLayout {
        array_stride: size_of::<LightVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 8,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 16,
                shader_location: 2,
            },
        ],
    })];
    let mk_pipeline = |label: &'static str,
                       module: &wgpu::ShaderModule,
                       layout: &wgpu::PipelineLayout,
                       format: wgpu::TextureFormat,
                       blend: wgpu::BlendState,
                       write_mask: wgpu::ColorWrites|
     -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("vs_main"),
                buffers: &vertex_buffers,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(blend),
                    write_mask,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        })
    };
    let pipeline_mask = mk_pipeline(
        "light2d_mask_pipeline",
        &mask_shader,
        &mask_pipeline_layout,
        MASK_FORMAT,
        MASK_BLEND,
        wgpu::ColorWrites::ALL,
    );
    let pipelines = [
        mk_pipeline(
            "light2d_add",
            &shader,
            &pipeline_layout,
            wgpu::TextureFormat::Rgba8Unorm,
            blend_state(LightBlend::Add),
            wgpu::ColorWrites::ALL,
        ),
        mk_pipeline(
            "light2d_sub",
            &shader,
            &pipeline_layout,
            wgpu::TextureFormat::Rgba8Unorm,
            blend_state(LightBlend::Sub),
            wgpu::ColorWrites::ALL,
        ),
        mk_pipeline(
            "light2d_mix",
            &shader,
            &pipeline_layout,
            wgpu::TextureFormat::Rgba8Unorm,
            blend_state(LightBlend::Mix),
            wgpu::ColorWrites::ALL,
        ),
    ];
    PassEntry {
        pipeline_mask,
        pipelines,
        ramp_layout,
        vertices,
        vertex_cap: 6,
        uniforms,
        uniform_bind,
        uniform_stride,
        uniform_slots,
        mask_texture,
        mask_view,
        mask_bind,
        mask_bind_empty,
        ramp_bind_empty,
        ramp_binds: HashMap::new(),
        applied_textures_gen: None,
        sampler,
        target: None,
        generation,
    }
}

fn create_texture_bind(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    label: &'static str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

/// A 1x1 texture of one solid colour, uploaded right away.
fn solid_view(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    rgba: [u8; 4],
    format: wgpu::TextureFormat,
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("light2d_solid"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Bake every light's quad and wedges, in accumulation order, and report the
/// ranges the draw calls use. Light quads go into light map clip space, wedges
/// into that light's mask clip space.
fn build_vertices(
    plan: &LightingPlan,
    world_rect: [f32; 4],
) -> (Vec<LightVertex>, Vec<LightRanges>) {
    let mut vertices = Vec::with_capacity(plan.lights.len() * 6 + plan.wedge_vertices());
    let mut ranges = Vec::with_capacity(plan.lights.len());
    for light in &plan.lights {
        let quad_start = vertices.len() as u32;
        let corners = [
            light.center - light.half_extent,
            Vec2::new(
                light.center.x + light.half_extent.x,
                light.center.y - light.half_extent.y,
            ),
            light.center + light.half_extent,
            Vec2::new(
                light.center.x - light.half_extent.x,
                light.center.y + light.half_extent.y,
            ),
        ];
        for corner in [
            corners[0], corners[1], corners[2], corners[0], corners[2], corners[3],
        ] {
            vertices.push(LightVertex {
                clip: lightmap_clip(corner, world_rect),
                world: [corner.x, corner.y],
                weights: [0.0; MAX_MASK_TAPS],
            });
        }
        let mask = if light.shadowed {
            let start = vertices.len() as u32;
            for wedge in &light.wedges {
                let quad = wedge.quad;
                for corner in [
                    quad[0], quad[1], quad[2], quad[0], quad[2], quad[3],
                ] {
                    vertices.push(LightVertex {
                        clip: mask_clip(corner, light.center, light.half_extent),
                        world: [corner.x, corner.y],
                        weights: wedge.mask_weights(),
                    });
                }
            }
            Some((start, vertices.len() as u32 - start))
        } else {
            None
        };
        ranges.push(LightRanges {
            quad: (quad_start, 6),
            mask,
        });
    }
    (vertices, ranges)
}

/// Pack one uniform per light, padded to the device's dynamic-offset stride.
fn build_uniforms(plan: &LightingPlan, stride: usize) -> (Vec<u8>, usize) {
    let stride = stride.max(size_of::<LightUniform>());
    let mut out = Vec::with_capacity(plan.lights.len() * stride);
    for light in &plan.lights {
        let mut stop_at = [[0.0f32; 4]; 2];
        let mut stop_rgba = [[0.0f32; 4]; 8];
        // Stops past the live count repeat the last one, so a miscount reads a
        // flat ramp rather than whatever zeroed memory holds.
        let last = if light.stop_count == 0 {
            0
        } else {
            light.stop_count as usize - 1
        };
        for i in 0..8 {
            let stop = light.stops[i.min(last)];
            stop_at[i / 4][i % 4] = stop.at;
            stop_rgba[i] = stop.color;
        }
        let uniform = LightUniform {
            center_half: [
                light.center.x,
                light.center.y,
                1.0 / light.half_extent.x,
                1.0 / light.half_extent.y,
            ],
            color_energy: [
                light.color[0],
                light.color[1],
                light.color[2],
                light.energy,
            ],
            params: [
                if light.stop_count == 0 { 1.0 } else { 0.0 },
                light.stop_count as f32,
                0.0,
                0.0,
            ],
            mask_xform: [
                light.center.x - light.half_extent.x,
                light.center.y - light.half_extent.y,
                1.0 / (2.0 * light.half_extent.x),
                1.0 / (2.0 * light.half_extent.y),
            ],
            tex: [
                light.texture_offset.x,
                light.texture_offset.y,
                1.0 / light.texture_half_extent.x,
                1.0 / light.texture_half_extent.y,
            ],
            stop_at,
            stop_rgba,
        };
        out.extend_from_slice(bytemuck::bytes_of(&uniform));
        out.resize((out.len() + stride - 1) / stride * stride, 0);
    }
    (out, plan.lights.len())
}

fn record_frame(
    encoder: &mut wgpu::CommandEncoder,
    entry: &PassEntry,
    target: &LightTarget,
    plan: &LightingPlan,
    ranges: &[LightRanges],
    world_rect: &[f32; 4],
    light_count: usize,
) {
    let clear_value = |layer: u32| {
        let rgb = plan.ambient_rgb(layer);
        wgpu::Color {
            r: rgb[0] as f64,
            g: rgb[1] as f64,
            b: rgb[2] as f64,
            a: 1.0,
        }
    };
    // Every layer starts at the ambient light for that layer, before any
    // light is blended in. The last layer is the identity by construction:
    // `ambient_rgb` never answers for it.
    for (layer, view) in target.views.iter().enumerate() {
        let _clear = begin_pass(
            encoder,
            "light2d_clear",
            view,
            clear_value(layer as u32),
            None,
        );
    }
    for (slot, light) in plan
        .lights
        .iter()
        .enumerate()
        .take(light_count.min(entry.uniform_slots))
    {
        let Some(range) = ranges.get(slot) else {
            continue;
        };
        if let Some((start, count)) = range.mask {
            let mut pass = begin_pass(
                encoder,
                "light2d_mask",
                &entry.mask_view,
                wgpu::Color::TRANSPARENT,
                None,
            );
            pass.set_pipeline(&entry.pipeline_mask);
            pass.set_vertex_buffer(0, entry.vertices.slice(..));
            pass.draw(start..start + count, 0..1);
        }
        let aabb = [
            light.center.x - light.half_extent.x,
            light.center.y - light.half_extent.y,
            light.center.x + light.half_extent.x,
            light.center.y + light.half_extent.y,
        ];
        let scissor = scissor_px(aabb, *world_rect, target.size);
        let mask_bind = if range.mask.is_some() {
            &entry.mask_bind
        } else {
            &entry.mask_bind_empty
        };
        let ramp_bind = entry
            .ramp_binds
            .get(&light.texture_slot)
            .map(|(_, bind)| bind)
            .unwrap_or(&entry.ramp_bind_empty);
        for layer in light.layer_indices() {
            let Some(view) = target.views.get(layer as usize) else {
                continue;
            };
            let mut pass = begin_pass_load(encoder, "light2d_light", view, scissor);
            pass.set_pipeline(&entry.pipelines[blend_index(light.blend)]);
            pass.set_bind_group(0, &entry.uniform_bind, &[slot as u32 * entry.uniform_stride]);
            pass.set_bind_group(1, mask_bind, &[]);
            pass.set_bind_group(2, ramp_bind, &[]);
            pass.set_vertex_buffer(0, entry.vertices.slice(..));
            pass.draw(range.quad.0..range.quad.0 + range.quad.1, 0..1);
        }
    }
}

fn begin_pass<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    label: &'static str,
    view: &'a wgpu::TextureView,
    clear: wgpu::Color,
    scissor: Option<[u32; 4]>,
) -> wgpu::RenderPass<'a> {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(clear),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    if let Some([x, y, w, h]) = scissor {
        pass.set_scissor_rect(x, y, w, h);
    }
    pass
}

fn begin_pass_load<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    label: &'static str,
    view: &'a wgpu::TextureView,
    scissor: Option<[u32; 4]>,
) -> wgpu::RenderPass<'a> {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    if let Some([x, y, w, h]) = scissor {
        pass.set_scissor_rect(x, y, w, h);
    }
    pass
}

impl WgpuCallback for Light2dPass {
    fn resource_key(&self) -> Option<&str> {
        Some(SHARED_CALLBACK_KEY)
    }

    fn resource_type_id(&self) -> std::any::TypeId {
        std::any::TypeId::of::<LightScope>()
    }

    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        screen: &ScreenDescriptor,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        self.prepare_frame(device, queue, encoder, screen, resources);
        Vec::new()
    }

    fn paint(
        &self,
        _info: repose_core::PaintCallbackInfo,
        _rpass: &mut repose_render_wgpu::CallbackRenderPass<'_, '_>,
        _resources: &CallbackResources,
    ) {
        // This pass only produces the light map batches consume; it paints
        // nothing of its own into the main target.
    }
}

#[cfg(test)]
mod tests {
    use super::super::light::{
        AmbientLight, Light2d, LightBudget, LightFalloff, LightStop, stops_at,
    };
    use super::*;

    fn light_plan(lights: &[Light2d], occluders: &[Occluder2d]) -> LightingPlan {
        plan_lighting(
            lights,
            occluders,
            &AmbientLight::default(),
            [-100.0, -50.0, 200.0, 100.0],
            &LightBudget::default(),
        )
    }

    #[test]
    fn lightmap_clip_puts_the_world_rect_on_the_unit_square() {
        let rect = [100.0, 200.0, 40.0, 20.0];
        // Top-left of the rect is clip (-1, +1); bottom-right (1, -1).
        assert_eq!(lightmap_clip(Vec2::new(100.0, 200.0), rect), [-1.0, 1.0]);
        let br = lightmap_clip(Vec2::new(140.0, 220.0), rect);
        assert!(
            (br[0] - 1.0).abs() < 1e-5 && (br[1] + 1.0).abs() < 1e-5,
            "bottom-right lands on (1, -1): {br:?}"
        );
        let centre = lightmap_clip(Vec2::new(120.0, 210.0), rect);
        assert!(centre[0].abs() < 1e-6 && centre[1].abs() < 1e-6);
    }

    #[test]
    fn mask_clip_stretches_the_footprint_over_the_mask() {
        let centre = Vec2::new(10.0, -4.0);
        let half = Vec2::new(8.0, 2.0);
        assert_eq!(mask_clip(centre - half, centre, half), [-1.0, 1.0]);
        assert_eq!(mask_clip(centre + half, centre, half), [1.0, -1.0]);
        // The disc shader's uv and the mask clip agree on the same corner.
        let corner = Vec2::new(2.0, -6.0);
        let uv = (corner - (centre - half)) / (half * 2.0);
        let clip = mask_clip(corner, centre, half);
        assert!((clip[0] - (uv.x * 2.0 - 1.0)).abs() < 1e-6);
        assert!((clip[1] - (1.0 - uv.y * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn scissor_covers_the_footprint_and_clamps_to_the_map() {
        let rect = [0.0, 0.0, 100.0, 50.0];
        let size = [200, 100];
        // A footprint in the middle: 2 px per world unit, y down.
        assert_eq!(
            scissor_px([10.0, 10.0, 20.0, 20.0], rect, size),
            Some([20, 20, 20, 20])
        );
        // Off-map boxes are skipped instead of clamped to a sliver.
        assert_eq!(scissor_px([-50.0, -50.0, -10.0, -10.0], rect, size), None);
        assert_eq!(scissor_px([200.0, 0.0, 300.0, 10.0], rect, size), None);
        // Partially visible boxes clamp.
        assert_eq!(
            scissor_px([-10.0, -10.0, 10.0, 10.0], rect, size),
            Some([0, 0, 20, 20])
        );
        assert_eq!(
            scissor_px([90.0, 40.0, 200.0, 100.0], rect, size),
            Some([180, 80, 20, 20])
        );
    }

    #[test]
    fn vertex_packing_orders_quads_then_wedges_per_light() {
        let lights = [
            Light2d {
                position: Vec2::new(0.0, 0.0),
                radius: 10.0,
                shadows: true,
                ..Default::default()
            },
            Light2d {
                position: Vec2::new(30.0, 0.0),
                radius: 5.0,
                ..Default::default()
            },
        ];
        let occluders = [Occluder2d::rect(Vec2::new(2.0, -2.0), Vec2::new(4.0, 2.0))];
        let plan = light_plan(&lights, &occluders);
        assert_eq!(plan.lights.len(), 2);
        assert!(plan.lights[0].shadowed && !plan.lights[1].shadowed);
        let world_rect = plan.world_rect;
        let (vertices, ranges) = build_vertices(&plan, world_rect);
        assert_eq!(vertices.len(), 6 + 4 * 6 + 6, "two quads and four wedges");
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0].quad, (0, 6));
        assert_eq!(ranges[0].mask, Some((6, 24)));
        assert_eq!(ranges[1].quad, (30, 6));
        assert_eq!(ranges[1].mask, None);
        // Every vertex is finite, and the light quads land on their world
        // corners.
        for vertex in &vertices {
            assert!(vertex.clip[0].is_finite() && vertex.clip[1].is_finite());
        }
        let top_left = vertices[0];
        assert_eq!(top_left.world, [-10.0, -10.0]);
        assert_eq!(
            top_left.clip,
            lightmap_clip(Vec2::new(-10.0, -10.0), world_rect)
        );
        assert_eq!(vertices[5].world, [-10.0, 10.0]);
        // Wedge vertices carry the tap weight in its own mask channel and the
        // mask clip coordinates.
        let wedge = vertices[6];
        assert_eq!(wedge.weights, [1.0, 0.0, 0.0, 0.0], "one hard tap");
        assert_eq!(
            wedge.clip,
            mask_clip(
                Vec2::from_array(wedge.world),
                Vec2::ZERO,
                Vec2::splat(10.0)
            )
        );
    }

    #[test]
    fn uniform_packing_matches_the_shader_layout() {
        let light = Light2d {
            position: Vec2::new(4.0, -2.0),
            radius: 8.0,
            color: [1.0, 0.5, 0.25],
            energy: 0.75,
            shadows: true,
            falloff: LightFalloff::stops(vec![
                LightStop {
                    at: 0.0,
                    color: [1.0, 1.0, 1.0, 1.0],
                },
                LightStop {
                    at: 1.0,
                    color: [0.0, 0.0, 0.0, 0.0],
                },
            ]),
            ..Default::default()
        };
        let plan = light_plan(&[light], &[]);
        let stride = 512;
        let (bytes, count) = build_uniforms(&plan, stride);
        assert_eq!(count, 1);
        assert_eq!(bytes.len(), stride);
        let packed: &LightUniform = bytemuck::from_bytes(&bytes[..size_of::<LightUniform>()]);
        assert_eq!(packed.center_half, [4.0, -2.0, 0.125, 0.125]);
        assert_eq!(packed.color_energy, [1.0, 0.5, 0.25, 0.75]);
        // Mode 0 = stops, two live stops, and the unused slots repeat the last.
        assert_eq!(packed.params, [0.0, 2.0, 0.0, 0.0]);
        assert_eq!(packed.mask_xform, [-4.0, -10.0, 1.0 / 16.0, 1.0 / 16.0]);
        assert_eq!(packed.stop_at[0], [0.0, 1.0, 1.0, 1.0]);
        assert_eq!(packed.stop_rgba[1], [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(packed.stop_rgba[7], [0.0, 0.0, 0.0, 0.0]);
        // Padding is wider than the uniform: the next slot starts at `stride`.
        let (bytes, _) = build_uniforms(&plan, stride);
        assert!(bytes[size_of::<LightUniform>()..].iter().all(|b| *b == 0));
    }

    #[test]
    fn packed_uniform_reproduces_the_shader_ramp_and_uvs() {
        // The shader is not executed by these checks, so its arithmetic is
        // replayed here against the CPU helpers it must agree with.
        let stops = vec![
            LightStop {
                at: 0.0,
                color: [1.0, 0.8, 0.6, 1.0],
            },
            LightStop {
                at: 0.4,
                color: [0.5, 0.4, 0.3, 0.5],
            },
            LightStop {
                at: 1.0,
                color: [0.0, 0.0, 0.0, 0.0],
            },
        ];
        let ramp_light = Light2d {
            position: Vec2::new(10.0, 20.0),
            radius: 40.0,
            falloff: LightFalloff::stops(stops.clone()),
            ..Default::default()
        };
        let plan = light_plan(&[ramp_light], &[]);
        let (bytes, _) = build_uniforms(&plan, 256);
        let packed: &LightUniform = bytemuck::from_bytes(&bytes[..size_of::<LightUniform>()]);
        assert_eq!(packed.params[0], 0.0, "stops mode");
        assert_eq!(packed.params[1], 3.0, "three live stops");
        // `light_disc.wgsl::stops_ramp`, replayed.
        let shader_ramp = |t: f32| {
            let mut ramp = packed.stop_rgba[0];
            let mut previous = 0.0f32;
            for i in 1..8 {
                if i as f32 >= packed.params[1] {
                    break;
                }
                let at = packed.stop_at[i / 4][i % 4];
                let span = (at - previous).max(1e-6);
                let w = ((t - previous) / span).clamp(0.0, 1.0);
                for c in 0..4 {
                    ramp[c] += (packed.stop_rgba[i][c] - ramp[c]) * w;
                }
                previous = at;
            }
            ramp
        };
        for t in [0.0, 0.1, 0.25, 0.4, 0.5, 0.75, 1.0, 1.5] {
            let cpu = stops_at(&stops, t);
            let gpu = shader_ramp(t);
            assert!(
                cpu.iter().zip(gpu.iter()).all(|(a, b)| (a - b).abs() < 1e-5),
                "t={t}: cpu {cpu:?} vs shader {gpu:?}"
            );
        }

        let texture_light = Light2d {
            position: Vec2::new(10.0, 20.0),
            radius: 40.0,
            falloff: LightFalloff::Texture {
                slot: 3,
                half_extent: Vec2::new(16.0, 8.0),
                offset: Vec2::new(2.0, -1.0),
            },
            ..Default::default()
        };
        let plan = light_plan(&[texture_light.clone()], &[]);
        let (bytes, _) = build_uniforms(&plan, 256);
        let packed: &LightUniform = bytemuck::from_bytes(&bytes[..size_of::<LightUniform>()]);
        assert_eq!(packed.params[0], 1.0, "texture mode");
        // `light_disc.wgsl`'s uv: (world - centre - offset) * tex.zw * 0.5 + 0.5.
        let shader_uv = |world: Vec2| {
            let d = world - Vec2::new(packed.center_half[0], packed.center_half[1])
                - Vec2::new(packed.tex[0], packed.tex[1]);
            Vec2::new(d.x * packed.tex[2], d.y * packed.tex[3]) * 0.5 + Vec2::splat(0.5)
        };
        for world in [
            Vec2::new(12.0, 19.0),
            Vec2::new(11.0, 20.0),
            Vec2::new(26.0, 28.0),
            Vec2::new(-4.0, 12.0),
        ] {
            let cpu = super::super::light::texture_uv(
                texture_light.position,
                Vec2::new(2.0, -1.0),
                Vec2::new(16.0, 8.0),
                world,
            );
            let gpu = shader_uv(world);
            assert!(
                (cpu - gpu).length() < 1e-5,
                "uv mismatch at {world:?}: cpu {cpu:?} vs shader {gpu:?}"
            );
        }
    }

    #[test]
    fn blend_states_are_godots_three_modes() {
        // Add: src * src.a + dst, multiplier channel untouched.
        let add = blend_state(LightBlend::Add);
        assert_eq!(add.color, ADD);
        assert_eq!(add.alpha, KEEP_MULTIPLIER);
        // Sub: dst - src * src.a.
        let sub = blend_state(LightBlend::Sub);
        assert_eq!(sub.color.operation, wgpu::BlendOperation::ReverseSubtract);
        assert_eq!(sub.color.src_factor, wgpu::BlendFactor::SrcAlpha);
        // Mix: src + dst * (1 - src.a), multiplier scaled the same way.
        let mix = blend_state(LightBlend::Mix);
        assert_eq!(mix.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
        assert_eq!(mix.color.src_factor, wgpu::BlendFactor::One);
        assert_eq!(mix.alpha.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
        assert_eq!(mix.alpha.src_factor, wgpu::BlendFactor::Zero);
        assert_eq!(
            [0, 1, 2],
            [
                blend_index(LightBlend::Add),
                blend_index(LightBlend::Sub),
                blend_index(LightBlend::Mix)
            ]
        );
        // The mask unions instead of adding: each tap has its own channel, and
        // the light shader sums the channels.
        assert_eq!(MASK_BLEND.color.operation, wgpu::BlendOperation::Max);
        assert_eq!(MASK_BLEND.alpha.operation, wgpu::BlendOperation::Max);
        assert_eq!(MASK_FORMAT, wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(MAX_MASK_TAPS, 4, "one channel per Pcf4 tap");
    }
}
