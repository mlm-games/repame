//! Optional post-process and anti-aliasing stages for the viewport pass chain.
//!
//! Every effect here is opt-in and expressed as extra [`RenderPass`]es, so it
//! composes with the passes a game already declares and costs nothing when it
//! is off. [`PostProcess::plan`] turns a settings block into that pass list:
//!
//! ```text
//! post.scene (Render)          the frame's geometry, batch format
//! post.depth (Depth)           view depth + camera velocity, Rgba16Float
//! post.bloom.d0..dn-1 (Fullscreen)   prefilter + 13-tap downsample pyramid
//! post.bloom.u0..un-2 (Fullscreen)   3x3 tent upsample, aux = the down level
//! post.color (Fullscreen)      scene + bloom * intensity
//! post.dof (Fullscreen)        depth-of-field, aux = post.depth
//! post.motionblur (Fullscreen) camera motion blur, aux = post.depth
//! post.fxaa (Fullscreen)       FXAA, presents
//! ```
//!
//! Costs, all at the scene resolution unless noted:
//!
//! | Stage | Passes | Fill cost |
//! |---|---|---|
//! | Bloom, `n` levels | `2n + 1` | ~0.75 scene passes (each level is 1/4 the previous) |
//! | Depth + velocity | 1 geometry replay | one extra scene draw, no shading |
//! | DOF | 1 | 16 taps |
//! | Motion blur | 1 | up to 16 taps |
//! | FXAA | 1 | 5 taps + early out |
//!
//! Bloom operates on the tone-mapped LDR scene buffer, because the viewport
//! has no HDR scene target yet; a threshold above 1.0 therefore cannot be
//! reached and bloom keys off the top of the display range instead. Moving to
//! an HDR scene target later only changes the source texture, not this chain.
//!
//! Depth of field and motion blur need the depth/velocity pass. Motion blur is
//! *camera* motion blur: the velocity is the reprojection delta between the
//! previous and current view-projection, so moving objects do not smear until
//! the engine grows a per-object velocity buffer. TAA is deliberately absent:
//! it needs correct per-pixel motion vectors and history rejection, and a
//! half-wired TAA is worse than none.

use glam::Mat4;

use crate::pass::{
    MAX_PASS_TARGETS, PassError, PassKind, RenderPass, RenderTarget, RenderTargetFormat,
    TargetSemantics,
};

/// Kind of screen-space effect one [`PostStage`] runs.
///
/// The indices are the shader's `mode` uniform, so they are part of the GPU
/// contract and must not be renumbered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PostEffect {
    /// Threshold the source with a soft knee and box-downsample it.
    BloomPrefilter,
    /// 13-tap downsample into the next-smaller mip.
    BloomDownsample,
    /// 3x3 tent upsample of the source, added to `aux` (the down level).
    BloomUpsample,
    /// Add `aux * intensity` (the bloom pyramid root) to the source.
    BloomCombine,
    /// Depth-of-field blur; `aux` holds view depth in `.r`.
    Dof,
    /// Camera motion blur along the velocity in `aux.gb`.
    MotionBlur,
    /// FXAA 3.11 edge blur.
    Fxaa,
    /// Ray-marched participating media; `aux` holds view depth in `.r`.
    VolumetricFog,
}

impl PostEffect {
    /// Value the shader switches on.
    pub const fn mode(self) -> f32 {
        match self {
            Self::BloomPrefilter => 1.0,
            Self::BloomDownsample => 2.0,
            Self::BloomUpsample => 3.0,
            Self::BloomCombine => 4.0,
            Self::Dof => 5.0,
            Self::MotionBlur => 6.0,
            Self::Fxaa => 7.0,
            Self::VolumetricFog => 8.0,
        }
    }
}

/// One fullscreen effect pass: what to run and up to two `[f32; 4]` parameter
/// blocks. `aux` is a second sampled target (`None` binds a 1x1 dummy).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PostStage {
    pub effect: PostEffect,
    pub params: [f32; 4],
    pub params2: [f32; 4],
}

impl PostStage {
    pub const fn new(effect: PostEffect, params: [f32; 4], params2: [f32; 4]) -> Self {
        Self {
            effect,
            params,
            params2,
        }
    }
}

/// Bloom: a soft-knee threshold plus a down/upsample pyramid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BloomSettings {
    /// Luminance above which a texel contributes. Clamped `>= 0`.
    pub threshold: f32,
    /// Soft knee width around the threshold. Clamped `>= 0`.
    pub soft_knee: f32,
    /// Added bloom strength. `0` disables the effect.
    pub intensity: f32,
    /// Pyramid depth, clamped `1..=`[`MAX_BLOOM_LEVELS`].
    pub levels: u32,
}

/// Deepest bloom pyramid [`BloomSettings`] accepts.
pub const MAX_BLOOM_LEVELS: u32 = 4;

impl Default for BloomSettings {
    fn default() -> Self {
        Self {
            threshold: 0.8,
            soft_knee: 0.4,
            intensity: 0.6,
            levels: 3,
        }
    }
}

impl BloomSettings {
    /// True when the settings would contribute anything.
    pub fn is_active(&self) -> bool {
        self.intensity > 0.0 && self.levels > 0
    }

    fn clamped_levels(&self) -> u32 {
        self.levels.clamp(1, MAX_BLOOM_LEVELS)
    }
}

/// Depth of field, expressed the way Godot's practical camera attributes are:
/// a focus distance, a sharp range around it, and separate near/far blur
/// amounts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DofSettings {
    /// View-space distance that stays perfectly sharp, in world units.
    pub focus_distance: f32,
    /// Half-width of the sharp band around [`focus_distance`].
    pub focus_range: f32,
    /// Blur multiplier in front of the focus band, `0..=1`.
    pub near_blur: f32,
    /// Blur multiplier behind the focus band, `0..=1`.
    pub far_blur: f32,
    /// Largest blur radius in pixels.
    pub max_radius_px: f32,
}

impl Default for DofSettings {
    fn default() -> Self {
        Self {
            focus_distance: 30.0,
            focus_range: 5.0,
            near_blur: 0.5,
            far_blur: 1.0,
            max_radius_px: 8.0,
        }
    }
}

/// Camera motion blur along the reprojection velocity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionBlurSettings {
    /// Fraction of the velocity to smear, `0..=1`.
    pub strength: f32,
    /// Largest blur radius in pixels.
    pub max_radius_px: f32,
    /// Taps along the velocity, clamped `1..=16`.
    pub samples: u32,
}

impl Default for MotionBlurSettings {
    fn default() -> Self {
        Self {
            strength: 0.5,
            max_radius_px: 12.0,
            samples: 8,
        }
    }
}

/// FXAA tuning, matching the reference shader's two thresholds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FxaaSettings {
    /// Contrast above which an edge is shaded, as a fraction of local luma.
    pub edge_threshold: f32,
    /// Absolute floor under [`edge_threshold`], so dark edges still resolve.
    pub edge_threshold_min: f32,
}

impl Default for FxaaSettings {
    fn default() -> Self {
        // The reference shader's defaults (Bevy ships the same pair).
        Self {
            edge_threshold: 0.125,
            edge_threshold_min: 0.0312,
        }
    }
}

/// Volumetric fog: participating media ray-marched per pixel against the
/// depth/velocity target, distinct from the per-material distance fog (which
/// keeps running inside the scene shader).
///
/// The march runs from the eye to the sampled view depth along the pixel's
/// view ray, accumulating an emission-absorption integral with a height
/// density profile (`density * exp(-height_falloff * max(y - height_base, 0))`)
/// and a cheap single-scatter phase term toward the frame's sun. It is applied
/// after bloom/DOF/motion blur (it reads neither) and before anti-aliasing, on
/// the tone-mapped scene buffer, so it composites in display space rather than
/// radiance; moving the chain to an HDR target later changes the sampled
/// source, not this stage.
///
/// Cost: one full-screen pass, `steps` samples and one dependent read per
/// sample; `steps` is clamped to `1..=64` and defaults to 24.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VolumetricFogSettings {
    /// Extinction per world unit at [`height_base`](Self::height_base).
    pub density: f32,
    /// How fast density decays above the base height, per world unit.
    /// `0` makes the medium uniform.
    pub height_falloff: f32,
    /// World-space Y where density is highest.
    pub height_base: f32,
    /// Media color (linear RGB), also the in-scattered light color.
    pub color: [f32; 3],
    /// Overall multiplier on the in-scattered radiance.
    pub intensity: f32,
    /// Ray-march distance ceiling in world units.
    pub max_distance: f32,
    /// Samples along the ray, clamped `1..=64`.
    pub steps: u32,
    /// Fraction of the scattering that is directionless, `0..=1`.
    pub ambient: f32,
    /// Forward-scattering anisotropy, clamped `0..=0.95`.
    pub anisotropy: f32,
}

impl Default for VolumetricFogSettings {
    fn default() -> Self {
        Self {
            density: 0.02,
            height_falloff: 0.12,
            height_base: 0.0,
            color: [0.6, 0.7, 0.85],
            intensity: 1.0,
            max_distance: 400.0,
            steps: 24,
            ambient: 0.6,
            anisotropy: 0.3,
        }
    }
}

impl VolumetricFogSettings {
    /// True when the settings would contribute anything.
    pub fn is_active(&self) -> bool {
        self.density > 0.0 && self.intensity > 0.0 && self.max_distance > 0.0
    }

    /// Sample count actually marched.
    pub fn clamped_steps(&self) -> u32 {
        self.steps.clamp(1, 64)
    }

    /// `(density, height_falloff, height_base, ambient)` for the shader.
    pub fn packed(&self) -> ([f32; 4], [f32; 4]) {
        (
            [
                self.density.max(0.0),
                self.height_falloff.max(0.0),
                self.height_base,
                self.ambient.clamp(0.0, 1.0),
            ],
            [
                self.intensity.max(0.0),
                self.max_distance.max(1e-3),
                self.clamped_steps() as f32,
                self.anisotropy.clamp(0.0, 0.95),
            ],
        )
    }
}

/// Which anti-aliasing stage runs last in the chain.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum AntiAliasing {
    /// No post AA; MSAA (if the surface was created with it) still applies.
    #[default]
    Off,
    /// FXAA: one pass, 5 taps, no history. Cheapest useful AA.
    Fxaa(FxaaSettings),
}

impl AntiAliasing {
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Fxaa(_))
    }
}

/// Frame-scoped post-process configuration.
///
/// All fields default to off, so `PostProcess::default()` plans nothing and the
/// viewport keeps its legacy path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PostProcess {
    pub bloom: Option<BloomSettings>,
    pub dof: Option<DofSettings>,
    pub motion_blur: Option<MotionBlurSettings>,
    pub anti_alias: AntiAliasing,
    /// Participating-media fog (ray-marched); `None` is off.
    pub fog: Option<VolumetricFogSettings>,
    /// Previous frame's view-projection, for camera motion vectors. `None`
    /// makes the viewport use its own per-viewport history instead.
    pub previous_view_proj: Option<Mat4>,
}

impl Default for PostProcess {
    fn default() -> Self {
        Self {
            bloom: None,
            dof: None,
            motion_blur: None,
            anti_alias: AntiAliasing::Off,
            fog: None,
            previous_view_proj: None,
        }
    }
}

impl PostProcess {
    /// A settings block with bloom, DOF and FXAA on and motion blur off.
    pub fn quality() -> Self {
        Self {
            bloom: Some(BloomSettings::default()),
            dof: Some(DofSettings::default()),
            motion_blur: None,
            anti_alias: AntiAliasing::Fxaa(FxaaSettings::default()),
            fog: None,
            previous_view_proj: None,
        }
    }

    /// True when any stage is enabled.
    pub fn is_active(&self) -> bool {
        self.bloom.is_some_and(|b| b.is_active())
            || self.dof.is_some()
            || self.motion_blur.is_some_and(|m| m.strength > 0.0)
            || self.anti_alias.is_active()
            || self.fog.is_some_and(|f| f.is_active())
    }

    /// True when the depth/velocity pass is needed.
    pub fn needs_depth(&self) -> bool {
        self.dof.is_some()
            || self.motion_blur.is_some_and(|m| m.strength > 0.0)
            || self.fog.is_some_and(|f| f.is_active())
    }

    /// Builds the frame's pass list, or `None` when nothing is enabled.
    ///
    /// `scene` is the scene size in pixels and `batch_format` must be the
    /// format the viewport's own pipelines render into, since the scene pass is
    /// drawn with them. The returned chain's `scene_pass` is the 1-based index
    /// to pass to
    /// [`SceneBatch::push_group_for_pass`](crate::SceneBatch::push_group_for_pass).
    pub fn plan(
        &self,
        scene: [u32; 2],
        batch_format: RenderTargetFormat,
    ) -> Result<Option<PostChain>, PostError> {
        if !self.is_active() {
            return Ok(None);
        }
        if scene[0] == 0 || scene[1] == 0 {
            return Err(PostError::ZeroSizedScene);
        }
        let scene_w = scene[0];
        let scene_h = scene[1];
        let mut passes: Vec<RenderPass> = Vec::with_capacity(8);
        let mut current = String::from(SCENE_TARGET);
        let mut current_pass = 0usize;

        // Scene: an ordinary geometry pass the viewport fills through
        // `push_group_for_pass`. It carries its own depth, so the batch's
        // depth-tested pipelines draw into it unchanged.
        let mut scene_pass =
            RenderPass::render(RenderTarget::color(SCENE_TARGET, scene_w, scene_h), Vec::new());
        scene_pass.target.format = batch_format;
        passes.push(scene_pass);
        let scene_index = 1u32;

        // Depth + camera velocity, written by replaying the scene geometry.
        let mut depth_pass = None;
        if self.needs_depth() {
            let target = RenderTarget::new(
                DEPTH_TARGET,
                scene_w,
                scene_h,
                RenderTargetFormat::Rgba16Float,
                TargetSemantics::LinearDepth,
            );
            passes.push(RenderPass {
                target,
                kind: PassKind::Depth,
                source: None,
                groups: Vec::new(),
                presents: false,
                overlay: [1.0; 4],
                post: None,
                aux: None,
            });
            depth_pass = Some(passes.len() as u32);
        }

        // Bloom pyramid.
        if let Some(bloom) = self.bloom.filter(|b| b.is_active()) {
            let levels = bloom.clamped_levels();
            let pyr = bloom_pyramid(scene_w, scene_h, levels);
            let mut prev = current.clone();
            for (i, dim) in pyr.iter().enumerate() {
                let name = format!("post.bloom.d{i}");
                let effect = if i == 0 {
                    PostStage::new(
                        PostEffect::BloomPrefilter,
                        [bloom.threshold.max(0.0), bloom.soft_knee.max(0.0), 0.0, 0.0],
                        [0.0; 4],
                    )
                } else {
                    PostStage::new(PostEffect::BloomDownsample, [0.0; 4], [0.0; 4])
                };
                passes.push(post_pass(&prev, name.clone(), *dim, batch_format, effect, None));
                prev = name;
            }
            // u_{n-1} is the last down level itself; then upsample back down
            // the pyramid, each level adding into the wider down level.
            let mut up = format!("post.bloom.d{}", levels - 1);
            for i in (0..levels.saturating_sub(1)).rev() {
                let down = format!("post.bloom.d{i}");
                let dst = format!("post.bloom.u{i}");
                passes.push(post_pass(
                    &up,
                    dst.clone(),
                    pyr[i as usize],
                    batch_format,
                    PostStage::new(PostEffect::BloomUpsample, [1.0, 0.0, 0.0, 0.0], [0.0; 4]),
                    Some(down),
                ));
                up = dst;
            }
            passes.push(post_pass(
                &current,
                "post.color".to_string(),
                [scene_w, scene_h],
                batch_format,
                PostStage::new(
                    PostEffect::BloomCombine,
                    [bloom.intensity.max(0.0), 0.0, 0.0, 0.0],
                    [0.0; 4],
                ),
                Some(up),
            ));
            current = String::from("post.color");
            current_pass = passes.len() - 1;
        }

        // Depth of field.
        if let Some(dof) = self.dof {
            if depth_pass.is_none() {
                return Err(PostError::MissingDepth);
            }
            let target = format!("post.dof");
            let stage = PostStage::new(
                PostEffect::Dof,
                [
                    dof.focus_distance.max(0.0),
                    dof.focus_range.max(1e-3),
                    dof.near_blur.clamp(0.0, 1.0),
                    dof.far_blur.clamp(0.0, 1.0),
                ],
                [dof.max_radius_px.max(0.0), 16.0, 0.0, 0.0],
            );
            passes.push(post_pass(
                &current,
                target.clone(),
                [scene_w, scene_h],
                batch_format,
                stage,
                Some(DEPTH_TARGET.to_string()),
            ));
            current = target;
            current_pass = passes.len() - 1;
        }

        // Camera motion blur.
        if let Some(mb) = self.motion_blur.filter(|m| m.strength > 0.0) {
            if depth_pass.is_none() {
                return Err(PostError::MissingDepth);
            }
            let target = format!("post.motionblur");
            let samples = mb.samples.clamp(1, 16) as f32;
            let stage = PostStage::new(
                PostEffect::MotionBlur,
                [mb.strength.clamp(0.0, 1.0), mb.max_radius_px.max(0.0), samples, 0.0],
                [0.0; 4],
            );
            passes.push(post_pass(
                &current,
                target.clone(),
                [scene_w, scene_h],
                batch_format,
                stage,
                Some(DEPTH_TARGET.to_string()),
            ));
            current = target;
            current_pass = passes.len() - 1;
        }

        // Volumetric fog, after the color stages and before AA.
        if let Some(fog) = self.fog.filter(|f| f.is_active()) {
            if depth_pass.is_none() {
                return Err(PostError::MissingDepth);
            }
            let (params, params2) = fog.packed();
            passes.push(post_pass(
                &current,
                "post.fog".to_string(),
                [scene_w, scene_h],
                batch_format,
                PostStage::new(PostEffect::VolumetricFog, params, params2),
                Some(DEPTH_TARGET.to_string()),
            ));
            current = String::from("post.fog");
            current_pass = passes.len() - 1;
        }

        // Anti-aliasing, last, and the pass that presents.
        if let AntiAliasing::Fxaa(fxaa) = self.anti_alias {
            let stage = PostStage::new(
                PostEffect::Fxaa,
                [fxaa.edge_threshold, fxaa.edge_threshold_min, 0.0, 0.0],
                [0.0; 4],
            );
            passes.push(post_pass(
                &current,
                "post.fxaa".to_string(),
                [scene_w, scene_h],
                batch_format,
                stage,
                None,
            ));
            current_pass = passes.len() - 1;
        }

        if passes.len() > MAX_PASS_TARGETS {
            return Err(PostError::TooManyPasses);
        }
        passes[current_pass].presents = true;
        Ok(Some(PostChain {
            passes,
            scene_pass: scene_index,
            depth_pass,
        }))
    }
}

/// Named target the frame's geometry renders into.
pub const SCENE_TARGET: &str = "post.scene";
/// Named target holding view depth (`.r`) and camera velocity (`.gb`).
pub const DEPTH_TARGET: &str = "post.depth";

/// Why a post-process chain could not be built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PostError {
    /// The scene had a zero dimension.
    ZeroSizedScene,
    /// A depth-consuming stage was requested without the depth pass.
    MissingDepth,
    /// The chain needs more targets than [`MAX_PASS_TARGETS`].
    TooManyPasses,
}

impl From<PostError> for PassError {
    fn from(_: PostError) -> Self {
        PassError::TooManyTargets
    }
}

impl std::fmt::Display for PostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroSizedScene => write!(f, "post-process scene is zero-sized"),
            Self::MissingDepth => write!(f, "post-process stage needs the depth pass"),
            Self::TooManyPasses => write!(f, "post-process chain exceeds MAX_PASS_TARGETS"),
        }
    }
}

impl std::error::Error for PostError {}

/// A planned post-process chain: the frame's passes plus the indices a caller
/// needs to submit geometry.
#[derive(Clone, Debug)]
pub struct PostChain {
    pub passes: Vec<RenderPass>,
    /// 1-based pass index the scene geometry belongs to.
    pub scene_pass: u32,
    /// 1-based pass index of the depth/velocity target, when planned.
    pub depth_pass: Option<u32>,
}

fn post_pass(
    source: &str,
    name: String,
    size: [u32; 2],
    format: RenderTargetFormat,
    stage: PostStage,
    aux: Option<String>,
) -> RenderPass {
    let mut pass = RenderPass::fullscreen(
        source.to_string(),
        RenderTarget::new(name, size[0].max(1), size[1].max(1), format, TargetSemantics::Auxiliary),
    );
    pass.post = Some(stage);
    pass.aux = aux;
    pass
}

/// Bloom pyramid dimensions: the source halved `levels` times, floored at 1
/// pixel and stopping early once a level would be 1x1 (repeating it would add
/// passes that only blur one pixel).
pub fn bloom_pyramid(width: u32, height: u32, levels: u32) -> Vec<[u32; 2]> {
    let mut out = Vec::new();
    let (mut w, mut h) = (width.max(1), height.max(1));
    for _ in 0..levels.clamp(1, MAX_BLOOM_LEVELS) {
        w = (w / 2).max(1);
        h = (h / 2).max(1);
        out.push([w, h]);
        if w == 1 && h == 1 {
            break;
        }
    }
    out
}

/// Circle-of-confusion radius in pixels for a view-space depth, matching the
/// shader's `coc_radius`. Zero inside the sharp band.
pub fn coc_radius(depth: f32, dof: &DofSettings) -> f32 {
    let range = dof.focus_range.max(1e-3);
    let signed = depth - dof.focus_distance;
    let blur = if signed < -range {
        ((-signed - range) / range).min(1.0) * dof.near_blur.clamp(0.0, 1.0)
    } else if signed > range {
        ((signed - range) / range).min(1.0) * dof.far_blur.clamp(0.0, 1.0)
    } else {
        0.0
    };
    (blur * dof.max_radius_px.max(0.0)).min(dof.max_radius_px.max(0.0))
}

/// Relative luminance the shader's FXAA uses (`0.299/0.587/0.114`).
pub fn fxaa_luma(rgb: [f32; 3]) -> f32 {
    rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114
}

/// FXAA's contrast early-out: true when an edge is strong enough to blend.
pub fn fxaa_edge_applies(luma_min: f32, luma_max: f32, fxaa: &FxaaSettings) -> bool {
    luma_max - luma_min >= luma_max * fxaa.edge_threshold.max(0.0) + fxaa.edge_threshold_min
}

/// FXAA's edge direction from the four diagonal luminances, before the
/// direction-reduce clamp the shader applies. Pins the sign convention: a
/// vertical edge (different luminance left vs right) yields `dir.x`.
pub fn fxaa_direction(nw: f32, ne: f32, sw: f32, se: f32) -> [f32; 2] {
    [
        -((nw + ne) - (sw + se)),
        (nw + sw) - (ne + se),
    ]
}

/// Fullscreen post-process shader: one module with a `mode` switch, sampling a
/// source and an optional auxiliary target.
pub const POST_WGSL: &str = r#"
struct PostParams {
    params: vec4<f32>,
    params2: vec4<f32>,
    texel: vec4<f32>,
    // `meta` is reserved in WGSL; the lane layout is unchanged.
    stage: vec4<f32>,
    // Camera state for the ray-marched fog stage: (xyz, tan_half_fov),
    // (xyz, aspect), (xyz right, unused).
    cam_pos: vec4<f32>,
    cam_forward: vec4<f32>,
    cam_right: vec4<f32>,
    // Sun direction (xyz, unit) and color (rgb, diffuse).
    light_dir: vec4<f32>,
    light_color: vec4<f32>,
    // Media color (rgb, unused).
    fog_color: vec4<f32>,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_smp: sampler;
@group(0) @binding(2) var<uniform> post: PostParams;
@group(0) @binding(3) var aux_tex: texture_2d<f32>;

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

fn src(uv: vec2<f32>) -> vec3<f32> {
    return textureSample(src_tex, src_smp, uv).rgb;
}

fn aux(uv: vec2<f32>) -> vec4<f32> {
    return textureSample(aux_tex, src_smp, uv);
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

fn soft_threshold(c: vec3<f32>, threshold: f32, knee: f32) -> vec3<f32> {
    let br = max(c.r, max(c.g, c.b));
    let k = max(knee, 1e-4);
    var soft = clamp(br - threshold + k, 0.0, 2.0 * k);
    soft = soft * soft / (4.0 * k);
    let contrib = max(soft, br - threshold) / max(br, 1e-4);
    return c * contrib;
}

fn downsample_13(uv: vec2<f32>) -> vec3<f32> {
    let t = post.texel.xy;
    let a = src(uv + vec2<f32>(-2.0, 2.0) * t);
    let b = src(uv + vec2<f32>(0.0, 2.0) * t);
    let c = src(uv + vec2<f32>(2.0, 2.0) * t);
    let d = src(uv + vec2<f32>(-2.0, 0.0) * t);
    let e = src(uv);
    let f = src(uv + vec2<f32>(2.0, 0.0) * t);
    let g = src(uv + vec2<f32>(-2.0, -2.0) * t);
    let h = src(uv + vec2<f32>(0.0, -2.0) * t);
    let i = src(uv + vec2<f32>(2.0, -2.0) * t);
    let j = src(uv + vec2<f32>(-1.0, 1.0) * t);
    let k = src(uv + vec2<f32>(1.0, 1.0) * t);
    let l = src(uv + vec2<f32>(-1.0, -1.0) * t);
    let m = src(uv + vec2<f32>(1.0, -1.0) * t);
    var o = e * 0.125;
    o = o + (a + c + g + i) * 0.03125;
    o = o + (b + d + f + h) * 0.0625;
    o = o + (j + k + l + m) * 0.125;
    return o;
}

fn upsample_tent(uv: vec2<f32>, radius: f32) -> vec3<f32> {
    let t = post.texel.xy * radius;
    var o = vec3<f32>(0.0);
    for (var y: i32 = -1; y <= 1; y = y + 1) {
        for (var x: i32 = -1; x <= 1; x = x + 1) {
            let w = f32(2 - abs(x)) * f32(2 - abs(y));
            o = o + src(uv + vec2<f32>(f32(x), f32(y)) * t) * w;
        }
    }
    return o * (1.0 / 16.0);
}

fn coc_px(depth: f32) -> f32 {
    let focus = post.params.x;
    let range = max(post.params.y, 1e-3);
    let signed = depth - focus;
    var blur = 0.0;
    if (signed < -range) {
        blur = min((-signed - range) / range, 1.0) * post.params.z;
    } else if (signed > range) {
        blur = min((signed - range) / range, 1.0) * post.params.w;
    }
    return clamp(blur * post.params2.x, 0.0, post.params2.x);
}

fn dof(uv: vec2<f32>) -> vec3<f32> {
    let depth = aux(uv).r;
    let px = coc_px(depth);
    if (px < 0.5) {
        return src(uv);
    }
    let radius = px * post.texel.xy;
    let samples = i32(post.params2.y + 0.5);
    var acc = src(uv);
    var wsum = 1.0;
    // Golden-angle spiral: even coverage without a lookup table.
    for (var i: i32 = 1; i < 16; i = i + 1) {
        if (i >= samples) {
            break;
        }
        let fi = f32(i);
        let ang = fi * 2.399963;
        let rad = sqrt(fi / f32(samples));
        acc = acc + src(uv + vec2<f32>(cos(ang), sin(ang)) * radius * rad);
        wsum = wsum + 1.0;
    }
    return acc / wsum;
}

fn motion_blur(uv: vec2<f32>) -> vec3<f32> {
    let vel = aux(uv).gb * post.params.x * post.params.y;
    let n = i32(post.params2.z + 0.5);
    var acc = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var i: i32 = 0; i < 16; i = i + 1) {
        if (i >= n) {
            break;
        }
        let t = f32(i) / f32(max(n - 1, 1)) - 0.5;
        acc = acc + src(uv - vel * post.texel.xy * t);
        wsum = wsum + 1.0;
    }
    return acc / max(wsum, 1.0);
}

// Ray-marched participating media. `params` = (density, height_falloff,
// height_base, ambient), `params2` = (intensity, max_distance, steps,
// anisotropy). The sampled view depth bounds the march: fog past the
// geometry at this pixel would show through an occluder.
fn volumetric_fog(uv: vec2<f32>) -> vec3<f32> {
    let scene = src(uv);
    let depth = aux(uv).r;
    let steps = max(i32(post.params2.z + 0.5), 1);
    let max_distance = max(post.params2.y, 1e-3);
    // Background pixels have no depth: fog them out to the distance ceiling.
    let limit = select(max_distance, min(depth, max_distance), depth > 0.0);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let up = cross(post.cam_right.xyz, post.cam_forward.xyz);
    let tan_half_fov = post.cam_pos.w;
    let dir = normalize(
        post.cam_forward.xyz
            + post.cam_right.xyz * (ndc.x * tan_half_fov * post.cam_forward.w)
            + up * (ndc.y * tan_half_fov));
    let axis = max(dot(dir, post.cam_forward.xyz), 1e-4);
    let step_len = limit / axis / f32(steps);
    let density = post.params.x;
    let falloff = post.params.y;
    let base = post.params.z;
    let ambient = post.params.w;
    let g = post.params2.w;
    let sun_dir = normalize(post.light_dir.xyz);
    let sun = max(dot(dir, sun_dir), 0.0);
    let knee = pow(sun, mix(2.0, 8.0, 1.0 - g));
    let radiance = post.fog_color.rgb * post.params2.x
        * (ambient + (1.0 - ambient) * knee)
        * (0.25 + post.light_color.w * 0.75);
    var trans = 1.0;
    var acc = vec3<f32>(0.0);
    for (var i: i32 = 0; i < 64; i = i + 1) {
        if (i >= steps) {
            break;
        }
        let t = (f32(i) + 0.5) * step_len;
        let p = post.cam_pos.xyz + dir * t;
        let height = max(p.y - base, 0.0);
        let local = density * exp(-falloff * height);
        let absorb = 1.0 - exp(-local * step_len);
        acc = acc + trans * absorb * radiance;
        trans = trans * (1.0 - absorb);
        if (trans < 0.002) {
            break;
        }
    }
    return scene * trans + acc;
}

fn fxaa(uv: vec2<f32>) -> vec3<f32> {    let t = post.texel.xy;
    let rgbM = src(uv);
    let nw = src(uv + vec2<f32>(-1.0, -1.0) * t);
    let ne = src(uv + vec2<f32>(1.0, -1.0) * t);
    let sw = src(uv + vec2<f32>(-1.0, 1.0) * t);
    let se = src(uv + vec2<f32>(1.0, 1.0) * t);
    let lM = luma(rgbM);
    let lNW = luma(nw);
    let lNE = luma(ne);
    let lSW = luma(sw);
    let lSE = luma(se);
    let lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));
    let lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));
    if (lMax - lMin < lMax * post.params.x + post.params.y) {
        return rgbM;
    }
    var dir = vec2<f32>(
        -((lNW + lNE) - (lSW + lSE)),
        (lNW + lSW) - (lNE + lSE),
    );
    let dirReduce = max((lNW + lNE + lSW + lSE) * 0.03125, 0.0078125);
    let rcp = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = clamp(dir * rcp, vec2<f32>(-8.0), vec2<f32>(8.0)) * t;
    let rgbA = 0.5 * (src(uv + dir * (1.0 / 3.0 - 0.5)) + src(uv + dir * (2.0 / 3.0 - 0.5)));
    let rgbB = rgbA * 0.5 + 0.25 * (src(uv - dir * 0.5) + src(uv + dir * 0.5));
    let lB = luma(rgbB);
    if (lB < lMin || lB > lMax) {
        return rgbA;
    }
    return rgbB;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let mode = post.stage.x;
    var rgb = src(in.uv);
    if (mode < 0.5) {
        return vec4<f32>(rgb, 1.0);
    } else if (mode < 1.5) {
        // Bloom prefilter: box-downsample to the half-res target, then threshold.
        let t = post.texel.xy;
        let c = src(in.uv + vec2<f32>(-0.5, -0.5) * t)
              + src(in.uv + vec2<f32>(0.5, -0.5) * t)
              + src(in.uv + vec2<f32>(-0.5, 0.5) * t)
              + src(in.uv + vec2<f32>(0.5, 0.5) * t);
        rgb = soft_threshold(c * 0.25, post.params.x, post.params.y);
    } else if (mode < 2.5) {
        rgb = downsample_13(in.uv);
    } else if (mode < 3.5) {
        rgb = upsample_tent(in.uv, max(post.params.x, 1.0)) + aux(in.uv).rgb;
    } else if (mode < 4.5) {
        rgb = rgb + aux(in.uv).rgb * post.params.x;
    } else if (mode < 5.5) {
        rgb = dof(in.uv);
    } else if (mode < 6.5) {
        rgb = motion_blur(in.uv);
    } else if (mode < 7.5) {
        rgb = fxaa(in.uv);
    } else {
        rgb = volumetric_fog(in.uv);
    }
    return vec4<f32>(rgb, 1.0);
}
"#;

/// Depth + camera velocity shader: the scene's own vertex layout, replayed to
/// write `(view_depth, velocity.x, velocity.y, 1)`.
pub const DEPTH_VELOCITY_WGSL: &str = r#"
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
};

struct PrevView {
    prev_view_proj: mat4x4<f32>,
};

struct SkinPalette {
    joints: array<mat4x4<f32>>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> skin_palette: SkinPalette;
@group(1) @binding(0) var<uniform> prev: PrevView;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) view_depth: f32,
    @location(1) velocity: vec2<f32>,
};

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1) joints: vec4<u32>,
    @location(2) weights: vec4<f32>,
    @location(3) skin_mix: f32,
    @location(4) skin_base: f32,
) -> VsOut {
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
    let clip = camera.view_proj * vec4<f32>(p, 1.0);
    let prev_clip = prev.prev_view_proj * vec4<f32>(p, 1.0);
    let ndc = clip.xy / max(clip.w, 1e-6);
    let prev_ndc = prev_clip.xy / max(prev_clip.w, 1e-6);
    var out: VsOut;
    out.pos = clip;
    out.view_depth = dot(p - camera.cam_pos, camera.cam_forward);
    out.velocity = (ndc - prev_ndc) * 0.5;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.view_depth, in.velocity.x, in.velocity.y, 1.0);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn parses(label: &str, src: &str) {
        if let Err(e) = naga::front::wgsl::parse_str(src) {
            panic!("{label} does not parse: {e:?}");
        }
    }

    #[test]
    fn post_shaders_parse() {
        parses("POST_WGSL", POST_WGSL);
        parses("DEPTH_VELOCITY_WGSL", DEPTH_VELOCITY_WGSL);
    }

    /// The bloom pyramid halves each level and stops at a single pixel: the
    /// exact dimension sequence the pass planner allocates.
    #[test]
    fn bloom_pyramid_halves_to_one_pixel() {
        assert_eq!(
            bloom_pyramid(1920, 1080, 4),
            vec![[960, 540], [480, 270], [240, 135], [120, 67]]
        );
        // A tiny source stops early instead of repeating 1x1 levels.
        assert_eq!(bloom_pyramid(4, 4, 4), vec![[2, 2], [1, 1]]);
        // Zero dimensions still yield one usable level.
        assert_eq!(bloom_pyramid(0, 0, 3), vec![[1, 1]]);
    }

    /// The planner's target count stays inside the pass ceiling at the deepest
    /// bloom, and the scene pass is always first.
    #[test]
    fn plan_fits_the_pass_ceiling() {
        let post = PostProcess {
            bloom: Some(BloomSettings {
                levels: MAX_BLOOM_LEVELS,
                ..BloomSettings::default()
            }),
            dof: Some(DofSettings::default()),
            motion_blur: Some(MotionBlurSettings::default()),
            anti_alias: AntiAliasing::Fxaa(FxaaSettings::default()),
            fog: Some(VolumetricFogSettings::default()),
            previous_view_proj: None,
        };
        let chain = post
            .plan([1920, 1080], RenderTargetFormat::Rgba8UnormSrgb)
            .expect("plans")
            .expect("active");
        assert!(chain.passes.len() <= MAX_PASS_TARGETS, "{}", chain.passes.len());
        assert_eq!(chain.passes[0].target.name, SCENE_TARGET);
        assert_eq!(chain.passes[0].kind, PassKind::Render);
        assert_eq!(chain.passes[0].target.format, RenderTargetFormat::Rgba8UnormSrgb);
        assert_eq!(chain.scene_pass, 1);
        assert_eq!(chain.depth_pass, Some(2));
        // Exactly one pass presents, and it is the last.
        let presenting: Vec<usize> = chain
            .passes
            .iter()
            .enumerate()
            .filter(|(_, p)| p.presents)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(presenting, vec![chain.passes.len() - 1]);
        assert_eq!(
            chain.passes[chain.passes.len() - 1].post.unwrap().effect,
            PostEffect::Fxaa
        );
        // The fog stage sits between motion blur and AA and reads depth.
        let fog = chain
            .passes
            .iter()
            .position(|p| p.post.is_some_and(|s| s.effect == PostEffect::VolumetricFog))
            .expect("fog planned");
        assert!(fog < chain.passes.len() - 1, "fog runs before AA");
        assert_eq!(chain.passes[fog].aux.as_deref(), Some(DEPTH_TARGET));
        // Every fullscreen stage names an existing source.
        for pass in &chain.passes {
            if let Some(src) = &pass.source {
                assert!(
                    chain.passes.iter().any(|p| &p.target.name == src),
                    "source {src} exists"
                );
            }
            if let Some(aux) = &pass.aux {
                assert!(
                    chain.passes.iter().any(|p| &p.target.name == aux),
                    "aux {aux} exists"
                );
            }
        }
    }

    /// Inactive settings plan nothing, so the viewport keeps its legacy path.
    #[test]
    fn inactive_post_plans_nothing() {
        assert!(PostProcess::default().plan([100, 100], RenderTargetFormat::Rgba8Unorm).unwrap().is_none());
        let aa_only = PostProcess {
            anti_alias: AntiAliasing::Fxaa(FxaaSettings::default()),
            ..Default::default()
        };
        let chain = aa_only
            .plan([100, 100], RenderTargetFormat::Rgba8Unorm)
            .unwrap()
            .unwrap();
        assert_eq!(chain.passes.len(), 2, "scene + fxaa");
        assert_eq!(chain.passes[0].target.name, SCENE_TARGET);
        assert_eq!(chain.depth_pass, None);
    }

    #[test]
    fn zero_sized_scene_is_an_error() {
        let post = PostProcess {
            bloom: Some(BloomSettings::default()),
            ..Default::default()
        };
        // `PostChain` is not `PartialEq`: match the error directly.
        assert!(matches!(
            post.plan([0, 100], RenderTargetFormat::Rgba8Unorm),
            Err(PostError::ZeroSizedScene)
        ));
    }

    /// The shader's mode numbers are the enum indices; a renumbering would
    /// silently run the wrong effect.
    #[test]
    fn effect_modes_are_pinned() {
        assert_eq!(PostEffect::BloomPrefilter.mode(), 1.0);
        assert_eq!(PostEffect::BloomDownsample.mode(), 2.0);
        assert_eq!(PostEffect::BloomUpsample.mode(), 3.0);
        assert_eq!(PostEffect::BloomCombine.mode(), 4.0);
        assert_eq!(PostEffect::Dof.mode(), 5.0);
        assert_eq!(PostEffect::MotionBlur.mode(), 6.0);
        assert_eq!(PostEffect::Fxaa.mode(), 7.0);
        assert_eq!(PostEffect::VolumetricFog.mode(), 8.0);
    }

    /// Fog settings clamp into the shader's ranges and pack into the two
    /// parameter blocks the stage carries.
    #[test]
    fn fog_settings_pack() {
        let fog = VolumetricFogSettings {
            density: -1.0,
            height_falloff: -2.0,
            height_base: 3.0,
            steps: 0,
            anisotropy: 4.0,
            ambient: 2.0,
            ..VolumetricFogSettings::default()
        };
        assert!(!fog.is_active(), "negative density is inert");
        assert_eq!(fog.clamped_steps(), 1);
        let fog = VolumetricFogSettings {
            density: 0.05,
            height_falloff: 0.25,
            height_base: 3.0,
            steps: 100,
            anisotropy: 0.5,
            ambient: 0.25,
            ..VolumetricFogSettings::default()
        };
        assert!(fog.is_active());
        let (params, params2) = fog.packed();
        assert_eq!(params, [0.05, 0.25, 3.0, 0.25]);
        assert_eq!(params2[0], fog.intensity);
        assert_eq!(params2[2], 64.0, "steps clamp to the march ceiling");
        assert_eq!(params2[3], 0.5);
        // A frame with only fog still needs the depth pass.
        let post = PostProcess {
            fog: Some(fog),
            ..Default::default()
        };
        assert!(post.needs_depth());
        let chain = post
            .plan([64, 64], RenderTargetFormat::Rgba8Unorm)
            .unwrap()
            .unwrap();
        assert!(chain.depth_pass.is_some());
        assert_eq!(chain.passes.len(), 3, "scene + depth + fog");
        assert!(chain.passes[2].presents);
    }

    /// CoC is zero inside the focus band and saturates at the max radius.
    #[test]
    fn coc_radius_pins_the_focus_band() {
        let dof = DofSettings {
            focus_distance: 10.0,
            focus_range: 2.0,
            near_blur: 0.5,
            far_blur: 1.0,
            max_radius_px: 8.0,
        };
        assert_eq!(coc_radius(10.0, &dof), 0.0);
        assert_eq!(coc_radius(12.0, &dof), 0.0, "band is inclusive");
        // Just past the far edge: half a range past -> half the far blur.
        assert!((coc_radius(13.0, &dof) - 4.0).abs() < 1e-5);
        // Far behind: saturates at the far blur times the radius.
        assert_eq!(coc_radius(100.0, &dof), 8.0);
        // Near side saturates at the (weaker) near blur.
        assert_eq!(coc_radius(0.0, &dof), 4.0);
    }

    /// FXAA's contrast gate and edge direction are pinned by value so the CPU
    /// reference and the shader cannot drift apart unnoticed.
    #[test]
    fn fxaa_edge_case_is_pinned() {
        let fxaa = FxaaSettings::default();
        // High-contrast step: blends.
        assert!(fxaa_edge_applies(0.0, 1.0, &fxaa));
        // Flat region with a tiny local luma: below the absolute floor.
        assert!(!fxaa_edge_applies(0.10, 0.12, &fxaa));
        // Silent drift check: 0.5 * 0.125 + 0.0312 = 0.0937.
        assert!(fxaa_edge_applies(0.5, 0.5 + 0.0937, &fxaa));
        assert!(!fxaa_edge_applies(0.5, 0.5 + 0.0936, &fxaa));
        // A vertical edge (bright right, dark left) points along -x.
        let dir = fxaa_direction(0.0, 1.0, 0.0, 1.0);
        assert_eq!(dir, [-2.0, 0.0]);
        // A horizontal edge (bright below, dark above) points along +y.
        let dir = fxaa_direction(0.0, 0.0, 1.0, 1.0);
        assert_eq!(dir, [0.0, 2.0]);
    }

    #[test]
    fn luma_matches_the_shader_weights() {
        assert!((fxaa_luma([1.0, 1.0, 1.0]) - 1.0).abs() < 1e-6);
        assert!((fxaa_luma([1.0, 0.0, 0.0]) - 0.299).abs() < 1e-6);
    }
}
