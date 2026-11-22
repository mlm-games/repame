//! Render targets and pass description for the offscreen path.
//!
//! The viewport currently paints one depth-tested scene into the UI pass.
//! Games that need water refraction, a reflection buffer, a post chain or a
//! cutscene composite describe the extra work as [`RenderTarget`]s and hand
//! [`Frame3d::passes`](crate::Frame3d::passes) to the batch, which runs them in
//! order and hands each resolved result back as a plain RGBA8 image the next
//! pass (or the game) can sample.
//!
//! [`crate::postfx::PostProcess`] builds such a list automatically; a declared
//! [`PassKind::Depth`] target is filled by replaying the frame's scene geometry
//! (see [`crate::render::prepare_scene_with_id`]), and the viewport appends a
//! [`PassKind::Occlusion`] pass of its own when a frame arms
//! [`crate::Frame3d::occlusion_feed`] and declares no depth pass.
//!
//! [`pass_order`] resolves the list into the order the batch executes: a pass
//! that samples a named target runs after the pass that writes it, and the
//! depth replays run after every scene-color pass they replay.

/// Pixel format of a render target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RenderTargetFormat {
    Rgba8Unorm,
    /// Matches the compositor's scene format, so a pass target can be
    /// sampled and drawn with the batch's own pipelines.
    Rgba8UnormSrgb,
    Bgra8Unorm,
    /// Half-float RGBA: the depth-kind replay format (view depth in `.r`),
    /// filterable.
    Rgba16Float,
    /// Single-lane float: a depth-kind target that feeds the occlusion
    /// chain (`textureLoad` only). Not filterable, so post stages cannot
    /// sample it.
    R32Float,
    /// Depth attachment format, for pipelines that render their own depth.
    Depth32Float,
}

impl RenderTargetFormat {
    pub const fn bytes_per_texel(self) -> u32 {
        match self {
            Self::Depth32Float | Self::R32Float => 4,
            Self::Rgba8Unorm | Self::Rgba8UnormSrgb | Self::Bgra8Unorm => 4,
            Self::Rgba16Float => 8,
        }
    }

    pub const fn is_depth(self) -> bool {
        matches!(self, Self::Depth32Float)
    }

    /// The format carries an alpha channel.
    pub const fn has_alpha(self) -> bool {
        matches!(
            self,
            Self::Rgba8Unorm | Self::Rgba8UnormSrgb | Self::Bgra8Unorm | Self::Rgba16Float
        )
    }

    /// The format stores display-encoded colour, so sampling it decodes.
    pub const fn is_srgb(self) -> bool {
        matches!(self, Self::Rgba8UnormSrgb)
    }

    /// The format can be sampled with a filtering sampler.
    pub const fn is_filterable(self) -> bool {
        !matches!(self, Self::R32Float | Self::Depth32Float)
    }

    /// The target format matching a GPU texture format, or `None` when the
    /// batch cannot render into it. Lets a caller planning passes (post
    /// processing) pick a scene format that the batch's pipelines accept.
    pub const fn from_wgpu(format: wgpu::TextureFormat) -> Option<Self> {
        match format {
            wgpu::TextureFormat::Rgba8Unorm => Some(Self::Rgba8Unorm),
            wgpu::TextureFormat::Rgba8UnormSrgb => Some(Self::Rgba8UnormSrgb),
            wgpu::TextureFormat::Bgra8Unorm => Some(Self::Bgra8Unorm),
            wgpu::TextureFormat::Rgba16Float => Some(Self::Rgba16Float),
            wgpu::TextureFormat::R32Float => Some(Self::R32Float),
            wgpu::TextureFormat::Depth32Float => Some(Self::Depth32Float),
            _ => None,
        }
    }
}

/// What a target holds, so the batch can pick a pipeline and a clear policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TargetSemantics {
    /// Scene color, tone-mapped on resolve.
    SceneColor,
    /// View-space or world normals.
    Normal,
    /// Linear depth for fog, volumetrics or soft particles.
    LinearDepth,
    /// Object ids for picking and deferred reads.
    ObjectId,
    /// Anything else; the game supplies the clear color.
    Auxiliary,
}

/// What a pass does with its input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PassKind {
    /// Draw groups into the target.
    Render,
    /// Draw scene groups into a G-buffer: the declared target (an
    /// `Rgba16Float` colour attachment) holds albedo, while two engine-owned
    /// `Rgba16Float` attachments hold view-space normals (plus the lit flag)
    /// and material params (metallic, roughness, emissive, view depth).
    Gbuffer,
    /// Fullscreen light accumulation over a [`PassKind::Gbuffer`] source.
    /// Samples all three of the source's attachments and the camera uniform,
    /// and writes one colour target. Requires `source`.
    Deferred,
    /// Fill the target by replaying the frame's scene geometry as view-depth
    /// and camera velocity (`Rgba16Float`: depth, velocity x, velocity y, 1;
    /// `R32Float` when only the chain reads it). The batch owns the geometry,
    /// so this pass takes no `groups`.
    Depth,
    /// The engine's own view-depth replay for the occlusion feed, with the
    /// same target requirements as [`PassKind::Depth`] but writing view depth
    /// alone (the `Depth` kind also writes camera velocity for the post
    /// chain). The viewport appends one when a frame arms
    /// [`crate::Frame3d::occlusion_feed`] and declares no depth pass, so
    /// occlusion culling never depends on the post chain; games do not
    /// declare it themselves.
    Occlusion,
    /// Resolve the source target into a sampled texture.
    Resolve,
    /// Fullscreen blit or post chain over the source.
    Fullscreen,
    /// Copy a region of the source target into the target, converting the
    /// pixel format on the way (the framebuffer copy the retail `GXCopyTex`
    /// performs).
    Copy,
}

/// Pixel format a [`PassKind::Copy`] pass writes into its target, mirroring
/// the formats the retail `GXSetTexCopyDst` accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CopyFormat {
    /// No conversion: the source's RGBA is written as it is.
    Rgba8,
    /// 4-bit intensity (`GX_TF_I4`): the source's luma, quantised to four
    /// bits and written to every channel.
    Intensity4,
    /// 8-bit intensity (`GX_TF_I8`).
    Intensity8,
    /// 4-bit intensity and 4-bit alpha (`GX_TF_IA4`).
    IntensityAlpha4,
    /// 8-bit intensity and 8-bit alpha (`GX_TF_IA8`).
    IntensityAlpha8,
    /// 16-bit colour with an opaque alpha (`GX_TF_RGB565`).
    Rgb565,
    /// 4-bit red (`GX_CTF_R4`).
    Red4,
    /// 4-bit red and 4-bit alpha (`GX_CTF_RA4`).
    RedAlpha4,
    /// 8-bit red and 8-bit alpha (`GX_CTF_RA8`).
    RedAlpha8,
    /// 8-bit alpha (`GX_CTF_A8`).
    Alpha8,
    /// 8-bit red (`GX_CTF_R8`).
    Red8,
    /// 8-bit green (`GX_CTF_G8`).
    Green8,
    /// 8-bit blue (`GX_CTF_B8`).
    Blue8,
    /// 8-bit red and green (`GX_CTF_RG8`).
    RedGreen8,
    /// 8-bit green and blue (`GX_CTF_GB8`).
    GreenBlue8,
}

impl CopyFormat {
    /// The copy shader's conversion mode. The numbers are the shader's
    /// contract and must not be renumbered.
    pub const fn mode(self) -> u32 {
        match self {
            Self::Rgba8 => 0,
            Self::Intensity4 => 1,
            Self::Intensity8 => 2,
            Self::IntensityAlpha4 => 3,
            Self::IntensityAlpha8 => 4,
            Self::Rgb565 => 5,
            Self::Red4 => 6,
            Self::RedAlpha4 => 7,
            Self::RedAlpha8 => 8,
            Self::Alpha8 => 9,
            Self::Red8 => 10,
            Self::Green8 => 11,
            Self::Blue8 => 12,
            Self::RedGreen8 => 13,
            Self::GreenBlue8 => 14,
        }
    }
}

/// Region of a source target a copy reads, in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CopyRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl CopyRect {
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// The whole source target.
    pub const fn full(width: u32, height: u32) -> Self {
        Self::new(0, 0, width, height)
    }
}

/// What one [`PassKind::Copy`] pass copies.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CopySpec {
    /// Region of the source target read.
    pub rect: CopyRect,
    /// Pixel format written into the target.
    pub format: CopyFormat,
    /// The source had no alpha channel, so the copy writes an opaque alpha.
    pub opaque_alpha: bool,
    /// Texture-array page the copied target is published into, so a later
    /// draw can sample it as an ordinary scene texture. The target's size
    /// must fit the page, and its format must match the array's.
    pub publish_page: Option<u32>,
}

impl CopySpec {
    pub const fn new(rect: CopyRect, format: CopyFormat) -> Self {
        Self {
            rect,
            format,
            opaque_alpha: false,
            publish_page: None,
        }
    }

    /// The copy writes an opaque alpha, as the retail copy does when the
    /// source framebuffer format has no alpha channel.
    pub const fn with_opaque_alpha(mut self, opaque: bool) -> Self {
        self.opaque_alpha = opaque;
        self
    }

    /// Publishes the copied target into a texture-array page.
    pub const fn with_page(mut self, page: u32) -> Self {
        self.publish_page = Some(page);
        self
    }
}

/// One render target in a frame's pass list.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderTarget {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub format: RenderTargetFormat,
    pub semantics: TargetSemantics,
    /// Linear clear color, applied when [`PassKind::Render`] clears.
    pub clear: [f32; 4],
    /// Sample count for MSAA; 1 disables it.
    pub samples: u32,
}

impl RenderTarget {
    pub fn new(
        name: impl Into<String>,
        width: u32,
        height: u32,
        format: RenderTargetFormat,
        semantics: TargetSemantics,
    ) -> Self {
        Self {
            name: name.into(),
            width,
            height,
            format,
            semantics,
            clear: [0.0, 0.0, 0.0, 0.0],
            samples: 1,
        }
    }

    /// Target the size of the frame being rendered.
    pub fn sized_like_frame(name: impl Into<String>, width: u32, height: u32) -> Self {
        Self::color(name, width, height)
    }

    pub fn color(name: impl Into<String>, width: u32, height: u32) -> Self {
        Self::new(
            name,
            width,
            height,
            RenderTargetFormat::Rgba8Unorm,
            TargetSemantics::SceneColor,
        )
    }

    pub fn with_clear(mut self, clear: [f32; 4]) -> Self {
        self.clear = clear;
        self
    }

    pub fn with_samples(mut self, samples: u32) -> Self {
        self.samples = samples.max(1);
        self
    }

    pub fn byte_size(&self) -> usize {
        self.width as usize
            * self.height as usize
            * self.format.bytes_per_texel() as usize
            * self.samples as usize
    }
}

/// One entry in a frame's pass list.
#[derive(Clone, Debug)]
pub struct RenderPass {
    pub target: RenderTarget,
    pub kind: PassKind,
    /// Target sampled by [`PassKind::Resolve`], [`PassKind::Fullscreen`],
    /// [`PassKind::Deferred`] and [`PassKind::Copy`].
    pub source: Option<String>,
    /// Groups drawn by [`PassKind::Render`] and [`PassKind::Gbuffer`]; empty
    /// for the other kinds.
    pub groups: Vec<crate::MeshGroup>,
    /// Composites this pass's target over the scene when the frame is
    /// painted. Only the presenting pass reaches the screen.
    pub presents: bool,
    /// Tint and opacity applied by a presenting composite.
    pub overlay: [f32; 4],
    /// Screen-space effect a [`PassKind::Fullscreen`] pass runs. `None` keeps
    /// the plain tinted blit.
    pub post: Option<crate::postfx::PostStage>,
    /// Second target a post stage samples (`None` binds a dummy).
    pub aux: Option<String>,
    /// Draws over the target's contents instead of clearing them. A
    /// declared target is written by its own pass only, so this keeps the
    /// *previous frame's* contents (an accumulation pass); attachment ops
    /// over an earlier pass's target are the engine's, not a declaration's.
    /// Defaults to `false`, the clearing pass.
    pub load: bool,
    /// What a [`PassKind::Copy`] pass copies.
    pub copy: Option<CopySpec>,
    /// Tint and opacity a [`PassKind::Fullscreen`] pass multiplies into the
    /// source it samples. Defaults to identity.
    pub tint: [f32; 4],
    /// Targets whose writers this pass runs after, for a pass that consumes
    /// their contents without sampling them (a scene drawing a texture a
    /// copy published, for instance).
    pub depends_on: Vec<String>,
}

impl RenderPass {
    pub fn render(target: RenderTarget, groups: Vec<crate::MeshGroup>) -> Self {
        Self {
            target,
            kind: PassKind::Render,
            source: None,
            groups,
            presents: false,
            overlay: [1.0, 1.0, 1.0, 1.0],
            post: None,
            aux: None,
            load: false,
            copy: None,
            tint: [1.0, 1.0, 1.0, 1.0],
            depends_on: Vec::new(),
        }
    }

    /// Draw scene groups into a G-buffer target. The target must be
    /// `Rgba16Float`; the batch owns its two auxiliary attachments.
    pub fn gbuffer(target: RenderTarget, groups: Vec<crate::MeshGroup>) -> Self {
        Self {
            target,
            kind: PassKind::Gbuffer,
            source: None,
            groups,
            presents: false,
            overlay: [1.0, 1.0, 1.0, 1.0],
            post: None,
            aux: None,
            load: false,
            copy: None,
            tint: [1.0, 1.0, 1.0, 1.0],
            depends_on: Vec::new(),
        }
    }

    /// A deferred light-accumulation pass sampling a G-buffer `source`.
    pub fn deferred(source: impl Into<String>, target: RenderTarget) -> Self {
        Self {
            target,
            kind: PassKind::Deferred,
            source: Some(source.into()),
            groups: Vec::new(),
            presents: false,
            overlay: [1.0, 1.0, 1.0, 1.0],
            post: None,
            aux: None,
            load: false,
            copy: None,
            tint: [1.0, 1.0, 1.0, 1.0],
            depends_on: Vec::new(),
        }
    }

    /// A fullscreen pass sampling `source` into `target`.
    pub fn fullscreen(source: impl Into<String>, target: RenderTarget) -> Self {
        Self {
            target,
            kind: PassKind::Fullscreen,
            source: Some(source.into()),
            groups: Vec::new(),
            presents: false,
            overlay: [1.0, 1.0, 1.0, 1.0],
            post: None,
            aux: None,
            load: false,
            copy: None,
            tint: [1.0, 1.0, 1.0, 1.0],
            depends_on: Vec::new(),
        }
    }

    /// A copy pass: the region `spec` names of `source` is copied into
    /// `target`, scaled to the target's size and converted to the spec's
    /// pixel format.
    pub fn copy(source: impl Into<String>, target: RenderTarget, spec: CopySpec) -> Self {
        Self {
            target,
            kind: PassKind::Copy,
            source: Some(source.into()),
            groups: Vec::new(),
            presents: false,
            overlay: [1.0, 1.0, 1.0, 1.0],
            post: None,
            aux: None,
            load: false,
            copy: Some(spec),
            tint: [1.0, 1.0, 1.0, 1.0],
            depends_on: Vec::new(),
        }
    }

    /// Draws over the target's previous contents instead of clearing them
    /// (an accumulation pass; see [`RenderPass::load`]).
    pub fn loading(mut self) -> Self {
        self.load = true;
        self
    }

    /// Marks this pass as the one composited to the screen.
    pub fn presenting(mut self, overlay: [f32; 4]) -> Self {
        self.presents = true;
        self.overlay = overlay;
        self
    }

    /// Runs `stage` in this fullscreen pass.
    pub fn with_post(mut self, stage: crate::postfx::PostStage) -> Self {
        self.post = Some(stage);
        self
    }

    /// Multiplies the sampled source by `tint` in this fullscreen pass.
    pub fn with_tint(mut self, tint: [f32; 4]) -> Self {
        self.tint = tint;
        self
    }

    /// Runs this pass after the pass that writes `target`.
    pub fn after(mut self, target: impl Into<String>) -> Self {
        self.depends_on.push(target.into());
        self
    }
}

/// A mesh group bound to a declared pass by name.
#[derive(Clone, Debug)]
pub struct PassGroup {
    /// Name of the pass's target (the [`RenderTarget::name`] it declares).
    pub pass: String,
    pub group: crate::MeshGroup,
}

impl PassGroup {
    pub fn new(pass: impl Into<String>, group: crate::MeshGroup) -> Self {
        Self {
            pass: pass.into(),
            group,
        }
    }
}

/// Why a pass list was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassError {
    ZeroSized,
    SampleCountUnsupported,
    MissingSource,
    UnknownSource,
    /// A dependency names a target no pass declares.
    UnknownDependency,
    DuplicateTarget,
    TooManyTargets,
    SizeMismatch,
    /// A copy pass writes into a target the batch cannot render into as a
    /// colour attachment.
    InvalidCopyTarget,
    /// More than one pass claims the screen. Only the first is composited,
    /// so the rest render every frame and are never seen.
    MultiplePresenting,
    /// A deferred pass samples a source that is not a G-buffer pass.
    InvalidDeferredSource,
    /// A dependency cycle: the `source`/`aux` names and the depth-after-
    /// scene rules walk back to a pass that depends on itself.
    Circular,
}

/// Ceiling on the length of one frame's pass list. A bloom pyramid plus DOF,
/// motion blur and FXAA lands around a dozen, so the old value of 8 is raised
/// to leave the post chain headroom.
pub const MAX_PASS_TARGETS: usize = 16;

/// Validates a frame's pass list before it reaches the GPU.
pub fn validate_passes(passes: &[RenderPass]) -> Result<(), PassError> {
    if passes.len() > MAX_PASS_TARGETS {
        return Err(PassError::TooManyTargets);
    }
    // Phase one: everything that does not depend on another pass, and the
    // complete target list (the scheduler, not the declaration order, decides
    // what runs first, so a source may be declared later).
    let mut names: Vec<(&str, &RenderTarget)> = Vec::with_capacity(passes.len());
    let mut presenting = 0usize;
    for pass in passes {
        if pass.presents {
            presenting += 1;
            if presenting > 1 {
                return Err(PassError::MultiplePresenting);
            }
        }
        if pass.target.width == 0 || pass.target.height == 0 {
            return Err(PassError::ZeroSized);
        }
        if pass.target.samples != 1 {
            return Err(PassError::SampleCountUnsupported);
        }
        if names
            .iter()
            .any(|(name, _)| *name == pass.target.name.as_str())
        {
            return Err(PassError::DuplicateTarget);
        }
        if matches!(
            pass.kind,
            PassKind::Resolve | PassKind::Fullscreen | PassKind::Deferred | PassKind::Copy
        ) && pass.source.is_none()
        {
            return Err(PassError::MissingSource);
        }
        if pass.kind == PassKind::Copy {
            if pass.target.format.is_depth() || pass.copy.is_none() {
                return Err(PassError::InvalidCopyTarget);
            }
        }
        names.push((&pass.target.name, &pass.target));
    }
    // Phase two: the sampled targets exist, and a resolve matches its source.
    for pass in passes {
        if let Some(source) = pass.source.as_deref() {
            let (_, source_target) = names
                .iter()
                .find(|(name, _)| *name == source)
                .ok_or(PassError::UnknownSource)?;
            if pass.kind == PassKind::Resolve
                && (source_target.width != pass.target.width
                    || source_target.height != pass.target.height)
            {
                return Err(PassError::SizeMismatch);
            }
            if pass.kind == PassKind::Deferred
                && !passes
                    .iter()
                    .any(|other| other.target.name == source && other.kind == PassKind::Gbuffer)
            {
                return Err(PassError::InvalidDeferredSource);
            }
        }
        if let Some(aux) = pass.aux.as_deref()
            && !names.iter().any(|(name, _)| *name == aux)
        {
            return Err(PassError::UnknownSource);
        }
        for dependency in &pass.depends_on {
            if !names.iter().any(|(name, _)| *name == dependency) {
                return Err(PassError::UnknownDependency);
            }
        }
    }
    // A dependency cycle would leave passes unordered: reject the list
    // instead of scheduling what can be scheduled and guessing the rest.
    let mut placed = vec![false; passes.len()];
    let mut remaining = passes.len();
    while remaining > 0 {
        let mut progressed = false;
        for index in 0..passes.len() {
            if placed[index]
                || dependencies(passes, index)
                    .iter()
                    .any(|dep| !placed[*dep])
            {
                continue;
            }
            placed[index] = true;
            remaining -= 1;
            progressed = true;
        }
        if !progressed {
            return Err(PassError::Circular);
        }
    }
    Ok(())
}

/// Resolves the pass list into the order the batch will execute.
///
/// Stable: ties keep declaration order. The rules are:
///
/// - a pass that samples a named target (`source`, `aux`) runs after the
///   pass that writes it;
/// - a pass that names a target in `depends_on` runs after the pass that
///   writes it;
/// - a [`PassKind::Depth`] or [`PassKind::Occlusion`] pass runs after every
///   [`TargetSemantics::SceneColor`] pass, whether it is a [`PassKind::Render`]
///   or [`PassKind::Gbuffer`] pass, because it replays that scene's geometry
///   (and the engine's same-frame occlusion phase draws after it).
///
/// [`validate_passes`] rejects a list whose dependencies cannot be
/// satisfied, so a validated list always has a full order.
pub fn pass_order(passes: &[RenderPass]) -> Vec<usize> {
    let mut order = Vec::with_capacity(passes.len());
    let mut placed = vec![false; passes.len()];
    while order.len() < passes.len() {
        let mut progressed = false;
        for index in 0..passes.len() {
            if placed[index]
                || dependencies(passes, index)
                    .iter()
                    .any(|dep| !placed[*dep])
            {
                continue;
            }
            placed[index] = true;
            order.push(index);
            progressed = true;
        }
        if !progressed {
            // Unreachable for a validated list: keep the remainder in
            // declaration order rather than looping.
            order.extend((0..passes.len()).filter(|index| !placed[*index]));
            break;
        }
    }
    order
}

/// Indices `index` must run after: the writers of the targets it samples or
/// depends on, plus, for the depth replays, every scene-color render or
/// G-buffer pass.
fn dependencies(passes: &[RenderPass], index: usize) -> Vec<usize> {
    let pass = &passes[index];
    let mut deps = Vec::new();
    let named = [
        pass.source.as_deref(),
        pass.aux.as_deref(),
    ]
    .into_iter()
    .flatten()
    .chain(pass.depends_on.iter().map(|name| name.as_str()));
    for name in named {
        if let Some(from) = passes.iter().position(|other| other.target.name == name)
            && !deps.contains(&from)
        {
            deps.push(from);
        }
    }
    if matches!(pass.kind, PassKind::Depth | PassKind::Occlusion) {
        for (scene, other) in passes.iter().enumerate() {
            if matches!(other.kind, PassKind::Render | PassKind::Gbuffer)
                && other.target.semantics == TargetSemantics::SceneColor
                && scene != index
                && !deps.contains(&scene)
            {
                deps.push(scene);
            }
        }
    }
    deps
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene() -> RenderPass {
        RenderPass::render(RenderTarget::color("scene", 64, 64), Vec::new())
    }

    fn blit(source: &str, name: &str) -> RenderPass {
        RenderPass::fullscreen(
            source,
            RenderTarget::new(
                name,
                64,
                64,
                RenderTargetFormat::Rgba8UnormSrgb,
                TargetSemantics::Auxiliary,
            ),
        )
    }

    fn depth(name: &str) -> RenderPass {
        RenderPass {
            target: RenderTarget::new(
                name,
                64,
                64,
                RenderTargetFormat::Rgba16Float,
                TargetSemantics::LinearDepth,
            ),
            kind: PassKind::Depth,
            source: None,
            groups: Vec::new(),
            presents: false,
            overlay: [1.0; 4],
            post: None,
            aux: None,
            load: false,
            copy: None,
            tint: [1.0; 4],
            depends_on: Vec::new(),
        }
    }

    /// A pass sampling a later-declared target is scheduled after it, and a
    /// list without dependencies keeps declaration order.
    #[test]
    fn order_follows_sources_and_keeps_ties() {
        let passes = [blit("scene", "blit"), scene()];
        assert_eq!(pass_order(&passes), vec![1, 0]);
        assert!(validate_passes(&passes).is_ok());
        let passes = [scene(), blit("scene", "blit")];
        assert_eq!(pass_order(&passes), vec![0, 1]);
        // No dependencies: declaration order.
        let passes = [blit("a", "b"), blit("b", "c"), scene()];
        assert_eq!(pass_order(&passes), vec![0, 1, 2]);
        // A deferred pass must name the G-buffer it samples.
        let deferred = RenderPass::deferred("scene", RenderTarget::color("lit", 64, 64));
        assert_eq!(
            validate_passes(&[scene(), deferred.clone()]),
            Err(PassError::InvalidDeferredSource)
        );
        let gbuffer = RenderPass::gbuffer(RenderTarget::color("scene", 64, 64), Vec::new());
        assert!(validate_passes(&[gbuffer, deferred.clone()]).is_ok());
        let mut missing = deferred;
        missing.source = None;
        assert_eq!(
            validate_passes(&[scene(), missing]),
            Err(PassError::MissingSource)
        );
    }

    /// A depth replay runs after every scene-color render pass, whatever the
    /// declaration says: it replays their geometry.
    #[test]
    fn depth_replays_follow_scene_renders() {
        let passes = [depth("depth"), scene()];
        assert_eq!(pass_order(&passes), vec![1, 0]);
        let passes = [scene(), depth("depth")];
        assert_eq!(pass_order(&passes), vec![0, 1]);
        let gbuffer = RenderPass::gbuffer(RenderTarget::color("gbuffer", 64, 64), Vec::new());
        let passes = [depth("depth"), gbuffer];
        assert_eq!(pass_order(&passes), vec![1, 0]);
    }

    /// A dependency cycle is rejected rather than scheduled by guesswork.
    #[test]
    fn cycles_are_rejected() {
        let passes = [blit("b", "a"), blit("a", "b")];
        assert_eq!(validate_passes(&passes), Err(PassError::Circular));
        // Self-sampling is the shortest cycle.
        let passes = [blit("a", "a")];
        assert_eq!(validate_passes(&passes), Err(PassError::Circular));
    }
}
