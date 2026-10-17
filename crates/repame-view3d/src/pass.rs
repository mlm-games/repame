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

/// Pixel format of a render target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RenderTargetFormat {
    Rgba8Unorm,
    /// Matches the compositor's scene format, so a pass target can be
    /// sampled and drawn with the batch's own pipelines.
    Rgba8UnormSrgb,
    Bgra8Unorm,
    Rgba16Float,
    R32Float,
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
    /// Fill the target by replaying the frame's scene geometry as view-depth
    /// and camera velocity (`RGBA16F`: depth, velocity x, velocity y, 1). The
    /// batch owns the geometry, so this pass takes no `groups`.
    Depth,
    /// The engine's own view-depth replay for the occlusion feed, identical
    /// in target requirements and content to [`PassKind::Depth`]. The
    /// viewport appends one when a frame arms
    /// [`crate::Frame3d::occlusion_feed`] and declares no depth pass, so
    /// occlusion culling never depends on the post chain; games do not
    /// declare it themselves.
    Occlusion,
    /// Resolve the source target into a sampled texture.
    Resolve,
    /// Fullscreen blit or post chain over the source.
    Fullscreen,
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
    /// Target sampled by [`PassKind::Resolve`] and [`PassKind::Fullscreen`].
    pub source: Option<String>,
    /// Groups drawn by [`PassKind::Render`]; empty for the other kinds.
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
        }
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
}

/// Why a pass list was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassError {
    ZeroSized,
    SampleCountUnsupported,
    MissingSource,
    UnknownSource,
    DuplicateTarget,
    TooManyTargets,
    SizeMismatch,
    /// More than one pass claims the screen. Only the first is composited,
    /// so the rest render every frame and are never seen.
    MultiplePresenting,
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
        match pass.kind {
            PassKind::Render | PassKind::Depth | PassKind::Occlusion => {}
            PassKind::Resolve | PassKind::Fullscreen => {
                let source = pass.source.as_deref().ok_or(PassError::MissingSource)?;
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
            }
        }
        if let Some(aux) = pass.aux.as_deref()
            && !names.iter().any(|(name, _)| *name == aux)
        {
            return Err(PassError::UnknownSource);
        }
        names.push((&pass.target.name, &pass.target));
    }
    Ok(())
}

/// Resolves the pass list into the order the batch will execute.
pub fn pass_order(passes: &[RenderPass]) -> Vec<usize> {
    (0..passes.len()).collect()
}
