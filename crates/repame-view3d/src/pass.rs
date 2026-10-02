//! Render targets and pass description for the offscreen path.
//!
//! The viewport currently paints one depth-tested scene into the UI pass.
//! Games that need water refraction, a reflection buffer, a post chain or a
//! cutscene composite describe the extra work as [`RenderTarget`]s and hand
//! [`Frame3d::passes`](crate::Frame3d::passes) to the batch, which runs them in
//! order and hands each resolved result back as a plain RGBA8 image the next
//! pass (or the game) can sample.

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
        }
    }

    /// Marks this pass as the one composited to the screen.
    pub fn presenting(mut self, overlay: [f32; 4]) -> Self {
        self.presents = true;
        self.overlay = overlay;
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
}

/// Hardware ceiling on simultaneously bound targets in one pass list.
pub const MAX_PASS_TARGETS: usize = 8;

/// Validates a frame's pass list before it reaches the GPU.
pub fn validate_passes(passes: &[RenderPass]) -> Result<(), PassError> {
    if passes.len() > MAX_PASS_TARGETS {
        return Err(PassError::TooManyTargets);
    }
    let mut names: Vec<(&str, &RenderTarget)> = Vec::with_capacity(passes.len());
    for pass in passes {
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
            PassKind::Render => {}
            PassKind::Resolve | PassKind::Fullscreen => {
                let source = pass.source.as_deref().ok_or(PassError::MissingSource)?;
                let (_, source) = names
                    .iter()
                    .find(|(name, _)| *name == source)
                    .ok_or(PassError::UnknownSource)?;
                if pass.kind == PassKind::Resolve
                    && (source.width != pass.target.width || source.height != pass.target.height)
                {
                    return Err(PassError::SizeMismatch);
                }
            }
        }
        names.push((&pass.target.name, &pass.target));
    }
    Ok(())
}

/// Resolves the pass list into the order the batch will execute.
pub fn pass_order(passes: &[RenderPass]) -> Vec<usize> {
    (0..passes.len()).collect()
}
