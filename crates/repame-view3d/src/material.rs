//! Console-era material model: texture-environment stages, light objects,
//! constant colors and fog, described as plain data and lowered to WGSL.
//!
//! The combinator algebra mirrors the hardware the ports target: every
//! channel of a stage resolves three operands (`a`, `b`, `c`) through one
//! of 64 operations, combines them with an RGB/alpha mode, then scales and
//! offsets the result. A program is an ordered list of stages writing the
//! output register, so a game maps its own shading script onto [`TevProgram`]
//! without the engine hard-coding one material set.
//!
//! [`crate::tev::evaluate`] is the reference implementation used by games that
//! need CPU-side colors (previews, minimaps, baked lighting) and by tests;
//! [`crate::tev::wgsl_program`] emits the equivalent GPU path.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::mesh::Rgb;

/// Hardware register count for stage outputs.
pub const MAX_TEV_STAGES: usize = 16;
/// Constant color registers (kcolors).
pub const MAX_KCOLORS: usize = 4;
/// Texture units a program may sample.
pub const MAX_TEX_UNITS: usize = 8;
/// Light objects a material may reference.
pub const MAX_LIGHTS: usize = 8;
/// Texture coordinate sets a program may read.
pub const MAX_TEXCOORD_SETS: usize = 8;

/// One of the 64 ways three operands fold into an intermediate value.
///
/// Named for the hardware's operand encoding rather than the arithmetic, so a
/// port's register dump can be transcribed directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TevOp {
    A,
    OneMinusA,
    C,
    OneMinusC,
    APlusB,
    AMinusB,
    AMinusC,
    ATimesB,
    APlusBHalf,
    BPlusCHalf,
    BMinusCHalf,
}

impl TevOp {
    /// Hardware operand code (0..12) used by register dumps.
    pub const fn code(self) -> u8 {
        match self {
            Self::A => 0,
            Self::OneMinusA => 1,
            Self::C => 2,
            Self::OneMinusC => 3,
            Self::APlusB => 4,
            Self::AMinusB => 5,
            Self::AMinusC => 6,
            Self::ATimesB => 7,
            Self::APlusBHalf => 8,
            Self::BPlusCHalf => 9,
            Self::BMinusCHalf => 10,
        }
    }

    pub fn from_code(code: u8) -> Option<Self> {
        Some(match code {
            0 => Self::A,
            1 => Self::OneMinusA,
            2 => Self::C,
            3 => Self::OneMinusC,
            4 => Self::APlusB,
            5 => Self::AMinusB,
            6 => Self::AMinusC,
            7 => Self::ATimesB,
            8 => Self::APlusBHalf,
            9 => Self::BPlusCHalf,
            10 => Self::BMinusCHalf,
            _ => return None,
        })
    }
}

/// How the first fold combines with the second and the destination register.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TevMode {
    Replace,
    Modulate,
    Add,
    Subtract,
}

impl TevMode {
    pub const fn code(self) -> u8 {
        match self {
            Self::Replace => 0,
            Self::Modulate => 1,
            Self::Add => 2,
            Self::Subtract => 3,
        }
    }

    pub fn from_code(code: u8) -> Option<Self> {
        Some(match code {
            0 => Self::Replace,
            1 => Self::Modulate,
            2 => Self::Add,
            3 => Self::Subtract,
            _ => return None,
        })
    }
}

/// Second fold against the destination (`d`), as used by the hardware's
/// post-`c` combiner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TevOp2 {
    D,
    OneMinusD,
    APlusD,
    APlusOneMinusD,
    ATimesD,
    ATimesOneMinusD,
    APlusB,
    AMinusB,
    ATimesA,
}

impl TevOp2 {
    pub const fn code(self) -> u8 {
        match self {
            Self::D => 0,
            Self::OneMinusD => 1,
            Self::APlusD => 2,
            Self::APlusOneMinusD => 3,
            Self::ATimesD => 4,
            Self::ATimesOneMinusD => 5,
            Self::APlusB => 6,
            Self::AMinusB => 7,
            Self::ATimesA => 8,
        }
    }

    pub fn from_code(code: u8) -> Option<Self> {
        Some(match code {
            0 => Self::D,
            1 => Self::OneMinusD,
            2 => Self::APlusD,
            3 => Self::APlusOneMinusD,
            4 => Self::ATimesD,
            5 => Self::ATimesOneMinusD,
            6 => Self::APlusB,
            7 => Self::AMinusB,
            8 => Self::ATimesA,
            _ => return None,
        })
    }
}

/// A source value a stage operand can read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TevArg {
    Zero,
    One,
    Half,
    Two,
    /// Incoming color, premultiplied by the stage alpha.
    Color,
    /// Incoming alpha.
    Alpha,
    /// Constant color register.
    KColor(u8),
    /// Sampled texture color of the stage's unit.
    TexColor,
    /// Sampled texture alpha of the stage's unit.
    TexAlpha,
    /// Sampled texture LOD blend factor.
    LodFrac,
    /// `1.0` for color channels, `0.0` for alpha (the "stub" operand).
    Stub,
    /// Texture color of an earlier unit, for multi-tex chains.
    TexColorOf(u8),
}

/// Scale/offset pair applied to an argument before it folds, in quarters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TevScale {
    /// Multiplier in quarters: -4..4 maps to -1.0, -0.75, .. 1.0.
    pub scale: i8,
    /// Additive bias in quarters, 0..=7 mapping to 0.0, 0.25, .. 1.75.
    pub bias: u8,
}

impl Default for TevScale {
    fn default() -> Self {
        Self { scale: 4, bias: 0 }
    }
}

impl TevScale {
    pub const IDENTITY: Self = Self { scale: 4, bias: 0 };

    /// Hardware encoding: signed scale in quarters, bias clamped to 7/4.
    pub fn new(scale: i8, bias: u8) -> Self {
        Self {
            scale: scale.clamp(-4, 4),
            bias: bias.min(7),
        }
    }

    pub fn multiplier(self) -> f32 {
        f32::from(self.scale) * 0.25
    }

    pub fn offset(self) -> f32 {
        f32::from(self.bias) * 0.25
    }
}

/// One channel of one operand slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TevOperand {
    pub arg: TevArg,
    pub negate: bool,
    pub scale: TevScale,
}

impl TevOperand {
    pub const fn new(arg: TevArg) -> Self {
        Self {
            arg,
            negate: false,
            scale: TevScale::IDENTITY,
        }
    }

    pub const fn rgb(arg: TevArg, scale: i8, bias: u8) -> Self {
        Self {
            arg,
            negate: false,
            scale: TevScale { scale, bias },
        }
    }

    pub const fn alpha(arg: TevArg, scale: i8, bias: u8) -> Self {
        Self {
            arg,
            negate: false,
            scale: TevScale { scale, bias },
        }
    }
}

impl Default for TevOperand {
    fn default() -> Self {
        Self::new(TevArg::Color)
    }
}

/// A texture-environment stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TevStage {
    /// Texture unit this stage samples.
    pub tex_unit: u8,
    /// Pre-sampled color/alpha written by the texture stage.
    pub color_arg: [TevOperand; 4],
    pub alpha_arg: [TevOperand; 3],
    pub color_op: [TevOp; 4],
    pub alpha_op: [TevOp; 3],
    pub op2: [TevOp2; 4],
    pub color_mode: TevMode,
    pub alpha_mode: TevMode,
    /// Scale/offset applied to the incoming color, per channel.
    pub color_scale: TevScale,
    pub alpha_scale: TevScale,
}

impl Default for TevStage {
    fn default() -> Self {
        Self {
            tex_unit: 0,
            color_arg: [TevOperand::new(TevArg::Color); 4],
            alpha_arg: [TevOperand::new(TevArg::Color); 3],
            color_op: [TevOp::APlusB; 4],
            alpha_op: [TevOp::APlusB; 3],
            op2: [TevOp2::D; 4],
            color_mode: TevMode::Modulate,
            alpha_mode: TevMode::Modulate,
            color_scale: TevScale::IDENTITY,
            alpha_scale: TevScale::IDENTITY,
        }
    }
}

/// Per-material constant color registers, RGBA each.
pub type KColors = [Rgba; MAX_KCOLORS];

/// Linear RGBA quad.
pub type Rgba = [f32; 4];

/// Identity kcolors: opaque black, matching a cleared material state.
pub const DEFAULT_KCOLORS: KColors = [
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// One hardware light object.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Light {
    /// Two color indices into [`KColors`], or None for the material ambient.
    pub colors: [Option<u8>; 2],
    pub direction: [f32; 3],
    pub position: [f32; 3],
    /// Cone angles: inner, outer, in cosine.
    pub angles: [f32; 3],
    /// Distance attenuation enabled and its range.
    pub distance: [f32; 2],
}

impl Default for Light {
    fn default() -> Self {
        Self {
            colors: [None, None],
            direction: [0.0, -1.0, 0.0],
            position: [0.0, 0.0, 0.0],
            angles: [-1.0, -1.0, 0.0],
            distance: [0.0, 0.0],
        }
    }
}

impl Light {
    /// A directional light with an explicit color register pair.
    pub fn directional(colors: [u8; 2], direction: [f32; 3]) -> Self {
        Self {
            colors: [Some(colors[0]), Some(colors[1])],
            direction,
            ..Self::default()
        }
    }

    /// A point light with a distance range.
    pub fn point(colors: [u8; 2], position: [f32; 3], range: f32) -> Self {
        Self {
            colors: [Some(colors[0]), Some(colors[1])],
            position,
            distance: [1.0, range],
            ..Self::default()
        }
    }
}

/// Material fog: start/end distance and a color register.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialFog {
    pub near: f32,
    pub far: f32,
    pub color: Rgb,
    pub enabled: bool,
}

impl Default for MaterialFog {
    fn default() -> Self {
        Self {
            near: 1.0,
            far: 100.0,
            color: [0.0, 0.0, 0.0],
            enabled: false,
        }
    }
}

/// A material as the shading system describes it: a stage program, its
/// constant registers, up to eight lights and optional material fog.
#[derive(Clone, Debug, PartialEq)]
pub struct ShadingModel {
    pub stages: Vec<TevStage>,
    pub kcolors: KColors,
    pub lights: Vec<Light>,
    pub ambient: u8,
    pub material_fog: MaterialFog,
    /// Ambient occlusion factor applied to indirect light.
    pub ao: f32,
}

impl Default for ShadingModel {
    fn default() -> Self {
        Self {
            stages: vec![TevStage::default()],
            kcolors: DEFAULT_KCOLORS,
            lights: Vec::new(),
            ambient: 0,
            material_fog: MaterialFog::default(),
            ao: 1.0,
        }
    }
}

impl ShadingModel {
    /// A single-stage modulate-texture model, the common opaque surface.
    pub fn textured(unit: u8) -> Self {
        let mut stage = TevStage {
            tex_unit: unit,
            ..TevStage::default()
        };
        stage.color_arg[0] = TevOperand::new(TevArg::Color);
        stage.color_arg[1] = TevOperand::new(TevArg::TexColor);
        stage.color_arg[2] = TevOperand::new(TevArg::One);
        stage.alpha_arg[0] = TevOperand::new(TevArg::Color);
        stage.alpha_arg[1] = TevOperand::new(TevArg::TexAlpha);
        stage.alpha_arg[2] = TevOperand::new(TevArg::One);
        stage.color_op = [TevOp::ATimesB; 4];
        stage.alpha_op = [TevOp::ATimesB; 3];
        stage.color_mode = TevMode::Replace;
        stage.alpha_mode = TevMode::Replace;
        Self {
            stages: vec![stage],
            ..Self::default()
        }
    }

    /// Fails when the program exceeds hardware limits or names bad registers.
    pub fn validate(&self) -> Result<(), ShadingError> {
        if self.stages.is_empty() {
            return Err(ShadingError::NoStages);
        }
        if self.stages.len() > MAX_TEV_STAGES {
            return Err(ShadingError::TooManyStages);
        }
        if self.lights.len() > MAX_LIGHTS {
            return Err(ShadingError::TooManyLights);
        }
        for stage in &self.stages {
            if usize::from(stage.tex_unit) >= MAX_TEX_UNITS {
                return Err(ShadingError::TexUnitOutOfRange(stage.tex_unit));
            }
        }
        let kc = |i: u8| usize::from(i) < MAX_KCOLORS;
        for light in &self.lights {
            for color in light.colors.iter().flatten() {
                if !kc(*color) {
                    return Err(ShadingError::KColorOutOfRange(*color));
                }
            }
        }
        if !kc(self.ambient) {
            return Err(ShadingError::KColorOutOfRange(self.ambient));
        }
        Ok(())
    }

    /// Stable identity of the generated shader, for pipeline caching.
    pub fn cache_key(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.stages.hash(&mut hasher);
        self.lights.len().hash(&mut hasher);
        self.ambient.hash(&mut hasher);
        self.material_fog.enabled.hash(&mut hasher);
        self.material_fog.near.to_bits().hash(&mut hasher);
        self.material_fog.far.to_bits().hash(&mut hasher);
        self.material_fog.color.map(f32::to_bits).hash(&mut hasher);
        hasher.finish()
    }
}

/// Rejection reasons for a [`ShadingModel`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShadingError {
    NoStages,
    TooManyStages,
    TooManyLights,
    TexUnitOutOfRange(u8),
    KColorOutOfRange(u8),
}

impl std::fmt::Display for ShadingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoStages => write!(f, "shading model has no stages"),
            Self::TooManyStages => write!(f, "shading model exceeds {MAX_TEV_STAGES} stages"),
            Self::TooManyLights => write!(f, "shading model exceeds {MAX_LIGHTS} lights"),
            Self::TexUnitOutOfRange(unit) => write!(f, "texture unit {unit} out of range"),
            Self::KColorOutOfRange(index) => write!(f, "constant color {index} out of range"),
        }
    }
}

impl std::error::Error for ShadingError {}
