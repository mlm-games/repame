//! Console-era material model: texture-environment stages, light objects,
//! constant colors and fog, described as plain data and lowered to WGSL.
//!
//! A stage is one combiner equation per channel: four arguments fold through
//! the texture-environment interpolation `reg = clamp(scale * (d + (1 - c) *
//! a + c * b + bias))`, and the channel writes one of four output registers.
//! [`TevMode`] expands to the five programs the console SDK's `GX_SetTevOp`
//! names, so the common texture cases stay one call while a port can still
//! declare any equation the hardware runs. A program is an ordered list of
//! stages, so a game maps its own shading script onto [`TevStage`] without
//! the engine hard-coding one material set.
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

/// Combiner operation: the two arithmetic codes the hardware models.
///
/// Code 0 adds the interpolated term, code 1 subtracts it. The compare codes
/// 8..15 exist in hardware but retail games do not rely on them, so they are
/// not modeled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TevOp {
    Add,
    Sub,
}

impl TevOp {
    pub const fn code(self) -> u8 {
        match self {
            Self::Add => 0,
            Self::Sub => 1,
        }
    }

    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Add),
            1 => Some(Self::Sub),
            _ => None,
        }
    }
}

/// Bias added after the interpolation, in halves of the channel range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TevBias {
    Zero,
    AddHalf,
    SubHalf,
}

impl TevBias {
    pub const fn code(self) -> u8 {
        match self {
            Self::Zero => 0,
            Self::AddHalf => 1,
            Self::SubHalf => 2,
        }
    }

    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Zero),
            1 => Some(Self::AddHalf),
            2 => Some(Self::SubHalf),
            _ => None,
        }
    }

    pub const fn offset(self) -> f32 {
        match self {
            Self::Zero => 0.0,
            Self::AddHalf => 0.5,
            Self::SubHalf => -0.5,
        }
    }
}

/// Output scale of a stage, as the hardware encodes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TevScale {
    X1,
    X2,
    X4,
    Divide2,
}

impl TevScale {
    pub const fn code(self) -> u8 {
        match self {
            Self::X1 => 0,
            Self::X2 => 1,
            Self::X4 => 2,
            Self::Divide2 => 3,
        }
    }

    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::X1),
            1 => Some(Self::X2),
            2 => Some(Self::X4),
            3 => Some(Self::Divide2),
            _ => None,
        }
    }

    pub const fn multiplier(self) -> f32 {
        match self {
            Self::X1 => 1.0,
            Self::X2 => 2.0,
            Self::X4 => 4.0,
            Self::Divide2 => 0.5,
        }
    }
}

/// Register a stage's channel writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TevDest {
    /// The previous-stage register every later stage reads.
    Prev,
    Reg0,
    Reg1,
    Reg2,
}

impl TevDest {
    pub const fn code(self) -> u8 {
        match self {
            Self::Prev => 0,
            Self::Reg0 => 1,
            Self::Reg1 => 2,
            Self::Reg2 => 3,
        }
    }

    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Prev),
            1 => Some(Self::Reg0),
            2 => Some(Self::Reg1),
            3 => Some(Self::Reg2),
            _ => None,
        }
    }

    /// Register slot index the evaluators write.
    pub const fn index(self) -> usize {
        self.code() as usize
    }
}

/// The five programs `GX_SetTevOp` names, as the hardware encodes them.
///
/// Each expands through [`TevStage::with_mode`] into the equation below,
/// with `C` the previous register, `A` its alpha and `T` the texel:
///
/// - `Modulate`: `C * T` and `A * At`
/// - `Decal`: `(1 - At) * C + At * T` and `A`
/// - `Blend`: `(1 - T) * C + T` and `A * At`
/// - `Replace`: `T` and `At`
/// - `PassClr`: `C` and `A`
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TevMode {
    Modulate,
    Decal,
    Blend,
    Replace,
    PassClr,
}

impl TevMode {
    pub const fn code(self) -> u8 {
        match self {
            Self::Modulate => 0,
            Self::Decal => 1,
            Self::Blend => 2,
            Self::Replace => 3,
            Self::PassClr => 4,
        }
    }

    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Modulate),
            1 => Some(Self::Decal),
            2 => Some(Self::Blend),
            3 => Some(Self::Replace),
            4 => Some(Self::PassClr),
            _ => None,
        }
    }
}

/// A source value a stage argument can read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TevArg {
    Zero,
    One,
    Half,
    Two,
    /// Previous register: its color in color channels, alpha in the alpha one.
    Color,
    /// Previous register alpha.
    Alpha,
    /// Output register 0, in the channel being computed.
    Reg0,
    /// Output register 1, in the channel being computed.
    Reg1,
    /// Output register 2, in the channel being computed.
    Reg2,
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

/// One texture-environment stage: the combiner equation and where it writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TevStage {
    /// Texture unit this stage samples.
    pub tex_unit: u8,
    /// Texture array layer that unit samples.
    pub page: u8,
    /// Color-channel arguments `a`, `b`, `c`, `d`.
    pub color_arg: [TevArg; 4],
    /// Alpha-channel arguments `a`, `b`, `c`, `d`.
    pub alpha_arg: [TevArg; 4],
    pub color_op: TevOp,
    pub alpha_op: TevOp,
    pub color_bias: TevBias,
    pub alpha_bias: TevBias,
    pub color_scale: TevScale,
    pub alpha_scale: TevScale,
    /// Clamp the channel into `0..1`, or else into the 10-bit range below.
    pub color_clamp: bool,
    pub alpha_clamp: bool,
    pub color_dest: TevDest,
    pub alpha_dest: TevDest,
}

impl Default for TevStage {
    fn default() -> Self {
        Self {
            tex_unit: 0,
            page: 0,
            color_arg: [TevArg::Zero, TevArg::Zero, TevArg::Zero, TevArg::Color],
            alpha_arg: [TevArg::Zero, TevArg::Zero, TevArg::Zero, TevArg::Alpha],
            color_op: TevOp::Add,
            alpha_op: TevOp::Add,
            color_bias: TevBias::Zero,
            alpha_bias: TevBias::Zero,
            color_scale: TevScale::X1,
            alpha_scale: TevScale::X1,
            color_clamp: true,
            alpha_clamp: true,
            color_dest: TevDest::Prev,
            alpha_dest: TevDest::Prev,
        }
    }
}

impl TevStage {
    /// The stage sampling `tex_unit` under one of the SDK's [`TevMode`] programs.
    ///
    /// Default bias, scale, clamping and destination are left in place; only
    /// the argument selects differ per program (see [`TevMode`]).
    pub fn with_mode(tex_unit: u8, mode: TevMode) -> Self {
        let mut stage = Self {
            tex_unit,
            ..Self::default()
        };
        match mode {
            TevMode::Modulate => {
                stage.color_arg = [TevArg::Zero, TevArg::Color, TevArg::TexColor, TevArg::Zero];
                stage.alpha_arg = [TevArg::Zero, TevArg::Alpha, TevArg::TexAlpha, TevArg::Zero];
            }
            TevMode::Decal => {
                stage.color_arg = [
                    TevArg::Color,
                    TevArg::TexColor,
                    TevArg::TexAlpha,
                    TevArg::Zero,
                ];
                stage.alpha_arg = [TevArg::Zero, TevArg::Zero, TevArg::Zero, TevArg::Alpha];
            }
            TevMode::Blend => {
                stage.color_arg = [TevArg::Color, TevArg::One, TevArg::TexColor, TevArg::Zero];
                stage.alpha_arg = [TevArg::Zero, TevArg::Alpha, TevArg::TexAlpha, TevArg::Zero];
            }
            TevMode::Replace => {
                stage.color_arg = [TevArg::Zero, TevArg::Zero, TevArg::Zero, TevArg::TexColor];
                stage.alpha_arg = [TevArg::Zero, TevArg::Zero, TevArg::Zero, TevArg::TexAlpha];
            }
            TevMode::PassClr => {}
        }
        stage
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

/// One light the shading evaluates: direction or position with its kcolor pair.
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

/// Material fog: start/end distance and its blend color.
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
        Self {
            stages: vec![TevStage::with_mode(unit, TevMode::Modulate)],
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
        let kc = |i: u8| usize::from(i) < MAX_KCOLORS;
        let mut sampled = [false; MAX_TEX_UNITS];
        for stage in &self.stages {
            if usize::from(stage.tex_unit) >= MAX_TEX_UNITS {
                return Err(ShadingError::TexUnitOutOfRange(stage.tex_unit));
            }
            sampled[usize::from(stage.tex_unit)] = true;
        }
        let check = |arg: TevArg, alpha: bool| -> Result<(), ShadingError> {
            match arg {
                TevArg::KColor(index) if !kc(index) => {
                    Err(ShadingError::KColorOutOfRange(index))
                }
                TevArg::TexColorOf(unit) if usize::from(unit) >= MAX_TEX_UNITS => {
                    Err(ShadingError::TexUnitOutOfRange(unit))
                }
                // The GPU binds one page per unit from the stages that sample
                // it, and the lowering only declares `texN` for those units.
                // An unsampled unit would emit a read of a binding that does
                // not exist, so the shader fails to compile at pipeline
                // creation rather than rendering something plausible.
                TevArg::TexColorOf(unit) if !sampled[usize::from(unit)] => {
                    Err(ShadingError::TexUnitNotSampled(unit))
                }
                TevArg::TexColor | TevArg::TexColorOf(_) if alpha => {
                    Err(ShadingError::AlphaArgTakesColor)
                }
                _ => Ok(()),
            }
        };
        for stage in &self.stages {
            for arg in stage.color_arg {
                check(arg, false)?;
            }
            for arg in stage.alpha_arg {
                check(arg, true)?;
            }
        }
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

    /// Stable identity of a registered material.
    ///
    /// A slot owns both the lowered shader *and* the constant uniform written
    /// for it, so every field that reaches either must be in here. Hashing
    /// only `lights.len()` and the fog window made two materials that differ
    /// in colour collapse onto one slot, and the second one then rendered
    /// with the first one's `kcolors` and `ao`.
    pub fn cache_key(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.stages.hash(&mut hasher);
        for light in &self.lights {
            light.colors.hash(&mut hasher);
            light.direction.map(f32::to_bits).hash(&mut hasher);
            light.position.map(f32::to_bits).hash(&mut hasher);
            light.angles.map(f32::to_bits).hash(&mut hasher);
            light.distance.map(f32::to_bits).hash(&mut hasher);
        }
        self.ambient.hash(&mut hasher);
        self.ao.to_bits().hash(&mut hasher);
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
    /// An argument reads a unit no stage samples, so no page is bound for it.
    TexUnitNotSampled(u8),
    KColorOutOfRange(u8),
    /// Texture color argument used in an alpha channel; alpha reads texel alpha.
    AlphaArgTakesColor,
}

impl std::fmt::Display for ShadingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoStages => write!(f, "shading model has no stages"),
            Self::TooManyStages => write!(f, "shading model exceeds {MAX_TEV_STAGES} stages"),
            Self::TooManyLights => write!(f, "shading model exceeds {MAX_LIGHTS} lights"),
            Self::TexUnitOutOfRange(unit) => write!(f, "texture unit {unit} out of range"),
            Self::TexUnitNotSampled(unit) => write!(
                f,
                "texture unit {unit} is read by an argument but sampled by no stage"
            ),
            Self::KColorOutOfRange(index) => write!(f, "constant color {index} out of range"),
            Self::AlphaArgTakesColor => write!(
                f,
                "alpha channel cannot read texture color; use TevArg::TexAlpha"
            ),
        }
    }
}

impl std::error::Error for ShadingError {}
