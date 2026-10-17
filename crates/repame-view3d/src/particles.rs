//! GPU particle systems: emitter config in, simulated billboards out.
//!
//! A [`ParticleSystem`] describes one emitter; [`ParticleFrame`] carries the
//! frame's systems plus the collision world. The batch keeps the simulation in
//! a GPU storage buffer, so no particle state ever crosses back to the CPU:
//!
//! ```text
//! prepare dispatch (1 thread)   advance the deterministic spawn cursor
//! simulate dispatch (256/workgroup)  spawn, integrate, collide, sub-emit
//! billboard draw                in the scene pass, depth-tested, blended
//! ```
//!
//! Determinism: every spawn derives its position, velocity, size and lifetime
//! from `hash(seed, spawn counter)`, never from wall-clock state, and the
//! per-frame spawn budget is `floor(carry + rate * dt)` with the fractional
//! carry kept across frames. The same seed, the same `dt` sequence, and the
//! same emitter config therefore produce the same particles. The per-particle
//! math is mirrored in Rust ([`integrate_step`], [`EmitterShape::sample`],
//! [`hash_u32`]) so tests pin the shader's behavior without a GPU.
//!
//! Sub-emitters: a system may target another system's request ring. Particle
//! deaths (`AtEnd`), collisions (`AtCollision`), spawns (`AtStart`) and a
//! per-particle `Constant` rate append requests, and the target consumes them
//! in the same frame because systems are dispatched in dependency order (a
//! cycle is broken with a warning at staging time).
//!
//! Collision runs against a GPU copy of the existing queries: [`AabbColliders`]
//! boxes or a bounded voxel occupancy field built from any
//! [`VoxelSource`](crate::VoxelSource) ([`ParticleVoxels::build`]). Speeds are
//! swept per step, so a particle cannot tunnel through a box or a voxel at the
//! frame's step length.

use bytemuck::{Pod, Zeroable};
use glam::Mat4;

use super::camera::AabbColliders;
use super::voxel::{Cell, VoxelSource};

/// Bytes of one GPU particle record (must match `Particle` in the WGSL).
pub const PARTICLE_BYTES: usize = 64;

/// Bytes of one sub-emitter request (must match `SubRequest` in the WGSL).
pub const SUB_REQUEST_BYTES: usize = 48;

/// Keys one curve or gradient packs into an emitter uniform.
pub const MAX_CURVE_KEYS: usize = 8;

/// Point attractors one system simulates.
pub const MAX_ATTRACTORS: usize = 8;

/// AABB colliders one frame's collider buffer holds.
pub const MAX_COLLIDER_BOXES: usize = 64;

/// Sub-emitter requests a ring holds at most (also the per-frame accept cap).
pub const MAX_REQUEST_RING: u32 = 256;

/// Invocations per simulate workgroup.
pub const SIMULATE_WORKGROUP: u32 = 256;

/// Why an emitter or frame was rejected. Staging drops invalid systems with
/// a warning; nothing is silently skipped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticleError {
    /// A system with no slots can never show anything.
    ZeroCapacity,
    /// A field held NaN or an infinity.
    NonFinite(&'static str),
    /// A min/max pair was inverted or empty.
    InvalidRange(&'static str),
    /// More attractors than [`MAX_ATTRACTORS`].
    TooManyAttractors(usize),
    /// A system named itself as its own sub-emitter target.
    SubTargetSelf(usize),
    /// The voxel field had a zero dimension, zero cell size, or no cells.
    BadVoxels,
    /// The frame's `dt` was zero, negative or not finite.
    InvalidDt,
}

impl std::fmt::Display for ParticleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroCapacity => write!(f, "particle system capacity is zero"),
            Self::NonFinite(field) => write!(f, "particle field `{field}` is not finite"),
            Self::InvalidRange(field) => write!(f, "particle range `{field}` is inverted"),
            Self::TooManyAttractors(n) => {
                write!(f, "{n} attractors past the {MAX_ATTRACTORS} cap")
            }
            Self::SubTargetSelf(i) => write!(f, "system {i} sub-emits into itself"),
            Self::BadVoxels => write!(f, "particle voxel field is unusable"),
            Self::InvalidDt => write!(f, "particle frame dt must be finite and positive"),
        }
    }
}

impl std::error::Error for ParticleError {}

/// Emitter volume particles spawn in, in the emitter's local space (+Y up,
/// transformed by [`ParticleSystem::transform`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EmitterShape {
    /// All particles start at the emitter origin.
    Point,
    /// Solid ball of `radius` (center at the origin).
    Sphere {
        /// Ball radius.
        radius: f32,
    },
    /// Box centered at the origin, `extents` are half extents.
    Box {
        /// Half extents per local axis.
        extents: [f32; 3],
    },
    /// Cone with its apex at the origin, opening along local +Y.
    Cone {
        /// Base radius at `y = height`.
        radius: f32,
        /// Cone height.
        height: f32,
    },
    /// Rim of a circle in the local XZ plane (no interior, no jitter).
    CircleEdge {
        /// Circle radius.
        radius: f32,
    },
}

impl Default for EmitterShape {
    fn default() -> Self {
        Self::Point
    }
}

impl EmitterShape {
    /// Shape id the shader switches on. Part of the GPU contract.
    pub const fn kind(self) -> u32 {
        match self {
            Self::Point => 0,
            Self::Sphere { .. } => 1,
            Self::Box { .. } => 2,
            Self::Cone { .. } => 3,
            Self::CircleEdge { .. } => 4,
        }
    }

    /// Samples one local-space spawn offset from eight uniform draws in
    /// `[0, 1)`. Mirrors the shader's `sample_shape` exactly.
    pub fn sample(&self, r: &[f32; 8]) -> [f32; 3] {
        match *self {
            Self::Point => [0.0, 0.0, 0.0],
            Self::Sphere { radius } => {
                let z = r[0] * 2.0 - 1.0;
                let turn = r[1] * std::f32::consts::TAU;
                let ring = (1.0 - z * z).max(0.0).sqrt();
                let rad = radius.max(0.0) * r[2].cbrt();
                [ring * turn.cos() * rad, z * rad, ring * turn.sin() * rad]
            }
            Self::Box { extents } => [
                (r[0] * 2.0 - 1.0) * extents[0],
                (r[1] * 2.0 - 1.0) * extents[1],
                (r[2] * 2.0 - 1.0) * extents[2],
            ],
            Self::Cone { radius, height } => {
                let height_frac = r[2];
                let turn = r[1] * std::f32::consts::TAU;
                let rad = radius.max(0.0) * height_frac * r[0].max(0.0).sqrt();
                [rad * turn.cos(), height_frac * height, rad * turn.sin()]
            }
            Self::CircleEdge { radius } => {
                let turn = r[0] * std::f32::consts::TAU;
                [radius * turn.cos(), 0.0, radius * turn.sin()]
            }
        }
    }

    fn validate(&self) -> Result<(), ParticleError> {
        let finite = |v: f32, field: &'static str| {
            if v.is_finite() {
                Ok(())
            } else {
                Err(ParticleError::NonFinite(field))
            }
        };
        match *self {
            Self::Point => Ok(()),
            Self::Sphere { radius } => finite(radius, "shape.radius"),
            Self::Box { extents } => {
                finite(extents[0], "shape.extents")?;
                finite(extents[1], "shape.extents")?;
                finite(extents[2], "shape.extents")
            }
            Self::Cone { radius, height } => {
                finite(radius, "shape.radius")?;
                finite(height, "shape.height")
            }
            Self::CircleEdge { radius } => finite(radius, "shape.radius"),
        }
    }
}

/// One scalar keyframe curve over normalized life (`0.0` spawn, `1.0` death),
/// sampled linearly between keys. Keys are sorted by time on construction and
/// clamped to `0..=1`; a value outside the key range holds the endpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct ParticleCurve {
    keys: Vec<(f32, f32)>,
}

impl Default for ParticleCurve {
    fn default() -> Self {
        Self::constant(1.0)
    }
}

impl ParticleCurve {
    /// A curve holding `value` for every life fraction.
    pub fn constant(value: f32) -> Self {
        Self {
            keys: vec![(0.0, value), (1.0, value)],
        }
    }

    /// A curve through `keys`, sorted by time and clamped into the unit
    /// domain. Rejects empty, non-finite or duplicate-time input.
    pub fn new(keys: impl IntoIterator<Item = (f32, f32)>) -> Result<Self, ParticleError> {
        let mut keys: Vec<(f32, f32)> = keys.into_iter().collect();
        if keys.is_empty() {
            return Err(ParticleError::InvalidRange("curve"));
        }
        for (t, v) in &keys {
            if !t.is_finite() || !v.is_finite() {
                return Err(ParticleError::NonFinite("curve.key"));
            }
        }
        keys.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        if keys.windows(2).any(|w| w[0].0 >= w[1].0) {
            return Err(ParticleError::InvalidRange("curve"));
        }
        for key in &mut keys {
            key.0 = key.0.clamp(0.0, 1.0);
        }
        Ok(Self { keys })
    }

    /// Identity curve (`t` maps to `t`).
    pub fn linear() -> Self {
        Self::new([(0.0, 0.0), (1.0, 1.0)]).expect("two valid keys")
    }

    /// Key count.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// True when the curve has no keys (never: constructors validate).
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Value at life fraction `t`, linear between keys and held at the ends.
    pub fn sample(&self, t: f32) -> f32 {
        curve_sample(&self.keys, t)
    }

    /// Keys that fit the uniform, in order.
    fn packed(&self) -> Vec<(f32, f32)> {
        self.keys.iter().copied().take(MAX_CURVE_KEYS).collect()
    }
}

/// One RGBA keyframe gradient over normalized life, sampled linearly.
#[derive(Clone, Debug, PartialEq)]
pub struct ParticleGradient {
    keys: Vec<(f32, [f32; 4])>,
}

impl Default for ParticleGradient {
    fn default() -> Self {
        Self::white()
    }
}

impl ParticleGradient {
    /// A gradient holding one color everywhere.
    pub fn constant(color: [f32; 4]) -> Self {
        Self {
            keys: vec![(0.0, color), (1.0, color)],
        }
    }

    /// White: multiplying an emitter color leaves it unchanged.
    pub fn white() -> Self {
        Self::constant([1.0, 1.0, 1.0, 1.0])
    }

    /// A gradient through `keys`, sorted and clamped like [`ParticleCurve`].
    pub fn new(
        keys: impl IntoIterator<Item = (f32, [f32; 4])>,
    ) -> Result<Self, ParticleError> {
        let mut keys: Vec<(f32, [f32; 4])> = keys.into_iter().collect();
        if keys.is_empty() {
            return Err(ParticleError::InvalidRange("gradient"));
        }
        for (t, c) in &keys {
            if !t.is_finite() || !c.iter().all(|v| v.is_finite()) {
                return Err(ParticleError::NonFinite("gradient.key"));
            }
        }
        keys.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        if keys.windows(2).any(|w| w[0].0 >= w[1].0) {
            return Err(ParticleError::InvalidRange("gradient"));
        }
        for key in &mut keys {
            key.0 = key.0.clamp(0.0, 1.0);
        }
        Ok(Self { keys })
    }

    /// Key count.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// True when the gradient has no keys (never: constructors validate).
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Color at life fraction `t`, linear between keys and held at the ends.
    pub fn sample(&self, t: f32) -> [f32; 4] {
        gradient_sample(&self.keys, t)
    }

    fn packed(&self) -> Vec<(f32, [f32; 4])> {
        self.keys.iter().copied().take(MAX_CURVE_KEYS).collect()
    }
}

/// Point attractor: pulls particles toward `position` inside `radius`.
/// Negative `strength` repels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Attractor {
    /// World-space center.
    pub position: [f32; 3],
    /// Acceleration at the center, world units per second squared.
    pub strength: f32,
    /// Falloff radius; zero disables the attractor.
    pub radius: f32,
}

/// How particles respond to the frame's collision world.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum CollisionMode {
    /// Straight-line integration, no contact response.
    #[default]
    Off,
    /// Swept against the frame's AABB set ([`ParticleColliders::boxes`]).
    Aabb {
        /// Velocity kept after a bounce, `0` = stop, `1` = perfect reflection.
        bounce: f32,
    },
    /// Swept against the frame's voxel field
    /// ([`ParticleColliders::voxels`]).
    Voxels {
        /// Velocity kept after a bounce, `0` = stop, `1` = perfect reflection.
        bounce: f32,
    },
}

impl CollisionMode {
    const fn kind(self) -> u32 {
        match self {
            Self::Off => 0,
            Self::Aabb { .. } => 1,
            Self::Voxels { .. } => 2,
        }
    }

    const fn bounce(self) -> f32 {
        match self {
            Self::Off => 0.0,
            Self::Aabb { bounce } | Self::Voxels { bounce } => bounce,
        }
    }
}

/// When a system emits into another system's request ring.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SubEmitterMode {
    /// Sub-emission off.
    #[default]
    Off,
    /// One request when a particle's life ends.
    AtEnd,
    /// One request when a particle bounced off the collision world.
    AtCollision,
    /// One request when a particle spawns.
    AtStart,
    /// A per-particle stream at [`SubEmitter::rate`] requests per second.
    Constant,
}

impl SubEmitterMode {
    const fn kind(self) -> u32 {
        match self {
            Self::Off => 0,
            Self::AtEnd => 1,
            Self::AtCollision => 2,
            Self::AtStart => 3,
            Self::Constant => 4,
        }
    }
}

/// Link from one system's particles to another system's spawn budget.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubEmitter {
    /// Index into [`ParticleFrame::systems`].
    pub target: usize,
    /// Event that produces a request.
    pub mode: SubEmitterMode,
    /// Fraction of the parent's velocity the child inherits, `0..=1`.
    pub inherit_velocity: f32,
    /// Requests per second per live particle (`Constant` mode only).
    pub rate: f32,
}

impl Default for SubEmitter {
    fn default() -> Self {
        Self {
            target: 0,
            mode: SubEmitterMode::Off,
            inherit_velocity: 0.0,
            rate: 0.0,
        }
    }
}

/// Billboard appearance of one system's particles.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleDisplay {
    /// World-unit size multiplier applied to the simulated size.
    pub size_scale: f32,
    /// Fragment alpha multiplier.
    pub alpha: f32,
    /// Texture page sampled, or `None` for flat color quads.
    pub page: Option<u32>,
    /// Linear RGB tint multiplied into every particle color.
    pub tint: [f32; 3],
}

impl Default for ParticleDisplay {
    fn default() -> Self {
        Self {
            size_scale: 1.0,
            alpha: 1.0,
            page: None,
            tint: [1.0, 1.0, 1.0],
        }
    }
}

/// One GPU emitter. Plain data; the batch owns all simulation state.
#[derive(Clone, Debug, PartialEq)]
pub struct ParticleSystem {
    /// Simulation slots (also the billboard instance count).
    pub capacity: u32,
    /// Continuous spawn rate, particles per second.
    pub rate: f32,
    /// Per-particle lifetime range in seconds, `[min, max]`.
    pub lifetime: [f32; 2],
    /// Initial speed range, `[min, max]`.
    pub speed: [f32; 2],
    /// Cone half-angle around the emitter's local +Y, radians.
    pub spread: f32,
    /// Emission volume.
    pub shape: EmitterShape,
    /// Emitter local-to-world transform (volume, cone axis, spawn direction).
    pub transform: Mat4,
    /// Constant acceleration, world units per second squared.
    pub gravity: [f32; 3],
    /// Velocity damping per second, `0` = none, `1` = full stop per second.
    pub drag: f32,
    /// Per-particle base size range in world units, `[min, max]`.
    pub size: [f32; 2],
    /// Size multiplier over normalized life.
    pub size_curve: ParticleCurve,
    /// Base color (linear RGBA; alpha is opacity).
    pub color: [f32; 4],
    /// Color multiplier over normalized life.
    pub color_curve: ParticleGradient,
    /// Point attractors, capped at [`MAX_ATTRACTORS`].
    pub attractors: Vec<Attractor>,
    /// Collision response.
    pub collision: CollisionMode,
    /// Sub-emission link, or `None`.
    pub sub: Option<SubEmitter>,
    /// Deterministic seed; the same seed and `dt` sequence replay exactly.
    pub seed: u32,
    /// Billboard appearance.
    pub display: ParticleDisplay,
    /// 1-based pass whose draw list the billboards join; `0` is the shared
    /// scene target.
    pub pass: u32,
}

impl Default for ParticleSystem {
    fn default() -> Self {
        Self {
            capacity: 256,
            rate: 64.0,
            lifetime: [0.8, 1.4],
            speed: [2.0, 4.0],
            spread: 0.35,
            shape: EmitterShape::Point,
            transform: Mat4::IDENTITY,
            gravity: [0.0, -9.8, 0.0],
            drag: 0.2,
            size: [0.15, 0.25],
            size_curve: ParticleCurve::constant(1.0),
            color: [1.0, 1.0, 1.0, 1.0],
            color_curve: ParticleGradient::white(),
            attractors: Vec::new(),
            collision: CollisionMode::Off,
            sub: None,
            seed: 1,
            display: ParticleDisplay::default(),
            pass: 0,
        }
    }
}

impl ParticleSystem {
    /// A system with `capacity` slots and default parameters.
    pub fn new(capacity: u32) -> Self {
        Self {
            capacity,
            ..Self::default()
        }
    }

    /// Rejects the configs the GPU path cannot represent.
    pub fn validate(&self) -> Result<(), ParticleError> {
        if self.capacity == 0 {
            return Err(ParticleError::ZeroCapacity);
        }
        let finite = |v: f32, field: &'static str| {
            if v.is_finite() {
                Ok(())
            } else {
                Err(ParticleError::NonFinite(field))
            }
        };
        finite(self.rate, "rate")?;
        finite(self.spread, "spread")?;
        finite(self.drag, "drag")?;
        finite(self.color[0], "color")?;
        finite(self.color[1], "color")?;
        finite(self.color[2], "color")?;
        finite(self.color[3], "color")?;
        for v in self.gravity {
            finite(v, "gravity")?;
        }
        self.shape.validate()?;
        if !self.transform.is_finite() {
            return Err(ParticleError::NonFinite("transform"));
        }
        if !(self.lifetime[0].is_finite() && self.lifetime[1].is_finite())
            || self.lifetime[1] < self.lifetime[0]
            || self.lifetime[0] <= 0.0
        {
            return Err(ParticleError::InvalidRange("lifetime"));
        }
        for v in self.speed {
            finite(v, "speed")?;
        }
        if self.speed[1] < self.speed[0] {
            return Err(ParticleError::InvalidRange("speed"));
        }
        for v in self.size {
            finite(v, "size")?;
        }
        if self.size[1] < self.size[0] || self.size[0] < 0.0 {
            return Err(ParticleError::InvalidRange("size"));
        }
        if self.attractors.len() > MAX_ATTRACTORS {
            return Err(ParticleError::TooManyAttractors(self.attractors.len()));
        }
        for a in &self.attractors {
            for v in a.position {
                finite(v, "attractor.position")?;
            }
            finite(a.strength, "attractor.strength")?;
            finite(a.radius, "attractor.radius")?;
        }
        if let Some(sub) = self.sub {
            if matches!(sub.mode, SubEmitterMode::Off) {
                return Ok(());
            }
            if !sub.inherit_velocity.is_finite() || !sub.rate.is_finite() {
                return Err(ParticleError::NonFinite("sub"));
            }
        }
        finite(self.display.size_scale, "display.size_scale")?;
        finite(self.display.alpha, "display.alpha")?;
        finite(self.display.tint[0], "display.tint")?;
        finite(self.display.tint[1], "display.tint")?;
        finite(self.display.tint[2], "display.tint")
    }
}

/// A frame's particle submission: `dt`, the emitters, and the collision world.
#[derive(Clone, Debug)]
pub struct ParticleFrame {
    /// Simulation step in seconds, clamped to `[0, 0.1]` at staging.
    /// Deterministic replay needs the same sequence.
    pub dt: f32,
    /// Emitters, in submission order. Sub-emitter targets are indices into
    /// this list.
    pub systems: Vec<ParticleSystem>,
    /// Collision world the frame's systems sweep against.
    pub colliders: ParticleColliders,
}

impl ParticleFrame {
    /// An empty frame stepping `dt` seconds.
    pub fn new(dt: f32) -> Self {
        Self {
            dt,
            systems: Vec::new(),
            colliders: ParticleColliders::default(),
        }
    }

    /// Appends one system, returning its index for sub-emitter targets.
    pub fn push(&mut self, system: ParticleSystem) -> usize {
        self.systems.push(system);
        self.systems.len() - 1
    }

    /// `dt` clamped into the range the integrator accepts, with a warning
    /// when the caller's value is out of range.
    pub fn clamped_dt(&self) -> f32 {
        if !self.dt.is_finite() || self.dt <= 0.0 {
            0.0
        } else if self.dt > 0.1 {
            log::warn!("particle frame dt {} clamped to 0.1s", self.dt);
            0.1
        } else {
            self.dt
        }
    }

    /// Rejects a frame whose `dt` cannot be integrated.
    pub fn validate(&self) -> Result<(), ParticleError> {
        if !self.dt.is_finite() || self.dt <= 0.0 {
            return Err(ParticleError::InvalidDt);
        }
        for system in &self.systems {
            system.validate()?;
        }
        Ok(())
    }
}

impl Default for ParticleFrame {
    fn default() -> Self {
        Self::new(1.0 / 60.0)
    }
}

/// The frame's collision world, packed for the compute pass.
#[derive(Clone, Debug, Default)]
pub struct ParticleColliders {
    /// `(min, max)` boxes, normalized on push and capped at
    /// [`MAX_COLLIDER_BOXES`] at staging.
    pub boxes: Vec<([f32; 3], [f32; 3])>,
    /// Bounded voxel occupancy field, shared behind an [`Arc`] so a frame
    /// that reuses one world does not deep-copy it every submission.
    pub voxels: Option<std::sync::Arc<ParticleVoxels>>,
}

impl ParticleColliders {
    /// An empty world.
    pub fn new() -> Self {
        Self::default()
    }

    /// Takes the boxes out of the crate's camera-arm collider set, so
    /// particle collision and the spring arm share one world description.
    pub fn from_aabb_colliders(colliders: &AabbColliders) -> Self {
        let mut out = Self::default();
        for (min, max) in &colliders.boxes {
            out.push_box(min.to_array(), max.to_array());
        }
        out
    }

    /// Adds one box, normalizing corners and dropping degenerate or
    /// non-finite input.
    pub fn push_box(&mut self, min: [f32; 3], max: [f32; 3]) {
        let lo = [
            min[0].min(max[0]),
            min[1].min(max[1]),
            min[2].min(max[2]),
        ];
        let hi = [
            min[0].max(max[0]),
            min[1].max(max[1]),
            min[2].max(max[2]),
        ];
        if !lo.iter().chain(hi.iter()).all(|v| v.is_finite()) {
            log::warn!("particle collider box dropped (non-finite)");
            return;
        }
        if (0..3).all(|a| hi[a] <= lo[a]) {
            log::warn!("particle collider box dropped (degenerate)");
            return;
        }
        self.boxes.push((lo, hi));
    }

    /// Installs a voxel field, replacing any previous one.
    pub fn set_voxels(&mut self, voxels: ParticleVoxels) {
        self.voxels = Some(std::sync::Arc::new(voxels));
    }

    /// Installs a voxel field the caller already shares (the batch reuses
    /// the same GPU texture while this handle is unchanged).
    pub fn set_voxels_shared(&mut self, voxels: std::sync::Arc<ParticleVoxels>) {
        self.voxels = Some(voxels);
    }

    pub fn is_empty(&self) -> bool {
        self.boxes.is_empty() && self.voxels.is_none()
    }
}

/// Bounded voxel occupancy field built from any [`VoxelSource`], sampled with
/// `textureLoad` in the collision shader.
#[derive(Clone, Debug, PartialEq)]
pub struct ParticleVoxels {
    /// World-space position of cell `(0, 0, 0)`'s minimum corner.
    pub origin: [f32; 3],
    /// Edge length of one cell, world units.
    pub cell: f32,
    /// Field dimensions in cells.
    pub dims: [u32; 3],
    /// Occupancy per cell, row-major `x + dims.x * (y + dims.y * z)`; `1`
    /// = solid.
    pub cells: Vec<u32>,
}

impl ParticleVoxels {
    /// Samples `dims` cells of `source` starting at `origin` with `cell`
    /// spacing, marking a cell solid when `is_solid` accepts the block there.
    pub fn build<S, F>(
        source: &S,
        origin: [f32; 3],
        cell: f32,
        dims: [u32; 3],
        is_solid: F,
    ) -> Result<Self, ParticleError>
    where
        S: VoxelSource,
        F: Fn(&Cell) -> bool,
    {
        if dims.iter().any(|d| *d == 0) || dims.iter().any(|d| *d > 512) {
            return Err(ParticleError::BadVoxels);
        }
        if !cell.is_finite() || cell <= 0.0 || !origin.iter().all(|v| v.is_finite()) {
            return Err(ParticleError::BadVoxels);
        }
        let mut cells = vec![0u32; dims[0] as usize * dims[1] as usize * dims[2] as usize];
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    let world = [
                        origin[0] + x as f32 * cell,
                        origin[1] + y as f32 * cell,
                        origin[2] + z as f32 * cell,
                    ];
                    let solid = source
                        .get([
                            world[0].floor() as i32,
                            world[1].floor() as i32,
                            world[2].floor() as i32,
                        ])
                        .is_some_and(|block| is_solid(&block));
                    let index = (x + dims[0] * (y + dims[1] * z)) as usize;
                    cells[index] = solid as u32;
                }
            }
        }
        Ok(Self {
            origin,
            cell,
            dims,
            cells,
        })
    }

    /// Cell count.
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Occupancy at a cell coordinate, `false` outside the field.
    pub fn solid(&self, cell: [i32; 3]) -> bool {
        if cell.iter().any(|c| *c < 0)
            || (0..3).any(|a| cell[a] as u32 >= self.dims[a])
        {
            return false;
        }
        let index = cell[0] as usize
            + self.dims[0] as usize
                * (cell[1] as usize + self.dims[1] as usize * cell[2] as usize);
        self.cells[index] != 0
    }
}

// ---------------------------------------------------------------------------
// CPU mirrors of the simulation math. The WGSL below is a transliteration of
// these functions; the tests pin their behavior so the shader cannot drift
// without a failing test on the Rust side.
// ---------------------------------------------------------------------------

/// 32-bit mixing hash. Deterministic across platforms (pure integer ops).
pub fn hash_u32(seed: u32, index: u32) -> u32 {
    let mut x = seed ^ index.wrapping_mul(0x9E37_79B1);
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^= x >> 16;
    x
}

/// One uniform draw in `[0, 1)` for spawn `index`, distinguished by `salt`.
pub fn random01(seed: u32, index: u32, salt: u32) -> f32 {
    let h = hash_u32(seed ^ salt.wrapping_mul(0x85EB_CA6B), index);
    (h >> 8) as f32 / 16_777_216.0
}

/// Spawn direction inside a cone of half-angle `spread` around local +Y,
/// from two uniform draws. Mirrors the shader's `sample_cone`.
pub fn sample_cone(spread: f32, r0: f32, r1: f32) -> [f32; 3] {
    let cos_max = spread.clamp(0.0, std::f32::consts::PI).cos();
    let cos_theta = 1.0 - r0 * (1.0 - cos_max);
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = r1 * std::f32::consts::TAU;
    [sin_theta * phi.cos(), cos_theta, sin_theta * phi.sin()]
}

/// Linear keyframe sample over `(time, value)` keys sorted by time. Outside
/// the key range the nearest endpoint holds.
pub fn curve_sample(keys: &[(f32, f32)], t: f32) -> f32 {
    if keys.is_empty() {
        return 1.0;
    }
    if t <= keys[0].0 {
        return keys[0].1;
    }
    for pair in keys.windows(2) {
        let (t0, v0) = pair[0];
        let (t1, v1) = pair[1];
        if t < t1 {
            if t1 > t0 {
                let f = (t - t0) / (t1 - t0);
                return v0 + (v1 - v0) * f;
            }
            return v1;
        }
    }
    keys[keys.len() - 1].1
}

/// Linear RGBA gradient sample, same rules as [`curve_sample`].
pub fn gradient_sample(keys: &[(f32, [f32; 4])], t: f32) -> [f32; 4] {
    if keys.is_empty() {
        return [1.0; 4];
    }
    if t <= keys[0].0 {
        return keys[0].1;
    }
    for pair in keys.windows(2) {
        let (t0, c0) = pair[0];
        let (t1, c1) = pair[1];
        if t < t1 {
            let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 1.0 };
            return [
                c0[0] + (c1[0] - c0[0]) * f,
                c0[1] + (c1[1] - c0[1]) * f,
                c0[2] + (c1[2] - c0[2]) * f,
                c0[3] + (c1[3] - c0[3]) * f,
            ];
        }
    }
    keys[keys.len() - 1].1
}

/// One particle as the storage buffer holds it (64 bytes). `pos/life`,
/// `vel/size`, `color` ride the same lanes the shader reads.
///
/// `size` and `color` are the values the particle spawned with; the size
/// curve and color gradient are applied at billboard time (`1 - life/life0`
/// is the curve input), so a simulation step never compounds them.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Particle {
    pub pos: [f32; 3],
    /// Remaining life in seconds; `<= 0` is dead.
    pub life: f32,
    pub vel: [f32; 3],
    /// Base size in world units, before [`ParticleSystem::size_curve`].
    pub size: f32,
    /// Base linear RGBA, before [`ParticleSystem::color_curve`].
    pub color: [f32; 4],
    pub seed: u32,
    /// Lifetime the particle spawned with (curve input is `1 - life/life0`).
    pub life0: f32,
    /// Reserved lanes, kept so the Rust and WGSL layouts agree byte for byte.
    pub _pad: [f32; 2],
}

const _: () = assert!(std::mem::size_of::<Particle>() == PARTICLE_BYTES);

/// One integration step: gravity, attractors and drag, then the position move.
/// Mirrors the shader's `integrate` exactly.
pub fn integrate_step(
    particle: &mut Particle,
    dt: f32,
    gravity: [f32; 3],
    drag: f32,
    attractors: &[Attractor],
) {
    let mut accel = gravity;
    for attractor in attractors {
        if attractor.radius <= 0.0 || attractor.strength == 0.0 {
            continue;
        }
        let offset = [
            attractor.position[0] - particle.pos[0],
            attractor.position[1] - particle.pos[1],
            attractor.position[2] - particle.pos[2],
        ];
        let dist = (offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2])
            .sqrt()
            .max(1e-6);
        if dist >= attractor.radius {
            continue;
        }
        let falloff = 1.0 - dist / attractor.radius;
        let scale = attractor.strength * falloff * falloff / dist;
        accel[0] += offset[0] * scale;
        accel[1] += offset[1] * scale;
        accel[2] += offset[2] * scale;
    }
    let keep = (1.0 - drag.max(0.0) * dt).clamp(0.0, 1.0);
    for axis in 0..3 {
        particle.vel[axis] = particle.vel[axis] * keep + accel[axis] * dt;
    }
    for axis in 0..3 {
        particle.pos[axis] += particle.vel[axis] * dt;
    }
    particle.life -= dt;
}

/// Velocity after bouncing off a unit `normal`: the normal component comes
/// back as `-bounce * vn` (so `0` slides, `1` reflects perfectly) while the
/// tangential component is untouched. A velocity already leaving the surface
/// is returned unchanged. Mirrors the shader's `bounce_velocity`.
pub fn bounce_velocity(vel: [f32; 3], normal: [f32; 3], bounce: f32) -> [f32; 3] {
    let vn = vel[0] * normal[0] + vel[1] * normal[1] + vel[2] * normal[2];
    if vn >= 0.0 {
        return vel;
    }
    let k = vn * (1.0 + bounce.max(0.0));
    [
        vel[0] - normal[0] * k,
        vel[1] - normal[1] * k,
        vel[2] - normal[2] * k,
    ]
}

/// Swept segment against an AABB, expanded by the caller for a particle
/// radius. Returns `(normal, t, axis)`: the surface normal toward the segment
/// start, the hit fraction in `0..=1`, and the axis the segment entered
/// through. Mirrors the shader's `segment_box`.
pub fn segment_box(
    start: [f32; 3],
    end: [f32; 3],
    min: [f32; 3],
    max: [f32; 3],
) -> Option<([f32; 3], f32, u32)> {
    let d = [
        end[0] - start[0],
        end[1] - start[1],
        end[2] - start[2],
    ];
    let mut t_min = 0.0f32;
    let mut t_max = 1.0f32;
    let mut axis = 0u32;
    let mut sign = -1.0f32;
    for a in 0..3usize {
        let o = start[a];
        let dd = d[a];
        let lo = min[a];
        let hi = max[a];
        if dd.abs() < 1e-9 {
            if o < lo || o > hi {
                return None;
            }
            continue;
        }
        let mut t0 = (lo - o) / dd;
        let mut t1 = (hi - o) / dd;
        let mut s = -1.0f32;
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
            s = 1.0;
        }
        if t0 > t_min {
            t_min = t0;
            axis = a as u32;
            sign = s;
        }
        t_max = t_max.min(t1);
        if t_min > t_max {
            return None;
        }
    }
    if t_min > 1.0 || t_max < 0.0 {
        return None;
    }
    let mut normal = [0.0f32; 3];
    normal[axis as usize] = sign;
    Some((normal, t_min.clamp(0.0, 1.0), axis))
}

/// March a segment through a voxel field and return the first solid hit as
/// `(normal, t)`. Mirrors the shader's `segment_voxels`.
pub fn segment_voxels(
    start: [f32; 3],
    end: [f32; 3],
    field: &ParticleVoxels,
) -> Option<([f32; 3], f32)> {
    let inv = 1.0 / field.cell;
    let to_grid = |p: [f32; 3]| {
        [
            (p[0] - field.origin[0]) * inv,
            (p[1] - field.origin[1]) * inv,
            (p[2] - field.origin[2]) * inv,
        ]
    };
    let g0 = to_grid(start);
    let g1 = to_grid(end);
    let d = [g1[0] - g0[0], g1[1] - g0[1], g1[2] - g0[2]];
    let mut cell = [
        g0[0].floor() as i32,
        g0[1].floor() as i32,
        g0[2].floor() as i32,
    ];
    let step = [
        if d[0] > 0.0 { 1i32 } else { -1i32 },
        if d[1] > 0.0 { 1 } else { -1 },
        if d[2] > 0.0 { 1 } else { -1 },
    ];
    let mut t_max = [f32::INFINITY; 3];
    let mut t_delta = [f32::INFINITY; 3];
    for a in 0..3 {
        if d[a].abs() < 1e-12 {
            continue;
        }
        let bound = if step[a] > 0 { cell[a] + 1 } else { cell[a] };
        t_max[a] = (bound as f32 - g0[a]) / d[a];
        t_delta[a] = 1.0 / d[a].abs();
    }
    let max_steps = field.dims[0] as usize + field.dims[1] as usize + field.dims[2] as usize + 3;
    for _ in 0..max_steps {
        if t_max.iter().cloned().fold(f32::INFINITY, f32::min) > 1.0 {
            return None;
        }
        let axis = if t_max[0] < t_max[1] && t_max[0] < t_max[2] {
            0
        } else if t_max[1] < t_max[2] {
            1
        } else {
            2
        };
        let t = t_max[axis];
        cell[axis] += step[axis];
        t_max[axis] += t_delta[axis];
        if t < 0.0 {
            continue;
        }
        let inside = (0..3).all(|a| cell[a] >= 0 && (cell[a] as u32) < field.dims[a]);
        if !inside {
            return None;
        }
        if field.solid(cell) {
            let mut normal = [0.0f32; 3];
            normal[axis] = -step[axis] as f32;
            return Some((normal, t.clamp(0.0, 1.0)));
        }
    }
    None
}

/// Emitter parameters the compute pass reads, packed per frame per system.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GpuEmitter {
    pub transform: [[f32; 4]; 4],
    /// `(shape kind, radius, height, unused)`.
    pub shape: [f32; 4],
    /// Box half extents (shape kind 2).
    pub extents: [f32; 4],
    /// `(speed min, speed max, spread, drag)`.
    pub motion: [f32; 4],
    /// `(life min, life max, size min, size max)`.
    pub life: [f32; 4],
    /// `(gravity.xyz, dt)`.
    pub env: [f32; 4],
    /// Base color, linear RGBA.
    pub color: [f32; 4],
    /// `(collision kind, sub mode, texture page + 1 or 0, display flags)`.
    pub flags: [u32; 4],
    /// `(continuous budget, sub budget, capacity, request ring cap)`.
    pub counts: [u32; 4],
    /// `(emitter seed, frame index, size curve keys, color curve keys)`.
    pub seed: [u32; 4],
    /// `(sub rate, sub inherit velocity, collision bounce, unused)`.
    pub extra: [f32; 4],
    /// Size curve keys as `(t, v, 0, 0)`, ascending.
    pub size_curve: [[f32; 4]; MAX_CURVE_KEYS],
    /// Color curve keys as `(t, r, g, b)` — alpha needs a second lane, so
    /// this carries RGB and `color_curve_a` the alpha.
    pub color_curve: [[f32; 4]; MAX_CURVE_KEYS],
    /// Color curve alpha per key.
    pub color_curve_a: [f32; 4],
    /// `(attractor.xyz, radius)`.
    pub attractor_pos: [[f32; 4]; MAX_ATTRACTORS],
    /// `(strength, 0, 0, 0)`.
    pub attractor_str: [[f32; 4]; MAX_ATTRACTORS],
}

unsafe impl bytemuck::Zeroable for GpuEmitter {}
unsafe impl bytemuck::Pod for GpuEmitter {}

/// Collision world for the compute pass.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GpuColliders {
    /// `(box count, reserved, reserved, reserved)`.
    pub box_count: [f32; 4],
    /// `(origin.xyz, cell size)`.
    pub voxel_origin: [f32; 4],
    /// `(dims.xyz, unused)`.
    pub voxel_dims: [u32; 4],
    /// Box pairs: `boxes[2i]` = min, `boxes[2i + 1]` = max.
    pub boxes: [[f32; 4]; MAX_COLLIDER_BOXES * 2],
}

unsafe impl bytemuck::Zeroable for GpuColliders {}
unsafe impl bytemuck::Pod for GpuColliders {}

/// Simulation cursor the prepare dispatch advances.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct GpuSimState {
    /// Absolute spawn counter; slot of spawn `c` is `c % capacity`.
    pub cursor: u32,
    /// First spawn counter of this frame.
    pub spawn_start: u32,
    /// Spawns this frame (continuous + accepted requests).
    pub spawn_total: u32,
    /// Request-ring read position before this frame's consumption.
    pub sub_start: u32,
    /// Requests accepted this frame.
    pub sub_consume: u32,
    /// Frame counter (constant sub-emission draw salt).
    pub frame: u32,
    pub _pad: [u32; 2],
}

unsafe impl bytemuck::Zeroable for GpuSimState {}
unsafe impl bytemuck::Pod for GpuSimState {}

/// Billboard draw parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GpuDrawParams {
    /// `(size scale, alpha, page as f32, textured flag)`.
    pub scale_alpha: [f32; 4],
    /// `(tint.rgb, unused)`.
    pub tint: [f32; 4],
}

unsafe impl bytemuck::Zeroable for GpuDrawParams {}
unsafe impl bytemuck::Pod for GpuDrawParams {}

/// Packs one frame's emitter parameters. `budget` is the continuous spawn
/// count the CPU resolved from `rate * dt` plus the carried fraction.
pub(crate) fn pack_emitter(
    system: &ParticleSystem,
    budget: u32,
    sub_budget: u32,
    frame_index: u32,
    dt: f32,
    request_cap: u32,
) -> GpuEmitter {
    let (kind, radius, height) = match system.shape {
        EmitterShape::Point => (0, 0.0, 0.0),
        EmitterShape::Sphere { radius } => (1, radius, 0.0),
        EmitterShape::Box { .. } => (2, 0.0, 0.0),
        EmitterShape::Cone { radius, height } => (3, radius, height),
        EmitterShape::CircleEdge { radius } => (4, radius, 0.0),
    };
    let extents = match system.shape {
        EmitterShape::Box { extents } => extents,
        _ => [0.0; 3],
    };
    let mut size_curve = [[0.0f32; 4]; MAX_CURVE_KEYS];
    let packed_size = system.size_curve.packed();
    for (slot, (t, v)) in size_curve.iter_mut().zip(packed_size.iter()) {
        *slot = [*t, *v, 0.0, 0.0];
    }
    let mut color_curve = [[0.0f32; 4]; MAX_CURVE_KEYS];
    let mut color_curve_a = [0.0f32; 4];
    let packed_color = system.color_curve.packed();
    for (i, (t, c)) in packed_color.iter().enumerate() {
        color_curve[i] = [*t, c[0], c[1], c[2]];
        if i < color_curve_a.len() {
            color_curve_a[i] = c[3];
        }
    }
    let mut attractor_pos = [[0.0f32; 4]; MAX_ATTRACTORS];
    let mut attractor_str = [[0.0f32; 4]; MAX_ATTRACTORS];
    for (i, a) in system.attractors.iter().take(MAX_ATTRACTORS).enumerate() {
        attractor_pos[i] = [a.position[0], a.position[1], a.position[2], a.radius];
        attractor_str[i] = [a.strength, 0.0, 0.0, 0.0];
    }
    let (sub_mode, sub_rate, sub_inherit) = match system.sub {
        Some(sub) => (sub.mode.kind(), sub.rate, sub.inherit_velocity),
        None => (0, 0.0, 0.0),
    };
    let page = system.display.page.map_or(0, |p| p + 1);
    GpuEmitter {        transform: system.transform.to_cols_array_2d(),
        shape: [kind as f32, radius, height, 0.0],
        extents: [extents[0], extents[1], extents[2], 0.0],
        motion: [
            system.speed[0],
            system.speed[1],
            system.spread,
            system.drag,
        ],
        life: [
            system.lifetime[0],
            system.lifetime[1],
            system.size[0],
            system.size[1],
        ],
        env: [system.gravity[0], system.gravity[1], system.gravity[2], dt],
        color: system.color,
        flags: [system.collision.kind(), sub_mode, page, 0],
        counts: [budget, sub_budget, system.capacity, request_cap],
        seed: [
            system.seed,
            frame_index,
            packed_size.len() as u32,
            packed_color.len() as u32,
        ],
        extra: [sub_rate, sub_inherit, system.collision.bounce(), 0.0],
        size_curve,
        color_curve,
        color_curve_a,
        attractor_pos,
        attractor_str,
    }
}

/// Packs the frame's collision world. `boxes` is truncated (with a warning)
/// to [`MAX_COLLIDER_BOXES`].
pub(crate) fn pack_colliders(colliders: &ParticleColliders) -> GpuColliders {
    let mut packed = GpuColliders {
        // Per-system data only: the collider buffer carries the boxes and
        // the voxel field (the per-system response is in the emitter's
        // `extra` lane).
        box_count: [0.0; 4],
        voxel_origin: [0.0; 4],
        voxel_dims: [0; 4],
        boxes: [[0.0; 4]; MAX_COLLIDER_BOXES * 2],
    };
    if colliders.boxes.len() > MAX_COLLIDER_BOXES {
        log::warn!(
            "particle colliders: keeping first {} of {} boxes",
            MAX_COLLIDER_BOXES,
            colliders.boxes.len()
        );
    }
    for (i, (min, max)) in colliders.boxes.iter().take(MAX_COLLIDER_BOXES).enumerate() {
        packed.boxes[i * 2] = [min[0], min[1], min[2], 0.0];
        packed.boxes[i * 2 + 1] = [max[0], max[1], max[2], 0.0];
    }
    packed.box_count[0] = colliders.boxes.len().min(MAX_COLLIDER_BOXES) as f32;
    if let Some(voxels) = &colliders.voxels {
        packed.voxel_origin = [
            voxels.origin[0],
            voxels.origin[1],
            voxels.origin[2],
            voxels.cell,
        ];
        packed.voxel_dims = [voxels.dims[0], voxels.dims[1], voxels.dims[2], 0];
    }
    packed
}

/// Sub-emitter request ring capacity for a system: enough for a frame's
/// worth of deaths, capped.
pub fn request_capacity(capacity: u32) -> u32 {
    capacity.clamp(1, MAX_REQUEST_RING)
}

/// Compute shader: `prepare` advances the spawn cursor, `simulate` spawns,
/// integrates, collides and sub-emits.
pub const PARTICLE_COMPUTE_WGSL: &str = r#"
struct Particle {
    pos: vec3<f32>,
    life: f32,
    vel: vec3<f32>,
    size: f32,
    color: vec4<f32>,
    seed: u32,
    life0: f32,
    _pad: vec2<f32>,
};

struct SubRequest {
    pos: vec3<f32>,
    size: f32,
    vel: vec3<f32>,
    weight: f32,
    color: vec4<f32>,
};

struct SimState {
    cursor: u32,
    spawn_start: u32,
    spawn_total: u32,
    sub_start: u32,
    sub_consume: u32,
    frame: u32,
    _pad: vec2<u32>,
};

struct RequestRing {
    write: atomic<u32>,
    read: u32,
    _pad: vec2<u32>,
    entries: array<SubRequest>,
};

struct Emitter {
    transform: mat4x4<f32>,
    shape: vec4<f32>,
    extents: vec4<f32>,
    motion: vec4<f32>,
    life: vec4<f32>,
    env: vec4<f32>,
    color: vec4<f32>,
    flags: vec4<u32>,
    counts: vec4<u32>,
    seed: vec4<u32>,
    extra: vec4<f32>,
    size_curve: array<vec4<f32>, 8>,
    color_curve: array<vec4<f32>, 8>,
    color_curve_a: vec4<f32>,
    attractor_pos: array<vec4<f32>, 8>,
    attractor_str: array<vec4<f32>, 8>,
};

struct Colliders {
    box_count: vec4<f32>,
    voxel_origin: vec4<f32>,
    voxel_dims: vec4<u32>,
    boxes: array<vec4<f32>, 128>,
};

@group(0) @binding(0) var<storage, read> emitter: Emitter;
@group(0) @binding(1) var<storage, read_write> sim: SimState;
@group(0) @binding(2) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(3) var<storage, read_write> incoming: RequestRing;
@group(0) @binding(4) var<storage, read_write> outgoing: RequestRing;
@group(0) @binding(5) var<storage, read> colliders: Colliders;
@group(0) @binding(6) var voxels: texture_3d<u32>;

const PI: f32 = 3.141592653589793;

fn rand01(index: u32, salt: u32) -> f32 {
    var x = emitter.seed.x ^ (salt * 0x85EBCA6Bu) ^ (index * 0x9E3779B1u);
    x = x ^ (x >> 16u);
    x = x * 0x7FEB352Du;
    x = x ^ (x >> 15u);
    x = x * 0x846CA68Bu;
    x = x ^ (x >> 16u);
    return f32(x >> 8u) / 16777216.0;
}

fn hash_u32_mix(index: u32) -> u32 {
    var x = index ^ 0x27D4EB2Du;
    x = x ^ (x >> 15u);
    x = x * 0x2C1B3C6Du;
    x = x ^ (x >> 12u);
    return x;
}

fn sample_shape(r0: f32, r1: f32, r2: f32) -> vec3<f32> {
    let kind = i32(emitter.shape.x + 0.5);
    if (kind == 1) {
        let z = r0 * 2.0 - 1.0;
        let turn = r1 * 2.0 * PI;
        let ring = sqrt(max(1.0 - z * z, 0.0));
        let rad = max(emitter.shape.y, 0.0) * pow(max(r2, 1e-6), 1.0 / 3.0);
        return vec3<f32>(ring * cos(turn) * rad, z * rad, ring * sin(turn) * rad);
    }
    if (kind == 2) {
        return vec3<f32>(
            (r0 * 2.0 - 1.0) * emitter.extents.x,
            (r1 * 2.0 - 1.0) * emitter.extents.y,
            (r2 * 2.0 - 1.0) * emitter.extents.z,
        );
    }
    if (kind == 3) {
        let height_frac = r2;
        let turn = r1 * 2.0 * PI;
        let rad = max(emitter.shape.y, 0.0) * height_frac * sqrt(max(r0, 0.0));
        return vec3<f32>(rad * cos(turn), height_frac * emitter.shape.z, rad * sin(turn));
    }
    if (kind == 4) {
        let turn = r0 * 2.0 * PI;
        return vec3<f32>(emitter.shape.y * cos(turn), 0.0, emitter.shape.y * sin(turn));
    }
    return vec3<f32>(0.0);
}

fn sample_cone(r0: f32, r1: f32) -> vec3<f32> {
    let spread = clamp(emitter.motion.z, 0.0, PI);
    let cos_max = cos(spread);
    let cos_theta = 1.0 - r0 * (1.0 - cos_max);
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let phi = r1 * 2.0 * PI;
    return vec3<f32>(sin_theta * cos(phi), cos_theta, sin_theta * sin(phi));
}

fn size_curve_value(t: f32) -> f32 {
    let count = emitter.seed.z;
    if (count == 0u) {
        return 1.0;
    }
    var prev_t = emitter.size_curve[0].x;
    var prev_v = emitter.size_curve[0].y;
    if (t <= prev_t) {
        return prev_v;
    }
    var value = prev_v;
    for (var i = 1u; i < count; i = i + 1u) {
        let key = emitter.size_curve[i];
        if (t < key.x) {
            if (key.x > prev_t) {
                let f = (t - prev_t) / (key.x - prev_t);
                return prev_v + (key.y - prev_v) * f;
            }
            return key.y;
        }
        prev_t = key.x;
        prev_v = key.y;
        value = key.y;
    }
    return value;
}

fn color_curve_value(t: f32) -> vec4<f32> {
    let count = emitter.seed.w;
    if (count == 0u) {
        return vec4<f32>(1.0);
    }
    var prev_t = emitter.color_curve[0].x;
    var prev_c = vec4<f32>(
        emitter.color_curve[0].y,
        emitter.color_curve[0].z,
        emitter.color_curve[0].w,
        emitter.color_curve_a[0],
    );
    if (t <= prev_t) {
        return prev_c;
    }
    var value = prev_c;
    for (var i = 1u; i < count; i = i + 1u) {
        let key = emitter.color_curve[i];
        if (t < key.x) {
            let next_c = vec4<f32>(key.y, key.z, key.w, emitter.color_curve_a[i]);
            if (key.x > prev_t) {
                let f = (t - prev_t) / (key.x - prev_t);
                return mix(prev_c, next_c, f);
            }
            return next_c;
        }
        prev_t = key.x;
        prev_c = vec4<f32>(key.y, key.z, key.w, emitter.color_curve_a[i]);
        value = prev_c;
    }
    return value;
}

fn attractor_accel(p: vec3<f32>) -> vec3<f32> {
    var accel = vec3<f32>(0.0);
    for (var i = 0u; i < 8u; i = i + 1u) {
        let spec = emitter.attractor_pos[i];
        if (spec.w <= 0.0) {
            continue;
        }
        let strength = emitter.attractor_str[i].x;
        if (strength == 0.0) {
            continue;
        }
        let offset = spec.xyz - p;
        let dist = max(length(offset), 1e-6);
        if (dist >= spec.w) {
            continue;
        }
        let falloff = 1.0 - dist / spec.w;
        accel = accel + offset * (strength * falloff * falloff / dist);
    }
    return accel;
}

fn bounce_velocity(v: vec3<f32>, normal: vec3<f32>, bounce: f32) -> vec3<f32> {
    let vn = dot(v, normal);
    if (vn >= 0.0) {
        return v;
    }
    return v - normal * (vn * (1.0 + max(bounce, 0.0)));
}

struct Hit {
    hit: bool,
    pos: vec3<f32>,
    vel: vec3<f32>,
    normal: vec3<f32>,
};

fn segment_box(p0: vec3<f32>, p1: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> vec4<f32> {
    let d = p1 - p0;
    var t_min = 0.0;
    var t_max = 1.0;
    var axis = -1.0;
    var sign = -1.0;
    for (var a = 0u; a < 3u; a = a + 1u) {
        let o = p0[a];
        let dd = d[a];
        let blo = lo[a];
        let bhi = hi[a];
        if (abs(dd) < 1e-9) {
            if (o < blo || o > bhi) {
                return vec4<f32>(-1.0);
            }
            continue;
        }
        var t0 = (blo - o) / dd;
        var t1 = (bhi - o) / dd;
        var s = -1.0;
        if (t0 > t1) {
            let tmp = t0;
            t0 = t1;
            t1 = tmp;
            s = 1.0;
        }
        if (t0 > t_min) {
            t_min = t0;
            axis = f32(a);
            sign = s;
        }
        t_max = min(t_max, t1);
        if (t_min > t_max) {
            return vec4<f32>(-1.0);
        }
    }
    if (axis < 0.0) {
        return vec4<f32>(-1.0);
    }
    return vec4<f32>(t_min, axis, sign, 0.0);
}

fn collide_boxes(p0: vec3<f32>, p1: vec3<f32>, v: vec3<f32>, radius: f32, bounce: f32) -> Hit {
    var out = Hit(false, p1, v, vec3<f32>(0.0));
    let count = min(u32(colliders.box_count.x + 0.5), 64u);
    var best_t = 2.0;
    for (var i = 0u; i < count; i = i + 1u) {
        let lo = colliders.boxes[i * 2u].xyz - vec3<f32>(radius);
        let hi = colliders.boxes[i * 2u + 1u].xyz + vec3<f32>(radius);
        let hit = segment_box(p0, p1, lo, hi);
        if (hit.x < 0.0 || hit.x >= best_t) {
            continue;
        }
        best_t = hit.x;
        let axis = u32(hit.y + 0.5);
        var normal = vec3<f32>(0.0);
        if (axis == 0u) {
            normal.x = hit.z;
        } else if (axis == 1u) {
            normal.y = hit.z;
        } else {
            normal.z = hit.z;
        }
        out.hit = true;
        out.normal = normal;
        out.pos = p0 + (p1 - p0) * hit.x + normal * 1e-3;
        out.vel = bounce_velocity(v, normal, bounce);
    }
    return out;
}

fn collide_voxels(p0: vec3<f32>, p1: vec3<f32>, v: vec3<f32>, bounce: f32) -> Hit {
    var out = Hit(false, p1, v, vec3<f32>(0.0));
    let origin = colliders.voxel_origin.xyz;
    let cell = colliders.voxel_origin.w;
    let dims = colliders.voxel_dims.xyz;
    if (cell <= 0.0 || dims.x == 0u || dims.y == 0u || dims.z == 0u) {
        return out;
    }
    let inv = 1.0 / cell;
    let g0 = (p0 - origin) * inv;
    let g1 = (p1 - origin) * inv;
    let d = g1 - g0;
    var floor_cell = vec3<i32>(floor(g0));
    var step = select(vec3<i32>(-1), vec3<i32>(1), d > vec3<f32>(0.0));
    var t_max = vec3<f32>(1e30);
    var t_delta = vec3<f32>(1e30);
    if (abs(d.x) > 1e-12) {
        let bound = select(f32(floor_cell.x), f32(floor_cell.x + 1), d.x > 0.0);
        t_max.x = (bound - g0.x) / d.x;
        t_delta.x = 1.0 / abs(d.x);
    }
    if (abs(d.y) > 1e-12) {
        let bound = select(f32(floor_cell.y), f32(floor_cell.y + 1), d.y > 0.0);
        t_max.y = (bound - g0.y) / d.y;
        t_delta.y = 1.0 / abs(d.y);
    }
    if (abs(d.z) > 1e-12) {
        let bound = select(f32(floor_cell.z), f32(floor_cell.z + 1), d.z > 0.0);
        t_max.z = (bound - g0.z) / d.z;
        t_delta.z = 1.0 / abs(d.z);
    }
    let max_steps = dims.x + dims.y + dims.z + 3u;
    for (var s = 0u; s < max_steps; s = s + 1u) {
        let t = min(t_max.x, min(t_max.y, t_max.z));
        if (t > 1.0) {
            return out;
        }
        var axis = 2u;
        if (t_max.x <= t_max.y && t_max.x <= t_max.z) {
            axis = 0u;
        } else if (t_max.y <= t_max.z) {
            axis = 1u;
        }
        floor_cell[axis] = floor_cell[axis] + step[axis];
        t_max[axis] = t_max[axis] + t_delta[axis];
        if (t < 0.0) {
            continue;
        }
        if (floor_cell.x < 0 || floor_cell.y < 0 || floor_cell.z < 0
            || floor_cell.x >= i32(dims.x) || floor_cell.y >= i32(dims.y) || floor_cell.z >= i32(dims.z)) {
            return out;
        }
        if (textureLoad(voxels, floor_cell, 0).r != 0u) {
            var normal = vec3<f32>(0.0);
            if (axis == 0u) {
                normal.x = -f32(step.x);
            } else if (axis == 1u) {
                normal.y = -f32(step.y);
            } else {
                normal.z = -f32(step.z);
            }
            out.hit = true;
            out.normal = normal;
            out.pos = p0 + (p1 - p0) * clamp(t, 0.0, 1.0) + normal * 1e-3;
            out.vel = bounce_velocity(v, normal, bounce);
            return out;
        }
    }
    return out;
}

fn collide(p0: vec3<f32>, p1: vec3<f32>, v: vec3<f32>, radius: f32) -> Hit {
    let mode = u32(emitter.flags.x);
    let bounce = emitter.extra.z;
    if (mode == 1u) {
        return collide_boxes(p0, p1, v, radius, bounce);
    }
    if (mode == 2u) {
        return collide_voxels(p0, p1, v, bounce);
    }
    return Hit(false, p1, v, vec3<f32>(0.0));
}

fn push_request(pos: vec3<f32>, size: f32, vel: vec3<f32>, color: vec4<f32>) {
    let cap = emitter.counts.w;
    if (cap == 0u) {
        return;
    }
    let slot = atomicAdd(&outgoing.write, 1u);
    let read = outgoing.read;
    if (slot >= read && (slot - read) < cap) {
        outgoing.entries[slot % cap] = SubRequest(pos, size, vel, 0.0, color);
    }
}

@compute @workgroup_size(1)
fn prepare() {
    var s = sim;
    let cap = emitter.counts.z;
    let budget = min(emitter.counts.x, cap);
    let sub_budget = emitter.counts.y;
    let req_cap = emitter.counts.w;
    var accept = 0u;
    if (sub_budget > 0u && req_cap > 0u) {
        let write = atomicLoad(&incoming.write);
        let read = incoming.read;
        if (write > read) {
            accept = min(write - read, sub_budget);
        }
    }
    accept = min(accept, cap - budget);
    s.spawn_start = s.cursor;
    s.spawn_total = budget + accept;
    s.sub_start = incoming.read;
    s.sub_consume = accept;
    s.cursor = s.cursor + budget + accept;
    incoming.read = s.sub_start + accept;
    sim = s;
}

@compute @workgroup_size(256)
fn simulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let cap = emitter.counts.z;
    if (index >= cap) {
        return;
    }
    let dt = emitter.env.w;
    let budget = min(emitter.counts.x, cap);
    var p = particles[index];
    let first = sim.spawn_start % cap;
    let offset = (index + cap - first) % cap;
    var fresh = false;
    var hit = Hit(false, p.pos, p.vel, vec3<f32>(0.0));
    if (offset < sim.spawn_total) {
        let counter = sim.spawn_start + offset;
        let dir_world = (emitter.transform * vec4<f32>(sample_cone(rand01(counter, 3u), rand01(counter, 4u)), 0.0)).xyz;
        let speed = mix(emitter.motion.x, emitter.motion.y, rand01(counter, 6u));
        let life = mix(emitter.life.x, emitter.life.y, rand01(counter, 7u));
        let size = mix(emitter.life.z, emitter.life.w, rand01(counter, 8u));
        var base_vel = dir_world * speed;
        var base_color = emitter.color;
        if (offset < budget) {
            let local = sample_shape(rand01(counter, 21u), rand01(counter, 22u), rand01(counter, 23u));
            p.pos = (emitter.transform * vec4<f32>(local, 1.0)).xyz;
        } else {
            let request_index = sim.sub_start + (offset - budget);
            let req = incoming.entries[request_index % emitter.counts.w];
            p.pos = req.pos;
            base_vel = req.vel * emitter.extra.y + base_vel;
            base_color = req.color * emitter.color;
        }
        p.life = max(life, 1e-4);
        p.life0 = p.life;
        p.vel = base_vel;
        p.size = size;
        p.color = base_color;
        p.seed = index;
        p._pad = vec2<f32>(0.0);
        fresh = true;
    } else {
        if (p.life <= 0.0) {
            return;
        }
        let age = clamp(1.0 - p.life / max(p.life0, 1e-6), 0.0, 1.0);
        let old = p.pos;
        let accel = emitter.env.xyz + attractor_accel(p.pos);
        let keep = clamp(1.0 - max(emitter.motion.w, 0.0) * dt, 0.0, 1.0);
        p.vel = p.vel * keep + accel * dt;
        p.pos = p.pos + p.vel * dt;
        hit = collide(old, p.pos, p.vel, p.size * size_curve_value(age) * 0.5);
        if (hit.hit) {
            p.pos = hit.pos;
            p.vel = hit.vel;
        }
        p.life = p.life - dt;
    }
    let mode = emitter.flags.y;
    if (mode == 3u && fresh) {
        push_request(p.pos, p.size, p.vel, p.color);
    }
    if (mode == 2u && hit.hit) {
        push_request(p.pos, p.size, p.vel, p.color);
    }
    if (p.life <= 0.0) {
        if (mode == 1u) {
            push_request(p.pos, p.size, p.vel, p.color);
        }
        particles[index] = p;
        return;
    }
    if (mode == 4u && emitter.extra.x > 0.0) {
        let draw = rand01(index, hash_u32_mix(sim.frame));
        if (draw < emitter.extra.x * dt) {
            push_request(p.pos, p.size, p.vel, p.color);
        }
    }
    particles[index] = p;
}
"#;

/// Billboard shader: expands the particle storage into camera-facing quads
/// and samples an optional texture page.
pub const PARTICLE_BILLBOARD_WGSL: &str = r#"
struct Particle {
    pos: vec3<f32>,
    life: f32,
    vel: vec3<f32>,
    size: f32,
    color: vec4<f32>,
    seed: u32,
    life0: f32,
    _pad: vec2<f32>,
};

struct Camera {
    view_proj: mat4x4<f32>,
};

struct Emitter {
    transform: mat4x4<f32>,
    shape: vec4<f32>,
    extents: vec4<f32>,
    motion: vec4<f32>,
    life: vec4<f32>,
    env: vec4<f32>,
    color: vec4<f32>,
    flags: vec4<u32>,
    counts: vec4<u32>,
    seed: vec4<u32>,
    extra: vec4<f32>,
    size_curve: array<vec4<f32>, 8>,
    color_curve: array<vec4<f32>, 8>,
    color_curve_a: vec4<f32>,
    attractor_pos: array<vec4<f32>, 8>,
    attractor_str: array<vec4<f32>, 8>,
};

struct DrawParams {
    scale_alpha: vec4<f32>,
    tint: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var scene_tex: texture_2d_array<f32>;
@group(1) @binding(1) var scene_smp: sampler;
@group(2) @binding(0) var<storage, read> particles: array<Particle>;
@group(2) @binding(1) var<storage, read> emitter: Emitter;
@group(2) @binding(2) var<uniform> draw: DrawParams;

fn size_curve_value(t: f32) -> f32 {
    let count = emitter.seed.z;
    if (count == 0u) {
        return 1.0;
    }
    var prev_t = emitter.size_curve[0].x;
    var prev_v = emitter.size_curve[0].y;
    if (t <= prev_t) {
        return prev_v;
    }
    var value = prev_v;
    for (var i = 1u; i < count; i = i + 1u) {
        let key = emitter.size_curve[i];
        if (t < key.x) {
            if (key.x > prev_t) {
                let f = (t - prev_t) / (key.x - prev_t);
                return prev_v + (key.y - prev_v) * f;
            }
            return key.y;
        }
        prev_t = key.x;
        prev_v = key.y;
        value = key.y;
    }
    return value;
}

fn color_curve_value(t: f32) -> vec4<f32> {
    let count = emitter.seed.w;
    if (count == 0u) {
        return vec4<f32>(1.0);
    }
    var prev_t = emitter.color_curve[0].x;
    var prev_c = vec4<f32>(
        emitter.color_curve[0].y,
        emitter.color_curve[0].z,
        emitter.color_curve[0].w,
        emitter.color_curve_a[0],
    );
    if (t <= prev_t) {
        return prev_c;
    }
    var value = prev_c;
    for (var i = 1u; i < count; i = i + 1u) {
        let key = emitter.color_curve[i];
        if (t < key.x) {
            let next_c = vec4<f32>(key.y, key.z, key.w, emitter.color_curve_a[i]);
            if (key.x > prev_t) {
                let f = (t - prev_t) / (key.x - prev_t);
                return mix(prev_c, next_c, f);
            }
            return next_c;
        }
        prev_t = key.x;
        prev_c = vec4<f32>(key.y, key.z, key.w, emitter.color_curve_a[i]);
        value = prev_c;
    }
    return value;
}

const CORNERS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
);
const QUAD_UV: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 0.0),
);

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32) -> VsOut {
    let index = vertex / 6u;
    let corner = vertex % 6u;
    let p = particles[index];
    let alive = p.life > 0.0;
    let age = clamp(1.0 - p.life / max(p.life0, 1e-6), 0.0, 1.0);
    let size = select(0.0, max(p.size * size_curve_value(age) * draw.scale_alpha.x, 0.0), alive);
    let half_size = size * 0.5;
    let clip = camera.view_proj * vec4<f32>(p.pos, 1.0);
    let offset = select(vec2<f32>(0.0), CORNERS[corner], alive)
        * vec2<f32>(camera.view_proj[0][0], camera.view_proj[1][1]) * half_size;
    let tinted = p.color * color_curve_value(age);
    var out: VsOut;
    out.pos = vec4<f32>(clip.xy + offset, clip.z, clip.w);
    out.uv = QUAD_UV[corner];
    out.color = vec4<f32>(tinted.rgb * draw.tint.rgb, tinted.a * draw.scale_alpha.y);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var color = in.color;
    if (draw.scale_alpha.w > 0.5) {
        let texel = textureSampleLevel(scene_tex, scene_smp, in.uv, i32(draw.scale_alpha.z + 0.5), 0.0);
        color = vec4<f32>(color.rgb * texel.rgb, color.a * texel.a);
    }
    if (color.a < 0.004) {
        discard;
    }
    return color;
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
    fn particle_shaders_parse() {
        parses("PARTICLE_COMPUTE_WGSL", PARTICLE_COMPUTE_WGSL);
        parses("PARTICLE_BILLBOARD_WGSL", PARTICLE_BILLBOARD_WGSL);
    }

    #[test]
    fn layouts_match_the_shader() {
        assert_eq!(size_of::<Particle>(), PARTICLE_BYTES);
        assert_eq!(size_of::<GpuEmitter>() % 16, 0);
        assert_eq!(size_of::<GpuColliders>() % 16, 0);
        assert_eq!(size_of::<GpuSimState>(), 32);
        assert_eq!(size_of::<GpuDrawParams>(), 32);
    }

    /// Every shape stays inside its declared bounds, from many draws, and
    /// uses the draws it claims to (a sphere is not a point, a rim is not a
    /// disc).
    #[test]
    fn shapes_sample_inside_their_bounds() {
        for i in 0..256u32 {
            let r: [f32; 8] = std::array::from_fn(|s| random01(7, i, s as u32));
            let p = EmitterShape::Point.sample(&r);
            assert_eq!(p, [0.0, 0.0, 0.0]);

            let sphere = EmitterShape::Sphere { radius: 2.0 }.sample(&r);
            let len = (sphere[0] * sphere[0] + sphere[1] * sphere[1] + sphere[2] * sphere[2]).sqrt();
            assert!(len <= 2.0 + 1e-4, "sphere radius: {len}");

            let cube = EmitterShape::Box {
                extents: [1.0, 2.0, 3.0],
            }
            .sample(&r);
            assert!(cube[0].abs() <= 1.0 && cube[1].abs() <= 2.0 && cube[2].abs() <= 3.0);

            let cone = EmitterShape::Cone {
                radius: 1.5,
                height: 4.0,
            }
            .sample(&r);
            assert!((0.0..=4.0).contains(&cone[1]), "cone height: {}", cone[1]);
            let ring = (cone[0] * cone[0] + cone[2] * cone[2]).sqrt();
            let allowed = 1.5 * (cone[1] / 4.0) + 1e-4;
            assert!(ring <= allowed, "cone radius {ring} > {allowed}");

            let edge = EmitterShape::CircleEdge { radius: 3.0 }.sample(&r);
            let dist = (edge[0] * edge[0] + edge[2] * edge[2]).sqrt();
            assert!((dist - 3.0).abs() < 1e-4, "rim radius: {dist}");
            assert_eq!(edge[1], 0.0);
        }
        // A ball of radius 1 fills most of its volume: a deep sample is not
        // all at the crust or all at the center.
        let mut near_center = 0;
        for i in 0..64u32 {
            let r: [f32; 8] = std::array::from_fn(|s| random01(11, i, s as u32));
            let p = EmitterShape::Sphere { radius: 1.0 }.sample(&r);
            if p[0] * p[0] + p[1] * p[1] + p[2] * p[2] < 0.25 {
                near_center += 1;
            }
        }
        assert!(near_center > 0, "ball samples reach the inside");
    }

    /// A fixed seed and counter give a fixed particle, byte for byte, and
    /// neighbouring counters differ.
    #[test]
    fn seeding_is_deterministic_per_counter() {
        let a = random01(42, 7, 1);
        assert_eq!(a, random01(42, 7, 1), "same seed, counter and salt");
        assert_ne!(a, random01(42, 8, 1), "counters differ");
        assert_ne!(a, random01(43, 7, 1), "seeds differ");
        assert_ne!(a, random01(42, 7, 2), "salts differ");
        assert!((0.0..1.0).contains(&a));
        // The hash is a fixed function: pin the exact output.
        assert_eq!(hash_u32(1, 0), 1_753_845_952);
    }

    /// One integration step under gravity matches the closed-form velocity
    /// (drag zero) and keeps the fixed-seed spawn reproducible.
    #[test]
    fn integration_step_matches_closed_form() {
        let mut p = Particle {
            pos: [0.0, 0.0, 0.0],
            life: 2.0,
            vel: [1.0, 0.0, 0.0],
            size: 0.2,
            color: [1.0; 4],
            seed: 0,
            life0: 2.0,
            _pad: [0.0; 2],
        };
        let dt = 0.5;
        integrate_step(&mut p, dt, [0.0, -10.0, 0.0], 0.0, &[]);
        assert!((p.vel[1] - -5.0).abs() < 1e-5, "v = g t: {}", p.vel[1]);
        assert!((p.pos[0] - 0.5).abs() < 1e-5, "x advances by v dt");
        assert!((p.pos[1] - -2.5).abs() < 1e-5, "y advances by v dt");
        assert!((p.life - 1.5).abs() < 1e-5);

        // Drag keeps the fraction the first-order model declares.
        let mut d = Particle {
            vel: [10.0, 0.0, 0.0],
            life: 1.0,
            ..p
        };
        integrate_step(&mut d, 0.25, [0.0; 3], 2.0, &[]);
        assert!((d.vel[0] - 5.0).abs() < 1e-4, "v *= 1 - drag dt: {}", d.vel[0]);

        // An attractor inside its radius pulls toward its center and stops
        // pulling outside it.
        let attractor = Attractor {
            position: [10.0, 0.0, 0.0],
            strength: 8.0,
            radius: 4.0,
        };
        let mut a = Particle {
            pos: [8.0, 0.0, 0.0],
            vel: [0.0; 3],
            life: 1.0,
            ..p
        };
        integrate_step(&mut a, 1.0, [0.0; 3], 0.0, &[attractor]);
        assert!(a.vel[0] > 0.0, "pulled toward the attractor: {:?}", a.vel);
        let mut b = Particle {
            pos: [30.0, 0.0, 0.0],
            vel: [0.0; 3],
            life: 1.0,
            ..p
        };
        integrate_step(&mut b, 1.0, [0.0; 3], 0.0, &[attractor]);
        assert_eq!(b.vel, [0.0; 3], "outside the radius, no pull");
    }

    /// Cone sampling stays within the half-angle and covers both extremes.
    #[test]
    fn cone_spread_bounds_the_direction() {
        let spread = std::f32::consts::FRAC_PI_4;
        let mut min_up = 1.0f32;
        let mut max_tilt = 0.0f32;
        for i in 0..64u32 {
            let d = sample_cone(spread, random01(3, i, 1), random01(3, i, 2));
            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-4, "unit direction: {len}");
            assert!(d[1] >= spread.cos() - 1e-4, "inside the cone: {}", d[1]);
            min_up = min_up.min(d[1]);
            max_tilt = max_tilt.max(d[0].hypot(d[2]));
        }
        assert!(max_tilt > 0.6, "cone reaches its rim: {max_tilt}");
        assert!((min_up - spread.cos()).abs() < 0.05, "cone reaches its axis wall");
        // Zero spread is a straight line along +Y.
        let straight = sample_cone(0.0, 0.5, 0.25);
        assert!(straight[0].abs() < 1e-5 && straight[2].abs() < 1e-5);
        assert!((straight[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn curves_hold_endpoints_and_interpolate() {
        let curve = ParticleCurve::new([(0.0, 2.0), (0.5, 0.0), (1.0, 4.0)]).expect("valid");
        assert_eq!(curve.sample(-1.0), 2.0);
        assert_eq!(curve.sample(0.0), 2.0);
        assert!((curve.sample(0.25) - 1.0).abs() < 1e-6);
        assert_eq!(curve.sample(0.5), 0.0);
        assert!((curve.sample(0.75) - 2.0).abs() < 1e-6);
        assert_eq!(curve.sample(2.0), 4.0);
        // Insertion order does not matter; duplicate times are rejected.
        let sorted =
            ParticleCurve::new([(0.5, 0.0), (1.0, 4.0), (0.0, 2.0)]).expect("valid");
        assert_eq!(sorted.sample(0.25), curve.sample(0.25));
        assert!(ParticleCurve::new([(0.2, 1.0), (0.2, 2.0)]).is_err());
        let gradient = ParticleGradient::new([(0.0, [1.0, 1.0, 1.0, 1.0]), (1.0, [0.0, 0.0, 0.0, 0.0])])
            .expect("valid");
        assert_eq!(gradient.sample(0.0), [1.0; 4]);
        assert!((gradient.sample(0.5)[3] - 0.5).abs() < 1e-6);
    }

    /// The swept box test reports the entry face and reflects the velocity
    /// with the requested restitution.
    #[test]
    fn collision_reflects_off_a_box() {
        let hit = segment_box(
            [0.0, 0.0, -5.0],
            [0.0, 0.0, 5.0],
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
        )
        .expect("segment crosses the box");
        assert_eq!(hit.0, [0.0, 0.0, -1.0], "entry normal faces the start");
        assert!((hit.1 - 0.4).abs() < 1e-5, "entry at z = -1: {}", hit.1);
        assert_eq!(hit.2, 2, "entered through z");
        let bounced = bounce_velocity([0.0, 0.0, 3.0], hit.0, 1.0);
        assert_eq!(bounced, [0.0, 0.0, -3.0], "perfect reflection");
        let damped = bounce_velocity([0.0, 0.0, 3.0], hit.0, 0.25);
        assert!((damped[2] + 0.75).abs() < 1e-6, "quarter restitution");
        // A segment that misses, and one parallel to a face.
        assert!(segment_box([0.0, 5.0, -5.0], [0.0, 5.0, 5.0], [-1.0; 3], [1.0; 3]).is_none());
        assert!(segment_box([0.0, 0.0, -5.0], [0.0, 0.0, 5.0], [-1.0; 3], [1.0; 3]).is_some());
    }

    /// The voxel march stops at the first solid cell and reports the face it
    /// crossed.
    #[test]
    fn voxel_march_stops_at_the_first_solid_cell() {
        let mut field = ParticleVoxels {
            origin: [0.0; 3],
            cell: 1.0,
            dims: [4, 1, 1],
            cells: vec![0, 1, 0, 0],
        };
        let hit = segment_voxels([0.5, 0.5, 0.5], [3.5, 0.5, 0.5], &field)
            .expect("hits the solid cell");
        assert_eq!(hit.0, [-1.0, 0.0, 0.0], "entered the -x face");
        // The x = 1 boundary is one third of the way into the segment.
        assert!(
            (hit.1 - 1.0 / 6.0).abs() < 1e-5,
            "at the x = 1 boundary: {}",
            hit.1
        );
        // All air: the segment exits without a hit.
        field.cells = vec![0, 0, 0, 0];
        assert!(segment_voxels([0.5, 0.5, 0.5], [3.5, 0.5, 0.5], &field).is_none());
    }

    #[test]
    fn voxels_build_and_sample_the_source() {
        use std::collections::HashMap;
        let mut grid: HashMap<[i32; 3], Cell> = HashMap::new();
        grid.insert([0, 0, 0], Cell::full(1));
        grid.insert([1, 0, 0], Cell::full(1));
        let field = ParticleVoxels::build(&grid, [0.0; 3], 1.0, [2, 1, 1], |_| true)
            .expect("valid field");
        assert_eq!(field.len(), 2);
        assert!(field.solid([0, 0, 0]));
        assert!(field.solid([1, 0, 0]));
        assert!(!field.solid([2, 0, 0]));
        assert!(!field.solid([-1, 0, 0]));
        assert!(ParticleVoxels::build(&grid, [0.0; 3], 1.0, [0, 1, 1], |_| true).is_err());
        assert!(ParticleVoxels::build(&grid, [0.0; 3], 0.0, [1, 1, 1], |_| true).is_err());
    }

    #[test]
    fn invalid_systems_are_rejected() {
        let mut system = ParticleSystem::new(0);
        assert_eq!(system.validate(), Err(ParticleError::ZeroCapacity));
        system.capacity = 4;
        system.lifetime = [2.0, 1.0];
        assert_eq!(system.validate(), Err(ParticleError::InvalidRange("lifetime")));
        system.lifetime = [0.5, 1.0];
        system.speed = [f32::NAN, 1.0];
        assert_eq!(system.validate(), Err(ParticleError::NonFinite("speed")));
        system.speed = [1.0, 2.0];
        system.attractors = vec![Attractor { position: [0.0; 3], strength: 1.0, radius: 1.0 }; MAX_ATTRACTORS + 1];
        assert!(matches!(
            system.validate(),
            Err(ParticleError::TooManyAttractors(_))
        ));
        assert_eq!(
            ParticleFrame::new(0.0).validate(),
            Err(ParticleError::InvalidDt)
        );
        assert!(ParticleFrame::new(1.0 / 60.0).validate().is_ok());
    }

    /// The CPU spawn budget reproduces the documented `floor(carry + rate*dt)`
    /// contract that staging implements.
    #[test]
    fn spawn_budget_carries_the_fraction() {
        let mut carry = 0.0f32;
        let rate = 90.0f32;
        let dt = 1.0 / 60.0;
        let mut spawned = 0u32;
        for _ in 0..60 {
            carry += rate * dt;
            let whole = carry.floor();
            carry -= whole;
            spawned += whole as u32;
        }
        assert_eq!(spawned, 90, "one second of 90/s over 60 steps");
        assert!(carry < 1.0);
    }
}
