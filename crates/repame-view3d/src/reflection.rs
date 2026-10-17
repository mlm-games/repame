//! Reflection probes: a cube map captured at a point, read by the material
//! path as an ambient + specular source with a falloff volume.
//!
//! A [`ProbeSettings`] describes where the cube is captured, how large its
//! influence is, and how bright its two contributions are:
//!
//! ```text
//! position   capture point (cube camera origin)
//! range      falloff radius; influence fades to zero at `range`
//! intensity  overall multiplier on the sampled radiance
//! ambient    how much of the cube feeds the diffuse/ambient term
//! specular   how much feeds the view-reflected specular term
//! resolution cube face edge in pixels, clamped 16..=512
//! interval   capture every N frames (1 = every frame)
//! ```
//!
//! The capture is a real render: the batch renders the frame's opaque,
//! depth-tested groups of the shared scene pass into the six faces with a
//! camera at `position` (fog, shadows and cascades off, so the probe holds
//! albedo/lighting rather than the main camera's post state). Groups routed
//! to a declared pass and particle billboards are not part of the cube; every
//! lit fragment samples it, with [`probe_weight`] as the falloff:
//!
//! ```text
//! weight = 1 inside range/2, smoothstep to 0 at `range`
//! ```
//!
//! Cost: six scene replays per capture per probe, so `interval` exists -
//! a static probe captures once and keeps its cube, a moving one re-captures
//! on its interval. Probe radiance is not prefiltered (single mip), so the
//! specular term is multiplied by `1 - roughness` and reads as a sharp
//! reflection on smooth surfaces.

/// Probes one frame accepts; each costs six render passes per capture.
pub const MAX_PROBES: usize = 4;

/// Smallest cube face edge a probe accepts.
pub const MIN_PROBE_RESOLUTION: u32 = 16;

/// Largest cube face edge a probe accepts.
pub const MAX_PROBE_RESOLUTION: u32 = 512;

/// One reflection probe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeSettings {
    /// Capture point, world space.
    pub position: [f32; 3],
    /// Influence radius in world units.
    pub range: f32,
    /// Multiplier on the sampled radiance. `0` disables the probe.
    pub intensity: f32,
    /// Weight of the cube's ambient term (diffuse fill).
    pub ambient: f32,
    /// Weight of the cube's specular term (view reflection).
    pub specular: f32,
    /// Cube face edge in pixels, clamped to
    /// [`MIN_PROBE_RESOLUTION`]`..=`[`MAX_PROBE_RESOLUTION`].
    pub resolution: u32,
    /// Capture every `interval` frames; `1` re-captures every frame.
    pub interval: u32,
}

impl Default for ProbeSettings {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            range: 30.0,
            intensity: 1.0,
            ambient: 0.5,
            specular: 1.0,
            resolution: 128,
            interval: 1,
        }
    }
}

impl ProbeSettings {
    /// A probe centered on `position` with the default look.
    pub fn new(position: [f32; 3]) -> Self {
        Self {
            position,
            ..Self::default()
        }
    }

    /// Face edge actually allocated.
    pub fn clamped_resolution(&self) -> u32 {
        self.resolution.clamp(MIN_PROBE_RESOLUTION, MAX_PROBE_RESOLUTION)
    }

    /// Capture period in frames; `0` behaves as `1`.
    pub fn clamped_interval(&self) -> u32 {
        self.interval.max(1)
    }

    /// True when this probe contributes anything.
    pub fn is_active(&self) -> bool {
        self.intensity > 0.0 && self.range > 0.0 && (self.ambient > 0.0 || self.specular > 0.0)
    }

    /// Rejects non-finite settings before they reach the uniform.
    pub fn validate(&self) -> Result<(), ProbeError> {
        let finite = |v: f32| v.is_finite();
        if !self.position.iter().all(|v| finite(*v))
            || !finite(self.range)
            || !finite(self.intensity)
            || !finite(self.ambient)
            || !finite(self.specular)
        {
            return Err(ProbeError::NonFinite);
        }
        Ok(())
    }
}

/// Why a probe set was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeError {
    /// A field held NaN or an infinity.
    NonFinite,
}

impl std::fmt::Display for ProbeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite => write!(f, "reflection probe has a non-finite field"),
        }
    }
}

impl std::error::Error for ProbeError {}

/// Falloff volume the shader applies at distance `distance` from a probe of
/// radius `range`: `1` out to half the range, then a smooth drop to `0` at
/// the range. Mirrors the WGSL `repame_probe_weight`.
pub fn probe_weight(distance: f32, range: f32) -> f32 {
    if !distance.is_finite() || !range.is_finite() || range <= 0.0 {
        return 0.0;
    }
    let inner = range * 0.5;
    if distance <= inner {
        return 1.0;
    }
    if distance >= range {
        return 0.0;
    }
    let t = (distance - inner) / (range - inner);
    1.0 - t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The falloff is full inside half the range, zero at the range, and
    /// strictly decreasing between.
    #[test]
    fn falloff_volume_is_monotonic() {
        let range = 20.0;
        assert_eq!(probe_weight(0.0, range), 1.0);
        assert_eq!(probe_weight(10.0, range), 1.0);
        assert_eq!(probe_weight(20.0, range), 0.0);
        assert_eq!(probe_weight(40.0, range), 0.0);
        let mid = probe_weight(15.0, range);
        assert!(mid > 0.0 && mid < 1.0, "halfway: {mid}");
        assert_eq!(mid, 0.5, "smoothstep midpoint");
        assert!(probe_weight(12.0, range) > mid);
        // Degenerate ranges never contribute.
        assert_eq!(probe_weight(1.0, 0.0), 0.0);
        assert_eq!(probe_weight(1.0, -3.0), 0.0);
        assert_eq!(probe_weight(f32::NAN, range), 0.0);
    }

    #[test]
    fn settings_clamp_and_validate() {
        let mut probe = ProbeSettings::new([1.0, 2.0, 3.0]);
        assert_eq!(probe.clamped_resolution(), 128);
        probe.resolution = 4;
        assert_eq!(probe.clamped_resolution(), MIN_PROBE_RESOLUTION);
        probe.resolution = 4096;
        assert_eq!(probe.clamped_resolution(), MAX_PROBE_RESOLUTION);
        probe.interval = 0;
        assert_eq!(probe.clamped_interval(), 1);
        assert!(probe.is_active());
        probe.intensity = 0.0;
        assert!(!probe.is_active());
        let mut bad = ProbeSettings::default();
        bad.position[1] = f32::NAN;
        assert_eq!(bad.validate(), Err(ProbeError::NonFinite));
    }
}
