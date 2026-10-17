//! 3D spatialization: [`AudioListener3D`], attenuation models, panning, and
//! Doppler.
//!
//! This is a pure game-side calculation, the same shape as
//! [`crate::spatial_2d`]: the caller positions a listener and a source, calls
//! [`spatial_3d`], and applies the returned linear gain, pan, and pitch to a
//! playing voice. Nothing here touches the audio thread.
//!
//! The attenuation curves, the stereo panning law, and the Doppler formula
//! follow Godot's `scene/3d/audio_stream_player_3d.cpp`:
//! - `inverse_distance`: `linear_to_db(1 / (d/unit + eps))`
//! - `inverse_square`: `linear_to_db(1 / ((d/unit)^2 + eps))`
//! - `exponential`: amplitude `e^(-d/unit)`, i.e.
//!   `-20 * (d/unit) * log10(e)` dB
//! - `linear`: `1 - d/max_distance` in amplitude, reaching zero exactly at
//!   `max_distance`; with no `max_distance` it falls back to
//!   `inverse_distance`
//! - `disabled`: no distance falloff at all
//!
//! Every dB model clamps at `max_db` (so a point-blank source cannot clip the
//! bus), and `max_distance` both tapers the dB models towards zero and makes
//! anything beyond it inaudible.
//!
//! Panning is Godot's stereo law: the source direction is flattened to the
//! horizontal plane, and `panning_strength` shrinks the left/right weight
//! ratio towards centre. `panning_strength = 1` puts a source at a right angle
//! fully in one channel; `0` keeps it centred.

use crate::channels::{db_to_linear, linear_to_db};

/// Godot's `CMP_EPSILON`, guarding the logarithm at zero distance.
pub const DISTANCE_EPSILON: f32 = 1e-5;

/// Speed of sound in m/s, as in Godot's Doppler formula.
pub const SPEED_OF_SOUND: f32 = 343.0;

/// Plain `f32` vector, so this crate needs no math dependency.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn dot(self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn cross(self, o: Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn normalize(self) -> Self {
        let len = self.length();
        if len <= 1e-9 {
            Self::ZERO
        } else {
            Self::new(self.x / len, self.y / len, self.z / len)
        }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

/// Distance falloff shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttenuationModel {
    /// `1 - d/max_distance`, zero at `max_distance`.
    Linear,
    /// `e^(-d/unit)`: fast but finite at the unit distance.
    Exponential,
    /// `1/(d/unit)`: 6 dB per doubling.
    #[default]
    InverseDistance,
    /// `1/(d/unit)^2`: 12 dB per doubling.
    InverseSquare,
    /// No falloff; only `volume_db` and `max_db` apply.
    Disabled,
}

/// Everything a positional 3D player brings to the calculation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialParams {
    pub model: AttenuationModel,
    /// Distance at which the model's reference gain is unity.
    pub unit_size: f32,
    /// Ceiling on the boosted attenuation in dB (Godot's `max_db`).
    pub max_db: f32,
    /// Player volume in dB, added before the `max_db` clamp.
    pub volume_db: f32,
    /// Beyond this distance the source is silent; `0` means unbounded.
    pub max_distance: f32,
    /// `1` is full left/right, `0` is centred.
    pub panning_strength: f32,
    pub doppler: bool,
}

impl Default for SpatialParams {
    fn default() -> Self {
        Self {
            model: AttenuationModel::InverseDistance,
            unit_size: 10.0,
            max_db: 3.0,
            volume_db: 0.0,
            max_distance: 0.0,
            panning_strength: 1.0,
            doppler: false,
        }
    }
}

/// Result of [`spatial_3d`], ready to apply to a voice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialOutput {
    /// Linear gain (attenuation, `volume_db`, `max_distance` taper).
    pub gain: f32,
    /// Stereo position `-1..=1`.
    pub pan: f32,
    /// Playback rate multiplier (Doppler; `1.0` when disabled or static).
    pub pitch: f32,
}

/// A live voice whose gain, pan, and pitch are recomputed from the listener
/// every frame, so attenuation, panning, and Doppler track a moving source.
///
/// [`crate::Audio::update`] re-evaluates every tracked voice and sends only
/// the changes, and drops a voice once the engine reports it finished.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialVoice3D {
    pub voice: u64,
    pub source: Vec3,
    /// Source velocity in world units per second, for Doppler.
    pub velocity: Vec3,
    pub params: SpatialParams,
    /// The cue's own gain, kept so a re-target does not lose it.
    pub base_gain: f32,
    /// Last applied output, so unchanged voices send nothing.
    pub last: SpatialOutput,
}

/// The scene listener.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioListener3D {
    pub position: Vec3,
    /// Look direction (Godot's forward is `-Z`).
    pub forward: Vec3,
    pub up: Vec3,
    /// Listener velocity in world units per second, for Doppler.
    pub velocity: Vec3,
}

impl Default for AudioListener3D {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioListener3D {
    /// At the origin, looking down `-Z` with `+Y` up.
    pub fn new() -> Self {
        Self {
            position: Vec3::ZERO,
            forward: Vec3::new(0.0, 0.0, -1.0),
            up: Vec3::new(0.0, 1.0, 0.0),
            velocity: Vec3::ZERO,
        }
    }

    /// Place the listener and aim it.
    pub fn with_transform(position: Vec3, forward: Vec3, up: Vec3) -> Self {
        Self {
            position,
            forward,
            up,
            velocity: Vec3::ZERO,
        }
    }

    /// Listener velocities for Doppler (`0` disables the term).
    pub fn with_velocity(mut self, velocity: Vec3) -> Self {
        self.velocity = velocity;
        self
    }

    /// Orthonormal listener basis: `(right, up, back)`, back being `-forward`.
    fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let back = (-self.forward).normalize();
        let back = if back == Vec3::ZERO {
            Vec3::new(0.0, 0.0, 1.0)
        } else {
            back
        };
        let up = if self.up == Vec3::ZERO {
            Vec3::new(0.0, 1.0, 0.0)
        } else {
            self.up
        };
        let right = up.cross(back).normalize();
        let right = if right == Vec3::ZERO {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            right
        };
        let up = back.cross(right).normalize();
        (right, up, back)
    }

    /// Express a world-space delta in the listener's local frame.
    pub fn to_local(&self, v: Vec3) -> Vec3 {
        let (right, up, back) = self.basis();
        Vec3::new(right.dot(v), up.dot(v), back.dot(v))
    }
}

/// Distance falloff in linear gain, before panning and Doppler.
pub fn attenuation_gain(params: &SpatialParams, distance: f32) -> f32 {
    let unit = params.unit_size.max(1e-6);
    let d = distance.max(0.0);
    let att_db = match params.model {
        AttenuationModel::Disabled => 0.0,
        AttenuationModel::InverseDistance => -20.0 * (d / unit + DISTANCE_EPSILON).log10(),
        AttenuationModel::InverseSquare => {
            -20.0 * ((d / unit) * (d / unit) + DISTANCE_EPSILON).log10()
        }
        AttenuationModel::Exponential => {
            -20.0 * (d / unit) * std::f32::consts::LOG10_E
        }
        AttenuationModel::Linear => {
            if params.max_distance > 0.0 {
                linear_to_db((1.0 - d / params.max_distance).clamp(0.0, 1.0))
            } else {
                -20.0 * (d / unit + DISTANCE_EPSILON).log10()
            }
        }
    };
    db_to_linear((att_db + params.volume_db).min(params.max_db))
}

/// Full positional calculation for one source.
pub fn spatial_3d(
    listener: &AudioListener3D,
    source: Vec3,
    source_velocity: Vec3,
    params: &SpatialParams,
) -> SpatialOutput {
    let offset = source - listener.position;
    let distance = offset.length();
    if params.max_distance > 0.0 && distance > params.max_distance {
        return SpatialOutput {
            gain: 0.0,
            pan: 0.0,
            pitch: 1.0,
        };
    }
    let mut gain = attenuation_gain(params, distance);
    if params.max_distance > 0.0 && params.model != AttenuationModel::Linear {
        gain *= (1.0 - distance / params.max_distance).clamp(0.0, 1.0);
    }

    // Godot's stereo panning law, applied to the flattened source direction.
    let local = listener.to_local(offset);
    let flat = (local.x * local.x + local.z * local.z).sqrt();
    let cosx = if flat <= 1e-9 {
        0.0
    } else {
        (local.x / flat).clamp(-1.0, 1.0)
    };
    let strength = params.panning_strength.clamp(0.0, 1.0);
    let g = (1.0 - strength) * (1.0 - strength);
    let f = (1.0 - g) / (1.0 + g);
    let fcosx = cosx * f;
    let pan = (4.0 / std::f32::consts::PI) * (((1.0 - fcosx) / 2.0).sqrt()).acos() - 1.0;

    let pitch = if params.doppler {
        doppler_pitch(listener, offset, distance, source_velocity)
    } else {
        1.0
    };

    SpatialOutput { gain, pan, pitch }
}

/// Godot's Doppler: pitch ratio `c / (c + v_radial)`, clamped to `1/8..8`.
/// `v_radial` is the source's velocity along the listener-to-source axis, in
/// the listener's frame, so a receding source pitches down and an approaching
/// one pitches up.
pub fn doppler_pitch(
    listener: &AudioListener3D,
    offset: Vec3,
    distance: f32,
    source_velocity: Vec3,
) -> f32 {
    if distance <= 1e-6 {
        return 1.0;
    }
    let local_velocity = listener.to_local(source_velocity - listener.velocity);
    let speed = local_velocity.length();
    if speed <= 1e-6 {
        return 1.0;
    }
    let approaching = offset.normalize().dot(local_velocity.normalize());
    (SPEED_OF_SOUND / (SPEED_OF_SOUND + speed * approaching)).clamp(1.0 / 8.0, 8.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    /// Golden attenuation curve: Godot's models at their reference distances,
    /// with `unit_size = 10`, no volume offset, and a `max_db` of `3`.
    #[test]
    fn attenuation_curves_are_pinned() {
        let p = SpatialParams::default();
        // Unity at the unit distance, 6 dB down at two units.
        assert!(close(attenuation_gain(&p, 10.0), 0.99999, 1e-4));
        assert!(close(attenuation_gain(&p, 20.0), 0.499997, 1e-4));
        assert!(close(attenuation_gain(&p, 100.0), 0.1, 1e-4));
        // Point-blank is clamped by max_db (3 dB).
        assert!(close(attenuation_gain(&p, 0.0), db_to_linear(3.0), 1e-5));

        let square = SpatialParams {
            model: AttenuationModel::InverseSquare,
            ..p
        };
        assert!(close(attenuation_gain(&square, 10.0), 0.99999, 1e-4));
        assert!(close(attenuation_gain(&square, 20.0), 0.25, 1e-4));
        assert!(close(attenuation_gain(&square, 40.0), 0.0625, 1e-4));

        let exp = SpatialParams {
            model: AttenuationModel::Exponential,
            ..p
        };
        assert!(close(attenuation_gain(&exp, 0.0), 1.0, 1e-5));
        assert!(close(attenuation_gain(&exp, 10.0), 0.36787945, 1e-4));
        assert!(close(attenuation_gain(&exp, 20.0), 0.13533528, 1e-4));

        // Linear reaches exactly zero at max_distance.
        let lin = SpatialParams {
            model: AttenuationModel::Linear,
            max_distance: 50.0,
            ..p
        };
        assert!(close(attenuation_gain(&lin, 0.0), 1.0, 1e-5));
        assert!(close(attenuation_gain(&lin, 25.0), 0.5, 1e-5));
        assert_eq!(attenuation_gain(&lin, 50.0), 0.0);

        let off = SpatialParams {
            model: AttenuationModel::Disabled,
            ..p
        };
        assert!(close(attenuation_gain(&off, 1000.0), 1.0, 1e-6));
    }

    /// Golden panning: a right-angle source is fully in the right channel,
    /// a 45-degree source sits at `0.5`, and a source behind the listener is
    /// centred because panning ignores elevation and front/back.
    #[test]
    fn panning_law_is_pinned() {
        let listener = AudioListener3D::new();
        let p = SpatialParams::default();
        let at = |x, y, z| spatial_3d(&listener, Vec3::new(x, y, z), Vec3::ZERO, &p);
        assert!(close(at(10.0, 0.0, 0.0).pan, 1.0, 1e-5));
        assert!(close(at(-10.0, 0.0, 0.0).pan, -1.0, 1e-5));
        assert!(close(at(0.0, 0.0, 10.0).pan, 0.0, 1e-5), "behind is centred");
        assert!(close(at(10.0, 0.0, 10.0).pan, 0.5, 1e-5), "45 degrees");
        let centred = SpatialParams {
            panning_strength: 0.0,
            ..p
        };
        let out = spatial_3d(&listener, Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO, &centred);
        assert!(close(out.pan, 0.0, 1e-6), "strength 0 disables panning");
    }

    /// Golden Doppler: `343/(343+v)` for a receding source and `343/(343-v)`
    /// for an approaching one.
    #[test]
    fn doppler_ratios_are_pinned() {
        let listener = AudioListener3D::new();
        let p = SpatialParams {
            doppler: true,
            model: AttenuationModel::Disabled,
            ..SpatialParams::default()
        };
        let source = Vec3::new(0.0, 0.0, 10.0);
        let away = spatial_3d(&listener, source, Vec3::new(0.0, 0.0, 34.3), &p);
        assert!(close(away.pitch, 0.9090909, 1e-5), "{}", away.pitch);
        let toward = spatial_3d(&listener, source, Vec3::new(0.0, 0.0, -34.3), &p);
        assert!(close(toward.pitch, 1.1111112, 1e-5), "{}", toward.pitch);
        // A static source never shifts pitch.
        let still = spatial_3d(&listener, source, Vec3::ZERO, &p);
        assert!(close(still.pitch, 1.0, 1e-6));
        // Doppler off ignores motion entirely.
        let off = SpatialParams {
            doppler: false,
            ..p
        };
        assert!(close(
            spatial_3d(&listener, source, Vec3::new(0.0, 0.0, 300.0), &off).pitch,
            1.0,
            1e-6
        ));
    }

    /// `max_distance` cuts the source off entirely, and the taper reaches
    /// zero before that for the dB models.
    #[test]
    fn max_distance_gates_and_tapers() {
        let listener = AudioListener3D::new();
        let p = SpatialParams {
            max_distance: 20.0,
            ..SpatialParams::default()
        };
        let out = spatial_3d(&listener, Vec3::new(0.0, 0.0, 25.0), Vec3::ZERO, &p);
        assert_eq!(out.gain, 0.0);
        let near = spatial_3d(&listener, Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO, &p);
        let mid = spatial_3d(&listener, Vec3::new(0.0, 0.0, 15.0), Vec3::ZERO, &p);
        assert!(near.gain > mid.gain && mid.gain > 0.0);
    }

    #[test]
    fn listener_frame_rotates_with_the_transform() {
        // Facing +X means a source at +X is dead ahead, and so centred.
        let listener =
            AudioListener3D::with_transform(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let p = SpatialParams::default();
        let ahead = spatial_3d(&listener, Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO, &p);
        assert!(close(ahead.pan, 0.0, 1e-5));
        // A source at -Y is below, which the stereo law flattens to centre.
        let lateral = spatial_3d(&listener, Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO, &p);
        assert!(close(lateral.pan, 1.0, 1e-5), "{}", lateral.pan);
    }

    #[test]
    fn vector_helpers_are_sane() {
        assert_eq!(Vec3::ZERO.length(), 0.0);
        assert!(close(Vec3::new(3.0, 4.0, 0.0).length(), 5.0, 1e-6));
        assert_eq!(Vec3::new(9.0, 0.0, 0.0).normalize(), Vec3::new(1.0, 0.0, 0.0));
        let c = Vec3::new(1.0, 0.0, 0.0).cross(Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(c, Vec3::new(0.0, 0.0, 1.0));
    }
}
