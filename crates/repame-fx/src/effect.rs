//! Effect definitions: spawner + init + update + render in one
//! serde struct (hanabi's modifier chain, flattened for fixed-tick CPU).
//! Code-built today, `.fx.ron` files later without format changes.

use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};

/// Base value plus symmetric jitter, sampled with any RNG.
/// (hanabi `Value`, `bevy_particle_systems::JitteredValue`.)
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Jittered {
    pub base: f32,
    pub range: f32,
}

impl Jittered {
    pub fn exact(base: f32) -> Self {
        Self { base, range: 0.0 }
    }

    pub fn sample(&self, rng: &mut impl Rng) -> f32 {
        if self.range <= 0.0 {
            self.base
        } else {
            self.base + rng.random_range(-self.range..self.range)
        }
    }
}

/// Color keys over normalized life 0..1, lerped in linear RGBA.
/// (hanabi `ColorOverLifetimeModifier`, enoki gradients.)
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Gradient {
    pub keys: Vec<(f32, [f32; 4])>,
}

impl Gradient {
    pub fn solid(color: [f32; 4]) -> Self {
        Self {
            keys: vec![(0.0, color), (1.0, color)],
        }
    }

    pub fn fade_out(color: [f32; 4]) -> Self {
        let mut end = color;
        end[3] = 0.0;
        Self {
            keys: vec![(0.0, color), (1.0, end)],
        }
    }

    /// Sample at life fraction `t` (clamped 0..1). Empty gradient is white.
    pub fn sample(&self, t: f32) -> [f32; 4] {
        let t = t.clamp(0.0, 1.0);
        if self.keys.is_empty() {
            return [1.0, 1.0, 1.0, 1.0];
        }
        let mut prev = self.keys[0];
        if t <= prev.0 {
            return prev.1;
        }
        for &next in &self.keys[1..] {
            if t <= next.0 {
                let span = (next.0 - prev.0).max(f32::EPSILON);
                let f = (t - prev.0) / span;
                return lerp_rgba(prev.1, next.1, f);
            }
            prev = next;
        }
        prev.1
    }
}

fn lerp_rgba(a: [f32; 4], b: [f32; 4], f: f32) -> [f32; 4] {
    [
        a[0] + (b[0] - a[0]) * f,
        a[1] + (b[1] - a[1]) * f,
        a[2] + (b[2] - a[2]) * f,
        a[3] + (b[3] - a[3]) * f,
    ]
}

/// Size/alpha curve over life.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum EaseKind {
    #[default]
    Linear,
    QuadOut,
    CubicOut,
    BackOut,
}

impl EaseKind {
    /// Ease life fraction 0..1 into curve value 0..1.
    pub fn apply(self, t: f32) -> f32 {
        use repose_core::animation::Easing;
        let ease = match self {
            EaseKind::Linear => Easing::Linear,
            EaseKind::QuadOut => Easing::EaseOut,
            EaseKind::CubicOut => Easing::CubicOut,
            EaseKind::BackOut => Easing::BackOut,
        };
        ease.interpolate(t.clamp(0.0, 1.0))
    }
}

/// Spawn policy: continuous rate and/or one-shot burst cap.
/// (hanabi `SpawnerSettings::rate`, enoki spawner state.)
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SpawnerDef {
    /// Particles per second while active (0 = burst-only).
    pub rate_per_sec: f32,
    /// Hard cap on live particles per spawner.
    pub max_alive: usize,
}

impl Default for SpawnerDef {
    fn default() -> Self {
        Self {
            rate_per_sec: 0.0,
            max_alive: 64,
        }
    }
}

/// Full effect: init (speed/lifetime/size jitter) + update
/// (gravity/drag) + render (gradient/size curve). One struct from
/// code today, from `.fx.ron` tomorrow.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EffectDef {
    pub spawner: SpawnerDef,
    pub speed_pps: Jittered,
    pub lifetime_ticks: Jittered,
    pub size_px: Jittered,
    pub gravity_pps2: f32,
    pub drag_per_sec: f32,
    pub gradient: Gradient,
    pub ease: EaseKind,
}

impl Default for EffectDef {
    fn default() -> Self {
        Self {
            spawner: SpawnerDef::default(),
            speed_pps: Jittered::exact(60.0),
            lifetime_ticks: Jittered::exact(40.0),
            size_px: Jittered::exact(6.0),
            gravity_pps2: 0.0,
            drag_per_sec: 0.0,
            gradient: Gradient::fade_out([1.0, 1.0, 1.0, 1.0]),
            ease: EaseKind::QuadOut,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_samples_keys() {
        let g = Gradient {
            keys: vec![(0.0, [1.0, 0.0, 0.0, 1.0]), (1.0, [0.0, 0.0, 1.0, 0.0])],
        };
        assert_eq!(g.sample(0.0), [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(g.sample(1.0), [0.0, 0.0, 1.0, 0.0]);
        let mid = g.sample(0.5);
        assert!((mid[0] - 0.5).abs() < 1e-5 && (mid[2] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn gradient_empty_is_white() {
        assert_eq!(Gradient::default().sample(0.3), [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn ease_endpoints_hold() {
        for ease in [
            EaseKind::Linear,
            EaseKind::QuadOut,
            EaseKind::CubicOut,
            EaseKind::BackOut,
        ] {
            assert!(ease.apply(0.0).abs() < 1e-4);
            assert!((ease.apply(1.0) - 1.0).abs() < 1e-4);
        }
        assert!(EaseKind::QuadOut.apply(0.5) > 0.5);
    }

    #[test]
    fn jittered_exact_needs_no_rng() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        assert_eq!(Jittered::exact(3.0).sample(&mut rng), 3.0);
        let j = Jittered {
            base: 10.0,
            range: 2.0,
        };
        for _ in 0..32 {
            let v = j.sample(&mut rng);
            assert!((8.0..12.0).contains(&v));
        }
    }
}
