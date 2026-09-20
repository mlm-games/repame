//! Effect definitions: spawner plus init, update, and render in one serde struct.
//! Built from code today; `.fx.ron` files later without format change.

use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};

/// Base value plus symmetric jitter.
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
/// Keys must be sorted ascending; unsorted keys trip a `debug_assert`.
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
        debug_assert!(
            self.keys.windows(2).all(|w| w[0].0 <= w[1].0),
            "gradient keys must be sorted ascending by time"
        );
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

/// Size and alpha curve over life. Shared with the UI tween family
/// so `.fx.ron` curves and UI tweens use one easing set.
pub use repose_core::animation::EaseKind;

/// Spawn policy: continuous rate plus one-shot burst cap.
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

/// Full effect: init jitter plus gravity, drag, gradient, size curve.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EffectDef {
    pub spawner: SpawnerDef,
    pub speed_pps: Jittered,
    /// Particle life in seconds. Accepts the legacy `lifetime_ticks` name
    /// (100 Hz quanta) on load and converts it. Heuristic boundary: values
    /// above 5.0 are treated as ticks (a 5 s particle is already absurdly
    /// long; a 5-tick life is common), values at or below as seconds.
    /// Ambiguous inputs (`base: 4.0` meaning 4 ticks = 0.04 s) misconvert:
    /// prefer writing new files with `lifetime_secs`.
    #[serde(alias = "lifetime_ticks", deserialize_with = "de_lifetime_secs")]
    pub lifetime_secs: Jittered,
    pub size_px: Jittered,
    pub gravity_pps2: f32,
    pub drag_per_sec: f32,
    pub gradient: Gradient,
    pub ease: EaseKind,
    /// Atlas page sampled by this effect's particles (default 0).
    #[serde(default)]
    pub page: u32,
    /// Atlas sub-rect: full page by default, one cell for textured puffs.
    #[serde(default = "uv_min_default")]
    pub uv_min: [f32; 2],
    #[serde(default = "uv_max_default")]
    pub uv_max: [f32; 2],
}

fn de_lifetime_secs<'de, D>(d: D) -> Result<Jittered, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum TicksOrSecs {
        Map { base: f32, range: f32 },
        Raw(f32),
    }
    let v = TicksOrSecs::deserialize(d)?;
    let conv = |base: f32, range: f32| {
        if base > 5.0 || range > 5.0 {
            Jittered {
                base: base / 100.0,
                range: range / 100.0,
            }
        } else {
            Jittered { base, range }
        }
    };
    match v {
        TicksOrSecs::Map { base, range } => Ok(conv(base, range)),
        TicksOrSecs::Raw(f) => Ok(Jittered::exact(if f > 5.0 { f / 100.0 } else { f })),
    }
}

fn uv_min_default() -> [f32; 2] {
    [0.0, 0.0]
}

fn uv_max_default() -> [f32; 2] {
    [1.0, 1.0]
}

impl Default for EffectDef {
    fn default() -> Self {
        Self {
            spawner: SpawnerDef::default(),
            speed_pps: Jittered::exact(60.0),
            lifetime_secs: Jittered::exact(0.4),
            size_px: Jittered::exact(6.0),
            gravity_pps2: 0.0,
            drag_per_sec: 0.0,
            gradient: Gradient::fade_out([1.0, 1.0, 1.0, 1.0]),
            ease: EaseKind::QuadOut,
            page: 0,
            uv_min: uv_min_default(),
            uv_max: uv_max_default(),
        }
    }
}

impl EffectDef {
    /// Legacy `lifetime_ticks` view (100 Hz quanta, rounded up).
    pub fn lifetime_ticks(&self) -> Jittered {
        Jittered {
            base: (self.lifetime_secs.base * 100.0).ceil().max(1.0),
            range: (self.lifetime_secs.range * 100.0).ceil().max(0.0),
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
