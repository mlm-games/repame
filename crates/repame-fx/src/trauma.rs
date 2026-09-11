//! Trauma shake: 0..1 impact memory with decay and a squared
//! response, Perlin-noise offsets applied to rendering only (never to
//! sim positions). Follows Bevy's `2d_screen_shake` example and the
//! retired `ScreenEffectsConfig` numbers (`trauma_decay` 1.5).

use bevy_ecs::prelude::*;
use noise::{NoiseFn, Perlin};

/// Screen-shake state. Plain data: the game owns one (resource, field,
/// whatever fits) and maps [`Trauma::offset`] onto its camera.
#[derive(Clone, Debug, Resource)]
pub struct Trauma {
    /// 0 (still) .. 1 (full shake). Clamped on add.
    pub amount: f32,
    /// Per-second decay (1.5 drains full trauma in ~0.67 s).
    pub decay_per_sec: f32,
    /// Max translation in px at full trauma.
    pub max_translation_px: f32,
    /// Max roll in radians at full trauma.
    pub max_roll_rad: f32,
    /// Noise traversal speed (arbitrary units per second).
    pub noise_speed: f32,
    seed: u32,
}

impl Default for Trauma {
    fn default() -> Self {
        Self {
            amount: 0.0,
            decay_per_sec: 1.5,
            max_translation_px: 20.0,
            max_roll_rad: 10.0f32.to_radians(),
            noise_speed: 20.0,
            seed: 1337,
        }
    }
}

impl Trauma {
    pub fn new() -> Self {
        Self::default()
    }

    /// Non-default noise seed: per-run shake variation, or two
    /// simultaneous sources that must not correlate. Same seed replays
    /// the same offsets (deterministic like the default).
    pub fn with_seed(seed: u32) -> Self {
        Self {
            seed,
            ..Self::default()
        }
    }

    fn noise(&self) -> Perlin {
        Perlin::new(self.seed)
    }

    /// Add impact. Clamped to 1.
    pub fn add(&mut self, amount: f32) {
        self.amount = (self.amount + amount).clamp(0.0, 1.0);
    }

    /// Decay over `ticks` (100 Hz). No-op at 0.
    pub fn decay(&mut self, ticks: i32) {
        if ticks <= 0 {
            return;
        }
        self.amount = (self.amount - self.decay_per_sec * ticks as f32 / 100.0).max(0.0);
    }

    /// `(dx_px, dy_px, roll_rad)` at `time_secs` (wall or sim time).
    /// Squared response: small hits barely move, big hits punch.
    /// Deterministic for the same inputs (headless-stable).
    pub fn offset(&self, time_secs: f32) -> (f32, f32, f32) {
        let shake = self.amount * self.amount;
        if shake <= 0.0 {
            return (0.0, 0.0, 0.0);
        }
        let noise = self.noise();
        let t = time_secs as f64 * self.noise_speed as f64;
        let nx = noise.get([t, 100.0]) as f32;
        let ny = noise.get([t, 200.0]) as f32;
        let nr = noise.get([t, 300.0]) as f32;
        (
            nx * shake * self.max_translation_px,
            ny * shake * self.max_translation_px,
            nr * shake * self.max_roll_rad,
        )
    }

    /// World-space camera offset for repame `Camera2d`.
    /// `units_per_pixel` and `zoom` match `Camera2d` fields.
    pub fn camera_offset(
        &self,
        time_secs: f32,
        units_per_pixel: f32,
        zoom: f32,
    ) -> (f32, f32, f32) {
        let (dx_px, dy_px, roll) = self.offset(time_secs);
        let upp = if units_per_pixel.is_finite() {
            units_per_pixel
        } else {
            1.0
        };
        let scale = (upp / zoom.max(1e-6)).max(0.0);
        (dx_px * scale, dy_px * scale, roll)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_clamps_and_decays() {
        let mut tr = Trauma::new();
        tr.add(0.4);
        assert_eq!(tr.amount, 0.4);
        tr.add(0.9);
        assert_eq!(tr.amount, 1.0);
        tr.decay(100);
        assert!(tr.amount <= 0.0, "1.5/s drains full trauma in a second");
    }

    #[test]
    fn zero_trauma_is_still() {
        let tr = Trauma::new();
        assert_eq!(tr.offset(3.7), (0.0, 0.0, 0.0));
    }

    #[test]
    fn offset_is_deterministic_and_bounded() {
        let mut tr = Trauma::new();
        tr.add(1.0);
        let a = tr.offset(1.23);
        let b = tr.offset(1.23);
        assert_eq!(a, b);
        assert!(a.0.abs() <= 20.0 && a.1.abs() <= 20.0);
        assert!(a.2.abs() <= 10.0f32.to_radians() + 1e-5);
    }

    #[test]
    fn response_is_squared() {
        let mut half = Trauma::new();
        half.add(0.5);
        let mut full = Trauma::new();
        full.add(1.0);
        let h = half.offset(0.7);
        let f = full.offset(0.7);
        assert!((h.0 / f.0 - 0.25).abs() < 1e-4);
    }

    #[test]
    fn seeds_vary_but_replay() {
        let mut a = Trauma::with_seed(1);
        let mut b = Trauma::with_seed(2);
        let mut a2 = Trauma::with_seed(1);
        a.add(1.0);
        b.add(1.0);
        a2.add(1.0);
        assert_eq!(a.offset(0.73), a2.offset(0.73));
        assert_ne!(a.offset(0.73), b.offset(0.73));
    }

    #[test]
    fn camera_offset_scales_px_to_world() {
        let mut tr = Trauma::new();
        tr.add(1.0);
        let (dx_px, dy_px, roll) = tr.offset(0.5);
        let (dx_w, dy_w, roll_w) = tr.camera_offset(0.5, 2.0, 2.0);
        assert!((dx_w - dx_px).abs() < 1e-5 && (dy_w - dy_px).abs() < 1e-5);
        assert_eq!(roll_w, roll);
        let (zx, zy, _) = tr.camera_offset(0.5, 1.0, 0.0);
        assert!(zx.is_finite() && zy.is_finite());
    }
}
