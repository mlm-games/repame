//! [`SoundBank`]: named cues with variations, cooldowns, polyphony caps.
//!
//! The game-idiomatic core: `bank.play("footstep")` picks a variation,
//! enforces per-cue cooldown and voice caps (stealing the oldest), and
//! sends a [`PlayCmd`](crate::PlayCmd) down the command channel. Time is
//! an explicit `now_ms` parameter on [`SoundBank::play_at_ms`] so tests
//! run on a fake clock; [`SoundBank::play`] stamps the real clock.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use web_workers::sync::mpsc::Sender;

use crate::command::{PlayCmd, RealtimeCommand, SharedFrames};
use crate::decode_bytes;
use crate::{AudioChannel, AudioState};

/// How a cue picks among its variations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Variation {
    /// Cycle in order (footsteps, UI ticks).
    #[default]
    RoundRobin,
    /// Seeded random (impacts, debris).
    Random,
}

/// Cue definition: everything except the sound data itself.
#[derive(Debug, Clone)]
pub struct CueDef {
    pub bus: AudioChannel,
    pub gain: f32,
    pub cooldown_ms: u64,
    pub max_voices: usize,
    /// Pitch wobble as a fraction (`0.05` = +/-5%).
    pub pitch_wobble: f32,
    pub variation: Variation,
}

impl Default for CueDef {
    fn default() -> Self {
        Self {
            bus: AudioChannel::Sfx,
            gain: 1.0,
            cooldown_ms: 0,
            max_voices: 8,
            pitch_wobble: 0.0,
            variation: Variation::RoundRobin,
        }
    }
}

/// One registered cue: decoded variations plus live-voice accounting.
struct Cue {
    def: CueDef,
    sounds: Vec<Arc<SharedFrames>>,
    rr_index: usize,
    rng: u64,
    last_play_ms: Option<u64>,
    live: Vec<u64>,
}

impl Cue {
    fn pick(&mut self) -> Arc<SharedFrames> {
        let i = match self.def.variation {
            Variation::RoundRobin => {
                let i = self.rr_index % self.sounds.len();
                self.rr_index += 1;
                i
            }
            Variation::Random => {
                // xorshift64star; deterministic per seed.
                self.rng ^= self.rng >> 12;
                self.rng ^= self.rng << 25;
                self.rng ^= self.rng >> 27;
                ((self.rng.wrapping_mul(0x2545F4914F6CDD1D) >> 32) as usize) % self.sounds.len()
            }
        };
        self.sounds[i].clone()
    }

    fn rate(&mut self) -> f32 {
        let w = self.def.pitch_wobble.max(0.0);
        if w <= 0.0 {
            return 1.0;
        }
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        let u = (self.rng >> 11) as f32 / (u64::MAX >> 11) as f32; // 0..1
        1.0 + (u * 2.0 - 1.0) * w
    }
}

/// 2D spatial shaping: distance falloff plus x-dominant pan.
/// Returns `(gain_scale, pan)`; dead center is `(1.0, 0.0)`.
pub fn spatial_2d(dx: f32, dy: f32, scale: f32) -> (f32, f32) {
    let dx = dx * scale;
    let dy = dy * scale;
    let dist = (dx * dx + dy * dy).sqrt();
    let gain = 1.0 / (1.0 + dist);
    let pan = (dx / (dx.abs() + dy.abs() + 1e-6)).clamp(-1.0, 1.0);
    (gain, pan)
}

/// Named-cue registry on the game thread. Silent without a link
/// (plays return `None`; definitions still validate).
pub struct SoundBank {
    cues: HashMap<String, Cue>,
    tx: Option<Sender<RealtimeCommand>>,
    state: Option<Arc<AudioState>>,
    seed: u64,
    boot: Instant,
    listener: [f32; 2],
    spatial_scale: f32,
}

impl SoundBank {
    pub fn new() -> Self {
        Self {
            cues: HashMap::new(),
            tx: None,
            state: None,
            seed: 0x9E3779B97F4A7C15,
            boot: Instant::now(),
            listener: [0.0, 0.0],
            spatial_scale: 1.0,
        }
    }

    /// Deterministic variation seed (tests; default is fixed anyway).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Wire to an audio thread (done by [`crate::Audio::try_init`]; call
    /// directly for custom plumbing or tests).
    pub fn attach(&mut self, tx: Sender<RealtimeCommand>, state: Arc<AudioState>) {
        self.tx = Some(tx);
        self.state = Some(state);
    }

    /// Register a cue from encoded files (Ogg/MP3/FLAC/WAV). Each file
    /// becomes one variation. Replaces any cue of the same name.
    pub fn load(&mut self, name: &str, def: CueDef, files: &[&[u8]]) -> Result<()> {
        if files.is_empty() {
            anyhow::bail!("cue `{name}` needs at least one file");
        }
        let mut sounds = Vec::with_capacity(files.len());
        for bytes in files {
            sounds.push(Arc::new(decode_bytes(bytes)?));
        }
        self.cues.insert(
            name.to_string(),
            Cue {
                def,
                sounds,
                rr_index: 0,
                rng: self.seed ^ name.len() as u64,
                last_play_ms: None,
                live: Vec::new(),
            },
        );
        Ok(())
    }

    /// Drop the finished ids from live-voice accounting. Called from
    /// [`crate::Audio::update`] with the drained completions.
    pub fn reap(&mut self, finished: &[u64]) {
        if finished.is_empty() {
            return;
        }
        for cue in self.cues.values_mut() {
            cue.live.retain(|id| !finished.contains(id));
        }
    }

    /// Current live-voice count for a cue (`0` when unknown).
    pub fn live_count(&self, name: &str) -> usize {
        self.cues.get(name).map(|c| c.live.len()).unwrap_or(0)
    }

    /// Fire-and-forget play. Returns the voice id, or `None` when the
    /// cue is unknown, cooling down, or there is no audio thread.
    pub fn play(&mut self, name: &str) -> Option<u64> {
        let now_ms = self.boot.elapsed().as_millis() as u64;
        self.play_at_ms(name, 1.0, 1.0, 0.0, now_ms)
            .map(|(id, _)| id)
    }

    /// Positional play in 2D world units (listener set via
    /// [`SoundBank::set_listener`]).
    pub fn play_at(&mut self, name: &str, x: f32, y: f32) -> Option<u64> {
        let now_ms = self.boot.elapsed().as_millis() as u64;
        let dx = x - self.listener[0];
        let dy = y - self.listener[1];
        let (g, pan) = spatial_2d(dx, dy, self.spatial_scale);
        self.play_at_ms(name, g, 1.0, pan, now_ms).map(|(id, _)| id)
    }

    /// Deterministic entry: explicit spatial shaping and clock.
    /// Returns `(voice, rate)` so tests can observe the wobble.
    pub fn play_at_ms(
        &mut self,
        name: &str,
        gain_scale: f32,
        rate_scale: f32,
        pan: f32,
        now_ms: u64,
    ) -> Option<(u64, f32)> {
        let (tx, state) = match (&self.tx, &self.state) {
            (Some(tx), Some(state)) => (tx, state),
            _ => return None,
        };
        let cue = self.cues.get_mut(name)?;
        if cue.sounds.is_empty() {
            return None;
        }
        if let Some(last) = cue.last_play_ms
            && now_ms.saturating_sub(last) < cue.def.cooldown_ms
        {
            return None;
        }
        // Steal the oldest voice past the cap.
        if cue.live.len() >= cue.def.max_voices.max(1)
            && let Some(oldest) = cue.live.first().copied()
        {
            let _ = tx.send_spin(RealtimeCommand::Stop(oldest));
            cue.live.remove(0);
        }
        let sound = cue.pick();
        let rate = cue.rate() * rate_scale.max(0.01);
        let id = state.alloc_voice();
        let cmd = PlayCmd {
            voice: id,
            sound,
            gain: cue.def.gain.max(0.0) * gain_scale.max(0.0),
            rate,
            pan: pan.clamp(-1.0, 1.0),
            bus: cue.def.bus,
            looping: false,
        };
        if tx.send_spin(RealtimeCommand::Play(cmd)).is_err() {
            return None;
        }
        cue.live.push(id);
        cue.last_play_ms = Some(now_ms);
        Some((id, rate))
    }

    /// Stop every live voice of a cue.
    pub fn stop_cue(&mut self, name: &str) {
        if let (Some(tx), Some(cue)) = (&self.tx, self.cues.get_mut(name)) {
            for id in cue.live.drain(..) {
                let _ = tx.send_spin(RealtimeCommand::Stop(id));
            }
        }
    }

    /// Scene listener in 2D world units.
    pub fn set_listener(&mut self, x: f32, y: f32) {
        self.listener = [x, y];
    }

    pub fn set_spatial_scale(&mut self, scale: f32) {
        self.spatial_scale = scale.max(0.0);
    }
}

impl Default for SoundBank {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{audio_link, synth_sine_wav};

    fn banked() -> (SoundBank, crate::GameAudioLink, crate::ThreadAudioLink) {
        let (game, thread) = audio_link();
        let mut bank = SoundBank::new().with_seed(42);
        bank.attach(game.tx.clone(), game.state.clone());
        (bank, game, thread)
    }

    fn wav(freq: f32) -> Vec<u8> {
        synth_sine_wav(freq, 0.05, 22050)
    }

    #[test]
    fn unknown_cue_is_silent_none() {
        let (mut bank, _game, _thread) = banked();
        assert_eq!(bank.play("nope"), None);
        assert_eq!(bank.live_count("nope"), 0);
    }

    #[test]
    fn cooldown_throttles_retrigger() {
        let (mut bank, _game, _thread) = banked();
        let a = wav(440.0);
        bank.load(
            "tick",
            CueDef {
                cooldown_ms: 100,
                ..Default::default()
            },
            &[&a],
        )
        .unwrap();
        assert!(bank.play_at_ms("tick", 1.0, 1.0, 0.0, 1000).is_some());
        assert!(bank.play_at_ms("tick", 1.0, 1.0, 0.0, 1050).is_none());
        assert!(bank.play_at_ms("tick", 1.0, 1.0, 0.0, 1100).is_some());
        assert_eq!(bank.live_count("tick"), 2);
    }

    #[test]
    fn polyphony_cap_steals_oldest() {
        let (mut bank, _game, thread) = banked();
        let a = wav(440.0);
        bank.load(
            "shot",
            CueDef {
                max_voices: 2,
                ..Default::default()
            },
            &[&a],
        )
        .unwrap();
        let v1 = bank.play_at_ms("shot", 1.0, 1.0, 0.0, 0).unwrap().0;
        let _v2 = bank.play_at_ms("shot", 1.0, 1.0, 0.0, 1).unwrap().0;
        assert_eq!(bank.live_count("shot"), 2);
        let v3 = bank.play_at_ms("shot", 1.0, 1.0, 0.0, 2).unwrap().0;
        assert_eq!(bank.live_count("shot"), 2);
        assert_ne!(v3, v1);
        // Command order on the thread side: Play, Play, Stop(oldest), Play.
        use crate::RealtimeCommand::*;
        assert!(matches!(thread.rx.try_recv(), Ok(Play(_))));
        assert!(matches!(thread.rx.try_recv(), Ok(Play(_))));
        assert!(matches!(thread.rx.try_recv(), Ok(Stop(id)) if id == v1));
        assert!(matches!(thread.rx.try_recv(), Ok(Play(_))));
        // v1 was already stolen out; reaping a live voice drops the count.
        bank.reap(&[v3]);
        assert_eq!(bank.live_count("shot"), 1);
    }

    #[test]
    fn pitch_wobble_is_deterministic() {
        let (mut bank, _game, _thread) = banked();
        let a = wav(440.0);
        bank.load(
            "debris",
            CueDef {
                pitch_wobble: 0.1,
                variation: Variation::Random,
                ..Default::default()
            },
            &[&a],
        )
        .unwrap();
        let rates: Vec<f32> = (0..8)
            .map(|t| bank.play_at_ms("debris", 1.0, 1.0, 0.0, t).unwrap().1)
            .collect();
        assert!(rates.iter().all(|r| (0.9..=1.1).contains(r)), "{rates:?}");
        assert!(rates.iter().any(|r| (*r - 1.0).abs() > 0.001), "{rates:?}");
        // Same seed replays the same sequence.
        let (mut bank2, _g2, _t2) = banked();
        bank2
            .load(
                "debris",
                CueDef {
                    pitch_wobble: 0.1,
                    variation: Variation::Random,
                    ..Default::default()
                },
                &[&a],
            )
            .unwrap();
        let rates2: Vec<f32> = (0..8)
            .map(|t| bank2.play_at_ms("debris", 1.0, 1.0, 0.0, t).unwrap().1)
            .collect();
        assert_eq!(rates, rates2);
    }

    #[test]
    fn spatial_shaping_behaves() {
        let (g, pan) = spatial_2d(0.0, 0.0, 1.0);
        assert!((g - 1.0).abs() < 1e-6 && pan.abs() < 1e-6);
        let (near, _) = spatial_2d(1.0, 0.0, 1.0);
        let (far, _) = spatial_2d(9.0, 0.0, 1.0);
        assert!((near - 0.5).abs() < 1e-6 && far < near);
        let (_, right) = spatial_2d(5.0, 0.0, 1.0);
        let (_, left) = spatial_2d(-5.0, 0.0, 1.0);
        assert!(right > 0.9 && left < -0.9);
    }

    #[test]
    fn stop_cue_clears_live() {
        let (mut bank, _game, _thread) = banked();
        let a = wav(440.0);
        bank.load("alarm", CueDef::default(), &[&a]).unwrap();
        bank.play_at_ms("alarm", 1.0, 1.0, 0.0, 0);
        bank.play_at_ms("alarm", 1.0, 1.0, 0.0, 1);
        assert_eq!(bank.live_count("alarm"), 2);
        bank.stop_cue("alarm");
        assert_eq!(bank.live_count("alarm"), 0);
    }
}
