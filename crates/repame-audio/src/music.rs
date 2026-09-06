//! [`Music`]: track director with crossfade, intensity stems, ducking.
//!
//! Game-idiomatic music: `play("battle", fade)` crossfades from whatever
//! runs; `set_intensity` mixes layered stems by windows; `duck` dips
//! under dialog or stingers and releases. All motion is [`Fade`](crate::RealtimeCommand::Fade)
//! commands on looping music-bus voices  - the engine owns the ramps,
//! this owns the policy. `update(dt)` advances the duck envelope.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use web_workers::sync::mpsc::Sender;

use crate::command::{PlayCmd, RealtimeCommand, SharedFrames};
use crate::decode_bytes;
use crate::{AudioChannel, AudioState};

/// One intensity stem: audible across the `[lo, hi]` intensity window
/// with a linear ramp at both edges.
#[derive(Debug, Clone)]
pub struct StemDef {
    pub lo: f32,
    pub hi: f32,
    pub gain: f32,
}

struct Stem {
    sound: Arc<SharedFrames>,
    lo: f32,
    hi: f32,
    gain: f32,
}

struct Track {
    gain: f32,
    main: Arc<SharedFrames>,
    stems: Vec<Stem>,
}

struct Playing {
    name: String,
    /// `(voice, stem_index)`; `None` stem is the main mix.
    voices: Vec<(u64, Option<usize>)>,
}

/// Duck envelope state.
#[derive(Debug, Clone, Copy)]
enum DuckEnv {
    Idle,
    Hold {
        left_secs: f32,
    },
    Release {
        from: f32,
        left_secs: f32,
        total_secs: f32,
    },
}

/// Looping-track director on the game thread. Silent without a link.
pub struct Music {
    tracks: HashMap<String, Track>,
    current: Option<Playing>,
    intensity: f32,
    duck_factor: f32,
    duck_target: f32,
    duck_env: DuckEnv,
    duck_release_secs: f32,
    last_sent_factor: f32,
    tx: Option<Sender<RealtimeCommand>>,
    state: Option<Arc<AudioState>>,
}

impl Music {
    pub fn new() -> Self {
        Self {
            tracks: HashMap::new(),
            current: None,
            intensity: 1.0,
            duck_factor: 1.0,
            duck_target: 1.0,
            duck_env: DuckEnv::Idle,
            duck_release_secs: 0.5,
            last_sent_factor: 1.0,
            tx: None,
            state: None,
        }
    }

    /// Wire to an audio thread (done by [`crate::Audio::try_init`]; call
    /// directly for custom plumbing or tests).
    pub fn attach(&mut self, tx: Sender<RealtimeCommand>, state: Arc<AudioState>) {
        self.tx = Some(tx);
        self.state = Some(state);
    }

    /// Register a looping track (main mix). Stems arrive via
    /// [`Music::add_stem`]. Replaces any track of the same name.
    pub fn load_track(&mut self, name: &str, gain: f32, main: &[u8]) -> Result<()> {
        let main = Arc::new(decode_bytes(main)?);
        self.tracks.insert(
            name.to_string(),
            Track {
                gain: gain.max(0.0),
                main,
                stems: Vec::new(),
            },
        );
        Ok(())
    }

    /// Add an intensity stem to a registered track.
    pub fn add_stem(&mut self, name: &str, bytes: &[u8], def: StemDef) -> Result<()> {
        let track = self
            .tracks
            .get_mut(name)
            .ok_or_else(|| anyhow::anyhow!("unknown music track `{name}`"))?;
        track.stems.push(Stem {
            sound: Arc::new(decode_bytes(bytes)?),
            lo: def.lo.clamp(0.0, 1.0),
            hi: def.hi.clamp(0.0, 1.0).max(def.lo.clamp(0.0, 1.0)),
            gain: def.gain.max(0.0),
        });
        Ok(())
    }

    /// Drop finished ids from the current track's voice list.
    pub fn reap(&mut self, finished: &[u64]) {
        if let Some(current) = &mut self.current {
            current.voices.retain(|(id, _)| !finished.contains(id));
            if current.voices.is_empty() {
                self.current = None;
            }
        }
    }

    pub fn now_playing(&self) -> Option<&str> {
        self.current.as_ref().map(|c| c.name.as_str())
    }

    /// Crossfade to a track (`0.0` = hard cut). Unknown names fail soft.
    pub fn play(&mut self, name: &str, fade_secs: f32) -> bool {
        let (tx, state) = match (&self.tx, &self.state) {
            (Some(tx), Some(state)) => (tx, state),
            _ => return false,
        };
        let track = match self.tracks.get(name) {
            Some(t) => t,
            None => {
                log::warn!("unknown music track `{name}`");
                return false;
            }
        };
        let fade = fade_secs.max(0.0);
        // Out with the old.
        if let Some(old) = self.current.take() {
            for (voice, _) in old.voices {
                let _ = tx.send_spin(RealtimeCommand::Fade {
                    voice,
                    target: 0.0,
                    secs: fade,
                });
            }
        }
        // In with the new: start silent, ramp to target.
        let mut voices = Vec::new();
        let start = |sound: Arc<SharedFrames>, gain: f32| -> Option<u64> {
            let id = state.alloc_voice();
            let play = RealtimeCommand::Play(PlayCmd {
                voice: id,
                sound,
                gain: 0.0,
                rate: 1.0,
                pan: 0.0,
                bus: AudioChannel::Music,
                looping: true,
            });
            if tx.send_spin(play).is_err() {
                return None;
            }
            let _ = tx.send_spin(RealtimeCommand::Fade {
                voice: id,
                target: gain,
                secs: fade.max(0.01),
            });
            Some(id)
        };
        if let Some(id) = start(track.main.clone(), self.stem_target(None, track)) {
            voices.push((id, None));
        }
        for (i, stem) in track.stems.iter().enumerate() {
            if let Some(id) = start(stem.sound.clone(), self.stem_target(Some(i), track)) {
                voices.push((id, Some(i)));
            }
        }
        if voices.is_empty() {
            return false;
        }
        self.current = Some(Playing {
            name: name.to_string(),
            voices,
        });
        true
    }

    /// Hard stop with a fade-out. Completions reap through `reap`.
    pub fn stop(&mut self, fade_secs: f32) {
        if let (Some(tx), Some(old)) = (&self.tx, self.current.take()) {
            for (voice, _) in old.voices {
                let _ = tx.send_spin(RealtimeCommand::Fade {
                    voice,
                    target: 0.0,
                    secs: fade_secs.max(0.0),
                });
            }
        }
    }

    /// Intensity `0..=1`: retargets stems through short fades.
    pub fn set_intensity(&mut self, v: f32) {
        self.intensity = v.clamp(0.0, 1.0);
        self.retarget(0.3);
    }

    /// Dip under dialog/stingers: instant attack to `amount_db`, hold,
    /// then linear release. Re-calling re-ducks from the current factor.
    pub fn duck(&mut self, amount_db: f32, hold_secs: f32, release_secs: f32) {
        self.duck_target = 10.0_f32.powf(amount_db / 20.0).clamp(0.0, 1.0);
        self.duck_release_secs = release_secs.max(0.05);
        self.duck_env = DuckEnv::Hold {
            left_secs: hold_secs.max(0.0),
        };
        self.duck_factor = self.duck_target;
        self.retarget(0.05);
    }

    /// Advance the duck envelope. Call from the frame pump with real dt.
    pub fn update(&mut self, dt_secs: f32) {
        let dt = dt_secs.max(0.0);
        match &mut self.duck_env {
            DuckEnv::Idle => return,
            DuckEnv::Hold { left_secs } => {
                *left_secs -= dt;
                if *left_secs <= 0.0 {
                    let from = self.duck_factor;
                    self.duck_env = DuckEnv::Release {
                        from,
                        left_secs: self.duck_release_secs,
                        total_secs: self.duck_release_secs,
                    };
                } else {
                    return;
                }
            }
            DuckEnv::Release {
                from,
                left_secs,
                total_secs,
            } => {
                *left_secs -= dt;
                let k = 1.0 - (*left_secs / *total_secs).clamp(0.0, 1.0);
                self.duck_factor = *from + (1.0 - *from) * k;
                if *left_secs <= 0.0 {
                    self.duck_factor = 1.0;
                    self.duck_target = 1.0;
                    self.duck_env = DuckEnv::Idle;
                }
            }
        }
        if (self.duck_factor - self.last_sent_factor).abs() > 0.005 {
            self.retarget(0.1);
        }
    }

    /// Absolute voice-gain target for one stem (main = `None`).
    fn stem_target(&self, stem: Option<usize>, track: &Track) -> f32 {
        let window = match stem {
            None => 1.0,
            Some(i) => {
                let s = &track.stems[i];
                if s.hi <= s.lo {
                    if self.intensity >= s.lo { 1.0 } else { 0.0 }
                } else {
                    ((self.intensity - s.lo) / (s.hi - s.lo)).clamp(0.0, 1.0)
                }
            }
        };
        let stem_gain = match stem {
            None => 1.0,
            Some(i) => track.stems[i].gain,
        };
        track.gain * window * stem_gain * self.duck_factor
    }

    /// Re-aim every current voice at its policy target.
    fn retarget(&mut self, secs: f32) {
        let (tx, current) = match (&self.tx, &self.current) {
            (Some(tx), Some(current)) => (tx, current),
            _ => return,
        };
        // Track lookup by name (current borrows self; re-borrow fields).
        let name = current.name.clone();
        let (track_gain, stems): (f32, Vec<(f32, f32, f32)>) = match self.tracks.get(&name) {
            Some(t) => (
                t.gain,
                t.stems.iter().map(|s| (s.lo, s.hi, s.gain)).collect(),
            ),
            None => return,
        };
        for (voice, stem) in &current.voices {
            let window = match stem {
                None => 1.0,
                Some(i) => {
                    let (lo, hi, _) = stems[*i];
                    if hi <= lo {
                        if self.intensity >= lo { 1.0 } else { 0.0 }
                    } else {
                        ((self.intensity - lo) / (hi - lo)).clamp(0.0, 1.0)
                    }
                }
            };
            let stem_gain = match stem {
                None => 1.0,
                Some(i) => stems[*i].2,
            };
            let target = track_gain * window * stem_gain * self.duck_factor;
            let _ = tx.send_spin(RealtimeCommand::Fade {
                voice: *voice,
                target,
                secs: secs.max(0.01),
            });
        }
        self.last_sent_factor = self.duck_factor;
    }
}

impl Default for Music {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{audio_link, synth_sine_wav};

    fn musicked() -> (Music, crate::GameAudioLink, crate::ThreadAudioLink) {
        let (game, thread) = audio_link();
        let mut music = Music::new();
        music.attach(game.tx.clone(), game.state.clone());
        (music, game, thread)
    }

    fn wav(freq: f32) -> Vec<u8> {
        synth_sine_wav(freq, 0.2, 22050)
    }

    fn load_two_stems(music: &mut Music) {
        let main = wav(220.0);
        let drums = wav(330.0);
        let lead = wav(440.0);
        music.load_track("battle", 0.8, &main).unwrap();
        music
            .add_stem(
                "battle",
                &drums,
                StemDef {
                    lo: 0.0,
                    hi: 0.5,
                    gain: 1.0,
                },
            )
            .unwrap();
        music
            .add_stem(
                "battle",
                &lead,
                StemDef {
                    lo: 0.5,
                    hi: 1.0,
                    gain: 1.0,
                },
            )
            .unwrap();
    }

    fn drain(thread: &crate::ThreadAudioLink) -> Vec<RealtimeCommand> {
        let mut out = Vec::new();
        while let Ok(cmd) = thread.rx.try_recv() {
            out.push(cmd);
        }
        out
    }

    #[test]
    fn play_starts_main_and_stems_silent_then_ramps() {
        let (mut music, _game, thread) = musicked();
        load_two_stems(&mut music);
        assert!(music.play("battle", 2.0));
        assert_eq!(music.now_playing(), Some("battle"));
        let cmds = drain(&thread);
        // 3 plays (gain 0) + 3 fade-ins.
        assert_eq!(cmds.len(), 6);
        assert!(
            cmds.iter()
                .filter(|c| matches!(c, RealtimeCommand::Play(_)))
                .count()
                == 3
        );
        let fades: Vec<(f32, f32)> = cmds
            .iter()
            .filter_map(|c| match c {
                RealtimeCommand::Fade { target, secs, .. } => Some((*target, *secs)),
                _ => None,
            })
            .collect();
        assert_eq!(fades.len(), 3);
        assert!(fades.iter().all(|(_, s)| (*s - 2.0).abs() < 1e-6));
        // Full intensity: main 0.8, drums window(1.0)=1 -> 0.8, lead window=1 -> 0.8.
        assert!(fades.iter().all(|(t, _)| (*t - 0.8).abs() < 1e-6));
        assert!(!music.play("nope", 1.0));
    }

    #[test]
    fn crossfade_fades_old_out() {
        let (mut music, _game, thread) = musicked();
        load_two_stems(&mut music);
        let calm = wav(110.0);
        music.load_track("calm", 0.6, &calm).unwrap();
        music.play("battle", 1.0);
        let _ = drain(&thread);
        music.play("calm", 3.0);
        assert_eq!(music.now_playing(), Some("calm"));
        let cmds = drain(&thread);
        // 3 fade-outs to zero + 1 play + 1 fade-in.
        let outs: Vec<f32> = cmds
            .iter()
            .filter_map(|c| match c {
                RealtimeCommand::Fade { target, .. } if *target == 0.0 => Some(*target),
                _ => None,
            })
            .collect();
        assert_eq!(outs.len(), 3);
        assert!(cmds.iter().any(|c| matches!(c, RealtimeCommand::Play(_))));
    }

    #[test]
    fn intensity_retargets_stems() {
        let (mut music, _game, thread) = musicked();
        load_two_stems(&mut music);
        music.play("battle", 0.0);
        let _ = drain(&thread);
        music.set_intensity(0.25);
        let cmds = drain(&thread);
        // 3 voices retargeted: main 0.8, drums window(0.25 in [0,.5])=0.5 -> 0.4, lead 0.
        let mut targets: Vec<f32> = cmds
            .iter()
            .filter_map(|c| match c {
                RealtimeCommand::Fade { target, .. } => Some(*target),
                _ => None,
            })
            .collect();
        targets.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(targets.len(), 3);
        assert!((targets[0] - 0.0).abs() < 1e-6, "{targets:?}");
        assert!((targets[1] - 0.4).abs() < 1e-6, "{targets:?}");
        assert!((targets[2] - 0.8).abs() < 1e-6, "{targets:?}");
    }

    #[test]
    fn duck_dips_and_releases() {
        let (mut music, _game, thread) = musicked();
        load_two_stems(&mut music);
        music.play("battle", 0.0);
        let _ = drain(&thread);
        music.duck(-12.0, 0.5, 1.0);
        let cmds = drain(&thread);
        assert_eq!(cmds.len(), 3); // attack retarget, all voices
        let first: f32 = match &cmds[0] {
            RealtimeCommand::Fade { target, .. } => *target,
            _ => panic!("expected fade"),
        };
        assert!((first - 0.8 * 0.251).abs() < 0.02, "first={first}");
        // Hold: no motion.
        music.update(0.4);
        assert!(drain(&thread).is_empty());
        // Release over 1 s: factor climbs back, fades stream once it moves.
        music.update(0.2); // crosses the hold boundary (no motion yet)
        music.update(0.2); // factor moves -> retarget fires
        assert!(!drain(&thread).is_empty());
        for _ in 0..10 {
            music.update(0.2);
        }
        let _ = drain(&thread);
        music.update(0.0);
        assert!(drain(&thread).is_empty()); // settled at 1.0, Idle
    }

    #[test]
    fn stop_fades_out_and_reap_clears() {
        let (mut music, _game, thread) = musicked();
        load_two_stems(&mut music);
        music.play("battle", 0.0);
        let _ = drain(&thread);
        music.stop(1.5);
        assert_eq!(music.now_playing(), None);
        let cmds = drain(&thread);
        assert_eq!(cmds.len(), 3);
        assert!(
            cmds.iter()
                .all(|c| matches!(c, RealtimeCommand::Fade { target: 0.0, .. }))
        );
        // Completions for already-cleared tracks are harmless.
        music.reap(&[7, 8, 9]);
    }
}
