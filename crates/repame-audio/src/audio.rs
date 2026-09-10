//! [`Audio`]: game-thread sound service.
//!
//! Owns the engine (when a device exists), the command link, bus mirrors,
//! and  - as later steps land  - the bank, music director, and rig map.
//! Games hold one `Audio`, call `play`/`music`/`rig_events` per frame,
//! and pump [`Audio::update`].

use anyhow::Result;

use crate::{
    AudioChannel, AudioChannels, AudioSource, CueDef, Engine, GameAudioLink, Music, RigAudio,
    SoundBank, StemDef, audio_link,
};

/// Game-thread sound service (single owner; share with `&mut` or a cell).
pub struct Audio {
    link: Option<GameAudioLink>,
    engine: Option<Engine>,
    channels: AudioChannels,
    finished: Vec<u64>,
    bank: SoundBank,
    music: Music,
    rig: RigAudio,
}

impl Audio {
    /// Open the real backend. Bails soft without an output device  -
    /// keep `Option<Audio>` (or [`Audio::noop`]) and skip sound.
    pub fn try_init() -> Result<Self> {
        let (game, thread) = audio_link();
        let engine = Engine::open(thread)?;
        let mut bank = SoundBank::new();
        bank.attach(game.tx.clone(), game.state.clone());
        let mut music = Music::new();
        music.attach(game.tx.clone(), game.state.clone());
        Ok(Self {
            link: Some(game),
            engine: Some(engine),
            channels: AudioChannels::default(),
            finished: Vec::new(),
            bank,
            music,
            rig: RigAudio::new(),
        })
    }

    /// Silent service: game code and tests run without a device.
    pub fn noop() -> Self {
        Self {
            link: None,
            engine: None,
            channels: AudioChannels::default(),
            finished: Vec::new(),
            bank: SoundBank::new(),
            music: Music::new(),
            rig: RigAudio::new(),
        }
    }

    /// True when a real output device backs this service.
    pub fn is_live(&self) -> bool {
        self.engine.is_some()
    }

    /// Device rate in Hz, or `0` with no stream.
    pub fn sample_rate(&self) -> u32 {
        self.engine.as_ref().map(|e| e.rate()).unwrap_or(0)
    }

    /// Validate a source for later registration (decoding lands in `bank`).
    pub fn add_source(&self, source: AudioSource) -> Result<()> {
        if !source.is_supported() {
            anyhow::bail!("unsupported audio format");
        }
        Ok(())
    }

    /// Per-frame pump with real dt: drain voice completions, prune bank
    /// accounting, advance the music duck envelope. Pair with
    /// [`take_finished`](Self::take_finished) each frame: completions
    /// accumulate until taken (each id is reported exactly once).
    pub fn update(&mut self, dt_secs: f32) {
        let mut fresh = Vec::new();
        if let Some(link) = &self.link {
            while let Ok(ev) = link.events.try_recv() {
                fresh.push(ev.voice());
            }
        }
        if !fresh.is_empty() {
            self.bank.reap(&fresh);
            self.music.reap(&fresh);
            self.finished.extend(fresh);
        }
        self.music.update(dt_secs);
    }

    /// Voice ids that completed since the last call.
    ///
    /// Drains the internal completion queue: each id appears exactly once,
    /// on the first call after its voice ends. Feed the result back into
    /// bank/music accounting or your own completion-gated logic (play the
    /// next line when the current one ends).
    ///
    /// Only natural completions appear here. Voices cut short by `stop`
    /// are removed silently and never report, so logic that waits for a
    /// completion must also handle the stopped path. Otherwise it waits
    /// forever after a manual stop.
    pub fn take_finished(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.finished)
    }

    /// Register a cue from encoded files (see [`SoundBank::load`]).
    pub fn load_cue(&mut self, name: &str, def: CueDef, files: &[&[u8]]) -> Result<()> {
        self.bank.load(name, def, files)
    }

    /// Fire-and-forget cue play. Returns the voice id, or `None` when
    /// throttled, unknown, or silent.
    pub fn play(&mut self, cue: &str) -> Option<u64> {
        self.bank.play(cue)
    }

    /// Positional play in 2D world units.
    pub fn play_at(&mut self, cue: &str, x: f32, y: f32) -> Option<u64> {
        self.bank.play_at(cue, x, y)
    }

    /// Stop every live voice of a cue.
    pub fn stop_cue(&mut self, cue: &str) {
        self.bank.stop_cue(cue);
    }

    /// Direct bank access (listener, scale, counts).
    pub fn bank(&mut self) -> &mut SoundBank {
        &mut self.bank
    }

    /// Register a looping music track (see [`Music::load_track`]).
    pub fn load_track(&mut self, name: &str, gain: f32, main: &[u8]) -> Result<()> {
        self.music.load_track(name, gain, main)
    }

    /// Add an intensity stem (see [`Music::add_stem`]).
    pub fn add_stem(&mut self, name: &str, bytes: &[u8], def: StemDef) -> Result<()> {
        self.music.add_stem(name, bytes, def)
    }

    /// Crossfade to a track. Unknown names fail soft (`false`).
    pub fn play_music(&mut self, name: &str, fade_secs: f32) -> bool {
        self.music.play(name, fade_secs)
    }

    /// Fade the music out.
    pub fn stop_music(&mut self, fade_secs: f32) {
        self.music.stop(fade_secs);
    }

    /// Intensity `0..=1` for layered stems.
    pub fn set_intensity(&mut self, v: f32) {
        self.music.set_intensity(v);
    }

    /// Dip under dialog/stingers, then release.
    pub fn duck_music(&mut self, amount_db: f32, hold_secs: f32, release_secs: f32) {
        self.music.duck(amount_db, hold_secs, release_secs);
    }

    /// Currently directed track, if any.
    pub fn now_playing(&self) -> Option<&str> {
        self.music.now_playing()
    }

    /// Route one rig event to a cue (see [`RigAudio::map_event`]).
    pub fn map_rig_event(&mut self, rig: &str, event: &str, cue: &str) {
        self.rig.map_event(rig, event, cue);
    }

    /// Route an event for every rig (see [`RigAudio::map_event_global`]).
    pub fn map_rig_event_global(&mut self, event: &str, cue: &str) {
        self.rig.map_event_global(event, cue);
    }

    /// Forward one tick's rig events into cue plays. Returns voices started.
    pub fn rig_events(&mut self, rig: &str, events: &[String]) -> Vec<u64> {
        self.rig.sync(&mut self.bank, rig, events)
    }

    /// Positional variant of [`Audio::rig_events`].
    pub fn rig_events_at(&mut self, rig: &str, events: &[String], x: f32, y: f32) -> Vec<u64> {
        self.rig.sync_at(&mut self.bank, rig, events, x, y)
    }

    /// Current bus volumes (game-side mirror of the audio thread truth).
    pub fn channels(&self) -> AudioChannels {
        self.channels
    }

    /// Set one bus: mirrors locally and commands the audio thread.
    pub fn set_channel(&mut self, channel: AudioChannel, gain: f32) {
        self.channels.set(channel, gain);
        if let Some(link) = &self.link {
            let _ = link.tx.send_spin(crate::RealtimeCommand::SetBus {
                bus: channel,
                gain: self.channels.get(channel),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth_sine_wav;

    #[test]
    fn stub_never_blocks_game_code() {
        let mut audio = Audio::noop();
        assert!(!audio.is_live());
        assert_eq!(audio.sample_rate(), 0);
        audio
            .add_source(AudioSource::from_bytes(synth_sine_wav(440.0, 0.1, 22050)))
            .expect("synth wav validates");
        assert!(audio.add_source(AudioSource::from_bytes(vec![7])).is_err());
        audio.set_channel(AudioChannel::Music, 0.5);
        assert!((audio.channels().music() - 0.5).abs() < 1e-6);
        audio.update(0.016);
        assert!(audio.take_finished().is_empty());
    }

    #[test]
    fn init_fails_soft_without_device() {
        // Headless CI has no device: must bail soft, never panic.
        let _ = Audio::try_init();
    }
}
