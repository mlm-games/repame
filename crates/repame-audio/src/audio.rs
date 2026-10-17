//! [`Audio`]: game-thread sound service.
//! Owns the engine, command link, banks, music director, bus tree, and rig
//! map. Games hold one `Audio`, call `play` and `music` per frame, and pump
//! update.

use std::sync::Arc;

use anyhow::Result;

use crate::{
    AudioChannel, AudioChannels, AudioListener3D, AudioSource, BusError, BusId, BusTree, CueDef,
    CueRequest, CueSpec, Engine, GameAudioLink, Music, RealtimeCommand, RigAudio, SoundBank,
    SpatialParams, SpatialVoice3D, StemDef, SyncPoint, audio_link, spatial_3d,
};

/// Re-target threshold before a spatial voice sends a fade command.
const SPATIAL_GAIN_EPSILON: f32 = 0.002;
/// Re-target threshold before a spatial voice sends a rate command.
const SPATIAL_PITCH_EPSILON: f32 = 0.0005;
/// Seconds a spatial re-target takes, so discrete updates do not step audibly.
const SPATIAL_RAMP_SECS: f32 = 0.05;

/// Game-thread sound service (single owner; share with `&mut` or a cell).
pub struct Audio {
    link: Option<GameAudioLink>,
    engine: Option<Engine>,
    channels: AudioChannels,
    finished: Vec<u64>,
    bank: SoundBank,
    music: Music,
    rig: RigAudio,
    tree: BusTree,
    tree_dirty: bool,
    listener: AudioListener3D,
    spatial: Vec<SpatialVoice3D>,
}

impl Audio {
    /// Open the real backend. Bails without an output device, so callers
    /// keep `Option<Audio>` or [`Audio::noop`] and skip sound.
    pub fn try_init() -> Result<Self> {
        let (game, thread) = audio_link();
        let engine = Engine::open(thread)?;
        let mut bank = SoundBank::new();
        bank.attach(game.tx.clone(), game.state.clone());
        let mut music = Music::new();
        music.attach(game.tx.clone(), game.state.clone());
        let tree = BusTree::new();
        let _ = game.tx.send_spin(RealtimeCommand::SetBusTree(Arc::new(tree.clone())));
        Ok(Self {
            link: Some(game),
            engine: Some(engine),
            channels: AudioChannels::default(),
            finished: Vec::new(),
            bank,
            music,
            rig: RigAudio::new(),
            tree,
            tree_dirty: false,
            listener: AudioListener3D::new(),
            spatial: Vec::new(),
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
            tree: BusTree::new(),
            tree_dirty: false,
            listener: AudioListener3D::new(),
            spatial: Vec::new(),
        }
    }

    /// True when a real output device backs this service.
    pub fn is_live(&self) -> bool {
        self.engine.is_some()
    }

    /// Retry a start the host blocked, after a user gesture.
    ///
    /// On the web the browser refuses to start an `AudioContext` that was
    /// created without one, so [`Audio::try_init`] succeeds and then runs
    /// silent forever. Hand this to a first-gesture hook and call it there;
    /// off wasm it does nothing.
    pub fn unlock(&self) {
        if let Some(engine) = &self.engine {
            engine.unlock();
        }
    }

    /// Hard cut: silence every voice and suspend the output device.
    ///
    /// A browser shell that quits cannot exit the process, so it parks and
    /// leaves the tab open; without this the mixer keeps rendering and
    /// looping music plays on forever over the parked frame. Off the web a
    /// quit ends the process, which the OS silences by itself.
    ///
    /// The transport gate lands within one audio block and pops nothing,
    /// which is why this cuts instead of fading: the frame pump a fade would
    /// need is the thing that just stopped. Terminal — there is no un-silence,
    /// and [`Audio::unlock`] will not lift the gate, so a game that wants
    /// sound later must not have silenced.
    pub fn silence(&self) {
        if let Some(link) = &self.link {
            let _ = link.tx.send_spin(RealtimeCommand::Pause(true));
        }
        if let Some(engine) = &self.engine {
            engine.pause();
        }
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

    /// Per-frame pump with real dt: drain completions, prune bank and spatial
    /// accounting, advance the music duck envelope, and push a changed bus
    /// tree to the audio thread. Each id is reported once; pair with
    /// [`take_finished`](Self::take_finished).
    pub fn update(&mut self, dt_secs: f32) {
        self.bank.drain_loader();
        let mut fresh = Vec::new();
        if let Some(link) = &self.link {
            while let Ok(ev) = link.events.try_recv() {
                fresh.push(ev.voice());
            }
        }
        if !fresh.is_empty() {
            self.bank.reap(&fresh);
            self.music.reap(&fresh);
            self.spatial.retain(|t| !fresh.contains(&t.voice));
            self.finished.extend(fresh);
        }
        self.music.update(dt_secs);
        self.sync_buses();
        self.update_spatial();
    }

    /// Voice ids that completed since the last call. Drains the queue.
    /// Stopped voices report nothing, so completion-gated logic must
    /// also handle the stopped path.
    pub fn take_finished(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.finished)
    }

    /// Register a cue from encoded files (see [`SoundBank::load`]).
    pub fn load_cue(&mut self, name: &str, def: CueDef, files: &[&[u8]]) -> Result<()> {
        self.bank.load(name, def, files)
    }

    /// Register a cue rendered from a [`CueSpec`] table: one spec per
    /// variation. The game keeps the tuning table and produces no bytes (see
    /// [`SoundBank::load_synth`]).
    pub fn load_synth(&mut self, name: &str, def: CueDef, specs: &[CueSpec]) -> Result<()> {
        self.bank.load_synth(name, def, specs)
    }

    /// Ask for a cue, decoding on the loader worker instead of the game
    /// thread (see [`SoundBank::request`]). `Ready` means play it now,
    /// `Queued` means audible on a later frame, `Failed` means stop asking.
    pub fn request_cue(&mut self, name: &str, def: CueDef, files: &[&[u8]]) -> CueRequest {
        self.bank.request(name, def, files)
    }

    /// Ask for a synth cue, rendering on the loader worker (see
    /// [`SoundBank::request_synth`]). Same three outcomes as
    /// [`Audio::request_cue`].
    pub fn request_synth(&mut self, name: &str, def: CueDef, specs: &[CueSpec]) -> CueRequest {
        self.bank.request_synth(name, def, specs)
    }

    /// Cues that failed to decode since the last call, for logging.
    /// [`Audio::request_cue`] already reports the same failure.
    pub fn take_cue_failures(&mut self) -> Vec<(String, String)> {
        self.bank.take_failed()
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

    /// Start a cue as a sustained looping voice (engine and skid beds).
    /// Returns the voice id; drive it with [`Self::set_voice_rate`],
    /// [`Self::fade_voice`] and [`Self::stop_voice`].
    pub fn play_loop(&mut self, cue: &str, gain: f32, rate: f32) -> Option<u64> {
        self.bank.play_loop(cue, gain, rate, 0.0)
    }

    /// Retune a loop started with [`Self::play_loop`].
    pub fn set_voice_rate(&mut self, voice: u64, rate: f32) {
        self.bank.set_voice_rate(voice, rate);
    }

    /// Ramp a loop's gain to `gain` over `secs` (0 applies immediately).
    pub fn fade_voice(&mut self, voice: u64, gain: f32, secs: f32) {
        self.bank.fade_voice(voice, gain, secs);
    }

    /// End a loop started with [`Self::play_loop`].
    pub fn stop_voice(&mut self, voice: u64) {
        self.bank.stop_voice(voice);
    }

    /// Direct bank access (listener, scale, counts).
    pub fn bank(&mut self) -> &mut SoundBank {
        &mut self.bank
    }

    /// Scene listener for 3D spatialization.
    pub fn listener_3d(&self) -> &AudioListener3D {
        &self.listener
    }

    /// Move the 3D listener. Tracked spatial voices follow it on the next
    /// [`Audio::update`].
    pub fn set_listener_3d(&mut self, listener: AudioListener3D) {
        self.listener = listener;
    }

    /// Play a cue at a 3D position and keep it tracked: attenuation, panning,
    /// and Doppler are re-evaluated every [`Audio::update`] until the voice
    /// finishes. Returns the voice id.
    pub fn play_3d(
        &mut self,
        cue: &str,
        source: crate::Vec3,
        velocity: crate::Vec3,
        params: SpatialParams,
    ) -> Option<u64> {
        let out = spatial_3d(&self.listener, source, velocity, &params);
        let (voice, base_gain) = self.bank.play_scaled(cue, out.gain, out.pitch, out.pan)?;
        self.spatial.push(SpatialVoice3D {
            voice,
            source,
            velocity,
            params,
            base_gain,
            last: out,
        });
        Some(voice)
    }

    /// Tracked 3D voices, for inspection or manual re-targeting.
    pub fn spatial_voices(&self) -> &[SpatialVoice3D] {
        &self.spatial
    }

    /// Mutable view of a tracked 3D voice (move the source, change params).
    pub fn spatial_voices_mut(&mut self) -> &mut Vec<SpatialVoice3D> {
        &mut self.spatial
    }

    /// Stop every tracked 3D voice and forget them.
    pub fn stop_3d(&mut self) {
        if let Some(link) = &self.link {
            for track in self.spatial.drain(..) {
                let _ = link.tx.send_spin(RealtimeCommand::Fade {
                    voice: track.voice,
                    target: 0.0,
                    secs: 0.05,
                });
            }
        } else {
            self.spatial.clear();
        }
    }

    /// Re-evaluate tracked 3D voices against the listener.
    fn update_spatial(&mut self) {
        if self.spatial.is_empty() {
            return;
        }
        let listener = self.listener;
        for track in &mut self.spatial {
            let out = spatial_3d(&listener, track.source, track.velocity, &track.params);
            if let Some(link) = &self.link {
                if (out.gain - track.last.gain).abs() > SPATIAL_GAIN_EPSILON {
                    let _ = link.tx.send_spin(RealtimeCommand::Fade {
                        voice: track.voice,
                        target: track.base_gain * out.gain,
                        secs: SPATIAL_RAMP_SECS,
                    });
                }
                if (out.pitch - track.last.pitch).abs() > SPATIAL_PITCH_EPSILON {
                    let _ = link.tx.send_spin(RealtimeCommand::SetRate {
                        voice: track.voice,
                        rate: out.pitch,
                    });
                }
            }
            track.last = out;
        }
    }

    /// The authored bus tree.
    pub fn bus_tree(&self) -> &BusTree {
        &self.tree
    }

    /// Edit the bus tree. The change reaches the audio thread on the next
    /// [`Audio::update`].
    pub fn bus_tree_mut(&mut self) -> &mut BusTree {
        self.tree_dirty = true;
        &mut self.tree
    }

    /// Add a bus by name under a parent (cycle-safe).
    pub fn add_bus(&mut self, name: &str, parent: BusId) -> Result<BusId, BusError> {
        let id = self.tree.add_bus(name, parent)?;
        self.tree_dirty = true;
        Ok(id)
    }

    /// Replace a bus effect chain (cycle-safe).
    pub fn set_bus_effects(
        &mut self,
        bus: BusId,
        effects: Vec<crate::EffectSpec>,
    ) -> Result<(), BusError> {
        self.tree.set_effects(bus, effects)?;
        self.tree_dirty = true;
        Ok(())
    }

    /// Set any bus fader, live: the value reaches the audio thread as a
    /// single atomic-free command and does not rebuild the DSP graph.
    pub fn set_bus_gain(&mut self, bus: BusId, gain: f32) {
        let _ = self.tree.set_gain(bus, gain);
        if bus == BusId::channel(AudioChannel::Master) {
            self.channels.set(AudioChannel::Master, gain);
        } else if let Some(channel) = channel_of(bus) {
            self.channels.set(channel, gain);
        }
        if let Some(link) = &self.link {
            let _ = link.tx.send_spin(RealtimeCommand::SetBusGain { bus, gain });
        }
    }

    /// Push the edited bus tree to the audio thread if it changed.
    pub fn sync_buses(&mut self) {
        if !self.tree_dirty {
            return;
        }
        self.tree_dirty = false;
        if let Some(link) = &self.link {
            let _ = link
                .tx
                .send_spin(RealtimeCommand::SetBusTree(Arc::new(self.tree.clone())));
        }
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

    /// Crossfade to a track at a musical branch point (see
    /// [`Music::play_synced`]).
    pub fn play_music_synced(&mut self, name: &str, fade_secs: f32, sync: SyncPoint) -> bool {
        self.music.play_synced(name, fade_secs, sync)
    }

    /// Fire a one-shot music stinger at a branch point (see
    /// [`Music::stinger`]).
    pub fn play_stinger(&mut self, name: &str, sync: SyncPoint, duck_db: f32) -> bool {
        self.music.stinger(name, sync, duck_db)
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

    /// Current bus volumes (game-side mirror of the legacy four channels).
    pub fn channels(&self) -> AudioChannels {
        self.channels
    }

    /// Set one of the four built-in buses: mirrors locally, into the tree,
    /// and commands the audio thread.
    pub fn set_channel(&mut self, channel: AudioChannel, gain: f32) {
        self.channels.set(channel, gain);
        let _ = self.tree.set_gain(BusId::channel(channel), self.channels.get(channel));
        if let Some(link) = &self.link {
            let _ = link.tx.send_spin(RealtimeCommand::SetBus {
                bus: channel,
                gain: self.channels.get(channel),
            });
        }
    }
}

/// Built-in channel behind a bus id, if it is one.
fn channel_of(bus: BusId) -> Option<AudioChannel> {
    [
        AudioChannel::Master,
        AudioChannel::Music,
        AudioChannel::Sfx,
        AudioChannel::Ui,
    ]
    .into_iter()
    .find(|c| BusId::channel(*c) == bus)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EffectSpec, Waveform, synth_sine_wav};

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
        assert!((audio.bus_tree().gain(BusId::channel(AudioChannel::Music)) - 0.5).abs() < 1e-6);
        audio.update(0.016);
        assert!(audio.take_finished().is_empty());
    }

    #[test]
    fn bus_tree_edits_are_cycle_safe_offline() {
        let mut audio = Audio::noop();
        let sfx = BusId::channel(AudioChannel::Sfx);
        let sub = audio.add_bus("sfx_weapons", sfx).expect("adds");
        assert_eq!(audio.bus_tree().parent(sub), Some(sfx));
        assert!(audio.add_bus("sfx_weapons", sfx).is_err());
        audio
            .set_bus_effects(
                sfx,
                vec![
                    EffectSpec::low_pass(4000.0, 2),
                    EffectSpec::limiter(-1.0),
                ],
            )
            .expect("effects");
        assert_eq!(audio.bus_tree().effects(sfx).len(), 2);
        audio.set_bus_gain(sub, 0.25);
        assert!((audio.bus_tree().gain(sub) - 0.25).abs() < 1e-6);
        // The synthetic bus is not one of the four mirrors.
        assert!((audio.channels().sfx() - 1.0).abs() < 1e-6);
        audio.update(0.016);
    }

    #[test]
    fn spatial_play_is_silent_but_tracked_offline() {
        let mut audio = Audio::noop();
        // No link: nothing plays, so no handle is registered.
        let voice = audio.play_3d(
            "gunfire",
            crate::Vec3::new(10.0, 0.0, 0.0),
            crate::Vec3::ZERO,
            SpatialParams::default(),
        );
        assert_eq!(voice, None);
        assert!(audio.spatial_voices().is_empty());
        audio.set_listener_3d(AudioListener3D::new());
        audio.stop_3d();
        audio.update(0.016);
    }

    #[test]
    fn init_fails_soft_without_device() {
        // Headless CI may lack a device: open must bail, not panic.
        let _ = Audio::try_init();
    }

    #[test]
    fn synth_cues_register_on_the_noop_service() {
        let mut audio = Audio::noop();
        let spec = CueSpec::new(Waveform::Square, 220.0, 0.1);
        audio
            .load_synth("blip", CueDef::default(), &[spec.clone()])
            .expect("renders");
        // Registered but silent without a link; asking again is `Ready`
        // because the cue is already in the bank.
        assert_eq!(
            audio.request_synth("blip", CueDef::default(), &[spec]),
            CueRequest::Ready
        );
        assert!(audio.play("blip").is_none());
        // A table the renderer refuses reports through the same channel as a
        // failed decode.
        assert!(audio.load_synth("none", CueDef::default(), &[]).is_err());
        let bad = CueSpec::new(Waveform::Sine, 440.0, 0.0);
        assert!(audio.load_synth("bad", CueDef::default(), &[bad]).is_err());
        audio.update(0.016);
    }
}
