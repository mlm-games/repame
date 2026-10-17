//! [`Engine`]: cpal output plus the audio-thread voice mixer and bus graph.
//! The stream callback owns an `EngineCore`: each block drains commands,
//! splits at any scheduled music branch point, renders while playing, and
//! reports completions. F32 stereo at 48 kHz is preferred; absence bails
//! without a device. [`render_block`] is testable without any audio device.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::bus::{BusId, BusTree};
use crate::command::{EngineEvent, MusicStart, MusicTransition, RealtimeCommand, SharedFrames, ThreadAudioLink};
use crate::mixer::BusGraph;
use crate::AudioState;

/// One live voice on the audio thread.
#[derive(Debug)]
pub(crate) struct Voice {
    sound: Arc<SharedFrames>,
    /// Fractional frame position (rate stepping).
    pos: f64,
    gain: f32,
    rate: f32,
    pan: f32,
    /// Index into the bus graph.
    bus: usize,
    looping: bool,
    fade: Option<FadeAnim>,
}

/// In-progress gain ramp toward an absolute voice-gain target, measured in
/// device frames so a ramp is sample-exact.
#[derive(Debug, Clone, Copy)]
struct FadeAnim {
    from: f32,
    to: f32,
    frames: u64,
    done: u64,
}

/// A music branch resolved to an absolute device frame.
struct MusicEvent {
    at_frame: u64,
    starts: Vec<MusicStart>,
    stops: Vec<u64>,
    fade_out_frames: u64,
    fade_in_frames: u64,
}

/// Audio-thread core: voices, bus graph, frame clock, scheduled branches.
pub(crate) struct EngineCore {
    link: ThreadAudioLink,
    voices: HashMap<u64, Voice>,
    graph: BusGraph,
    /// Device frames rendered since the stream opened. Only advances while
    /// the transport is playing.
    clock: u64,
    /// Scheduled music events, sorted by frame.
    pending: Vec<MusicEvent>,
}

impl EngineCore {
    fn new(link: ThreadAudioLink) -> Self {
        let mut graph = BusGraph::new(0);
        graph.sync(&BusTree::new());
        Self {
            link,
            voices: HashMap::new(),
            graph,
            clock: 0,
            pending: Vec::new(),
        }
    }
}

/// Equal-power stereo gains for `pan` in `-1..=1`.
fn pan_gains(pan: f32) -> (f32, f32) {
    let angle = (pan.clamp(-1.0, 1.0) + 1.0) * std::f32::consts::FRAC_PI_4;
    (angle.cos(), angle.sin())
}

/// Catmull-Rom through `p1`..`p2` at fractional offset `t`. Exact at
/// `t = 0`, so integer steps stay sample-identical.
fn catmull(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
    let b = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
    let c = 0.5 * (p2 - p0);
    ((a * t + b) * t + c) * t + p1
}

/// Drain one command batch, then mix all voices into `out` through the bus
/// graph. `out` is overwritten. The block is split wherever a scheduled music
/// branch falls, so a branch point lands on an exact sample.
pub(crate) fn render_block(core: &mut EngineCore, out: &mut [f32], out_channels: usize) {
    drain_commands(core);

    let state: &AudioState = &core.link.state;
    if !state.playing.load(std::sync::atomic::Ordering::Relaxed) {
        out.fill(0.0);
        return;
    }
    let rate = state.sample_rate();
    let total = if out_channels == 0 {
        0
    } else {
        out.len() / out_channels
    };
    if rate == 0 || total == 0 {
        out.fill(0.0);
        return;
    }
    core.graph.prepare(rate);

    let mut frame = 0usize;
    while frame < total {
        apply_due(core, rate);
        let limit = core.clock + (total - frame) as u64;
        let next = core
            .pending
            .iter()
            .map(|e| e.at_frame)
            .filter(|f| *f > core.clock && *f <= limit)
            .min();
        let segment = match next {
            Some(f) => (f - core.clock) as usize,
            None => total - frame,
        };
        if segment == 0 {
            break;
        }
        core.graph.begin(segment);
        mix_voices(core, segment, rate);
        let slice = &mut out[frame * out_channels..(frame + segment) * out_channels];
        core.graph.process(segment, slice, out_channels);
        core.clock += segment as u64;
        frame += segment;
    }
}

fn drain_commands(core: &mut EngineCore) {
    let rate = core.link.state.sample_rate().max(1);
    while let Ok(cmd) = core.link.rx.try_recv() {
        match cmd {
            RealtimeCommand::Play(p) => {
                core.voices.insert(
                    p.voice,
                    Voice {
                        sound: p.sound,
                        pos: 0.0,
                        gain: p.gain.max(0.0),
                        rate: p.rate,
                        pan: p.pan.clamp(-1.0, 1.0),
                        bus: BusId::channel(p.bus).index(),
                        looping: p.looping,
                        fade: None,
                    },
                );
            }
            RealtimeCommand::Stop(id) => {
                core.voices.remove(&id);
            }
            RealtimeCommand::StopBus(bus) => {
                let root = BusId::channel(bus).index();
                let nodes = &core.graph;
                core.voices
                    .retain(|_, v| !nodes.descends_from(v.bus, root));
            }
            RealtimeCommand::SetBus { bus, gain } => {
                core.graph.set_gain(BusId::channel(bus).index(), gain);
                core.link.state.set_bus_gain(bus, gain);
            }
            RealtimeCommand::SetBusGain { bus, gain } => {
                core.graph.set_gain(bus.index(), gain);
            }
            RealtimeCommand::SetBusTree(tree) => {
                core.graph.sync(&tree);
            }
            RealtimeCommand::Pause(paused) => {
                core.link
                    .state
                    .playing
                    .store(!paused, std::sync::atomic::Ordering::Relaxed);
            }
            RealtimeCommand::Fade {
                voice,
                target,
                secs,
            } => {
                if let Some(v) = core.voices.get_mut(&voice) {
                    let frames = ((secs.max(0.001) * rate as f32).round() as u64).max(1);
                    v.fade = Some(FadeAnim {
                        from: v.gain,
                        to: target.max(0.0),
                        frames,
                        done: 0,
                    });
                }
            }
            RealtimeCommand::SetRate { voice, rate } => {
                if let Some(v) = core.voices.get_mut(&voice) {
                    v.rate = rate.max(0.01);
                }
            }
            RealtimeCommand::MusicTransition(t) => schedule_transition(core, t, rate),
        }
    }
}

/// Resolve a transition's branch point to an absolute frame and queue it.
///
/// With a quantum and a living phase voice, the branch lands on the next
/// multiple of `quantum_frames` of that voice's loop position, which is what
/// makes the hand-off land on a musical boundary and on an exact sample.
fn schedule_transition(core: &mut EngineCore, t: MusicTransition, rate: u32) {
    if t.starts.is_empty() && t.outgoing.is_empty() {
        return;
    }
    let delay = match (t.phase_from, t.quantum_frames.filter(|q| *q > 0)) {
        (Some(voice), Some(quantum)) => core.voices.get(&voice).map_or(0, |v| {
            let r = v.pos.rem_euclid(quantum as f64);
            if r <= 1e-9 {
                0
            } else {
                (quantum as f64 - r).round() as u64
            }
        }),
        _ => 0,
    };
    let to_frames = |secs: f32| ((secs.max(0.0) * rate as f32).round() as u64).max(0);
    let event = MusicEvent {
        at_frame: core.clock + delay,
        starts: t.starts,
        stops: t.outgoing,
        fade_out_frames: to_frames(t.fade_out_secs),
        fade_in_frames: to_frames(t.fade_in_secs),
    };
    let at = event.at_frame;
    let pos = core
        .pending
        .iter()
        .position(|e| e.at_frame > at)
        .unwrap_or(core.pending.len());
    core.pending.insert(pos, event);
}

fn apply_due(core: &mut EngineCore, rate: u32) {
    while core.pending.first().is_some_and(|e| e.at_frame <= core.clock) {
        let event = core.pending.remove(0);
        for id in event.stops {
            let frames = event.fade_out_frames;
            match core.voices.get_mut(&id) {
                Some(v) if frames > 0 => {
                    v.fade = Some(FadeAnim {
                        from: v.gain,
                        to: 0.0,
                        frames,
                        done: 0,
                    });
                }
                Some(_) => {
                    core.voices.remove(&id);
                }
                None => {}
            }
        }
        for start in event.starts {
            let fade = event.fade_in_frames;
            let gain = start.gain.max(0.0);
            core.voices.insert(
                start.voice,
                Voice {
                    sound: start.sound,
                    pos: 0.0,
                    gain: if fade == 0 { gain } else { 0.0 },
                    rate: 1.0,
                    pan: 0.0,
                    bus: BusId::channel(start.bus).index(),
                    looping: start.looping,
                    fade: if fade == 0 {
                        None
                    } else {
                        Some(FadeAnim {
                            from: 0.0,
                            to: gain,
                            frames: fade,
                            done: 0,
                        })
                    },
                },
            );
        }
    }
    let _ = rate;
}

fn mix_voices(core: &mut EngineCore, frames: usize, rate: u32) {
    let EngineCore { voices, graph, .. } = core;
    let mut finished = Vec::new();
    for (id, voice) in voices.iter_mut() {
        let len = voice.sound.len_frames();
        if len == 0 {
            finished.push(*id);
            continue;
        }
        let step = voice.rate.max(0.01) as f64 * voice.sound.sample_rate as f64 / rate as f64;
        let (lg, rg) = pan_gains(voice.pan);
        let n_ch = voice.sound.channels.max(1) as usize;
        let mut ended = false;
        let buf = graph.bus_buffer(voice.bus, frames);
        for frame in buf.chunks_mut(2) {
            if let Some(fade) = &mut voice.fade {
                fade.done = (fade.done + 1).min(fade.frames);
                let k = fade.done as f32 / fade.frames.max(1) as f32;
                voice.gain = fade.from + (fade.to - fade.from) * k;
                if fade.done >= fade.frames {
                    let to = fade.to;
                    voice.fade = None;
                    if to <= 0.0 {
                        ended = true;
                        break;
                    }
                }
            }
            if voice.pos >= len as f64 {
                if voice.looping {
                    voice.pos %= len as f64;
                } else {
                    ended = true;
                    break;
                }
            }
            let i0 = voice.pos as usize;
            let frac = (voice.pos - i0 as f64) as f32;
            let im1 = i0.saturating_sub(1);
            let i1 = (i0 + 1).min(len - 1);
            let i2 = (i0 + 2).min(len - 1);
            let (l, r) = if n_ch >= 2 {
                let l = catmull(
                    voice.sound.frames[im1 * 2],
                    voice.sound.frames[i0 * 2],
                    voice.sound.frames[i1 * 2],
                    voice.sound.frames[i2 * 2],
                    frac,
                );
                let r = catmull(
                    voice.sound.frames[im1 * 2 + 1],
                    voice.sound.frames[i0 * 2 + 1],
                    voice.sound.frames[i1 * 2 + 1],
                    voice.sound.frames[i2 * 2 + 1],
                    frac,
                );
                (l, r)
            } else {
                let m = catmull(
                    voice.sound.frames[im1],
                    voice.sound.frames[i0],
                    voice.sound.frames[i1],
                    voice.sound.frames[i2],
                    frac,
                );
                (m, m)
            };
            let g = voice.gain;
            if g > 0.0 {
                frame[0] += l * lg * g;
                frame[1] += r * rg * g;
            }
            voice.pos += step;
        }
        if ended || (voice.pos >= len as f64 && !voice.looping) {
            finished.push(*id);
        }
    }
    for id in finished {
        voices.remove(&id);
        let _ = core.link.events.send_spin(EngineEvent::Finished(id));
    }
}

/// cpal stream owner. Dropping stops the callback.
pub struct Engine {
    stream: cpal::Stream,
    rate: u32,
    channels: u16,
}

impl Engine {
    /// Open the default output and start the callback.
    /// Bails without a device; callers continue silent.
    pub fn open(link: ThreadAudioLink) -> Result<Self> {
        // Wasm: explicit AudioWorklet host when atomics are on (the only
        // low-latency web host); default host otherwise so plain checks
        // without `+atomics` still compile.
        #[cfg(target_arch = "wasm32")]
        let host = {
            #[cfg(target_feature = "atomics")]
            let worklet: Option<cpal::Host> = cpal::available_hosts()
                .into_iter()
                .find(|id| *id == cpal::HostId::AudioWorklet)
                .and_then(|id| cpal::host_from_id(id).ok());
            #[cfg(not(target_feature = "atomics"))]
            let worklet: Option<cpal::Host> = None;
            worklet.unwrap_or_else(cpal::default_host)
        };
        #[cfg(not(target_arch = "wasm32"))]
        let host = cpal::default_host();

        let device = host
            .default_output_device()
            .ok_or_else(|| anyhow!("no audio output device"))?;

        // Prefer F32 stereo; fall back to whatever the device offers.
        // The mixer adapts to the native channel count (1 or 2).
        let options: Vec<_> = device.supported_output_configs()?.collect();
        let range = options
            .iter()
            .find(|c| c.sample_format() == cpal::SampleFormat::F32 && c.channels() >= 2)
            .or_else(|| {
                options
                    .iter()
                    .find(|c| c.sample_format() == cpal::SampleFormat::F32)
            })
            .cloned();
        let picked = match range {
            Some(r) => {
                let (min_hz, max_hz) = (r.min_sample_rate(), r.max_sample_rate());
                let rate = if (min_hz..=max_hz).contains(&48000) {
                    48000
                } else {
                    max_hz
                };
                r.with_sample_rate(rate)
            }
            None => device.default_output_config()?,
        };
        let format = picked.sample_format();
        let channels = picked.channels().max(1);
        let rate = picked.sample_rate();
        let config = picked.config();

        link.state.set_sample_rate(rate);
        let mut core = EngineCore::new(link);
        let err_fn = |e| log::error!("audio stream error: {e}");
        let stream = match format {
            cpal::SampleFormat::F32 => device.build_output_stream(
                config,
                move |data: &mut [f32], _| {
                    render_block(&mut core, data, channels as usize);
                },
                err_fn,
                None,
            )?,
            cpal::SampleFormat::I16 => {
                let mut scratch = Vec::<f32>::new();
                device.build_output_stream(
                    config,
                    move |data: &mut [i16], _| {
                        let n = data.len();
                        scratch.clear();
                        scratch.resize(n, 0.0);
                        render_block(&mut core, &mut scratch, channels as usize);
                        for (o, s) in data.iter_mut().zip(scratch.iter()) {
                            *o = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                        }
                    },
                    err_fn,
                    None,
                )?
            }
            cpal::SampleFormat::U16 => {
                let mut scratch = Vec::<f32>::new();
                device.build_output_stream(
                    config,
                    move |data: &mut [u16], _| {
                        let n = data.len();
                        scratch.clear();
                        scratch.resize(n, 0.0);
                        render_block(&mut core, &mut scratch, channels as usize);
                        for (o, s) in data.iter_mut().zip(scratch.iter()) {
                            *o = ((s.clamp(-1.0, 1.0) * 0.5 + 0.5) * u16::MAX as f32) as u16;
                        }
                    },
                    err_fn,
                    None,
                )?
            }
            f => anyhow::bail!("unsupported sample format: {f:?}"),
        };
        stream.play()?;
        Ok(Self {
            stream,
            rate,
            channels,
        })
    }

    /// Re-arm a device a host policy blocked at open.
    ///
    /// The only real case is the browser autoplay gate: cpal opens the
    /// `AudioContext` and calls `resume()` from `play()`, which the browser
    /// refuses outside a user gesture, so the context comes up suspended and
    /// stays silent for the session. Calling `play()` again once a gesture has
    /// happened resumes it. cpal guards its one-shot setup with a
    /// `compare_exchange`, so a repeat `play()` only re-resumes.
    ///
    /// No-op off wasm, where [`Engine::open`] already started the device.
    pub fn unlock(&self) {
        #[cfg(target_arch = "wasm32")]
        let _ = self.stream.play();
        #[cfg(not(target_arch = "wasm32"))]
        let _ = &self.stream;
    }

    /// Suspend the output device, releasing the audio hardware.
    ///
    /// On the web this is `AudioContext.suspend()`: a live context keeps the
    /// device awake and keeps the tab flagged as playing audio, so a shell
    /// that parks instead of exiting has to suspend it. Off wasm it is
    /// best-effort — cpal pauses at the hardware level where the backend
    /// supports it and stops only the callback where it does not, so errors
    /// are ignored and [`crate::Audio::silence`] carries the guarantee.
    pub fn pause(&self) {
        let _ = self.stream.pause();
    }

    pub fn rate(&self) -> u32 {
        self.rate
    }

    pub fn channels(&self) -> u16 {
        self.channels
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioChannel, PlayCmd, audio_link, decode_bytes, synth_sine_wav};

    fn core() -> (EngineCore, crate::GameAudioLink) {
        let (game, thread) = audio_link();
        thread.state.set_sample_rate(44100);
        (EngineCore::new(thread), game)
    }

    fn play_cmd(voice: u64, freq: f32) -> PlayCmd {
        let wav = synth_sine_wav(freq, 0.05, 44100);
        let decoded = decode_bytes(&wav).unwrap();
        PlayCmd {
            voice,
            sound: Arc::new(decoded),
            gain: 1.0,
            rate: 1.0,
            pan: 0.0,
            bus: AudioChannel::Sfx,
            looping: false,
        }
    }

    /// Flat-envelope sine so block energies scale with gain.
    fn flat_cmd(voice: u64, freq: f32, pan: f32) -> PlayCmd {
        let frames: Vec<f32> = (0..4410)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / 44100.0).sin() * 0.5)
            .collect();
        PlayCmd {
            voice,
            sound: Arc::new(SharedFrames {
                sample_rate: 44100,
                channels: 1,
                frames,
            }),
            gain: 1.0,
            rate: 1.0,
            pan,
            bus: AudioChannel::Sfx,
            looping: false,
        }
    }

    fn render(core: &mut EngineCore, frames: usize) -> Vec<f32> {
        let mut out = vec![0.0; frames * 2];
        render_block(core, &mut out, 2);
        out
    }

    #[test]
    fn voice_renders_then_finishes() {
        let (mut core, game) = core();
        let id = game.state.alloc_voice();
        game.tx
            .send_spin(RealtimeCommand::Play(play_cmd(id, 440.0)))
            .expect("send");
        let out = render(&mut core, 512);
        let peak: f32 = out.iter().map(|v| v.abs()).fold(0.0, f32::max);
        assert!(peak > 0.1, "peak = {peak}");
        // 0.05 s at 44.1 kHz = 2205 frames; render past the end.
        render(&mut core, 2205);
        assert!(!core.voices.contains_key(&id));
        assert_eq!(
            game.events.try_recv().expect("completion"),
            EngineEvent::Finished(id)
        );
    }

    #[test]
    fn cubic_step_hits_exact_taps() {
        let (game, thread) = audio_link();
        thread.state.set_sample_rate(48000);
        let mut core = EngineCore::new(thread);
        let id = game.state.alloc_voice();
        let mut frames = vec![0.0f32; 8];
        frames[3] = 1.0;
        game.tx
            .send_spin(RealtimeCommand::Play(PlayCmd {
                voice: id,
                sound: Arc::new(SharedFrames {
                    sample_rate: 24000,
                    channels: 1,
                    frames,
                }),
                gain: 1.0,
                rate: 1.0,
                pan: 0.0,
                bus: AudioChannel::Sfx,
                looping: false,
            }))
            .expect("send");
        // Step 24000/48000 = 0.5: frame 5 lands half a frame before the
        // impulse, where Catmull-Rom gives 0.5625 (linear would give 0.5).
        let out = render(&mut core, 7);
        let p = std::f32::consts::FRAC_1_SQRT_2;
        assert!((out[4 * 2] - 0.0).abs() < 1e-6, "integer taps stay exact");
        assert!((out[5 * 2] - 0.5625 * p).abs() < 1e-4, "cubic = 0.5625");
        assert!((out[6 * 2] - p).abs() < 1e-4, "integer taps stay exact");
    }

    #[test]
    fn pause_renders_silence_and_stop_removes() {
        let (mut core, game) = core();
        let id = game.state.alloc_voice();
        game.tx
            .send_spin(RealtimeCommand::Play(play_cmd(id, 440.0)))
            .expect("send");
        game.tx
            .send_spin(RealtimeCommand::Pause(true))
            .expect("send");
        let out = render(&mut core, 512);
        assert!(out.iter().all(|v| *v == 0.0));
        assert!(core.voices.contains_key(&id));
        game.tx
            .send_spin(RealtimeCommand::Pause(false))
            .expect("send");
        game.tx.send_spin(RealtimeCommand::Stop(id)).expect("send");
        render(&mut core, 64);
        assert!(!core.voices.contains_key(&id));
        // Stopped voices report nothing.
        assert!(game.events.try_recv().is_err());
    }

    #[test]
    fn looping_wraps_without_completion() {
        let (mut core, game) = core();
        let id = game.state.alloc_voice();
        let mut cmd = play_cmd(id, 330.0);
        cmd.looping = true;
        game.tx.send_spin(RealtimeCommand::Play(cmd)).expect("send");
        for _ in 0..4 {
            render(&mut core, 2205);
        }
        assert!(core.voices.contains_key(&id));
        assert!(game.events.try_recv().is_err());
    }

    #[test]
    fn bus_gain_and_pan_shape_output() {
        let (mut core, game) = core();
        let id = game.state.alloc_voice();
        game.tx
            .send_spin(RealtimeCommand::Play(flat_cmd(id, 440.0, -1.0)))
            .expect("send");
        let out = render(&mut core, 512);
        // Block energy (phase-independent) rather than peak.
        let energy: f32 = out.iter().step_by(2).map(|v| v.abs()).sum();
        let right: f32 = out
            .iter()
            .skip(1)
            .step_by(2)
            .map(|v| v.abs())
            .fold(0.0, f32::max);
        assert!(energy > 10.0 && right < 0.01, "energy={energy} r={right}");
        // Halve the sfx bus: block energy halves on the next block.
        game.tx
            .send_spin(RealtimeCommand::SetBus {
                bus: AudioChannel::Sfx,
                gain: 0.5,
            })
            .expect("send");
        let out = render(&mut core, 512);
        let half_energy: f32 = out.iter().step_by(2).map(|v| v.abs()).sum();
        assert!(
            (half_energy - energy * 0.5).abs() < energy * 0.02,
            "half={half_energy} energy={energy}"
        );
    }

    #[test]
    fn fade_ramps_gain_and_finishes_at_zero() {
        let (mut core, game) = core();
        let id = game.state.alloc_voice();
        game.tx
            .send_spin(RealtimeCommand::Play(flat_cmd(id, 440.0, 0.0)))
            .expect("send");
        let pre = render(&mut core, 256);
        let pre_peak: f32 = pre.iter().map(|v| v.abs()).fold(0.0, f32::max);
        // Flat 0.5 sine through center equal-power pan (0.707/side).
        assert!((pre_peak - 0.354).abs() < 0.02, "pre={pre_peak}");
        // 0.05 s fade ~ 2205 frames at 44.1 kHz.
        game.tx
            .send_spin(RealtimeCommand::Fade {
                voice: id,
                target: 0.0,
                secs: 0.05,
            })
            .expect("send");
        let mid = render(&mut core, 256);
        let mid_peak: f32 = mid.iter().map(|v| v.abs()).fold(0.0, f32::max);
        assert!(mid_peak < pre_peak, "mid={mid_peak} pre={pre_peak}");
        render(&mut core, 2205);
        assert!(!core.voices.contains_key(&id));
        assert_eq!(
            game.events.try_recv().expect("completion"),
            EngineEvent::Finished(id)
        );
    }

    #[test]
    fn master_bus_gates_all_voices() {
        let (mut core, game) = core();
        let id = game.state.alloc_voice();
        game.tx
            .send_spin(RealtimeCommand::Play(flat_cmd(id, 440.0, 0.0)))
            .expect("send");
        let out = render(&mut core, 512);
        let peak: f32 = out.iter().map(|v| v.abs()).fold(0.0, f32::max);
        assert!(peak > 0.1, "audible before mute, peak={peak}");
        game.tx
            .send_spin(RealtimeCommand::SetBus {
                bus: AudioChannel::Master,
                gain: 0.0,
            })
            .expect("send");
        let out = render(&mut core, 512);
        assert!(
            out.iter().all(|v| *v == 0.0),
            "master at 0 must silence sfx voices"
        );
    }

    /// Golden sample accuracy: a branch resolved to the next quantum of a
    /// looping voice's phase starts the incoming voice on exactly that frame.
    #[test]
    fn music_branch_lands_on_an_exact_sample() {
        let (mut core, game) = core();
        let phase = game.state.alloc_voice();
        // 400-frame loop of silence; the incoming voice is a single spike.
        let mut phase_cmd = play_cmd(phase, 440.0);
        phase_cmd.sound = Arc::new(SharedFrames {
            sample_rate: 44100,
            channels: 1,
            frames: vec![0.0; 400],
        });
        phase_cmd.looping = true;
        phase_cmd.gain = 1.0;
        game.tx
            .send_spin(RealtimeCommand::Play(phase_cmd))
            .expect("send");
        // Run 250 frames so the loop phase is 250.
        core.clock = 0;
        render(&mut core, 250);
        assert_eq!(core.voices[&phase].pos, 250.0);
        let incoming = game.state.alloc_voice();
        let mut spike = vec![0.0f32; 8];
        spike[0] = 1.0;
        game.tx
            .send_spin(RealtimeCommand::MusicTransition(MusicTransition {
                phase_from: Some(phase),
                outgoing: vec![],
                starts: vec![MusicStart {
                    voice: incoming,
                    sound: Arc::new(SharedFrames {
                        sample_rate: 44100,
                        channels: 1,
                        frames: spike,
                    }),
                    gain: 1.0,
                    looping: false,
                    bus: AudioChannel::Music,
                }],
                quantum_frames: Some(400),
                fade_out_secs: 0.0,
                fade_in_secs: 0.0,
            }))
            .expect("send");
        // Next multiple of 400 after 250 is 400, so the spike is due 150
        // frames after the command. Render a 512-frame block: the spike must
        // appear at frame 150 of it, not at frame 0.
        let out = render(&mut core, 512);
        let first = out
            .iter()
            .step_by(2)
            .position(|v| v.abs() > 1e-6)
            .expect("spike arrived");
        assert_eq!(first, 150, "branch point must be sample-exact");
        let p = std::f32::consts::FRAC_1_SQRT_2;
        assert!((out[150 * 2] - p).abs() < 1e-4);
    }

    #[test]
    fn open_never_panics_without_device() {
        let (game, thread) = audio_link();
        // Headless CI may lack a device: open must bail, not panic.
        let _ = (game, Engine::open(thread));
    }
}
