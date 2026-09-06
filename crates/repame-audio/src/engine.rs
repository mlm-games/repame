//! [`Engine`]: cpal output plus the audio-thread voice mixer.
//!
//! The stream callback owns an `EngineCore` (voices + thread link):
//! each block it drains [`RealtimeCommand`]s with `try_recv`, renders
//! when [`AudioState::playing`] holds, and reports completions as
//! [`EngineEvent::Finished`]. Host selection is explicit on wasm
//! (`AudioWorklet`, the only low-latency web host); elsewhere the
//! default host covers desktop and Android AAudio. F32 stereo at
//! 48 kHz is preferred, anything else is adapted, absence bails soft.
//!
//! [`render_block`] is a free function so the mixer is unit-testable
//! without any audio device.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::command::{EngineEvent, RealtimeCommand, SharedFrames, ThreadAudioLink};
use crate::{AudioChannel, AudioState};

/// One live voice on the audio thread.
#[derive(Debug)]
pub(crate) struct Voice {
    sound: Arc<SharedFrames>,
    /// Fractional frame position (rate stepping).
    pos: f64,
    gain: f32,
    rate: f32,
    pan: f32,
    bus: AudioChannel,
    looping: bool,
    fade: Option<FadeAnim>,
}

/// In-progress gain ramp toward an absolute voice-gain target.
#[derive(Debug, Clone, Copy)]
struct FadeAnim {
    from: f32,
    to: f32,
    secs: f32,
    t: f32,
}

/// Audio-thread core: voices plus the thread endpoint. Lives inside the
/// cpal callback; the game thread never touches it.
pub(crate) struct EngineCore {
    link: ThreadAudioLink,
    voices: HashMap<u64, Voice>,
}

impl EngineCore {
    fn new(link: ThreadAudioLink) -> Self {
        Self {
            link,
            voices: HashMap::new(),
        }
    }
}

/// Equal-power stereo gains for `pan` in `-1..=1`.
fn pan_gains(pan: f32) -> (f32, f32) {
    let angle = (pan.clamp(-1.0, 1.0) + 1.0) * std::f32::consts::FRAC_PI_4;
    (angle.cos(), angle.sin())
}

/// Drain one command batch, then mix all voices into `out` (added, so
/// zero it first). `out_channels` is 1 or 2; extra device channels are
/// left silent by the caller.
pub(crate) fn render_block(core: &mut EngineCore, out: &mut [f32], out_channels: usize) {
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
                        bus: p.bus,
                        looping: p.looping,
                        fade: None,
                    },
                );
            }
            RealtimeCommand::Stop(id) => {
                core.voices.remove(&id);
            }
            RealtimeCommand::StopBus(bus) => {
                core.voices.retain(|_, v| v.bus != bus);
            }
            RealtimeCommand::SetBus { bus, gain } => {
                core.link.state.set_bus_gain(bus, gain);
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
                    v.fade = Some(FadeAnim {
                        from: v.gain,
                        to: target.max(0.0),
                        secs: secs.max(0.001),
                        t: 0.0,
                    });
                }
            }
        }
    }

    let state: &AudioState = &core.link.state;
    if !state.playing.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let dev_rate = state.sample_rate();
    if dev_rate == 0 || out_channels == 0 {
        return;
    }

    let mut finished = Vec::new();
    let block_secs = out.len() as f32 / out_channels.max(1) as f32 / dev_rate as f32;
    for (id, voice) in core.voices.iter_mut() {
        let len = voice.sound.len_frames();
        if len == 0 {
            finished.push(*id);
            continue;
        }
        // Gain ramps advance first so the mix below uses this block's gain.
        // Ramping to zero finishes the voice like a natural end.
        if let Some(f) = &mut voice.fade {
            f.t += block_secs;
            let k = (f.t / f.secs).min(1.0);
            voice.gain = f.from + (f.to - f.from) * k;
            if k >= 1.0 {
                let to = f.to;
                voice.fade = None;
                if to <= 0.0 {
                    finished.push(*id);
                    continue;
                }
            }
        }
        let step = voice.rate.max(0.01) as f64 * voice.sound.sample_rate as f64 / dev_rate as f64;
        let g = voice.gain * state.bus_gain(voice.bus);
        let (lg, rg) = pan_gains(voice.pan);
        let n_ch = voice.sound.channels.max(1) as usize;
        let stereo_out = out_channels >= 2;
        for frame in out.chunks_mut(out_channels) {
            if voice.pos >= len as f64 {
                if voice.looping {
                    voice.pos %= len as f64;
                } else {
                    break;
                }
            }
            let i0 = voice.pos as usize;
            let frac = (voice.pos - i0 as f64) as f32;
            let i1 = (i0 + 1).min(len - 1);
            let (l, r) = if n_ch >= 2 {
                let base0 = i0 * 2;
                let base1 = i1 * 2;
                let l = voice.sound.frames[base0]
                    + (voice.sound.frames[base1] - voice.sound.frames[base0]) * frac;
                let r = voice.sound.frames[base0 + 1]
                    + (voice.sound.frames[base1 + 1] - voice.sound.frames[base0 + 1]) * frac;
                (l, r)
            } else {
                let a = voice.sound.frames[i0];
                let b = voice.sound.frames[i1];
                let m = a + (b - a) * frac;
                (m, m)
            };
            if g > 0.0 {
                if stereo_out {
                    frame[0] += l * lg * g;
                    frame[1] += r * rg * g;
                } else {
                    frame[0] += (l + r) * 0.5 * g;
                }
            }
            voice.pos += step;
        }
        if voice.pos >= len as f64 && !voice.looping {
            finished.push(*id);
        }
    }
    for id in finished {
        core.voices.remove(&id);
        let _ = core.link.events.send_spin(EngineEvent::Finished(id));
    }
}

/// cpal stream owner. Dropping stops the callback; the game thread keeps
/// the [`ThreadAudioLink`]'s game half for commands and events.
pub struct Engine {
    _stream: cpal::Stream,
    rate: u32,
    channels: u16,
}

impl Engine {
    /// Open the default output (AudioWorklet host on wasm) and start the
    /// callback. Bails soft when no device exists  - callers keep going
    /// silent, never crash.
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
                    data.fill(0.0);
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
            _stream: stream,
            rate,
            channels,
        })
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
    use crate::{PlayCmd, audio_link, decode_bytes, synth_sine_wav};

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

    /// Flat-envelope sine (no decay): block energies scale exactly with gain.
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
    fn open_never_panics_without_device() {
        let (game, thread) = audio_link();
        // Headless CI has no device: must bail soft, never panic.
        let _ = (game, Engine::open(thread));
    }
}
