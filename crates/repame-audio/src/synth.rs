//! [`CueSpec`]: declarative cue synthesis, so a game keeps its tuning table
//! and drops its DSP and its WAV encoder.
//!
//! A cue table is plain data: one [`CueSpec`] per variation. Register it with
//! [`SoundBank::load_synth`](crate::SoundBank::load_synth) (renders on the game
//! thread) or [`SoundBank::request_synth`](crate::SoundBank::request_synth)
//! (renders on the decode worker), and the cue plays through the existing
//! bank, bus, and spatial paths. Rendering is a pure function of the spec and
//! a sample rate: it produces the interleaved mono f32 [`SharedFrames`] the
//! mixer already plays, or PCM16 WAV bytes through [`CueSpec::render_wav`]
//! for the byte-based cue path. [`wav_pcm16`] encodes any buffer standalone.
//!
//! Rendering happens on the game thread or on the loader worker, never in the
//! audio callback: it allocates, and nothing here is called from the mixer.
//! Nothing is allocated per sample either - the frame buffer is sized once
//! before the loop.
//!
//! ## What a spec covers
//! - [`Waveform`]: sine, triangle, square, saw, and white noise. The
//!   discontinuous saw and square carry a 2-point PolyBLEP correction, so a
//!   short cue does not alias when it sweeps.
//! - Frequency: `from_hz` to `to_hz` over the cue, [`Sweep::Constant`],
//!   [`Sweep::Linear`] or [`Sweep::Exponential`] (`from * (to / from)^p`),
//!   plus optional [`Vibrato`] as a fraction of the swept frequency.
//! - Amplitude: [`AmplitudeEnvelope::Adsr`] or a
//!   [`AmplitudeEnvelope::Points`] breakpoint list. Values are always
//!   `0..=1`, so an oscillating envelope cannot flip phase.
//! - `noise_mix` blends white noise over the waveform - the grit the
//!   hand-written impact cues all added by hand - and `noise_seed` makes that
//!   noise reproducible: the stream is drawn once per frame whatever the
//!   waveform, so one seed fixes the same noise for every spec.
//!   [`NoiseShape`] picks whether the noise rides the same envelope as the
//!   waveform or its square, so grit can die ahead of the tone it sits on.
//! - `gain` (clamped to `>= 0`) scales the mixed signal; every frame is
//!   clamped to `-1.0..=1.0` afterwards, so an over-driven spec distorts
//!   instead of wrapping the 16-bit encoder.
//! - `secs` is the cue length and `sample_rate` the rate the spec is written
//!   for. A bank renders at the device rate when one is known and at
//!   `sample_rate` otherwise; [`Audio::update`](crate::Audio::update) then
//!   converts a pre-attach cue like any other.
//!
//! Rendering is deterministic: same spec and rate, same frames, on every
//! platform (the noise RNG is integer-only). The spec types derive `serde`,
//! so a table can live in content:
//!
//! ```ignore
//! // A game keeps only this: no DSP, no WAV encoder, no per-cue helper.
//! let table = [
//!     ("laser", CueSpec::new(Waveform::Square, 880.0, 0.12)
//!         .with_sweep(220.0, Sweep::Exponential)
//!         .with_gain(0.5)),
//!     ("impact", CueSpec::new(Waveform::Triangle, 140.0, 0.9)
//!         .with_noise(0.4, 0x51ED)
//!         .with_amplitude(AmplitudeEnvelope::Adsr {
//!             attack_s: 0.001,
//!             decay_s: 0.22,
//!             sustain: 0.0,
//!             release_s: 0.4,
//!         })),
//! ];
//! for (name, spec) in table {
//!     audio.request_synth(name, CueDef::default(), &[spec]);
//! }
//! ```
//!
//! ## Out of scope in v1
//! Music beds. This renders one voice per spec: no layering into a kit, no
//! note sequences, no loop-seam snapping and no seamless-wrap guarantee. A
//! layered bed belongs in the music path ([`crate::Music`],
//! [`crate::StemDef`]) or in shipped media, not here.

use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::command::SharedFrames;
use crate::decode::MAX_FRAMES;
use crate::state::AudioState;

/// Rate [`CueSpec::new`] writes at.
const DEFAULT_SAMPLE_RATE: u32 = 44_100;

/// Oscillator shape of a [`CueSpec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Waveform {
    #[default]
    Sine,
    Triangle,
    Square,
    Saw,
    /// White noise in `-1.0..=1.0`; `from_hz`/`to_hz` are ignored.
    Noise,
}

/// How the frequency travels from `from_hz` to `to_hz` across the cue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sweep {
    /// Hold `from_hz`; `to_hz` is ignored.
    #[default]
    Constant,
    /// `from + (to - from) * p`, `p` the cue fraction.
    Linear,
    /// `from * (to / from)^p`: constant octaves per second. Both endpoints
    /// must be positive; [`CueSpec::validate`] rejects a pair that is not,
    /// and [`CueSpec::frequency_at`] holds `from_hz` for one.
    Exponential,
}

/// Frequency wobble around the swept frequency.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vibrato {
    /// Wobble rate in Hz.
    pub rate_hz: f32,
    /// Deviation as a fraction (`0.05` = +/-5%).
    pub depth: f32,
}

impl Default for Vibrato {
    /// 6 Hz, +/-2%: a string-ish flutter.
    fn default() -> Self {
        Self {
            rate_hz: 6.0,
            depth: 0.02,
        }
    }
}

/// Amplitude over the cue, always evaluated in `0..=1`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AmplitudeEnvelope {
    /// Attack (linear 0 to 1), then an exponential decay toward `sustain`
    /// with `decay_s` as the time constant, then the cue's final `release_s`
    /// as a linear ramp scaling whatever the decay reached. The decay is
    /// asymptotic like an analog envelope's, so a zero sustain fades out
    /// instead of stepping to silence, and a release that overlaps the attack
    /// or decay continues from the current level.
    Adsr {
        /// Linear 0 to 1 ramp; `0` starts at full scale.
        attack_s: f32,
        /// Exponential decay time constant toward `sustain`; `0` jumps there.
        decay_s: f32,
        /// Level the decay approaches (`0..=1`).
        sustain: f32,
        /// Linear ramp to zero over the cue's final `release_s`.
        release_s: f32,
    },
    /// Piecewise-linear `(time_s, value)` breakpoints, held before the first
    /// and after the last. The timeline is the cue's own, so the cue length
    /// does not enter; at least one point is required.
    Points(Vec<(f32, f32)>),
}

impl Default for AmplitudeEnvelope {
    /// A short percussive pluck: 5 ms attack, 80 ms exponential decay into
    /// silence, 50 ms release.
    fn default() -> Self {
        Self::Adsr {
            attack_s: 0.005,
            decay_s: 0.08,
            sustain: 0.0,
            release_s: 0.05,
        }
    }
}

impl AmplitudeEnvelope {
    /// Envelope value in `0..=1` at `t_secs`.
    ///
    /// `duration_secs` only places the ADSR release: the release ramp
    /// occupies the final `release_s` of a cue that long, and a `release_s`
    /// that runs past the cue end still lands on zero. It is ignored by
    /// [`AmplitudeEnvelope::Points`], which carries its own timeline. The
    /// result is clamped, so nothing reads below zero.
    pub fn value_at(&self, t_secs: f32, duration_secs: f32) -> f32 {
        match self {
            Self::Adsr {
                attack_s,
                decay_s,
                sustain,
                release_s,
            } => {
                let t = t_secs.max(0.0);
                let attack = attack_s.max(0.0);
                let decay = decay_s.max(0.0);
                let sustain = sustain.clamp(0.0, 1.0);
                let body = if attack > 0.0 && t < attack {
                    (t / attack).clamp(0.0, 1.0)
                } else if decay > 0.0 {
                    // `exp(0)` at the attack edge keeps the segment continuous.
                    sustain + (1.0 - sustain) * (-(t - attack) / decay).exp()
                } else {
                    sustain
                };
                let release = release_s.max(0.0);
                let release_ramp = if release > 0.0 {
                    ((duration_secs - t) / release).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                (body * release_ramp).clamp(0.0, 1.0)
            }
            Self::Points(points) => {
                let (Some(&(first_t, first_v)), Some(&(last_t, last_v))) =
                    (points.first(), points.last())
                else {
                    return 0.0;
                };
                let t = t_secs.max(0.0);
                if t <= first_t {
                    return first_v.clamp(0.0, 1.0);
                }
                if t >= last_t {
                    return last_v.clamp(0.0, 1.0);
                }
                for pair in points.windows(2) {
                    let (t0, v0) = pair[0];
                    let (t1, v1) = pair[1];
                    if t >= t0 && t <= t1 {
                        let fraction = if t1 > t0 { (t - t0) / (t1 - t0) } else { 1.0 };
                        return (v0 + (v1 - v0) * fraction).clamp(0.0, 1.0);
                    }
                }
                last_v.clamp(0.0, 1.0)
            }
        }
    }
}

/// Why a [`CueSpec`] could not render or encode.
#[derive(Debug, Clone, PartialEq)]
pub enum SynthError {
    /// A render or WAV encode needs a non-zero rate.
    ZeroSampleRate,
    /// `secs` is not finite, is not positive, or is shorter than one frame at
    /// the render rate.
    InvalidDuration { secs: f32 },
    /// The output would exceed the size cap; `max_frames` names the limit
    /// (the render cap, or the 16-bit WAV data field).
    TooLong { frames: usize, max_frames: usize },
    /// A spec field is NaN or infinite.
    NonFiniteField { field: &'static str },
    /// [`Sweep::Exponential`] needs `from_hz > 0` and `to_hz > 0`.
    ExponentialSweepNeedsPositiveFrequencies { from_hz: f32, to_hz: f32 },
    /// [`AmplitudeEnvelope::Points`] with no points.
    EmptyBreakpoints,
    /// Breakpoint times must not go backwards.
    BreakpointsNotSorted { index: usize },
    /// [`wav_pcm16`] writes mono or stereo only.
    UnsupportedChannels { channels: u8 },
}

impl fmt::Display for SynthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSampleRate => write!(f, "sample rate must be non-zero"),
            Self::InvalidDuration { secs } => {
                write!(f, "duration {secs} s does not make a whole frame")
            }
            Self::TooLong { frames, max_frames } => write!(
                f,
                "render of {frames} frames exceeds the {max_frames}-frame cap"
            ),
            Self::NonFiniteField { field } => write!(f, "field `{field}` is not finite"),
            Self::ExponentialSweepNeedsPositiveFrequencies { from_hz, to_hz } => write!(
                f,
                "exponential sweep needs positive frequencies, got {from_hz} -> {to_hz}"
            ),
            Self::EmptyBreakpoints => write!(f, "breakpoint list is empty"),
            Self::BreakpointsNotSorted { index } => {
                write!(f, "breakpoint {index} goes backwards in time")
            }
            Self::UnsupportedChannels { channels } => {
                write!(f, "WAV encoding needs 1 or 2 channels, got {channels}")
            }
        }
    }
}

impl std::error::Error for SynthError {}

/// How the white-noise blend follows the amplitude envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoiseShape {
    /// Noise rides the same envelope as the waveform.
    #[default]
    Envelope,
    /// Noise rides the envelope's square, so it dies twice as fast as the
    /// tone it sits on: the law the hand-rolled cue tables used.
    Squared,
}

/// One cue, declaratively: waveform, frequency path, envelope, and render
/// parameters. All plain data, so a table of these can live in content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CueSpec {
    /// Oscillator shape.
    pub waveform: Waveform,
    /// Frequency at the cue start, pitch before any vibrato.
    pub from_hz: f32,
    /// Frequency at the cue end, or the same as `from_hz` under
    /// [`Sweep::Constant`].
    pub to_hz: f32,
    /// How `from_hz` travels to `to_hz`.
    pub sweep: Sweep,
    /// Amplitude over the cue.
    pub amplitude: AmplitudeEnvelope,
    /// Frequency wobble, or a steady pitch when `None`.
    pub vibrato: Option<Vibrato>,
    /// White-noise blend in `0..=1` over the waveform; `1` is noise alone.
    pub noise_mix: f32,
    /// Seed of the noise stream. `0` maps to a fixed non-zero fallback,
    /// because a zero xorshift state never advances.
    pub noise_seed: u64,
    /// Whether the noise blend follows the envelope or its square.
    pub noise_shape: NoiseShape,
    /// Cue length in seconds; the envelope runs over exactly this long.
    pub secs: f32,
    /// Linear output gain, clamped to `>= 0` and applied before clipping.
    pub gain: f32,
    /// Rate the spec is written for.
    pub sample_rate: u32,
}

impl CueSpec {
    /// A sine at `from_hz` for `secs`, at the default rate, with the default
    /// pluck envelope and unity gain.
    pub fn new(waveform: Waveform, from_hz: f32, secs: f32) -> Self {
        Self {
            waveform,
            from_hz,
            to_hz: from_hz,
            secs,
            ..Self::default()
        }
    }

    /// Sweep to `to_hz` across the cue.
    pub fn with_sweep(mut self, to_hz: f32, sweep: Sweep) -> Self {
        self.to_hz = to_hz;
        self.sweep = sweep;
        self
    }

    /// Replace the amplitude envelope.
    pub fn with_amplitude(mut self, amplitude: AmplitudeEnvelope) -> Self {
        self.amplitude = amplitude;
        self
    }

    /// Output gain.
    pub fn with_gain(mut self, gain: f32) -> Self {
        self.gain = gain;
        self
    }

    /// Vibrato as a fraction of the swept frequency.
    pub fn with_vibrato(mut self, rate_hz: f32, depth: f32) -> Self {
        self.vibrato = Some(Vibrato { rate_hz, depth });
        self
    }

    /// White-noise blend and its seed.
    pub fn with_noise(mut self, mix: f32, seed: u64) -> Self {
        self.noise_mix = mix;
        self.noise_seed = seed;
        self
    }

    /// Whether the noise blend follows the envelope or its square.
    pub fn with_noise_shape(mut self, shape: NoiseShape) -> Self {
        self.noise_shape = shape;
        self
    }

    /// Rate the spec is written for.
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Frame count an unclamped render would have at `sample_rate`, or why it
    /// cannot render. The count is the nearest whole sample, so a duration
    /// written as `frames / rate` in `f32` renders exactly that many frames.
    /// The cap keeps a typo in `secs` from eating memory.
    pub fn duration_frames(&self, sample_rate: u32) -> Result<usize, SynthError> {
        if sample_rate == 0 {
            return Err(SynthError::ZeroSampleRate);
        }
        if !self.secs.is_finite() || self.secs <= 0.0 {
            return Err(SynthError::InvalidDuration { secs: self.secs });
        }
        let frames = (self.secs as f64 * sample_rate as f64).round() as usize;
        if frames == 0 {
            return Err(SynthError::InvalidDuration { secs: self.secs });
        }
        if frames > MAX_FRAMES {
            return Err(SynthError::TooLong {
                frames,
                max_frames: MAX_FRAMES,
            });
        }
        Ok(frames)
    }

    /// Everything a render would reject, at the spec's own rate: finite
    /// fields, a renderable duration, positive exponential endpoints, and a
    /// usable breakpoint list. Check a content table with this instead of
    /// waiting for `load_synth` to fail.
    pub fn validate(&self) -> Result<(), SynthError> {
        self.validate_at(self.sample_rate)
    }

    fn validate_at(&self, sample_rate: u32) -> Result<(), SynthError> {
        self.duration_frames(sample_rate)?;
        for (field, value) in [
            ("from_hz", self.from_hz),
            ("to_hz", self.to_hz),
            ("noise_mix", self.noise_mix),
            ("secs", self.secs),
            ("gain", self.gain),
        ] {
            if !value.is_finite() {
                return Err(SynthError::NonFiniteField { field });
            }
        }
        if self.sweep == Sweep::Exponential && (self.from_hz <= 0.0 || self.to_hz <= 0.0) {
            return Err(SynthError::ExponentialSweepNeedsPositiveFrequencies {
                from_hz: self.from_hz,
                to_hz: self.to_hz,
            });
        }
        if let Some(vibrato) = &self.vibrato {
            if !vibrato.rate_hz.is_finite() {
                return Err(SynthError::NonFiniteField {
                    field: "vibrato.rate_hz",
                });
            }
            if !vibrato.depth.is_finite() {
                return Err(SynthError::NonFiniteField {
                    field: "vibrato.depth",
                });
            }
        }
        match &self.amplitude {
            AmplitudeEnvelope::Adsr {
                attack_s,
                decay_s,
                sustain,
                release_s,
            } => {
                for (field, value) in [
                    ("amplitude.attack_s", *attack_s),
                    ("amplitude.decay_s", *decay_s),
                    ("amplitude.sustain", *sustain),
                    ("amplitude.release_s", *release_s),
                ] {
                    if !value.is_finite() {
                        return Err(SynthError::NonFiniteField { field });
                    }
                }
            }
            AmplitudeEnvelope::Points(points) => {
                if points.is_empty() {
                    return Err(SynthError::EmptyBreakpoints);
                }
                for (index, (time, value)) in points.iter().enumerate() {
                    if !time.is_finite() || !value.is_finite() {
                        return Err(SynthError::NonFiniteField {
                            field: "amplitude.points",
                        });
                    }
                    if index > 0 && *time < points[index - 1].0 {
                        return Err(SynthError::BreakpointsNotSorted { index });
                    }
                }
            }
        }
        Ok(())
    }

    /// Interleaved mono f32 frames at the spec's own `sample_rate`.
    pub fn render(&self) -> Result<SharedFrames, SynthError> {
        self.render_at(self.sample_rate)
    }

    /// Interleaved mono f32 frames at `sample_rate`, every sample clamped to
    /// `-1.0..=1.0`. Mono keeps a one-shot out of the pan law; spatial and
    /// field shaping happen on the bus and voice, as for any other cue.
    pub fn render_at(&self, sample_rate: u32) -> Result<SharedFrames, SynthError> {
        self.validate_at(sample_rate)?;
        let count = self.duration_frames(sample_rate)?;
        let rate = sample_rate as f32;
        let inv_rate = 1.0 / rate;
        let gain = self.gain.max(0.0);
        let noise_mix = self.noise_mix.clamp(0.0, 1.0);
        let mut frames = Vec::with_capacity(count);
        // Phase is accumulated in cycles; `f64` keeps an extreme `from_hz`
        // from overflowing the accumulator into NaN frames.
        let mut phase = 0.0f64;
        let mut noise = Noise::new(self.noise_seed);
        for i in 0..count {
            let t = i as f32 * inv_rate;
            let frequency = self.frequency_at(t);
            let noise_sample = noise.next_sample();
            let dt = (frequency.abs() * inv_rate).min(0.5);
            let tuned = match self.waveform {
                Waveform::Sine => (std::f64::consts::TAU * phase).sin() as f32,
                Waveform::Triangle => triangle(phase as f32),
                Waveform::Square => square(phase as f32, dt),
                Waveform::Saw => saw(phase as f32, dt),
                Waveform::Noise => noise_sample,
            };
            let mixed = if noise_mix >= 1.0 {
                noise_sample
            } else if noise_mix <= 0.0 {
                tuned
            } else {
                tuned + (noise_sample - tuned) * noise_mix
            };
            let amplitude = self.amplitude.value_at(t, self.secs);
            let sample = match self.noise_shape {
                NoiseShape::Envelope => mixed * amplitude * gain,
                NoiseShape::Squared => {
                    let noise_amplitude = amplitude * amplitude;
                    let blended = if noise_mix >= 1.0 {
                        noise_sample * noise_amplitude
                    } else if noise_mix <= 0.0 {
                        tuned * amplitude
                    } else {
                        tuned * (1.0 - noise_mix) * amplitude
                            + noise_sample * noise_mix * noise_amplitude
                    };
                    blended * gain
                }
            };
            frames.push(sample.clamp(-1.0, 1.0));
            phase = (phase + frequency as f64 * inv_rate as f64).rem_euclid(1.0);
        }
        Ok(SharedFrames {
            sample_rate,
            channels: 1,
            frames,
        })
    }

    /// PCM16 WAV bytes of [`CueSpec::render`], ready for
    /// [`SoundBank::load`](crate::SoundBank::load) or
    /// [`Audio::request_cue`](crate::Audio::request_cue) when a game still
    /// wants bytes.
    pub fn render_wav(&self) -> Result<Vec<u8>, SynthError> {
        wav_pcm16(&self.render()?)
    }

    /// Frequency in Hz at `t_secs`, sweep and vibrato included. Time outside
    /// the cue is clamped to the endpoints.
    ///
    /// Always finite for finite fields: an exponential sweep whose endpoints
    /// cannot form the path (a non-positive one, which [`CueSpec::validate`]
    /// rejects) holds `from_hz` rather than returning a non-finite value.
    pub fn frequency_at(&self, t_secs: f32) -> f32 {
        let progress = if self.secs > 0.0 {
            (t_secs / self.secs).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let base = match self.sweep {
            Sweep::Constant => self.from_hz,
            Sweep::Linear => self.from_hz + (self.to_hz - self.from_hz) * progress,
            Sweep::Exponential if self.from_hz > 0.0 && self.to_hz > 0.0 => {
                self.from_hz * (self.to_hz / self.from_hz).powf(progress)
            }
            Sweep::Exponential => self.from_hz,
        };
        match &self.vibrato {
            Some(vibrato) if vibrato.depth > 0.0 => {
                let phase = std::f32::consts::TAU * vibrato.rate_hz * t_secs;
                base * (1.0 + vibrato.depth * phase.sin())
            }
            _ => base,
        }
    }
}

impl Default for CueSpec {
    /// A 440 Hz sine pluck, 0.25 s at 44.1 kHz.
    fn default() -> Self {
        Self {
            waveform: Waveform::Sine,
            from_hz: 440.0,
            to_hz: 440.0,
            sweep: Sweep::Constant,
            amplitude: AmplitudeEnvelope::default(),
            vibrato: None,
            noise_mix: 0.0,
            noise_seed: 0,
            noise_shape: NoiseShape::default(),
            secs: 0.25,
            gain: 1.0,
            sample_rate: DEFAULT_SAMPLE_RATE,
        }
    }
}

/// Encode interleaved frames as 16-bit PCM WAV, mono or stereo.
///
/// Samples are clamped to `-1.0..=1.0` and truncated toward zero, the
/// conversion the hand-written encoders used; a valid header is written even
/// for an empty buffer. Three or more channels are rejected instead of
/// downmixed, so a mix mistake is visible.
pub fn wav_pcm16(frames: &SharedFrames) -> Result<Vec<u8>, SynthError> {
    if frames.channels == 0 || frames.channels > 2 {
        return Err(SynthError::UnsupportedChannels {
            channels: frames.channels,
        });
    }
    let channels = frames.channels as u16;
    let sample_count = frames.frames.len();
    let max_frames = u32::MAX as usize / 2;
    let data_len = sample_count
        .checked_mul(2)
        .and_then(|bytes| u32::try_from(bytes).ok())
        .ok_or(SynthError::TooLong {
            frames: sample_count,
            max_frames,
        })?;
    let mut out = Vec::with_capacity(44 + sample_count * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&frames.sample_rate.to_le_bytes());
    out.extend_from_slice(
        &frames
            .sample_rate
            .saturating_mul(channels as u32)
            .saturating_mul(2)
            .to_le_bytes(),
    );
    out.extend_from_slice(&(channels * 2).to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes()); // bits
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for sample in &frames.frames {
        let value = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        out.extend_from_slice(&value.to_le_bytes());
    }
    Ok(out)
}

/// Rendered frames at the device rate when one is open, else at the spec's
/// rate. The bank retargets a pre-attach cue the same way as a decoded one.
pub(crate) fn render_for_device(
    spec: &CueSpec,
    state: Option<&AudioState>,
) -> Result<Arc<SharedFrames>, SynthError> {
    let target = state
        .map(AudioState::sample_rate)
        .filter(|rate| *rate != 0)
        .unwrap_or(spec.sample_rate);
    Ok(Arc::new(spec.render_at(target)?))
}

/// 2-point PolyBLEP correction for the discontinuity at phase 0.
fn poly_blep(phase: f32, dt: f32) -> f32 {
    if phase < dt {
        let x = phase / dt;
        x + x - x * x - 1.0
    } else if phase > 1.0 - dt {
        let x = (phase - 1.0) / dt;
        x * x + x + x + 1.0
    } else {
        0.0
    }
}

/// Triangle from normalized phase, starting at zero.
fn triangle(phase: f32) -> f32 {
    if phase < 0.25 {
        phase * 4.0
    } else if phase < 0.75 {
        2.0 - phase * 4.0
    } else {
        phase * 4.0 - 4.0
    }
}

/// Square with the rising and falling edges band-limited.
fn square(phase: f32, dt: f32) -> f32 {
    let naive = if phase < 0.5 { 1.0 } else { -1.0 };
    let opposite = if phase < 0.5 {
        phase + 0.5
    } else {
        phase - 0.5
    };
    naive + poly_blep(phase, dt) - poly_blep(opposite, dt)
}

/// Rising saw with the wrap band-limited.
fn saw(phase: f32, dt: f32) -> f32 {
    phase * 2.0 - 1.0 - poly_blep(phase, dt)
}

/// xorshift64 noise: integer-only, so a seed fixes the same stream on every
/// platform and at every sample rate.
#[derive(Debug, Clone, Copy)]
struct Noise {
    state: u64,
}

impl Noise {
    /// Zero would lock a xorshift at zero; the fallback matches the bank's
    /// variation RNG.
    const ZERO_SEED_FALLBACK: u64 = 0x9E37_79B9_7F4A_7C15;

    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                Self::ZERO_SEED_FALLBACK
            } else {
                seed
            },
        }
    }

    /// Next white sample in `-1.0..=1.0`.
    fn next_sample(&mut self) -> f32 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        let unit =
            (self.state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f32 / (u64::MAX >> 11) as f32;
        unit * 2.0 - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Frequency in the last `window_secs` of a render, from sign changes:
    /// two crossings per cycle.
    fn measure_hz(frames: &SharedFrames, window_secs: f32) -> f32 {
        let total = frames.len_frames();
        let window = ((frames.sample_rate as f32 * window_secs) as usize).min(total);
        let start = total - window;
        let mut crossings = 0usize;
        for i in start + 1..total {
            if (frames.frames[i - 1] < 0.0) != (frames.frames[i] < 0.0) {
                crossings += 1;
            }
        }
        crossings as f32 * frames.sample_rate as f32 / (2.0 * window as f32)
    }

    /// Constant full-scale envelope, so measurements are not shaded by the
    /// default pluck.
    fn flat() -> AmplitudeEnvelope {
        AmplitudeEnvelope::Points(vec![(0.0, 1.0)])
    }

    #[test]
    fn adsr_amplitude_is_golden_at_known_times() {
        let env = AmplitudeEnvelope::Adsr {
            attack_s: 0.1,
            decay_s: 0.2,
            sustain: 0.5,
            release_s: 0.1,
        };
        let secs = 1.0;
        // Attack is linear from zero.
        assert!((env.value_at(0.0, secs) - 0.0).abs() < 1e-6);
        assert!((env.value_at(0.05, secs) - 0.5).abs() < 1e-6);
        // The attack edge lands on full scale.
        assert!((env.value_at(0.1, secs) - 1.0).abs() < 1e-6);
        // Decay: `-0.75` and `-2.0` time constants past the attack edge, with
        // the release ramp still at full.
        let at_three_quarters = 0.5 + 0.5 * (-0.75f64).exp();
        let got = env.value_at(0.25, secs) as f64;
        assert!(
            (got - at_three_quarters).abs() < 1e-5,
            "got {got}, want {at_three_quarters}"
        );
        let at_two = 0.5 + 0.5 * (-2.0f64).exp();
        let got = env.value_at(0.5, secs) as f64;
        assert!((got - at_two).abs() < 1e-5, "got {got}, want {at_two}");
        // Release: the last 0.1 s scales the decayed body by a ramp to zero.
        let expected = (0.5 + 0.5 * (-4.25f64).exp()) * 0.5;
        let got = env.value_at(0.95, secs) as f64;
        assert!((got - expected).abs() < 1e-5, "got {got}, want {expected}");
        assert!(env.value_at(1.0, secs).abs() < 1e-6);
        // Past the end the ramp keeps falling and reads zero, never negative.
        assert!(env.value_at(2.0, secs).abs() < 1e-6);
        // A sustain of one is a flat hold after the attack.
        let held = AmplitudeEnvelope::Adsr {
            attack_s: 0.01,
            decay_s: 0.05,
            sustain: 1.0,
            release_s: 0.0,
        };
        assert!((held.value_at(0.005, 1.0) - 0.5).abs() < 1e-6);
        assert!((held.value_at(0.5, 1.0) - 1.0).abs() < 1e-6);
        // A release longer than the cue scales the body instead of cutting it
        // to `sustain`: the cue ends at zero without a step.
        let overrun = AmplitudeEnvelope::Adsr {
            attack_s: 0.0,
            decay_s: 0.01,
            sustain: 1.0,
            release_s: 2.0,
        };
        assert!((overrun.value_at(0.5, 1.0) - 0.25).abs() < 1e-6);
        assert!(overrun.value_at(1.0, 1.0).abs() < 1e-6);
        assert!(overrun.value_at(2.0, 1.0).abs() < 1e-6);
        // A release as long as the cue is a full-length ramp.
        let whole = AmplitudeEnvelope::Adsr {
            attack_s: 0.0,
            decay_s: 0.01,
            sustain: 1.0,
            release_s: 1.0,
        };
        assert!((whole.value_at(0.25, 1.0) - 0.75).abs() < 1e-6);
        assert!(whole.value_at(1.0, 1.0).abs() < 1e-6);
    }

    #[test]
    fn breakpoint_amplitude_interpolates_and_holds() {
        let env = AmplitudeEnvelope::Points(vec![(0.0, 0.0), (0.1, 1.0), (0.2, 0.5)]);
        assert!((env.value_at(0.0, 1.0) - 0.0).abs() < 1e-6);
        assert!((env.value_at(0.05, 1.0) - 0.5).abs() < 1e-6);
        assert!((env.value_at(0.15, 1.0) - 0.75).abs() < 1e-6);
        assert!((env.value_at(0.9, 1.0) - 0.5).abs() < 1e-6);
        assert!((env.value_at(-1.0, 1.0) - 0.0).abs() < 1e-6);
        // The cue length only places the ADSR release; points carry their own
        // timeline.
        assert!((env.value_at(0.05, 999.0) - 0.5).abs() < 1e-6);
        // A single point is a constant.
        let constant = AmplitudeEnvelope::Points(vec![(0.0, 0.25)]);
        assert!((constant.value_at(0.0, 1.0) - 0.25).abs() < 1e-6);
        assert!((constant.value_at(5.0, 1.0) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn rendered_sine_keeps_the_analytical_phase() {
        let spec = CueSpec::new(Waveform::Sine, 100.0, 0.05)
            .with_amplitude(flat())
            .with_gain(1.0);
        let frames = spec.render_at(1000).expect("renders");
        assert_eq!(frames.channels, 1);
        assert_eq!(frames.sample_rate, 1000);
        assert_eq!(frames.len_frames(), 50);
        // 10 samples per cycle: sample i sits at `sin(2 pi f i / rate)`.
        for i in 0..50 {
            let expected = (std::f32::consts::TAU * 100.0 * i as f32 / 1000.0).sin();
            let got = frames.frames[i];
            assert!(
                (got - expected).abs() < 1e-3,
                "frame {i}: {got} vs {expected}"
            );
        }
    }

    #[test]
    fn linear_sweep_reaches_its_end_frequency() {
        let spec = CueSpec::new(Waveform::Sine, 200.0, 0.5)
            .with_sweep(800.0, Sweep::Linear)
            .with_amplitude(flat());
        assert!((spec.frequency_at(0.0) - 200.0).abs() < 1e-3);
        assert!((spec.frequency_at(0.25) - 500.0).abs() < 1e-3);
        assert!((spec.frequency_at(spec.secs) - 800.0).abs() < 1e-3);
        // Time past the end clamps to the endpoint rather than overshooting.
        assert!((spec.frequency_at(10.0) - 800.0).abs() < 1e-3);
        let frames = spec.render_at(22050).expect("renders");
        let measured = measure_hz(&frames, 0.02);
        assert!(
            (measured - 800.0).abs() / 800.0 < 0.08,
            "linear sweep ended at {measured} Hz"
        );
    }

    #[test]
    fn exponential_sweep_reaches_its_end_frequency() {
        let spec = CueSpec::new(Waveform::Sine, 300.0, 0.5)
            .with_sweep(1200.0, Sweep::Exponential)
            .with_amplitude(flat());
        // Halfway is the geometric mean, not the arithmetic one.
        assert!((spec.frequency_at(0.25) - 600.0).abs() < 1.0);
        assert!((spec.frequency_at(spec.secs) - 1200.0).abs() < 1e-2);
        let frames = spec.render_at(22050).expect("renders");
        let measured = measure_hz(&frames, 0.02);
        assert!(
            (measured - 1200.0).abs() / 1200.0 < 0.08,
            "exponential sweep ended at {measured} Hz"
        );
        // An exponential sweep through zero has no defined path: the render
        // is rejected, and the accessor holds `from_hz` instead of going
        // non-finite.
        let through_zero = spec.clone().with_sweep(0.0, Sweep::Exponential);
        assert!(matches!(
            through_zero.validate(),
            Err(SynthError::ExponentialSweepNeedsPositiveFrequencies { .. })
        ));
        assert!((through_zero.frequency_at(0.25) - 300.0).abs() < 1e-3);
        assert!(through_zero.render().is_err());
    }

    #[test]
    fn vibrato_deviates_and_returns() {
        let spec = CueSpec::new(Waveform::Sine, 400.0, 1.0)
            .with_sweep(400.0, Sweep::Constant)
            .with_vibrato(5.0, 0.05);
        let at_zero = spec.frequency_at(0.0);
        assert!((at_zero - 400.0).abs() < 1e-3);
        let quarter = spec.frequency_at(1.0 / 20.0); // vibrato peak
        assert!((quarter - 420.0).abs() < 1e-2, "{quarter}");
        let trough = spec.frequency_at(3.0 / 20.0);
        assert!((trough - 380.0).abs() < 1e-2, "{trough}");
        assert!((spec.frequency_at(1.0 / 5.0) - 400.0).abs() < 1e-3);
    }

    #[test]
    fn noise_is_deterministic_for_a_fixed_seed() {
        let spec = CueSpec::new(Waveform::Noise, 0.0, 0.05).with_noise(1.0, 7);
        let first = spec.render().expect("renders");
        let second = spec.render().expect("renders");
        assert_eq!(first.frames, second.frames);
        let other = spec.clone().with_noise(1.0, 8).render().expect("renders");
        assert_ne!(first.frames, other.frames);
        // Noise is bounded and audible, not a stuck stream.
        assert!(first.frames.iter().all(|s| s.abs() <= 1.0));
        assert!(first.frames.iter().any(|s| s.abs() > 0.05));
        // A full noise blend is exactly the noise stream, whatever the
        // waveform underneath; a half blend is exactly half of each.
        let saw_only = CueSpec::new(Waveform::Saw, 500.0, 0.05)
            .with_amplitude(flat())
            .render()
            .expect("renders");
        let noise_only = CueSpec::new(Waveform::Noise, 0.0, 0.05)
            .with_noise(1.0, 7)
            .with_amplitude(flat())
            .render()
            .expect("renders");
        let mixed = CueSpec::new(Waveform::Saw, 500.0, 0.05)
            .with_noise(1.0, 7)
            .with_amplitude(flat())
            .render()
            .expect("renders");
        assert_eq!(mixed.frames, noise_only.frames);
        let half = CueSpec::new(Waveform::Saw, 500.0, 0.05)
            .with_noise(0.5, 7)
            .with_amplitude(flat())
            .render()
            .expect("renders");
        for (index, ((blend, saw), noise)) in half
            .frames
            .iter()
            .zip(&saw_only.frames)
            .zip(&noise_only.frames)
            .enumerate()
        {
            let expected = 0.5 * (saw + noise);
            assert!(
                (blend - expected).abs() < 1e-6,
                "frame {index}: {blend} vs {expected}"
            );
        }
        // Seed zero is remapped, so the stream still advances.
        let zero_seed = CueSpec::new(Waveform::Noise, 0.0, 0.02)
            .render()
            .expect("renders");
        assert!(zero_seed.frames.iter().any(|s| *s != 0.0));
    }

    #[test]
    fn discontinuous_waveforms_stay_bounded() {
        for waveform in [Waveform::Triangle, Waveform::Square, Waveform::Saw] {
            let spec = CueSpec::new(waveform, 1000.0, 0.05)
                .with_amplitude(flat())
                .with_gain(1.0);
            let frames = spec.render_at(44_100).expect("renders");
            assert!(
                frames
                    .frames
                    .iter()
                    .all(|s| s.is_finite() && s.abs() <= 1.0),
                "{waveform:?} left the range"
            );
            // One cycle holds 44 samples, so the edges cost little amplitude.
            let peak = frames.frames.iter().fold(0.0f32, |p, s| p.max(s.abs()));
            assert!((0.9..=1.0).contains(&peak), "{waveform:?} peak {peak}");
        }
    }

    #[test]
    fn gain_clips_and_bad_specs_fail_loudly() {
        // Over-driven gain distorts at the rails, it does not wrap.
        let loud = CueSpec::new(Waveform::Sine, 100.0, 0.05)
            .with_amplitude(flat())
            .with_gain(8.0)
            .render_at(8000)
            .expect("renders");
        let peak = loud.frames.iter().fold(0.0f32, |p, s| p.max(s.abs()));
        assert!((peak - 1.0).abs() < 1e-6, "peak {peak}");
        // Negative gain is muted, not inverted (the crate's gain convention).
        let inverted = CueSpec::new(Waveform::Sine, 100.0, 0.05)
            .with_amplitude(flat())
            .with_gain(-1.0)
            .render_at(8000)
            .expect("renders");
        assert!(inverted.frames.iter().all(|s| *s == 0.0));

        let base = CueSpec::new(Waveform::Sine, 440.0, 0.1);
        assert!(matches!(base.render_at(0), Err(SynthError::ZeroSampleRate)));
        assert!(matches!(
            CueSpec::new(Waveform::Sine, 440.0, 0.0).render(),
            Err(SynthError::InvalidDuration { .. })
        ));
        assert!(matches!(
            CueSpec::new(Waveform::Sine, 440.0, f32::NAN).validate(),
            Err(SynthError::InvalidDuration { .. })
        ));
        assert!(matches!(
            CueSpec::new(Waveform::Sine, 440.0, 1e9).render_at(44_100),
            Err(SynthError::TooLong { .. })
        ));
        assert!(matches!(
            CueSpec::new(Waveform::Sine, f32::NAN, 0.1).validate(),
            Err(SynthError::NonFiniteField { field: "from_hz" })
        ));
        let empty = CueSpec {
            amplitude: AmplitudeEnvelope::Points(Vec::new()),
            ..CueSpec::default()
        };
        assert!(matches!(
            empty.validate(),
            Err(SynthError::EmptyBreakpoints)
        ));
        assert!(empty.render().is_err());
        let unsorted = CueSpec {
            amplitude: AmplitudeEnvelope::Points(vec![(0.1, 1.0), (0.05, 0.5)]),
            ..CueSpec::default()
        };
        assert!(matches!(
            unsorted.validate(),
            Err(SynthError::BreakpointsNotSorted { index: 1 })
        ));
    }

    #[test]
    fn wav_encoding_covers_mono_and_stereo() {
        let mono = SharedFrames {
            sample_rate: 22_050,
            channels: 1,
            frames: vec![0.0, 0.5, -0.5],
        };
        let wav = wav_pcm16(&mono).expect("encodes");
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[36..40], b"data");
        assert_eq!(wav.len(), 44 + 3 * 2);
        assert_eq!(u16::from_le_bytes([wav[22], wav[23]]), 1);
        assert_eq!(
            u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]),
            22_050
        );
        assert_eq!(u32::from_le_bytes([wav[40], wav[41], wav[42], wav[43]]), 6);
        assert_eq!(i16::from_le_bytes([wav[44], wav[45]]), 0);
        let half = i16::from_le_bytes([wav[46], wav[47]]) as f32 / i16::MAX as f32;
        assert!((half - 0.5).abs() < 1e-3, "half is {half}");

        let stereo = SharedFrames {
            sample_rate: 44_100,
            channels: 2,
            frames: vec![0.25, -0.25],
        };
        let wav = wav_pcm16(&stereo).expect("encodes");
        assert_eq!(u16::from_le_bytes([wav[22], wav[23]]), 2);
        assert_eq!(
            u32::from_le_bytes([wav[28], wav[29], wav[30], wav[31]]),
            44_100 * 4
        );
        assert_eq!(u16::from_le_bytes([wav[32], wav[33]]), 4);

        let surround = SharedFrames {
            sample_rate: 44_100,
            channels: 3,
            frames: vec![0.0; 3],
        };
        assert!(matches!(
            wav_pcm16(&surround),
            Err(SynthError::UnsupportedChannels { channels: 3 })
        ));
        // A spec's WAV round-trips through the sniffer the loader uses.
        let wav = CueSpec::new(Waveform::Triangle, 330.0, 0.05)
            .render_wav()
            .expect("encodes");
        assert_eq!(crate::sniff_format(&wav), crate::AudioFormat::Wav);
    }

    #[test]
    fn durations_written_in_frames_render_those_frames() {
        // `frames / rate` in `f32` lands a hair below the integer for these,
        // and a truncated count silently dropped the final sample.
        for (frames, rate) in [
            (2200u32, 22_050u32),
            (1600, 22_050),
            (900, 22_050),
            (1800, 22_050),
            (2400, 22_050),
            (3400, 22_050),
            (5000, 22_050),
            (11_000, 22_050),
            (4410, 44_100),
            (100, 1000),
        ] {
            let spec = CueSpec::new(Waveform::Sine, 440.0, frames as f32 / rate as f32);
            assert_eq!(
                spec.duration_frames(rate).expect("fits"),
                frames as usize,
                "{frames} frames at {rate} Hz"
            );
            assert_eq!(
                spec.render_at(rate).expect("renders").len_frames(),
                frames as usize,
                "{frames} frames at {rate} Hz, rendered"
            );
        }
    }

    #[test]
    fn squared_noise_rides_the_envelope_squared() {
        let envelope = AmplitudeEnvelope::Adsr {
            attack_s: 0.0,
            decay_s: 0.05,
            sustain: 0.0,
            release_s: 0.0,
        };
        let render = |shape: NoiseShape| {
            CueSpec::new(Waveform::Noise, 440.0, 0.1)
                .with_noise(1.0, 0x51ED)
                .with_noise_shape(shape)
                .with_amplitude(envelope.clone())
                .with_sample_rate(16_000)
                .render()
                .expect("renders")
                .frames
        };
        let linear = render(NoiseShape::Envelope);
        let squared = render(NoiseShape::Squared);
        assert_eq!(linear.len(), squared.len());
        for (i, (a, b)) in linear.iter().zip(squared.iter()).enumerate() {
            let env = envelope.value_at(i as f32 / 16_000.0, 0.1);
            if env <= 0.01 {
                continue;
            }
            // Both shapes draw the same noise sample at the same instant, so
            // dividing the envelope back out leaves one number.
            assert!(
                (b / (env * env) - a / env).abs() < 1e-6,
                "sample {i}: {a} at env {env} vs {b}"
            );
        }
        // With noise at zero the blend is unity, so the shape cannot matter.
        let quiet = |shape: NoiseShape| {
            CueSpec::new(Waveform::Sine, 440.0, 0.05)
                .with_noise(0.0, 7)
                .with_noise_shape(shape)
                .render()
                .expect("renders")
                .frames
        };
        assert_eq!(quiet(NoiseShape::Envelope), quiet(NoiseShape::Squared));
        assert_eq!(CueSpec::default().noise_shape, NoiseShape::Envelope);
    }

    #[test]
    fn render_honours_the_requested_rate_and_duration() {
        let spec = CueSpec::new(Waveform::Sine, 440.0, 0.1);
        for rate in [16_000u32, 22_050, 44_100, 48_000] {
            let frames = spec.render_at(rate).expect("renders");
            assert_eq!(frames.sample_rate, rate);
            assert_eq!(frames.len_frames(), (0.1f64 * rate as f64) as usize);
            assert_eq!(
                frames.len_frames(),
                spec.duration_frames(rate).expect("fits")
            );
            assert!(frames.frames.iter().all(|s| s.is_finite()));
        }
        // The spec's own rate is what `render` uses.
        let spec = spec.with_sample_rate(22_050);
        assert_eq!(spec.render().expect("renders").len_frames(), 2205);
    }
}
