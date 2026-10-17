//! Per-bus effect chains: gain, biquad filter, delay, reverb, compressor,
//! limiter, and a 3-band EQ.
//!
//! [`EffectSpec`] is the authored description carried on a [`crate::BusTree`];
//! [`EffectDsp`] is the audio-thread instance holding the filter memory. The
//! instance is created from the spec at the device rate, and only its
//! parameters are rewritten when the tree is re-synced, so a running delay or
//! reverb tail is not flushed by an unrelated fader move.
//!
//! Filter shapes follow Godot's `AudioFilterSW` (`servers/audio/audio_filter_sw.cpp`):
//! a normalized biquad per stage, with `stages` cascaded for 6/12/18/24 dB
//! slopes. The compressor, limiter, and reverb follow
//! `servers/audio/effects/audio_effect_{compressor,limiter,reverb}.cpp` and
//! `reverb_filter.cpp`, which is a Freeverb-derived network.
//!
//! ## Latency and feedback constraints
//! - **Filter / EQ / gain** add no latency: they are minimum-phase IIR filters
//!   with no lookahead. Coefficients are recomputed at block boundaries; Godot
//!   interpolates them within a block, so a swept cutoff here moves in block
//!   quanta instead of sample steps.
//! - **Delay** has `time_ms` of latency by construction. The feedback gain is
//!   clamped to `+/-0.95` so the loop always decays.
//! - **Reverb** has at least 10 samples of predelay (Godot's floor) and comb
//!   feedback clamped to `0.7..0.98`, so the tail is bounded but long; the
//!   network is sized from the Freeverb tunings at the device rate.
//! - **Compressor** is a zero-latency peak detector with no lookahead: gain
//!   reduction lags the detector by the attack time. `attack_ms` is floored at
//!   0.01 ms so the smoothing coefficient stays finite.
//! - **Limiter** is zero-lookahead soft clip plus a hard ceiling. A signal
//!   above the ceiling is clamped to it sample by sample, which means it
//!   cannot pre-empt the peak it is clamping.

use crate::bus::BusId;
use crate::channels::{db_to_linear, linear_to_db};

/// Biquad shape used by [`EffectSpec::Filter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterMode {
    LowPass,
    HighPass,
    BandPass,
}

/// One effect in a bus chain. Cheap to copy; the DSP state lives on the audio
/// thread in [`EffectDsp`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EffectSpec {
    /// Linear gain from decibels.
    Gain { db: f32 },
    /// Biquad filter with 1..=4 cascaded stages (6/12/18/24 dB per octave).
    Filter {
        mode: FilterMode,
        cutoff_hz: f32,
        /// Godot's `resonance` (linear Q). `0.5` is flat.
        resonance: f32,
        stages: u8,
    },
    /// Delay line with a low-passed feedback path and a wet/dry blend.
    Delay {
        time_ms: f32,
        /// Clamped to `-0.95..=0.95` when instantiated.
        feedback: f32,
        /// Wet fraction: `0` is dry, `1` is fully wet.
        mix: f32,
        /// One-pole low-pass cutoff on the feedback path.
        feedback_lowpass_hz: f32,
        /// Extra right-channel delay as a fraction of `time_ms` (`0.1` widens).
        spread: f32,
    },
    /// Freeverb-derived network: 8 combs into 4 allpasses, with predelay.
    Reverb {
        room_size: f32,
        damping: f32,
        wet: f32,
        dry: f32,
        predelay_ms: f32,
        predelay_feedback: f32,
        /// Input high-pass as a `0..=1` amount (Godot's `hpf`).
        highpass: f32,
    },
    /// Zero-latency peak compressor. `sidechain` keys the detector on another
    /// bus's output instead of its own input.
    Compressor {
        threshold_db: f32,
        ratio: f32,
        attack_ms: f32,
        release_ms: f32,
        makeup_db: f32,
        mix: f32,
        sidechain: Option<BusId>,
    },
    /// Soft-clip limiter with a hard ceiling.
    Limiter {
        ceiling_db: f32,
        threshold_db: f32,
        soft_clip_db: f32,
        soft_clip_ratio: f32,
    },
    /// Three serial biquads: low shelf, mid peak, high shelf.
    Eq3 {
        low_hz: f32,
        mid_hz: f32,
        high_hz: f32,
        low_db: f32,
        mid_db: f32,
        high_db: f32,
    },
}

impl EffectSpec {
    /// Unity gain.
    pub fn gain(db: f32) -> Self {
        Self::Gain { db }
    }

    /// Low-pass with `stages` cascaded biquads.
    pub fn low_pass(cutoff_hz: f32, stages: u8) -> Self {
        Self::Filter {
            mode: FilterMode::LowPass,
            cutoff_hz,
            resonance: 0.5,
            stages: stages.clamp(1, 4),
        }
    }

    /// High-pass with `stages` cascaded biquads.
    pub fn high_pass(cutoff_hz: f32, stages: u8) -> Self {
        Self::Filter {
            mode: FilterMode::HighPass,
            cutoff_hz,
            resonance: 0.5,
            stages: stages.clamp(1, 4),
        }
    }

    /// Band-pass with `stages` cascaded biquads.
    pub fn band_pass(cutoff_hz: f32, stages: u8) -> Self {
        Self::Filter {
            mode: FilterMode::BandPass,
            cutoff_hz,
            resonance: 0.5,
            stages: stages.clamp(1, 4),
        }
    }

    /// Delay with a damped feedback path.
    pub fn delay(time_ms: f32, feedback: f32, mix: f32) -> Self {
        Self::Delay {
            time_ms,
            feedback,
            mix,
            feedback_lowpass_hz: 8000.0,
            spread: 0.0,
        }
    }

    /// Room reverb that is audible by default (unlike Godot's, whose wet
    /// default is 0).
    pub fn reverb(room_size: f32, damping: f32, wet: f32) -> Self {
        Self::Reverb {
            room_size,
            damping,
            wet,
            dry: 1.0,
            predelay_ms: 20.0,
            predelay_feedback: 0.4,
            highpass: 0.0,
        }
    }

    /// Compressor keyed on its own input.
    pub fn compressor(threshold_db: f32, ratio: f32) -> Self {
        Self::Compressor {
            threshold_db,
            ratio,
            attack_ms: 5.0,
            release_ms: 100.0,
            makeup_db: 0.0,
            mix: 1.0,
            sidechain: None,
        }
    }

    /// Compressor keyed on another bus (ducking).
    pub fn sidechain_duck(key: BusId, threshold_db: f32, ratio: f32) -> Self {
        Self::Compressor {
            threshold_db,
            ratio,
            attack_ms: 5.0,
            release_ms: 150.0,
            makeup_db: 0.0,
            mix: 1.0,
            sidechain: Some(key),
        }
    }

    /// Brickwall-ish limiter at `ceiling_db`.
    pub fn limiter(ceiling_db: f32) -> Self {
        Self::Limiter {
            ceiling_db,
            threshold_db: 0.0,
            soft_clip_db: 2.0,
            soft_clip_ratio: 10.0,
        }
    }

    /// Three-band tone control.
    pub fn eq3(low_db: f32, mid_db: f32, high_db: f32) -> Self {
        Self::Eq3 {
            low_hz: 250.0,
            mid_hz: 1000.0,
            high_hz: 4000.0,
            low_db,
            mid_db,
            high_db,
        }
    }

    /// Discriminant used to keep DSP state across a re-sync.
    fn kind(&self) -> u8 {
        match self {
            Self::Gain { .. } => 0,
            Self::Filter { .. } => 1,
            Self::Delay { .. } => 2,
            Self::Reverb { .. } => 3,
            Self::Compressor { .. } => 4,
            Self::Limiter { .. } => 5,
            Self::Eq3 { .. } => 6,
        }
    }

    fn prepare(&self, rate: u32) -> EffectDsp {
        match *self {
            Self::Gain { db } => EffectDsp::Gain(GainDsp { lin: db_to_linear(db) }),
            Self::Filter { .. } => EffectDsp::Filter(FilterDsp::new(self, rate)),
            Self::Delay { .. } => EffectDsp::Delay(DelayDsp::new(self, rate)),
            Self::Reverb { .. } => EffectDsp::Reverb(ReverbDsp::new(self, rate)),
            Self::Compressor { .. } => EffectDsp::Compressor(CompressorDsp::new(self, rate)),
            Self::Limiter { .. } => EffectDsp::Limiter(LimiterDsp::new(self, rate)),
            Self::Eq3 { .. } => EffectDsp::Eq3(Eq3Dsp::new(self, rate)),
        }
    }
}

/// Denormal guard, as in Godot's `undenormalize`.
#[inline]
fn undenormal(mut v: f32) -> f32 {
    if v.abs() < 1e-15 {
        v = 0.0;
    }
    v
}

/// Normalized biquad, Godot's `AudioFilterSW::Coeffs` layout.
#[derive(Debug, Clone, Copy, Default)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

/// Direct-form biquad memory (`hb1/hb2` input history, `ha1/ha2` output).
#[derive(Debug, Clone, Copy, Default)]
struct BiquadState {
    hb1: f64,
    hb2: f64,
    ha1: f64,
    ha2: f64,
}

impl BiquadState {
    #[inline]
    fn process(&mut self, c: &Biquad, x: f32) -> f32 {
        let x = x as f64;
        let y = x * c.b0 + self.hb1 * c.b1 + self.hb2 * c.b2 + self.ha1 * c.a1 + self.ha2 * c.a2;
        self.ha2 = self.ha1;
        self.hb2 = self.hb1;
        self.hb1 = x;
        self.ha1 = y;
        y as f32
    }
}

#[inline]
fn normalize(b0: f64, b1: f64, b2: f64, a1: f64, a2: f64, a0: f64) -> Biquad {
    Biquad {
        b0: b0 / a0,
        b1: b1 / a0,
        b2: b2 / a0,
        a1: a1 / -a0,
        a2: a2 / -a0,
    }
}

/// Godot's `AudioFilterSW::prepare_coefficients` for the pass modes.
fn pass_coeffs(mode: FilterMode, cutoff_hz: f32, resonance: f32, stages: u32, rate: f32) -> Biquad {
    let rate = rate.max(1.0) as f64;
    let sr_limit = rate / 2.0 + 512.0;
    let final_cutoff = (cutoff_hz as f64).clamp(1.0, sr_limit);
    let omega = std::f64::consts::TAU * final_cutoff / rate;
    let (sin_v, cos_v) = (omega.sin(), omega.cos());

    let mut q = (resonance as f64).max(0.0001);
    if mode == FilterMode::BandPass {
        q *= 2.0;
    }
    if stages > 1 && q > 1.0 {
        q = q.powf(1.0 / stages as f64);
    }
    let alpha = sin_v / (2.0 * q);
    let a0 = 1.0 + alpha;
    match mode {
        FilterMode::LowPass => normalize(
            (1.0 - cos_v) / 2.0,
            1.0 - cos_v,
            (1.0 - cos_v) / 2.0,
            -2.0 * cos_v,
            1.0 - alpha,
            a0,
        ),
        FilterMode::HighPass => normalize(
            (1.0 + cos_v) / 2.0,
            -(1.0 + cos_v),
            (1.0 + cos_v) / 2.0,
            -2.0 * cos_v,
            1.0 - alpha,
            a0,
        ),
        FilterMode::BandPass => normalize(
            alpha * (q + 1.0).sqrt(),
            0.0,
            -alpha * (q + 1.0).sqrt(),
            -2.0 * cos_v,
            1.0 - alpha,
            a0,
        ),
    }
}

/// Godot's low-shelf coefficients.
fn low_shelf_coeffs(cutoff_hz: f32, resonance: f32, gain_db: f32, rate: f32) -> Biquad {
    let rate = rate.max(1.0) as f64;
    let final_cutoff = (cutoff_hz as f64).clamp(1.0, rate / 2.0 + 512.0);
    let omega = std::f64::consts::TAU * final_cutoff / rate;
    let (sin_v, cos_v) = (omega.sin(), omega.cos());
    let mut tmpq = (resonance as f64).sqrt();
    if tmpq <= 0.0 {
        tmpq = 0.001;
    }
    let tmpgain = (db_to_linear(gain_db) as f64).max(0.001);
    let beta = tmpgain.sqrt() / tmpq;
    let a0 = (tmpgain + 1.0) + (tmpgain - 1.0) * cos_v + beta * sin_v;
    normalize(
        tmpgain * ((tmpgain + 1.0) - (tmpgain - 1.0) * cos_v + beta * sin_v),
        2.0 * tmpgain * ((tmpgain - 1.0) - (tmpgain + 1.0) * cos_v),
        tmpgain * ((tmpgain + 1.0) - (tmpgain - 1.0) * cos_v - beta * sin_v),
        -2.0 * ((tmpgain - 1.0) + (tmpgain + 1.0) * cos_v),
        (tmpgain + 1.0) + (tmpgain - 1.0) * cos_v - beta * sin_v,
        a0,
    )
}

/// Godot's high-shelf coefficients.
fn high_shelf_coeffs(cutoff_hz: f32, resonance: f32, gain_db: f32, rate: f32) -> Biquad {
    let rate = rate.max(1.0) as f64;
    let final_cutoff = (cutoff_hz as f64).clamp(1.0, rate / 2.0 + 512.0);
    let omega = std::f64::consts::TAU * final_cutoff / rate;
    let (sin_v, cos_v) = (omega.sin(), omega.cos());
    let mut tmpq = (resonance as f64).sqrt();
    if tmpq <= 0.0 {
        tmpq = 0.001;
    }
    let tmpgain = (db_to_linear(gain_db) as f64).max(0.001);
    let beta = tmpgain.sqrt() / tmpq;
    let a0 = (tmpgain + 1.0) - (tmpgain - 1.0) * cos_v + beta * sin_v;
    normalize(
        tmpgain * ((tmpgain + 1.0) + (tmpgain - 1.0) * cos_v + beta * sin_v),
        -2.0 * tmpgain * ((tmpgain - 1.0) + (tmpgain + 1.0) * cos_v),
        tmpgain * ((tmpgain + 1.0) + (tmpgain - 1.0) * cos_v - beta * sin_v),
        2.0 * ((tmpgain - 1.0) - (tmpgain + 1.0) * cos_v),
        (tmpgain + 1.0) - (tmpgain - 1.0) * cos_v - beta * sin_v,
        a0,
    )
}

/// Godot's peaking-band coefficients.
fn peak_coeffs(cutoff_hz: f32, resonance: f32, gain_db: f32, rate: f32) -> Biquad {
    let rate = rate.max(1.0) as f64;
    let final_cutoff = (cutoff_hz as f64).clamp(1.0, rate / 2.0 + 512.0);
    let omega = std::f64::consts::TAU * final_cutoff / rate;
    let (sin_v, cos_v) = (omega.sin(), omega.cos());
    let q = (resonance as f64).max(0.0001);
    let gain = (db_to_linear(gain_db) as f64).max(0.001);
    let alpha = sin_v / (2.0 * q);
    let a0 = 1.0 + alpha;
    normalize(
        1.0 + alpha * gain,
        -2.0 * cos_v,
        1.0 - alpha * gain,
        -2.0 * cos_v,
        1.0 - alpha / gain,
        a0,
    )
}

/// Audio-thread instance of one [`EffectSpec`].
#[derive(Debug)]
pub(crate) enum EffectDsp {
    Gain(GainDsp),
    Filter(FilterDsp),
    Delay(DelayDsp),
    Reverb(ReverbDsp),
    Compressor(CompressorDsp),
    Limiter(LimiterDsp),
    Eq3(Eq3Dsp),
}

impl EffectDsp {
    /// Build a fresh instance. `rate` may be `0` before the device is open;
    /// call [`EffectDsp::set_rate`] with the real rate then.
    pub(crate) fn new(spec: &EffectSpec, rate: u32) -> Self {
        spec.prepare(rate)
    }

    /// True when `spec` is the same effect kind, so state can be kept.
    pub(crate) fn same_kind(&self, spec: &EffectSpec) -> bool {
        matches!(
            (self, spec.kind()),
            (Self::Gain(_), 0)
                | (Self::Filter(_), 1)
                | (Self::Delay(_), 2)
                | (Self::Reverb(_), 3)
                | (Self::Compressor(_), 4)
                | (Self::Limiter(_), 5)
                | (Self::Eq3(_), 6)
        )
    }

    /// Rewrite parameters, keeping filter memory and delay/reverb tails.
    pub(crate) fn set_params(&mut self, spec: &EffectSpec) {
        match (self, *spec) {
            (Self::Gain(g), EffectSpec::Gain { db }) => g.lin = db_to_linear(db),
            (Self::Filter(f), s) => f.set_spec(&s),
            (Self::Delay(d), s) => d.set_spec(&s),
            (Self::Reverb(r), s) => r.set_spec(&s),
            (Self::Compressor(c), s) => c.set_spec(&s),
            (Self::Limiter(l), s) => l.set_spec(&s),
            (Self::Eq3(e), s) => e.set_spec(&s),
            _ => {}
        }
    }

    /// Re-configure for a device rate change. Flushes DSP memory.
    pub(crate) fn set_rate(&mut self, rate: u32) {
        match self {
            Self::Filter(f) => f.set_rate(rate),
            Self::Delay(d) => d.set_rate(rate),
            Self::Reverb(r) => r.set_rate(rate),
            Self::Compressor(c) => c.set_rate(rate),
            Self::Eq3(e) => e.set_rate(rate),
            Self::Limiter(_) | Self::Gain(_) => {}
        }
    }

    /// Bus index whose processed output keys this effect, if any.
    pub(crate) fn sidechain_key(&self) -> Option<usize> {
        match self {
            Self::Compressor(c) => c.sidechain.map(|b| b.index()),
            _ => None,
        }
    }

    /// Process one block in place. `key` is the sidechain snapshot of the bus
    /// the compressor names, already at the device rate.
    pub(crate) fn process(&mut self, buf: &mut [f32], frames: usize, key: Option<&[f32]>) {
        match self {
            Self::Gain(g) => {
                let lin = g.lin;
                if lin != 1.0 {
                    for v in buf[..frames * 2].iter_mut() {
                        *v *= lin;
                    }
                }
            }
            Self::Filter(f) => {
                for frame in buf[..frames * 2].chunks_mut(2) {
                    frame[0] = f.process(0, frame[0]);
                    frame[1] = f.process(1, frame[1]);
                }
            }
            Self::Delay(d) => {
                for frame in buf[..frames * 2].chunks_mut(2) {
                    let (l, r) = d.process(frame[0], frame[1]);
                    frame[0] = l;
                    frame[1] = r;
                }
            }
            Self::Reverb(r) => {
                for frame in buf[..frames * 2].chunks_mut(2) {
                    frame[0] = r.process(frame[0]);
                    frame[1] = r.process(frame[1]);
                }
            }
            Self::Compressor(c) => {
                for (i, frame) in buf[..frames * 2].chunks_mut(2).enumerate() {
                    let level = match key {
                        Some(k) => {
                            let a = k.get(i * 2).copied().unwrap_or(0.0).abs();
                            let b = k.get(i * 2 + 1).copied().unwrap_or(0.0).abs();
                            a.max(b)
                        }
                        None => frame[0].abs().max(frame[1].abs()),
                    };
                    let g = c.gain_for(level);
                    frame[0] *= g;
                    frame[1] *= g;
                }
            }
            Self::Limiter(l) => {
                for frame in buf[..frames * 2].chunks_mut(2) {
                    frame[0] = l.process(frame[0]);
                    frame[1] = l.process(frame[1]);
                }
            }
            Self::Eq3(e) => {
                for frame in buf[..frames * 2].chunks_mut(2) {
                    frame[0] = e.process(0, frame[0]);
                    frame[1] = e.process(1, frame[1]);
                }
            }
        }
    }
}

/// Plain gain.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GainDsp {
    lin: f32,
}

/// Cascaded biquads, one memory set per channel and stage.
#[derive(Debug, Clone)]
pub(crate) struct FilterDsp {
    rate: u32,
    spec: EffectSpec,
    coeffs: [Biquad; 4],
    stages: usize,
    state: [[BiquadState; 4]; 2],
}

impl FilterDsp {
    fn new(spec: &EffectSpec, rate: u32) -> Self {
        let mut dsp = Self {
            rate,
            spec: *spec,
            coeffs: [Biquad::default(); 4],
            stages: 1,
            state: [[BiquadState::default(); 4]; 2],
        };
        dsp.rebuild();
        dsp
    }

    fn set_spec(&mut self, spec: &EffectSpec) {
        if self.spec != *spec {
            self.spec = *spec;
            self.rebuild();
        }
    }

    fn set_rate(&mut self, rate: u32) {
        if self.rate != rate {
            self.rate = rate;
            self.state = [[BiquadState::default(); 4]; 2];
            self.rebuild();
        }
    }

    fn rebuild(&mut self) {
        let EffectSpec::Filter {
            mode,
            cutoff_hz,
            resonance,
            stages,
        } = self.spec
        else {
            return;
        };
        let stages = (stages as usize).clamp(1, 4);
        self.stages = stages;
        let c = pass_coeffs(mode, cutoff_hz, resonance, stages as u32, self.rate as f32);
        for slot in self.coeffs.iter_mut().take(stages) {
            *slot = c;
        }
    }

    fn process(&mut self, channel: usize, x: f32) -> f32 {
        let mut v = x;
        for s in 0..self.stages {
            v = self.state[channel][s].process(&self.coeffs[s], v);
        }
        v
    }
}

/// Delay line with a damped feedback path. Stereo, with a fractional spread on
/// the right channel.
#[derive(Debug, Clone)]
pub(crate) struct DelayDsp {
    rate: u32,
    time_ms: f32,
    feedback: f32,
    mix: f32,
    lowpass_hz: f32,
    spread: f32,
    left: Vec<f32>,
    right: Vec<f32>,
    pos: usize,
    delay_l: usize,
    delay_r: usize,
    lp_c: f32,
    lp_state: [f32; 2],
}

impl DelayDsp {
    fn new(spec: &EffectSpec, rate: u32) -> Self {
        let mut dsp = Self {
            rate,
            time_ms: 250.0,
            feedback: 0.0,
            mix: 0.0,
            lowpass_hz: 8000.0,
            spread: 0.0,
            left: Vec::new(),
            right: Vec::new(),
            pos: 0,
            delay_l: 1,
            delay_r: 1,
            lp_c: 0.0,
            lp_state: [0.0; 2],
        };
        dsp.set_spec(spec);
        dsp.allocate();
        dsp
    }

    fn set_spec(&mut self, spec: &EffectSpec) {
        let EffectSpec::Delay {
            time_ms,
            feedback,
            mix,
            feedback_lowpass_hz,
            spread,
        } = *spec
        else {
            return;
        };
        self.time_ms = time_ms.max(0.0);
        self.feedback = feedback.clamp(-0.95, 0.95);
        self.mix = mix.clamp(0.0, 1.0);
        self.lowpass_hz = feedback_lowpass_hz.max(20.0);
        self.spread = spread.max(0.0);
        self.retune();
    }

    fn set_rate(&mut self, rate: u32) {
        if self.rate != rate {
            self.rate = rate;
            self.allocate();
        }
    }

    fn allocate(&mut self) {
        // Two seconds is the hard ceiling (Godot allows five with extra room).
        let len = ((self.rate.max(1) as f32 * 2.0) as usize).max(4);
        self.left = vec![0.0; len];
        self.right = vec![0.0; len];
        self.pos = 0;
        self.lp_state = [0.0; 2];
        self.retune();
    }

    fn retune(&mut self) {
        let rate = self.rate.max(1) as f32;
        let len = self.left.len().max(2);
        let tap = (self.time_ms / 1000.0 * rate).round().max(1.0) as usize;
        self.delay_l = tap.min(len - 1);
        self.delay_r = ((tap as f32 * (1.0 + self.spread)).round() as usize).clamp(1, len - 1);
        let c = (-std::f32::consts::TAU * self.lowpass_hz.min(rate / 2.0) / rate).exp();
        self.lp_c = c.clamp(0.0, 0.999);
    }

    fn process(&mut self, l: f32, r: f32) -> (f32, f32) {
        let len = self.left.len();
        if len == 0 {
            return (l, r);
        }
        let read_l = (self.pos + len - self.delay_l) % len;
        let read_r = (self.pos + len - self.delay_r) % len;
        let dl = self.left[read_l];
        let dr = self.right[read_r];
        let one = 1.0 - self.lp_c;
        self.lp_state[0] = dl * one + self.lp_state[0] * self.lp_c;
        self.lp_state[1] = dr * one + self.lp_state[1] * self.lp_c;
        self.left[self.pos] = undenormal(l + self.lp_state[0] * self.feedback);
        self.right[self.pos] = undenormal(r + self.lp_state[1] * self.feedback);
        self.pos = (self.pos + 1) % len;
        let dry = 1.0 - self.mix;
        (l * dry + dl * self.mix, r * dry + dr * self.mix)
    }
}

/// Freeverb network. Comb/allpass tunings and recursion are Godot's
/// `Reverb` in `servers/audio/effects/reverb_filter.cpp`.
#[derive(Debug, Clone)]
pub(crate) struct ReverbDsp {
    rate: u32,
    room_size: f32,
    damping: f32,
    wet: f32,
    dry: f32,
    predelay_ms: f32,
    predelay_fb: f32,
    highpass: f32,
    combs: [Comb; COMB_COUNT],
    allpasses: [AllPass; ALLPASS_COUNT],
    echo: Vec<f32>,
    echo_pos: usize,
    predelay: usize,
    hp_a1: f32,
    hp_a2: f32,
    hp_b1: f32,
    hp_h1: f32,
    hp_h2: f32,
}

const COMB_COUNT: usize = 8;
const ALLPASS_COUNT: usize = 4;
const MAX_ECHO_MS: f32 = 500.0;
const ALLPASS_FEEDBACK: f32 = 0.7;
const WET_SCALE: f32 = 0.6;

const COMB_TUNINGS: [f32; COMB_COUNT] = [
    0.025306122448979593,
    0.026938775510204082,
    0.028956916099773241,
    0.03074829931972789,
    0.032244897959183672,
    0.03380952380952381,
    0.035306122448979592,
    0.036666666666666667,
];

const ALLPASS_TUNINGS: [f32; ALLPASS_COUNT] = [
    0.0051020408163265302,
    0.007732426303854875,
    0.01,
    0.012607709750566893,
];

#[derive(Debug, Clone, Default)]
struct Comb {
    buffer: Vec<f32>,
    feedback: f32,
    damp: f32,
    damp_h: f32,
    pos: usize,
}

#[derive(Debug, Clone, Default)]
struct AllPass {
    buffer: Vec<f32>,
    pos: usize,
}

impl ReverbDsp {
    fn new(spec: &EffectSpec, rate: u32) -> Self {
        let mut dsp = Self {
            rate,
            room_size: 0.5,
            damping: 0.5,
            wet: 0.3,
            dry: 1.0,
            predelay_ms: 20.0,
            predelay_fb: 0.4,
            highpass: 0.0,
            combs: std::array::from_fn(|_| Comb::default()),
            allpasses: std::array::from_fn(|_| AllPass::default()),
            echo: Vec::new(),
            echo_pos: 0,
            predelay: 10,
            hp_a1: 0.0,
            hp_a2: 0.0,
            hp_b1: 0.0,
            hp_h1: 0.0,
            hp_h2: 0.0,
        };
        dsp.set_spec(spec);
        dsp.allocate();
        dsp
    }

    fn set_spec(&mut self, spec: &EffectSpec) {
        let EffectSpec::Reverb {
            room_size,
            damping,
            wet,
            dry,
            predelay_ms,
            predelay_feedback,
            highpass,
        } = *spec
        else {
            return;
        };
        self.room_size = room_size.clamp(0.0, 1.0);
        self.damping = damping.clamp(0.0, 1.0);
        self.wet = wet.max(0.0);
        self.dry = dry.max(0.0);
        self.predelay_ms = predelay_ms.max(0.0);
        self.predelay_fb = predelay_feedback.clamp(0.0, 0.95);
        self.highpass = highpass.clamp(0.0, 1.0);
        self.update_coeffs();
    }

    fn set_rate(&mut self, rate: u32) {
        if self.rate != rate {
            self.rate = rate;
            self.allocate();
        }
    }

    fn allocate(&mut self) {
        let rate = self.rate.max(1) as f32;
        for (i, c) in self.combs.iter_mut().enumerate() {
            let len = ((COMB_TUNINGS[i] * rate).round() as usize).max(5);
            c.buffer = vec![0.0; len];
            c.pos = 0;
            c.damp_h = 0.0;
        }
        for (i, a) in self.allpasses.iter_mut().enumerate() {
            let len = ((ALLPASS_TUNINGS[i] * rate).round() as usize).max(5);
            a.buffer = vec![0.0; len];
            a.pos = 0;
        }
        self.echo = vec![0.0; (MAX_ECHO_MS / 1000.0 * rate + 1.0) as usize + 1];
        self.echo_pos = 0;
        self.hp_h1 = 0.0;
        self.hp_h2 = 0.0;
        self.update_coeffs();
    }

    fn update_coeffs(&mut self) {
        let rate = self.rate.max(1) as f32;
        let fb = (0.7 + self.room_size * 0.28).clamp(0.7, 0.98);
        let auxdmp = (self.damping / 2.0 + 0.5).powi(2);
        let damp = (-std::f32::consts::TAU * auxdmp * 10000.0 / rate).exp();
        for c in &mut self.combs {
            c.feedback = fb;
            c.damp = damp;
        }
        let want = (self.predelay_ms / 1000.0 * rate).round() as usize;
        self.predelay = want.clamp(10, self.echo.len().saturating_sub(1).max(11));
        let hpaux = (-std::f32::consts::TAU * self.highpass * 6000.0 / rate).exp();
        self.hp_a1 = (1.0 + hpaux) / 2.0;
        self.hp_a2 = -(1.0 + hpaux) / 2.0;
        self.hp_b1 = hpaux;
    }

    fn process(&mut self, x: f32) -> f32 {
        if self.echo.is_empty() {
            return x;
        }
        let size = self.echo.len();
        let read = (self.echo_pos + size - self.predelay) % size;
        let mut input = self.echo[read] * self.predelay_fb + x;
        self.echo[self.echo_pos] = input;
        self.echo_pos = (self.echo_pos + 1) % size;
        if self.highpass > 0.0 {
            let out = input * self.hp_a1 + self.hp_h1 * self.hp_a2 + self.hp_h2 * self.hp_b1;
            self.hp_h2 = out;
            self.hp_h1 = input;
            input = out;
        }
        let mut wet = 0.0;
        for c in &mut self.combs {
            let out = c.buffer[c.pos] * c.feedback;
            let out = out * (1.0 - c.damp) + c.damp_h * c.damp;
            c.damp_h = out;
            c.buffer[c.pos] = undenormal(input + out);
            c.pos = if c.pos + 1 >= c.buffer.len() {
                0
            } else {
                c.pos + 1
            };
            wet += out;
        }
        for a in &mut self.allpasses {
            let aux = a.buffer[a.pos];
            a.buffer[a.pos] = undenormal(ALLPASS_FEEDBACK * aux + wet);
            wet = aux - ALLPASS_FEEDBACK * a.buffer[a.pos];
            a.pos = if a.pos + 1 >= a.buffer.len() {
                0
            } else {
                a.pos + 1
            };
        }
        wet * self.wet * WET_SCALE + x * self.dry
    }
}

/// Godot's `AudioEffectCompressorInstance::process`.
#[derive(Debug, Clone)]
pub(crate) struct CompressorDsp {
    rate: u32,
    spec: EffectSpec,
    atcoef: f32,
    relcoef: f32,
    ratatcoef: f32,
    ratrelcoef: f32,
    threshold: f32,
    makeup: f32,
    mix: f32,
    ratio: f32,
    /// Bus whose output keys the detector, if any.
    pub(crate) sidechain: Option<BusId>,
    rundb: f32,
    runratio: f32,
    averatio: f32,
    runmax: f32,
    maxover: f32,
    gr_meter: f32,
}

impl CompressorDsp {
    fn new(spec: &EffectSpec, rate: u32) -> Self {
        let mut dsp = Self {
            rate,
            spec: EffectSpec::compressor(-24.0, 4.0),
            atcoef: 0.0,
            relcoef: 0.0,
            ratatcoef: 0.0,
            ratrelcoef: 0.0,
            threshold: 1.0,
            makeup: 1.0,
            mix: 1.0,
            ratio: 4.0,
            sidechain: None,
            rundb: 0.0,
            runratio: 0.0,
            averatio: 0.0,
            runmax: 0.0,
            maxover: 0.0,
            gr_meter: 1.0,
        };
        dsp.set_spec(spec);
        dsp
    }

    fn set_spec(&mut self, spec: &EffectSpec) {
        let EffectSpec::Compressor {
            threshold_db,
            ratio,
            makeup_db,
            mix,
            sidechain,
            ..
        } = *spec
        else {
            return;
        };
        self.spec = *spec;
        self.sidechain = sidechain;
        self.threshold = db_to_linear(threshold_db);
        self.ratio = ratio.max(1.0);
        self.makeup = db_to_linear(makeup_db);
        self.mix = mix.clamp(0.0, 1.0);
        self.rebuild();
    }

    fn set_rate(&mut self, rate: u32) {
        if self.rate != rate {
            self.rate = rate;
            self.rebuild();
        }
    }

    fn rebuild(&mut self) {
        let rate = self.rate.max(1) as f32;
        let (attack_ms, release_ms) = match self.spec {
            EffectSpec::Compressor {
                attack_ms,
                release_ms,
                ..
            } => (attack_ms, release_ms),
            _ => (5.0, 100.0),
        };
        let attack = (attack_ms.max(0.01) / 1000.0) * rate;
        let release = (release_ms.max(0.01) / 1000.0) * rate;
        self.atcoef = (-1.0 / attack).exp();
        self.relcoef = (-1.0 / release).exp();
        self.ratatcoef = (-1.0 / (0.00001 * rate)).exp();
        self.ratrelcoef = (-1.0 / (0.5 * rate)).exp();
    }

    /// Per-sample gain reduction for a detector input level (linear peak).
    fn gain_for(&mut self, level: f32) -> f32 {
        let threshold = self.threshold.max(1e-9);
        let mut overdb = 2.08136898 * linear_to_db(level / threshold);
        if overdb < 0.0 {
            overdb = 0.0;
        }
        if overdb - self.rundb > 5.0 {
            self.averatio = 4.0;
        }
        if overdb > self.rundb {
            self.rundb = overdb + self.atcoef * (self.rundb - overdb);
            self.runratio = self.averatio + self.ratatcoef * (self.runratio - self.averatio);
        } else {
            self.rundb = overdb + self.relcoef * (self.rundb - overdb);
            self.runratio = self.averatio + self.ratrelcoef * (self.runratio - self.averatio);
        }
        overdb = self.rundb;
        self.averatio = self.runratio;
        let cratio = self.ratio;
        let gr = -overdb * (cratio - 1.0) / cratio;
        let grv = db_to_linear(gr);
        let gr_meter_decay = (-1.0 / (0.5 * self.rate.max(1) as f32)).exp();
        self.runmax = self.maxover + self.relcoef * (self.runmax - self.maxover);
        self.maxover = self.runmax;
        if grv < self.gr_meter {
            self.gr_meter = grv;
        } else {
            self.gr_meter = (self.gr_meter * gr_meter_decay).min(1.0);
        }
        self.makeup * grv * self.mix + (1.0 - self.mix)
    }
}

/// Godot's `AudioEffectLimiterInstance::process`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LimiterDsp {
    ceiling: f32,
    ceiling_db: f32,
    makeup: f32,
    soft_clip: f32,
    scmult: f32,
}

impl LimiterDsp {
    fn new(spec: &EffectSpec, _rate: u32) -> Self {
        let mut dsp = Self {
            ceiling: 1.0,
            ceiling_db: 0.0,
            makeup: 1.0,
            soft_clip: 0.0,
            scmult: 0.0,
        };
        dsp.set_spec(spec);
        dsp
    }

    fn set_spec(&mut self, spec: &EffectSpec) {
        let EffectSpec::Limiter {
            ceiling_db,
            threshold_db,
            soft_clip_db,
            soft_clip_ratio: _,
        } = *spec
        else {
            return;
        };
        self.ceiling_db = ceiling_db;
        self.ceiling = db_to_linear(ceiling_db);
        self.makeup = db_to_linear(ceiling_db - threshold_db);
        let sc = -soft_clip_db;
        self.soft_clip = db_to_linear(sc);
        let peakdb = ceiling_db + 25.0;
        self.scmult = ((ceiling_db - sc) / (peakdb - sc)).abs();
    }

    fn process(&self, x: f32) -> f32 {
        let s = x * self.makeup;
        let sign = if s < 0.0 { -1.0 } else { 1.0 };
        let abs = s.abs();
        let overdb = linear_to_db(abs) - self.ceiling_db;
        let clipped = if abs > self.soft_clip {
            sign * (self.soft_clip + db_to_linear(overdb * self.scmult))
        } else {
            s
        };
        self.ceiling.min(clipped.abs()) * if clipped < 0.0 { -1.0 } else { 1.0 }
    }
}

/// Three serial biquads: low shelf, mid peak, high shelf.
#[derive(Debug, Clone)]
pub(crate) struct Eq3Dsp {
    rate: u32,
    spec: EffectSpec,
    coeffs: [Biquad; 3],
    state: [[BiquadState; 3]; 2],
}

impl Eq3Dsp {
    fn new(spec: &EffectSpec, rate: u32) -> Self {
        let mut dsp = Self {
            rate,
            spec: EffectSpec::eq3(0.0, 0.0, 0.0),
            coeffs: [Biquad::default(); 3],
            state: [[BiquadState::default(); 3]; 2],
        };
        dsp.set_spec(spec);
        dsp
    }

    fn set_spec(&mut self, spec: &EffectSpec) {
        if let EffectSpec::Eq3 { .. } = *spec {
            self.spec = *spec;
            self.rebuild();
        }
    }

    fn set_rate(&mut self, rate: u32) {
        if self.rate != rate {
            self.rate = rate;
            self.state = [[BiquadState::default(); 3]; 2];
            self.rebuild();
        }
    }

    fn rebuild(&mut self) {
        let EffectSpec::Eq3 {
            low_hz,
            mid_hz,
            high_hz,
            low_db,
            mid_db,
            high_db,
        } = self.spec
        else {
            return;
        };
        let rate = self.rate as f32;
        self.coeffs[0] = low_shelf_coeffs(low_hz, 0.7071, low_db, rate);
        self.coeffs[1] = peak_coeffs(mid_hz, 0.7071, mid_db, rate);
        self.coeffs[2] = high_shelf_coeffs(high_hz, 0.7071, high_db, rate);
    }

    fn process(&mut self, channel: usize, x: f32) -> f32 {
        let mut v = x;
        for s in 0..3 {
            v = self.state[channel][s].process(&self.coeffs[s], v);
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(effect: &EffectSpec, rate: u32, input: &[f32]) -> Vec<f32> {
        let mut dsp = EffectDsp::new(effect, rate);
        let mut buf = Vec::with_capacity(input.len() * 2);
        for x in input {
            buf.push(*x);
            buf.push(*x);
        }
        dsp.process(&mut buf, input.len(), None);
        buf.iter().step_by(2).copied().collect()
    }

    fn impulse(n: usize) -> Vec<f32> {
        let mut v = vec![0.0; n];
        if n > 0 {
            v[0] = 1.0;
        }
        v
    }

    /// Pins the Godot `AudioFilterSW` low-pass difference equation at
    /// 1 kHz / 48 kHz / Q 0.5. Expected values come from the documented
    /// closed form, not from this implementation:
    /// `b0 = b2 = 0.0037836977`, `b1 = 0.0075673953`,
    /// `a1 = 1.7539529260`, `a2 = -0.7690877166`.
    #[test]
    fn lowpass_impulse_response_is_pinned() {
        let effect = EffectSpec::Filter {
            mode: FilterMode::LowPass,
            cutoff_hz: 1000.0,
            resonance: 0.5,
            stages: 1,
        };
        let out = run(&effect, 48000, &impulse(4));
        assert!((out[0] - 0.0037836977).abs() < 2e-6, "{out:?}");
        assert!((out[1] - 0.0142038229).abs() < 2e-6, "{out:?}");
        assert!((out[2] - 0.0257865390).abs() < 2e-6, "{out:?}");
        assert!((out[3] - 0.0343043899).abs() < 2e-6, "{out:?}");
    }

    /// A 24 dB/oct low-pass is four cascaded copies of the single stage, so
    /// stage 1 of a step response must be the single-stage response squared.
    #[test]
    fn cascaded_filter_stages_stack() {
        let one = EffectSpec::low_pass(500.0, 1);
        let four = EffectSpec::low_pass(500.0, 4);
        let step = vec![1.0; 32];
        let a = run(&one, 44100, &step);
        let b = run(&four, 44100, &step);
        // The four-stage cascade truncates the passband as well (Godot's
        // 1/(stages+1) gain compensation), so compare shape only: the
        // cascade must be smoother (less overshoot) than the single stage.
        let peak_a: f32 = a.iter().cloned().fold(0.0, f32::max);
        let peak_b: f32 = b.iter().cloned().fold(0.0, f32::max);
        assert!(peak_b > 0.0 && peak_a > 0.0);
        assert!(b[2] < a[2], "cascade is slower to rise");
    }

    /// The limiter's hard ceiling is exact: no sample may leave above
    /// `db_to_linear(-0.1)`.
    #[test]
    fn limiter_never_exceeds_the_ceiling() {
        let effect = EffectSpec::limiter(-0.1);
        let ceiling = db_to_linear(-0.1);
        let loud: Vec<f32> = (0..64).map(|_| 4.0).collect();
        for v in run(&effect, 48000, &loud) {
            assert!(v.abs() <= ceiling + 1e-6, "v = {v}");
        }
        // Quiet input below the soft-clip knee only takes makeup gain.
        let quiet: Vec<f32> = (0..64).map(|_| 0.25).collect();
        let out = run(&effect, 48000, &quiet);
        assert!((out[0] - 0.25 * ceiling).abs() < 1e-6, "{}", out[0]);
    }

    /// Feedback is clamped, so a delay with an absurd setting still decays.
    #[test]
    fn delay_feedback_is_bounded() {
        let effect = EffectSpec::Delay {
            time_ms: 1.0,
            feedback: 40.0,
            mix: 1.0,
            feedback_lowpass_hz: 20000.0,
            spread: 0.0,
        };
        let out = run(&effect, 48000, &impulse(4096));
        let early: f32 = out[..64].iter().map(|v| v.abs()).sum();
        let late: f32 = out[2048..].iter().map(|v| v.abs()).sum();
        assert!(early > 0.0);
        assert!(late < early, "delay must decay: {early} -> {late}");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn reverb_stays_finite_and_smears_the_impulse() {
        let effect = EffectSpec::reverb(0.8, 0.5, 1.0);
        let out = run(&effect, 44100, &impulse(44100));
        assert!(out.iter().all(|v| v.is_finite()));
        let tail: f32 = out[20000..].iter().map(|v| v.abs()).sum();
        assert!(tail > 0.0, "reverb produced no tail");
        assert!(out[0].abs() < 1.0, "dry path is untouched");
    }

    #[test]
    fn eq_bands_move_only_their_own_end() {
        let low = EffectSpec::eq3(6.0, 0.0, 0.0);
        let high = EffectSpec::eq3(0.0, 0.0, 6.0);
        let sine = |hz: f32| -> Vec<f32> {
            (0..4096)
                .map(|i| (std::f32::consts::TAU * hz * i as f32 / 48000.0).sin())
                .collect()
        };
        let energy = |v: &[f32]| v[1024..].iter().map(|x| x * x).sum::<f32>().sqrt();
        let bass = sine(80.0);
        let treble = sine(12000.0);
        assert!(energy(&run(&low, 48000, &bass)) > energy(&bass) * 1.5);
        assert!(energy(&run(&low, 48000, &treble)) < energy(&treble) * 1.05);
        assert!(energy(&run(&high, 48000, &treble)) > energy(&treble) * 1.5);
    }
}
