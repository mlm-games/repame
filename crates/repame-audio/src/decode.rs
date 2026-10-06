//! [`decode_bytes`]: encoded bytes to [`SharedFrames`] via symphonia.
//! Supports the Vorbis, MP3, FLAC, WAV, and AAC inputs in the pin.
//! Also hosts the device-rate converters: `resample` (windowed-sinc,
//! load time) and [`resample_linear`] (legacy realtime matcher).

use std::io::Cursor;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::default::get_codecs;

use crate::command::SharedFrames;
use crate::state::AudioState;

/// Decoded-output cap in frames. Music tracks pass; accidents do not.
const MAX_FRAMES: usize = 32_000_000;

/// Decode a whole sound file into interleaved f32 frames.
/// Stereo stays stereo; mono stays mono; 3+ channels keep L/R.
pub fn decode_bytes(bytes: &[u8]) -> Result<SharedFrames> {
    if bytes.is_empty() {
        anyhow::bail!("cannot decode empty input");
    }
    let mss = MediaSourceStream::new(Box::new(Cursor::new(bytes.to_vec())), Default::default());
    let mut format = symphonia::default::get_probe().probe(
        &Hint::new(),
        mss,
        FormatOptions::default(),
        MetadataOptions::default(),
    )?;
    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| anyhow!("no audio track found"))?;
    let track_id = track.id;
    let codec_params = track
        .codec_params
        .clone()
        .ok_or_else(|| anyhow!("track has no codec parameters"))?;
    let params = codec_params
        .audio()
        .ok_or_else(|| anyhow!("track is not an audio track"))?;
    let rate = params
        .sample_rate
        .ok_or_else(|| anyhow!("unknown sample rate"))?;
    let src_channels = params.channels.clone().map(|c| c.count()).unwrap_or(1);
    let channels = src_channels.min(2) as u8;

    let mut decoder = get_codecs().make_audio_decoder(params, &AudioDecoderOptions::default())?;
    let mut interleaved: Vec<f32> = Vec::new();
    loop {
        match format.next_packet() {
            Ok(Some(packet)) => {
                if packet.track_id != track_id {
                    continue;
                }
                match decoder.decode(&packet) {
                    Ok(decoded) => {
                        let mut packet_samples: Vec<f32> = Vec::new();
                        decoded.copy_to_vec_interleaved(&mut packet_samples);
                        if src_channels <= 2 {
                            interleaved.extend_from_slice(&packet_samples);
                        } else {
                            // Keep L/R, drop the rest.
                            for frame in packet_samples.chunks(src_channels) {
                                interleaved.push(frame[0]);
                                interleaved.push(frame.get(1).copied().unwrap_or(0.0));
                            }
                        }
                    }
                    Err(Error::DecodeError(e)) => {
                        log::warn!("skipping corrupt audio packet: {e}");
                        continue;
                    }
                    Err(e) => return Err(anyhow!("audio decode failed: {e}")),
                }
            }
            Ok(None) => break,
            Err(Error::ResetRequired) => break,
            Err(e) => return Err(e.into()),
        }
        if interleaved.len() / channels as usize > MAX_FRAMES {
            anyhow::bail!("decoded audio exceeds the {MAX_FRAMES}-frame cap");
        }
    }
    if interleaved.is_empty() {
        anyhow::bail!("decoded audio is empty");
    }
    Ok(SharedFrames {
        sample_rate: rate,
        channels,
        frames: interleaved,
    })
}

/// Linear resample to `target_hz`. No-op when rates match.
pub fn resample_linear(sound: &SharedFrames, target_hz: u32) -> SharedFrames {
    if sound.sample_rate == target_hz || sound.sample_rate == 0 || target_hz == 0 {
        return sound.clone();
    }
    let ratio = sound.sample_rate as f64 / target_hz as f64;
    let n_ch = sound.channels.max(1) as usize;
    let n_in = sound.len_frames();
    if n_in == 0 {
        return SharedFrames {
            sample_rate: target_hz,
            channels: sound.channels,
            frames: Vec::new(),
        };
    }
    let n_out = ((n_in as f64 / ratio) as usize).max(1);
    let mut frames = Vec::with_capacity(n_out * n_ch);
    for i in 0..n_out {
        let pos = i as f64 * ratio;
        let i0 = pos as usize;
        let frac = (pos - i0 as f64) as f32;
        let i1 = (i0 + 1).min(n_in.saturating_sub(1));
        for c in 0..n_ch {
            let a = sound.frames[i0 * n_ch + c];
            let b = sound.frames[i1 * n_ch + c];
            frames.push(a + (b - a) * frac);
        }
    }
    SharedFrames {
        sample_rate: target_hz,
        channels: sound.channels,
        frames,
    }
}

/// Phase-table resolution of the windowed-sinc converter (linear interp
/// between rows keeps phase error near -120 dB).
const SINC_PHASES: usize = 1024;
/// Sinc zero crossings per side at cutoff 0.5; stretched when decimating.
const SINC_CROSSINGS: f32 = 24.0;

/// Four-term Blackman-Harris window over `x` in `-1..=1`.
fn blackman_harris(x: f32) -> f32 {
    0.35875
        + 0.48829 * (std::f32::consts::PI * x).cos()
        + 0.14128 * (2.0 * std::f32::consts::PI * x).cos()
        + 0.01168 * (3.0 * std::f32::consts::PI * x).cos()
}

/// Windowed-sinc resample to `target_hz`. Each phase-table row is
/// normalized to unit DC gain; input edges replicate. Empty, zero-rate,
/// and matching-rate inputs pass through.
fn resample(sound: &SharedFrames, target_hz: u32) -> Result<SharedFrames> {
    if target_hz == 0 || sound.sample_rate == 0 || sound.sample_rate == target_hz {
        return Ok(sound.clone());
    }
    let n_ch = sound.channels.max(1) as usize;
    let n_in = sound.len_frames();
    if n_in == 0 {
        return Ok(SharedFrames {
            sample_rate: target_hz,
            channels: sound.channels,
            frames: Vec::new(),
        });
    }
    let ratio = sound.sample_rate as f64 / target_hz as f64;
    let n_out = ((n_in as f64 / ratio).round() as usize).max(1);
    if n_out > MAX_FRAMES {
        anyhow::bail!("resampled audio exceeds the {MAX_FRAMES}-frame cap");
    }
    let cutoff = 0.5f32 * (target_hz as f32 / sound.sample_rate as f32).min(1.0);
    let half = ((SINC_CROSSINGS / (2.0 * cutoff)).ceil() as usize).max(2);
    let taps = half * 2 + 1;
    let mut table = vec![0.0f32; (SINC_PHASES + 1) * taps];
    for p in 0..=SINC_PHASES {
        let frac = p as f32 / SINC_PHASES as f32;
        let row = p * taps;
        let mut sum = 0.0f32;
        for (j, v) in table[row..row + taps].iter_mut().enumerate() {
            let u = j as f32 - half as f32 - frac;
            *v = if u.abs() <= half as f32 {
                let s = 2.0 * cutoff * u;
                let sinc = if s.abs() < 1e-6 {
                    1.0
                } else {
                    (std::f32::consts::PI * s).sin() / (std::f32::consts::PI * s)
                };
                2.0 * cutoff * sinc * blackman_harris(u / half as f32)
            } else {
                0.0
            };
            sum += *v;
        }
        if sum.is_finite() && sum.abs() > 1e-6 {
            for v in &mut table[row..row + taps] {
                *v /= sum;
            }
        }
    }
    let mut frames = vec![0.0f32; n_out * n_ch];
    let mut coefs = Vec::with_capacity(taps);
    let last = (n_in - 1) as i64;
    for m in 0..n_out {
        let pos = m as f64 * ratio;
        let base = pos.floor();
        let ph = ((pos - base) as f32) * SINC_PHASES as f32;
        let p = (ph as usize).min(SINC_PHASES - 1);
        let a = ph - p as f32;
        let row0 = p * taps;
        let row1 = row0 + taps;
        coefs.clear();
        for j in 0..taps {
            let v0 = table[row0 + j];
            coefs.push(v0 + a * (table[row1 + j] - v0));
        }
        let start = base as i64 - half as i64;
        for (j, &gain) in coefs.iter().enumerate() {
            let frame = (start + j as i64).clamp(0, last) as usize * n_ch;
            for c in 0..n_ch {
                frames[m * n_ch + c] += gain * sound.frames[frame + c];
            }
        }
    }
    Ok(SharedFrames {
        sample_rate: target_hz,
        channels: sound.channels,
        frames,
    })
}

/// Convert `sound` to the device rate in `state` when one is known.
fn to_device_rate(sound: SharedFrames, state: Option<&AudioState>) -> Result<SharedFrames> {
    let Some(state) = state else { return Ok(sound) };
    let target = state.sample_rate();
    if target == 0 || target == sound.sample_rate {
        return Ok(sound);
    }
    resample(&sound, target)
}

/// Decode encoded bytes straight to the device rate, ready to hand to a voice.
pub(crate) fn decode_to_device(
    bytes: &[u8],
    state: Option<&AudioState>,
) -> Result<Arc<SharedFrames>> {
    Ok(Arc::new(to_device_rate(decode_bytes(bytes)?, state)?))
}

/// Swap `sound` to `target` in place; failures keep the original.
pub fn retarget_to(sound: &mut Arc<SharedFrames>, target: u32) {
    if target == 0 || sound.sample_rate == target {
        return;
    }
    match resample(sound, target) {
        Ok(frames) => *sound = Arc::new(frames),
        Err(e) => log::warn!("sample-rate conversion skipped: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth_sine_wav;

    #[test]
    fn wav_round_trips() {
        let wav = synth_sine_wav(440.0, 0.1, 22050);
        let decoded = decode_bytes(&wav).expect("synth wav decodes");
        assert_eq!(decoded.sample_rate, 22050);
        assert_eq!(decoded.channels, 1);
        assert_eq!(decoded.len_frames(), (22050.0 * 0.1) as usize);
        assert!(decoded.frames.iter().all(|v| v.is_finite()));
        let peak: f32 = decoded.frames.iter().map(|v| v.abs()).fold(0.0, f32::max);
        assert!(peak > 0.3, "peak = {peak}");
        assert!(decoded.frames[0].abs() < 0.05);
    }

    #[test]
    fn garbage_and_empty_fail_cleanly() {
        assert!(decode_bytes(&[]).is_err());
        assert!(decode_bytes(&[7, 7, 7, 7]).is_err());
        assert!(decode_bytes(b"ID3....nope").is_err());
    }

    #[test]
    fn resample_empty_stays_empty() {
        let empty = SharedFrames {
            sample_rate: 44100,
            channels: 1,
            frames: Vec::new(),
        };
        let out = resample_linear(&empty, 48000);
        assert_eq!(out.sample_rate, 48000);
        assert!(out.frames.is_empty());
    }

    #[test]
    fn resample_changes_length_not_shape() {
        let wav = synth_sine_wav(440.0, 0.2, 44100);
        let decoded = decode_bytes(&wav).unwrap();
        let up = resample_linear(&decoded, 48000);
        assert_eq!(up.sample_rate, 48000);
        let expect = (decoded.len_frames() as f64 * 48000.0 / 44100.0) as usize;
        assert!((up.len_frames() as i64 - expect as i64).abs() <= 2);
        let peak: f32 = up.frames.iter().map(|v| v.abs()).fold(0.0, f32::max);
        assert!(peak > 0.3, "peak = {peak}");
        let same = resample_linear(&decoded, 44100);
        assert_eq!(same.frames, decoded.frames);
    }

    fn sine_at(hz: f32, rate: u32) -> SharedFrames {
        SharedFrames {
            sample_rate: rate,
            channels: 1,
            frames: (0..rate as usize)
                .map(|i| {
                    (2.0 * std::f64::consts::PI * hz as f64 * i as f64 / rate as f64).sin() as f32
                        * 0.5
                })
                .collect(),
        }
    }

    #[test]
    fn sinc_resample_keeps_sine_intact() {
        for (src, dst) in [(32000u32, 48000u32), (48000, 32000), (44100, 48000)] {
            let sound = sine_at(1000.0, src);
            let out = resample(&sound, dst).unwrap();
            assert_eq!(out.sample_rate, dst);
            let expect = (sound.len_frames() as f64 * dst as f64 / src as f64).round() as usize;
            assert_eq!(out.len_frames(), expect, "{src} -> {dst}");
            // Skip the replicated-edge transient; compare against the ideal sine.
            let skip = 512;
            let (mut sig, mut err) = (0.0f64, 0.0f64);
            for m in skip..out.len_frames() - skip {
                let ideal =
                    0.5 * (2.0 * std::f64::consts::PI * 1000.0 * m as f64 / dst as f64).sin();
                let d = out.frames[m] as f64 - ideal;
                sig += ideal * ideal;
                err += d * d;
            }
            let snr = 10.0 * (sig / err).log10();
            assert!(snr > 60.0, "{src} -> {dst}: snr = {snr:.1} dB");
        }
    }

    #[test]
    fn sinc_resample_passes_through_noop_and_empty() {
        let sound = sine_at(440.0, 32000);
        let same = resample(&sound, 32000).unwrap();
        assert!(same.frames == sound.frames);
        let empty = SharedFrames {
            sample_rate: 44100,
            channels: 1,
            frames: Vec::new(),
        };
        let out = resample(&empty, 48000).unwrap();
        assert_eq!(out.sample_rate, 48000);
        assert!(out.frames.is_empty());
    }
}
