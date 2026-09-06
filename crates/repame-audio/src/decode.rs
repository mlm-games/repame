//! [`decode_bytes`]: encoded bytes -> [`SharedFrames`] via symphonia.
//!
//! Registry symphonia 0.6 (same pin as the shipping DAW): Vorbis/Ogg,
//! MP3, FLAC, WAV/PCM, AAC and the rest of the `all` feature set. Kenney
//! packs ship Vorbis Ogg  - no conversion step needed. Deliberately out:
//! Opus (needs the LGPL `symphonia-adapter-oporus`; Vorbis covers the
//! game asset pipeline). Also hosts [`resample_linear`], the device-rate
//! matcher the engine uses per voice.

use std::io::Cursor;

use anyhow::{Result, anyhow};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::default::get_codecs;

use crate::command::SharedFrames;

/// Decoded-output safety cap (frames, all channels counted once).
/// ~12 min of mono at 44.1 kHz; music tracks pass, accidents do not.
const MAX_FRAMES: usize = 32_000_000;

/// Decode a whole sound file into interleaved f32 frames.
/// Stereo stays stereo; mono stays mono; 3+ channels keep L/R.
/// Returns the file's native rate (the engine resamples to the device).
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

/// Linear-interpolation resample to `target_hz`. No-op when rates match.
/// Pure (no device) so banks can pre-match the engine rate at load time.
pub fn resample_linear(sound: &SharedFrames, target_hz: u32) -> SharedFrames {
    if sound.sample_rate == target_hz || sound.sample_rate == 0 || target_hz == 0 {
        return sound.clone();
    }
    let ratio = sound.sample_rate as f64 / target_hz as f64;
    let n_ch = sound.channels.max(1) as usize;
    let n_in = sound.len_frames();
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
}
