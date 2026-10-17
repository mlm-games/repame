//! [`AudioSource`]: encoded sound bytes plus format sniffing.
//! Sources stay encoded until decode; Ogg Vorbis needs no conversion.
//! [`synth_sine_wav`] is the oldest synth helper here; new cue tables should
//! use [`crate::CueSpec`] and [`crate::wav_pcm16`] instead.

use crate::command::SharedFrames;
use crate::synth::{AmplitudeEnvelope, CueSpec, Waveform, wav_pcm16};

/// Container/codec of encoded sound bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AudioFormat {
    Ogg,
    Mp3,
    Flac,
    Wav,
    #[default]
    Unknown,
}

/// Magic-byte sniffing: `OggS`, `ID3`/frame-sync, `fLaC`, `RIFF....WAVE`.
pub fn sniff_format(bytes: &[u8]) -> AudioFormat {
    if bytes.starts_with(b"OggS") {
        AudioFormat::Ogg
    } else if bytes.starts_with(b"ID3")
        || (bytes.len() > 2 && bytes[0] == 0xFF && bytes[1] & 0xE0 == 0xE0)
    {
        AudioFormat::Mp3
    } else if bytes.starts_with(b"fLaC") {
        AudioFormat::Flac
    } else if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
        AudioFormat::Wav
    } else {
        AudioFormat::Unknown
    }
}

/// Encoded sound bytes, registered once via add.
#[derive(Debug, Clone)]
pub struct AudioSource {
    bytes: Vec<u8>,
    format: AudioFormat,
}

impl AudioSource {
    /// Wrap encoded bytes; format is sniffed, not trusted from extension.
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        let format = sniff_format(&bytes);
        Self { bytes, format }
    }

    /// Raw encoded bytes (for the backend decoder).
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Sniffed container/codec.
    pub fn format(&self) -> AudioFormat {
        self.format
    }

    /// False for [`AudioFormat::Unknown`]; the backend rejects those.
    pub fn is_supported(&self) -> bool {
        self.format != AudioFormat::Unknown
    }
}

/// Minimal mono 16-bit WAV writer: sine burst with decay.
///
/// The first synth helper the games shipped with; it is now a thin
/// [`CueSpec`] preset (sine, a `1/30` s exponential decay, gain `0.5`), so
/// tuning and rendering live in [`crate::synth`]. Arguments that cannot render
/// (a zero rate, a non-positive length) yield a valid but empty WAV after a
/// warning, because the signature has no way to report the failure.
pub fn synth_sine_wav(freq_hz: f32, secs: f32, sample_rate: u32) -> Vec<u8> {
    let spec = CueSpec {
        waveform: Waveform::Sine,
        from_hz: freq_hz,
        to_hz: freq_hz,
        amplitude: AmplitudeEnvelope::Adsr {
            attack_s: 0.0,
            decay_s: 1.0 / 30.0,
            sustain: 0.0,
            release_s: 0.0,
        },
        gain: 0.5,
        secs,
        sample_rate,
        ..CueSpec::default()
    };
    match spec.render_wav() {
        Ok(wav) => wav,
        Err(error) => {
            log::warn!("synth_sine_wav({freq_hz}, {secs}, {sample_rate}): {error}");
            let empty = SharedFrames {
                sample_rate,
                channels: 1,
                frames: Vec::new(),
            };
            wav_pcm16(&empty).unwrap_or_default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sniffing_reads_magic_bytes() {
        assert_eq!(sniff_format(b"OggS...."), AudioFormat::Ogg);
        assert_eq!(sniff_format(b"ID3...."), AudioFormat::Mp3);
        assert_eq!(sniff_format(&[0xFF, 0xFB, 0x00]), AudioFormat::Mp3);
        assert_eq!(sniff_format(b"fLaC...."), AudioFormat::Flac);
        let mut wav = b"RIFF....WAVE".to_vec();
        wav[4..8].copy_from_slice(&[1, 2, 3, 4]);
        assert_eq!(sniff_format(&wav), AudioFormat::Wav);
        assert_eq!(sniff_format(b"nope"), AudioFormat::Unknown);
        assert_eq!(sniff_format(b""), AudioFormat::Unknown);
    }

    #[test]
    fn synth_wav_is_valid_mono_pcm() {
        let wav = synth_sine_wav(440.0, 0.1, 44100);
        let src = AudioSource::from_bytes(wav.clone());
        assert_eq!(src.format(), AudioFormat::Wav);
        assert!(src.is_supported());
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(&wav[36..40], b"data");
        let n = (44100.0 * 0.1) as usize;
        assert_eq!(wav.len(), 44 + n * 2);
        // Starts near zero (no click), finite throughout.
        let first = i16::from_le_bytes([wav[44], wav[45]]);
        assert!(first.abs() < 2000, "first = {first}");
        assert!(!AudioSource::from_bytes(vec![0, 1, 2]).is_supported());
    }
}
