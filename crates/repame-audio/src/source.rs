//! [`AudioSource`]: encoded sound bytes plus format sniffing.
//!
//! Sources stay encoded until [`crate::Audio::add_source`] decodes them
//! into the backend. Kenney packs ship Ogg Vorbis, which kira decodes
//! natively  - no conversion step needed.

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

/// Encoded sound bytes, registered once via [`crate::Audio::add_source`].
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

/// Minimal mono 16-bit WAV writer: sine burst with exponential decay.
/// Pure (no kira) so tests and placeholders work on every platform.
pub fn synth_sine_wav(freq_hz: f32, secs: f32, sample_rate: u32) -> Vec<u8> {
    let n = (sample_rate as f32 * secs) as usize;
    let mut out = Vec::with_capacity(44 + n * 2);
    let data_len = (n * 2) as u32;
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for i in 0..n {
        let t = i as f32 / sample_rate as f32;
        let v = (2.0 * std::f32::consts::PI * freq_hz * t).sin() * (-t * 30.0).exp() * 0.5;
        out.extend_from_slice(&((v.clamp(-1.0, 1.0) * i16::MAX as f32) as i16).to_le_bytes());
    }
    out
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
