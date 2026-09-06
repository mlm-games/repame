//! [`AudioState`]: atomics shared by the game thread and the audio thread.
//!
//! Transport flags, bus gains, voice-id generation, overrun counter.
//! All `Relaxed` (same as the shipping DAW engine): these are signals,
//! not synchronization points. Sample-accurate work travels through
//! [`crate::command`] instead.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use crate::AudioChannel;

/// Thread-shared audio state. Clone shares (wrap in `Arc` once).
pub struct AudioState {
    /// Master transport gate: the callback renders silence while false
    /// (app suspend, Android lifecycle, explicit mute).
    pub playing: AtomicBool,
    /// Device sample rate in Hz. `0` until the stream opens.
    sample_rate: AtomicU32,
    /// Per-bus linear gains as f32 bits: master, music, sfx, ui.
    bus_gains: [AtomicU32; 4],
    /// Voice-id generator. `0` is reserved (invalid id).
    next_voice: AtomicU64,
    /// Callback overrun counter (debugging / telemetry).
    pub xruns: AtomicU64,
}

impl AudioState {
    pub fn new() -> Self {
        Self {
            playing: AtomicBool::new(true),
            sample_rate: AtomicU32::new(0),
            bus_gains: [
                AtomicU32::new(1.0_f32.to_bits()),
                AtomicU32::new(0.8_f32.to_bits()),
                AtomicU32::new(1.0_f32.to_bits()),
                AtomicU32::new(1.0_f32.to_bits()),
            ],
            next_voice: AtomicU64::new(1),
            xruns: AtomicU64::new(0),
        }
    }

    fn bus_index(channel: AudioChannel) -> usize {
        match channel {
            AudioChannel::Master => 0,
            AudioChannel::Music => 1,
            AudioChannel::Sfx => 2,
            AudioChannel::Ui => 3,
        }
    }

    /// Allocate a fresh voice id (never `0`).
    pub fn alloc_voice(&self) -> u64 {
        self.next_voice.fetch_add(1, Ordering::Relaxed)
    }

    /// Device rate in Hz, or `0` when no stream is open.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate.load(Ordering::Relaxed)
    }

    pub fn set_sample_rate(&self, hz: u32) {
        self.sample_rate.store(hz, Ordering::Relaxed);
    }

    /// Linear bus gain (clamped to `>= 0` on write).
    pub fn bus_gain(&self, channel: AudioChannel) -> f32 {
        f32::from_bits(self.bus_gains[Self::bus_index(channel)].load(Ordering::Relaxed))
    }

    pub fn set_bus_gain(&self, channel: AudioChannel, gain: f32) {
        self.bus_gains[Self::bus_index(channel)].store(gain.max(0.0).to_bits(), Ordering::Relaxed);
    }
}

impl Default for AudioState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_channels() {
        let s = AudioState::new();
        assert!(s.playing.load(Ordering::Relaxed));
        assert_eq!(s.sample_rate(), 0);
        assert!((s.bus_gain(AudioChannel::Music) - 0.8).abs() < 1e-6);
        assert!((s.bus_gain(AudioChannel::Sfx) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn gains_round_trip_and_clamp() {
        let s = AudioState::new();
        s.set_bus_gain(AudioChannel::Master, 0.5);
        assert!((s.bus_gain(AudioChannel::Master) - 0.5).abs() < 1e-6);
        s.set_bus_gain(AudioChannel::Ui, -3.0);
        assert_eq!(s.bus_gain(AudioChannel::Ui), 0.0);
        s.set_sample_rate(48000);
        assert_eq!(s.sample_rate(), 48000);
    }

    #[test]
    fn voice_ids_are_unique_and_nonzero() {
        use std::collections::HashSet;
        let s = AudioState::new();
        let ids: HashSet<u64> = (0..256).map(|_| s.alloc_voice()).collect();
        assert_eq!(ids.len(), 256);
        assert!(!ids.contains(&0));
    }
}
