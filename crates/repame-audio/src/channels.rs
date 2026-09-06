//! [`AudioChannels`]: master / music / sfx / ui buses (linear gains).
//!
//! Matches the volume settings both shells already persist
//! (`master_volume`, `sfx_volume`, `music_volume` in `save.ron`)
//! plus a `ui` bus, mirroring `game_utils_bevy::audio::AudioChannels`.

/// Named bus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AudioChannel {
    Master,
    Music,
    Sfx,
    Ui,
}

/// Four-bus mixer. Effective gain = `master * bus`, linear.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioChannels {
    master: f32,
    music: f32,
    sfx: f32,
    ui: f32,
}

impl Default for AudioChannels {
    /// Mirrors the shipped defaults (`music 0.8` in `rozvp::app`).
    fn default() -> Self {
        Self {
            master: 1.0,
            music: 0.8,
            sfx: 1.0,
            ui: 1.0,
        }
    }
}

impl AudioChannels {
    /// Raw bus gain (master included).
    pub fn get(&self, channel: AudioChannel) -> f32 {
        match channel {
            AudioChannel::Master => self.master,
            AudioChannel::Music => self.music,
            AudioChannel::Sfx => self.sfx,
            AudioChannel::Ui => self.ui,
        }
    }

    /// Set one bus (clamped to `>= 0`; master included).
    pub fn set(&mut self, channel: AudioChannel, gain: f32) {
        let gain = gain.max(0.0);
        match channel {
            AudioChannel::Master => self.master = gain,
            AudioChannel::Music => self.music = gain,
            AudioChannel::Sfx => self.sfx = gain,
            AudioChannel::Ui => self.ui = gain,
        }
    }

    /// Linear gain for a bus after the master stage.
    pub fn effective(&self, channel: AudioChannel) -> f32 {
        if channel == AudioChannel::Master {
            self.master
        } else {
            self.master * self.get(channel)
        }
    }

    pub fn master(&self) -> f32 {
        self.master
    }

    pub fn music(&self) -> f32 {
        self.music
    }

    pub fn sfx(&self) -> f32 {
        self.sfx
    }

    pub fn ui(&self) -> f32 {
        self.ui
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buses_stage_behind_master() {
        let mut ch = AudioChannels::default();
        assert!((ch.effective(AudioChannel::Sfx) - 1.0).abs() < 1e-6);
        assert!((ch.effective(AudioChannel::Music) - 0.8).abs() < 1e-6);
        assert_eq!(ch.ui(), 1.0);
        ch.set(AudioChannel::Master, 0.5);
        assert!((ch.effective(AudioChannel::Sfx) - 0.5).abs() < 1e-6);
        ch.set(AudioChannel::Sfx, -2.0);
        assert_eq!(ch.get(AudioChannel::Sfx), 0.0);
    }
}
