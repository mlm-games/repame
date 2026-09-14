//! [`AudioChannels`]: master, music, sfx, and ui buses (linear gains).

/// Named bus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AudioChannel {
    Master,
    Music,
    Sfx,
    Ui,
}

/// Linear gain (`0..1`, `1` is unchanged) to decibels.
/// Volume work is logarithmic: each -6 dB halves amplitude.
/// `0.0` maps to negative infinity, so mute the bus at zero instead.
///
/// ```rust
/// use repame_audio::{db_to_linear, linear_to_db};
///
/// assert!((linear_to_db(1.0)).abs() < 1e-6);
/// assert!((linear_to_db(0.5) + 6.0206).abs() < 1e-3);
/// assert_eq!(linear_to_db(0.0), f32::NEG_INFINITY);
/// ```
pub fn linear_to_db(linear: f32) -> f32 {
    if linear <= 0.0 {
        f32::NEG_INFINITY
    } else {
        20.0 * linear.log10()
    }
}

/// Decibels to linear bus gain. Negative infinity maps to `0.0`,
/// as does anything at or below -80 dB.
///
/// ```rust
/// use repame_audio::{db_to_linear, linear_to_db};
///
/// assert!((db_to_linear(0.0) - 1.0).abs() < 1e-6);
/// assert_eq!(db_to_linear(f32::NEG_INFINITY), 0.0);
/// for v in [0.1, 0.5, 0.8, 1.0] {
///     assert!((db_to_linear(linear_to_db(v)) - v).abs() < 1e-5);
/// }
/// ```
pub fn db_to_linear(db: f32) -> f32 {
    if !db.is_finite() || db <= -80.0 {
        0.0
    } else {
        10.0f32.powf(db / 20.0)
    }
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

    /// Set one bus (clamped to `>= 0`).
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

    /// Bus volume in decibels (raw bus gain, master not included).
    pub fn get_db(&self, channel: AudioChannel) -> f32 {
        linear_to_db(self.get(channel))
    }

    /// Set one bus from decibels. Values at or below -80 dB mute.
    pub fn set_db(&mut self, channel: AudioChannel, db: f32) {
        self.set(channel, db_to_linear(db));
    }

    /// Effective volume in decibels after the master stage.
    pub fn effective_db(&self, channel: AudioChannel) -> f32 {
        linear_to_db(self.effective(channel))
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

    #[test]
    fn db_and_linear_round_trip() {
        assert!((linear_to_db(1.0)).abs() < 1e-6);
        assert!((linear_to_db(0.5) + 6.0206).abs() < 1e-3);
        assert_eq!(linear_to_db(0.0), f32::NEG_INFINITY);
        assert_eq!(db_to_linear(f32::NEG_INFINITY), 0.0);
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-6);
        // Slider round trip: linear -> db -> linear.
        for v in [0.1, 0.5, 0.8, 1.0] {
            assert!((db_to_linear(linear_to_db(v)) - v).abs() < 1e-5);
        }
        // Bus -6 dB halves the amplitude.
        let mut ch = AudioChannels::default();
        ch.set_db(AudioChannel::Music, -6.0206);
        assert!((ch.get(AudioChannel::Music) - 0.5).abs() < 1e-3);
        assert!((ch.effective_db(AudioChannel::Music) - -6.0206).abs() < 1e-3);
    }
}
