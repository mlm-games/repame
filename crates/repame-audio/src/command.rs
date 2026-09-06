//! Command + event channels between the game thread and the audio thread.
//!
//! Game -> audio: [`RealtimeCommand`], sent with `send_spin` (never parks),
//! drained with `try_recv` on the callback. Audio -> game: [`EngineEvent`]
//! (voice completions for reaping and music handoff). Queues are unbounded;
//! game command rates are tiny, and overflow policy is the caller's
//! (banks enforce polyphony before sending).

use std::sync::Arc;

use web_workers::sync::mpsc::{Receiver, Sender, channel};

use crate::AudioChannel;
use crate::state::AudioState;

/// Decoded PCM shared with the audio thread. Interleaved f32,
/// mono (`channels == 1`) or stereo (`channels == 2`).
#[derive(Debug, Clone)]
pub struct SharedFrames {
    pub sample_rate: u32,
    pub channels: u8,
    pub frames: Vec<f32>,
}

impl SharedFrames {
    /// Frame count (samples divided by channel count).
    pub fn len_frames(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.frames.len() / self.channels as usize
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len_frames() == 0
    }
}

/// Voice start request. `voice` comes from [`AudioState::alloc_voice`].
#[derive(Debug, Clone)]
pub struct PlayCmd {
    pub voice: u64,
    pub sound: Arc<SharedFrames>,
    /// Linear gain before bus staging.
    pub gain: f32,
    /// `1.0` = natural (also shifts pitch).
    pub rate: f32,
    /// Stereo position, `-1.0` (left) .. `1.0` (right).
    pub pan: f32,
    pub bus: AudioChannel,
    pub looping: bool,
}

/// Game -> audio commands. Never blocks; unknown voice ids are ignored.
#[derive(Debug, Clone)]
pub enum RealtimeCommand {
    Play(PlayCmd),
    Stop(u64),
    StopBus(AudioChannel),
    SetBus {
        bus: AudioChannel,
        gain: f32,
    },
    Pause(bool),
    /// Ramp a voice's gain to `target` over `secs` (linear). Completing
    /// at `0.0` finishes the voice like a natural end.
    Fade {
        voice: u64,
        target: f32,
        secs: f32,
    },
}

/// Audio -> game events, drained in [`crate::Audio::update`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineEvent {
    /// A non-looping voice ran to completion (reap accounting,
    /// music-track handoff).
    Finished(u64),
}

impl EngineEvent {
    /// Voice id behind the event.
    pub fn voice(&self) -> u64 {
        match self {
            Self::Finished(v) => *v,
        }
    }
}

/// Game-thread endpoint: send commands, read completions, share state.
pub struct GameAudioLink {
    pub tx: Sender<RealtimeCommand>,
    pub events: Receiver<EngineEvent>,
    pub state: Arc<AudioState>,
}

/// Audio-thread endpoint: drain commands, report completions.
pub struct ThreadAudioLink {
    pub rx: Receiver<RealtimeCommand>,
    pub events: Sender<EngineEvent>,
    pub state: Arc<AudioState>,
}

/// Linked pair plus fresh shared state.
pub fn audio_link() -> (GameAudioLink, ThreadAudioLink) {
    let (tx, rx) = channel();
    let (events_tx, events_rx) = channel();
    let state = Arc::new(AudioState::new());
    (
        GameAudioLink {
            tx,
            events: events_rx,
            state: state.clone(),
        },
        ThreadAudioLink {
            rx,
            events: events_tx,
            state,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frames() -> Arc<SharedFrames> {
        Arc::new(SharedFrames {
            sample_rate: 44100,
            channels: 1,
            frames: vec![0.0; 100],
        })
    }

    #[test]
    fn frames_len_counts_channels() {
        let mono = SharedFrames {
            sample_rate: 44100,
            channels: 1,
            frames: vec![0.0; 100],
        };
        assert_eq!(mono.len_frames(), 100);
        let stereo = SharedFrames {
            sample_rate: 48000,
            channels: 2,
            frames: vec![0.0; 100],
        };
        assert_eq!(stereo.len_frames(), 50);
        assert!(
            SharedFrames {
                sample_rate: 0,
                channels: 0,
                frames: vec![],
            }
            .is_empty()
        );
    }

    #[test]
    fn commands_round_trip_without_blocking() {
        let (game, thread) = audio_link();
        game.tx
            .send_spin(RealtimeCommand::Play(PlayCmd {
                voice: game.state.alloc_voice(),
                sound: frames(),
                gain: 0.8,
                rate: 1.0,
                pan: -0.5,
                bus: AudioChannel::Sfx,
                looping: false,
            }))
            .expect("send");
        game.tx
            .send_spin(RealtimeCommand::StopBus(AudioChannel::Music))
            .expect("send");
        let first = thread.rx.try_recv().expect("first command");
        assert!(matches!(first, RealtimeCommand::Play(_)));
        let second = thread.rx.try_recv().expect("second command");
        assert!(matches!(
            second,
            RealtimeCommand::StopBus(AudioChannel::Music)
        ));
        assert!(thread.rx.try_recv().is_err());
        // Back-channel works too.
        thread
            .events
            .send_spin(EngineEvent::Finished(1))
            .expect("event");
        assert_eq!(
            game.events.try_recv().expect("event"),
            EngineEvent::Finished(1)
        );
    }
}
