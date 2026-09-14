//! Game sound on cpal plus symphonia with an atomic command channel.
//! [`Audio`] is the game-thread handle: [`SoundBank`] one-shots,
//! [`Music`] direction, and [`RigAudio`] rig-event routing.
//! [`Audio::try_init`] bails without a device; [`Audio::noop`] runs silent.

mod audio;
mod bank;
mod channels;
mod command;
mod decode;
mod engine;
mod music;
mod rig;
mod source;
mod state;

pub use audio::Audio;
pub use bank::{CueDef, SoundBank, Variation, spatial_2d};
pub use channels::{AudioChannel, AudioChannels, db_to_linear, linear_to_db};
pub use command::{
    EngineEvent, GameAudioLink, PlayCmd, RealtimeCommand, SharedFrames, ThreadAudioLink, audio_link,
};
pub use decode::{decode_bytes, resample_linear};
pub use engine::Engine;
pub use music::{Music, StemDef};
pub use rig::RigAudio;
pub use source::{AudioFormat, AudioSource, sniff_format, synth_sine_wav};
pub use state::AudioState;

pub mod prelude {
    pub use crate::{
        Audio, AudioChannel, AudioChannels, AudioFormat, AudioSource, AudioState, CueDef, Music,
        RigAudio, SoundBank, StemDef, Variation, db_to_linear, linear_to_db,
    };
}
