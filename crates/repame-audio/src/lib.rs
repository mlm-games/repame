//! Game-idiomatic sound on the house stack (cpal + symphonia + atomics).
//!
//! [`Audio`] is the game-thread handle: [`SoundBank`] one-shots
//! (`play("footstep")` with variations, cooldowns, polyphony caps),
//! [`Music`] direction (crossfade, intensity stems, ducking), and
//! [`RigAudio`] forwarding of renamite rig events (`footstep`, `bite`)
//! into cues. Per frame the game pumps [`Audio::update`] with real dt.
//!
//! A lock-free command channel carries work to the audio thread, which
//! renders voices over cpal (desktop, Android AAudio, wasm AudioWorklet
//! host)  - the same shape as the shipping DAW engine, written fresh for
//! this crate's license. [`Audio::try_init`] fails soft without a device;
//! [`Audio::noop`] keeps game code and tests running silent.

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
