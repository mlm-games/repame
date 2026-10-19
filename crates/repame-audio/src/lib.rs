//! Game sound on cpal plus symphonia with an atomic command channel.
//! [`Audio`] is the game-thread handle: [`SoundBank`] one-shots,
//! [`Music`] direction, [`RigAudio`] rig-event routing, the [`BusTree`]
//! mixer, and [`AudioListener3D`] spatial playback.
//! [`CueSpec`] renders a declarative cue table to frames, so a game keeps its
//! tuning table instead of a DSP and a WAV encoder.
//! [`Audio::try_init`] bails without a device; [`Audio::noop`] runs silent.

mod audio;
mod bank;
mod bus;
mod channels;
mod command;
mod cue;
mod decode;
mod effects;
mod engine;
mod loader;
mod mixer;
mod music;
mod rig;
mod source;
mod spatial;
mod state;
mod synth;

pub use audio::Audio;
pub use bank::{CueDef, CueRequest, SoundBank, Variation, spatial_2d};
pub use bus::{BusDef, BusError, BusId, BusSend, BusTree, MASTER};
pub use channels::{AudioChannel, AudioChannels, db_to_linear, linear_to_db};
pub use command::{
    EngineEvent, GameAudioLink, MusicStart, MusicTransition, PlayCmd, RealtimeCommand,
    SharedFrames, ThreadAudioLink, audio_link,
};
pub use cue::Cue;
pub use decode::{decode_bytes, resample_linear};
pub(crate) use decode::{decode_to_device, retarget_to};
pub use effects::{EffectSpec, FilterMode};
pub use engine::Engine;
pub use music::{Music, StemDef, SyncPoint};
pub use rig::RigAudio;
pub use source::{AudioFormat, AudioSource, sniff_format, synth_sine_wav};
pub use spatial::{
    AttenuationModel, AudioListener3D, SpatialOutput, SpatialParams, SpatialVoice3D, Vec3,
    attenuation_gain, doppler_pitch, spatial_3d,
};
pub use state::AudioState;
pub use synth::{
    AmplitudeEnvelope, CueSpec, NoiseShape, Sweep, SynthError, Vibrato, Waveform, wav_pcm16,
};

pub mod prelude {
    pub use crate::{
        AmplitudeEnvelope, AttenuationModel, Audio, AudioChannel, AudioChannels, AudioFormat,
        AudioListener3D, AudioSource, AudioState, BusError, BusId, BusTree, CueDef, CueRequest,
        CueSpec, EffectSpec, FilterMode, Music, RigAudio, SoundBank, SpatialParams, StemDef, Sweep,
        SyncPoint, SynthError, Variation, Vibrato, Waveform, db_to_linear, linear_to_db,
        spatial_2d, wav_pcm16,
    };
}
