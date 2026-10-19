//! Game sound on cpal plus symphonia with an atomic command channel.
//!
//! [`Audio`] is the game-thread handle: [`SoundBank`] one-shots,
//! [`Music`] direction, [`RigAudio`] rig-event routing, the [`BusTree`]
//! mixer, and [`AudioListener3D`] spatial playback.
//! [`Audio::try_init`] bails without a device; [`Audio::noop`] runs silent.
//!
//! A cue is either bytes or a table, and the table is the newer half.
//! [`CueSpec`] carries the tuning: waveform, frequency path, amplitude
//! envelope, vibrato, and a white-noise blend whose [`NoiseShape`] says
//! whether the noise rides the envelope or its square, so grit can die ahead
//! of the tone it sits on. [`CueSpec::render`] makes frames,
//! [`CueSpec::render_wav`] makes PCM16 bytes, and [`SoundBank::load_synth`]
//! and [`SoundBank::request_synth`] register a whole table of variations, so a
//! game keeps no DSP and no encoder of its own.
//!
//! Under the bank sit the parts that shape what a cue becomes: the
//! [`BusTree`]'s sends, sidechains and mix order, the [`EffectSpec`] chain of
//! delays, reverbs and filters, spatial panning, and the loader worker that
//! decodes off the game thread.

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
