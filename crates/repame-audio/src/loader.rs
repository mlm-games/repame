//! Background decode worker: symphonia decode + device-rate resample run
//! off the game thread (repadio pattern), `Music` and `SoundBank` drain
//! [`DecodeDone`] in `update`. One worker per service (`Music`,
//! `SoundBank`), FIFO, so a track always lands before its stems.

use std::sync::Arc;

use web_workers::sync::mpsc::{Receiver, Sender, channel};

use crate::command::SharedFrames;
use crate::bank::CueDef;
use crate::music::StemDef;
use crate::{AudioState, decode_to_device};

/// One queued decode job.
pub(crate) enum DecodeJob {
    Track {
        name: String,
        gain: f32,
        bytes: Vec<u8>,
    },
    Stem {
        track: String,
        bytes: Vec<u8>,
        def: StemDef,
    },
    Cue {
        name: String,
        def: CueDef,
        files: Vec<Vec<u8>>,
    },
}

/// A finished (or failed) decode coming back from the worker.
pub(crate) enum DecodeDone {
    Track {
        name: String,
        gain: f32,
        main: Arc<SharedFrames>,
    },
    Stem {
        track: String,
        sound: Arc<SharedFrames>,
        def: StemDef,
    },
    Cue {
        name: String,
        def: CueDef,
        sounds: Vec<Arc<SharedFrames>>,
    },
    TrackFailed {
        name: String,
        error: String,
    },
    StemFailed {
        track: String,
        error: String,
    },
    CueFailed {
        name: String,
        error: String,
    },
}

/// Loader worker handle: jobs out, results back.
pub(crate) struct Loader {
    pub(crate) jobs: Sender<DecodeJob>,
    pub(crate) done: Receiver<DecodeDone>,
}

fn decode_job(job: DecodeJob, state: &AudioState) -> DecodeDone {
    match job {
        DecodeJob::Track { name, gain, bytes } => match decode_to_device(&bytes, Some(state)) {
            Ok(frames) => DecodeDone::Track {
                name,
                gain,
                main: frames,
            },
            Err(e) => DecodeDone::TrackFailed {
                name,
                error: e.to_string(),
            },
        },
        DecodeJob::Stem { track, bytes, def } => match decode_to_device(&bytes, Some(state)) {
            Ok(frames) => DecodeDone::Stem {
                track,
                sound: frames,
                def,
            },
            Err(e) => DecodeDone::StemFailed {
                track,
                error: e.to_string(),
            },
        },
        DecodeJob::Cue { name, def, files } => {
            let mut sounds = Vec::with_capacity(files.len());
            let mut failure = None;
            for bytes in &files {
                match decode_to_device(bytes, Some(state)) {
                    Ok(sound) => sounds.push(sound),
                    Err(e) => {
                        failure = Some(e.to_string());
                        break;
                    }
                }
            }
            match failure {
                None => DecodeDone::Cue { name, def, sounds },
                Some(error) => DecodeDone::CueFailed { name, error },
            }
        }
    }
}

/// Start the decode worker. Returns `None` where threads are unavailable
/// (noop engines, thread-less wasm); callers then decode inline.
pub(crate) fn spawn(state: Arc<AudioState>) -> Option<Loader> {
    #[cfg(all(target_family = "wasm", target_os = "unknown"))]
    if !web_workers::web::has_spawn_support() {
        return None;
    }
    let (job_tx, job_rx) = channel::<DecodeJob>();
    let (done_tx, done_rx) = channel::<DecodeDone>();
    web_workers::spawn(move || {
        while let Ok(job) = job_rx.recv_block() {
            if done_tx.send_block(decode_job(job, &state)).is_err() {
                break;
            }
        }
    });
    Some(Loader {
        jobs: job_tx,
        done: done_rx,
    })
}
