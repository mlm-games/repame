use repame_audio::{Audio, Cue};

pub fn drain_cues(audio: &mut Audio, cues: impl IntoIterator<Item = Cue>, pitch: impl Fn() -> f32) {
    for cue in cues {
        let _ = cue.pitch(pitch());
        audio.play(cue.name);
    }
}
