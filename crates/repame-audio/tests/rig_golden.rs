//! Golden chain: vendored `zombie.ren` -> renamite events -> rig map -> bank.
//!
//! The asset is a copy of rozvp's zombie rig (vendored so this crate is
//! self-contained): walk must emit `footstep`, and the `eating` input must
//! reach the attack state that emits `bite`. Both must resolve to started voices.

use renamite_player::Player;
use repame_audio::{CueDef, RigAudio, SoundBank, audio_link, synth_sine_wav};

static ZOMBIE_REN: &str = include_str!("assets/zombie.ren");

fn wired() -> (SoundBank, RigAudio, Player, repame_audio::ThreadAudioLink) {
    let player = Player::from_ren_str(ZOMBIE_REN).expect("zombie.ren parses");
    let (game, thread) = audio_link();
    let mut bank = SoundBank::new();
    bank.attach(game.tx.clone(), game.state.clone());
    let step = synth_sine_wav(300.0, 0.05, 22050);
    let chomp = synth_sine_wav(150.0, 0.1, 22050);
    bank.load("step", CueDef::default(), &[&step]).unwrap();
    bank.load("chomp", CueDef::default(), &[&chomp]).unwrap();
    let mut rig = RigAudio::new();
    rig.map_event("zombie", "footstep", "step");
    rig.map_event("zombie", "bite", "chomp");
    (bank, rig, player, thread)
}

#[test]
fn walk_footsteps_reach_voices() {
    let (mut bank, mut rig, mut player, _thread) = wired();
    let mut voices = 0;
    for _ in 0..360 {
        // End the tick borrow before syncing (both need `&mut` elsewhere).
        let events: Vec<String> = player.tick(1.0 / 60.0).to_vec();
        voices += rig.sync(&mut bank, "zombie", &events).len();
    }
    assert!(voices >= 2, "footstep voices = {voices}");
}

#[test]
fn eating_bite_reaches_voices() {
    let (mut bank, mut rig, mut player, _thread) = wired();
    assert!(player.set_bool("eating", true), "eating input exists");
    let mut bite_seen = false;
    let mut bite_voiced = false;
    for _ in 0..360 {
        let events: Vec<String> = player.tick(1.0 / 60.0).to_vec();
        if events.iter().any(|e| e == "bite") {
            bite_seen = true;
            // Sync the bite alone so the voice attributes to it.
            bite_voiced |= !rig
                .sync(&mut bank, "zombie", &["bite".to_string()])
                .is_empty();
        } else {
            rig.sync(&mut bank, "zombie", &events);
        }
    }
    assert!(bite_seen, "attack state never emitted `bite`");
    assert!(bite_voiced, "`bite` never became a voice");
}
