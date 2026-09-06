//! [`RigAudio`]: renamite rig-event strings -> bank cues.
//!
//! Rigs author sound (`EventKey` frames like `footstep`/`bite` in
//! `zombie.ren`); this maps event names to cues per rig (plus a global
//! fallback) and forwards each tick's [`events`](https://github.com/mlm-games/renamite)
//! slice into the bank. Zero game code per animation.

use std::collections::HashMap;

use crate::SoundBank;

/// Event-name to cue-name routing. Per-rig maps win over the fallback.
#[derive(Debug, Default)]
pub struct RigAudio {
    rigs: HashMap<String, HashMap<String, String>>,
    fallback: HashMap<String, String>,
}

impl RigAudio {
    pub fn new() -> Self {
        Self::default()
    }

    /// Route one rig event to a cue (`rig` names the animation set,
    /// e.g. `"zombie"`).
    pub fn map_event(&mut self, rig: &str, event: &str, cue: &str) {
        self.rigs
            .entry(rig.to_string())
            .or_default()
            .insert(event.to_string(), cue.to_string());
    }

    /// Route an event for every rig without a specific entry.
    pub fn map_event_global(&mut self, event: &str, cue: &str) {
        self.fallback.insert(event.to_string(), cue.to_string());
    }

    /// Resolve an event to a cue name (rig map, else fallback).
    pub fn resolve(&self, rig: &str, event: &str) -> Option<&str> {
        self.rigs
            .get(rig)
            .and_then(|m| m.get(event))
            .or_else(|| self.fallback.get(event))
            .map(String::as_str)
    }

    /// Forward one tick's rig events into cue plays.
    /// Returns started voice ids (unmapped events and throttled or
    /// unloaded cues contribute nothing).
    pub fn sync(&mut self, bank: &mut SoundBank, rig: &str, events: &[String]) -> Vec<u64> {
        let cues: Vec<String> = events
            .iter()
            .filter_map(|e| self.resolve(rig, e).map(str::to_string))
            .collect();
        cues.into_iter().filter_map(|c| bank.play(&c)).collect()
    }

    /// Positional variant: every event plays at `(x, y)`.
    pub fn sync_at(
        &mut self,
        bank: &mut SoundBank,
        rig: &str,
        events: &[String],
        x: f32,
        y: f32,
    ) -> Vec<u64> {
        let cues: Vec<String> = events
            .iter()
            .filter_map(|e| self.resolve(rig, e).map(str::to_string))
            .collect();
        cues.into_iter()
            .filter_map(|c| bank.play_at(&c, x, y))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CueDef, audio_link, synth_sine_wav};

    fn banked() -> (SoundBank, crate::GameAudioLink, crate::ThreadAudioLink) {
        let (game, thread) = audio_link();
        let mut bank = SoundBank::new();
        bank.attach(game.tx.clone(), game.state.clone());
        let step = synth_sine_wav(300.0, 0.05, 22050);
        let chomp = synth_sine_wav(150.0, 0.1, 22050);
        bank.load("step", CueDef::default(), &[&step]).unwrap();
        bank.load("chomp", CueDef::default(), &[&chomp]).unwrap();
        (bank, game, thread)
    }

    #[test]
    fn routes_per_rig_with_global_fallback() {
        let (mut bank, _game, _thread) = banked();
        let mut rig = RigAudio::new();
        rig.map_event("zombie", "footstep", "step");
        rig.map_event("zombie", "bite", "chomp");
        rig.map_event_global("ui_click", "step");

        let ev = |s: &str| s.to_string();
        let voices = rig.sync(
            &mut bank,
            "zombie",
            &[ev("footstep"), ev("bite"), ev("sneeze")],
        );
        assert_eq!(voices.len(), 2);
        // Fallback covers rigs without a map; unknown rigs stay silent.
        assert_eq!(rig.sync(&mut bank, "pea", &[ev("ui_click")]).len(), 1);
        assert!(rig.sync(&mut bank, "pea", &[ev("footstep")]).is_empty());
        assert_eq!(rig.resolve("zombie", "bite"), Some("chomp"));
        assert_eq!(rig.resolve("pea", "bite"), None);
    }

    #[test]
    fn positional_sync_plays_at_point() {
        let (mut bank, _game, _thread) = banked();
        let mut rig = RigAudio::new();
        rig.map_event("zombie", "footstep", "step");
        bank.set_listener(100.0, 0.0);
        let voices = rig.sync_at(&mut bank, "zombie", &["footstep".to_string()], 100.0, 0.0);
        assert_eq!(voices.len(), 1);
    }
}
