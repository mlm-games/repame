//! Per-tick action state: levels, edges, consumption, mocking.

use std::collections::{HashMap, HashSet};

use bevy_ecs::prelude::*;
use repose_core::input::{GamepadAxis, GamepadButton, PointerButton};
use repose_core::shortcuts::KeyChord;

use super::binding::Binding;
use super::map::ActionMap;
use super::ActionLike;

/// Sim-side gameplay input: fed from repose events each frame, read by
/// fixed-step systems each tick. Edges (`just_pressed`/`just_released`)
/// live exactly one tick: call [`begin_tick`](Self::begin_tick) first in
/// the schedule (see [`begin_tick_system`](super::begin_tick_system)),
/// then feed events, then run gameplay systems.
#[derive(Resource, Debug)]
pub struct ActionState<A: ActionLike> {
    map: ActionMap<A>,
    pressed: HashSet<A>,
    just_pressed: HashSet<A>,
    just_released: HashSet<A>,
    consumed: HashSet<A>,
    axes: HashMap<GamepadAxis, f32>,
    active_contexts: HashSet<String>,
}

impl<A: ActionLike> ActionState<A> {
    pub fn new(map: ActionMap<A>) -> Self {
        Self {
            map,
            pressed: HashSet::new(),
            just_pressed: HashSet::new(),
            just_released: HashSet::new(),
            consumed: HashSet::new(),
            axes: HashMap::new(),
            active_contexts: HashSet::new(),
        }
    }

    pub fn map(&self) -> &ActionMap<A> {
        &self.map
    }

    /// Edge rollover for the new tick: edges clear, levels persist.
    pub fn begin_tick(&mut self) {
        self.just_pressed.clear();
        self.just_released.clear();
    }

    /// Switch the active context set (phase gating, e.g. menu vs play).
    /// Empty set with contexts defined means only context-free actions fire.
    pub fn set_contexts(&mut self, contexts: &[&str]) {
        self.active_contexts = contexts.iter().map(|s| s.to_string()).collect();
    }

    fn live(&self, action: &A) -> bool {
        self.map.live_in(action, &self.active_contexts) && !self.consumed.contains(action)
    }

    fn press(&mut self, action: &A) {
        if !self.live(action) {
            return;
        }
        if self.pressed.insert(action.clone()) {
            self.just_pressed.insert(action.clone());
        }
    }

    fn release(&mut self, action: &A) {
        if self.pressed.remove(action) {
            self.just_released.insert(action.clone());
        }
    }

    fn fire_binding(&mut self, binding: &Binding, down: bool) {
        let actions: Vec<A> = self
            .map
            .actions()
            .filter(|a| self.map.bindings_for(a).contains(binding))
            .cloned()
            .collect();
        for action in actions {
            if down {
                self.press(&action);
            } else {
                self.release(&action);
            }
        }
    }

    /// Feed a keyboard chord press/release from the runner.
    pub fn key(&mut self, chord: &KeyChord, down: bool) {
        self.fire_binding(&Binding::Key(chord.clone()), down);
    }

    /// Feed a mouse button press/release.
    pub fn mouse(&mut self, button: PointerButton, down: bool) {
        self.fire_binding(&Binding::Mouse(button), down);
    }

    /// Feed a gamepad button press/release.
    pub fn pad(&mut self, button: GamepadButton, down: bool) {
        self.fire_binding(&Binding::Pad(button), down);
    }

    /// Feed an axis value; threshold bindings flip on crossing.
    pub fn axis(&mut self, axis: GamepadAxis, value: f32) {
        self.axes.insert(axis, value);
        let flips: Vec<(A, bool)> = self
            .map
            .actions()
            .flat_map(|a| {
                self.map.bindings_for(a).iter().filter_map(move |b| match b {
                    Binding::Axis { axis: ba, threshold } if *ba == axis => {
                        Some((a.clone(), Binding::axis_active(*threshold, value)))
                    }
                    _ => None,
                })
            })
            .collect();
        for (action, active) in flips {
            if active {
                self.press(&action);
            } else {
                self.release(&action);
            }
        }
    }

    /// Latest value for an axis (0.0 if never fed).
    pub fn axis_value(&self, axis: GamepadAxis) -> f32 {
        self.axes.get(&axis).copied().unwrap_or(0.0)
    }

    /// Held (and not consumed).
    pub fn pressed(&self, action: &A) -> bool {
        self.pressed.contains(action) && !self.consumed.contains(action)
    }

    /// Started this tick (and not consumed).
    pub fn just_pressed(&self, action: &A) -> bool {
        self.just_pressed.contains(action) && !self.consumed.contains(action)
    }

    /// Released this tick.
    pub fn just_released(&self, action: &A) -> bool {
        self.just_released.contains(action)
    }

    /// Swallow an action so later readers skip it (UI consumed the click).
    /// Returns true while the action is held.
    pub fn consume(&mut self, action: &A) -> bool {
        let held = self.pressed.contains(action);
        self.consumed.insert(action.clone());
        held
    }

    pub fn consume_all(&mut self) {
        self.consumed.extend(self.pressed.iter().cloned());
    }

    /// Drive an action without hardware (cutscenes, AI, tests).
    pub fn mock(&mut self, action: &A, down: bool) {
        if down {
            self.press(action);
        } else {
            self.release(action);
        }
    }

    /// Clear consumption (fresh frame handoff from UI).
    pub fn clear_consumed(&mut self) {
        self.consumed.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use repose_core::input::{Key, Modifiers};

    fn jump_map() -> ActionMap<&'static str> {
        let mut map = ActionMap::new();
        map.bind(
            "jump",
            Binding::Key(KeyChord::new(Key::Space, Modifiers::default())),
        );
        map.bind("jump", Binding::Pad(GamepadButton::South));
        map
    }

    fn space() -> KeyChord {
        KeyChord::new(Key::Space, Modifiers::default())
    }

    #[test]
    fn edges_live_one_tick() {
        let mut st = ActionState::new(jump_map());
        st.key(&space(), true);
        assert!(st.just_pressed(&"jump"));
        assert!(st.pressed(&"jump"));
        st.begin_tick();
        assert!(!st.just_pressed(&"jump"));
        assert!(st.pressed(&"jump"));
        st.key(&space(), false);
        assert!(st.just_released(&"jump"));
        assert!(!st.pressed(&"jump"));
        st.begin_tick();
        assert!(!st.just_released(&"jump"));
    }

    #[test]
    fn pad_and_key_drive_same_action() {
        let mut st = ActionState::new(jump_map());
        st.pad(GamepadButton::South, true);
        assert!(st.just_pressed(&"jump"));
        st.pad(GamepadButton::South, false);
        st.begin_tick();
        st.key(&space(), true);
        assert!(st.just_pressed(&"jump"));
    }

    #[test]
    fn axis_threshold_flips() {
        let mut map = ActionMap::new();
        map.bind(
            "left",
            Binding::Axis {
                axis: GamepadAxis::LeftStickX,
                threshold: -0.5,
            },
        );
        let mut st = ActionState::new(map);
        st.axis(GamepadAxis::LeftStickX, -0.2);
        assert!(!st.pressed(&"left"));
        st.axis(GamepadAxis::LeftStickX, -0.8);
        assert!(st.just_pressed(&"left"));
        assert_eq!(st.axis_value(GamepadAxis::LeftStickX), -0.8);
        st.axis(GamepadAxis::LeftStickX, 0.0);
        assert!(st.just_released(&"left"));
    }

    #[test]
    fn consume_hides_from_late_readers() {
        let mut st = ActionState::new(jump_map());
        st.key(&space(), true);
        assert!(st.consume(&"jump"));
        assert!(!st.pressed(&"jump"));
        assert!(!st.just_pressed(&"jump"));
        st.clear_consumed();
        assert!(st.pressed(&"jump"));
    }

    #[test]
    fn context_gates_sim_action() {
        let mut map = jump_map();
        map.in_context("gameplay", "jump");
        let mut st = ActionState::new(map);
        st.set_contexts(&["menu"]);
        st.key(&space(), true);
        assert!(!st.pressed(&"jump"));
        st.set_contexts(&["gameplay"]);
        st.key(&space(), true);
        assert!(st.pressed(&"jump"));
    }

    #[test]
    fn mock_drives_without_hardware() {
        let mut st = ActionState::new(jump_map());
        st.mock(&"jump", true);
        assert!(st.just_pressed(&"jump"));
        st.mock(&"jump", false);
        assert!(st.just_released(&"jump"));
    }
}
