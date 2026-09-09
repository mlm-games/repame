//! Per-tick action state: levels, edges, consumption, mocking.

use std::collections::{HashMap, HashSet};

use bevy_ecs::prelude::*;
use repose_core::input::{GamepadAxis, GamepadButton, PointerButton};
use repose_core::shortcuts::KeyChord;

use super::ActionLike;
use super::binding::Binding;
use super::map::ActionMap;

/// Sim-side gameplay input: fed from repose events each frame, read by
/// fixed-step systems each tick. Edges (`just_pressed`/`just_released`)
/// clear at the END of each tick ([`end_tick`](Self::end_tick),
/// registered last), so compose-fed events survive until the tick's
/// systems run. Frame-scope readers call
/// [`clear_edges`](Self::clear_edges) after consuming.
#[derive(Resource, Debug)]
pub struct ActionState<A: ActionLike> {
    map: ActionMap<A>,
    pressed: HashSet<A>,
    just_pressed: HashSet<A>,
    just_released: HashSet<A>,
    consumed: HashSet<A>,
    axes: HashMap<GamepadAxis, f32>,
    active_contexts: HashSet<String>,
    /// Currently-held button bindings (key/mouse/pad). Releases only drop
    /// the action when *no* binding for it remains down (OR semantics).
    down_buttons: Vec<Binding>,
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
            down_buttons: Vec::new(),
        }
    }

    pub fn map(&self) -> &ActionMap<A> {
        &self.map
    }

    /// Edge rollover for the tick end: edges clear, levels persist.
    /// Frame-scope readers use [`clear_edges`](Self::clear_edges) instead.
    pub fn end_tick(&mut self) {
        self.just_pressed.clear();
        self.just_released.clear();
    }

    /// Clear edges outside the schedule (after frame-scope consumption).
    pub fn clear_edges(&mut self) {
        self.end_tick();
    }

    /// Switch the active context set (phase gating, e.g. menu vs play).
    /// Empty set with contexts defined means only context-free actions fire.
    /// Pending edges drop only when the set actually changes: re-setting
    /// the same contexts every frame (the normal pump pattern) keeps them.
    /// Levels (`pressed`) always track hardware truthfully.
    pub fn set_contexts(&mut self, contexts: &[&str]) {
        let next: HashSet<String> = contexts.iter().map(|s| s.to_string()).collect();
        if next != self.active_contexts {
            self.active_contexts = next;
            self.just_pressed.clear();
            self.just_released.clear();
        }
    }

    fn live(&self, action: &A) -> bool {
        self.map.live_in(action, &self.active_contexts) && !self.consumed.contains(action)
    }

    fn press(&mut self, action: &A) {
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
        match binding {
            Binding::Key(_) | Binding::Mouse(_) | Binding::Pad(_) => {
                if down {
                    if !self.down_buttons.iter().any(|b| b == binding) {
                        self.down_buttons.push(binding.clone());
                    }
                } else {
                    self.down_buttons.retain(|b| b != binding);
                }
            }
            Binding::Axis { .. } => {}
        }
        let actions: Vec<A> = self
            .map
            .actions()
            .filter(|a| self.map.bindings_for(a).contains(binding))
            .cloned()
            .collect();
        for action in actions {
            if self.action_down(&action) {
                self.press(&action);
            } else {
                self.release(&action);
            }
        }
    }

    fn binding_down(&self, binding: &Binding) -> bool {
        match binding {
            Binding::Key(_) | Binding::Mouse(_) | Binding::Pad(_) => {
                self.down_buttons.iter().any(|b| b == binding)
            }
            Binding::Axis { axis, threshold } => {
                Binding::axis_active(*threshold, self.axis_value(*axis))
            }
        }
    }

    fn action_down(&self, action: &A) -> bool {
        self.map
            .bindings_for(action)
            .iter()
            .any(|b| self.binding_down(b))
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
    /// Re-evaluates every action bound to this axis with OR semantics, so
    /// a second binding holding the action keeps it pressed.
    pub fn axis(&mut self, axis: GamepadAxis, value: f32) {
        self.axes.insert(axis, value);
        let actions: Vec<A> = self
            .map
            .actions()
            .filter(|a| {
                self.map.bindings_for(a).iter().any(|b| match b {
                    Binding::Axis { axis: ba, .. } => *ba == axis,
                    _ => false,
                })
            })
            .cloned()
            .collect();
        for action in actions {
            if self.action_down(&action) {
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

    /// Action strength in `0..1`: `1.0` for held buttons, `|value|`
    /// clamped for active axis bindings (strongest binding wins), `0.0`
    /// when inactive or gated by context/consumption.
    pub fn strength(&self, action: &A) -> f32 {
        if !self.live(action) {
            return 0.0;
        }
        let mut best = 0.0f32;
        for b in self.map.bindings_for(action) {
            let s = match b {
                Binding::Key(_) | Binding::Mouse(_) | Binding::Pad(_) => {
                    if self.binding_down(b) {
                        1.0
                    } else {
                        0.0
                    }
                }
                Binding::Axis { axis, threshold } => {
                    let v = self.axis_value(*axis);
                    if Binding::axis_active(*threshold, v) {
                        v.abs().clamp(0.0, 1.0).max(1e-6)
                    } else {
                        0.0
                    }
                }
            };
            best = best.max(s);
        }
        best.clamp(0.0, 1.0)
    }

    /// Signed 2D input vector from four actions: `x = pos_x - neg_x`,
    /// `y = pos_y - neg_y` from strengths, deadzoned by `deadzone` and
    /// length-clamped to `1.0` (analog-friendly movement from d-pad or
    /// sticks).
    pub fn vector(&self, neg_x: &A, pos_x: &A, neg_y: &A, pos_y: &A, deadzone: f32) -> (f32, f32) {
        let mut x = self.strength(pos_x) - self.strength(neg_x);
        let mut y = self.strength(pos_y) - self.strength(neg_y);
        let len = (x * x + y * y).sqrt();
        if len < deadzone.max(0.0) {
            return (0.0, 0.0);
        }
        if len > 1.0 {
            x /= len;
            y /= len;
        }
        (x, y)
    }

    /// Held (and live in the active context, not consumed).
    pub fn pressed(&self, action: &A) -> bool {
        self.pressed.contains(action) && self.live(action)
    }

    /// Started this tick (and live in the active context, not consumed).
    pub fn just_pressed(&self, action: &A) -> bool {
        self.just_pressed.contains(action) && self.live(action)
    }

    /// Released this tick (and live in the active context, not consumed —
    /// same gating as `pressed`/`just_pressed` so context-gated and
    /// UI-consumed releases never leak to late readers).
    pub fn just_released(&self, action: &A) -> bool {
        self.just_released.contains(action) && self.live(action)
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
    fn edges_clear_at_tick_end() {
        let mut st = ActionState::new(jump_map());
        st.key(&space(), true);
        assert!(st.just_pressed(&"jump"));
        assert!(st.pressed(&"jump"));
        // Edge survives (no schedule ran yet); systems read, then clear.
        assert!(st.just_pressed(&"jump"));
        st.end_tick();
        assert!(!st.just_pressed(&"jump"));
        assert!(st.pressed(&"jump"));
        st.key(&space(), false);
        assert!(st.just_released(&"jump"));
        assert!(!st.pressed(&"jump"));
        st.end_tick();
        assert!(!st.just_released(&"jump"));
    }

    #[test]
    fn pad_and_key_drive_same_action() {
        let mut st = ActionState::new(jump_map());
        st.pad(GamepadButton::South, true);
        assert!(st.just_pressed(&"jump"));
        st.end_tick();
        st.pad(GamepadButton::South, false);
        st.end_tick();
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
    fn context_switch_drops_pending_edges() {
        let mut map = jump_map();
        map.in_context("gameplay", "jump");
        let mut st = ActionState::new(map);
        st.set_contexts(&["gameplay"]);
        st.key(&space(), true);
        assert!(st.just_pressed(&"jump"));
        st.set_contexts(&["menu"]);
        assert!(!st.just_pressed(&"jump"));
        assert!(!st.pressed(&"jump"));
        st.key(&space(), false);
        st.set_contexts(&["gameplay"]);
        assert!(!st.just_released(&"jump"));
    }

    #[test]
    fn mock_drives_without_hardware() {
        let mut st = ActionState::new(jump_map());
        st.mock(&"jump", true);
        assert!(st.just_pressed(&"jump"));
        st.mock(&"jump", false);
        assert!(st.just_released(&"jump"));
    }

    #[test]
    fn multi_binding_release_is_or() {
        let mut st = ActionState::new(jump_map());
        st.key(&space(), true);
        st.pad(GamepadButton::South, true);
        assert!(st.pressed(&"jump"));
        st.end_tick();
        st.key(&space(), false);
        assert!(st.pressed(&"jump"), "South still held");
        assert!(!st.just_released(&"jump"), "no release while OR-held");
        st.pad(GamepadButton::South, false);
        assert!(!st.pressed(&"jump"));
        assert!(st.just_released(&"jump"));
    }

    #[test]
    fn consume_hides_release_too() {
        let mut st = ActionState::new(jump_map());
        st.key(&space(), true);
        st.end_tick();
        st.key(&space(), false);
        assert!(st.just_released(&"jump"));
        st.consume(&"jump");
        assert!(!st.just_released(&"jump"));
    }

    #[test]
    fn strength_and_vector_behaviour() {
        use repose_core::input::Key;
        let mut map = ActionMap::new();
        map.bind(
            "right",
            Binding::Axis {
                axis: GamepadAxis::LeftStickX,
                threshold: 0.2,
            },
        );
        map.bind(
            "left",
            Binding::Axis {
                axis: GamepadAxis::LeftStickX,
                threshold: -0.2,
            },
        );
        map.bind(
            "jump",
            Binding::Key(KeyChord::new(Key::Space, Modifiers::default())),
        );
        let mut st = ActionState::new(map);
        // Rest: all zero.
        assert_eq!(st.strength(&"right"), 0.0);
        assert_eq!(st.vector(&"left", &"right", &"jump", &"jump", 0.2), (0.0, 0.0));
        // Half deflection right: proportional strength.
        st.axis(GamepadAxis::LeftStickX, 0.5);
        assert!((st.strength(&"right") - 0.5).abs() < 1e-6);
        assert_eq!(st.strength(&"left"), 0.0);
        let (x, y) = st.vector(&"left", &"right", &"jump", &"jump", 0.2);
        assert!((x - 0.5).abs() < 1e-6 && y.abs() < 1e-6);
        // Vector deadzone snaps small-but-active sticks to zero.
        assert_eq!(st.vector(&"left", &"right", &"jump", &"jump", 0.6), (0.0, 0.0));
        // Below deadzone the vector snaps to zero even if bound.
        st.axis(GamepadAxis::LeftStickX, 0.1);
        assert_eq!(st.strength(&"right"), 0.0);
        // Button: full strength.
        st.key(&space(), true);
        assert_eq!(st.strength(&"jump"), 1.0);
    }
}
