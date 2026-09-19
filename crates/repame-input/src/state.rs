//! Per-tick action state: levels, edges, consumption, mocking.

use std::collections::{HashMap, HashSet};

use bevy_ecs::prelude::*;
use repose_core::input::{GamepadAxis, GamepadButton, PhysicalKey, PointerButton};
use repose_core::shortcuts::KeyChord;

use super::ActionLike;
use super::binding::Binding;
use super::map::ActionMap;

/// Sim-side gameplay input fed from repose events, read per tick.
/// Edges clear at tick end via `end_tick` (registered last).
/// Frame-scope readers call `clear_edges` after consuming.
#[derive(Resource, Debug)]
pub struct ActionState<A: ActionLike> {
    map: ActionMap<A>,
    pressed: HashSet<A>,
    just_pressed: HashSet<A>,
    just_released: HashSet<A>,
    consumed: HashSet<A>,
    axes: HashMap<GamepadAxis, f32>,
    active_contexts: HashSet<String>,
    /// Held button bindings. Releases drop the action when no binding stays down.
    down_buttons: HashSet<Binding>,
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
            down_buttons: HashSet::new(),
        }
    }

    pub fn map(&self) -> &ActionMap<A> {
        &self.map
    }

    pub fn replace_map(&mut self, map: ActionMap<A>) {
        self.map = map;
        let live: Vec<Binding> = self
            .map
            .actions()
            .flat_map(|a| self.map.bindings_for(a).iter().cloned())
            .collect();
        self.down_buttons.retain(|b| live.contains(b));
        self.pressed.retain(|a| !self.map.bindings_for(a).is_empty());
        self.just_pressed
            .retain(|a| !self.map.bindings_for(a).is_empty());
        self.just_released.clear();
    }

    /// Edge rollover for tick end: edges clear, levels persist.
    pub fn end_tick(&mut self) {
        self.just_pressed.clear();
        self.just_released.clear();
    }

    /// Clear edges outside the schedule.
    pub fn clear_edges(&mut self) {
        self.end_tick();
    }

    /// Switch the active context set. Pending edges drop on change;
    /// re-setting the same set keeps them. Levels track hardware.
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
            Binding::Key(_) | Binding::Physical(_) | Binding::Mouse(_) | Binding::Pad(_) => {
                if down {
                    self.down_buttons.insert(binding.clone());
                } else {
                    self.down_buttons.remove(binding);
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
            Binding::Key(_) | Binding::Physical(_) | Binding::Mouse(_) | Binding::Pad(_) => {
                self.down_buttons.contains(binding)
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

    /// Feed a keyboard chord press/release.
    pub fn key(&mut self, chord: &KeyChord, down: bool) {
        self.fire_binding(&Binding::Key(chord.clone()), down);
    }

    /// Feed a physical key position press/release.
    pub fn physical(&mut self, key: PhysicalKey, down: bool) {
        self.fire_binding(&Binding::Physical(key), down);
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

    /// Latest raw value fed for an axis (`0.0` if not fed).
    ///
    /// Raw deflection in `-1..1` (sticks) or `0..1` (triggers), no threshold applied.
    pub fn axis_value(&self, axis: GamepadAxis) -> f32 {
        self.axes.get(&axis).copied().unwrap_or(0.0)
    }

    /// Intensity from `0.0` (inactive) to `1.0` (held). Buttons read `0`/`1`;
    /// axes remap `threshold..1` to `0..1`. Strongest binding wins.
    /// Gated or consumed actions read `0.0`.
    pub fn strength(&self, action: &A) -> f32 {
        if !self.live(action) {
            return 0.0;
        }
        let mut best = 0.0f32;
        for b in self.map.bindings_for(action) {
            let s =
                match b {
                    Binding::Key(_) | Binding::Physical(_) | Binding::Mouse(_) | Binding::Pad(_) => {
                        if self.binding_down(b) { 1.0 } else { 0.0 }
                    }
                    Binding::Axis { axis, threshold } => {
                        let v = self.axis_value(*axis);
                        if !Binding::axis_active(*threshold, v) {
                            0.0
                        } else {
                            // Radial deadzone remap t..1 to 0..1.
                            let t = threshold.abs().clamp(0.0, 0.95);
                            let a = v.abs();
                            if a <= t {
                                0.0
                            } else {
                                ((a - t) / (1.0 - t)).clamp(0.0, 1.0).max(1e-6)
                            }
                        }
                    }
                };
            best = best.max(s);
        }
        best.clamp(0.0, 1.0)
    }

    /// Movement vector from four actions. Lengths below `deadzone` snap to zero;
    /// lengths above `1` normalize back. `y` is screen convention (down-positive).
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

    /// Held, live, and not consumed. True each tick a binding is down.
    pub fn pressed(&self, action: &A) -> bool {
        self.pressed.contains(action) && self.live(action)
    }

    /// Started this tick. Edge set on press, cleared at `end_tick`.
    pub fn just_pressed(&self, action: &A) -> bool {
        self.just_pressed.contains(action) && self.live(action)
    }

    /// Released this tick. Clears at `end_tick` like `just_pressed`.
    pub fn just_released(&self, action: &A) -> bool {
        self.just_released.contains(action) && self.live(action)
    }

    /// Mark consumed so later readers skip it. Returns true while held.
    pub fn consume(&mut self, action: &A) -> bool {
        let held = self.pressed.contains(action);
        self.consumed.insert(action.clone());
        held
    }

    pub fn consume_all(&mut self) {
        self.consumed.extend(self.pressed.iter().cloned());
    }

    /// Drive an action without hardware.
    pub fn mock(&mut self, action: &A, down: bool) {
        if down {
            self.press(action);
        } else {
            self.release(action);
        }
    }

    /// Clear consumption.
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
        assert!(st.pressed(&"jump"), "South still held");
        // Edges persist until tick end; systems read, then clear.
        assert!(st.just_pressed(&"jump"));
        st.end_tick();
        assert!(!st.just_pressed(&"jump"));
        assert!(st.pressed(&"jump"), "South still held");
        st.key(&space(), false);
        assert!(st.just_released(&"jump"));
        assert!(!st.pressed(&"jump"));
        st.end_tick();
        assert!(!st.just_released(&"jump"), "no release while OR-held");
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
        assert!(st.pressed(&"jump"), "South still held");
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
        assert!(st.pressed(&"jump"), "South still held");
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
        assert!(!st.just_released(&"jump"), "no release while OR-held");
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
        assert!(st.pressed(&"jump"), "South still held");
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
        assert!(!st.just_released(&"jump"), "no release while OR-held");
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
        // Rest reads zero.
        assert_eq!(st.strength(&"right"), 0.0);
        assert_eq!(
            st.vector(&"left", &"right", &"jump", &"jump", 0.2),
            (0.0, 0.0)
        );
        // Half deflection right, deadzone remapped.
        st.axis(GamepadAxis::LeftStickX, 0.5);
        assert!((st.strength(&"right") - 0.375).abs() < 1e-6);
        assert_eq!(st.strength(&"left"), 0.0);
        let (x, y) = st.vector(&"left", &"right", &"jump", &"jump", 0.2);
        assert!((x - 0.375).abs() < 1e-6 && y.abs() < 1e-6);
        // Deadzone snaps small sticks to zero.
        assert_eq!(
            st.vector(&"left", &"right", &"jump", &"jump", 0.6),
            (0.0, 0.0)
        );
        // Below threshold the strength reads zero.
        st.axis(GamepadAxis::LeftStickX, 0.1);
        assert_eq!(st.strength(&"right"), 0.0);
        // Buttons read full strength.
        st.key(&space(), true);
        assert_eq!(st.strength(&"jump"), 1.0);
    }
}
