use std::collections::HashMap;

use glam::Vec2;
use repame_input::GamepadState;
use repose_core::input::{GamepadAxis, GamepadButton, GamepadEvent, GamepadId};

pub const TRIGGER_HELD: f32 = 0.5;

#[derive(Default)]
pub struct PadBridge {
    lx: f32,
    ly: f32,
    rx: f32,
    ry: f32,
    lt_held: bool,
    rt_held: bool,
    south: bool,
    east: bool,
    dpad_l: bool,
    dpad_u: bool,
    dpad_r: bool,
    north: bool,
    lt_edge: bool,
    rt_edge: bool,
    south_edge: bool,
    east_edge: bool,
    dpad_l_edge: bool,
    dpad_u_edge: bool,
    dpad_r_edge: bool,
    north_edge: bool,
}

impl PadBridge {
    pub fn button(&mut self, button: GamepadButton, pressed: bool) {
        let (held, edge) = match button {
            GamepadButton::South => (&mut self.south, &mut self.south_edge),
            GamepadButton::East => (&mut self.east, &mut self.east_edge),
            GamepadButton::North => (&mut self.north, &mut self.north_edge),
            GamepadButton::DPadLeft => (&mut self.dpad_l, &mut self.dpad_l_edge),
            GamepadButton::DPadUp => (&mut self.dpad_u, &mut self.dpad_u_edge),
            GamepadButton::DPadRight => (&mut self.dpad_r, &mut self.dpad_r_edge),
            _ => return,
        };
        if pressed && !*held {
            *edge = true;
        }
        *held = pressed;
    }

    pub fn axis(&mut self, axis: GamepadAxis, value: f32) {
        match axis {
            GamepadAxis::LeftStickX => self.lx = value,
            GamepadAxis::LeftStickY => self.ly = value,
            GamepadAxis::RightStickX => self.rx = value,
            GamepadAxis::RightStickY => self.ry = value,
            GamepadAxis::LeftTrigger => {
                let held = value > TRIGGER_HELD;
                if held && !self.lt_held {
                    self.lt_edge = true;
                }
                self.lt_held = held;
            }
            GamepadAxis::RightTrigger => {
                let held = value > TRIGGER_HELD;
                if held && !self.rt_held {
                    self.rt_edge = true;
                }
                self.rt_held = held;
            }
        }
    }

    pub fn snapshot(&self) -> GamepadState {
        GamepadState {
            left_stick: Vec2::new(self.lx, self.ly),
            right_stick: Vec2::new(self.rx, self.ry),
            right_trigger_held: self.rt_held,
            right_trigger_pressed: self.rt_edge,
            left_trigger_held: self.lt_held,
            left_trigger_pressed: self.lt_edge,
            south_pressed: self.south_edge,
            east_pressed: self.east_edge,
            dpad_left_pressed: self.dpad_l_edge,
            dpad_up_pressed: self.dpad_u_edge,
            dpad_right_pressed: self.dpad_r_edge,
            north_pressed: self.north_edge,
        }
    }

    pub fn clear_edges(&mut self) {
        self.lt_edge = false;
        self.rt_edge = false;
        self.south_edge = false;
        self.east_edge = false;
        self.dpad_l_edge = false;
        self.dpad_u_edge = false;
        self.dpad_r_edge = false;
        self.north_edge = false;
    }
}

#[derive(Default)]
pub struct PadBank {
    pads: HashMap<GamepadId, PadBridge>,
}

impl PadBank {
    pub fn feed(&mut self, events: Vec<GamepadEvent>) {
        for ev in events {
            match ev {
                GamepadEvent::Connected { .. } => {}
                GamepadEvent::Disconnected { id } => {
                    // Drop the bridge: a stale handle never aliases a new
                    // device, and held buttons do not strand (the entry is
                    // gone, so no snapshot reports them held). Reconnect
                    // starts from a released zero state.
                    self.pads.remove(&id);
                }
                GamepadEvent::Button { id, button, pressed } => {
                    self.pads.entry(id).or_default().button(button, pressed);
                }
                GamepadEvent::Axis { id, axis, value } => {
                    self.pads.entry(id).or_default().axis(axis, value);
                }
            }
        }
    }

    pub fn drain(&mut self) -> Vec<GamepadState> {
        let mut ids: Vec<GamepadId> = self.pads.keys().copied().collect();
        ids.sort_by_key(|id| id.0);
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(bridge) = self.pads.get_mut(&id) {
                out.push(bridge.snapshot());
                bridge.clear_edges();
            }
        }
        out
    }

    pub fn live(&self) -> bool {
        !self.pads.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rising_edge_fires_once() {
        let mut b = PadBridge::default();
        b.button(GamepadButton::South, true);
        assert!(b.snapshot().south_pressed);
        b.clear_edges();
        b.button(GamepadButton::South, true);
        assert!(!b.snapshot().south_pressed);
    }

    #[test]
    fn trigger_threshold_edges() {
        let mut b = PadBridge::default();
        b.axis(GamepadAxis::RightTrigger, 0.8);
        let s = b.snapshot();
        assert!(s.right_trigger_held && s.right_trigger_pressed);
    }
}
