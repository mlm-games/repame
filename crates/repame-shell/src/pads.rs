use std::collections::HashMap;

use glam::Vec2;
use repame_input::GamepadState;
use repose_core::input::{GamepadAxis, GamepadButton, GamepadEvent, GamepadId};

pub const TRIGGER_HELD: f32 = 0.5;

#[derive(Default)]
pub struct PadBridge {
    vendor_id: u16,
    product_id: u16,
    lx: f32,
    ly: f32,
    rx: f32,
    ry: f32,
    lt_held: bool,
    rt_held: bool,
    south: bool,
    east: bool,
    west: bool,
    dpad_l: bool,
    dpad_u: bool,
    dpad_r: bool,
    dpad_d: bool,
    north: bool,
    lb: bool,
    rb: bool,
    start: bool,
    select: bool,
    l3: bool,
    r3: bool,
    lt_edge: bool,
    rt_edge: bool,
    south_edge: bool,
    east_edge: bool,
    west_edge: bool,
    dpad_l_edge: bool,
    dpad_u_edge: bool,
    dpad_r_edge: bool,
    dpad_d_edge: bool,
    north_edge: bool,
    lb_edge: bool,
    rb_edge: bool,
    start_edge: bool,
    select_edge: bool,
    l3_edge: bool,
    r3_edge: bool,
}

impl PadBridge {
    pub fn button(&mut self, button: GamepadButton, pressed: bool) {
        let (held, edge) = match button {
            GamepadButton::South => (&mut self.south, &mut self.south_edge),
            GamepadButton::East => (&mut self.east, &mut self.east_edge),
            GamepadButton::West => (&mut self.west, &mut self.west_edge),
            GamepadButton::North => (&mut self.north, &mut self.north_edge),
            GamepadButton::DPadLeft => (&mut self.dpad_l, &mut self.dpad_l_edge),
            GamepadButton::DPadUp => (&mut self.dpad_u, &mut self.dpad_u_edge),
            GamepadButton::DPadRight => (&mut self.dpad_r, &mut self.dpad_r_edge),
            GamepadButton::DPadDown => (&mut self.dpad_d, &mut self.dpad_d_edge),
            GamepadButton::LeftShoulder => (&mut self.lb, &mut self.lb_edge),
            GamepadButton::RightShoulder => (&mut self.rb, &mut self.rb_edge),
            GamepadButton::Start => (&mut self.start, &mut self.start_edge),
            GamepadButton::Select => (&mut self.select, &mut self.select_edge),
            GamepadButton::LeftStick => (&mut self.l3, &mut self.l3_edge),
            GamepadButton::RightStick => (&mut self.r3, &mut self.r3_edge),
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
            vendor_id: self.vendor_id,
            product_id: self.product_id,
            left_stick: Vec2::new(self.lx, self.ly),
            right_stick: Vec2::new(self.rx, self.ry),
            left_trigger_held: self.lt_held,
            left_trigger_pressed: self.lt_edge,
            right_trigger_held: self.rt_held,
            right_trigger_pressed: self.rt_edge,
            south_held: self.south,
            south_pressed: self.south_edge,
            east_held: self.east,
            east_pressed: self.east_edge,
            west_held: self.west,
            west_pressed: self.west_edge,
            north_held: self.north,
            north_pressed: self.north_edge,
            dpad_left_held: self.dpad_l,
            dpad_left_pressed: self.dpad_l_edge,
            dpad_up_held: self.dpad_u,
            dpad_up_pressed: self.dpad_u_edge,
            dpad_right_held: self.dpad_r,
            dpad_right_pressed: self.dpad_r_edge,
            dpad_down_held: self.dpad_d,
            dpad_down_pressed: self.dpad_d_edge,
            left_shoulder_held: self.lb,
            left_shoulder_pressed: self.lb_edge,
            right_shoulder_held: self.rb,
            right_shoulder_pressed: self.rb_edge,
            start_held: self.start,
            start_pressed: self.start_edge,
            select_held: self.select,
            select_pressed: self.select_edge,
            left_stick_click_held: self.l3,
            left_stick_click_pressed: self.l3_edge,
            right_stick_click_held: self.r3,
            right_stick_click_pressed: self.r3_edge,
        }
    }

    pub fn clear_edges(&mut self) {
        self.lt_edge = false;
        self.rt_edge = false;
        self.south_edge = false;
        self.east_edge = false;
        self.west_edge = false;
        self.dpad_l_edge = false;
        self.dpad_u_edge = false;
        self.dpad_r_edge = false;
        self.dpad_d_edge = false;
        self.north_edge = false;
        self.lb_edge = false;
        self.rb_edge = false;
        self.start_edge = false;
        self.select_edge = false;
        self.l3_edge = false;
        self.r3_edge = false;
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
                GamepadEvent::Connected {
                    id,
                    vendor_id,
                    product_id,
                    ..
                } => {
                    let pad = self.pads.entry(id).or_default();
                    pad.vendor_id = vendor_id;
                    pad.product_id = product_id;
                }
                GamepadEvent::Disconnected { id } => {
                    self.pads.remove(&id);
                }
                GamepadEvent::Button {
                    id,
                    button,
                    pressed,
                } => {
                    self.pads.entry(id).or_default().button(button, pressed);
                }
                GamepadEvent::Axis { id, axis, value } => {
                    self.pads.entry(id).or_default().axis(axis, value);
                }
            }
        }
    }

    /// Snapshots paired with their device id (id-sorted): callers route
    /// per-player pads before the id is discarded. Prefer this over
    /// [`PadBank::drain`], which drops the routing key.
    pub fn drain_with_id(&mut self) -> Vec<(GamepadId, GamepadState)> {
        let mut ids: Vec<GamepadId> = self.pads.keys().copied().collect();
        ids.sort_by_key(|id| id.0);
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(bridge) = self.pads.get_mut(&id) {
                out.push((id, bridge.snapshot()));
                bridge.clear_edges();
            }
        }
        out
    }

    pub fn drain(&mut self) -> Vec<GamepadState> {
        self.drain_with_id().into_iter().map(|(_, s)| s).collect()
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

    #[test]
    fn shoulder_and_dpad_down_buttons_track() {
        let mut b = PadBridge::default();
        b.button(GamepadButton::RightShoulder, true);
        b.button(GamepadButton::LeftShoulder, true);
        b.button(GamepadButton::DPadDown, true);
        b.button(GamepadButton::West, true);
        let s = b.snapshot();
        assert!(s.right_shoulder_held && s.right_shoulder_pressed);
        assert!(s.left_shoulder_held && s.left_shoulder_pressed);
        assert!(s.dpad_down_pressed);
        assert!(s.west_pressed);
    }

    #[test]
    fn drain_with_id_routes_devices() {
        let mut bank = PadBank::default();
        let id0 = GamepadId(3);
        let id1 = GamepadId(1);
        bank.feed(vec![
            GamepadEvent::Connected {
                id: id0,
                name: "GameCube Adapter".to_string(),
                vendor_id: 0x057e,
                product_id: 0x0337,
            },
            GamepadEvent::Button {
                id: id0,
                button: GamepadButton::South,
                pressed: true,
            },
            GamepadEvent::Button {
                id: id1,
                button: GamepadButton::East,
                pressed: true,
            },
        ]);
        let drained = bank.drain_with_id();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].0, id1, "id-sorted");
        assert!(drained[1].1.south_pressed);
        assert!(drained[0].1.east_pressed);
        assert_eq!(
            (drained[1].1.vendor_id, drained[1].1.product_id),
            (0x057e, 0x0337),
            "pad identity must reach the game"
        );
        assert_eq!(
            (drained[0].1.vendor_id, drained[0].1.product_id),
            (0, 0),
            "a pad that never reported ids reads as zero"
        );
    }
}
