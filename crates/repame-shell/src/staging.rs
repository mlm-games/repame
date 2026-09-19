use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use glam::Vec2;
use repame_input::{AimTracker, GamepadState, MouseState, StagedClick, TouchContact};
use repose_core::input::{Key, KeyEvent, KeyEventType, PhysicalKey, PointerButton};
use repose_core::runtime::Scheduler;

#[derive(Default)]
pub struct Staging {
    pub held: HashSet<PhysicalKey>,
    pub edges: Vec<PhysicalKey>,
    pub window_focused: bool,
    pub clicks: Vec<StagedClick>,
    pub aim: AimTracker,
    pub mouse_edges: Vec<(PointerButton, bool)>,
    pub lmb_held: bool,
    pub rmb_held: bool,
    pub rmb_down_edge: bool,
    pub touch_active: HashMap<u64, (Vec2, Vec2)>,
    pub touch_new: HashSet<u64>,
    pub pads: Vec<GamepadState>,
    pub pad_live: bool,
    pub view_width: f32,
}

impl Staging {
    pub fn shared() -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            window_focused: true,
            view_width: 1280.0,
            ..Default::default()
        }))
    }

    pub fn handle_key(&mut self, ke: &KeyEvent) {
        let down = matches!(ke.event_type, KeyEventType::Down);
        match &ke.key {
            Key::Escape | Key::Enter => {}
            Key::Character(_) => {
                if let Some(key) = ke.physical {
                    self.stage_physical(key, down, ke.is_repeat);
                }
            }
            Key::ShiftLeft => self.stage_code(PhysicalKey::ShiftLeft, down, ke.is_repeat),
            Key::ShiftRight => self.stage_code(PhysicalKey::ShiftRight, down, ke.is_repeat),
            Key::Space => self.stage_code(PhysicalKey::Space, down, ke.is_repeat),
            Key::Tab => self.stage_code(PhysicalKey::Tab, down, ke.is_repeat),
            Key::ArrowUp => self.stage_code(PhysicalKey::ArrowUp, down, ke.is_repeat),
            Key::ArrowDown => self.stage_code(PhysicalKey::ArrowDown, down, ke.is_repeat),
            Key::ArrowLeft => self.stage_code(PhysicalKey::ArrowLeft, down, ke.is_repeat),
            Key::ArrowRight => self.stage_code(PhysicalKey::ArrowRight, down, ke.is_repeat),
            _ => {}
        }
    }

    pub fn stage_physical(&mut self, key: PhysicalKey, down: bool, is_repeat: bool) {
        if down {
            if self.held.insert(key) && !is_repeat {
                self.edges.push(key);
            }
        } else {
            self.held.remove(&key);
        }
    }

    fn stage_code(&mut self, code: PhysicalKey, down: bool, is_repeat: bool) {
        self.stage_physical(code, down, is_repeat);
    }

    pub fn stage_physical_key(&mut self, key: PhysicalKey, down: bool) {
        self.stage_physical(key, down, false);
    }

    pub fn set_window_focused(&mut self, focused: bool) {
        self.window_focused = focused;
        if !focused {
            self.held.clear();
            self.edges.clear();
            self.lmb_held = false;
            self.rmb_held = false;
        }
    }

    pub fn feed_polled(&mut self, sched: &Scheduler) {
        super::apply_scheduler_levels(
            &mut self.held,
            &mut self.lmb_held,
            &mut self.rmb_held,
            &mut self.window_focused,
            sched,
            |k| Some(k),
            |_| None,
        );
        if !self.window_focused {
            self.edges.clear();
        }
    }

    pub fn pick_down(&mut self, button: PointerButton) {
        self.mouse_edges.push((button, true));
        if button == PointerButton::Primary {
            self.lmb_held = true;
        } else {
            self.rmb_down_edge = true;
            self.rmb_held = true;
        }
    }

    pub fn pick_up(&mut self, button: PointerButton) {
        self.mouse_edges.push((button, false));
        if button == PointerButton::Primary {
            self.lmb_held = false;
        } else {
            self.rmb_held = false;
        }
    }

    pub fn take_rmb_down(&mut self) -> bool {
        std::mem::replace(&mut self.rmb_down_edge, false)
    }

    pub fn stage_click(&mut self, world: Vec2, screen: [f32; 2], density: f32) {
        let d = density.max(1e-6);
        self.clicks.push(StagedClick {
            world,
            dp: [screen[0] / d, screen[1] / d],
        });
    }

    pub fn take_clicks(&mut self) -> Vec<StagedClick> {
        std::mem::take(&mut self.clicks)
    }

    pub fn take_edges(&mut self) -> Vec<PhysicalKey> {
        std::mem::take(&mut self.edges)
    }

    pub fn take_mouse_edges(&mut self) -> Vec<(PointerButton, bool)> {
        std::mem::take(&mut self.mouse_edges)
    }

    pub fn take_pads(&mut self) -> Vec<GamepadState> {
        self.pad_live = !self.pads.is_empty();
        std::mem::take(&mut self.pads)
    }

    pub fn stage_gamepad(&mut self, pad: GamepadState) {
        self.pads.push(pad);
        self.pad_live = true;
    }

    pub fn touch_down(&mut self, id: u64, screen: Vec2) {
        self.touch_active.insert(id, (screen, screen));
        self.touch_new.insert(id);
    }

    pub fn touch_move(&mut self, id: u64, screen: Vec2) {
        if let Some(contact) = self.touch_active.get_mut(&id) {
            contact.1 = screen;
        }
    }

    pub fn touch_up(&mut self, id: u64) {
        self.touch_active.remove(&id);
        self.touch_new.remove(&id);
    }

    pub fn touch_contacts(&mut self, density: f32) -> Vec<TouchContact> {
        let d = density.max(1e-6);
        let out: Vec<TouchContact> = self
            .touch_active
            .iter()
            .map(|(id, (start, pos))| TouchContact {
                start: *start / d,
                pos: *pos / d,
                just_pressed: self.touch_new.contains(id),
            })
            .collect();
        self.touch_new.clear();
        out
    }

    pub fn cursor_move(&mut self, phys_px: Vec2) {
        self.aim.cursor_move(phys_px);
    }

    pub fn stage_hover(&mut self, world: Vec2, screen: [f32; 2]) {
        self.aim.stage_hover(world, screen);
    }

    pub fn mouse_state(&self, allow_fire: bool) -> MouseState {
        MouseState {
            left_held: self.lmb_held && allow_fire,
            left_pressed: !self.clicks.is_empty() && allow_fire,
            right_held: self.rmb_held,
            right_pressed: self.rmb_down_edge,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use repose_core::input::{Key, Modifiers};

    fn key(physical: PhysicalKey) -> KeyEvent {
        KeyEvent {
            key: Key::Character('x'),
            modifiers: Modifiers::default(),
            is_repeat: false,
            event_type: KeyEventType::Down,
            utf16_code_point: 0,
            physical: Some(physical),
        }
    }

    #[test]
    fn edges_dedupe_while_held() {
        let mut s = Staging::default();
        s.handle_key(&key(PhysicalKey::KeyW));
        s.handle_key(&key(PhysicalKey::KeyW));
        assert_eq!(s.take_edges().len(), 1);
    }

    #[test]
    fn focus_loss_clears_levels() {
        let mut s = Staging::default();
        s.handle_key(&key(PhysicalKey::KeyW));
        s.set_window_focused(false);
        assert!(s.held.is_empty());
    }
}
