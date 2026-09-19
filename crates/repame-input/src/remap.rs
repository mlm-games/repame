use bevy_ecs::prelude::*;
use super::keymap::{Keymap, KeymapCapture, KeymapDevice, KeymapEntry};
use repose_core::input::{Key, Modifiers};
use repose_core::shortcuts::KeyChord;

#[derive(Resource, Clone, Debug)]
pub struct RemapSession<A> {
    pub map: Keymap<A>,
    pub capture: Option<KeymapCapture<A>>,
}

impl<A: Clone + Eq + std::hash::Hash> Default for RemapSession<A> {
    fn default() -> Self {
        Self {
            map: Keymap::new(),
            capture: None,
        }
    }
}

impl<A: Clone + Eq + std::hash::Hash> RemapSession<A> {
    pub fn armed(&self) -> bool {
        self.capture.is_some()
    }

    pub fn begin(&mut self, action: A, device: KeymapDevice) {
        self.capture = Some(self.map.begin_capture(action, device));
    }

    pub fn cancel(&mut self) {
        self.capture = None;
    }

    pub fn resolve(&mut self, pressed: Option<KeymapEntry>) -> bool {
        if let Some(capture) = self.capture.take() {
            self.map.resolve_capture(&capture, pressed);
            true
        } else {
            false
        }
    }

    pub fn resolve_physical(&mut self, key: repose_core::input::PhysicalKey) -> bool {
        self.resolve(Some(KeymapEntry::Physical(key)))
    }

    pub fn resolve_key(&mut self, key: &Key) -> bool {
        let chord = match key {
            Key::Space => KeyChord::new(Key::Space, Modifiers::default()),
            Key::Tab => KeyChord::new(Key::Tab, Modifiers::default()),
            Key::Enter => KeyChord::new(Key::Enter, Modifiers::default()),
            Key::Escape => KeyChord::new(Key::Escape, Modifiers::default()),
            Key::ShiftLeft => KeyChord::new(Key::ShiftLeft, Modifiers::default()),
            Key::ShiftRight => KeyChord::new(Key::ShiftRight, Modifiers::default()),
            Key::Character(c) => {
                KeyChord::new(Key::Character(c.to_ascii_lowercase()), Modifiers::default())
            }
            _ => return false,
        };
        self.resolve(Some(KeymapEntry::Key(chord)))
    }

    pub fn resolve_mouse(&mut self, left: bool) -> bool {
        let entry = if left {
            KeymapEntry::Mouse(repose_core::input::PointerButton::Primary)
        } else {
            KeymapEntry::Mouse(repose_core::input::PointerButton::Secondary)
        };
        if self
            .capture
            .as_ref()
            .is_some_and(|c| c.device == KeymapDevice::KeyboardMouse)
        {
            self.resolve(Some(entry))
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_clears_capture() {
        let mut s: RemapSession<&str> = RemapSession::default();
        s.begin("north", KeymapDevice::KeyboardMouse);
        assert!(s.armed());
        assert!(s.resolve_physical(repose_core::input::PhysicalKey::KeyZ));
        assert!(!s.armed());
    }
}
