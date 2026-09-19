
use std::collections::HashMap;
use std::hash::Hash;

use repose_core::input::{GamepadAxis, GamepadButton, Key, Modifiers, PhysicalKey, PointerButton};
use repose_core::shortcuts::KeyChord;

/// One side of a remappable control: what the keyboard/mouse entry or
/// the gamepad entry currently holds.
#[derive(Clone, Debug, PartialEq)]
pub enum KeymapEntry {
    None,
    Key(KeyChord),
    Physical(PhysicalKey),
    Mouse(PointerButton),
    Pad(GamepadButton),
    Axis { axis: GamepadAxis, threshold: f32 },
}

impl Default for KeymapEntry {
    fn default() -> Self {
        KeymapEntry::None
    }
}

impl std::hash::Hash for KeymapEntry {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            KeymapEntry::None => {}
            KeymapEntry::Key(chord) => chord.hash(state),
            KeymapEntry::Physical(key) => key.hash(state),
            KeymapEntry::Mouse(button) => {
                (*button as u8).hash(state);
            }
            KeymapEntry::Pad(button) => button.hash(state),
            KeymapEntry::Axis { axis, threshold } => {
                axis.hash(state);
                let t = if *threshold == 0.0 { 0.0 } else { *threshold };
                t.to_bits().hash(state);
            }
        }
    }
}

/// next gamepad button when a pad drove the gesture, else the next
/// keyboard key / mouse button (`Other_10:374-404`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KeymapDevice {
    KeyboardMouse,
    Gamepad,
}

/// Pending rebind gesture: which action, which side, armed by the
/// options screen and resolved by the next pressed input.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KeymapCapture<A> {
    pub action: A,
    pub device: KeymapDevice,
}

/// Per-action remappable bindings: a keyboard/mouse entry plus a
/// (`Key[$ key][0]` keyboard, `Key[$ key][1]` gamepad).
#[derive(Clone, Debug, Default)]
pub struct Keymap<A> {
    keyboard: HashMap<A, KeymapEntry>,
    gamepad: HashMap<A, KeymapEntry>,
}

impl<A: Clone + Eq + Hash> Keymap<A> {
    pub fn new() -> Self {
        Self {
            keyboard: HashMap::new(),
            gamepad: HashMap::new(),
        }
    }

    pub fn set_keyboard(&mut self, action: A, entry: KeymapEntry) {
        self.keyboard.insert(action, entry);
    }

    pub fn set_gamepad(&mut self, action: A, entry: KeymapEntry) {
        self.gamepad.insert(action, entry);
    }

    pub fn keyboard(&self, action: &A) -> KeymapEntry {
        self.keyboard.get(action).cloned().unwrap_or_default()
    }

    pub fn gamepad(&self, action: &A) -> KeymapEntry {
        self.gamepad.get(action).cloned().unwrap_or_default()
    }

    /// Active entry for an action: gamepad side when a pad drives
    pub fn active(&self, action: &A, gamepad: bool) -> KeymapEntry {
        if gamepad {
            let entry = self.gamepad(action);
            if entry != KeymapEntry::None {
                return entry;
            }
        }
        self.keyboard(action)
    }

    /// Begin a rebind gesture for `action` on `device`
    pub fn begin_capture(&self, action: A, device: KeymapDevice) -> KeymapCapture<A> {
        KeymapCapture { action, device }
    }

    /// Resolve a capture with the next pressed input. `None` clears
    /// mouse button, or pad button/axis overwrites that side
    pub fn resolve_capture(
        &mut self,
        capture: &KeymapCapture<A>,
        pressed: Option<KeymapEntry>,
    ) {
        let entry = pressed.unwrap_or(KeymapEntry::None);
        match capture.device {
            KeymapDevice::KeyboardMouse => {
                self.keyboard.insert(capture.action.clone(), entry);
            }
            KeymapDevice::Gamepad => {
                self.gamepad.insert(capture.action.clone(), entry);
            }
        }
    }

    pub fn actions(&self) -> impl Iterator<Item = &A> {
        self.keyboard.keys().chain(self.gamepad.keys())
    }

    pub fn to_action_map(&self) -> super::map::ActionMap<A> {
        let mut out = super::map::ActionMap::new();
        for action in self.actions() {
            for entry in [self.keyboard(action), self.gamepad(action)] {
                if let Some(binding) = super::binding::Binding::from_entry(&entry) {
                    out.bind(action.clone(), binding);
                }
            }
        }
        out
    }

    pub fn to_action_map_contexts(
        &self,
        gameplay: &[A],
        menu: &[A],
    ) -> super::map::ActionMap<A> {
        let mut out = self.to_action_map();
        for action in gameplay {
            out.in_context("gameplay", action.clone());
        }
        for action in menu {
            out.in_context("menu", action.clone());
        }
        out
    }
}

/// Serializable row: one action name plus both entries as strings.
/// Games map `action` back to their enum; the strings stay readable
/// in the save file (`KeyW`, `Space`, `MouseLeft`, `PadSouth`, ...).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct KeymapRow {
    pub action: String,
    pub keyboard: String,
    pub gamepad: String,
}

pub fn encode_keymap_entry(entry: &KeymapEntry) -> String {
    match entry {
        KeymapEntry::None => String::new(),
        KeymapEntry::Key(chord) => encode_chord(chord),
        KeymapEntry::Physical(key) => key.name().to_string(),
        KeymapEntry::Mouse(PointerButton::Primary) => "MouseLeft".to_string(),
        KeymapEntry::Mouse(PointerButton::Secondary) => "MouseRight".to_string(),
        KeymapEntry::Mouse(PointerButton::Tertiary) => "MouseMiddle".to_string(),
        KeymapEntry::Pad(button) => format!("Pad{button:?}"),
        KeymapEntry::Axis { axis, threshold } => format!("Axis{axis:?}@{threshold}"),
    }
}

pub fn decode_keymap_entry(text: &str) -> KeymapEntry {
    let text = text.trim();
    if text.is_empty() {
        return KeymapEntry::None;
    }
    match text {
        "MouseLeft" => return KeymapEntry::Mouse(PointerButton::Primary),
        "MouseRight" => return KeymapEntry::Mouse(PointerButton::Secondary),
        "MouseMiddle" => return KeymapEntry::Mouse(PointerButton::Tertiary),
        _ => {}
    }
    if let Some(name) = text.strip_prefix("Pad")
        && let Some(button) = decode_pad_button(name)
    {
        return KeymapEntry::Pad(button);
    }
    if let Some(rest) = text.strip_prefix("Axis")
        && let Some((axis_name, threshold)) = rest.split_once('@')
        && let Some(axis) = decode_axis(axis_name)
        && let Ok(threshold) = threshold.parse::<f32>()
    {
        return KeymapEntry::Axis { axis, threshold };
    }
    let physical = PhysicalKey::from_name(text);
    if !matches!(physical, PhysicalKey::Unidentified) || text == "Unidentified" {
        return KeymapEntry::Physical(physical);
    }
    if let Some(chord) = decode_chord(text) {
        return KeymapEntry::Key(chord);
    }
    KeymapEntry::None
}

/// REMAP capture helper: every physical key position maps to a
/// plain (no-modifier) [`KeyChord`] so ANY key is capturable.
pub fn chord_for_physical(key: PhysicalKey) -> Option<KeyChord> {
    use repose_core::input::Key;
    let chord_key = match key {
        PhysicalKey::Space => Key::Space,
        PhysicalKey::Tab => Key::Tab,
        PhysicalKey::ShiftLeft => Key::ShiftLeft,
        PhysicalKey::ShiftRight => Key::ShiftRight,
        PhysicalKey::ArrowUp => Key::ArrowUp,
        PhysicalKey::ArrowDown => Key::ArrowDown,
        PhysicalKey::ArrowLeft => Key::ArrowLeft,
        PhysicalKey::ArrowRight => Key::ArrowRight,
        PhysicalKey::Enter => Key::Enter,
        PhysicalKey::Escape => Key::Escape,
        PhysicalKey::Backspace => Key::Backspace,
        PhysicalKey::Delete => Key::Delete,
        PhysicalKey::Insert => Key::Insert,
        PhysicalKey::Home => Key::Home,
        PhysicalKey::End => Key::End,
        PhysicalKey::PageUp => Key::PageUp,
        PhysicalKey::PageDown => Key::PageDown,
        _ => Key::Character(glyph_for_physical(key)?),
    };
    Some(KeyChord::new(chord_key, Modifiers::default()))
}

/// Physical-position -> glyph table.
pub fn glyph_for_physical(key: PhysicalKey) -> Option<char> {
    let name = key.name();
    if let Some(tail) = name
        .strip_prefix("Key")
        .or_else(|| name.strip_prefix("Digit"))
    {
        let mut chars = tail.chars();
        match (chars.next(), chars.next()) {
            (Some(c), None) => return Some(c.to_ascii_lowercase()),
            _ => return None,
        }
    }
    Some(match key {
        PhysicalKey::Backquote => '`',
        PhysicalKey::Minus => '-',
        PhysicalKey::Equal => '=',
        PhysicalKey::BracketLeft => '[',
        PhysicalKey::BracketRight => ']',
        PhysicalKey::Backslash => '\\',
        PhysicalKey::Semicolon => ';',
        PhysicalKey::Quote => '\'',
        PhysicalKey::Comma => ',',
        PhysicalKey::Period => '.',
        PhysicalKey::Slash => '/',
        _ => return None,
    })
}

fn encode_chord(chord: &KeyChord) -> String {
    let mut out = String::new();
    if chord.modifiers.ctrl {
        out.push_str("Ctrl+");
    }
    if chord.modifiers.shift {
        out.push_str("Shift+");
    }
    if chord.modifiers.alt {
        out.push_str("Alt+");
    }
    if chord.modifiers.meta {
        out.push_str("Meta+");
    }
    out.push_str(&encode_key(&chord.key));
    out
}

fn encode_key(key: &Key) -> String {
    match key {
        Key::Character(c) => {
            if *c == ' ' {
                "Space".to_string()
            } else {
                c.to_ascii_uppercase().to_string()
            }
        }
        Key::Enter => "Enter".to_string(),
        Key::Tab => "Tab".to_string(),
        Key::Backspace => "Backspace".to_string(),
        Key::Delete => "Delete".to_string(),
        Key::Insert => "Insert".to_string(),
        Key::Escape => "Escape".to_string(),
        Key::ArrowLeft => "ArrowLeft".to_string(),
        Key::ArrowRight => "ArrowRight".to_string(),
        Key::ArrowUp => "ArrowUp".to_string(),
        Key::ArrowDown => "ArrowDown".to_string(),
        Key::Home => "Home".to_string(),
        Key::End => "End".to_string(),
        Key::PageUp => "PageUp".to_string(),
        Key::PageDown => "PageDown".to_string(),
        Key::Space => "Space".to_string(),
        Key::ShiftLeft => "ShiftLeft".to_string(),
        Key::ShiftRight => "ShiftRight".to_string(),
        Key::F(n) => format!("F{n}"),
        Key::Unknown => String::new(),
    }
}

fn decode_chord(text: &str) -> Option<KeyChord> {
    let mut rest = text;
    let mut modifiers = Modifiers::default();
    loop {
        if let Some(tail) = rest.strip_prefix("Ctrl+") {
            modifiers.ctrl = true;
            rest = tail;
        } else if let Some(tail) = rest.strip_prefix("Shift+") {
            modifiers.shift = true;
            rest = tail;
        } else if let Some(tail) = rest.strip_prefix("Alt+") {
            modifiers.alt = true;
            rest = tail;
        } else if let Some(tail) = rest.strip_prefix("Meta+") {
            modifiers.meta = true;
            rest = tail;
        } else {
            break;
        }
    }
    decode_key(rest).map(|key| KeyChord::new(key, modifiers))
}

fn decode_key(text: &str) -> Option<Key> {
    match text {
        "Enter" => Some(Key::Enter),
        "Tab" => Some(Key::Tab),
        "Backspace" => Some(Key::Backspace),
        "Delete" => Some(Key::Delete),
        "Insert" => Some(Key::Insert),
        "Escape" => Some(Key::Escape),
        "ArrowLeft" => Some(Key::ArrowLeft),
        "ArrowRight" => Some(Key::ArrowRight),
        "ArrowUp" => Some(Key::ArrowUp),
        "ArrowDown" => Some(Key::ArrowDown),
        "Home" => Some(Key::Home),
        "End" => Some(Key::End),
        "PageUp" => Some(Key::PageUp),
        "PageDown" => Some(Key::PageDown),
        "Space" => Some(Key::Space),
        "ShiftLeft" => Some(Key::ShiftLeft),
        "ShiftRight" => Some(Key::ShiftRight),
        _ => {
            if let Some(n) = text.strip_prefix('F').and_then(|n| n.parse::<u8>().ok())
                && (1..=12).contains(&n)
            {
                return Some(Key::F(n));
            }
            let mut chars = text.chars();
            if let (Some(c), None) = (chars.next(), chars.next()) {
                Some(Key::Character(c.to_ascii_lowercase()))
            } else {
                None
            }
        }
    }
}

fn decode_pad_button(name: &str) -> Option<GamepadButton> {
    Some(match name {
        "South" => GamepadButton::South,
        "East" => GamepadButton::East,
        "West" => GamepadButton::West,
        "North" => GamepadButton::North,
        "Start" => GamepadButton::Start,
        "Select" => GamepadButton::Select,
        "LeftShoulder" => GamepadButton::LeftShoulder,
        "RightShoulder" => GamepadButton::RightShoulder,
        "LeftStick" => GamepadButton::LeftStick,
        "RightStick" => GamepadButton::RightStick,
        "DPadUp" => GamepadButton::DPadUp,
        "DPadDown" => GamepadButton::DPadDown,
        "DPadLeft" => GamepadButton::DPadLeft,
        "DPadRight" => GamepadButton::DPadRight,
        _ => return None,
    })
}

fn decode_axis(name: &str) -> Option<GamepadAxis> {
    Some(match name {
        "LeftStickX" => GamepadAxis::LeftStickX,
        "LeftStickY" => GamepadAxis::LeftStickY,
        "RightStickX" => GamepadAxis::RightStickX,
        "RightStickY" => GamepadAxis::RightStickY,
        "LeftTrigger" => GamepadAxis::LeftTrigger,
        "RightTrigger" => GamepadAxis::RightTrigger,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chord(c: char) -> KeymapEntry {
        KeymapEntry::Key(KeyChord::new(Key::Character(c), Modifiers::default()))
    }

    #[test]
    fn chord_covers_full_physical_alphabet() {
        use repose_core::input::PhysicalKey;
        assert_eq!(
            chord_for_physical(PhysicalKey::KeyW),
            Some(KeyChord::new(Key::Character('w'), Modifiers::default()))
        );
        assert_eq!(
            chord_for_physical(PhysicalKey::Digit1),
            Some(KeyChord::new(Key::Character('1'), Modifiers::default()))
        );
        assert_eq!(
            chord_for_physical(PhysicalKey::Minus),
            Some(KeyChord::new(Key::Character('-'), Modifiers::default()))
        );
        assert_eq!(
            chord_for_physical(PhysicalKey::Space),
            Some(KeyChord::new(Key::Space, Modifiers::default()))
        );
        assert_eq!(chord_for_physical(PhysicalKey::F5), None);
        assert_eq!(chord_for_physical(PhysicalKey::Unidentified), None);
    }

    #[test]
    fn glyph_covers_letters_digits_punctuation() {
        use repose_core::input::PhysicalKey;
        assert_eq!(glyph_for_physical(PhysicalKey::KeyQ), Some('q'));
        assert_eq!(glyph_for_physical(PhysicalKey::Digit7), Some('7'));
        assert_eq!(glyph_for_physical(PhysicalKey::Minus), Some('-'));
        assert_eq!(glyph_for_physical(PhysicalKey::Slash), Some('/'));
        assert_eq!(glyph_for_physical(PhysicalKey::Space), None);
        assert_eq!(glyph_for_physical(PhysicalKey::F1), None);
    }

    #[test]
    fn keyboard_and_gamepad_sides_stay_separate() {
        let mut map = Keymap::new();
        map.set_keyboard("fire", KeymapEntry::Mouse(PointerButton::Primary));
        map.set_gamepad("fire", KeymapEntry::Pad(GamepadButton::South));
        assert_eq!(
            map.active(&"fire", false),
            KeymapEntry::Mouse(PointerButton::Primary)
        );
        assert_eq!(
            map.active(&"fire", true),
            KeymapEntry::Pad(GamepadButton::South)
        );
    }

    #[test]
    fn gamepad_falls_back_to_keyboard_when_unbound() {
        let mut map = Keymap::new();
        map.set_keyboard("north", chord('w'));
        assert_eq!(map.active(&"north", true), chord('w'));
        assert_eq!(map.active(&"missing", true), KeymapEntry::None);
    }

    #[test]
    fn capture_overwrites_one_side_only() {
        let mut map = Keymap::new();
        map.set_keyboard("fire", KeymapEntry::Mouse(PointerButton::Primary));
        map.set_gamepad("fire", KeymapEntry::Pad(GamepadButton::South));
        let capture = map.begin_capture("fire", KeymapDevice::KeyboardMouse);
        map.resolve_capture(&capture, Some(chord('f')));
        assert_eq!(map.keyboard(&"fire"), chord('f'));
        assert_eq!(
            map.gamepad(&"fire"),
            KeymapEntry::Pad(GamepadButton::South)
        );
        let capture = map.begin_capture("fire", KeymapDevice::Gamepad);
        map.resolve_capture(&capture, None);
        assert_eq!(map.gamepad(&"fire"), KeymapEntry::None);
        assert_eq!(map.active(&"fire", true), chord('f'));
    }

    #[test]
    fn entries_round_trip_through_strings() {
        let entries = [
            KeymapEntry::None,
            chord('w'),
            KeymapEntry::Key(KeyChord::new(
                Key::Space,
                Modifiers {
                    shift: true,
                    ..Modifiers::default()
                },
            )),
            KeymapEntry::Mouse(PointerButton::Primary),
            KeymapEntry::Mouse(PointerButton::Secondary),
            KeymapEntry::Mouse(PointerButton::Tertiary),
            KeymapEntry::Pad(GamepadButton::DPadUp),
            KeymapEntry::Axis {
                axis: GamepadAxis::LeftTrigger,
                threshold: 0.5,
            },
        ];
        for entry in entries {
            let text = encode_keymap_entry(&entry);
            assert_eq!(decode_keymap_entry(&text), entry, "round trip {text:?}");
        }
        assert_eq!(decode_keymap_entry("bogus"), KeymapEntry::None);
    }
}
