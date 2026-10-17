//! Where an xr-standard gamepad carries each declared input.
//!
//! WebXR has no interaction profiles to bind against: a session's input
//! sources expose one gamepad each, and which control its buttons and axes are
//! comes from the gamepad's own `mapping`. The `xr-standard` mapping is the one
//! WebXR defines, and it is the only one this crate reads:
//!
//! ```text
//! buttons[0]  trigger, and the select input that feeds it
//! buttons[1]  squeeze, and the grip input that feeds it
//! buttons[3]  the thumbstick's click
//! axes[2]     the thumbstick's x
//! axes[3]     the thumbstick's y
//! ```
//!
//! A gamepad that reports another mapping, or a shorter one (browsers list only
//! the controls the device has), leaves the actions bound to those paths
//! inactive; nothing here guesses an index for them. Controls the standard
//! mapping does define but this crate does not read, the touchpad at
//! `buttons[2]` and `axes[0..2]` and the menu button at `buttons[4]`, are left
//! out for the same reason: reading them would be a guess about a control the
//! app did not name.

use glam::Vec2;

use crate::action::{ActionKind, ActionValue, InputPath, UserPath};

/// One button of a gamepad, as a sync read it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GamepadButton {
    /// `buttons[i].pressed`.
    pub pressed: bool,
    /// `buttons[i].value`.
    pub value: f32,
}

/// Where an xr-standard gamepad carries one declared input, as the gamepad
/// specification indexes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GamepadElement {
    /// A button's press state: `buttons[index].pressed`.
    Button(u32),
    /// A button's analog value: `buttons[index].value`.
    ButtonValue(u32),
    /// A two-axis control: `axes[index]` and `axes[index + 1]`.
    Axes(u32),
}

/// The element an action of `kind` bound to `path` reads on an xr-standard
/// gamepad, or `None` when the standard mapping carries no such control.
///
/// The input decides the button or stick, the component and the action kind
/// decide which reading of it (a press or an analog value), and the two have
/// to agree: a boolean action bound to a value component is not a press, and
/// this says so rather than reading the other one.
pub fn gamepad_element(kind: ActionKind, path: &InputPath) -> Option<GamepadElement> {
    let UserPath::Hand(_) = path.user else {
        // A path on `/user/gamepad`, `/user/head` or anything else is not an
        // input source's hand, and WebXR reports no handedness for those.
        return None;
    };
    let component = path.component.as_deref();
    match (path.input.as_str(), component, kind) {
        ("select" | "trigger", None | Some("click"), ActionKind::Boolean) => {
            Some(GamepadElement::Button(0))
        }
        ("select" | "trigger", None | Some("value"), ActionKind::Float) => {
            Some(GamepadElement::ButtonValue(0))
        }
        ("squeeze", None | Some("click"), ActionKind::Boolean) => Some(GamepadElement::Button(1)),
        ("squeeze", None | Some("value"), ActionKind::Float) => {
            Some(GamepadElement::ButtonValue(1))
        }
        ("thumbstick", None | Some("click"), ActionKind::Boolean) => {
            Some(GamepadElement::Button(3))
        }
        ("thumbstick", None, ActionKind::Vector2) => Some(GamepadElement::Axes(2)),
        _ => None,
    }
}

/// The reading an xr-standard gamepad carries at `element`, or `None` when the
/// gamepad does not have that control. A shorter gamepad leaves the action
/// bound past its end inactive rather than reading a neighbouring control.
pub fn element_reading(
    buttons: &[GamepadButton],
    axes: &[f32],
    element: GamepadElement,
) -> Option<ActionValue> {
    match element {
        GamepadElement::Button(index) => buttons
            .get(index as usize)
            .map(|button| ActionValue::Bool(button.pressed)),
        GamepadElement::ButtonValue(index) => buttons
            .get(index as usize)
            .map(|button| ActionValue::Float(button.value)),
        GamepadElement::Axes(index) => {
            let index = index as usize;
            let x = *axes.get(index)?;
            let y = *axes.get(index + 1)?;
            Some(ActionValue::Vector2(Vec2::new(x, y)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(text: &str) -> InputPath {
        InputPath::parse(text).expect("a well-formed test path")
    }

    #[test]
    fn xr_standard_indices_come_from_the_declared_bindings() {
        let cases: [(ActionKind, &str, GamepadElement); 8] = [
            (
                ActionKind::Boolean,
                "/user/hand/left/input/select/click",
                GamepadElement::Button(0),
            ),
            (
                ActionKind::Boolean,
                "/user/hand/right/input/select",
                GamepadElement::Button(0),
            ),
            (
                ActionKind::Float,
                "/user/hand/right/input/select",
                GamepadElement::ButtonValue(0),
            ),
            (
                ActionKind::Float,
                "/user/hand/left/input/trigger/value",
                GamepadElement::ButtonValue(0),
            ),
            (
                ActionKind::Boolean,
                "/user/hand/left/input/squeeze/click",
                GamepadElement::Button(1),
            ),
            (
                ActionKind::Float,
                "/user/hand/right/input/squeeze/value",
                GamepadElement::ButtonValue(1),
            ),
            (
                ActionKind::Vector2,
                "/user/hand/left/input/thumbstick",
                GamepadElement::Axes(2),
            ),
            (
                ActionKind::Boolean,
                "/user/hand/right/input/thumbstick/click",
                GamepadElement::Button(3),
            ),
        ];
        for (kind, text, element) in cases {
            assert_eq!(gamepad_element(kind, &path(text)), Some(element), "{text}");
        }
    }

    #[test]
    fn unmapped_paths_are_not_guessed_at() {
        let cases: [(ActionKind, &str); 12] = [
            (ActionKind::Boolean, "/user/hand/left/input/select/value"),
            (ActionKind::Float, "/user/hand/left/input/select/click"),
            (ActionKind::Vector2, "/user/hand/left/input/select/value"),
            (ActionKind::Float, "/user/hand/left/input/thumbstick/value"),
            (ActionKind::Vector2, "/user/hand/left/input/squeeze/value"),
            (ActionKind::Boolean, "/user/hand/left/input/select/touch"),
            (ActionKind::Boolean, "/user/hand/left/input/menu/click"),
            (ActionKind::Vector2, "/user/hand/left/input/trackpad"),
            (ActionKind::Boolean, "/user/head/input/select/click"),
            (ActionKind::Boolean, "/user/gamepad/input/select/click"),
            (ActionKind::Boolean, "/user/hand/left/input/aim/activate"),
            (ActionKind::Float, "/user/hand/left/input/trigger/touch"),
        ];
        for (kind, text) in cases {
            assert_eq!(
                gamepad_element(kind, &path(text)),
                None,
                "{text} read as {kind:?}"
            );
        }
    }

    #[test]
    fn shorter_gamepads_leave_controls_past_their_end_unread() {
        // A gamepad with only a touchpad: two buttons and two axes.
        let buttons = [GamepadButton {
            pressed: true,
            value: 0.5,
        }; 2];
        let axes = [0.25, -0.5];
        assert_eq!(
            element_reading(&buttons, &axes, GamepadElement::Button(1)),
            Some(ActionValue::Bool(true))
        );
        assert_eq!(
            element_reading(&buttons, &axes, GamepadElement::ButtonValue(1)),
            Some(ActionValue::Float(0.5))
        );
        assert_eq!(element_reading(&buttons, &axes, GamepadElement::Button(3)), None);
        assert_eq!(element_reading(&buttons, &axes, GamepadElement::Axes(2)), None);
        assert_eq!(element_reading(&[], &[], GamepadElement::Button(0)), None);

        // A thumbstick needs both of its axes.
        let axes = [0.25, -0.5, 0.75, -1.0];
        assert_eq!(
            element_reading(&buttons, &axes, GamepadElement::Axes(2)),
            Some(ActionValue::Vector2(Vec2::new(0.75, -1.0)))
        );
        assert_eq!(
            element_reading(&buttons, &[0.0, 0.0, 0.5], GamepadElement::Axes(2)),
            None
        );
    }
}
