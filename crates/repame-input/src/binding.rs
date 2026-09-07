//! Physical input bindings: one action fires from many inputs.

use repose_core::input::{GamepadAxis, GamepadButton, PointerButton};
use repose_core::shortcuts::KeyChord;

/// One physical source that can drive an action. Many-to-one: several
/// bindings can point at the same action (keyboard + pad + axis).
#[derive(Clone, Debug)]
pub enum Binding {
    /// Keyboard chord (key + modifiers), same shape as repose shortcuts.
    Key(KeyChord),
    /// Mouse button press.
    Mouse(PointerButton),
    /// Gamepad button press.
    Pad(GamepadButton),
    /// Analog axis crossing. Active while `value >= threshold` for a
    /// positive threshold, `value <= threshold` for a negative one, so
    /// `-0.5` on `LeftStickX` means "pushed left past half".
    Axis { axis: GamepadAxis, threshold: f32 },
}

// Manual `PartialEq`: repose `PointerButton` has no `PartialEq`, so
// compare mouse variants by discriminant.
impl PartialEq for Binding {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Binding::Key(a), Binding::Key(b)) => a == b,
            (Binding::Mouse(a), Binding::Mouse(b)) => {
                std::mem::discriminant(a) == std::mem::discriminant(b)
            }
            (Binding::Pad(a), Binding::Pad(b)) => a == b,
            (
                Binding::Axis { axis: a, threshold: t },
                Binding::Axis { axis: b, threshold: u },
            ) => a == b && t == u,
            _ => false,
        }
    }
}

impl Binding {
    /// True while the latest axis value holds past the threshold.
    pub fn axis_active(threshold: f32, value: f32) -> bool {
        if threshold >= 0.0 {
            value >= threshold
        } else {
            value <= threshold
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_threshold_sign_selects_side() {
        assert!(Binding::axis_active(0.5, 0.7));
        assert!(!Binding::axis_active(0.5, 0.3));
        assert!(Binding::axis_active(-0.5, -0.7));
        assert!(!Binding::axis_active(-0.5, -0.3));
        assert!(Binding::axis_active(0.0, 0.0));
    }
}
