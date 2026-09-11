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

// Manual `PartialEq` + `Eq` + `Hash`: repose `PointerButton` has no
// `PartialEq`, so compare/hash mouse variants by discriminant.
impl PartialEq for Binding {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Binding::Key(a), Binding::Key(b)) => a == b,
            (Binding::Mouse(a), Binding::Mouse(b)) => {
                std::mem::discriminant(a) == std::mem::discriminant(b)
            }
            (Binding::Pad(a), Binding::Pad(b)) => a == b,
            (
                Binding::Axis {
                    axis: a,
                    threshold: t,
                },
                Binding::Axis {
                    axis: b,
                    threshold: u,
                },
            ) => a == b && t == u,
            _ => false,
        }
    }
}

impl Eq for Binding {}

impl std::hash::Hash for Binding {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Binding::Key(chord) => chord.hash(state),
            Binding::Mouse(b) => std::mem::discriminant(b).hash(state),
            Binding::Pad(b) => b.hash(state),
            Binding::Axis { axis, threshold } => {
                axis.hash(state);
                let t = if *threshold == 0.0 { 0.0 } else { *threshold };
                t.to_bits().hash(state);
            }
        }
    }
}

impl Binding {
    /// True while the latest axis value holds past the threshold.
    /// A zero threshold means "any non-zero deflection" (rest stick at
    /// exactly 0.0 is inactive), so a centered stick never holds an
    /// action forever.
    pub fn axis_active(threshold: f32, value: f32) -> bool {
        if !value.is_finite() {
            return false;
        }
        if threshold == 0.0 {
            value != 0.0
        } else if threshold > 0.0 {
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
        assert!(!Binding::axis_active(0.0, 0.0));
        assert!(Binding::axis_active(0.0, 0.1));
        assert!(Binding::axis_active(0.0, -0.1));
        assert!(!Binding::axis_active(0.0, f32::NAN));
        assert!(!Binding::axis_active(0.5, f32::NAN));
        assert!(!Binding::axis_active(-0.5, f32::NAN));
    }

    #[test]
    fn bindings_hash_for_sets() {
        use repose_core::input::{GamepadAxis, GamepadButton, Key, Modifiers, PointerButton};
        use repose_core::shortcuts::KeyChord;
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Binding::Pad(GamepadButton::South));
        assert!(set.contains(&Binding::Pad(GamepadButton::South)));
        set.insert(Binding::Mouse(PointerButton::Primary));
        assert!(set.contains(&Binding::Mouse(PointerButton::Primary)));
        assert!(!set.contains(&Binding::Mouse(PointerButton::Secondary)));
        set.insert(Binding::Axis {
            axis: GamepadAxis::LeftStickX,
            threshold: 0.5,
        });
        assert!(set.contains(&Binding::Axis {
            axis: GamepadAxis::LeftStickX,
            threshold: 0.5,
        }));
        assert!(!set.contains(&Binding::Axis {
            axis: GamepadAxis::LeftStickX,
            threshold: 0.6,
        }));
        let _ = KeyChord::new(Key::Space, Modifiers::default());
    }
}
