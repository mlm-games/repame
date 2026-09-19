use glam::Vec2;

#[derive(Clone, Copy, Debug, Default)]
pub struct MouseState {
    pub left_held: bool,
    pub left_pressed: bool,
    pub right_held: bool,
    pub right_pressed: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GamepadState {
    pub left_stick: Vec2,
    pub right_stick: Vec2,
    pub right_trigger_held: bool,
    pub right_trigger_pressed: bool,
    pub left_trigger_held: bool,
    pub left_trigger_pressed: bool,
    pub south_pressed: bool,
    pub east_pressed: bool,
    pub dpad_left_pressed: bool,
    pub dpad_up_pressed: bool,
    pub dpad_right_pressed: bool,
    pub north_pressed: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TouchContact {
    pub start: Vec2,
    pub pos: Vec2,
    pub just_pressed: bool,
}

pub fn dead_zone(value: Vec2) -> Vec2 {
    const DEAD_ZONE: f32 = 0.22;

    let length = value.length();
    if length <= DEAD_ZONE {
        return Vec2::ZERO;
    }

    let scaled = ((length - DEAD_ZONE) / (1.0 - DEAD_ZONE)).clamp(0.0, 1.0);
    value.normalize_or_zero() * scaled
}

pub fn apply_stick(raw: Vec2) -> Vec2 {
    dead_zone(raw).clamp_length_max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dead_zone_kills_rest() {
        assert_eq!(dead_zone(Vec2::new(0.1, 0.0)), Vec2::ZERO);
    }

    #[test]
    fn full_deflection_survives() {
        let out = apply_stick(Vec2::new(1.0, 0.0));
        assert!((out.length() - 1.0).abs() < 1e-6);
    }
}
