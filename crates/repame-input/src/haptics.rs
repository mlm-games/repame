use std::collections::BTreeMap;

/// A constant-strength motor command. `duration` of zero runs until stopped.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HapticEffect {
    pub strong: f32,
    pub weak: f32,
    pub duration: f32,
}

impl HapticEffect {
    pub fn constant(strong: f32, weak: f32) -> Self {
        Self {
            strong,
            weak,
            duration: 0.0,
        }
    }

    pub fn for_seconds(strong: f32, weak: f32, duration: f32) -> Self {
        Self {
            strong,
            weak,
            duration: duration.max(0.0),
        }
    }
}

/// Platform hook that receives resolved motor strengths for one device.
pub trait HapticSink {
    fn apply(&mut self, device: u64, strong: f32, weak: f32);
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Motor {
    strong: f32,
    weak: f32,
    remaining: f32,
    timed: bool,
}

/// Per-device haptic state with tick-accurate effect expiry.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HapticBus {
    devices: BTreeMap<u64, Motor>,
}

impl HapticBus {
    pub fn new() -> Self {
        Self::default()
    }

    /// Starts (or replaces) the running effect on a device.
    pub fn play(&mut self, device: u64, effect: HapticEffect) {
        self.devices.insert(
            device,
            Motor {
                strong: clamp_motor(effect.strong),
                weak: clamp_motor(effect.weak),
                remaining: effect.duration,
                timed: effect.duration > 0.0,
            },
        );
    }

    pub fn stop(&mut self, device: u64) {
        self.devices.remove(&device);
    }

    pub fn stop_all(&mut self) {
        self.devices.clear();
    }

    /// Ages every running effect, dropping the ones that ran out.
    pub fn advance(&mut self, dt_seconds: f32) {
        if dt_seconds <= 0.0 {
            return;
        }
        for motor in self.devices.values_mut() {
            if motor.timed {
                motor.remaining = (motor.remaining - dt_seconds).max(0.0);
            }
        }
        self.devices
            .retain(|_, motor| !motor.timed || motor.remaining > 0.0);
    }

    /// Resolved motor strengths in `0..=1`; both zero once the effect expired.
    pub fn motors(&self, device: u64) -> (f32, f32) {
        match self.devices.get(&device) {
            Some(motor) => (motor.strong, motor.weak),
            None => (0.0, 0.0),
        }
    }

    pub fn is_playing(&self, device: u64) -> bool {
        self.devices.contains_key(&device)
    }

    pub fn active_devices(&self) -> impl Iterator<Item = u64> + '_ {
        self.devices.keys().copied()
    }

    /// Feeds every active device to a platform sink.
    pub fn apply_to(&self, sink: &mut dyn HapticSink) {
        for (&device, motor) in &self.devices {
            sink.apply(device, motor.strong, motor.weak);
        }
    }
}

fn clamp_motor(value: f32) -> f32 {
    if value.is_nan() {
        return 0.0;
    }
    value.clamp(0.0, 1.0)
}
