use std::collections::HashMap;

use glam::Vec3;
use repame_input::{MotionSample, MotionSource, SensorKind};
use repose_core::input::SensorKind as ReposeSensorKind;
use repose_core::input::{GamepadEvent, GamepadId};
use repose_platform::sensor::{SensorBackend, SensorReading, create_backend};

const DEG_TO_RAD: f32 = std::f32::consts::PI / 180.0;
const G_TO_MS2: f32 = 9.806_65;

/// One pad's motion, ready for [`MotionTracker`](repame_input::MotionTracker).
/// Holds its own copy of the reading so it stays a plain value the sim can
/// carry between frames.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PadMotion {
    kind: SensorKind,
    sample: MotionSample,
    reported: bool,
}

impl PadMotion {
    /// Whether this pad has ever reported motion.
    pub fn reported(&self) -> bool {
        self.reported
    }
}

impl MotionSource for PadMotion {
    fn kind(&self) -> SensorKind {
        self.kind
    }

    fn sample(&self) -> Option<MotionSample> {
        self.reported.then_some(self.sample)
    }
}

/// Motion-sensor polling on top of the shared platform backend. Sensor nodes
/// are separate evdev nodes from the pad, so readings are routed back to pads
/// by name; feed [`MotionPoller::feed_gamepad`] the same events
/// [`GamepadPoller`](crate::GamepadPoller) drains.
pub struct MotionPoller {
    backend: Option<Box<dyn SensorBackend>>,
    pads: HashMap<u32, String>,
    gyroscope: HashMap<u32, [f32; 3]>,
    accelerometer: HashMap<u32, [f32; 3]>,
}

impl MotionPoller {
    pub fn new() -> Self {
        Self {
            backend: create_backend().map(|b| Box::new(b) as Box<dyn SensorBackend>),
            pads: HashMap::new(),
            gyroscope: HashMap::new(),
            accelerometer: HashMap::new(),
        }
    }

    /// Track pad names so sensor devices can be routed to them.
    pub fn feed_gamepad(&mut self, events: &[GamepadEvent]) {
        for event in events {
            match event {
                GamepadEvent::Connected { id, name } => {
                    self.pads.insert(id.0, name.clone());
                }
                GamepadEvent::Disconnected { id } => {
                    self.pads.remove(&id.0);
                    self.gyroscope.remove(&id.0);
                    self.accelerometer.remove(&id.0);
                }
                _ => {}
            }
        }
    }

    /// Drain sensor readings since the last call. Devices with no matching
    /// pad are dropped rather than invented.
    pub fn poll(&mut self) {
        let Some(backend) = &mut self.backend else {
            return;
        };
        let readings = backend.poll_sensors();
        self.feed_readings(readings);
    }

    /// Route readings to pads by device name, latest wins.
    pub fn feed_readings(&mut self, readings: Vec<SensorReading>) {
        for reading in readings {
            let Some(id) = self.resolve(&reading.device) else {
                log::debug!(
                    "motion: no pad named {:?}, dropping {:?}",
                    reading.device,
                    reading.sample.kind
                );
                continue;
            };
            match reading.sample.kind {
                ReposeSensorKind::Gyroscope => {
                    self.gyroscope.insert(id, reading.sample.data);
                }
                ReposeSensorKind::Accelerometer => {
                    self.accelerometer.insert(id, reading.sample.data);
                }
            }
        }
    }

    /// Current motion for `id`, in repame's units: angular velocity in
    /// radians per second and acceleration in m/s^2, both device-frame.
    pub fn source(&self, id: GamepadId) -> PadMotion {
        let gyroscope = self.gyroscope.get(&id.0).copied();
        let accelerometer = self.accelerometer.get(&id.0).copied();
        let kind = match (gyroscope.is_some(), accelerometer.is_some()) {
            (true, true) => SensorKind::GyroscopeAccelerometer,
            (true, false) => SensorKind::Gyroscope,
            (false, true) => SensorKind::Accelerometer,
            (false, false) => SensorKind::None,
        };
        let angular_velocity = gyroscope
            .map(|d| Vec3::new(d[0], d[1], d[2]) * DEG_TO_RAD)
            .unwrap_or(Vec3::ZERO);
        let acceleration = accelerometer
            .map(|d| Vec3::new(d[0], d[1], d[2]) * G_TO_MS2)
            .unwrap_or(Vec3::ZERO);
        PadMotion {
            kind,
            sample: MotionSample {
                acceleration,
                angular_velocity,
            },
            reported: kind != SensorKind::None,
        }
    }

    /// Pads that have reported motion, id-sorted.
    pub fn motion_ids(&self) -> Vec<GamepadId> {
        let mut ids: Vec<u32> = self
            .gyroscope
            .keys()
            .chain(self.accelerometer.keys())
            .copied()
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids.into_iter().map(GamepadId).collect()
    }

    fn resolve(&self, device: &str) -> Option<u32> {
        if let Some((id, _)) = self.pads.iter().find(|(_, name)| name.as_str() == device) {
            return Some(*id);
        }
        let needle = repose_core::input::normalize_device_name(device);
        let mut matches = self
            .pads
            .iter()
            .filter(|(_, name)| repose_core::input::normalize_device_name(name) == needle)
            .map(|(id, _)| *id);
        let first = matches.next()?;
        matches.next().is_none().then_some(first)
    }
}

impl Default for MotionPoller {
    fn default() -> Self {
        Self::new()
    }
}
