use glam::{Quat, Vec3};

/// Sensor axes a motion-capable controller reports.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SensorKind {
    #[default]
    None,
    Gyroscope,
    Accelerometer,
    GyroscopeAccelerometer,
}

impl SensorKind {
    pub fn has_gyroscope(self) -> bool {
        matches!(self, Self::Gyroscope | Self::GyroscopeAccelerometer)
    }

    pub fn has_accelerometer(self) -> bool {
        matches!(self, Self::Accelerometer | Self::GyroscopeAccelerometer)
    }
}

/// One reading from a device's motion sensors, in the device frame.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MotionSample {
    pub acceleration: Vec3,
    pub angular_velocity: Vec3,
}

/// Platform hook that feeds sensor readings for one device.
pub trait MotionSource {
    fn kind(&self) -> SensorKind;
    fn sample(&self) -> Option<MotionSample>;
}

/// Per-tick readout of a device's tracked motion.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MotionSnapshot {
    pub kind: SensorKind,
    /// Device-to-world orientation.
    pub orientation: Quat,
    pub angular_velocity: Vec3,
    pub acceleration: Vec3,
}

/// Gravity magnitude used by [`MotionTracker::gravity`], in m/s^2.
pub const GRAVITY: f32 = 9.806_65;

/// Integrates gyro readings into a device orientation, as console SDKs do for
/// motion controllers: `orientation` maps the device frame into the world,
/// device-frame angular velocity composes on its right, and the accelerometer
/// is reported in the device frame alongside it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionTracker {
    kind: SensorKind,
    orientation: Quat,
    angular_velocity: Vec3,
    acceleration: Vec3,
}

impl MotionTracker {
    pub fn new(kind: SensorKind) -> Self {
        Self {
            kind,
            orientation: Quat::IDENTITY,
            angular_velocity: Vec3::ZERO,
            acceleration: Vec3::ZERO,
        }
    }

    pub fn kind(&self) -> SensorKind {
        self.kind
    }

    /// Folds one sensor reading in. A missing axis leaves its previous value, so
    /// gyro-only and accel-only devices both integrate cleanly.
    pub fn integrate(&mut self, sample: MotionSample, dt_seconds: f32) {
        if dt_seconds > 0.0 && self.kind.has_gyroscope() {
            let spin = sample.angular_velocity * dt_seconds;
            if spin.length_squared() > 0.0 {
                self.orientation = (self.orientation * Quat::from_scaled_axis(spin)).normalize();
            }
            self.angular_velocity = sample.angular_velocity;
        }
        if self.kind.has_accelerometer() {
            self.acceleration = sample.acceleration;
        }
    }

    /// Resets orientation to identity and drops the held axes.
    pub fn recenter(&mut self) {
        self.orientation = Quat::IDENTITY;
        self.angular_velocity = Vec3::ZERO;
        self.acceleration = Vec3::ZERO;
    }

    pub fn orientation(&self) -> Quat {
        self.orientation
    }

    pub fn angular_velocity(&self) -> Vec3 {
        self.angular_velocity
    }

    pub fn acceleration(&self) -> Vec3 {
        self.acceleration
    }

    /// Gravity direction in the device frame: world down through the
    /// orientation's inverse, since `orientation` maps device into world.
    pub fn gravity(&self) -> Vec3 {
        self.orientation.inverse() * Vec3::new(0.0, -GRAVITY, 0.0)
    }

    pub fn snapshot(&self) -> MotionSnapshot {
        MotionSnapshot {
            kind: self.kind,
            orientation: self.orientation,
            angular_velocity: self.angular_velocity,
            acceleration: self.acceleration,
        }
    }
}
