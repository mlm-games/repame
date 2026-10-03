use std::collections::{HashMap, HashSet};

use repame_input::{HapticBus, HapticSink};
use repose_app::ReposeRuntime;
use repose_core::input::GamepadId;
use web_time::{Duration, Instant};

/// Length of each rumble upload. The backend sends a fresh force-feedback
/// effect per call, so uploads are spaced out rather than repeated per frame.
pub const RUMBLE_UPLOAD_MS: u32 = 1000;

/// Upload again once this much of [`RUMBLE_UPLOAD_MS`] has elapsed, which
/// keeps a held effect alive without restarting it every frame.
pub const RUMBLE_REFRESH: Duration = Duration::from_millis(800);

/// Strengths below this count as stopped.
const MOTOR_EPSILON: f32 = 0.001;

#[derive(Clone, Copy, Debug)]
struct Live {
    strong: f32,
    weak: f32,
    uploaded: Instant,
}

/// Routes [`HapticBus`] motors to `ReposeRuntime::request_rumble`. Uploads
/// only on change and on refresh, and stops devices the bus dropped, since
/// the bus stops mentioning a device rather than zeroing it.
#[derive(Default)]
pub struct RumbleBridge {
    live: HashMap<u64, Live>,
    queued: Vec<(GamepadId, f32, f32, u32)>,
}

impl RumbleBridge {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed every motor the bus is running, stopping what it dropped.
    pub fn sync(&mut self, bus: &HapticBus) {
        let now = Instant::now();
        self.sync_at(bus, now);
    }

    /// [`RumbleBridge::sync`] against an explicit clock.
    pub fn sync_at(&mut self, bus: &HapticBus, now: Instant) {
        let active: HashSet<u64> = bus.active_devices().collect();
        let stopped: Vec<u64> = self
            .live
            .keys()
            .copied()
            .filter(|device| !active.contains(device))
            .collect();
        for device in stopped {
            self.live.remove(&device);
            self.queued.push((GamepadId(device as u32), 0.0, 0.0, 0));
        }
        for device in active {
            let (strong, weak) = bus.motors(device);
            self.set(device, strong, weak, now);
        }
    }

    /// Hand the queued rumble to the runtime. Call once per frame after
    /// [`RumbleBridge::sync`]; the runner drains the runtime queue in turn.
    pub fn forward_to(&mut self, runtime: &mut ReposeRuntime) {
        for (id, strong, weak, duration_ms) in self.drain() {
            runtime.request_rumble(id, strong, weak, duration_ms);
        }
    }

    pub fn drain(&mut self) -> Vec<(GamepadId, f32, f32, u32)> {
        std::mem::take(&mut self.queued)
    }

    fn set(&mut self, device: u64, strong: f32, weak: f32, now: Instant) {
        let strong = clamp_motor(strong);
        let weak = clamp_motor(weak);
        let stopped = strong <= MOTOR_EPSILON && weak <= MOTOR_EPSILON;
        let stale = self
            .live
            .get(&device)
            .is_some_and(|live| now.duration_since(live.uploaded) >= RUMBLE_REFRESH);
        let changed = self
            .live
            .get(&device)
            .is_none_or(|live| live.strong != strong || live.weak != weak);
        if !stopped && !stale && !changed {
            return;
        }
        if stopped {
            if self.live.remove(&device).is_some() {
                self.queued.push((GamepadId(device as u32), 0.0, 0.0, 0));
            }
            return;
        }
        self.live.insert(
            device,
            Live {
                strong,
                weak,
                uploaded: now,
            },
        );
        self.queued
            .push((GamepadId(device as u32), strong, weak, RUMBLE_UPLOAD_MS));
    }
}

impl HapticSink for RumbleBridge {
    fn apply(&mut self, device: u64, strong: f32, weak: f32) {
        let now = Instant::now();
        self.set(device, strong, weak, now);
    }
}

fn clamp_motor(value: f32) -> f32 {
    if value.is_nan() {
        return 0.0;
    }
    value.clamp(0.0, 1.0)
}
