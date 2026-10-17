//! Orbit camera: yaw/pitch/dist around ground target. Y-up, right-handed.
//! Math only (glam). Shared by render uniform and picking.

use glam::camera::rh::proj::opengl::perspective as perspective_gl;
use glam::camera::rh::view::look_at_mat4;
use glam::{Mat4, Vec2, Vec3};

/// Maps OpenGL NDC depth [-1, 1] to wgpu [0, 1].
pub const OPENGL_TO_WGPU: Mat4 = Mat4::from_cols(
    glam::Vec4::new(1.0, 0.0, 0.0, 0.0),
    glam::Vec4::new(0.0, 1.0, 0.0, 0.0),
    glam::Vec4::new(0.0, 0.0, 0.5, 0.0),
    glam::Vec4::new(0.0, 0.0, 0.5, 1.0),
);

/// Near plane. Matches [`OrbitCamera::proj_matrix`].
pub const NEAR: f32 = 0.5;

/// Far plane for [`OrbitCamera::proj_matrix`].
pub const FAR: f32 = 2000.0;

/// Orbit camera around ground target. Angles in radians.
/// Game owns this in signals and copies it into [`Frame3d`](crate::Frame3d) per frame.
#[derive(Clone, Copy, Debug)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub dist: f32,
    pub fov_y_deg: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::new(0.0, 0.0, -2.0),
            yaw: -0.7,
            pitch: 0.85,
            dist: 110.0,
            fov_y_deg: 45.0,
        }
    }
}

impl OrbitCamera {
    pub fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        self.target + Vec3::new(cp * cy, sp, cp * sy) * self.dist
    }

    /// Unit vector the camera looks along, `target - eye` normalized.
    ///
    /// View depth is measured along this, not as the distance from the eye:
    /// fog and cascade splits are both depths, so an off-axis fragment at the
    /// same depth must read the same value as one under the eye.
    pub fn forward(&self) -> Vec3 {
        (self.target - self.eye()).normalize_or(-Vec3::Z)
    }

    pub fn view_matrix(&self) -> Mat4 {
        look_at_mat4(self.eye(), self.target, Vec3::Y)
    }

    pub fn proj_matrix(&self, aspect: f32) -> Mat4 {
        OPENGL_TO_WGPU * perspective_gl(self.fov_y_deg.to_radians(), aspect.max(0.01), NEAR, FAR)
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        self.proj_matrix(aspect) * self.view_matrix()
    }

    /// Drag deltas in dp. Horizontal orbits, vertical tilts.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.005;
        self.pitch = (self.pitch + dy * 0.005).clamp(0.12, 1.45);
    }

    /// Grab pan. Content follows cursor. Deltas in dp.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let fwd = (self.target - self.eye()).normalize();
        let right = fwd.cross(Vec3::Y).normalize();
        let up = right.cross(fwd).normalize();
        let s = self.dist / 800.0;
        self.target += -right * dx * s + up * dy * s;
        self.target.y = 0.0;
    }

    /// Multiplicative zoom, e.g. `1.0 + -wheel_y * 0.002`.
    pub fn zoom(&mut self, factor: f32) {
        self.dist = (self.dist * factor).clamp(15.0, 600.0);
    }

    /// World ray for viewport pixel. Units cancel in the division.
    pub fn screen_ray(&self, aspect: f32, viewport_px: Vec2, px: Vec2) -> (Vec3, Vec3) {
        let ndc = Vec2::new(
            (px.x / viewport_px.x) * 2.0 - 1.0,
            1.0 - (px.y / viewport_px.y) * 2.0,
        );
        let inv = self.view_proj(aspect).inverse();
        let p0 = inv.project_point3(Vec3::new(ndc.x, ndc.y, 0.0));
        let p1 = inv.project_point3(Vec3::new(ndc.x, ndc.y, 1.0));
        (p0, (p1 - p0).normalize())
    }

    /// Ground (y = 0) point under pixel, if ray hits.
    pub fn ground_point(&self, aspect: f32, viewport_px: Vec2, px: Vec2) -> Option<Vec2> {
        let (origin, dir) = self.screen_ray(aspect, viewport_px, px);
        if dir.y > -1e-6 {
            return None;
        }
        let t = -origin.y / dir.y;
        if t < 0.0 {
            return None;
        }
        Some(Vec2::new(origin.x + dir.x * t, origin.z + dir.z * t))
    }
}

/// A value the smoothing helpers can interpolate: `f32` and `Vec3` here.
/// Scale/add/subtract is all a critically damped spring needs.
pub trait DampValue: Copy {
    fn damp_zero() -> Self;
    fn damp_sub(self, other: Self) -> Self;
    fn damp_add(self, other: Self) -> Self;
    fn damp_scale(self, scale: f32) -> Self;
}

impl DampValue for f32 {
    fn damp_zero() -> Self {
        0.0
    }

    fn damp_sub(self, other: Self) -> Self {
        self - other
    }

    fn damp_add(self, other: Self) -> Self {
        self + other
    }

    fn damp_scale(self, scale: f32) -> Self {
        self * scale
    }
}

impl DampValue for Vec3 {
    fn damp_zero() -> Self {
        Vec3::ZERO
    }

    fn damp_sub(self, other: Self) -> Self {
        self - other
    }

    fn damp_add(self, other: Self) -> Self {
        self + other
    }

    fn damp_scale(self, scale: f32) -> Self {
        self * scale
    }
}

/// Frame-rate-independent exponential smoothing: the fraction of the gap to
/// `target` closed this step for a given decay `rate` (1/seconds). Unlike a
/// raw `lerp(current, target, k)` this is exact for any `dt`, so behavior does
/// not change with frame rate.
pub fn exp_decay_alpha(rate: f32, dt: f32) -> f32 {
    if !rate.is_finite() || !dt.is_finite() || rate <= 0.0 || dt <= 0.0 {
        return 0.0;
    }
    (1.0 - (-rate * dt).exp()).clamp(0.0, 1.0)
}

/// Exponential smoothing step toward `target`. Monotonic and never overshoots,
/// but only asymptotic: prefer [`Smoothed`] when a settle time must be met.
pub fn exp_decay<T: DampValue>(current: T, target: T, rate: f32, dt: f32) -> T {
    current.damp_add(
        target
            .damp_sub(current)
            .damp_scale(exp_decay_alpha(rate, dt)),
    )
}

/// Critically damped spring smoothing (Game Programming Gems "smooth damp").
///
/// Reaches the target in about `smooth_time` seconds with no overshoot, and is
/// frame-rate independent. `max_step` clamps a single integration step so a
/// long hitch cannot blow the spring up.
///
/// This is the smoothing `OrbitCamera` drives use; a plain lerp would overshoot
/// under a large step and is deliberately not offered.
#[derive(Clone, Copy, Debug)]
pub struct Smoothed<T: DampValue> {
    /// Approximate time to settle, in seconds. Clamped to `[1e-4, 10]`.
    pub smooth_time: f32,
    /// Largest `dt` integrated in one step, in seconds.
    pub max_step: f32,
    /// Last value the spring settled on.
    pub value: T,
    velocity: T,
}

impl<T: DampValue> Smoothed<T> {
    /// A spring holding `value`, settling in `smooth_time` seconds.
    pub fn new(value: T, smooth_time: f32) -> Self {
        Self {
            smooth_time,
            max_step: 0.05,
            value,
            velocity: T::damp_zero(),
        }
    }

    /// Drops the spring onto `value` and stops it (no easing).
    pub fn snap(&mut self, value: T) {
        self.value = value;
        self.velocity = T::damp_zero();
    }

    /// Advances one step toward `target` and returns the new value.
    pub fn step(&mut self, target: T, dt: f32) -> T {
        if !dt.is_finite() || dt <= 0.0 {
            return self.value;
        }
        let dt = dt.min(self.max_step.max(1e-4));
        let omega = 2.0 / self.smooth_time.clamp(1e-4, 10.0);
        let x = omega * dt;
        let exp = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x);
        let change = self.value.damp_sub(target);
        let temp = self.velocity.damp_add(change.damp_scale(omega)).damp_scale(dt);
        self.velocity = self
            .velocity
            .damp_sub(temp.damp_scale(omega))
            .damp_scale(exp);
        self.value = target.damp_add(change.damp_add(temp).damp_scale(exp));
        self.value
    }
}

/// Smooths the orbit parameters independently, so pan (target), orbit
/// (yaw/pitch) and zoom (distance) each settle on their own time constant.
///
/// Yaw eases the short way around a full turn, so a camera crossing the
/// `-pi`/`pi` seam does not spin the long way.
#[derive(Clone, Copy, Debug)]
pub struct SmoothedCamera {
    pub target: Smoothed<Vec3>,
    pub yaw: Smoothed<f32>,
    pub pitch: Smoothed<f32>,
    pub dist: Smoothed<f32>,
    pub fov: Smoothed<f32>,
}

impl SmoothedCamera {
    /// Springs seeded on `cam`. `smooth_time` seconds; `0` snaps every field.
    pub fn from_camera(cam: &OrbitCamera, smooth_time: f32) -> Self {
        Self {
            target: Smoothed::new(cam.target, smooth_time),
            yaw: Smoothed::new(cam.yaw, smooth_time),
            pitch: Smoothed::new(cam.pitch, smooth_time),
            dist: Smoothed::new(cam.dist, smooth_time),
            fov: Smoothed::new(cam.fov_y_deg, smooth_time),
        }
    }

    /// Drops every spring onto `cam`.
    pub fn snap_to(&mut self, cam: &OrbitCamera) {
        self.target.snap(cam.target);
        self.yaw.snap(cam.yaw);
        self.pitch.snap(cam.pitch);
        self.dist.snap(cam.dist);
        self.fov.snap(cam.fov_y_deg);
    }

    /// Advances every spring toward `goal` and returns the smoothed camera.
    pub fn step(&mut self, goal: &OrbitCamera, dt: f32) -> OrbitCamera {
        // Nearest-angle yaw so the short way around is taken.
        let yaw_goal = self.yaw.value + wrap_pi(goal.yaw - self.yaw.value);
        OrbitCamera {
            target: self.target.step(goal.target, dt),
            yaw: self.yaw.step(yaw_goal, dt),
            pitch: self.pitch.step(goal.pitch, dt),
            dist: self.dist.step(goal.dist, dt),
            fov_y_deg: self.fov.step(goal.fov_y_deg, dt),
        }
    }
}

/// Wraps an angle into `-pi..pi`.
pub fn wrap_pi(angle: f32) -> f32 {
    let two_pi = std::f32::consts::TAU;
    let wrapped = (angle + std::f32::consts::PI).rem_euclid(two_pi) - std::f32::consts::PI;
    if wrapped <= -std::f32::consts::PI {
        wrapped + two_pi
    } else {
        wrapped
    }
}

/// Scene query a [`SpringArm`] uses to keep the camera out of geometry.
///
/// Implement it over whatever the game already has (a physics ray, a voxel
/// walk, a mesh pick); the arm only needs the distance to the first surface.
pub trait CameraCollision {
    /// Distance along the unit `dir` from `origin` to the first surface within
    /// `max_dist`, or `None` for a clear line.
    fn cast_ray(&self, origin: Vec3, dir: Vec3, max_dist: f32) -> Option<f32>;
}

/// A world with nothing to collide against: the arm stays at full length.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoCollision;

impl CameraCollision for NoCollision {
    fn cast_ray(&self, _origin: Vec3, _dir: Vec3, _max_dist: f32) -> Option<f32> {
        None
    }
}

impl<F: Fn(Vec3, Vec3, f32) -> Option<f32>> CameraCollision for F {
    fn cast_ray(&self, origin: Vec3, dir: Vec3, max_dist: f32) -> Option<f32> {
        self(origin, dir, max_dist)
    }
}

/// A set of axis-aligned boxes the arm ray-tests against (slab method). Pair
/// it with the game's static bounds when no physics query exists.
#[derive(Clone, Debug, Default)]
pub struct AabbColliders {
    /// `(min, max)` corners, normalized on push.
    pub boxes: Vec<(Vec3, Vec3)>,
}

impl AabbColliders {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a box, normalizing the corners and ignoring non-finite ones.
    pub fn push(&mut self, min: Vec3, max: Vec3) {
        if !min.is_finite() || !max.is_finite() {
            return;
        }
        let lo = min.min(max);
        let hi = min.max(max);
        if (hi - lo).length_squared() <= 0.0 {
            return;
        }
        self.boxes.push((lo, hi));
    }

    pub fn is_empty(&self) -> bool {
        self.boxes.is_empty()
    }

    /// Kills the colliders a box set no longer needs.
    pub fn clear(&mut self) {
        self.boxes.clear();
    }
}

/// Slab ray/box entry distance, for one box. `None` when the ray misses or
/// hits beyond `max_dist`; `Some(0)` when `origin` starts inside.
pub fn ray_aabb_entry(origin: Vec3, dir: Vec3, min: Vec3, max: Vec3, max_dist: f32) -> Option<f32> {
    let mut tmin = 0.0f32;
    let mut tmax = max_dist;
    for axis in 0..3 {
        let o = origin[axis];
        let d = dir[axis];
        let lo = min[axis];
        let hi = max[axis];
        if d.abs() < 1e-8 {
            if o < lo || o > hi {
                return None;
            }
            continue;
        }
        let inv = 1.0 / d;
        let mut t1 = (lo - o) * inv;
        let mut t2 = (hi - o) * inv;
        if t1 > t2 {
            std::mem::swap(&mut t1, &mut t2);
        }
        tmin = tmin.max(t1);
        tmax = tmax.min(t2);
        if tmin > tmax {
            return None;
        }
    }
    Some(tmin.max(0.0))
}

impl CameraCollision for AabbColliders {
    fn cast_ray(&self, origin: Vec3, dir: Vec3, max_dist: f32) -> Option<f32> {
        let mut nearest: Option<f32> = None;
        for (min, max) in &self.boxes {
            if let Some(t) = ray_aabb_entry(origin, dir, *min, *max, max_dist)
                && nearest.is_none_or(|n| t < n)
            {
                nearest = Some(t);
            }
        }
        nearest
    }
}

/// A spring arm: an offset held `length` world units behind an anchor along
/// the look direction, shortened when geometry intrudes.
///
/// Godot's `SpringArm3D` casts from its origin along its forward axis and
/// parks its children at the hit; this keeps the same shape with an explicit
/// anchor and direction so it works for any camera rig. `margin` pulls the
/// camera in slightly past the contact point so near-plane clipping never
/// shows the surface.
#[derive(Clone, Copy, Debug)]
pub struct SpringArm {
    /// Rest distance from the anchor, in world units.
    pub length: f32,
    /// Extra pull-in past a contact, in world units.
    pub margin: f32,
    /// Camera offset added to the resolved position (e.g. a shoulder offset).
    pub offset: Vec3,
}

impl Default for SpringArm {
    fn default() -> Self {
        Self {
            length: 5.0,
            margin: 0.2,
            offset: Vec3::ZERO,
        }
    }
}

/// Where a [`SpringArm`] resolved to this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpringArmState {
    /// World position the camera should sit at.
    pub position: Vec3,
    /// Distance from the anchor actually used (<= the arm length).
    pub distance: f32,
    /// True when a contact shortened the arm.
    pub blocked: bool,
}

impl SpringArm {
    /// Resolves the arm for a unit `dir` from `anchor`, ignoring smoothing.
    pub fn resolve(
        &self,
        anchor: Vec3,
        dir: Vec3,
        colliders: &impl CameraCollision,
    ) -> SpringArmState {
        let d = dir.normalize_or_zero();
        let length = if self.length.is_finite() {
            self.length.max(0.0)
        } else {
            0.0
        };
        if d == Vec3::ZERO || length == 0.0 {
            return SpringArmState {
                position: anchor + self.offset,
                distance: 0.0,
                blocked: false,
            };
        }
        let blocked = colliders.cast_ray(anchor, d, length);
        let distance = blocked
            .map(|t| (t - self.margin.max(0.0)).clamp(0.0, length))
            .unwrap_or(length);
        SpringArmState {
            position: anchor + d * distance + self.offset,
            distance,
            blocked: blocked.is_some(),
        }
    }

    /// Resolves with rate-limited length: an intrusion shortens the arm
    /// instantly (never clipping a wall mid-lerp), while recovery eases in over
    /// `settle` seconds through `length_spring`.
    pub fn resolve_smoothed(
        &self,
        anchor: Vec3,
        dir: Vec3,
        colliders: &impl CameraCollision,
        length_spring: &mut Smoothed<f32>,
        dt: f32,
    ) -> SpringArmState {
        let raw = self.resolve(anchor, dir, colliders);
        if raw.distance <= length_spring.value && raw.blocked {
            // Instant shorten: a smoothed pull-in would let the wall through.
            length_spring.snap(raw.distance);
        } else {
            length_spring.step(raw.distance, dt);
        }
        let d = dir.normalize_or_zero();
        SpringArmState {
            position: anchor + d * length_spring.value + self.offset,
            distance: length_spring.value,
            blocked: raw.blocked,
        }
    }
}

/// Exposure and depth-of-field attributes attached to a camera, mirroring
/// Godot's `CameraAttributes` split from `Camera3D` itself.
///
/// Only exposure is plumbed to the tonemap here: [`CameraAttributes::apply_to`]
/// writes the resolved multiplier into [`SceneLight::exposure`], which is the
/// value [`repame_tonemap`](crate::render) already reads. DOF rides
/// [`crate::postfx::DofSettings`] into the post chain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraAttributes {
    /// Linear exposure multiplier, `1.0` neutral. Folded with `exposure_ev`.
    pub exposure: f32,
    /// Additional exposure in stops (EV), `0.0` neutral.
    pub exposure_ev: f32,
    /// Depth-of-field parameters, `None` renders everything sharp.
    pub dof: Option<crate::postfx::DofSettings>,
}

impl Default for CameraAttributes {
    fn default() -> Self {
        Self {
            exposure: 1.0,
            exposure_ev: 0.0,
            dof: None,
        }
    }
}

impl CameraAttributes {
    /// A neutral attribute set: exposure 1.0, no EV lift, no DOF.
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolved linear exposure: `exposure * 2^exposure_ev`, clamped to a
    /// positive finite range.
    pub fn exposure_value(&self) -> f32 {
        let base = if self.exposure.is_finite() {
            self.exposure
        } else {
            1.0
        };
        let ev = if self.exposure_ev.is_finite() {
            self.exposure_ev.clamp(-16.0, 16.0)
        } else {
            0.0
        };
        (base * 2.0f32.powf(ev)).clamp(1e-4, 1e4)
    }

    /// Writes the resolved exposure into the frame light.
    pub fn apply_to(&self, light: &mut crate::render::SceneLight) {
        light.exposure = self.exposure_value();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aspect() -> f32 {
        16.0 / 9.0
    }

    #[test]
    fn default_camera_sees_origin() {
        let cam = OrbitCamera::default();
        let vp = cam.view_proj(aspect());
        let clip = vp * glam::Vec4::new(0.0, 2.0, 0.0, 1.0);
        let ndc = Vec3::new(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w);
        assert!(clip.w > 0.0, "origin must be in front of camera");
        assert!(
            ndc.x.abs() < 1.0 && ndc.y.abs() < 1.0,
            "origin on screen: {ndc:?}"
        );
        assert!((0.0..=1.0).contains(&ndc.z), "depth remapped: {}", ndc.z);
    }

    #[test]
    fn orbit_zoom_pan_keep_camera_valid() {
        let mut cam = OrbitCamera::default();
        cam.orbit(400.0, -300.0);
        cam.zoom(0.05);
        cam.zoom(50.0);
        cam.pan(1000.0, -1000.0);
        assert!((15.0..=600.0).contains(&cam.dist));
        assert!((0.12..=1.45).contains(&cam.pitch));
        assert_eq!(cam.target.y, 0.0);
        assert!(cam.eye().y > 0.0);
    }

    #[test]
    fn ground_point_at_viewport_center_is_near_target() {
        let cam = OrbitCamera::default();
        let vp = Vec2::new(1600.0, 900.0);
        let g = cam
            .ground_point(aspect(), vp, Vec2::new(800.0, 450.0))
            .expect("center ray hits ground");
        let t = Vec2::new(cam.target.x, cam.target.z);
        assert!(
            (g - t).length() < cam.dist * 0.75,
            "center ~= target: {g:?}"
        );
    }

    #[test]
    fn screen_ray_through_center_hits_near_target() {
        // Close-up camera pins the unproject math end to end.
        let cam = OrbitCamera {
            dist: 30.0,
            pitch: 0.9,
            ..OrbitCamera::default()
        };
        let vp = Vec2::new(800.0, 600.0);
        let a = vp.x / vp.y;
        let g = cam
            .ground_point(a, vp, Vec2::new(400.0, 300.0))
            .expect("center ray hits ground");
        let t = Vec2::new(cam.target.x, cam.target.z);
        assert!((g - t).length() < 8.0, "center ~= target: {g:?} vs {t:?}");
    }

    /// The spring reaches the target without ever crossing it, at both 30 and
    /// 60 Hz: the no-overshoot contract a plain lerp cannot keep for large dt.
    #[test]
    fn critically_damped_step_response_has_no_overshoot() {
        for dt in [1.0 / 60.0, 1.0 / 30.0] {
            let mut s = Smoothed::new(0.0f32, 0.2);
            let mut peak: f32 = 0.0;
            for _ in 0..240 {
                let v = s.step(1.0, dt);
                peak = peak.max(v);
            }
            assert!(peak <= 1.0 + 1e-4, "overshoot at dt {dt}: {peak}");
            assert!(
                (s.value - 1.0).abs() < 1e-3,
                "settled at dt {dt}: {}",
                s.value
            );
        }
    }

    /// Frame-rate independence: the same 4 s of travel at 30 and 60 Hz lands
    /// within a thousandth, so a camera does not move differently per refresh.
    #[test]
    fn damping_is_frame_rate_independent() {
        let run = |dt: f32| {
            let mut s = Smoothed::new(0.0f32, 0.35);
            let steps = (4.0 / dt).round() as usize;
            for _ in 0..steps {
                s.step(1.0, dt);
            }
            s.value
        };
        let a = run(1.0 / 30.0);
        let b = run(1.0 / 60.0);
        assert!((a - b).abs() < 1e-3, "30Hz {a} vs 60Hz {b}");
    }

    /// Exponential smoothing never overshoots a step either, and closes a
    /// fixed fraction of the gap per second.
    #[test]
    fn exp_decay_is_monotonic_and_frame_rate_independent() {
        let mut a = 0.0f32;
        let mut b = 0.0f32;
        for _ in 0..120 {
            a = exp_decay(a, 1.0, 8.0, 1.0 / 60.0);
        }
        for _ in 0..240 {
            b = exp_decay(b, 1.0, 8.0, 1.0 / 120.0);
        }
        assert!(a <= 1.0 && b <= 1.0);
        assert!((a - b).abs() < 5e-3, "{a} vs {b}");
    }

    /// Yaw eases the short way around: a target just past +pi from -pi does
    /// not spin the camera the long way.
    #[test]
    fn smoothed_camera_yaw_takes_the_short_way() {
        let from = OrbitCamera {
            yaw: -3.0,
            ..OrbitCamera::default()
        };
        let mut s = SmoothedCamera::from_camera(&from, 0.2);
        let goal = OrbitCamera {
            yaw: 3.0, // equivalent to -3 + (-0.28), i.e. short way
            ..OrbitCamera::default()
        };
        let mut cam = from;
        for _ in 0..120 {
            cam = s.step(&goal, 1.0 / 60.0);
        }
        assert!(
            (wrap_pi(cam.yaw - goal.yaw)).abs() < 0.02,
            "settled near the goal: {} vs {}",
            cam.yaw,
            goal.yaw
        );
        assert!(cam.yaw > from.yaw, "moved the short way (up), got {}", cam.yaw);
    }

    /// A spring arm holds its full length in open space and shortens to the
    /// contact minus the margin.
    #[test]
    fn spring_arm_shortens_at_a_contact() {
        let arm = SpringArm {
            length: 10.0,
            margin: 0.5,
            offset: Vec3::ZERO,
        };
        let open = arm.resolve(Vec3::ZERO, Vec3::NEG_Z, &NoCollision);
        assert_eq!(open.distance, 10.0);
        assert!(!open.blocked);
        assert_eq!(open.position, Vec3::new(0.0, 0.0, -10.0));
        let mut world = AabbColliders::new();
        // A wall at z = -6.
        world.push(Vec3::new(-5.0, -5.0, -7.0), Vec3::new(5.0, 5.0, -6.0));
        let hit = arm.resolve(Vec3::ZERO, Vec3::NEG_Z, &world);
        assert!(hit.blocked);
        assert!((hit.distance - 5.5).abs() < 1e-4, "6 - 0.5: {}", hit.distance);
    }

    /// Intrusions snap in instantly while recovery eases, so the camera never
    /// spends a frame inside the wall it just hit.
    #[test]
    fn spring_arm_snaps_in_and_eases_out() {
        let arm = SpringArm {
            length: 10.0,
            margin: 0.0,
            offset: Vec3::ZERO,
        };
        let mut world = AabbColliders::new();
        world.push(Vec3::new(-5.0, -5.0, -4.0), Vec3::new(5.0, 5.0, -3.0));
        let mut len = Smoothed::new(10.0f32, 0.2);
        let blocked = arm.resolve_smoothed(Vec3::ZERO, Vec3::NEG_Z, &world, &mut len, 1.0 / 60.0);
        assert!(blocked.blocked);
        assert!((blocked.distance - 4.0).abs() < 1e-4, "instant in: {}", blocked.distance);
        // Wall gone: the arm recovers, never overshooting the rest length.
        let open = NoCollision;
        let mut max = 0.0f32;
        for _ in 0..120 {
            let s = arm.resolve_smoothed(Vec3::ZERO, Vec3::NEG_Z, &open, &mut len, 1.0 / 60.0);
            max = max.max(s.distance);
        }
        assert!((len.value - 10.0).abs() < 1e-2, "recovered: {}", len.value);
        assert!(max <= 10.0 + 1e-3, "no overshoot: {max}");
    }

    /// Slab test: a hit inside the box returns its entry, a miss returns None,
    /// and an origin inside returns zero.
    #[test]
    fn ray_aabb_entry_is_the_slab_distance() {
        let min = Vec3::new(-1.0, -1.0, -1.0);
        let max = Vec3::splat(1.0);
        assert_eq!(
            ray_aabb_entry(Vec3::new(0.0, 0.0, -5.0), Vec3::Z, min, max, 100.0),
            Some(4.0)
        );
        assert_eq!(
            ray_aabb_entry(Vec3::new(0.0, 5.0, -5.0), Vec3::Z, min, max, 100.0),
            None
        );
        assert_eq!(
            ray_aabb_entry(Vec3::new(0.0, 0.0, 0.0), Vec3::Z, min, max, 100.0),
            Some(0.0)
        );
        assert_eq!(
            ray_aabb_entry(Vec3::new(0.0, 0.0, -5.0), Vec3::Z, min, max, 1.0),
            None,
            "beyond max_dist"
        );
    }

    /// Exposure resolves multiplier * 2^EV and lands in the tonemap uniform.
    #[test]
    fn camera_attributes_plumb_exposure_to_the_light() {
        let attrs = CameraAttributes {
            exposure: 2.0,
            exposure_ev: 3.0,
            dof: None,
        };
        assert!((attrs.exposure_value() - 16.0).abs() < 1e-4);
        let mut light = crate::render::SceneLight::default();
        attrs.apply_to(&mut light);
        assert!((light.exposure - 16.0).abs() < 1e-4);
        // NaN folds back to a neutral value rather than poisoning the uniform.
        let bad = CameraAttributes {
            exposure: f32::NAN,
            exposure_ev: f32::NAN,
            dof: None,
        };
        assert_eq!(bad.exposure_value(), 1.0);
    }
}
