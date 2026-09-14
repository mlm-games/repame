//! CPU skinning behind [`MeshGroup`](super::mesh::MeshGroup).
use std::collections::HashMap;

use glam::{Mat4, Quat, Vec3, Vec4};

use super::mesh::MeshGroup;

/// One joint's local transform (glTF TRS, game-driven per frame).
#[derive(Clone, Copy, Debug)]
pub struct JointPose {
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

impl Default for JointPose {
    fn default() -> Self {
        Self {
            translation: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
        }
    }
}

impl JointPose {
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(
            Vec3::from(self.scale),
            Quat::from_array(self.rotation),
            Vec3::from(self.translation),
        )
    }

    /// Blend toward `other` by `weight` (0 = self, 1 = other): lerp for
    /// TRS, slerp for rotation. Used by crossfade layers and additive
    /// partial blends.
    pub fn blend(&self, other: &JointPose, weight: f32) -> JointPose {
        let w = weight.clamp(0.0, 1.0);
        let lerp3 = |a: [f32; 3], b: [f32; 3]| {
            [
                a[0] + (b[0] - a[0]) * w,
                a[1] + (b[1] - a[1]) * w,
                a[2] + (b[2] - a[2]) * w,
            ]
        };
        JointPose {
            translation: lerp3(self.translation, other.translation),
            rotation: Quat::from_array(self.rotation)
                .slerp(Quat::from_array(other.rotation), w)
                .to_array(),
            scale: lerp3(self.scale, other.scale),
        }
    }
}

/// Per-channel interpolation for a [`NodePose`] track, mirroring the glTF
/// sampler the channel was imported from. Stored parallel to the value
/// channels so [`NodePose::sample`] can mix STEP holds, LINEAR blends, and
/// CUBICSPLINE tangents key-by-key.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Interp {
    /// Linear blend (lerp for T/S, slerp for R).
    #[default]
    Linear,
    /// Hold the left key until the next time (no blending).
    Step,
    /// Hermite blend with per-key tangents (translation/scale; slerp with
    /// tangent-nudged endpoints for rotation).
    CubicSpline,
}

/// Node animation track: timestamped TRS keys sampled per frame.
/// Times are seconds, ascending; out-of-range samples clamp (hold first /
/// last — matches `AnimPlayer::Once` hold semantics in `repame-anim`).
#[derive(Clone, Debug, Default)]
pub struct NodePose {
    pub times: Vec<f32>,
    pub translations: Vec<[f32; 3]>,
    pub rotations: Vec<[f32; 4]>,
    pub scales: Vec<[f32; 3]>,
    /// Interpolation per key (translation channel).
    pub translation_interp: Vec<Interp>,
    /// Interpolation per key (rotation channel).
    pub rotation_interp: Vec<Interp>,
    /// Interpolation per key (scale channel).
    pub scale_interp: Vec<Interp>,
    /// Cubic-spline in/out tangents per key (translation channel, per
    /// second). Empty unless [`Interp::CubicSpline`] is used.
    pub translation_tangents: Vec<([f32; 3], [f32; 3])>,
    /// Cubic-spline in/out tangents per key (rotation channel, xyz of the
    /// tangent quaternions). Empty unless cubic.
    pub rotation_tangents: Vec<([f32; 4], [f32; 4])>,
    /// Cubic-spline in/out tangents per key (scale channel). Empty unless
    /// cubic.
    pub scale_tangents: Vec<([f32; 3], [f32; 3])>,
}

/// Hermite basis at `alpha` in `[0, 1]`: position and endpoint velocity
/// weights for a segment of duration `dt` (tangent inputs are per-second).
fn hermite(alpha: f32, dt: f32) -> (f32, f32, f32, f32) {
    let a2 = alpha * alpha;
    let a3 = a2 * alpha;
    let h00 = 2.0 * a3 - 3.0 * a2 + 1.0;
    let h10 = a3 - 2.0 * a2 + alpha;
    let h01 = -2.0 * a3 + 3.0 * a2;
    let h11 = a3 - a2;
    (h00, h10 * dt, h01, h11 * dt)
}

fn hermite3(
    a: [f32; 3],
    out_a: [f32; 3],
    b: [f32; 3],
    in_b: [f32; 3],
    h: (f32, f32, f32, f32),
) -> [f32; 3] {
    [
        h.0 * a[0] + h.1 * out_a[0] + h.2 * b[0] + h.3 * in_b[0],
        h.0 * a[1] + h.1 * out_a[1] + h.2 * b[1] + h.3 * in_b[1],
        h.0 * a[2] + h.1 * out_a[2] + h.2 * b[2] + h.3 * in_b[2],
    ]
}

impl NodePose {
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    pub fn duration(&self) -> f32 {
        self.times.last().copied().unwrap_or(0.0)
    }

    /// Sample at `t` seconds (clamped). Missing channels hold bind values
    /// (identity rotation, unit scale, zero translation). Per-key
    /// interpolation: STEP holds the left key, LINEAR lerps/slerps, and
    /// CUBICSPLINE hermite-blends with the stored tangents.
    pub fn sample(&self, t: f32) -> JointPose {
        if self.times.is_empty() {
            return JointPose::default();
        }
        let n = self.times.len();
        let mut i = 0;
        while i + 1 < n && self.times[i + 1] <= t {
            i += 1;
        }
        let j = (i + 1).min(n - 1);
        let dt = self.times[j] - self.times[i];
        let alpha = if j == i || dt <= 0.0 {
            0.0
        } else {
            ((t - self.times[i]) / dt).clamp(0.0, 1.0)
        };
        let ti = |v: &[Interp]| v.get(i).copied().unwrap_or(Interp::Linear);
        let lerp3 = |a: [f32; 3], b: [f32; 3]| {
            [
                a[0] + (b[0] - a[0]) * alpha,
                a[1] + (b[1] - a[1]) * alpha,
                a[2] + (b[2] - a[2]) * alpha,
            ]
        };
        let sample3 = |vals: &[[f32; 3]],
                       tang: &[([f32; 3], [f32; 3])],
                       interp: Interp,
                       fallback: [f32; 3]| {
            match (vals.get(i), vals.get(j)) {
                (Some(a), Some(b)) => match interp {
                    Interp::Step => *a,
                    Interp::Linear => lerp3(*a, *b),
                    Interp::CubicSpline => match (tang.get(i), tang.get(j)) {
                        (Some((_, out_a)), Some((in_b, _))) if dt > 0.0 => {
                            hermite3(*a, *out_a, *b, *in_b, hermite(alpha, dt))
                        }
                        _ => lerp3(*a, *b),
                    },
                },
                (Some(a), None) | (None, Some(a)) => *a,
                (None, None) => fallback,
            }
        };
        // Rotation blends on the quaternion sphere; cubic tangents nudge
        // the slerp endpoints (tangent xyz scaled by dt, w held).
        let sample_rot = || match (self.rotations.get(i), self.rotations.get(j)) {
            (Some(a), Some(b)) => match ti(&self.rotation_interp) {
                Interp::Step => *a,
                Interp::Linear => Quat::from_array(*a)
                    .slerp(Quat::from_array(*b), alpha)
                    .to_array(),
                Interp::CubicSpline => {
                    match (self.rotation_tangents.get(i), self.rotation_tangents.get(j)) {
                        (Some((_, out_a)), Some((in_b, _))) if dt > 0.0 => {
                            // Tangent-nudged slerp: offset each endpoint by its
                            // tangent (xyz scaled by dt/3, w held), normalize,
                            // then slerp. Exact for zero tangents.
                            let nudge = |q: [f32; 4], tan: [f32; 4]| {
                                let qv = Quat::from_array(q);
                                let off = Vec3::new(tan[0], tan[1], tan[2]) * (dt / 3.0);
                                let nudged =
                                    Vec4::new(qv.x + off.x, qv.y + off.y, qv.z + off.z, qv.w);
                                Quat::from_vec4(nudged).normalize().to_array()
                            };
                            Quat::from_array(nudge(*a, *out_a))
                                .slerp(Quat::from_array(nudge(*b, *in_b)), alpha)
                                .to_array()
                        }
                        _ => Quat::from_array(*a)
                            .slerp(Quat::from_array(*b), alpha)
                            .to_array(),
                    }
                }
            },
            (Some(a), None) | (None, Some(a)) => *a,
            (None, None) => [0.0, 0.0, 0.0, 1.0],
        };
        JointPose {
            translation: sample3(
                &self.translations,
                &self.translation_tangents,
                ti(&self.translation_interp),
                [0.0, 0.0, 0.0],
            ),
            rotation: sample_rot(),
            scale: sample3(
                &self.scales,
                &self.scale_tangents,
                ti(&self.scale_interp),
                [1.0, 1.0, 1.0],
            ),
        }
    }
}

/// One glTF animation: named node tracks sampled by [`NodePose::sample`].
/// Keys are node indices (map through [`SkinnedMesh::node_to_joint`] to
/// drive joint matrices; node hierarchy composes through [`Skeleton`]).
/// Morph `weights` targets (mesh index + target count) sample into
/// [`SkeletonPlayer::morph_weights`].
#[derive(Clone, Debug, Default)]
pub struct Animation {
    pub name: String,
    pub tracks: HashMap<usize, NodePose>,
    /// Morph-target weight tracks: (mesh index, target count, times, weights
    /// flattened per-key as `target_count` scalars). Sampled by the player
    /// into per-mesh weight vectors.
    pub morph_tracks: HashMap<usize, MorphTrack>,
}

impl Animation {
    pub fn duration(&self) -> f32 {
        let track_max = self
            .tracks
            .values()
            .map(|t| t.duration())
            .fold(0.0f32, f32::max);
        let morph_max = self
            .morph_tracks
            .values()
            .map(|t| t.duration())
            .fold(0.0f32, f32::max);
        track_max.max(morph_max)
    }

    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty() && self.morph_tracks.is_empty()
    }

    /// World-space joint matrices for `skeleton` at time `t` seconds:
    /// sample every track, fall back to bind locals for untracked nodes,
    /// compose down the tree. Output order matches
    /// [`SkinnedMesh::inverse_bind`] (skin joint order); joints for other
    /// skins in the file are identity (caller picks one skin per player).
    pub fn joint_matrices(&self, skeleton: &Skeleton, mesh: &SkinnedMesh, t: f32) -> Vec<Mat4> {
        let mut locals: HashMap<usize, JointPose> = HashMap::new();
        for (node, track) in &self.tracks {
            locals.insert(*node, track.sample(t));
        }
        let world = skeleton.world_matrices(&locals);
        mesh.inverse_bind
            .iter()
            .enumerate()
            .map(|(slot, _)| {
                let node = mesh.joint_nodes.get(slot).copied().unwrap_or(usize::MAX);
                world.get(&node).copied().unwrap_or(Mat4::IDENTITY)
            })
            .collect()
    }
}

/// Morph-target weight track for one mesh: `target_count` weights per key.
#[derive(Clone, Debug, Default)]
pub struct MorphTrack {
    /// Key times in seconds, ascending.
    pub times: Vec<f32>,
    /// Weights flattened per key (`times.len() * target_count`).
    pub weights: Vec<f32>,
    pub target_count: usize,
}

impl MorphTrack {
    pub fn duration(&self) -> f32 {
        self.times.last().copied().unwrap_or(0.0)
    }

    /// Sample weights at `t` seconds (clamped, linear blend).
    pub fn sample(&self, t: f32) -> Vec<f32> {
        if self.times.is_empty() || self.target_count == 0 {
            return vec![0.0; self.target_count];
        }
        let n = self.times.len();
        let mut i = 0;
        while i + 1 < n && self.times[i + 1] <= t {
            i += 1;
        }
        let j = (i + 1).min(n - 1);
        let alpha = if j == i || self.times[j] <= self.times[i] {
            0.0
        } else {
            ((t - self.times[i]) / (self.times[j] - self.times[i])).clamp(0.0, 1.0)
        };
        let at = |k: usize| {
            let base = k * self.target_count;
            &self.weights[base..base + self.target_count]
        };
        match (self.weights.len() >= (i + 1) * self.target_count, j == i) {
            (true, _) => at(i)
                .iter()
                .zip(at(j).iter())
                .map(|(a, b)| a + (b - a) * alpha)
                .collect(),
            _ => vec![0.0; self.target_count],
        }
    }
}

/// Scene-graph skeleton: parent links plus bind local TRS for every node
/// in the file. Built once by [`import_skeleton`]; [`Animation`] samples
/// local poses against it and composes world matrices down the tree, so a
/// shoulder rotation carries the whole arm (without this, joint matrices
/// are node-local and children never follow their parents).
#[derive(Clone, Debug, Default)]
pub struct Skeleton {
    /// Parent node index per node (`None` = scene root).
    pub parents: HashMap<usize, Option<usize>>,
    /// Bind local transform per node (from the node TRS/matrix at import).
    pub bind_local: HashMap<usize, Mat4>,
}

impl Skeleton {
    pub fn node_count(&self) -> usize {
        self.parents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parents.is_empty()
    }

    /// Compose world matrices for every node: `world[node]` is
    /// `world[parent] * local[node]`, where `local` is the animated pose
    /// when present and the bind local otherwise. Missing parents
    /// (untracked roots) compose from identity — same fallback as the bind
    /// pose.
    pub fn world_matrices(&self, locals: &HashMap<usize, JointPose>) -> HashMap<usize, Mat4> {
        let mut world: HashMap<usize, Mat4> = HashMap::new();
        let mut order: Vec<usize> = self.parents.keys().copied().collect();
        order.sort();
        // Depth-first via repeated passes: parents always resolve before
        // children (bounded by node count; cycles break at identity).
        for _ in 0..order.len() + 1 {
            let mut progressed = false;
            for node in &order {
                if world.contains_key(node) {
                    continue;
                }
                let parent_world = match self.parents.get(node).copied().flatten() {
                    Some(p) => match world.get(&p) {
                        Some(m) => *m,
                        None => continue,
                    },
                    None => Mat4::IDENTITY,
                };
                let local = locals
                    .get(node)
                    .map(|pose| pose.matrix())
                    .or_else(|| self.bind_local.get(node).copied())
                    .unwrap_or(Mat4::IDENTITY);
                world.insert(*node, parent_world * local);
                progressed = true;
            }
            if !progressed {
                break;
            }
        }
        world
    }
}

/// Whether a [`SkeletonPlayer`] keeps playing past the end.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SkeletonLoop {
    /// Wrap to the start and keep playing (walk cycles, idles).
    #[default]
    Loop,
    /// Hold the last frame and stop (one-shot attacks, deaths).
    Once,
    /// Bounce between the ends and keep playing (same fold as
    /// `repame-anim` PingPong).
    PingPong,
}

/// Outcome of one [`SkeletonPlayer::advance`] call.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SkeletalAdvance {
    /// True when the clock crossed into a new animation time (always true
    /// on a real advance — the mesh must re-bake; false only on no-ops).
    pub pose_changed: bool,
    /// Full laps (Loop) or bounces (PingPong) crossed, even across
    /// hitches, so per-lap effects never drop laps. Once: 1 on finish.
    pub wraps: u32,
}

impl SkeletalAdvance {
    pub fn ended(&self) -> bool {
        self.wraps > 0
    }
}

/// Playback clock over one [`Animation`]: `pos` seconds, signed
/// `speed_scale` (sign = direction, like `repame-anim` `AnimPlayer` — no
/// separate direction flag), Loop/Once/PingPong, `advance(dt)` with
/// wraps/ended. The game samples joint matrices + morph weights from the
/// player each frame and bakes [`MeshGroup`]s via [`SkinnedMesh::pose`]
/// and [`MorphSet::apply`].
#[derive(Clone, Debug)]
pub struct SkeletonPlayer {
    duration: f32,
    pos: f32,
    playing: bool,
    speed_scale: f32,
    loop_mode: SkeletonLoop,
    finished: bool,
}

impl SkeletonPlayer {
    /// New player over `anim` with the given loop behavior. Starts paused
    /// at `t = 0` with rate `1.0`.
    pub fn new(anim: &Animation, loop_mode: SkeletonLoop) -> Self {
        Self {
            duration: anim.duration().max(0.0),
            pos: 0.0,
            playing: false,
            speed_scale: 1.0,
            loop_mode,
            finished: false,
        }
    }

    pub fn play(&mut self) {
        if self.finished {
            self.pos = 0.0;
            self.finished = false;
        }
        self.playing = true;
    }

    /// Play in reverse: negative rate, jumping to the end first when
    /// stopped or finished.
    pub fn play_backwards(&mut self) {
        if (!self.playing || self.finished) && self.duration > 0.0 {
            self.pos = self.duration;
        }
        self.speed_scale = -self.speed_scale.abs();
        self.finished = false;
        self.playing = true;
    }

    pub fn pause(&mut self) {
        self.playing = false;
    }

    /// Stop and reset to `t = 0` (rate sign kept).
    pub fn stop(&mut self) {
        self.playing = false;
        self.pos = 0.0;
        self.finished = false;
    }

    /// Speed multiplier; negative plays in reverse, `0` freezes while
    /// "playing", non-finite treated as `0`.
    pub fn set_speed_scale(&mut self, s: f32) {
        self.speed_scale = if s.is_finite() { s } else { 0.0 };
    }

    pub fn speed_scale(&self) -> f32 {
        self.speed_scale
    }

    pub fn is_playing(&self) -> bool {
        self.playing
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Current animation time in seconds.
    pub fn time(&self) -> f32 {
        self.pos
    }

    pub fn duration(&self) -> f32 {
        self.duration
    }

    /// Jump to `t` seconds (clamped to the duration); clears `finished`.
    pub fn seek(&mut self, t: f32) {
        self.pos = t.clamp(0.0, self.duration);
        self.finished = false;
    }

    /// Advance the clock by `dt` seconds. No-op while paused, at rate `0`,
    /// on empty animations, or for non-positive/non-finite `dt`.
    pub fn advance(&mut self, dt: f32) -> SkeletalAdvance {
        if !self.playing || self.duration <= 0.0 {
            return SkeletalAdvance::default();
        }
        if !dt.is_finite() || dt <= 0.0 {
            return SkeletalAdvance::default();
        }
        let step = dt * self.speed_scale;
        if step == 0.0 {
            return SkeletalAdvance::default();
        }
        let pos_before = self.pos;
        match self.loop_mode {
            SkeletonLoop::Loop => {
                let n = self.duration;
                self.pos = (self.pos + step).rem_euclid(n);
                let wraps =
                    (pos_before.div_euclid(n) - (pos_before + step).div_euclid(n)).abs() as u32;
                SkeletalAdvance {
                    pose_changed: true,
                    wraps,
                }
            }
            SkeletonLoop::Once => {
                self.pos += step;
                if self.pos >= self.duration || self.pos < 0.0 {
                    self.pos = if step > 0.0 { self.duration } else { 0.0 };
                    self.playing = false;
                    self.finished = true;
                    SkeletalAdvance {
                        pose_changed: true,
                        wraps: 1,
                    }
                } else {
                    SkeletalAdvance {
                        pose_changed: true,
                        wraps: 0,
                    }
                }
            }
            SkeletonLoop::PingPong => {
                let period = 2.0 * self.duration;
                if period <= 0.0 {
                    return SkeletalAdvance::default();
                }
                let raw = self.pos + step;
                let wraps = (raw.div_euclid(period) - pos_before.div_euclid(period)).abs() as u32;
                self.pos = raw.rem_euclid(period);
                if self.pos > self.duration {
                    self.pos = period - self.pos;
                }
                SkeletalAdvance {
                    pose_changed: true,
                    wraps,
                }
            }
        }
    }

    /// Sampled time after the PingPong fold (Loop/Once: `pos` directly).
    pub fn sample_time(&self) -> f32 {
        match self.loop_mode {
            SkeletonLoop::PingPong => {
                let period = 2.0 * self.duration;
                if period <= 0.0 {
                    return 0.0;
                }
                let tri = self.pos.rem_euclid(period);
                if tri <= self.duration {
                    tri
                } else {
                    period - tri
                }
            }
            _ => self.pos,
        }
    }

    /// Joint matrices for `mesh` under `skeleton` at the current time.
    pub fn joint_matrices(
        &self,
        anim: &Animation,
        skeleton: &Skeleton,
        mesh: &SkinnedMesh,
    ) -> Vec<Mat4> {
        anim.joint_matrices(skeleton, mesh, self.sample_time())
    }

    /// Morph weights per mesh at the current time: (mesh index, weights).
    /// Meshes without a track are absent (caller holds bind = zeros).
    pub fn morph_weights(&self, anim: &Animation) -> HashMap<usize, Vec<f32>> {
        let t = self.sample_time();
        anim.morph_tracks
            .iter()
            .map(|(mesh, track)| (*mesh, track.sample(t)))
            .collect()
    }
}

/// One mesh's morph targets: position/normal deltas per target, parsed
/// once by [`import_morphs`]. [`apply`](MorphSet::apply) blends them onto
/// a baked [`MeshGroup`] with per-target weights — CPU-side, no GPU morph
/// path (few targets, few verts, per-frame blend).
#[derive(Clone, Debug, Default)]
pub struct MorphSet {
    /// Position deltas per target (`targets[k][i]` = delta for vertex `i`).
    pub position_deltas: Vec<Vec<[f32; 3]>>,
    /// Normal deltas per target (empty when the file omits them).
    pub normal_deltas: Vec<Vec<[f32; 3]>>,
}

impl MorphSet {
    pub fn target_count(&self) -> usize {
        self.position_deltas.len()
    }

    pub fn is_empty(&self) -> bool {
        self.position_deltas.is_empty()
    }

    /// Blend `weights[k]` of target `k` into `group` (in place). Weights
    /// clamp to `0..1`; mismatched-length targets are skipped, never a
    /// panic. Normals renormalize after the blend.
    pub fn apply(&self, group: &mut MeshGroup, weights: &[f32]) {
        if weights.iter().all(|w| *w == 0.0) {
            return;
        }
        let n = group.positions.len();
        for (k, deltas) in self.position_deltas.iter().enumerate() {
            let w = weights.get(k).copied().unwrap_or(0.0).clamp(0.0, 1.0);
            if w == 0.0 || deltas.len() != n {
                continue;
            }
            for (p, d) in group.positions.iter_mut().zip(deltas.iter()) {
                p[0] += d[0] * w;
                p[1] += d[1] * w;
                p[2] += d[2] * w;
            }
        }
        if group.normals.len() == n {
            for (k, deltas) in self.normal_deltas.iter().enumerate() {
                let w = weights.get(k).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                if w == 0.0 || deltas.len() != n {
                    continue;
                }
                for (normal, d) in group.normals.iter_mut().zip(deltas.iter()) {
                    normal[0] += d[0] * w;
                    normal[1] += d[1] * w;
                    normal[2] += d[2] * w;
                }
            }
            for normal in group.normals.iter_mut() {
                let v = Vec3::from(*normal);
                *normal = v.try_normalize().unwrap_or(Vec3::Y).into();
            }
        }
    }
}

/// Build the scene-graph skeleton: parent links + bind locals for every
/// node in every scene of the file. Bind locals come from the node
/// TRS/matrix exactly as [`gltf`](super::gltf) composes them for static
/// import, so animated and static paths agree on the bind pose.
pub fn import_skeleton(bytes: &[u8]) -> Result<Skeleton, gltf::Error> {
    let (doc, _, _) = gltf::import_slice(bytes)?;
    let mut parents: HashMap<usize, Option<usize>> = HashMap::new();
    let mut bind_local: HashMap<usize, Mat4> = HashMap::new();
    for scene in doc.scenes() {
        for node in scene.nodes() {
            collect_skeleton(&node, None, &mut parents, &mut bind_local);
        }
    }
    Ok(Skeleton {
        parents,
        bind_local,
    })
}

fn collect_skeleton(
    node: &gltf::Node<'_>,
    parent: Option<usize>,
    parents: &mut HashMap<usize, Option<usize>>,
    bind_local: &mut HashMap<usize, Mat4>,
) {
    use gltf::scene::Transform as T;
    let local = match node.transform() {
        T::Matrix { matrix } => Mat4::from_cols_array_2d(&matrix),
        T::Decomposed {
            translation,
            rotation,
            scale,
        } => Mat4::from_scale_rotation_translation(
            Vec3::from(scale),
            Quat::from_array(rotation),
            Vec3::from(translation),
        ),
    };
    parents.insert(node.index(), parent);
    bind_local.insert(node.index(), local);
    for child in node.children() {
        collect_skeleton(&child, Some(node.index()), parents, bind_local);
    }
}

/// Bind-pose skinned mesh parsed once from a glTF primitive (+ skin).
/// `joints`/`weights` are per-vertex `[u16; 4]`/`[f32; 4]` (normalized on
/// parse); `inverse_bind` is per-joint `Mat4` (identity when the file omits
/// them, i.e. pre-applied). `node_to_joint` maps glTF node indices to joint
/// slots so animation tracks (keyed by node) drive the right matrices;
/// `joint_nodes` is the inverse (joint slot -> node index) for hierarchy
/// composition in [`Animation::joint_matrices`].
///
/// `transparent`/`alpha`/`alpha_cutoff` carry the material's alpha mode
/// (see [`alpha_mode`](super::gltf::alpha_mode)): [`pose`](SkinnedMesh::pose)
/// copies them into every baked group, so animated BLEND/MASK materials
/// fade and cut out exactly like static ones.
#[derive(Clone, Debug, Default)]
pub struct SkinnedMesh {
    pub name: String,
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub colors: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
    pub joints: Vec<[u16; 4]>,
    pub weights: Vec<[f32; 4]>,
    pub inverse_bind: Vec<Mat4>,
    pub node_to_joint: HashMap<usize, usize>,
    pub joint_nodes: Vec<usize>,
    pub texture_page: u32,
    /// Document image index behind the base-color texture (`None` =
    /// untextured material). Games resolve it through the textured import's
    /// page map and call [`assign_page`](SkinnedMesh::assign_page) once —
    /// the baked pose copies `texture_page` per frame, so the bind mesh is
    /// the single place to set it.
    pub base_image: Option<usize>,
    pub pick_id: u32,
    /// Alpha-blend pass (glTF `BLEND`).
    pub transparent: bool,
    /// Base-color alpha (group multiplier; glTF `BLEND` fades, `MASK`
    /// gates on it when the texel is opaque).
    pub alpha: f32,
    /// Alpha cutoff (glTF `MASK` threshold; 0.0 = keep everything).
    pub alpha_cutoff: f32,
    /// Surface material (glTF metallic/roughness/emissive factors).
    pub material: super::mesh::Material,
    pub depth_test: bool,
}

impl SkinnedMesh {
    pub fn joint_count(&self) -> usize {
        self.inverse_bind.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty() || self.positions.is_empty()
    }

    /// Bake `joint_matrices` (world-space per joint, same order as
    /// `inverse_bind`) into a world-space [`MeshGroup`]. Normals rotate by
    /// the blended matrix's 3x3 and renormalize. Unweighted verts (all-zero
    /// weights) hold bind pose.
    ///
    /// Transparency/alpha/cutoff carry through from the bind mesh (usually
    /// identity/opaque — see [`SkinnedMesh`] defaults); the group is
    /// otherwise rebuilt per frame from the pose.
    pub fn pose(&self, joint_matrices: &[Mat4]) -> MeshGroup {
        let mut group = MeshGroup {
            texture_page: self.texture_page,
            pick_id: self.pick_id,
            depth_test: self.depth_test,
            transparent: self.transparent,
            alpha: self.alpha,
            alpha_cutoff: self.alpha_cutoff,
            material: self.material,
            ..Default::default()
        };
        if self.is_empty() {
            return group;
        }
        let n = self.positions.len();
        group.positions.reserve_exact(n);
        group.colors.extend_from_slice(&self.colors);
        let has_normals = self.normals.len() == n;
        if has_normals {
            group.normals.reserve_exact(n);
        }
        if self.uvs.len() == n {
            group.uvs.extend_from_slice(&self.uvs);
        }
        group.indices.extend_from_slice(&self.indices);

        let skin_count = joint_matrices.len().min(self.inverse_bind.len());
        for i in 0..n {
            let pos = Vec3::from(self.positions[i]);
            let joints = self.joints.get(i).copied().unwrap_or([0; 4]);
            let weights = self.weights.get(i).copied().unwrap_or([0.0; 4]);
            let wsum: f32 = weights.iter().sum();
            if wsum <= 1e-8 || skin_count == 0 {
                group.positions.push(self.positions[i]);
                if has_normals {
                    group.normals.push(self.normals[i]);
                }
                continue;
            }
            let mut skinned = Vec3::ZERO;
            let mut nrm = Vec3::ZERO;
            let has_n = has_normals;
            let bind_n = if has_n {
                Vec3::from(self.normals[i])
            } else {
                Vec3::Y
            };
            for k in 0..4 {
                let w = weights[k] / wsum;
                if w <= 0.0 {
                    continue;
                }
                let j = (joints[k] as usize).min(skin_count.saturating_sub(1));
                let m = joint_matrices[j] * self.inverse_bind[j];
                skinned += m.transform_point3(pos) * w;
                if has_n {
                    nrm += m.transform_vector3(bind_n) * w;
                }
            }
            group.positions.push(skinned.into());
            if has_n {
                group
                    .normals
                    .push(nrm.try_normalize().unwrap_or(Vec3::Y).into());
            }
        }
        group
    }

    /// Assign a packed texture page to the bind mesh: sets `texture_page`
    /// and scales uvs into the placed rect (`placed_w`/`placed_h` inside a
    /// `layer_size` layer at the origin). The baked pose copies both per
    /// frame, so call once after import — never per frame. Uv-less meshes
    /// only record the page (nothing to scale); callers with no pixels for
    /// this mesh should clear `uvs` instead (see
    /// [`import_slice_textured`](super::gltf::import_slice_textured)).
    pub fn assign_page(&mut self, page: u32, placed_w: u32, placed_h: u32, layer_size: u32) {
        self.texture_page = page;
        if self.uvs.is_empty() || layer_size == 0 {
            return;
        }
        let sx = placed_w as f32 / layer_size as f32;
        let sy = placed_h as f32 / layer_size as f32;
        for uv in &mut self.uvs {
            uv[0] *= sx;
            uv[1] *= sy;
        }
    }
}

/// Parse every skinned primitive in `bytes` into [`SkinnedMesh`]s (bind
/// pose). Unskinned primitives are skipped (the [`gltf`](super::gltf)
/// importer owns those). Joint slots follow skin joint order; node indices
/// map through `node_to_joint` for animation tracks.
pub fn import_skinned(bytes: &[u8]) -> Result<Vec<SkinnedMesh>, gltf::Error> {
    let (doc, buffers, _) = gltf::import_slice(bytes)?;
    let mut out = Vec::new();
    for mesh in doc.meshes() {
        for prim in mesh.primitives() {
            let reader = prim.reader(|b| buffers.get(b.index()).map(|d| d.0.as_slice()));
            let joints: Option<Vec<[u16; 4]>> = reader.read_joints(0).map(|it| {
                use gltf::mesh::util::ReadJoints as J;
                match it {
                    J::U8(iter) => iter
                        .map(|j| [j[0] as u16, j[1] as u16, j[2] as u16, j[3] as u16])
                        .collect(),
                    J::U16(iter) => iter.collect(),
                }
            });
            let weights: Option<Vec<[f32; 4]>> = reader.read_weights(0).map(|it| {
                use gltf::mesh::util::ReadWeights as W;
                match it {
                    W::U8(iter) => iter
                        .map(|w| {
                            [
                                w[0] as f32 / 255.0,
                                w[1] as f32 / 255.0,
                                w[2] as f32 / 255.0,
                                w[3] as f32 / 255.0,
                            ]
                        })
                        .collect(),
                    W::U16(iter) => iter
                        .map(|w| {
                            [
                                w[0] as f32 / 65535.0,
                                w[1] as f32 / 65535.0,
                                w[2] as f32 / 65535.0,
                                w[3] as f32 / 65535.0,
                            ]
                        })
                        .collect(),
                    W::F32(iter) => iter.collect(),
                }
            });
            let (Some(joints), Some(weights)) = (joints, weights) else {
                continue; // unskinned: the static importer owns it
            };
            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .map(|it| it.collect())
                .unwrap_or_default();
            if positions.is_empty() {
                continue;
            }
            let skin = doc.skins().next();
            let (inverse_bind, node_to_joint, joint_nodes) = match skin {
                Some(s) => {
                    let ibm: Vec<Mat4> = s
                        .reader(|b| buffers.get(b.index()).map(|d| d.0.as_slice()))
                        .read_inverse_bind_matrices()
                        .map(|it| it.map(|m| Mat4::from_cols_array_2d(&m)).collect())
                        .unwrap_or_default();
                    let joints: Vec<gltf::Node> = s.joints().collect();
                    let map: HashMap<usize, usize> = joints
                        .iter()
                        .enumerate()
                        .map(|(slot, node)| (node.index(), slot))
                        .collect();
                    let nodes: Vec<usize> = joints.iter().map(|n| n.index()).collect();
                    let count = joints.len().max(1);
                    let ibm = if ibm.len() >= count {
                        ibm
                    } else {
                        let mut full = ibm;
                        full.resize(count, Mat4::IDENTITY);
                        full
                    };
                    (ibm, map, nodes)
                }
                None => (vec![Mat4::IDENTITY], HashMap::new(), vec![usize::MAX]),
            };
            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|it| it.collect())
                .unwrap_or_default();
            // Same texcoord rule as the static importer: the material's
            // base-color texture selects the set, never silently set 0.
            let tex_coord = prim
                .material()
                .pbr_metallic_roughness()
                .base_color_texture()
                .map(|t| t.tex_coord())
                .unwrap_or(0);
            let uvs: Vec<[f32; 2]> = reader
                .read_tex_coords(tex_coord)
                .map(|it| it.into_f32().collect())
                .unwrap_or_default();
            let indices: Vec<u32> = match reader.read_indices() {
                Some(gltf::mesh::util::ReadIndices::U8(it)) => it.map(u32::from).collect(),
                Some(gltf::mesh::util::ReadIndices::U16(it)) => it.map(u32::from).collect(),
                Some(gltf::mesh::util::ReadIndices::U32(it)) => it.collect(),
                None => (0..positions.len() as u32).collect(),
            };
            let bc = prim.material().pbr_metallic_roughness().base_color_factor();
            let tint = [bc[0], bc[1], bc[2]];
            let (transparent, alpha, alpha_cutoff) = super::gltf::alpha_mode(&prim.material());
            // Same guarded link as the static importer (see `gltf.rs`).
            let base_image = prim
                .material()
                .pbr_metallic_roughness()
                .base_color_texture()
                .and_then(|t| {
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        t.texture().source().index()
                    }))
                    .ok()
                });
            out.push(SkinnedMesh {
                name: format!("skin_{}", mesh.index()),
                colors: vec![tint; positions.len()],
                positions,
                normals,
                uvs: uvs.iter().map(|[u, v]| [*u, 1.0 - *v]).collect(),
                indices,
                joints,
                weights,
                inverse_bind,
                node_to_joint,
                joint_nodes,
                texture_page: 0,
                base_image,
                pick_id: 0,
                transparent,
                alpha,
                alpha_cutoff,
                material: super::gltf::material_of(&prim.material()),
                depth_test: true,
            });
        }
    }
    Ok(out)
}

/// One imported keyframe: time, channel, value, interpolation, and
/// CUBICSPLINE tangents (identity when unused).
#[derive(Clone, Copy, Debug, Default)]
struct TrackKey {
    t: f32,
    /// 0 = translation, 1 = rotation, 2 = scale.
    ch: u8,
    interp: Interp,
    value: JointPose,
    tan_in: JointPose,
    tan_out: JointPose,
}

/// Sample every animation track in `bytes` into node-indexed [`NodePose`]s
/// plus [`MorphTrack`]s. Per-channel interpolation comes from the glTF
/// sampler (`channel.sampler().interpolation()`): LINEAR lerps/slerps, STEP
/// holds, CUBICSPLINE carries tangents (translation/scale hermite, rotation
/// tangent-nudged slerp). CUBICSPLINE outputs arrive triple-wide
/// (in-tangent, value, out-tangent) per key: the middle third drives values,
/// the outer thirds drive tangents. `weights` (morph) outputs arrive
/// `target_count`-wide per key and split per mesh. Unknown targets skip
/// silently — never a panic, never partial tracks.
pub fn import_animations(bytes: &[u8]) -> Result<Vec<Animation>, gltf::Error> {
    let (doc, buffers, _) = gltf::import_slice(bytes)?;
    let mesh_targets: Vec<usize> = doc
        .meshes()
        .map(|m| {
            m.primitives()
                .next()
                .map(|p| {
                    let reader = p.reader(|b| buffers.get(b.index()).map(|d| d.0.as_slice()));
                    reader.read_morph_targets().len()
                })
                .unwrap_or(0)
        })
        .collect();
    let mesh_of_node: HashMap<usize, usize> = {
        let mut map = HashMap::new();
        for scene in doc.scenes() {
            for node in scene.nodes() {
                let mut stack = vec![node];
                while let Some(n) = stack.pop() {
                    if let Some(mesh) = n.mesh() {
                        map.entry(n.index()).or_insert(mesh.index());
                    }
                    for child in n.children() {
                        stack.push(child);
                    }
                }
            }
        }
        map
    };
    let mut out = Vec::new();
    for anim in doc.animations() {
        let name = anim.name().unwrap_or("anim").to_string();
        let mut tracks: HashMap<usize, Vec<TrackKey>> = HashMap::new();
        let mut morphs: HashMap<usize, Vec<(f32, Vec<f32>)>> = HashMap::new();
        for channel in anim.channels() {
            let target = channel.target();
            let node = target.node().index();
            let interp = match channel.sampler().interpolation() {
                gltf::animation::Interpolation::Step => Interp::Step,
                gltf::animation::Interpolation::CubicSpline => Interp::CubicSpline,
                _ => Interp::Linear,
            };
            let reader = channel.reader(|b| buffers.get(b.index()).map(|d| d.0.as_slice()));
            let inputs: Vec<f32> = reader
                .read_inputs()
                .map(|it| it.collect())
                .unwrap_or_default();
            if inputs.is_empty() {
                continue;
            }
            use gltf::animation::util::ReadOutputs as O;
            match reader.read_outputs() {
                Some(O::Translations(it)) => {
                    let vals: Vec<[f32; 3]> = it.collect();
                    if interp == Interp::CubicSpline {
                        // Triple-wide per key: (in-tangent, value, out-tangent).
                        let n = inputs.len().min(vals.len() / 3);
                        for k in 0..n {
                            tracks.entry(node).or_default().push(TrackKey {
                                t: inputs[k],
                                ch: 0,
                                interp,
                                value: JointPose {
                                    translation: vals[k * 3 + 1],
                                    ..Default::default()
                                },
                                tan_in: JointPose {
                                    translation: vals[k * 3],
                                    ..Default::default()
                                },
                                tan_out: JointPose {
                                    translation: vals[k * 3 + 2],
                                    ..Default::default()
                                },
                            });
                        }
                    } else {
                        for (t, v) in inputs.into_iter().zip(vals) {
                            tracks.entry(node).or_default().push(TrackKey {
                                t,
                                ch: 0,
                                interp,
                                value: JointPose {
                                    translation: v,
                                    ..Default::default()
                                },
                                ..Default::default()
                            });
                        }
                    }
                }
                Some(O::Rotations(rot)) => {
                    let vals: Vec<[f32; 4]> = rot.into_f32().collect();
                    if interp == Interp::CubicSpline {
                        let n = inputs.len().min(vals.len() / 3);
                        for k in 0..n {
                            tracks.entry(node).or_default().push(TrackKey {
                                t: inputs[k],
                                ch: 1,
                                interp,
                                value: JointPose {
                                    rotation: vals[k * 3 + 1],
                                    ..Default::default()
                                },
                                tan_in: JointPose {
                                    rotation: vals[k * 3],
                                    ..Default::default()
                                },
                                tan_out: JointPose {
                                    rotation: vals[k * 3 + 2],
                                    ..Default::default()
                                },
                            });
                        }
                    } else {
                        for (t, v) in inputs.into_iter().zip(vals) {
                            tracks.entry(node).or_default().push(TrackKey {
                                t,
                                ch: 1,
                                interp,
                                value: JointPose {
                                    rotation: v,
                                    ..Default::default()
                                },
                                ..Default::default()
                            });
                        }
                    }
                }
                Some(O::Scales(it)) => {
                    let vals: Vec<[f32; 3]> = it.collect();
                    if interp == Interp::CubicSpline {
                        let n = inputs.len().min(vals.len() / 3);
                        for k in 0..n {
                            tracks.entry(node).or_default().push(TrackKey {
                                t: inputs[k],
                                ch: 2,
                                interp,
                                value: JointPose {
                                    scale: vals[k * 3 + 1],
                                    ..Default::default()
                                },
                                tan_in: JointPose {
                                    scale: vals[k * 3],
                                    ..Default::default()
                                },
                                tan_out: JointPose {
                                    scale: vals[k * 3 + 2],
                                    ..Default::default()
                                },
                            });
                        }
                    } else {
                        for (t, v) in inputs.into_iter().zip(vals) {
                            tracks.entry(node).or_default().push(TrackKey {
                                t,
                                ch: 2,
                                interp,
                                value: JointPose {
                                    scale: v,
                                    ..Default::default()
                                },
                                ..Default::default()
                            });
                        }
                    }
                }
                Some(O::MorphTargetWeights(w)) => {
                    use gltf::animation::util::MorphTargetWeights as M;
                    let flat: Vec<f32> = match w {
                        M::I8(it) => it.map(f32::from).collect(),
                        M::U8(it) => it.map(f32::from).collect(),
                        M::I16(it) => it.map(f32::from).collect(),
                        M::U16(it) => it.map(f32::from).collect(),
                        M::F32(it) => it.collect(),
                    };
                    let mesh = match mesh_of_node.get(&node) {
                        Some(m) => *m,
                        None => continue,
                    };
                    let count = mesh_targets.get(mesh).copied().unwrap_or(0);
                    if count == 0 {
                        continue;
                    }
                    let n = inputs.len().min(flat.len() / count);
                    for k in 0..n {
                        morphs
                            .entry(mesh)
                            .or_default()
                            .push((inputs[k], flat[k * count..(k + 1) * count].to_vec()));
                    }
                }
                _ => {}
            }
        }
        let mut poses: HashMap<usize, NodePose> = HashMap::new();
        for (node, keys) in tracks {
            let mut pose = NodePose::default();
            for key in keys {
                match pose.times.iter().position(|&e| (e - key.t).abs() < 1e-9) {
                    Some(i) => match key.ch {
                        0 => {
                            if pose.translations.len() == pose.times.len() {
                                pose.translations[i] = key.value.translation;
                                if pose.translation_interp.len() == pose.times.len() {
                                    pose.translation_interp[i] = key.interp;
                                }
                                if pose.translation_tangents.len() == pose.times.len() {
                                    pose.translation_tangents[i] =
                                        (key.tan_in.translation, key.tan_out.translation);
                                }
                            }
                        }
                        1 => {
                            if pose.rotations.len() == pose.times.len() {
                                pose.rotations[i] = key.value.rotation;
                                if pose.rotation_interp.len() == pose.times.len() {
                                    pose.rotation_interp[i] = key.interp;
                                }
                                if pose.rotation_tangents.len() == pose.times.len() {
                                    pose.rotation_tangents[i] =
                                        (key.tan_in.rotation, key.tan_out.rotation);
                                }
                            }
                        }
                        _ => {
                            if pose.scales.len() == pose.times.len() {
                                pose.scales[i] = key.value.scale;
                                if pose.scale_interp.len() == pose.times.len() {
                                    pose.scale_interp[i] = key.interp;
                                }
                                if pose.scale_tangents.len() == pose.times.len() {
                                    pose.scale_tangents[i] = (key.tan_in.scale, key.tan_out.scale);
                                }
                            }
                        }
                    },
                    None => {
                        pose.times.push(key.t);
                        let last_t = pose.translations.last().copied().unwrap_or([0.0, 0.0, 0.0]);
                        let last_r = pose
                            .rotations
                            .last()
                            .copied()
                            .unwrap_or([0.0, 0.0, 0.0, 1.0]);
                        let last_s = pose.scales.last().copied().unwrap_or([1.0, 1.0, 1.0]);
                        pose.translations.push(match key.ch {
                            0 => key.value.translation,
                            _ => last_t,
                        });
                        pose.rotations.push(match key.ch {
                            1 => key.value.rotation,
                            _ => last_r,
                        });
                        pose.scales.push(match key.ch {
                            2 => key.value.scale,
                            _ => last_s,
                        });
                        pose.translation_interp.push(match key.ch {
                            0 => key.interp,
                            _ => Interp::Linear,
                        });
                        pose.rotation_interp.push(match key.ch {
                            1 => key.interp,
                            _ => Interp::Linear,
                        });
                        pose.scale_interp.push(match key.ch {
                            2 => key.interp,
                            _ => Interp::Linear,
                        });
                        pose.translation_tangents
                            .push((key.tan_in.translation, key.tan_out.translation));
                        pose.rotation_tangents
                            .push((key.tan_in.rotation, key.tan_out.rotation));
                        pose.scale_tangents
                            .push((key.tan_in.scale, key.tan_out.scale));
                    }
                }
            }
            let mut order: Vec<usize> = (0..pose.times.len()).collect();
            order.sort_by(|&a, &b| {
                pose.times[a]
                    .partial_cmp(&pose.times[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let gather = |v: &Vec<[f32; 3]>| order.iter().map(|&i| v[i]).collect::<Vec<_>>();
            let gather4 = |v: &Vec<[f32; 4]>| order.iter().map(|&i| v[i]).collect::<Vec<_>>();
            let gather_i = |v: &Vec<Interp>| order.iter().map(|&i| v[i]).collect::<Vec<_>>();
            let gather_t3 =
                |v: &Vec<([f32; 3], [f32; 3])>| order.iter().map(|&i| v[i]).collect::<Vec<_>>();
            let gather_t4 =
                |v: &Vec<([f32; 4], [f32; 4])>| order.iter().map(|&i| v[i]).collect::<Vec<_>>();
            pose = NodePose {
                times: order.iter().map(|&i| pose.times[i]).collect(),
                translations: gather(&pose.translations),
                rotations: gather4(&pose.rotations),
                scales: gather(&pose.scales),
                translation_interp: gather_i(&pose.translation_interp),
                rotation_interp: gather_i(&pose.rotation_interp),
                scale_interp: gather_i(&pose.scale_interp),
                translation_tangents: gather_t3(&pose.translation_tangents),
                rotation_tangents: gather_t4(&pose.rotation_tangents),
                scale_tangents: gather_t3(&pose.scale_tangents),
            };
            poses.insert(node, pose);
        }
        let mut morph_out: HashMap<usize, MorphTrack> = HashMap::new();
        for (mesh, keys) in morphs {
            let count = mesh_targets.get(mesh).copied().unwrap_or(0);
            if count == 0 {
                continue;
            }
            let mut times: Vec<f32> = Vec::with_capacity(keys.len());
            let mut weights: Vec<f32> = Vec::with_capacity(keys.len() * count);
            for (t, w) in keys {
                times.push(t);
                if w.len() == count {
                    weights.extend_from_slice(&w);
                } else {
                    weights.extend(std::iter::repeat_n(0.0, count));
                }
            }
            let mut order: Vec<usize> = (0..times.len()).collect();
            order.sort_by(|&a, &b| {
                times[a]
                    .partial_cmp(&times[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            morph_out.insert(
                mesh,
                MorphTrack {
                    times: order.iter().map(|&i| times[i]).collect(),
                    weights: order
                        .iter()
                        .flat_map(|&i| weights[i * count..(i + 1) * count].to_vec())
                        .collect(),
                    target_count: count,
                },
            );
        }
        out.push(Animation {
            name,
            tracks: poses,
            morph_tracks: morph_out,
        });
    }
    Ok(out)
}

/// Parse morph targets (position/normal deltas) per mesh, in primitive
/// order. `out[mesh_index]` holds that mesh's targets; meshes without
/// targets hold an empty [`MorphSet`]. Weights arrive per frame from
/// animation tracks ([`Animation::morph_tracks`]) or game code, and
/// [`MorphSet::apply`] blends them onto the baked group.
pub fn import_morphs(bytes: &[u8]) -> Result<Vec<MorphSet>, gltf::Error> {
    let (doc, buffers, _) = gltf::import_slice(bytes)?;
    let mut out = Vec::new();
    for mesh in doc.meshes() {
        let mut set = MorphSet::default();
        for prim in mesh.primitives() {
            let reader = prim.reader(|b| buffers.get(b.index()).map(|d| d.0.as_slice()));
            for (positions, normals, _) in reader.read_morph_targets() {
                let pos: Vec<[f32; 3]> = positions.map(|it| it.collect()).unwrap_or_default();
                if pos.is_empty() {
                    continue;
                }
                let nrm: Vec<[f32; 3]> = normals.map(|it| it.collect()).unwrap_or_default();
                set.position_deltas.push(pos);
                set.normal_deltas.push(nrm);
            }
            if !set.is_empty() {
                break; // first morph-carrying primitive wins per mesh
            }
        }
        out.push(set);
    }
    Ok(out)
}

/// Bone attachment (Godot `BoneAttachment3D`): fix a prop group to a joint.
///
/// `group` is baked in mesh-local space (an unskinned prop: a hat, a held
/// gun, a pickup marker). `node` is the joint-space world matrix of the
/// attach joint (from [`Skeleton::world_matrices`] — usually animated, so
/// call this per frame after sampling the player), and `offset` is the
/// prop's local transform relative to the bone (identity = prop origin on
/// the joint). Returns the group moved into world space, in place.
///
/// Attributes (normals rotate, uvs untouched), material, pick id, and
/// transparency ride along: the prop keeps its look and stays clickable.
/// The prop must be rigid (no per-vertex skinning) — the whole group takes
/// one matrix.
pub fn attach_to_joint(group: &mut MeshGroup, node: &Mat4, offset: &Mat4) {
    let m = *node * *offset;
    let n = Mat4::from_quat(Quat::from_mat4(&m));
    for p in &mut group.positions {
        *p = m.transform_point3(Vec3::from(*p)).into();
    }
    for normal in &mut group.normals {
        let v = n.transform_vector3(Vec3::from(*normal));
        *normal = v.try_normalize().unwrap_or(Vec3::Y).into();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two-vertex "limb": vert 0 bound to joint 0, vert 1 split 50/50.
    fn limb() -> SkinnedMesh {
        SkinnedMesh {
            name: "limb".into(),
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            normals: vec![[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            uvs: vec![],
            colors: vec![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            indices: vec![0, 1, 0],
            joints: vec![[0, 0, 0, 0], [0, 1, 0, 0]],
            weights: vec![[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]],
            inverse_bind: vec![Mat4::IDENTITY, Mat4::IDENTITY],
            node_to_joint: HashMap::from([(10, 0), (11, 1)]),
            joint_nodes: vec![10, 11],
            texture_page: 0,
            pick_id: 3,
            depth_test: true,
            ..Default::default()
        }
    }

    #[test]
    fn identity_pose_holds_bind() {
        let g = limb().pose(&[Mat4::IDENTITY, Mat4::IDENTITY]);
        assert_eq!(g.positions, vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        assert_eq!(g.normals.len(), 2);
        assert_eq!(g.pick_id, 3);
    }

    #[test]
    fn joint_translation_blends_by_weight() {
        let joints = [
            Mat4::IDENTITY,
            Mat4::from_translation(Vec3::new(2.0, 0.0, 0.0)),
        ];
        let g = limb().pose(&joints);
        assert!((g.positions[0][0] - 0.0).abs() < 1e-6);
        assert!(
            (g.positions[1][0] - 2.0).abs() < 1e-6,
            "{:?}",
            g.positions[1]
        );
    }

    #[test]
    fn unweighted_verts_hold_bind_pose() {
        let mut m = limb();
        m.weights[0] = [0.0; 4];
        let joints = [Mat4::from_translation(Vec3::new(9.0, 9.0, 9.0)); 2];
        let g = m.pose(&joints);
        assert_eq!(g.positions[0], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn out_of_range_joints_clamp_instead_of_panic() {
        let mut m = limb();
        m.joints[1] = [9, 9, 9, 9];
        let g = m.pose(&[Mat4::IDENTITY, Mat4::IDENTITY]);
        assert_eq!(g.positions.len(), 2);
    }

    #[test]
    fn node_pose_samples_and_clamps() {
        let pose = NodePose {
            times: vec![0.0, 1.0],
            translations: vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            rotations: vec![],
            scales: vec![],
            ..Default::default()
        };
        assert_eq!(pose.sample(-1.0).translation, [0.0, 0.0, 0.0]);
        assert_eq!(pose.sample(0.5).translation, [1.0, 0.0, 0.0]);
        assert_eq!(pose.sample(9.0).translation, [2.0, 0.0, 0.0]);
        assert_eq!(pose.duration(), 1.0);
        assert!(NodePose::default().is_empty());
    }

    #[test]
    fn joint_pose_matrix_composes_trs() {
        let p = JointPose {
            translation: [1.0, 0.0, 0.0],
            ..Default::default()
        };
        let v = p.matrix().transform_point3(Vec3::ZERO);
        assert!((v - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn gnome_skin_and_anims_evaluate() {
        let path = "/home/ymsr/Downloads/pilot-garden-linux/assets/gnome.glb";
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("SKIP gnome test (no fixture): {e}");
                return;
            }
        };
        let skinned = import_skinned(&bytes).expect("gnome parses");
        assert_eq!(skinned.len(), 1);
        let m = &skinned[0];
        assert_eq!(m.joint_count(), 9);
        let id = vec![Mat4::IDENTITY; m.joint_count()];
        let bind = m.pose(&id);
        assert_eq!(bind.tri_count(), 715);
        let anims = import_animations(&bytes).expect("gnome anims parse");
        assert_eq!(anims.len(), 4);
        let a = &anims[0];
        let skeleton = import_skeleton(&bytes).expect("gnome skeleton parses");
        assert!(skeleton.node_count() >= m.joint_count());
        let joints = a.joint_matrices(&skeleton, m, a.duration() / 2.0);
        assert_eq!(joints.len(), m.joint_count());
        let posed = m.pose(&joints);
        let moved = posed
            .positions
            .iter()
            .zip(bind.positions.iter())
            .filter(|(a, b)| {
                let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
                dx * dx + dy * dy + dz * dz > 1e-10
            })
            .count();
        assert_eq!(moved, posed.positions.len(), "anim moves the mesh");
        // Player path agrees with the manual path at the same time.
        let mut player = SkeletonPlayer::new(a, SkeletonLoop::Loop);
        player.play();
        player.seek(a.duration() / 2.0);
        let via_player = player.joint_matrices(a, &skeleton, m);
        assert_eq!(via_player.len(), joints.len());
        for (x, y) in via_player.iter().zip(joints.iter()) {
            assert!(
                (x.to_cols_array().iter().zip(y.to_cols_array().iter()))
                    .all(|(p, q)| (p - q).abs() < 1e-6),
                "player matches manual sample"
            );
        }
    }

    #[test]
    fn step_interp_holds_left_key() {
        let pose = NodePose {
            times: vec![0.0, 1.0],
            translations: vec![[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            translation_interp: vec![Interp::Step, Interp::Step],
            ..Default::default()
        };
        assert_eq!(pose.sample(0.99).translation, [0.0, 0.0, 0.0]);
        assert_eq!(pose.sample(1.0).translation, [4.0, 0.0, 0.0]);
    }

    #[test]
    fn cubic_hermite_midpoint_matches_basis() {
        // Zero in-tangent, unit out-tangent over dt=1: h00/h10/h01/h11 at
        // alpha=0.5 are (0.5, 0.125, 0.5, -0.125), so x = 0.5*0 + 0.125*0
        // + 0.5*2 + -0.125*0 = 1.0.
        let pose = NodePose {
            times: vec![0.0, 1.0],
            translations: vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            translation_interp: vec![Interp::CubicSpline, Interp::CubicSpline],
            translation_tangents: vec![([0.0; 3], [0.0; 3]), ([0.0; 3], [0.0; 3])],
            ..Default::default()
        };
        let mid = pose.sample(0.5).translation[0];
        assert!((mid - 1.0).abs() < 1e-5, "mid = {mid}");
        // Out-tangent of 3.0 on x pushes the midpoint up: x = 0.5*0 +
        // 0.125*3 + 0.5*2 = 1.375.
        let pose = NodePose {
            translation_tangents: vec![([0.0; 3], [3.0, 0.0, 0.0]), ([0.0; 3], [0.0; 3])],
            ..pose
        };
        let mid = pose.sample(0.5).translation[0];
        assert!((mid - 1.375).abs() < 1e-5, "mid = {mid}");
    }

    #[test]
    fn children_follow_parents_through_world_matrices() {
        let skeleton = Skeleton {
            parents: HashMap::from([(0, None), (1, Some(0))]),
            bind_local: HashMap::from([(0, Mat4::IDENTITY), (1, Mat4::IDENTITY)]),
        };
        let anim = Animation {
            name: "wave".into(),
            tracks: HashMap::from([(
                0,
                NodePose {
                    times: vec![0.0, 1.0],
                    translations: vec![[0.0; 3], [5.0, 0.0, 0.0]],
                    ..Default::default()
                },
            )]),
            ..Default::default()
        };
        let mesh = SkinnedMesh {
            inverse_bind: vec![Mat4::IDENTITY, Mat4::IDENTITY],
            joint_nodes: vec![0, 1],
            ..Default::default()
        };
        let joints = anim.joint_matrices(&skeleton, &mesh, 1.0);
        let child_origin = joints[1].transform_point3(Vec3::ZERO);
        assert!(
            (child_origin - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-5,
            "child follows parent: {child_origin:?}"
        );
    }

    #[test]
    fn player_loops_holds_and_bounces_like_anim_player() {
        let anim = Animation {
            name: "t".into(),
            tracks: HashMap::from([(
                0,
                NodePose {
                    times: vec![0.0, 2.0],
                    translations: vec![[0.0; 3], [2.0, 0.0, 0.0]],
                    ..Default::default()
                },
            )]),
            ..Default::default()
        };
        assert_eq!(anim.duration(), 2.0);
        // Loop wraps.
        let mut p = SkeletonPlayer::new(&anim, SkeletonLoop::Loop);
        p.play();
        let adv = p.advance(2.5);
        assert_eq!(adv.wraps, 1);
        assert!((p.time() - 0.5).abs() < 1e-5, "t = {}", p.time());
        // Once holds and finishes.
        let mut p = SkeletonPlayer::new(&anim, SkeletonLoop::Once);
        p.play();
        let adv = p.advance(9.0);
        assert!(p.is_finished() && !p.is_playing());
        assert_eq!(adv.wraps, 1);
        assert_eq!(p.time(), 2.0);
        // PingPong folds: 2.5s into a 2s clip reads t=1.5.
        let mut p = SkeletonPlayer::new(&anim, SkeletonLoop::PingPong);
        p.play();
        p.advance(2.5);
        assert!(
            (p.sample_time() - 1.5).abs() < 1e-5,
            "t = {}",
            p.sample_time()
        );
        // Reverse plays backwards from the end.
        let mut p = SkeletonPlayer::new(&anim, SkeletonLoop::Loop);
        p.play_backwards();
        assert_eq!(p.time(), 2.0);
        p.advance(0.5);
        assert!((p.time() - 1.5).abs() < 1e-5, "t = {}", p.time());
        // Paused / zero-rate / bad dt are no-ops.
        let mut p = SkeletonPlayer::new(&anim, SkeletonLoop::Loop);
        assert!(!p.advance(1.0).pose_changed);
        p.play();
        p.set_speed_scale(0.0);
        assert!(!p.advance(1.0).pose_changed);
        p.set_speed_scale(f32::NAN);
        assert!(!p.advance(1.0).pose_changed);
        assert!(!p.advance(-1.0).pose_changed);
    }

    #[test]
    fn attached_prop_follows_the_joint() {
        let mut prop = MeshGroup {
            pick_id: 5,
            depth_test: true,
            ..Default::default()
        };
        let v = [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ];
        for (a, b, c, d) in [
            (0, 1, 2, 3),
            (4, 6, 5, 7),
            (0, 4, 5, 1),
            (2, 6, 7, 3),
            (0, 3, 7, 4),
            (1, 5, 6, 2),
        ] {
            prop.push_quad_lit(v[a], v[b], v[c], v[d], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        }
        prop.material = super::super::mesh::Material {
            metallic: 0.5,
            ..Default::default()
        };
        let node = Mat4::from_translation(Vec3::new(0.0, 3.0, 0.0))
            * Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let offset = Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0));
        attach_to_joint(&mut prop, &node, &offset);
        let c = prop
            .positions
            .iter()
            .fold(Vec3::ZERO, |a, p| a + Vec3::from(*p))
            / prop.positions.len() as f32;
        assert!(
            (c.x - 0.0).abs() < 1e-4 && (c.z + 1.0).abs() < 1e-4,
            "centroid carried by the joint: {c:?}"
        );
        assert!(
            (c.y - 3.0).abs() <= 1.0,
            "centroid inside the moved cube: {c:?}"
        );
        assert!(prop.normals.iter().all(|n| *n == [0.0, 1.0, 0.0]));
        assert_eq!(prop.pick_id, 5);
        assert_eq!(prop.material.metallic, 0.5);
        assert_eq!(prop.tri_count(), 12);
    }

    #[test]
    fn joint_pose_blend_is_identity_at_zero_and_full_at_one() {
        let a = JointPose::default();
        let b = JointPose {
            translation: [4.0, 0.0, 0.0],
            rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2).to_array(),
            scale: [2.0, 2.0, 2.0],
        };
        let z = a.blend(&b, 0.0);
        assert_eq!(z.translation, a.translation);
        let f = a.blend(&b, 1.0);
        assert_eq!(f.translation, b.translation);
        let h = a.blend(&b, 0.5);
        assert_eq!(h.translation, [2.0, 0.0, 0.0]);
    }

    #[test]
    fn morph_set_blends_positions_and_renormalizes() {
        let mut group = MeshGroup {
            depth_test: true,
            ..Default::default()
        };
        group.push_quad_lit(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
        );
        let n = group.positions.len();
        let set = MorphSet {
            position_deltas: vec![vec![[1.0, 0.0, 0.0]; n]],
            normal_deltas: vec![vec![[1.0, 0.0, 0.0]; n]],
        };
        // Zero weights: untouched.
        set.apply(&mut group, &[0.0]);
        assert_eq!(group.positions[0], [0.0, 0.0, 0.0]);
        // Half weight: half delta.
        set.apply(&mut group, &[0.5]);
        assert!((group.positions[0][0] - 0.5).abs() < 1e-6);
        assert!(
            group.normals.iter().all(|nor| {
                let l = Vec3::from(*nor).length();
                (l - 1.0).abs() < 1e-5
            }),
            "renormalized"
        );
    }

    #[test]
    fn morph_track_samples_weights_linearly() {
        let track = MorphTrack {
            times: vec![0.0, 1.0],
            weights: vec![0.0, 0.0, 1.0, 0.5],
            target_count: 2,
        };
        assert_eq!(track.sample(-1.0), vec![0.0, 0.0]);
        assert_eq!(track.sample(0.5), vec![0.5, 0.25]);
        assert_eq!(track.sample(9.0), vec![1.0, 0.5]);
    }
}
