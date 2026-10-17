//! Immediate-mode debug draw: frame-scoped lines and wire shapes.
//!
//! Games call [`Gizmos::line`] and friends while composing a frame, then hand
//! the accumulator to [`Frame3d::push_gizmos`](crate::Frame3d::push_gizmos)
//! with the current camera eye. The accumulator expands every primitive into
//! triangles and emits them as a `depth_test = false` overlay
//! [`MeshGroup`](crate::MeshGroup), the same never-culled channel the viewport
//! already uses for ground decals and editor handles, so gizmos draw on top of
//! the scene no matter where they sit in the frustum.
//!
//! The batch's pipelines are triangle-list only, so a segment becomes a
//! camera-facing ribbon (`width` world units) rather than a hardware line.
//! Coordinates are world space, y-up right-handed, matching the rest of the
//! crate. Everything is plain data: nothing here touches the GPU or a global,
//! and `clear` (or a fresh accumulator) starts the next frame.
//!
//! ```
//! use repame_view3d::gizmos::{AXIS_X, Gizmos};
//! use glam::Vec3;
//!
//! let mut g = Gizmos::new();
//! g.line(Vec3::ZERO, Vec3::X, AXIS_X);
//! g.wire_aabb(Vec3::splat(-1.0), Vec3::splat(1.0), [0.2, 1.0, 0.4, 1.0]);
//! let group = g.to_group(Vec3::new(0.0, 0.0, 10.0)).expect("two shapes");
//! assert!(!group.depth_test);
//! ```

use glam::Vec3;

use crate::mesh::MeshGroup;

/// Linear RGBA color, matching [`crate::material::Rgba`].
pub type GizmoColor = [f32; 4];

/// X axis red, the conventional debug-draw tint.
pub const AXIS_X: GizmoColor = [0.9, 0.15, 0.2, 1.0];
/// Y axis green.
pub const AXIS_Y: GizmoColor = [0.25, 0.85, 0.3, 1.0];
/// Z axis blue.
pub const AXIS_Z: GizmoColor = [0.25, 0.45, 0.95, 1.0];
/// Neutral white, the default shape tint.
pub const WHITE: GizmoColor = [1.0, 1.0, 1.0, 1.0];
/// Yellow, the conventional highlight tint.
pub const YELLOW: GizmoColor = [0.95, 0.85, 0.2, 1.0];

/// One segment awaiting expansion, with per-endpoint colors so a gradient
/// line keeps both ends.
#[derive(Clone, Copy, Debug)]
struct Segment {
    a: Vec3,
    b: Vec3,
    ca: GizmoColor,
    cb: GizmoColor,
}

/// Frame-scoped immediate-mode debug-draw batch.
///
/// Push primitives in any order; they are expanded to triangles by
/// [`Gizmos::to_group`]. Vertices are world space and unlit, and the resulting
/// group is a transparent `depth_test = false` overlay, so shape order is the
/// only ordering that matters and self-intersection never hides a primitive.
#[derive(Clone, Debug)]
pub struct Gizmos {
    segments: Vec<Segment>,
    /// Ribbon half-extent in world units. See [`Gizmos::set_line_width`].
    width: f32,
}

impl Default for Gizmos {
    fn default() -> Self {
        Self {
            segments: Vec::new(),
            width: 0.1,
        }
    }
}

impl Gizmos {
    /// A fresh accumulator with a 0.1 world-unit line width.
    pub fn new() -> Self {
        Self::default()
    }

    /// Drops every primitive, keeping the line width. Call once per frame when
    /// reusing one accumulator (the immediate-mode contract).
    pub fn clear(&mut self) {
        self.segments.clear();
    }

    /// Segment ribbon width, i.e. the full thickness in world units. Clamped
    /// non-negative and finite; `0` collapses ribbons to nothing.
    pub fn set_line_width(&mut self, width: f32) {
        self.width = if width.is_finite() { width.max(0.0) } else { 0.0 };
    }

    /// Current ribbon width in world units.
    pub fn line_width(&self) -> f32 {
        self.width
    }

    /// Number of segments queued (a wire box is 12, a circle is its segment
    /// count).
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// One segment with a single color. Non-finite endpoints or colors are
    /// dropped so a bad value cannot poison the whole group's bounds.
    pub fn line(&mut self, a: Vec3, b: Vec3, color: GizmoColor) {
        self.line_gradient(a, b, color, color);
    }

    /// One segment whose endpoints carry different colors.
    pub fn line_gradient(&mut self, a: Vec3, b: Vec3, ca: GizmoColor, cb: GizmoColor) {
        if !finite3(a) || !finite3(b) || !finite4(ca) || !finite4(cb) {
            return;
        }
        if a == b {
            return;
        }
        self.segments.push(Segment { a, b, ca, cb });
    }

    /// A polyline through `points`, one segment per adjacent pair.
    pub fn line_strip(&mut self, points: &[Vec3], color: GizmoColor) {
        for pair in points.windows(2) {
            self.line(pair[0], pair[1], color);
        }
    }

    /// Closed polyline through `points`, connecting the last back to the first.
    pub fn line_loop(&mut self, points: &[Vec3], color: GizmoColor) {
        self.line_strip(points, color);
        if let (Some(first), Some(last)) = (points.first(), points.last()) {
            self.line(*last, *first, color);
        }
    }

    /// A shaft with a two-stroke arrowhead at `to`. `head` is the head length
    /// in world units (clamped to the shaft length); the head opens at ~25
    /// degrees.
    pub fn arrow(&mut self, from: Vec3, to: Vec3, color: GizmoColor) {
        self.arrow_sized(from, to, 0.0, color);
    }

    /// [`Gizmos::arrow`] with an explicit head length; `0` auto-sizes to 20% of
    /// the shaft, clamped to the shaft length.
    pub fn arrow_sized(&mut self, from: Vec3, to: Vec3, head: f32, color: GizmoColor) {
        self.line(from, to, color);
        let dir = to - from;
        let len = dir.length();
        if len < 1e-6 {
            return;
        }
        let unit = dir / len;
        let (side, up) = orthonormal(unit);
        let h = if head.is_finite() && head > 0.0 {
            head.min(len)
        } else {
            len * 0.2
        };
        let back = to - unit * h;
        let spread = h * 0.5;
        self.line(to, back + side * spread, color);
        self.line(to, back - side * spread, color);
        self.line(to, back + up * spread, color);
        self.line(to, back - up * spread, color);
    }

    /// Axis-aligned wire box from two opposite corners: twelve segments.
    pub fn wire_aabb(&mut self, min: Vec3, max: Vec3, color: GizmoColor) {
        let lo = min.min(max);
        let hi = min.max(max);
        let x0 = lo.x;
        let y0 = lo.y;
        let z0 = lo.z;
        let x1 = hi.x;
        let y1 = hi.y;
        let z1 = hi.z;
        let c = color;
        let corners = [
            Vec3::new(x0, y0, z0),
            Vec3::new(x1, y0, z0),
            Vec3::new(x1, y0, z1),
            Vec3::new(x0, y0, z1),
            Vec3::new(x0, y1, z0),
            Vec3::new(x1, y1, z0),
            Vec3::new(x1, y1, z1),
            Vec3::new(x0, y1, z1),
        ];
        const EDGES: [(usize, usize); 12] = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ];
        for (i, j) in EDGES {
            self.line(corners[i], corners[j], c);
        }
    }

    /// Wire sphere as three great circles (XY, XZ, YZ), `segments` steps each.
    pub fn wire_sphere(&mut self, center: Vec3, radius: f32, color: GizmoColor) {
        self.wire_sphere_sized(center, radius, 24, color);
    }

    /// [`Gizmos::wire_sphere`] with an explicit segment count per circle.
    pub fn wire_sphere_sized(
        &mut self,
        center: Vec3,
        radius: f32,
        segments: u32,
        color: GizmoColor,
    ) {
        if !finite3(center) || !radius.is_finite() || radius <= 0.0 {
            return;
        }
        let n = segments.clamp(3, 512);
        self.circle(center, Vec3::Z, radius, n, color);
        self.circle(center, Vec3::Y, radius, n, color);
        self.circle(center, Vec3::X, radius, n, color);
    }

    /// Wire circle in the plane whose normal is `normal`, `segments` steps.
    pub fn circle(&mut self, center: Vec3, normal: Vec3, radius: f32, segments: u32, color: GizmoColor) {
        if !finite3(center) || !finite3(normal) || !radius.is_finite() || radius <= 0.0 {
            return;
        }
        let n = normal.normalize_or_zero();
        if n == Vec3::ZERO {
            return;
        }
        let (u, v) = orthonormal(n);
        let steps = segments.clamp(3, 512);
        let mut prev = center + u * radius;
        for i in 1..=steps {
            let t = i as f32 / steps as f32 * std::f32::consts::TAU;
            let (s, c) = t.sin_cos();
            let p = center + (u * c + v * s) * radius;
            self.line(prev, p, color);
            prev = p;
        }
    }

    /// Ground grid on the XZ plane, `half` cells each way around `origin`
    /// (so `half = 1` is a 3x3 lattice). `cell` is the spacing in world units.
    pub fn grid(&mut self, origin: Vec3, cell: f32, half: u32, color: GizmoColor) {
        if !finite3(origin) || !cell.is_finite() || cell <= 0.0 {
            return;
        }
        let half = half.min(1024);
        let span = cell * half as f32;
        for i in 0..=half * 2 {
            let o = -span + i as f32 * cell;
            self.line(
                origin + Vec3::new(o, 0.0, -span),
                origin + Vec3::new(o, 0.0, span),
                color,
            );
            self.line(
                origin + Vec3::new(-span, 0.0, o),
                origin + Vec3::new(span, 0.0, o),
                color,
            );
        }
    }

    /// World axes at `origin`, `len` world units long: X red, Y green, Z blue.
    pub fn axes(&mut self, origin: Vec3, len: f32) {
        self.arrow(origin, origin + Vec3::X * len, AXIS_X);
        self.arrow(origin, origin + Vec3::Y * len, AXIS_Y);
        self.arrow(origin, origin + Vec3::Z * len, AXIS_Z);
    }

    /// Three-segment cross at `p`, `size` world units on each arm. `up` and
    /// `right` pick the screen-facing plane, so a caller with a camera frame
    /// gets a readable cross.
    pub fn cross(&mut self, p: Vec3, size: f32, up: Vec3, right: Vec3, color: GizmoColor) {
        if !finite3(p) || !size.is_finite() || size <= 0.0 {
            return;
        }
        let u = up.normalize_or_zero();
        let r = right.normalize_or_zero();
        if u == Vec3::ZERO || r == Vec3::ZERO {
            return;
        }
        let f = u.cross(r).normalize_or_zero();
        if f == Vec3::ZERO {
            return;
        }
        self.line(p - u * size, p + u * size, color);
        self.line(p - r * size, p + r * size, color);
        self.line(p - f * size, p + f * size, color);
    }

    /// Ray from `origin` along `dir`, `length` world units long.
    pub fn ray(&mut self, origin: Vec3, dir: Vec3, length: f32, color: GizmoColor) {
        if !dir.is_finite() || !length.is_finite() {
            return;
        }
        let d = dir.normalize_or_zero();
        if d == Vec3::ZERO {
            return;
        }
        self.line(origin, origin + d * length, color);
    }

    /// Expands the accumulator into one unlit overlay group, or `None` when
    /// nothing was queued (or the ribbon width is zero, which would draw
    /// nothing anyway).
    ///
    /// `eye` is the camera world position; ribbons billboard around their own
    /// axis toward it, so a segment stays visible from any view.
    pub fn to_group(&self, eye: Vec3) -> Option<MeshGroup> {
        if self.segments.is_empty() || self.width <= 0.0 || !finite3(eye) {
            return None;
        }
        let mut group = MeshGroup {
            depth_test: false,
            transparent: true,
            alpha: 1.0,
            ..Default::default()
        };
        // One shared half-width for every segment: a per-segment width would
        // let a caller thread state through the whole API for no debug value.
        let half = self.width * 0.5;
        for seg in &self.segments {
            let dir = seg.b - seg.a;
            let mid = (seg.a + seg.b) * 0.5;
            let view = (eye - mid).normalize_or_zero();
            // Screen-space perpendicular: perpendicular to both the segment
            // and the view direction, so the ribbon faces the camera.
            let mut side = dir.cross(view).normalize_or_zero();
            if side == Vec3::ZERO {
                // Segment points at the camera (or `eye == mid`): any
                // perpendicular keeps a visible sliver instead of collapsing.
                let (u, _) = orthonormal(dir.normalize_or_zero());
                side = u;
            }
            let aw = seg.a + side * half;
            let bw = seg.b + side * half;
            let bo = seg.b - side * half;
            let ao = seg.a - side * half;
            // Keep the quad's winding toward the eye: the overlay pipelines
            // cull back faces, and a ribbon seen from behind would vanish.
            let face = (bw - ao).cross(bo - ao);
            let (c0, c1, c2, c3) = if face.dot(eye - mid) >= 0.0 {
                (ao, aw, bw, bo)
            } else {
                (ao, bo, bw, aw)
            };
            let base = group.positions.len() as u32;
            group.positions.extend_from_slice(&[
                c0.to_array(),
                c1.to_array(),
                c2.to_array(),
                c3.to_array(),
            ]);
            group
                .colors
                .extend_from_slice(&[rgb(seg.ca), rgb(seg.ca), rgb(seg.cb), rgb(seg.cb)]);
            group
                .indices
                .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        Some(group)
    }
}

fn rgb(c: GizmoColor) -> [f32; 3] {
    [c[0], c[1], c[2]]
}

fn finite3(v: Vec3) -> bool {
    v.is_finite()
}

fn finite4(c: GizmoColor) -> bool {
    c[0].is_finite() && c[1].is_finite() && c[2].is_finite() && c[3].is_finite()
}

/// Two unit vectors spanning the plane perpendicular to `n`. `n` is assumed
/// unit; falls back to world up/right when `n` is degenerate.
fn orthonormal(n: Vec3) -> (Vec3, Vec3) {
    let axis = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let u = n.cross(axis).normalize_or_zero();
    let u = if u == Vec3::ZERO {
        n.cross(Vec3::Z).normalize_or_zero()
    } else {
        u
    };
    let v = n.cross(u).normalize_or_zero();
    (u, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A known segment billboarded from a known eye pins the expansion math:
    /// the ribbon lies in the plane spanned by the segment and the screen
    /// perpendicular, centered on the segment, `width` thick.
    #[test]
    fn line_expands_to_a_camera_facing_ribbon() {
        let mut g = Gizmos::new();
        g.set_line_width(0.2);
        g.line(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), WHITE);
        let group = g.to_group(Vec3::new(1.0, 0.0, 5.0)).expect("one segment");
        assert!(!group.depth_test, "gizmos ride the overlay channel");
        assert!(group.transparent, "overlay blends over the scene");
        assert_eq!(group.positions.len(), 4);
        assert_eq!(group.indices, vec![0, 1, 2, 0, 2, 3]);
        // Segment along +X, eye along +Z from the midpoint: the perpendicular
        // is exactly -Y, so the ribbon is (0, +/-0.1) at x=0 and x=2.
        let expect = [
            [0.0, 0.1, 0.0],
            [0.0, -0.1, 0.0],
            [2.0, -0.1, 0.0],
            [2.0, 0.1, 0.0],
        ];
        for (got, want) in group.positions.iter().zip(expect.iter()) {
            for k in 0..3 {
                assert!(
                    (got[k] - want[k]).abs() < 1e-5,
                    "vertex {got:?} vs {want:?}"
                );
            }
        }
        // Ribbon faces the eye (winding survives back-face culling).
        let a = Vec3::from(group.positions[0]);
        let b = Vec3::from(group.positions[1]);
        let c = Vec3::from(group.positions[2]);
        assert!(
            (b - a).cross(c - a).dot(Vec3::new(0.0, 0.0, 1.0)) > 0.0,
            "quad winds toward the eye"
        );
    }

    /// The same segment seen from behind flips its winding instead of
    /// disappearing under back-face culling.
    #[test]
    fn ribbon_winding_follows_the_eye() {
        let mut g = Gizmos::new();
        g.line(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), WHITE);
        let front = g.to_group(Vec3::new(1.0, 0.0, 5.0)).unwrap();
        let back = g.to_group(Vec3::new(1.0, 0.0, -5.0)).unwrap();
        let normal = |m: &MeshGroup| {
            let a = Vec3::from(m.positions[0]);
            let b = Vec3::from(m.positions[1]);
            let c = Vec3::from(m.positions[2]);
            (b - a).cross(c - a)
        };
        assert!(normal(&front).dot(Vec3::Z) > 0.0);
        assert!(normal(&back).dot(Vec3::new(0.0, 0.0, -1.0)) > 0.0);
    }

    /// A wire box is exactly twelve segments, and its bounds are the box.
    #[test]
    fn wire_aabb_is_twelve_edges_at_the_bounds() {
        let mut g = Gizmos::new();
        g.wire_aabb(Vec3::splat(-1.0), Vec3::new(1.0, 2.0, 3.0), WHITE);
        assert_eq!(g.segment_count(), 12);
        let group = g.to_group(Vec3::new(20.0, 20.0, 20.0)).unwrap();
        let (min, max) = group.bounds().expect("finite bounds");
        for k in 0..3 {
            assert!((min[k] - (-1.0)).abs() < 0.2, "min {min:?}");
        }
        assert!((max[0] - 1.0).abs() < 0.2 && (max[1] - 2.0).abs() < 0.2);
        assert!((max[2] - 3.0).abs() < 0.2, "max {max:?}");
    }

    /// A wire circle lies in the plane its normal defines and every point is
    /// exactly `radius` from `center`.
    #[test]
    fn circle_points_sit_on_the_radius_in_plane() {
        let center = Vec3::new(3.0, 4.0, 5.0);
        let normal = Vec3::Y;
        let mut g = Gizmos::new();
        g.set_line_width(0.001);
        g.circle(center, normal, 2.0, 32, WHITE);
        let group = g.to_group(Vec3::new(100.0, 100.0, 100.0)).unwrap();
        for p in &group.positions {
            let v = Vec3::from(*p) - center;
            assert!((v.length() - 2.0).abs() < 0.01, "off radius: {v:?}");
            assert!(v.dot(normal).abs() < 0.05, "off plane: {v:?}");
        }
    }

    /// A grid on the XZ plane is flat in Y and closes on both axes.
    #[test]
    fn grid_stays_on_the_ground_plane() {
        let mut g = Gizmos::new();
        g.grid(Vec3::new(0.0, 1.0, 0.0), 2.0, 2, WHITE);
        // 5 lines each way.
        assert_eq!(g.segment_count(), 10);
        let group = g.to_group(Vec3::new(0.0, 50.0, 0.0)).unwrap();
        let (min, max) = group.bounds().unwrap();
        assert!(min[1] > 0.9 && max[1] < 1.1, "flat in Y: {min:?} {max:?}");
        assert!((max[0] - min[0] - 8.0).abs() < 0.2, "spans 4 cells");
    }

    /// Axis gizmo colors are the conventional red/green/blue and use arrows.
    #[test]
    fn axes_are_colored_arrows() {
        let mut g = Gizmos::new();
        g.axes(Vec3::ZERO, 1.0);
        assert_eq!(g.segment_count(), 3 * 5, "shaft + four head strokes");
        let colors: Vec<GizmoColor> = g.segments.iter().map(|s| s.ca).collect();
        assert_eq!(&colors[..5], &[AXIS_X; 5]);
        assert_eq!(&colors[5..10], &[AXIS_Y; 5]);
        assert_eq!(&colors[10..], &[AXIS_Z; 5]);
    }

    /// Empty and degenerate inputs draw nothing rather than emitting a group
    /// with a bad vertex.
    #[test]
    fn degenerate_input_draws_nothing() {
        let mut g = Gizmos::new();
        assert!(g.to_group(Vec3::ZERO).is_none(), "empty stays empty");
        g.line(Vec3::ZERO, Vec3::ZERO, WHITE);
        assert!(g.is_empty(), "zero-length segment dropped");
        g.line(Vec3::ZERO, Vec3::new(f32::NAN, 0.0, 0.0), WHITE);
        assert!(g.is_empty(), "non-finite endpoint dropped");
        g.line(Vec3::ZERO, Vec3::X, WHITE);
        g.set_line_width(0.0);
        assert!(g.to_group(Vec3::Z).is_none(), "zero width draws nothing");
    }

    /// Clearing drops the frame's primitives but keeps the width.
    #[test]
    fn clear_is_frame_scoped() {
        let mut g = Gizmos::new();
        g.set_line_width(0.5);
        g.line(Vec3::ZERO, Vec3::X, WHITE);
        g.clear();
        assert!(g.is_empty());
        assert_eq!(g.line_width(), 0.5);
    }
}
