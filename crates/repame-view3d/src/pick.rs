//! CPU ray picking over mesh snapshots: ray in, nearest pickable hit out.
//!
//! The GPU path stays untouched: picking runs on the [`MeshGroup`](super::mesh::MeshGroup)
//! snapshot the game already built for the frame, through the same
//! [`OrbitCamera`](super::camera::OrbitCamera) the renderer used, so content
//! and picks cannot disagree. Groups opt in with
//! [`MeshGroup::pick_id`](super::mesh::MeshGroup::pick_id) (`0` = skipped:
//! leave bulk terrain unpickable and use ground-plane picks for it).
//!
//! Broadphase is a per-group AABB, recomputed per call. That is linear in
//! the pickable vertex count per pointer event — fine at starter/editor
//! scale, and bulk groups stay free by staying unpickable. A cached spatial
//! index can slot in behind [`pick_ray`] later without changing callers.
//! Narrow phase is backface-culled Möller–Trumbore, matching the renderer's
//! `FrontFace::Ccw` + `CullMode::Back`: only drawn faces pick.

use glam::{Vec2, Vec3};

use super::camera::OrbitCamera;
use super::mesh::MeshGroup;

/// Nearest pickable hit along a ray.
#[derive(Clone, Copy, Debug)]
pub struct MeshHit {
    /// The hit group's pick id.
    pub pick_id: u32,
    /// Index into the groups slice (submission order).
    pub group: usize,
    /// Triangle index within the group (0-based, `indices.len() / 3` space).
    pub tri: u32,
    /// Ray distance from the origin to the hit point (in `dir` units —
    /// world units when `dir` is normalized, as [`pick_screen`] passes).
    pub distance: f32,
    /// World-space hit point.
    pub point: [f32; 3],
    /// Unit face normal at the hit (world space).
    pub normal: [f32; 3],
}

/// AABB of a group's positions, or `None` when empty or non-finite.
pub fn group_bounds(group: &MeshGroup) -> Option<(Vec3, Vec3)> {
    group
        .bounds()
        .map(|(min, max)| (Vec3::from(min), Vec3::from(max)))
}

/// Slab test: does the ray touch the box? Axis-parallel rays (near-zero
/// direction components) test containment on that axis instead.
pub fn ray_aabb(origin: Vec3, dir: Vec3, min: Vec3, max: Vec3) -> bool {
    let mut tmin = 0.0f32;
    let mut tmax = f32::INFINITY;
    for i in 0..3 {
        let o = origin[i];
        let d = dir[i];
        if d.abs() < 1e-12 {
            if o < min[i] || o > max[i] {
                return false;
            }
        } else {
            let inv = 1.0 / d;
            let (mut t0, mut t1) = ((min[i] - o) * inv, (max[i] - o) * inv);
            if t0 > t1 {
                std::mem::swap(&mut t0, &mut t1);
            }
            tmin = tmin.max(t0);
            tmax = tmax.min(t1);
            if tmin > tmax {
                return false;
            }
        }
    }
    true
}

/// Backface-culled Möller–Trumbore: `Some((t, point, normal))` for a front
/// face hit with `t >= 0`, else `None`. Degenerate triangles miss; NaN
/// coordinates miss (every comparison goes false) — never panics.
pub fn ray_triangle(
    origin: Vec3,
    dir: Vec3,
    a: Vec3,
    b: Vec3,
    c: Vec3,
) -> Option<(f32, Vec3, Vec3)> {
    let e1 = b - a;
    let e2 = c - a;
    let cross = e1.cross(e2);
    let area2 = cross.length();
    if area2 <= 1e-12 {
        return None; // degenerate (also rejects NaN: comparison is false)
    }
    let normal = cross / area2;
    // Match the renderer (CCW front, backfaces culled): front faces oppose
    // the ray. A coplanar ray (dot == 0) hits nothing.
    if normal.dot(dir) >= 0.0 {
        return None;
    }
    let p = dir.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-12 {
        return None; // parallel (post-cull: grazing)
    }
    let inv = 1.0 / det;
    let to_origin = origin - a;
    let u = to_origin.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = to_origin.cross(e1);
    let v = dir.dot(q) * inv;
    if !(0.0..=1.0).contains(&v) || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv;
    if t < 0.0 {
        return None;
    }
    Some((t, origin + dir * t, normal))
}

/// Nearest pickable hit along `origin` + `dir`. Groups with `pick_id == 0`
/// are skipped outright (no bounds cost); out-of-range indices are skipped
/// tri-by-tri, mirroring the batch validator. `depth_test` is ignored: ray
/// order decides, so overlay gizmos stay clickable.
pub fn pick_ray(origin: Vec3, dir: Vec3, groups: &[MeshGroup]) -> Option<MeshHit> {
    let mut best: Option<MeshHit> = None;
    for (gi, group) in groups.iter().enumerate() {
        if group.pick_id == 0 || group.indices.len() < 3 {
            continue;
        }
        let Some((bmin, bmax)) = group_bounds(group) else {
            continue;
        };
        if !ray_aabb(origin, dir, bmin, bmax) {
            continue;
        }
        let count = group.positions.len();
        let (chunks, _) = group.indices.as_chunks::<3>();
        for (ti, tri) in chunks.iter().enumerate() {
            let (ia, ib, ic) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            if ia >= count || ib >= count || ic >= count {
                continue;
            }
            let a = Vec3::from(group.positions[ia]);
            let b = Vec3::from(group.positions[ib]);
            let c = Vec3::from(group.positions[ic]);
            let Some((t, point, normal)) = ray_triangle(origin, dir, a, b, c) else {
                continue;
            };
            if best.is_none_or(|hit: MeshHit| t < hit.distance) {
                best = Some(MeshHit {
                    pick_id: group.pick_id,
                    group: gi,
                    tri: ti as u32,
                    distance: t,
                    point: point.into(),
                    normal: normal.into(),
                });
            }
        }
    }
    best
}

/// Screen-space entry: unproject `px` through `cam` (the same camera the
/// GPU used: `cam.view_proj(aspect)`) and pick. `viewport_px` and `px` must
/// use the same units (either is fine, the division cancels out).
/// The ray starts at the near plane — closer-than-near geometry is clipped
/// on screen too, so picks and pixels agree.
pub fn pick_screen(
    cam: &OrbitCamera,
    aspect: f32,
    viewport_px: Vec2,
    px: Vec2,
    groups: &[MeshGroup],
) -> Option<MeshHit> {
    let (origin, dir) = cam.screen_ray(aspect, viewport_px, px);
    pick_ray(origin, dir, groups)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Up-facing triangle (CCW from above, +Y normal) at the origin.
    fn up_tri() -> (Vec3, Vec3, Vec3) {
        (
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(-1.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
        )
    }

    #[test]
    fn front_face_hits_with_distance_point_normal() {
        let (a, b, c) = up_tri();
        let (t, p, n) = ray_triangle(Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y, a, b, c)
            .expect("straight-down ray hits");
        assert!((t - 5.0).abs() < 1e-5, "t = {t}");
        assert!((p - Vec3::ZERO).length() < 1e-5, "p = {p}");
        assert!((n - Vec3::Y).length() < 1e-6, "n = {n}");
    }

    #[test]
    fn backface_misses() {
        let (a, b, c) = up_tri();
        // Same triangle wound the other way: normal follows the ray.
        assert!(ray_triangle(Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y, a, c, b).is_none());
        // Same winding, ray from below strikes the back: miss.
        assert!(ray_triangle(Vec3::new(0.0, -5.0, 0.0), Vec3::Y, a, b, c).is_none());
    }

    #[test]
    fn degenerate_and_nan_miss_without_panic() {
        let (a, b, c) = up_tri();
        let z = Vec3::ZERO;
        assert!(ray_triangle(z, Vec3::NEG_Y, z, z, z).is_none());
        assert!(ray_triangle(Vec3::NAN, Vec3::NEG_Y, a, b, c).is_none());
        let nan = Vec3::NAN;
        assert!(ray_triangle(Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y, nan, nan, nan).is_none());
        assert!(ray_triangle(Vec3::new(0.0, 5.0, 0.0), Vec3::NAN, a, b, c).is_none());
    }

    #[test]
    fn aabb_handles_parallel_rays() {
        let min = Vec3::new(-1.0, -1.0, -1.0);
        let max = Vec3::ONE;
        assert!(ray_aabb(Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y, min, max));
        // Parallel to Y but outside XZ: miss.
        assert!(!ray_aabb(Vec3::new(5.0, 5.0, 0.0), Vec3::NEG_Y, min, max));
        // Parallel to Y, inside XZ: hit.
        assert!(ray_aabb(Vec3::new(0.5, 5.0, 0.0), Vec3::NEG_Y, min, max));
        // Origin inside the box: hit.
        assert!(ray_aabb(Vec3::ZERO, Vec3::X, min, max));
    }

    /// CCW-from-above quad at height `y` (same corner order as the
    /// renderer's depth-test fixtures).
    fn sheet(id: u32, y: f32, half: f32) -> MeshGroup {
        let mut g = MeshGroup {
            pick_id: id,
            depth_test: true,
            ..Default::default()
        };
        g.push_quad(
            [-half, y, half],
            [half, y, half],
            [half, y, -half],
            [-half, y, -half],
            [1.0, 1.0, 1.0],
        );
        g
    }

    #[test]
    fn nearest_pickable_wins_and_zero_id_is_invisible() {
        let groups = vec![
            sheet(1, 0.0, 8.0), // far, pushed first
            sheet(2, 2.0, 8.0), // near
            sheet(0, 4.0, 8.0), // unpickable, floating between eye and near
        ];
        let hit = pick_ray(Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y, &groups).expect("hits near");
        assert_eq!(hit.pick_id, 2);
        assert!((hit.distance - 3.0).abs() < 1e-5, "d = {}", hit.distance);
        assert_eq!(hit.group, 1);
        assert!((Vec3::from(hit.point) - Vec3::new(0.0, 2.0, 0.0)).length() < 1e-4);
    }

    #[test]
    fn empty_and_miss_return_none() {
        assert!(pick_ray(Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y, &[]).is_none());
        let groups = vec![sheet(1, 0.0, 8.0)];
        // Offset ray parallel to Y but outside the sheet: AABB rejects.
        assert!(pick_ray(Vec3::new(50.0, 5.0, 0.0), Vec3::NEG_Y, &groups).is_none());
        // Upward ray: backfaces, and nothing ahead.
        assert!(pick_ray(Vec3::new(0.0, 5.0, 0.0), Vec3::Y, &groups).is_none());
    }

    #[test]
    fn screen_roundtrip_hits_pickable_box() {
        use super::super::camera::OrbitCamera;
        let cam = OrbitCamera {
            target: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.9,
            dist: 30.0,
            fov_y_deg: 30.0,
        };
        let mut g = MeshGroup {
            pick_id: 9,
            depth_test: true,
            ..Default::default()
        };
        g.push_box(0.0, 0.0, 0.0, 20.0, 4.0, 20.0, [1.0, 1.0, 1.0], cam.eye());
        let vp = Vec2::new(800.0, 600.0);
        let hit = pick_screen(&cam, 800.0 / 600.0, vp, Vec2::new(400.0, 300.0), &[g])
            .expect("center ray hits the box top");
        assert_eq!(hit.pick_id, 9);
        assert!(
            (hit.point[1] - 4.0).abs() < 1e-3,
            "top face: {:?}",
            hit.point
        );
    }
}
