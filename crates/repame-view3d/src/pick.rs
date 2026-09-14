//! CPU ray picking over mesh snapshot.
//! Uses the same camera as render, so content and picks agree.
//! Groups opt in with `pick_id` (0 = skipped).
//! Broadphase is per-group AABB. Narrow phase is backface-culled Moller-Trumbore.

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
    /// Triangle index within group.
    pub tri: u32,
    /// Ray distance to hit. World units when dir is normalized.
    pub distance: f32,
    /// World-space hit point.
    pub point: [f32; 3],
    /// Unit face normal at the hit (world space).
    pub normal: [f32; 3],
}

/// AABB of group positions. None when empty or non-finite.
pub fn group_bounds(group: &MeshGroup) -> Option<(Vec3, Vec3)> {
    group
        .bounds()
        .map(|(min, max)| (Vec3::from(min), Vec3::from(max)))
}

/// Slab test. Near-zero dir components test containment on that axis.
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

/// Backface-culled Moller-Trumbore. Degenerate and NaN tris miss.
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
    // CCW front, backfaces culled. Coplanar ray hits nothing.
    if normal.dot(dir) >= 0.0 {
        return None;
    }
    let p = dir.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-12 {
        return None;
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

/// Nearest pickable hit along ray. Skips pick_id 0 and bad indices.
/// Ignores depth_test: ray order decides.
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

/// Screen-space pick: unproject `px` through `cam`, then `pick_ray`.
/// Ray starts at near plane. Units cancel in the division.
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

    /// Up triangle, CCW from above.
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
        // Reversed winding: normal follows ray.
        assert!(ray_triangle(Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y, a, c, b).is_none());
        // Ray from below hits back: miss.
        assert!(ray_triangle(Vec3::new(0.0, -5.0, 0.0), Vec3::Y, a, b, c).is_none());
    }

    #[test]
    fn degenerate_and_nan_miss() {
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

    /// Quad at height y, CCW from above.
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
    fn nearest_pickable_wins_and_zero_id_skips() {
        let groups = vec![sheet(1, 0.0, 8.0), sheet(2, 2.0, 8.0), sheet(0, 4.0, 8.0)];
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
        assert!(pick_ray(Vec3::new(50.0, 5.0, 0.0), Vec3::NEG_Y, &groups).is_none());
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
