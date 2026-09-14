//! Mesh snapshots: CPU-owned geometry the renderer uploads per frame.
//!
//! Long-term containers (glTF import, skinning, chunk meshing with dirty
//! tracking) plug in behind these types; the renderer only ever sees
//! vertex/index/tint/normal lists, so the GPU path stays stable while the
//! asset side grows.
//!
//! Lighting is opt-in per group: groups without normals draw flat
//! (backwards-compatible with the original flat path); groups with normals
//! are shaded by the frame's [`SceneLight`](crate::SceneLight) as
//! `base * (ambient + diffuse * max(dot(N, L), 0))`.

use glam::Vec3;

/// Linear-space RGB triplets (authored flat, output raw).
pub type Rgb = [f32; 3];

/// One draw group: indexed triangles in world space with a per-vertex
/// tint. Games rebuild these per frame from their sim state (see
/// [`Frame3d::push`](crate::Frame3d::push)); chunked/voxel worlds submit
/// one group per material and keep the lists across frames.
///
/// Normals are optional: when `normals` is empty the group draws flat
/// (legacy path). When present it must match `positions` in length, and
/// the frame's [`SceneLight`](crate::SceneLight) shades the group.
#[derive(Clone, Debug, Default)]
pub struct MeshGroup {
    /// World-space positions, Y-up right-handed.
    pub positions: Vec<[f32; 3]>,
    /// Per-vertex tint (linear RGB). Without normals this is the final
    /// color (shading already baked per face by the producer, see
    /// [`shade_for_dir`]); with normals it is the albedo the light
    /// modulates.
    pub colors: Vec<[f32; 3]>,
    /// Per-vertex normals (unit length, world space). Empty = unlit.
    pub normals: Vec<[f32; 3]>,
    /// Triangle indices into `positions` / `colors` / `normals`.
    pub indices: Vec<u32>,
    /// Opaque geometry occludes (`true`) or always draws (`false`, e.g.
    /// flat ground overlays and editor gizmo quads that must stay visible
    /// under grazing angles).
    pub depth_test: bool,
}

impl MeshGroup {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty() || self.positions.is_empty()
    }

    pub fn tri_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Push one triangle (counter-clockwise when viewed from outside).
    pub fn push_tri(&mut self, a: [f32; 3], b: [f32; 3], c: [f32; 3], color: Rgb) {
        let base = self.positions.len() as u32;
        self.positions.extend_from_slice(&[a, b, c]);
        self.colors.extend_from_slice(&[color, color, color]);
        self.indices.extend_from_slice(&[base, base + 1, base + 2]);
    }

    /// Push one triangle with an explicit face normal (unit length, world
    /// space). Groups mixing `push_tri` and `push_tri_lit` are dropped at
    /// batch time (all-or-nothing normals), so pick one per group.
    pub fn push_tri_lit(
        &mut self,
        a: [f32; 3],
        b: [f32; 3],
        c: [f32; 3],
        color: Rgb,
        normal: [f32; 3],
    ) {
        let base = self.positions.len() as u32;
        self.positions.extend_from_slice(&[a, b, c]);
        self.colors.extend_from_slice(&[color, color, color]);
        self.normals.extend_from_slice(&[normal, normal, normal]);
        self.indices.extend_from_slice(&[base, base + 1, base + 2]);
    }

    /// Push one quad as two triangles (a, b, c) + (a, c, d).
    pub fn push_quad(&mut self, a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3], color: Rgb) {
        self.push_tri(a, b, c, color);
        self.push_tri(a, c, d, color);
    }

    /// Push one lit quad as two `push_tri_lit` triangles.
    pub fn push_quad_lit(
        &mut self,
        a: [f32; 3],
        b: [f32; 3],
        c: [f32; 3],
        d: [f32; 3],
        color: Rgb,
        normal: [f32; 3],
    ) {
        self.push_tri_lit(a, b, c, color, normal);
        self.push_tri_lit(a, c, d, color, normal);
    }

    /// Push an axis-aligned shaded box centered at (`cx`, base `y0`, `cz`),
    /// only emitting camera-facing planes (winding-independent, analytic).
    /// Matches the resims starter-scene boxes: top at full tint, sides
    /// shaded (see [`shade_for_dir`]).
    #[allow(clippy::too_many_arguments)] // box spec is position + size + tint + eye, one struct would churn call sites
    pub fn push_box(
        &mut self,
        cx: f32,
        y0: f32,
        cz: f32,
        w: f32,
        h: f32,
        d: f32,
        color: Rgb,
        eye: Vec3,
    ) {
        let x0 = cx - w / 2.0;
        let x1 = cx + w / 2.0;
        let z0 = cz - d / 2.0;
        let z1 = cz + d / 2.0;
        let y1 = y0 + h;
        let f = |dir: Vec3, pt: Vec3| (eye - pt).dot(dir) > 0.02 * (eye - pt).length().max(1e-6);
        if f(Vec3::Y, Vec3::new(cx, y1, cz)) {
            self.push_quad(
                [x0, y1, z0],
                [x0, y1, z1],
                [x1, y1, z1],
                [x1, y1, z0],
                shade(color, [0, 1, 0]),
            );
        }
        if f(Vec3::X, Vec3::new(x1, y0 + h / 2.0, cz)) {
            self.push_quad(
                [x1, y0, z0],
                [x1, y0, z1],
                [x1, y1, z1],
                [x1, y1, z0],
                shade(color, [1, 0, 0]),
            );
        }
        if f(Vec3::NEG_X, Vec3::new(x0, y0 + h / 2.0, cz)) {
            self.push_quad(
                [x0, y0, z1],
                [x0, y0, z0],
                [x0, y1, z0],
                [x0, y1, z1],
                shade(color, [-1, 0, 0]),
            );
        }
        if f(Vec3::Z, Vec3::new(cx, y0 + h / 2.0, z1)) {
            self.push_quad(
                [x1, y0, z1],
                [x0, y0, z1],
                [x0, y1, z1],
                [x1, y1, z1],
                shade(color, [0, 0, 1]),
            );
        }
        if f(Vec3::NEG_Z, Vec3::new(cx, y0 + h / 2.0, z0)) {
            self.push_quad(
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y1, z0],
                [x0, y1, z0],
                shade(color, [0, 0, -1]),
            );
        }
    }
}

/// Directional shade baked per face: top full, sides graded, bottom dim.
/// (No sunlight sim yet; contrast is data, not material hacks.)
pub fn shade_for_dir(dir: [i32; 3]) -> f32 {
    if dir == [0, 1, 0] {
        1.0
    } else if dir == [0, -1, 0] {
        0.55
    } else if dir[0] != 0 {
        0.8
    } else {
        0.7
    }
}

/// Shade a linear color for a face direction.
pub fn shade(color: Rgb, dir: [i32; 3]) -> Rgb {
    let s = shade_for_dir(dir);
    [color[0] * s, color[1] * s, color[2] * s]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_emits_only_facing_planes() {
        let mut g = MeshGroup {
            depth_test: true,
            ..Default::default()
        };
        let eye = Vec3::new(0.0, 10.0, 20.0);
        g.push_box(0.0, 0.0, 0.0, 2.0, 2.0, 2.0, [1.0, 1.0, 1.0], eye);
        // Top + the one facing side (+Z toward the eye at +z); ±X are
        // edge-on and -Z faces away: 2 quads = 4 tris.
        assert_eq!(g.tri_count(), 4, "got {} tris", g.tri_count());
        assert!(g.positions.iter().all(|p| p.iter().all(|v| v.is_finite())));
    }

    #[test]
    fn shade_grades_top_side_bottom() {
        assert_eq!(shade_for_dir([0, 1, 0]), 1.0);
        assert!(shade_for_dir([1, 0, 0]) > shade_for_dir([0, 0, 1]));
        assert!(shade_for_dir([0, -1, 0]) < 0.6);
    }

    #[test]
    fn lit_tris_carry_matching_normals() {
        let mut g = MeshGroup {
            depth_test: true,
            ..Default::default()
        };
        g.push_quad_lit(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        );
        assert_eq!(g.tri_count(), 2);
        assert_eq!(g.normals.len(), g.positions.len());
        assert!(g.normals.iter().all(|n| *n == [0.0, 1.0, 0.0]));
    }
}
