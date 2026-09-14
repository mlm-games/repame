//! Mesh snapshot: CPU-owned geometry uploaded per frame.
//! Lighting and textures are opt-in per group.
//! Normals off = flat tint. Uvs off = no texture sample.

use glam::Vec3;

/// Linear RGB triplets.
pub type Rgb = [f32; 3];

/// Surface material for lit groups. Flat groups ignore it.
#[derive(Clone, Copy, Debug)]
pub struct Material {
    /// 0 = dielectric, 1 = metal. Kills diffuse at 1.
    pub metallic: f32,
    /// 0 = mirror, 1 = matte. Spec lobe narrows as this drops.
    pub roughness: f32,
    /// Added unlit on top of lit result. May exceed 1.0.
    pub emissive: Rgb,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            metallic: 0.0,
            roughness: 1.0,
            emissive: [0.0, 0.0, 0.0],
        }
    }
}

/// One draw group: indexed triangles in world space with per-vertex tint.
///
/// Normals opt in to lighting. Must match positions when present.
/// Uvs opt in to texture. Must match positions when present.
/// One material and one texture page per group.
#[derive(Clone, Debug)]
pub struct MeshGroup {
    /// World-space positions, Y-up right-handed.
    pub positions: Vec<[f32; 3]>,
    /// Per-vertex tint. With normals this is albedo. Texel multiplies first.
    /// glTF COLOR_0 folds in here at import.
    ///
    /// Hazard: uvs without an uploaded page sample empty texels
    /// (transparent black that discards). Upload the page or strip uvs.
    pub colors: Vec<[f32; 3]>,
    /// Per-vertex normals, unit length, world space. Empty = unlit.
    pub normals: Vec<[f32; 3]>,
    /// Per-vertex uvs, 0..1, y-down. Empty = untextured.
    pub uvs: Vec<[f32; 2]>,
    /// Texture array layer sampled when `uvs` is non-empty.
    pub texture_page: u32,
    /// Document image behind base-color texture. None = untextured.
    /// Textured import resolves this to `texture_page` after upload.
    pub base_image: Option<usize>,
    /// Pick id for CPU ray picking. 0 = skipped.
    pub pick_id: u32,
    /// Alpha-blend pass when true, opaque when false.
    pub transparent: bool,
    /// Group alpha multiplier, 0..1, default 1.0.
    pub alpha: f32,
    /// Alpha cutoff, default 0.0 = keep all. Discards below it.
    pub alpha_cutoff: f32,
    /// Surface material (lit groups only; flat groups ignore it).
    pub material: Material,
    /// Indices into positions/colors/normals/uvs.
    pub indices: Vec<u32>,
    /// Opaque geometry occludes when true. False draws on top.
    pub depth_test: bool,
}

impl Default for MeshGroup {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            colors: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            texture_page: 0,
            base_image: None,
            pick_id: 0,
            transparent: false,
            alpha: 1.0,
            alpha_cutoff: 0.0,
            material: Material::default(),
            indices: Vec::new(),
            depth_test: false,
        }
    }
}

impl MeshGroup {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty() || self.positions.is_empty()
    }

    pub fn tri_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// AABB as (min, max). None when empty or non-finite.
    /// Used by frustum cull and picking.
    pub fn bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        let mut verts = self.positions.iter();
        let first = Vec3::from(*verts.next()?);
        let mut min = first;
        let mut max = first;
        for p in verts {
            let v = Vec3::from(*p);
            min = min.min(v);
            max = max.max(v);
        }
        if min.is_finite() && max.is_finite() && min.x <= max.x && min.y <= max.y && min.z <= max.z
        {
            Some((min.into(), max.into()))
        } else {
            None
        }
    }

    /// Push triangle, CCW from outside. No normals, no uvs.
    /// Use one push style per group; mixed styles drop at batch time.
    pub fn push_tri(&mut self, a: [f32; 3], b: [f32; 3], c: [f32; 3], color: Rgb) {
        let base = self.positions.len() as u32;
        self.positions.extend_from_slice(&[a, b, c]);
        self.colors.extend_from_slice(&[color, color, color]);
        self.indices.extend_from_slice(&[base, base + 1, base + 2]);
    }

    /// Push textured triangle, CCW from outside. No normals.
    pub fn push_tri_textured(
        &mut self,
        a: [f32; 3],
        b: [f32; 3],
        c: [f32; 3],
        color: Rgb,
        uvs: [[f32; 2]; 3],
    ) {
        let base = self.positions.len() as u32;
        self.positions.extend_from_slice(&[a, b, c]);
        self.colors.extend_from_slice(&[color, color, color]);
        self.uvs.extend_from_slice(&uvs);
        self.indices.extend_from_slice(&[base, base + 1, base + 2]);
    }

    /// Push triangle with face normal. Unit length, world space.
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

    /// Push triangle with face normal and uvs. For lit textured groups.
    pub fn push_tri_lit_textured(
        &mut self,
        a: [f32; 3],
        b: [f32; 3],
        c: [f32; 3],
        color: Rgb,
        normal: [f32; 3],
        uvs: [[f32; 2]; 3],
    ) {
        let base = self.positions.len() as u32;
        self.positions.extend_from_slice(&[a, b, c]);
        self.colors.extend_from_slice(&[color, color, color]);
        self.normals.extend_from_slice(&[normal, normal, normal]);
        self.uvs.extend_from_slice(&uvs);
        self.indices.extend_from_slice(&[base, base + 1, base + 2]);
    }

    /// Push quad as (a, b, c) + (a, c, d).
    pub fn push_quad(&mut self, a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3], color: Rgb) {
        self.push_tri(a, b, c, color);
        self.push_tri(a, c, d, color);
    }

    /// Push unlit textured quad. Uvs in corner order a/b/c/d.
    #[allow(clippy::too_many_arguments)] // same shape as `push_quad_lit_textured`
    pub fn push_quad_textured(
        &mut self,
        a: [f32; 3],
        b: [f32; 3],
        c: [f32; 3],
        d: [f32; 3],
        color: Rgb,
        uvs: [[f32; 2]; 4],
    ) {
        self.push_tri_textured(a, b, c, color, [uvs[0], uvs[1], uvs[2]]);
        self.push_tri_textured(a, c, d, color, [uvs[0], uvs[2], uvs[3]]);
    }

    /// Push lit quad as two `push_tri_lit` triangles.
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

    /// Push lit textured quad. Uvs in corner order a/b/c/d.
    #[allow(clippy::too_many_arguments)] // quad spec is position + tint + normal + uvs
    pub fn push_quad_lit_textured(
        &mut self,
        a: [f32; 3],
        b: [f32; 3],
        c: [f32; 3],
        d: [f32; 3],
        color: Rgb,
        normal: [f32; 3],
        uvs: [[f32; 2]; 4],
    ) {
        self.push_tri_lit_textured(a, b, c, color, normal, [uvs[0], uvs[1], uvs[2]]);
        self.push_tri_lit_textured(a, c, d, color, normal, [uvs[0], uvs[2], uvs[3]]);
    }

    /// Push axis-aligned shaded box. Emits camera-facing planes only.
    #[allow(clippy::too_many_arguments)] // box spec is position + size + tint + eye
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

/// Face shade factor: top 1.0, sides graded, bottom dim.
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
        // Top + one facing side: 2 quads = 4 tris.
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

    #[test]
    fn textured_tris_carry_matching_uvs_and_page() {
        let mut g = MeshGroup {
            texture_page: 2,
            depth_test: true,
            ..Default::default()
        };
        g.push_quad_textured(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        );
        assert_eq!(g.tri_count(), 2);
        assert_eq!(g.uvs.len(), g.positions.len());
        assert_eq!(g.normals.len(), 0);
        assert_eq!(g.texture_page, 2);
        assert_eq!(g.uvs[0], [0.0, 0.0]);
        assert_eq!(g.uvs[2], [1.0, 1.0]);
    }

    #[test]
    fn lit_textured_tris_carry_both() {
        let mut g = MeshGroup {
            depth_test: true,
            ..Default::default()
        };
        g.push_quad_lit_textured(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        );
        assert_eq!(g.tri_count(), 2);
        assert_eq!(g.uvs.len(), g.positions.len());
        assert_eq!(g.normals.len(), g.positions.len());
    }
}
