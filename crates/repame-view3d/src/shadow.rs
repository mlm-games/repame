//! GPU shadow maps: one directional-light depth pass behind [`MeshGroup`].
//!
//! Two rigs live here: cascaded directional shadows (up to
//! [`MAX_CASCADES`] ortho slices fitted to the camera frustum, texel-snapped
//! so the shadows don't shimmer while the camera moves) and point-light
//! cube shadows (one shadow-casting point light renders its surroundings
//! into a distance cube; the rest shade unshadowed). The batch in
//! [`SceneBatch`](crate::SceneBatch) owns the textures and passes; this
//! module owns the math plus the plain-data descriptors.

use glam::{Mat4, Vec3};

use super::camera::OPENGL_TO_WGPU;

/// Max directional cascades in one frame. Four 1024px cascades cost four
/// depth passes; games on weak GPUs set `count` to 1 (identical to the
/// legacy single map) or 2.
pub const MAX_CASCADES: usize = 4;

/// Max point lights evaluated in the scene shader. Extra lights submitted
/// on [`LightRig`](crate::LightRig) are dropped with a warning, brightest
/// (highest color luminance times range) kept.
pub const MAX_POINTS: usize = 8;

/// Max skinned joints per GPU draw. glTF files with more joints fall back
/// to the CPU [`pose`](crate::SkinnedMesh::pose) path (the batch checks and
/// warns); 128 joints is 8 KiB of palette uniforms per draw.
pub const MAX_SKIN_JOINTS: usize = 128;

/// Shadow-casting configuration. `None` on [`Frame3d`](crate::Frame3d)
/// disables the depth pass and reproduces legacy pixels.
#[derive(Clone, Copy, Debug)]
pub struct ShadowDesc {
    /// Depth texture edge in texels. Clamped to `64..=4096` on use.
    pub size: u32,
    /// Constant depth bias in light NDC depth. Clamped to `0..=0.05`.
    /// Too low acne-prone, too high floats shadows (peter-panning).
    pub bias: f32,
    /// How fully shadows darken: 1.0 = physical, 0.0 = no visible shadow
    /// (pass still runs; use `None` to skip it). Clamped to `0..=1`.
    pub strength: f32,
}

impl Default for ShadowDesc {
    fn default() -> Self {
        Self {
            size: 1024,
            bias: 0.001,
            strength: 1.0,
        }
    }
}

impl ShadowDesc {
    /// Texture edge after clamping (what the batch actually allocates).
    pub fn clamped_size(&self) -> u32 {
        self.size.clamp(64, 4096)
    }

    /// Bias after clamping.
    pub fn clamped_bias(&self) -> f32 {
        if self.bias.is_finite() {
            self.bias.clamp(0.0, 0.05)
        } else {
            0.001
        }
    }

    /// Strength after clamping.
    pub fn clamped_strength(&self) -> f32 {
        if self.strength.is_finite() {
            self.strength.clamp(0.0, 1.0)
        } else {
            1.0
        }
    }

    /// Texel size in uv units for the PCF kernel.
    pub fn texel(&self) -> f32 {
        1.0 / self.clamped_size() as f32
    }
}

/// Ortho half-extent in world units for a camera at `dist`.
/// Covers the view around the target with margin; clamped so close-ups keep
/// texel density and far views keep finite depth precision.
/// Legacy single-map path ([`SceneBatch::set_shadow`](crate::SceneBatch::set_shadow)).
pub fn shadow_extent(dist: f32) -> f32 {
    if dist.is_finite() {
        (dist * 0.6).clamp(25.0, 500.0)
    } else {
        110.0
    }
}

/// Light-space view-projection for the shadow pass: orthographic box of
/// half-`extent` around `center`, looking along `-dir` (dir points toward
/// the light, same convention as [`SceneLight`](crate::SceneLight)).
///
/// Near/far bracket the box (`eye` sits `2 * extent` out, planes at
/// `extent` and `3 * extent`): content inside the box maps to depth
/// `0..1`, content outside clamps. Degenerate directions fall back to +Y
/// (same fallback as [`SceneBatch::set_light`](crate::SceneBatch::set_light)).
pub fn light_view_proj(center: Vec3, dir: Vec3, extent: f32) -> Mat4 {
    let extent = if extent.is_finite() {
        extent.clamp(1.0, 2000.0)
    } else {
        110.0
    };
    let d = if dir.length_squared() > 1e-8 {
        dir.normalize()
    } else {
        Vec3::Y
    };
    let eye = center + d * extent * 2.0;
    let view = glam::camera::rh::view::look_at_mat4(eye, center, Vec3::Y);
    let view = if view.is_finite() {
        view
    } else {
        glam::camera::rh::view::look_at_mat4(eye, center, Vec3::Z)
    };
    let proj = ortho_gl(-extent, extent, -extent, extent, extent, extent * 3.0);
    OPENGL_TO_WGPU * proj * view
}

/// Right-handed OpenGL-style orthographic projection (NDC depth `-1..1`,
/// remapped to wgpu `0..1` by the caller via [`OPENGL_TO_WGPU`]).
fn ortho_gl(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Mat4 {
    let tx = -(right + left) / (right - left);
    let ty = -(top + bottom) / (top - bottom);
    let tz = -(far + near) / (far - near);
    Mat4::from_cols(
        glam::Vec4::new(2.0 / (right - left), 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, 2.0 / (top - bottom), 0.0, 0.0),
        glam::Vec4::new(0.0, 0.0, -2.0 / (far - near), 0.0),
        glam::Vec4::new(tx, ty, tz, 1.0),
    )
}

/// Normalize a light direction with the batch fallback (+Y on degenerate).
fn norm_dir(dir: Vec3) -> Vec3 {
    if dir.length_squared() > 1e-8 {
        dir.normalize()
    } else {
        Vec3::Y
    }
}

/// One cascade slice of a directional shadow rig: light-space matrix plus
/// the camera depth range it covers (view-space distance from the eye).
#[derive(Clone, Copy, Debug)]
pub struct CascadeSlice {
    /// Light view-projection for this slice's ortho box.
    pub view_proj: Mat4,
    /// Slice near/far in camera view distance (eye-space, world units).
    pub near: f32,
    /// Slice far in camera view distance.
    pub far: f32,
    /// Ortho half-extent actually used (post-snap, for texel math).
    pub extent: f32,
    /// Snap texel in uv units (`1 / size`), echoed for the uniform.
    pub texel: f32,
}

/// Cascaded directional shadow rig. `count` slices split `near..far`
/// (log-uniform blend by `lambda`, 0 = uniform, 1 = logarithmic); each
/// slice fits its own ortho box to the camera-frustum corners in that
/// depth range, snapped to whole texels so static geometry doesn't
/// shimmer when the camera drifts.
#[derive(Clone, Copy, Debug)]
pub struct CascadeDesc {
    /// Slice count, clamped to `1..=MAX_CASCADES` on use.
    pub count: usize,
    /// Split blend: 0 = uniform slices, 1 = fully logarithmic.
    pub lambda: f32,
    /// Depth texture edge per cascade. Clamped to `64..=2048` on use
    /// (four 2048px cascades is the sane ceiling; above that the batch
    /// warns and clamps).
    pub size: u32,
    /// Constant depth bias in light NDC depth. Clamped to `0..=0.05`.
    pub bias: f32,
    /// Normal-offset (slope-scale-ish world push along the normal)
    /// in world units. Clamped to `0..=1`. Kills acne on steep faces
    /// where a constant bias alone floats.
    pub normal_bias: f32,
    /// How fully shadows darken: 1.0 = physical, 0.0 = no visible
    /// shadow (pass still runs; `None` on the frame skips it).
    pub strength: f32,
}

impl Default for CascadeDesc {
    fn default() -> Self {
        Self {
            count: 3,
            lambda: 0.75,
            size: 1024,
            bias: 0.001,
            normal_bias: 0.05,
            strength: 1.0,
        }
    }
}

impl CascadeDesc {
    /// Slice count after clamping.
    pub fn clamped_count(&self) -> usize {
        self.count.clamp(1, MAX_CASCADES)
    }

    /// Texture edge after clamping.
    pub fn clamped_size(&self) -> u32 {
        self.size.clamp(64, 2048)
    }

    /// Bias after clamping.
    pub fn clamped_bias(&self) -> f32 {
        if self.bias.is_finite() {
            self.bias.clamp(0.0, 0.05)
        } else {
            0.001
        }
    }

    /// Normal bias after clamping.
    pub fn clamped_normal_bias(&self) -> f32 {
        if self.normal_bias.is_finite() {
            self.normal_bias.clamp(0.0, 1.0)
        } else {
            0.05
        }
    }

    /// Strength after clamping.
    pub fn clamped_strength(&self) -> f32 {
        if self.strength.is_finite() {
            self.strength.clamp(0.0, 1.0)
        } else {
            1.0
        }
    }

    /// Texel size in uv units for the PCF kernel.
    pub fn texel(&self) -> f32 {
        1.0 / self.clamped_size() as f32
    }

    /// Split depths: `count + 1` entries from `near` to `far` (eye-space
    /// view distance). Log-uniform blend: `d = lambda * log + (1-lambda) *
    /// uniform`, the standard PSSM fit. Non-finite or non-positive-near
    /// inputs fall back to `1..100` (the log branch needs `near > 0`:
    /// `0 * inf` is NaN, so a zero near plane forces the fallback rather
    /// than poisoning every split).
    pub fn splits(&self, near: f32, far: f32) -> Vec<f32> {
        let (mut near, mut far) = (near, far);
        if !near.is_finite() || !far.is_finite() || far <= near || near <= 0.0 {
            near = 1.0;
            far = 100.0;
        }
        let n = self.clamped_count();
        let lambda = if self.lambda.is_finite() {
            self.lambda.clamp(0.0, 1.0)
        } else {
            0.75
        };
        let mut out = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let t = i as f32 / n as f32;
            let log = near * (far / near).powf(t);
            let uni = near + (far - near) * t;
            out.push(lambda * log + (1.0 - lambda) * uni);
        }
        out
    }

    /// Fit one cascade slice to the camera frustum corners between
    /// `slice_near` and `slice_far` (eye-space view distance). Builds the
    /// eight frustum corners in world space from the camera matrices, maps
    /// them into light space, and sizes the ortho box to the min/max,
    /// snapped down/up to whole texels (stable: sub-texel camera motion
    /// never moves the box).
    pub fn fit_slice(
        &self,
        cam_eye: Vec3,
        cam_view: Mat4,
        cam_inv_view_proj: Mat4,
        light_dir: Vec3,
        slice_near: f32,
        slice_far: f32,
    ) -> CascadeSlice {
        let dir = norm_dir(light_dir);
        let size = self.clamped_size() as f32;
        //
        let mut corners = Vec::with_capacity(8);
        for &x in &[-1.0f32, 1.0] {
            for &y in &[-1.0f32, 1.0] {
                for &z in &[0.0f32, 1.0f32] {
                    let p = cam_inv_view_proj.project_point3(Vec3::new(x, y, z));
                    corners.push(p);
                }
            }
        }
        let depths: Vec<f32> = corners
            .iter()
            .map(|p| {
                let v = cam_view.transform_point3(*p);
                (-v.z).max(0.0)
            })
            .collect();
        let mut slabbed = Vec::with_capacity(8);
        for (c, d) in corners.iter().zip(depths.iter()) {
            let t = (*d).clamp(slice_near, slice_far);
            let ray = (*c - cam_eye).normalize_or_zero();
            let eye_depth = |p: Vec3| -cam_view.transform_point3(p).z;
            let d0 = eye_depth(cam_eye);
            let d1 = eye_depth(*c);
            let s = if (d1 - d0).abs() > 1e-6 {
                ((t - d0) / (d1 - d0)).clamp(0.0, 1.0)
            } else {
                1.0
            };
            slabbed.push(cam_eye + ray * (c - cam_eye).length() * s);
        }
        let center = slabbed.iter().fold(Vec3::ZERO, |a, p| a + *p) / slabbed.len().max(1) as f32;
        let up = if dir.dot(Vec3::Y).abs() > 0.99 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        let eye = center + dir * 1000.0;
        let mut view = glam::camera::rh::view::look_at_mat4(eye, center, up);
        if !view.is_finite() {
            view = glam::camera::rh::view::look_at_mat4(eye, center, Vec3::Z);
        }
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for p in &slabbed {
            let q = view.transform_point3(*p);
            min = min.min(q);
            max = max.max(q);
        }
        if !min.is_finite() || !max.is_finite() {
            let vp = light_view_proj(center, dir, 110.0);
            return CascadeSlice {
                view_proj: vp,
                near: slice_near,
                far: slice_far,
                extent: 110.0,
                texel: self.texel(),
            };
        }
        let pad = 1.0f32;
        let (mut lx0, mut lx1) = (min.x - pad, max.x + pad);
        let (mut ly0, mut ly1) = (min.y - pad, max.y + pad);
        let extent = ((lx1 - lx0).max(ly1 - ly0) * 0.5)
            .clamp(1.0, 2000.0)
            .max(1.0);
        let world_texel = extent * 2.0 / size;
        if world_texel > 0.0 && world_texel.is_finite() {
            lx0 = (lx0 / world_texel).floor() * world_texel;
            lx1 = (lx1 / world_texel).ceil() * world_texel;
            ly0 = (ly0 / world_texel).floor() * world_texel;
            ly1 = (ly1 / world_texel).ceil() * world_texel;
        }
        let cx = (lx0 + lx1) * 0.5;
        let cy = (ly0 + ly1) * 0.5;
        let half = ((lx1 - lx0).max(ly1 - ly0) * 0.5).max(1.0);
        let lz0 = min.z - 50.0;
        let lz1 = max.z + 50.0;
        let (near, far) = (lz0.min(lz1), lz0.max(lz1));
        let proj = ortho_gl(cx - half, cx + half, cy - half, cy + half, -far, -near);
        let view_proj = OPENGL_TO_WGPU * proj * view;
        CascadeSlice {
            view_proj: if view_proj.is_finite() {
                view_proj
            } else {
                light_view_proj(center, dir, 110.0)
            },
            near: slice_near,
            far: slice_far,
            extent: half,
            texel: self.texel(),
        }
    }

    /// Fit all slices for a frame. `cam_near`/`cam_far` in eye-space view
    /// distance (pass `NEAR`/`FAR` unless the game narrows them).
    pub fn fit_all(
        &self,
        cam_eye: Vec3,
        cam_view: Mat4,
        cam_inv_view_proj: Mat4,
        light_dir: Vec3,
        cam_near: f32,
        cam_far: f32,
    ) -> Vec<CascadeSlice> {
        let splits = self.splits(cam_near, cam_far);
        let mut out = Vec::with_capacity(self.clamped_count());
        for w in splits.windows(2) {
            out.push(self.fit_slice(cam_eye, cam_view, cam_inv_view_proj, light_dir, w[0], w[1]));
        }
        out
    }
}

/// One shadow-casting point light: position, warm color, range, and size.
/// The batch renders the scene's opaque depth-tested groups into a
/// per-light distance cube (single pass, six faces via instanced
/// layer selection in the geometry stage... implemented as six depth-only
/// passes, one per face, since wgpu has no geometry shader), then the
/// scene shader samples the nearest face by dominant axis.
#[derive(Clone, Copy, Debug)]
pub struct PointLight {
    /// World-space position.
    pub position: [f32; 3],
    /// Light color, linear RGB (luminance picks survivors past MAX_POINTS).
    pub color: [f32; 3],
    /// Diffuse strength multiplier.
    pub intensity: f32,
    /// World-unit cutoff. Fragments past `range` get no contribution.
    /// Clamped to `1..=500` on use.
    pub range: f32,
    /// Depth cube edge in texels. Clamped to `64..=1024` on use.
    pub size: u32,
    /// Constant depth bias in cube NDC depth. Clamped to `0..=0.05`.
    pub bias: f32,
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            position: [0.0, 5.0, 0.0],
            color: [1.0, 0.9, 0.8],
            intensity: 1.0,
            range: 30.0,
            size: 512,
            bias: 0.005,
        }
    }
}

impl PointLight {
    /// Range after clamping.
    pub fn clamped_range(&self) -> f32 {
        if self.range.is_finite() {
            self.range.clamp(1.0, 500.0)
        } else {
            30.0
        }
    }

    /// Cube edge after clamping.
    pub fn clamped_size(&self) -> u32 {
        self.size.clamp(64, 1024)
    }

    /// Bias after clamping.
    pub fn clamped_bias(&self) -> f32 {
        if self.bias.is_finite() {
            self.bias.clamp(0.0, 0.05)
        } else {
            0.005
        }
    }

    /// Luminance-ish score for survivor selection (max channel of color
    /// times clamped intensity, NaN-safe).
    pub fn score(&self) -> f32 {
        let m = self.color[0].max(self.color[1]).max(self.color[2]);
        let i = if self.intensity.is_finite() {
            self.intensity.max(0.0)
        } else {
            0.0
        };
        if m.is_finite() { m.max(0.0) * i } else { 0.0 }
    }

    /// Six face view-projection matrices (+X, -X, +Y, -Y, +Z, -Z) with a
    /// 90-degree perspective (near 0.05, far = range). Near 0.05 (not
    /// 0.5): a 0.5 near plane blinds the light to everything within half
    /// a meter, and `clamped_range` floors at 1.0, so 0.5 is never
    /// degenerate here; the `None` case is a non-finite position.
    /// Returns `None` when the position is non-finite.
    pub fn cube_faces(&self) -> Option<[[[f32; 4]; 4]; 6]> {
        use glam::camera::rh::proj::opengl::perspective as perspective_gl;
        let range = self.clamped_range();
        let pos = Vec3::from(self.position);
        if !pos.is_finite() {
            return None;
        }
        let proj = OPENGL_TO_WGPU * perspective_gl(std::f32::consts::FRAC_PI_2, 1.0, 0.05, range);
        let faces = [
            (Vec3::X, Vec3::NEG_Y),
            (Vec3::NEG_X, Vec3::NEG_Y),
            (Vec3::Y, Vec3::NEG_Z),
            (Vec3::NEG_Y, Vec3::Z),
            (Vec3::Z, Vec3::NEG_Y),
            (Vec3::NEG_Z, Vec3::NEG_Y),
        ];
        let mut out = [[[0.0f32; 4]; 4]; 6];
        for (i, (fwd, up)) in faces.iter().enumerate() {
            let view = glam::camera::rh::view::look_at_mat4(pos, pos + *fwd, *up);
            let vp = (proj * view).to_cols_array_2d();
            out[i] = vp;
        }
        Some(out)
    }

    /// Dominant-axis face index for a fragment-to-light vector (matches
    /// the shader's face pick exactly, pinned by tests).
    pub fn face_for_dir(d: Vec3) -> usize {
        let ax = d.x.abs();
        let ay = d.y.abs();
        let az = d.z.abs();
        if ax >= ay && ax >= az {
            if d.x >= 0.0 { 0 } else { 1 }
        } else if ay >= ax && ay >= az {
            if d.y >= 0.0 { 2 } else { 3 }
        } else if d.z >= 0.0 {
            4
        } else {
            5
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box_corners(center: Vec3, half: f32) -> Vec<Vec3> {
        let mut out = Vec::with_capacity(8);
        for &x in &[-1.0, 1.0] {
            for &y in &[-1.0, 1.0] {
                for &z in &[-1.0, 1.0] {
                    out.push(center + Vec3::new(x * half, y * half, z * half));
                }
            }
        }
        out
    }

    /// NDC of a world point under a light matrix (None when behind the eye).
    fn ndc_of(vp: &Mat4, p: Vec3) -> Option<Vec3> {
        let clip = *vp * glam::Vec4::new(p.x, p.y, p.z, 1.0);
        if clip.w <= 0.0 {
            return None;
        }
        Some(Vec3::new(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w))
    }

    #[test]
    fn light_box_covers_its_center_content() {
        let center = Vec3::ZERO;
        let vp = light_view_proj(center, Vec3::new(0.3, 1.0, 0.2), 25.0);
        for c in box_corners(center, 10.0) {
            let ndc = ndc_of(&vp, c).expect("content in front of the light eye");
            assert!(
                ndc.x.abs() <= 1.0 && ndc.y.abs() <= 1.0,
                "inside ortho bounds: {c:?} -> {ndc:?}"
            );
            assert!(
                (0.0..=1.0).contains(&ndc.z),
                "depth remapped to wgpu range: {c:?} -> {ndc:?}"
            );
        }
    }

    #[test]
    fn degenerate_direction_falls_back_to_up() {
        let vp = light_view_proj(Vec3::ZERO, Vec3::ZERO, 25.0);
        assert!(vp.is_finite(), "fallback matrix is valid");
        let ndc = ndc_of(&vp, Vec3::ZERO).expect("origin visible");
        assert!(ndc.x.abs() <= 1.0 && ndc.y.abs() <= 1.0);
    }

    #[test]
    fn straight_down_light_stays_valid() {
        let vp = light_view_proj(Vec3::ZERO, Vec3::Y, 25.0);
        assert!(vp.is_finite(), "straight-down light is valid");
        let ndc = ndc_of(&vp, Vec3::ZERO).expect("origin visible");
        assert!((0.0..=1.0).contains(&ndc.z), "{ndc:?}");
    }

    #[test]
    fn outside_content_leaves_the_box() {
        let vp = light_view_proj(Vec3::ZERO, Vec3::Y, 25.0);
        let far = Vec3::new(500.0, 0.0, 0.0);
        let ndc = ndc_of(&vp, far).expect("in front");
        assert!(ndc.x.abs() > 1.0, "{ndc:?}");
    }

    #[test]
    fn extent_clamps_both_ends() {
        assert_eq!(shadow_extent(110.0), 66.0);
        assert_eq!(shadow_extent(1.0), 25.0);
        assert_eq!(shadow_extent(5000.0), 500.0);
        assert_eq!(shadow_extent(f32::NAN), 110.0);
        assert_eq!(shadow_extent(f32::INFINITY), 110.0);
    }

    #[test]
    fn desc_clamps_instead_of_misallocating() {
        let d = ShadowDesc {
            size: 1,
            bias: -1.0,
            strength: 9.0,
        };
        assert_eq!(d.clamped_size(), 64);
        assert_eq!(d.clamped_bias(), 0.0);
        assert_eq!(d.clamped_strength(), 1.0);
        let d = ShadowDesc {
            size: 1 << 30,
            bias: f32::NAN,
            strength: f32::NAN,
        };
        assert_eq!(d.clamped_size(), 4096);
        assert_eq!(d.clamped_bias(), 0.001);
        assert_eq!(d.clamped_strength(), 1.0);
        assert!((ShadowDesc::default().texel() - 1.0 / 1024.0).abs() < 1e-9);
    }

    fn test_cam() -> (Vec3, Mat4, Mat4) {
        use super::super::camera::OrbitCamera;
        let cam = OrbitCamera {
            target: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.9,
            dist: 60.0,
            fov_y_deg: 45.0,
        };
        let aspect = 16.0 / 9.0;
        let vp = cam.view_proj(aspect);
        (cam.eye(), cam.view_matrix(), vp.inverse())
    }

    #[test]
    fn cascade_splits_span_near_to_far() {
        let d = CascadeDesc {
            count: 3,
            lambda: 0.75,
            ..CascadeDesc::default()
        };
        let s = d.splits(1.0, 100.0);
        assert_eq!(s.len(), 4);
        assert!((s[0] - 1.0).abs() < 1e-4, "{s:?}");
        assert!((s[3] - 100.0).abs() < 1e-4, "{s:?}");
        for w in s.windows(2) {
            assert!(w[1] > w[0], "{s:?}");
        }
        let u = CascadeDesc {
            count: 2,
            lambda: 0.0,
            ..CascadeDesc::default()
        }
        .splits(1.0, 100.0);
        assert!((u[1] - 50.5).abs() < 1e-3, "{u:?}");
        let bad = CascadeDesc::default().splits(f32::NAN, -5.0);
        assert!(bad.windows(2).all(|w| w[1] > w[0]), "{bad:?}");
    }

    #[test]
    fn cascade_count_and_fields_clamp() {
        let d = CascadeDesc {
            count: 99,
            lambda: f32::NAN,
            size: 1 << 30,
            bias: -1.0,
            normal_bias: f32::NAN,
            strength: 9.0,
        };
        assert_eq!(d.clamped_count(), MAX_CASCADES);
        assert_eq!(d.clamped_size(), 2048);
        assert_eq!(d.clamped_bias(), 0.0);
        assert_eq!(d.clamped_normal_bias(), 0.05);
        assert_eq!(d.clamped_strength(), 1.0);
        assert_eq!(CascadeDesc { count: 0, ..d }.clamped_count(), 1);
    }

    #[test]
    fn fitted_slices_cover_the_frustum() {
        let (eye, view, inv_vp) = test_cam();
        let d = CascadeDesc {
            count: 3,
            ..CascadeDesc::default()
        };
        let slices = d.fit_all(eye, view, inv_vp, Vec3::new(0.3, 1.0, 0.2), 0.5, 2000.0);
        assert_eq!(slices.len(), 3);
        assert!((slices[0].near - 0.5).abs() < 1e-3);
        assert!((slices[2].far - 2000.0).abs() < 1e-3);
        for w in slices.windows(2) {
            assert!((w[1].near - w[0].far).abs() < 1e-3);
        }
        for s in &slices {
            assert!(s.view_proj.is_finite(), "valid matrix");
            assert!(s.extent >= 1.0 && s.extent <= 2000.0, "{}", s.extent);
        }
        let close = d.fit_all(eye, view, inv_vp, Vec3::new(0.3, 1.0, 0.2), 0.5, 120.0);
        assert!(
            close[0].extent < shadow_extent(60.0),
            "near slice {} vs legacy {}",
            close[0].extent,
            shadow_extent(60.0)
        );
    }

    #[test]
    fn slice_fit_snaps_to_texels() {
        let d = CascadeDesc::default();
        let (eye, view, inv_vp) = test_cam();
        let a = d.fit_slice(eye, view, inv_vp, Vec3::Y, 0.5, 50.0);
        let world_texel = a.extent * 2.0 / CascadeDesc::default().clamped_size() as f32;
        let nudge = Vec3::new(world_texel * 0.4, 0.0, 0.0);
        let b = d.fit_slice(eye + nudge, view, inv_vp, Vec3::Y, 0.5, 50.0);
        let center = |m: Mat4| {
            let e = m.to_cols_array_2d();
            (e[3][0], e[3][1])
        };
        let (ax, ay) = center(a.view_proj);
        let (bx, by) = center(b.view_proj);
        let drift = (ax - bx).abs() + (ay - by).abs();
        assert!(
            drift < a.texel * 8.0 + 1e-6,
            "snapped drift bounded: {drift}"
        );
        assert!(
            (a.extent - b.extent).abs() < a.extent * 0.05 + 1e-3,
            "{} vs {}",
            a.extent,
            b.extent
        );
    }

    #[test]
    fn point_cube_faces_cover_all_directions() {
        let p = PointLight::default();
        let faces = p.cube_faces().expect("valid range");
        let dirs = [
            (Vec3::X, 0),
            (Vec3::NEG_X, 1),
            (Vec3::Y, 2),
            (Vec3::NEG_Y, 3),
            (Vec3::Z, 4),
            (Vec3::NEG_Z, 5),
        ];
        for (dir, face) in dirs {
            assert_eq!(PointLight::face_for_dir(dir), face);
            let vp = Mat4::from_cols_array_2d(&faces[face]);
            let probe = Vec3::from(p.position) + dir * (p.clamped_range() * 0.5);
            let clip = vp * glam::Vec4::new(probe.x, probe.y, probe.z, 1.0);
            assert!(clip.w > 0.0, "in front of face {face}");
            let ndc = Vec3::new(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w);
            assert!(
                ndc.x.abs() <= 1.0 && ndc.y.abs() <= 1.0,
                "face {face}: {ndc:?}"
            );
            assert!((0.0..=1.0).contains(&ndc.z), "face {face}: {ndc:?}");
        }
        assert!(
            PointLight {
                position: [f32::NAN, 0.0, 0.0],
                ..p
            }
            .cube_faces()
            .is_none()
        );
        assert!(
            PointLight { range: 0.0, ..p }.cube_faces().is_some(),
            "clamped range stays usable"
        );
    }

    #[test]
    fn point_fields_clamp_and_score() {
        let p = PointLight {
            range: 99999.0,
            size: 1 << 20,
            bias: f32::NAN,
            intensity: f32::NAN,
            ..PointLight::default()
        };
        assert_eq!(p.clamped_range(), 500.0);
        assert_eq!(p.clamped_size(), 1024);
        assert_eq!(p.clamped_bias(), 0.005);
        assert_eq!(p.score(), 0.0, "NaN intensity scores zero");
        let dim = PointLight {
            color: [0.1, 0.1, 0.1],
            intensity: 1.0,
            ..PointLight::default()
        };
        let bright = PointLight {
            color: [2.0, 1.0, 1.0],
            intensity: 1.0,
            ..PointLight::default()
        };
        assert!(bright.score() > dim.score());
    }
}
