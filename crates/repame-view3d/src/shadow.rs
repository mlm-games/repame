//! GPU shadow maps: one directional-light depth pass behind [`MeshGroup`].

use glam::{Mat4, Vec3};

use super::camera::OPENGL_TO_WGPU;

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
}
