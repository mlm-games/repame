//! Stereo view configuration: per-eye pose, fov, and the view/projection pair.
//!
//! The projection lives here rather than in the runtime because the runtime only
//! hands over an asymmetric [`Fov`] and the near/far planes are the renderer's
//! choice. [`projection_from_fov`] is the OpenXR projection formula for a
//! `[0, 1]` depth clip space, which is what `wgpu` (and so `repame-view3d`) uses:
//! a right-handed eye space looking down `-z`, `z = near` mapping to `0` and
//! `z = far` to `1`. Passing a runtime's `[-1, 1]` OpenGL-style matrix through
//! unchanged is the classic source of half-depth-range stereo bugs, so the
//! conversion is done from fov data instead of from a handed-over matrix.

use glam::{Mat4, Vec4};

use crate::Pose;

/// Which half of a stereo pair a view describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Eye {
    Left,
    Right,
}

impl Eye {
    /// Both eyes, left first, in the order runtimes enumerate views.
    pub const ALL: [Eye; 2] = [Eye::Left, Eye::Right];

    /// Position in a per-eye array. Also the layer index inside a stereo swapchain.
    pub fn index(self) -> usize {
        match self {
            Eye::Left => 0,
            Eye::Right => 1,
        }
    }

    /// `"left"` or `"right"`.
    pub fn name(self) -> &'static str {
        match self {
            Eye::Left => "left",
            Eye::Right => "right",
        }
    }
}

/// Pixel format of a runtime-owned eye image.
///
/// Wide-gamut and float formats are included because runtimes do hand them out;
/// an app that cannot import the one it is given must say so rather than
/// pretending a different format's bytes are correct.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// 8-bit sRGB RGBA.
    Rgba8UnormSrgb,
    /// 8-bit linear RGBA.
    Rgba8Unorm,
    /// 8-bit sRGB BGRA.
    Bgra8UnormSrgb,
    /// 8-bit linear BGRA.
    Bgra8Unorm,
    /// 16-bit float RGBA.
    Rgba16Float,
}

impl PixelFormat {
    /// True when the format's channels are sRGB encoded.
    pub fn is_srgb(self) -> bool {
        matches!(
            self,
            PixelFormat::Rgba8UnormSrgb | PixelFormat::Bgra8UnormSrgb
        )
    }

    /// Bytes per pixel.
    pub fn bytes_per_pixel(self) -> u32 {
        match self {
            PixelFormat::Rgba8UnormSrgb
            | PixelFormat::Rgba8Unorm
            | PixelFormat::Bgra8UnormSrgb
            | PixelFormat::Bgra8Unorm => 4,
            PixelFormat::Rgba16Float => 8,
        }
    }
}

/// How the runtime composites the rendered layer over the real world.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BlendMode {
    /// Nothing of the real world shows through.
    #[default]
    Opaque,
    /// The rendered image is added to the real world.
    Additive,
    /// The rendered image's alpha composites over the real world.
    AlphaBlend,
}

/// Which views a session presents.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ViewConfiguration {
    /// One view per eye.
    #[default]
    PrimaryStereo,
    /// A single view, for handheld AR.
    PrimaryMono,
}

impl ViewConfiguration {
    /// Number of views this configuration asks for.
    pub fn view_count(self) -> usize {
        match self {
            ViewConfiguration::PrimaryStereo => 2,
            ViewConfiguration::PrimaryMono => 1,
        }
    }

    /// Eyes in enumeration order. A mono configuration still uses the left eye,
    /// which is what runtimes report for it.
    pub fn eyes(self) -> &'static [Eye] {
        match self {
            ViewConfiguration::PrimaryStereo => &Eye::ALL,
            ViewConfiguration::PrimaryMono => &Eye::ALL[..1],
        }
    }

    /// Eyes present in a stereo configuration.
    pub fn is_stereo(self) -> bool {
        matches!(self, ViewConfiguration::PrimaryStereo)
    }
}

/// An asymmetric field of view, as the runtime reports it: four half-angles in
/// radians around the view's `-z` axis, in the runtime's own sign convention
/// (`angle_left` and `angle_down` are normally negative).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fov {
    /// Angle from `-z` towards `-x`.
    pub angle_left: f32,
    /// Angle from `-z` towards `+x`.
    pub angle_right: f32,
    /// Angle from `-z` towards `+y`.
    pub angle_up: f32,
    /// Angle from `-z` towards `-y`.
    pub angle_down: f32,
}

impl Fov {
    /// A centred frustum from a vertical field of view, the shape an ordinary
    /// perspective camera uses. Useful for synthesizing a mono view.
    pub fn symmetric(fov_y_radians: f32, aspect: f32) -> Self {
        let tan_up = (fov_y_radians * 0.5).tan();
        let tan_right = tan_up * aspect;
        Self {
            angle_left: (-tan_right).atan(),
            angle_right: tan_right.atan(),
            angle_up: tan_up.atan(),
            angle_down: (-tan_up).atan(),
        }
    }

    /// Every angle finite.
    pub fn is_finite(&self) -> bool {
        self.angle_left.is_finite()
            && self.angle_right.is_finite()
            && self.angle_up.is_finite()
            && self.angle_down.is_finite()
    }

    /// Horizontal and vertical extent in radians.
    pub fn extents(&self) -> (f32, f32) {
        (
            self.angle_right - self.angle_left,
            self.angle_up - self.angle_down,
        )
    }

    /// Recover the fov of an OpenGL-convention perspective projection, the
    /// shape WebGL and GL runtimes hand over, so the crate's own `[0, 1]`-depth
    /// projection can be rebuilt from [`projection_from_fov`] instead of
    /// forwarding a matrix in the wrong depth range.
    ///
    /// `None` when the matrix is not a perspective projection, e.g. when its x
    /// or y scale is zero.
    pub fn from_gl_projection(matrix: Mat4) -> Option<Fov> {
        let x_scale = matrix.x_axis.x;
        let y_scale = matrix.y_axis.y;
        if x_scale == 0.0 || y_scale == 0.0 {
            return None;
        }
        // Inverse of projection_from_fov for a `[-1, 1]` depth range: the z
        // column's x and y hold `(tan_right + tan_left) / width` and
        // `(tan_up + tan_down) / height`.
        let tan_right = (matrix.z_axis.x + 1.0) / x_scale;
        let tan_left = (matrix.z_axis.x - 1.0) / x_scale;
        let tan_up = (matrix.z_axis.y + 1.0) / y_scale;
        let tan_down = (matrix.z_axis.y - 1.0) / y_scale;
        let fov = Fov {
            angle_left: tan_left.atan(),
            angle_right: tan_right.atan(),
            angle_up: tan_up.atan(),
            angle_down: tan_down.atan(),
        };
        fov.is_finite().then_some(fov)
    }
}

/// Eye-space-to-clip projection from runtime fov data, for a `[0, 1]` depth
/// range. `far <= near` places the far plane at infinity.
///
/// Degenerate input (a zero-width or zero-height fov) yields a matrix of NaNs
/// rather than a silent identity; runtimes do not produce one, and a NaN matrix
/// fails visibly in the renderer instead of drawing a wrong image.
pub fn projection_from_fov(fov: &Fov, near: f32, far: f32) -> Mat4 {
    let tan_left = fov.angle_left.tan();
    let tan_right = fov.angle_right.tan();
    let tan_up = fov.angle_up.tan();
    let tan_down = fov.angle_down.tan();

    let width = tan_right - tan_left;
    let height = tan_up - tan_down;

    let (m22, m32) = if far > near {
        (far / (near - far), near * far / (near - far))
    } else {
        (-1.0, -near)
    };

    Mat4::from_cols(
        Vec4::new(2.0 / width, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 2.0 / height, 0.0, 0.0),
        Vec4::new(
            (tan_right + tan_left) / width,
            (tan_up + tan_down) / height,
            m22,
            -1.0,
        ),
        Vec4::new(0.0, 0.0, m32, 0.0),
    )
}

/// One eye's placement and frustum for a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewSetup {
    /// Which eye this is.
    pub eye: Eye,
    /// Eye pose in the play space, absent when the runtime is not tracking it.
    pub pose: Option<Pose>,
    /// The runtime's fov. Absent when the backend hands over a finished matrix
    /// instead (WebXR does).
    pub fov: Option<Fov>,
    /// Reference-space-to-eye transform. Absent exactly when [`ViewSetup::pose`] is.
    pub view: Option<Mat4>,
    /// Eye-to-clip transform, always `[0, 1]` depth.
    pub projection: Mat4,
    /// `[x, y, width, height]` of this eye inside the swapchain image, when the
    /// runtime packs both eyes into one image.
    pub viewport: Option<[u32; 4]>,
}

impl ViewSetup {
    /// Build from an eye pose and fov, deriving both matrices.
    pub fn from_pose_fov(eye: Eye, pose: Option<Pose>, fov: Fov, near: f32, far: f32) -> Self {
        Self {
            eye,
            pose,
            fov: Some(fov),
            view: pose.map(|pose| pose.view_matrix()),
            projection: projection_from_fov(&fov, near, far),
            viewport: None,
        }
    }

    /// A view whose pose is known but whose projection was produced by the
    /// backend directly.
    pub fn from_matrices(eye: Eye, pose: Option<Pose>, view: Mat4, projection: Mat4) -> Self {
        Self {
            eye,
            pose,
            fov: None,
            view: Some(view),
            projection,
            viewport: None,
        }
    }

    /// Both matrices, or an error when the runtime is not tracking the eye.
    pub fn matrices(&self) -> Result<(Mat4, Mat4), crate::XrError> {
        let view = self.view.ok_or(crate::XrError::Untracked {
            what: match self.eye {
                Eye::Left => "left eye",
                Eye::Right => "right eye",
            },
        })?;
        if !view.is_finite() || !self.projection.is_finite() {
            return Err(crate::XrError::InvalidData {
                what: "view matrices",
            });
        }
        Ok((view, self.projection))
    }
}

/// Everything the renderer needs to draw a stereo frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ViewConfig {
    /// One entry per view, in the order the runtime enumerated them.
    pub views: Vec<ViewSetup>,
    /// Per-eye render target size the runtime recommends.
    pub recommended_size: [u32; 2],
    /// Near plane used for every projection.
    pub near: f32,
    /// Far plane used for every projection.
    pub far: f32,
    /// Compositing mode the session is running in.
    pub blend_mode: BlendMode,
}

impl ViewConfig {
    /// The view for one eye, if the configuration has one.
    pub fn view(&self, eye: Eye) -> Option<&ViewSetup> {
        self.views.iter().find(|view| view.eye == eye)
    }

    /// True for a two-view configuration.
    pub fn is_stereo(&self) -> bool {
        self.views.len() > 1
    }

    /// True when every view has a usable pose.
    pub fn is_tracked(&self) -> bool {
        self.views.iter().all(|view| view.view.is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, vec3};

    fn close(a: Mat4, b: Mat4) -> bool {
        (a - b).to_cols_array().iter().all(|x| x.abs() < 1e-6)
    }

    #[test]
    fn symmetric_fov_matches_the_standard_perspective() {
        let fov_y = 1.1_f32;
        let aspect = 1.4_f32;
        let (near, far) = (0.1_f32, 250.0_f32);
        let fov = Fov::symmetric(fov_y, aspect);
        let ours = projection_from_fov(&fov, near, far);
        let reference = glam::camera::rh::proj::directx::perspective(fov_y, aspect, near, far);
        assert!(close(ours, reference), "ours {ours:?} glam {reference:?}");
    }

    #[test]
    fn asymmetric_fov_pins_every_matrix_entry() {
        // tan(angle) == the given value.
        let fov = Fov {
            angle_left: (-2.0f32).atan(),
            angle_right: 2.0f32.atan(),
            angle_up: 3.0f32.atan(),
            angle_down: (-1.0f32).atan(),
        };
        let (near, far) = (1.0_f32, 11.0_f32);
        let m = projection_from_fov(&fov, near, far);
        let expected = Mat4::from_cols(
            Vec4::new(0.5, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 0.5, 0.0, 0.0),
            Vec4::new(0.0, 0.5, -1.1, -1.0),
            Vec4::new(0.0, 0.0, -1.1, 0.0),
        );
        assert!(close(m, expected), "{m:?}");

        // near maps to depth 0, far to 1, and the half-plane above centre is +y.
        let near_point = m * Vec4::new(0.0, 0.0, -near, 1.0);
        assert!(near_point.z.abs() < 1e-6);
        let far_point = m * Vec4::new(0.0, 0.0, -far, 1.0);
        assert!((far_point.z / far_point.w - 1.0).abs() < 1e-6);
        let above = m * Vec4::new(0.0, 1.0, -1.0, 1.0);
        assert!(above.y > 0.0);

        // An off-centre fov puts the frustum's own corners, not the -z axis, on
        // the clip volume's edges; an offset in the wrong matrix entry shows up
        // here and nowhere else.
        let mut off_centre = fov;
        off_centre.angle_left = (-1.0f32).atan();
        off_centre.angle_right = 3.0f32.atan();
        let m = projection_from_fov(&off_centre, near, far);
        for (x, y) in [(-1.0f32, -1.0f32), (3.0, -1.0), (-1.0, 3.0), (3.0, 3.0)] {
            let corner = m * Vec4::new(x, y, -1.0, 1.0);
            assert!(
                (corner.x / corner.w - x.signum()).abs() < 1e-5,
                "corner {x},{y} -> {corner:?}"
            );
            assert!(
                (corner.y / corner.w - y.signum()).abs() < 1e-5,
                "corner {x},{y} -> {corner:?}"
            );
        }
    }

    #[test]
    fn a_gl_projection_round_trips_back_to_its_fov() {
        let fov = Fov {
            angle_left: (-1.1f32).atan(),
            angle_right: 0.9f32.atan(),
            angle_up: 0.8f32.atan(),
            angle_down: (-0.9f32).atan(),
        };
        let (tan_left, tan_right) = (fov.angle_left.tan(), fov.angle_right.tan());
        let (tan_up, tan_down) = (fov.angle_up.tan(), fov.angle_down.tan());
        let (width, height) = (tan_right - tan_left, tan_up - tan_down);
        let (near, far) = (0.1_f32, 100.0_f32);
        // The OpenGL-convention matrix a GL runtime hands over, built the way
        // the XR matrix helpers do it: offsets in the z column, `-1` beneath.
        let gl = Mat4::from_cols(
            Vec4::new(2.0 / width, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 2.0 / height, 0.0, 0.0),
            Vec4::new(
                (tan_right + tan_left) / width,
                (tan_up + tan_down) / height,
                -(far + near) / (far - near),
                -1.0,
            ),
            Vec4::new(0.0, 0.0, -2.0 * far * near / (far - near), 0.0),
        );

        let restored = Fov::from_gl_projection(gl).expect("a perspective matrix");
        assert!((restored.angle_left - fov.angle_left).abs() < 1e-6);
        assert!((restored.angle_right - fov.angle_right).abs() < 1e-6);
        assert!((restored.angle_up - fov.angle_up).abs() < 1e-6);
        assert!((restored.angle_down - fov.angle_down).abs() < 1e-6);
        assert!(close(
            projection_from_fov(&restored, near, far),
            projection_from_fov(&fov, near, far)
        ));

        let degenerate = Mat4::from_cols(
            Vec4::new(0.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 0.5, 0.0, 0.0),
            Vec4::new(0.0, 0.0, -1.0, -1.0),
            Vec4::new(0.0, 0.0, -1.0, 0.0),
        );
        assert!(Fov::from_gl_projection(degenerate).is_none());
    }

    #[test]
    fn infinite_far_plane_keeps_depth_in_range() {
        let fov = Fov::symmetric(0.9, 1.0);
        let m = projection_from_fov(&fov, 0.5, 0.0);
        assert!((m.z_axis.z + 1.0).abs() < 1e-6);
        assert!((m.w_axis.z + 0.5).abs() < 1e-6);
        assert!((m.z_axis.w + 1.0).abs() < 1e-6);
        assert!(m.w_axis.w.abs() < 1e-6);

        let near_point = m * Vec4::new(0.0, 0.0, -0.5, 1.0);
        assert!(near_point.z.abs() < 1e-6);
        let far_point = m * Vec4::new(0.0, 0.0, -1.0e6, 1.0);
        let depth = far_point.z / far_point.w;
        assert!(depth > 0.9999 && depth < 1.0);
    }

    #[test]
    fn view_setup_derives_matrices_from_the_pose() {
        let pose = Pose::new(vec3(0.03, 1.7, 0.0), Quat::from_rotation_y(0.2));
        let fov = Fov::symmetric(1.0, 1.0);
        let view = ViewSetup::from_pose_fov(Eye::Left, Some(pose), fov, 0.1, 100.0);
        let (view_matrix, projection) = view.matrices().unwrap();
        assert!(close(view_matrix, pose.view_matrix()));
        assert!(close(projection, projection_from_fov(&fov, 0.1, 100.0)));
        assert_eq!(view.fov, Some(fov));

        // An untracked eye reports no matrices instead of an identity view.
        let untracked = ViewSetup::from_pose_fov(Eye::Right, None, fov, 0.1, 100.0);
        assert!(untracked.view.is_none());
        assert!(untracked.matrices().is_err());

        let config = ViewConfig {
            views: vec![view, untracked],
            recommended_size: [1832, 1920],
            near: 0.1,
            far: 100.0,
            blend_mode: BlendMode::Opaque,
        };
        assert!(config.is_stereo());
        assert!(!config.is_tracked());
        assert!(config.view(Eye::Left).is_some());
        assert_eq!(config.view(Eye::Right).unwrap().eye.index(), 1);
        assert!(config.views[0].viewport.is_none());
    }

    #[test]
    fn fov_helpers_and_view_configurations() {
        let fov = Fov::symmetric(core::f32::consts::FRAC_PI_2, 1.0);
        assert!(fov.is_finite());
        let (h, v) = fov.extents();
        assert!((h - v).abs() < 1e-6, "square frustum at aspect 1");
        assert!((v - core::f32::consts::FRAC_PI_2).abs() < 1e-5);
        assert!(fov.angle_left < 0.0 && fov.angle_down < 0.0);
        assert!(
            !Fov {
                angle_left: f32::NAN,
                ..fov
            }
            .is_finite()
        );

        assert_eq!(ViewConfiguration::PrimaryStereo.view_count(), 2);
        assert_eq!(ViewConfiguration::PrimaryMono.view_count(), 1);
        assert_eq!(ViewConfiguration::PrimaryMono.eyes()[0], Eye::Left);
        assert!(ViewConfiguration::PrimaryStereo.is_stereo());
        assert!(!ViewConfiguration::PrimaryMono.is_stereo());
    }
}
