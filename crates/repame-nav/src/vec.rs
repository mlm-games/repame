//! Coordinate abstraction shared by the 2D and 3D navigation meshes.

use glam::{Vec2, Vec3};

/// A point or vector a navmesh can be built from: [`glam::Vec2`] for flat 2D
/// navigation, [`glam::Vec3`] for a 3D heightfield.
///
/// The navigation plane is the XY plane for `Vec2` and the XZ plane for
/// `Vec3`. The remaining axis (`height`) is the surface offset that lets the
/// funnel follow a slope.
pub trait NavVec: Copy + PartialEq + core::fmt::Debug + Send + Sync + 'static {
    /// Additive identity.
    const ZERO: Self;

    /// Componentwise sum.
    fn add(self, other: Self) -> Self;
    /// Componentwise difference.
    fn sub(self, other: Self) -> Self;
    /// Scalar product.
    fn mul(self, scale: f32) -> Self;
    /// Dot product.
    fn dot(self, other: Self) -> f32;
    /// Signed area of the parallelogram spanned by the two vectors, measured
    /// against the navigation plane normal (`+Z` for 2D, `+Y` for 3D).
    fn cross_up(self, other: Self) -> f32;
    /// Euclidean length.
    fn length(self) -> f32;
    /// Position projected onto the navigation plane.
    fn nav2(self) -> [f32; 2];
    /// Offset outside the navigation plane (`0` for 2D).
    fn height(self) -> f32;
    /// Rebuild from a navigation plane position plus a height.
    fn from_nav2(plane: [f32; 2], height: f32) -> Self;

    /// Distance between two points.
    fn distance(self, other: Self) -> f32 {
        self.sub(other).length()
    }

    /// Signed side test used by the funnel, measured from `apex`.
    fn cross_apex(apex: Self, a: Self, b: Self) -> f32 {
        a.sub(apex).cross_up(b.sub(apex))
    }

    /// Unit vector, or [`Self::ZERO`] when the length underflows.
    fn normalized(self) -> Self {
        let len = self.length();
        if len > 1e-20 { self.mul(1.0 / len) } else { Self::ZERO }
    }
}

impl NavVec for Vec2 {
    const ZERO: Self = Vec2::ZERO;

    fn add(self, other: Self) -> Self {
        self + other
    }
    fn sub(self, other: Self) -> Self {
        self - other
    }
    fn mul(self, scale: f32) -> Self {
        self * scale
    }
    fn dot(self, other: Self) -> f32 {
        Vec2::dot(self, other)
    }
    fn cross_up(self, other: Self) -> f32 {
        self.x * other.y - self.y * other.x
    }
    fn length(self) -> f32 {
        Vec2::length(self)
    }
    fn nav2(self) -> [f32; 2] {
        [self.x, self.y]
    }
    fn height(self) -> f32 {
        0.0
    }
    fn from_nav2(plane: [f32; 2], _height: f32) -> Self {
        Vec2::new(plane[0], plane[1])
    }
}

impl NavVec for Vec3 {
    const ZERO: Self = Vec3::ZERO;

    fn add(self, other: Self) -> Self {
        self + other
    }
    fn sub(self, other: Self) -> Self {
        self - other
    }
    fn mul(self, scale: f32) -> Self {
        self * scale
    }
    fn dot(self, other: Self) -> f32 {
        Vec3::dot(self, other)
    }
    fn cross_up(self, other: Self) -> f32 {
        self.z * other.x - self.x * other.z
    }
    fn length(self) -> f32 {
        Vec3::length(self)
    }
    fn nav2(self) -> [f32; 2] {
        [self.x, self.z]
    }
    fn height(self) -> f32 {
        self.y
    }
    fn from_nav2(plane: [f32; 2], height: f32) -> Self {
        Vec3::new(plane[0], height, plane[1])
    }
}
