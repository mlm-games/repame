//! Spatial navigation for Repame: a heightfield navmesh you bake from the
//! engine's own collision input, A* path queries over the region graph,
//! funnel (string-pull) corridors, and path-following agents with reciprocal
//! velocity obstacle avoidance. Both 2D ([`glam::Vec2`]) and 3D
//! ([`glam::Vec3`]) meshes share one implementation.
//!
//! # Pipeline
//! 1. Implement [`Source2d`] or [`Source3d`] (any
//!    [`repame_physics::VoxelQuery`] is a [`Source3d`]) and call [`bake_2d`]
//!    or [`bake_3d`] to get a [`NavMesh`].
//! 2. Query with [`NavMesh::find_path`], or wrap the mesh in a [`NavMap`] to
//!    layer dynamic [`NavObstacle`]s and off-mesh [`NavLink`]s on top.
//! 3. Drive a [`NavigationAgent`] with the returned path.
//!
//! # Determinism
//! Bakes emit regions in row-major cell order and deduplicate shared corners,
//! so vertex numbering is fixed. Adjacency and path results never iterate a
//! hash map; A* breaks cost ties by polygon index and the funnel is
//! order-stable. Obstacles and links are read in insertion order, and the
//! avoidance resolver samples a fixed fan around sorted neighbours. Within one
//! binary, the same inputs give bit-identical output. Cross-target float
//! equality (native libm vs wasm, notably `sqrt`/`sin_cos`) is not claimed,
//! the same caveat as [`repame_sim`]: hash a trace and diff to prove it.
//!
//! ```rust
//! use repame_nav::{bake_2d, Bake2d, Source2d};
//! use glam::Vec2;
//!
//! struct Open;
//! impl Source2d for Open {
//!     fn is_solid(&self, _cell: [i32; 2]) -> bool { false }
//! }
//!
//! let mesh = bake_2d(&Open, &Bake2d { min: [0, 0], max: [4, 4], cell_size: 1.0, agent_radius: 0.0 })
//!     .expect("open field bakes");
//! let path = mesh.find_path(Vec2::new(0.5, 0.5), Vec2::new(3.5, 3.5)).expect("connected");
//! assert_eq!(path.len(), 2);
//! ```

mod agent;
mod avoid;
mod bake;
mod funnel;
mod map;
mod mesh;
mod path;
mod vec;

pub use agent::{AgentNeighbor, NavigationAgent};
pub use avoid::{resolve, AvoidanceConfig, Neighbor2};
pub use bake::{bake_2d, bake_3d, Bake2d, Bake3d, Source2d, Source3d, VOXEL};
pub use funnel::funnel;
pub use map::{NavLink, NavMap, NavObstacle};
pub use mesh::{NavMesh, Neighbor, Polygon};
pub use vec::NavVec;

/// A navigation setup or query refused to build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NavError {
    /// The bake region is empty or inverted.
    EmptyRegion,
    /// A polygon index pointed outside the vertex buffer.
    BadIndex,
    /// A polygon had fewer than three vertices.
    DegeneratePolygon,
    /// The region held no walkable cell after erosion.
    NoWalkableArea,
    /// A scale or radius was not positive.
    InvalidConfig,
}

impl core::fmt::Display for NavError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let message = match self {
            NavError::EmptyRegion => "navigation bake region is empty",
            NavError::BadIndex => "polygon index is out of range",
            NavError::DegeneratePolygon => "polygon has fewer than three vertices",
            NavError::NoWalkableArea => "no walkable area after agent-radius erosion",
            NavError::InvalidConfig => "navigation configuration is invalid",
        };
        f.write_str(message)
    }
}

impl std::error::Error for NavError {}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Vec2, Vec3};

    struct Grid {
        solid: Vec<[i32; 2]>,
    }

    impl Source2d for Grid {
        fn is_solid(&self, cell: [i32; 2]) -> bool {
            self.solid.contains(&cell)
        }
    }

    fn walled_corridor() -> Grid {
        Grid { solid: vec![[2, 0], [2, 1]] }
    }

    fn corridor() -> NavMesh<Vec2> {
        bake_2d(
            &walled_corridor(),
            &Bake2d { min: [0, 0], max: [5, 3], cell_size: 1.0, agent_radius: 0.0 },
        )
        .expect("corridor bakes")
    }

    fn path_length(path: &[Vec2]) -> f32 {
        path.windows(2).map(|w| w[0].distance(w[1])).sum()
    }

    #[test]
    fn two_d_path_hugs_the_wall_corners() {
        let mesh = corridor();
        let path = mesh.find_path(Vec2::new(0.5, 1.5), Vec2::new(4.5, 1.5)).expect("connected");
        assert_eq!(path.len(), 4, "start, two corners, end: {path:?}");
        assert!((path[1].x - 2.0).abs() < 1e-4 && (path[1].y - 2.0).abs() < 1e-4, "corner {}", path[1]);
        assert!((path[2].x - 3.0).abs() < 1e-4 && (path[2].y - 2.0).abs() < 1e-4, "corner {}", path[2]);
        let expected = 2.0 * 2.5f32.sqrt() + 1.0;
        assert!(
            (path_length(&path) - expected).abs() < 1e-3,
            "length {} expected {expected}",
            path_length(&path)
        );
    }

    #[test]
    fn path_is_deterministic() {
        let mesh = corridor();
        let from = Vec2::new(0.5, 0.75);
        let to = Vec2::new(4.5, 2.25);
        let first = mesh.find_path(from, to);
        let second = mesh.find_path(from, to);
        assert!(first.is_some());
        assert_eq!(first, second, "same query must give the identical path");
    }

    #[test]
    fn obstacle_carves_the_only_gap() {
        let mesh = corridor();
        let mut map = NavMap::new(mesh);
        let from = Vec2::new(0.5, 1.5);
        let to = Vec2::new(4.5, 1.5);
        assert!(map.find_path(from, to).is_some(), "open gap");
        map.add_obstacle(NavObstacle::new(vec![[2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 3.0]]));
        assert!(map.find_path(from, to).is_none(), "carved gap blocks the route");
    }

    #[test]
    fn off_mesh_link_shortens_the_route() {
        let mesh = corridor();
        let from = Vec2::new(0.5, 1.5);
        let to = Vec2::new(4.5, 1.5);
        let detour = path_length(&mesh.find_path(from, to).expect("connected"));

        let mut map = NavMap::new(mesh);
        map.add_link(NavLink::new(Vec2::new(1.5, 1.5), Vec2::new(3.5, 1.5)));
        let direct = path_length(&map.find_path(from, to).expect("linked route"));
        assert!((direct - 4.0).abs() < 1e-3, "link path length {direct}");
        assert!(direct < detour, "link {direct} must beat detour {detour}");
    }

    struct Ramp;

    impl Source3d for Ramp {
        fn is_solid(&self, cell: [i32; 3]) -> bool {
            cell[1] <= 0 && (-4..=8).contains(&cell[0]) && (-4..=8).contains(&cell[2])
        }
        fn surface_top(&self, cell: [i32; 3], wx: f32, _wz: f32) -> Option<f32> {
            (cell[1] == 0).then_some(0.5 * wx)
        }
    }

    #[test]
    fn three_d_path_follows_a_slope() {
        let mesh = bake_3d(
            &Ramp,
            &Bake3d { min: [0, 0, 0], max: [4, 1, 2], ..Bake3d::default() },
        )
        .expect("ramp bakes");
        let from = Vec3::new(0.5, 0.0, 0.5);
        let to = Vec3::new(3.5, 0.0, 0.5);
        let path = mesh.find_path(from, to).expect("connected");
        assert!((path[0].y - 0.25).abs() < 1e-3, "start lifted to the surface: {}", path[0].y);
        assert!((path[path.len() - 1].y - 1.75).abs() < 1e-3, "end lifted: {}", path[path.len() - 1].y);
        let length: f32 = path.windows(2).map(|w| w[0].distance(w[1])).sum();
        let expected = (3.0f32 * 3.0 + 1.5 * 1.5).sqrt();
        assert!((length - expected).abs() < 0.02, "slope length {length} expected {expected}");
    }

    #[test]
    fn agent_seeks_and_arrives() {
        let mesh = corridor();
        let mut agent = NavigationAgent::<Vec2>::new(2.0);
        agent.arrival_radius = 0.25;
        agent.set_path(mesh.find_path(Vec2::new(0.5, 1.5), Vec2::new(4.5, 1.5)).expect("connected"));

        let mut position = Vec2::new(0.5, 1.5);
        for _ in 0..600 {
            let velocity = agent.desired_velocity(position);
            if velocity == Vec2::ZERO {
                break;
            }
            position += velocity * (1.0 / 60.0);
        }
        assert!(agent.is_finished(position), "agent stopped at {position}");
        assert!(position.distance(Vec2::new(4.5, 1.5)) <= agent.arrival_radius + 1e-3);
    }

    #[test]
    fn avoidance_is_deterministic_and_diverges_from_desired() {
        let neighbor = [AgentNeighbor {
            position: Vec2::new(1.0, 0.0),
            velocity: Vec2::new(-1.0, 0.0),
            radius: 0.4,
        }];
        let mut agent = NavigationAgent::<Vec2>::new(1.0);
        agent.avoidance_enabled = true;
        agent.desired_velocity = Vec2::new(1.0, 0.0);
        let first = agent.avoidance_velocity(Vec2::ZERO, &neighbor);
        let second = agent.avoidance_velocity(Vec2::ZERO, &neighbor);
        assert_eq!(first, second, "avoidance must be deterministic");
        assert!(first.y.abs() > 1e-3 || first.x < 1.0, "avoidance steers off a head-on course: {first}");
    }
}
