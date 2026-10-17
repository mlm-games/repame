//! Navmesh bake from engine geometry.
//!
//! Godot bakes a `NavigationMesh` from render geometry; here the source is the
//! engine's own collision input. A caller implements [`Source2d`] or
//! [`Source3d`] (any [`repame_physics::VoxelQuery`] is a `Source3d` for free),
//! so a bake needs no renderer and is fully testable.
//!
//! The bake is a heightfield build: one walkable column becomes one quad
//! region whose corners sit on the sampled surface, so a ramp produces a
//! tilted quad and the funnel follows the slope. Regions are not merged, which
//! keeps shared edges exact and adjacency unambiguous.

use crate::mesh::NavMesh;
use crate::NavError;
use glam::{Vec2, Vec3};
use std::collections::HashMap;

/// Unit horizontal size of one voxel column, in world units.
pub const VOXEL: f32 = 1.0;

/// 2D solidity sampler: a cell is either solid or free.
pub trait Source2d {
    /// Is the 2D cell solid?
    fn is_solid(&self, cell: [i32; 2]) -> bool;
}

/// 3D collision sampler: solidity plus the top surface height behind a cell
/// (ramps live behind [`Source3d::surface_top`]).
pub trait Source3d {
    /// Is the 3D cell solid?
    fn is_solid(&self, cell: [i32; 3]) -> bool;
    /// Top surface height of `cell` at world (`wx`, `wz`), or `None` in air.
    fn surface_top(&self, cell: [i32; 3], wx: f32, wz: f32) -> Option<f32>;
}

impl<Q: repame_physics::VoxelQuery> Source3d for Q {
    fn is_solid(&self, cell: [i32; 3]) -> bool {
        repame_physics::VoxelQuery::is_solid(self, cell)
    }
    fn surface_top(&self, cell: [i32; 3], wx: f32, wz: f32) -> Option<f32> {
        repame_physics::VoxelQuery::surface_top(self, cell, wx, wz)
    }
}

/// 2D bake parameters. `min` is inclusive, `max` exclusive.
#[derive(Clone, Copy, Debug)]
pub struct Bake2d {
    /// First cell column, inclusive.
    pub min: [i32; 2],
    /// One past the last cell column.
    pub max: [i32; 2],
    /// World size of one grid cell.
    pub cell_size: f32,
    /// Agent radius. Walkable cells closer than this to a solid cell are
    /// dropped, so agents do not hug walls.
    pub agent_radius: f32,
}

impl Default for Bake2d {
    fn default() -> Self {
        Self {
            min: [0, 0],
            max: [0, 0],
            cell_size: 1.0,
            agent_radius: 0.4,
        }
    }
}

/// 3D bake parameters. `min` is inclusive, `max` exclusive; a voxel is one
/// world unit.
#[derive(Clone, Copy, Debug)]
pub struct Bake3d {
    /// First cell, inclusive.
    pub min: [i32; 3],
    /// One past the last cell.
    pub max: [i32; 3],
    /// Vertical clearance a walkable column must have above its surface.
    pub agent_height: f32,
    /// Agent radius; walkable columns nearer than this to a non-walkable
    /// column are dropped.
    pub agent_radius: f32,
    /// Steepest walkable slope, in degrees.
    pub max_slope_deg: f32,
}

impl Default for Bake3d {
    fn default() -> Self {
        Self {
            min: [0, 0, 0],
            max: [0, 0, 0],
            agent_height: 1.8,
            agent_radius: 0.4,
            max_slope_deg: 45.0,
        }
    }
}

/// Bake a flat 2D navmesh of unit squares from a solidity sampler.
///
/// The 3D voxel world can drive this too by sampling one height layer, but a
/// dedicated 2D source keeps the crate usable without a voxel world.
pub fn bake_2d<S: Source2d>(source: &S, config: &Bake2d) -> Result<NavMesh<Vec2>, NavError> {
    if config.max[0] <= config.min[0] || config.max[1] <= config.min[1] {
        return Err(NavError::EmptyRegion);
    }
    if config.cell_size <= 0.0 {
        return Err(NavError::InvalidConfig);
    }
    let radius_cells = (config.agent_radius / config.cell_size).ceil() as i32;
    let walkable = |x: i32, z: i32| -> bool {
        if source.is_solid([x, z]) {
            return false;
        }
        for dx in -radius_cells..=radius_cells {
            for dz in -radius_cells..=radius_cells {
                if dx * dx + dz * dz <= radius_cells * radius_cells && source.is_solid([x + dx, z + dz])
                {
                    return false;
                }
            }
        }
        true
    };

    let mut vertices: Vec<Vec2> = Vec::new();
    let mut corners: HashMap<(i32, i32), u32> = HashMap::new();
    let mut polygons: Vec<Vec<u32>> = Vec::new();
    let scale = config.cell_size;
    let vertex = |x: i32, z: i32, vertices: &mut Vec<Vec2>, corners: &mut HashMap<(i32, i32), u32>| {
        *corners.entry((x, z)).or_insert_with(|| {
            vertices.push(Vec2::new(x as f32 * scale, z as f32 * scale));
            (vertices.len() - 1) as u32
        })
    };

    for x in config.min[0]..config.max[0] {
        for z in config.min[1]..config.max[1] {
            if !walkable(x, z) {
                continue;
            }
            let c0 = vertex(x, z, &mut vertices, &mut corners);
            let c1 = vertex(x + 1, z, &mut vertices, &mut corners);
            let c2 = vertex(x + 1, z + 1, &mut vertices, &mut corners);
            let c3 = vertex(x, z + 1, &mut vertices, &mut corners);
            polygons.push(vec![c0, c1, c2, c3]);
        }
    }
    if polygons.is_empty() {
        return Err(NavError::NoWalkableArea);
    }
    NavMesh::from_polygons(vertices, polygons)
}

/// Bake a heightfield 3D navmesh from a collision sampler.
pub fn bake_3d<S: Source3d>(source: &S, config: &Bake3d) -> Result<NavMesh<Vec3>, NavError> {
    if config.max[0] <= config.min[0] || config.max[1] <= config.min[1] || config.max[2] <= config.min[2]
    {
        return Err(NavError::EmptyRegion);
    }
    let radius_cells = (config.agent_radius / VOXEL).ceil() as i32;
    let margin = radius_cells + 1;
    let clearance_cells = config.agent_height.ceil() as i32;

    let scan_min = [config.min[0] - margin, config.min[2] - margin];
    let scan_max = [config.max[0] + margin, config.max[2] + margin];
    let width = (scan_max[0] - scan_min[0] + 1) as usize;
    let height = (scan_max[1] - scan_min[1] + 1) as usize;

    let mut ground: Vec<Option<[i32; 3]>> = vec![None; width * height];
    let index = |x: i32, z: i32| -> usize {
        ((x - scan_min[0]) as usize) + ((z - scan_min[1]) as usize) * width
    };

    for x in scan_min[0]..=scan_max[0] {
        for z in scan_min[1]..=scan_max[1] {
            let mut found = None;
            for y in (config.min[1]..config.max[1]).rev() {
                if !source.is_solid([x, y, z]) {
                    continue;
                }
                let mut clear = true;
                for k in 1..=clearance_cells {
                    if source.is_solid([x, y + k, z]) {
                        clear = false;
                        break;
                    }
                }
                if clear {
                    found = Some([x, y, z]);
                    break;
                }
            }
            ground[index(x, z)] = found;
        }
    }

    let column = |x: i32, z: i32| -> Option<[i32; 3]> {
        if x < scan_min[0] || z < scan_min[1] || x > scan_max[0] || z > scan_max[1] {
            None
        } else {
            ground[index(x, z)]
        }
    };
    let corner_height = |x: i32, z: i32| -> Option<f32> {
        column(x, z)
            .map(|cell| source.surface_top(cell, x as f32, z as f32).unwrap_or((cell[1] + 1) as f32))
    };
    let erode = |x: i32, z: i32| -> bool {
        if column(x, z).is_none() {
            return false;
        }
        for dx in -radius_cells..=radius_cells {
            for dz in -radius_cells..=radius_cells {
                if dx * dx + dz * dz <= radius_cells * radius_cells && column(x + dx, z + dz).is_none() {
                    return false;
                }
            }
        }
        true
    };

    let cos_limit = config.max_slope_deg.to_radians().cos();
    let mut vertices: Vec<Vec3> = Vec::new();
    let mut corners: HashMap<(i32, i32), u32> = HashMap::new();
    let mut polygons: Vec<Vec<u32>> = Vec::new();

    for x in config.min[0]..config.max[0] {
        for z in config.min[2]..config.max[2] {
            if !erode(x, z) {
                continue;
            }
            let (Some(h00), Some(h10), Some(h11), Some(h01)) = (
                corner_height(x, z),
                corner_height(x + 1, z),
                corner_height(x + 1, z + 1),
                corner_height(x, z + 1),
            ) else {
                continue;
            };
            let p00 = Vec3::new(x as f32, h00, z as f32);
            let p10 = Vec3::new((x + 1) as f32, h10, z as f32);
            let p11 = Vec3::new((x + 1) as f32, h11, (z + 1) as f32);
            let p01 = Vec3::new(x as f32, h01, (z + 1) as f32);
            let normal = (p10 - p00).cross(p11 - p00) + (p11 - p00).cross(p01 - p00);
            let length = normal.length();
            if length <= 1e-6 {
                continue;
            }
            if normal.y.abs() / length < cos_limit {
                continue;
            }
            let key = |x: i32, z: i32, vertices: &mut Vec<Vec3>, corners: &mut HashMap<(i32, i32), u32>| {
                *corners.entry((x, z)).or_insert_with(|| {
                    let height = corner_height(x, z).unwrap_or(0.0);
                    vertices.push(Vec3::new(x as f32, height, z as f32));
                    (vertices.len() - 1) as u32
                })
            };
            let c0 = key(x, z, &mut vertices, &mut corners);
            let c1 = key(x + 1, z, &mut vertices, &mut corners);
            let c2 = key(x + 1, z + 1, &mut vertices, &mut corners);
            let c3 = key(x, z + 1, &mut vertices, &mut corners);
            polygons.push(vec![c0, c1, c2, c3]);
        }
    }
    if polygons.is_empty() {
        return Err(NavError::NoWalkableArea);
    }
    NavMesh::from_polygons(vertices, polygons)
}
