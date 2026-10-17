//! Navmesh storage: a shared vertex buffer, convex polygon regions, and the
//! edge adjacency between them.
//!
//! Adjacency is derived from the geometry: two polygons are neighbours when
//! they traverse the same vertex pair in opposite directions. Building it from
//! geometry (rather than from bake-time bookkeeping) is what makes a merged or
//! hand-authored mesh and a baked grid mesh share one query path.

use crate::vec::NavVec;
use crate::NavError;
use std::collections::HashMap;

/// One convex region, stored as indices into the mesh vertex buffer.
#[derive(Clone, Debug, PartialEq)]
pub struct Polygon<V> {
    /// Vertex indices, wound consistently (one direction around the region).
    pub indices: Vec<u32>,
    /// Mean of the vertices. Used as the A* heuristic reference and the
    /// obstacle containment test point.
    pub center: V,
    /// Largest distance from `center` to a vertex.
    pub radius: f32,
}

/// The region across one directed edge of a polygon.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Neighbor {
    /// The adjacent polygon index.
    pub polygon: u32,
    /// The neighbour's own edge index for the shared edge.
    pub edge: u32,
}

/// A navigation mesh: vertices plus convex polygon regions plus adjacency.
///
/// Queries never iterate a hash map, so the same mesh always answers the same
/// way. The map is only used while [`NavMesh::from_polygons`] builds adjacency.
#[derive(Clone, Debug)]
pub struct NavMesh<V: NavVec> {
    /// Shared vertex buffer. Baked meshes deduplicate shared corners here.
    pub vertices: Vec<V>,
    /// Convex regions over `vertices`.
    pub polygons: Vec<Polygon<V>>,
    neighbors: Vec<Vec<Option<Neighbor>>>,
}

impl<V: NavVec> NavMesh<V> {
    /// Build a mesh from vertices and per-polygon vertex index lists.
    ///
    /// Returns [`NavError::DegeneratePolygon`] for a region with fewer than
    /// three vertices and [`NavError::BadIndex`] for an out-of-range index.
    pub fn from_polygons(vertices: Vec<V>, polygons: Vec<Vec<u32>>) -> Result<Self, NavError> {
        if polygons.is_empty() {
            return Err(NavError::EmptyRegion);
        }
        let mut regions = Vec::with_capacity(polygons.len());
        for indices in polygons {
            if indices.len() < 3 {
                return Err(NavError::DegeneratePolygon);
            }
            let mut center = V::ZERO;
            for &index in &indices {
                let vertex = *vertices.get(index as usize).ok_or(NavError::BadIndex)?;
                center = center.add(vertex);
            }
            center = center.mul(1.0 / indices.len() as f32);
            let mut radius = 0.0f32;
            for &index in &indices {
                radius = radius.max(center.distance(vertices[index as usize]));
            }
            regions.push(Polygon { indices, center, radius });
        }

        let mut neighbors = vec![Vec::new(); regions.len()];
        let mut directed: HashMap<(u32, u32), (u32, u32)> = HashMap::new();
        for (polygon, region) in regions.iter().enumerate() {
            let count = region.indices.len();
            neighbors[polygon] = vec![None; count];
            for edge in 0..count {
                let a = region.indices[edge];
                let b = region.indices[(edge + 1) % count];
                directed.insert((a, b), (polygon as u32, edge as u32));
            }
        }
        for (polygon, region) in regions.iter().enumerate() {
            let count = region.indices.len();
            for edge in 0..count {
                let a = region.indices[edge];
                let b = region.indices[(edge + 1) % count];
                if let Some(&(other, other_edge)) = directed.get(&(b, a)) {
                    if other as usize != polygon {
                        neighbors[polygon][edge] = Some(Neighbor { polygon: other, edge: other_edge });
                    }
                }
            }
        }

        Ok(Self { vertices, polygons: regions, neighbors })
    }

    /// Number of regions.
    pub fn polygon_count(&self) -> usize {
        self.polygons.len()
    }

    /// One region.
    pub fn polygon(&self, index: u32) -> Option<&Polygon<V>> {
        self.polygons.get(index as usize)
    }

    /// Vertex indices of one region.
    pub fn polygon_indices(&self, index: u32) -> Option<&[u32]> {
        self.polygons.get(index as usize).map(|p| p.indices.as_slice())
    }

    /// Neighbour across directed edge `edge` of `polygon`, if the edge is shared.
    pub fn neighbor(&self, polygon: u32, edge: u32) -> Option<Neighbor> {
        self.neighbors.get(polygon as usize)?.get(edge as usize).copied().flatten()
    }

    /// Endpoints of directed edge `edge` of `polygon`.
    pub fn portal(&self, polygon: u32, edge: u32) -> Option<[V; 2]> {
        let region = self.polygons.get(polygon as usize)?;
        let count = region.indices.len();
        let a = *region.indices.get(edge as usize)?;
        let b = region.indices[(edge as usize + 1) % count];
        Some([self.vertices[a as usize], self.vertices[b as usize]])
    }

    /// Region containing `point`, if any. Ties (they cannot occur in a normal
    /// tiling) resolve to the lowest region index.
    pub fn locate(&self, point: V) -> Option<u32> {
        let mut best: Option<(u32, f32)> = None;
        for (index, region) in self.polygons.iter().enumerate() {
            let region_vertices = self.region_vertices(region);
            if point_in_polygon(&region_vertices, point).is_none() {
                continue;
            }
            let distance = region.center.distance(point);
            match best {
                Some((_, d)) if d <= distance => {}
                _ => best = Some((index as u32, distance)),
            }
        }
        best.map(|(index, _)| index)
    }

    /// Region nearest to `point` when it is outside the mesh.
    pub fn closest_polygon(&self, point: V) -> Option<u32> {
        let mut best: Option<(u32, f32)> = None;
        for (index, region) in self.polygons.iter().enumerate() {
            let candidate = closest_point_on_polygon(&self.vertices, region, point);
            let distance = point.distance(candidate);
            match best {
                Some((_, d)) if d <= distance => {}
                _ => best = Some((index as u32, distance)),
            }
        }
        best.map(|(index, _)| index)
    }

    /// Closest point on a region to `point`.
    ///
    /// Inside the region the point is lifted onto the (possibly tilted)
    /// polygon surface by barycentric interpolation; outside it is projected
    /// onto the closest boundary edge.
    pub fn closest_point_on_polygon(&self, polygon: u32, point: V) -> V {
        match self.polygons.get(polygon as usize) {
            Some(region) => closest_point_on_polygon(&self.vertices, region, point),
            None => point,
        }
    }

    pub(crate) fn region_vertices(&self, region: &Polygon<V>) -> Vec<V> {
        region
            .indices
            .iter()
            .map(|&index| self.vertices[index as usize])
            .collect()
    }
}

/// Barycentric weights of `point` in the navigation plane triangle `a b c`,
/// or `None` when the point is outside it.
pub(crate) fn barycentric(plane: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> Option<[f32; 3]> {
    let v0 = [b[0] - a[0], b[1] - a[1]];
    let v1 = [c[0] - a[0], c[1] - a[1]];
    let v2 = [plane[0] - a[0], plane[1] - a[1]];
    let denom = v0[0] * v1[1] - v1[0] * v0[1];
    if denom.abs() < 1e-12 {
        return None;
    }
    let w1 = (v2[0] * v1[1] - v1[0] * v2[1]) / denom;
    let w2 = (v0[0] * v2[1] - v2[0] * v0[1]) / denom;
    let w0 = 1.0 - w1 - w2;
    const EPS: f32 = 1e-5;
    if w0 >= -EPS && w1 >= -EPS && w2 >= -EPS {
        Some([w0, w1, w2])
    } else {
        None
    }
}

/// First fan triangle of `vertices` containing `point` in the navigation
/// plane, with the interpolation weights.
pub(crate) fn point_in_polygon<V: NavVec>(vertices: &[V], point: V) -> Option<(usize, [f32; 3])> {
    if vertices.len() < 3 {
        return None;
    }
    let plane = point.nav2();
    let a = vertices[0].nav2();
    for i in 1..vertices.len() - 1 {
        let b = vertices[i].nav2();
        let c = vertices[i + 1].nav2();
        if let Some(weights) = barycentric(plane, a, b, c) {
            return Some((i, weights));
        }
    }
    None
}

/// Closest point on the segment `a b` to `point`, in full space.
pub(crate) fn closest_on_segment<V: NavVec>(point: V, a: V, b: V) -> V {
    let ab = b.sub(a);
    let denom = ab.dot(ab);
    if denom <= 1e-20 {
        return a;
    }
    let t = (point.sub(a).dot(ab) / denom).clamp(0.0, 1.0);
    a.add(ab.mul(t))
}

pub(crate) fn closest_point_on_polygon<V: NavVec>(
    vertices: &[V],
    region: &Polygon<V>,
    point: V,
) -> V {
    let region_vertices: Vec<V> =
        region.indices.iter().map(|&i| vertices[i as usize]).collect();
    if let Some((triangle, weights)) = point_in_polygon(&region_vertices, point) {
        let a = region_vertices[0].mul(weights[0]);
        let b = region_vertices[triangle].mul(weights[1]);
        let c = region_vertices[triangle + 1].mul(weights[2]);
        return a.add(b).add(c);
    }
    let mut best = region_vertices[0];
    let mut best_distance = point.distance(best);
    for i in 0..region_vertices.len() {
        let a = region_vertices[i];
        let b = region_vertices[(i + 1) % region_vertices.len()];
        let candidate = closest_on_segment(point, a, b);
        let distance = point.distance(candidate);
        if distance < best_distance {
            best_distance = distance;
            best = candidate;
        }
    }
    best
}
