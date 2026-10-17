//! Runtime navigation map: a baked [`NavMesh`] plus dynamic carved obstacles
//! and off-mesh links.
//!
//! Obstacles carve at polygon granularity: a region whose navigation-plane
//! centre falls inside an obstacle is removed from the search, and a crossing
//! whose portal midpoint falls inside an obstacle is removed too, so a path
//! cannot cut a corner through a thin obstacle. Off-mesh links add an edge
//! between the regions holding their endpoints.
//!
//! Both lists are consumed in insertion order, which keeps queries
//! deterministic.

use crate::mesh::NavMesh;
use crate::path::{astar_with, ResolvedLink};
use crate::vec::NavVec;
use crate::funnel;

/// A dynamic carved region, in navigation-plane coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct NavObstacle {
    /// Outline polygon (XY for 2D, XZ for 3D). Any winding.
    pub outline: Vec<[f32; 2]>,
    /// Carving is skipped while disabled.
    pub enabled: bool,
}

impl NavObstacle {
    /// Enabled obstacle from an outline.
    pub fn new(outline: Vec<[f32; 2]>) -> Self {
        Self { outline, enabled: true }
    }

    fn contains_point(&self, point: [f32; 2]) -> bool {
        if !self.enabled || self.outline.len() < 3 {
            return false;
        }
        let mut inside = false;
        let count = self.outline.len();
        let mut j = count - 1;
        for i in 0..count {
            let a = self.outline[i];
            let b = self.outline[j];
            if (a[1] > point[1]) != (b[1] > point[1]) {
                let t = (point[1] - a[1]) / (b[1] - a[1]);
                if point[0] < a[0] + t * (b[0] - a[0]) {
                    inside = !inside;
                }
            }
            j = i;
        }
        inside
    }
}

/// A one-way or bidirectional off-mesh connection between two world points.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NavLink<V> {
    /// Link start, approached from a region containing it.
    pub start: V,
    /// Link end; the agent arrives here and continues into its region.
    pub end: V,
    /// When set, the link is usable in both directions.
    pub bidirectional: bool,
    /// Traversal cost. Defaults to the straight length.
    pub cost: Option<f32>,
    /// Usable while enabled.
    pub enabled: bool,
}

impl<V> NavLink<V> {
    /// Enabled one-way link with default cost.
    pub fn new(start: V, end: V) -> Self {
        Self { start, end, bidirectional: false, cost: None, enabled: true }
    }
}

/// A baked mesh plus the dynamic data layered on top of it for queries.
#[derive(Clone, Debug)]
pub struct NavMap<V: NavVec> {
    /// The baked mesh.
    pub mesh: NavMesh<V>,
    /// Dynamic carved regions.
    pub obstacles: Vec<NavObstacle>,
    /// Off-mesh connections.
    pub links: Vec<NavLink<V>>,
}

impl<V: NavVec> NavMap<V> {
    /// Wrap a baked mesh.
    pub fn new(mesh: NavMesh<V>) -> Self {
        Self { mesh, obstacles: Vec::new(), links: Vec::new() }
    }

    /// Add a carved obstacle. Returns its index.
    pub fn add_obstacle(&mut self, obstacle: NavObstacle) -> usize {
        self.obstacles.push(obstacle);
        self.obstacles.len() - 1
    }

    /// Add an off-mesh link. Returns its index.
    pub fn add_link(&mut self, link: NavLink<V>) -> usize {
        self.links.push(link);
        self.links.len() - 1
    }

    fn blocked_point(&self, point: V) -> bool {
        let plane = point.nav2();
        self.obstacles.iter().any(|o| o.contains_point(plane))
    }

    fn blocked_polygon(&self, polygon: u32) -> bool {
        match self.mesh.polygon(polygon) {
            Some(region) => self.blocked_point(region.center),
            None => false,
        }
    }

    fn blocked_portal(&self, portal: [V; 2]) -> bool {
        let a = portal[0].nav2();
        let b = portal[1].nav2();
        let midpoint = V::from_nav2([(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5], 0.0);
        self.blocked_point(midpoint)
    }

    fn resolve_links(&self) -> Vec<ResolvedLink<V>> {
        let mut resolved = Vec::new();
        for link in &self.links {
            if !link.enabled || self.blocked_point(link.start) || self.blocked_point(link.end) {
                continue;
            }
            let from = self.mesh.locate(link.start).or_else(|| self.mesh.closest_polygon(link.start));
            let to = self.mesh.locate(link.end).or_else(|| self.mesh.closest_polygon(link.end));
            let (Some(from), Some(to)) = (from, to) else {
                continue;
            };
            if from == to {
                continue;
            }
            let cost = link.cost.unwrap_or(link.start.distance(link.end));
            resolved.push(ResolvedLink {
                from,
                to,
                portal: [link.start, link.end],
                cost,
                bidirectional: link.bidirectional,
            });
        }
        resolved
    }

    /// Shortest path through the mesh with obstacles and links applied.
    ///
    /// Deterministic: the same map state and endpoints give the same path.
    pub fn find_path(&self, from: V, to: V) -> Option<Vec<V>> {
        let start_polygon = self.mesh.locate(from).or_else(|| self.mesh.closest_polygon(from))?;
        let end_polygon = self.mesh.locate(to).or_else(|| self.mesh.closest_polygon(to))?;
        let start = self.mesh.closest_point_on_polygon(start_polygon, from);
        let end = self.mesh.closest_point_on_polygon(end_polygon, to);
        if start_polygon == end_polygon {
            return Some(vec![start, end]);
        }
        let links = self.resolve_links();
        let (_, portals) = astar_with(
            &self.mesh,
            start_polygon,
            end_polygon,
            start,
            end,
            |polygon| self.blocked_polygon(polygon),
            |portal| self.blocked_portal(portal),
            &links,
        )?;
        let mut path = funnel::funnel(start, end, &portals);
        funnel::dedup(&mut path);
        Some(path)
    }
}
