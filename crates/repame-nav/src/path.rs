//! A* over the polygon graph, then a funnel pass to straighten the corridor.

use crate::funnel::{dedup, funnel};
use crate::mesh::{closest_on_segment, NavMesh};
use crate::vec::NavVec;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A caller-supplied off-mesh link already resolved onto the mesh.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ResolvedLink<V> {
    pub from: u32,
    pub to: u32,
    pub portal: [V; 2],
    pub cost: f32,
    pub bidirectional: bool,
}

/// Open-set entry. Ordered by `f` then polygon index, so equal-cost nodes pop
/// in a fixed order regardless of heap internals.
#[derive(Clone, Copy, Debug)]
struct OpenNode {
    f: f32,
    polygon: u32,
}

impl PartialEq for OpenNode {
    fn eq(&self, other: &Self) -> bool {
        self.f.to_bits() == other.f.to_bits() && self.polygon == other.polygon
    }
}
impl Eq for OpenNode {}
impl PartialOrd for OpenNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OpenNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.f.total_cmp(&other.f).then(self.polygon.cmp(&other.polygon))
    }
}

fn heuristic<V: NavVec>(mesh: &NavMesh<V>, polygon: u32, target: V) -> f32 {
    let region = &mesh.polygons[polygon as usize];
    (region.center.distance(target) - region.radius).max(0.0)
}

/// Search the polygon graph from `start_poly` to `end_poly`.
///
/// Returns the corridor of polygon indices and, in travel order, the portal
/// crossed to enter each polygon after the first. `blocked_poly` removes a
/// region; `blocked_portal` removes a single crossing; `links` adds off-mesh
/// edges.
pub(crate) fn astar_with<V: NavVec>(
    mesh: &NavMesh<V>,
    start_poly: u32,
    end_poly: u32,
    start_point: V,
    end_point: V,
    blocked_poly: impl Fn(u32) -> bool,
    blocked_portal: impl Fn([V; 2]) -> bool,
    links: &[ResolvedLink<V>],
) -> Option<(Vec<u32>, Vec<[V; 2]>)> {
    if start_poly == end_poly {
        return Some((vec![start_poly], Vec::new()));
    }
    let count = mesh.polygons.len();
    let mut g = vec![f32::INFINITY; count];
    let mut came: Vec<Option<(u32, [V; 2])>> = vec![None; count];
    let mut entry = vec![V::ZERO; count];
    let mut visited = vec![false; count];

    let mut open: BinaryHeap<std::cmp::Reverse<OpenNode>> = BinaryHeap::new();
    g[start_poly as usize] = 0.0;
    entry[start_poly as usize] = start_point;
    open.push(std::cmp::Reverse(OpenNode {
        f: heuristic(mesh, start_poly, end_point),
        polygon: start_poly,
    }));

    while let Some(std::cmp::Reverse(node)) = open.pop() {
        let polygon = node.polygon;
        let pi = polygon as usize;
        if visited[pi] {
            continue;
        }
        visited[pi] = true;
        if polygon == end_poly {
            break;
        }
        let entry_point = entry[pi];

        for edge in 0..mesh.polygons[pi].indices.len() as u32 {
            let Some(neighbor) = mesh.neighbor(polygon, edge) else {
                continue;
            };
            let q = neighbor.polygon;
            if q == polygon {
                continue;
            }
            if q != end_poly && blocked_poly(q) {
                continue;
            }
            let Some(portal) = mesh.portal(polygon, edge) else {
                continue;
            };
            if blocked_portal(portal) {
                continue;
            }
            let q_entry = closest_on_segment(entry_point, portal[0], portal[1]);
            let ng = g[pi] + entry_point.distance(q_entry);
            if ng < g[q as usize] {
                g[q as usize] = ng;
                came[q as usize] = Some((polygon, portal));
                entry[q as usize] = q_entry;
                open.push(std::cmp::Reverse(OpenNode {
                    f: ng + heuristic(mesh, q, end_point),
                    polygon: q,
                }));
            }
        }

        for link in links {
            let (q, portal, cost) = if link.from == polygon {
                (link.to, link.portal, entry_point.distance(link.portal[0]) + link.cost)
            } else if link.bidirectional && link.to == polygon {
                (
                    link.from,
                    [link.portal[1], link.portal[0]],
                    entry_point.distance(link.portal[1]) + link.cost,
                )
            } else {
                continue;
            };
            if q != end_poly && blocked_poly(q) {
                continue;
            }
            let q_entry = portal[1];
            let ng = g[pi] + cost;
            if ng < g[q as usize] {
                g[q as usize] = ng;
                came[q as usize] = Some((polygon, portal));
                entry[q as usize] = q_entry;
                open.push(std::cmp::Reverse(OpenNode {
                    f: ng + heuristic(mesh, q, end_point),
                    polygon: q,
                }));
            }
        }
    }

    let mut corridor = vec![end_poly];
    let mut portals = Vec::new();
    let mut current = end_poly;
    while current != start_poly {
        let (previous, portal) = came[current as usize]?;
        portals.push(portal);
        corridor.push(previous);
        current = previous;
    }
    corridor.reverse();
    portals.reverse();
    Some((corridor, portals))
}

impl<V: NavVec> NavMesh<V> {
    /// Shortest path through the mesh from `from` to `to`.
    ///
    /// Endpoints are projected onto their region (a heightfield lift or a
    /// boundary projection). Returns `None` when either endpoint has no
    /// region or no corridor connects them.
    pub fn find_path(&self, from: V, to: V) -> Option<Vec<V>> {
        let start_polygon = self.locate(from).or_else(|| self.closest_polygon(from))?;
        let end_polygon = self.locate(to).or_else(|| self.closest_polygon(to))?;
        let start = self.closest_point_on_polygon(start_polygon, from);
        let end = self.closest_point_on_polygon(end_polygon, to);
        if start_polygon == end_polygon {
            return Some(vec![start, end]);
        }
        let (_, portals) =
            astar_with(self, start_polygon, end_polygon, start, end, |_| false, |_| false, &[])?;
        let mut path = funnel(start, end, &portals);
        dedup(&mut path);
        Some(path)
    }
}
