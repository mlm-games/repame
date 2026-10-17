//! The funnel (string-pull) algorithm: turns a polygon corridor into the
//! straight-line corner sequence an agent actually walks.
//!
//! For 2D the side test uses the z cross product; for 3D it uses the cross
//! product against the navigation up axis, so a slope stays correct. Portals
//! arrive ordered from the start polygon to the end polygon and each is
//! normalised against the current apex, so the polygon winding does not matter.

use crate::vec::NavVec;

/// Pull a straight path through an ordered list of portal segments.
///
/// `start` and `end` are the query endpoints; each portal is the shared edge
/// between two consecutive corridor polygons, in travel order.
pub fn funnel<V: NavVec>(start: V, end: V, portals: &[[V; 2]]) -> Vec<V> {
    let mut path = Vec::with_capacity(portals.len() + 2);
    path.push(start);
    if portals.is_empty() {
        path.push(end);
        return path;
    }

    let mut apex = start;
    let mut left = start;
    let mut left_index = 0usize;
    let mut right = start;
    let mut right_index = 0usize;

    let mut i = 0usize;
    while i < portals.len() {
        let mut portal_left = portals[i][0];
        let mut portal_right = portals[i][1];
        if V::cross_apex(apex, portal_left, portal_right) < 0.0 {
            core::mem::swap(&mut portal_left, &mut portal_right);
        }

        if V::cross_apex(apex, left, portal_left) >= 0.0 {
            if left == apex || V::cross_apex(apex, portal_left, right) > 0.0 {
                left = portal_left;
                left_index = i;
            } else {
                path.push(right);
                apex = right;
                left = apex;
                right = apex;
                left_index = right_index;
                i = right_index;
                continue;
            }
        }

        if V::cross_apex(apex, right, portal_right) <= 0.0 {
            if right == apex || V::cross_apex(apex, portal_right, left) < 0.0 {
                right = portal_right;
                right_index = i;
            } else {
                path.push(left);
                apex = left;
                left = apex;
                right = apex;
                right_index = left_index;
                i = left_index;
                continue;
            }
        }

        i += 1;
    }

    if path.last().copied() != Some(end) {
        path.push(end);
    }
    path
}

/// Drop consecutive points closer than `1e-4`, so a corridor that flattens to
/// a single segment does not report duplicate corners.
pub fn dedup<V: NavVec>(path: &mut Vec<V>) {
    let mut write = 1;
    for read in 1..path.len() {
        if path[read].distance(path[write - 1]) > 1e-4 {
            path[write] = path[read];
            write += 1;
        }
    }
    path.truncate(write.max(1));
}
