//! Flat-2D mover kit: swept-circle vs AABB settlements, contacts, bounces.
//! y-down world pixels; a body is a circle of `radius`, obstacles are AABBs
//! (props) plus a solid-cell callback for tile grids. Everything here is pure
//! over caller-supplied data, so tests feed fixtures without a world.
//!
//! The four boundary slabs grow from `bounds` on every call, so a mover can
//! never leave the arena; [`clamp_to_bounds`] is the same guarantee for
//! callers that integrate positions by hand.

use bevy_ecs::prelude::Entity;
use glam::Vec2;

/// Cell edge in world pixels for the swept solid-cell window.
pub const CELL_SIZE: f32 = 16.0;
/// Reflections per move before the remaining displacement is applied as-is.
const MAX_REFLECTIONS: usize = 4;

/// Solid cells overlapping a world rect: the kit walks a swept window and
/// asks the grid for its solid cells as `(center, size)` pairs. The
/// implementation owns which cells are solid and emits only those.
pub trait CellSolids {
    fn for_each_cell(&self, min: Vec2, max: Vec2, f: &mut dyn FnMut(Vec2, Vec2));
}

/// Static world description for one move: AABB props, solid cells and arena
/// bounds. Build one per move (it borrows), or keep one alive for repeated
/// point tests at a settled position.
#[derive(Clone, Copy)]
pub struct SolidWorld<'a> {
    pub props: &'a [(Vec2, Vec2)],
    pub cells: Option<&'a dyn CellSolids>,
    pub bounds: (Vec2, Vec2),
}

impl<'a> SolidWorld<'a> {
    pub fn new(props: &'a [(Vec2, Vec2)], bounds: (Vec2, Vec2)) -> Self {
        Self {
            props,
            cells: None,
            bounds,
        }
    }

    pub fn with_cells(mut self, cells: &'a dyn CellSolids) -> Self {
        self.cells = Some(cells);
        self
    }
}

/// A resolved contact against the solid set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolidContact {
    pub normal: Vec2,
    pub penetration: f32,
    pub fraction: f32,
}

/// A swept circle's first touch of one AABB.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SweptCircleAabb {
    pub normal: Vec2,
    pub penetration: f32,
    pub fraction: f32,
}

#[derive(Clone, Copy, Debug)]
struct SolidAabb {
    min: Vec2,
    max: Vec2,
}

impl SolidAabb {
    fn from_center_size(center: Vec2, size: Vec2) -> Self {
        let half = size.abs() * 0.5;
        Self {
            min: center - half,
            max: center + half,
        }
    }

    fn center_half(self) -> (Vec2, Vec2) {
        ((self.min + self.max) * 0.5, (self.max - self.min) * 0.5)
    }
}

pub fn circle_aabb_normal(pos: Vec2, radius: f32, center: Vec2, half: Vec2) -> Option<Vec2> {
    let half = half.abs();
    let closest = Vec2::new(
        pos.x.clamp(center.x - half.x, center.x + half.x),
        pos.y.clamp(center.y - half.y, center.y + half.y),
    );
    let delta = pos - closest;
    let d2 = delta.length_squared();
    if d2 > radius * radius {
        return None;
    }
    if d2 > 1e-8 {
        return Some(delta.normalize());
    }

    let dx = half.x - (pos.x - center.x).abs();
    let dy = half.y - (pos.y - center.y).abs();
    let sx = if pos.x < center.x { -1.0 } else { 1.0 };
    let sy = if pos.y < center.y { -1.0 } else { 1.0 };
    if dx < dy {
        Some(Vec2::new(sx, 0.0))
    } else {
        Some(Vec2::new(0.0, sy))
    }
}

pub fn circle_aabb_penetration(pos: Vec2, radius: f32, center: Vec2, half: Vec2) -> Option<f32> {
    let half = half.abs();
    let closest = Vec2::new(
        pos.x.clamp(center.x - half.x, center.x + half.x),
        pos.y.clamp(center.y - half.y, center.y + half.y),
    );
    let delta = pos - closest;
    let distance = delta.length();
    if distance > radius {
        return None;
    }
    if distance > 1e-8 {
        return Some(radius - distance);
    }
    let dx = half.x - (pos.x - center.x).abs();
    let dy = half.y - (pos.y - center.y).abs();
    Some(radius + dx.min(dy))
}

pub fn circle_aabb_overlap(
    pos: Vec2,
    radius: f32,
    center: Vec2,
    half: Vec2,
) -> Option<(Vec2, f32)> {
    let half = half.abs();
    let closest = Vec2::new(
        pos.x.clamp(center.x - half.x, center.x + half.x),
        pos.y.clamp(center.y - half.y, center.y + half.y),
    );
    let delta = pos - closest;
    let d2 = delta.length_squared();
    if d2 > radius * radius {
        return None;
    }
    let normal = if d2 > 1e-8 {
        delta.normalize()
    } else {
        let dx = half.x - (pos.x - center.x).abs();
        let dy = half.y - (pos.y - center.y).abs();
        let sx = if pos.x < center.x { -1.0 } else { 1.0 };
        let sy = if pos.y < center.y { -1.0 } else { 1.0 };
        if dx < dy {
            Vec2::new(sx, 0.0)
        } else {
            Vec2::new(0.0, sy)
        }
    };
    let distance = delta.length();
    if distance > radius {
        return None;
    }
    let penetration = if distance > 1e-8 {
        radius - distance
    } else {
        let dx = half.x - (pos.x - center.x).abs();
        let dy = half.y - (pos.y - center.y).abs();
        radius + dx.min(dy)
    };
    Some((normal, penetration))
}

pub fn swept_circle_aabb(
    start: Vec2,
    displacement: Vec2,
    radius: f32,
    center: Vec2,
    half: Vec2,
) -> Option<SweptCircleAabb> {
    let half = half.abs();
    let radius = radius.max(0.0);
    // Broad phase: any contact needs some point of the path inside the box
    // grown by `radius` (the narrow phase also tolerates `1e-4` of slack, so
    // grow a little more). That grown box is a superset of the rounded
    // region, so a segment miss proves `None` and skips the searches below;
    // hits fall through untouched.
    let grow = radius + 1e-3;
    let end = start + displacement;
    let mut t_lo = 0.0f32;
    let mut t_hi = 1.0f32;
    for (p, d, lo, hi) in [
        (
            start.x,
            end.x - start.x,
            center.x - half.x - grow,
            center.x + half.x + grow,
        ),
        (
            start.y,
            end.y - start.y,
            center.y - half.y - grow,
            center.y + half.y + grow,
        ),
    ] {
        if d == 0.0 {
            if p < lo || p > hi {
                return None;
            }
        } else {
            let mut a = (lo - p) / d;
            let mut b = (hi - p) / d;
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
            t_lo = t_lo.max(a);
            t_hi = t_hi.min(b);
            if t_lo > t_hi {
                return None;
            }
        }
    }
    let distance = |t: f32| {
        let p = start + displacement * t;
        let closest = Vec2::new(
            p.x.clamp(center.x - half.x, center.x + half.x),
            p.y.clamp(center.y - half.y, center.y + half.y),
        );
        (p - closest).length()
    };
    let distance_squared = |t: f32| {
        let p = start + displacement * t;
        let closest = Vec2::new(
            p.x.clamp(center.x - half.x, center.x + half.x),
            p.y.clamp(center.y - half.y, center.y + half.y),
        );
        (p - closest).length_squared()
    };
    let d0 = distance(0.0);
    if d0 <= radius + 1e-5 {
        let normal = circle_aabb_normal(start, radius, center, half).unwrap_or_else(|| {
            let p = start - center;
            if p.x.abs() > p.y.abs() {
                Vec2::new(if p.x < 0.0 { -1.0 } else { 1.0 }, 0.0)
            } else {
                Vec2::new(0.0, if p.y < 0.0 { -1.0 } else { 1.0 })
            }
        });
        return Some(SweptCircleAabb {
            normal,
            penetration: circle_aabb_penetration(start, radius, center, half).unwrap_or(0.0),
            fraction: 0.0,
        });
    }

    let mut lo = 0.0;
    let mut hi = 1.0;
    for _ in 0..48 {
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        if distance_squared(m1) < distance_squared(m2) {
            hi = m2;
        } else {
            lo = m1;
        }
    }
    let min_t = (lo + hi) * 0.5;
    if distance(min_t) > radius + 1e-4 {
        return None;
    }

    let mut lo = 0.0;
    let mut hi = min_t;
    for _ in 0..40 {
        let mid = (lo + hi) * 0.5;
        if distance(mid) <= radius + 1e-5 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let fraction = hi.clamp(0.0, 1.0);
    let point = start + displacement * fraction;
    let normal = circle_aabb_normal(point, radius + 1e-3, center, half).unwrap_or_else(|| {
        let closest = Vec2::new(
            point.x.clamp(center.x - half.x, center.x + half.x),
            point.y.clamp(center.y - half.y, center.y + half.y),
        );
        (point - closest).normalize_or_zero()
    });
    Some(SweptCircleAabb {
        normal,
        penetration: circle_aabb_penetration(point, radius, center, half).unwrap_or(0.0),
        fraction,
    })
}

pub fn bounce_velocity(vel: Vec2, normal: Vec2) -> Vec2 {
    vel - 2.0 * vel.dot(normal) * normal
}

/// Boundary contact at `pos` for a circle: `Some(normal)` when the circle
/// touches or passes an arena edge.
pub fn arena_wall_normal(pos: Vec2, radius: f32, bounds: (Vec2, Vec2)) -> Option<Vec2> {
    let hx = bounds.1.x - radius;
    let hy = bounds.1.y - radius;
    let mut n = Vec2::ZERO;
    if pos.x > hx {
        n.x = 1.0;
    } else if pos.x < bounds.0.x + radius {
        n.x = -1.0;
    }
    if pos.y > hy {
        n.y = 1.0;
    } else if pos.y < bounds.0.y + radius {
        n.y = -1.0;
    }
    if n == Vec2::ZERO {
        None
    } else {
        Some(n.normalize_or_zero())
    }
}

/// Pellet directions fanned around `base_dir`: `samples[i]` in `-1..1`
/// scales the half-`spread` (radians), extras clamp to the last sample.
pub fn split_directions(base_dir: Vec2, pellets: u8, spread: f32, samples: &[f32]) -> Vec<Vec2> {
    let base_angle = if base_dir.length_squared() > 1e-6 {
        base_dir.y.atan2(base_dir.x)
    } else {
        0.0
    };
    let mut out = Vec::with_capacity(pellets as usize);
    for i in 0..pellets as usize {
        let t = samples.get(i).copied().unwrap_or(0.0).clamp(-1.0, 1.0);
        let angle = base_angle + t * spread;
        out.push(Vec2::new(angle.cos(), angle.sin()));
    }
    out
}

/// Pierce book-keeping: `(despawn, pierce_left_after)` for a hit.
pub fn should_despawn_after_hit(
    damaged: bool,
    pierce_left_before: Option<u8>,
) -> (bool, Option<u8>) {
    match pierce_left_before {
        Some(n) if n > 0 && damaged => (false, Some(n - 1)),
        Some(n) if n > 0 && !damaged => (false, Some(n)),
        Some(0) | None if damaged => (true, pierce_left_before.map(|_| 0)),
        _ => (true, pierce_left_before),
    }
}

/// Record a hit target once per shot; false when already recorded.
pub fn record_hit(set: &mut Vec<Entity>, target: Entity) -> bool {
    if set.contains(&target) {
        false
    } else {
        set.push(target);
        true
    }
}

/// Clamp a body inside the arena rectangle.
pub fn clamp_to_bounds(pos: &mut Vec2, radius: f32, bounds: (Vec2, Vec2)) {
    pos.x = pos.x.clamp(bounds.0.x + radius, bounds.1.x - radius);
    pos.y = pos.y.clamp(bounds.0.y + radius, bounds.1.y - radius);
}

/// Push a circle out of AABB props (`(center, size)` pairs). Pure over
/// an iterator so tests feed fixtures without a world.
pub fn resolve_aabb_pushout(
    pos: &mut Vec2,
    radius: f32,
    props: impl Iterator<Item = (Vec2, Vec2)>,
) {
    for (center, size) in props {
        let half = size / 2.0;
        let closest = Vec2::new(
            pos.x.clamp(center.x - half.x, center.x + half.x),
            pos.y.clamp(center.y - half.y, center.y + half.y),
        );
        let d = *pos - closest;
        let dist = d.length();
        if dist >= radius || dist <= 0.0001 {
            continue;
        }
        pos.x += d.x / dist * (radius - dist);
        pos.y += d.y / dist * (radius - dist);
    }
}

fn fill_solid_shapes(
    shapes: &mut Vec<SolidAabb>,
    start: Vec2,
    displacement: Vec2,
    radius: f32,
    world: &SolidWorld,
) {
    for &(center, size) in world.props {
        if size.x > 0.0 || size.y > 0.0 {
            shapes.push(SolidAabb::from_center_size(center, size));
        }
    }

    let (min, max) = world.bounds;
    let extent = 100_000.0;
    shapes.push(SolidAabb {
        min: Vec2::new(min.x - extent, min.y - extent),
        max: Vec2::new(min.x, max.y + extent),
    });
    shapes.push(SolidAabb {
        min: Vec2::new(max.x, min.y - extent),
        max: Vec2::new(max.x + extent, max.y + extent),
    });
    shapes.push(SolidAabb {
        min: Vec2::new(min.x - extent, min.y - extent),
        max: Vec2::new(max.x + extent, min.y),
    });
    shapes.push(SolidAabb {
        min: Vec2::new(min.x - extent, max.y),
        max: Vec2::new(max.x + extent, max.y + extent),
    });

    if let Some(grid) = world.cells {
        let reach = radius + CELL_SIZE;
        let window_min = start.min(start + displacement) - Vec2::splat(reach);
        let window_max = start.max(start + displacement) + Vec2::splat(reach);
        grid.for_each_cell(window_min, window_max, &mut |center, size| {
            shapes.push(SolidAabb::from_center_size(center, size));
        });
    }
}

/// Pre-built solid set for repeated point tests at a settled position.
pub struct SolidShapes(Vec<SolidAabb>);

pub fn build_solid_shapes(
    start: Vec2,
    displacement: Vec2,
    radius: f32,
    world: &SolidWorld,
) -> SolidShapes {
    let mut shapes = Vec::with_capacity(world.props.len() + 5);
    fill_solid_shapes(&mut shapes, start, displacement, radius, world);
    SolidShapes(shapes)
}

fn first_swept_contact(
    start: Vec2,
    displacement: Vec2,
    radius: f32,
    shapes: &[SolidAabb],
) -> Option<SolidContact> {
    let mut first: Option<SolidContact> = None;
    for shape in shapes {
        let (center, half) = shape.center_half();
        let Some(contact) = swept_circle_aabb(start, displacement, radius, center, half) else {
            continue;
        };
        if contact.fraction <= 1e-6
            && contact.penetration <= 1e-5
            && displacement.dot(contact.normal) >= -1e-6
        {
            continue;
        }
        if first.is_none_or(|old| contact.fraction < old.fraction) {
            first = Some(SolidContact {
                normal: contact.normal,
                penetration: contact.penetration,
                fraction: contact.fraction,
            });
        }
    }
    first
}

fn first_current_contact(pos: Vec2, radius: f32, shapes: &[SolidAabb]) -> Option<SolidContact> {
    let mut first: Option<SolidContact> = None;
    for shape in shapes {
        let (center, half) = shape.center_half();
        let Some((normal, penetration)) = circle_aabb_overlap(pos, radius, center, half) else {
            continue;
        };
        let candidate = SolidContact {
            normal,
            penetration,
            fraction: 0.0,
        };
        if first.is_none_or(|old| candidate.penetration > old.penetration) {
            first = Some(candidate);
        }
    }
    first
}

pub fn contact_at(pos: Vec2, radius: f32, shapes: &SolidShapes) -> Option<SolidContact> {
    first_current_contact(pos, radius, &shapes.0)
}

pub fn solid_contact(pos: Vec2, radius: f32, world: &SolidWorld) -> Option<SolidContact> {
    let shapes = build_solid_shapes(pos, Vec2::ZERO, radius, world);
    contact_at(pos, radius, &shapes)
}

/// Move along `displacement`, sliding along contacts (one reflection per
/// wall). Returns the first contact when one happened.
pub fn move_contact_solid(
    pos: &mut Vec2,
    displacement: Vec2,
    radius: f32,
    world: &SolidWorld,
) -> Option<SolidContact> {
    clamp_to_bounds(pos, radius, world.bounds);
    let shapes = build_solid_shapes(*pos, displacement, radius, world);
    let mut remaining = displacement;
    let mut first = None;
    for _ in 0..MAX_REFLECTIONS {
        let Some(contact) = first_swept_contact(*pos, remaining, radius, &shapes.0) else {
            *pos += remaining;
            clamp_to_bounds(pos, radius, world.bounds);
            break;
        };
        if first.is_none() {
            first = Some(contact);
        }
        *pos += remaining * contact.fraction;
        if contact.penetration > 0.0 {
            *pos += contact.normal * (contact.penetration + 0.001);
        }
        let consumed = remaining * contact.fraction;
        remaining -= consumed;
        remaining -= contact.normal * remaining.dot(contact.normal);
        if remaining.length_squared() <= 1e-8 {
            break;
        }
    }
    clamp_to_bounds(pos, radius, world.bounds);
    first
}

/// Move along `velocity * dt`; with `bounce`, reflect the velocity (and the
/// remaining displacement) off every contact instead of sliding.
pub fn move_bounce_solid(
    pos: &mut Vec2,
    velocity: &mut Vec2,
    radius: f32,
    dt: f32,
    world: &SolidWorld,
    bounce: bool,
) -> Option<SolidContact> {
    move_bounce_solid_displacement(
        pos,
        velocity,
        *velocity * dt.max(0.0),
        radius,
        world,
        bounce,
    )
}

pub fn move_bounce_solid_displacement(
    pos: &mut Vec2,
    velocity: &mut Vec2,
    displacement: Vec2,
    radius: f32,
    world: &SolidWorld,
    bounce: bool,
) -> Option<SolidContact> {
    clamp_to_bounds(pos, radius, world.bounds);
    let shapes = build_solid_shapes(*pos, displacement, radius, world);
    let mut remaining = displacement;
    let mut first = None;
    for _ in 0..MAX_REFLECTIONS {
        let Some(contact) = first_swept_contact(*pos, remaining, radius, &shapes.0) else {
            *pos += remaining;
            clamp_to_bounds(pos, radius, world.bounds);
            return first;
        };
        if first.is_none() {
            first = Some(contact);
        }
        *pos += remaining * contact.fraction;
        if contact.penetration > 0.0 {
            *pos += contact.normal * (contact.penetration + 0.001);
        }
        let consumed = remaining * contact.fraction;
        remaining -= consumed;
        if bounce {
            if velocity.dot(contact.normal) < 0.0 {
                *velocity = bounce_velocity(*velocity, contact.normal);
            }
            if remaining.dot(contact.normal) < 0.0 {
                remaining = bounce_velocity(remaining, contact.normal);
            }
        } else {
            remaining -= contact.normal * remaining.dot(contact.normal);
        }
        if remaining.length_squared() <= 1e-8 {
            return first;
        }
    }
    // Out of reflections with displacement left: a diagonal path into a corner
    // burns one reflection per wall, so apply the remainder rather than dropping
    // it and stalling the mover in the corner.
    *pos += remaining;
    clamp_to_bounds(pos, radius, world.bounds);
    first
}

/// One step of "walk toward the target": move up to `distance` along the
/// heading to `target`, settling on contacts.
pub fn potential_step_solid(
    pos: &mut Vec2,
    target: Vec2,
    distance: f32,
    radius: f32,
    world: &SolidWorld,
) -> Option<SolidContact> {
    let delta = target - *pos;
    if delta.length_squared() <= 1e-8 {
        return None;
    }
    let step = delta.normalize() * distance.max(0.0);
    move_contact_solid(pos, step, radius, world)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A single 16x16 wall cell centered at `(cx + 0.5, cy + 0.5) * 16`.
    struct OneCell(i32, i32);
    impl CellSolids for OneCell {
        fn for_each_cell(&self, min: Vec2, max: Vec2, f: &mut dyn FnMut(Vec2, Vec2)) {
            let center = Vec2::new(
                self.0 as f32 * CELL_SIZE + CELL_SIZE * 0.5,
                self.1 as f32 * CELL_SIZE + CELL_SIZE * 0.5,
            );
            if center.x + CELL_SIZE * 0.5 < min.x
                || center.x - CELL_SIZE * 0.5 > max.x
                || center.y + CELL_SIZE * 0.5 < min.y
                || center.y - CELL_SIZE * 0.5 > max.y
            {
                return;
            }
            f(center, Vec2::splat(CELL_SIZE));
        }
    }

    const BOUNDS: (Vec2, Vec2) = (Vec2::new(-64.0, -64.0), Vec2::new(64.0, 64.0));

    fn world_with(grid: &dyn CellSolids) -> SolidWorld<'_> {
        SolidWorld::new(&[], BOUNDS).with_cells(grid)
    }

    #[test]
    fn normal_points_out_of_the_face() {
        // Circle left of a box: normal points -x.
        let n =
            circle_aabb_normal(Vec2::new(-4.0, 0.0), 2.0, Vec2::ZERO, Vec2::splat(2.0)).unwrap();
        assert!((n - Vec2::new(-1.0, 0.0)).length() < 1e-6);
        // Fully outside: no contact.
        assert!(
            circle_aabb_normal(Vec2::new(-10.0, 0.0), 2.0, Vec2::ZERO, Vec2::splat(2.0)).is_none()
        );
        // Inside: normal escapes along the shallow axis (x here).
        let n =
            circle_aabb_normal(Vec2::new(-1.0, 0.0), 2.0, Vec2::ZERO, Vec2::splat(4.0)).unwrap();
        assert!((n - Vec2::new(-1.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn sweep_hits_ahead_at_the_face() {
        let hit = swept_circle_aabb(
            Vec2::new(-20.0, 0.0),
            Vec2::new(40.0, 0.0),
            2.0,
            Vec2::ZERO,
            Vec2::splat(2.0),
        )
        .unwrap();
        // Face at x=-2, circle radius 2: contact at x=-4, t = 16/40.
        assert!((hit.fraction - 0.4).abs() < 1e-3);
        assert!((hit.normal - Vec2::new(-1.0, 0.0)).length() < 1e-3);
    }

    #[test]
    fn sweep_that_clears_the_box_is_none() {
        assert!(
            swept_circle_aabb(
                Vec2::new(-20.0, 40.0),
                Vec2::new(40.0, 0.0),
                2.0,
                Vec2::ZERO,
                Vec2::splat(2.0),
            )
            .is_none()
        );
    }

    #[test]
    fn move_contact_stops_at_a_cell_wall() {
        let grid = OneCell(0, 0); // center (8, 8), 16x16
        let mut pos = Vec2::new(-8.0, 8.0);
        let contact = move_contact_solid(&mut pos, Vec2::new(32.0, 0.0), 4.0, &world_with(&grid));
        assert!(contact.is_some());
        // Settled just left of the wall face at x = 0, less one radius.
        assert!((pos.x - -4.0).abs() < 0.5);
    }

    #[test]
    fn move_contact_slides_along_a_diagonal_wall() {
        let grid = OneCell(0, 0);
        let mut pos = Vec2::new(-8.0, 8.0);
        move_contact_solid(&mut pos, Vec2::new(16.0, 8.0), 4.0, &world_with(&grid));
        // Slides down the wall face instead of stopping dead.
        assert!(pos.y > 9.0);
    }

    #[test]
    fn bounce_reflects_velocity_off_the_wall() {
        let grid = OneCell(0, 0);
        let mut pos = Vec2::new(-8.0, 8.0);
        let mut vel = Vec2::new(300.0, 30.0);
        let hit = move_bounce_solid(
            &mut pos,
            &mut vel,
            4.0,
            1.0 / 30.0,
            &world_with(&grid),
            true,
        );
        assert!(hit.is_some());
        assert!(vel.x < 0.0);
    }

    #[test]
    fn clamp_keeps_the_circle_inside() {
        let mut p = Vec2::new(100.0, -100.0);
        clamp_to_bounds(&mut p, 4.0, BOUNDS);
        assert_eq!(p, Vec2::new(60.0, -60.0));
    }

    #[test]
    fn arena_normal_ignores_the_middle() {
        assert!(arena_wall_normal(Vec2::ZERO, 4.0, BOUNDS).is_none());
        let n = arena_wall_normal(Vec2::new(61.0, 0.0), 4.0, BOUNDS).unwrap();
        assert!(n.x > 0.0 && n.y == 0.0);
    }

    #[test]
    fn pushout_settles_on_the_face() {
        // Center at x=2.5 against a box spanning x -2..2: one radius out.
        let mut p = Vec2::new(2.5, 0.0);
        resolve_aabb_pushout(&mut p, 2.0, [(Vec2::ZERO, Vec2::splat(4.0))].into_iter());
        assert!((p.x - 4.0).abs() < 1e-3);
    }

    #[test]
    fn split_directions_center_and_edges() {
        let dirs = split_directions(Vec2::X, 3, 1.0, &[-1.0, 0.0, 1.0]);
        assert_eq!(dirs.len(), 3);
        assert!((dirs[1] - Vec2::X).length() < 1e-6);
        assert!((dirs[0].y.atan2(dirs[0].x) - -1.0).abs() < 1e-6);
        assert!((dirs[2].y.atan2(dirs[2].x) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn pierce_table() {
        assert_eq!(should_despawn_after_hit(true, None), (true, None));
        assert_eq!(should_despawn_after_hit(true, Some(2)), (false, Some(1)));
        assert_eq!(should_despawn_after_hit(false, Some(2)), (false, Some(2)));
        assert_eq!(should_despawn_after_hit(true, Some(0)), (true, Some(0)));
        assert_eq!(should_despawn_after_hit(false, Some(0)), (true, Some(0)));
    }
}
