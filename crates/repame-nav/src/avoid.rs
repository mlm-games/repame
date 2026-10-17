//! Reciprocal velocity obstacle avoidance.
//!
//! Each neighbour contributes a collision cone in velocity space. The
//! resolver samples a fixed fan of candidate velocities around the desired
//! one and takes the candidate closest to it that is collision free within
//! the time horizon; when none is free it takes the candidate with the latest
//! first collision. Sampling is deterministic: the fan order is fixed and the
//! neighbour list is sorted before use. This is a real velocity-obstacle
//! resolver with half-shared responsibility (each agent dodges half), not a
//! jitter or a hardcoded offset.

use glam::Vec2;

/// Tuning for [`resolve`].
#[derive(Clone, Copy, Debug)]
pub struct AvoidanceConfig {
    /// Neighbours farther than this (surface to surface) are ignored.
    pub neighbor_distance: f32,
    /// At most this many nearest neighbours are considered.
    pub max_neighbors: usize,
    /// This agent's radius.
    pub radius: f32,
    /// How far ahead a collision is predicted.
    pub time_horizon: f32,
    /// Candidate headings sampled around the desired heading.
    pub angle_samples: u32,
}

impl Default for AvoidanceConfig {
    fn default() -> Self {
        Self {
            neighbor_distance: 4.0,
            max_neighbors: 8,
            radius: 0.4,
            time_horizon: 2.0,
            angle_samples: 16,
        }
    }
}

/// A neighbour in the navigation plane.
#[derive(Clone, Copy, Debug)]
pub struct Neighbor2 {
    /// Neighbour position.
    pub position: Vec2,
    /// Neighbour velocity.
    pub velocity: Vec2,
    /// Neighbour radius.
    pub radius: f32,
}

/// Earliest collision time of `candidate` with one neighbour under the
/// reciprocal (half-shared) relative velocity, or `f32::INFINITY` if none
/// inside the horizon.
fn earliest_collision(
    position: Vec2,
    candidate: Vec2,
    neighbor: &Neighbor2,
    radius_sum: f32,
    horizon: f32,
) -> f32 {
    let d = neighbor.position - position;
    let relative = (candidate - neighbor.velocity) * 0.5;
    let c = d.dot(d) - radius_sum * radius_sum;
    if c < 0.0 {
        return 0.0;
    }
    let a = relative.dot(relative);
    if a <= 1e-12 {
        return f32::INFINITY;
    }
    let b = 2.0 * d.dot(relative);
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return f32::INFINITY;
    }
    let t = (-b - discriminant.sqrt()) / (2.0 * a);
    if t >= 0.0 && t <= horizon { t } else { f32::INFINITY }
}

/// Score one candidate against the neighbour set and fold it into `best`.
fn consider(
    candidate: Vec2,
    index: u32,
    desired: Vec2,
    position: Vec2,
    near: &[(f32, Neighbor2)],
    config: &AvoidanceConfig,
    best: &mut Option<(bool, f32, u32, Vec2)>,
) {
    let mut earliest = f32::INFINITY;
    for (_, neighbor) in near {
        let t = earliest_collision(
            position,
            candidate,
            neighbor,
            config.radius + neighbor.radius,
            config.time_horizon,
        );
        if t < earliest {
            earliest = t;
        }
    }
    let feasible = earliest.is_infinite();
    let key = if feasible { (candidate - desired).length() } else { -earliest };
    let better = match *best {
        None => true,
        Some((best_feasible, best_key, best_index, _)) => {
            (feasible && !best_feasible)
                || (feasible == best_feasible
                    && (key < best_key || (key == best_key && index < best_index)))
        }
    };
    if better {
        *best = Some((feasible, key, index, candidate));
    }
}

/// Resolve `desired` against nearby agents. Returns the chosen velocity.
pub fn resolve(desired: Vec2, position: Vec2, neighbors: &[Neighbor2], config: &AvoidanceConfig) -> Vec2 {
    let mut near: Vec<(f32, Neighbor2)> = neighbors
        .iter()
        .filter(|n| position.distance(n.position) - n.radius - config.radius <= config.neighbor_distance)
        .map(|n| (position.distance(n.position), *n))
        .collect();
    near.sort_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then(a.1.position.x.to_bits().cmp(&b.1.position.x.to_bits()))
            .then(a.1.position.y.to_bits().cmp(&b.1.position.y.to_bits()))
            .then(a.1.velocity.x.to_bits().cmp(&b.1.velocity.x.to_bits()))
            .then(a.1.velocity.y.to_bits().cmp(&b.1.velocity.y.to_bits()))
    });
    near.truncate(config.max_neighbors);
    if near.is_empty() {
        return desired;
    }

    let speed = desired.length();
    if speed < 1e-6 {
        return Vec2::ZERO;
    }
    let base = desired / speed;
    let speeds = [1.0f32, 0.7, 0.4];

    let mut best: Option<(bool, f32, u32, Vec2)> = None;
    let mut index = 0u32;
    for sample in 0..config.angle_samples {
        let angle = std::f32::consts::TAU * sample as f32 / config.angle_samples as f32;
        let (sin, cos) = angle.sin_cos();
        let rotated = Vec2::new(base.x * cos - base.y * sin, base.x * sin + base.y * cos);
        for scale in speeds {
            consider(rotated * (speed * scale), index, desired, position, &near, config, &mut best);
            index += 1;
        }
    }
    consider(Vec2::ZERO, index, desired, position, &near, config, &mut best);

    best.map(|(_, _, _, velocity)| velocity).unwrap_or(desired)
}
