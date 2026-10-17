//! [`NavigationAgent`]: path following with seek plus arrival, and an
//! avoidance hook that resolves the desired velocity against nearby agents.

use crate::avoid::{resolve, AvoidanceConfig, Neighbor2};
use crate::vec::NavVec;

/// Another agent seen by the avoidance resolver.
#[derive(Clone, Copy, Debug)]
pub struct AgentNeighbor<V> {
    /// Agent position.
    pub position: V,
    /// Agent velocity.
    pub velocity: V,
    /// Agent radius.
    pub radius: f32,
}

/// Follows a path returned by [`NavMesh::find_path`](crate::NavMesh::find_path)
/// or [`NavMap::find_path`](crate::NavMap::find_path).
///
/// `desired_velocity` seeks the current waypoint and slows into the final one
/// by `arrival_radius`. `avoidance_velocity` then feeds that desired velocity
/// through the reciprocal velocity obstacle resolver.
#[derive(Clone, Debug)]
pub struct NavigationAgent<V: NavVec> {
    /// Current path, first point nearest the agent.
    pub path: Vec<V>,
    /// Index of the waypoint being sought.
    pub waypoint: usize,
    /// Desired speed toward a waypoint.
    pub max_speed: f32,
    /// Distance to the final waypoint within which the agent stops.
    pub arrival_radius: f32,
    /// Distance within which an intermediate waypoint counts as reached.
    pub waypoint_advance: f32,
    /// Last velocity produced by [`NavigationAgent::desired_velocity`].
    pub desired_velocity: V,
    /// Avoidance tuning.
    pub avoidance: AvoidanceConfig,
    /// When false, avoidance is a pass-through.
    pub avoidance_enabled: bool,
}

impl<V: NavVec> NavigationAgent<V> {
    /// New agent with no path.
    pub fn new(max_speed: f32) -> Self {
        Self {
            path: Vec::new(),
            waypoint: 0,
            max_speed,
            arrival_radius: 0.5,
            waypoint_advance: 0.3,
            desired_velocity: V::ZERO,
            avoidance: AvoidanceConfig::default(),
            avoidance_enabled: false,
        }
    }

    /// Replace the path and reset following to its first point.
    pub fn set_path(&mut self, path: Vec<V>) {
        self.waypoint = 0;
        self.path = path;
        self.desired_velocity = V::ZERO;
    }

    /// Drop the path.
    pub fn clear_path(&mut self) {
        self.path.clear();
        self.waypoint = 0;
        self.desired_velocity = V::ZERO;
    }

    /// The final waypoint, if any.
    pub fn target(&self) -> Option<V> {
        self.path.last().copied()
    }

    /// Whether the path is empty or the final waypoint is within
    /// [`NavigationAgent::arrival_radius`].
    pub fn is_finished(&self, position: V) -> bool {
        match self.target() {
            Some(target) => position.distance(target) <= self.arrival_radius,
            None => true,
        }
    }

    /// Advance past every waypoint the agent has already reached.
    pub fn advance(&mut self, position: V) {
        while self.waypoint + 1 < self.path.len()
            && position.distance(self.path[self.waypoint]) <= self.waypoint_advance
        {
            self.waypoint += 1;
        }
    }

    /// Seek the current waypoint, slowing into the final one. Caches the result
    /// in [`NavigationAgent::desired_velocity`].
    pub fn desired_velocity(&mut self, position: V) -> V {
        self.advance(position);
        if self.path.is_empty() {
            self.desired_velocity = V::ZERO;
            return V::ZERO;
        }
        let waypoint = self.path[self.waypoint];
        let distance = position.distance(waypoint);
        let is_final = self.waypoint + 1 >= self.path.len();
        if is_final && distance <= self.arrival_radius {
            self.desired_velocity = V::ZERO;
            return V::ZERO;
        }
        let direction = waypoint.sub(position).normalized();
        let speed = if is_final {
            let span = (distance - self.arrival_radius).max(0.0) / self.arrival_radius.max(1e-4);
            self.max_speed * span.min(1.0)
        } else {
            self.max_speed
        };
        self.desired_velocity = direction.mul(speed);
        self.desired_velocity
    }

    /// Apply avoidance to the cached desired velocity.
    ///
    /// Avoidance runs in the navigation plane, so a 3D agent dodges on the
    /// ground plane while keeping the vertical component of its desired
    /// velocity (the slope). Deterministic for a given neighbour list.
    pub fn avoidance_velocity(&self, position: V, neighbors: &[AgentNeighbor<V>]) -> V {
        if !self.avoidance_enabled || neighbors.is_empty() {
            return self.desired_velocity;
        }
        let plane: Vec<Neighbor2> = neighbors
            .iter()
            .map(|n| {
                let p = n.position.nav2();
                let v = n.velocity.nav2();
                Neighbor2 {
                    position: glam::Vec2::new(p[0], p[1]),
                    velocity: glam::Vec2::new(v[0], v[1]),
                    radius: n.radius,
                }
            })
            .collect();
        let desired = self.desired_velocity.nav2();
        let resolved = resolve(
            glam::Vec2::new(desired[0], desired[1]),
            glam::Vec2::new(position.nav2()[0], position.nav2()[1]),
            &plane,
            &self.avoidance,
        );
        V::from_nav2([resolved.x, resolved.y], self.desired_velocity.height())
    }
}
