//! Mesh level-of-detail: distance thresholds with hysteresis.
//!
//! A [`LodGroup`] owns the per-level meshes of one object (or voxel chunk) and
//! a [`LodSelector`] that picks a level from the eye distance. Selection is
//! pure math: [`select_level`] is the whole policy, and a test pins it.
//!
//! Hysteresis exists because a level boundary is a discontinuity: without it,
//! an eye distance oscillating by a centimetre across the threshold would swap
//! meshes every frame. The rule is one-sided per direction:
//!
//! ```text
//! coarsen (level n -> n+1) only above threshold[n] * (1 + hysteresis)
//! refine  (level n -> n-1) only below threshold[n-1] * (1 - hysteresis)
//! ```
//!
//! `threshold[i]` is the largest eye distance that still selects level `i`;
//! the coarsest level has no threshold and covers everything beyond the last
//! one. Distances are measured from the eye to [`LodGroup::center`] (pass the
//! group's bounds center, or the voxel chunk's center), so a group never
//! flickers because its geometry moved within its own bounds.
//!
//! The voxel path builds the meshes: [`build_chunk_mesh_lod`](crate::voxel::build_chunk_mesh_lod)
//! aggregates blocks at `stride` and meshes the coarse grid, so a chunk at
//! level `n` is a real coarser mesh, not the same triangles at a lower rate.

use glam::Vec3;

use super::mesh::MeshGroup;
use super::viewport::Frame3d;

/// Levels one selector accepts (thresholds + the unbounded coarsest level).
pub const MAX_LOD_LEVELS: usize = 8;

/// Why an LOD chain was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LodError {
    /// No levels at all.
    Empty,
    /// More levels than [`MAX_LOD_LEVELS`].
    TooManyLevels(usize),
    /// The thresholds were not finite and strictly increasing.
    InvalidThresholds,
}

impl std::fmt::Display for LodError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "lod chain has no levels"),
            Self::TooManyLevels(n) => write!(f, "{n} lod levels past the {MAX_LOD_LEVELS} cap"),
            Self::InvalidThresholds => write!(f, "lod thresholds must be finite and increasing"),
        }
    }
}

impl std::error::Error for LodError {}

/// Pure selection policy: the level for `distance`, biased by `previous` and
/// `hysteresis`. See the module docs for the rule.
pub fn select_level(
    max_distances: &[f32],
    distance: f32,
    previous: Option<usize>,
    hysteresis: f32,
) -> usize {
    let levels = max_distances.len() + 1;
    if !distance.is_finite() {
        return previous.unwrap_or(levels - 1).min(levels - 1);
    }
    let distance = distance.max(0.0);
    let target = max_distances
        .iter()
        .position(|threshold| distance <= *threshold)
        .unwrap_or(levels - 1);
    let hysteresis = if hysteresis.is_finite() {
        hysteresis.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let Some(previous) = previous.filter(|level| *level < levels) else {
        return target;
    };
    if target == previous {
        return previous;
    }
    if target > previous {
        let threshold = max_distances[previous];
        if distance > threshold * (1.0 + hysteresis) {
            target
        } else {
            previous
        }
    } else {
        let threshold = max_distances[target];
        if distance < threshold * (1.0 - hysteresis) {
            target
        } else {
            previous
        }
    }
}

/// Per-group level picker: thresholds, hysteresis and the last level, so the
/// hysteresis has state to work with.
#[derive(Clone, Debug, PartialEq)]
pub struct LodSelector {
    max_distances: Vec<f32>,
    hysteresis: f32,
    level: usize,
    changes: u64,
}

impl LodSelector {
    /// Validates `max_distances` (finite, strictly increasing, at most
    /// [`MAX_LOD_LEVELS`] - 1 entries) and starts on the finest level.
    pub fn new(max_distances: Vec<f32>, hysteresis: f32) -> Result<Self, LodError> {
        if max_distances.len() + 1 > MAX_LOD_LEVELS {
            return Err(LodError::TooManyLevels(max_distances.len() + 1));
        }
        if max_distances
            .iter()
            .any(|d| !d.is_finite() || *d <= 0.0)
            || max_distances.windows(2).any(|w| w[0] >= w[1])
        {
            return Err(LodError::InvalidThresholds);
        }
        Ok(Self {
            max_distances,
            hysteresis: if hysteresis.is_finite() {
                hysteresis.clamp(0.0, 1.0)
            } else {
                0.0
            },
            level: 0,
            changes: 0,
        })
    }

    /// Picks a level for `distance` and remembers it.
    pub fn select(&mut self, distance: f32) -> usize {
        let next = select_level(
            &self.max_distances,
            distance,
            Some(self.level),
            self.hysteresis,
        );
        if next != self.level {
            self.level = next;
            self.changes += 1;
        }
        next
    }

    /// Last selected level.
    pub fn level(&self) -> usize {
        self.level
    }

    /// Level switches so far (a quality readout; a flickering setup shows a
    /// climbing count).
    pub fn changes(&self) -> u64 {
        self.changes
    }

    /// Number of levels this selector can pick.
    pub fn levels(&self) -> usize {
        self.max_distances.len() + 1
    }

    /// Thresholds, finest first.
    pub fn max_distances(&self) -> &[f32] {
        &self.max_distances
    }
}

/// One object's level meshes plus the selector that picks among them.
#[derive(Clone, Debug)]
pub struct LodGroup {
    /// Meshes, finest first; `levels[i]` is drawn while the selector sits on
    /// level `i`.
    pub levels: Vec<MeshGroup>,
    /// Picks the level from the eye distance.
    pub selector: LodSelector,
    /// Bounds center distances are measured from.
    pub center: [f32; 3],
}

impl LodGroup {
    /// A group whose `levels[i]` covers up to `max_distances[i]` (last level
    /// unbounded). Requires one threshold fewer than meshes.
    pub fn new(
        levels: Vec<MeshGroup>,
        max_distances: Vec<f32>,
        hysteresis: f32,
        center: [f32; 3],
    ) -> Result<Self, LodError> {
        if levels.is_empty() {
            return Err(LodError::Empty);
        }
        if max_distances.len() + 1 != levels.len() {
            return Err(LodError::InvalidThresholds);
        }
        Ok(Self {
            levels,
            selector: LodSelector::new(max_distances, hysteresis)?,
            center,
        })
    }

    /// Updates the selection for `eye` and returns the chosen level.
    pub fn update(&mut self, eye: Vec3) -> usize {
        let distance = (eye - Vec3::from(self.center)).length();
        self.selector.select(distance)
    }

    /// The mesh `eye` selects, or `None` when the level slot is missing.
    pub fn visible(&mut self, eye: Vec3) -> Option<&MeshGroup> {
        let level = self.update(eye);
        self.levels.get(level)
    }

    /// Appends the mesh `eye` selects to a frame. Returns false (and pushes
    /// nothing) when the level slot is missing.
    pub fn push_visible(&mut self, frame: &mut Frame3d, eye: Vec3) -> bool {
        match self.visible(eye) {
            Some(group) if !group.is_empty() => {
                frame.push(group.clone());
                true
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tri_group(tag: f32) -> MeshGroup {
        let mut group = MeshGroup {
            depth_test: true,
            ..Default::default()
        };
        group.push_tri(
            [tag, 0.0, 0.0],
            [tag + 1.0, 0.0, 0.0],
            [tag, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        );
        group
    }

    /// The hysteresis band is one-sided: the threshold is crossed generously
    /// when coarsening and required to be cleared when refining.
    #[test]
    fn hysteresis_bands_pin_both_directions() {
        let thresholds = [10.0f32, 20.0];
        // Stateless selection: exactly the buckets.
        assert_eq!(select_level(&thresholds, 5.0, None, 0.1), 0);
        assert_eq!(select_level(&thresholds, 10.0, None, 0.1), 0);
        assert_eq!(select_level(&thresholds, 10.5, None, 0.1), 1);
        assert_eq!(select_level(&thresholds, 20.0, None, 0.1), 1);
        assert_eq!(select_level(&thresholds, 100.0, None, 0.1), 2);

        // Coarsening from level 0 needs distance > 10 * 1.1.
        assert_eq!(select_level(&thresholds, 10.5, Some(0), 0.1), 0);
        assert_eq!(select_level(&thresholds, 11.0, Some(0), 0.1), 0);
        assert_eq!(select_level(&thresholds, 11.5, Some(0), 0.1), 1);

        // Refining from level 1 needs distance < 10 * 0.9.
        assert_eq!(select_level(&thresholds, 9.5, Some(1), 0.1), 1);
        assert_eq!(select_level(&thresholds, 9.0, Some(1), 0.1), 1);
        assert_eq!(select_level(&thresholds, 8.5, Some(1), 0.1), 0);

        // Zero hysteresis is the plain bucket: both boundaries switch.
        assert_eq!(select_level(&thresholds, 10.1, Some(0), 0.0), 1);
        assert_eq!(select_level(&thresholds, 9.9, Some(1), 0.0), 0);

        // Coarsening from the middle straight past the coarsest threshold
        // needs the level-1 band, not the final one.
        assert_eq!(select_level(&thresholds, 25.0, Some(1), 0.1), 2);
        assert_eq!(select_level(&thresholds, 21.5, Some(1), 0.1), 1);

        // Non-finite distances keep the current level.
        assert_eq!(select_level(&thresholds, f32::NAN, Some(1), 0.1), 1);
    }

    /// The selector tracks its state and counts only real switches.
    #[test]
    fn selector_counts_switches() {
        let mut selector = LodSelector::new(vec![10.0, 20.0], 0.1).expect("valid");
        assert_eq!(selector.levels(), 3);
        assert_eq!(selector.select(5.0), 0);
        assert_eq!(selector.changes(), 0);
        assert_eq!(selector.select(15.0), 1);
        assert_eq!(selector.select(30.0), 2);
        assert_eq!(selector.select(15.0), 1);
        assert_eq!(selector.changes(), 3);
        assert_eq!(selector.level(), 1);
        // Bouncing inside the hysteresis band does not switch.
        let mut selector = LodSelector::new(vec![10.0], 0.5).expect("valid");
        assert_eq!(selector.select(13.0), 0);
        assert_eq!(selector.select(9.0), 0);
        assert_eq!(selector.changes(), 0);
    }

    /// Threshold validation rejects inverted, empty and oversized chains.
    #[test]
    fn selector_validation() {
        assert_eq!(LodSelector::new(vec![], 0.1).map(|s| s.levels()), Ok(1));
        assert_eq!(
            LodSelector::new(vec![20.0, 10.0], 0.1),
            Err(LodError::InvalidThresholds)
        );
        assert_eq!(
            LodSelector::new(vec![0.0], 0.1),
            Err(LodError::InvalidThresholds)
        );
        assert_eq!(
            LodSelector::new(vec![f32::NAN], 0.1),
            Err(LodError::InvalidThresholds)
        );
        assert!(matches!(
            LodSelector::new(vec![1.0; MAX_LOD_LEVELS], 0.1),
            Err(LodError::TooManyLevels(_))
        ));
        assert!(matches!(
            LodGroup::new(Vec::new(), Vec::new(), 0.1, [0.0; 3]),
            Err(LodError::Empty)
        ));
        assert!(matches!(
            LodGroup::new(vec![tri_group(0.0)], vec![10.0], 0.1, [0.0; 3]),
            Err(LodError::InvalidThresholds)
        ));
    }

    /// A group swaps meshes with the eye distance and reports the level it
    /// drew, taking distance from its own center.
    #[test]
    fn group_swaps_meshes_by_distance() {
        let mut group =
            LodGroup::new(vec![tri_group(0.0), tri_group(10.0)], vec![10.0], 0.1, [0.0; 3])
                .expect("valid");
        let eye = Vec3::new(5.0, 0.0, 0.0);
        assert_eq!(group.update(eye), 0);
        assert_eq!(group.visible(eye).map(|g| g.positions[0][0]), Some(0.0));
        let far = Vec3::new(30.0, 0.0, 0.0);
        assert_eq!(group.update(far), 1);
        assert_eq!(group.visible(far).map(|g| g.positions[0][0]), Some(10.0));
        // Back inside the hysteresis band: still the coarse mesh.
        let band = Vec3::new(9.5, 0.0, 0.0);
        assert_eq!(group.update(band), 1);
        // Well inside: back to the fine mesh.
        let close = Vec3::new(8.0, 0.0, 0.0);
        assert_eq!(group.update(close), 0);
    }
}
