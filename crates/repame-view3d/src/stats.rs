//! Per-frame draw counters, the numbers a port needs to find its own cliffs.
//!
//! Counters accumulate during the batch's prepare and are readable after
//! paint, so a game can log them, drive a debug overlay, or gate a quality
//! setting on them without reaching into the renderer.

/// Counters for one frame's scene submission.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameStats {
    /// Draw calls issued, after degenerate-triangle culling.
    pub draw_calls: u32,
    /// Draw calls the batch merged because they shared pipeline state.
    pub merged_draw_calls: u32,
    /// Triangles submitted.
    pub triangles: u32,
    /// Vertices uploaded.
    pub vertices: u32,
    /// Indices uploaded.
    pub indices: u32,
    /// Pipeline creations, i.e. how often shader state changed shape.
    pub pipelines_created: u32,
    /// Texture bytes staged for upload.
    pub texture_bytes: u64,
    /// Groups dropped before submission.
    pub culled_groups: u32,
    /// Shadow depth draws.
    pub shadow_draws: u32,
    /// Skinned draws.
    pub skinned_draws: u32,
    /// Offscreen passes executed.
    pub passes: u32,
    /// Triangles removed as degenerate.
    pub degenerate_triangles: u32,
}

impl FrameStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Folds another frame's counters in, for rolling windows.
    pub fn accumulate(&mut self, other: &FrameStats) {
        self.draw_calls += other.draw_calls;
        self.merged_draw_calls += other.merged_draw_calls;
        self.triangles += other.triangles;
        self.vertices += other.vertices;
        self.indices += other.indices;
        self.pipelines_created += other.pipelines_created;
        self.texture_bytes += other.texture_bytes;
        self.culled_groups += other.culled_groups;
        self.shadow_draws += other.shadow_draws;
        self.skinned_draws += other.skinned_draws;
        self.passes += other.passes;
        self.degenerate_triangles += other.degenerate_triangles;
    }

    /// Draw calls that were not merged away.
    pub fn effective_draw_calls(&self) -> u32 {
        self.draw_calls.saturating_sub(self.merged_draw_calls)
    }

    /// Mean triangles per effective draw call.
    pub fn triangles_per_draw(&self) -> f32 {
        let calls = self.effective_draw_calls();
        if calls == 0 {
            0.0
        } else {
            self.triangles as f32 / calls as f32
        }
    }

    pub fn is_empty(&self) -> bool {
        *self == FrameStats::default()
    }
}

/// Where the batch publishes its counters each frame.
pub trait StatsSink {
    fn publish(&mut self, stats: FrameStats);
}

impl StatsSink for Option<Box<dyn StatsSink>> {
    fn publish(&mut self, stats: FrameStats) {
        if let Some(sink) = self.as_mut() {
            sink.publish(stats);
        }
    }
}

/// A sink that keeps the most recent frame, for debug overlays.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LatestFrame {
    pub stats: FrameStats,
    pub frames_seen: u64,
}

impl LatestFrame {
    pub fn new() -> Self {
        Self::default()
    }
}

impl StatsSink for LatestFrame {
    fn publish(&mut self, stats: FrameStats) {
        self.stats = stats;
        self.frames_seen += 1;
    }
}

/// A sink that averages over a sliding window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RollingAverage {
    pub average: FrameStats,
    pub samples: u64,
    pub capacity: u64,
}

impl RollingAverage {
    pub fn new(capacity: u64) -> Self {
        Self {
            average: FrameStats::default(),
            samples: 0,
            capacity: capacity.max(1),
        }
    }
}

impl StatsSink for RollingAverage {
    fn publish(&mut self, stats: FrameStats) {
        let count = (self.samples + 1) as f32;
        let weight = 1.0 / count;
        let blend = |slot: &mut u32, value: u32| {
            *slot = (*slot as f32 + (value as f32 - *slot as f32) * weight).round() as u32;
        };
        blend(&mut self.average.draw_calls, stats.draw_calls);
        blend(&mut self.average.triangles, stats.triangles);
        blend(&mut self.average.vertices, stats.vertices);
        blend(&mut self.average.indices, stats.indices);
        blend(&mut self.average.pipelines_created, stats.pipelines_created);
        blend(&mut self.average.culled_groups, stats.culled_groups);
        blend(&mut self.average.shadow_draws, stats.shadow_draws);
        blend(&mut self.average.skinned_draws, stats.skinned_draws);
        blend(&mut self.average.passes, stats.passes);
        self.average.texture_bytes = ((self.average.texture_bytes as f32
            + (stats.texture_bytes as f32 - self.average.texture_bytes as f32) * weight)
            .round()) as u64;
        self.samples += 1;
    }
}
