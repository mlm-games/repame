//! Feature-gated GPU pass timing and pipeline statistics.
//!
//! The device decides which queries exist. Without `TIMESTAMP_QUERY` the
//! profiler is inert; `PIPELINE_STATISTICS_QUERY` is native-only and likewise
//! optional. A single frame's results are read back one frame after resolve.
//! The app must request the features when creating the device; otherwise the
//! profiler is inert.

use std::sync::{
    Arc,
    atomic::{AtomicU8, Ordering},
};

use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoder, ComputePassTimestampWrites, Device,
    PipelineStatisticsTypes, QuerySet, QuerySetDescriptor, QueryType, Queue, RenderPass,
    RenderPassTimestampWrites,
};

/// Statistics collected per rendered pass when the device supports them.
pub const DEFAULT_STATS: PipelineStatisticsTypes = PipelineStatisticsTypes::from_bits_truncate(
    PipelineStatisticsTypes::VERTEX_SHADER_INVOCATIONS.bits()
        | PipelineStatisticsTypes::CLIPPER_PRIMITIVES_OUT.bits()
        | PipelineStatisticsTypes::FRAGMENT_SHADER_INVOCATIONS.bits(),
);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProfilerSupport {
    pub timestamps: bool,
    pub statistics: bool,
}

impl ProfilerSupport {
    pub fn detect(device: &Device) -> Self {
        let features = device.features();
        Self {
            timestamps: features.contains(wgpu::Features::TIMESTAMP_QUERY),
            statistics: features.contains(wgpu::Features::PIPELINE_STATISTICS_QUERY),
        }
    }

    pub fn any(&self) -> bool {
        self.timestamps || self.statistics
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FrameTiming {
    pub pass_nanos: Vec<f64>,
    pub vertex_shader_invocations: u64,
    pub primitives_out: u64,
    pub fragment_shader_invocations: u64,
}

pub struct GpuProfiler {
    timestamps: Option<QuerySet>,
    statistics: Option<QuerySet>,
    stats: PipelineStatisticsTypes,
    stats_slots: u32,
    capacity: u32,
    next: u32,
    stats_next: u32,
    last_used: u32,
    last_stats: u32,
    resolve: Buffer,
    readback: [Buffer; 2],
    pending_index: usize,
    mapping_index: usize,
    status: Arc<AtomicU8>,
    mapping: bool,
    support: ProfilerSupport,
}

impl GpuProfiler {
    pub fn new(device: &Device, pass_capacity: u32, stats: PipelineStatisticsTypes) -> Self {
        let support = ProfilerSupport::detect(device);
        let capacity = pass_capacity.max(1);
        let stats_slots = stats.bits().count_ones().max(1);
        let timestamp_bytes = timestamp_region(capacity);
        let stats_bytes = if support.statistics {
            align256(u64::from(capacity) * u64::from(stats_slots) * 8)
        } else {
            0
        };
        let size = timestamp_bytes + stats_bytes;
        let timestamps = support.timestamps.then(|| {
            device.create_query_set(&QuerySetDescriptor {
                label: Some("repame_view3d_timestamps"),
                ty: QueryType::Timestamp,
                count: capacity * 2,
            })
        });
        let statistics = support.statistics.then(|| {
            device.create_query_set(&QuerySetDescriptor {
                label: Some("repame_view3d_pipeline_stats"),
                ty: QueryType::PipelineStatistics(stats),
                count: capacity * stats_slots,
            })
        });
        let resolve = device.create_buffer(&BufferDescriptor {
            label: Some("repame_view3d_query_resolve"),
            size,
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = std::array::from_fn(|index| {
            device.create_buffer(&BufferDescriptor {
                label: Some(if index == 0 {
                    "repame_view3d_query_readback_a"
                } else {
                    "repame_view3d_query_readback_b"
                }),
                size,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        });
        Self {
            timestamps,
            statistics,
            stats,
            stats_slots,
            capacity,
            next: 0,
            stats_next: 0,
            last_used: 0,
            last_stats: 0,
            resolve,
            readback,
            pending_index: 0,
            mapping_index: 0,
            status: Arc::new(AtomicU8::new(0)),
            mapping: false,
            support,
        }
    }

    pub fn support(&self) -> ProfilerSupport {
        self.support
    }

    pub fn timestamp_writes(&mut self) -> Option<RenderPassTimestampWrites<'_>> {
        let (index, set) = self.advance()?;
        let base = index * 2;
        Some(RenderPassTimestampWrites {
            query_set: set,
            beginning_of_pass_write_index: Some(base),
            end_of_pass_write_index: Some(base + 1),
        })
    }

    pub fn compute_timestamp_writes(&mut self) -> Option<ComputePassTimestampWrites<'_>> {
        let (index, set) = self.advance()?;
        let base = index * 2;
        Some(ComputePassTimestampWrites {
            query_set: set,
            beginning_of_pass_write_index: Some(base),
            end_of_pass_write_index: Some(base + 1),
        })
    }

    fn advance(&mut self) -> Option<(u32, &QuerySet)> {
        let set = self.timestamps.as_ref()?;
        if self.next >= self.capacity {
            return None;
        }
        let index = self.next;
        self.next += 1;
        Some((index, set))
    }

    pub fn begin_statistics(&mut self, pass: &mut RenderPass<'_>) {
        if self.stats_next >= self.capacity {
            return;
        }
        if let Some(set) = self.statistics.as_ref() {
            pass.begin_pipeline_statistics_query(set, self.stats_next * self.stats_slots);
            self.stats_next += 1;
        }
    }

    pub fn end_statistics(&self, pass: &mut RenderPass<'_>) {
        if self.statistics.is_some() {
            pass.end_pipeline_statistics_query();
        }
    }

    pub fn resolve(&mut self, encoder: &mut CommandEncoder) {
        let used = self.next;
        let stats_used = self.stats_next;
        if !self.support.any() || (used == 0 && stats_used == 0) {
            return;
        }
        if used > 0
            && let Some(set) = self.timestamps.as_ref()
        {
            encoder.resolve_query_set(set, 0..used * 2, &self.resolve, 0);
        }
        if stats_used > 0
            && let Some(set) = self.statistics.as_ref()
        {
            encoder.resolve_query_set(
                set,
                0..stats_used * self.stats_slots,
                &self.resolve,
                timestamp_region(self.capacity),
            );
        }
        let target = 1 - self.pending_index;
        if self.mapping && self.mapping_index == target {
            self.last_used = used;
            self.last_stats = stats_used;
            self.next = 0;
            self.stats_next = 0;
            return;
        }
        encoder.copy_buffer_to_buffer(
            &self.resolve,
            0,
            &self.readback[target],
            0,
            self.resolve.size(),
        );
        self.pending_index = target;
        self.last_used = used;
        self.last_stats = stats_used;
        self.next = 0;
        self.stats_next = 0;
    }

    pub fn read(&mut self, device: &Device, queue: &Queue) -> Option<FrameTiming> {
        if self.mapping {
            let _ = device.poll(wgpu::PollType::Poll);
            if self.status.load(Ordering::Acquire) == 0 {
                return None;
            }
            let ok = self.status.load(Ordering::Acquire) == 1;
            let timing = if ok {
                let view = self
                    .readback[self.mapping_index]
                    .slice(..)
                    .get_mapped_range()
                    .expect("readback is mapped");
                Some(self.decode(&view, queue))
            } else {
                None
            };
            if ok {
                self.readback[self.mapping_index].unmap();
            }
            self.mapping = false;
            self.status.store(0, Ordering::Release);
            return timing;
        }
        if !self.support.any() {
            return None;
        }
        let status = Arc::clone(&self.status);
        self.readback[self.pending_index]
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
            status.store(
                if result.is_ok() { 1 } else { 2 },
                Ordering::Release,
            );
        });
        self.mapping_index = self.pending_index;
        self.mapping = true;
        let _ = device.poll(wgpu::PollType::Poll);
        None
    }

    fn decode(&self, view: &[u8], queue: &Queue) -> FrameTiming {
        let words: &[u64] = bytemuck::cast_slice(view);
        let period = queue.get_timestamp_period() as f64;
        let pass_nanos = if self.support.timestamps {
            (0..self.last_used as usize)
                .filter_map(|pass| {
                    let start = words.get(pass * 2).copied()?;
                    let end = words.get(pass * 2 + 1).copied()?;
                    let delta = end.wrapping_sub(start) as f64;
                    Some(delta * period)
                })
                .collect()
        } else {
            Vec::new()
        };
        let stats_start = (timestamp_region(self.capacity) / 8) as usize;
        let mut out = FrameTiming {
            pass_nanos,
            ..Default::default()
        };
        if self.support.statistics {
            let slots: Vec<u64> = (0..self.last_stats as usize)
                .flat_map(|pass| {
                    (0..self.stats_slots as usize).map(move |slot| {
                        words
                            .get(stats_start + pass * self.stats_slots as usize + slot)
                            .copied()
                            .unwrap_or(0)
                    })
                })
                .collect();
            let mut slot = 0;
            let stats = self.stats;
            let flags = [
                PipelineStatisticsTypes::VERTEX_SHADER_INVOCATIONS,
                PipelineStatisticsTypes::CLIPPER_INVOCATIONS,
                PipelineStatisticsTypes::CLIPPER_PRIMITIVES_OUT,
                PipelineStatisticsTypes::FRAGMENT_SHADER_INVOCATIONS,
                PipelineStatisticsTypes::COMPUTE_SHADER_INVOCATIONS,
            ];
            let mut values = [0u64; 5];
            for (index, flag) in flags.iter().enumerate() {
                if stats.contains(*flag) {
                    values[index] = slots.get(slot).copied().unwrap_or(0);
                    slot += 1;
                }
            }
            out.vertex_shader_invocations = values[0];
            out.primitives_out = values[2];
            out.fragment_shader_invocations = values[3];
        }
        out
    }
}

fn align256(value: u64) -> u64 {
    value.div_ceil(256) * 256
}

fn timestamp_region(capacity: u32) -> u64 {
    align256(u64::from(capacity) * 2 * 8)
}
