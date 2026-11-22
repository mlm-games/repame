//! Compute-side ray picking over mesh snapshots.
//!
//! One invocation per triangle, nearest hit wins through an atomic compare
//! loop. The result is a `pick_id` read back one frame later. CPU picking in
//! [`crate::pick`] stays the authoritative synchronous path; this feeds games
//! that need GPU throughput on dense scenes.

use std::sync::{
    Arc,
    atomic::{AtomicU8, Ordering},
};

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::mesh::MeshGroup;

const WORKGROUP: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct RayUniform {
    origin: [f32; 4],
    dir: [f32; 4],
    count: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PickTriangle {
    a: [f32; 4],
    b: [f32; 4],
    c: [f32; 4],
    pick_id: u32,
    _pad: [u32; 3],
}

impl PickTriangle {
    fn from_vertices(a: [f32; 3], b: [f32; 3], c: [f32; 3], pick_id: u32) -> Self {
        Self {
            a: [a[0], a[1], a[2], 0.0],
            b: [b[0], b[1], b[2], 0.0],
            c: [c[0], c[1], c[2], 0.0],
            pick_id,
            _pad: [0; 3],
        }
    }
}

/// Triangles flattened for the compute pass. Empty when no group opts in.
#[derive(Clone, Debug, Default)]
pub struct PickTriangles {
    pub items: Vec<PickTriangle>,
}

impl PickTriangles {
    pub fn from_groups(groups: &[MeshGroup]) -> Self {
        let mut items = Vec::new();
        for group in groups {
            if group.pick_id == 0 {
                continue;
            }
            for tri in group.indices.as_chunks::<3>().0 {
                let Some(a) = group.positions.get(tri[0] as usize) else {
                    continue;
                };
                let Some(b) = group.positions.get(tri[1] as usize) else {
                    continue;
                };
                let Some(c) = group.positions.get(tri[2] as usize) else {
                    continue;
                };
                items.push(PickTriangle::from_vertices(*a, *b, *c, group.pick_id));
            }
        }
        Self { items }
    }
}

const PICK_WGSL: &str = r#"
struct Ray {
    origin: vec4<f32>,
    dir: vec4<f32>,
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};
struct Tri {
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
    pick_id: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};
struct Result {
    distance: atomic<u32>,
    pick_id: atomic<u32>,
    lock: atomic<u32>,
};
@group(0) @binding(0) var<uniform> ray: Ray;
@group(0) @binding(1) var<storage, read> tris: array<Tri>;
@group(0) @binding(2) var<storage, read_write> result: Result;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= ray.count) { return; }
    let tri = tris[index];
    let e1 = tri.b.xyz - tri.a.xyz;
    let e2 = tri.c.xyz - tri.a.xyz;
    let p = cross(ray.dir.xyz, e2);
    let det = dot(e1, p);
    if (abs(det) < 1e-12) { return; }
    let inv = 1.0 / det;
    let t = ray.origin.xyz - tri.a.xyz;
    let u = dot(t, p) * inv;
    if (u < 0.0 || u > 1.0) { return; }
    let q = cross(t, e1);
    let v = dot(ray.dir.xyz, q) * inv;
    if (v < 0.0 || u + v > 1.0) { return; }
    let distance = dot(e2, q) * inv;
    if (distance <= 0.0) { return; }
    let key = bitcast<u32>(max(distance, 0.0));
    loop {
        if (atomicCompareExchangeWeak(&result.lock, 0u, 1u).exchanged) {
            if (key < atomicLoad(&result.distance)) {
                atomicStore(&result.distance, key);
                atomicStore(&result.pick_id, tri.pick_id);
            }
            atomicStore(&result.lock, 0u);
            break;
        }
    }
}
"#;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PickKind {
    Hover,
    Click,
}

#[derive(Clone, Debug)]
pub struct PickRequest {
    pub id: u64,
    pub kind: PickKind,
    pub origin: Vec3,
    pub dir: Vec3,
    pub triangles: PickTriangles,
}

#[derive(Clone, Copy, Debug)]
pub struct PickResult {
    pub id: u64,
    pub kind: PickKind,
    pub origin: Vec3,
    pub dir: Vec3,
    pub pick_id: u32,
    pub distance: f32,
}

struct PickMeta {
    id: u64,
    kind: PickKind,
    origin: Vec3,
    dir: Vec3,
}

pub struct PickFeed {
    picker: GpuPicker,
    pending: std::collections::VecDeque<PickMeta>,
}

impl PickFeed {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            picker: GpuPicker::new(device),
            pending: std::collections::VecDeque::new(),
        }
    }

    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        request: &PickRequest,
    ) {
        self.picker.encode(
            device,
            queue,
            encoder,
            &request.triangles.items,
            request.origin,
            request.dir,
        );
        self.pending.push_back(PickMeta {
            id: request.id,
            kind: request.kind,
            origin: request.origin,
            dir: request.dir,
        });
    }

    pub fn take(&mut self, device: &wgpu::Device) -> Option<PickResult> {
        let (pick_id, distance) = self.picker.take(device)?;
        let meta = self.pending.pop_front()?;
        Some(PickResult {
            id: meta.id,
            kind: meta.kind,
            origin: meta.origin,
            dir: meta.dir,
            pick_id,
            distance,
        })
    }
}

pub struct GpuPicker {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    ray: wgpu::Buffer,
    triangles: wgpu::Buffer,
    capacity: usize,
    result: wgpu::Buffer,
    readback: [wgpu::Buffer; 2],
    pending_index: usize,
    mapping_index: usize,
    status: Arc<AtomicU8>,
    mapping: bool,
}

impl GpuPicker {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("repame_view3d_pick"),
            source: wgpu::ShaderSource::Wgsl(PICK_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("repame_view3d_pick_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("repame_view3d_pick_pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("repame_view3d_pick"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let buffer = |label: &'static str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        Self {
            pipeline,
            layout,
            ray: buffer(
                "repame_view3d_pick_ray",
                size_of::<RayUniform>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            triangles: buffer(
                "repame_view3d_pick_tris",
                64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            capacity: 0,
            result: buffer(
                "repame_view3d_pick_result",
                12,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            readback: std::array::from_fn(|index| {
                buffer(
                    if index == 0 {
                        "repame_view3d_pick_readback_a"
                    } else {
                        "repame_view3d_pick_readback_b"
                    },
                    12,
                    wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                )
            }),
            pending_index: 0,
            mapping_index: 0,
            status: Arc::new(AtomicU8::new(0)),
            mapping: false,
        }
    }

    /// Uploads `triangles` and the ray, clears the result and dispatches.
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        triangles: &[PickTriangle],
        origin: Vec3,
        dir: Vec3,
    ) {
        if self.mapping {
            device.poll(wgpu::PollType::Poll).ok();
        }
        let uniform = RayUniform {
            origin: [origin.x, origin.y, origin.z, 0.0],
            dir: [dir.x, dir.y, dir.z, 0.0],
            count: triangles.len() as u32,
            _pad: [0; 3],
        };
        queue.write_buffer(&self.ray, 0, bytemuck::bytes_of(&uniform));
        if triangles.len() > self.capacity {
            self.capacity = triangles.len().next_power_of_two().max(64);
            self.triangles = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("repame_view3d_pick_tris"),
                size: (self.capacity * size_of::<PickTriangle>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if !triangles.is_empty() {
            queue.write_buffer(&self.triangles, 0, bytemuck::cast_slice(triangles));
        }
        queue.write_buffer(
            &self.result,
            0,
            &[0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0],
        );
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("repame_view3d_pick_bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.ray.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.triangles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.result.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("repame_view3d_pick"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(triangles.len().div_ceil(WORKGROUP as usize) as u32, 1, 1);
        drop(pass);
        let target = 1 - self.pending_index;
        if !(self.mapping && self.mapping_index == target) {
            encoder.copy_buffer_to_buffer(&self.result, 0, &self.readback[target], 0, 12);
            self.pending_index = target;
        }
    }

    /// Returns `(pick_id, distance)` of the nearest hit, or `None` when no
    /// hit or not ready.
    pub fn take(&mut self, device: &wgpu::Device) -> Option<(u32, f32)> {
        if self.mapping {
            device.poll(wgpu::PollType::Poll).ok();
            if self.status.load(Ordering::Acquire) == 0 {
                return None;
            }
            let ok = self.status.load(Ordering::Acquire) == 1;
            let id = if ok {
                let view = self.readback[self.mapping_index]
                    .slice(..)
                    .get_mapped_range()
                    .expect("pick readback is mapped");
                let words: &[u32] = bytemuck::cast_slice(&view);
                let id = *words.get(1)?;
                let distance_bits = *words.get(0)?;
                (id != 0).then_some((id, f32::from_bits(distance_bits)))
            } else {
                None
            };
            if ok {
                self.readback[self.mapping_index].unmap();
            }
            self.mapping = false;
            self.status.store(0, Ordering::Release);
            return id;
        }
        let status = Arc::clone(&self.status);
        self.readback[self.pending_index]
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                status.store(if result.is_ok() { 1 } else { 2 }, Ordering::Release);
            });
        self.mapping_index = self.pending_index;
        self.mapping = true;
        device.poll(wgpu::PollType::Poll).ok();
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_shader_parses() {
        let module = naga::front::wgsl::parse_str(PICK_WGSL).expect("pick WGSL parses");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("pick WGSL validates");
    }

    #[test]
    fn groups_without_pick_id_are_skipped() {
        let mut group = MeshGroup::default();
        group.pick_id = 0;
        group.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        group.indices = vec![0, 1, 2];
        assert!(PickTriangles::from_groups(&[group]).items.is_empty());

        let mut group = MeshGroup::default();
        group.pick_id = 7;
        group.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        group.indices = vec![0, 1, 2];
        let tris = PickTriangles::from_groups(&[group]);
        assert_eq!(tris.items.len(), 1);
        assert_eq!(tris.items[0].pick_id, 7);
    }
}
