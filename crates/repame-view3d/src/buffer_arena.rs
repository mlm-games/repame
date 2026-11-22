//! Sub-allocation of one GPU buffer into aligned ranges.
//!
//! A single allocation is carved into offsets instead of creating a buffer per
//! draw. Freed ranges are coalesced so a later allocation can reuse them.

use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Alloc {
    pub offset: u64,
    pub size: u64,
}

impl Alloc {
    pub fn end(&self) -> u64 {
        self.offset + self.size
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArenaError {
    InvalidAlignment,
    InvalidSize,
    OutOfMemory,
}

#[derive(Clone, Debug)]
pub struct BufferArena {
    capacity: u64,
    alignment: u64,
    used: u64,
    free: BTreeMap<u64, u64>,
}

impl BufferArena {
    pub fn new(capacity: u64, alignment: u64) -> Result<Self, ArenaError> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(ArenaError::InvalidAlignment);
        }
        let mut free = BTreeMap::new();
        if capacity > 0 {
            free.insert(0, capacity);
        }
        Ok(Self {
            capacity,
            alignment,
            used: 0,
            free,
        })
    }

    pub fn alloc(&mut self, size: u64) -> Result<Alloc, ArenaError> {
        if size == 0 {
            return Err(ArenaError::InvalidSize);
        }
        let mut found = None;
        for (&offset, &len) in &self.free {
            let base = align_up(offset, self.alignment);
            let padding = base - offset;
            if len >= padding && len - padding >= size {
                found = Some((offset, len, base, padding));
                break;
            }
        }
        let (offset, len, base, padding) = found.ok_or(ArenaError::OutOfMemory)?;
        self.free.remove(&offset);
        if padding > 0 {
            self.free.insert(offset, padding);
        }
        let rest = len - padding - size;
        if rest > 0 {
            self.free.insert(base + size, rest);
        }
        self.used += size;
        Ok(Alloc {
            offset: base,
            size,
        })
    }

    pub fn free(&mut self, alloc: Alloc) {
        if alloc.size == 0 || alloc.offset % self.alignment != 0 {
            return;
        }
        let mut start = alloc.offset;
        let mut end = alloc.end();
        if let Some((&before, &len)) = self.free.range(..=start).next_back() {
            if before + len == start {
                self.free.remove(&before);
                start = before;
            }
        }
        if let Some((&after, &len)) = self.free.range(start..).next() {
            if end == after {
                self.free.remove(&after);
                end = after + len;
            }
        }
        self.free.insert(start, end - start);
        self.used = self.used.saturating_sub(alloc.size);
    }

    pub fn clear(&mut self) {
        self.free.clear();
        if self.capacity > 0 {
            self.free.insert(0, self.capacity);
        }
        self.used = 0;
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    pub fn used(&self) -> u64 {
        self.used
    }

    pub fn free_bytes(&self) -> u64 {
        self.capacity - self.used
    }

    pub fn regions(&self) -> usize {
        self.free.len()
    }
}

fn align_up(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

pub struct GpuArena {
    buffer: wgpu::Buffer,
    arena: BufferArena,
}

impl GpuArena {
    pub fn new(
        device: &wgpu::Device,
        label: &str,
        capacity: u64,
        alignment: u64,
        usage: wgpu::BufferUsages,
    ) -> Result<Self, ArenaError> {
        let arena = BufferArena::new(capacity, alignment)?;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: capacity,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self { buffer, arena })
    }

    pub fn alloc(&mut self, size: u64) -> Result<Alloc, ArenaError> {
        self.arena.alloc(size)
    }

    pub fn free(&mut self, alloc: Alloc) {
        self.arena.free(alloc);
    }

    pub fn clear(&mut self) {
        self.arena.clear();
    }

    pub fn write(&self, queue: &wgpu::Queue, alloc: Alloc, bytes: &[u8]) {
        queue.write_buffer(&self.buffer, alloc.offset, bytes);
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn used(&self) -> u64 {
        self.arena.used()
    }

    pub fn capacity(&self) -> u64 {
        self.arena.capacity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_allocation_reuses_coalesced_ranges() {
        let mut arena = BufferArena::new(64, 16).unwrap();
        let a = arena.alloc(20).unwrap();
        let b = arena.alloc(8).unwrap();
        let c = arena.alloc(8).unwrap();
        assert_eq!((a.offset, b.offset, c.offset), (0, 32, 48));
        assert_eq!(arena.alloc(16), Err(ArenaError::OutOfMemory));

        arena.free(b);
        arena.free(a);
        arena.free(c);
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.regions(), 1);

        let whole = arena.alloc(64).unwrap();
        assert_eq!(whole.offset, 0);
        assert_eq!(arena.alloc(1), Err(ArenaError::OutOfMemory));
        assert_eq!(arena.alloc(0), Err(ArenaError::InvalidSize));
    }
}
