//! Stable key to dense index mapping with a reserved fallback slot.
//!
//! Shaders address a bounded texture array by index. A table maps a stable
//! 64-bit key to such an index so an asset's slot survives unrelated loads.
//! Index zero is reserved for misses.

pub const FALLBACK_INDEX: u32 = 0;
pub const DEFAULT_PROBE_LIMIT: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DescriptorTableError {
    InvalidCapacity,
    Full,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    Empty,
    Tombstone,
    Used { key: u64, index: u32 },
}

#[derive(Clone, Debug)]
pub struct DescriptorTable {
    slots: Vec<Slot>,
    mask: usize,
    free: Vec<u32>,
    next: u32,
    max_index: u32,
    used: usize,
    tombstones: usize,
    probe_limit: usize,
}

impl DescriptorTable {
    pub fn new(requested_slots: usize) -> Result<Self, DescriptorTableError> {
        Self::with_probe_limit(requested_slots, DEFAULT_PROBE_LIMIT)
    }

    pub fn with_probe_limit(
        requested_slots: usize,
        probe_limit: usize,
    ) -> Result<Self, DescriptorTableError> {
        if requested_slots < 2 || probe_limit == 0 {
            return Err(DescriptorTableError::InvalidCapacity);
        }
        let mut slots = 1usize;
        while slots < requested_slots {
            slots <<= 1;
        }
        if slots > u32::MAX as usize {
            return Err(DescriptorTableError::InvalidCapacity);
        }
        Ok(Self {
            slots: vec![Slot::Empty; slots],
            mask: slots - 1,
            free: Vec::new(),
            next: 1,
            max_index: (slots - 1) as u32,
            used: 0,
            tombstones: 0,
            probe_limit,
        })
    }

    pub fn insert(&mut self, key: u64) -> Result<u32, DescriptorTableError> {
        if let Some((_, index)) = self.find(key) {
            return Ok(index);
        }
        let index = if let Some(index) = self.free.pop() {
            index
        } else if self.next <= self.max_index {
            let index = self.next;
            self.next += 1;
            index
        } else {
            return Err(DescriptorTableError::Full);
        };

        let mut target = None;
        let mut first_tombstone = None;
        let start = Self::hash(key) & self.mask;
        for step in 0..self.probe_limit {
            let at = (start + step) & self.mask;
            match self.slots[at] {
                Slot::Empty => {
                    target = Some(at);
                    break;
                }
                Slot::Tombstone => {
                    if first_tombstone.is_none() {
                        first_tombstone = Some(at);
                    }
                }
                Slot::Used { .. } => {}
            }
        }
        let at = match target.or(first_tombstone) {
            Some(at) => at,
            None => {
                if self.used + self.tombstones + 1 >= self.slots.len() {
                    return Err(DescriptorTableError::Full);
                }
                self.rehash();
                let start = Self::hash(key) & self.mask;
                (0..self.slots.len())
                    .map(|step| (start + step) & self.mask)
                    .find(|at| self.slots[*at] == Slot::Empty)
                    .ok_or(DescriptorTableError::Full)?
            }
        };
        if self.slots[at] == Slot::Tombstone {
            self.tombstones -= 1;
        }
        self.slots[at] = Slot::Used { key, index };
        self.used += 1;
        if (self.used + self.tombstones) * 4 >= self.slots.len() * 3 {
            self.rehash();
        }
        Ok(index)
    }

    pub fn get(&self, key: u64) -> Option<u32> {
        self.find(key).map(|(_, index)| index)
    }

    pub fn resolve(&self, key: u64) -> u32 {
        self.get(key).unwrap_or(FALLBACK_INDEX)
    }

    pub fn remove(&mut self, key: u64) -> bool {
        let Some((at, index)) = self.find(key) else {
            return false;
        };
        self.slots[at] = Slot::Tombstone;
        self.free.push(index);
        self.used -= 1;
        self.tombstones += 1;
        true
    }

    pub fn clear(&mut self) {
        self.slots.fill(Slot::Empty);
        self.free.clear();
        self.next = 1;
        self.used = 0;
        self.tombstones = 0;
    }

    pub fn len(&self) -> usize {
        self.used
    }

    pub fn is_empty(&self) -> bool {
        self.used == 0
    }

    pub fn capacity(&self) -> u32 {
        self.max_index
    }

    fn find(&self, key: u64) -> Option<(usize, u32)> {
        Self::find_in(&self.slots, self.mask, key, self.probe_limit)
    }

    fn find_in(slots: &[Slot], mask: usize, key: u64, limit: usize) -> Option<(usize, u32)> {
        let start = Self::hash(key) & mask;
        for step in 0..limit {
            let at = (start + step) & mask;
            match slots[at] {
                Slot::Empty => return None,
                Slot::Tombstone => {}
                Slot::Used {
                    key: candidate,
                    index,
                } => {
                    if candidate == key {
                        return Some((at, index));
                    }
                }
            }
        }
        None
    }

    fn rehash(&mut self) {
        let live: Vec<(u64, u32)> = self
            .slots
            .iter()
            .filter_map(|slot| match slot {
                Slot::Used { key, index } => Some((*key, *index)),
                _ => None,
            })
            .collect();
        self.slots.fill(Slot::Empty);
        self.tombstones = 0;
        for (key, index) in live {
            let start = Self::hash(key) & self.mask;
            let at = (0..self.slots.len())
                .map(|step| (start + step) & self.mask)
                .find(|at| *self.slots.get(*at).unwrap() == Slot::Empty)
                .expect("live entries fit in the table");
            self.slots[at] = Slot::Used { key, index };
        }
    }

    fn hash(key: u64) -> usize {
        fnv1a64(&key.to_le_bytes()) as usize
    }
}

pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv_vectors_and_stable_indices_hold() {
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);

        let mut table = DescriptorTable::new(128).unwrap();
        assert_eq!(table.resolve(9), FALLBACK_INDEX);
        let a = table.insert(9).unwrap();
        assert_eq!(table.insert(9).unwrap(), a);
        for key in 10..64 {
            table.insert(key).unwrap();
        }
        let snapshots: Vec<(u64, u32)> = (9..64)
            .map(|key| (key, table.get(key).unwrap()))
            .collect();
        for (key, index) in snapshots {
            assert_eq!(table.get(key), Some(index));
            assert_ne!(index, FALLBACK_INDEX);
        }
        assert!(table.remove(9));
        assert!(!table.remove(9));
        assert_eq!(table.resolve(9), FALLBACK_INDEX);
        assert_eq!(table.insert(100).unwrap(), a);
    }
}
