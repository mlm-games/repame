//! Texture atlas: CPU-side rectangle packing + upload queue.
//!
//! Packing is shelf-based per square page ([`pack`]); pages spill over
//! automatically and the page index feeds straight into
//! `repame_sprite::SpriteInstance.page`.
//!
//! ```rust
//! use repame_atlas::{Atlas, AtlasDesc};
//!
//! let mut atlas = Atlas::new(AtlasDesc { size: 512, max_pages: 4 });
//! let uv = atlas.alloc_str("hero_idle_0", 48, 32).expect("fits");
//! assert_eq!(uv.page, 0);
//! let writes = atlas.drain_writes();
//! assert_eq!(writes.len(), 1);
//! ```

mod pack;
mod upload;

pub use pack::{Placement, ShelfPage};
pub use upload::{AtlasWrite, PageClear, UploadQueue};

use std::collections::HashMap;

/// Opaque sprite key. Games hash their asset names with [`atlas_id`];
/// explicit ids are fine too as long as they are unique per atlas.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AtlasId(pub u64);

/// Deterministic FNV-1a name hash, stable across runs and platforms so
/// packs (and tests) reproduce byte-identically.
pub fn atlas_id(name: &str) -> AtlasId {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut hash = OFFSET;
    for byte in name.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    AtlasId(hash)
}

/// Normalized texture coords for one sprite: `page` selects the atlas
/// layer, `min`/`max` the sub-rectangle (y-down, matching PNG row order).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UvRect {
    pub page: u32,
    pub min: [f32; 2],
    pub max: [f32; 2],
}

/// Atlas construction parameters.
#[derive(Clone, Copy, Debug)]
pub struct AtlasDesc {
    /// Square page edge in pixels (e.g. 1024, 2048).
    pub size: u32,
    /// Hard cap on page count; allocations beyond it fail cleanly.
    pub max_pages: u32,
}

/// Why an allocation failed. Never panics: oversize and exhaustion are
/// normal load-time conditions the game resolves (bigger atlas, split
/// pack, streaming).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocError {
    /// A live entry already owns `key`; remove it first (or keep it -
    /// allocating the same sprite twice is usually a game bug).
    Duplicate,
    /// Larger than one page in either dimension.
    TooLarge,
    /// Every page is full (and `max_pages` is reached).
    OutOfSpace,
}

/// The atlas: multi-page shelf packer + upload queue facade.
#[derive(Debug)]
pub struct Atlas {
    desc: AtlasDesc,
    pages: Vec<ShelfPage>,
    entries: HashMap<AtlasId, (u32, Placement)>,
    queue: UploadQueue,
}

impl Atlas {
    pub fn new(desc: AtlasDesc) -> Self {
        debug_assert!(desc.size > 0, "atlas pages need a size");
        debug_assert!(desc.max_pages > 0, "atlas needs at least one page");
        Self {
            desc,
            pages: Vec::new(),
            entries: HashMap::new(),
            queue: UploadQueue::default(),
        }
    }

    /// Place a `w`x`h` sprite under `key`. Queues exactly one
    /// [`AtlasWrite`] so the backend knows what to upload where.
    pub fn alloc(&mut self, key: AtlasId, w: u32, h: u32) -> Result<UvRect, AllocError> {
        if self.entries.contains_key(&key) {
            return Err(AllocError::Duplicate);
        }
        if w == 0 || h == 0 || w > self.desc.size || h > self.desc.size {
            return Err(AllocError::TooLarge);
        }
        for (page, shelf) in self.pages.iter_mut().enumerate() {
            if let Some(p) = shelf.alloc(w, h) {
                return Ok(self.insert(key, page as u32, p));
            }
        }
        if self.pages.len() as u32 >= self.desc.max_pages {
            return Err(AllocError::OutOfSpace);
        }
        let mut shelf = ShelfPage::new(self.desc.size);
        let page = self.pages.len() as u32;
        let p = shelf
            .alloc(w, h)
            .expect("fits on a fresh page: bounds checked above");
        self.pages.push(shelf);
        Ok(self.insert(key, page, p))
    }

    /// Convenience wrapper hashing a name with [`atlas_id`].
    pub fn alloc_str(&mut self, name: &str, w: u32, h: u32) -> Result<UvRect, AllocError> {
        self.alloc(atlas_id(name), w, h)
    }

    fn insert(&mut self, key: AtlasId, page: u32, p: Placement) -> UvRect {
        let uv = self.uv_of(page, p);
        self.entries.insert(key, (page, p));
        self.queue.push_write(AtlasWrite {
            key,
            page,
            x: p.x,
            y: p.y,
            w: p.w,
            h: p.h,
            uv,
        });
        uv
    }

    /// Look up a live entry's uv rect.
    pub fn uv_rect(&self, key: AtlasId) -> Option<UvRect> {
        let (page, p) = *self.entries.get(&key)?;
        Some(self.uv_of(page, p))
    }

    fn uv_of(&self, page: u32, p: Placement) -> UvRect {
        let s = self.desc.size as f32;
        UvRect {
            page,
            min: [p.x as f32 / s, p.y as f32 / s],
            max: [(p.x + p.w) as f32 / s, (p.y + p.h) as f32 / s],
        }
    }

    /// Remove an entry, returning its pixels to the page's gap pool.
    /// Returns false when `key` was not live. Removals queue no upload:
    /// stale texels are simply never sampled again.
    pub fn remove(&mut self, key: AtlasId) -> bool {
        let Some((page, p)) = self.entries.remove(&key) else {
            return false;
        };
        if let Some(shelf) = self.pages.get_mut(page as usize) {
            shelf.free(p);
        }
        true
    }

    /// Drop everything; queues one [`PageClear`] per touched page so the
    /// backend can recycle whole textures.
    pub fn clear(&mut self) {
        for page in 0..self.pages.len() as u32 {
            self.queue.push_clear(PageClear { page });
        }
        self.pages.clear();
        self.entries.clear();
    }

    /// Live entry count.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Pages currently backing the pack (<= `max_pages`).
    pub fn page_count(&self) -> u32 {
        self.pages.len() as u32
    }

    /// Approximate fill ratio over touched pages (0..1).
    pub fn utilization(&self) -> f32 {
        if self.pages.is_empty() {
            return 0.0;
        }
        let page_area = self.desc.size as u64 * self.desc.size as u64;
        let used: u64 = self.pages.iter().map(ShelfPage::used_area).sum();
        (used as f32 / (page_area * self.pages.len() as u64) as f32).clamp(0.0, 1.0)
    }

    pub fn drain_writes(&mut self) -> Vec<AtlasWrite> {
        self.queue.drain_writes()
    }

    pub fn drain_clears(&mut self) -> Vec<PageClear> {
        self.queue.drain_clears()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_many_frames_onto_one_page() {
        let mut atlas = Atlas::new(AtlasDesc {
            size: 256,
            max_pages: 2,
        });
        // nt-style strip frames: 48x32 cells.
        for i in 0..32 {
            let uv = atlas.alloc(AtlasId(i), 48, 32).expect("fits");
            assert_eq!(uv.page, 0, "frame {i} spilled early");
        }
        assert_eq!(atlas.len(), 32);
        assert!(atlas.utilization() > 0.3);
    }

    #[test]
    fn spills_to_new_pages_then_fails_cleanly() {
        let mut atlas = Atlas::new(AtlasDesc {
            size: 64,
            max_pages: 2,
        });
        let mut pages = std::collections::HashSet::new();
        let mut i = 0u64;
        loop {
            match atlas.alloc(AtlasId(i), 32, 32) {
                Ok(uv) => {
                    pages.insert(uv.page);
                }
                Err(AllocError::OutOfSpace) => break,
                Err(e) => panic!("unexpected error: {e:?}"),
            }
            i += 1;
            assert!(i < 100, "runaway pack");
        }
        assert_eq!(pages.len(), 2, "both pages used before failing");
        assert_eq!(atlas.page_count(), 2);
    }

    #[test]
    fn duplicate_and_oversize_are_errors_not_panics() {
        let mut atlas = Atlas::new(AtlasDesc {
            size: 64,
            max_pages: 1,
        });
        atlas.alloc_str("hero", 16, 16).unwrap();
        assert_eq!(atlas.alloc_str("hero", 16, 16), Err(AllocError::Duplicate));
        assert_eq!(atlas.alloc_str("boss", 65, 65), Err(AllocError::TooLarge));
        assert_eq!(atlas.alloc_str("zero", 0, 8), Err(AllocError::TooLarge));
    }

    #[test]
    fn uv_rects_are_stable_and_look_up() {
        let mut atlas = Atlas::new(AtlasDesc {
            size: 128,
            max_pages: 1,
        });
        let uv = atlas.alloc_str("a", 64, 64).unwrap();
        assert_eq!(uv.min, [0.0, 0.0]);
        assert_eq!(uv.max, [0.5, 0.5]);
        assert_eq!(atlas.uv_rect(atlas_id("a")), Some(uv));
        assert_eq!(atlas.uv_rect(atlas_id("missing")), None);
    }

    #[test]
    fn remove_frees_for_reuse_and_clear_resets() {
        let mut atlas = Atlas::new(AtlasDesc {
            size: 64,
            max_pages: 1,
        });
        let key = atlas_id("temp");
        let uv = atlas.alloc(key, 32, 32).unwrap();
        assert!(atlas.remove(key));
        assert!(!atlas.remove(key));
        assert_eq!(atlas.uv_rect(key), None);
        let uv2 = atlas.alloc(atlas_id("next"), 32, 32).unwrap();
        assert_eq!((uv2.min, uv2.max), (uv.min, uv.max));
        atlas.clear();
        assert!(atlas.is_empty());
        assert_eq!(atlas.page_count(), 0);
        assert_eq!(atlas.drain_clears().len(), 1);
    }

    #[test]
    fn name_hash_is_deterministic() {
        assert_eq!(atlas_id("sprAllyBullet"), atlas_id("sprAllyBullet"));
        assert_ne!(atlas_id("a"), atlas_id("b"));
    }
}
