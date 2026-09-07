//! Upload queue: the atlas owns geometry, never pixels.
//!
//! Games decode their own images (PNG, generated, etc) and blit them into
//! backend textures. The atlas tells the backend *where* each sprite
//! landed via [`AtlasWrite`]; removals that free a whole page surface as
//! [`PageClear`]. Backends drain these once per frame (or once per load)
//! and translate them into their own upload calls
//! (`RenderContext::set_image_rgba8`, native texture updates, etc).

use super::{AtlasId, UvRect};

/// One pending pixel upload: the game blits the sprite's RGBA8 pixels
/// (identified by `key`) into `page` at (`x`, `y`, `w`, `h`).
#[derive(Clone, Debug, PartialEq)]
pub struct AtlasWrite {
    pub key: AtlasId,
    pub page: u32,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    pub uv: UvRect,
}

/// A page whose contents were bulk-invalidated (after `Atlas::clear`);
/// the backend should drop or re-upload it wholesale.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PageClear {
    pub page: u32,
}

/// FIFO queue drained by the backend. Order is allocation order, which
/// keeps load-time uploads sequential per page.
#[derive(Clone, Debug, Default)]
pub struct UploadQueue {
    writes: Vec<AtlasWrite>,
    clears: Vec<PageClear>,
}

impl UploadQueue {
    pub fn push_write(&mut self, write: AtlasWrite) {
        self.writes.push(write);
    }

    pub fn push_clear(&mut self, clear: PageClear) {
        self.clears.push(clear);
    }

    pub fn drain_writes(&mut self) -> Vec<AtlasWrite> {
        std::mem::take(&mut self.writes)
    }

    pub fn drain_clears(&mut self) -> Vec<PageClear> {
        std::mem::take(&mut self.clears)
    }

    pub fn is_empty(&self) -> bool {
        self.writes.is_empty() && self.clears.is_empty()
    }

    pub fn len(&self) -> usize {
        self.writes.len() + self.clears.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drain_empties_in_order() {
        let mut q = UploadQueue::default();
        let uv = UvRect {
            page: 0,
            min: [0.0, 0.0],
            max: [0.5, 0.5],
        };
        for i in 0..3 {
            q.push_write(AtlasWrite {
                key: AtlasId(i),
                page: 0,
                x: i as u32 * 8,
                y: 0,
                w: 8,
                h: 8,
                uv,
            });
        }
        q.push_clear(PageClear { page: 1 });
        let writes = q.drain_writes();
        assert_eq!(writes.len(), 3);
        assert_eq!(writes[0].x, 0);
        assert_eq!(writes[2].x, 16);
        assert!(q.drain_writes().is_empty());
        assert_eq!(q.drain_clears().len(), 1);
        assert!(q.is_empty());
    }
}
