//! Shelf rectangle packer for one square page.
//!
//! Classic shelf algorithm: allocations stack left-to-right on shelves;
//! a freed rect returns a gap to its shelf, and gaps are reused
//! first-fit with adjacent-gap coalescing. Shelves never shrink, so
//! workloads with stable working sets (sprite frames) don't fragment.

/// Placement of one allocation inside a page, in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

#[derive(Clone, Debug)]
struct Gap {
    x: u32,
    w: u32,
}

#[derive(Clone, Debug)]
struct Shelf {
    y: u32,
    h: u32,
    cursor: u32,
    gaps: Vec<Gap>,
}

/// One packable page. Not thread-safe; games drive it at load time or
/// behind their own lock.
#[derive(Clone, Debug)]
pub struct ShelfPage {
    size: u32,
    shelves: Vec<Shelf>,
    cursor_y: u32,
}

impl ShelfPage {
    pub fn new(size: u32) -> Self {
        Self {
            size,
            shelves: Vec::new(),
            cursor_y: 0,
        }
    }

    /// Try to place a `w`x`h` rect. Returns its placement on success.
    pub fn alloc(&mut self, w: u32, h: u32) -> Option<Placement> {
        if w == 0 || h == 0 || w > self.size || h > self.size {
            return None;
        }
        // Best-fit open shelf: shortest shelf that fits, to keep squat
        // shelves for squat sprites.
        let mut best: Option<usize> = None;
        for (i, shelf) in self.shelves.iter().enumerate() {
            if shelf.h < h || shelf.cursor + w > self.size {
                continue;
            }
            let fits_gap = shelf.gaps.iter().any(|g| g.w >= w);
            if shelf.cursor + w <= self.size || fits_gap {
                match best {
                    Some(b) if self.shelves[b].h <= shelf.h => {}
                    _ => best = Some(i),
                }
            }
        }
        if let Some(i) = best {
            return self.place_on_shelf(i, w, h);
        }
        // New shelf below the cursor.
        if self.cursor_y + h > self.size {
            return None;
        }
        let y = self.cursor_y;
        self.cursor_y += h;
        self.shelves.push(Shelf {
            y,
            h,
            cursor: 0,
            gaps: Vec::new(),
        });
        self.place_on_shelf(self.shelves.len() - 1, w, h)
    }

    fn place_on_shelf(&mut self, i: usize, w: u32, h: u32) -> Option<Placement> {
        let shelf = &mut self.shelves[i];
        debug_assert!(shelf.h >= h);
        // Prefer reusing a freed gap over growing the cursor.
        for (gi, gap) in shelf.gaps.iter_mut().enumerate() {
            if gap.w >= w {
                let x = gap.x;
                gap.x += w;
                gap.w -= w;
                if gap.w == 0 {
                    shelf.gaps.remove(gi);
                }
                return Some(Placement {
                    x,
                    y: shelf.y,
                    w,
                    h,
                });
            }
        }
        if shelf.cursor + w > self.size {
            return None;
        }
        let x = shelf.cursor;
        shelf.cursor += w;
        Some(Placement {
            x,
            y: shelf.y,
            w,
            h,
        })
    }

    /// Return a placement to its shelf's gap list, coalescing neighbours.
    pub fn free(&mut self, p: Placement) {
        let Some(shelf) = self.shelves.iter_mut().find(|s| s.y == p.y && s.h >= p.h) else {
            return;
        };
        shelf.gaps.push(Gap { x: p.x, w: p.w });
        shelf.gaps.sort_by_key(|g| g.x);
        // Coalesce adjacent gaps.
        let mut merged: Vec<Gap> = Vec::with_capacity(shelf.gaps.len());
        for gap in shelf.gaps.drain(..) {
            if let Some(last) = merged.last_mut()
                && last.x + last.w == gap.x
            {
                last.w += gap.w;
            } else {
                merged.push(gap);
            }
        }
        shelf.gaps = merged;
    }

    /// Allocated pixel area (approximate: freed gaps still count until
    /// their shelf is reused, which is exactly the fragmentation cost).
    pub fn used_area(&self) -> u64 {
        let mut area = 0u64;
        for shelf in &self.shelves {
            area += shelf.cursor as u64 * shelf.h as u64;
        }
        area
    }

    pub fn page_size(&self) -> u32 {
        self.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_rows_without_overlap() {
        let mut page = ShelfPage::new(64);
        let mut rects = Vec::new();
        for _ in 0..8 {
            let p = page.alloc(16, 16).expect("fits");
            rects.push(p);
        }
        for (i, a) in rects.iter().enumerate() {
            for b in &rects[i + 1..] {
                let overlap =
                    a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
                assert!(!overlap, "overlap: {a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn rejects_oversize_and_zero() {
        let mut page = ShelfPage::new(64);
        assert!(page.alloc(65, 16).is_none());
        assert!(page.alloc(16, 65).is_none());
        assert!(page.alloc(0, 16).is_none());
        assert!(page.alloc(16, 0).is_none());
        assert!(page.alloc(64, 64).is_some());
    }

    #[test]
    fn freed_gaps_are_reused() {
        let mut page = ShelfPage::new(64);
        let a = page.alloc(16, 16).unwrap();
        let _b = page.alloc(16, 16).unwrap();
        page.free(a);
        let c = page.alloc(16, 16).expect("gap reused");
        assert_eq!((c.x, c.y), (a.x, a.y));
    }

    #[test]
    fn adjacent_gaps_coalesce() {
        let mut page = ShelfPage::new(64);
        let a = page.alloc(16, 16).unwrap();
        let b = page.alloc(16, 16).unwrap();
        page.free(a);
        page.free(b);
        // A 32-wide rect now fits where two 16-wide gaps merged.
        let c = page.alloc(32, 16).expect("merged gap reused");
        assert_eq!((c.x, c.y), (0, 0));
    }
}
