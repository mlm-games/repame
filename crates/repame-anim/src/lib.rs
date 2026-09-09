//! Animation catalogs: nt-style `{name: {frames, w, h, fps, xorigin,
//! yorigin}}` JSON parsed, packed frame-by-frame into a [`repame_atlas`]
//! [`Atlas`], and served back as uv rects.

use std::collections::HashMap;

use repame_atlas::{AllocError, AtlasId, atlas_id};
use serde::Deserialize;

/// Atlas types this API surfaces (re-exported so catalog users need no
/// direct atlas dependency).
pub use repame_atlas::{Atlas, AtlasDesc, UvRect};

/// One animation definition: a horizontal strip of `frames` cells.
#[derive(Clone, Debug, Deserialize)]
pub struct AnimDef {
    /// Frame count in the strip.
    pub frames: u32,
    /// Cell size in pixels.
    pub w: u32,
    pub h: u32,
    /// Playback rate; game-side timers consume this.
    pub fps: f32,
    /// Origin in pixels (bevy `Anchor` source).
    pub xorigin: f32,
    pub yorigin: f32,
}

impl AnimDef {
    /// Normalized anchor (`[0.5, 0.5]` = centered), straight into
    /// `SpriteInstance.anchor`. Zero-size cells anchor top-left.
    pub fn anchor(&self) -> [f32; 2] {
        if self.w == 0 || self.h == 0 {
            return [0.0, 0.0];
        }
        [
            (self.xorigin / self.w as f32).clamp(0.0, 1.0),
            (self.yorigin / self.h as f32).clamp(0.0, 1.0),
        ]
    }

    /// Frame index wrapped to the strip (negative-safe for ping-pong).
    pub fn wrap_frame(&self, frame: i32) -> u32 {
        if self.frames == 0 {
            return 0;
        }
        frame.rem_euclid(self.frames as i32) as u32
    }
}

/// Why catalog construction failed.
#[derive(Debug)]
pub enum CatalogError {
    /// Malformed JSON.
    Json(serde_json::Error),
    /// Atlas exhausted mid-pack; `placed` frames landed before it.
    AtlasFull { placed: usize },
    /// A single cell is larger than one page.
    CellTooLarge { name: String },
}

impl std::fmt::Display for CatalogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json(e) => write!(f, "anim catalog json: {e}"),
            Self::AtlasFull { placed } => {
                write!(f, "atlas full after {placed} frames")
            }
            Self::CellTooLarge { name } => {
                write!(f, "anim cell larger than one page: {name}")
            }
        }
    }
}

impl std::error::Error for CatalogError {}

/// Parsed + packed catalog. Owns the [`Atlas`]; games drain its upload
/// queue into their backend after construction (and after streaming
/// more anims in later).
#[derive(bevy_ecs::prelude::Resource)]
pub struct AnimCatalog {
    defs: HashMap<String, AnimDef>,
    atlas: Atlas,
}

impl Default for AnimCatalog {
    fn default() -> Self {
        Self::empty()
    }
}

impl AnimCatalog {
    /// Empty catalog (no defs, empty atlas): systems resolve nothing and
    /// skip via `def()` returning `None`. Startup default so schedules
    /// can boot before art loads; the game replaces it on load.
    pub fn empty() -> Self {
        Self {
            defs: HashMap::new(),
            atlas: Atlas::new(AtlasDesc {
                size: 64,
                max_pages: 1,
            }),
        }
    }
    /// Parse JSON and pack every frame. `serde_json` maps sort keys, so
    /// packs are deterministic regardless of file order. Definition keys
    /// are stemmed (`stem`), so bare names and full paths address the
    /// same entry.
    pub fn from_json(json: &str, desc: AtlasDesc) -> Result<Self, CatalogError> {
        let raw: HashMap<String, AnimDef> =
            serde_json::from_str(json).map_err(CatalogError::Json)?;
        let mut defs = HashMap::with_capacity(raw.len());
        for (name, def) in raw {
            defs.insert(stem(&name).to_string(), def);
        }
        let mut atlas = Atlas::new(desc);
        let mut placed = 0usize;
        let mut names: Vec<&String> = defs.keys().collect();
        names.sort();
        for name in names {
            let def = &defs[name];
            for frame in 0..def.frames {
                let key = frame_key(name, frame);
                match atlas.alloc(key, def.w, def.h) {
                    Ok(_) => placed += 1,
                    Err(AllocError::TooLarge) => {
                        return Err(CatalogError::CellTooLarge { name: name.clone() });
                    }
                    Err(_) => return Err(CatalogError::AtlasFull { placed }),
                }
            }
        }
        Ok(Self { defs, atlas })
    }

    /// Animation definition by strip name or full path.
    pub fn def(&self, name: &str) -> Option<&AnimDef> {
        self.defs.get(name).or_else(|| self.defs.get(stem(name)))
    }

    /// Strip names, sorted.
    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.defs.keys().map(String::as_str).collect();
        names.sort();
        names
    }

    /// Atlas uv rect for one frame (wraps like nt's `image_index`).
    pub fn uv(&self, name: &str, frame: i32) -> Option<UvRect> {
        let def = self.def(name)?;
        self.atlas
            .uv_rect(frame_key(stem(name), def.wrap_frame(frame)))
    }

    /// Source rectangle inside the strip PNG (`{name}.png`, horizontal
    /// frames): `[x, y, w, h]`. The game blits this into the atlas
    /// placement from the matching drain entry.
    pub fn src_rect(&self, name: &str, frame: i32) -> Option<[u32; 4]> {
        let def = self.def(name)?;
        let f = def.wrap_frame(frame);
        Some([f * def.w, 0, def.w, def.h])
    }

    /// Total packed frames.
    pub fn frame_count(&self) -> usize {
        self.atlas.len()
    }

    pub fn atlas(&self) -> &Atlas {
        &self.atlas
    }

    pub fn atlas_mut(&mut self) -> &mut Atlas {
        &mut self.atlas
    }
}

/// Strip a lookup key to its anim stem: `"images/sprRad.png"` ->
/// `"sprRad"`, `"sprRad"` -> `"sprRad"`. The bevy build keyed its
/// catalog by full paths while `anims.json` uses bare names; stemming
/// makes both spellings resolve to one entry.
pub fn stem(path: &str) -> &str {
    let base = path.rsplit('/').next().unwrap_or(path);
    base.strip_suffix(".png").unwrap_or(base)
}

/// Deterministic per-frame key: `"{name}#{frame}"` hashed.
pub fn frame_key(name: &str, frame: u32) -> AtlasId {
    atlas_id(&format!("{name}#{frame}"))
}

/// Deterministic per-frame key from a path *or* name.
pub fn frame_key_for(path: &str, frame: u32) -> AtlasId {
    frame_key(stem(path), frame)
}

#[cfg(test)]
mod tests {
    use super::*;

    const JSON: &str = r#"{
        "hero": {"frames": 4, "w": 16, "h": 16, "fps": 12.0, "xorigin": 8.0, "yorigin": 16.0},
        "bullet": {"frames": 2, "w": 8, "h": 8, "fps": 8.0, "xorigin": 4.0, "yorigin": 4.0}
    }"#;

    fn catalog() -> AnimCatalog {
        AnimCatalog::from_json(
            JSON,
            AtlasDesc {
                size: 64,
                max_pages: 1,
            },
        )
        .expect("fits")
    }

    #[test]
    fn packs_all_frames_with_stable_uvs() {
        let cat = catalog();
        assert_eq!(cat.frame_count(), 6);
        assert_eq!(cat.names(), vec!["bullet", "hero"]);
        let uv = cat.uv("hero", 0).unwrap();
        assert_eq!(cat.uv("hero", 0), Some(uv));
        // Wrap-around like image_index.
        assert_eq!(cat.uv("hero", 4), cat.uv("hero", 0));
        assert_eq!(cat.uv("hero", -1), cat.uv("hero", 3));
        assert_eq!(cat.uv("missing", 0), None);
    }

    #[test]
    fn full_paths_and_names_resolve_identically() {
        let cat = catalog();
        // nt addresses strips by full path; packs key by stem.
        assert_eq!(
            cat.def("images/hero.png").map(|d| d.frames),
            cat.def("hero").map(|d| d.frames)
        );
        assert_eq!(cat.uv("images/hero.png", 1), cat.uv("hero", 1));
        assert_eq!(cat.src_rect("images/bullet.png", 1), Some([8, 0, 8, 8]));
        assert_eq!(
            frame_key_for("images/hero.png", 2),
            frame_key_for("hero", 2)
        );
    }

    #[test]
    fn anchor_and_src_rect_match_bevy_conventions() {
        let cat = catalog();
        // Origin bottom-center (8,16 of 16x16) -> anchor [0.5, 1.0].
        assert_eq!(cat.def("hero").unwrap().anchor(), [0.5, 1.0]);
        assert_eq!(cat.src_rect("hero", 2), Some([32, 0, 16, 16]));
        assert_eq!(cat.src_rect("bullet", 1), Some([8, 0, 8, 8]));
    }

    #[test]
    fn errors_are_typed_not_panics() {
        assert!(matches!(
            AnimCatalog::from_json(
                "not json",
                AtlasDesc {
                    size: 64,
                    max_pages: 1
                }
            ),
            Err(CatalogError::Json(_))
        ));
        assert!(matches!(
            AnimCatalog::from_json(
                JSON,
                AtlasDesc {
                    size: 8,
                    max_pages: 1
                }
            ),
            Err(CatalogError::AtlasFull { .. })
        ));
        let big = r#"{"huge": {"frames": 1, "w": 512, "h": 512, "fps": 1.0, "xorigin": 0.0, "yorigin": 0.0}}"#;
        assert!(matches!(
            AnimCatalog::from_json(
                big,
                AtlasDesc {
                    size: 64,
                    max_pages: 4
                }
            ),
            Err(CatalogError::CellTooLarge { .. })
        ));
    }

    /// Full nt catalog pack: proves the real content budget. Reads
    /// `$NT_ASSETS/images/anims.json` (else the nt checkout next to
    /// Repos); skips gracefully when absent.
    #[test]
    fn packs_full_nt_catalog() {
        let path = std::env::var("NT_ASSETS").map_or_else(
            |_| {
                let home = std::env::var("HOME").unwrap_or_default();
                format!("{home}/Documents/nt-recreated-bevy/assets/images/anims.json")
            },
            |dir| format!("{dir}/images/anims.json"),
        );
        let Ok(json) = std::fs::read_to_string(&path) else {
            eprintln!("SKIP nt catalog pack (no assets at {path})");
            return;
        };
        let cat = AnimCatalog::from_json(
            &json,
            AtlasDesc {
                size: 2048,
                max_pages: 16,
            },
        )
        .expect("nt catalog fits in 16 pages @2048");
        assert_eq!(cat.names().len(), 2066);
        assert_eq!(cat.frame_count(), 11796);
        eprintln!(
            "nt catalog: {} frames on {} pages @2048 ({:.0}% fill)",
            cat.frame_count(),
            cat.atlas().page_count(),
            cat.atlas().utilization() * 100.0
        );
        // Spot checks: first/last anims resolve with sane uvs.
        for name in ["spr360Big", "sprVlambeer"] {
            let uv = cat.uv(name, 0).expect("resolves");
            assert!(uv.max[0] <= 1.0 && uv.max[1] <= 1.0);
            assert!(uv.max[0] > uv.min[0] && uv.max[1] > uv.min[1]);
        }
    }
}
