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
    /// Two definition keys stem to the same name (e.g.
    /// `images/a/foo.png` and `images/b/foo.png` both -> `foo`).
    DuplicateStem {
        stem: String,
        first: String,
        second: String,
    },
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
            Self::DuplicateStem {
                stem,
                first,
                second,
            } => write!(
                f,
                "anim stem collision `{stem}` from `{first}` and `{second}`"
            ),
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
        let mut stems: HashMap<String, String> = HashMap::with_capacity(raw.len());
        for (name, def) in raw {
            let s = stem(&name).to_string();
            if let Some(first) = stems.get(&s) {
                return Err(CatalogError::DuplicateStem {
                    stem: s,
                    first: first.clone(),
                    second: name,
                });
            }
            stems.insert(s.clone(), name);
            defs.insert(s, def);
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

/// What happens when playback reaches the end of the strip.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LoopMode {
    /// Wrap to the start and keep playing. The default: walk cycles, idle
    /// loops, spinning coins.
    #[default]
    Loop,
    /// Hold the last cell and stop. Reports finished once through the
    /// `ended` flag of [`advance`](AnimPlayer::advance) and stays there
    /// until [`play`](AnimPlayer::play) restarts it: one-shot attacks,
    /// death animations, UI pop-ins.
    Once,
    /// Bounce between the ends (cell `0` to cell `N-1` and back) and keep
    /// playing: patrol pacing, bobbing pickups, swinging lanterns.
    PingPong,
}

/// Frame player for one animation strip.
///
/// Owns playback timing over a strip shape (`frames` cells at `fps`,
/// copied from an [`AnimDef`] at construction). The game drives it with
/// variable frame time and reads [`frame`](AnimPlayer::frame) to look up
/// the atlas cell in the catalog:
///
/// ```ignore
/// player.advance(dt);
/// let uv = catalog.uv("hero", player.frame() as i32).unwrap();
/// ```
///
/// Playback state at a glance:
///
/// - [`play`](AnimPlayer::play) starts or resumes; [`pause`](AnimPlayer::pause)
///   freezes in place; [`stop`](AnimPlayer::stop) resets to cell `0`.
/// - `speed_scale` multiplies the rate: `1` is normal speed, `0.5` half
///   speed, `2` double speed. A negative value plays in reverse and `0`
///   freezes the frame while staying "playing" (see
///   [`is_playing`](AnimPlayer::is_playing)).
/// - [`frame`](AnimPlayer::frame) is the current cell; [`frame_progress`](AnimPlayer::frame_progress)
///   is `0..1` toward the next cell (`1..0` in reverse).
///
/// **Note:** a player snapshots `frames`/`fps` at construction. Streaming a
/// replacement definition for the same name needs a fresh player.
#[derive(Clone, Debug)]
pub struct AnimPlayer {
    frames: u32,
    fps: f32,
    pos: f32,
    playing: bool,
    speed_scale: f32,
    loop_mode: LoopMode,
    forward: bool,
    finished: bool,
}

impl AnimPlayer {
    /// New player over `def`'s strip with the given loop behavior.
    ///
    /// Starts paused at cell `0` with `speed_scale` of `1.0`. Call
    /// [`play`](AnimPlayer::play) to start the clock.
    pub fn new(def: &AnimDef, loop_mode: LoopMode) -> Self {
        Self {
            frames: def.frames,
            fps: def.fps,
            pos: 0.0,
            playing: false,
            speed_scale: 1.0,
            loop_mode,
            forward: true,
            finished: false,
        }
    }

    /// Start (or resume) playing from the current position.
    ///
    /// Resuming a paused animation continues from the kept cell and
    /// progress. Replaying a finished one-shot ([`is_finished`](AnimPlayer::is_finished))
    /// restarts it from cell `0` first.
    pub fn play(&mut self) {
        if self.finished {
            self.pos = 0.0;
            self.forward = true;
            self.finished = false;
        }
        self.playing = true;
    }

    /// Play in reverse: flips the direction and resumes from the current
    /// cell.
    ///
    /// Jumps to the last cell first when stopped or after a finished
    /// one-shot, so a fresh player starts at the end of the strip. This is
    /// shorthand for a negative [`speed_scale`](AnimPlayer::set_speed_scale)
    /// starting at the far end.
    pub fn play_backwards(&mut self) {
        if (!self.playing || self.finished) && self.frames > 0 {
            self.pos = (self.frames.saturating_sub(1)) as f32;
        }
        self.forward = false;
        self.finished = false;
        self.playing = true;
    }

    /// Pause, keeping the current cell and progress.
    ///
    /// [`play`](AnimPlayer::play) (or [`play_backwards`](AnimPlayer::play_backwards))
    /// resumes from the exact same spot. See also [`stop`](AnimPlayer::stop),
    /// which resets instead of holding.
    pub fn pause(&mut self) {
        self.playing = false;
    }

    /// Stop and reset to cell `0`.
    ///
    /// Clears the finished flag, restores forward direction, and zeroes the
    /// progress. The `speed_scale` is kept. See also
    /// [`pause`](AnimPlayer::pause), which holds the position instead.
    pub fn stop(&mut self) {
        self.playing = false;
        self.pos = 0.0;
        self.forward = true;
        self.finished = false;
    }

    /// Speed multiplier for [`advance`](AnimPlayer::advance).
    ///
    /// `1.0` is normal speed, `0.5` half speed, `2.0` double speed. A
    /// negative value plays in reverse; `0.0` freezes the frame while the
    /// player stays "playing". Non-finite values (`NaN`, infinity) are
    /// treated as `0.0` so a bad calculation pauses instead of corrupting
    /// the clock.
    pub fn set_speed_scale(&mut self, s: f32) {
        self.speed_scale = if s.is_finite() { s } else { 0.0 };
    }

    /// Current speed multiplier (see [`set_speed_scale`](AnimPlayer::set_speed_scale)).
    /// Defaults to `1.0`.
    pub fn speed_scale(&self) -> f32 {
        self.speed_scale
    }

    /// Whether the clock is running.
    ///
    /// Returns `true` after [`play`](AnimPlayer::play) until
    /// [`pause`](AnimPlayer::pause), [`stop`](AnimPlayer::stop), or a
    /// one-shot reaching its end, even at `speed_scale` of `0`, where time
    /// is frozen but the intent to play remains.
    pub fn is_playing(&self) -> bool {
        self.playing
    }

    /// Whether a [`Once`](LoopMode::Once) animation has reached its end.
    ///
    /// Set when the one-shot finishes and cleared by [`play`](AnimPlayer::play)
    /// (which restarts the strip) or [`stop`](AnimPlayer::stop). Looping
    /// animations never set this; watch the `ended` flag of
    /// [`advance`](AnimPlayer::advance) for their wrap moments instead.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Current cell index, always a valid strip cell.
    ///
    /// Returns `0` for an empty strip. Otherwise floors the internal clock
    /// and clamps to the last cell, so it never points past the strip even
    /// on the exact end boundary.
    pub fn frame(&self) -> u32 {
        if self.frames == 0 {
            return 0;
        }
        (self.pos.floor() as u32).min(self.frames - 1)
    }

    /// Progress toward the next cell, from `0.0` to `1.0`.
    ///
    /// Playing forward, the value rises from `0.0` (just entered the cell)
    /// to `1.0` (about to leave it). Playing in reverse it runs `1.0`
    /// down to `0.0`. Useful for blending or for syncing effects to a
    /// mid-cell moment. To jump cells while keeping a specific progress,
    /// use [`set_frame_and_progress`](AnimPlayer::set_frame_and_progress).
    pub fn frame_progress(&self) -> f32 {
        let f = (self.pos - self.pos.floor()).clamp(0.0, 1.0);
        if self.backward() { 1.0 - f } else { f }
    }

    fn backward(&self) -> bool {
        self.speed_scale < 0.0 || !self.forward
    }

    /// Set the cell and progress together, preserving playback state.
    ///
    /// Unlike [`stop`](AnimPlayer::stop), nothing else resets: whether the
    /// player is playing, its direction, and its finished flag are kept
    /// (except that the finished flag clears, since the position is fresh).
    /// Out-of-range inputs clamp (`frame` to the last cell, `progress` to
    /// `0..1`), and an empty strip ignores the call.
    ///
    /// Useful for handing the exact cycle position to a fresh player, e.g.
    /// when swapping to a same-length variant skin mid-motion:
    ///
    /// ```ignore
    /// let f = player.frame();
    /// let p = player.frame_progress();
    /// let mut other = AnimPlayer::new(catalog.def("hero_alt").unwrap(), LoopMode::Loop);
    /// other.set_frame_and_progress(f, p);
    /// other.play();
    /// ```
    pub fn set_frame_and_progress(&mut self, frame: u32, progress: f32) {
        if self.frames == 0 {
            return;
        }
        let f = frame.min(self.frames - 1) as f32;
        let p = if self.backward() {
            1.0 - progress.clamp(0.0, 1.0)
        } else {
            progress.clamp(0.0, 1.0)
        };
        self.pos = (f + p).clamp(0.0, (self.frames - 1) as f32);
        self.finished = false;
    }

    /// Advance the clock by `dt` seconds.
    ///
    /// Returns `(frame_changed, ended)`:
    ///
    /// - `frame_changed` is true when the advance crossed into a different
    ///   cell. Use it to refresh atlas lookups and to fire per-step effects
    ///   (footstep sounds, particles) exactly once per cell.
    /// - `ended` is true when a [`Once`](LoopMode::Once) animation finishes
    ///   on this advance, or when a [`Loop`](LoopMode::Loop) /
    ///   [`PingPong`](LoopMode::PingPong) animation wraps or bounces. A
    ///   looping animation never sets [`is_finished`](AnimPlayer::is_finished),
    ///   so this flag is the way to count its laps.
    ///
    /// The call is a no-op (returning `(false, false)`) while paused, at a
    /// `speed_scale` of `0`, on strips with fewer than two cells, at
    /// non-positive `fps`, or for non-positive/non-finite `dt`. Oversized
    /// `dt` values (hitch frames) cross as many cells as they cover: a loop
    /// wraps once per advance at most for the flag, while the cell always
    /// lands exactly.
    pub fn advance(&mut self, dt: f32) -> (bool, bool) {
        if !self.playing || self.frames <= 1 || self.fps <= 0.0 {
            return (false, false);
        }
        if !dt.is_finite() || dt <= 0.0 {
            return (false, false);
        }
        let before = self.frame();
        let step = dt * self.fps * self.speed_scale;
        if step == 0.0 {
            return (false, false);
        }
        let estep = if self.backward() {
            -step.abs()
        } else {
            step.abs()
        };
        let last = (self.frames - 1) as f32;
        let pos_before = self.pos;
        match self.loop_mode {
            LoopMode::Loop => {
                self.pos = (self.pos + estep).rem_euclid(self.frames as f32);
                let wrapped = if estep > 0.0 {
                    pos_before + estep >= self.frames as f32
                } else {
                    pos_before + estep < 0.0
                };
                (self.frame() != before, wrapped)
            }
            LoopMode::Once => {
                self.pos += estep;
                if self.pos >= self.frames as f32 || self.pos < 0.0 {
                    self.pos = if estep > 0.0 { last } else { 0.0 };
                    self.playing = false;
                    self.finished = true;
                    (self.frame() != before, true)
                } else {
                    (self.frame() != before, false)
                }
            }
            LoopMode::PingPong => {
                // Triangle wave with period 2*(N-1): 0..N-1..0.
                let period = 2.0 * last;
                let tri0 = if self.forward {
                    self.pos
                } else {
                    period - self.pos
                }
                .rem_euclid(period);
                let raw = tri0 + step;
                let ended = raw >= period || raw < 0.0;
                let tri = raw.rem_euclid(period);
                self.forward = tri <= last;
                self.pos = if self.forward { tri } else { period - tri };
                self.pos = self.pos.clamp(0.0, last);
                (self.frame() != before, ended)
            }
        }
    }
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
        let dup = r#"{"images/a/foo.png": {"frames": 1, "w": 8, "h": 8, "fps": 1.0, "xorigin": 0.0, "yorigin": 0.0}, "images/b/foo.png": {"frames": 1, "w": 8, "h": 8, "fps": 1.0, "xorigin": 0.0, "yorigin": 0.0}}"#;
        assert!(matches!(
            AnimCatalog::from_json(
                dup,
                AtlasDesc {
                    size: 64,
                    max_pages: 1
                }
            ),
            Err(CatalogError::DuplicateStem { .. })
        ));
    }

    #[test]
    fn player_loops_and_reports_progress() {
        let def = AnimDef {
            frames: 4,
            w: 16,
            h: 16,
            fps: 10.0,
            xorigin: 8.0,
            yorigin: 8.0,
        };
        let mut p = AnimPlayer::new(&def, LoopMode::Loop);
        assert!(!p.is_playing());
        p.play();
        assert_eq!(p.frame(), 0);
        assert_eq!(p.frame_progress(), 0.0);
        // Half a cell at 10 fps.
        let (changed, ended) = p.advance(0.05);
        assert!(!changed && !ended);
        assert!((p.frame_progress() - 0.5).abs() < 1e-6);
        // Full wrap: 4 cells at 10 fps = 0.4 s per loop.
        let mut looped = false;
        for _ in 0..8 {
            let (_, e) = p.advance(0.05);
            looped |= e;
        }
        assert!(looped);
        assert!(p.is_playing() && !p.is_finished());
    }

    #[test]
    fn player_once_holds_last_frame() {
        let def = AnimDef {
            frames: 3,
            w: 8,
            h: 8,
            fps: 10.0,
            xorigin: 0.0,
            yorigin: 0.0,
        };
        let mut p = AnimPlayer::new(&def, LoopMode::Once);
        p.play();
        let mut ended = false;
        for _ in 0..10 {
            let (_, e) = p.advance(0.05);
            ended |= e;
        }
        assert!(ended);
        assert_eq!(p.frame(), 2);
        assert!(p.is_finished() && !p.is_playing());
        // Replay restarts from cell 0.
        p.play();
        assert_eq!(p.frame(), 0);
        assert!(!p.is_finished());
    }

    #[test]
    fn player_pingpong_bounces() {
        let def = AnimDef {
            frames: 3,
            w: 8,
            h: 8,
            fps: 10.0,
            xorigin: 0.0,
            yorigin: 0.0,
        };
        let mut p = AnimPlayer::new(&def, LoopMode::PingPong);
        p.play();
        // 0.1 s per cell: 0,1,2,1,0,1,...
        let mut seq = vec![p.frame()];
        for _ in 0..5 {
            p.advance(0.1);
            seq.push(p.frame());
        }
        assert_eq!(seq, vec![0, 1, 2, 1, 0, 1]);
    }

    #[test]
    fn player_speed_scale_and_backwards() {
        let def = AnimDef {
            frames: 4,
            w: 8,
            h: 8,
            fps: 10.0,
            xorigin: 0.0,
            yorigin: 0.0,
        };
        let mut p = AnimPlayer::new(&def, LoopMode::Loop);
        p.set_speed_scale(2.0);
        p.play();
        p.advance(0.05);
        assert_eq!(p.frame(), 1, "double speed advances a full cell");
        p.set_speed_scale(-1.0);
        p.advance(0.05);
        assert_eq!(p.frame(), 0, "reverse steps back");
        // Reverse progress runs 1 -> 0.
        p.set_frame_and_progress(1, 0.5);
        assert!((p.frame_progress() - 0.5).abs() < 1e-6);
        p.play_backwards();
        assert_eq!(p.frame(), 1, "backwards resumes in place while playing");
        p.advance(0.1);
        assert_eq!(p.frame(), 0, "reverse steps back");
        // Fresh backwards play starts at the last cell.
        let mut q = AnimPlayer::new(&def, LoopMode::Loop);
        q.play_backwards();
        assert_eq!(q.frame(), 3);
        assert!(q.is_playing());
        p.stop();
        assert_eq!(p.frame(), 0);
        assert!(!p.is_playing());
    }

    #[test]
    fn loop_play_backwards_reverses() {
        let def = AnimDef {
            frames: 4,
            w: 8,
            h: 8,
            fps: 10.0,
            xorigin: 0.0,
            yorigin: 0.0,
        };
        let mut p = AnimPlayer::new(&def, LoopMode::Loop);
        p.play_backwards();
        assert_eq!(p.frame(), 3);
        p.advance(0.05);
        assert_eq!(p.frame(), 2, "reverse steps back, got {}", p.frame());
        let mut q = AnimPlayer::new(&def, LoopMode::Once);
        q.play();
        q.set_frame_and_progress(2, 0.0);
        q.play_backwards();
        let (changed, ended) = q.advance(0.05);
        assert!(changed && !ended);
        assert_eq!(q.frame(), 1);
    }

    #[test]
    fn set_frame_and_progress_round_trips_in_reverse() {
        let def = AnimDef {
            frames: 4,
            w: 8,
            h: 8,
            fps: 10.0,
            xorigin: 0.0,
            yorigin: 0.0,
        };
        let mut p = AnimPlayer::new(&def, LoopMode::Loop);
        p.set_speed_scale(-1.0);
        p.play();
        p.set_frame_and_progress(1, 0.8);
        assert!((p.frame_progress() - 0.8).abs() < 1e-6);
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
