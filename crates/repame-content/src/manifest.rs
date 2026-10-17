//! The generated content manifest and the one content-path rule.
//!
//! A build step scans a source directory once and writes the asset table the
//! engine loads ([`ContentManifest`], [`ASSET_MANIFEST_FILE`]) plus the packed
//! archive ([`ASSET_PACK_FILE`]) beside it. The manifest is **data, not
//! generated Rust**: it needs no build-script codegen step, no `OUT_DIR`
//! plumbing and no proc macro, it loads identically on native and wasm, it
//! diffs in git, and a typed loader rejects a hand-edited manifest with a
//! typed error instead of recompiling a game to discover a mistyped path.
//! Generated Rust would need a codegen run inside `build.rs` for every game,
//! would recompile a 3,000-line table on every art change, and would still
//! have to be loaded at runtime to list or hash the shipped assets.
//!
//! Every asset is addressed by an [`AssetPath`]: a `/`-separated path relative
//! to the content root. That one string is the whole path rule — loose files
//! resolve it as `<content root>/<path>`, a packed archive resolves it as the
//! entry of the same name — so the same key names the same asset in
//! development and in a packaged build, and neither mode ever sees an absolute
//! source path. The records carry the dimensions, kind and content hash a game
//! used to hand-maintain, and the [`AtlasPlan`] lays the sprite pages out once,
//! at build time, instead of packing at every boot.

use std::fmt;
use std::path::{Component, Path, PathBuf};

use repame_atlas::{AllocError, Atlas, AtlasDesc, atlas_id};
use serde::{Deserialize, Deserializer, Serialize, de};

use crate::error::ContentError;
use crate::import::AssetKind;

/// Name of the manifest under the content root, and its entry name inside the
/// packed archive: the manifest is a content path like any other.
pub const ASSET_MANIFEST_FILE: &str = "assets.ron";
/// Name of the packed archive under the content root.
pub const ASSET_PACK_FILE: &str = "assets.pack";
/// Manifest schema version.
pub const ASSET_MANIFEST_FORMAT: u32 = 1;
/// Manifest generator version; bump when a scan result changes meaning, so a
/// stale manifest written by an older generator can be told apart.
pub const ASSET_GENERATOR_VERSION: u32 = 1;

/// Segment count ceiling; a path longer than this is refused rather than
/// passing a filesystem limit failure through as a mystery.
const MAX_PATH_SEGMENTS: usize = 64;

/// A content path: the key every asset is addressed by, in development and in
/// a packed archive.
///
/// The rule, in full: a content path is a non-empty `/`-separated path
/// relative to the content root, with no empty, `.` or `..` segment, no
/// leading `/`, no `\` and no `:` (so a Windows drive or separator cannot
/// sneak in and the same string is valid on every platform). Loose files
/// resolve it as `<content root>/<path>` ([`AssetPath::dev_path`]); a packed
/// archive resolves it as the entry of the same name
/// ([`AssetPath::entry_name`]). Nothing resolves an absolute path at runtime,
/// so a build never bakes a source checkout location into a game.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct AssetPath(String);

impl AssetPath {
    /// Validate and normalize a content path.
    pub fn parse(value: &str) -> Result<Self, ContentError> {
        validate_path(value).map_err(|message| ContentError::format(value, message))?;
        Ok(Self(value.to_string()))
    }

    /// A content path from an OS-relative path, joining its components with
    /// `/`. Refuses a non-UTF-8 component and any component above the current
    /// directory, with a typed error naming the path.
    pub fn from_relative(path: impl AsRef<Path>) -> Result<Self, ContentError> {
        let path = path.as_ref();
        let mut value = String::new();
        for component in path.components() {
            match component {
                Component::Normal(segment) => {
                    let segment = segment.to_str().ok_or_else(|| {
                        ContentError::format(path, "content path is not valid UTF-8")
                    })?;
                    if !value.is_empty() {
                        value.push('/');
                    }
                    value.push_str(segment);
                }
                _ => {
                    return Err(ContentError::format(
                        path,
                        format!(
                            "content path {} must be relative and inside the content root",
                            path.display()
                        ),
                    ));
                }
            }
        }
        Self::parse(&value)
    }

    /// A content path from a source file: `absolute` with `root` stripped.
    /// A source outside the content root is a typed error naming the source,
    /// never a silently mangled key.
    pub fn from_absolute(
        root: impl AsRef<Path>,
        absolute: impl AsRef<Path>,
    ) -> Result<Self, ContentError> {
        let root = root.as_ref();
        let absolute = absolute.as_ref();
        let relative = absolute.strip_prefix(root).map_err(|_| {
            ContentError::new(
                absolute,
                format!(
                    "source {} is outside the content root {}",
                    absolute.display(),
                    root.display()
                ),
            )
        })?;
        Self::from_relative(relative)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The archive entry name this path resolves to when packaged. Equal to
    /// [`AssetPath::as_str`] by the path rule, and named separately so call
    /// sites say which mode they mean.
    pub fn entry_name(&self) -> &str {
        &self.0
    }

    /// The loose file this path resolves to under `root` when not packaged.
    pub fn dev_path(&self, root: impl AsRef<Path>) -> PathBuf {
        root.as_ref().join(&self.0)
    }

    pub fn file_name(&self) -> Option<&str> {
        self.0.rsplit('/').next()
    }

    /// Lowercase extension, for classification and pre-flight checks.
    pub fn extension(&self) -> Option<String> {
        Path::new(&self.0)
            .extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase)
    }

    /// True when `prefix` is this path or a directory above it.
    pub fn starts_with_dir(&self, prefix: &str) -> bool {
        self.0 == prefix
            || (self.0.len() > prefix.len()
                && self.0.starts_with(prefix)
                && self.0.as_bytes()[prefix.len()] == b'/')
    }
}

fn validate_path(value: &str) -> Result<(), String> {
    if value.is_empty() {
        return Err("content path is empty".to_string());
    }
    if value.chars().any(char::is_control) {
        return Err(format!("content path `{value}` contains a control character"));
    }
    if value.starts_with('/') || value.ends_with('/') || value.contains('\\') || value.contains(':') {
        return Err(format!(
            "content path `{value}` must be relative and use `/` separators"
        ));
    }
    let segments: Vec<&str> = value.split('/').collect();
    if segments.len() > MAX_PATH_SEGMENTS {
        return Err(format!("content path `{value}` is nested too deeply"));
    }
    if segments
        .iter()
        .any(|segment| segment.is_empty() || *segment == "." || *segment == "..")
    {
        return Err(format!(
            "content path `{value}` has an empty, `.` or `..` segment"
        ));
    }
    Ok(())
}

impl fmt::Display for AssetPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for AssetPath {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl TryFrom<&str> for AssetPath {
    type Error = ContentError;

    fn try_from(value: &str) -> Result<Self, ContentError> {
        Self::parse(value)
    }
}

impl TryFrom<String> for AssetPath {
    type Error = ContentError;

    fn try_from(value: String) -> Result<Self, ContentError> {
        Self::parse(&value)
    }
}

impl std::str::FromStr for AssetPath {
    type Err = ContentError;

    fn from_str(value: &str) -> Result<Self, ContentError> {
        Self::parse(value)
    }
}

impl<'de> Deserialize<'de> for AssetPath {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        Self::parse(&value).map_err(|error| de::Error::custom(format!("{}: {}", value, error.message())))
    }
}

/// Scanned image metadata: pixel dimensions, plus the container details a
/// per-mip reader needs when the source is a DDS or KTX 2 texture.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    /// Present when the source is a texture container the crate can open a
    /// level table for; absent for a plain decodable image file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub texture: Option<TextureMeta>,
}

impl ImageInfo {
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// True for a DDS/KTX 2 container, i.e. a source whose mip levels can be
    /// read one at a time through [`AssetVfs`](crate::AssetVfs).
    pub fn is_texture(&self) -> bool {
        self.texture.is_some()
    }
}

/// Texture container details read from a header, without decoding pixels.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextureMeta {
    /// `"dds"` or `"ktx2"`.
    pub container: String,
    /// [`TextureFormat::as_str`](crate::TextureFormat::as_str), e.g. `"bc1"`.
    pub format: String,
    pub mip_levels: u32,
}

/// Scanned audio stream parameters. Both fields are optional because a codec
/// may describe a stream without declaring them; the manifest records what the
/// probe reported and nothing more.
#[derive(Clone, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct AudioInfo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channels: Option<u32>,
}

/// One scanned source asset.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContentEntry {
    /// The content path, and the archive entry name when packed.
    pub path: AssetPath,
    pub kind: AssetKind,
    /// Source byte length.
    pub size: u64,
    /// FNV-1a 64-bit content hash, the same fingerprint the import cache uses.
    /// It detects a changed source; it is not a collision-resistant digest.
    pub hash: u64,
    /// Present exactly when `kind` is [`AssetKind::Image`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<ImageInfo>,
    /// Present exactly when `kind` is [`AssetKind::Audio`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<AudioInfo>,
}

impl ContentEntry {
    /// True when the source is a DDS/KTX 2 container, so its mip levels can be
    /// read out of the archive without loading the whole texture.
    pub fn is_texture(&self) -> bool {
        self.image.as_ref().is_some_and(ImageInfo::is_texture)
    }
}

/// The generated asset table: format, generator version, one record per source
/// asset in path order, and the optional atlas plan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContentManifest {
    pub format: u32,
    /// Manifest generator version, so a stale table is recognisable: the
    /// loader refuses a table written by a different generator version rather
    /// than guessing at its meaning.
    #[serde(default)]
    pub version: u32,
    /// Free-form label the build set, e.g. the game name.
    #[serde(default)]
    pub name: String,
    /// One record per source asset, strictly sorted by path.
    #[serde(default)]
    pub entries: Vec<ContentEntry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub atlas: Option<AtlasPlan>,
}

impl ContentManifest {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            format: ASSET_MANIFEST_FORMAT,
            version: ASSET_GENERATOR_VERSION,
            name: name.into(),
            entries: Vec::new(),
            atlas: None,
        }
    }

    /// One record per source asset, sorted by path.
    pub fn entries(&self) -> &[ContentEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entry(&self, path: &AssetPath) -> Option<&ContentEntry> {
        self.entries
            .binary_search_by(|entry| entry.path.cmp(path))
            .ok()
            .map(|index| &self.entries[index])
    }

    /// True when the manifest lists this path. Lookups are by the manifest's
    /// own key, so a caller cannot ask a question the manifest cannot answer
    /// without walking the archive.
    pub fn contains(&self, path: &AssetPath) -> bool {
        self.entry(path).is_some()
    }

    pub fn of_kind(&self, kind: AssetKind) -> impl Iterator<Item = &ContentEntry> {
        self.entries.iter().filter(move |entry| entry.kind == kind)
    }

    /// Contiguous containers (DDS/KTX 2) among the image entries.
    pub fn textures(&self) -> impl Iterator<Item = &ContentEntry> {
        self.entries.iter().filter(|entry| entry.is_texture())
    }

    pub fn atlas(&self) -> Option<&AtlasPlan> {
        self.atlas.as_ref()
    }

    pub fn total_size(&self) -> u64 {
        self.entries.iter().map(|entry| entry.size).sum()
    }

    /// Serialize for writing: pretty RON plus a trailing newline, byte-stable
    /// for the same entries.
    pub fn to_ron(&self) -> Result<String, ContentError> {
        let mut text = ron::ser::to_string_pretty(self, Default::default())
            .map_err(|error| ContentError::format(ASSET_MANIFEST_FILE, format!("serialize manifest: {error}")))?;
        text.push('\n');
        Ok(text)
    }

    /// Parse and validate a manifest, naming `source` in every error.
    pub fn from_ron(text: &str, source: &Path) -> Result<Self, ContentError> {
        let manifest: Self = ron::from_str(text)
            .map_err(|error| ContentError::format(source, format!("invalid RON: {error}")))?;
        manifest.validate(source)?;
        Ok(manifest)
    }

    /// Check the invariants a writer and a loader both rely on: one sorted
    /// record per path, kind-specific fields present exactly when they apply,
    /// and a self-consistent atlas plan. Nothing is dropped or repaired here;
    /// an inconsistent manifest is a typed error.
    pub fn validate(&self, source: &Path) -> Result<(), ContentError> {
        if self.format != ASSET_MANIFEST_FORMAT {
            return Err(ContentError::unsupported(
                source,
                format!(
                    "manifest format {}; expected {ASSET_MANIFEST_FORMAT}",
                    self.format
                ),
            ));
        }
        if self.version != ASSET_GENERATOR_VERSION {
            return Err(ContentError::unsupported(
                source,
                format!(
                    "manifest was written by generator version {}; regenerate it with {ASSET_GENERATOR_VERSION}",
                    self.version
                ),
            ));
        }
        for (index, entry) in self.entries.iter().enumerate() {
            if index > 0 && self.entries[index - 1].path >= entry.path {
                return Err(ContentError::format(
                    source,
                    if self.entries[index - 1].path == entry.path {
                        format!("manifest repeats entry `{}`", entry.path)
                    } else {
                        format!("manifest entries are not sorted by path at `{}`", entry.path)
                    },
                ));
            }
            match entry.kind {
                AssetKind::Image => {
                    if entry.image.is_none() {
                        return Err(ContentError::format(
                            source,
                            format!("image `{}` has no scanned dimensions", entry.path),
                        ));
                    }
                }
                AssetKind::Audio => {
                    if entry.audio.is_none() {
                        return Err(ContentError::format(
                            source,
                            format!("audio `{}` has no scanned stream parameters", entry.path),
                        ));
                    }
                }
                _ => {
                    if entry.image.is_some() || entry.audio.is_some() {
                        return Err(ContentError::format(
                            source,
                            format!(
                                "`{}` is a {} but carries image or audio metadata",
                                entry.path,
                                entry.kind.as_str()
                            ),
                        ));
                    }
                }
            }
        }
        if let Some(atlas) = &self.atlas {
            atlas.validate(self, source)?;
        }
        Ok(())
    }
}

/// Which sources an atlas plan packs: the caller's explicit list, or every
/// plain image the manifest lists (a container texture is not a sprite).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AtlasSelection {
    AllImages,
    Sprites(Vec<AssetPath>),
}

/// A build-time request for an atlas plan.
#[derive(Clone, Debug)]
pub struct AtlasPlanRequest {
    desc: AtlasDesc,
    selection: AtlasSelection,
}

impl AtlasPlanRequest {
    /// Pack the given sprite paths. The plan sorts them, so the same set of
    /// sprites always produces the same plan.
    pub fn sprites<S: AsRef<str>>(
        desc: AtlasDesc,
        sprites: impl IntoIterator<Item = S>,
    ) -> Result<Self, ContentError> {
        validate_desc(desc)?;
        let mut paths = Vec::new();
        for sprite in sprites {
            paths.push(AssetPath::parse(sprite.as_ref())?);
        }
        Ok(Self {
            desc,
            selection: AtlasSelection::Sprites(paths),
        })
    }

    /// Pack every plain image in the manifest. A sprite that does not fit a
    /// page is a typed error naming it, never a silent omission.
    pub fn all_images(desc: AtlasDesc) -> Result<Self, ContentError> {
        validate_desc(desc)?;
        Ok(Self {
            desc,
            selection: AtlasSelection::AllImages,
        })
    }

    pub fn desc(&self) -> AtlasDesc {
        self.desc
    }

    pub fn selection(&self) -> &AtlasSelection {
        &self.selection
    }
}

fn validate_desc(desc: AtlasDesc) -> Result<(), ContentError> {
    if desc.size == 0 || desc.max_pages == 0 {
        return Err(ContentError::new(
            ASSET_MANIFEST_FILE,
            format!(
                "atlas page size {} and page count {} must both be non-zero",
                desc.size, desc.max_pages
            ),
        ));
    }
    Ok(())
}

/// One packed sprite: where it sits in the page and the UV rect covering it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AtlasPlacement {
    /// The content path, and the key the plan is stable under.
    pub path: AssetPath,
    /// `repame_atlas::atlas_id(path)`, so a caller can look a placement up with
    /// the same key the runtime atlas allocates under.
    pub key: u64,
    pub page: u32,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    /// `min x, min y, max x, max y` in page UV space, covering the sprite's
    /// inner rect (no padding).
    pub uv: [f32; 4],
}

/// A build-time atlas layout: page geometry plus one placement per sprite, in
/// path order. A game uploads the pack once and reads UVs from here, so it
/// never packs rectangles at runtime: allocate `pages` square buffers of
/// `page_size`, blit each sprite's pixels at `(x, y)`, upload, and sample the
/// `uv` rect.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AtlasPlan {
    pub page_size: u32,
    pub padding: u32,
    pub pages: u32,
    pub sprites: Vec<AtlasPlacement>,
}

impl AtlasPlan {
    /// Lay out the request's sprites over the manifest's scanned dimensions.
    ///
    /// Sprites are allocated in sorted path order, which is what makes the plan
    /// reproducible: replaying the same request through `repame_atlas::Atlas`
    /// yields the same rects.
    pub fn build(
        manifest: &ContentManifest,
        request: &AtlasPlanRequest,
    ) -> Result<Self, ContentError> {
        let mut paths: Vec<AssetPath> = match &request.selection {
            AtlasSelection::AllImages => manifest
                .entries
                .iter()
                .filter(|entry| entry.kind == AssetKind::Image)
                .filter(|entry| entry.image.as_ref().is_some_and(|image| !image.is_texture()))
                .map(|entry| entry.path.clone())
                .collect(),
            AtlasSelection::Sprites(paths) => paths.clone(),
        };
        paths.sort();
        if paths.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(ContentError::new(
                ASSET_MANIFEST_FILE,
                "atlas plan lists a sprite twice",
            ));
        }

        let desc = request.desc;
        let entries: std::collections::BTreeMap<&AssetPath, &ContentEntry> = manifest
            .entries
            .iter()
            .map(|entry| (&entry.path, entry))
            .collect();
        let mut atlas = Atlas::new(desc);
        for path in &paths {
            let entry = entries.get(path).copied().ok_or_else(|| {
                ContentError::new(path.as_str(), "atlas sprite is not a content entry")
            })?;
            let image = entry.image.as_ref().ok_or_else(|| {
                ContentError::new(path.as_str(), "atlas sprite has no scanned dimensions")
            })?;
            if image.is_texture() {
                return Err(ContentError::new(
                    path.as_str(),
                    "atlas sprites must be plain images, not DDS/KTX 2 containers",
                ));
            }
            atlas
                .alloc(atlas_id(path.as_str()), image.width, image.height)
                .map_err(|error| match error {
                    AllocError::TooLarge => ContentError::new(
                        path.as_str(),
                        format!(
                            "sprite {}x{} does not fit a {} px page with {} px padding",
                            image.width, image.height, desc.size, desc.padding
                        ),
                    ),
                    AllocError::OutOfSpace => ContentError::new(
                        path.as_str(),
                        format!(
                            "atlas is full after {} pages; raise max_pages or page_size",
                            desc.max_pages
                        ),
                    ),
                    AllocError::Duplicate => ContentError::new(
                        path.as_str(),
                        "atlas plan lists a sprite twice",
                    ),
                })?;
        }

        let writes = atlas.drain_writes();
        if writes.len() != paths.len() {
            return Err(ContentError::new(
                ASSET_MANIFEST_FILE,
                format!(
                    "atlas planned {} sprites but allocated {}",
                    paths.len(),
                    writes.len()
                ),
            ));
        }
        let sprites = writes
            .iter()
            .zip(&paths)
            .map(|(write, path)| AtlasPlacement {
                path: path.clone(),
                key: write.key.0,
                page: write.page,
                x: write.x,
                y: write.y,
                w: write.w,
                h: write.h,
                uv: [write.uv.min[0], write.uv.min[1], write.uv.max[0], write.uv.max[1]],
            })
            .collect();
        Ok(Self {
            page_size: desc.size,
            padding: desc.padding,
            pages: atlas.page_count(),
            sprites,
        })
    }

    pub fn len(&self) -> usize {
        self.sprites.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sprites.is_empty()
    }

    pub fn placement(&self, path: &AssetPath) -> Option<&AtlasPlacement> {
        self.sprites
            .binary_search_by(|sprite| sprite.path.cmp(path))
            .ok()
            .map(|index| &self.sprites[index])
    }

    /// Look a placement up by the key a runtime atlas would allocate under.
    pub fn placement_by_key(&self, key: u64) -> Option<&AtlasPlacement> {
        self.sprites.iter().find(|sprite| sprite.key == key)
    }

    /// Check the plan against the manifest it belongs to: sorted and unique
    /// placements, dims that match the scanned sprite, rectangles inside their
    /// page, UVs that agree with the pixel rect, and shelf rows that do not
    /// overlap. A hand-edited or corrupt plan fails instead of drawing garbage.
    pub fn validate(&self, manifest: &ContentManifest, source: &Path) -> Result<(), ContentError> {
        if self.page_size == 0
            || self.padding.saturating_mul(2) >= self.page_size
            || (self.pages == 0 && !self.sprites.is_empty())
        {
            return Err(ContentError::format(
                source,
                format!(
                    "atlas plan has page size {}, {} pages, {} px padding and {} sprites",
                    self.page_size,
                    self.pages,
                    self.padding,
                    self.sprites.len()
                ),
            ));
        }
        let mut rows: std::collections::BTreeMap<(u32, u32), Vec<(u32, u32, &AssetPath)>> =
            std::collections::BTreeMap::new();
        for (index, sprite) in self.sprites.iter().enumerate() {
            if index > 0 && self.sprites[index - 1].path >= sprite.path {
                return Err(ContentError::format(
                    source,
                    format!("atlas plan is not sorted by path at `{}`", sprite.path),
                ));
            }
            if sprite.key != atlas_id(sprite.path.as_str()).0 {
                return Err(ContentError::format(
                    source,
                    format!("atlas placement `{}` has a foreign key", sprite.path),
                ));
            }
            if sprite.page >= self.pages {
                return Err(ContentError::format(
                    source,
                    format!(
                        "atlas placement `{}` names page {} of {}",
                        sprite.path, sprite.page, self.pages
                    ),
                ));
            }
            if sprite.w == 0
                || sprite.h == 0
                || sprite.x.checked_add(sprite.w).is_none_or(|end| end > self.page_size)
                || sprite.y.checked_add(sprite.h).is_none_or(|end| end > self.page_size)
            {
                return Err(ContentError::format(
                    source,
                    format!(
                        "atlas placement `{}` at {}x{} {}x{} leaves its {} px page",
                        sprite.path, sprite.x, sprite.y, sprite.w, sprite.h, self.page_size
                    ),
                ));
            }
            let entry = manifest.entry(&sprite.path).ok_or_else(|| {
                ContentError::format(
                    source,
                    format!("atlas placement `{}` is not a content entry", sprite.path),
                )
            })?;
            let image = entry.image.as_ref().ok_or_else(|| {
                ContentError::format(
                    source,
                    format!("atlas placement `{}` is not an image", sprite.path),
                )
            })?;
            if image.width != sprite.w || image.height != sprite.h {
                return Err(ContentError::format(
                    source,
                    format!(
                        "atlas placement `{}` is {}x{} but the sprite is {}x{}",
                        sprite.path, sprite.w, sprite.h, image.width, image.height
                    ),
                ));
            }
            let scale = self.page_size as f32;
            let expected = [
                sprite.x as f32 / scale,
                sprite.y as f32 / scale,
                (sprite.x + sprite.w) as f32 / scale,
                (sprite.y + sprite.h) as f32 / scale,
            ];
            if !sprite.uv.iter().all(|value| value.is_finite() && (0.0..=1.0).contains(value))
                || sprite
                    .uv
                    .iter()
                    .zip(expected.iter())
                    .any(|(stored, computed)| (stored - computed).abs() > 1.0 / scale)
            {
                return Err(ContentError::format(
                    source,
                    format!("atlas placement `{}` has UVs that leave its rect", sprite.path),
                ));
            }
            rows.entry((sprite.page, sprite.y))
                .or_default()
                .push((sprite.x, sprite.x + sprite.w, &sprite.path));
        }
        for ((page, y), mut row) in rows {
            row.sort_by_key(|entry| entry.0);
            for pair in row.windows(2) {
                if pair[0].1 > pair[1].0 {
                    return Err(ContentError::format(
                        source,
                        format!(
                            "atlas placements `{}` and `{}` overlap on page {page} at y {y}",
                            pair[0].2, pair[1].2
                        ),
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_paths_reject_anything_that_could_escape_the_root() {
        for bad in [
            "",
            "/abs/x.png",
            "a//b.png",
            "a/./b.png",
            "a/../b.png",
            "..",
            "a\\b.png",
            "C:/x.png",
            "a/b.png/",
        ] {
            assert!(AssetPath::parse(bad).is_err(), "`{bad}` must be refused");
        }
        let path = AssetPath::parse("gfx/ui/cursor.png").expect("valid");
        assert_eq!(path.dev_path("root").to_string_lossy(), "root/gfx/ui/cursor.png");
        assert_eq!(path.entry_name(), path.as_str());
        assert_eq!(path.file_name(), Some("cursor.png"));
        assert_eq!(path.extension().as_deref(), Some("png"));
        assert!(path.starts_with_dir("gfx/ui"));
        assert!(!path.starts_with_dir("gfx/u"));
    }

    #[test]
    fn manifest_round_trips_and_refuses_inconsistent_records() {
        let mut manifest = ContentManifest::new("game");
        manifest.entries.push(ContentEntry {
            path: AssetPath::parse("beep.wav").expect("path"),
            kind: AssetKind::Audio,
            size: 5,
            hash: 6,
            image: None,
            audio: Some(AudioInfo {
                sample_rate: Some(8000),
                channels: Some(1),
            }),
        });
        manifest.entries.push(ContentEntry {
            path: AssetPath::parse("hero.png").expect("path"),
            kind: AssetKind::Image,
            size: 12,
            hash: 34,
            image: Some(ImageInfo {
                width: 4,
                height: 3,
                texture: None,
            }),
            audio: None,
        });
        let text = manifest.to_ron().expect("serialize");
        let parsed = ContentManifest::from_ron(&text, Path::new("assets.ron")).expect("parse");
        assert_eq!(parsed, manifest);

        let mut broken = manifest.clone();
        broken.entries[1].image = None;
        let error = broken
            .validate(Path::new("assets.ron"))
            .expect_err("image without dimensions");
        assert_eq!(error.kind(), crate::ContentErrorKind::Format);
        assert!(error.message().contains("hero.png"), "{}", error.message());

        let mut unordered = manifest.clone();
        unordered.entries.swap(0, 1);
        assert!(unordered.validate(Path::new("assets.ron")).is_err());
    }
}
