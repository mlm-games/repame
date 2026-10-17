//! Plugin-shaped asset import: one [`Importer`] per source kind, selected
//! through an [`ImporterRegistry`], with provenance written to a sidecar.
//!
//! This mirrors Godot's `EditorImportPlugin` + `.import` sidecar model and
//! Bevy's asset processors: the source bytes are read once, an importer keyed
//! by asset kind, file extension or magic produces metadata, and a record
//! (importer, source hash, importer version, dependency fingerprints) is
//! written next to the cached artifact so an unchanged source is never
//! re-imported.

use std::collections::BTreeMap;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use image::GenericImageView;
use serde::{Deserialize, Serialize};
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use crate::error::{ContentError, ron_error};
use crate::project::{AssetSource, content_hash};
use game_utils::storage::{FsStorage, Storage};

pub const IMPORT_FORMAT: u32 = 1;
pub const IMPORTER_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssetKind {
    Image,
    Audio,
    Gltf,
    Ron,
    Text,
    Binary,
}

impl AssetKind {
    pub fn parse(value: &str, path: &Path) -> Result<Self, ContentError> {
        match value.to_ascii_lowercase().as_str() {
            "image" => Ok(Self::Image),
            "audio" => Ok(Self::Audio),
            "gltf" | "glb" => Ok(Self::Gltf),
            "ron" => Ok(Self::Ron),
            "text" => Ok(Self::Text),
            "binary" => Ok(Self::Binary),
            _ => Err(ContentError::new(
                path,
                format!("unsupported asset kind `{value}`"),
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Audio => "audio",
            Self::Gltf => "gltf",
            Self::Ron => "ron",
            Self::Text => "text",
            Self::Binary => "binary",
        }
    }
}

/// One import request: the source path and its bytes.
#[derive(Clone, Copy, Debug)]
pub struct ImportContext<'a> {
    pub path: &'a Path,
    pub bytes: &'a [u8],
}

/// What an [`Importer`] produces from a source. The artifact itself is the
/// source bytes; importers only derive metadata and validate the source.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImportOutput {
    pub metadata: BTreeMap<String, ron::Value>,
}

impl ImportOutput {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, key: impl Into<String>, value: impl Into<ron::Value>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// A per-source-kind importer: bytes + path in, [`ImportOutput`] out.
pub trait Importer: Send + Sync {
    /// Stable provenance name written to the sidecar record.
    fn name(&self) -> &'static str;

    /// Provenance version; bump when the derived metadata changes meaning.
    fn version(&self) -> u32 {
        IMPORTER_VERSION
    }

    /// The declared asset kind this importer serves.
    fn kind(&self) -> AssetKind;

    /// Lowercase extensions this importer claims, for detection and disambiguation.
    fn extensions(&self) -> &'static [&'static str] {
        &[]
    }

    /// True when the leading bytes identify this importer's format.
    fn matches(&self, bytes: &[u8]) -> bool {
        let _ = bytes;
        false
    }

    fn import(&self, context: &ImportContext<'_>) -> Result<ImportOutput, ContentError>;
}

/// Importer lookup, keyed by asset kind plus extension/magic.
pub struct ImporterRegistry {
    importers: Vec<Box<dyn Importer>>,
}

impl ImporterRegistry {
    /// A registry populated with the built-in importers.
    pub fn new() -> Self {
        Self::with_builtins()
    }

    /// An empty registry, for callers that register their own importers.
    pub fn empty() -> Self {
        Self {
            importers: Vec::new(),
        }
    }

    /// A registry populated with the built-in importers.
    pub fn with_builtins() -> Self {
        let mut registry = Self::empty();
        for importer in builtins() {
            registry.importers.push(importer);
        }
        registry
    }

    /// Add an importer. Fails on a duplicate provenance name.
    pub fn register(&mut self, importer: Box<dyn Importer>) -> Result<(), String> {
        if self
            .importers
            .iter()
            .any(|existing| existing.name() == importer.name())
        {
            return Err(format!("importer `{}` is already registered", importer.name()));
        }
        self.importers.push(importer);
        Ok(())
    }

    pub fn importers(&self) -> impl Iterator<Item = &dyn Importer> {
        self.importers.iter().map(Box::as_ref)
    }

    pub fn is_empty(&self) -> bool {
        self.importers.is_empty()
    }

    /// First importer claiming this file extension.
    pub fn by_extension(&self, path: &Path) -> Option<&dyn Importer> {
        let extension = path
            .extension()
            .and_then(|value| value.to_str())
            .map(str::to_ascii_lowercase)?;
        self.importers
            .iter()
            .map(Box::as_ref)
            .find(|importer| importer.extensions().contains(&extension.as_str()))
    }

    /// First importer whose magic matches.
    pub fn by_magic(&self, bytes: &[u8]) -> Option<&dyn Importer> {
        self.importers
            .iter()
            .map(Box::as_ref)
            .find(|importer| importer.matches(bytes))
    }

    /// Resolve the importer for a declared kind, disambiguated by extension then magic.
    pub fn resolve(
        &self,
        kind: AssetKind,
        path: &Path,
        bytes: &[u8],
    ) -> Result<&dyn Importer, ContentError> {
        let candidates: Vec<&dyn Importer> = self
            .importers
            .iter()
            .map(Box::as_ref)
            .filter(|importer| importer.kind() == kind)
            .collect();
        match candidates.len() {
            0 => Err(ContentError::unsupported(
                path,
                format!("no importer registered for asset kind `{}`", kind.as_str()),
            )),
            1 => Ok(candidates[0]),
            _ => {
                let extension = path
                    .extension()
                    .and_then(|value| value.to_str())
                    .map(str::to_ascii_lowercase);
                if let Some(extension) = extension {
                    if let Some(importer) = candidates
                        .iter()
                        .copied()
                        .find(|importer| importer.extensions().contains(&extension.as_str()))
                    {
                        return Ok(importer);
                    }
                }
                if let Some(importer) = candidates.iter().copied().find(|importer| importer.matches(bytes)) {
                    return Ok(importer);
                }
                Err(ContentError::format(
                    path,
                    format!(
                        "ambiguous importers for asset kind `{}`",
                        kind.as_str()
                    ),
                ))
            }
        }
    }
}

impl Default for ImporterRegistry {
    fn default() -> Self {
        Self::with_builtins()
    }
}

fn builtins() -> Vec<Box<dyn Importer>> {
    vec![
        Box::new(ImageImporter),
        Box::new(AudioImporter),
        Box::new(GltfImporter),
        Box::new(RonImporter),
        Box::new(TextImporter),
        Box::new(BinaryImporter),
    ]
}

/// Decodes images through the `image` crate and reports their dimensions.
#[derive(Clone, Copy, Debug, Default)]
pub struct ImageImporter;

impl Importer for ImageImporter {
    fn name(&self) -> &'static str {
        "image"
    }

    fn kind(&self) -> AssetKind {
        AssetKind::Image
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["png", "jpg", "jpeg", "webp"]
    }

    fn matches(&self, bytes: &[u8]) -> bool {
        bytes.starts_with(&[0x89, b'P', b'N', b'G'])
            || bytes.starts_with(&[0xFF, 0xD8, 0xFF])
            || (bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP")
    }

    fn import(&self, context: &ImportContext<'_>) -> Result<ImportOutput, ContentError> {
        let image = image::load_from_memory(context.bytes)
            .map_err(|error| ContentError::new(context.path, format!("decode image: {error}")))?;
        let (width, height) = image.dimensions();
        Ok(ImportOutput::new()
            .with("width", ron::Value::from(width as i64))
            .with("height", ron::Value::from(height as i64)))
    }
}

/// Probes audio through `symphonia` and reports stream parameters.
#[derive(Clone, Copy, Debug, Default)]
pub struct AudioImporter;

impl Importer for AudioImporter {
    fn name(&self) -> &'static str {
        "audio"
    }

    fn kind(&self) -> AssetKind {
        AssetKind::Audio
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["wav", "flac", "mp3", "ogg", "oga", "m4a", "mp4", "mkv", "caf"]
    }

    fn matches(&self, bytes: &[u8]) -> bool {
        bytes.starts_with(b"RIFF")
            || bytes.starts_with(b"fLaC")
            || bytes.starts_with(b"OggS")
            || bytes.starts_with(b"ID3")
            || bytes.starts_with(b"caff")
            || bytes.starts_with(&[0xFF, 0xFB])
    }

    fn import(&self, context: &ImportContext<'_>) -> Result<ImportOutput, ContentError> {
        let path = context.path;
        let stream =
            MediaSourceStream::new(Box::new(Cursor::new(context.bytes.to_vec())), Default::default());
        let format = symphonia::default::get_probe()
            .probe(
                &Hint::new(),
                stream,
                FormatOptions::default(),
                MetadataOptions::default(),
            )
            .map_err(|error| ContentError::new(path, format!("probe audio: {error}")))?;
        let track = format
            .default_track(TrackType::Audio)
            .ok_or_else(|| ContentError::new(path, "audio has no audio track"))?;
        let params = track
            .codec_params
            .clone()
            .ok_or_else(|| ContentError::new(path, "audio track has no codec parameters"))?;
        let audio = params
            .audio()
            .ok_or_else(|| ContentError::new(path, "audio track is not an audio track"))?;
        let mut output = ImportOutput::new();
        if let Some(rate) = audio.sample_rate {
            output
                .metadata
                .insert("sample_rate".to_string(), ron::Value::from(rate as i64));
        }
        if let Some(channels) = &audio.channels {
            output.metadata.insert(
                "channels".to_string(),
                ron::Value::from(channels.count() as i64),
            );
        }
        Ok(output)
    }
}

/// Counts glTF documents through the `gltf` crate.
#[derive(Clone, Copy, Debug, Default)]
pub struct GltfImporter;

impl Importer for GltfImporter {
    fn name(&self) -> &'static str {
        "gltf"
    }

    fn kind(&self) -> AssetKind {
        AssetKind::Gltf
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["gltf", "glb"]
    }

    fn matches(&self, bytes: &[u8]) -> bool {
        bytes.starts_with(b"glTF")
            || bytes
                .iter()
                .find(|byte| !byte.is_ascii_whitespace())
                .is_some_and(|byte| *byte == b'{')
    }

    fn import(&self, context: &ImportContext<'_>) -> Result<ImportOutput, ContentError> {
        let path = context.path;
        let (document, _buffers, _images) = gltf::import_slice(context.bytes)
            .map_err(|error| ContentError::new(path, format!("decode glTF: {error}")))?;
        Ok(ImportOutput::new()
            .with("meshes", ron::Value::from(document.meshes().count() as i64))
            .with(
                "materials",
                ron::Value::from(document.materials().count() as i64),
            )
            .with(
                "textures",
                ron::Value::from(document.textures().count() as i64),
            ))
    }
}

/// Validates RON syntax.
#[derive(Clone, Copy, Debug, Default)]
pub struct RonImporter;

impl Importer for RonImporter {
    fn name(&self) -> &'static str {
        "ron"
    }

    fn kind(&self) -> AssetKind {
        AssetKind::Ron
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["ron"]
    }

    fn import(&self, context: &ImportContext<'_>) -> Result<ImportOutput, ContentError> {
        let text = std::str::from_utf8(context.bytes).map_err(|error| {
            ContentError::new(context.path, format!("RON is not UTF-8: {error}"))
        })?;
        ron::from_str::<ron::Value>(text).map_err(|error| ron_error(context.path, error))?;
        Ok(ImportOutput::new().with("valid", ron::Value::Bool(true)))
    }
}

/// Validates UTF-8 text.
#[derive(Clone, Copy, Debug, Default)]
pub struct TextImporter;

impl Importer for TextImporter {
    fn name(&self) -> &'static str {
        "text"
    }

    fn kind(&self) -> AssetKind {
        AssetKind::Text
    }

    fn extensions(&self) -> &'static [&'static str] {
        &[
            "txt", "md", "csv", "json", "toml", "yaml", "yml", "xml", "wgsl", "glsl", "frag",
            "vert",
        ]
    }

    fn import(&self, context: &ImportContext<'_>) -> Result<ImportOutput, ContentError> {
        std::str::from_utf8(context.bytes).map_err(|error| {
            ContentError::new(context.path, format!("text is not UTF-8: {error}"))
        })?;
        Ok(ImportOutput::new().with("valid", ron::Value::Bool(true)))
    }
}

/// Accepts any bytes and records their length.
#[derive(Clone, Copy, Debug, Default)]
pub struct BinaryImporter;

impl Importer for BinaryImporter {
    fn name(&self) -> &'static str {
        "binary"
    }

    fn kind(&self) -> AssetKind {
        AssetKind::Binary
    }

    fn import(&self, context: &ImportContext<'_>) -> Result<ImportOutput, ContentError> {
        Ok(ImportOutput::new().with("bytes", ron::Value::from(context.bytes.len() as i64)))
    }
}

/// Provenance sidecar for one imported source, mirroring Godot's `.import`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ImportRecord {
    pub format: u32,
    pub id: String,
    pub kind: String,
    /// Provenance: the importer that produced this record.
    #[serde(default)]
    pub importer: String,
    pub source_fingerprint: u64,
    pub importer_version: u32,
    pub artifact: String,
    pub dependencies: BTreeMap<String, u64>,
    pub metadata: BTreeMap<String, ron::Value>,
}

#[derive(Clone, Debug)]
pub struct ImportedAsset {
    pub id: String,
    pub kind: AssetKind,
    pub artifact: PathBuf,
    pub bytes: Vec<u8>,
    pub metadata: BTreeMap<String, ron::Value>,
    pub reused: bool,
}

pub struct ImportCache<S: Storage = FsStorage> {
    root: PathBuf,
    storage: S,
    registry: ImporterRegistry,
}

impl ImportCache<FsStorage> {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self, ContentError> {
        Self::new_with_storage(root, FsStorage)
    }
}

impl<S: Storage> ImportCache<S> {
    pub fn new_with_storage(root: impl Into<PathBuf>, storage: S) -> Result<Self, ContentError> {
        let root = root.into();
        storage
            .create_dir_all(&root)
            .map_err(|error| ContentError::new(&root, format!("create import cache: {error}")))?;
        Ok(Self {
            root,
            storage,
            registry: ImporterRegistry::default(),
        })
    }

    /// Replace the importer registry, e.g. with one carrying project importers.
    pub fn with_registry(mut self, registry: ImporterRegistry) -> Self {
        self.registry = registry;
        self
    }

    pub fn registry(&self) -> &ImporterRegistry {
        &self.registry
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn storage(&self) -> &S {
        &self.storage
    }

    pub fn import(
        &self,
        source: &AssetSource,
        dependency_fingerprints: &BTreeMap<String, u64>,
    ) -> Result<ImportedAsset, ContentError> {
        let kind = AssetKind::parse(&source.kind, &source.path)?;
        let bytes = self.read_required(&source.path, "asset source")?;
        let importer = self.registry.resolve(kind, &source.path, &bytes)?;
        let fingerprint = content_hash(&bytes);
        let cache_key = cache_key(&source.id);
        let record_path = self.root.join("records").join(format!("{cache_key}.ron"));
        let artifact_path = self.root.join("artifacts").join(format!("{cache_key}.bin"));

        if let Some(record) = self.read_record(&record_path)? {
            let cached_artifact = self.root.join(&record.artifact);
            let valid = record.format == IMPORT_FORMAT
                && record.id == source.id
                && record.kind == kind.as_str()
                && (record.importer.is_empty() || record.importer == importer.name())
                && record.source_fingerprint == fingerprint
                && record.importer_version == importer.version()
                && record.dependencies == *dependency_fingerprints
                && cached_artifact == artifact_path
                && self.storage.is_file(&artifact_path);
            if valid {
                let bytes = self.read_required(&artifact_path, "cached asset artifact")?;
                return Ok(ImportedAsset {
                    id: source.id.clone(),
                    kind,
                    artifact: artifact_path,
                    bytes,
                    metadata: record.metadata,
                    reused: true,
                });
            }
        }

        let output = importer.import(&ImportContext {
            path: &source.path,
            bytes: &bytes,
        })?;
        self.write_atomic(&artifact_path, &bytes)?;
        let record = ImportRecord {
            format: IMPORT_FORMAT,
            id: source.id.clone(),
            kind: kind.as_str().to_string(),
            importer: importer.name().to_string(),
            source_fingerprint: fingerprint,
            importer_version: importer.version(),
            artifact: format!("artifacts/{cache_key}.bin"),
            dependencies: dependency_fingerprints.clone(),
            metadata: output.metadata.clone(),
        };
        let record_text = ron::ser::to_string_pretty(&record, Default::default())
            .map_err(|error| ron_error(&record_path, error))?;
        self.write_atomic(&record_path, record_text.as_bytes())?;

        Ok(ImportedAsset {
            id: source.id.clone(),
            kind,
            artifact: artifact_path,
            bytes,
            metadata: output.metadata,
            reused: false,
        })
    }

    fn read_record(&self, path: &Path) -> Result<Option<ImportRecord>, ContentError> {
        let Some(bytes) = self
            .storage
            .read(path)
            .map_err(|error| ContentError::new(path, format!("read import record: {error}")))?
        else {
            return Ok(None);
        };
        let text = std::str::from_utf8(&bytes).map_err(|error| {
            ContentError::new(path, format!("import record is not UTF-8: {error}"))
        })?;
        Ok(ron::from_str(text).ok())
    }

    fn read_required(&self, path: &Path, kind: &str) -> Result<Vec<u8>, ContentError> {
        self.storage
            .read(path)
            .map_err(|error| ContentError::new(path, format!("read {kind}: {error}")))?
            .ok_or_else(|| ContentError::new(path, format!("{kind} is missing")))
    }

    fn write_atomic(&self, path: &Path, bytes: &[u8]) -> Result<(), ContentError> {
        if let Some(parent) = path.parent() {
            self.storage.create_dir_all(parent).map_err(|error| {
                ContentError::new(parent, format!("create import directory: {error}"))
            })?;
        }
        let temporary = path.with_extension("tmp");
        self.storage.write(&temporary, bytes).map_err(|error| {
            ContentError::new(&temporary, format!("write import artifact: {error}"))
        })?;
        if let Err(error) = self.storage.rename(&temporary, path) {
            let _ = self.storage.remove_file(&temporary);
            return Err(ContentError::new(
                path,
                format!("replace import artifact: {error}"),
            ));
        }
        Ok(())
    }
}

fn cache_key(id: &str) -> String {
    format!("{:016x}", content_hash(id.as_bytes()))
}
