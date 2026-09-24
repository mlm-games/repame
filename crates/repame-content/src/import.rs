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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ImportRecord {
    pub format: u32,
    pub id: String,
    pub kind: String,
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
        Ok(Self { root, storage })
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
        let fingerprint = content_hash(&bytes);
        let cache_key = cache_key(&source.id);
        let record_path = self.root.join("records").join(format!("{cache_key}.ron"));
        let artifact_path = self.root.join("artifacts").join(format!("{cache_key}.bin"));

        if let Some(record) = self.read_record(&record_path)? {
            let cached_artifact = self.root.join(&record.artifact);
            let valid = record.format == IMPORT_FORMAT
                && record.id == source.id
                && record.kind == kind.as_str()
                && record.source_fingerprint == fingerprint
                && record.importer_version == IMPORTER_VERSION
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

        let metadata = inspect(kind, &bytes, &source.path)?;
        self.write_atomic(&artifact_path, &bytes)?;
        let record = ImportRecord {
            format: IMPORT_FORMAT,
            id: source.id.clone(),
            kind: kind.as_str().to_string(),
            source_fingerprint: fingerprint,
            importer_version: IMPORTER_VERSION,
            artifact: format!("artifacts/{cache_key}.bin"),
            dependencies: dependency_fingerprints.clone(),
            metadata: metadata.clone(),
        };
        let record_text = ron::ser::to_string_pretty(&record, Default::default())
            .map_err(|error| ron_error(&record_path, error))?;
        self.write_atomic(&record_path, record_text.as_bytes())?;

        Ok(ImportedAsset {
            id: source.id.clone(),
            kind,
            artifact: artifact_path,
            bytes,
            metadata,
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

fn inspect(
    kind: AssetKind,
    bytes: &[u8],
    path: &Path,
) -> Result<BTreeMap<String, ron::Value>, ContentError> {
    let mut metadata = BTreeMap::new();
    match kind {
        AssetKind::Image => {
            let image = image::load_from_memory(bytes)
                .map_err(|error| ContentError::new(path, format!("decode image: {error}")))?;
            let (width, height) = image.dimensions();
            metadata.insert("width".to_string(), ron::Value::from(width as i64));
            metadata.insert("height".to_string(), ron::Value::from(height as i64));
        }
        AssetKind::Audio => {
            let stream =
                MediaSourceStream::new(Box::new(Cursor::new(bytes.to_vec())), Default::default());
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
            if let Some(rate) = audio.sample_rate {
                metadata.insert("sample_rate".to_string(), ron::Value::from(rate as i64));
            }
            if let Some(channels) = &audio.channels {
                metadata.insert(
                    "channels".to_string(),
                    ron::Value::from(channels.count() as i64),
                );
            }
        }
        AssetKind::Gltf => {
            let (document, _buffers, _images) = gltf::import_slice(bytes)
                .map_err(|error| ContentError::new(path, format!("decode glTF: {error}")))?;
            metadata.insert(
                "meshes".to_string(),
                ron::Value::from(document.meshes().count() as i64),
            );
            metadata.insert(
                "materials".to_string(),
                ron::Value::from(document.materials().count() as i64),
            );
            metadata.insert(
                "textures".to_string(),
                ron::Value::from(document.textures().count() as i64),
            );
        }
        AssetKind::Ron => {
            let text = std::str::from_utf8(bytes)
                .map_err(|error| ContentError::new(path, format!("RON is not UTF-8: {error}")))?;
            ron::from_str::<ron::Value>(text).map_err(|error| ron_error(path, error))?;
            metadata.insert("valid".to_string(), ron::Value::Bool(true));
        }
        AssetKind::Text => {
            std::str::from_utf8(bytes)
                .map_err(|error| ContentError::new(path, format!("text is not UTF-8: {error}")))?;
            metadata.insert("valid".to_string(), ron::Value::Bool(true));
        }
        AssetKind::Binary => {
            metadata.insert("bytes".to_string(), ron::Value::from(bytes.len() as i64));
        }
    }
    Ok(metadata)
}

fn cache_key(id: &str) -> String {
    format!("{:016x}", content_hash(id.as_bytes()))
}
