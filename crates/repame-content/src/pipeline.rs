//! Build-time asset pipeline: scan a source directory, write the manifest and
//! pack the archive.
//!
//! [`AssetPipeline`] is the one entry point a game calls from a `build.rs` or
//! its `main`: [`AssetPipeline::build`] walks the source tree once, classifies
//! and measures every file, lays out the atlas plan if one was requested, and
//! writes `<output root>/assets.ron` plus `<output root>/assets.pack`.
//! [`build_assets`] is that with the output root set to the source root, which
//! is the layout a game that owns its art directory wants: the same directory
//! serves the content in development and holds what a packaged build ships.
//! A game whose art lives in an external (possibly read-only) checkout writes
//! the manifest and pack elsewhere and serves development content with
//! [`LooseAssets::open_with_manifest`](crate::LooseAssets::open_with_manifest),
//! so the content paths stay relative to the source root either way.
//!
//! There is deliberately no CLI framework and no plugin system: a game owns its
//! own build step and calls one function. The pipeline reads sources through
//! [`Storage`], so a build runs on the real filesystem or, in tests, entirely
//! in memory.
//!
//! Images are measured without decoding pixels (a header read for PNG/JPEG/
//! WebP, the container header for DDS/KTX 2), audio is probed through
//! symphonia, and every other file is classified by the importer registry's
//! extension and magic tables. A source whose bytes do not match its kind is a
//! typed error naming its content path; nothing is dropped from the manifest
//! because it could not be measured.

use std::collections::BTreeSet;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use game_utils::storage::{FsStorage, Storage};
use image::ImageReader;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use crate::codec::{self, DDS_MAGIC, KTX2_IDENTIFIER};
use crate::error::{ContentError, io_error};
use crate::import::{AssetKind, AudioImporter, ImageImporter, Importer, ImporterRegistry};
use crate::manifest::{
    ASSET_MANIFEST_FILE, ASSET_PACK_FILE, AssetPath, AtlasPlan, AtlasPlanRequest, AudioInfo,
    ContentEntry, ContentManifest, ImageInfo, TextureMeta,
};
use crate::package::{Compression, PackageWriter};
use crate::project::content_hash;

/// Texture container extensions measured through the crate's own container
/// parsers rather than the `image` crate.
const TEXTURE_EXTENSIONS: [&str; 3] = ["dds", "ktx2", "ktx"];

/// What [`AssetPipeline::build`] wrote.
#[derive(Clone, Debug)]
pub struct AssetBuild {
    /// The manifest that was written, and packed as an archive entry.
    pub manifest: ContentManifest,
    pub manifest_path: PathBuf,
    pub archive_path: PathBuf,
    /// Packed archive length in bytes.
    pub archive_size: u64,
}

impl AssetBuild {
    pub fn entries(&self) -> usize {
        self.manifest.len()
    }

    /// Atlas pages in the written plan, when one was requested.
    pub fn atlas_pages(&self) -> Option<u32> {
        self.manifest.atlas().map(|atlas| atlas.pages)
    }
}

/// Scans a content root and writes the asset table and its packed archive.
pub struct AssetPipeline<S: Storage = FsStorage> {
    source_root: PathBuf,
    storage: S,
    name: String,
    registry: ImporterRegistry,
    atlas: Option<AtlasPlanRequest>,
    compression: Option<Compression>,
    excludes: Vec<String>,
}

impl AssetPipeline<FsStorage> {
    /// Scan `source_root` on the real filesystem.
    pub fn new(source_root: impl Into<PathBuf>) -> Result<Self, ContentError> {
        Self::with_storage(source_root, FsStorage)
    }
}

impl<S: Storage> AssetPipeline<S> {
    /// Scan `source_root` through `storage`. The root must already be a
    /// directory; a missing source tree is a typed error naming it, never an
    /// empty manifest.
    pub fn with_storage(source_root: impl Into<PathBuf>, storage: S) -> Result<Self, ContentError> {
        let source_root = source_root.into();
        if !storage.is_dir(&source_root) {
            return Err(ContentError::missing(
                &source_root,
                format!("content root {}", source_root.display()),
            ));
        }
        Ok(Self {
            source_root,
            storage,
            name: String::new(),
            registry: ImporterRegistry::default(),
            atlas: None,
            compression: None,
            excludes: Vec::new(),
        })
    }

    /// A label stored in the manifest, e.g. the game name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Resolve a sprite atlas plan at scan time, so nothing packs at runtime.
    pub fn atlas(mut self, request: AtlasPlanRequest) -> Self {
        self.atlas = Some(request);
        self
    }

    /// Force one compression method for every packed entry. Without it, image
    /// and audio containers (already compressed) are stored and everything else
    /// is deflated.
    pub fn compression(mut self, compression: Compression) -> Self {
        self.compression = Some(compression);
        self
    }

    /// Leave a content path or directory out of the manifest and the archive.
    /// Only explicit calls exclude a file, and each one is the caller's.
    pub fn exclude(mut self, prefix: impl AsRef<str>) -> Self {
        self.excludes.push(prefix.as_ref().to_string());
        self
    }

    pub fn source_root(&self) -> &Path {
        &self.source_root
    }

    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// Walk the source tree and build the manifest for it: one entry per file
    /// in sorted content-path order, plus the atlas plan when requested.
    pub fn scan(&self) -> Result<ContentManifest, ContentError> {
        if !self.storage.is_dir(&self.source_root) {
            return Err(ContentError::missing(
                &self.source_root,
                format!("content root {}", self.source_root.display()),
            ));
        }
        let paths = self.sources()?;
        let mut manifest = ContentManifest::new(self.name.clone());
        manifest.entries.reserve(paths.len());
        for path in paths {
            let bytes = self.read_required(&path)?;
            manifest.entries.push(self.entry(path, &bytes)?);
        }
        if let Some(request) = &self.atlas {
            manifest.atlas = Some(AtlasPlan::build(&manifest, request)?);
        }
        // Never emit a manifest the loader would refuse.
        manifest.validate(Path::new(ASSET_MANIFEST_FILE))?;
        Ok(manifest)
    }

    /// Write only the manifest, into `output_root`.
    pub fn write_manifest(&self, output_root: impl AsRef<Path>) -> Result<ContentManifest, ContentError> {
        let manifest = self.scan()?;
        let path = output_root.as_ref().join(ASSET_MANIFEST_FILE);
        write_atomic(&self.storage, &path, manifest.to_ron()?.as_bytes())?;
        Ok(manifest)
    }

    /// Scan once and write both the manifest and the packed archive into
    /// `output_root`. This is the entry point a game calls.
    ///
    /// Content paths stay relative to the source root whatever the output root
    /// is: development serves them with
    /// [`LooseAssets::open_with_manifest`](crate::LooseAssets::open_with_manifest)
    /// when the two differ, and a packaged build carries the manifest inside
    /// the archive, so it needs no separate copy.
    pub fn build(&self, output_root: impl AsRef<Path>) -> Result<AssetBuild, ContentError> {
        let output_root = output_root.as_ref();
        let manifest = self.scan()?;
        let manifest_path = output_root.join(ASSET_MANIFEST_FILE);
        let archive_path = output_root.join(ASSET_PACK_FILE);
        write_atomic(&self.storage, &manifest_path, manifest.to_ron()?.as_bytes())?;
        let archive = self.pack(&manifest, &archive_path)?;
        let archive_size = archive.len() as u64;
        write_atomic(&self.storage, &archive_path, &archive)?;
        Ok(AssetBuild {
            manifest,
            manifest_path,
            archive_path,
            archive_size,
        })
    }

    /// Serialize the manifest and every source into one ZIP archive. The
    /// manifest is entry [`ASSET_MANIFEST_FILE`], then every content path in
    /// the same order the manifest lists them, so the archive is byte-stable
    /// for the same sources.
    pub fn pack(
        &self,
        manifest: &ContentManifest,
        archive: &Path,
    ) -> Result<Vec<u8>, ContentError> {
        let mut writer = PackageWriter::new();
        writer.add(
            ASSET_MANIFEST_FILE,
            manifest.to_ron()?.into_bytes(),
            Compression::Deflated,
        )?;
        for entry in manifest.entries() {
            let bytes = self.read_required(&entry.path)?;
            let compression = self
                .compression
                .unwrap_or_else(|| default_compression(entry.kind));
            writer.add(entry.path.entry_name(), bytes, compression)?;
        }
        writer.finish(archive)
    }

    fn sources(&self) -> Result<Vec<AssetPath>, ContentError> {
        let mut files = Vec::new();
        let mut directories = vec![self.source_root.clone()];
        let mut seen = BTreeSet::new();
        while let Some(directory) = directories.pop() {
            if !seen.insert(directory.clone()) {
                continue;
            }
            let mut children = self
                .storage
                .read_dir(&directory)
                .map_err(|error| io_error(&directory, "read content directory", error))?;
            children.sort();
            for child in children {
                let Some(name) = child.file_name().and_then(|name| name.to_str()) else {
                    return Err(ContentError::format(
                        &child,
                        "content file name is not valid UTF-8",
                    ));
                };
                if is_output(name) || name.starts_with('.') {
                    continue;
                }
                if self.storage.is_dir(&child) {
                    directories.push(child);
                } else if self.storage.is_file(&child) {
                    let path = AssetPath::from_absolute(&self.source_root, &child)?;
                    if !self
                        .excludes
                        .iter()
                        .any(|exclude| path.starts_with_dir(exclude))
                    {
                        files.push(path);
                    }
                } else {
                    return Err(ContentError::new(
                        &child,
                        "content entry is neither a file nor a directory",
                    ));
                }
            }
        }
        files.sort();
        Ok(files)
    }

    fn read_required(&self, path: &AssetPath) -> Result<Vec<u8>, ContentError> {
        let file = path.dev_path(&self.source_root);
        self.storage
            .read(&file)
            .map_err(|error| io_error(&file, "read content source", error))?
            .ok_or_else(|| {
                ContentError::missing(path.as_str(), format!("source {}", file.display()))
            })
    }

    fn entry(&self, path: AssetPath, bytes: &[u8]) -> Result<ContentEntry, ContentError> {
        let (kind, image, audio) = self.classify(&path, bytes)?;
        Ok(ContentEntry {
            path,
            kind,
            size: bytes.len() as u64,
            hash: content_hash(bytes),
            image,
            audio,
        })
    }

    /// Classify and measure one source. Image and audio sources are measured
    /// directly (cheap header reads); every other kind comes from the importer
    /// registry's extension and magic tables, falling back to binary — which
    /// accepts any bytes — rather than dropping the file.
    fn classify(
        &self,
        path: &AssetPath,
        bytes: &[u8],
    ) -> Result<(AssetKind, Option<ImageInfo>, Option<AudioInfo>), ContentError> {
        let kind = self.kind_of(path, bytes);
        match kind {
            AssetKind::Image => Ok((kind, Some(image_info(path, bytes)?), None)),
            AssetKind::Audio => Ok((kind, None, Some(audio_info(path, bytes)?))),
            kind => Ok((kind, None, None)),
        }
    }

    /// The declared kind of one source: the extension tables this crate
    /// measures itself, then the importer registry's extension and magic.
    fn kind_of(&self, path: &AssetPath, bytes: &[u8]) -> AssetKind {
        if let Some(extension) = path.extension() {
            if image_extensions().contains(&extension.as_str()) {
                return AssetKind::Image;
            }
            if AudioImporter.extensions().contains(&extension.as_str()) {
                return AssetKind::Audio;
            }
        }
        self.registry
            .by_extension(Path::new(path.as_str()))
            .or_else(|| self.registry.by_magic(bytes))
            .map(|importer| importer.kind())
            .unwrap_or(AssetKind::Binary)
    }
}

/// Scan `source_root` and write `<source_root>/assets.ron` and
/// `<source_root>/assets.pack`: the whole build step, with the outputs beside
/// the sources so one directory serves both a development run and what is
/// shipped. Use [`AssetPipeline`] to write them somewhere else.
pub fn build_assets(source_root: impl AsRef<Path>) -> Result<AssetBuild, ContentError> {
    let root = source_root.as_ref().to_path_buf();
    AssetPipeline::new(root.clone())?.build(root)
}

/// Extensions this pipeline measures as images: the image importer's own list
/// plus the texture containers the crate parses itself.
fn image_extensions() -> &'static [&'static str] {
    static EXTENSIONS: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    EXTENSIONS.get_or_init(|| {
        let mut extensions: Vec<&'static str> = ImageImporter.extensions().to_vec();
        extensions.extend(TEXTURE_EXTENSIONS);
        extensions
    })
}

fn is_output(name: &str) -> bool {
    name == ASSET_MANIFEST_FILE || name == ASSET_PACK_FILE
}

/// Already-compressed containers are stored; everything else is deflated.
fn default_compression(kind: AssetKind) -> Compression {
    match kind {
        AssetKind::Image | AssetKind::Audio => Compression::Stored,
        _ => Compression::Deflated,
    }
}

fn image_info(path: &AssetPath, bytes: &[u8]) -> Result<ImageInfo, ContentError> {
    let source = Path::new(path.as_str());
    let is_dds = bytes.len() >= 4 && codec::read_u32_le(bytes, 0, source)? == DDS_MAGIC;
    if bytes.starts_with(&KTX2_IDENTIFIER) || bytes.starts_with(&codec::KTX1_IDENTIFIER) || is_dds {
        // Refuses KTX 1 with its typed unsupported error, and never
        // approximates a container it cannot parse.
        let info = codec::parse_container(bytes, source)?;
        let container = if bytes.starts_with(&KTX2_IDENTIFIER) {
            "ktx2"
        } else if is_dds {
            "dds"
        } else {
            "ktx1"
        };
        return Ok(ImageInfo {
            width: info.width,
            height: info.height,
            texture: Some(TextureMeta {
                container: container.to_string(),
                format: info.format.as_str().to_string(),
                mip_levels: info.mip_levels,
            }),
        });
    }
    let (width, height) = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|error| ContentError::format(source, format!("read image header: {error}")))?
        .into_dimensions()
        .map_err(|error| ContentError::format(source, format!("read image dimensions: {error}")))?;
    Ok(ImageInfo {
        width,
        height,
        texture: None,
    })
}

fn audio_info(path: &AssetPath, bytes: &[u8]) -> Result<AudioInfo, ContentError> {
    let source = Path::new(path.as_str());
    let stream = MediaSourceStream::new(Box::new(Cursor::new(bytes.to_vec())), Default::default());
    let format = symphonia::default::get_probe()
        .probe(
            &Hint::new(),
            stream,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|error| ContentError::format(source, format!("probe audio: {error}")))?;
    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| ContentError::format(source, "audio has no audio track"))?;
    let params = track
        .codec_params
        .clone()
        .ok_or_else(|| ContentError::format(source, "audio track has no codec parameters"))?;
    let audio = params
        .audio()
        .ok_or_else(|| ContentError::format(source, "audio track is not an audio track"))?;
    Ok(AudioInfo {
        sample_rate: audio.sample_rate,
        channels: audio
            .channels
            .as_ref()
            .map(|channels| channels.count() as u32),
    })
}

fn write_atomic<S: Storage>(storage: &S, path: &Path, bytes: &[u8]) -> Result<(), ContentError> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| ContentError::new(path, "output path has no file name"))?;
    let temporary = path.with_file_name(format!(".{file_name}.tmp"));
    if let Some(parent) = path.parent() {
        storage
            .create_dir_all(parent)
            .map_err(|error| io_error(parent, "create output directory", error))?;
    }
    storage
        .write(&temporary, bytes)
        .map_err(|error| io_error(&temporary, "write output", error))?;
    if let Err(error) = storage.rename(&temporary, path) {
        let _ = storage.remove_file(&temporary);
        return Err(io_error(path, "replace output", error));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_utils::storage::MemoryStorage;
    use repame_atlas::AtlasDesc;

    use crate::package::Package;

    fn tiny_png(width: u32, height: u32) -> Vec<u8> {
        let image = image::RgbaImage::from_pixel(width, height, image::Rgba([10, 20, 30, 255]));
        let mut bytes = Vec::new();
        image
            .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
            .expect("encode png");
        bytes
    }

    /// A minimal 16-bit mono PCM WAV with 16 silent samples.
    fn tiny_wav() -> Vec<u8> {
        let data = vec![0u8; 32];
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
        bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
        bytes.extend_from_slice(&8000u32.to_le_bytes());
        bytes.extend_from_slice(&16000u32.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&data);
        bytes
    }

    fn game_tree() -> MemoryStorage {
        let storage = MemoryStorage::new();
        storage.create_dir_all(Path::new("game/art/ui")).unwrap();
        storage.write(Path::new("game/art/ui/hero.png"), &tiny_png(4, 3)).unwrap();
        storage.write(Path::new("game/art/tile.png"), &tiny_png(16, 8)).unwrap();
        storage.write(Path::new("game/art/beep.wav"), &tiny_wav()).unwrap();
        storage.write(Path::new("game/art/notes.txt"), b"hello").unwrap();
        storage
    }

    #[test]
    fn manifest_is_byte_identical_across_two_runs() {
        let storage = game_tree();
        let request = AtlasPlanRequest::all_images(AtlasDesc {
            size: 64,
            max_pages: 2,
            padding: 1,
        })
        .expect("request");
        // The outputs land inside the scanned tree: a second run must skip them
        // and produce the same bytes.
        let first = AssetPipeline::with_storage("game/art", storage.clone())
            .expect("pipeline")
            .name("game")
            .atlas(request.clone())
            .build("game/art")
            .expect("first build");
        assert_eq!(first.manifest_path, PathBuf::from("game/art/assets.ron"));
        assert_eq!(first.archive_path, PathBuf::from("game/art/assets.pack"));
        let first_manifest = storage
            .read(Path::new("game/art/assets.ron"))
            .unwrap()
            .expect("manifest");
        let first_pack = storage
            .read(Path::new("game/art/assets.pack"))
            .unwrap()
            .expect("pack");

        let second = AssetPipeline::with_storage("game/art", storage.clone())
            .expect("pipeline")
            .name("game")
            .atlas(request)
            .build("game/art")
            .expect("second build");
        let second_manifest = storage
            .read(Path::new("game/art/assets.ron"))
            .unwrap()
            .expect("manifest");
        let second_pack = storage
            .read(Path::new("game/art/assets.pack"))
            .unwrap()
            .expect("pack");

        assert_eq!(first.manifest, second.manifest);
        assert_eq!(first_manifest, second_manifest, "manifest bytes differ");
        assert_eq!(first_pack, second_pack, "archive bytes differ");

        let paths: Vec<&str> = first
            .manifest
            .entries()
            .iter()
            .map(|entry| entry.path.as_str())
            .collect();
        assert_eq!(paths, vec!["beep.wav", "notes.txt", "tile.png", "ui/hero.png"]);
        let hero = AssetPath::parse("ui/hero.png").expect("path");
        let hero_entry = first.manifest.entry(&hero).expect("hero");
        assert_eq!(hero_entry.kind, AssetKind::Image);
        assert_eq!(hero_entry.image.as_ref().expect("dimensions").dimensions(), (4, 3));
        let beep = first.manifest.entry(&AssetPath::parse("beep.wav").expect("path")).expect("beep");
        assert_eq!(beep.kind, AssetKind::Audio);
        assert_eq!(
            beep.audio,
            Some(AudioInfo {
                sample_rate: Some(8000),
                channels: Some(1)
            })
        );
        let notes = first.manifest.entry(&AssetPath::parse("notes.txt").expect("path")).expect("notes");
        assert_eq!(notes.kind, AssetKind::Text);
        let summed: u64 = first.manifest.entries().iter().map(|entry| entry.size).sum();
        assert_eq!(first.manifest.total_size(), summed);
        assert_eq!(first.atlas_pages(), Some(1));
        assert!(first.manifest.atlas().is_some(), "atlas plan is in the manifest");
    }

    #[test]
    fn atlas_plan_is_stable_and_replays_through_repame_atlas() {
        let storage = game_tree();
        let desc = AtlasDesc {
            size: 64,
            max_pages: 2,
            padding: 1,
        };
        let manifest = AssetPipeline::with_storage("game/art", storage.clone())
            .expect("pipeline")
            .atlas(AtlasPlanRequest::all_images(desc).expect("request"))
            .scan()
            .expect("scan");
        let plan = manifest.atlas().expect("plan");
        assert_eq!(plan.len(), 2, "both plain images pack; the wav and text do not");
        assert_eq!(plan.pages, 1);
        assert!(plan.placement(&AssetPath::parse("tile.png").expect("path")).is_some());

        // Replaying the plan's own order through repame-atlas reproduces it.
        let mut atlas = repame_atlas::Atlas::new(desc);
        for sprite in &plan.sprites {
            let uv = atlas
                .alloc(repame_atlas::AtlasId(sprite.key), sprite.w, sprite.h)
                .expect("fits");
            assert_eq!(uv.page, sprite.page);
            assert_eq!([uv.min[0], uv.min[1], uv.max[0], uv.max[1]], sprite.uv);
        }
        assert_eq!(atlas.drain_writes().len(), plan.len());
    }

    #[test]
    fn every_image_importer_extension_is_measured_as_an_image() {
        let storage = MemoryStorage::new();
        storage.create_dir_all(Path::new("art")).unwrap();
        let pipeline = AssetPipeline::with_storage("art", storage.clone()).expect("pipeline");
        let extensions = image_extensions();
        for extension in ImageImporter.extensions() {
            assert!(
                extensions.contains(extension),
                "image importer extension `{extension}` is not scanned as an image"
            );
        }
        let png = AssetPath::parse("hero.png").expect("path");
        let (kind, image, _) = pipeline.classify(&png, &tiny_png(2, 2)).expect("png");
        assert_eq!(kind, AssetKind::Image);
        assert!(!image.expect("dimensions").is_texture());
        // A source with no extension is still measured, not listed as binary.
        let bare = AssetPath::parse("hero").expect("path");
        let (kind, image, _) = pipeline.classify(&bare, &tiny_png(2, 2)).expect("bare png");
        assert_eq!(kind, AssetKind::Image);
        assert_eq!(image.expect("dimensions").dimensions(), (2, 2));
    }

    #[test]
    fn a_missing_or_unreadable_source_is_a_typed_error_naming_the_path() {
        let storage = MemoryStorage::new();
        let error = match AssetPipeline::with_storage("nowhere", storage.clone()) {
            Ok(_) => panic!("a missing content root must not build a pipeline"),
            Err(error) => error,
        };
        assert!(error.is_missing(), "{error}");
        assert!(error.to_string().contains("nowhere"), "{error}");

        storage.create_dir_all(Path::new("art")).unwrap();
        storage.write(Path::new("art/broken.png"), b"not a png").unwrap();
        let error = AssetPipeline::with_storage("art", storage.clone())
            .expect("pipeline")
            .scan()
            .expect_err("unreadable image");
        assert_eq!(error.path(), Path::new("broken.png"));
        assert_eq!(error.kind(), crate::ContentErrorKind::Format);

        // An absolute source outside the content root never becomes a key.
        let error = AssetPath::from_absolute(Path::new("art"), Path::new("/elsewhere/hero.png"))
            .expect_err("outside the root");
        assert!(error.to_string().contains("outside the content root"), "{error}");
    }

    #[test]
    fn packing_selects_the_documented_compression_and_entry_names() {
        let storage = game_tree();
        let build = AssetPipeline::with_storage("game/art", storage.clone())
            .expect("pipeline")
            .build("out")
            .expect("build");
        assert_eq!(build.entries(), 4);
        assert_eq!(build.atlas_pages(), None);
        let pack = storage.read(Path::new("out/assets.pack")).unwrap().expect("pack");
        assert_eq!(build.archive_size, pack.len() as u64);
        let package = Package::read(pack, Path::new(ASSET_PACK_FILE)).expect("package");
        assert_eq!(package.len(), 5, "manifest plus four sources");
        assert!(package.entry(ASSET_MANIFEST_FILE).is_some());
        assert!(package.entry("ui/hero.png").is_some());
        let hero = package.entry("ui/hero.png").expect("hero");
        assert_eq!(hero.compression, Compression::Stored);
        let notes = package.entry("notes.txt").expect("notes");
        assert_eq!(notes.compression, Compression::Deflated);
    }
}
