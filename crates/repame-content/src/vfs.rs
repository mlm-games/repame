//! Runtime content resolution: loose files in development, one archive when
//! packaged.
//!
//! [`AssetStore`] is the runtime entry point. It reads a content root the same
//! way in both modes: `<root>/assets.pack` present means the archive is served
//! ([`AssetStore::Packed`]), otherwise loose files under `<root>` are
//! ([`AssetStore::Dev`]). A web build fetches (or embeds) the one archive and
//! hands its bytes to [`AssetVfs::from_bytes`]; a desktop build either reads
//! the archive from disk or runs loose. The game calls `read`, `contains` or
//! `open_texture` and never learns which mode it is in except through
//! [`AssetStore::mode`].
//!
//! Both modes resolve a content path by the same key: [`AssetPath::dev_path`]
//! for loose files, [`AssetPath::entry_name`] for archive entries. Neither mode
//! ever needs an absolute source path — no `GODOT_ROOT`, no per-file
//! `include_bytes!` table.
//!
//! Textures stream per mip out of the archive: [`AssetStore::open_texture`]
//! opens a [`MipReader`] over an [`AssetRangeSource`], so a level read touches
//! exactly that level's stored bytes whether they live in a file or in a ZIP
//! entry. Unsupported containers stay typed
//! [`Unsupported`](crate::ContentErrorKind::Unsupported), a missing asset is
//! [`Missing`](crate::ContentErrorKind::Missing) naming its content path, and
//! neither is a short read or a silent placeholder.

use std::fmt;
use std::path::{Path, PathBuf};

use game_utils::storage::{FsStorage, Storage};

use crate::codec::stream::{ByteRangeSource, FileRangeSource, MipReader};
use crate::error::{ContentError, io_error};
use crate::manifest::{ASSET_MANIFEST_FILE, ASSET_PACK_FILE, AssetPath, ContentManifest};
use crate::package::{Package, PackageRangeSource};

/// How a content root is served.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssetMode {
    /// Loose files under the content root: the development layout.
    Dev,
    /// One packed archive: the shipped layout.
    Packed,
}

impl AssetMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dev => "dev",
            Self::Packed => "packed",
        }
    }
}

/// Where one content path resolves to in the mode that is running.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AssetLocation {
    /// The loose file the path names.
    File(PathBuf),
    /// The archive entry the path names.
    Entry { archive: PathBuf, name: String },
}

impl fmt::Display for AssetLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::File(path) => write!(f, "{}", path.display()),
            Self::Entry { archive, name } => write!(f, "{}!{name}", archive.display()),
        }
    }
}

/// A random-access byte source inside a content store: a loose file or an
/// archive entry, behind the one [`ByteRangeSource`] the per-mip reader uses.
pub enum AssetRangeSource<'a> {
    File(FileRangeSource),
    Packed(PackageRangeSource<'a>),
}

impl ByteRangeSource for AssetRangeSource<'_> {
    fn byte_len(&self) -> Result<u64, ContentError> {
        match self {
            Self::File(source) => source.byte_len(),
            Self::Packed(source) => source.byte_len(),
        }
    }

    fn read_range(&self, offset: u64, length: usize) -> Result<Vec<u8>, ContentError> {
        match self {
            Self::File(source) => source.read_range(offset, length),
            Self::Packed(source) => source.read_range(offset, length),
        }
    }
}

/// Loose content under a root directory, with the manifest that describes it.
#[derive(Debug)]
pub struct LooseAssets<S: Storage = FsStorage> {
    root: PathBuf,
    storage: S,
    manifest: ContentManifest,
}

impl LooseAssets<FsStorage> {
    /// Open `<root>/assets.ron` from the real filesystem.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, ContentError> {
        Self::open_with_storage(root, FsStorage)
    }
}

impl<S: Storage> LooseAssets<S> {
    /// Open the manifest under `root` and serve the files it names. A missing
    /// or inconsistent manifest is a typed error naming it, never an empty
    /// store.
    pub fn open_with_storage(
        root: impl Into<PathBuf>,
        storage: S,
    ) -> Result<Self, ContentError> {
        let root = root.into();
        let manifest_path = root.join(ASSET_MANIFEST_FILE);
        Self::open_with_manifest(root, manifest_path, storage)
    }

    /// Serve loose content under `root` from a manifest read at `manifest`.
    ///
    /// For a game whose sources live in an external or read-only checkout: the
    /// manifest and archive are generated elsewhere, and the content paths stay
    /// relative to `root`, exactly as in a packaged build.
    pub fn open_with_manifest(
        root: impl Into<PathBuf>,
        manifest: impl AsRef<Path>,
        storage: S,
    ) -> Result<Self, ContentError> {
        let root = root.into();
        let manifest_path = manifest.as_ref();
        let bytes = storage
            .read(manifest_path)
            .map_err(|error| io_error(manifest_path, "read asset manifest", error))?
            .ok_or_else(|| {
                ContentError::missing(
                    manifest_path,
                    format!(
                        "asset manifest {} (run the asset pipeline to write it)",
                        manifest_path.display()
                    ),
                )
            })?;
        let text = std::str::from_utf8(&bytes).map_err(|error| {
            ContentError::format(manifest_path, format!("asset manifest is not UTF-8: {error}"))
        })?;
        let manifest = ContentManifest::from_ron(text, manifest_path)?;
        Ok(Self {
            root,
            storage,
            manifest,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn storage(&self) -> &S {
        &self.storage
    }

    pub fn manifest(&self) -> &ContentManifest {
        &self.manifest
    }

    /// True when the file this path names exists. The manifest is the table of
    /// what the build scanned; a file that is not in it is still readable in
    /// development, so a hot-reloaded source never lags a generated table.
    pub fn contains(&self, path: &AssetPath) -> bool {
        self.storage.is_file(&path.dev_path(&self.root))
    }

    pub fn location(&self, path: &AssetPath) -> AssetLocation {
        AssetLocation::File(path.dev_path(&self.root))
    }

    pub fn read(&self, path: &AssetPath) -> Result<Vec<u8>, ContentError> {
        let file = path.dev_path(&self.root);
        self.storage
            .read(&file)
            .map_err(|error| io_error(&file, "read asset", error))?
            .ok_or_else(|| ContentError::missing(path.as_str(), "asset"))
    }

    /// Open a random-access source over the loose file.
    pub fn range_source(&self, path: &AssetPath) -> Result<AssetRangeSource<'static>, ContentError> {
        let file = path.dev_path(&self.root);
        if !self.storage.is_file(&file) {
            return Err(ContentError::missing(path.as_str(), "asset"));
        }
        Ok(AssetRangeSource::File(FileRangeSource::open(&file)?))
    }

    /// Open the texture this path names and read single mip levels from it.
    pub fn open_texture(
        &self,
        path: &AssetPath,
    ) -> Result<MipReader<AssetRangeSource<'static>>, ContentError> {
        MipReader::open(self.range_source(path)?, Path::new(path.as_str()))
    }
}

/// A packed content archive with the manifest it carries.
#[derive(Debug)]
pub struct AssetVfs {
    package: Package,
    source: PathBuf,
    manifest: ContentManifest,
}

impl AssetVfs {
    /// Open a packed archive from the real filesystem.
    pub fn open(archive: impl AsRef<Path>) -> Result<Self, ContentError> {
        Self::open_with_storage(&FsStorage, archive)
    }

    /// Open a packed archive through `storage`.
    pub fn open_with_storage<S: Storage>(
        storage: &S,
        archive: impl AsRef<Path>,
    ) -> Result<Self, ContentError> {
        let archive = archive.as_ref();
        let bytes = storage
            .read(archive)
            .map_err(|error| io_error(archive, "read asset archive", error))?
            .ok_or_else(|| ContentError::missing(archive, "asset archive"))?;
        Self::from_bytes(bytes, archive)
    }

    /// Open an archive already in memory — how a web build serves the one file
    /// it fetched. `source` only names the archive in errors.
    pub fn from_bytes(bytes: Vec<u8>, source: impl Into<PathBuf>) -> Result<Self, ContentError> {
        let source = source.into();
        let package = Package::read(bytes, &source)?;
        if package.entry(ASSET_MANIFEST_FILE).is_none() {
            return Err(ContentError::missing(
                ASSET_MANIFEST_FILE,
                format!("asset manifest entry in {}", source.display()),
            ));
        }
        let manifest_bytes = package.read_file(ASSET_MANIFEST_FILE, &source)?;
        let text = std::str::from_utf8(&manifest_bytes).map_err(|error| {
            ContentError::format(&source, format!("asset manifest is not UTF-8: {error}"))
        })?;
        let manifest = ContentManifest::from_ron(text, &source)?;
        // Every asset the manifest promises must be in the archive: a build
        // that packed a stale table is caught at open, naming the asset.
        for entry in manifest.entries() {
            if package.entry(entry.path.entry_name()).is_none() {
                return Err(ContentError::missing(
                    entry.path.as_str(),
                    format!("asset in {}", source.display()),
                ));
            }
        }
        Ok(Self {
            package,
            source,
            manifest,
        })
    }

    /// The archive this store reads from, as named in errors.
    pub fn archive_path(&self) -> &Path {
        &self.source
    }

    /// The indexed archive, for callers that need the raw entry table.
    pub fn package(&self) -> &Package {
        &self.package
    }

    pub fn manifest(&self) -> &ContentManifest {
        &self.manifest
    }

    /// Archive entry count, including the manifest entry.
    pub fn len(&self) -> usize {
        self.package.len()
    }

    pub fn is_empty(&self) -> bool {
        self.package.is_empty()
    }

    pub fn contains(&self, path: &AssetPath) -> bool {
        self.package.entry(path.entry_name()).is_some()
    }

    pub fn location(&self, path: &AssetPath) -> AssetLocation {
        AssetLocation::Entry {
            archive: self.source.clone(),
            name: path.entry_name().to_string(),
        }
    }

    /// Extract one asset, checking the entry's size and CRC-32.
    pub fn read(&self, path: &AssetPath) -> Result<Vec<u8>, ContentError> {
        if self.package.entry(path.entry_name()).is_none() {
            return Err(ContentError::missing(
                path.as_str(),
                format!("asset in {}", self.source.display()),
            ));
        }
        self.package.read_file(path.entry_name(), &self.source)
    }

    /// Read one asset as UTF-8 text.
    pub fn read_str(&self, path: &AssetPath) -> Result<String, ContentError> {
        let bytes = self.read(path)?;
        String::from_utf8(bytes).map_err(|error| {
            ContentError::format(path.as_str(), format!("asset is not UTF-8: {error}"))
        })
    }

    /// Open a random-access source over one archive entry, so a caller can read
    /// a single mip level without extracting the whole entry.
    pub fn range_source<'a>(
        &'a self,
        path: &AssetPath,
    ) -> Result<AssetRangeSource<'a>, ContentError> {
        if self.package.entry(path.entry_name()).is_none() {
            return Err(ContentError::missing(
                path.as_str(),
                format!("asset in {}", self.source.display()),
            ));
        }
        Ok(AssetRangeSource::Packed(
            self.package.range_source(path.entry_name(), &self.source)?,
        ))
    }

    /// Open a texture out of the archive and read single mip levels from it.
    /// Nothing else in the archive is read.
    pub fn open_texture(
        &self,
        path: &AssetPath,
    ) -> Result<MipReader<AssetRangeSource<'_>>, ContentError> {
        MipReader::open(self.range_source(path)?, Path::new(path.as_str()))
    }
}

/// The runtime content store: loose files or a packed archive, chosen by
/// whether the content root holds [`ASSET_PACK_FILE`].
#[derive(Debug)]
pub enum AssetStore<S: Storage = FsStorage> {
    Dev(LooseAssets<S>),
    Packed(AssetVfs),
}

impl AssetStore<FsStorage> {
    /// Serve `root`: the packed archive when `<root>/assets.pack` exists — a
    /// shipped build — and loose files otherwise. The manifest is read from
    /// the archive in the first case and from `<root>/assets.ron` in the
    /// second, so a game passes the same directory in both modes.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, ContentError> {
        Self::open_with_storage(root, FsStorage)
    }
}

impl<S: Storage> AssetStore<S> {
    pub fn open_with_storage(
        root: impl Into<PathBuf>,
        storage: S,
    ) -> Result<Self, ContentError> {
        let root = root.into();
        let archive = root.join(ASSET_PACK_FILE);
        if storage.is_file(&archive) {
            Ok(Self::Packed(AssetVfs::open_with_storage(&storage, &archive)?))
        } else {
            Ok(Self::Dev(LooseAssets::open_with_storage(root, storage)?))
        }
    }

    /// Serve an already-opened loose store.
    pub fn dev(assets: LooseAssets<S>) -> Self {
        Self::Dev(assets)
    }

    /// Serve an already-opened archive, e.g. one whose bytes were fetched.
    pub fn packed(vfs: AssetVfs) -> Self {
        Self::Packed(vfs)
    }

    pub fn mode(&self) -> AssetMode {
        match self {
            Self::Dev(_) => AssetMode::Dev,
            Self::Packed(_) => AssetMode::Packed,
        }
    }

    pub fn dev_store(&self) -> Option<&LooseAssets<S>> {
        match self {
            Self::Dev(assets) => Some(assets),
            Self::Packed(_) => None,
        }
    }

    pub fn packed_store(&self) -> Option<&AssetVfs> {
        match self {
            Self::Packed(vfs) => Some(vfs),
            Self::Dev(_) => None,
        }
    }

    pub fn manifest(&self) -> &ContentManifest {
        match self {
            Self::Dev(assets) => assets.manifest(),
            Self::Packed(vfs) => vfs.manifest(),
        }
    }

    pub fn contains(&self, path: &AssetPath) -> bool {
        match self {
            Self::Dev(assets) => assets.contains(path),
            Self::Packed(vfs) => vfs.contains(path),
        }
    }

    pub fn location(&self, path: &AssetPath) -> AssetLocation {
        match self {
            Self::Dev(assets) => assets.location(path),
            Self::Packed(vfs) => vfs.location(path),
        }
    }

    pub fn read(&self, path: &AssetPath) -> Result<Vec<u8>, ContentError> {
        match self {
            Self::Dev(assets) => assets.read(path),
            Self::Packed(vfs) => vfs.read(path),
        }
    }

    pub fn range_source(&self, path: &AssetPath) -> Result<AssetRangeSource<'_>, ContentError> {
        match self {
            Self::Dev(assets) => assets.range_source(path),
            Self::Packed(vfs) => vfs.range_source(path),
        }
    }

    /// Open a texture from whichever store is running and read single mip
    /// levels from it.
    pub fn open_texture(
        &self,
        path: &AssetPath,
    ) -> Result<MipReader<AssetRangeSource<'_>>, ContentError> {
        MipReader::open(self.range_source(path)?, Path::new(path.as_str()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_utils::storage::MemoryStorage;

    use crate::codec::BlockCodecRegistry;
    use crate::codec::DDS_MAGIC;
    use crate::manifest::{AtlasPlanRequest, ContentManifest};
    use crate::package::{Compression, PackageWriter};
    use crate::pipeline::AssetPipeline;

    fn tiny_png(width: u32, height: u32) -> Vec<u8> {
        let image = image::RgbaImage::from_pixel(width, height, image::Rgba([1, 2, 3, 255]));
        let mut bytes = Vec::new();
        image
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .expect("encode png");
        bytes
    }

    /// A two-level BC1 DDS: 8x8, then 4x4, payload at byte 128.
    fn bc1_dds_mipped() -> Vec<u8> {
        let mut bytes = Vec::new();
        let push_u32 = |bytes: &mut Vec<u8>, value: u32| {
            bytes.extend_from_slice(&value.to_le_bytes());
        };
        push_u32(&mut bytes, DDS_MAGIC);
        push_u32(&mut bytes, 124);
        push_u32(&mut bytes, 0x0002_1007);
        push_u32(&mut bytes, 8);
        push_u32(&mut bytes, 8);
        push_u32(&mut bytes, 32);
        push_u32(&mut bytes, 0);
        push_u32(&mut bytes, 2);
        for _ in 0..11 {
            push_u32(&mut bytes, 0);
        }
        push_u32(&mut bytes, 32);
        push_u32(&mut bytes, 0x0000_0004);
        push_u32(&mut bytes, u32::from_le_bytes(*b"DXT1"));
        push_u32(&mut bytes, 0);
        push_u32(&mut bytes, 0);
        push_u32(&mut bytes, 0);
        push_u32(&mut bytes, 0);
        push_u32(&mut bytes, 0);
        push_u32(&mut bytes, 0x4010_0008);
        push_u32(&mut bytes, 0);
        push_u32(&mut bytes, 0);
        push_u32(&mut bytes, 0);
        assert_eq!(bytes.len(), 128);
        for level in 0..2 {
            let blocks = if level == 0 { 4 } else { 1 };
            for _ in 0..blocks {
                bytes.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF, 0x1B, 0x1B, 0x1B, 0x1B]);
            }
        }
        assert_eq!(bytes.len(), 168);
        bytes
    }

    /// A small project: one PNG, one DDS texture, one text file, and the
    /// manifest and archive the pipeline writes for them. The sources stay
    /// under `game/art` while the outputs go to `game`, which is the layout a
    /// game with an external art checkout has.
    fn packed_project() -> (MemoryStorage, PathBuf) {
        let storage = MemoryStorage::new();
        storage.create_dir_all(Path::new("game/art/tex")).unwrap();
        storage
            .write(Path::new("game/art/hero.png"), &tiny_png(4, 3))
            .unwrap();
        storage
            .write(Path::new("game/art/tex/tile.dds"), &bc1_dds_mipped())
            .unwrap();
        storage
            .write(Path::new("game/art/level.ron"), b"(name: \"one\")")
            .unwrap();
        AssetPipeline::with_storage("game/art", storage.clone())
            .expect("pipeline")
            .atlas(
                AtlasPlanRequest::all_images(repame_atlas::AtlasDesc {
                    size: 32,
                    max_pages: 1,
                    padding: 1,
                })
                .expect("request"),
            )
            .build("game")
            .expect("build");
        (storage, PathBuf::from("game"))
    }

    /// The development store of [`packed_project`]: loose sources under
    /// `game/art`, manifest generated into `game`.
    fn dev_store(storage: &MemoryStorage, root: &Path) -> LooseAssets<MemoryStorage> {
        LooseAssets::open_with_manifest(
            "game/art",
            root.join(ASSET_MANIFEST_FILE),
            storage.clone(),
        )
        .expect("dev store")
    }

    #[test]
    fn dev_and_packed_resolve_the_same_content_path_to_the_same_bytes() {
        let (storage, root) = packed_project();
        let dev = dev_store(&storage, &root);
        let pack = storage
            .read(&root.join(ASSET_PACK_FILE))
            .unwrap()
            .expect("archive");
        let packed =
            AssetVfs::from_bytes(pack, root.join(ASSET_PACK_FILE)).expect("packed store");

        assert_eq!(dev.manifest(), packed.manifest());
        for entry in dev.manifest().entries() {
            assert_eq!(
                dev.read(&entry.path).expect("dev read"),
                packed.read(&entry.path).expect("packed read"),
                "reading `{}` differs between dev and packed",
                entry.path
            );
        }
        // The manifest is a content path like any other.
        let manifest_path = AssetPath::parse(ASSET_MANIFEST_FILE).expect("path");
        assert_eq!(
            dev.read(&manifest_path).expect("dev manifest"),
            packed.read(&manifest_path).expect("packed manifest")
        );

        let hero = AssetPath::parse("hero.png").expect("path");
        match dev.location(&hero) {
            AssetLocation::File(path) => assert!(path.ends_with("game/art/hero.png")),
            location => panic!("dev store must resolve a file, got {location}"),
        }
        match packed.location(&hero) {
            AssetLocation::Entry { name, .. } => assert_eq!(name, hero.entry_name()),
            location => panic!("packed store must resolve an entry, got {location}"),
        }

        // A shipped root serves the archive; without it the same project runs
        // loose from the external source root.
        let store = AssetStore::open_with_storage(&root, storage.clone()).expect("store");
        assert_eq!(store.mode(), AssetMode::Packed);
        assert_eq!(store.read(&hero).expect("store read"), dev.read(&hero).expect("dev read"));
        storage
            .remove_file(&root.join(ASSET_PACK_FILE))
            .expect("remove archive");
        let store = AssetStore::dev(dev_store(&storage, &root));
        assert_eq!(store.mode(), AssetMode::Dev);
        assert_eq!(store.read(&hero).expect("store read"), dev.read(&hero).expect("dev read"));
    }

    #[test]
    fn a_missing_asset_is_a_typed_error_naming_the_content_path() {
        let (storage, root) = packed_project();
        let dev = dev_store(&storage, &root);
        let pack = storage
            .read(&root.join(ASSET_PACK_FILE))
            .unwrap()
            .expect("archive");
        let packed = AssetVfs::from_bytes(pack, root.join(ASSET_PACK_FILE)).expect("packed");

        let absent = AssetPath::parse("ui/missing.png").expect("path");
        for error in [
            dev.read(&absent).expect_err("dev miss"),
            packed.read(&absent).expect_err("packed miss"),
            packed
                .range_source(&absent)
                .err()
                .expect("packed range miss"),
        ] {
            assert!(error.is_missing(), "{error}");
            assert_eq!(error.path(), Path::new("ui/missing.png"));
        }
        assert!(!dev.contains(&absent));
        assert!(!packed.contains(&absent));

        // A root with neither manifest nor archive reports what is missing.
        let error = AssetStore::open_with_storage("empty", MemoryStorage::new())
            .expect_err("no content");
        assert!(error.is_missing(), "{error}");
        assert!(error.to_string().contains(ASSET_MANIFEST_FILE), "{error}");

        // An archive whose table promises an entry it does not hold is caught
        // at open, naming the asset.
        let mut manifest = ContentManifest::new("stale");
        manifest.entries.push(crate::manifest::ContentEntry {
            path: AssetPath::parse("gone.png").expect("path"),
            kind: crate::AssetKind::Image,
            size: 3,
            hash: 4,
            image: Some(crate::manifest::ImageInfo {
                width: 1,
                height: 1,
                texture: None,
            }),
            audio: None,
        });
        let mut writer = PackageWriter::new();
        writer
            .add(
                ASSET_MANIFEST_FILE,
                manifest.to_ron().expect("ron").into_bytes(),
                Compression::Stored,
            )
            .expect("add");
        let archive = writer.finish(Path::new("stale.pack")).expect("finish");
        let error = AssetVfs::from_bytes(archive, "stale.pack").expect_err("stale archive");
        assert!(error.is_missing(), "{error}");
        assert_eq!(error.path(), Path::new("gone.png"));
    }

    #[test]
    fn an_archive_round_trips_through_the_vfs_and_streams_a_single_mip() {
        let (storage, root) = packed_project();
        let pack = storage
            .read(&root.join(ASSET_PACK_FILE))
            .unwrap()
            .expect("archive");
        let vfs = AssetVfs::from_bytes(pack, root.join(ASSET_PACK_FILE)).expect("packed");

        assert_eq!(vfs.len(), 4, "manifest plus three sources");
        assert_eq!(vfs.manifest().len(), 3);
        assert!(vfs.manifest().atlas().is_some(), "the plan travels with the archive");
        assert_eq!(
            vfs.read(&AssetPath::parse("level.ron").expect("path")).expect("text"),
            b"(name: \"one\")"
        );
        assert_eq!(
            vfs.read_str(&AssetPath::parse("level.ron").expect("path")).expect("text"),
            "(name: \"one\")"
        );

        // Per-mip streaming out of the archive reads one level, and matches the
        // whole-container decode of the same texture.
        let tile = AssetPath::parse("tex/tile.dds").expect("path");
        assert!(vfs.manifest().entry(&tile).expect("entry").is_texture());
        let reader = vfs.open_texture(&tile).expect("open texture");
        assert_eq!(reader.info().mip_levels, 2);
        assert_eq!(reader.level_range(1).expect("range"), 160..168);
        let level = reader
            .read_level(&BlockCodecRegistry::default(), 1)
            .expect("level 1");
        assert_eq!((level.width, level.height), (4, 4));
        let dds = bc1_dds_mipped();
        let whole = BlockCodecRegistry::default()
            .decode_level(
                &crate::parse_container(&dds, Path::new("tex/tile.dds")).expect("container"),
                1,
                &dds,
                Path::new("tex/tile.dds"),
            )
            .expect("whole decode");
        assert_eq!(level, whole);
    }
}
