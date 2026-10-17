//! Headless RON project, scene, resource and asset content foundation.
//!
//! * [`Project`] loads and validates a `project.ron` manifest with its scenes,
//!   assets and resources, and spawns scenes into a `bevy_ecs` [`World`].
//! * [`ImporterRegistry`] is the plugin-shaped import pipeline: one [`Importer`]
//!   per source kind, with provenance written to a sidecar record.
//! * [`AssetPipeline`] is the build-time asset step: it scans a source tree
//!   once and writes the generated [`ContentManifest`] plus the packed archive,
//!   so a game ships a data table and one archive instead of a hand-maintained
//!   sprite table and a hundred `include_bytes!` calls.
//! * [`AssetStore`] is the runtime side of the same rule: loose files in
//!   development, the packed archive when shipped, and both resolve an
//!   [`AssetPath`] to the same asset.
//! * [`HotReloader`] watches the content root in development, re-imports only
//!   what changed and publishes [`ArtifactSwap`]s to apply at a boundary.
//! * [`codec`] parses DDS and KTX 2 containers and decodes BC1, BC3 and BC4
//!   blocks; [`codec::stream`] opens a header and reads one mip level at a time
//!   from a file, a slice or a package entry; [`package`] reads and writes ZIP
//!   archives and PCK-like indexes.
//!
//! [`World`]: bevy_ecs::prelude::World

pub mod codec;
pub mod manifest;
pub mod package;
pub mod pipeline;
pub mod vfs;

mod error;
mod import;
mod model;
mod project;
mod reload;
mod watch;

pub const PROJECT_FORMAT: u32 = 1;
pub const PROJECT_FILE: &str = "project.ron";

pub use codec::{
    BlockCodec, BlockCodecRegistry, DecodedImage, TextureFormat, TextureInfo, TextureKind,
    TextureLevel, parse_container,
};
pub use codec::stream::{ByteRangeSource, FileRangeSource, MipReader};
pub use error::{ContentError, ContentErrorKind};
pub use import::{
    AssetKind, AudioImporter, BinaryImporter, GltfImporter, IMPORT_FORMAT, IMPORTER_VERSION,
    ImageImporter, ImportCache, ImportContext, ImportOutput, ImportRecord, ImportedAsset, Importer,
    ImporterRegistry, RonImporter, TextImporter,
};
pub use manifest::{
    ASSET_GENERATOR_VERSION, ASSET_MANIFEST_FILE, ASSET_MANIFEST_FORMAT, ASSET_PACK_FILE, AssetPath,
    AtlasPlacement, AtlasPlan, AtlasPlanRequest, AtlasSelection, AudioInfo, ContentEntry,
    ContentManifest, ImageInfo, TextureMeta,
};
pub use model::{
    AssetEntry, ProjectManifest, ResourceEntry, SceneDocument, SceneEntity, SceneEntityData,
    SceneEntry, SceneInstance, TypeRegistry,
};
pub use package::{
    Compression, Package, PackageEntry, PackageRangeSource, PackageWriter, PckDirectory, PckEntry,
    crc32,
};
pub use pipeline::{AssetBuild, AssetPipeline, build_assets};
pub use project::{AssetSource, Project, ResourceSource, content_hash};
pub use reload::{ArtifactSwap, HotReloader, ReloadOutcome};
pub use vfs::{AssetLocation, AssetMode, AssetRangeSource, AssetStore, AssetVfs, LooseAssets};
pub use watch::{AssetWatcher, FileStamp, WatchEvent, WatchEventKind, diff_snapshots};
