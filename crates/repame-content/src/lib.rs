mod error;
mod model;
mod project;

pub const PROJECT_FORMAT: u32 = 1;
pub const PROJECT_FILE: &str = "project.ron";

pub use error::ContentError;
pub use model::{
    AssetEntry, ProjectManifest, ResourceEntry, SceneDocument, SceneEntity, SceneEntityData,
    SceneEntry, SceneInstance,
};
pub use project::{AssetSource, Project, ResourceSource, content_hash};
