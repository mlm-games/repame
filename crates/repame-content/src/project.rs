use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use bevy_ecs::prelude::World;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::error::{ContentError, io_error, ron_error};
use crate::import::{ImportCache, ImportedAsset};
use crate::model::{
    AssetEntry, ProjectManifest, ResourceEntry, SceneDocument, SceneEntry, SceneInstance,
    TypeRegistry, validate_id,
};
use crate::{PROJECT_FILE, PROJECT_FORMAT};

pub fn content_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssetSource {
    pub id: String,
    pub kind: String,
    pub path: PathBuf,
    pub size: u64,
    pub fingerprint: u64,
    pub dependencies: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceSource {
    pub id: String,
    pub kind: String,
    pub path: PathBuf,
    pub size: u64,
    pub fingerprint: u64,
}

#[derive(Clone, Debug)]
pub struct Project {
    root: PathBuf,
    manifest: ProjectManifest,
    assets: BTreeMap<String, AssetEntry>,
    resources: BTreeMap<String, ResourceEntry>,
    scenes: BTreeMap<String, SceneEntry>,
    asset_paths: BTreeMap<String, PathBuf>,
    resource_paths: BTreeMap<String, PathBuf>,
    scene_paths: BTreeMap<String, PathBuf>,
    entry_scene_path: PathBuf,
}

impl Project {
    pub fn load(root: impl AsRef<Path>) -> Result<Self, ContentError> {
        let root = canonical_root(root.as_ref())?;
        let manifest_path = resolve_file(&root, PROJECT_FILE, &root, "project")?;
        let manifest = read_ron(&manifest_path)?;
        Self::from_manifest(root, manifest)
    }

    pub fn from_manifest(
        root: impl AsRef<Path>,
        manifest: ProjectManifest,
    ) -> Result<Self, ContentError> {
        let root = canonical_root(root.as_ref())?;
        if manifest.format != PROJECT_FORMAT {
            return Err(ContentError::new(
                root.join(PROJECT_FILE),
                format!(
                    "unsupported project format {}; expected {PROJECT_FORMAT}",
                    manifest.format
                ),
            ));
        }
        if manifest.name.trim().is_empty()
            || manifest
                .name
                .chars()
                .any(|character| character.is_control())
        {
            return Err(ContentError::new(
                root.join(PROJECT_FILE),
                "project name is invalid",
            ));
        }
        if manifest.entry_scene.trim().is_empty() {
            return Err(ContentError::new(
                root.join(PROJECT_FILE),
                "entry scene path is empty",
            ));
        }

        let mut assets = BTreeMap::new();
        let mut asset_paths = BTreeMap::new();
        for entry in &manifest.assets {
            validate_entry(
                &entry.id,
                &entry.kind,
                &entry.path,
                &root.join(PROJECT_FILE),
            )?;
            if assets.insert(entry.id.clone(), entry.clone()).is_some() {
                return Err(ContentError::new(
                    root.join(PROJECT_FILE),
                    format!("duplicate asset id `{}`", entry.id),
                ));
            }
            asset_paths.insert(
                entry.id.clone(),
                resolve_file(&root, &entry.path, &root.join(PROJECT_FILE), "asset")?,
            );
        }

        let mut resources = BTreeMap::new();
        let mut resource_paths = BTreeMap::new();
        for entry in &manifest.resources {
            validate_entry(
                &entry.id,
                &entry.kind,
                &entry.path,
                &root.join(PROJECT_FILE),
            )?;
            if resources.insert(entry.id.clone(), entry.clone()).is_some() {
                return Err(ContentError::new(
                    root.join(PROJECT_FILE),
                    format!("duplicate resource id `{}`", entry.id),
                ));
            }
            resource_paths.insert(
                entry.id.clone(),
                resolve_file(&root, &entry.path, &root.join(PROJECT_FILE), "resource")?,
            );
        }

        let mut scenes = BTreeMap::new();
        let mut scene_paths = BTreeMap::new();
        for entry in &manifest.scenes {
            validate_id("scene", &entry.id, &root.join(PROJECT_FILE))?;
            validate_relative_path(&entry.path, &root.join(PROJECT_FILE), "scene")?;
            if scenes.insert(entry.id.clone(), entry.clone()).is_some() {
                return Err(ContentError::new(
                    root.join(PROJECT_FILE),
                    format!("duplicate scene id `{}`", entry.id),
                ));
            }
            scene_paths.insert(
                entry.id.clone(),
                resolve_file(&root, &entry.path, &root.join(PROJECT_FILE), "scene")?,
            );
        }

        let entry_scene_path = resolve_file(
            &root,
            &manifest.entry_scene,
            &root.join(PROJECT_FILE),
            "entry scene",
        )?;
        if !scenes.is_empty() && !scene_paths.values().any(|path| path == &entry_scene_path) {
            return Err(ContentError::new(
                root.join(PROJECT_FILE),
                format!(
                    "entry scene `{}` is not present in the scenes list",
                    manifest.entry_scene
                ),
            ));
        }

        validate_dependency_graph(
            &dependency_map(
                manifest
                    .assets
                    .iter()
                    .map(|entry| (&entry.id, &entry.dependencies)),
            ),
            &root.join(PROJECT_FILE),
            "asset",
        )?;
        validate_dependency_graph(
            &dependency_map(
                manifest
                    .resources
                    .iter()
                    .map(|entry| (&entry.id, &entry.dependencies)),
            ),
            &root.join(PROJECT_FILE),
            "resource",
        )?;
        validate_dependency_graph(
            &dependency_map(
                manifest
                    .scenes
                    .iter()
                    .map(|entry| (&entry.id, &entry.dependencies)),
            ),
            &root.join(PROJECT_FILE),
            "scene",
        )?;

        let project = Self {
            root,
            manifest,
            assets,
            resources,
            scenes,
            asset_paths,
            resource_paths,
            scene_paths,
            entry_scene_path,
        };
        project.validate()?;
        Ok(project)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn manifest(&self) -> &ProjectManifest {
        &self.manifest
    }

    pub fn project_path(&self) -> PathBuf {
        self.root.join(PROJECT_FILE)
    }

    pub fn entry_scene_path(&self) -> &Path {
        &self.entry_scene_path
    }

    pub fn scene_count(&self) -> usize {
        let mut paths = BTreeSet::new();
        paths.insert(self.entry_scene_path.clone());
        paths.extend(self.scene_paths.values().cloned());
        paths.len()
    }

    pub fn asset_count(&self) -> usize {
        self.assets.len()
    }

    pub fn resource_count(&self) -> usize {
        self.resources.len()
    }

    pub fn asset(&self, id: &str) -> Option<&AssetEntry> {
        self.assets.get(id)
    }

    pub fn resource(&self, id: &str) -> Option<&ResourceEntry> {
        self.resources.get(id)
    }

    pub fn scene(&self, id: &str) -> Option<&SceneEntry> {
        self.scenes.get(id)
    }

    pub fn asset_path(&self, id: &str) -> Option<&Path> {
        self.asset_paths.get(id).map(PathBuf::as_path)
    }

    pub fn resource_path(&self, id: &str) -> Option<&Path> {
        self.resource_paths.get(id).map(PathBuf::as_path)
    }

    pub fn scene_path(&self, id: &str) -> Option<&Path> {
        self.scene_paths.get(id).map(PathBuf::as_path)
    }

    pub fn entry_scene(&self) -> Result<SceneDocument, ContentError> {
        self.load_scene_path(&self.entry_scene_path)
    }

    pub fn entry_scene_hash(&self) -> Result<u64, ContentError> {
        self.entry_scene()?.content_hash()
    }

    pub fn scene_by_id(&self, id: &str) -> Result<SceneDocument, ContentError> {
        let path = self.scene_paths.get(id).ok_or_else(|| {
            ContentError::new(self.project_path(), format!("unknown scene id `{id}`"))
        })?;
        self.load_scene_path(path)
    }

    pub fn scene_hash(&self, id: &str) -> Result<u64, ContentError> {
        self.scene_by_id(id)?.content_hash()
    }

    pub fn scene_by_path(&self, path: &str) -> Result<SceneDocument, ContentError> {
        let path = resolve_file(&self.root, path, &self.project_path(), "scene")?;
        self.load_scene_path(&path)
    }

    pub fn spawn_entry_scene(&self, world: &mut World) -> Result<SceneInstance, ContentError> {
        let scene = self.entry_scene()?;
        scene.spawn(world)
    }

    pub fn spawn_scene(&self, id: &str, world: &mut World) -> Result<SceneInstance, ContentError> {
        let scene = self.scene_by_id(id)?;
        scene.spawn(world)
    }

    pub fn read_asset(&self, id: &str) -> Result<Vec<u8>, ContentError> {
        let path = self.asset_paths.get(id).ok_or_else(|| {
            ContentError::new(self.project_path(), format!("unknown asset id `{id}`"))
        })?;
        fs::read(path).map_err(|error| io_error(path, "read asset", error))
    }

    pub fn read_resource(&self, id: &str) -> Result<Vec<u8>, ContentError> {
        let path = self.resource_paths.get(id).ok_or_else(|| {
            ContentError::new(self.project_path(), format!("unknown resource id `{id}`"))
        })?;
        fs::read(path).map_err(|error| io_error(path, "read resource", error))
    }

    pub fn read_resource_ron<T: DeserializeOwned>(&self, id: &str) -> Result<T, ContentError> {
        let path = self.resource_paths.get(id).ok_or_else(|| {
            ContentError::new(self.project_path(), format!("unknown resource id `{id}`"))
        })?;
        read_ron(path)
    }

    pub fn asset_sources(&self) -> Result<Vec<AssetSource>, ContentError> {
        self.assets
            .values()
            .map(|entry| {
                let path = self.asset_paths[&entry.id].clone();
                let size = fs::metadata(&path)
                    .map_err(|error| io_error(&path, "read asset metadata", error))?
                    .len();
                Ok(AssetSource {
                    id: entry.id.clone(),
                    kind: entry.kind.clone(),
                    fingerprint: file_fingerprint(&path)?,
                    path,
                    size,
                    dependencies: entry.dependencies.clone(),
                })
            })
            .collect()
    }

    pub fn resource_sources(&self) -> Result<Vec<ResourceSource>, ContentError> {
        self.resources
            .values()
            .map(|entry| {
                let path = self.resource_paths[&entry.id].clone();
                let size = fs::metadata(&path)
                    .map_err(|error| io_error(&path, "read resource metadata", error))?
                    .len();
                Ok(ResourceSource {
                    id: entry.id.clone(),
                    kind: entry.kind.clone(),
                    fingerprint: file_fingerprint(&path)?,
                    path,
                    size,
                })
            })
            .collect()
    }

    pub fn asset_import_order(&self) -> Vec<String> {
        dependency_order(&dependency_map(
            self.assets
                .values()
                .map(|entry| (&entry.id, &entry.dependencies)),
        ))
    }

    pub fn resource_import_order(&self) -> Vec<String> {
        dependency_order(&dependency_map(
            self.resources
                .values()
                .map(|entry| (&entry.id, &entry.dependencies)),
        ))
    }

    pub fn scene_load_order(&self) -> Vec<String> {
        dependency_order(&dependency_map(
            self.scenes
                .values()
                .map(|entry| (&entry.id, &entry.dependencies)),
        ))
    }

    pub fn import_cache(&self) -> Result<ImportCache, ContentError> {
        ImportCache::new(self.root.join(".repame").join("imports"))
    }

    pub fn import_assets(&self) -> Result<Vec<ImportedAsset>, ContentError> {
        let sources = self.asset_sources()?;
        let by_id: BTreeMap<String, AssetSource> = sources
            .into_iter()
            .map(|source| (source.id.clone(), source))
            .collect();
        let cache = self.import_cache()?;
        let mut imported = Vec::with_capacity(by_id.len());
        for id in self.asset_import_order() {
            let source = &by_id[&id];
            let dependencies = source
                .dependencies
                .iter()
                .filter_map(|dependency| {
                    by_id
                        .get(dependency)
                        .map(|entry| (dependency.clone(), entry.fingerprint))
                })
                .collect();
            imported.push(cache.import(source, &dependencies)?);
        }
        Ok(imported)
    }

    pub fn validate(&self) -> Result<(), ContentError> {
        let mut scene_paths = BTreeSet::new();
        scene_paths.insert(self.entry_scene_path.clone());
        scene_paths.extend(self.scene_paths.values().cloned());
        for path in scene_paths {
            self.load_scene_path(&path)?;
        }
        Ok(())
    }

    pub fn validate_with_types(&self, types: &TypeRegistry) -> Result<(), ContentError> {
        self.validate()?;
        types.validate_resource_kinds(&self.resources, &self.project_path())?;
        let mut scene_paths = BTreeSet::new();
        scene_paths.insert(self.entry_scene_path.clone());
        scene_paths.extend(self.scene_paths.values().cloned());
        for path in scene_paths {
            let scene = self.load_scene_path(&path)?;
            types.validate_scene(&scene, &self.assets, &self.resources, &path)?;
        }
        Ok(())
    }

    pub fn write_manifest(&self) -> Result<(), ContentError> {
        let path = self.project_path();
        let text = serialize_ron(&path, &self.manifest)?;
        write_atomic(&path, text.as_bytes())
    }

    pub fn write_entry_scene(&self, scene: &SceneDocument) -> Result<(), ContentError> {
        self.write_scene_path(&self.entry_scene_path, scene)
    }

    pub fn write_scene(&self, id: &str, scene: &SceneDocument) -> Result<(), ContentError> {
        let path = self.scene_paths.get(id).ok_or_else(|| {
            ContentError::new(self.project_path(), format!("unknown scene id `{id}`"))
        })?;
        self.write_scene_path(path, scene)
    }

    fn write_scene_path(&self, path: &Path, scene: &SceneDocument) -> Result<(), ContentError> {
        scene.validate(&self.assets, &self.resources, path)?;
        let text = serialize_ron(path, scene)?;
        write_atomic(path, text.as_bytes())
    }

    fn load_scene_path(&self, path: &Path) -> Result<SceneDocument, ContentError> {
        let scene: SceneDocument = read_ron(path)?;
        scene.validate(&self.assets, &self.resources, path)?;
        Ok(scene)
    }
}

fn canonical_root(root: &Path) -> Result<PathBuf, ContentError> {
    let canonical =
        fs::canonicalize(root).map_err(|error| io_error(root, "resolve project root", error))?;
    if !canonical.is_dir() {
        return Err(ContentError::new(root, "project root is not a directory"));
    }
    Ok(canonical)
}

fn validate_entry(id: &str, kind: &str, path: &str, owner: &Path) -> Result<(), ContentError> {
    validate_id("content", id, owner)?;
    if kind.trim().is_empty() || kind.chars().any(|character| character.is_control()) {
        return Err(ContentError::new(
            owner,
            format!("content `{id}` has an invalid kind"),
        ));
    }
    validate_relative_path(path, owner, "content")
}

fn validate_relative_path(path: &str, owner: &Path, kind: &str) -> Result<(), ContentError> {
    let relative = Path::new(path);
    if path.trim().is_empty()
        || relative.is_absolute()
        || relative.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(ContentError::new(
            owner,
            format!("{kind} path `{path}` must stay inside the project"),
        ));
    }
    Ok(())
}

fn resolve_file(
    root: &Path,
    path: &str,
    owner: &Path,
    kind: &str,
) -> Result<PathBuf, ContentError> {
    validate_relative_path(path, owner, kind)?;
    let candidate = root.join(path);
    let canonical = fs::canonicalize(&candidate)
        .map_err(|error| io_error(&candidate, format!("resolve {kind}"), error))?;
    if !canonical.starts_with(root) {
        return Err(ContentError::new(
            owner,
            format!("{kind} path `{path}` escapes the project root"),
        ));
    }
    if !canonical.is_file() {
        return Err(ContentError::new(
            &canonical,
            format!("{kind} path is not a file"),
        ));
    }
    Ok(canonical)
}

fn dependency_map<'a>(
    entries: impl Iterator<Item = (&'a String, &'a Vec<String>)>,
) -> BTreeMap<String, Vec<String>> {
    entries
        .map(|(id, dependencies)| (id.clone(), dependencies.clone()))
        .collect()
}

fn dependency_order(graph: &BTreeMap<String, Vec<String>>) -> Vec<String> {
    fn visit(
        id: &str,
        graph: &BTreeMap<String, Vec<String>>,
        visiting: &mut BTreeSet<String>,
        visited: &mut BTreeSet<String>,
        order: &mut Vec<String>,
    ) {
        if visited.contains(id) || !visiting.insert(id.to_string()) {
            return;
        }
        if let Some(dependencies) = graph.get(id) {
            for dependency in dependencies {
                visit(dependency, graph, visiting, visited, order);
            }
        }
        visiting.remove(id);
        visited.insert(id.to_string());
        order.push(id.to_string());
    }

    let mut order = Vec::with_capacity(graph.len());
    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    for id in graph.keys() {
        visit(id, graph, &mut visiting, &mut visited, &mut order);
    }
    order
}

fn validate_dependency_graph(
    graph: &BTreeMap<String, Vec<String>>,
    owner: &Path,
    kind: &str,
) -> Result<(), ContentError> {
    for (id, dependencies) in graph {
        let mut seen = BTreeSet::new();
        for dependency in dependencies {
            if !seen.insert(dependency) {
                return Err(ContentError::new(
                    owner,
                    format!("{kind} `{id}` repeats dependency `{dependency}`"),
                ));
            }
            if dependency == id {
                return Err(ContentError::new(
                    owner,
                    format!("{kind} `{id}` cannot depend on itself"),
                ));
            }
            if !graph.contains_key(dependency) {
                return Err(ContentError::new(
                    owner,
                    format!("{kind} `{id}` references missing dependency `{dependency}`"),
                ));
            }
        }
    }

    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    for id in graph.keys() {
        visit_dependency(id, graph, &mut visiting, &mut visited, owner, kind)?;
    }
    Ok(())
}

fn visit_dependency(
    id: &str,
    graph: &BTreeMap<String, Vec<String>>,
    visiting: &mut BTreeSet<String>,
    visited: &mut BTreeSet<String>,
    owner: &Path,
    kind: &str,
) -> Result<(), ContentError> {
    if visited.contains(id) {
        return Ok(());
    }
    if !visiting.insert(id.to_string()) {
        return Err(ContentError::new(
            owner,
            format!("{kind} dependency graph contains a cycle at `{id}`"),
        ));
    }
    if let Some(dependencies) = graph.get(id) {
        for dependency in dependencies {
            visit_dependency(dependency, graph, visiting, visited, owner, kind)?;
        }
    }
    visiting.remove(id);
    visited.insert(id.to_string());
    Ok(())
}

fn file_fingerprint(path: &Path) -> Result<u64, ContentError> {
    let mut file = fs::File::open(path).map_err(|error| io_error(path, "open content", error))?;
    let mut hash = 0xcbf29ce484222325;
    let mut buffer = [0u8; 8192];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| io_error(path, "read content", error))?;
        if read == 0 {
            break;
        }
        for byte in &buffer[..read] {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    Ok(hash)
}

fn read_ron<T: DeserializeOwned>(path: &Path) -> Result<T, ContentError> {
    let text = fs::read_to_string(path).map_err(|error| io_error(path, "read RON", error))?;
    ron::from_str(&text).map_err(|error| ron_error(path, error))
}

fn serialize_ron<T: Serialize>(path: &Path, value: &T) -> Result<String, ContentError> {
    let mut text = ron::ser::to_string_pretty(value, Default::default())
        .map_err(|error| ron_error(path, error))?;
    text.push('\n');
    Ok(text)
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), ContentError> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| ContentError::new(path, "content path has no file name"))?;
    let temporary = path.with_file_name(format!(".{file_name}.tmp"));
    if let Err(error) = fs::write(&temporary, bytes) {
        let _ = fs::remove_file(&temporary);
        return Err(io_error(&temporary, "write temporary content", error));
    }
    if let Err(error) = fs::rename(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(io_error(path, "replace content", error));
    }
    Ok(())
}
