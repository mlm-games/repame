use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use bevy_ecs::prelude::{Component, Entity, World};
use ron::Value;
use serde::{Deserialize, Serialize};

use crate::PROJECT_FORMAT;
use crate::error::{ContentError, ron_error};

pub const SCENE_FORMAT: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ProjectManifest {
    pub format: u32,
    pub name: String,
    pub entry_scene: String,
    #[serde(default)]
    pub scenes: Vec<SceneEntry>,
    #[serde(default)]
    pub assets: Vec<AssetEntry>,
    #[serde(default)]
    pub resources: Vec<ResourceEntry>,
}

impl Default for ProjectManifest {
    fn default() -> Self {
        Self {
            format: PROJECT_FORMAT,
            name: "untitled".to_string(),
            entry_scene: "scenes/main.ron".to_string(),
            scenes: Vec::new(),
            assets: Vec::new(),
            resources: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SceneEntry {
    pub id: String,
    pub path: String,
    #[serde(default)]
    pub dependencies: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AssetEntry {
    pub id: String,
    pub kind: String,
    pub path: String,
    #[serde(default)]
    pub dependencies: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResourceEntry {
    pub id: String,
    pub kind: String,
    pub path: String,
    #[serde(default)]
    pub dependencies: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SceneDocument {
    pub format: u32,
    pub id: String,
    #[serde(default)]
    pub entities: Vec<SceneEntity>,
    #[serde(default)]
    pub resources: BTreeMap<String, String>,
}

impl Default for SceneDocument {
    fn default() -> Self {
        Self {
            format: SCENE_FORMAT,
            id: "main".to_string(),
            entities: Vec::new(),
            resources: BTreeMap::new(),
        }
    }
}

impl SceneDocument {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ..Self::default()
        }
    }

    pub fn entity(&self, id: &str) -> Option<&SceneEntity> {
        self.entities.iter().find(|entity| entity.id == id)
    }

    pub fn spawn(&self, world: &mut World) -> Result<SceneInstance, ContentError> {
        self.validate_structure(Path::new("<scene>"))?;
        let mut entities = BTreeMap::new();
        for entity in &self.entities {
            let ecs_entity = world
                .spawn(SceneEntityData {
                    scene_id: self.id.clone(),
                    id: entity.id.clone(),
                    name: entity.name.clone(),
                    parent: None,
                    components: entity.components.clone(),
                    assets: entity.assets.clone(),
                    resources: entity.resources.clone(),
                })
                .id();
            entities.insert(entity.id.clone(), ecs_entity);
        }
        for entity in &self.entities {
            let Some(parent_id) = &entity.parent else {
                continue;
            };
            let Some(parent) = entities.get(parent_id).copied() else {
                continue;
            };
            if let Some(mut data) = world
                .entity_mut(entities[&entity.id])
                .get_mut::<SceneEntityData>()
            {
                data.parent = Some(parent);
            }
        }
        Ok(SceneInstance {
            scene_id: self.id.clone(),
            entities,
        })
    }

    pub fn validate(
        &self,
        assets: &BTreeMap<String, AssetEntry>,
        resources: &BTreeMap<String, ResourceEntry>,
        path: &Path,
    ) -> Result<(), ContentError> {
        self.validate_structure(path)?;
        for entity in &self.entities {
            validate_references(&entity.assets, assets, "asset", &entity.id, path)?;
            validate_references(&entity.resources, resources, "resource", &entity.id, path)?;
        }
        validate_references(&self.resources, resources, "resource", "scene", path)
    }

    fn validate_structure(&self, path: &Path) -> Result<(), ContentError> {
        if self.format != SCENE_FORMAT {
            return Err(ContentError::new(
                path,
                format!(
                    "unsupported scene format {}; expected {SCENE_FORMAT}",
                    self.format
                ),
            ));
        }
        validate_id("scene", &self.id, path)?;

        let mut ids = BTreeSet::new();
        for entity in &self.entities {
            validate_id("entity", &entity.id, path)?;
            if !ids.insert(entity.id.clone()) {
                return Err(ContentError::new(
                    path,
                    format!("duplicate entity id `{}`", entity.id),
                ));
            }
            if entity.name.chars().any(|character| character.is_control()) {
                return Err(ContentError::new(
                    path,
                    format!("entity `{}` has an invalid name", entity.id),
                ));
            }
            for component_name in entity.components.keys() {
                if component_name.trim().is_empty() {
                    return Err(ContentError::new(
                        path,
                        format!("entity `{}` has an empty component name", entity.id),
                    ));
                }
                if !matches!(entity.components[component_name], Value::Map(_)) {
                    return Err(ContentError::new(
                        path,
                        format!(
                            "entity `{}` component `{component_name}` must be an object",
                            entity.id
                        ),
                    ));
                }
            }
        }
        validate_parent_graph(&self.entities, path)
    }

    pub fn to_ron(&self) -> Result<String, ContentError> {
        ron::ser::to_string_pretty(self, Default::default())
            .map_err(|error| ron_error(Path::new("<scene>"), error))
    }

    pub fn content_hash(&self) -> Result<u64, ContentError> {
        Ok(crate::project::content_hash(self.to_ron()?.as_bytes()))
    }
}

#[derive(Clone, Debug, Default)]
pub struct TypeRegistry {
    components: BTreeSet<String>,
    resource_kinds: BTreeSet<String>,
    strict: bool,
}

impl TypeRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    pub fn register_component(&mut self, name: impl Into<String>) -> Result<(), String> {
        let name = name.into();
        if name.trim().is_empty() || name.trim() != name {
            return Err(format!("invalid component type `{name}`"));
        }
        self.components.insert(name);
        Ok(())
    }

    pub fn register_resource_kind(&mut self, kind: impl Into<String>) -> Result<(), String> {
        let kind = kind.into();
        if kind.trim().is_empty() || kind.trim() != kind {
            return Err(format!("invalid resource kind `{kind}`"));
        }
        self.resource_kinds.insert(kind);
        Ok(())
    }

    pub fn validate_scene(
        &self,
        scene: &SceneDocument,
        assets: &BTreeMap<String, AssetEntry>,
        resources: &BTreeMap<String, ResourceEntry>,
        path: &Path,
    ) -> Result<(), ContentError> {
        scene.validate(assets, resources, path)?;
        if !self.strict {
            return Ok(());
        }
        for entity in &scene.entities {
            for component in entity.components.keys() {
                if !self.components.contains(component) {
                    return Err(ContentError::new(
                        path,
                        format!(
                            "entity `{}` uses unregistered component `{component}`",
                            entity.id
                        ),
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn validate_resource_kinds(
        &self,
        resources: &BTreeMap<String, ResourceEntry>,
        path: &Path,
    ) -> Result<(), ContentError> {
        if !self.strict {
            return Ok(());
        }
        for resource in resources.values() {
            if !self.resource_kinds.contains(&resource.kind) {
                return Err(ContentError::new(
                    path,
                    format!(
                        "resource `{}` uses unregistered kind `{}`",
                        resource.id, resource.kind
                    ),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct SceneEntity {
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub parent: Option<String>,
    #[serde(default)]
    pub components: BTreeMap<String, Value>,
    #[serde(default)]
    pub assets: BTreeMap<String, String>,
    #[serde(default)]
    pub resources: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Component)]
pub struct SceneEntityData {
    pub scene_id: String,
    pub id: String,
    pub name: String,
    pub parent: Option<Entity>,
    pub components: BTreeMap<String, Value>,
    pub assets: BTreeMap<String, String>,
    pub resources: BTreeMap<String, String>,
}

#[derive(Clone, Debug)]
pub struct SceneInstance {
    pub scene_id: String,
    pub entities: BTreeMap<String, Entity>,
}

impl SceneInstance {
    pub fn entity(&self, id: &str) -> Option<Entity> {
        self.entities.get(id).copied()
    }
}

pub(crate) fn validate_id(kind: &str, id: &str, path: &Path) -> Result<(), ContentError> {
    if id.is_empty() || id.trim() != id || id.chars().any(|character| character.is_control()) {
        return Err(ContentError::new(path, format!("invalid {kind} id `{id}`")));
    }
    Ok(())
}

fn validate_references<T>(
    references: &BTreeMap<String, String>,
    entries: &BTreeMap<String, T>,
    kind: &str,
    owner: &str,
    path: &Path,
) -> Result<(), ContentError> {
    for (name, id) in references {
        if name.trim().is_empty() {
            return Err(ContentError::new(
                path,
                format!("{owner} has an empty {kind} reference name"),
            ));
        }
        if !entries.contains_key(id) {
            return Err(ContentError::new(
                path,
                format!("{owner} references missing {kind} `{id}`"),
            ));
        }
    }
    Ok(())
}

fn validate_parent_graph(entities: &[SceneEntity], path: &Path) -> Result<(), ContentError> {
    let mut parents = BTreeMap::new();
    let ids: BTreeSet<&str> = entities.iter().map(|entity| entity.id.as_str()).collect();
    for entity in entities {
        if let Some(parent) = &entity.parent {
            if parent == &entity.id {
                return Err(ContentError::new(
                    path,
                    format!("entity `{}` cannot parent itself", entity.id),
                ));
            }
            if !ids.contains(parent.as_str()) {
                return Err(ContentError::new(
                    path,
                    format!(
                        "entity `{}` references missing parent `{parent}`",
                        entity.id
                    ),
                ));
            }
        }
        parents.insert(entity.id.clone(), entity.parent.clone());
    }

    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    for entity in entities {
        visit_parent(&entity.id, &parents, &mut visiting, &mut visited, path)?;
    }
    Ok(())
}

fn visit_parent(
    id: &str,
    parents: &BTreeMap<String, Option<String>>,
    visiting: &mut BTreeSet<String>,
    visited: &mut BTreeSet<String>,
    path: &Path,
) -> Result<(), ContentError> {
    if visited.contains(id) {
        return Ok(());
    }
    if !visiting.insert(id.to_string()) {
        return Err(ContentError::new(
            path,
            format!("scene parent graph contains a cycle at `{id}`"),
        ));
    }
    if let Some(Some(parent)) = parents.get(id) {
        visit_parent(parent, parents, visiting, visited, path)?;
    }
    visiting.remove(id);
    visited.insert(id.to_string());
    Ok(())
}
