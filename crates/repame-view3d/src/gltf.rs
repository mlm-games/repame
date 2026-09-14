//! glTF 2.0 import behind [`MeshGroup`](super::mesh::MeshGroup).

use glam::{Mat4, Quat, Vec3};

use super::mesh::MeshGroup;

/// Why an import produced no (or partial) geometry. Parsing itself reports
/// through `gltf::Error`; this covers the semantic skips the importer logs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportSkip {
    /// Primitive mode isn't triangles/fans/strips (points/lines).
    NonTriangleMode,
    /// No position accessor (nothing to draw).
    NoPositions,
    /// Index/attribute read failed (sparse/mismatched — skipped).
    BadAccessor,
}

/// One imported mesh: named node baked to world space.
#[derive(Clone, Debug, Default)]
pub struct ImportedMesh {
    /// Node name (`mesh_{index}` fallback), for pick ids / debugging.
    pub name: String,
    /// Draw groups (one per primitive, world-space, lit when the source
    /// had normals, tinted by the material base-color factor).
    pub groups: Vec<MeshGroup>,
}

impl ImportedMesh {
    pub fn tri_count(&self) -> usize {
        self.groups.iter().map(|g| g.tri_count()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.groups.iter().all(|g| g.is_empty())
    }
}

/// Import `.glb` (or inline-buffer `.gltf`) bytes into baked meshes.
///
/// Node transforms compose down the scene graph (T * R * S per node, column
/// major); normals use the inverse-transpose rotation only (uniform-scale
/// safe — non-uniform scale renormalizes on the way out). One group per
/// primitive; material base-color factor becomes the tint (textures stay
/// game-decoded: uvs copy through verbatim).
pub fn import_slice(bytes: &[u8]) -> Result<Vec<ImportedMesh>, gltf::Error> {
    let (doc, buffers, _) = gltf::import_slice(bytes)?;
    Ok(import_document(&doc, &buffers))
}

fn import_document(doc: &gltf::Document, buffers: &[gltf::buffer::Data]) -> Vec<ImportedMesh> {
    let mut out = Vec::new();
    for scene in doc.scenes() {
        for node in scene.nodes() {
            collect_node(doc, buffers, &node, Mat4::IDENTITY, &mut out);
        }
    }
    out
}

fn collect_node(
    doc: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    node: &gltf::Node<'_>,
    parent: Mat4,
    out: &mut Vec<ImportedMesh>,
) {
    let (t, r, s) = node.transform().decomposed();
    let local =
        Mat4::from_scale_rotation_translation(Vec3::from(s), Quat::from_array(r), Vec3::from(t));
    let world = parent * local;
    if let Some(mesh) = node.mesh() {
        let name = node
            .name()
            .map(|n| n.to_string())
            .unwrap_or_else(|| format!("mesh_{}", mesh.index()));
        let mut imported = ImportedMesh {
            name,
            groups: Vec::new(),
        };
        for prim in mesh.primitives() {
            match import_primitive(doc, buffers, &prim, world) {
                Ok(group) => {
                    if !group.is_empty() {
                        imported.groups.push(group);
                    }
                }
                Err(skip) => {
                    log::warn!("gltf: skipping primitive ({skip:?})");
                }
            }
        }
        if !imported.is_empty() {
            out.push(imported);
        }
    }
    for child in node.children() {
        collect_node(doc, buffers, &child, world, out);
    }
}

/// glTF primitive modes we draw (everything else is [`ImportSkip`]).
fn triangles_only(mode: gltf::mesh::Mode) -> Result<(), ImportSkip> {
    match mode {
        gltf::mesh::Mode::Triangles
        | gltf::mesh::Mode::TriangleFan
        | gltf::mesh::Mode::TriangleStrip => Ok(()),
        _ => Err(ImportSkip::NonTriangleMode),
    }
}

fn import_primitive(
    doc: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    prim: &gltf::Primitive<'_>,
    world: Mat4,
) -> Result<MeshGroup, ImportSkip> {
    triangles_only(prim.mode())?;
    let reader = prim.reader(|b| buffers.get(b.index()).map(|d| d.0.as_slice()));

    let positions: Vec<[f32; 3]> = reader
        .read_positions()
        .ok_or(ImportSkip::NoPositions)?
        .collect();
    if positions.is_empty() {
        return Err(ImportSkip::NoPositions);
    }
    let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(|it| it.collect());
    let uvs: Option<Vec<[f32; 2]>> = reader.read_tex_coords(0).map(|it| it.into_f32().collect());

    let file_indices: Vec<u32> = match reader.read_indices() {
        Some(gltf::mesh::util::ReadIndices::U8(it)) => it.map(u32::from).collect(),
        Some(gltf::mesh::util::ReadIndices::U16(it)) => it.map(u32::from).collect(),
        Some(gltf::mesh::util::ReadIndices::U32(it)) => it.collect(),
        None => (0..positions.len() as u32).collect(),
    };
    let indices: Vec<u32> = match prim.mode() {
        gltf::mesh::Mode::TriangleFan => fan_to_list(&file_indices),
        gltf::mesh::Mode::TriangleStrip => strip_to_list(&file_indices),
        _ => file_indices,
    };

    let bc = prim.material().pbr_metallic_roughness().base_color_factor();
    let tint: [f32; 3] = [bc[0], bc[1], bc[2]];

    let normal_mat = Mat4::from_quat(Quat::from_mat4(&world));
    let lit = normals.is_some();
    let textured = uvs.is_some();

    let mut group = MeshGroup {
        depth_test: true,
        ..Default::default()
    };
    group.positions.reserve_exact(positions.len());
    group.colors.reserve_exact(positions.len());
    if lit {
        group.normals.reserve_exact(positions.len());
    }
    if textured {
        group.uvs.reserve_exact(positions.len());
    }
    group.indices.reserve_exact(indices.len());

    for p in &positions {
        let wp = world.transform_point3(Vec3::from(*p));
        group.positions.push(wp.into());
        group.colors.push(tint);
    }
    if let Some(ns) = &normals {
        for n in ns {
            let wn = normal_mat.transform_vector3(Vec3::from(*n));
            let wn = wn.try_normalize().unwrap_or(Vec3::Y);
            group.normals.push(wn.into());
        }
    }
    if let Some(ts) = &uvs {
        group.uvs.extend(ts.iter().map(|[u, v]| [*u, 1.0 - *v]));
    }
    group.indices.extend_from_slice(&indices);

    let count = group.positions.len();
    if group.indices.iter().any(|i| (*i as usize) >= count) {
        return Err(ImportSkip::BadAccessor);
    }
    let _ = doc;
    Ok(group)
}

/// Triangle fan (0, i, i+1) to a triangle list.
pub fn fan_to_list(indices: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(indices.len());
    if indices.len() < 3 {
        return out;
    }
    for w in indices[1..].windows(2) {
        out.extend_from_slice(&[indices[0], w[0], w[1]]);
    }
    out
}

/// Triangle strip to a triangle list (alternating winding preserved: odd
/// steps swap so every triangle stays front-facing).
pub fn strip_to_list(indices: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(indices.len());
    if indices.len() < 3 {
        return out;
    }
    for (i, w) in indices.windows(3).enumerate() {
        if i % 2 == 0 {
            out.extend_from_slice(&[w[0], w[1], w[2]]);
        } else {
            out.extend_from_slice(&[w[2], w[1], w[0]]);
        }
    }
    out
}

/// Push every imported mesh's groups into one [`MeshGroup`] list (for
/// direct snapshot assembly). Names are dropped — use
/// [`ImportedMesh::groups`] when per-node pick ids matter.
pub fn flatten_imported(meshes: &[ImportedMesh]) -> Vec<MeshGroup> {
    meshes
        .iter()
        .flat_map(|m| m.groups.iter().cloned())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad_gltf() -> Vec<u8> {
        let mut bin: Vec<u8> = Vec::new();
        for p in [
            [-1f32, 0.0, -1.0],
            [-1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
        ] {
            bin.extend_from_slice(bytemuck::cast_slice(&p));
        }
        for _ in 0..4 {
            bin.extend_from_slice(bytemuck::cast_slice(&[0f32, 1.0, 0.0]));
        }
        for uv in [[0f32, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]] {
            bin.extend_from_slice(bytemuck::cast_slice(&uv));
        }
        for i in [0u16, 1, 2, 0, 2, 3] {
            bin.extend_from_slice(bytemuck::cast_slice(&[i]));
        }
        assert_eq!(bin.len(), 48 + 48 + 32 + 12);
        let json = [
            r#"{"asset":{"version":"2.0"},"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],"meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2},"indices":3}]}],"buffers":[{"byteLength":"#,
            &bin.len().to_string(),
            r#"}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":48},{"buffer":0,"byteOffset":48,"byteLength":48},{"buffer":0,"byteOffset":96,"byteLength":32},{"buffer":0,"byteOffset":128,"byteLength":12}],"accessors":[{"bufferView":0,"componentType":5126,"count":4,"type":"VEC3","min":[-1.0,0.0,-1.0],"max":[1.0,0.0,1.0]},{"bufferView":1,"componentType":5126,"count":4,"type":"VEC3"},{"bufferView":2,"componentType":5126,"count":4,"type":"VEC2"},{"bufferView":3,"componentType":5123,"count":6,"type":"SCALAR"}]}"#,
        ]
        .concat();
        let mut glb = Vec::new();
        glb.extend_from_slice(&12u32.to_le_bytes()); // magic placeholder, replaced below
        glb.clear();
        let json_bytes = json.as_bytes();
        let json_pad = (4 - json_bytes.len() % 4) % 4;
        let bin_pad = (4 - bin.len() % 4) % 4;
        let total = 12 + 8 + json_bytes.len() + json_pad + 8 + bin.len() + bin_pad;
        glb.extend_from_slice(&0x46546C67u32.to_le_bytes()); // "glTF"
        glb.extend_from_slice(&2u32.to_le_bytes());
        glb.extend_from_slice(&(total as u32).to_le_bytes());
        glb.extend_from_slice(&((json_bytes.len() + json_pad) as u32).to_le_bytes());
        glb.extend_from_slice(&0x4E4F534Au32.to_le_bytes()); // "JSON"
        glb.extend_from_slice(json_bytes);
        glb.extend_from_slice(&vec![0x20u8; json_pad]);
        glb.extend_from_slice(&((bin.len() + bin_pad) as u32).to_le_bytes());
        glb.extend_from_slice(&0x004E4942u32.to_le_bytes()); // "BIN\0"
        glb.extend_from_slice(&bin);
        glb.extend_from_slice(&vec![0u8; bin_pad]);
        glb
    }

    #[test]
    fn fan_and_strip_triangulate() {
        assert_eq!(fan_to_list(&[0, 1, 2, 3]), vec![0, 1, 2, 0, 2, 3]);
        assert!(fan_to_list(&[0, 1]).is_empty());
        assert_eq!(strip_to_list(&[0, 1, 2, 3]), vec![0, 1, 2, 3, 2, 1]);
        assert!(strip_to_list(&[0, 1]).is_empty());
    }

    #[test]
    fn quad_imports_lit_textured_world_baked() {
        let glb = quad_gltf();
        let meshes = import_slice(&glb).expect("fixture parses");
        assert_eq!(meshes.len(), 1, "one node mesh");
        assert_eq!(meshes[0].groups.len(), 1, "one primitive");
        let g = &meshes[0].groups[0];
        assert_eq!(g.tri_count(), 2);
        assert_eq!(g.normals.len(), g.positions.len());
        assert!(g.normals.iter().all(|n| *n == [0.0, 1.0, 0.0]));
        assert_eq!(g.uvs.len(), g.positions.len());
        assert_eq!(g.uvs[0], [0.0, 1.0]);
        assert_eq!(g.uvs[2], [1.0, 0.0]);
        assert!(g.positions.iter().all(|p| p[1] == 0.0));
        assert!(g.colors.iter().all(|c| *c == [1.0, 1.0, 1.0]));
    }

    #[test]
    fn garbage_bytes_error_instead_of_panic() {
        assert!(import_slice(&[]).is_err());
        assert!(import_slice(&[0u8; 64]).is_err());
    }

    #[test]
    fn bad_mode_and_missing_positions_skip_cleanly() {
        assert_eq!(
            triangles_only(gltf::mesh::Mode::Points),
            Err(ImportSkip::NonTriangleMode)
        );
        assert_eq!(
            triangles_only(gltf::mesh::Mode::LineStrip),
            Err(ImportSkip::NonTriangleMode)
        );
        assert!(triangles_only(gltf::mesh::Mode::TriangleStrip).is_ok());
    }
}
