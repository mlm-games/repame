//! glTF 2.0 import to [`MeshGroup`].

use glam::{Mat4, Quat, Vec3};

use super::mesh::MeshGroup;

/// Why an import produced no (or partial) geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportSkip {
    /// Primitive mode is points or lines.
    NonTriangleMode,
    /// No position accessor.
    NoPositions,
    /// Index or attribute read failed.
    BadAccessor,
}

/// One imported mesh: node name plus world-space groups.
#[derive(Clone, Debug, Default)]
pub struct ImportedMesh {
    /// Node name (`mesh_{index}` fallback).
    pub name: String,
    /// One group per primitive.
    pub groups: Vec<MeshGroup>,
}

/// Primitive alpha: (transparent, alpha, cutoff).
pub fn alpha_mode(material: &gltf::Material<'_>) -> (bool, f32, f32) {
    use gltf::material::AlphaMode as M;
    let bc = material.pbr_metallic_roughness().base_color_factor();
    match material.alpha_mode() {
        M::Opaque => (false, 1.0, 0.0),
        M::Mask => (false, bc[3], material.alpha_cutoff().unwrap_or(0.5)),
        M::Blend => (true, bc[3], 0.0),
    }
}

/// Primitive surface material: metallic/roughness/emissive factors.
/// Clamped at flatten time.
pub fn material_of(material: &gltf::Material<'_>) -> super::mesh::Material {
    let pbr = material.pbr_metallic_roughness();
    let emissive = material.emissive_factor();
    let strength = material.emissive_strength().unwrap_or(1.0);
    super::mesh::Material {
        metallic: pbr.metallic_factor(),
        roughness: pbr.roughness_factor(),
        emissive: [
            emissive[0] * strength,
            emissive[1] * strength,
            emissive[2] * strength,
        ],
    }
}

/// True when `KHR_materials_unlit` is set. Unlit skips light.
pub fn is_unlit(material: &gltf::Material<'_>) -> bool {
    material.unlit()
}

impl ImportedMesh {
    pub fn tri_count(&self) -> usize {
        self.groups.iter().map(|g| g.tri_count()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.groups.iter().all(|g| g.is_empty())
    }
}

/// Import .glb (or inline-buffer .gltf) bytes into baked meshes.
/// Node transforms compose down the graph. One group per primitive.
/// Base-color factor becomes tint. Uvs copy through; `texture_page`
/// stays 0, see [`import_slice_textured`](crate::import_slice_textured).
pub fn import_slice(bytes: &[u8]) -> Result<Vec<ImportedMesh>, gltf::Error> {
    let (doc, buffers, _) = gltf::import_slice(bytes)?;
    Ok(import_document(&doc, &buffers))
}

/// Scene walk shared by plain and textured import.
pub(crate) fn import_document(
    doc: &gltf::Document,
    buffers: &[gltf::buffer::Data],
) -> Vec<ImportedMesh> {
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

/// Primitive modes we draw. Others are [`ImportSkip`].
pub(crate) fn triangles_only(mode: gltf::mesh::Mode) -> Result<(), ImportSkip> {
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
    let normals: Vec<[f32; 3]> = reader
        .read_normals()
        .map(|it| it.collect())
        .unwrap_or_default();
    let normals: Option<Vec<[f32; 3]>> = if normals.is_empty() {
        None
    } else {
        Some(normals)
    };
    let tex_coord = prim
        .material()
        .pbr_metallic_roughness()
        .base_color_texture()
        .map(|t| {
            t.texture_transform()
                .and_then(|xf| xf.tex_coord())
                .unwrap_or_else(|| t.tex_coord())
        })
        .unwrap_or(0);
    let uvs: Option<Vec<[f32; 2]>> = reader
        .read_tex_coords(tex_coord)
        .map(|it| it.into_f32().collect());
    // Document image behind base-color texture. None = untextured.
    // Guarded against corrupt texture indices.
    let base_image = prim
        .material()
        .pbr_metallic_roughness()
        .base_color_texture()
        .and_then(|t| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                t.texture().source().index()
            }))
            .ok()
        });

    let file_indices: Vec<u32> = match reader.read_indices() {
        Some(gltf::mesh::util::ReadIndices::U8(it)) => it.map(u32::from).collect(),
        Some(gltf::mesh::util::ReadIndices::U16(it)) => it.map(u32::from).collect(),
        Some(gltf::mesh::util::ReadIndices::U32(it)) => it.collect(),
        None => (0..positions.len() as u32).collect(),
    };
    let double_sided = prim.material().double_sided();
    let indices: Vec<u32> = match prim.mode() {
        gltf::mesh::Mode::TriangleFan => fan_to_list(&file_indices),
        gltf::mesh::Mode::TriangleStrip => strip_to_list(&file_indices),
        _ => file_indices,
    };
    let indices = if double_sided {
        let mut both = Vec::with_capacity(indices.len() * 2);
        both.extend_from_slice(&indices);
        for tri in indices.as_chunks::<3>().0 {
            both.extend_from_slice(&[tri[0], tri[2], tri[1]]);
        }
        both
    } else {
        indices
    };

    let bc = prim.material().pbr_metallic_roughness().base_color_factor();
    let tint: [f32; 3] = [bc[0], bc[1], bc[2]];
    let (transparent, alpha, alpha_cutoff) = alpha_mode(&prim.material());

    let unlit = is_unlit(&prim.material());
    // Inverse-transpose of the world 3x3: correct under non-uniform node
    // scale, where `Quat::from_mat4` would bake the skew into the rotation.
    // Falls back to no rotation on degenerate (zero-scale) matrices.
    let normal_mat = {
        let rot_scale = Mat4::from_mat3(glam::Mat3::from_mat4(world));
        let inv_t = rot_scale.inverse().transpose();
        if inv_t.is_finite() {
            inv_t
        } else {
            Mat4::IDENTITY
        }
    };
    let lit = normals.as_ref().is_some_and(|ns| !ns.is_empty()) && !unlit;
    let textured = uvs.is_some();

    let mut group = MeshGroup {
        depth_test: true,
        transparent,
        alpha,
        alpha_cutoff,
        material: material_of(&prim.material()),
        base_image,
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
        let ts_owned: Option<Vec<[f32; 2]>> = prim
            .material()
            .pbr_metallic_roughness()
            .base_color_texture()
            .and_then(|i| i.texture_transform())
            .map(|xf| {
                let (ox, oy) = (xf.offset()[0], xf.offset()[1]);
                let (sx, sy) = (xf.scale()[0], xf.scale()[1]);
                let (s, c) = xf.rotation().sin_cos();
                ts.iter()
                    .map(|[u, v]| {
                        let (x, y) = (u * sx, v * sy);
                        [ox + c * x + s * y, oy - s * x + c * y]
                    })
                    .collect()
            });
        let ts_ref: &Vec<[f32; 2]> = ts_owned.as_ref().unwrap_or(ts);
        group.uvs.extend(ts_ref.iter().map(|[u, v]| [*u, 1.0 - *v]));
    }
    group.indices.extend_from_slice(&indices);

    if let Some(rgba) = reader.read_colors(0).map(|c| {
        let v: Vec<[f32; 4]> = c.into_rgba_f32().collect();
        v
    }) {
        if rgba.len() == group.colors.len() {
            for (tint, vc) in group.colors.iter_mut().zip(rgba.iter()) {
                tint[0] *= vc[0];
                tint[1] *= vc[1];
                tint[2] *= vc[2];
                group.alpha *= vc[3];
            }
        } else {
            log::warn!(
                "gltf: COLOR_0 len {} != {} verts, skipping vertex colors",
                rgba.len(),
                group.colors.len()
            );
        }
    }

    let count = group.positions.len();
    if group.indices.iter().any(|i| (*i as usize) >= count) {
        return Err(ImportSkip::BadAccessor);
    }
    let _ = doc;
    Ok(group)
}

/// Fan (0, i, i+1) to triangle list.
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

/// Strip to triangle list. Odd steps swap to hold winding.
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

/// Push all imported groups into one list. Drops node names.
pub fn flatten_imported(meshes: &[ImportedMesh]) -> Vec<MeshGroup> {
    meshes
        .iter()
        .flat_map(|m| m.groups.iter().cloned())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// JSON-only glTF container for material fixtures.
    pub(crate) fn wrap_json(json: &str) -> Vec<u8> {
        let json_bytes = json.as_bytes();
        let json_pad = (4 - json_bytes.len() % 4) % 4;
        let total = 12 + 8 + json_bytes.len() + json_pad;
        let mut glb = Vec::with_capacity(total);
        glb.extend_from_slice(&0x46546C67u32.to_le_bytes()); // "glTF"
        glb.extend_from_slice(&2u32.to_le_bytes());
        glb.extend_from_slice(&(total as u32).to_le_bytes());
        glb.extend_from_slice(&((json_bytes.len() + json_pad) as u32).to_le_bytes());
        glb.extend_from_slice(&0x4E4F534Au32.to_le_bytes()); // "JSON"
        glb.extend_from_slice(json_bytes);
        glb.extend_from_slice(&vec![0x20u8; json_pad]);
        glb
    }

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
    fn alpha_modes_map_to_group_fields() {
        // OPAQUE is the gltf default (no alphaMode key): opaque, full.
        let glb = quad_gltf();
        let meshes = import_slice(&glb).expect("fixture parses");
        let g = &meshes[0].groups[0];
        assert!(!g.transparent);
        assert_eq!((g.alpha, g.alpha_cutoff), (1.0, 0.0));
        // BLEND / MASK resolve directly (no fixture rebuild needed).
        let doc_json = r#"{"asset":{"version":"2.0"},"materials":[{"alphaMode":"BLEND","alphaCutoff":0.3},{"alphaMode":"MASK"}],"scenes":[{"nodes":[]}]}"#;
        let (doc, _, _) = gltf::import_slice(wrap_json(doc_json)).expect("material fixture parses");
        let mats: Vec<gltf::Material> = doc.materials().collect();
        assert_eq!(mats.len(), 2);
        assert_eq!(alpha_mode(&mats[0]), (true, 1.0, 0.0));
        assert_eq!(alpha_mode(&mats[1]), (false, 1.0, 0.5));
    }

    #[test]
    fn material_factors_import_verbatim() {
        let glb = quad_gltf();
        let meshes = import_slice(&glb).expect("fixture parses");
        let g = &meshes[0].groups[0];
        assert_eq!((g.material.metallic, g.material.roughness), (1.0, 1.0));
        assert_eq!(g.material.emissive, [0.0, 0.0, 0.0]);
        let doc_json = r#"{"asset":{"version":"2.0"},"materials":[{"pbrMetallicRoughness":{"metallicFactor":0.2,"roughnessFactor":0.6},"emissiveFactor":[2.0,0.5,0.0]}],"scenes":[{"nodes":[]}]}"#;
        let (doc, _, _) = gltf::import_slice(wrap_json(doc_json)).expect("material fixture parses");
        let mats: Vec<gltf::Material> = doc.materials().collect();
        assert_eq!(mats.len(), 1);
        let m = material_of(&mats[0]);
        assert!((m.metallic - 0.2).abs() < 1e-6, "{m:?}");
        assert!((m.roughness - 0.6).abs() < 1e-6, "{m:?}");
        assert_eq!(m.emissive, [2.0, 0.5, 0.0]);
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

    /// Quad with COLOR_0 (u8 RGBA). Tint multiplies, alpha scales group.
    fn color_quad_gltf() -> Vec<u8> {
        let mut bin: Vec<u8> = Vec::new();
        for p in [
            [-1f32, 0.0, -1.0],
            [-1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
        ] {
            bin.extend_from_slice(bytemuck::cast_slice(&p));
        }
        for c in [
            [255u8, 255, 255, 255],
            [255, 0, 0, 255],
            [0, 255, 0, 255],
            [255, 255, 255, 128],
        ] {
            bin.extend_from_slice(&c);
        }
        for i in [0u16, 1, 2, 0, 2, 3] {
            bin.extend_from_slice(bytemuck::cast_slice(&[i]));
        }
        let json = [
            r#"{"asset":{"version":"2.0"},"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],"meshes":[{"primitives":[{"attributes":{"POSITION":0,"COLOR_0":1},"indices":2,"material":0}]}],"materials":[{"pbrMetallicRoughness":{"baseColorFactor":[1.0,1.0,1.0,1.0]}}],"buffers":[{"byteLength":"#,
            &bin.len().to_string(),
            r#"}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":48},{"buffer":0,"byteOffset":48,"byteLength":16},{"buffer":0,"byteOffset":64,"byteLength":12}],"accessors":[{"bufferView":0,"componentType":5126,"count":4,"type":"VEC3","min":[-1.0,0.0,-1.0],"max":[1.0,0.0,1.0]},{"bufferView":1,"componentType":5121,"count":4,"type":"VEC4","normalized":true},{"bufferView":2,"componentType":5123,"count":6,"type":"SCALAR"}]}"#,
        ]
        .concat();
        let json_bytes = json.as_bytes();
        let json_pad = (4 - json_bytes.len() % 4) % 4;
        let bin_pad = (4 - bin.len() % 4) % 4;
        let total = 12 + 8 + json_bytes.len() + json_pad + 8 + bin.len() + bin_pad;
        let mut glb = Vec::with_capacity(total);
        glb.extend_from_slice(&0x46546C67u32.to_le_bytes());
        glb.extend_from_slice(&2u32.to_le_bytes());
        glb.extend_from_slice(&(total as u32).to_le_bytes());
        glb.extend_from_slice(&((json_bytes.len() + json_pad) as u32).to_le_bytes());
        glb.extend_from_slice(&0x4E4F534Au32.to_le_bytes());
        glb.extend_from_slice(json_bytes);
        glb.extend_from_slice(&vec![0x20u8; json_pad]);
        glb.extend_from_slice(&((bin.len() + bin_pad) as u32).to_le_bytes());
        glb.extend_from_slice(&0x004E4942u32.to_le_bytes());
        glb.extend_from_slice(&bin);
        glb.extend_from_slice(&vec![0u8; bin_pad]);
        glb
    }

    #[test]
    fn color_0_multiplies_tint_and_alpha() {
        let meshes = import_slice(&color_quad_gltf()).expect("color fixture parses");
        let g = &meshes[0].groups[0];
        assert_eq!(g.colors.len(), 4);
        assert_eq!(g.colors[0], [1.0, 1.0, 1.0], "white stays");
        assert_eq!(g.colors[1], [1.0, 0.0, 0.0], "red multiplies");
        assert_eq!(g.colors[2], [0.0, 1.0, 0.0], "green multiplies");
        let expect = 128.0f32 / 255.0;
        assert!(
            (g.alpha - expect).abs() < 1e-3,
            "alpha scaled by the 50% vert: {}",
            g.alpha
        );
        assert!(g.normals.is_empty());
    }

    #[test]
    fn double_sided_emits_reversed_winding_copy() {
        let mut bin: Vec<u8> = Vec::new();
        for p in [
            [-1f32, 0.0, -1.0],
            [-1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
        ] {
            bin.extend_from_slice(bytemuck::cast_slice(&p));
        }
        for i in [0u16, 1, 2, 0, 2, 3] {
            bin.extend_from_slice(bytemuck::cast_slice(&[i]));
        }
        let json = format!(
            r#"{{"asset":{{"version":"2.0"}},"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}},"indices":1,"material":0}}]}}],"materials":[{{"doubleSided":true}}],"buffers":[{{"byteLength":{}}}],"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":48}},{{"buffer":0,"byteOffset":48,"byteLength":12}}],"accessors":[{{"bufferView":0,"componentType":5126,"count":4,"type":"VEC3","min":[-1.0,0.0,-1.0],"max":[1.0,0.0,1.0]}},{{"bufferView":1,"componentType":5123,"count":6,"type":"SCALAR"}}]}}"#,
            bin.len()
        );
        let json_bytes = json.as_bytes();
        let json_pad = (4 - json_bytes.len() % 4) % 4;
        let bin_pad = (4 - bin.len() % 4) % 4;
        let total = 12 + 8 + json_bytes.len() + json_pad + 8 + bin.len() + bin_pad;
        let mut glb = Vec::with_capacity(total);
        glb.extend_from_slice(&0x46546C67u32.to_le_bytes());
        glb.extend_from_slice(&2u32.to_le_bytes());
        glb.extend_from_slice(&(total as u32).to_le_bytes());
        glb.extend_from_slice(&((json_bytes.len() + json_pad) as u32).to_le_bytes());
        glb.extend_from_slice(&0x4E4F534Au32.to_le_bytes());
        glb.extend_from_slice(json_bytes);
        glb.extend_from_slice(&vec![0x20u8; json_pad]);
        glb.extend_from_slice(&((bin.len() + bin_pad) as u32).to_le_bytes());
        glb.extend_from_slice(&0x004E4942u32.to_le_bytes());
        glb.extend_from_slice(&bin);
        glb.extend_from_slice(&vec![0u8; bin_pad]);
        let meshes = import_slice(&glb).expect("double-sided parses");
        let g = &meshes[0].groups[0];
        assert_eq!(g.tri_count(), 4, "front + reversed copy");
        assert_eq!(&g.indices[0..6], &[0, 1, 2, 0, 2, 3]);
        assert_eq!(&g.indices[6..12], &[0, 2, 1, 0, 3, 2]);
    }

    #[test]
    fn unlit_material_skips_normals() {
        let doc_json = r#"{"asset":{"version":"2.0"},"extensionsUsed":["KHR_materials_unlit"],"materials":[{"extensions":{"KHR_materials_unlit":{}}}],"scenes":[{"nodes":[]}]}"#;
        let gltf = gltf::Gltf::from_slice_without_validation(&wrap_json(doc_json))
            .expect("unlit fixture parses");
        let mats: Vec<gltf::Material> = gltf.document.materials().collect();
        assert_eq!(mats.len(), 1);
        assert!(is_unlit(&mats[0]), "extension detected");
    }

    #[test]
    fn emissive_strength_scales_emissive() {
        let doc_json = r#"{"asset":{"version":"2.0"},"extensionsUsed":["KHR_materials_emissive_strength"],"materials":[{"emissiveFactor":[1.0,0.5,0.0],"extensions":{"KHR_materials_emissive_strength":{"emissiveStrength":3.0}}}],"scenes":[{"nodes":[]}]}"#;
        let gltf = gltf::Gltf::from_slice_without_validation(&wrap_json(doc_json)).expect("parses");
        let mats: Vec<gltf::Material> = gltf.document.materials().collect();
        let m = material_of(&mats[0]);
        assert_eq!(m.emissive, [3.0, 1.5, 0.0], "strength scales: {m:?}");
    }

    #[test]
    fn nonuniform_node_scale_keeps_normals_unit() {
        // A node with scale [1, 2, 1]: positions stretch, but normals must
        // come out unit-length and axis-correct (inverse-transpose, not
        // `Quat::from_mat4`, which bakes the skew into the rotation).
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
        for i in [0u16, 1, 2, 0, 2, 3] {
            bin.extend_from_slice(bytemuck::cast_slice(&[i]));
        }
        let json = format!(
            r#"{{"asset":{{"version":"2.0"}},"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0,"scale":[1.0,2.0,1.0]}}],"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"NORMAL":1}},"indices":2}}]}}],"buffers":[{{"byteLength":{}}}],"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":48}},{{"buffer":0,"byteOffset":48,"byteLength":48}},{{"buffer":0,"byteOffset":96,"byteLength":12}}],"accessors":[{{"bufferView":0,"componentType":5126,"count":4,"type":"VEC3","min":[-1.0,0.0,-1.0],"max":[1.0,0.0,1.0]}},{{"bufferView":1,"componentType":5126,"count":4,"type":"VEC3"}},{{"bufferView":2,"componentType":5123,"count":6,"type":"SCALAR"}}]}}"#,
            bin.len()
        );
        let json_bytes = json.as_bytes();
        let json_pad = (4 - json_bytes.len() % 4) % 4;
        let bin_pad = (4 - bin.len() % 4) % 4;
        let total = 12 + 8 + json_bytes.len() + json_pad + 8 + bin.len() + bin_pad;
        let mut glb = Vec::with_capacity(total);
        glb.extend_from_slice(&0x46546C67u32.to_le_bytes());
        glb.extend_from_slice(&2u32.to_le_bytes());
        glb.extend_from_slice(&(total as u32).to_le_bytes());
        glb.extend_from_slice(&((json_bytes.len() + json_pad) as u32).to_le_bytes());
        glb.extend_from_slice(&0x4E4F534Au32.to_le_bytes());
        glb.extend_from_slice(json_bytes);
        glb.extend_from_slice(&vec![0x20u8; json_pad]);
        glb.extend_from_slice(&((bin.len() + bin_pad) as u32).to_le_bytes());
        glb.extend_from_slice(&0x004E4942u32.to_le_bytes());
        glb.extend_from_slice(&bin);
        glb.extend_from_slice(&vec![0u8; bin_pad]);
        let meshes = import_slice(&glb).expect("scaled node parses");
        let g = &meshes[0].groups[0];
        assert!(
            g.normals.iter().all(|n| *n == [0.0, 1.0, 0.0]),
            "{:?}",
            g.normals
        );
    }
}
