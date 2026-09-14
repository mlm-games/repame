//! CPU skinning behind [`MeshGroup`](super::mesh::MeshGroup).
//!
//! Asset-side only, like [`gltf`](super::gltf): a [`SkinnedMesh`] holds the
//! bind-pose vertices plus joint/weight attributes parsed once; per frame
//! the game computes joint matrices (from [`NodePose`] animation tracks or
//! procedural poses) and calls [`pose`] to bake a plain world-space group
//! the renderer already draws. No GPU path changes, no shader skinning —
//! the batch sees positions/normals like any procedural group.
//!
//! Math (glTF 2.0 spec): `skinned_pos = sum_w joint_mat[j] * ibm[j] * pos`,
//! normals rotate by the blended 3x3 (renormalized). Weights normalize on
//! the way in (unweighted verts hold bind pose); joint indices past the
//! matrix list clamp to joint 0 with a warning-free skip (malformed data
//! holds still instead of panicking).

use std::collections::HashMap;

use glam::{Mat4, Quat, Vec3};

use super::mesh::MeshGroup;

/// One joint's local transform (glTF TRS, game-driven per frame).
#[derive(Clone, Copy, Debug)]
pub struct JointPose {
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

impl Default for JointPose {
    fn default() -> Self {
        Self {
            translation: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
        }
    }
}

impl JointPose {
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(
            Vec3::from(self.scale),
            Quat::from_array(self.rotation),
            Vec3::from(self.translation),
        )
    }
}

/// Node animation track: timestamped TRS keys sampled per frame.
/// Times are seconds, ascending; out-of-range samples clamp (hold first /
/// last — matches `AnimPlayer::Once` hold semantics in `repame-anim`).
#[derive(Clone, Debug, Default)]
pub struct NodePose {
    pub times: Vec<f32>,
    pub translations: Vec<[f32; 3]>,
    pub rotations: Vec<[f32; 4]>,
    pub scales: Vec<[f32; 3]>,
}

impl NodePose {
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    pub fn duration(&self) -> f32 {
        self.times.last().copied().unwrap_or(0.0)
    }

    /// Sample at `t` seconds (clamped). Missing channels hold bind values
    /// (identity rotation, unit scale, zero translation).
    pub fn sample(&self, t: f32) -> JointPose {
        if self.times.is_empty() {
            return JointPose::default();
        }
        let n = self.times.len();
        let mut i = 0;
        while i + 1 < n && self.times[i + 1] <= t {
            i += 1;
        }
        let j = (i + 1).min(n - 1);
        let alpha = if j == i || self.times[j] <= self.times[i] {
            0.0
        } else {
            ((t - self.times[i]) / (self.times[j] - self.times[i])).clamp(0.0, 1.0)
        };
        let lerp3 = |a: [f32; 3], b: [f32; 3]| {
            [
                a[0] + (b[0] - a[0]) * alpha,
                a[1] + (b[1] - a[1]) * alpha,
                a[2] + (b[2] - a[2]) * alpha,
            ]
        };
        JointPose {
            translation: match (self.translations.get(i), self.translations.get(j)) {
                (Some(a), Some(b)) => lerp3(*a, *b),
                (Some(a), None) | (None, Some(a)) => *a,
                (None, None) => [0.0, 0.0, 0.0],
            },
            rotation: match (self.rotations.get(i), self.rotations.get(j)) {
                (Some(a), Some(b)) => Quat::from_array(*a)
                    .slerp(Quat::from_array(*b), alpha)
                    .to_array(),
                (Some(a), None) | (None, Some(a)) => *a,
                (None, None) => [0.0, 0.0, 0.0, 1.0],
            },
            scale: match (self.scales.get(i), self.scales.get(j)) {
                (Some(a), Some(b)) => lerp3(*a, *b),
                (Some(a), None) | (None, Some(a)) => *a,
                (None, None) => [1.0, 1.0, 1.0],
            },
        }
    }
}

/// One glTF animation: named node tracks sampled by [`NodePose::sample`].
/// Keys are node indices (map through [`SkinnedMesh::node_to_joint`] to
/// drive joint matrices).
#[derive(Clone, Debug, Default)]
pub struct Animation {
    pub name: String,
    pub tracks: HashMap<usize, NodePose>,
}

impl Animation {
    pub fn duration(&self) -> f32 {
        self.tracks
            .values()
            .map(|t| t.duration())
            .fold(0.0f32, f32::max)
    }

    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }
}

/// Bind-pose skinned mesh parsed once from a glTF primitive (+ skin).
/// `joints`/`weights` are per-vertex `[u16; 4]`/`[f32; 4]` (normalized on
/// parse); `inverse_bind` is per-joint `Mat4` (identity when the file omits
/// them, i.e. pre-applied). `node_to_joint` maps glTF node indices to joint
/// slots so animation tracks (keyed by node) drive the right matrices.
#[derive(Clone, Debug, Default)]
pub struct SkinnedMesh {
    pub name: String,
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub colors: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
    pub joints: Vec<[u16; 4]>,
    pub weights: Vec<[f32; 4]>,
    pub inverse_bind: Vec<Mat4>,
    pub node_to_joint: HashMap<usize, usize>,
    pub texture_page: u32,
    pub pick_id: u32,
    pub depth_test: bool,
}

impl SkinnedMesh {
    pub fn joint_count(&self) -> usize {
        self.inverse_bind.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty() || self.positions.is_empty()
    }

    /// Bake `joint_matrices` (world-space per joint, same order as
    /// `inverse_bind`) into a world-space [`MeshGroup`]. Normals rotate by
    /// the blended matrix's 3x3 and renormalize. Unweighted verts (all-zero
    /// weights) hold bind pose.
    pub fn pose(&self, joint_matrices: &[Mat4]) -> MeshGroup {
        let mut group = MeshGroup {
            texture_page: self.texture_page,
            pick_id: self.pick_id,
            depth_test: self.depth_test,
            ..Default::default()
        };
        if self.is_empty() {
            return group;
        }
        let n = self.positions.len();
        group.positions.reserve_exact(n);
        group.colors.extend_from_slice(&self.colors);
        let has_normals = self.normals.len() == n;
        if has_normals {
            group.normals.reserve_exact(n);
        }
        if self.uvs.len() == n {
            group.uvs.extend_from_slice(&self.uvs);
        }
        group.indices.extend_from_slice(&self.indices);

        let skin_count = joint_matrices.len().min(self.inverse_bind.len());
        for i in 0..n {
            let pos = Vec3::from(self.positions[i]);
            let joints = self.joints.get(i).copied().unwrap_or([0; 4]);
            let weights = self.weights.get(i).copied().unwrap_or([0.0; 4]);
            let wsum: f32 = weights.iter().sum();
            if wsum <= 1e-8 || skin_count == 0 {
                group.positions.push(self.positions[i]);
                if has_normals {
                    group.normals.push(self.normals[i]);
                }
                continue;
            }
            let mut skinned = Vec3::ZERO;
            let mut nrm = Vec3::ZERO;
            let has_n = has_normals;
            let bind_n = if has_n {
                Vec3::from(self.normals[i])
            } else {
                Vec3::Y
            };
            for k in 0..4 {
                let w = weights[k] / wsum;
                if w <= 0.0 {
                    continue;
                }
                let j = (joints[k] as usize).min(skin_count.saturating_sub(1));
                let m = joint_matrices[j] * self.inverse_bind[j];
                skinned += m.transform_point3(pos) * w;
                if has_n {
                    nrm += m.transform_vector3(bind_n) * w;
                }
            }
            group.positions.push(skinned.into());
            if has_n {
                group
                    .normals
                    .push(nrm.try_normalize().unwrap_or(Vec3::Y).into());
            }
        }
        group
    }
}

/// Parse every skinned primitive in `bytes` into [`SkinnedMesh`]s (bind
/// pose). Unskinned primitives are skipped (the [`gltf`](super::gltf)
/// importer owns those). Joint slots follow skin joint order; node indices
/// map through `node_to_joint` for animation tracks.
pub fn import_skinned(bytes: &[u8]) -> Result<Vec<SkinnedMesh>, gltf::Error> {
    let (doc, buffers, _) = gltf::import_slice(bytes)?;
    let mut out = Vec::new();
    for mesh in doc.meshes() {
        for prim in mesh.primitives() {
            let reader = prim.reader(|b| buffers.get(b.index()).map(|d| d.0.as_slice()));
            let joints: Option<Vec<[u16; 4]>> = reader.read_joints(0).map(|it| {
                use gltf::mesh::util::ReadJoints as J;
                match it {
                    J::U8(iter) => iter
                        .map(|j| [j[0] as u16, j[1] as u16, j[2] as u16, j[3] as u16])
                        .collect(),
                    J::U16(iter) => iter.collect(),
                }
            });
            let weights: Option<Vec<[f32; 4]>> = reader.read_weights(0).map(|it| {
                use gltf::mesh::util::ReadWeights as W;
                match it {
                    W::U8(iter) => iter
                        .map(|w| {
                            [
                                w[0] as f32 / 255.0,
                                w[1] as f32 / 255.0,
                                w[2] as f32 / 255.0,
                                w[3] as f32 / 255.0,
                            ]
                        })
                        .collect(),
                    W::U16(iter) => iter
                        .map(|w| {
                            [
                                w[0] as f32 / 65535.0,
                                w[1] as f32 / 65535.0,
                                w[2] as f32 / 65535.0,
                                w[3] as f32 / 65535.0,
                            ]
                        })
                        .collect(),
                    W::F32(iter) => iter.collect(),
                }
            });
            let (Some(joints), Some(weights)) = (joints, weights) else {
                continue; // unskinned: the static importer owns it
            };
            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .map(|it| it.collect())
                .unwrap_or_default();
            if positions.is_empty() {
                continue;
            }
            let skin = doc.skins().next();
            let (inverse_bind, node_to_joint) = match skin {
                Some(s) => {
                    let ibm: Vec<Mat4> = s
                        .reader(|b| buffers.get(b.index()).map(|d| d.0.as_slice()))
                        .read_inverse_bind_matrices()
                        .map(|it| it.map(|m| Mat4::from_cols_array_2d(&m)).collect())
                        .unwrap_or_default();
                    let map: HashMap<usize, usize> = s
                        .joints()
                        .enumerate()
                        .map(|(slot, node)| (node.index(), slot))
                        .collect();
                    let count = s.joints().len().max(1);
                    let ibm = if ibm.len() >= count {
                        ibm
                    } else {
                        let mut full = ibm;
                        full.resize(count, Mat4::IDENTITY);
                        full
                    };
                    (ibm, map)
                }
                None => (vec![Mat4::IDENTITY], HashMap::new()),
            };
            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|it| it.collect())
                .unwrap_or_default();
            let uvs: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|it| it.into_f32().collect())
                .unwrap_or_default();
            let indices: Vec<u32> = match reader.read_indices() {
                Some(gltf::mesh::util::ReadIndices::U8(it)) => it.map(u32::from).collect(),
                Some(gltf::mesh::util::ReadIndices::U16(it)) => it.map(u32::from).collect(),
                Some(gltf::mesh::util::ReadIndices::U32(it)) => it.collect(),
                None => (0..positions.len() as u32).collect(),
            };
            let bc = prim.material().pbr_metallic_roughness().base_color_factor();
            let tint = [bc[0], bc[1], bc[2]];
            out.push(SkinnedMesh {
                name: format!("skin_{}", mesh.index()),
                colors: vec![tint; positions.len()],
                positions,
                normals,
                uvs: uvs.iter().map(|[u, v]| [*u, 1.0 - *v]).collect(),
                indices,
                joints,
                weights,
                inverse_bind,
                node_to_joint,
                texture_page: 0,
                pick_id: 0,
                depth_test: true,
            });
        }
    }
    Ok(out)
}

/// Sample every animation track in `bytes` into node-indexed [`NodePose`]s.
pub fn import_animations(bytes: &[u8]) -> Result<Vec<Animation>, gltf::Error> {
    let (doc, buffers, _) = gltf::import_slice(bytes)?;
    let mut out = Vec::new();
    for anim in doc.animations() {
        let name = anim.name().unwrap_or("anim").to_string();
        let mut tracks: HashMap<usize, Vec<(f32, JointPose, u8)>> = HashMap::new();
        for channel in anim.channels() {
            let target = channel.target();
            let node = target.node().index();
            let reader = channel.reader(|b| buffers.get(b.index()).map(|d| d.0.as_slice()));
            let inputs: Vec<f32> = reader
                .read_inputs()
                .map(|it| it.collect())
                .unwrap_or_default();
            if inputs.is_empty() {
                continue;
            }
            use gltf::animation::util::ReadOutputs as O;
            match reader.read_outputs() {
                Some(O::Translations(it)) => {
                    let vals: Vec<[f32; 3]> = it.collect();
                    for (t, v) in inputs.into_iter().zip(vals) {
                        tracks.entry(node).or_default().push((
                            t,
                            JointPose {
                                translation: v,
                                ..Default::default()
                            },
                            0,
                        ));
                    }
                }
                Some(O::Rotations(rot)) => {
                    let vals: Vec<[f32; 4]> = rot.into_f32().collect();
                    for (t, v) in inputs.into_iter().zip(vals) {
                        tracks.entry(node).or_default().push((
                            t,
                            JointPose {
                                rotation: v,
                                ..Default::default()
                            },
                            1,
                        ));
                    }
                }
                Some(O::Scales(it)) => {
                    let vals: Vec<[f32; 3]> = it.collect();
                    for (t, v) in inputs.into_iter().zip(vals) {
                        tracks.entry(node).or_default().push((
                            t,
                            JointPose {
                                scale: v,
                                ..Default::default()
                            },
                            2,
                        ));
                    }
                }
                _ => {}
            }
        }
        let mut poses: HashMap<usize, NodePose> = HashMap::new();
        for (node, keys) in tracks {
            let mut pose = NodePose::default();
            for (t, j, ch) in keys {
                match pose.times.iter().position(|&e| (e - t).abs() < 1e-9) {
                    Some(i) => match ch {
                        0 => {
                            if pose.translations.len() == pose.times.len() {
                                pose.translations[i] = j.translation;
                            }
                        }
                        1 => {
                            if pose.rotations.len() == pose.times.len() {
                                pose.rotations[i] = j.rotation;
                            }
                        }
                        _ => {
                            if pose.scales.len() == pose.times.len() {
                                pose.scales[i] = j.scale;
                            }
                        }
                    },
                    None => {
                        pose.times.push(t);
                        let last_t = pose.translations.last().copied().unwrap_or([0.0, 0.0, 0.0]);
                        let last_r = pose
                            .rotations
                            .last()
                            .copied()
                            .unwrap_or([0.0, 0.0, 0.0, 1.0]);
                        let last_s = pose.scales.last().copied().unwrap_or([1.0, 1.0, 1.0]);
                        pose.translations
                            .push(if ch == 0 { j.translation } else { last_t });
                        pose.rotations
                            .push(if ch == 1 { j.rotation } else { last_r });
                        pose.scales.push(if ch == 2 { j.scale } else { last_s });
                    }
                }
            }
            let mut order: Vec<usize> = (0..pose.times.len()).collect();
            order.sort_by(|&a, &b| {
                pose.times[a]
                    .partial_cmp(&pose.times[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let gather = |v: &Vec<[f32; 3]>| order.iter().map(|&i| v[i]).collect::<Vec<_>>();
            let gather4 = |v: &Vec<[f32; 4]>| order.iter().map(|&i| v[i]).collect::<Vec<_>>();
            pose = NodePose {
                times: order.iter().map(|&i| pose.times[i]).collect(),
                translations: gather(&pose.translations),
                rotations: gather4(&pose.rotations),
                scales: gather(&pose.scales),
            };
            poses.insert(node, pose);
        }
        out.push(Animation {
            name,
            tracks: poses,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two-vertex "limb": vert 0 bound to joint 0, vert 1 split 50/50.
    fn limb() -> SkinnedMesh {
        SkinnedMesh {
            name: "limb".into(),
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            normals: vec![[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            uvs: vec![],
            colors: vec![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            indices: vec![0, 1, 0],
            joints: vec![[0, 0, 0, 0], [0, 1, 0, 0]],
            weights: vec![[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]],
            inverse_bind: vec![Mat4::IDENTITY, Mat4::IDENTITY],
            node_to_joint: HashMap::from([(10, 0), (11, 1)]),
            texture_page: 0,
            pick_id: 3,
            depth_test: true,
        }
    }

    #[test]
    fn identity_pose_holds_bind() {
        let g = limb().pose(&[Mat4::IDENTITY, Mat4::IDENTITY]);
        assert_eq!(g.positions, vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        assert_eq!(g.normals.len(), 2);
        assert_eq!(g.pick_id, 3);
    }

    #[test]
    fn joint_translation_blends_by_weight() {
        let joints = [
            Mat4::IDENTITY,
            Mat4::from_translation(Vec3::new(2.0, 0.0, 0.0)),
        ];
        let g = limb().pose(&joints);
        assert!((g.positions[0][0] - 0.0).abs() < 1e-6);
        assert!(
            (g.positions[1][0] - 2.0).abs() < 1e-6,
            "{:?}",
            g.positions[1]
        );
    }

    #[test]
    fn unweighted_verts_hold_bind_pose() {
        let mut m = limb();
        m.weights[0] = [0.0; 4];
        let joints = [Mat4::from_translation(Vec3::new(9.0, 9.0, 9.0)); 2];
        let g = m.pose(&joints);
        assert_eq!(g.positions[0], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn out_of_range_joints_clamp_instead_of_panic() {
        let mut m = limb();
        m.joints[1] = [9, 9, 9, 9];
        let g = m.pose(&[Mat4::IDENTITY, Mat4::IDENTITY]);
        assert_eq!(g.positions.len(), 2);
    }

    #[test]
    fn node_pose_samples_and_clamps() {
        let pose = NodePose {
            times: vec![0.0, 1.0],
            translations: vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            rotations: vec![],
            scales: vec![],
        };
        assert_eq!(pose.sample(-1.0).translation, [0.0, 0.0, 0.0]);
        assert_eq!(pose.sample(0.5).translation, [1.0, 0.0, 0.0]);
        assert_eq!(pose.sample(9.0).translation, [2.0, 0.0, 0.0]);
        assert_eq!(pose.duration(), 1.0);
        assert!(NodePose::default().is_empty());
    }

    #[test]
    fn joint_pose_matrix_composes_trs() {
        let p = JointPose {
            translation: [1.0, 0.0, 0.0],
            ..Default::default()
        };
        let v = p.matrix().transform_point3(Vec3::ZERO);
        assert!((v - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn gnome_skin_and_anims_evaluate() {
        let path = "/home/ymsr/Downloads/pilot-garden-linux/assets/gnome.glb";
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("SKIP gnome test (no fixture): {e}");
                return;
            }
        };
        let skinned = import_skinned(&bytes).expect("gnome parses");
        assert_eq!(skinned.len(), 1);
        let m = &skinned[0];
        assert_eq!(m.joint_count(), 9);
        let id = vec![Mat4::IDENTITY; m.joint_count()];
        let bind = m.pose(&id);
        assert_eq!(bind.tri_count(), 715);
        let anims = import_animations(&bytes).expect("gnome anims parse");
        assert_eq!(anims.len(), 4);
        let a = &anims[0];
        let mut joints = vec![Mat4::IDENTITY; m.joint_count()];
        for (node, track) in &a.tracks {
            if let Some(&slot) = m.node_to_joint.get(node) {
                joints[slot] = track.sample(a.duration() / 2.0).matrix();
            }
        }
        let posed = m.pose(&joints);
        let moved = posed
            .positions
            .iter()
            .zip(bind.positions.iter())
            .filter(|(a, b)| {
                let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
                dx * dx + dy * dy + dz * dz > 1e-10
            })
            .count();
        assert_eq!(moved, posed.positions.len(), "anim moves the mesh");
    }
}
