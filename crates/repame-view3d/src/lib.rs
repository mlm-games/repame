//! 3D viewport: snapshot in, pixels out.
//!
//! Long-term companion to `repame-sprite`: games build a [`Frame3d`] per
//! frame (plain data, cheap to rebuild during composition) and mount
//! [`Viewport3d`] as a Repose view, which draws the mesh snapshot through
//! a depth-tested wgpu pass and reports orbit gestures + mesh/ground picks
//! back. Camera state lives in game signals, never in the renderer.
//!
//! Scope (full 3D, deeply — lands behind [`MeshGroup`] / [`Frame3d`]):
//! flat-shaded and lit indexed meshes (single directional + ambient, with
//! per-group PBR-lite [`Material`](crate::Material): metallic/roughness/
//! emissive) with a real GPU depth buffer, linear distance fog + Reinhard
//! exposure tonemap on the frame light, an orbit camera, ground-plane
//! picking, and CPU mesh picking (ray vs groups through the same camera;
//! groups opt in with [`MeshGroup::pick_id`]). Bone attachments ride
//! [`attach_to_joint`](crate::attach_to_joint) (Godot `BoneAttachment3D`:
//! rigid props fixed to animated joints). Base-color textures decode from
//! embedded buffer views ([`textures`]: PNG/JPEG/WebP by magic bytes, one
//! image per array layer, aspect-preserving downscale only) and link
//! through each
//! primitive's material ([`import_slice_textured`]) — tint, texture,
//! and light compose in that order; groups without pixels keep their tint
//! (never an invisible discard). Transparency is a second pass
//! (alpha-blend, no depth writes, back-to-front after opaque), with a
//! per-group cutoff for MASK-style cutouts; glTF alpha modes resolve via
//! [`alpha_mode`]. Frustum culling drops fully-off-screen depth-tested
//! groups before flattening (`groups_culled` reports the count). glTF
//! static import ([`gltf`]) and CPU skinning + animation tracks ([`skin`])
//! emit the same groups, so imported scenes compose with procedural ones;
//! the chunk mesher ([`voxel`]: greedy Full faces, rotation-aware occlusion,
//! exact shaped fallback, water pass) plugs in behind [`ChunkCache`] /
//! [`Frame3d`]. The renderer only ever sees vertex/index/tint/normal/uv lists, so the GPU
//! path stays stable while the asset side grows. The resims
//! `resims-view3d` starter scene (CPU painter sort, no depth) is the
//! reference producer, not a dependency.
//!
//! ```ignore
//! let mut frame = Frame3d::default();
//! frame.push(ground_group());
//! Viewport3d(frame, GeomHandle::new(), "scene.main", BatchDesc::default(), |ev| match ev {
//!     View3dEvent::Orbit { dx, dy } => cam.update(|c| c.orbit(dx, dy)),
//!     View3dEvent::GroundClick { x, z } => select(x, z),
//!     _ => {}
//! })
//! ```

pub mod camera;
pub mod chunk;
pub mod gltf;
pub mod mesh;
pub mod pick;
pub mod render;
pub mod skin;
pub mod textures;
pub mod viewport;
pub mod voxel;

pub use camera::{FAR, NEAR, OPENGL_TO_WGPU, OrbitCamera};
pub use chunk::{ChunkCache, ChunkDraw, ChunkEntry, validate_group};
pub use gltf::{
    ImportSkip, ImportedMesh, alpha_mode, fan_to_list, flatten_imported, import_slice, material_of,
    strip_to_list,
};
pub use mesh::{Material, MeshGroup, Rgb, shade, shade_for_dir};
pub use pick::{MeshHit, group_bounds, pick_ray, pick_screen, ray_aabb, ray_triangle};
pub use render::{BatchDesc, SceneBatch, SceneFilter, SceneLight, SceneUpload};
pub use render::{paint_scene_with_id, prepare_scene_with_id};
pub use skin::{
    Animation, Interp, JointPose, MorphSet, MorphTrack, NodePose, SkeletalAdvance, Skeleton,
    SkeletonLoop, SkeletonPlayer, SkinnedMesh, attach_to_joint, import_animations, import_morphs,
    import_skeleton, import_skinned,
};
pub use textures::{
    ImageSkip, PlacedPage, TextureImage, TexturedImport, decode_document_images,
    decode_image_bytes, decode_slice_images, import_slice_textured,
};
pub use viewport::{CLICK_SLOP_PX, Frame3d, GeomHandle, View3dEvent, Viewport3d, ViewportGeom};
pub use voxel::{
    CHUNK_SIZE, Cell, ChunkMeshInput, ChunkMeshOutput, DIRS, FaceKind, THIN_HEIGHT, VoxelShape,
    VoxelSource, build_chunk_mesh, neighbor_occludes, world_dir_to_local,
};
