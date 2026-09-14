//! 3D viewport: snapshot in, pixels out.
//!
//! Games build a [`Frame3d`] per frame and mount [`Viewport3d`] as a Repose view.
//! The batch draws mesh groups with depth testing and reports orbit and pick events.
//! Camera state lives in game signals. Groups carry verts, indices, tint, normals, uvs.
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
