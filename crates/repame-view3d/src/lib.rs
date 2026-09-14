//! 3D viewport: snapshot in, pixels out.
//!
//! Long-term companion to `repame-sprite`: games build a [`Frame3d`] per
//! frame (plain data, cheap to rebuild during composition) and mount
//! [`Viewport3d`] as a Repose view, which draws the mesh snapshot through
//! a depth-tested wgpu pass and reports orbit gestures + ground picks back.
//! Camera state lives in game signals, never in the renderer.
//!
//! Scope (full 3D, deeply — lands behind [`MeshGroup`] / [`Frame3d`]):
//! flat-shaded and single-light lit indexed meshes with a real GPU depth
//! buffer, an orbit camera, ground-plane picking, and CPU mesh picking
//! (ray vs groups through the same camera; groups opt in with
//! [`MeshGroup::pick_id`]). Textures plug into
//! the same path: mesh groups carry uvs + one array page, and the batch
//! owns the texture array (fed from per-frame uploads) — tint, texture,
//! and light compose in that order. glTF static import ([`gltf`]) and CPU
//! skinning + animation tracks ([`skin`]) emit the same groups, so imported
//! scenes compose with procedural ones; the chunk mesher plugs in behind
//! [`ChunkCache`] / [`Frame3d`]. The renderer only ever sees
//! vertex/index/tint/normal/uv lists, so the GPU path stays stable while
//! the asset side grows. The resims `resims-view3d` starter scene (CPU
//! painter sort, no depth) is the reference producer, not a dependency.
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
pub mod viewport;

pub use camera::{FAR, NEAR, OPENGL_TO_WGPU, OrbitCamera};
pub use chunk::{ChunkCache, ChunkDraw, ChunkEntry};
pub use gltf::{
    ImportSkip, ImportedMesh, fan_to_list, flatten_imported, import_slice, strip_to_list,
};
pub use mesh::{MeshGroup, Rgb, shade, shade_for_dir};
pub use pick::{MeshHit, group_bounds, pick_ray, pick_screen, ray_aabb, ray_triangle};
pub use render::{BatchDesc, SceneBatch, SceneFilter, SceneLight, SceneUpload};
pub use render::{paint_scene_with_id, prepare_scene_with_id};
pub use skin::{
    Animation, Interp, JointPose, MorphSet, MorphTrack, NodePose, SkeletalAdvance, Skeleton,
    SkeletonLoop, SkeletonPlayer, SkinnedMesh, import_animations, import_morphs, import_skeleton,
    import_skinned,
};
pub use viewport::{CLICK_SLOP_PX, Frame3d, GeomHandle, View3dEvent, Viewport3d, ViewportGeom};
