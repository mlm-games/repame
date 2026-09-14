//! 3D viewport: snapshot in, pixels out.
//!
//! Long-term companion to `repame-sprite`: games build a [`Frame3d`] per
//! frame (plain data, cheap to rebuild during composition) and mount
//! [`Viewport3d`] as a Repose view, which draws the mesh snapshot through
//! a depth-tested wgpu pass and reports orbit gestures + ground picks back.
//! Camera state lives in game signals, never in the renderer.
//!
//! Scope today (deliberately narrow — this crate grows slowly):
//! flat-shaded and single-light lit indexed meshes with a real GPU depth
//! buffer, an orbit camera, and ground-plane picking. Textures plug into
//! the same path: mesh groups carry uvs + one array page, and the batch
//! owns the texture array (fed from per-frame uploads) — tint, texture,
//! and light compose in that order. No glTF/skinning, no chunk mesher
//! yet: those plug in behind [`MeshGroup`] / [`Frame3d`], which the
//! renderer only ever sees as vertex/index/tint/normal/uv lists, so the
//! GPU path stays stable while the asset side grows. The resims
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
pub mod mesh;
pub mod render;
pub mod viewport;

pub use camera::{FAR, NEAR, OPENGL_TO_WGPU, OrbitCamera};
pub use mesh::{MeshGroup, Rgb, shade, shade_for_dir};
pub use render::{BatchDesc, SceneBatch, SceneFilter, SceneLight, SceneUpload};
pub use render::{paint_scene_with_id, prepare_scene_with_id};
pub use viewport::{CLICK_SLOP_PX, Frame3d, GeomHandle, View3dEvent, Viewport3d, ViewportGeom};
