//! 3D viewport: snapshot in, pixels out.
//!
//! Games build a [`Frame3d`] per frame and mount [`Viewport3d`] as a Repose view.
//! The batch draws mesh groups with depth testing and reports orbit and pick events.
//! Camera state lives in game signals. Groups carry verts, indices, tint, normals, uvs.
//!
//! Beyond the mesh path the frame can carry:
//!
//! * [`ParticleFrame`] - GPU particle emitters ([`particles`]): a compute pass
//!   spawns/integrates/collides/sub-emits, and camera-facing billboards draw
//!   through the same depth buffer and pass chain as the scene.
//! * [`ProbeSettings`] - reflection probes ([`reflection`]): a cube captured
//!   at a point and sampled by the material path with a falloff volume.
//! * [`OcclusionCulling`] - hierarchical-Z occlusion from the previous frame's
//!   depth ([`occlusion`]), applied before the frustum phase. The viewport
//!   drives the readback itself with [`OcclusionFeedDesc`] (armed by default on
//!   frames that declare a depth pass), so games need no feed of their own.
//!   The feed is GPU work and exists only where [`GpuOcclusion`] says the
//!   device can run it; a device without it keeps the host-side path, a
//!   pyramid the game stages itself.
//! * [`PostProcess`] - bloom, DOF, camera motion blur, FXAA and ray-marched
//!   volumetric fog ([`postfx`]).
//! * [`LodGroup`] - distance/hysteresis mesh LOD ([`lod`]), with coarse voxel
//!   chunks from [`build_chunk_mesh_lod`].
//! * [`TextureResidency`] - streamed texture mip residency ([`texture_stream`]):
//!   a byte budget, desired levels from the projected on-screen size, level
//!   hysteresis, and background level loads whose stale arrivals drop exactly
//!   like stale chunk builds. The frame carries the drain
//!   ([`Frame3d::mip_uploads`]) and the resolved per-page levels
//!   ([`Frame3d::publish_page_level`]), so a streaming game needs no direct
//!   [`SceneBatch`] access.
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
pub mod gizmos;
pub mod gltf;
pub mod lod;
pub mod material;
pub mod mesh;
pub mod occlusion;
pub mod particles;
pub mod pass;
pub mod pick;
pub mod postfx;
pub mod reflection;
pub mod render;
pub mod shading;
pub mod shadow;
pub mod skin;
pub mod stats;
pub mod stream;
pub mod tev;
pub mod texfmt;
pub mod texture_stream;
pub mod textures;
pub mod vertex;
pub mod viewport;
pub mod voxel;

pub use camera::{
    AabbColliders, CameraAttributes, CameraCollision, DampValue, FAR, NEAR, NoCollision,
    OPENGL_TO_WGPU, OrbitCamera, Smoothed, SmoothedCamera, SpringArm, SpringArmState,
    exp_decay, exp_decay_alpha, ray_aabb_entry, wrap_pi,
};
pub use chunk::{ChunkCache, ChunkDraw, ChunkEntry, validate_group};
pub use gizmos::{AXIS_X, AXIS_Y, AXIS_Z, GizmoColor, Gizmos, WHITE, YELLOW};
pub use gltf::{
    ImportSkip, ImportedMesh, alpha_mode, fan_to_list, flatten_imported, import_slice, material_of,
    strip_to_list,
};
pub use lod::{LodError, LodGroup, LodSelector, MAX_LOD_LEVELS, select_level};
pub use material::{
    DEFAULT_KCOLORS, KColors, Light, MAX_KCOLORS, MAX_LIGHTS, MAX_TEV_STAGES, MAX_TEX_UNITS,
    MaterialFog, Rgba, ShadingError, ShadingModel, TevArg, TevBias, TevDest, TevMode, TevOp,
    TevScale, TevStage,
};
pub use mesh::{Material, MeshGroup, Rgb, shade, shade_for_dir};
pub use occlusion::{
    DEFAULT_BIAS, DEFAULT_TILE, DepthPyramid, DepthSource, GpuOcclusion, MAX_TILE, OcclusionCulling,
    OcclusionFeedDesc, aabb_ndc, decode_f16, nearest_view_depth,
};
pub use pass::{
    MAX_PASS_TARGETS, PassError, PassKind, RenderPass, RenderTarget, RenderTargetFormat,
    TargetSemantics, pass_order, validate_passes,
};
pub use particles::{
    Attractor, CollisionMode, EmitterShape, MAX_ATTRACTORS, MAX_COLLIDER_BOXES, MAX_CURVE_KEYS,
    MAX_REQUEST_RING, PARTICLE_BILLBOARD_WGSL, PARTICLE_BYTES, PARTICLE_COMPUTE_WGSL, Particle,
    ParticleColliders, ParticleCurve, ParticleDisplay, ParticleError, ParticleFrame,
    ParticleGradient, ParticleSystem, ParticleVoxels, SIMULATE_WORKGROUP, SUB_REQUEST_BYTES,
    SubEmitter, SubEmitterMode, bounce_velocity, curve_sample, gradient_sample, hash_u32,
    integrate_step, random01, sample_cone, segment_box, segment_voxels,
};
pub use pick::{MeshHit, group_bounds, pick_ray, pick_screen, ray_aabb, ray_triangle};
pub use postfx::{
    AntiAliasing, BloomSettings, DEPTH_TARGET, DofSettings, FxaaSettings, MAX_BLOOM_LEVELS,
    MotionBlurSettings, PostChain, PostEffect, PostError, PostProcess, PostStage, SCENE_TARGET,
    VolumetricFogSettings, bloom_pyramid, coc_radius, fxaa_direction, fxaa_edge_applies, fxaa_luma,
};
pub use reflection::{
    MAX_PROBES, MAX_PROBE_RESOLUTION, MIN_PROBE_RESOLUTION, ProbeError, ProbeSettings,
    probe_weight,
};
pub use render::{
    BatchDesc, LightRig, MAX_RESIDENT_PAGES, MipUpload, SceneBatch, SceneFilter, SceneLight,
    SceneUpload,
};
pub use render::{paint_scene_with_id, prepare_scene_with_id};
pub use shading::{ShadedBatch, ShadedGroup, ShaderCache, ShadingUniform, kcolor_view};
pub use shadow::{
    CascadeDesc, CascadeSlice, PointLight, ShadowDesc, light_view_proj, shadow_extent,
};
pub use shadow::{MAX_CASCADES, MAX_POINTS, MAX_SKIN_JOINTS};
pub use skin::{
    Animation, GpuSkinning, Interp, JointPose, MorphSet, MorphTrack, NodePose, SkeletalAdvance,
    Skeleton, SkeletonLoop, SkeletonPlayer, SkinnedDraw, SkinnedMesh, attach_to_joint,
    import_animations, import_morphs, import_skeleton, import_skinned,
};
pub use stats::{FrameStats, LatestFrame, RollingAverage, StatsSink};
pub use stream::{ChunkBuild, ChunkPriority, ChunkStreamer, StreamPlan, plan_stream};
pub use texture_stream::{
    MAX_MIP_LEVELS, MipArrival, MipEviction, MipLevelDesc, MipPayload, MipPriority, MipRequest,
    MipStreamer, ProjectedSize, StreamError, StreamTuning, TextureChain, TextureResidency,
    TextureStreamPlan, TextureStreamStats, desired_level, project_bounds, select_mip_level,
};
pub use tev::{TexInput, evaluate, evaluate_lights, evaluate_stages, fog_factor, wgsl_program};
pub use texfmt::{
    Palette, PaletteBits, TexError, TexFormat, TextureOverrides, TextureSource, decode_palette,
    decode_texels, mip_chain, palettize_rgba, resample_half, rgba_from_texel, texel_from_rgba,
};
pub use textures::{
    ImageSkip, PlacedPage, TextureImage, TexturedImport, decode_document_images,
    decode_image_bytes, decode_slice_images, import_slice_textured,
};
pub use vertex::{
    IndexFormat, LayoutError, VertexAttribute, VertexFormat, VertexLayout, VertexSemantic,
};
pub use viewport::{
    CLICK_SLOP_PX, Frame3d, GeomHandle, PageLevel, View3dEvent, Viewport3d, ViewportGeom,
};
pub use voxel::{
    CHUNK_SIZE, Cell, ChunkMeshInput, ChunkMeshOutput, DIRS, FaceKind, THIN_HEIGHT, VoxelShape,
    VoxelSource, build_chunk_mesh, build_chunk_mesh_lod, neighbor_occludes, world_dir_to_local,
};
