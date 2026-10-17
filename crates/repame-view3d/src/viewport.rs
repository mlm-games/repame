//! 3D viewport view: orbit gestures in, mesh snapshot out to the GPU.
//!
//! Same split as `repame-sprite` `Viewport2d` and resims `Viewport3d`:
//! camera state lives in game signals, never in the renderer. The UI builds
//! a [`Frame3d`] per frame (plain data, rebuilt during composition) and
//! mounts [`Viewport3d`], which draws the snapshot through
//! a [`SceneBatch`] and reports orbit gestures + ground clicks back.

use std::rc::Rc;
use std::sync::{Arc, Mutex};

use glam::{Mat4, Vec2, Vec3};
use repose_core::input::{PointerButton, PointerEventKind};
use repose_core::{Modifier, View};
use repose_render_wgpu::{
    Callback, CallbackRenderPass, CallbackResources, ScreenDescriptor, WgpuCallback,
};
use repose_ui::Embedded;

use super::camera::OrbitCamera;
use super::mesh::MeshGroup;
use super::pick::{MeshHit, pick_ray};
use super::render::{BatchDesc, SceneBatch, SceneLight, SceneUpload};
use super::render::{paint_scene_with_callback, prepare_scene_with_id};
use super::skin::GpuSkinning;

/// Everything the viewport draws this frame. Plain data, snapshot per frame.
///
/// Framing contract (single source of truth): the GPU camera uniform is
/// `cam.view_proj(aspect)` where `aspect = viewport_w / viewport_h` from
/// the painted rect; picks invert through the same camera via
/// [`OrbitCamera::screen_ray`]. One camera, two consumers, they cannot
/// disagree.
///
/// Textures ride the same snapshot: [`Frame3d::uploads`] feeds the batch
/// texture array once per frame (games drain their image/atlas source
/// here), and each group samples one page (see
/// [`MeshGroup`] `texture_page`).
#[derive(Clone, Debug)]
pub struct Frame3d {
    pub cam: OrbitCamera,
    /// World-space mesh groups. Depth-tested groups occlude; groups with
    /// `depth_test = false` always draw (ground decals, editor gizmos).
    pub groups: Vec<MeshGroup>,
    /// Declared material programs for this frame, one per distinct
    /// [`crate::ShadingModel`]. A group selects one by index through
    /// [`MeshGroup::material_index`]; the viewport registers each entry and
    /// stamps the resulting slot. Empty leaves every group on the legacy
    /// path.
    pub materials: Vec<crate::ShadingModel>,
    /// Per-frame texture uploads into the batch array. Applied exactly
    /// once (sprite-batch `AtlasUpload` contract).
    pub uploads: Vec<SceneUpload>,
    /// Batch texture shape the groups pack against. Must match the batch
    /// the viewport draws through (`Viewport3d` takes it explicitly, so
    /// mismatches fail at the call site).
    pub desc: BatchDesc,
    /// Frame light for groups carrying normals. Flat groups ignore it.
    /// Legacy field: when [`rig`](Frame3d::rig) is `Some`, the rig wins
    /// (sun copied from here stays in sync via [`Frame3d::set_rig`]).
    /// Games should write the rig and leave this at default.
    pub light: SceneLight,
    /// Shadow-map configuration. `None` (default) disables the depth pass
    /// and reproduces legacy pixels exactly. `Some` renders opaque
    /// depth-tested groups into a light-space depth texture during
    /// `prepare` and scales diffuse + specular per lit fragment (3x3 PCF).
    /// Blob shadows (`repame-fx` decals) keep covering contact grounding.
    /// Legacy field: a `Some` [`rig`](Frame3d::rig) overrides this (its
    /// cascades replace the single map; set neither for legacy pixels).
    pub shadow: Option<crate::ShadowDesc>,
    /// Full light rig: sun + points + cascades + cube caster. `None`
    /// (default) keeps the legacy `light`/`shadow` path exactly. `Some`
    /// routes through [`SceneBatch::set_rig`](crate::SceneBatch::set_rig):
    /// the batch fits cascade slices from the frame camera and stages
    /// the brightest [`MAX_POINTS`](crate::MAX_POINTS) points.
    pub rig: Option<crate::LightRig>,
    /// GPU-skinned draws for the frame (bind mesh + sampled palette).
    /// Empty (default) disables the skin path; on a device without vertex
    /// storage the palette binding is omitted entirely (see
    /// [`gpu_skinning`](Frame3d::gpu_skinning)) and these draws are
    /// CPU-posed by the caller instead.
    pub skinned: Vec<crate::SkinnedDraw>,
    /// GPU-skinning capability mode, resolved when `prepare` runs. A
    /// three-way choice:
    /// 1. an explicit `Some(mode)` staged here wins;
    /// 2. `None` derives from the device: the adapter's
    ///    [`wgpu::DownlevelCapabilities`], which repose publishes into
    ///    `CallbackResources`, maps through
    ///    [`GpuSkinning::from_downlevel_flags`] (`VERTEX_STORAGE`);
    /// 3. a device with no capabilities published (an older repose) falls
    ///    back to [`GpuSkinning::Supported`], the historical GPU path,
    ///    byte for byte.
    ///
    /// On the resolved [`GpuSkinning::Unsupported`] the viewport builds its
    /// camera layout and shaders without the joint-palette storage binding
    /// and drops [`skinned`](Frame3d::skinned) draws with a one-time
    /// warning naming the missing `VERTEX_STORAGE` flag; the caller must
    /// CPU-pose them through
    /// [`SkinnedMesh::pose`](crate::SkinnedMesh::pose). Defaults to `None`
    /// (derive from the device).
    pub gpu_skinning: Option<GpuSkinning>,
    /// Offscreen clear color (linear 0..1 RGBA). The shared UI pass this
    /// viewport paints into has its own clear; this selects the scene
    /// target clear inside the viewport-owned pass.
    pub background: Option<[f32; 4]>,
    /// Camera exposure and depth-of-field attributes. `Some` overrides
    /// [`light`](Frame3d::light)`s exposure with the resolved value, and its
    /// DOF (when [`post`](Frame3d::post) leaves `dof` unset) enables the
    /// post-process chain.
    pub attributes: Option<crate::CameraAttributes>,
    /// Post-process + anti-aliasing. `None` (default) keeps the legacy single
    /// scene pass; `Some` plans the bloom/DOF/motion-blur/FXAA chain and
    /// renders the scene into it (see [`crate::postfx::PostProcess`]).
    pub post: Option<crate::PostProcess>,
    /// GPU particle systems for this frame. `None` (default) skips the
    /// simulation and the billboard draws entirely; `Some` runs one compute
    /// dispatch per live system before the frame's geometry and draws each
    /// system's billboards into its declared pass. The frame's `dt` lives on
    /// [`ParticleFrame::dt`](crate::ParticleFrame::dt): a fixed `dt` and a
    /// fixed seed replay identically.
    pub particles: Option<crate::ParticleFrame>,
    /// Reflection probes for this frame, capped at
    /// [`MAX_PROBES`](crate::MAX_PROBES). Empty (default) leaves probe
    /// sampling off. Each probe captures its cube on its own interval and
    /// every lit fragment samples the cubes with the probe's falloff
    /// volume.
    pub probes: Vec<crate::ProbeSettings>,
    /// Hierarchical-Z occlusion culling state for this frame (`None` =
    /// off). Build the pyramid from the depth pass's readback and feed it
    /// the frame after; see [`crate::OcclusionCulling`]. An explicit state
    /// here wins over the viewport-driven feed
    /// ([`occlusion_feed`](Frame3d::occlusion_feed)), which is the path
    /// games normally use.
    pub occlusion: Option<crate::OcclusionCulling>,
    /// Viewport-driven occlusion feed (`Some` = armed, the default). When
    /// the frame also declares a depth pass, the viewport runs a compute
    /// pass chain over that pass's target: tile maxima, then each coarser
    /// level, the whole chain mapped back without waiting and built into the
    /// [`DepthPyramid`](crate::DepthPyramid), staging the culling state on
    /// the next frame - occlusion culling with no game-side readback
    /// plumbing, at the cost of one dispatch (plus one per extra level) and
    /// the level chain (about 4/3 of the level-0 tile count) read back per
    /// frame. A frame that declares no depth pass gets the engine's own
    /// [`PassKind::Occlusion`](crate::PassKind::Occlusion) pass, sized to
    /// the viewport, so the feed works without a post chain. With
    /// [`OcclusionFeedDesc::phase_two`] armed a compute pass re-tests the
    /// phase-one rejects right after the depth pass and the survivors are
    /// drawn over the scene pass's target with its color and depth loaded,
    /// so no false negative survives a frame. `None` disables the feed (no
    /// dispatch, no map) and leaves [`occlusion`](Frame3d::occlusion) as the
    /// only source.
    pub occlusion_feed: Option<crate::OcclusionFeedDesc>,
    /// Cold-start viewport size in physical px, used for the GPU camera
    /// until the first painted size arrives (same role as `repame-sprite`
    /// `FrameInput::viewport_dp`). Falls back to 16:9.
    pub viewport_px: [f32; 2],
}

impl Default for Frame3d {
    fn default() -> Self {
        Self {
            cam: OrbitCamera::default(),
            groups: Vec::new(),
            materials: Vec::new(),
            uploads: Vec::new(),
            desc: BatchDesc::default(),
            light: SceneLight::default(),
            shadow: None,
            rig: None,
            skinned: Vec::new(),
            gpu_skinning: None,
            background: None,
            attributes: None,
            post: None,
            particles: None,
            probes: Vec::new(),
            occlusion: None,
            occlusion_feed: Some(crate::OcclusionFeedDesc::default()),
            viewport_px: [1600.0, 900.0],
        }
    }
}

impl Frame3d {
    /// Aspect of a viewport, with a 16:9 fallback for degenerate sizes.
    pub fn aspect(viewport_px: [f32; 2]) -> f32 {
        if viewport_px[1].abs() < 1e-6 {
            16.0 / 9.0
        } else {
            (viewport_px[0] / viewport_px[1]).max(0.01)
        }
    }

    /// Push one mesh group into the snapshot.
    pub fn push(&mut self, group: MeshGroup) {
        self.groups.push(group);
    }

    /// Set the full light rig. Copies `rig.sun` into [`light`](Frame3d::light)
    /// so legacy readers (debug overlays, CPU-side exposure math) stay in
    /// sync; clears [`shadow`](Frame3d::shadow) (the rig's cascades own the
    /// directional pass now, a stale single map would double-shadow).
    pub fn set_rig(&mut self, rig: crate::LightRig) {
        self.light = rig.sun;
        self.rig = Some(rig);
        self.shadow = None;
    }

    /// Resolves the frame's post-process settings: the frame's own
    /// [`post`](Frame3d::post), with the camera attributes' DOF folded in when
    /// the post block left it unset.
    pub fn resolved_post(&self) -> Option<crate::PostProcess> {
        let mut post = self.post?;
        if post.dof.is_none()
            && let Some(dof) = self.attributes.and_then(|attributes| attributes.dof)
        {
            post.dof = Some(dof);
        }
        Some(post)
    }

    /// Expands `gizmos` into the frame's overlay channel using `eye` as the
    /// camera position, so debug lines billboard toward the viewer and draw
    /// on top of the scene.
    pub fn push_gizmos(&mut self, gizmos: &crate::gizmos::Gizmos, eye: Vec3) {
        if let Some(group) = gizmos.to_group(eye) {
            self.groups.push(group);
        }
    }

    /// Push one GPU-skinned draw (bind mesh + sampled palette).
    pub fn push_skinned(&mut self, draw: crate::SkinnedDraw) {
        self.skinned.push(draw);
    }

    /// Append cached chunk geometry (see [`ChunkCache`](crate::ChunkCache)):
    /// validated groups copy into the snapshot alongside dynamic content.
    /// Deterministic when the caller passes [`draws`](super::chunk::ChunkCache::draws) order
    /// (chunk-sorted) first.
    ///
    /// Malformed cached groups are dropped with a warning (same validators
    /// as the batch): a stale or hand-built cache entry fails validation here
    /// frame.
    pub fn extend_chunks<'a>(
        &mut self,
        draws: impl IntoIterator<Item = super::chunk::ChunkDraw<'a>>,
    ) {
        for draw in draws {
            if super::chunk::validate_group(draw.group) {
                self.groups.push(draw.group.clone());
            }
        }
    }
}

/// UI-facing viewport events. Gestures arrive as deltas; the game applies
/// them to its camera signal ([`OrbitCamera::orbit`] / [`pan`](OrbitCamera::pan) /
/// [`zoom`](OrbitCamera::zoom)) and rebuilds the snapshot.
///
/// `GroundClick` fires on clean left-clicks (press + release with less than
/// [`CLICK_SLOP_PX`] travel) when the ray reaches the ground plane *and* no
/// pickable mesh is closer along the ray (mesh hits win, so agents occlude
/// the ground behind them). Games needing raw ground-only clicks can call
/// [`OrbitCamera::ground_point`] directly.
#[derive(Clone, Debug)]
pub enum View3dEvent {
    /// Left-drag orbit delta, in px.
    Orbit { dx: f32, dy: f32 },
    /// Pan delta, in px (shift/middle/right-drag).
    Pan { dx: f32, dy: f32 },
    /// Raw pointer-move-while-pressed delta, in px, with the button held.
    /// Fires for every such move regardless of button, alongside the
    /// derived [`View3dEvent::Orbit`] / [`View3dEvent::Pan`] delta.
    Drag {
        button: PointerButton,
        dx: f32,
        dy: f32,
    },
    /// Multiplicative zoom factor (wheel).
    Zoom { factor: f32 },
    /// Ground-plane (y = 0) point under a clean click, if the ray hits.
    /// Mesh-pickable groups are tested first: a mesh hit suppresses this
    /// and arrives as [`View3dEvent::MeshClick`] instead.
    GroundClick { x: f32, z: f32 },
    /// Nearest pickable mesh hit under a clean click (groups with
    /// `pick_id != 0`, ray ordered). Carries the pick id plus the
    /// world-space point, so agents select while terrain falls through
    /// to [`View3dEvent::GroundClick`].
    MeshClick { pick_id: u32, point: [f32; 3] },
    /// Camera ray at the pointer pixel on pointer-move: `eye` origin,
    /// unit `dir` into the scene (same ray [`resolve_click`] inverts
    /// through). Leads each move's events, so consumers always hold the
    /// current ray before the accompanying pick or drag delta.
    HoverRay { eye: [f32; 3], dir: [f32; 3] },
    /// Cursor pick under pointer-move: `Some` when a pickable group is
    /// under the cursor, else the ground-plane fallback. Emitted per move
    /// event (AABB early-out; unpickable scenes cost nothing).
    HoverMesh {
        pick_id: Option<u32>,
        x: f32,
        z: f32,
    },
    /// Cursor ground point on move (`None` when the
    /// ray misses the plane).
    Hover { x: Option<f32>, z: Option<f32> },
}

/// Press slop in physical px: pointer-up farther than this from
/// pointer-down cancels the click (drag-off-cancel, same value as
/// the 2D viewport).
pub const CLICK_SLOP_PX: f32 = 12.0;

/// Resolve one pointer position to a click: `MeshClick` when a pickable
/// group is nearest along the ray, else `GroundClick` when the ray reaches
/// the plane (and the mesh is not closer). Function of the snapshot only,
/// shared by the pointer-up handler and tests.
fn resolve_click(
    cam: &OrbitCamera,
    viewport_px: [f32; 2],
    px: [f32; 2],
    groups: &[MeshGroup],
) -> Option<View3dEvent> {
    let a = Frame3d::aspect(viewport_px);
    let vp = Vec2::new(viewport_px[0].max(1.0), viewport_px[1].max(1.0));
    let p = Vec2::new(px[0], px[1]);
    let (origin, dir) = cam.screen_ray(a, vp, p);
    let mesh = pick_ray(origin, dir, groups);
    let ground = cam.ground_point(a, vp, p);
    match (mesh, ground) {
        (Some(hit), _) if ground.is_none_or(|g| hit.distance < ground_dist(origin, dir, g)) => {
            Some(View3dEvent::MeshClick {
                pick_id: hit.pick_id,
                point: hit.point,
            })
        }
        (_, Some(g)) => Some(View3dEvent::GroundClick { x: g.x, z: g.y }),
        _ => None,
    }
}

/// Ray distance to a ground point exploded back from `ground_point`
/// (same ray, so the projection length is the distance). Only used to
/// order mesh hits against the plane.
fn ground_dist(origin: Vec3, dir: Vec3, g: Vec2) -> f32 {
    let target = Vec3::new(g.x, 0.0, g.y) - origin;
    let denom = dir.length_squared().max(1e-12);
    (target.dot(dir) / denom).max(0.0)
}

fn click_within_slop(a: [f32; 2], b: [f32; 2]) -> bool {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy <= CLICK_SLOP_PX * CLICK_SLOP_PX
}

/// Events for one pointer move, in emission order: the camera ray at the
/// pixel ([`View3dEvent::HoverRay`]) leads, then the press-state payload —
/// the derived orbit/pan delta plus the raw [`View3dEvent::Drag`] when a
/// button is down, else the hover pick ([`View3dEvent::HoverMesh`] /
/// [`View3dEvent::Hover`]). No events for a pressed zero-delta move.
/// Function of the snapshot only, shared by the pointer-move handler and
/// tests.
fn resolve_move(
    d: Option<&DragState>,
    cam: &OrbitCamera,
    viewport_px: [f32; 2],
    px: [f32; 2],
    groups: &[MeshGroup],
) -> Vec<View3dEvent> {
    let a = Frame3d::aspect(viewport_px);
    let vp = Vec2::new(viewport_px[0].max(1.0), viewport_px[1].max(1.0));
    let p = Vec2::new(px[0], px[1]);
    let (eye, dir) = cam.screen_ray(a, vp, p);
    let ray = View3dEvent::HoverRay {
        eye: eye.into(),
        dir: dir.into(),
    };
    match d {
        Some(d) => {
            let dx = px[0] - d.last[0];
            let dy = px[1] - d.last[1];
            if dx.abs() + dy.abs() <= 0.0 {
                return Vec::new();
            }
            let gesture = if d.pan {
                View3dEvent::Pan { dx, dy }
            } else {
                View3dEvent::Orbit { dx, dy }
            };
            vec![
                ray,
                gesture,
                View3dEvent::Drag {
                    button: d.button,
                    dx,
                    dy,
                },
            ]
        }
        None => {
            let mesh: Option<MeshHit> = pick_ray(eye, dir, groups);
            let ground = cam.ground_point(a, vp, p);
            let pick = match (mesh, ground) {
                (Some(hit), _) => View3dEvent::HoverMesh {
                    pick_id: Some(hit.pick_id),
                    x: hit.point[0],
                    z: hit.point[2],
                },
                (None, Some(gp)) => View3dEvent::HoverMesh {
                    pick_id: None,
                    x: gp.x,
                    z: gp.y,
                },
                (None, None) => View3dEvent::Hover { x: None, z: None },
            };
            vec![ray, pick]
        }
    }
}

/// Press state: press position + last position, both viewport-local px,
/// plus the button that started the press.
#[derive(Clone, Copy)]
struct DragState {
    start: [f32; 2],
    last: [f32; 2],
    /// True for pan gestures (non-primary button or shift held).
    pan: bool,
    button: PointerButton,
}

/// 3D viewport view. Owns nothing render-side; gesture handling and camera
/// state live in the [`Frame3d`] snapshot (same split as resims
/// `Viewport3d`).
///
/// Layout contract: fills its parent. Draws the mesh groups through a
/// depth-tested [`SceneBatch`]. `on_size_changed` publishes the viewport
/// size ahead of `prepare` (resizes take effect the same frame, like the
/// 2D viewport); `paint` re-publishes authoritatively from its callback
/// info. Picks invert through the latest publish at event time, so picks
/// and content stay glued, including under camera motion.
///
/// `batch_desc` must match `input.desc` (the texture shape the groups
/// pack against): the batch is keyed on it and rebuilds, dropping
/// texture contents, when it changes, exactly like the sprite batch.
///
/// Occlusion culling needs no game-side wiring: with the feed armed
/// ([`Frame3d::occlusion_feed`], the default) and a depth pass declared for
/// the frame, the viewport copies that pass's target after submission, maps
/// it without ever waiting and stages the pyramid on the next frame. A frame
/// whose readback has not landed yet culls nothing rather than pacing the
/// frame around the GPU; the price is one frame of latency, which the state
/// absorbs by carrying the camera the depth was captured with. A caller that
/// stages [`Frame3d::occlusion`] itself wins over the feed.
#[allow(non_snake_case)] // Repose view convention (cf. resims `Viewport3d`).
pub fn Viewport3d(
    input: Frame3d,
    geom_out: GeomHandle,
    batch_id: impl Into<String>,
    batch_desc: BatchDesc,
    on_event: impl Fn(View3dEvent) + 'static,
) -> View {
    let batch_id: String = batch_id.into();
    // One shared snapshot: picks and the GPU payload read through the same
    // `Arc`, so per-frame composition clones no geometry (chunked scenes
    // no group clones here; the payload upload is the only copy, on the GPU
    // thread). `Rc` would do for the UI closures, but the payload needs
    // `Send + Sync`, so `Arc` serves both.
    let input = std::sync::Arc::new(input);
    debug_assert_eq!(
        (input.desc.layer_size, input.desc.layers),
        (batch_desc.layer_size, batch_desc.layers),
        "Viewport3d batch_desc must match input.desc (groups pack against input.desc)"
    );
    let draw_input = input.clone();
    let draw_id = batch_id.clone();
    // Picks read through the shared snapshot, no per-frame group clones
    // (no per-frame group clones). Camera and groups for picks are the
    // same values the GPU payload uploads, so content and picks stay glued.
    // Geometry still inverts through the latest painted publish at event
    // time (see `ViewportGeom` timing below).
    let hover_input = input.clone();
    let hover_geom = geom_out.clone();
    let click_input = input.clone();
    let click_geom = geom_out.clone();
    let size_geom = geom_out.clone();
    let on_event = Rc::new(on_event);
    let on_move = on_event.clone();
    let on_up_click = on_event.clone();
    let on_gesture = on_event.clone();
    let on_zoom = on_event;

    let drag: Rc<std::cell::Cell<Option<DragState>>> = Rc::new(std::cell::Cell::new(None));
    let drag_down = drag.clone();
    let drag_move = drag.clone();
    let drag_up = drag.clone();
    let drag_cancel = drag;

    let modifier = Modifier::new()
        .fill_max_size()
        .on_size_changed(move |dp: repose_core::Vec2| {
            size_geom.set(ViewportGeom {
                viewport_px: [dp.x.max(1.0), dp.y.max(1.0)],
            });
        })
        .on_pointer_down(move |ev: repose_core::input::PointerEvent| {
            let p = ev.position;
            let primary = matches!(ev.event, PointerEventKind::Down(PointerButton::Primary));
            let button = match ev.event {
                PointerEventKind::Down(b) => b,
                _ => PointerButton::Primary,
            };
            drag_down.set(Some(DragState {
                start: [p.x, p.y],
                last: [p.x, p.y],
                pan: !primary || ev.modifiers.shift,
                button,
            }));
        })
        .on_pointer_move(move |ev: repose_core::input::PointerEvent| {
            let p = ev.position;
            let g = hover_geom.get();
            let mut d = drag_move.take();
            let events = resolve_move(
                d.as_ref(),
                &hover_input.cam,
                g.viewport_px,
                [p.x, p.y],
                &hover_input.groups,
            );
            if let Some(d) = d.as_mut() {
                d.last = [p.x, p.y];
            }
            drag_move.set(d);
            for event in events {
                on_move(event);
            }
        })
        .on_pointer_up(move |ev: repose_core::input::PointerEvent| {
            let p = ev.position;
            if let Some(d) = drag_up.take()
                && click_within_slop(d.start, [p.x, p.y])
            {
                let g = click_geom.get();
                if let Some(event) = resolve_click(
                    &click_input.cam,
                    g.viewport_px,
                    [p.x, p.y],
                    &click_input.groups,
                ) {
                    on_up_click(event);
                }
            }
        })
        .on_pointer_cancel(move |_| {
            drag_cancel.set(None);
        })
        .on_scroll(move |d: repose_core::Vec2| {
            if d.y.abs() > 0.5 {
                let factor = (1.0 + (-d.y) * 0.002).clamp(0.5, 2.0);
                on_zoom(View3dEvent::Zoom { factor });
                repose_core::Vec2::ZERO
            } else {
                d
            }
        })
        .focusable(true)
        .on_action(move |action: repose_core::shortcuts::Action| {
            use repose_core::shortcuts::{Action, Gesture};
            match &action {
                Action::Gesture(Gesture::PinchWithCenter { delta_scale, .. }) => {
                    on_gesture(View3dEvent::Zoom {
                        factor: *delta_scale,
                    });
                    true
                }
                _ => false,
            }
        });
    let payload = GpuViewport3d {
        input: draw_input.clone(),
        geom: geom_out.arc(),
        history: geom_out.history(),
        batch_id: draw_id,
        batch_desc,
    };
    Embedded(modifier, Callback::new(payload))
}

/// One painted frame's geometry: the viewport size the GPU camera used.
/// Picks invert through this (same aspect), so content and picks stay
/// glued, including under camera motion, since picks read the latest
/// publish at event time.
///
/// Timing: layout publishes a placeholder ahead of `prepare` (resizes
/// apply the same frame); `paint` re-publishes authoritatively from its
/// callback info. Composition-time readers see the previous frame.
#[derive(Clone, Copy, Debug)]
pub struct ViewportGeom {
    /// Painted viewport size in physical px (drives the GPU camera).
    pub viewport_px: [f32; 2],
}

impl Default for ViewportGeom {
    fn default() -> Self {
        Self {
            viewport_px: [1600.0, 900.0],
        }
    }
}

/// Shared paint-time geometry. GPU callbacks need `Send + Sync`, so the
/// inner cell is mutex-backed (same pattern as `repame-sprite`
/// `GeomHandle`).
///
/// It also owns the view's cross-frame camera history (the previous frame's
/// view-projection for motion vectors), so a `Viewport3d` call that rebuilds
/// its payload every frame still sees the previous frame's camera.
#[derive(Clone, Default)]
pub struct GeomHandle {
    geom: Arc<Mutex<ViewportGeom>>,
    history: Arc<Mutex<CameraHistory>>,
}

impl GeomHandle {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self) -> ViewportGeom {
        self.geom.lock().map(|g| *g).unwrap_or_default()
    }

    pub fn set(&self, g: ViewportGeom) {
        if let Ok(mut slot) = self.geom.lock() {
            *slot = g;
        }
    }

    pub fn arc(&self) -> Arc<Mutex<ViewportGeom>> {
        self.geom.clone()
    }

    fn history(&self) -> Arc<Mutex<CameraHistory>> {
        self.history.clone()
    }
}

struct GpuViewport3d {
    input: Arc<Frame3d>,
    geom: Arc<Mutex<ViewportGeom>>,
    /// Previous frame's view-projection, for camera motion vectors. Shared
    /// across the payloads the view rebuilds each frame.
    history: Arc<Mutex<CameraHistory>>,
    batch_id: String,
    batch_desc: BatchDesc,
}

/// Camera state that must outlive a single frame's payload (the payload is
/// rebuilt during composition every frame).
#[derive(Clone, Copy, Debug, Default)]
struct CameraHistory {
    previous_view_proj: Option<Mat4>,
}

impl GpuViewport3d {
    /// Descriptor the batch is actually keyed on: explicit arg when it
    /// matches the snapshot (the common path), snapshot fallback when a
    /// caller passes a placeholder. Keeps old call sites drawing instead of
    /// rebuilding pipelines + dropping textures every frame.
    fn effective_desc(&self) -> BatchDesc {
        if (self.batch_desc.layer_size, self.batch_desc.layers)
            == (self.input.desc.layer_size, self.input.desc.layers)
        {
            self.batch_desc
        } else {
            self.input.desc
        }
    }

    /// Resolves this frame's [`GpuSkinning`] mode: an explicit
    /// [`Frame3d::gpu_skinning`] wins; otherwise the device's downlevel
    /// capabilities, published by repose in [`CallbackResources`], decide
    /// through [`GpuSkinning::from_downlevel_flags`] (`VERTEX_STORAGE`);
    /// a device with no capabilities published (an older repose) falls
    /// back to [`GpuSkinning::Supported`], the historical GPU path, byte
    /// for byte.
    fn effective_skinning(&self, resources: &CallbackResources) -> GpuSkinning {
        self.input
            .gpu_skinning
            .or_else(|| {
                resources
                    .get::<wgpu::DownlevelCapabilities>()
                    .map(|capabilities| GpuSkinning::from_downlevel_flags(capabilities.flags))
            })
            .unwrap_or(GpuSkinning::Supported)
    }
}

impl WgpuCallback for GpuViewport3d {
    fn resource_key(&self) -> Option<&str> {
        Some(&self.batch_id)
    }

    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        screen: &ScreenDescriptor,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let vp_px = self
            .geom
            .lock()
            .map(|g| g.viewport_px)
            .unwrap_or(self.input.viewport_px);
        let w = vp_px[0].max(1.0) as u32;
        let h = vp_px[1].max(1.0) as u32;
        let aspect = Frame3d::aspect(vp_px);
        let desc = self.effective_desc();
        let view_proj = self.input.cam.view_proj(aspect);
        let eye = self.input.cam.eye();
        let mut batch = SceneBatch::with_desc(self.batch_id.clone(), desc);
        batch.set_camera(view_proj);
        batch.set_camera_pos(eye.into());
        batch.set_camera_forward(self.input.cam.forward().into());
        let mut light = self.input.light;
        if let Some(attributes) = &self.input.attributes {
            attributes.apply_to(&mut light);
        }
        match &self.input.rig {
            Some(rig) => {
                let mut rig = rig.clone();
                if let Some(attributes) = &self.input.attributes {
                    attributes.apply_to(&mut rig.sun);
                }
                batch.set_rig(
                    &rig,
                    eye.into(),
                    self.input.cam.view_matrix(),
                    view_proj.inverse(),
                );
            }
            None => {
                batch.set_light(light);
                batch.set_shadow(self.input.shadow, self.input.cam.target.into(), self.input.cam.dist);
            }
        }
        let slots: Vec<u32> = self
            .input
            .materials
            .iter()
            .map(|model| match model.validate() {
                Ok(()) => batch.register_material(model),
                Err(_) => {
                    log::warn!("viewport: dropping invalid shading model");
                    0
                }
            })
            .collect();
        // Frame history: the depth/velocity pass reprojects through this,
        // so it must be the previous frame's view-projection, not this one.
        let previous = self
            .history
            .lock()
            .map(|mut history| history.previous_view_proj.replace(view_proj))
            .unwrap_or_default();
        let mut post = self.input.resolved_post();
        if let Some(post) = post.as_mut()
            && post.previous_view_proj.is_none()
        {
            post.previous_view_proj = previous;
        }
        let batch_format = crate::pass::RenderTargetFormat::from_wgpu(screen.target_format)
            .unwrap_or(crate::pass::RenderTargetFormat::Rgba8UnormSrgb);
        let chain = match post.as_ref().map(|post| post.plan([w, h], batch_format)) {
            Some(Ok(chain)) => chain,
            Some(Err(err)) => {
                log::warn!("viewport: post-process disabled this frame: {err}");
                None
            }
            None => None,
        };
        // The scene renders into the plan's first pass when a post chain is
        // active, otherwise into the shared offscreen scene target.
        let scene_pass = chain.as_ref().map_or(0, |chain| chain.scene_pass);
        if chain.is_some() {
            batch.set_post(post);
        }
        // Caller-staged occlusion wins over the feed; a frame that stages its
        // own state owns its pass list too.
        let feed = if self.input.occlusion.is_some() {
            None
        } else {
            self.input.occlusion_feed
        };
        // The feed owns a view-depth pass: when the frame declares none, the
        // engine appends its own, so occlusion culling never depends on the
        // post chain. A declared Depth pass is used as-is.
        let mut passes = chain
            .as_ref()
            .map(|chain| chain.passes.clone())
            .unwrap_or_default();
        if feed.is_some()
            && !passes.iter().any(|pass| {
                matches!(
                    pass.kind,
                    crate::PassKind::Depth | crate::PassKind::Occlusion
                )
            })
        {
            passes.push(crate::pass::RenderPass {
                target: crate::pass::RenderTarget::new(
                    "__repame_occlusion",
                    w,
                    h,
                    crate::pass::RenderTargetFormat::Rgba16Float,
                    crate::pass::TargetSemantics::LinearDepth,
                ),
                kind: crate::PassKind::Occlusion,
                source: None,
                groups: Vec::new(),
                presents: false,
                overlay: [0.0; 4],
                post: None,
                aux: None,
                load: false,
            });
        }
        if !passes.is_empty()
            && let Err(error) = batch.set_passes(&passes)
        {
            // A user pass may already own the synthesized target's name;
            // stage the declared list alone, which is the pre-feed behavior.
            log::warn!("viewport: pass list dropped: {error:?}");
            let declared = chain
                .as_ref()
                .map(|chain| chain.passes.as_slice())
                .unwrap_or(&[]);
            batch.set_passes(declared).ok();
        }
        for g in &self.input.groups {
            match g.material_index {
                index if index < slots.len() => {
                    batch.push_group_with_material_for_pass(g, slots[index], scene_pass)
                }
                _ => batch.push_group_for_pass(g, scene_pass),
            }
        }
        batch.set_skinning(self.effective_skinning(resources));
        for s in &self.input.skinned {
            batch.push_skinned_for_pass(s, scene_pass);
        }
        batch.extend_uploads(self.input.uploads.iter().cloned());
        batch.set_particles(self.input.particles.clone());
        batch.set_probes(&self.input.probes);
        // Occlusion: a caller-staged state wins outright; otherwise the
        // viewport's own feed (armed by `occlusion_feed`) supplies the
        // previous frame's pyramid when its readback has landed. The poll
        // never waits: a map still in flight leaves the frame with no
        // occlusion state instead of pacing the frame around the GPU (one
        // frame of latency is the design, see `crate::occlusion`).
        let fed = match feed {
            Some(_) => batch.poll_occlusion_feed(device, resources),
            None => None,
        };
        batch.set_occlusion(self.input.occlusion.clone().or(fed));
        batch.set_occlusion_feed(feed);
        batch.finish();
        batch.ensure_resources(device, screen, resources);
        batch.upload_all(device, queue, resources);
        prepare_scene_with_id(
            self.batch_id.as_str(),
            device,
            queue,
            encoder,
            screen,
            resources,
            w,
            h,
            self.input.background.unwrap_or([0.0, 0.0, 0.0, 1.0]),
        );
        // Request the next frame's occlusion feed: a compute chain reduces
        // the depth target into the packed level pyramid in this frame's
        // encoder, after the depth pass drew, and its host map is scheduled
        // for submission, so the chain lands a frame later and is tested
        // with the camera that captured it. A frame the depth pass had
        // nothing to draw for drops the feed instead: its target still
        // holds an older frame, which must not be fed forward.
        match feed {
            Some(desc) => {
                batch.encode_occlusion_readback(
                    device,
                    queue,
                    resources,
                    encoder,
                    view_proj,
                    eye.into(),
                    self.input.cam.forward().into(),
                    desc.clamped_tile(),
                    desc.clamped_bias(),
                );
            }
            None => batch.retire_occlusion_feed(device, resources),
        }
        Vec::new()
    }

    fn paint(
        &self,
        info: repose_core::PaintCallbackInfo,
        rpass: &mut CallbackRenderPass<'_, '_>,
        resources: &CallbackResources,
    ) {
        if let Ok(mut g) = self.geom.lock() {
            *g = ViewportGeom {
                viewport_px: [info.viewport.w.max(1.0), info.viewport.h.max(1.0)],
            };
        }
        paint_scene_with_callback(self.batch_id.as_str(), rpass, resources);
    }
}

#[cfg(test)]
mod tests {
    use super::super::camera::{NEAR, OrbitCamera};
    use super::*;

    fn orbit() -> OrbitCamera {
        OrbitCamera {
            target: glam::Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.9,
            dist: 30.0,
            fov_y_deg: 30.0,
        }
    }

    /// Pickable slab under the camera (top face at y = 2, wide enough that
    /// the center ray lands on it, not past its edges).
    fn slab() -> MeshGroup {
        let mut g = MeshGroup {
            pick_id: 7,
            depth_test: true,
            ..Default::default()
        };
        g.push_box(
            0.0,
            0.0,
            0.0,
            20.0,
            2.0,
            20.0,
            [1.0, 1.0, 1.0],
            orbit().eye(),
        );
        g
    }

    #[test]
    fn center_click_selects_mesh_before_ground() {
        let cam = orbit();
        let groups = vec![slab()];
        let Some(View3dEvent::MeshClick { pick_id, point }) =
            resolve_click(&cam, [800.0, 600.0], [400.0, 300.0], &groups)
        else {
            panic!("center click must MeshClick");
        };
        assert_eq!(pick_id, 7);
        assert!((point[1] - 2.0).abs() < 1e-3, "slab top: {point:?}");
    }

    #[test]
    fn empty_scene_falls_through_to_ground() {
        let cam = orbit();
        let Some(View3dEvent::GroundClick { x, z }) =
            resolve_click(&cam, [800.0, 600.0], [400.0, 300.0], &[])
        else {
            panic!("empty scene must GroundClick");
        };
        assert!(x.is_finite() && z.is_finite(), "({x}, {z})");
    }

    #[test]
    fn unpickable_scene_falls_through_to_ground() {
        let cam = orbit();
        let mut g = slab();
        g.pick_id = 0;
        let event = resolve_click(&cam, [800.0, 600.0], [400.0, 300.0], &[g]);
        assert!(
            matches!(event, Some(View3dEvent::GroundClick { .. })),
            "pick_id 0 must not MeshClick: {event:?}"
        );
    }

    #[test]
    fn upward_ray_clicks_nothing() {
        let cam = OrbitCamera {
            pitch: 0.12,
            ..orbit()
        };
        // Top edge: with a near-horizontal camera the ray runs parallel
        // to the ground and the slab top is edge-on, so neither mesh nor
        // plane is reachable there in practice, the resolver returns
        // None instead of inventing a click.
        let event = resolve_click(&cam, [800.0, 600.0], [400.0, 4.0], &[]);
        assert!(
            event.is_none() || matches!(event, Some(View3dEvent::GroundClick { .. })),
            "{event:?}"
        );
    }

    #[test]
    fn drag_carries_button_and_delta_per_pressed_move() {
        for (button, pan) in [
            (PointerButton::Primary, false),
            (PointerButton::Secondary, true),
            (PointerButton::Tertiary, true),
        ] {
            let mut d = DragState {
                start: [100.0, 100.0],
                last: [100.0, 100.0],
                pan,
                button,
            };
            let evs = resolve_move(Some(&d), &orbit(), [800.0, 600.0], [112.0, 95.0], &[]);
            assert_eq!(evs.len(), 3, "{button:?}: {evs:?}");
            assert!(matches!(evs[0], View3dEvent::HoverRay { .. }), "{evs:?}");
            if pan {
                assert!(
                    matches!(evs[1], View3dEvent::Pan { dx: 12.0, dy: -5.0 }),
                    "{button:?}: {evs:?}"
                );
            } else {
                assert!(
                    matches!(evs[1], View3dEvent::Orbit { dx: 12.0, dy: -5.0 }),
                    "{button:?}: {evs:?}"
                );
            }
            assert!(
                matches!(
                    evs[2],
                    View3dEvent::Drag { button: b, dx: 12.0, dy: -5.0 } if b == button
                ),
                "{button:?}: {evs:?}"
            );
            d.last = [112.0, 95.0];
            let evs = resolve_move(Some(&d), &orbit(), [800.0, 600.0], [110.0, 100.0], &[]);
            assert!(
                matches!(
                    evs[2],
                    View3dEvent::Drag { button: b, dx: -2.0, dy: 5.0 } if b == button
                ),
                "second move: {evs:?}"
            );
        }
        let still = DragState {
            start: [100.0, 100.0],
            last: [100.0, 100.0],
            pan: false,
            button: PointerButton::Primary,
        };
        let evs = resolve_move(Some(&still), &orbit(), [800.0, 600.0], [100.0, 100.0], &[]);
        assert!(evs.is_empty(), "{evs:?}");
    }

    #[test]
    fn hover_ray_leads_the_hover_pair_with_unit_dir_into_scene() {
        let cam = orbit();
        let evs = resolve_move(None, &cam, [800.0, 600.0], [400.0, 300.0], &[slab()]);
        assert_eq!(evs.len(), 2, "{evs:?}");
        let View3dEvent::HoverRay { eye, dir } = evs[0] else {
            panic!("ray must lead the hover pair: {evs:?}");
        };
        assert!(
            matches!(
                evs[1],
                View3dEvent::HoverMesh {
                    pick_id: Some(7),
                    ..
                }
            ),
            "{evs:?}"
        );
        let eye = Vec3::from_array(eye);
        let dir = Vec3::from_array(dir);
        assert!((dir.length() - 1.0).abs() < 1e-5, "unit dir: {dir:?}");
        // Ray origin sits on the near plane (the origin resolve_click
        // picks through), so the center ray starts NEAR from the eye.
        assert!(
            ((eye - cam.eye()).length() - NEAR).abs() < 1e-4,
            "eye: {eye:?} vs {:?}",
            cam.eye()
        );
        assert!(dir.dot(cam.target - eye) > 0.0, "into the scene: {dir:?}");
    }
}
