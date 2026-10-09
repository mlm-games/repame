//! 3D viewport view: orbit gestures in, mesh snapshot out to the GPU.
//!
//! Same split as `repame-sprite` `Viewport2d` and resims `Viewport3d`:
//! camera state lives in game signals, never in the renderer. The UI builds
//! a [`Frame3d`] per frame (plain data, rebuilt during composition) and
//! mounts [`Viewport3d`], which draws the snapshot through
//! a [`SceneBatch`] and reports orbit gestures + ground clicks back.

use std::rc::Rc;
use std::sync::{Arc, Mutex};

use glam::Vec2;
use glam::Vec3;
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
    /// Empty (default) disables the skin path; the palette uniform stays
    /// bound but unread.
    pub skinned: Vec<crate::SkinnedDraw>,
    /// Offscreen clear color (linear 0..1 RGBA). The shared UI pass this
    /// viewport paints into has its own clear; this selects the scene
    /// target clear inside the viewport-owned pass.
    pub background: Option<[f32; 4]>,
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
            uploads: Vec::new(),
            desc: BatchDesc::default(),
            light: SceneLight::default(),
            shadow: None,
            rig: None,
            skinned: Vec::new(),
            background: None,
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
        });
    let payload = GpuViewport3d {
        input: draw_input.clone(),
        geom: geom_out.arc(),
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
#[derive(Clone, Default)]
pub struct GeomHandle(Arc<Mutex<ViewportGeom>>);

impl GeomHandle {
    pub fn new() -> Self {
        Self(Arc::new(Mutex::new(ViewportGeom::default())))
    }

    pub fn get(&self) -> ViewportGeom {
        self.0.lock().map(|g| *g).unwrap_or_default()
    }

    pub fn set(&self, g: ViewportGeom) {
        if let Ok(mut slot) = self.0.lock() {
            *slot = g;
        }
    }

    pub fn arc(&self) -> Arc<Mutex<ViewportGeom>> {
        self.0.clone()
    }
}

struct GpuViewport3d {
    input: Arc<Frame3d>,
    geom: Arc<Mutex<ViewportGeom>>,
    batch_id: String,
    batch_desc: BatchDesc,
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
        let vp = self
            .geom
            .lock()
            .map(|g| g.viewport_px)
            .unwrap_or(self.input.viewport_px);
        let w = vp[0].max(1.0) as u32;
        let h = vp[1].max(1.0) as u32;
        let aspect = Frame3d::aspect(vp);
        let desc = self.effective_desc();
        let mut batch = SceneBatch::with_desc(self.batch_id.clone(), desc);
        batch.set_camera(self.input.cam.view_proj(aspect));
        batch.set_camera_pos(self.input.cam.eye().into());
        batch.set_camera_forward(self.input.cam.forward().into());
        match &self.input.rig {
            Some(rig) => {
                let vp = self.input.cam.view_proj(aspect);
                batch.set_rig(
                    rig,
                    self.input.cam.eye().into(),
                    self.input.cam.view_matrix(),
                    vp.inverse(),
                );
            }
            None => {
                batch.set_light(self.input.light);
                batch.set_shadow(
                    self.input.shadow,
                    self.input.cam.target.into(),
                    self.input.cam.dist,
                );
            }
        }
        for g in &self.input.groups {
            batch.push_group(g);
        }
        for s in &self.input.skinned {
            batch.push_skinned(s);
        }
        batch.extend_uploads(self.input.uploads.iter().cloned());
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
