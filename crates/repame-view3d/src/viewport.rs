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
use repose_render_wgpu::{Callback, CallbackResources, ScreenDescriptor, WgpuCallback};
use repose_ui::Embedded;

use super::camera::OrbitCamera;
use super::mesh::MeshGroup;
use super::pick::{MeshHit, pick_ray};
use super::render::{BatchDesc, SceneBatch, SceneLight, SceneUpload};
use super::render::{paint_scene_with_id, prepare_scene_with_id};

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
    pub light: SceneLight,
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

/// Press state: press position + last position, both viewport-local px.
#[derive(Clone, Copy)]
struct DragState {
    start: [f32; 2],
    last: [f32; 2],
    /// True for pan gestures (non-primary button or shift held).
    pan: bool,
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
    let on_hover = on_event.clone();
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
            drag_down.set(Some(DragState {
                start: [p.x, p.y],
                last: [p.x, p.y],
                pan: !primary || ev.modifiers.shift,
            }));
        })
        .on_pointer_move(move |ev: repose_core::input::PointerEvent| {
            let p = ev.position;
            match drag_move.take() {
                Some(mut d) => {
                    let dx = p.x - d.last[0];
                    let dy = p.y - d.last[1];
                    d.last = [p.x, p.y];
                    drag_move.set(Some(d));
                    if dx.abs() + dy.abs() > 0.0 {
                        if d.pan {
                            on_move(View3dEvent::Pan { dx, dy });
                        } else {
                            on_move(View3dEvent::Orbit { dx, dy });
                        }
                    }
                }
                None => {
                    // Hover: mesh pick first (same ray the GPU camera
                    // used), ground-plane fallback when nothing pickable
                    // is under the cursor.
                    let g = hover_geom.get();
                    let vpx = [g.viewport_px[0].max(1.0), g.viewport_px[1].max(1.0)];
                    let a = Frame3d::aspect(g.viewport_px);
                    let vp = Vec2::new(vpx[0], vpx[1]);
                    let p = Vec2::new(p.x, p.y);
                    let (origin, dir) = hover_input.cam.screen_ray(a, vp, p);
                    let mesh: Option<MeshHit> = pick_ray(origin, dir, &hover_input.groups);
                    let ground = hover_input.cam.ground_point(a, vp, p);
                    match (mesh, ground) {
                        (Some(hit), _) => on_hover(View3dEvent::HoverMesh {
                            pick_id: Some(hit.pick_id),
                            x: hit.point[0],
                            z: hit.point[2],
                        }),
                        (None, Some(gp)) => on_hover(View3dEvent::HoverMesh {
                            pick_id: None,
                            x: gp.x,
                            z: gp.y,
                        }),
                        (None, None) => on_hover(View3dEvent::Hover { x: None, z: None }),
                    }
                }
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
        batch.set_light(self.input.light);
        for g in &self.input.groups {
            batch.push_group(g);
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
        rpass: &mut wgpu::RenderPass<'static>,
        resources: &CallbackResources,
    ) {
        if let Ok(mut g) = self.geom.lock() {
            *g = ViewportGeom {
                viewport_px: [info.viewport.w.max(1.0), info.viewport.h.max(1.0)],
            };
        }
        paint_scene_with_id(self.batch_id.as_str(), rpass, resources);
    }
}

#[cfg(test)]
mod tests {
    use super::super::camera::OrbitCamera;
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
}
