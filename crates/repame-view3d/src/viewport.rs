//! 3D viewport view: orbit gestures in, mesh snapshot out to the GPU.
//!
//! Same split as `repame-sprite` `Viewport2d` and resims `Viewport3d`:
//! camera state lives in game signals, never in the renderer. The UI builds
//! a [`Frame3d`] per frame (plain data, cheap to rebuild during
//! composition) and mounts [`Viewport3d`], which draws the snapshot through
//! a [`SceneBatch`] and reports orbit gestures + ground clicks back.

use std::rc::Rc;
use std::sync::{Arc, Mutex};

use glam::Vec2;
use repose_core::input::{PointerButton, PointerEventKind};
use repose_core::{Modifier, View};
use repose_render_wgpu::{Callback, CallbackResources, ScreenDescriptor, WgpuCallback};
use repose_ui::Embedded;

use super::camera::OrbitCamera;
use super::mesh::MeshGroup;
use super::render::{BatchDesc, SceneBatch, SceneLight, SceneUpload};
use super::render::{paint_scene_with_id, prepare_scene_with_id};

/// Everything the viewport draws this frame. Plain data, snapshot per frame.
///
/// Framing contract (single source of truth): the GPU camera uniform is
/// `cam.view_proj(aspect)` where `aspect = viewport_w / viewport_h` from
/// the painted rect; picks invert through the same camera via
/// [`OrbitCamera::screen_ray`]. One camera, two consumers — they cannot
/// disagree.
///
/// Textures ride the same snapshot: [`Frame3d::uploads`] feeds the batch
/// texture array once per frame (games drain their image/atlas source
/// here), and each group samples one page (see
/// [`MeshGroup`](super::mesh::MeshGroup) `texture_page`).
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
    /// mismatches fail at the call site, not silently on the GPU).
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
}

/// UI-facing viewport events. Gestures arrive as deltas; the game applies
/// them to its camera signal ([`OrbitCamera::orbit`] / [`pan`](OrbitCamera::pan) /
/// [`zoom`](OrbitCamera::zoom)) and rebuilds the snapshot.
///
/// `GroundClick` fires on clean left-clicks (press + release with less than
/// [`CLICK_SLOP_PX`] travel), like the 2D viewport's `Click`.
#[derive(Clone, Debug)]
pub enum View3dEvent {
    /// Left-drag orbit delta, in px.
    Orbit { dx: f32, dy: f32 },
    /// Pan delta, in px (shift/middle/right-drag).
    Pan { dx: f32, dy: f32 },
    /// Multiplicative zoom factor (wheel).
    Zoom { factor: f32 },
    /// Ground-plane (y = 0) point under a clean click, if the ray hits.
    GroundClick { x: f32, z: f32 },
    /// Cursor ground point on move (cheap hover readout; `None` when the
    /// ray misses the plane).
    Hover { x: Option<f32>, z: Option<f32> },
}

/// Press slop in physical px: pointer-up farther than this from
/// pointer-down cancels the `GroundClick` (drag-off-cancel, same value as
/// the 2D viewport).
pub const CLICK_SLOP_PX: f32 = 12.0;

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
/// and content stay glued — including under camera motion.
///
/// `batch_desc` must match `input.desc` (the texture shape the groups
/// pack against): the batch is keyed on it and rebuilds — dropping
/// texture contents — when it changes, exactly like the sprite batch.
#[allow(non_snake_case)] // Repose view convention (cf. resims `Viewport3d`).
pub fn Viewport3d(
    input: Frame3d,
    geom_out: GeomHandle,
    batch_id: impl Into<String>,
    batch_desc: BatchDesc,
    on_event: impl Fn(View3dEvent) + 'static,
) -> View {
    let batch_id: String = batch_id.into();
    let input = Rc::new(input);
    let draw_input = input.clone();
    let draw_id = batch_id.clone();
    let hover_cam = input.cam;
    let hover_geom = geom_out.clone();
    let click_cam = input.cam;
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
                    // Hover readout through the latest painted geometry.
                    let g = hover_geom.get();
                    let a = Frame3d::aspect(g.viewport_px);
                    let hit = hover_cam.ground_point(
                        a,
                        Vec2::new(g.viewport_px[0].max(1.0), g.viewport_px[1].max(1.0)),
                        Vec2::new(p.x, p.y),
                    );
                    on_hover(View3dEvent::Hover {
                        x: hit.map(|v| v.x),
                        z: hit.map(|v| v.y),
                    });
                }
            }
        })
        .on_pointer_up(move |ev: repose_core::input::PointerEvent| {
            let p = ev.position;
            if let Some(d) = drag_up.take()
                && click_within_slop(d.start, [p.x, p.y])
            {
                let g = click_geom.get();
                let a = Frame3d::aspect(g.viewport_px);
                if let Some(hit) = click_cam.ground_point(
                    a,
                    Vec2::new(g.viewport_px[0].max(1.0), g.viewport_px[1].max(1.0)),
                    Vec2::new(p.x, p.y),
                ) {
                    on_up_click(View3dEvent::GroundClick { x: hit.x, z: hit.y });
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
        input: Arc::new(draw_input.as_ref().clone()),
        geom: geom_out.arc(),
        batch_id: draw_id,
        batch_desc,
    };
    Embedded(modifier, Callback::new(payload))
}

/// One painted frame's geometry: the viewport size the GPU camera used.
/// Picks invert through this (same aspect), so content and picks stay
/// glued — including under camera motion, since picks read the latest
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
        let mut batch = SceneBatch::with_desc(self.batch_id.clone(), self.batch_desc);
        batch.set_camera(self.input.cam.view_proj(aspect));
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
