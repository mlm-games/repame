//! World-anchored instanced draw list.
//!
//! [`actors_view`] presents one rig filling one surface. A game needs the
//! opposite: one rig drawn many times per frame, all sharing a single world
//! camera, ordered by z, each with its own placement and paint. That is this
//! module.
//!
//! One [`ActorLibrary`] owns every rig and one [`SceneRenderer`], so tessellation
//! is shared. Per frame the list is walked in z order; each distinct
//! `(entry, frame, tint)` is prepared once and its [`PreparedScene`] reused by
//! every instance with that combination. Placement rides in each draw's
//! `transform` field, which is *not* part of the mesh cache key
//! (`renamite_render_bridge::SceneRenderer`), so hundreds of instances of one
//! rig cost one tessellation.
//!
//! Camera [`roll`](repame_sprite::Camera2d::roll) is not supported: renamite
//! composes a view as scale plus offset, so a rotation has nowhere to enter.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use glam::{DVec2, DVec4};
use renamite_model::{Scene, ScenePaint};
use renamite_render_bridge::{PreparedDraw, PreparedScene};
use repame_sprite::{Camera2d, PickEvent, click_within_slop_dp, dp_to_world, effective_fit};
use repose_core::input::{PointerButton, PointerEvent, PointerEventKind, PointerKind};
use repose_core::locals::effective_density_scale;
use repose_core::{BlendMode, Modifier, Rect, RenderContext, SceneNode, View};
use repose_ui::{Box as UiBox, ViewExt, ZStack};

use crate::{PlayerHost, PlayerHostRef, ViewTransform};

/// Index into an [`ActorLibrary`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ActorId(pub usize);

/// One rig plus the facts about it the draw loop needs.
struct Entry {
    host: PlayerHostRef,
    /// Cached from the first prepared scene: whether any item paints an image.
    images: Option<bool>,
}

pub struct ActorLibrary {
    entries: Vec<Entry>,
}

impl Default for ActorLibrary {
    fn default() -> Self {
        Self::new()
    }
}

impl ActorLibrary {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add a rig from `.ren` source text.
    pub fn push_ren_str(&mut self, source: &str) -> Result<ActorId, renamite_player::PlayerError> {
        Ok(self.push(PlayerHost::from_ren_str(source)?))
    }

    /// Add a rig from packed `.renb` bytes.
    pub fn push_ren_bytes(
        &mut self,
        bytes: &[u8],
    ) -> Result<ActorId, renamite_player::PlayerError> {
        Ok(self.push(PlayerHost::from_ren_bytes(bytes)?))
    }

    pub fn push(&mut self, host: PlayerHost) -> ActorId {
        self.entries.push(Entry {
            host: Rc::new(RefCell::new(host)),
            images: None,
        });
        ActorId(self.entries.len() - 1)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn host(&self, id: ActorId) -> Option<&PlayerHostRef> {
        self.entries.get(id.0).map(|entry| &entry.host)
    }
}

/// What a placement draws. One draw list carries all three media so a single
/// surface can hold them in one z order; a rig is one kind of item, not the
/// only kind of input.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActorItem {
    /// A frame of a rig from an [`ActorLibrary`], drawn as vector geometry and
    /// instanced when several placements share the frame.
    Rig { actor: ActorId, frame: f64 },
    /// One cell of an atlas page, drawn by the instanced sprite batch. `uv` is
    /// the cell rectangle in page space; `size` is its unrotated extent in
    /// pixels, before the placement's scale.
    Raster {
        page: u32,
        uv_min: [f32; 2],
        uv_max: [f32; 2],
        size: [f32; 2],
    },
    /// A solid rectangle in the placement's own space, centred on the anchor
    /// and sized `half`. Drawn through Repose's instanced rect pipe, so runs of
    /// them batch.
    Rect { half: [f32; 2] },
}

/// One placement of one item. Cheaper to clone than to configure.
#[derive(Clone, Debug)]
pub struct ActorDraw {
    pub item: ActorItem,
    /// World position of the anchor point.
    pub world: DVec2,
    pub rotation: f64,
    /// Per-axis scale; multiply by `-1` to flip, so `flip` is sugar.
    pub scale: DVec2,
    /// Normalised point of the cell placed at `world`. `(0.5, 0.5)` centres it.
    pub anchor: DVec2,
    /// Multiplied into every solid fill. `1,1,1,1` leaves the rig untouched.
    pub tint: DVec4,
    /// Cell-local `[x, y, w, h]` to clip to, in the rig's own coordinates.
    /// `None` draws the whole cell. A rotated clip is covered by the axis-aligned
    /// bounds of its transformed corners, so it over-covers rather than cuts.
    pub clip: Option<[f64; 4]>,
    pub blend: BlendMode,
    /// Higher paints later. Sorted once per frame.
    pub z: f32,
}

impl Default for ActorDraw {
    fn default() -> Self {
        Self {
            item: ActorItem::Rect { half: [0.0, 0.0] },
            world: DVec2::ZERO,
            rotation: 0.0,
            scale: DVec2::ONE,
            anchor: DVec2::splat(0.5),
            tint: DVec4::ONE,
            clip: None,
            blend: BlendMode::Alpha,
            z: 0.0,
        }
    }
}

#[derive(Default)]
pub struct ActorDrawList {
    draws: Vec<ActorDraw>,
}

impl ActorDrawList {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adopt a frame's placements wholesale. The caller keeps ownership of the
    /// vector and hands it over per frame, so this only moves it.
    pub fn from_draws(draws: Vec<ActorDraw>) -> Self {
        Self { draws }
    }

    pub fn clear(&mut self) {
        self.draws.clear();
    }

    pub fn push(&mut self, draw: ActorDraw) {
        self.draws.push(draw);
    }

    pub fn len(&self) -> usize {
        self.draws.len()
    }

    pub fn is_empty(&self) -> bool {
        self.draws.is_empty()
    }

    pub fn as_slice(&self) -> &[ActorDraw] {
        &self.draws
    }
}

/// `out = b` applied after `a`, in renamite's column-major layout:
/// `out.x = v[0] * x + v[2] * y + v[4]` and `out.y = v[1] * x + v[3] * y + v[5]`.
fn compose(a: [f64; 6], b: [f64; 6]) -> [f64; 6] {
    [
        b[0] * a[0] + b[2] * a[1],
        b[1] * a[0] + b[3] * a[1],
        b[0] * a[2] + b[2] * a[3],
        b[1] * a[2] + b[3] * a[3],
        b[0] * a[4] + b[2] * a[5] + b[4],
        b[1] * a[4] + b[3] * a[5] + b[5],
    ]
}

/// Local cell point -> world: scale, then rotate, then translate, about the
/// anchor. Same column-major layout as [`compose`].
fn instance_affine(draw: &ActorDraw, size: DVec2) -> [f64; 6] {
    let (sin, cos) = draw.rotation.sin_cos();
    let (sx, sy) = (draw.scale.x, draw.scale.y);
    let affine = [cos * sx, sin * sx, -sin * sy, cos * sy, 0.0, 0.0];
    let pivot = DVec2::new(draw.anchor.x * size.x, draw.anchor.y * size.y);
    let rotated = apply(affine, pivot);
    [
        affine[0],
        affine[1],
        affine[2],
        affine[3],
        draw.world.x - rotated.x,
        draw.world.y - rotated.y,
    ]
}

fn view_affine(view: &ViewTransform) -> [f64; 6] {
    [
        view.scale,
        0.0,
        0.0,
        view.scale,
        view.offset.x,
        view.offset.y,
    ]
}

fn tint_key(tint: DVec4) -> [u64; 4] {
    [
        tint.x.to_bits(),
        tint.y.to_bits(),
        tint.z.to_bits(),
        tint.w.to_bits(),
    ]
}

fn artboard(player: &renamite_player::Player) -> DVec2 {
    let comp = player.composition();
    match player.project.document.compositions.get(comp) {
        Some(c) => DVec2::new(c.size.0 as f64, c.size.1 as f64),
        None => DVec2::ONE,
    }
}

fn clip_mesh(x0: f64, y0: f64, x1: f64, y1: f64) -> Arc<repose_core::VectorMeshData> {
    let vertices: Arc<[repose_core::VectorVertex]> = [
        ([x0 as f32, y0 as f32], [0.0, 0.0]),
        ([x1 as f32, y0 as f32], [1.0, 0.0]),
        ([x1 as f32, y1 as f32], [1.0, 1.0]),
        ([x0 as f32, y1 as f32], [0.0, 1.0]),
    ]
    .iter()
    .map(|(pos, uv)| repose_core::VectorVertex {
        pos: *pos,
        color: [1.0, 1.0, 1.0, 1.0],
        uv: *uv,
    })
    .collect();
    Arc::new(repose_core::VectorMeshData {
        vertices,
        indices: [0u32, 1, 2, 0, 2, 3].into(),
    })
}

fn apply(a: [f64; 6], p: DVec2) -> DVec2 {
    DVec2::new(
        a[0] * p.x + a[2] * p.y + a[4],
        a[1] * p.x + a[3] * p.y + a[5],
    )
}

/// Multiplying every solid fill by `tint`. Gradients and images are left
/// alone: a uniform multiply over a gradient or a cropped bitmap has no
/// defined meaning, and silently picking one would be worse than not tinting.
fn tint_scene(scene: &Scene, tint: DVec4) -> Scene {
    let mut out = scene.clone();
    for item in &mut out.items {
        if let ScenePaint::Solid(color) = &mut item.paint {
            color.r *= tint.x;
            color.g *= tint.y;
            color.b *= tint.z;
            color.a *= tint.w;
        }
    }
    out
}

/// Fold an instance placement into every image item's local->world affine.
///
/// Vector draws retarget through `PreparedDraw::transform` instead, because
/// that field is outside the mesh cache key. Images have no mesh, so folding
/// into the affine costs nothing and keeps them exact under rotation.
fn place_images(scene: &mut Scene, instance: [f64; 6]) {
    for item in &mut scene.items {
        if let ScenePaint::Image { affine, .. } = &mut item.paint {
            *affine = compose(instance, *affine);
        }
    }
}

fn has_images(scene: &Scene) -> bool {
    scene
        .items
        .iter()
        .any(|item| matches!(item.paint, ScenePaint::Image { .. }))
}

/// Camera as renamite's scale-plus-offset view, matching `repame_sprite`'s
/// `world_to_dp` exactly. That law is in dp; a `DrawScope` addresses physical
/// px, so the caller scales by the density to land in the space the draws are
/// emitted in.
fn camera_view(camera: &Camera2d, canvas_dp: [f32; 2], world_size: [f32; 2]) -> ViewTransform {
    let (scale, ox, oy) = effective_fit(canvas_dp, world_size, camera);
    let [cx, cy] = camera.effective_center();
    ViewTransform {
        scale: scale as f64,
        offset: DVec2::new(
            f64::from(ox) - f64::from(cx - world_size[0] * 0.5) * f64::from(scale),
            f64::from(oy) - f64::from(cy - world_size[1] * 0.5) * f64::from(scale),
        ),
    }
}

/// Camera view rescaled from dp into the canvas's physical px.
fn camera_view_px(
    camera: &Camera2d,
    canvas_dp: [f32; 2],
    world_size: [f32; 2],
    density: f64,
) -> ViewTransform {
    let view = camera_view(camera, canvas_dp, world_size);
    ViewTransform {
        scale: view.scale * density,
        offset: view.offset * density,
    }
}

/// Retarget prepared vector draws onto one instance. Meshes are arcs, so this
/// is a handful of f32 per draw.
fn retarget_draws(base: &[PreparedDraw], instance: [f64; 6]) -> Vec<PreparedDraw> {
    base.iter()
        .map(|draw| match draw {
            PreparedDraw::Vector {
                mesh,
                transform,
                paint,
                clips,
                blend,
            } => PreparedDraw::Vector {
                mesh: mesh.clone(),
                transform: compose(instance, transform.map(f64::from)).map(|value| value as f32),
                paint: *paint,
                clips: clips.clone(),
                blend: *blend,
            },
            PreparedDraw::Image {
                handle,
                rect,
                transform,
                tint,
                fit,
                clips,
                blend,
            } => PreparedDraw::Image {
                handle: *handle,
                rect: *rect,
                transform: *transform,
                tint: *tint,
                fit: *fit,
                clips: clips.clone(),
                blend: *blend,
            },
        })
        .collect()
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PrepKey {
    actor: usize,
    frame: u64,
    tint: [u64; 4],
}

/// Atlas resources backing [`ActorItem::Raster`] placements. The pixel data
/// rides an `Arc`; every view batches its own rows per frame from it.
#[derive(Clone)]
pub struct RasterAtlas {
    pub desc: repame_sprite::BatchDesc,
    pub uploads: std::sync::Arc<[repame_sprite::AtlasUpload]>,
    pub batch_id: String,
}

/// One z-ordered world surface for all three [`ActorItem`] media. Rigs and
/// rects are mesh placements and share the instanced mesh path; raster runs go
/// to the atlas batch, which keeps its instancing. Because the whole list is
/// emitted into one scene, the media interleave freely in z.
pub fn actors_view_mixed(
    library: Rc<RefCell<ActorLibrary>>,
    draws: Rc<ActorDrawList>,
    camera: Camera2d,
    world_size: [f32; 2],
    ctx: RenderContext,
    atlas: Option<Rc<RasterAtlas>>,
) -> View {
    debug_assert!(
        camera.roll.abs() < f32::EPSILON,
        "actors_view_mixed does not apply Camera2d::roll"
    );

    let prepared: Rc<RefCell<HashMap<PrepKey, PreparedScene>>> = Rc::default();
    let order: Rc<RefCell<Vec<u32>>> = Rc::default();

    let modifier = Modifier::new().fill_max_size().hit_passthrough().painter(
        move |scene: &mut repose_core::Scene, rect: Rect, _alpha: f32| {
            let density = effective_density_scale().max(1e-6);
            let canvas_dp = [rect.w / density, rect.h / density];
            if canvas_dp[0] <= 1.0 || canvas_dp[1] <= 1.0 {
                return;
            }
            let view = camera_view_px(&camera, canvas_dp, world_size, f64::from(density));

            let entries = draws.as_slice();
            {
                let mut order = order.borrow_mut();
                order.clear();
                order.extend(0..entries.len() as u32);
                order.sort_by(|a, b| compare_draws(&entries[*a as usize], &entries[*b as usize]));
            }

            prepared.borrow_mut().clear();
            let sorted: Vec<u32> = order.borrow().clone();
            let mut library = library.borrow_mut();
            let mut cursor = 0usize;
            let mut run_index = 0u32;
            while cursor < sorted.len() {
                let head = &entries[sorted[cursor] as usize];
                let group = (medium_key(head.item), item_key(head.item), head.z.to_bits());
                let mut end = cursor + 1;
                while end < sorted.len() {
                    let next = &entries[sorted[end] as usize];
                    if (medium_key(next.item), item_key(next.item), next.z.to_bits()) != group
                        || next.clip.is_some()
                    {
                        break;
                    }
                    end += 1;
                }
                let run = cursor..end;
                cursor = end;
                let run_id = run_index;
                run_index += 1;
                emit_run(
                    run,
                    run_id,
                    &sorted,
                    entries,
                    &group.0,
                    &mut library,
                    &prepared,
                    &view,
                    &camera,
                    canvas_dp,
                    world_size,
                    density,
                    atlas.as_deref(),
                    rect,
                    scene,
                    &ctx,
                );
            }
        },
    );
    UiBox(modifier)
}

#[allow(clippy::too_many_arguments)]
fn emit_run(
    run: std::ops::Range<usize>,
    run_id: u32,
    sorted: &[u32],
    entries: &[ActorDraw],
    medium: &Medium,
    library: &mut ActorLibrary,
    prepared: &Rc<RefCell<HashMap<PrepKey, PreparedScene>>>,
    view: &ViewTransform,
    camera: &Camera2d,
    canvas_dp: [f32; 2],
    world_size: [f32; 2],
    density: f32,
    atlas: Option<&RasterAtlas>,
    rect: Rect,
    scene: &mut repose_core::Scene,
    ctx: &RenderContext,
) {
    match medium {
        Medium::Rect => {
            let mesh = unit_quad();
            let mut instances = Vec::with_capacity(run.len());
            for i in run {
                let draw = &entries[sorted[i] as usize];
                let ActorItem::Rect { half } = draw.item else {
                    continue;
                };
                instances.push(rect_instance(draw, half, view));
            }
            if instances.is_empty() {
                return;
            }
            scene.nodes.push(SceneNode::VectorMeshInstanced {
                mesh,
                transforms: instances,
                paint: repose_core::PaintDesc::Solid,
                blend: BlendMode::Alpha,
            });
        }

        Medium::Raster => {
            let Some(atlas) = atlas else {
                return;
            };
            let mut batch = repame_sprite::SpriteBatch::with_id(
                format!("{}.r{}", atlas.batch_id, run_id),
                atlas.desc.clone(),
            );
            batch.set_camera(camera.fit_matrix(canvas_dp, world_size));
            for i in run {
                let draw = &entries[sorted[i] as usize];
                let ActorItem::Raster {
                    page,
                    uv_min,
                    uv_max,
                    size,
                } = draw.item
                else {
                    continue;
                };
                batch.push_sprite(&sprite_instance(draw, page, uv_min, uv_max, size));
            }
            let fit = repame_sprite::effective_fit(canvas_dp, world_size, camera);
            scene.nodes.push(
                repame_sprite::BatchFrame::new(batch, atlas.uploads.clone()).node(Rect {
                    x: rect.x + fit.1 * density,
                    y: rect.y + fit.2 * density,
                    w: world_size[0] * fit.0 * density,
                    h: world_size[1] * fit.0 * density,
                }),
            );
        }

        Medium::Rig => {
            let head = &entries[sorted[run.start] as usize];
            let ActorItem::Rig { actor, frame } = head.item else {
                return;
            };
            let Some(entry) = library.entries.get_mut(actor.0) else {
                return;
            };
            let PlayerHost {
                player,
                renderer,
                dirty_images,
                ..
            } = &mut *entry.host.borrow_mut();
            player.scrub(frame);
            let raw = player.scene();
            let size = artboard(player);
            let images = *entry.images.get_or_insert_with(|| has_images(raw));
            if *dirty_images {
                renderer.sync_document_images(&player.project.document, ctx);
                *dirty_images = false;
            }

            let key = PrepKey {
                actor: actor.0,
                frame: frame.to_bits(),
                tint: tint_key(head.tint),
            };
            let source = if head.tint == DVec4::ONE {
                raw.clone()
            } else {
                tint_scene(raw, head.tint)
            };

            if !images {
                let mut cache = prepared.borrow_mut();
                if !cache.contains_key(&key) {
                    cache.insert(key, renderer.prepare(&source, view));
                }
                let base = cache.get(&key).expect("inserted above");
                if run.len() > 1 && base.is_batchable() {
                    let instances: Vec<[f64; 6]> = run
                        .clone()
                        .map(|i| instance_affine(&entries[sorted[i] as usize], size))
                        .collect();
                    renderer.append_repose_scene_instanced(base, &instances, scene);
                    return;
                }
            }

            for i in run {
                let draw = &entries[sorted[i] as usize];
                let instance = instance_affine(draw, size);
                if images {
                    let mut placed = source.clone();
                    place_images(&mut placed, instance);
                    let base = renderer.prepare(&placed, view);
                    let retargeted = PreparedScene {
                        draws: retarget_draws(&base.draws, instance),
                        clips: base.clips,
                    };
                    open_clip(scene, draw.clip, instance, view);
                    renderer.append_repose_scene(&retargeted, scene);
                    close_clip(scene, draw.clip.is_some());
                    continue;
                }
                let mut cache = prepared.borrow_mut();
                let base = cache.get_mut(&key).expect("inserted above");
                let retargeted = PreparedScene {
                    draws: retarget_draws(&base.draws, instance),
                    clips: std::mem::take(&mut base.clips),
                };
                drop(cache);
                open_clip(scene, draw.clip, instance, view);
                renderer.append_repose_scene(&retargeted, scene);
                close_clip(scene, draw.clip.is_some());
                if let Some(base) = prepared.borrow_mut().get_mut(&key) {
                    base.clips = retargeted.clips;
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Medium {
    Rig,
    Raster,
    Rect,
}

fn medium_key(item: ActorItem) -> Medium {
    match item {
        ActorItem::Rig { .. } => Medium::Rig,
        ActorItem::Raster { .. } => Medium::Raster,
        ActorItem::Rect { .. } => Medium::Rect,
    }
}

fn compare_draws(a: &ActorDraw, b: &ActorDraw) -> std::cmp::Ordering {
    a.z.partial_cmp(&b.z)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| medium_key(a.item).cmp(&medium_key(b.item)))
        .then_with(|| item_key(a.item).cmp(&item_key(b.item)))
        .then_with(|| tint_key(a.tint).cmp(&tint_key(b.tint)))
}

fn item_key(item: ActorItem) -> u64 {
    match item {
        ActorItem::Rig { actor, frame } => (actor.0 as u64) << 32 | (frame.to_bits() & 0xffff_ffff),
        ActorItem::Raster { page, .. } => page as u64,
        ActorItem::Rect { .. } => 0,
    }
}

fn unit_quad() -> Arc<repose_core::VectorMeshData> {
    thread_local! {
        static QUAD: Arc<repose_core::VectorMeshData> = Arc::new(repose_core::VectorMeshData {
            vertices: [
                ([-1.0, -1.0], [0.0, 0.0]),
                ([1.0, -1.0], [1.0, 0.0]),
                ([1.0, 1.0], [1.0, 1.0]),
                ([-1.0, 1.0], [0.0, 1.0]),
            ]
            .iter()
            .map(|(pos, uv)| repose_core::VectorVertex {
                pos: *pos,
                color: [1.0, 1.0, 1.0, 1.0],
                uv: *uv,
            })
            .collect::<Arc<[_]>>(),
            indices: [0u32, 1, 2, 0, 2, 3].into(),
        });
    }
    QUAD.with(Arc::clone)
}

fn rect_instance(draw: &ActorDraw, half: [f32; 2], view: &ViewTransform) -> [f32; 6] {
    let full = [2.0 * half[0], 2.0 * half[1]];
    let world = compose(instance_affine_size(draw, full), view_affine(view));
    [
        world[0] as f32,
        world[1] as f32,
        world[2] as f32,
        world[3] as f32,
        world[4] as f32,
        world[5] as f32,
    ]
}

/// Same placement law as [`instance_affine`] but with an explicit cell size,
/// so a rect can size its own quad.
fn instance_affine_size(draw: &ActorDraw, size: [f32; 2]) -> [f64; 6] {
    let (sin, cos) = draw.rotation.sin_cos();
    let (sx, sy) = (draw.scale.x, draw.scale.y);
    let affine = [cos * sx, sin * sx, -sin * sy, cos * sy, 0.0, 0.0];
    let pivot = DVec2::new(
        draw.anchor.x * f64::from(size[0]),
        draw.anchor.y * f64::from(size[1]),
    );
    let rotated = apply(affine, pivot);
    [
        affine[0],
        affine[1],
        affine[2],
        affine[3],
        draw.world.x - rotated.x,
        draw.world.y - rotated.y,
    ]
}

fn sprite_instance(
    draw: &ActorDraw,
    page: u32,
    uv_min: [f32; 2],
    uv_max: [f32; 2],
    size: [f32; 2],
) -> repame_sprite::SpriteInstance {
    let (sx, sy) = (draw.scale.x as f32, draw.scale.y as f32);
    repame_sprite::SpriteInstance {
        center: glam::Vec2::new(draw.world.x as f32, draw.world.y as f32),
        rotation: draw.rotation as f32,
        size: glam::Vec2::new(size[0] * sx.abs(), size[1] * sy.abs()),
        anchor: glam::Vec2::new(draw.anchor.x as f32, draw.anchor.y as f32),
        flip_x: sx < 0.0,
        flip_y: sy < 0.0,
        uv_min: glam::Vec2::new(uv_min[0], uv_min[1]),
        uv_max: glam::Vec2::new(uv_max[0], uv_max[1]),
        color: [
            draw.tint.x as f32,
            draw.tint.y as f32,
            draw.tint.z as f32,
            draw.tint.w as f32,
        ],
        page,
        z: draw.z,
        blend: match draw.blend {
            BlendMode::Add => repame_sprite::SpriteBlend::Additive,
            BlendMode::Multiply => repame_sprite::SpriteBlend::Multiply,
            _ => repame_sprite::SpriteBlend::Alpha,
        },
    }
}

fn open_clip(
    scene: &mut repose_core::Scene,
    clip: Option<[f64; 4]>,
    instance: [f64; 6],
    view: &ViewTransform,
) {
    let Some([x, y, w, h]) = clip else {
        return;
    };
    let world = compose(instance, view_affine(view));
    let corners = [
        apply(world, DVec2::new(x, y)),
        apply(world, DVec2::new(x + w, y)),
        apply(world, DVec2::new(x + w, y + h)),
        apply(world, DVec2::new(x, y + h)),
    ];
    let (lo, hi) = corners.iter().fold(
        (DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY)),
        |(lo, hi), p| (lo.min(*p), hi.max(*p)),
    );
    scene.nodes.push(SceneNode::PushVectorClip {
        mesh: clip_mesh(lo.x, lo.y, hi.x, hi.y),
        op: repose_core::ClipOp::Intersect,
    });
}

fn close_clip(scene: &mut repose_core::Scene, pushed: bool) {
    if pushed {
        scene.nodes.push(SceneNode::PopVectorClip);
    }
}

/// World camera surface: every rig in `library`, instanced by `draws`.
pub fn actors_view_world(
    library: Rc<RefCell<ActorLibrary>>,
    draws: Rc<ActorDrawList>,
    camera: Camera2d,
    world_size: [f32; 2],
    ctx: RenderContext,
) -> View {
    actors_view_mixed(library, draws, camera, world_size, ctx, None)
}

/// Pointer plumbing for [`actors_view_world_with_events`]. The `DrawScope` half
/// of a viewport has no hit channel of its own, so the camera law is replayed
/// here to rebuild the `PickEvent`s a sprite viewport emits.
struct PickState {
    sink: Rc<dyn Fn(PickEvent)>,
    camera: Camera2d,
    world_size: [f32; 2],
    canvas_dp: Rc<RefCell<[f32; 2]>>,
    press: HashMap<u64, [f32; 2]>,
}

impl PickState {
    fn project(&self, position: repose_core::Vec2) -> ([f32; 2], glam::Vec2) {
        let density = effective_density_scale().max(1e-6);
        let dp = [position.x as f32 / density, position.y as f32 / density];
        let fit = effective_fit(*self.canvas_dp.borrow(), self.world_size, &self.camera);
        let world = dp_to_world(dp, self.world_size, self.camera.effective_center(), fit);
        // `PickEvent.screen` is window-PHYSICAL px, not dp, so consumers
        // convert once. Callers must pass `window_position`: Repose rewrites
        // `event.position` to be hit-region-local (`repose-app` runtime), and
        // a pick surface nested in an offset container would otherwise shift
        // every pick by that origin.
        (
            [position.x as f32, position.y as f32],
            glam::Vec2::new(world[0], world[1]),
        )
    }

    fn is_contact(&self, event: &PointerEvent) -> bool {
        matches!(event.kind, PointerKind::Touch | PointerKind::Pen)
    }
}

/// World camera surface that also reports [`PickEvent`]s: press/release in
/// world and screen space, hover, a click gated behind
/// [`click_within_slop_dp`], and the touch triplet.
pub fn actors_view_world_with_events(
    library: Rc<RefCell<ActorLibrary>>,
    draws: Rc<ActorDrawList>,
    camera: Camera2d,
    world_size: [f32; 2],
    ctx: RenderContext,
    on_event: Rc<dyn Fn(PickEvent)>,
) -> View {
    actors_view_mixed_with_events(library, draws, camera, world_size, ctx, None, on_event)
}

/// [`actors_view_mixed`] with the [`PickEvent`] surface of
/// [`actors_view_world_with_events`].
pub fn actors_view_mixed_with_events(
    library: Rc<RefCell<ActorLibrary>>,
    draws: Rc<ActorDrawList>,
    camera: Camera2d,
    world_size: [f32; 2],
    ctx: RenderContext,
    atlas: Option<Rc<RasterAtlas>>,
    on_event: Rc<dyn Fn(PickEvent)>,
) -> View {
    let surface: Rc<RefCell<[f32; 2]>> = Rc::new(RefCell::new([1.0, 1.0]));
    let state: Rc<RefCell<PickState>> = Rc::new(RefCell::new(PickState {
        sink: on_event,
        camera,
        world_size,
        canvas_dp: surface.clone(),
        press: HashMap::new(),
    }));

    let surface_view = actors_view_mixed(library, draws, camera, world_size, ctx, atlas);

    let down = state.clone();
    let up = state.clone();
    let move_state = state.clone();
    let leave = state.clone();
    let sized = surface.clone();
    let modifier = Modifier::new()
        .fill_max_size()
        .on_size_changed(move |size| {
            *sized.borrow_mut() = [size.x, size.y];
        })
        .on_pointer_down(move |event| {
            let mut state = down.borrow_mut();
            let (screen, world) = state.project(event.window_position);
            let button = match event.event {
                PointerEventKind::Down(button) => button,
                _ => PointerButton::Primary,
            };
            state.press.insert(event.id.0, screen);
            (state.sink)(PickEvent::Press {
                world,
                screen,
                button,
            });
            if state.is_contact(&event) {
                (state.sink)(PickEvent::TouchDown {
                    id: event.id.0,
                    screen,
                });
            }
        })
        .on_pointer_up(move |event| {
            let mut state = up.borrow_mut();
            let (screen, world) = state.project(event.window_position);
            let button = match event.event {
                PointerEventKind::Up(button) => button,
                _ => PointerButton::Primary,
            };
            let origin = state.press.remove(&event.id.0);
            let density = effective_density_scale().max(1e-6);
            (state.sink)(PickEvent::Release {
                world,
                screen,
                button,
            });
            if state.is_contact(&event) {
                (state.sink)(PickEvent::TouchUp { id: event.id.0 });
            }
            if origin.is_some_and(|start| click_within_slop_dp(start, screen, density)) {
                (state.sink)(PickEvent::Click {
                    world,
                    screen,
                    button,
                });
            }
        })
        .on_pointer_move(move |event| {
            let state = move_state.borrow();
            let (screen, world) = state.project(event.window_position);
            (state.sink)(PickEvent::Hover { world, screen });
            if state.is_contact(&event) {
                (state.sink)(PickEvent::TouchMove {
                    id: event.id.0,
                    screen,
                });
            }
        })
        .on_pointer_leave(move |event| {
            let mut state = leave.borrow_mut();
            if state.press.remove(&event.id.0).is_some() {
                (state.sink)(PickEvent::TouchUp { id: event.id.0 });
            }
        });

    // Later children stack on top, so the transparent hit region sits above
    // the surface it reports picks for.
    ZStack(Modifier::new().fill_max_size())
        .child(surface_view)
        .child(UiBox(modifier))
}

#[cfg(test)]
mod tests {
    use super::*;
    use renamite_animation::{Animated, Frame, FrameRate};
    use renamite_model::{
        Color, Document, FillRule, Node, NodeKind, Parent, ShapeKind, StyleKind, StylePaint,
    };
    use renamite_render_bridge::SceneRenderer;
    use repame_sprite::{effective_fit, world_to_dp};
    use std::cell::RefCell;
    use std::rc::Rc;

    fn rig(size: DVec2, color: Color) -> Document {
        let mut doc = Document::empty();
        {
            let comp = doc.main_composition_mut().unwrap();
            comp.size = (size.x as u32, size.y as u32);
            comp.rate = FrameRate { num: 12, den: 1 };
            comp.range = (Frame(0), Frame(1));
        }
        let root = doc.create_node(Node::new("rig", NodeKind::Group));
        doc.attach(root, Parent::Comp(doc.main), 0).unwrap();
        let shape = doc.create_node(Node::new(
            "box",
            NodeKind::Shape(ShapeKind::Rect {
                pos: Animated::new(DVec2::new(size.x * 0.5, size.y * 0.5)),
                size: Animated::new(size),
                rounded: Animated::new(0.0),
            }),
        ));
        doc.attach(shape, Parent::Node(root), 0).unwrap();
        let fill = doc.create_node(Node::new(
            "fill",
            NodeKind::Style(StyleKind::Fill {
                paint: StylePaint::Solid {
                    color: Animated::new(color),
                },
                rule: FillRule::NonZero,
            }),
        ));
        doc.attach(fill, Parent::Node(root), 1).unwrap();
        doc
    }

    fn view_affine(view: &ViewTransform) -> [f64; 6] {
        [
            view.scale,
            0.0,
            0.0,
            view.scale,
            view.offset.x,
            view.offset.y,
        ]
    }

    fn close(a: DVec2, b: DVec2) -> bool {
        (a - b).length() < 1e-4
    }

    /// The camera mapping must agree with the engine's own forward projection, or
    /// every actor lands somewhere the picking code does not expect.
    #[test]
    fn camera_view_agrees_with_world_to_dp() {
        let canvas_dp = [640.0, 360.0];
        let world_size = [800.0, 600.0];
        for center in [
            DVec2::new(0.0, 0.0),
            DVec2::new(400.0, 300.0),
            DVec2::new(913.0, -47.0),
        ] {
            for zoom in [1.0f32, 2.5, 0.4] {
                let camera = Camera2d {
                    center: center.as_vec2(),
                    zoom,
                    ..Camera2d::default()
                };
                let view = camera_view(&camera, canvas_dp, world_size);
                let affine = view_affine(&view);
                for world in [
                    DVec2::new(0.0, 0.0),
                    DVec2::new(400.0, 300.0),
                    DVec2::new(799.0, 599.0),
                ] {
                    let via_engine = world_to_dp(
                        [world.x as f32, world.y as f32],
                        world_size,
                        camera.effective_center(),
                        effective_fit(canvas_dp, world_size, &camera),
                    );
                    let via_actor = apply(affine, world);
                    assert!(
                        close(
                            via_actor,
                            DVec2::new(via_engine[0] as f64, via_engine[1] as f64)
                        ),
                        "center={center:?} zoom={zoom} world={world:?}: actor {via_actor:?} vs engine {via_engine:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn compose_is_apply_a_then_b() {
        let a = [2.0, 0.0, 0.0, 2.0, 10.0, 20.0];
        let b = [1.0, 0.0, 0.0, 1.0, -4.0, 6.0];
        let p = DVec2::new(3.0, 7.0);
        assert!(close(apply(compose(a, b), p), apply(b, apply(a, p))));
    }

    #[test]
    fn compose_is_associative() {
        let a = [2.0, 0.5, -0.5, 3.0, 10.0, 20.0];
        let b = [1.0, 0.25, 0.75, 1.0, -4.0, 6.0];
        let c = [0.5, 0.0, 0.0, 0.5, 100.0, -50.0];
        let left = compose(compose(a, b), c);
        let right = compose(a, compose(b, c));
        for p in [DVec2::ZERO, DVec2::new(1.0, 2.0), DVec2::new(-9.0, 4.5)] {
            assert!(close(apply(left, p), apply(right, p)));
        }
    }

    /// Anchored placement: the anchor point of the cell must land on `world`.
    #[test]
    fn instance_affine_places_the_anchor_on_world() {
        let size = DVec2::new(24.0, 32.0);
        for anchor in [
            DVec2::new(0.5, 0.5),
            DVec2::ZERO,
            DVec2::ONE,
            DVec2::new(0.25, 0.75),
        ] {
            let draw = ActorDraw {
                world: DVec2::new(120.0, -40.0),
                anchor,
                ..ActorDraw::default()
            };
            let affine = instance_affine(&draw, size);
            assert!(
                close(apply(affine, anchor * size), draw.world),
                "anchor {anchor:?}"
            );
        }
    }

    #[test]
    fn instance_affine_rotation_and_flip() {
        let size = DVec2::new(10.0, 10.0);

        // A quarter turn sends the point half a cell right of the anchor to half a
        // cell below it, because y grows downward.
        let turned = ActorDraw {
            world: DVec2::ZERO,
            anchor: DVec2::splat(0.5),
            rotation: std::f64::consts::FRAC_PI_2,
            ..ActorDraw::default()
        };
        assert!(close(
            apply(instance_affine(&turned, size), DVec2::new(10.0, 5.0)),
            DVec2::new(0.0, 5.0)
        ));

        // A negative x scale mirrors about the anchor, which is not the same as
        // mirroring about the cell origin.
        let mirrored = ActorDraw {
            world: DVec2::splat(5.0),
            anchor: DVec2::splat(0.5),
            scale: DVec2::new(-1.0, 1.0),
            ..ActorDraw::default()
        };
        assert!(close(
            apply(instance_affine(&mirrored, size), DVec2::new(10.0, 5.0)),
            DVec2::new(0.0, 5.0)
        ));
    }

    /// The load-bearing claim: retargeting the prepared transform must place every
    /// vertex exactly where baking the same placement into the scene before
    /// `prepare` would have.
    #[test]
    fn retarget_matches_baking_the_placement_into_the_scene() {
        let size = DVec2::new(24.0, 32.0);
        let doc = rig(size, Color::rgba(0.2, 0.6, 0.9, 1.0));
        let scene = renamite_model::evaluate(&doc, doc.main, 0.0);
        let view = ViewTransform {
            scale: 2.5,
            offset: DVec2::new(13.0, -7.0),
        };

        for draw in [
            ActorDraw::default(),
            ActorDraw {
                world: DVec2::new(300.0, -125.0),
                ..ActorDraw::default()
            },
            ActorDraw {
                world: DVec2::new(-64.0, 41.0),
                rotation: 0.7,
                scale: DVec2::new(1.75, 0.5),
                anchor: DVec2::new(0.0, 1.0),
                ..ActorDraw::default()
            },
            ActorDraw {
                world: DVec2::new(10.0, 20.0),
                rotation: -2.1,
                scale: DVec2::new(-1.0, 2.0),
                anchor: DVec2::new(0.25, 0.75),
                ..ActorDraw::default()
            },
        ] {
            let instance = instance_affine(&draw, size);

            let fast = SceneRenderer::new().prepare(&scene, &view);
            let fast = retarget_draws(&fast.draws, instance);

            let mut baked = scene.clone();
            for item in &mut baked.items {
                item.path.apply_affine(kurbo::Affine::new(instance));
            }
            let slow = SceneRenderer::new().prepare(&baked, &view);

            assert_eq!(fast.len(), slow.draws.len());
            for (fast, slow) in fast.iter().zip(&slow.draws) {
                let (
                    PreparedDraw::Vector {
                        mesh: fast_mesh,
                        transform: fast_transform,
                        ..
                    },
                    PreparedDraw::Vector {
                        mesh: slow_mesh,
                        transform: slow_transform,
                        ..
                    },
                ) = (fast, slow)
                else {
                    panic!("expected vector draws");
                };
                // Tessellation quantises, so compare the transformed extents
                // rather than vertex-for-vertex identity.
                let bounds = |mesh: &repose_core::VectorMeshData, t: &[f32; 6]| {
                    mesh.vertices
                        .iter()
                        .map(|v| {
                            DVec2::new(
                                f64::from(t[0]) * f64::from(v.pos[0])
                                    + f64::from(t[2]) * f64::from(v.pos[1])
                                    + f64::from(t[4]),
                                f64::from(t[1]) * f64::from(v.pos[0])
                                    + f64::from(t[3]) * f64::from(v.pos[1])
                                    + f64::from(t[5]),
                            )
                        })
                        .fold(
                            (DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY)),
                            |(lo, hi), p| (lo.min(p), hi.max(p)),
                        )
                };
                let (fast_bounds, slow_bounds) = (
                    bounds(fast_mesh, fast_transform),
                    bounds(slow_mesh, slow_transform),
                );
                assert!(
                    close(fast_bounds.0, slow_bounds.0) && close(fast_bounds.1, slow_bounds.1),
                    "placement {draw:?}\nfast {fast_bounds:?}\nslow {slow_bounds:?}"
                );
            }
        }
    }

    #[test]
    fn tint_scene_scales_only_solid_fills() {
        let doc = rig(DVec2::new(4.0, 4.0), Color::rgba(0.5, 0.5, 0.5, 1.0));
        let scene = renamite_model::evaluate(&doc, doc.main, 0.0);
        let tinted = tint_scene(&scene, DVec4::new(0.5, 0.25, 1.0, 0.5));
        let ScenePaint::Solid(color) = tinted.items[0].paint else {
            panic!("expected solid");
        };
        assert!((color.r - 0.25).abs() < 1e-9, "r was {}", color.r);
        assert!((color.g - 0.125).abs() < 1e-9, "g was {}", color.g);
        assert!((color.b - 0.5).abs() < 1e-9, "b was {}", color.b);
        assert!((color.a - 0.5).abs() < 1e-9, "a was {}", color.a);
    }

    #[test]
    fn library_round_trips_rigs_and_reports_ids() {
        let mut library = ActorLibrary::new();
        assert!(library.is_empty());
        let first = library
            .push_ren_str(crate::MINIMAL_REN)
            .expect("minimal document parses");
        let second = library
            .push_ren_str(crate::MINIMAL_REN)
            .expect("minimal document parses");
        assert_eq!(library.len(), 2);
        assert_eq!(first, ActorId(0));
        assert_eq!(second, ActorId(1));
        assert!(library.host(second).is_some());
        assert!(library.host(ActorId(2)).is_none());
    }

    #[test]
    fn draw_list_sorts_by_z() {
        let mut list = ActorDrawList::new();
        for z in [3.0, -1.0, 2.0, 0.0] {
            list.push(ActorDraw {
                z,
                ..ActorDraw::default()
            });
        }
        let mut sorted = list.as_slice().to_vec();
        sorted.sort_by(|a, b| a.z.partial_cmp(&b.z).unwrap());
        assert_eq!(
            sorted.iter().map(|d| d.z).collect::<Vec<_>>(),
            vec![-1.0, 0.0, 2.0, 3.0]
        );
    }

    #[test]
    fn mixed_media_share_one_z_order() {
        let rig = |z: f32| ActorDraw {
            item: ActorItem::Rig {
                actor: ActorId(0),
                frame: 0.0,
            },
            z,
            ..ActorDraw::default()
        };
        let raster = |z: f32| ActorDraw {
            item: ActorItem::Raster {
                page: 0,
                uv_min: [0.0, 0.0],
                uv_max: [1.0, 1.0],
                size: [8.0, 8.0],
            },
            z,
            ..ActorDraw::default()
        };
        let rect = |z: f32| ActorDraw {
            item: ActorItem::Rect { half: [4.0, 4.0] },
            z,
            ..ActorDraw::default()
        };
        let mut draws = [rig(1.0), raster(0.0), rect(2.0), rig(0.5)];
        draws.sort_by(compare_draws);
        let media: Vec<Medium> = draws.iter().map(|d| medium_key(d.item)).collect();
        assert_eq!(
            media,
            vec![Medium::Raster, Medium::Rig, Medium::Rig, Medium::Rect],
            "z is the major key across media: raster@0, rig@0.5, rig@1, rect@2"
        );
    }

    #[test]
    fn world_view_constructs_without_a_backend() {
        let mut library = ActorLibrary::new();
        library.push_ren_str(crate::MINIMAL_REN).unwrap();
        let mut draws = ActorDrawList::new();
        draws.push(ActorDraw {
            world: DVec2::new(12.0, 34.0),
            ..ActorDraw::default()
        });
        let ctx = RenderContext::new();
        let _view = actors_view_world(
            Rc::new(RefCell::new(library)),
            Rc::new(draws),
            Camera2d::default(),
            [800.0, 600.0],
            ctx,
        );
    }

    /// `PickState::project` returns window-physical px and unprojects the world
    /// point from dp. Emitting dp made every consumer's own px->dp conversion a
    /// second divide, landing clicks at half their position.
    #[test]
    fn project_emits_physical_px_and_unprojects_from_dp() {
        let canvas_dp = [800.0, 600.0];
        let world_size = [320.0, 240.0];
        let camera = Camera2d::default();
        let density = effective_density_scale().max(1e-6);
        let pick = PickState {
            sink: Rc::new(|_| {}),
            camera,
            world_size,
            canvas_dp: Rc::new(RefCell::new(canvas_dp)),
            press: HashMap::new(),
        };
        let physical = repose_core::Vec2 { x: 640.0, y: 480.0 };
        let (screen, world) = pick.project(physical);
        assert_eq!(screen, [640.0, 480.0], "screen must stay physical px");
        let dp = [640.0 / density, 480.0 / density];
        let fit = effective_fit(canvas_dp, world_size, &camera);
        let want = dp_to_world(dp, world_size, camera.effective_center(), fit);
        assert!(
            (world.x - want[0]).abs() < 1e-3 && (world.y - want[1]).abs() < 1e-3,
            "world must unproject through the dp screen, got {world:?} want {want:?}"
        );
    }
}
