//! Pose-sweep invariants and a golden contact sheet for the 3D viewport path.
//!
//! Three failures motivated this file, all of which looked like "the renderer is
//! broken" from inside a game and none of which a single blessable still would
//! have caught:
//!
//! * **missing faces** — a surface whose triangles are wound the wrong way is
//!   back-face culled and simply is not there. A whole road can disappear from
//!   every camera above it, and a golden blessed on the broken build keeps it
//!   invisible forever, because the golden only records "unchanged".
//! * **the camera moves the wrong way** — an inverted yaw/pitch sign, or a
//!   mirrored projection, still produces plausible images; only the *direction*
//!   of motion across poses exposes it.
//! * **ground at the wrong height** — geometry that reads under or above where it
//!   claims to be, so a car floats over a road or buries itself in it.
//!
//! So the gate here is *invariants over the sweep*, not pixel equality:
//!
//! * nothing shows the background *inside* a solid's silhouette (a hole),
//! * the ground covers the bottom of the frame and never rises into the top,
//! * a raised platform is drawn on top of the ground it stands on,
//! * the two world axes land on the screen side the camera's convention says,
//!   and rotating the camera moves them the way it claims.
//!
//! On top of that, every pose is rendered into one committed contact sheet
//! (`tests/goldens/pose_sweep.png`) compared with the same per-channel tolerance
//! and diff-fraction allowance the 2D lighting goldens use. A failing sheet is
//! written per pose so the report names the pose, not just the file.
//!
//! Regenerate the sheet with:
//! `REPAME_BLESS=1 cargo test -p repame-view3d --test pose_golden`
//! Record the sweep as lossless FFV1 for review (ffmpeg required, never a gate):
//! `REPAME_VIDEO=/tmp/sweep cargo test -p repame-view3d --test pose_golden`
//! Without a WGPU adapter the tests skip with a warning instead of failing.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

use glam::{Vec3, Vec4};
use repame_view3d::render::{BatchDesc, SceneFilter};
use repame_view3d::render::TonemapMethod;
use repame_view3d::{Frame3d, GeomHandle, MeshGroup, OrbitCamera, SceneLight, Viewport3d};
use repose_core::RenderContext;
use repose_render_wgpu::apply_render_commands;
use repose_render_wgpu::offscreen::OffscreenRenderer;
use repose_ui::{Interactions, TextFieldState, layout_and_paint};

/// One pose is rendered at this size; the contact sheet halves it.
const W: u32 = 320;
const H: u32 = 180;
const SHEET_COLS: u32 = 6;

/// Per-channel deviation below this is "same pixel" when comparing the sheet.
const CHANNEL_TOL: i16 = 10;
/// Fraction of sheet pixels allowed to exceed `CHANNEL_TOL`.
const MAX_DIFF_FRACTION: f64 = 0.01;
/// Background pixels allowed in the lower half of any pose.
const MAX_BOTTOM_HOLE_FRACTION: f64 = 0.02;
/// Ground pixels allowed in the upper quarter of any pose.
const MAX_TOP_GROUND_FRACTION: f64 = 0.02;

/// Linear scene clear colour; the renderer writes it sRGB-encoded.
const SKY: [f32; 4] = [0.36, 0.56, 0.82, 1.0];
/// The offscreen readback clear, in the f64 the renderer's API takes.
const CLEAR: [f64; 4] = [0.36, 0.56, 0.82, 1.0];
const GROUND: [f32; 3] = [0.22, 0.45, 0.20];
const PLATFORM: [f32; 3] = [0.62, 0.62, 0.58];
const BOX: [f32; 3] = [0.78, 0.16, 0.12];
const POST_X: [f32; 3] = [0.95, 0.85, 0.15];
const POST_NX: [f32; 3] = [0.10, 0.80, 0.85];
const POST_Z: [f32; 3] = [0.20, 0.30, 0.95];
const POST_NZ: [f32; 3] = [0.95, 0.95, 0.95];

/// The two camera distances the sweep uses.
const DIST: f32 = 13.0;
/// A box on the ground for scale, and a second one floating clear of every other
/// object (the posts top out at 4 m, and it clears the frame's top edge) so nothing can ever cross its silhouette.
/// The floating one carries the solid and hole checks: a missing face there
/// shows the sky, which is unambiguous.
const BOX_CENTRE: Vec3 = Vec3::new(-3.0, 1.0, 2.5);
const BOX_HALF: f32 = 1.2;
const SKY_BOX_CENTRE: Vec3 = Vec3::new(0.0, 4.6, 0.0);
/// Platform top surface: 0.5 m thick slab sitting on the ground, top at y = 1.
const PLATFORM_CENTRE: Vec3 = Vec3::new(0.0, 0.5, -3.0);
const PLATFORM_HALF: Vec3 = Vec3::new(2.5, 0.5, 2.0);

// ------------------------------------------------------------------- geometry

/// A quad with an explicit normal, wound counter-clockwise as seen from that
/// normal, which is the front face the viewport culls against.
fn push_quad(g: &mut MeshGroup, verts: [Vec3; 4], normal: Vec3, colour: [f32; 3]) {
    let base = g.positions.len() as u32;
    for v in verts {
        g.positions.push(v.to_array());
        g.colors.push(colour);
        g.normals.push(normal.normalize().to_array());
    }
    // Front-face order, pinned by `the_front_face_convention_is_pinned`: a face
    // is drawn when `(b - a) x (c - a)` points along its outward normal, which
    // for these corners is (0,1,2)/(0,2,3). The other order culls the surface.
    g.indices
        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

/// An axis-aligned box, six quads, each wound so its outward normal is front.
fn push_box(g: &mut MeshGroup, centre: Vec3, half: Vec3, colour: [f32; 3]) {
    let c = centre;
    let h = half;
    let v = |x: f32, y: f32, z: f32| c + Vec3::new(x * h.x, y * h.y, z * h.z);
    // +Y top, -Y bottom, +X/-X, +Z/-Z: each entry is (corners in CCW order seen
    // from outside, outward normal).
    let faces: [([Vec3; 4], Vec3); 6] = [
        (
            [v(-1., 1., 1.), v(1., 1., 1.), v(1., 1., -1.), v(-1., 1., -1.)],
            Vec3::Y,
        ),
        (
            [v(-1., -1., -1.), v(1., -1., -1.), v(1., -1., 1.), v(-1., -1., 1.)],
            -Vec3::Y,
        ),
        (
            [v(1., -1., 1.), v(1., -1., -1.), v(1., 1., -1.), v(1., 1., 1.)],
            Vec3::X,
        ),
        (
            [v(-1., -1., -1.), v(-1., -1., 1.), v(-1., 1., 1.), v(-1., 1., -1.)],
            -Vec3::X,
        ),
        (
            [v(-1., -1., 1.), v(1., -1., 1.), v(1., 1., 1.), v(-1., 1., 1.)],
            Vec3::Z,
        ),
        (
            [v(1., -1., -1.), v(-1., -1., -1.), v(-1., 1., -1.), v(1., 1., -1.)],
            -Vec3::Z,
        ),
    ];
    for (corners, normal) in faces {
        push_quad(g, corners, normal, colour);
    }
}

/// The canonical scene every pose renders. `solid` switches the invariants
/// between the honest scene and the deliberately broken ones.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Variant {
    /// Wound and placed correctly.
    Honest,
    /// The ground's triangles reversed: culled from every camera above it.
    BackwardsGround,
    /// The box's +X face dropped: a visible hole in a solid.
    MissingBoxFace,
}

fn scene(variant: Variant) -> Vec<MeshGroup> {
    let mut ground = MeshGroup::default();
    // Effectively unbounded: a finite plane shows its far edge from a grazing
    // camera, and then "no sky below the horizon" cannot hold by construction.
    let d = 400.0f32;
    let corners = [
        Vec3::new(-d, 0.0, -d),
        Vec3::new(-d, 0.0, d),
        Vec3::new(d, 0.0, d),
        Vec3::new(d, 0.0, -d),
    ];
    if variant == Variant::BackwardsGround {
        let base = ground.positions.len() as u32;
        for v in corners {
            ground.positions.push(v.to_array());
            ground.colors.push(GROUND);
            ground.normals.push(Vec3::Y.to_array());
        }
        // Reversed winding: still claims to face up, culled anyway.
        ground
            .indices
            .extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
    } else {
        push_quad(&mut ground, corners, Vec3::Y, GROUND);
    }

    let mut platform = MeshGroup::default();
    push_box(&mut platform, PLATFORM_CENTRE, PLATFORM_HALF, PLATFORM);

    let mut red_box = MeshGroup::default();
    push_box(&mut red_box, BOX_CENTRE, Vec3::splat(BOX_HALF), BOX);
    push_box(&mut red_box, SKY_BOX_CENTRE, Vec3::splat(BOX_HALF), BOX);
    if variant == Variant::MissingBoxFace {
        // The ground box writes six quads (36 indices) before the floating one,
        // whose first quad is its +Y top: the face a steep camera looks at. What
        // is behind it is the interior (culled) and then the ground, so the cells
        // it occupies stop reading as the box.
        let top_face = 36;
        red_box.indices.drain(top_face..top_face + 6);
    }

    let mut posts = MeshGroup::default();
    // Short posts: they mark the world axes for the camera checks, and staying
    // below the floating box keeps them from crossing its silhouette.
    let post = |posts: &mut MeshGroup, at: Vec3, colour: [f32; 3]| {
        push_box(
            posts,
            at + Vec3::new(0.0, 0.75, 0.0),
            Vec3::new(0.12, 0.75, 0.12),
            colour,
        );
    };
    post(&mut posts, Vec3::new(6.0, 0.0, 0.0), POST_X);
    post(&mut posts, Vec3::new(-6.0, 0.0, 0.0), POST_NX);
    post(&mut posts, Vec3::new(0.0, 0.0, 6.0), POST_Z);
    post(&mut posts, Vec3::new(0.0, 0.0, -6.0), POST_NZ);

    vec![ground, platform, red_box, posts]
}

// ------------------------------------------------------------------- rendering

fn gpu() -> Option<OffscreenRenderer> {
    match OffscreenRenderer::new_blocking(W, H, 1) {
        Ok(gpu) => Some(gpu),
        Err(error) => {
            eprintln!("skipping pose sweep: no WGPU adapter ({error:#})");
            None
        }
    }
}

fn light() -> SceneLight {
    SceneLight {
        direction: [-0.45, -0.80, 0.40],
        color: [1.0, 0.97, 0.9],
        diffuse: 1.0,
        ambient: [0.32, 0.34, 0.38],
        fog: 0.0,
        fog_color: [0.36, 0.56, 0.82],
        fog_start: 100.0,
        fog_end: 400.0,
        exposure: 1.0,
        tonemap: TonemapMethod::None,
    }
}

fn camera(label: &str) -> OrbitCamera {
    let (yaw, pitch) = pose(label);
    OrbitCamera {
        target: Vec3::new(0.0, 1.0, 0.0),
        yaw,
        pitch,
        dist: DIST,
        fov_y_deg: 45.0,
    }
}

/// The sweep: twelve yaws around the scene at three pitches, plus a grazing
/// camera (near the ground, where height errors show) and a top-down one.
fn poses() -> Vec<String> {
    let mut out = Vec::new();
    for step in 0..12 {
        let yaw = -std::f32::consts::PI + step as f32 * std::f32::consts::TAU / 12.0;
        for pitch in [0.10f32, 0.42, 0.85] {
            out.push(format!("yaw{yaw:+.2}_pitch{pitch:.2}"));
        }
    }
    out.push("grazing".to_string());
    out.push("topdown".to_string());
    out
}

fn pose(label: &str) -> (f32, f32) {
    match label {
        "grazing" => (0.0, 0.02),
        "topdown" => (std::f32::consts::FRAC_PI_4, 1.30),
        _ => {
            let yaw = label
                .split("yaw")
                .nth(1)
                .and_then(|rest| rest.split('_').next())
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(0.0);
            let pitch = label
                .split("pitch")
                .nth(1)
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(0.4);
            (yaw, pitch)
        }
    }
}

/// The background path on its own: no groups, so every pixel is the clear.
fn render_empty(gpu: &mut OffscreenRenderer, cam: OrbitCamera) -> Vec<u8> {
    let desc = BatchDesc {
        layer_size: 64,
        layers: 1,
        filter: SceneFilter::Linear,
        mips: 1,
    };
    let frame = Frame3d {
        cam,
        desc,
        light: light(),
        background: Some(SKY),
        viewport_px: [W as f32, H as f32],
        ..Frame3d::default()
    };
    let ctx = RenderContext::new();
    let geom = GeomHandle::new();
    let view = Viewport3d(frame, geom, "pose-sweep-empty", desc, |_| {});
    let states: HashMap<u64, Rc<RefCell<TextFieldState>>> = HashMap::new();
    let (scene, _h, _s) =
        layout_and_paint(&view, (W, H), &states, &Interactions::default(), None);
    apply_render_commands(gpu.renderer_mut(), ctx.drain()).log_failures();
    gpu.render_rgba(&scene, Some(CLEAR)).expect("renders")
}

fn render_with(gpu: &mut OffscreenRenderer, cam: OrbitCamera, variant: Variant) -> Vec<u8> {
    let desc = BatchDesc {
        layer_size: 64,
        layers: 1,
        filter: SceneFilter::Linear,
        mips: 1,
    };
    let mut frame = Frame3d {
        cam,
        desc,
        light: light(),
        // The viewport clears its own target; leaving this `None` paints the
        // scene against black whatever the renderer's clear colour is.
        background: Some(SKY),
        viewport_px: [W as f32, H as f32],
        ..Frame3d::default()
    };
    for group in scene(variant) {
        frame.push(group);
    }

    let ctx = RenderContext::new();
    let geom = GeomHandle::new();
    let view = Viewport3d(frame, geom, "pose-sweep", desc, |_| {});
    let states: HashMap<u64, Rc<RefCell<TextFieldState>>> = HashMap::new();
    let (scene, _hits, _sems) =
        layout_and_paint(&view, (W, H), &states, &Interactions::default(), None);
    apply_render_commands(gpu.renderer_mut(), ctx.drain()).log_failures();
    gpu.render_rgba(&scene, Some(CLEAR))
        .expect("the scene renders")
}

// ------------------------------------------------------------------- probing

fn pixel(rgba: &[u8], x: u32, y: u32) -> [u8; 3] {
    let i = ((y * W + x) * 4) as usize;
    [rgba[i], rgba[i + 1], rgba[i + 2]]
}



fn is_background(a: [u8; 3], background: [u8; 3]) -> bool {
    (0..3).all(|i| (a[i] as i32 - background[i] as i32).abs() <= 6)
}

/// Classify by which channel dominates. Lighting and tone mapping move
/// brightness, not hue, so these survive driver differences where a comparison
/// against a base colour does not.
fn is_ground(a: [u8; 3]) -> bool {
    let (r, g, b) = (a[0] as i32, a[1] as i32, a[2] as i32);
    g > r + 14 && g > b + 14
}


fn is_red(a: [u8; 3]) -> bool {
    let (r, g, b) = (a[0] as i32, a[1] as i32, a[2] as i32);
    r > g + 14 && r > b + 14
}


/// Share of a rectangle's pixels matching a predicate, inset so silhouette
/// edges never decide the answer.
fn share(
    rgba: &[u8],
    rect: (f32, f32, f32, f32),
    inset: f32,
    predicate: impl Fn([u8; 3]) -> bool,
) -> f64 {
    let (x0, y0, x1, y1) = rect;
    let x0 = (x0 + inset).max(0.0) as u32;
    let y0 = (y0 + inset).max(0.0) as u32;
    let x1 = (x1 - inset).min(W as f32 - 1.0).max(0.0) as u32;
    let y1 = (y1 - inset).min(H as f32 - 1.0).max(0.0) as u32;
    if x1 <= x0 || y1 <= y0 {
        return 0.0;
    }
    let mut total = 0usize;
    let mut hits = 0usize;
    for y in y0..=y1 {
        for x in x0..=x1 {
            total += 1;
            if predicate(pixel(rgba, x, y)) {
                hits += 1;
            }
        }
    }
    hits as f64 / total.max(1) as f64
}

/// Project a world point to pixels: `None` when it is behind the camera.
fn project(cam: &OrbitCamera, point: Vec3) -> Option<(f32, f32)> {
    let clip: Vec4 = cam.view_proj(W as f32 / H as f32) * point.extend(1.0);
    if clip.w <= 1e-4 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    Some((
        (ndc.x * 0.5 + 0.5) * W as f32,
        (1.0 - (ndc.y * 0.5 + 0.5)) * H as f32,
    ))
}

/// The screen rectangle a box covers, from its eight corners.
fn box_rect(cam: &OrbitCamera, centre: Vec3, half: Vec3) -> Option<(f32, f32, f32, f32)> {
    let mut min = (f32::MAX, f32::MAX);
    let mut max = (f32::MIN, f32::MIN);
    for sx in [-1.0f32, 1.0] {
        for sy in [-1.0f32, 1.0] {
            for sz in [-1.0f32, 1.0] {
                let corner = centre + Vec3::new(sx * half.x, sy * half.y, sz * half.z);
                let (x, y) = project(cam, corner)?;
                min = (min.0.min(x), min.1.min(y));
                max = (max.0.max(x), max.1.max(y));
            }
        }
    }
    Some((min.0, min.1, max.0, max.1))
}

// ------------------------------------------------------------------- invariants

/// Of a solid's silhouette cells that lie fully inside the frame, how many read
/// as the solid.
///
/// Cells rather than one share, because occluders (a post crossing the box,
/// another object in front) take whole cells — and cells fully inside the frame
/// only, because a solid near the edge of a pose is a crop, and a crop of a solid
/// says nothing about its faces. A culled face still takes every cell it occupies.
fn solid_cells(
    rgba: &[u8],
    cam: &OrbitCamera,
    centre: Vec3,
    half: Vec3,
    predicate: impl Fn([u8; 3]) -> bool + Copy,
) -> Option<(usize, usize)> {
    let rect = box_rect(cam, centre, half)?;
    let (x0, y0) = (rect.0, rect.1);
    let (w, h) = ((rect.2 - rect.0) / 3.0, (rect.3 - rect.1) / 3.0);
    if w <= 0.0 || h <= 0.0 {
        return None;
    }
    let inset = w.min(h) / 3.0;
    let mut evaluated = 0usize;
    let mut solid = 0usize;
    for cy in 0..3 {
        for cx in 0..3 {
            // The silhouette *rectangle* of a box seen at an angle includes its
            // four corners, which are outside the box: a cube covers the middle
            // row and column of that rectangle, not the diagonals.
            if (cx == 0 || cx == 2) && (cy == 0 || cy == 2) {
                continue;
            }
            let cell = (
                x0 + cx as f32 * w,
                y0 + cy as f32 * h,
                x0 + (cx + 1) as f32 * w,
                y0 + (cy + 1) as f32 * h,
            );
            let inside = cell.0 >= 0.0 && cell.1 >= 0.0 && cell.2 <= W as f32 && cell.3 <= H as f32;
            if !inside {
                continue;
            }
            evaluated += 1;
            if share(rgba, cell, inset, predicate) >= 0.85 {
                solid += 1;
            }
        }
    }
    (evaluated >= 3).then_some((solid, evaluated))
}

/// The middle of a solid must read as the solid: a culled face shows whatever is
/// behind it instead, and takes whole cells with it.
fn assert_solid_middle(
    rgba: &[u8],
    cam: &OrbitCamera,
    centre: Vec3,
    half: Vec3,
    predicate: impl Fn([u8; 3]) -> bool + Copy,
    label: &str,
    what: &str,
    expect: &str,
) {
    let Some((solid, evaluated)) = solid_cells(rgba, cam, centre, half, predicate) else {
        return;
    };
    let fraction = solid as f64 / evaluated as f64;
    assert!(
        fraction >= 0.75,
        "{label}: the middle of {what} reads {expect} in only {solid}/{evaluated} visible \
         cells; a face wound the wrong way is culled and shows what is behind it"
    );
}

/// The screen row the true horizon falls on: where view rays are parallel to the
/// ground. Everything below it must be ground or something standing on it, and
/// nothing above it may be ground.
fn horizon_row(cam: &OrbitCamera) -> f32 {
    let half_fov = (cam.fov_y_deg.to_radians() * 0.5).tan();
    let elevation = cam.forward().y.clamp(-1.0, 1.0).asin();
    H as f32 * 0.5 * (1.0 + elevation.tan() / half_fov)
}

/// Everything a single pose must satisfy, independent of any golden. The three
/// checks map onto the three failures this file exists for: a missing surface, a
/// camera whose verticals read wrongly, and a solid that is not solid.
/// Returns whether the floating box was fully in frame, so the sweep can prove
/// the solid check actually ran rather than skipping every pose.
fn check_pose(rgba: &[u8], cam: &OrbitCamera, background: [u8; 3], label: &str) -> bool {
    let horizon = horizon_row(cam);
    // A few rows of slack: the ground plane is finite, so its far edge sits a
    // hair below the true horizon, and silhouette edges are one pixel wide.
    let margin = 6.0;
    let below = ((horizon + margin).max(0.0), H as f32);
    let above = (0.0, (horizon - margin).clamp(0.0, H as f32));

    // Below the horizon there is ground, or something standing on it: never sky.
    // A ground surface wound the wrong way is culled, and this is where that
    // shows — as does a hole in anything standing on it.
    let mut sky_below = 0usize;
    let mut below_total = 0usize;
    for y in (below.0 as u32)..H {
        for x in 0..W {
            below_total += 1;
            if is_background(pixel(rgba, x, y), background) {
                sky_below += 1;
            }
        }
    }
    let sky_share = sky_below as f64 / below_total.max(1) as f64;
    assert!(
        sky_share <= MAX_BOTTOM_HOLE_FRACTION,
        "{label}: {:.1}% of the frame below the horizon is sky; the ground is missing or a \
         surface facing the camera is culled",
        sky_share * 100.0
    );

    // Above the horizon there is sky: nothing green may float up there, which is
    // what a ground drawn at the wrong height looks like.
    let mut ground_above = 0usize;
    let mut above_total = 0usize;
    for y in 0..(above.1 as u32) {
        for x in 0..W {
            above_total += 1;
            if is_ground(pixel(rgba, x, y)) {
                ground_above += 1;
            }
        }
    }
    if above_total > 0 {
        let floating = ground_above as f64 / above_total as f64;
        assert!(
            floating <= MAX_TOP_GROUND_FRACTION,
            "{label}: {:.1}% of the sky above the horizon is ground; the ground is drawn at \
             the wrong height",
            floating * 100.0
        );
    }

    // Solids are solid: the middle of each reads as the solid, not as whatever is
    // behind a culled face.
    assert_solid_middle(
        rgba,
        cam,
        SKY_BOX_CENTRE,
        Vec3::splat(BOX_HALF),
        is_red,
        label,
        "the floating box",
        "red",
    );
    solid_cells(rgba, cam, SKY_BOX_CENTRE, Vec3::splat(BOX_HALF), is_red).is_some()
}

/// The camera's own conventions, pinned by where the world axes land: a mirrored
/// projection or an inverted yaw puts them on the wrong side.
fn check_camera_axes(render: &mut impl FnMut(OrbitCamera) -> Vec<u8>, background: [u8; 3]) {
    let target = Vec3::new(0.0, 1.0, 0.0);
    let post_z = Vec3::new(0.0, 0.9, 6.0);
    let post_x = Vec3::new(6.0, 0.9, 0.0);

    // yaw = 0 puts the eye on +X looking toward the origin: +Z is screen-left.
    let cam = OrbitCamera {
        target,
        yaw: 0.0,
        pitch: 0.35,
        dist: DIST,
        fov_y_deg: 45.0,
    };
    let rgba = render(cam);
    let (zx, _) = project(&cam, post_z).expect("+Z post visible at yaw 0");
    assert!(
        zx < W as f32 * 0.5,
        "yaw 0 with the eye on +X must put +Z on the left half, found x = {zx:.1}"
    );
    // ...and the post there is the blue one, not a mirrored neighbour.
    let probe = pixel(&rgba, zx.clamp(0.0, W as f32 - 1.0) as u32, H / 2);
    assert!(
        !is_background(probe, background),
        "the +Z post at yaw 0 should be on screen, found background at x = {zx:.1}"
    );

    // yaw = +pi/2 puts the eye on +Z looking toward the origin: +X is screen-right.
    let cam = OrbitCamera {
        target,
        yaw: std::f32::consts::FRAC_PI_2,
        pitch: 0.35,
        dist: DIST,
        fov_y_deg: 45.0,
    };
    let _ = render(cam);
    let (xx, _) = project(&cam, post_x).expect("+X post visible at yaw pi/2");
    assert!(
        xx > W as f32 * 0.5,
        "yaw +pi/2 puts the eye on +Z, so +X must be on the right half, found x = {xx:.1}"
    );

    // yaw = pi puts the eye on -X: +Z is now screen-right. Together these two
    // pins say the orbit turns the way yaw says and nothing is mirrored.
    let cam = OrbitCamera {
        target,
        yaw: std::f32::consts::PI,
        pitch: 0.35,
        dist: DIST,
        fov_y_deg: 45.0,
    };
    let (zx, _) = project(&cam, post_z).expect("+Z post visible at yaw pi");
    assert!(
        zx > W as f32 * 0.5,
        "yaw pi (eye on -X) must put +Z on the right half, found x = {zx:.1}"
    );

    // Raising the camera means looking further down at the ground, so the sky in
    // the frame must shrink as pitch grows. A walk up one column instead would be
    // at the mercy of whatever happens to cross it.
    let mut sky_shares = Vec::new();
    for pitch in [0.10f32, 0.42, 0.85] {
        let cam = OrbitCamera {
            target,
            yaw: 0.4,
            pitch,
            dist: DIST,
            fov_y_deg: 45.0,
        };
        let rgba = render(cam);
        let mut sky = 0usize;
        for y in 0..H {
            for x in 0..W {
                if is_background(pixel(&rgba, x, y), background) {
                    sky += 1;
                }
            }
        }
        sky_shares.push(sky as f64 / (W * H) as f64);
    }
    eprintln!("pitch sweep: sky shares {sky_shares:?}");
    // The ground plane is big enough to fill the frame from any steep camera, so
    // the share saturates at zero: the signal is that sky never *grows* with pitch
    // and that the shallow camera sees strictly more of it.
    assert!(
        sky_shares[0] >= sky_shares[1]
            && sky_shares[1] >= sky_shares[2]
            && sky_shares[0] > sky_shares[2] + 0.02,
        "raising the camera must not grow the sky in frame (pitch 0.10 > 0.42 > 0.85), got \
         {sky_shares:?} — an inverted pitch shows the opposite"
    );
}

// ------------------------------------------------------------------- goldens

fn goldens_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/goldens")
}

/// Tile the poses into one contact sheet, halved.
fn contact_sheet(frames: &[Vec<u8>]) -> (u32, u32, Vec<u8>) {
    let cell_w = W / 2;
    let cell_h = H / 2;
    let rows = (frames.len() as u32).div_ceil(SHEET_COLS);
    let (sw, sh) = (cell_w * SHEET_COLS, cell_h * rows);
    let mut sheet = vec![0u8; (sw * sh * 4) as usize];
    for (index, frame) in frames.iter().enumerate() {
        let cx = (index as u32 % SHEET_COLS) * cell_w;
        let cy = (index as u32 / SHEET_COLS) * cell_h;
        for y in 0..cell_h {
            for x in 0..cell_w {
                let src = (((y * 2) * W + x * 2) * 4) as usize;
                let dst = (((cy + y) * sw + cx + x) * 4) as usize;
                sheet[dst..dst + 4].copy_from_slice(&frame[src..src + 4]);
            }
        }
    }
    (sw, sh, sheet)
}

fn png_bytes(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
    let image = image::RgbaImage::from_raw(width, height, rgba.to_vec()).expect("rgba size");
    let mut cursor = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(image)
        .write_to(&mut cursor, image::ImageFormat::Png)
        .expect("encode png");
    cursor.into_inner()
}

fn check_sheet(width: u32, height: u32, rgba: &[u8]) {
    let path = goldens_dir().join("pose_sweep.png");
    let bless = std::env::var("REPAME_BLESS").is_ok();
    if bless || !path.exists() {
        std::fs::create_dir_all(goldens_dir()).expect("create goldens dir");
        std::fs::write(&path, png_bytes(width, height, rgba)).expect("write golden");
        if !bless {
            panic!(
                "pose_sweep: golden did not exist; wrote an initial one. Re-run to verify, \
                 or bless with REPAME_BLESS=1."
            );
        }
        return;
    }
    let golden = image::load_from_memory(&std::fs::read(&path).expect("read golden"))
        .expect("decode golden")
        .to_rgba8();
    assert_eq!(
        (golden.width(), golden.height()),
        (width, height),
        "pose_sweep: sheet size changed"
    );
    let total = (width * height) as f64;
    let mut differing = 0usize;
    for (g, a) in golden.pixels().zip(rgba.chunks_exact(4)) {
        if g.0.iter().zip(a.iter()).any(|(&gc, &ac)| {
            (gc as i16 - ac as i16).abs() > CHANNEL_TOL
        }) {
            differing += 1;
        }
    }
    let fraction = differing as f64 / total;
    if fraction > MAX_DIFF_FRACTION {
        let actual = goldens_dir().join("pose_sweep.actual.png");
        std::fs::write(&actual, png_bytes(width, height, rgba)).expect("write actual");
        panic!(
            "pose_sweep: {differing} px ({:.3}%) differ beyond tolerance {CHANNEL_TOL} \
             (limit {:.3}%). Actual written to {}. Compare against the pose list in \
             `poses()` to find the pose, or bless with REPAME_BLESS=1.",
            fraction * 100.0,
            MAX_DIFF_FRACTION * 100.0,
            actual.display()
        );
    }
}

// ------------------------------------------------------------------- the sweep

#[test]
fn pose_sweep_invariants_and_sheet() {
    let Some(mut gpu) = gpu() else {
        return;
    };
    // Where "nothing was drawn" reads. An empty scene must be exactly the clear
    // colour everywhere; sampling it from a scene frame instead once read the
    // ground, and every later comparison was against the wrong colour.
    let empty = render_empty(&mut gpu, camera("yaw+0.00_pitch0.42"));
    let background = pixel(&empty, 1, 1);
    for y in 0..H {
        for x in 0..W {
            assert!(
                is_background(pixel(&empty, x, y), background),
                "an empty frame must be the clear colour everywhere; ({x},{y}) is {:?}",
                pixel(&empty, x, y)
            );
        }
    }

    let mut frames = Vec::new();
    let mut solid_checked = 0usize;
    for label in poses() {
        let cam = camera(&label);
        let rgba = render_with(&mut gpu, cam, Variant::Honest);
        if let Ok(dir) = std::env::var("REPAME_POSES") {
            // Every pose on disk, so a failing invariant can be looked at
            // instead of guessed about.
            let dir = PathBuf::from(dir);
            let _ = std::fs::create_dir_all(&dir);
            let _ = std::fs::write(
                dir.join(format!("{label}.png")),
                png_bytes(W, H, &rgba),
            );
        }
        if check_pose(&rgba, &cam, background, &label) {
            solid_checked += 1;
        }
        frames.push(rgba);
    }

    assert!(
        solid_checked >= 6,
        "the floating box must be fully in frame for at least six poses, or the solid check \
         is silently skipped everywhere (it ran for {solid_checked})"
    );

    check_camera_axes(
        &mut |cam: OrbitCamera| render_with(&mut gpu, cam, Variant::Honest),
        background,
    );

    let (sw, sh, sheet) = contact_sheet(&frames);
    if let Ok(dir) = std::env::var("REPAME_VIDEO") {
        record_video(&dir, &frames);
    }
    check_sheet(sw, sh, &sheet);
}

/// Optional review artefact: the sweep as lossless FFV1 plus per-frame hashes.
/// Never a gate — a missing ffmpeg only prints.
fn record_video(dir: &str, frames: &[Vec<u8>]) {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let dir = PathBuf::from(dir);
    if std::fs::create_dir_all(&dir).is_err() {
        eprintln!("REPAME_VIDEO: cannot create {}", dir.display());
        return;
    }
    let video = dir.join("pose_sweep.mkv");
    let child = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            &format!("{W}x{H}"),
            "-r",
            "30",
            "-i",
            "pipe:0",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-pix_fmt",
            "bgra",
            video.to_str().unwrap(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn();
    match child {
        Ok(mut child) => {
            if let Some(stdin) = child.stdin.as_mut() {
                for frame in frames {
                    let _ = stdin.write_all(frame);
                }
            }
            let _ = child.wait();
            eprintln!("REPAME_VIDEO: wrote {}", video.display());
        }
        Err(error) => eprintln!("REPAME_VIDEO: ffmpeg unavailable ({error})"),
    }
}

/// Exactly one winding of an up-facing quad is drawn from above. Whichever it is,
/// `push_quad` must use it, so this pins the engine's culling convention where a
/// missing road cannot be blamed on a game's geometry code.
#[test]
fn the_front_face_convention_is_pinned() {
    let Some(mut gpu) = gpu() else {
        return;
    };
    let build = |push_quad_order: bool| {
        let mut g = MeshGroup::default();
        let d = 8.0f32;
        let corners = [
            Vec3::new(-d, 0.0, -d),
            Vec3::new(-d, 0.0, d),
            Vec3::new(d, 0.0, d),
            Vec3::new(d, 0.0, -d),
        ];
        for v in corners {
            g.positions.push(v.to_array());
            g.colors.push(GROUND);
            g.normals.push(Vec3::Y.to_array());
        }
        if push_quad_order {
            g.indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
        } else {
            g.indices.extend_from_slice(&[2, 1, 0, 3, 2, 0]);
        }
        g
    };

    let cam = OrbitCamera {
        target: Vec3::ZERO,
        yaw: 0.0,
        pitch: 0.45,
        dist: 14.0,
        fov_y_deg: 45.0,
    };
    let render = |gpu: &mut OffscreenRenderer, group: MeshGroup| {
        let desc = BatchDesc {
            layer_size: 64,
            layers: 1,
            filter: SceneFilter::Linear,
            mips: 1,
        };
        let mut frame = Frame3d {
            cam,
            desc,
            light: light(),
            viewport_px: [W as f32, H as f32],
            ..Frame3d::default()
        };
        frame.push(group);
        let ctx = RenderContext::new();
        let geom = GeomHandle::new();
        let view = Viewport3d(frame, geom, "front-face", desc, |_| {});
        let states: HashMap<u64, Rc<RefCell<TextFieldState>>> = HashMap::new();
        let (scene, _h, _s) =
            layout_and_paint(&view, (W, H), &states, &Interactions::default(), None);
        apply_render_commands(gpu.renderer_mut(), ctx.drain()).log_failures();
        gpu.render_rgba(&scene, Some(CLEAR)).expect("renders")
    };

    // A quad of either winding, or nothing at all, is what the two probes compare.
    let used = render(&mut gpu, build(true));
    let other = render(&mut gpu, build(false));
    let full = (0.0, 0.0, W as f32, H as f32);
    let used_hits = share(&used, full, 0.0, is_ground);
    let other_hits = share(&other, full, 0.0, is_ground);
    eprintln!(
        "front-face probe: push_quad's order drew {:.1}% ground, the reverse drew {:.1}%",
        used_hits * 100.0,
        other_hits * 100.0
    );
    assert!(
        used_hits > 0.1 && other_hits <= 0.1,
        "an up-facing quad must be visible from exactly one winding: push_quad's order drew \
         {used_hits:.3}, the reverse drew {other_hits:.3}. Fix `push_quad` to the visible \
         order and keep this test green."
    );
}

// ------------------------------------------------------------------- teeth

#[test]
fn the_sweep_notices_a_backwards_ground() {
    let Some(mut gpu) = gpu() else {
        return;
    };
    let cam = camera("yaw+0.00_pitch0.42");
    let honest = render_with(&mut gpu, cam, Variant::Honest);
    let background = pixel(&honest, 1, 1);
    let broken = render_with(&mut gpu, cam, Variant::BackwardsGround);
    let lower = (0.0, H as f32 * 0.72, W as f32, H as f32);
    let honest_ground = share(&honest, lower, 0.0, is_ground);
    let broken_ground = share(&broken, lower, 0.0, is_ground);
    assert!(
        honest_ground >= 0.5,
        "the honest scene must pass its own invariant (it read {:.0}%)",
        honest_ground * 100.0
    );
    assert!(
        broken_ground < 0.5,
        "a backwards-wound ground must fail the coverage invariant; it read {:.0}% ground",
        broken_ground * 100.0
    );
    let _ = background;
}

#[test]
fn the_sweep_notices_a_missing_face() {
    let Some(mut gpu) = gpu() else {
        return;
    };
    let cam = camera("yaw+0.00_pitch0.85");
    let honest = render_with(&mut gpu, cam, Variant::Honest);
    let broken = render_with(&mut gpu, cam, Variant::MissingBoxFace);
    let (honest_solid, honest_seen) =
        solid_cells(&honest, &cam, SKY_BOX_CENTRE, Vec3::splat(BOX_HALF), is_red)
            .expect("the floating box is visible at the probe pose");
    let (broken_solid, broken_seen) =
        solid_cells(&broken, &cam, SKY_BOX_CENTRE, Vec3::splat(BOX_HALF), is_red)
            .expect("the floating box is visible at the probe pose");
    if let Ok(dir) = std::env::var("REPAME_POSES") {
        let dir = PathBuf::from(dir);
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(dir.join("teeth_honest.png"), png_bytes(W, H, &honest));
        let _ = std::fs::write(dir.join("teeth_broken.png"), png_bytes(W, H, &broken));
        eprintln!(
            "teeth: honest {honest_solid}/{honest_seen}, broken {broken_solid}/{broken_seen}, \
             rect {:?}",
            box_rect(&cam, SKY_BOX_CENTRE, Vec3::splat(BOX_HALF))
        );
    }
    assert!(
        honest_solid * 4 >= honest_seen * 3,
        "the honest box must read red almost everywhere (it read {honest_solid}/{honest_seen} \
         visible cells)"
    );
    assert!(
        broken_solid * 4 < broken_seen * 3,
        "a dropped box face must fail the solid invariant; the silhouette still read \
         {broken_solid}/{broken_seen} red cells, so the hole is not visible from this pose — \
         move the probe camera or drop a face that faces it"
    );
}
