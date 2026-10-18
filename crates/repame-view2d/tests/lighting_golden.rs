//! Offscreen golden-image harness for the 2D lighting path: the light pass
//! (`shaders/light_disc.wgsl`, `shaders/light_mask.wgsl`) and the sprite
//! shaders that consume the light map (`shaders/sprite.wgsl`).
//!
//! Every other check in this crate is CPU-side (`light.rs` plans, wedge
//! geometry) or by construction (`lightpass.rs` pipeline state). This file is
//! the only place the lighting WGSL actually executes, and it runs through the
//! real game route rather than a re-implementation of it:
//!
//! * one [`Light2dPass`] and one [`BatchFrame`] are mounted as `repose`
//!   callbacks (`SceneNode::Callback`) in a single [`Scene`];
//! * the scene is rendered by
//!   [`OffscreenRenderer`](repose_render_wgpu::offscreen::OffscreenRenderer),
//!   which runs every callback's `prepare` (the light pass records its light
//!   map render passes there) and `finish_prepare` (where the batch resolves
//!   the published light map and builds its bind group) before the main pass
//!   paints the sprites;
//! * so what is asserted covers the whole chain: plan -> accumulation blend ->
//!   mask pass -> sprite sampling -> sRGB target.
//!
//! The offscreen callback route is the one the crate's other GPU tests use
//! (`src/batch.rs`'s `offscreen_batch_draws_oriented_textured_quads` mounts a
//! batch this way), and `repose-render-wgpu`'s own `tests/offscreen_contracts.rs`
//! locks the readback contract this harness relies on. The only thing given up
//! against a windowed run is the surface: swapchain formats, the real
//! `pixels_per_point`, and present timing are out of scope here.
//!
//! Scenes are built in code so what is under test is reviewable: world units
//! map 1:1 onto pixels (`screen_camera`), the sprite is a full-target white
//! quad, and the batch runs with `set_light_modulate([0.0; 3])` so the item's
//! own colour contributes nothing and every probe byte measures *light*, not
//! art. That makes several probes exact: with a 1.0 light-map scale the light
//! map is sampled at texel centres, so a probe's value is the ramp (or the
//! falloff texel) the shader evaluated at a computable distance.
//!
//! Regenerate the goldens with:
//! `REPAME_BLESS=1 cargo test -p repame-view2d --test lighting_golden`
//! Without the variable a missing golden is written and the test fails once,
//! so a new scene cannot silently pass. Without a WGPU adapter every test
//! skips with a warning instead of failing.
//!
//! Comparison is per-channel tolerance plus a diff-fraction allowance (AA and
//! filtering variance), never exact equality. The structural assertions on top
//! (lit brighter than unlit, shadowed equals ambient, penumbra wider than the
//! hard edge, an item untouched by a light whose mask it does not share) are
//! the ones that catch a plausible-looking broken pass.

use std::path::PathBuf;
use std::sync::Arc;

use glam::Mat4;
use glam::Vec2;
use repame_view2d::{
    AmbientLight, AtlasUpload, BatchDesc, BatchFrame, Light2d, Light2dDesc, Light2dPass,
    LightFalloff, LightStop, LightTextureUpload, Occluder2d, ShadowFilter, SpriteBatch,
    TextureFilter, screen_camera, stops_at,
};
use repose_core::{Color, Rect, Scene};
use repose_render_wgpu::offscreen::OffscreenRenderer;

/// Offscreen target edge in pixels. World units map 1:1 onto these pixels.
const SIZE: u32 = 256;
const SIZE_F: f32 = 256.0;
/// Per-channel deviation below this is "same pixel".
const CHANNEL_TOL: i16 = 8;
/// Fraction of pixels allowed to exceed `CHANNEL_TOL` (AA/filter variance).
const MAX_DIFF_FRACTION: f64 = 0.005;
/// Per-channel tolerance for probes whose value is pinned in closed form.
const PROBE_TOL: i16 = 3;

/// Atlas page 0 (the batch-path baseline): R G / B W, row-major, top first.
const PAGE_QUADRANTS: [u8; 16] = [
    255, 0, 0, 255, //
    0, 255, 0, 255, //
    0, 0, 255, 255, //
    255, 255, 255, 255,
];
/// Atlas page 1: solid white, so a lit probe measures only the light map.
const PAGE_WHITE: [u8; 16] = [255; 16];

/// The scene clear: opaque black, so anything the batch does not draw (and
/// anything a light does not reach) reads back as zero.
const BLACK_OPAQUE: Color = Color(0, 0, 0, 255);

fn goldens_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/goldens")
}

fn png_bytes(rgba: &[u8]) -> Vec<u8> {
    let image = image::RgbaImage::from_raw(SIZE, SIZE, rgba.to_vec()).expect("rgba size");
    let mut cursor = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(image)
        .write_to(&mut cursor, image::ImageFormat::Png)
        .expect("encode png");
    cursor.into_inner()
}

/// One offscreen renderer, or a warning: no adapter means the test skips
/// rather than passing by accident on a machine that cannot run the shaders.
fn gpu() -> Option<OffscreenRenderer> {
    match OffscreenRenderer::new_blocking(SIZE, SIZE, 1) {
        Ok(gpu) => Some(gpu),
        Err(error) => {
            eprintln!("skipping lighting golden test: no WGPU adapter ({error:#})");
            None
        }
    }
}

/// Compare against the committed golden: per-channel tolerance plus a
/// diff-fraction allowance, with `REPAME_BLESS=1` to rewrite it.
fn check_golden(name: &str, rgba: &[u8]) {
    let path = goldens_dir().join(format!("{name}.png"));
    let bless = std::env::var("REPAME_BLESS").is_ok();
    if bless || !path.exists() {
        std::fs::create_dir_all(goldens_dir()).expect("create goldens dir");
        std::fs::write(&path, png_bytes(rgba)).expect("write golden");
        if !bless {
            panic!(
                "{name}: golden did not exist; wrote an initial one. Re-run to verify, \
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
        (SIZE, SIZE),
        "{name}: golden size changed"
    );
    let total = (SIZE * SIZE) as f64;
    let mut differing = 0usize;
    for (g, a) in golden.pixels().zip(rgba.chunks_exact(4)) {
        let over_tol = g
            .0
            .iter()
            .zip(a.iter())
            .any(|(&gc, &ac)| (gc as i16 - ac as i16).abs() > CHANNEL_TOL);
        if over_tol {
            differing += 1;
        }
    }
    let fraction = differing as f64 / total;
    if fraction > MAX_DIFF_FRACTION {
        let actual_path = goldens_dir().join(format!("{name}.actual.png"));
        std::fs::write(&actual_path, png_bytes(rgba)).expect("write actual");
        panic!(
            "{name}: {differing} px ({:.3}%) differ beyond tolerance (limit {:.3}%). \
             Actual written to {}. Bless with REPAME_BLESS=1 if intentional.",
            fraction * 100.0,
            MAX_DIFF_FRACTION * 100.0,
            actual_path.display(),
        );
    }
}

fn render(gpu: &mut OffscreenRenderer, scene: &Scene) -> Vec<u8> {
    gpu.render_rgba(scene, Some([0.0, 0.0, 0.0, 1.0]))
        .expect("offscreen render")
}

fn pixel(rgba: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * SIZE + x) * 4) as usize;
    [rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]]
}

fn rgb(rgba: &[u8], x: u32, y: u32) -> [u8; 3] {
    let p = pixel(rgba, x, y);
    [p[0], p[1], p[2]]
}

/// sRGB encode of a linear channel, as the `Rgba8UnormSrgb` target applies it.
fn srgb_encode(linear: f32) -> u8 {
    let l = linear.clamp(0.0, 1.0);
    let s = if l <= 0.003_130_8 {
        12.92 * l
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as u8
}

/// sRGB decode, as the target's bytes read back.
fn srgb_decode(byte: u8) -> f32 {
    let s = byte as f32 / 255.0;
    if s <= 0.040_45 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

/// The byte a light-map channel `linear` produces: the pass stores it in an
/// 8-bit linear `Rgba8Unorm` layer (quantised), the sprite writes it into the
/// sRGB target (encoded). Both transfers are reproduced here.
fn lit_byte(linear: f32) -> u8 {
    let stored = (linear.clamp(0.0, 1.0) * 255.0).round() / 255.0;
    srgb_encode(stored)
}

fn assert_probe(rgba: &[u8], x: u32, y: u32, want: u8, tol: i16, what: &str) {
    let got = rgb(rgba, x, y);
    for (channel, value) in got.iter().enumerate() {
        assert!(
            (*value as i16 - want as i16).abs() <= tol,
            "{what}: pixel ({x}, {y}) channel {channel} is {value}, want {want} +- {tol} \
             (full pixel {got:?})"
        );
    }
}

// ---------------------------------------------------------------------------
// Scene building
// ---------------------------------------------------------------------------

fn camera() -> Mat4 {
    screen_camera([SIZE_F, SIZE_F])
}

fn full_rect() -> Rect {
    Rect {
        x: 0.0,
        y: 0.0,
        w: SIZE_F,
        h: SIZE_F,
    }
}

/// Falloff textures are level data, so the generation changes only when the
/// uploaded set does.
fn desc(textures_gen: u64) -> Light2dDesc {
    Light2dDesc {
        // The light map is sampled 1 texel per target pixel, so probes read
        // exactly the value the light shader evaluated.
        lightmap_scale: 1.0,
        // 2048 over these lights' footprints is well under a third of a world
        // unit per mask texel, so a hard shadow edge stays hard and the
        // penumbra width is the filter's, not the mask's.
        mask_size: 2048,
        filter: TextureFilter::Linear,
        textures_gen,
        ..Default::default()
    }
}

fn atlas() -> Arc<[AtlasUpload]> {
    Arc::from(vec![
        AtlasUpload {
            page: 0,
            x: 0,
            y: 0,
            w: 2,
            h: 2,
            rgba: PAGE_QUADRANTS.to_vec(),
        },
        AtlasUpload {
            page: 1,
            x: 0,
            y: 0,
            w: 2,
            h: 2,
            rgba: PAGE_WHITE.to_vec(),
        },
    ])
}

fn batch_desc() -> BatchDesc {
    BatchDesc {
        layer_size: 2,
        layers: 2,
        filter: TextureFilter::Nearest,
        uploads_gen: 0,
    }
}

/// A batch drawing one full-target quad of `page`, lit on `light_mask`.
fn full_quad_batch(id: &str, page: u32, light_mask: u32) -> SpriteBatch {
    let mut batch = SpriteBatch::with_id(id, batch_desc());
    batch.set_camera(camera());
    batch.push_masked(
        [SIZE_F / 2.0, SIZE_F / 2.0],
        [SIZE_F, SIZE_F],
        0.0,
        [0.5, 0.5],
        false,
        false,
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0; 4],
        page,
        light_mask,
    );
    batch
}

fn lit_scene(pass: Light2dPass, batch: SpriteBatch) -> Scene {
    Scene {
        clear_color: BLACK_OPAQUE,
        nodes: vec![
            pass.node(full_rect()),
            BatchFrame::new(batch, atlas()).node(full_rect()),
        ],
    }
}

// ---------------------------------------------------------------------------
// Scene 1: unlit sprite - the batch path baseline
// ---------------------------------------------------------------------------

/// No light pass at all: the batch falls back to the identity light map, so
/// this pins the plain instanced quad path every lit scene is built on.
/// The quad's uv rect is inset to texel centres, so the four atlas texels land
/// on the four image quadrants exactly.
#[test]
fn golden_unlit_sprite() {
    let Some(mut gpu) = gpu() else { return };
    let scene = Scene {
        clear_color: BLACK_OPAQUE,
        nodes: vec![
            BatchFrame::new(full_quad_batch("test.lighting.unlit", 0, 1), atlas())
                .node(full_rect()),
        ],
    };
    let rgba = render(&mut gpu, &scene);

    assert_eq!(rgb(&rgba, 64, 64), [255, 0, 0], "top-left red texel");
    assert_eq!(rgb(&rgba, 192, 64), [0, 255, 0], "top-right green texel");
    assert_eq!(rgb(&rgba, 64, 192), [0, 0, 255], "bottom-left blue texel");
    assert_eq!(
        rgb(&rgba, 192, 192),
        [255, 255, 255],
        "bottom-right white texel"
    );

    check_golden("unlit_sprite", &rgba);
}

// ---------------------------------------------------------------------------
// Scene 2: a point light at a known distance
// ---------------------------------------------------------------------------

/// Stops falloff at computable distances. The light sits on a pixel corner so
/// probes on its row are exactly 16 / 32 / 64 world units away, and the ramp
/// has a stop at `t = 0.25`, so `stops_at` (the CPU reference the shader
/// copies) pins the expected byte rather than a hand-written number.
#[test]
fn golden_point_light_stops() {
    let Some(mut gpu) = gpu() else { return };
    const CENTER: Vec2 = Vec2::new(128.5, 128.5);
    const RADIUS: f32 = 128.0;
    let stops = vec![
        LightStop {
            at: 0.0,
            color: [1.0, 1.0, 1.0, 1.0],
        },
        LightStop {
            at: 0.25,
            color: [0.5, 0.5, 0.5, 1.0],
        },
        LightStop {
            at: 1.0,
            color: [0.0, 0.0, 0.0, 1.0],
        },
    ];
    let frame = |gpu: &mut OffscreenRenderer, light_mask: u32| {
        let mut pass = Light2dPass::with_id("test.lighting.stops", desc(0));
        pass.set_camera(camera());
        pass.push_light(Light2d {
            position: CENTER,
            radius: RADIUS,
            color: [1.0, 1.0, 1.0],
            energy: 1.0,
            falloff: LightFalloff::stops(stops.clone()),
            light_mask,
            ..Light2d::default()
        });
        let mut batch = full_quad_batch("test.lighting.stops.batch", 1, 1);
        batch.set_light_source("test.lighting.stops");
        batch.set_light_modulate([0.0; 3]);
        render(gpu, &lit_scene(pass, batch))
    };
    let rgba = frame(&mut gpu, 1);

    let probes = [
        ((144u32, 128u32), 16.0f32),
        ((160, 128), 32.0),
        ((192, 128), 64.0),
        // Same distance on the other side: the ramp is radial, not signed.
        ((112, 128), 16.0),
    ];
    for (probe, distance) in probes {
        let ramp = stops_at(&stops, distance / RADIUS);
        assert_probe(
            &rgba,
            probe.0,
            probe.1,
            lit_byte(ramp[0]),
            PROBE_TOL,
            &format!("stopped ramp at {distance} world units"),
        );
    }

    // Past the rim the disc discards its fragments and the map keeps the
    // ambient (black here): nothing lights the sprite outside the radius.
    assert_probe(&rgba, 8, 8, 0, 1, "outside the light's radius");

    // Falloff is actually sampled: brightness must fall with distance.
    let (near, mid, far) = (
        rgb(&rgba, 144, 128)[0],
        rgb(&rgba, 160, 128)[0],
        rgb(&rgba, 192, 128)[0],
    );
    assert!(
        near > mid && mid > far,
        "ramp is not monotone with distance: {near} > {mid} > {far}"
    );
    assert!(
        near >= far + 20,
        "ramp barely fell over 48 world units: {near} vs {far}"
    );

    // Control: the same frame with the light's mask zeroed renders black at
    // every probe, so the bytes above measure the light, not the item.
    let unlit = frame(&mut gpu, 0);
    for (probe, _) in probes {
        assert_probe(
            &unlit,
            probe.0,
            probe.1,
            0,
            1,
            "masked-out light leaves the sprite unlit",
        );
    }

    check_golden("point_light_stops", &rgba);
}

/// Texture falloff: a 2x2 ramp of sRGB fixed points (white and black, so the
/// sampled linear values are exact) laid out W B / B W, which pins the uv
/// mapping in both axes; a u/v flip or a transpose lands a different texel.
#[test]
fn golden_point_light_texture() {
    let Some(mut gpu) = gpu() else { return };
    const CENTER: Vec2 = Vec2::new(128.5, 128.5);
    const HALF: Vec2 = Vec2::new(64.0, 64.0);
    /// Non-zero, so the texture's footprint is offset from the light quad.
    const OFFSET: Vec2 = Vec2::new(16.0, 0.0);

    let mut pass = Light2dPass::with_id("test.lighting.texture", desc(1));
    pass.set_camera(camera());
    pass.queue_texture(LightTextureUpload {
        slot: 0,
        w: 2,
        h: 2,
        rgba: vec![
            255, 255, 255, 255, //
            0, 0, 0, 255, //
            0, 0, 0, 255, //
            255, 255, 255, 255,
        ],
    });
    pass.push_light(Light2d {
        position: CENTER,
        radius: HALF.x,
        color: [1.0, 1.0, 1.0],
        energy: 1.0,
        falloff: LightFalloff::Texture {
            slot: 0,
            half_extent: HALF,
            offset: OFFSET,
        },
        ..Light2d::default()
    });
    let mut batch = full_quad_batch("test.lighting.texture.batch", 1, 1);
    batch.set_light_source("test.lighting.texture");
    batch.set_light_modulate([0.0; 3]);
    let rgba = render(&mut gpu, &lit_scene(pass, batch));

    // The inverse of `Light2d::texture_uv`: the world point a given ramp uv
    // lands on. Pixel centres are at half-integer world coordinates.
    let world_of = |u: f32, v: f32| {
        CENTER + OFFSET + (Vec2::new(u, v) - Vec2::splat(0.5)) * HALF * 2.0
    };
    // Texel centres of the 2x2 ramp: W, B on the top row, B, W on the bottom.
    for (u, v, want) in [
        (0.25, 0.25, 255),
        (0.75, 0.25, 0),
        (0.25, 0.75, 0),
        (0.75, 0.75, 255),
    ] {
        let world = world_of(u, v);
        assert_probe(
            &rgba,
            (world.x - 0.5) as u32,
            (world.y - 0.5) as u32,
            want,
            2,
            &format!("falloff texel at uv ({u}, {v})"),
        );
    }

    // Inside the light quad but outside the texture's footprint (the offset
    // moved it right): the shader discards, so clamping instead of discarding
    // would paint the edge texel here.
    assert_probe(
        &rgba,
        72,
        128,
        0,
        1,
        "quad area left of the offset falloff texture",
    );
    // Outside the light quad entirely: untouched by this light.
    assert_probe(&rgba, 200, 40, 0, 1, "outside the light quad");

    check_golden("point_light_texture", &rgba);
}

// ---------------------------------------------------------------------------
// Scene 3: two lights with disjoint light masks
// ---------------------------------------------------------------------------

/// Two lights 80 units apart, both covering both sprites, each on its own
/// mask bit. A sprite on mask bit 0 must see only the left light: the two
/// sprites sit inside *both* footprints, so a pass that ignores masks (or
/// resolves them to one layer) still renders something plausible, and only the
/// cross-scene equality with a single-light render catches it.
#[test]
fn golden_two_lights_disjoint_masks() {
    let Some(mut gpu) = gpu() else { return };
    const LEFT: Vec2 = Vec2::new(88.5, 128.5);
    const RIGHT: Vec2 = Vec2::new(168.5, 128.5);
    const RADIUS: f32 = 140.0;
    const ENERGY: f32 = 0.4;
    // Probes: each sprite's centre (on top of its light) and its edge facing
    // the other light (28 units out, 52 units from the other light).
    const A_CENTER: (u32, u32) = (88, 128);
    const B_CENTER: (u32, u32) = (168, 128);
    const A_PROBE: (u32, u32) = (116, 128);
    const B_PROBE: (u32, u32) = (140, 128);

    let light = |position: Vec2, mask: u32| Light2d {
        position,
        radius: RADIUS,
        color: [1.0, 1.0, 1.0],
        energy: ENERGY,
        falloff: LightFalloff::stops(vec![
            LightStop {
                at: 0.0,
                color: [1.0, 1.0, 1.0, 1.0],
            },
            LightStop {
                at: 1.0,
                color: [0.0, 0.0, 0.0, 1.0],
            },
        ]),
        light_mask: mask,
        ..Light2d::default()
    };

    let frame =
        |gpu: &mut OffscreenRenderer, lights: Vec<Light2d>, sprites: [(Vec2, u32); 2]| {
            let mut pass = Light2dPass::with_id("test.lighting.masks", desc(0));
            pass.set_camera(camera());
            pass.extend_lights(lights);
            let mut batch = SpriteBatch::with_id("test.lighting.masks.batch", batch_desc());
            batch.set_camera(camera());
            batch.set_light_source("test.lighting.masks");
            batch.set_light_modulate([0.0; 3]);
            for (center, mask) in sprites {
                batch.push_masked(
                    [center.x, center.y],
                    [64.0, 64.0],
                    0.0,
                    [0.5, 0.5],
                    false,
                    false,
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [1.0; 4],
                    1,
                    mask,
                );
            }
            render(gpu, &lit_scene(pass, batch))
        };

    let left_only = (LEFT, 0b01);
    let right_only = (RIGHT, 0b10);
    let two = frame(
        &mut gpu,
        vec![light(LEFT, 0b01), light(RIGHT, 0b10)],
        [(left_only.0, 0b01), (right_only.0, 0b10)],
    );
    // Single-light controls: both sprites carry the mask of the light present,
    // so each probe measures one light at a known distance.
    let only_left = frame(
        &mut gpu,
        vec![light(LEFT, 0b01)],
        [(left_only.0, 0b01), (right_only.0, 0b01)],
    );
    let only_right = frame(
        &mut gpu,
        vec![light(RIGHT, 0b10)],
        [(left_only.0, 0b10), (right_only.0, 0b10)],
    );
    let both = frame(
        &mut gpu,
        vec![light(LEFT, 0b01), light(RIGHT, 0b01)],
        [(left_only.0, 0b01), (right_only.0, 0b01)],
    );
    // Crossed: the same two lights, swapped sprite masks. The left sprite now
    // reads the right light's layer, and vice versa.
    let crossed = frame(
        &mut gpu,
        vec![light(LEFT, 0b01), light(RIGHT, 0b10)],
        [(left_only.0, 0b10), (right_only.0, 0b01)],
    );

    // Closed form: `energy * ramp(distance / radius)`, ramp = 1 - t.
    let ramp = |distance: f32| ENERGY * (1.0 - distance / RADIUS);

    // The assertion this scene exists for: an item is untouched by a light
    // whose mask it does not share. Each sprite sits inside both footprints.
    assert_eq!(
        rgb(&two, A_PROBE.0, A_PROBE.1),
        rgb(&only_left, A_PROBE.0, A_PROBE.1),
        "left sprite (mask 0b01) changed when the 0b10 light appeared"
    );
    assert_eq!(
        rgb(&two, B_PROBE.0, B_PROBE.1),
        rgb(&only_right, B_PROBE.0, B_PROBE.1),
        "right sprite (mask 0b10) changed when the 0b01 light appeared"
    );
    assert_eq!(
        rgb(&two, A_CENTER.0, A_CENTER.1),
        rgb(&two, B_CENTER.0, B_CENTER.1),
        "symmetric lights lit their sprites differently"
    );

    // The value the shared bit selects is the light's own: 28 units out on the
    // left light, and 52 units out on the right light.
    assert_probe(
        &only_left,
        A_PROBE.0,
        A_PROBE.1,
        lit_byte(ramp(28.0)),
        PROBE_TOL,
        "left light alone at the left sprite's edge",
    );
    assert_probe(
        &only_right,
        A_PROBE.0,
        A_PROBE.1,
        lit_byte(ramp(52.0)),
        PROBE_TOL,
        "right light alone at the left sprite's edge",
    );
    // The two lights genuinely differ at that probe (a mask bug that always
    // answered with the same light would make the equality checks vacuous).
    assert!(
        rgb(&only_left, A_PROBE.0, A_PROBE.1)[0]
            >= rgb(&only_right, A_PROBE.0, A_PROBE.1)[0] + 10,
        "the probes do not distinguish the two lights"
    );

    // The masks are honored in both directions: a sprite on the other mask bit
    // reads the other light, so this is not "everything is unlit".
    assert_eq!(
        rgb(&crossed, A_PROBE.0, A_PROBE.1),
        rgb(&only_right, A_PROBE.0, A_PROBE.1),
        "left sprite on mask 0b10 does not read the right light"
    );
    assert_eq!(
        rgb(&crossed, B_PROBE.0, B_PROBE.1),
        rgb(&only_left, B_PROBE.0, B_PROBE.1),
        "right sprite on mask 0b01 does not read the left light"
    );

    // Positive control for the comparison above: with both lights on the
    // sprite's bit, the same probe is much brighter, so a pass that dropped
    // the second light for an unrelated reason would be caught too.
    let bright = rgb(&both, A_PROBE.0, A_PROBE.1)[0];
    let masked = rgb(&two, A_PROBE.0, A_PROBE.1)[0];
    assert!(
        bright >= masked + 20,
        "sharing a bit with both lights did not brighten the probe: {bright} vs {masked}"
    );
    assert_probe(
        &both,
        A_PROBE.0,
        A_PROBE.1,
        lit_byte(ramp(28.0) + ramp(52.0)),
        PROBE_TOL,
        "both lights summed on the shared bit",
    );

    check_golden("two_lights_disjoint_masks", &two);
}

// ---------------------------------------------------------------------------
// Scene 4: a shadow edge at a fixed angle, and Pcf4 against None
// ---------------------------------------------------------------------------

/// The occluder: a thin wall (an open 2-point polyline) whose two corners cast
/// the umbra. At the probe row the shadow interval is exactly
/// `[x_light + (corner.x - x_light) * t, ...]` per corner, which is what the
/// edge assertions below solve for. `closed: false` keeps this to one edge, so
/// the penumbra's only source is that edge: every edge of a closed polygon
/// extrudes its own rotated wedges under `Pcf4`, and a body's interior edges
/// then widen the shadow outward past the silhouette, which is checked (for
/// the hard filter, where the union is exact) in
/// `shadow_hard_edge_of_a_closed_polygon`.
fn wall(closed: bool) -> Occluder2d {
    let mut occluder = Occluder2d {
        points: vec![Vec2::new(96.0, 96.0), Vec2::new(160.0, 96.0)],
        closed,
        light_mask: 1,
    };
    if closed {
        occluder.points = vec![
            Vec2::new(96.0, 96.0),
            Vec2::new(160.0, 96.0),
            Vec2::new(160.0, 108.0),
            Vec2::new(96.0, 108.0),
        ];
    }
    occluder
}

/// Linear red channel of one pixel row.
fn row_linear(rgba: &[u8], y: u32) -> Vec<f32> {
    (0..SIZE)
        .map(|x| srgb_decode(rgb(rgba, x, y)[0]))
        .collect()
}

/// Which way a shadow edge turns along the row.
#[derive(Clone, Copy)]
enum Edge {
    /// Lit on the left, shadowed on the right.
    LitThenShadow,
    /// Shadowed on the left, lit on the right.
    ShadowThenLit,
}

/// Edge position in pixel-index space inside `window`, whose value falls from
/// the lit level to the shadow level (`LitThenShadow`) or rises back.
///
/// `Pcf4`'s stepped penumbra has a plateau exactly at the mid level and the
/// plateau's centre is the edge. A hard edge has no plateau, so the estimate
/// falls back to the midpoint between the last lit and the first shadowed
/// pixel; the outermost partial pixel of a soft penumbra sits well outside the
/// mid band either way, so neither branch is biased by it.
fn edge_index(row: &[f32], lit: f32, shadow: f32, window: (usize, usize), edge: Edge) -> f32 {
    let mid = 0.5 * (lit + shadow);
    let band = 0.125 * (lit - shadow);
    let (lo, hi) = (window.0, window.1.min(SIZE as usize - 1));
    let in_band = |i: usize| (row[i] - mid).abs() <= band;
    if let (Some(first), Some(last)) = (
        (lo..=hi).find(|i| in_band(*i)),
        (lo..=hi).rev().find(|i| in_band(*i)),
    ) && last > first
    {
        return (first as f32 + last as f32) * 0.5;
    }
    let (below, above) = match edge {
        Edge::LitThenShadow => (
            (lo..=hi)
                .find(|i| row[*i] < mid - band)
                .expect("no pixel reached the shadow level in this window"),
            (lo..=hi)
                .rev()
                .find(|i| row[*i] > mid + band)
                .expect("no pixel kept the lit level in this window"),
        ),
        Edge::ShadowThenLit => (
            (lo..=hi)
                .rev()
                .find(|i| row[*i] < mid - band)
                .expect("no pixel reached the shadow level in this window"),
            (lo..=hi)
                .find(|i| row[*i] > mid + band)
                .expect("no pixel kept the lit level in this window"),
        ),
    };
    (below as f32 + above as f32) * 0.5
}

/// Pixels inside the transition band (10%..90% of the lit-to-shadow range) in
/// `window`, i.e. the penumbra width.
fn penumbra_width(row: &[f32], lit: f32, shadow: f32, window: (usize, usize)) -> usize {
    let (lo, hi) = window;
    let low = shadow + 0.1 * (lit - shadow);
    let high = lit - 0.1 * (lit - shadow);
    let hits: Vec<usize> = (lo..=hi)
        .filter(|i| row[*i] > low && row[*i] < high)
        .collect();
    match (hits.first(), hits.last()) {
        (Some(first), Some(last)) => last - first + 1,
        _ => 0,
    }
}

/// Distinct 8-bitlevels strictly inside the transition band: a stepped
/// `Pcf4` penumbra has several, a hard edge at most the AA pixel.
fn transition_levels(row: &[f32], lit: f32, shadow: f32) -> usize {
    let low = shadow + 0.1 * (lit - shadow);
    let high = lit - 0.1 * (lit - shadow);
    let mut levels: Vec<u8> = row
        .iter()
        .filter(|v| **v > low && **v < high)
        .map(|v| (v * 255.0).round() as u8)
        .collect();
    levels.sort_unstable();
    levels.dedup();
    levels.len()
}

/// One shadow render of the wall scene: ambient 0.25 everywhere, a flat white
/// light of `energy` above the wall, and the sprite modulated to black, so
/// every byte on the probe row is the light map. `energy == 0.0` is the
/// ambient-only control.
///
/// The light sits at y 52, not on the wall's own y-span, so the occluder's
/// bounds overlap the light's with margin instead of sitting on a boundary
/// (see `aabb_overlap` in `light.rs`).
fn shadow_frame(
    gpu: &mut OffscreenRenderer,
    pass_id: &str,
    filter: ShadowFilter,
    closed: bool,
    energy: f32,
) -> Vec<u8> {
    let mut pass = Light2dPass::with_id(pass_id, desc(0));
    pass.set_camera(camera());
    pass.set_ambient(AmbientLight {
        color: [1.0, 1.0, 1.0],
        energy: 0.25,
        light_mask: 1,
    });
    pass.push_light(Light2d {
        position: SHADOW_LIGHT,
        radius: 288.0,
        color: [1.0, 1.0, 1.0],
        energy,
        falloff: LightFalloff::stops(vec![LightStop {
            at: 0.0,
            color: [1.0, 1.0, 1.0, 1.0],
        }]),
        shadows: true,
        shadow_filter: filter,
        shadow_softness: 8.0,
        light_mask: 1,
        ..Light2d::default()
    });
    pass.push_occluder(wall(closed));
    let mut batch = full_quad_batch("test.lighting.shadow.batch", 1, 1);
    batch.set_light_source(pass_id);
    batch.set_light_modulate([0.0; 3]);
    render(gpu, &lit_scene(pass, batch))
}

/// The shadow scene's probe row, its light, and the analytic edge positions.
const SHADOW_ROW: u32 = 176;
const SHADOW_ROW_CENTER: f32 = SHADOW_ROW as f32 + 0.5;
/// The shadow scene's light: above the wall, footprint covering the target.
const SHADOW_LIGHT: Vec2 = Vec2::new(128.0, 52.0);
/// Pixel windows holding each shadow edge and nothing else: the umbra covers
/// roughly columns 37..219 at the probe row, so each window sees one crossing.
const EDGE_WINDOWS: [(usize, usize); 2] = [(0, 90), (170, SIZE as usize - 1)];

/// Where the ray from `light` through `corner` crosses the probe row.
fn silhouette_x(light: Vec2, corner: Vec2, row_center: f32) -> f32 {
    light.x + (corner.x - light.x) * (row_center - light.y) / (corner.y - light.y)
}

/// The shadow edge is where the geometry says it is, and `Pcf4` widens the
/// transition around that same edge without moving it.
#[test]
fn golden_shadow_edge_and_penumbra() {
    let Some(mut gpu) = gpu() else { return };
    const ENERGY: f32 = 0.5;
    const AMBIENT: f32 = 0.25;
    // The wall's corners: the near face of the closed variant is this edge.
    const CORNERS: [Vec2; 2] = [Vec2::new(96.0, 96.0), Vec2::new(160.0, 96.0)];

    let ambient_byte = lit_byte(AMBIENT);
    let lit_value = AMBIENT + ENERGY;
    let lit = srgb_decode(lit_byte(lit_value));
    let shadow = srgb_decode(ambient_byte);
    // Expected edge positions: solved from the light-corner rays, and the
    // pixel index whose centre sits on the ray (centres are at `x + 0.5`).
    let expect: Vec<f32> = CORNERS
        .iter()
        .map(|corner| silhouette_x(SHADOW_LIGHT, *corner, SHADOW_ROW_CENTER) - 0.5)
        .collect();

    let hard = shadow_frame(
        &mut gpu,
        "test.lighting.shadow.hard",
        ShadowFilter::None,
        false,
        ENERGY,
    );
    let soft = shadow_frame(
        &mut gpu,
        "test.lighting.shadow.soft",
        ShadowFilter::Pcf4,
        false,
        ENERGY,
    );
    // Control: no light at all beyond the ambient, so the shadowed bytes have
    // an independent reference instead of the test's own arithmetic.
    let ambient_only = shadow_frame(
        &mut gpu,
        "test.lighting.shadow.ambient",
        ShadowFilter::None,
        false,
        0.0,
    );

    for (name, rgba) in [("hard", &hard), ("pcf4", &soft)] {
        let row = row_linear(rgba, SHADOW_ROW);
        // Shadowed equals ambient: deep inside the umbra the light contributes
        // nothing, and the ambient-only control agrees pixel for pixel.
        assert_probe(
            rgba,
            128,
            SHADOW_ROW,
            ambient_byte,
            PROBE_TOL,
            &format!("{name}: deep shadow is the ambient"),
        );
        assert_eq!(
            rgb(rgba, 128, SHADOW_ROW),
            rgb(&ambient_only, 128, SHADOW_ROW),
            "{name}: deep shadow and the ambient-only control disagree"
        );
        // Lit on both sides of the umbra: outside the light's reach here is
        // only the wall's shadow, so the probes are at the full light level.
        assert_probe(
            rgba,
            6,
            SHADOW_ROW,
            lit_byte(lit_value),
            PROBE_TOL,
            &format!("{name}: lit left of the shadow"),
        );
        assert_probe(
            rgba,
            250,
            SHADOW_ROW,
            lit_byte(lit_value),
            PROBE_TOL,
            &format!("{name}: lit right of the shadow"),
        );
        let left = edge_index(&row, lit, shadow, EDGE_WINDOWS[0], Edge::LitThenShadow);
        let right = edge_index(&row, lit, shadow, EDGE_WINDOWS[1], Edge::ShadowThenLit);
        // Both filters keep the edge on the silhouette: for `None` the wedge
        // boundaries are the unrotated corner rays, for `Pcf4` the taps are
        // symmetric about them.
        assert!(
            (left - expect[0]).abs() <= 3.0,
            "{name}: left shadow edge at {left}, geometry says {}",
            expect[0]
        );
        assert!(
            (right - expect[1]).abs() <= 3.0,
            "{name}: right shadow edge at {right}, geometry says {}",
            expect[1]
        );
    }

    // The assertion the filter exists for: at identical geometry and light,
    // Pcf4's transition is far wider than None's, and stepped rather than a
    // single AA pixel.
    let hard_row = row_linear(&hard, SHADOW_ROW);
    let soft_row = row_linear(&soft, SHADOW_ROW);
    let hard_left = penumbra_width(&hard_row, lit, shadow, (0, 90));
    let soft_left = penumbra_width(&soft_row, lit, shadow, (0, 90));
    let hard_right = penumbra_width(&hard_row, lit, shadow, (170, SIZE as usize - 1));
    let soft_right = penumbra_width(&soft_row, lit, shadow, (170, SIZE as usize - 1));
    assert!(
        hard_left <= 4 && hard_right <= 4,
        "hard edges are not sharp: left {hard_left} px, right {hard_right} px"
    );
    assert!(
        soft_left >= 20 && soft_right >= 20,
        "pcf4 penumbra is too narrow to be one: left {soft_left} px, right {soft_right} px"
    );
    assert!(
        soft_left > hard_left + 15 && soft_right > hard_right + 15,
        "pcf4 did not widen the transition: hard {hard_left}/{hard_right}, soft {soft_left}/{soft_right}"
    );
    assert!(
        transition_levels(&hard_row, lit, shadow) <= 2,
        "hard edge has intermediate levels: {}",
        transition_levels(&hard_row, lit, shadow)
    );
    assert!(
        transition_levels(&soft_row, lit, shadow) >= 3,
        "pcf4 transition is not stepped: {} distinct levels",
        transition_levels(&soft_row, lit, shadow)
    );

    check_golden("shadow_hard", &hard);
    check_golden("shadow_pcf4", &soft);
}

/// The same light against a closed polygon: the umbra is still bounded by the
/// silhouette rays of the near face's corners, so both hard edges land where
/// the open wall's do, and the side and far faces' wedges must not brighten or
/// darken the interior.
#[test]
fn shadow_hard_edge_of_a_closed_polygon() {
    let Some(mut gpu) = gpu() else { return };
    const ENERGY: f32 = 0.5;
    const AMBIENT: f32 = 0.25;
    let rgba = shadow_frame(
        &mut gpu,
        "test.lighting.shadow.polygon",
        ShadowFilter::None,
        true,
        ENERGY,
    );
    let row = row_linear(&rgba, SHADOW_ROW);
    let lit = srgb_decode(lit_byte(AMBIENT + ENERGY));
    let shadow = srgb_decode(lit_byte(AMBIENT));
    let expect = [
        silhouette_x(SHADOW_LIGHT, Vec2::new(96.0, 96.0), SHADOW_ROW_CENTER) - 0.5,
        silhouette_x(SHADOW_LIGHT, Vec2::new(160.0, 96.0), SHADOW_ROW_CENTER) - 0.5,
    ];
    let left = edge_index(&row, lit, shadow, EDGE_WINDOWS[0], Edge::LitThenShadow);
    let right = edge_index(&row, lit, shadow, EDGE_WINDOWS[1], Edge::ShadowThenLit);
    assert!(
        (left - expect[0]).abs() <= 3.0,
        "polygon: left shadow edge at {left}, geometry says {}",
        expect[0]
    );
    assert!(
        (right - expect[1]).abs() <= 3.0,
        "polygon: right shadow edge at {right}, geometry says {}",
        expect[1]
    );
    assert_probe(
        &rgba,
        128,
        SHADOW_ROW,
        lit_byte(AMBIENT),
        PROBE_TOL,
        "polygon: deep shadow is the ambient",
    );
    assert_probe(
        &rgba,
        8,
        SHADOW_ROW,
        lit_byte(AMBIENT + ENERGY),
        PROBE_TOL,
        "polygon: lit left of the shadow",
    );
}

/// Pcf4 against the same closed polygon: the filter is an angular spread about
/// the light, so the penumbra must straddle the silhouette symmetrically, as
/// it does for the open wall, and full shadow may only start past the edge.
///
/// This is the regression the mask's per-tap union exists for: with every
/// wedge adding into one channel, the polygon's side and far faces piled extra
/// coverage on the lit side of the near face's silhouette, so the penumbra
/// collapsed onto the lit side and reached full shadow ~6 px *before* the
/// edge: a one-sided penumbra.
#[test]
fn shadow_pcf4_penumbra_of_a_closed_polygon() {
    let Some(mut gpu) = gpu() else { return };
    const ENERGY: f32 = 0.5;
    const AMBIENT: f32 = 0.25;
    const CORNERS: [Vec2; 2] = [Vec2::new(96.0, 96.0), Vec2::new(160.0, 96.0)];
    let rgba = shadow_frame(
        &mut gpu,
        "test.lighting.shadow.pcf4.polygon",
        ShadowFilter::Pcf4,
        true,
        ENERGY,
    );
    let row = row_linear(&rgba, SHADOW_ROW);
    let lit = srgb_decode(lit_byte(AMBIENT + ENERGY));
    let shadow = srgb_decode(lit_byte(AMBIENT));
    let expect: Vec<f32> = CORNERS
        .iter()
        .map(|corner| silhouette_x(SHADOW_LIGHT, *corner, SHADOW_ROW_CENTER) - 0.5)
        .collect();

    assert_probe(
        &rgba,
        128,
        SHADOW_ROW,
        lit_byte(AMBIENT),
        PROBE_TOL,
        "polygon pcf4: deep shadow is the ambient",
    );
    assert_probe(
        &rgba,
        6,
        SHADOW_ROW,
        lit_byte(AMBIENT + ENERGY),
        PROBE_TOL,
        "polygon pcf4: lit left of the shadow",
    );

    // Both edges keep the same silhouette as the hard and open-wall cases.
    let left = edge_index(&row, lit, shadow, EDGE_WINDOWS[0], Edge::LitThenShadow);
    let right = edge_index(&row, lit, shadow, EDGE_WINDOWS[1], Edge::ShadowThenLit);
    for (name, got, want) in [("left", left, expect[0]), ("right", right, expect[1])] {
        assert!(
            (got - want).abs() <= 3.0,
            "polygon pcf4: {name} penumbra centres on {got}, the silhouette is at {want}"
        );
    }

    // The penumbra is the filter's, so it is wide on both edges and the two
    // edges (mirror images of each other) are equally wide.
    let left_width = penumbra_width(&row, lit, shadow, EDGE_WINDOWS[0]);
    let right_width = penumbra_width(&row, lit, shadow, EDGE_WINDOWS[1]);
    assert!(
        left_width >= 20 && right_width >= 20,
        "polygon pcf4: penumbra too narrow: {left_width} px / {right_width} px"
    );
    assert!(
        left_width.abs_diff(right_width) <= 4,
        "polygon pcf4: the two edges' penumbras disagree: {left_width} px vs {right_width} px"
    );
    assert!(
        transition_levels(&row, lit, shadow) >= 3,
        "polygon pcf4: transition is not stepped: {} distinct levels",
        transition_levels(&row, lit, shadow)
    );

    // The penumbra straddles the silhouette: partial coverage on the lit side
    // and on the shadow side of it. A one-sided penumbra fails the second half.
    let mid = 0.5 * (lit + shadow);
    let band = 0.125 * (lit - shadow);
    let partial: Vec<usize> = (0..90)
        .filter(|x| (row[*x] - mid).abs() < band)
        .collect();
    let (first, last) = (
        *partial.first().expect("no partial pixels left of the silhouette"),
        *partial.last().expect("no partial pixels"),
    );
    assert!(
        (first as f32) <= expect[0] - 5.0,
        "polygon pcf4: the penumbra starts at {first}, not on the lit side of the \
         silhouette {}",
        expect[0]
    );
    assert!(
        (last as f32) >= expect[0] + 5.0,
        "polygon pcf4: the penumbra ends at {last}, before the silhouette {} \
         (full shadow on the lit side)",
        expect[0]
    );

    check_golden("shadow_pcf4_polygon", &rgba);
}

/// Every light in these scenes is inside the camera's world rect and the
/// camera is the batch's camera; if this drifts, the light map lands offset
/// from what the probes assume and the tests above fail with confusing bytes.
#[test]
fn lighting_scene_camera_maps_onto_the_target() {
    let mut pass = Light2dPass::with_id("test.lighting.camera", desc(0));
    pass.set_camera(camera());
    let rect = pass.plan().world_rect;
    let want = [0.0, 0.0, SIZE_F, SIZE_F];
    for (got, want) in rect.iter().zip(want.iter()) {
        assert!(
            (got - want).abs() < 0.01,
            "the camera no longer maps onto the target: {rect:?}"
        );
    }
}
