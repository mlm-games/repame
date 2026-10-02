//! One golden over the capabilities added against the console-runtime
//! reference: motion input, haptics, material shading, texture formats,
//! vertex layouts, render passes and frame statistics.
//!
//! Each block pins a value the reference SDK fixes, so a regression in any of
//! them surfaces here rather than as wrong pixels or dead input.

use std::f32::consts::{FRAC_PI_2, PI};

use glam::{Quat, Vec3};
use repame_input::{
    GRAVITY, HapticBus, HapticEffect, HapticSink, MotionSample, MotionTracker, SensorKind,
};
use repame_view3d::texfmt::TexError;
use repame_view3d::vertex::{expand_10_10_10_2, expand_component};
use repame_view3d::{
    FrameStats, IndexFormat, LatestFrame, Light, MaterialFog, MeshGroup, Palette, PaletteBits,
    PassError, PassKind, RenderPass, RenderTarget, RenderTargetFormat, RollingAverage, ShadedBatch,
    ShadedGroup, ShaderCache, ShadingError, ShadingModel, StatsSink, TargetSemantics, TevArg,
    TevMode, TevOp, TevOp2, TevOperand, TevStage, TexFormat, TexInput, TextureOverrides,
    VertexAttribute, VertexFormat, VertexLayout, VertexSemantic, decode_palette, evaluate_lights,
    evaluate_stages, fog_factor, mip_chain, palettize_rgba, rgba_from_texel, texel_from_rgba,
    validate_passes, wgsl_program,
};

fn approx(left: f32, right: f32, tolerance: f32) -> bool {
    (left - right).abs() <= tolerance
}

// ---------------------------------------------------------------- motion input

#[test]
fn golden_motion_orientation_is_a_quarter_turn_about_y() {
    let mut tracker = MotionTracker::new(SensorKind::GyroscopeAccelerometer);
    tracker.integrate(
        MotionSample {
            acceleration: Vec3::ZERO,
            angular_velocity: Vec3::new(0.0, FRAC_PI_2, 0.0),
        },
        1.0,
    );
    let forward = tracker.orientation() * Vec3::X;
    assert!(
        approx(forward.x, 0.0, 1e-5) && approx(forward.z, -1.0, 1e-5),
        "{forward:?}"
    );
    assert!(approx(tracker.gravity().length(), GRAVITY, 1e-4));
}

#[test]
fn golden_half_roll_puts_gravity_on_the_x_axis() {
    let mut tracker = MotionTracker::new(SensorKind::GyroscopeAccelerometer);
    tracker.integrate(
        MotionSample {
            acceleration: Vec3::ZERO,
            angular_velocity: Vec3::new(0.0, 0.0, FRAC_PI_2),
        },
        1.0,
    );
    let gravity = tracker.gravity();
    assert!(
        approx(gravity.x, GRAVITY, 1e-4) && approx(gravity.y, 0.0, 1e-4),
        "{gravity:?}"
    );
}

#[test]
fn golden_accelerometer_only_device_keeps_identity_orientation() {
    let mut tracker = MotionTracker::new(SensorKind::Accelerometer);
    tracker.integrate(
        MotionSample {
            acceleration: Vec3::new(0.0, -GRAVITY, 1.0),
            angular_velocity: Vec3::splat(9.0),
        },
        0.5,
    );
    assert_eq!(tracker.orientation(), Quat::IDENTITY);
    assert_eq!(tracker.acceleration(), Vec3::new(0.0, -GRAVITY, 1.0));
}

// ---------------------------------------------------------------- haptics

struct Sink {
    seen: Vec<(u64, f32, f32)>,
}

impl HapticSink for Sink {
    fn apply(&mut self, device: u64, strong: f32, weak: f32) {
        self.seen.push((device, strong, weak));
    }
}

#[test]
fn golden_haptic_effect_expires_exactly_on_its_duration() {
    let mut bus = HapticBus::new();
    bus.play(0, HapticEffect::for_seconds(1.0, 0.5, 0.25));
    bus.play(1, HapticEffect::constant(0.25, 0.0));
    bus.advance(0.25);
    assert_eq!(bus.motors(0), (0.0, 0.0), "expired at the boundary");
    assert_eq!(bus.motors(1), (0.25, 0.0), "constant runs until stopped");
    let mut sink = Sink { seen: Vec::new() };
    bus.apply_to(&mut sink);
    assert_eq!(sink.seen, vec![(1, 0.25, 0.0)]);
}

#[test]
fn golden_haptic_motors_clamp_into_the_unit_range() {
    let mut bus = HapticBus::new();
    bus.play(2, HapticEffect::constant(4.0, -1.0));
    bus.play(3, HapticEffect::constant(f32::NAN, f32::NAN));
    assert_eq!(bus.motors(2), (1.0, 0.0));
    assert_eq!(bus.motors(3), (0.0, 0.0));
}

// ---------------------------------------------------------------- shading

fn sample(color: [f32; 3], alpha: f32) -> Option<TexInput> {
    Some(TexInput {
        color,
        alpha,
        lod_frac: 0.0,
    })
}

#[test]
fn golden_textured_stage_multiplies_vertex_color_by_texel() {
    let model = ShadingModel::textured(0);
    model.validate().expect("model is valid");
    let out = evaluate_stages(
        &model,
        [0.5, 0.5, 0.5, 1.0],
        &[sample([1.0, 0.5, 0.25], 0.5)],
    );
    assert!(approx(out[0], 0.5, 1e-5), "{out:?}");
    assert!(approx(out[1], 0.25, 1e-5), "{out:?}");
    assert!(approx(out[2], 0.125, 1e-5), "{out:?}");
    assert!(approx(out[3], 0.5, 1e-5), "{out:?}");
}

#[test]
fn golden_second_stage_reads_the_first_stages_output() {
    let mut halve = TevStage::default();
    halve.color_arg[0] = TevOperand::rgb(TevArg::Color, 4, 0);
    halve.color_arg[1] = TevOperand::rgb(TevArg::Half, 4, 0);
    halve.color_arg[2] = TevOperand::rgb(TevArg::One, 4, 0);
    halve.color_arg[3] = TevOperand::rgb(TevArg::One, 4, 0);
    halve.color_op = [TevOp::ATimesB; 4];
    halve.color_mode = TevMode::Replace;
    halve.alpha_mode = TevMode::Replace;
    let model = ShadingModel {
        stages: vec![ShadingModel::textured(0).stages[0], halve],
        ..Default::default()
    };
    let out = evaluate_stages(
        &model,
        [0.8, 0.8, 0.8, 1.0],
        &[sample([0.5, 0.5, 0.5], 1.0)],
    );
    assert!(approx(out[0], 0.2, 1e-5), "{out:?}");
}

#[test]
fn golden_unbound_texture_unit_reads_white_and_keeps_alpha() {
    let out = evaluate_stages(&ShadingModel::textured(7), [0.4, 0.4, 0.4, 1.0], &[None]);
    assert!(
        approx(out[0], 0.4, 1e-5) && approx(out[3], 1.0, 1e-5),
        "{out:?}"
    );
}

#[test]
fn golden_directional_light_is_clamped_at_the_lambert_term() {
    let mut model = ShadingModel::textured(0);
    model.kcolors[0] = [1.0, 0.5, 0.0, 1.0];
    model.ambient = 1;
    model.kcolors[1] = [0.0, 0.0, 0.0, 1.0];
    model
        .lights
        .push(Light::directional([0, 1], [0.0, 1.0, 0.0]));
    let facing = evaluate_lights(&model, [0.0, 1.0, 0.0], [0.0; 3]);
    assert!(
        approx(facing[0], 1.0, 1e-5) && approx(facing[1], 0.5, 1e-5),
        "{facing:?}"
    );
    let away = evaluate_lights(&model, [0.0, -1.0, 0.0], [0.0; 3]);
    assert!(approx(away[0], 0.0, 1e-5), "{away:?}");
}

#[test]
fn golden_point_light_fades_linearly_to_zero_at_range() {
    let mut model = ShadingModel::default();
    model.kcolors[0] = [1.0, 1.0, 1.0, 1.0];
    model.ambient = 1;
    model.kcolors[1] = [0.0, 0.0, 0.0, 1.0];
    model
        .lights
        .push(Light::point([0, 0], [0.0, 0.0, 0.0], 10.0));
    let half = evaluate_lights(&model, [0.0, 0.0, -1.0], [0.0, 0.0, 5.0]);
    assert!(approx(half[2], 0.5, 1e-4), "{half:?}");
    let past = evaluate_lights(&model, [0.0, 0.0, -1.0], [0.0, 0.0, 40.0]);
    assert!(approx(past[2], 0.0, 1e-5), "{past:?}");
}

#[test]
fn golden_fog_reaches_full_strength_at_the_far_plane() {
    let fog = MaterialFog {
        near: 10.0,
        far: 20.0,
        color: [0.0; 3],
        enabled: true,
    };
    assert!(approx(fog_factor(&fog, 5.0), 0.0, 1e-6));
    assert!(approx(fog_factor(&fog, 15.0), 0.5, 1e-6));
    assert!(approx(fog_factor(&fog, 25.0), 1.0, 1e-6));
    let off = MaterialFog {
        enabled: false,
        ..fog
    };
    assert!(approx(fog_factor(&off, 25.0), 0.0, 1e-6));
}

#[test]
fn golden_operand_and_mode_codes_match_the_hardware_encoding() {
    assert_eq!(TevOp::from_code(7), Some(TevOp::ATimesB));
    assert_eq!(TevOp::from_code(11), None);
    assert_eq!(TevOp2::from_code(4), Some(TevOp2::ATimesD));
    assert_eq!(TevOp2::from_code(9), None);
    assert_eq!(TevMode::from_code(3), Some(TevMode::Subtract));
    assert_eq!(TevMode::from_code(4), None);
}

#[test]
fn golden_shading_models_beyond_hardware_limits_are_rejected() {
    let too_many = ShadingModel {
        stages: vec![TevStage::default(); 17],
        ..Default::default()
    };
    assert!(too_many.validate().is_err());
    let too_many_lights = ShadingModel {
        lights: vec![Light::default(); 9],
        ..Default::default()
    };
    assert!(too_many_lights.validate().is_err());
    let bad_unit = ShadingModel {
        stages: vec![TevStage {
            tex_unit: 8,
            ..TevStage::default()
        }],
        ..Default::default()
    };
    assert_eq!(bad_unit.validate(), Err(ShadingError::TexUnitOutOfRange(8)));
}

#[test]
fn golden_shader_lowering_names_every_sampled_unit() {
    let wgsl = wgsl_program(&ShadingModel::textured(5));
    assert!(wgsl.contains("tex5"), "{wgsl}");
    assert!(wgsl.contains("dst[3]"), "alpha channel written: {wgsl}");
    assert!(wgsl.contains("outColor = src"), "{}", wgsl);
}

#[test]
fn golden_shader_cache_lowers_each_distinct_program_once() {
    let mut cache = ShaderCache::new();
    let a = ShadingModel::textured(0);
    let b = ShadingModel::textured(1);
    let first = cache.get(&a).expect("lowers").to_string();
    let again = cache.get(&a).expect("cached").to_string();
    assert_eq!(first, again);
    cache.get(&b).expect("second program");
    assert_eq!(cache.len(), 2);
}

#[test]
fn golden_shaded_group_takes_the_page_of_each_stage() {
    let mut model = ShadingModel::textured(0);
    model.stages.push(TevStage {
        tex_unit: 3,
        ..TevStage::default()
    });
    let mut group = MeshGroup {
        depth_test: true,
        ..Default::default()
    };
    group.push_tri([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0; 3]);
    let shaded = ShadedGroup::new(group, model).expect("valid model");
    assert_eq!(shaded.texture_pages, vec![0, 3]);
    assert_eq!(shaded.tri_count(), 1);
}

#[test]
fn golden_shaded_batch_counts_draws_and_skips_empty_groups() {
    let mut batch = ShadedBatch::new();
    let mut group = MeshGroup {
        depth_test: true,
        ..Default::default()
    };
    group.push_quad(
        [0.0; 3],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0; 3],
    );
    batch.push(ShadedGroup::new(group, ShadingModel::default()).unwrap());
    batch.push(ShadedGroup::new(MeshGroup::default(), ShadingModel::default()).unwrap());
    assert_eq!(batch.len(), 1);
    assert_eq!(batch.stats.draw_calls, 1);
    assert_eq!(batch.stats.triangles, 2);
    assert_eq!(batch.stats.culled_groups, 1);
}

// ---------------------------------------------------------------- texture formats

#[test]
fn golden_four_bit_rows_pack_two_texels_per_byte() {
    assert_eq!(TexFormat::I4.row_bytes(4), 2);
    assert_eq!(TexFormat::I4.image_size(4, 2), 4);
    assert_eq!(TexFormat::Rgba8.image_size(4, 2), 32);
    assert_eq!(TexFormat::Rgb565.bytes_per_texel(), 2);
}

#[test]
fn golden_palette_indices_decode_in_row_order() {
    let palette = Palette::new(vec![[255, 0, 0], [0, 255, 0], [0, 0, 255]], PaletteBits::I4);
    let out = decode_palette(
        &[0x01, 0x20],
        TexFormat::Indexed {
            bits: PaletteBits::I4,
        },
        &palette,
        4,
        1,
    )
    .expect("decodes");
    assert_eq!(&out[0..4], &[255, 0, 0, 0xFF]);
    assert_eq!(&out[4..8], &[0, 255, 0, 0xFF]);
    assert_eq!(&out[8..12], &[0, 0, 255, 0xFF]);
}

#[test]
fn golden_palette_short_table_is_an_error_not_a_panic() {
    let palette = Palette::new(vec![[1, 2, 3]], PaletteBits::I4);
    let out = decode_palette(
        &[0xFF],
        TexFormat::Indexed {
            bits: PaletteBits::I4,
        },
        &palette,
        2,
        1,
    );
    assert_eq!(out, Err(TexError::PaletteOutOfRange));
}

#[test]
fn golden_rgb565_and_rgb5a3_expand_to_eight_bit() {
    assert_eq!(
        rgba_from_texel(&[0xF8, 0x00], TexFormat::Rgb565),
        [255, 0, 0, 0xFF]
    );
    assert_eq!(rgba_from_texel(&[0xFC, 0x00], TexFormat::Rgb5A3)[3], 0xFF);
    assert_eq!(rgba_from_texel(&[0x00, 0x00], TexFormat::Rgb5A3)[3], 0x00);
    assert_eq!(
        rgba_from_texel(&[0x80, 0x40], TexFormat::Ia8),
        [128, 128, 128, 64]
    );
}

#[test]
fn golden_texel_repacking_round_trips_through_rgba() {
    for format in [TexFormat::Rgb565, TexFormat::Rgba8] {
        let packed = texel_from_rgba([64, 128, 192, 255], format);
        let back = rgba_from_texel(&packed, format);
        assert!(approx(back[0] as f32, 64.0, 12.0), "{format:?} {back:?}");
    }
}

#[test]
fn golden_palettize_picks_the_nearest_entry() {
    let palette = Palette::new(vec![[0, 0, 0], [255, 255, 255]], PaletteBits::I4);
    assert_eq!(palettize_rgba([250, 250, 250], &palette), Some(1));
    assert_eq!(palettize_rgba([4, 4, 4], &palette), Some(0));
}

#[test]
fn golden_mip_chain_halves_down_to_one_by_one() {
    let levels = mip_chain(&vec![200_u8; 16 * 16 * 4], 16, 16);
    assert_eq!(levels.len(), 5);
    assert_eq!((levels[0].0, levels[0].1), (16, 16));
    assert_eq!((levels[4].0, levels[4].1), (1, 1));
    assert!(
        levels
            .iter()
            .all(|(_, _, data)| data.iter().all(|b| *b == 200))
    );
}

#[test]
fn golden_texture_overrides_replace_pages_after_upload() {
    let mut overrides = TextureOverrides::new();
    overrides.insert(3, vec![1, 2, 3, 4]);
    assert_eq!(overrides.get(3), Some(&[1_u8, 2, 3, 4][..]));
    overrides.remove(3);
    assert!(overrides.get(3).is_none());
    assert!(overrides.is_empty());
}

// ---------------------------------------------------------------- vertex layouts

#[test]
fn golden_console_standard_layout_packs_into_twenty_bytes() {
    let layout = VertexLayout::console_standard();
    assert_eq!(layout.stride, 20);
    assert_eq!(
        layout.attribute(VertexSemantic::Position).unwrap().offset,
        0
    );
    assert_eq!(layout.attribute(VertexSemantic::Normal).unwrap().offset, 12);
    assert_eq!(
        layout.attribute(VertexSemantic::Binormal).unwrap().offset,
        13
    );
    assert_eq!(layout.attribute(VertexSemantic::Uv(0)).unwrap().offset, 16);
}

#[test]
fn golden_layout_packing_aligns_and_rejects_duplicates() {
    let packed = VertexLayout::pack(
        vec![
            VertexAttribute::new(VertexSemantic::Color, VertexFormat::U8, 3),
            VertexAttribute::new(VertexSemantic::Position, VertexFormat::F32, 3),
        ],
        4,
    )
    .expect("packs");
    assert_eq!(
        packed.attribute(VertexSemantic::Position).unwrap().offset,
        4
    );
    assert_eq!(packed.stride, 16);
    let duplicate = VertexLayout::pack(
        vec![
            VertexAttribute::new(VertexSemantic::Position, VertexFormat::F32, 3),
            VertexAttribute::new(VertexSemantic::Position, VertexFormat::F32, 3),
        ],
        4,
    );
    assert!(duplicate.is_err());
}

#[test]
fn golden_index_format_picks_the_narrowest_width() {
    assert_eq!(IndexFormat::for_vertex_count(200), IndexFormat::U8);
    assert_eq!(IndexFormat::for_vertex_count(300), IndexFormat::U16);
    assert_eq!(IndexFormat::for_vertex_count(70_000), IndexFormat::U32);
    assert_eq!(IndexFormat::U16.byte_size(3), 6);
}

#[test]
fn golden_packed_components_expand_to_their_declared_ranges() {
    assert!(approx(
        expand_component(0xF, VertexFormat::U8x4, 0),
        1.0,
        1e-6
    ));
    assert!(approx(
        expand_component(0x8, VertexFormat::S8x4, 0),
        0.0,
        1e-6
    ));
    assert!(approx(
        expand_component(0xF, VertexFormat::S8x4, 0),
        1.0,
        1e-6
    ));
    let packed = 1023 | (512 << 20);
    assert!(approx(expand_10_10_10_2(packed, 0, false), 1.0, 1e-6));
    assert!(approx(expand_10_10_10_2(packed, 1, false), 0.0, 1e-6));
}

// ---------------------------------------------------------------- render passes

fn render_pass(name: &str) -> RenderPass {
    RenderPass {
        target: RenderTarget::color(name, 64, 64),
        kind: PassKind::Render,
        source: None,
        groups: Vec::new(),
    }
}

#[test]
fn golden_pass_lists_chain_forward_only() {
    let mut blit = render_pass("blit");
    blit.kind = PassKind::Fullscreen;
    blit.source = Some("scene".to_string());
    assert_eq!(validate_passes(&[render_pass("scene"), blit]), Ok(()));
    let mut forward = render_pass("blit");
    forward.kind = PassKind::Resolve;
    forward.source = Some("later".to_string());
    assert_eq!(validate_passes(&[forward]), Err(PassError::UnknownSource));
}

#[test]
fn golden_pass_validation_rejects_bad_declarations() {
    let mut zero = render_pass("scene");
    zero.target.width = 0;
    assert_eq!(validate_passes(&[zero]), Err(PassError::ZeroSized));
    let mut samples = render_pass("scene");
    samples.target.samples = 3;
    assert_eq!(
        validate_passes(&[samples]),
        Err(PassError::SampleCountUnsupported)
    );
    assert_eq!(
        validate_passes(&[render_pass("a"), render_pass("a")]),
        Err(PassError::DuplicateTarget)
    );
    let mut orphan = render_pass("a");
    orphan.kind = PassKind::Fullscreen;
    assert_eq!(validate_passes(&[orphan]), Err(PassError::MissingSource));
}

#[test]
fn golden_render_target_size_accounts_for_samples_and_format() {
    let target = RenderTarget::new(
        "normals",
        32,
        32,
        RenderTargetFormat::Rgba16Float,
        TargetSemantics::Normal,
    )
    .with_samples(4);
    assert_eq!(target.byte_size(), 32 * 32 * 8 * 4);
    assert!(!target.format.is_depth());
    assert!(RenderTargetFormat::Depth32Float.is_depth());
}

// ---------------------------------------------------------------- statistics

#[test]
fn golden_stats_separate_merged_from_effective_draws() {
    let stats = FrameStats {
        draw_calls: 10,
        merged_draw_calls: 4,
        triangles: 100,
        ..FrameStats::default()
    };
    assert_eq!(stats.effective_draw_calls(), 6);
    assert!(approx(stats.triangles_per_draw(), 100.0 / 6.0, 1e-4));
    assert!(FrameStats::default().is_empty());
}

#[test]
fn golden_sinks_track_latest_and_rolling_frames() {
    let mut latest = LatestFrame::new();
    latest.publish(FrameStats {
        draw_calls: 1,
        ..Default::default()
    });
    latest.publish(FrameStats {
        draw_calls: 7,
        ..Default::default()
    });
    assert_eq!(latest.stats.draw_calls, 7);
    assert_eq!(latest.frames_seen, 2);

    let mut rolling = RollingAverage::new(8);
    for _ in 0..10 {
        rolling.publish(FrameStats {
            draw_calls: 10,
            triangles: 100,
            ..Default::default()
        });
    }
    assert_eq!(rolling.average.draw_calls, 10);
    assert_eq!(rolling.samples, 10);
}

#[test]
fn golden_full_turn_returns_the_device_to_its_starting_orientation() {
    let mut tracker = MotionTracker::new(SensorKind::GyroscopeAccelerometer);
    for _ in 0..4 {
        tracker.integrate(
            MotionSample {
                acceleration: Vec3::ZERO,
                angular_velocity: Vec3::new(0.0, PI / 2.0, 0.0),
            },
            1.0,
        );
    }
    let forward = tracker.orientation() * Vec3::X;
    assert!(
        approx(forward.x, 1.0, 1e-4) && forward.z.abs() < 1e-4,
        "{forward:?}"
    );
}
