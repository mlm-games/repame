//! Reference evaluation and WGSL lowering for [`ShadingModel`].
//!
//! The CPU path exists so a game can resolve a material without a GPU, and so
//! the lowering has something to be checked against: [`wgsl_program`] emits
//! the same arithmetic in WGSL for the GPU path.

use crate::material::{
    KColors, Light, MaterialFog, Rgba, ShadingModel, TevArg, TevMode, TevOp, TevOp2, TevOperand,
    TevStage,
};

/// Sampled texel for one texture unit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TexInput {
    pub color: [f32; 3],
    pub alpha: f32,
    pub lod_frac: f32,
}

impl Default for TexInput {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0],
            alpha: 1.0,
            lod_frac: 0.0,
        }
    }
}

/// A texel lookup failure at unit index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TexUnitMissing(pub u8);

fn arg_value(arg: TevArg, unit: u8, channel: usize, texels: &[Option<TexInput>]) -> f32 {
    match arg {
        TevArg::Zero => 0.0,
        TevArg::One => 1.0,
        TevArg::Half => 0.5,
        TevArg::Two => 2.0,
        TevArg::Color => 0.0,
        TevArg::Alpha => 0.0,
        TevArg::KColor(_) => 0.0,
        TevArg::TexColor | TevArg::TexAlpha | TevArg::LodFrac => {
            match texels.get(usize::from(unit)).and_then(|slot| *slot) {
                Some(texel) => match arg {
                    TevArg::LodFrac => texel.lod_frac,
                    TevArg::TexAlpha => texel.alpha,
                    _ => texel.color[channel.min(3)],
                },
                None => 1.0,
            }
        }
        TevArg::Stub => {
            if channel == 3 {
                0.0
            } else {
                1.0
            }
        }
        TevArg::TexColorOf(other) => match texels.get(usize::from(other)).and_then(|slot| *slot) {
            Some(texel) => texel.color[channel.min(3)],
            None => 1.0,
        },
    }
}

fn scaled(
    operand: TevOperand,
    unit: u8,
    channel: usize,
    texels: &[Option<TexInput>],
    current: Rgba,
    kcolors: &KColors,
) -> f32 {
    let raw = match operand.arg {
        TevArg::Color => current[channel.min(3)],
        TevArg::Alpha => current[3],
        TevArg::KColor(index) => kcolors
            .get(usize::from(index))
            .copied()
            .unwrap_or([0.0, 0.0, 0.0, 1.0])[channel.min(3)],
        other => arg_value(other, unit, channel, texels),
    };
    let scaled = raw * operand.scale.multiplier() + operand.scale.offset();
    if operand.negate { -scaled } else { scaled }
}

fn fold(op: TevOp, a: f32, b: f32, c: f32) -> f32 {
    match op {
        TevOp::A => a,
        TevOp::OneMinusA => 1.0 - a,
        TevOp::C => c,
        TevOp::OneMinusC => 1.0 - c,
        TevOp::APlusB => a + b,
        TevOp::AMinusB => a - b,
        TevOp::AMinusC => a - c,
        TevOp::ATimesB => a * b,
        TevOp::APlusBHalf => (a + b) * 0.5,
        TevOp::BPlusCHalf => (b + c) * 0.5,
        TevOp::BMinusCHalf => (b - c) * 0.5,
    }
}

fn fold2(op: TevOp2, a: f32, b: f32, d: f32) -> f32 {
    match op {
        TevOp2::D => d,
        TevOp2::OneMinusD => 1.0 - d,
        TevOp2::APlusD => a + d,
        TevOp2::APlusOneMinusD => a + (1.0 - d),
        TevOp2::ATimesD => a * d,
        TevOp2::ATimesOneMinusD => a * (1.0 - d),
        TevOp2::APlusB => a + b,
        TevOp2::AMinusB => a - b,
        TevOp2::ATimesA => a * a,
    }
}

fn combine(mode: TevMode, a: f32, b: f32) -> f32 {
    match mode {
        TevMode::Replace => a,
        TevMode::Modulate => a * b,
        TevMode::Add => a + b,
        TevMode::Subtract => a - b,
    }
}

fn stage_channel(
    stage: &TevStage,
    channel: usize,
    texels: &[Option<TexInput>],
    current: Rgba,
    destination: Rgba,
    kcolors: &KColors,
) -> f32 {
    let operand_count = if channel == 3 { 3 } else { 4 };
    let mut values = [0.0_f32; 4];
    for (slot, value) in values.iter_mut().enumerate().take(operand_count) {
        let operand = if channel == 3 {
            stage.alpha_arg[slot]
        } else {
            stage.color_arg[slot]
        };
        *value = scaled(operand, stage.tex_unit, channel, texels, current, kcolors);
    }
    let op = if channel == 3 {
        stage.alpha_op[0]
    } else {
        stage.color_op[channel]
    };
    let mut combined = fold(op, values[0], values[1], values[2]);
    if operand_count == 4 {
        let fold_c = if channel == 3 {
            stage.alpha_op[1]
        } else {
            stage.color_op[1]
        };
        combined = fold(fold_c, combined, values[2], values[2]);
    }
    let second = fold2(
        stage.op2[channel.min(3)],
        combined,
        values[1],
        destination[channel],
    );
    let mode = if channel == 3 {
        stage.alpha_mode
    } else {
        stage.color_mode
    };
    let scale = if channel == 3 {
        stage.alpha_scale
    } else {
        stage.color_scale
    };
    combine(mode, combined, second) * scale.multiplier() + scale.offset()
}

/// Runs the stage program over an incoming color.
pub fn evaluate_stages(model: &ShadingModel, incoming: Rgba, texels: &[Option<TexInput>]) -> Rgba {
    let mut current = incoming;
    for stage in &model.stages {
        let mut next = [0.0_f32; 4];
        for (channel, slot) in next.iter_mut().enumerate() {
            *slot = stage_channel(stage, channel, texels, current, current, &model.kcolors);
        }
        current = next;
    }
    [
        current[0].clamp(0.0, 1.0),
        current[1].clamp(0.0, 1.0),
        current[2].clamp(0.0, 1.0),
        current[3].clamp(0.0, 1.0),
    ]
}

/// Diffuse and specular contribution of the model's lights.
pub fn evaluate_lights(model: &ShadingModel, normal: [f32; 3], world_pos: [f32; 3]) -> [f32; 3] {
    let mut accum = [0.0_f32; 3];
    for light in &model.lights {
        let kcolor = light
            .colors
            .iter()
            .flatten()
            .next()
            .and_then(|i| model.kcolors.get(usize::from(*i)).copied())
            .unwrap_or([0.0, 0.0, 0.0, 1.0]);
        let (direction, attenuation) = light_direction(light, world_pos);
        let lambert =
            (normal[0] * direction[0] + normal[1] * direction[1] + normal[2] * direction[2])
                .max(0.0);
        for channel in 0..3 {
            accum[channel] += lambert * kcolor[channel] * attenuation;
        }
    }
    let ambient = model
        .kcolors
        .get(usize::from(model.ambient))
        .copied()
        .unwrap_or([0.0, 0.0, 0.0, 1.0]);
    for channel in 0..3 {
        accum[channel] += ambient[channel] * model.ao;
    }
    accum
}

fn light_direction(light: &Light, world_pos: [f32; 3]) -> ([f32; 3], f32) {
    if light.distance[0] == 0.0 {
        return (normalize3(light.direction), 1.0);
    }
    let to_light = [
        light.position[0] - world_pos[0],
        light.position[1] - world_pos[1],
        light.position[2] - world_pos[2],
    ];
    let distance = (to_light[0].powi(2) + to_light[1].powi(2) + to_light[2].powi(2)).sqrt();
    let direction = normalize3(to_light);
    let range = light.distance[1].max(1e-6);
    let mut attenuation = 1.0 - (distance / range).clamp(0.0, 1.0);
    if light.angles[1] > -0.999 {
        let cos_angle = -(direction[0] * light.direction[0]
            + direction[1] * light.direction[1]
            + direction[2] * light.direction[2]);
        if cos_angle < light.angles[1] {
            attenuation = 0.0;
        } else if light.angles[0] > light.angles[1] {
            let t = (cos_angle - light.angles[1]) / (light.angles[0] - light.angles[1]);
            attenuation *= t.clamp(0.0, 1.0);
        }
    }
    (direction, attenuation)
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let length = (v[0].powi(2) + v[1].powi(2) + v[2].powi(2)).sqrt();
    if length < 1e-6 {
        [0.0, 1.0, 0.0]
    } else {
        [v[0] / length, v[1] / length, v[2] / length]
    }
}

/// Fog blend factor for a view distance.
pub fn fog_factor(fog: &MaterialFog, distance: f32) -> f32 {
    if !fog.enabled {
        return 0.0;
    }
    let span = (fog.far - fog.near).max(1e-6);
    ((distance - fog.near) / span).clamp(0.0, 1.0)
}

/// Full CPU reference: light, stage program, then fog.
#[allow(clippy::too_many_arguments)] // shading inputs are the hardware's inputs
pub fn evaluate(
    model: &ShadingModel,
    vertex_color: Rgba,
    normal: [f32; 3],
    world_pos: [f32; 3],
    eye: [f32; 3],
    texels: &[Option<TexInput>],
) -> Rgba {
    let lit = evaluate_lights(model, normal, world_pos);
    let incoming = [
        vertex_color[0] * lit[0],
        vertex_color[1] * lit[1],
        vertex_color[2] * lit[2],
        vertex_color[3],
    ];
    let mut out = evaluate_stages(model, incoming, texels);
    let distance = ((world_pos[0] - eye[0]).powi(2)
        + (world_pos[1] - eye[1]).powi(2)
        + (world_pos[2] - eye[2]).powi(2))
    .sqrt();
    let t = fog_factor(&model.material_fog, distance);
    out = [
        out[0] + (model.material_fog.color[0] - out[0]) * t,
        out[1] + (model.material_fog.color[1] - out[1]) * t,
        out[2] + (model.material_fog.color[2] - out[2]) * t,
        out[3],
    ];
    out
}

fn arg_wgsl(arg: TevArg, unit: usize, alpha_channel: bool) -> String {
    match arg {
        TevArg::Zero => "0.0".to_string(),
        TevArg::One => "1.0".to_string(),
        TevArg::Half => "0.5".to_string(),
        TevArg::Two => "2.0".to_string(),
        TevArg::Color => "src".to_string(),
        TevArg::Alpha => "src.a".to_string(),
        TevArg::KColor(index) => format!("kcolor[{}]", index.min(3)),
        TevArg::TexColor => format!("tex{}", unit),
        TevArg::TexAlpha => format!("tex{}.a", unit),
        TevArg::LodFrac => format!("lod{}", unit),
        TevArg::Stub => {
            if alpha_channel {
                "0.0".to_string()
            } else {
                "1.0".to_string()
            }
        }
        TevArg::TexColorOf(other) => format!("tex{}", usize::from(other).min(7)),
    }
}

fn operand_wgsl(operand: TevOperand, unit: usize, channel: usize) -> String {
    let raw = arg_wgsl(operand.arg, unit, channel == 3);
    let mut expr = format!(
        "({raw} * {} + {})",
        operand.scale.multiplier(),
        operand.scale.offset()
    );
    if operand.negate {
        expr = format!("(-{expr})");
    }
    expr
}

fn op_wgsl(op: TevOp, a: &str, b: &str, c: &str) -> String {
    let expr = match op {
        TevOp::A => a.to_string(),
        TevOp::OneMinusA => format!("(1.0 - {a})"),
        TevOp::C => c.to_string(),
        TevOp::OneMinusC => format!("(1.0 - {c})"),
        TevOp::APlusB => format!("({a} + {b})"),
        TevOp::AMinusB => format!("({a} - {b})"),
        TevOp::AMinusC => format!("({a} - {c})"),
        TevOp::ATimesB => format!("({a} * {b})"),
        TevOp::APlusBHalf => format!("(({a} + {b}) * 0.5)"),
        TevOp::BPlusCHalf => format!("(({b} + {c}) * 0.5)"),
        TevOp::BMinusCHalf => format!("(({b} - {c}) * 0.5)"),
    };
    format!("clamp({expr}, -4.0, 4.0)")
}

fn op2_wgsl(op: TevOp2, a: &str, d: &str) -> String {
    let expr = match op {
        TevOp2::D => d.to_string(),
        TevOp2::OneMinusD => format!("(1.0 - {d})"),
        TevOp2::APlusD => format!("({a} + {d})"),
        TevOp2::APlusOneMinusD => format!("({a} + (1.0 - {d}))"),
        TevOp2::ATimesD => format!("({a} * {d})"),
        TevOp2::ATimesOneMinusD => format!("({a} * (1.0 - {d}))"),
        TevOp2::APlusB => format!("({a} + {a})"),
        TevOp2::AMinusB => format!("({a} - {a})"),
        TevOp2::ATimesA => format!("({a} * {a})"),
    };
    format!("clamp({expr}, -4.0, 4.0)")
}

fn mode_wgsl(mode: TevMode, a: &str, b: &str) -> String {
    let expr = match mode {
        TevMode::Replace => a.to_string(),
        TevMode::Modulate => format!("({a} * {b})"),
        TevMode::Add => format!("({a} + {b})"),
        TevMode::Subtract => format!("({a} - {b})"),
    };
    format!("clamp({expr}, 0.0, 1.0)")
}

fn stage_channel_wgsl(stage: &TevStage, channel: usize) -> String {
    let unit = usize::from(stage.tex_unit).min(7);
    let count = if channel == 3 { 3 } else { 4 };
    let mut names: Vec<String> = Vec::with_capacity(count);
    for slot in 0..count {
        let operand = if channel == 3 {
            stage.alpha_arg[slot]
        } else {
            stage.color_arg[slot]
        };
        names.push(operand_wgsl(operand, unit, channel));
    }
    let op = if channel == 3 {
        stage.alpha_op[0]
    } else {
        stage.color_op[channel]
    };
    let mut combined = op_wgsl(op, &names[0], &names[1], &names[2]);
    if count == 4 {
        let second_op = if channel == 3 {
            stage.alpha_op[1]
        } else {
            stage.color_op[1]
        };
        combined = op_wgsl(second_op, &combined, &names[1], &names[2]);
    }
    let second = op2_wgsl(
        stage.op2[channel.min(3)],
        &combined,
        &format!("dst[{}]", channel),
    );
    let mode = if channel == 3 {
        stage.alpha_mode
    } else {
        stage.color_mode
    };
    let scale = if channel == 3 {
        stage.alpha_scale
    } else {
        stage.color_scale
    };
    format!(
        "clamp(({} * {} + {}), 0.0, 1.0)",
        mode_wgsl(mode, &combined, &second),
        scale.multiplier(),
        scale.offset()
    )
}

/// WGSL for the stage program, as a fragment-shader function body.
///
/// `texN` / `lodN` bindings are expected in scope per sampled unit, plus
/// `kcolor` (4 RGBA constants) and the incoming `src` color. Assigns `outColor`.
pub fn wgsl_program(model: &ShadingModel) -> String {
    let mut body = String::new();
    body.push_str("var src: vec4<f32> = stage_in;\n");
    for stage in &model.stages {
        body.push_str("var dst: vec4<f32> = src;\n");
        for channel in 0..4 {
            let expr = stage_channel_wgsl(stage, channel);
            body.push_str(&format!("dst[{channel}] = {expr};\n"));
        }
        body.push_str("src = dst;\n");
    }
    body.push_str("outColor = src;\n");
    body
}
