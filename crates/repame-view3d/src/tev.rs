//! Reference evaluation and WGSL lowering for [`ShadingModel`].
//!
//! The CPU path exists so a game can resolve a material without a GPU, and so
//! the lowering has something to be checked against: both paths read the same
//! per-channel equation, one folding it over registers, [`wgsl_program`]
//! emitting it in WGSL.

use crate::material::{
    KColors, Light, MaterialFog, Rgba, ShadingModel, TevArg, TevBias, TevDest, TevOp, TevScale,
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

/// One channel of one stage: its arguments and the fold applied to them.
///
/// Channels 0..2 read the stage's color side, channel 3 the alpha side, so
/// the CPU and WGSL paths cannot drift apart per channel.
struct StageTerms {
    args: [TevArg; 4],
    op: TevOp,
    bias: TevBias,
    scale: TevScale,
    clamp: bool,
    dest: TevDest,
}

fn stage_terms(stage: &TevStage, channel: usize) -> StageTerms {
    if channel == 3 {
        StageTerms {
            args: stage.alpha_arg,
            op: stage.alpha_op,
            bias: stage.alpha_bias,
            scale: stage.alpha_scale,
            clamp: stage.alpha_clamp,
            dest: stage.alpha_dest,
        }
    } else {
        StageTerms {
            args: stage.color_arg,
            op: stage.color_op,
            bias: stage.color_bias,
            scale: stage.color_scale,
            clamp: stage.color_clamp,
            dest: stage.color_dest,
        }
    }
}

/// One channel's equation: `scale * (d +/- ((1 - c) * a + c * b) + bias)`.
///
/// Clamped channels fold into `0..1`; unclamped ones keep the 10-bit signed
/// intermediate range the hardware rounds from.
fn combine_channel(terms: &StageTerms, args: [f32; 4]) -> f32 {
    let [a, b, c, d] = args;
    let lerp = (1.0 - c) * a + c * b;
    let mixed = match terms.op {
        TevOp::Add => d + lerp,
        TevOp::Sub => d - lerp,
    };
    let scaled = (mixed + terms.bias.offset()) * terms.scale.multiplier();
    if terms.clamp {
        scaled.clamp(0.0, 1.0)
    } else {
        scaled.clamp(-4.0, 1023.0 / 256.0)
    }
}

/// Reads one argument against the register file; a missing texel reads white.
fn arg_value(
    arg: TevArg,
    unit: u8,
    channel: usize,
    texels: &[Option<TexInput>],
    regs: &[[f32; 4]; 4],
    kcolors: &KColors,
    raster: &[f32; 4],
) -> f32 {
    let texel = |other: u8| {
        texels
            .get(usize::from(other))
            .and_then(|slot| *slot)
            .unwrap_or(TexInput {
                color: [1.0; 3],
                alpha: 1.0,
                lod_frac: 1.0,
            })
    };
    match arg {
        TevArg::Zero => 0.0,
        TevArg::One => 1.0,
        TevArg::Half => 0.5,
        TevArg::Two => 2.0,
        TevArg::Color => regs[0][channel],
        TevArg::Alpha => regs[0][3],
        TevArg::Reg0 => regs[1][channel],
        TevArg::Reg1 => regs[2][channel],
        TevArg::Reg2 => regs[3][channel],
        TevArg::Rasc => {
            if channel == 3 {
                raster[3]
            } else {
                raster[channel]
            }
        }
        TevArg::Rasa => raster[3],
        TevArg::KColor(index) => kcolors
            .get(usize::from(index))
            .copied()
            .unwrap_or([0.0, 0.0, 0.0, 1.0])[channel],
        TevArg::TexColor => {
            let texel = texel(unit);
            if channel == 3 {
                texel.alpha
            } else {
                texel.color[channel]
            }
        }
        TevArg::TexColorOf(other) => {
            let texel = texel(other);
            if channel == 3 {
                texel.alpha
            } else {
                texel.color[channel]
            }
        }
        TevArg::TexAlpha => texel(unit).alpha,
        TevArg::LodFrac => texel(unit).lod_frac,
        TevArg::Stub => {
            if channel == 3 {
                0.0
            } else {
                1.0
            }
        }
    }
}

/// Runs the stage program over an incoming color.
///
/// The rasterized color is register 0 at the first stage (the hardware's
/// `prev` register), registers 0..2 start at zero, and every stage reads the
/// register file before writing its color and alpha destinations. The result
/// is the last stage's computed channels, wherever they landed.
pub fn evaluate_stages(model: &ShadingModel, incoming: Rgba, texels: &[Option<TexInput>]) -> Rgba {
    let mut regs = [[0.0_f32; 4]; 4];
    regs[0] = incoming;
    let mut last = incoming;
    for stage in &model.stages {
        let mut result = [0.0_f32; 4];
        for (channel, out) in result.iter_mut().enumerate() {
            let terms = stage_terms(stage, channel);
            let args = terms.args.map(|arg| {
                arg_value(
                    arg,
                    stage.tex_unit,
                    channel,
                    texels,
                    &regs,
                    &model.kcolors,
                    &incoming,
                )
            });
            *out = combine_channel(&terms, args);
        }
        let color_at = stage_terms(stage, 0).dest.index();
        let alpha_at = stage_terms(stage, 3).dest.index();
        regs[color_at][0..3].copy_from_slice(&result[0..3]);
        regs[alpha_at][3] = result[3];
        last = result;
    }
    [
        last[0].clamp(0.0, 1.0),
        last[1].clamp(0.0, 1.0),
        last[2].clamp(0.0, 1.0),
        last[3].clamp(0.0, 1.0),
    ]
}

/// Diffuse contribution of the model's lights, plus its ambient.
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

fn arg_wgsl(arg: TevArg, unit: usize, channel: usize) -> String {
    let slot = ["r", "g", "b", "a"][channel];
    match arg {
        TevArg::Zero => "0.0".to_string(),
        TevArg::One => "1.0".to_string(),
        TevArg::Half => "0.5".to_string(),
        TevArg::Two => "2.0".to_string(),
        TevArg::Color => format!("tev_regs[0].{slot}"),
        TevArg::Alpha => "tev_regs[0].a".to_string(),
        TevArg::Reg0 => format!("tev_regs[1].{slot}"),
        TevArg::Reg1 => format!("tev_regs[2].{slot}"),
        TevArg::Reg2 => format!("tev_regs[3].{slot}"),
        TevArg::Rasc => {
            if channel == 3 {
                "lit_in.a".to_string()
            } else {
                format!("lit_in.{slot}")
            }
        }
        TevArg::Rasa => "lit_in.a".to_string(),
        TevArg::KColor(index) => format!("kcolor[{}].{slot}", index.min(3)),
        TevArg::TexColor => {
            if channel == 3 {
                format!("tex{unit}.a")
            } else {
                format!("tex{unit}.{slot}")
            }
        }
        TevArg::TexColorOf(other) => {
            let other = usize::from(other).min(7);
            if channel == 3 {
                format!("tex{other}.a")
            } else {
                format!("tex{other}.{slot}")
            }
        }
        TevArg::TexAlpha => format!("tex{unit}.a"),
        TevArg::LodFrac => format!("lod{unit}"),
        TevArg::Stub => {
            if channel == 3 {
                "0.0".to_string()
            } else {
                "1.0".to_string()
            }
        }
    }
}

fn stage_channel_wgsl(stage: &TevStage, channel: usize) -> String {
    let unit = usize::from(stage.tex_unit).min(7);
    let terms = stage_terms(stage, channel);
    let [a, b, c, d] = terms.args.map(|arg| arg_wgsl(arg, unit, channel));
    let lerp = format!("((1.0 - {c}) * {a} + {c} * {b})");
    let mixed = match terms.op {
        TevOp::Add => format!("({d} + {lerp})"),
        TevOp::Sub => format!("({d} - {lerp})"),
    };
    let biased = match terms.bias {
        TevBias::Zero => mixed,
        TevBias::AddHalf => format!("({mixed} + 0.5)"),
        TevBias::SubHalf => format!("({mixed} - 0.5)"),
    };
    let scaled = match terms.scale {
        TevScale::X1 => biased,
        TevScale::X2 => format!("({biased} * 2.0)"),
        TevScale::X4 => format!("({biased} * 4.0)"),
        TevScale::Divide2 => format!("({biased} * 0.5)"),
    };
    if terms.clamp {
        format!("clamp({scaled}, 0.0, 1.0)")
    } else {
        format!("clamp({scaled}, -4.0, 3.99609375)")
    }
}

/// WGSL for the stage program, as a fragment-shader function body.
///
/// `texN` / `lodN` bindings are expected in scope per sampled unit (the
/// sampled texel, alpha included), plus `kcolor` (4 RGBA constants) and the
/// incoming `lit_in` color. Assigns `outColor`.
pub fn wgsl_program(model: &ShadingModel) -> String {
    let mut body = String::from(
        "var tev_regs: array<vec4<f32>, 4> = array<vec4<f32>, 4>(lit_in, vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));\n",
    );
    body.push_str("var tev_out: vec4<f32> = lit_in;\n");
    for (stage_index, stage) in model.stages.iter().enumerate() {
        for channel in 0..4 {
            let expr = stage_channel_wgsl(stage, channel);
            body.push_str(&format!("let s{stage_index}_{channel} = {expr};\n"));
        }
        let color_at = stage.color_dest.index();
        let alpha_at = stage.alpha_dest.index();
        body.push_str(&format!(
            "tev_regs[{color_at}] = vec4<f32>(s{stage_index}_0, s{stage_index}_1, s{stage_index}_2, tev_regs[{color_at}].a);\n"
        ));
        body.push_str(&format!("tev_regs[{alpha_at}].a = s{stage_index}_3;\n"));
        body.push_str(&format!(
            "tev_out = vec4<f32>(s{stage_index}_0, s{stage_index}_1, s{stage_index}_2, s{stage_index}_3);\n"
        ));
    }
    body.push_str("outColor = tev_out;\n");
    body
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::{ShadingModel, TevArg, TevMode};

    /// `Rasc`/`Rasa` read the incoming tint, which the hardware's stage 0
    /// seeds before any texture enters. The GPU names it `lit_in`; if the two
    /// paths ever disagree on where it comes from, every untextured but
    /// tinted surface shifts silently, so both are pinned here.
    #[test]
    fn raster_args_read_the_incoming_tint() {
        let mut model = ShadingModel::default();
        let mut stage = TevStage::with_mode(0, TevMode::Modulate);
        // `d + RASC`: the console's stage-0 accumulator for rasterized color.
        stage.color_arg = [TevArg::Zero, TevArg::Zero, TevArg::Zero, TevArg::Rasc];
        stage.alpha_arg = [TevArg::Zero, TevArg::Zero, TevArg::Zero, TevArg::Rasa];
        model.stages = vec![stage];

        let incoming = [0.25, 0.5, 0.75, 0.5];
        let out = evaluate_stages(&model, incoming, &[]);
        assert_eq!(out, incoming, "raster args must pass the tint through");
        assert!(model.validate().is_ok(), "raster args are valid in both channels");

        let wgsl = wgsl_program(&model);
        assert!(wgsl.contains("lit_in.g"), "color must read the tint: {wgsl}");
        assert!(wgsl.contains("lit_in.a"), "alpha must read its alpha: {wgsl}");
    }
}
