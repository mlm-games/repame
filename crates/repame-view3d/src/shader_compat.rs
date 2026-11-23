//! WGSL compatibility for backends that cannot translate explicit-LOD sampling.
//!
//! The scene shaders sample streamed texture pages at a chosen mip level with
//! `textureSampleLevel`. Naga's GLSL backend only emits that when the target
//! supports `Features::TEXTURE_LEVELS` (GLSL 1.30 / ES 3.00 or later). Old GL
//! adapters and WebGL1-style targets lack it and fail pipeline creation with
//! "The selected version doesn't support Features(TEXTURE_LEVELS)".
//!
//! On those devices we rewrite the source so samples use implicit LOD
//! (`textureSample`), which every fragment stage supports. For ClearSleeper,
//! which uploads every page fully resident (level 0), streamed-page samples are
//! byte-identical to the explicit-level-0 sample. There is a quality trade-off
//! on the fallback path: reflection probes lose roughness-based LOD and the
//! prefilter loop uses implicit derivatives, so reflections are blurrier or
//! less precise on old GL. Rendering still works everywhere; only fidelity
//! changes. Modern GL 4.x and all browser devices also take the fallback path
//! (conservative), costing the same fidelity there.

/// Whether explicit-LOD sampling (`textureSampleLevel`) is available.
///
/// Naga's GLSL backend only emits `textureSampleLevel` when the target supports
/// `TEXTURE_LEVELS` (GLSL 1.30 / ES 3.00+). Old GL adapters and WebGL1-style
/// targets lack it, so we conservatively avoid explicit LOD on the GL backend.
/// All other backends (Vulkan, Metal, DX12, WebGPU) support it. This must be a
/// synchronous check: an error-scope probe would return a JS Promise on the
/// WebGPU backend, which cannot be resolved from a WASM render callback.
pub fn explicit_lod_supported(device: &wgpu::Device) -> bool {
    match device.adapter_info().backend {
        // Native GL: naga's GLSL backend lacks TEXTURE_LEVELS on old drivers.
        wgpu::Backend::Gl => false,
        // In the browser wgpu reports BrowserWebGpu even when Firefox/Dawn
        // implements WebGPU on top of an old OpenGL driver (e.g. Radeon HD
        // 3200), where shader translation rejects TEXTURE_LEVELS. We cannot
        // query the underlying driver from here, so take the safe path for all
        // browser devices: implicit LOD renders everywhere, explicit LOD may not.
        wgpu::Backend::BrowserWebGpu => false,
        // Vulkan, Metal, DX12 and null backends all support explicit LOD.
        _ => true,
    }
}

/// Create a WGSL shader module, rewriting explicit-LOD samples when the
/// device cannot translate them. All repame-view3d shader creation goes
/// through here so the old-GL fallback is applied uniformly.
pub fn module(device: &wgpu::Device, label: &str, source: String) -> wgpu::ShaderModule {
    // `explicit_lod_supported` reads only `adapter_info().backend` — cheap and
    // synchronous. Computing it per call avoids a process-wide cache so two
    // devices on different backends each get the right shader variant.
    let source = if explicit_lod_supported(device) {
        source
    } else {
        without_explicit_lod(&source)
    };
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

/// Rewrite explicit-LOD samples to implicit-LOD samples for a device lacking
/// `TEXTURE_LEVELS`. The transformation is applied to the final WGSL source.
pub fn without_explicit_lod(source: &str) -> String {
    let mut out = source.to_string();
    // Scene page samples: `textureSampleLevel(tex, smp, uv, page, resident)` ->
    // `textureSample(tex, smp, uv, page)`. The resident level is dropped; the
    // page (array layer) argument is kept.
    out = rewrite_calls(&out, "textureSampleLevel", "textureSample", 5);
    // Probe / prefilter samples: `textureSampleLevel(tex, smp, coord, lod)` ->
    // `textureSample(tex, smp, coord)`.
    out = rewrite_calls(&out, "textureSampleLevel", "textureSample", 4);
    out
}

/// Replace calls to `from` having exactly `arity` comma-separated top-level
/// arguments with calls to `to` whose final `arity - 3` arguments are dropped.
/// Handles the known WGSL forms without a full parser by tracking nesting.
fn rewrite_calls(source: &str, from: &str, to: &str, arity: usize) -> String {
    let mut result = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(pos) = rest.find(from) {
        // Ensure it is an identifier boundary, not part of a longer name.
        let before_ok = pos == 0
            || !rest[..pos]
                .chars()
                .last()
                .is_some_and(|c| c.is_alphanumeric() || c == '_');
        if !before_ok {
            let end = pos + from.len();
            result.push_str(&rest[..end]);
            rest = &rest[end..];
            continue;
        }
        let after = &rest[pos + from.len()..];
        if !after.starts_with('(') {
            let end = pos + from.len();
            result.push_str(&rest[..end]);
            rest = &rest[end..];
            continue;
        }
        // Find the matching close paren and split top-level arguments.
        let open = pos + from.len();
        let mut depth = 0i32;
        let mut close = None;
        for (i, c) in rest[open..].char_indices() {
            let i = open + i;
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(close) = close else {
            result.push_str(rest);
            rest = "";
            break;
        };
        let args_text = &rest[open + 1..close];
        let args = split_top_level(args_text);
        if args.len() == arity {
            let keep = if arity == 5 { 4 } else { 3 };
            result.push_str(&rest[..pos]);
            result.push_str(to);
            result.push('(');
            result.push_str(&args[..keep].join(", "));
            result.push(')');
        } else {
            result.push_str(&rest[..=close]);
        }
        rest = &rest[close + 1..];
    }
    result.push_str(rest);
    result
}

/// Split on commas at paren/bracket/brace depth zero.
fn split_top_level(text: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (i, c) in text.char_indices() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(text[start..i].trim().to_string());
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(text[start..].trim().to_string());
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrites_scene_sample() {
        let src = "let tex = textureSampleLevel(scene_tex, scene_smp, in.uv, i32(in.page + 0.5), resident);";
        let out = without_explicit_lod(src);
        assert!(out.contains("textureSample(scene_tex, scene_smp, in.uv, i32(in.page + 0.5))"));
        assert!(!out.contains("textureSampleLevel"));
    }

    #[test]
    fn rewrites_probe_sample() {
        let src = "textureSampleLevel(probe_tex0, probe_smp, dir, lod)";
        let out = without_explicit_lod(src);
        assert!(out.contains("textureSample(probe_tex0, probe_smp, dir)"));
    }

    #[test]
    fn preserves_when_arity_differs() {
        let src = "textureSampleLevel(a, b, c)";
        assert!(without_explicit_lod(src).contains("textureSampleLevel"));
    }

    #[test]
    fn rewrites_after_multibyte_text() {
        // The matching-paren scan must use byte offsets even when multibyte
        // characters precede the call (WGSL identifiers/comments can be UTF-8).
        let src = "αβ let tex = textureSampleLevel(scene_tex, smp, uv, page, resident);";
        let out = without_explicit_lod(src);
        assert!(out.contains("textureSample(scene_tex, smp, uv, page)"));
        assert!(!out.contains("textureSampleLevel"));
    }
}
