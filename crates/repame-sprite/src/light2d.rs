//! 2D point lights and shadow occluders for the GPU viewport.
//!
//! Plain snapshot data: the game rebuilds [`Light2d`] / [`Occluder2d`]
//! next to [`FrameInput`](crate::FrameInput) and the renderer turns each
//! shadowed light into an angular (polar) occluder map on the CPU, then
//! shades the composited frame with an analytic radial falloff. Lighting
//! is GPU-only: the canvas [`Viewport2d`](crate::Viewport2d) path reads
//! none of this (same limitation as [`FrameInput::chroma`]).

use glam::Vec2;

/// Lights shaded per viewport. Extra entries in
/// [`FrameInput::lights`](crate::FrameInput::lights) are ignored.
pub(crate) const MAX_LIGHTS: usize = 4;

/// Floor for a shadowed light's angular map after budget clamping.
pub(crate) const MIN_SHADOW_BINS: u32 = 128;

/// Total angular bins across all shadowed lights of one viewport:
/// 4096 bins x 8 bytes = 32 KiB of uniform data, inside the 64 KiB
/// `maxUniformBufferBindingSize` binding cap.
pub(crate) const MAX_TOTAL_SHADOW_BINS: usize = 4096;

const TAU: f32 = std::f32::consts::TAU;

/// One stop of a light's analytic radial ramp. Stops must ascend in
/// [`at`](LightStop::at) over `0..=1`; the shader ramps linearly between
/// them (no gradient texture).
#[derive(Clone, Copy, Debug, Default)]
pub struct LightStop {
    /// Normalized distance from the light centre, ascending in `0..=1`.
    pub at: f32,
    /// RGBA ramp value at `at`: RGB scales the light colour, A scales
    /// the ramp.
    pub color: [f32; 4],
}

/// GPU point light (Godot `PointLight2D` analog) in world space.
///
/// Shading contract: `scene.rgb * clamp(ambient + Σ(color * energy *
/// ramp(d / radius) * shadow), 0, 1)`, where `ramp` interpolates
/// [`falloff`](Light2d::falloff) and `shadow` is `1.0` when
/// [`shadows`](Light2d::shadows) is off or `d >= radius`. The GPU
/// shades at most the first four lights of a frame.
#[derive(Clone, Copy, Debug)]
pub struct Light2d {
    /// World-space light centre.
    pub position: Vec2,
    /// World-unit falloff radius. Non-positive disables the light.
    pub radius: f32,
    /// Peak light colour (RGB).
    pub color: [f32; 3],
    /// Brightness multiplier applied to the ramped colour.
    pub energy: f32,
    /// Three ascending stops of the radial ramp (no texture).
    pub falloff: [LightStop; 3],
    /// Casts shadows from [`FrameInput::occluders`](crate::FrameInput::occluders).
    pub shadows: bool,
    /// Penumbra half-width in angular bins: `0` is a hard edge, `2` is
    /// the five-tap (PCF5) width, clamped to `8.0` at upload. The
    /// world-space penumbra widens with distance from the light for
    /// free.
    pub shadow_softness: f32,
    /// Requested angular map resolution, clamped to the uniform budget
    /// (see [`MAX_TOTAL_SHADOW_BINS`]) at prepare time.
    pub shadow_bins: u32,
}

impl Default for Light2d {
    fn default() -> Self {
        Self {
            position: Vec2::ZERO,
            radius: 1.0,
            color: [1.0, 1.0, 1.0],
            energy: 1.0,
            falloff: [
                LightStop {
                    at: 0.0,
                    color: [1.0, 1.0, 1.0, 1.0],
                },
                LightStop {
                    at: 0.5,
                    color: [0.5, 0.5, 0.5, 1.0],
                },
                LightStop {
                    at: 1.0,
                    color: [0.0, 0.0, 0.0, 1.0],
                },
            ],
            shadows: false,
            shadow_softness: 2.0,
            shadow_bins: 1024,
        }
    }
}

/// Shadow-casting rectangle (Godot `LightOccluder2D` analog).
///
/// Corners satisfy `min <= max`, as built by [`rect`](Self::rect) and
/// [`cell`](Self::cell).
#[derive(Clone, Copy, Debug)]
pub struct Occluder2d {
    /// Inclusive lower corner in world units.
    pub min: Vec2,
    /// Inclusive upper corner in world units.
    pub max: Vec2,
}

impl Occluder2d {
    /// Occluder spanning the two corners (any corner order).
    pub fn rect(min: Vec2, max: Vec2) -> Self {
        Self {
            min: min.min(max),
            max: min.max(max),
        }
    }

    /// One level cell: `[col, row]` index at `cell_size` world units.
    pub fn cell(cell: [i32; 2], cell_size: f32) -> Self {
        let min = Vec2::new(cell[0] as f32 * cell_size, cell[1] as f32 * cell_size);
        Self {
            min,
            max: min + Vec2::splat(cell_size),
        }
    }

    /// Inclusive point-in-rect test.
    pub fn contains(&self, p: Vec2) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
    }
}

/// Min-hit distance per angular bin, normalized `0..1` of
/// `light.radius`: `[distance, hit]`, where `hit` is `1.0` when the bin
/// ray reaches an occluder inside the radius and `0.0` otherwise
/// (`distance` is then `1.0`). Bin `i` covers the centre angle
/// `(i + 0.5) / bins * 2π`, the same convention the shader samples.
///
/// `out` is cleared and resized to `light.shadow_bins` (`0` bins yields
/// an empty map).
pub fn angular_shadow_map(light: &Light2d, occluders: &[Occluder2d], out: &mut Vec<[f32; 2]>) {
    out.clear();
    let bins = light.shadow_bins as usize;
    out.resize(bins, [1.0, 0.0]);
    let radius = light.radius;
    if bins == 0 || occluders.is_empty() || !(radius.is_finite() && radius > 0.0) {
        return;
    }
    let origin = light.position;
    for occ in occluders {
        if !circle_touches_rect(origin, radius, *occ) {
            continue;
        }
        for (i, cell) in out.iter_mut().enumerate() {
            let ang = ((i as f32 + 0.5) / bins as f32) * TAU;
            let dir = Vec2::new(ang.cos(), ang.sin());
            let Some(t) = ray_rect_min(origin, dir, radius, *occ) else {
                continue;
            };
            let hit = (t / radius).clamp(0.0, 1.0);
            if cell[1] < 0.5 || hit < cell[0] {
                *cell = [hit, 1.0];
            }
        }
    }
}

/// Effective per-light bin counts for one frame, indexed like the
/// lights they belong to (`0` = no map). Clamps every shadowed light to
/// an even count of at least [`MIN_SHADOW_BINS`] within
/// [`MAX_TOTAL_SHADOW_BINS`], split evenly across the shadowed lights,
/// and warns when a request is reduced. Lights past [`MAX_LIGHTS`] are
/// not budgeted (the GPU does not shade them).
pub(crate) fn resolve_shadow_bins(lights: &[Light2d]) -> Vec<u32> {
    let n = lights.len().min(MAX_LIGHTS);
    let shadowed = lights[..n].iter().filter(|l| l.shadows).count();
    if shadowed == 0 {
        return vec![0; n];
    }
    let cap = ((MAX_TOTAL_SHADOW_BINS / shadowed) as u32) & !1;
    let mut out = Vec::with_capacity(n);
    for l in &lights[..n] {
        if !l.shadows {
            out.push(0);
            continue;
        }
        let want = l.shadow_bins.max(MIN_SHADOW_BINS).min(cap) & !1;
        if want < l.shadow_bins {
            log::warn!(
                "light2d: shadow_bins {} clamped to {} ({} shadowed lights, budget {} bins)",
                l.shadow_bins,
                want,
                shadowed,
                MAX_TOTAL_SHADOW_BINS
            );
        }
        out.push(want);
    }
    out
}

/// Circle-vs-AABB cull: everything the light radius could reach.
fn circle_touches_rect(c: Vec2, radius: f32, occ: Occluder2d) -> bool {
    let nx = c.x.min(occ.max.x).max(occ.min.x);
    let ny = c.y.min(occ.max.y).max(occ.min.y);
    let dx = c.x - nx;
    let dy = c.y - ny;
    dx * dx + dy * dy <= radius * radius
}

/// Nearest forward ray/AABB hit at or before `max_t`, else `None`.
fn ray_rect_min(origin: Vec2, dir: Vec2, max_t: f32, occ: Occluder2d) -> Option<f32> {
    let o = [origin.x, origin.y];
    let d = [dir.x, dir.y];
    let lo = [occ.min.x.min(occ.max.x), occ.min.y.min(occ.max.y)];
    let hi = [occ.min.x.max(occ.max.x), occ.min.y.max(occ.max.y)];
    let mut tmin = 0.0f32;
    let mut tmax = max_t;
    for a in 0..2 {
        if d[a].abs() <= 1e-9 {
            if o[a] < lo[a] || o[a] > hi[a] {
                return None;
            }
            continue;
        }
        let inv = 1.0 / d[a];
        let (mut t0, mut t1) = ((lo[a] - o[a]) * inv, (hi[a] - o[a]) * inv);
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }
        tmin = tmin.max(t0);
        tmax = tmax.min(t1);
        if tmin > tmax {
            return None;
        }
    }
    Some(tmin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angular_shadow_map_marks_the_covered_sector() {
        let light = Light2d {
            position: Vec2::ZERO,
            radius: 10.0,
            shadows: true,
            shadow_bins: 64,
            ..Default::default()
        };
        let wall = Occluder2d::rect(Vec2::new(2.0, -2.0), Vec2::new(5.0, 2.0));
        let mut out = vec![[9.0, 9.0]; 3];
        angular_shadow_map(&light, &[wall], &mut out);
        assert_eq!(out.len(), 64, "out is resized to shadow_bins");
        // The wall spans +-45 deg as seen from the light; bin centres are
        // (i + 0.5) * 5.625 deg, so 0..=7 and 56..=63 are shadowed.
        for (i, cell) in out.iter().enumerate() {
            let covered = !(8..56).contains(&i);
            assert_eq!(cell[1] > 0.5, covered, "bin {i} hit flag, {cell:?}");
        }
        // Uncovered bins stay fully open.
        assert!(out[8..56].iter().all(|c| *c == [1.0, 0.0]));
        // Covered bins hit the near face at x = 2: distance 2 / cos(angle).
        let expect = |i: usize| {
            let ang = (i as f32 + 0.5) * (TAU / 64.0);
            2.0 / ang.cos() / light.radius
        };
        for i in [0usize, 7, 56, 63] {
            assert!(
                (out[i][0] - expect(i)).abs() < 1e-4,
                "bin {i} distance {} vs {}",
                out[i][0],
                expect(i)
            );
        }
        // No occluders: nothing shadows anything.
        angular_shadow_map(&light, &[], &mut out);
        assert!(out.iter().all(|c| *c == [1.0, 0.0]));
    }
}
