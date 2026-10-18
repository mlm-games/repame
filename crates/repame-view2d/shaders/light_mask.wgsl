// Shadow mask pass for the repame-view2d lighting subsystem.
//! Consumed by `lightpass.rs`. Each vertex carries a position the CPU has
//! already baked into the mask target's clip space plus the wedge's weight in
//! its own tap's channel, so the pass needs no bind groups at all.
//!
//! One channel per tap, and the pipeline max-blends, so the wedges that share
//! a tap (every edge of a solid occluder, seen under one rotation of the
//! silhouette) union into that channel instead of summing: a point is shadowed
//! for a tap when the ray to the light crosses *any* edge, and adding the
//! overlapped wedges would darken the lit side of the silhouette. The light
//! shader then sums the channels, which is the taps' average: the penumbra.
//! A single-edge occluder has one wedge per tap, so a union and a sum agree
//! there and the open-wall penumbra is unchanged.

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) weights: vec4<f32>,
};

@vertex
fn vs_main(
    @location(0) clip: vec2<f32>,
    @location(1) world: vec2<f32>,
    @location(2) weights: vec4<f32>,
) -> VertexOut {
    var out: VertexOut;
    out.pos = vec4<f32>(clip, 0.0, 1.0);
    out.weights = weights;
    // `world` is unused here; it is declared so the mask and light pipelines
    // can share one vertex buffer layout.
    _ = world;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return in.weights;
}
