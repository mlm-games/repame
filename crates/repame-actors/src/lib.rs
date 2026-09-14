//! Live vector actors: `.ren` rigs played through `renamite-player`.
//! Timelines and state machines drive playback; the Canvas bridge paints.
//! [`actors_view`] is presentational, [`actors_view_with`] takes opts.
//!
//! ```no_run
//! use repame_actors::{actors_view, host_from_str};
//! use repose_core::RenderContext;
//!
//! let host = host_from_str(repame_actors::MINIMAL_REN).unwrap();
//! let ctx = RenderContext::new();
//! let _view = actors_view(host, ctx);
//! ```

use std::cell::RefCell;
use std::rc::Rc;

use glam::DVec2;
use renamite_render_bridge::SceneRenderer;
use repose_canvas::{Canvas, DrawScope};
use repose_core::input::PointerEvent;
use repose_core::{Modifier, RenderContext, Vec2, View, request_frame, theme};

pub use renamite_player::{Player, PlayerError};
pub use renamite_player_ui::{PlayerHost, PlayerHostRef, ViewTransform};

/// Minimal `.ren` document (empty 64x64 composition, no nodes).
/// Fixture helper for hosts built without asset files.
pub const MINIMAL_REN: &str = "RenFile(format_version: 1, meta: Meta(name: \"\", author: \"\", generator: \"\"), document: Document(format_version: 1, compositions: [SerdeSlot(value: None, version: 0), SerdeSlot(value: Some(Composition(name: \"\", size: (64, 64), rate: FrameRate(num: 60, den: 1), range: (Frame(0), Frame(1)), children: [])), version: 1)], nodes: [SerdeSlot(value: None, version: 0)], assets: [SerdeSlot(value: None, version: 0)], main: SerKey(idx: 1, version: 1)))";

/// Host a `.ren` rig from source text. The game ticks the handle per frame
/// and paints it via [`actors_view`] or [`actors_view_with`].
/// Hosts start playing; sim-driven games call `pause` to hold.
pub fn host_from_str(source: &str) -> Result<PlayerHostRef, PlayerError> {
    Ok(Rc::new(RefCell::new(PlayerHost::from_ren_str(source)?)))
}

/// How the artboard maps onto the surface.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ActorFit {
    /// Exact artboard fit each draw: scale from live surface size.
    /// Letterboxes when aspects differ; margins paint no art.
    #[default]
    Exact,
    /// Margin fit (56 px), re-applied on resize so zoom survives redraws.
    Margin,
}

/// Full-control options for [`actors_view_with`].
#[derive(Clone, Copy, Debug)]
pub struct ActorViewOpts {
    /// Artboard mapping (default [`ActorFit::Exact`]).
    pub fit: ActorFit,
    /// Checkerboard backplate behind the art (default off).
    pub chrome: bool,
    /// Scroll-zoom plus pointer forwarding in world coords (default off).
    pub interactive: bool,
    /// Tick playback each draw while playing (default off).
    pub auto_tick: bool,
}

impl Default for ActorViewOpts {
    /// Presentational defaults.
    fn default() -> Self {
        Self {
            fit: ActorFit::Exact,
            chrome: false,
            interactive: false,
            auto_tick: false,
        }
    }
}

impl ActorViewOpts {
    /// Editor defaults: margin fit, chrome, interaction, auto-tick.
    pub fn editor() -> Self {
        Self {
            fit: ActorFit::Margin,
            chrome: true,
            interactive: true,
            auto_tick: true,
        }
    }
}

/// Presentational rig surface with default opts: transparent, exact fit,
/// no input, no auto-tick. The game applies machine inputs and ticks.
pub fn actors_view(host: PlayerHostRef, ctx: RenderContext) -> View {
    actors_view_with(host, ctx, ActorViewOpts::default())
}

/// Rig surface with full control. Canvas size comes from layout
/// (`fill_max_size`); the game sizes and layers the surface.
pub fn actors_view_with(host: PlayerHostRef, ctx: RenderContext, opts: ActorViewOpts) -> View {
    let draw = host.clone();

    let mut modifier = Modifier::new().fill_max_size();
    if opts.chrome {
        modifier = modifier.background(theme().surface_container_lowest);
    }
    if opts.interactive {
        modifier = modifier
            .on_scroll({
                let host = host.clone();
                move |delta: Vec2| {
                    let mut h = host.borrow_mut();
                    let factor = (1.0 + (-delta.y as f64) * 0.002).clamp(0.5, 2.0);
                    let anchor = h.last_pointer;
                    h.zoom_at(anchor, factor);
                    request_frame();
                    Vec2::ZERO
                }
            })
            .on_pointer_down({
                let host = host.clone();
                move |pe: PointerEvent| {
                    let mut h = host.borrow_mut();
                    let p = pe_position(&pe);
                    h.last_pointer = p;
                    let world = h.view.screen_to_world(p);
                    h.player.pointer_down(world);
                    request_frame();
                }
            })
            .on_pointer_up({
                let host = host.clone();
                move |pe: PointerEvent| {
                    let mut h = host.borrow_mut();
                    let world = h.view.screen_to_world(pe_position(&pe));
                    h.player.pointer_up(world);
                    request_frame();
                }
            })
            .on_pointer_move({
                let host = host.clone();
                move |pe: PointerEvent| {
                    let mut h = host.borrow_mut();
                    let p = pe_position(&pe);
                    h.last_pointer = p;
                    let world = h.view.screen_to_world(p);
                    h.player.pointer_move(world);
                    request_frame();
                }
            })
            .on_pointer_leave({
                let host = host.clone();
                move |_pe: PointerEvent| {
                    host.borrow_mut().player.pointer_leave();
                    request_frame();
                }
            });
    } else {
        modifier = modifier.hit_passthrough();
    }

    Canvas(modifier, move |scope| {
        let mut h = draw.borrow_mut();

        if opts.auto_tick && h.tick_playback() {
            request_frame();
        }

        let sw = scope.size.width as f64;
        let sh = scope.size.height as f64;
        if sw <= 1.0 || sh <= 1.0 {
            return;
        }
        // One artboard lookup serves both fit modes and the chrome backplate.
        let art = h.artboard();
        let art_valid = art.x > 0.0 && art.y > 0.0;
        let surface = DVec2::new(sw, sh);
        match opts.fit {
            ActorFit::Exact => {
                if !art_valid {
                    return;
                }
                // Per-draw map with no zoom state; shared upstream helper.
                h.fit_exact(surface);
            }
            ActorFit::Margin => {
                // Margin fit tracks resizes so zoom survives redraws.
                h.fit(surface);
            }
        }
        if h.dirty_images {
            let host = &mut *h;
            host.renderer
                .sync_document_images(&host.player.project.document, &ctx);
            host.dirty_images = false;
        }
        if opts.chrome && art_valid {
            paint_chrome(scope, art, h.view);
        }
        let scene = h.player.scene().clone();
        let view = h.view;
        let prepared = h.renderer.prepare(&scene, &view);
        h.renderer.paint_prepared(&prepared, scope);
    })
}

fn pe_position(pe: &PointerEvent) -> DVec2 {
    DVec2::new(pe.position.x as f64, pe.position.y as f64)
}

/// Checkerboard backplate plus border behind the artboard.
fn paint_chrome(scope: &mut DrawScope, artboard: DVec2, view: ViewTransform) {
    SceneRenderer::paint_artboard_chrome(scope, artboard, &view);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opts_default_is_presentational() {
        let opts = ActorViewOpts::default();
        assert_eq!(opts.fit, ActorFit::Exact);
        assert!(!opts.chrome && !opts.interactive && !opts.auto_tick);
    }

    #[test]
    fn opts_editor_is_full_control() {
        let opts = ActorViewOpts::editor();
        assert_eq!(opts.fit, ActorFit::Margin);
        assert!(opts.chrome && opts.interactive && opts.auto_tick);
    }

    #[test]
    fn host_parses_minimal_document() {
        let host = host_from_str(MINIMAL_REN).expect("minimal doc must parse");
        // New hosts upload images on first draw.
        assert!(host.borrow().dirty_images);
    }

    #[test]
    fn views_construct_without_a_backend() {
        // Construction is data only; paint closures run later on the render thread.
        let ctx = RenderContext::new();
        let _presentational = actors_view(host_from_str(MINIMAL_REN).unwrap(), ctx.clone());
        let _full = actors_view_with(
            host_from_str(MINIMAL_REN).unwrap(),
            ctx.clone(),
            ActorViewOpts::editor(),
        );
        let _mixed = actors_view_with(
            host_from_str(MINIMAL_REN).unwrap(),
            ctx,
            ActorViewOpts {
                interactive: true,
                ..ActorViewOpts::default()
            },
        );
    }
}
