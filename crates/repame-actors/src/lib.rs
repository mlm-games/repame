//! Live vector actors: `.ren` rigs played through `renamite-player`.
//!
//! Actors animate via timelines + state machines (walk/attack/die),
//! driven by the game through host overrides and machine inputs, and
//! painted through the repose Canvas bridge. This replaces the bake-to-
//! sprite pipeline (a bevy-shell workaround) with live playback; `bake`
//! stays as the perf fallback and conformance reference.
//!
//! Repose types cross the renamite boundary here (`View`, `RenderContext`
//! inside player-ui), which proves the version alignment: a single repose
//! source graph-wide.
//!
//! Two embeddings, one data path:
//! - [`actors_view`]: presentational (rozvp zombie semantics).
//!   Transparent surface, exact artboard fit, no input, no auto-tick.
//!   Playback is ticked game-side (e.g. from sim markers) so pause
//!   freezes and there is no double-tick speedup.
//! - [`actors_view_with`]: full control via [`ActorViewOpts`]. Editor
//!   chrome (checkerboard backplate), margin fit with resize refit,
//!   scroll-zoom + pointer forwarding, and per-frame auto-tick.
//!
//! Positioning, mirroring, and layering stay game-side composition
//! (e.g. `repame_sprite::ActorFrame` + `ZStack`): this crate paints one
//! rig into one surface. See rozvp's `pilot/views.rs::rigs_layer` for
//! the entity-keyed multi-actor pattern (kept there; rozvp is frozen).
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

/// Minimal valid `.ren` document (empty 64x64 composition, no nodes).
/// Test/fixture helper so hosts can be built without asset files.
pub const MINIMAL_REN: &str = "RenFile(format_version: 1, meta: Meta(name: \"\", author: \"\", generator: \"\"), document: Document(format_version: 1, compositions: [SerdeSlot(value: None, version: 0), SerdeSlot(value: Some(Composition(name: \"\", size: (64, 64), rate: FrameRate(num: 60, den: 1), range: (Frame(0), Frame(1)), children: [])), version: 1)], nodes: [SerdeSlot(value: None, version: 0)], assets: [SerdeSlot(value: None, version: 0)], main: SerKey(idx: 1, version: 1)))";

/// Host a `.ren` rig from source text. The returned handle owns engine +
/// tessellator + playback state; the game ticks it per frame and mounts
/// [`actors_view`] (or [`actors_view_with`]) to paint it.
///
/// Hosts start `playing`; sim-driven games that tick manually call
/// `host.borrow_mut().pause()` to hold, or gate their tick on pause.
pub fn host_from_str(source: &str) -> Result<PlayerHostRef, PlayerError> {
    Ok(Rc::new(RefCell::new(PlayerHost::from_ren_str(source)?)))
}

/// How the artboard maps onto the surface.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ActorFit {
    /// Exact-fit the artboard every draw (rozvp zombie semantics): no
    /// editor margin, scale recomputed from the live surface size, so a
    /// 64x80 surface over a 256x320 artboard yields exactly 0.25.
    /// Letterboxes inside the surface when aspects differ: the empty
    /// margins still belong to the surface for layout/hit purposes but
    /// paint no art (presentational OK; interactive editors want Margin).
    #[default]
    Exact,
    /// Upstream margin fit (`PlayerHost::fit`, 56 px margin), re-applied
    /// only when the surface resizes so interactive zoom survives
    /// redraws.
    Margin,
}

/// Full-control options for [`actors_view_with`].
#[derive(Clone, Copy, Debug)]
pub struct ActorViewOpts {
    /// Artboard mapping (default [`ActorFit::Exact`]).
    pub fit: ActorFit,
    /// Checkerboard backplate + border behind the art (default off;
    /// presentational surfaces stay transparent).
    pub chrome: bool,
    /// Scroll-zoom plus pointer down/move/up/leave forwarding into the
    /// rig's machine listeners, in world coordinates (default off;
    /// presentational surfaces are `hit_passthrough`).
    pub interactive: bool,
    /// Tick playback every draw while playing, requesting follow-up
    /// frames (default off; sim-driven games tick from their own
    /// schedule so pause freezes).
    pub auto_tick: bool,
}

impl Default for ActorViewOpts {
    /// Presentational defaults (rozvp semantics).
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
    /// Editor-style defaults: margin fit, chrome, interaction, auto-tick.
    /// (For a pure editor embed, upstream
    /// `renamite_player_ui::RenamitePlayer` is the same shape.)
    pub fn editor() -> Self {
        Self {
            fit: ActorFit::Margin,
            chrome: true,
            interactive: true,
            auto_tick: true,
        }
    }
}

/// Presentational rig surface with [`ActorViewOpts::default`]: transparent,
/// exact-fit, no input, no auto-tick. Machine inputs (`set_bool`,
/// triggers) and playback ticking are applied by game code.
pub fn actors_view(host: PlayerHostRef, ctx: RenderContext) -> View {
    actors_view_with(host, ctx, ActorViewOpts::default())
}

/// Rig surface with full control. See [`ActorViewOpts`] for the matrix.
///
/// Canvas size comes from layout (`fill_max_size`); the game sizes and
/// positions the surface (and mirrors, tints, layers it) around this
/// call.
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
        // Artboard is needed by both fit modes (Margin uses it inside
        // `fit`, chrome needs it for the backplate): bail once here so
        // neither paints degenerate geometry on an empty composition.
        let art = h.artboard();
        let art_valid = art.x > 0.0 && art.y > 0.0;
        let surface = DVec2::new(sw, sh);
        match opts.fit {
            ActorFit::Exact => {
                if !art_valid {
                    return;
                }
                // Exact-fit is a per-draw map (no zoom state): delegate the
                // math to the shared upstream helper.
                h.fit_exact(surface);
            }
            ActorFit::Margin => {
                // Upstream margin fit with resize tracking, so interactive
                // zoom survives redraws.
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

/// Checkerboard backplate + border behind the artboard (editor chrome).
/// Shared upstream paint (`SceneRenderer::paint_artboard_chrome`), so the
/// editor, the player embed, and game actor surfaces all match.
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
        // Fresh host paints (dirty images upload on first draw).
        assert!(host.borrow().dirty_images);
    }

    #[test]
    fn views_construct_without_a_backend() {
        // View construction is pure data (paint closures run later on
        // the render thread), so all four corners build headless.
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
