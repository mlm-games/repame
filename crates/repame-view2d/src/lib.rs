//! 2D viewport: snapshot in, pixels out.
//!
//! The whole 2D screen path of a game lives here, plain sprites included: a
//! game that draws sprite quads over an atlas needs nothing else, and uses
//! [`SpriteInstance`] rows through the instanced [`SpriteBatch`], framed by a
//! dp-space [`Camera2d`] ([`FitMode`]) whose fit law the canvas, the GPU batch
//! and the CPU picks all read from the same functions ([`effective_fit`],
//! [`fit_view_proj`], [`world_to_dp`], [`dp_to_world`]).
//!
//! Around that core:
//!
//! * [`RasterAtlas`] - raster sheets drawn through the same batch, and the
//!   world-anchored [`ActorDrawList`] that puts them, `.ren` vector rigs and
//!   solid rects in one z order.
//! * [`WorldText`] - world-anchored text floaters, and [`design`] - the design
//!   box onto dp canvas mapping games would otherwise hand-roll.
//! * [`PickEvent`] - presses, hovers and the slop rule behind a click, plus
//!   [`CursorFrame`]s for a hardware cursor.
//! * [`fullscreen`] - the fullscreen post pass hook, driven by a WGSL shader
//!   and per-pass uniforms/textures.
//! * [`actors_view`] - `.ren` rigs played through `renamite-player`, one rig
//!   filling one presented surface.
//!
//! Unit discipline lives here, once: games work in world/dp units and do not
//! touch physical px.

pub mod actors;
pub use actors::{
    ActorFit, ActorViewOpts, MINIMAL_REN, actors_view, actors_view_with, host_from_bytes,
    host_from_str,
};
pub use renamite_player::{Player, PlayerError};
pub use renamite_player_ui::{PlayerHost, PlayerHostRef, ViewTransform};

pub mod batch;
pub use batch::{
    AtlasUpload, BatchDesc, BatchFrame, SpriteBatch, TextureFilter, draw_batch, draw_batch_with_id,
    frame_uv, instance_rows, screen_camera, sprite_aabb,
};

pub mod camera;
pub use camera::{
    Camera2d, FitMode, contain_fit, dp_to_world, dp_to_world_with_roll, effective_fit,
    fit_view_proj, fit_view_proj_with_roll, unproject_px, world_to_dp, world_to_dp_with_roll,
};

pub mod cursor;
pub use cursor::{CursorFrame, cursor_frame};

pub mod design;
pub use design::{DesignFit, DesignMapping, cover_fit};

pub mod draw;
pub use draw::{
    ActorDraw, ActorDrawList, ActorId, ActorItem, ActorLibrary, RasterAtlas, actors_view_mixed,
    actors_view_mixed_with_events, actors_view_world, actors_view_world_with_events,
};

pub mod fullscreen;
pub use fullscreen::{
    FullscreenDesc, FullscreenPass, FullscreenTexture, FullscreenTextureRef,
    FullscreenTextureUpload,
};

pub mod pick;
pub use pick::{CLICK_SLOP_DP, PickEvent, click_within_slop_dp};

pub mod sprite;
pub use sprite::{SpriteBlend, SpriteInstance, WorldText, rgba8};
