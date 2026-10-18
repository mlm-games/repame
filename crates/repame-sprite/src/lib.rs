//! Deprecated: superseded by [`repame_view2d`], which merges this crate with
//! `repame-actors` into the one 2D viewport crate. Deleted at the next
//! breaking bump; import `repame-view2d` in new code.
//!
//! The re-export below keeps every path that lived here resolving, module
//! paths included (`repame_sprite::batch::…`, `repame_sprite::design::…`,
//! `repame_sprite::fullscreen::…`), so a consumer can move at its own pace.

pub use repame_view2d::*;
