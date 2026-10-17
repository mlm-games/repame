//! Shared game UI kit for the Repame template line: one action queue, one run
//! flow, focus-navigable controls, one palette and one touch overlay.
//!
//! The kit exists because a measurement across the template-derived games found
//! the same four things in every `ui/` module: eleven files define the same
//! `mix`/`lighten`, eight to ten define the same `rect`/`chunky`/`label`
//! shapes, thirteen files across nine games — four of them outside the template
//! line — hand-roll the same `Sink`/`UiActions`/`drain_actions` action queue,
//! and eight ship the same virtual stick. Nothing in those games can be
//! navigated with a keyboard or a gamepad, because the overlay root asks for
//! focus for itself and the buttons are boxes with a click handler.
//!
//! ## What the kit owns
//!
//! * [`queue`]: [`ActionQueue`]/[`Sink`] to push from a click closure and drain
//!   from the sim, plus [`Intents`], the ordered edge view of a drained batch
//!   that replaces a per-action boolean struct.
//! * [`flow`]: [`Flow`], [`Transition`] and [`Suspension`] as
//!   `repame_sim::state` state sets with enter/exit/transition schedules, plus
//!   [`FlowState`], the playing / modal / transitional / suspended predicates
//!   every game's `Phase` and `RunState` re-derive.
//! * [`nav`]: focus traversal over the kit's own controls, built on repose's
//!   `spatial_focus_next` and `FocusRequester`, with an entry point, focus
//!   groups, boundary wrap and analog-stick repeat. This is the part the
//!   template games are missing entirely.
//! * [`palette`] and [`kit`]: palette tokens and the button/chip/panel/label
//!   voices every game redraws, with the colour blending in one place.
//! * [`touch`]: the virtual stick and tap buttons as kit controls, driving the
//!   same queue and the same `apply_stick` curve as the pads.
//!
//! ## Adoption
//!
//! Nothing here has to be an ECS resource, and the crate deliberately carries no
//! `bevy_ecs` dependency. A game keeps the wrapper resource it already declares
//! and puts the kit's handle inside it:
//!
//! ```ignore
//! #[derive(Resource, Clone, Default)]
//! pub struct UiActions(pub repame_ui::Sink<UiAction>);
//!
//! #[derive(Resource, Default)]
//! pub struct UiIntents(pub repame_ui::Intents<UiAction>);
//! ```
//!
//! The frame then drains once, and hands the same sink to the kit:
//!
//! ```ignore
//! let mut ui = self.sim.world.resource_mut::<UiActions>();
//! let mut intents = self.sim.world.resource_mut::<UiIntents>();
//! repame_ui::absorb(&ui.0, &mut intents.0);
//!
//! let kit = repame_ui::Kit::new(scale, repame_ui::Tokens::default(), ui.0.clone());
//! kit.begin_frame();
//! sim.add_chained_systems(update.run_if(repame_ui::flow::run_live));
//! ```
//!
//! What stays in the game: its action enum, its panel list, its art colours and
//! its font registration. What the kit removes: the copies of the queue, the
//! flow predicates, the palette blending, the shapes, the stick, and the reason
//! the menus could not be navigated.

pub mod flow;
pub mod kit;
pub mod layout;
pub mod nav;
pub mod palette;
pub mod queue;
pub mod touch;

pub use flow::{Flow, FlowState, InputOwner, Suspension, Transition, panel_owns_input, run_live};
pub use kit::{Button, Kit, overlay_root};
pub use layout::{
    DESIGN, Scale, ScaleError, anchor_edges, anchor_rect, design_units, disc, ring,
};
pub use nav::{
    GroupId, NavDir, NavDriver, NavEntry, NavMap, NavStep, StickNav, Wrap, chain_of, dpad_dirs,
    pad_confirm,
};
pub use palette::{
    FONT, ICON_BUTTON, ICON_GLYPH, MATERIAL_SYMBOLS, Tokens, chip, chunky, darken, hand_cut,
    lighten, mix, state_colors,
};
pub use queue::{ActionQueue, Intents, Sink, UiAction, absorb};
pub use touch::{STICK_RADIUS, TouchError, TouchPad};
