//! The virtual stick and tap buttons, as kit controls.
//!
//! Eight games ship the same `ui/touch.rs`: a left-side pointer zone that
//! becomes a stick origin on press, a drag that moves a knob, and a release
//! that zeroes it — plus, in the games that have them, round tap buttons built
//! from `RBox` and `on_click` that push a game-local action. They are all wired
//! to the same pointer path (repose's `on_pointer_*` callbacks on a hit region)
//! and all gated on `paused || modal`, each with its own copy of that condition
//! and of the stick arithmetic.
//!
//! [`TouchPad`] is that once. Two differences from the copies:
//!
//! * The gate is [`FlowState::game_has_input`], the same predicate the sim's
//!   gameplay systems read, so the overlay cannot be live while the sim thinks
//!   the game is frozen (or the reverse).
//! * The stick's output runs through `repame_input::apply_stick`, the dead zone
//!   the pads use, so a touch stick and a gamepad stick mean the same thing to
//!   gameplay. The copies wrote a raw `delta / radius`, which drifts at rest.
//!
//! Tap buttons push into the kit's action sink, so they arrive in the same
//! queue as every other control. The stick is a *polled* value
//! ([`TouchPad::stick`]): movement is continuous, not an edge, and every copy
//! polls it per tick.

use std::cell::RefCell;
use std::rc::Rc;

use glam::Vec2;
use repose_core::prelude::{Modifier, View};
use repose_core::input::{PointerEvent, PointerKind};
use repose_core::{AlignItems, Dp, JustifyContent};
use repose_ui::{Box as RBox, Spacer, ViewExt as _, ZStack};

use crate::flow::FlowState;
use crate::kit::Kit;
use crate::nav::NavDriver;
use crate::palette::Tokens;
use crate::queue::UiAction;

/// Radius of the stick's travel, in design units.
pub const STICK_RADIUS: f32 = 46.0;

/// Fraction of the viewport width the stick's touch zone covers.
pub const STICK_ZONE: f32 = 0.45;

/// Knob radius in design units.
const KNOB_RADIUS: f32 = 16.0;

/// Why a touch pad setting was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TouchError {
    /// The stick zone must cover a positive fraction of the viewport width, and
    /// no more than all of it.
    ZoneOutOfRange,
}

impl std::fmt::Display for TouchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TouchError::ZoneOutOfRange => {
                f.write_str("stick zone must be a fraction in 0..=1 of the viewport width")
            }
        }
    }
}

impl std::error::Error for TouchError {}

/// Re-anchor a drag so the knob stays within `radius` of the stick's origin.
///
/// Returns the origin to use and the knob's offset from it. Moving past the
/// radius drags the origin along, which is what the copies do: the stick follows
/// the finger instead of pinning to where the touch started.
pub fn drag(origin: Vec2, here: Vec2, radius: f32) -> (Vec2, Vec2) {
    if !radius.is_finite() || radius <= 0.0 {
        return (origin, Vec2::ZERO);
    }
    let delta = here - origin;
    if delta.length() <= radius {
        return (origin, delta);
    }
    let direction = delta.normalize_or_zero();
    (here - direction * radius, direction * radius)
}

/// The stick's output for a knob offset: dead-zoned through the pads' curve and
/// clamped to the unit disc.
///
/// A non-positive radius reads as rest rather than scaling to infinity.
pub fn axis(delta: Vec2, radius: f32) -> Vec2 {
    if !radius.is_finite() || radius <= 0.0 {
        return Vec2::ZERO;
    }
    repame_input::apply_stick(delta / radius)
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Owner {
    kind: PointerKind,
    id: u64,
}

fn owner_of(event: &PointerEvent) -> Owner {
    Owner {
        kind: event.kind,
        id: event.id.0,
    }
}

struct PadState {
    enabled: bool,
    scale: f32,
    zone: f32,
    owner: Option<Owner>,
    active: bool,
    origin: Vec2,
    raw: Vec2,
}

impl PadState {
    fn release(&mut self) {
        self.active = false;
        self.owner = None;
        self.origin = Vec2::ZERO;
        self.raw = Vec2::ZERO;
    }
}

/// The virtual stick and its tap buttons, pushing into the kit's action sink.
///
/// Owned by the app for the life of the game, like the games' own touch state:
/// the pad holds the finger that is down, its origin and the gate.
#[derive(Clone)]
pub struct TouchPad<A: UiAction> {
    kit: Kit<A>,
    inner: Rc<RefCell<PadState>>,
}

impl<A: UiAction> TouchPad<A> {
    /// A pad sharing `kit`'s sink, palette and scale.
    pub fn new(kit: Kit<A>) -> Self {
        Self {
            kit,
            inner: Rc::new(RefCell::new(PadState {
                enabled: false,
                scale: 1.0,
                zone: STICK_ZONE,
                owner: None,
                active: false,
                origin: Vec2::ZERO,
                raw: Vec2::ZERO,
            })),
        }
    }

    /// Cover a different fraction of the viewport width with the stick's zone.
    pub fn with_zone(self, fraction: f32) -> Result<Self, TouchError> {
        if !fraction.is_finite() || fraction <= 0.0 || fraction > 1.0 {
            return Err(TouchError::ZoneOutOfRange);
        }
        self.inner.borrow_mut().zone = fraction;
        Ok(self)
    }

    pub fn kit(&self) -> &Kit<A> {
        &self.kit
    }

    /// Whether the pad is taking pointers this frame.
    pub fn enabled(&self) -> bool {
        self.inner.borrow().enabled
    }

    /// Whether a finger is on the stick.
    pub fn active(&self) -> bool {
        self.inner.borrow().active
    }

    /// The stick's value: dead-zoned, clamped to the unit disc, y down.
    ///
    /// Poll it where the game already combines its move axes; it is not an
    /// action because movement is a level, not an edge.
    pub fn stick(&self) -> Vec2 {
        let inner = self.inner.borrow();
        axis(inner.raw, STICK_RADIUS)
    }

    /// Per-frame gate: the pad takes pointers only while the game owns input.
    ///
    /// Call before composing (or before the sim step) with the same
    /// [`FlowState`] the gameplay systems read. Closing the gate releases a
    /// finger that is still down, so the game cannot keep moving under a modal.
    pub fn begin_frame(&self, flow: FlowState) {
        let mut inner = self.inner.borrow_mut();
        inner.enabled = flow.game_has_input();
        if !inner.enabled {
            inner.release();
        }
    }

    /// Per-frame gate read straight from the sim world.
    ///
    /// A world where the flow was never installed reads as
    /// [`FlowState::CLOSED`], so the overlay is closed rather than live.
    pub fn begin_frame_in(&self, world: &repame_sim::bevy_ecs::prelude::World) {
        self.begin_frame(FlowState::from_world(world).unwrap_or(FlowState::CLOSED));
    }

    /// Forget the stick without touching the gate.
    pub fn release(&self) {
        self.inner.borrow_mut().release();
    }

    /// The stick's touch zone and ring, to be composed over the game.
    ///
    /// Nothing is composed while the gate is closed.
    pub fn view(&self, viewport: [f32; 2]) -> View {
        {
            let mut inner = self.inner.borrow_mut();
            if !inner.enabled {
                return Spacer();
            }
            // The pointer callbacks convert physical pixels with this scale;
            // layout reports the frame's fit scale.
            inner.scale = self.kit.scale().fit();
        }

        let tokens = self.kit.tokens();
        let zone_fraction = self.inner.borrow().zone;
        let zone = RBox(
            self.kit
                .place(viewport, [0.0, 0.0, zone_fraction, 1.0], [0.0; 4])
                .on_pointer_down(self.on_down())
                .on_pointer_move(self.on_move())
                .on_pointer_up(self.on_release())
                .on_pointer_cancel(self.on_release()),
        )
        .child(Spacer());
        ZStack(Modifier::new().fill_max_size()).child((zone, self.ring(viewport, &tokens)))
    }

    /// A round tap button at a design-space point, pushing `action` on release.
    ///
    /// Sized like the glyph buttons and placed absolutely, so it can sit under
    /// the thumb. Touch buttons stay out of keyboard traversal: a finger is not
    /// a focus.
    pub fn tap(
        &self,
        viewport: [f32; 2],
        glyph: &str,
        at: [f32; 2],
        radius: f32,
        action: A,
    ) -> View {
        let scale = self.kit.scale().fit();
        let tokens = self.kit.tokens();
        let enabled = self.inner.borrow().enabled;
        let modifier = self
            .kit
            .place(
                viewport,
                [0.0; 4],
                [
                    at[0] - radius,
                    at[1] - radius,
                    at[0] + radius,
                    at[1] + radius,
                ],
            )
            .background(tokens.cream)
            .border(Dp(3.0 * scale), tokens.ink, Dp(radius * scale))
            .clip_rounded(Dp(radius * scale))
            .shadow_with_color(Dp(0.0), Dp(4.0 * scale), tokens.button_shadow)
            .align_items(AlignItems::CENTER)
            .justify_content(JustifyContent::CENTER)
            .focusable(false);
        let modifier = if enabled {
            let sink = self.kit.sink().clone();
            modifier.on_click(move || sink.push(action.clone()))
        } else {
            modifier.disabled()
        };
        RBox(modifier).child(self.kit.icon(glyph, 26.0, tokens.ink))
    }

    /// The traversal driver for the pad's controls, when a game wants a panel's
    /// buttons reachable from a stick as well as from a finger: a touch stick is
    /// a pointer, so this is only for a game that also reads it as a
    /// directional axis.
    pub fn driver(&self) -> NavDriver {
        self.kit.driver()
    }

    fn on_down(&self) -> impl Fn(PointerEvent) + 'static {
        let inner = self.inner.clone();
        move |event: PointerEvent| {
            let mut state = inner.borrow_mut();
            if !state.enabled || state.owner.is_some() {
                return;
            }
            state.owner = Some(owner_of(&event));
            state.active = true;
            state.raw = Vec2::ZERO;
            state.origin = design_units(event.window_position, state.scale);
            event.consume();
        }
    }

    fn on_move(&self) -> impl Fn(PointerEvent) + 'static {
        let inner = self.inner.clone();
        move |event: PointerEvent| {
            let mut state = inner.borrow_mut();
            if !state.enabled || state.owner != Some(owner_of(&event)) {
                return;
            }
            let here = design_units(event.window_position, state.scale);
            let (origin, delta) = drag(state.origin, here, STICK_RADIUS);
            state.origin = origin;
            state.raw = delta;
            event.consume();
        }
    }

    fn on_release(&self) -> impl Fn(PointerEvent) + 'static {
        let inner = self.inner.clone();
        move |event: PointerEvent| {
            let mut state = inner.borrow_mut();
            if state.owner != Some(owner_of(&event)) {
                return;
            }
            state.release();
        }
    }

    fn ring(&self, viewport: [f32; 2], tokens: &Tokens) -> View {
        let inner = self.inner.borrow();
        if !inner.active {
            return Spacer();
        }
        let origin = inner.origin.to_array();
        let knob = (inner.origin + inner.raw).to_array();
        let vouchers = [
            self.kit.disc(viewport, origin, STICK_RADIUS + 13.0, tokens.touch_shadow),
            self.kit.disc(viewport, origin, STICK_RADIUS + 3.0, tokens.touch_bg),
            self.kit.ring(viewport, origin, STICK_RADIUS + 3.0, 3.0, tokens.touch_ring),
            self.kit.disc(viewport, origin, 4.0, tokens.touch_center),
            self.kit.disc(viewport, knob, KNOB_RADIUS - 3.0, tokens.touch_knob_dark),
            self.kit.disc(viewport, knob, KNOB_RADIUS - 6.0, tokens.touch_knob),
        ];
        ZStack(Modifier::new().fill_max_size()).child(vouchers.to_vec())
    }
}

/// Pointer position in physical pixels to design units.
fn design_units(pixels: repose_core::Vec2, scale: f32) -> Vec2 {
    crate::layout::design_units(pixels, scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::{Flow, Suspension, Transition};
    use crate::layout::Scale;
    use crate::nav::Wrap;
    use crate::palette::Tokens;
    use crate::queue::ActionQueue;
    use repame_input::{GamepadState, TouchContact};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Act {
        Jump,
        Dive,
    }

    fn pad() -> (TouchPad<Act>, ActionQueue<Act>) {
        let (queue, sink) = ActionQueue::pair();
        let kit = Kit::new(Scale::unit(), Tokens::default(), sink);
        (TouchPad::new(kit), queue)
    }

    fn playing() -> FlowState {
        FlowState::new(Flow::Playing, Transition::Idle, Suspension::Live)
    }

    fn paused() -> FlowState {
        FlowState::new(Flow::Paused, Transition::Idle, Suspension::Live)
    }

    #[test]
    fn drag_within_the_radius_keeps_the_origin() {
        let origin = Vec2::new(100.0, 100.0);
        let (kept, delta) = drag(origin, Vec2::new(120.0, 100.0), 46.0);
        assert_eq!(kept, origin);
        assert_eq!(delta, Vec2::new(20.0, 0.0));
    }

    #[test]
    fn drag_past_the_radius_pulls_the_origin_along() {
        let origin = Vec2::new(100.0, 100.0);
        let (moved, delta) = drag(origin, Vec2::new(200.0, 100.0), 46.0);
        assert_eq!(moved, Vec2::new(154.0, 100.0));
        assert_eq!(delta, Vec2::new(46.0, 0.0));
        assert!((delta.length() - 46.0).abs() < 1e-6);
    }

    #[test]
    fn a_degenerate_radius_reads_as_rest() {
        let origin = Vec2::new(10.0, 10.0);
        assert_eq!(drag(origin, Vec2::new(50.0, 10.0), 0.0), (origin, Vec2::ZERO));
        assert_eq!(axis(Vec2::new(46.0, 0.0), 0.0), Vec2::ZERO);
    }

    #[test]
    fn axis_kills_the_dead_zone_and_reaches_full_deflection() {
        assert_eq!(axis(Vec2::ZERO, STICK_RADIUS), Vec2::ZERO);
        assert_eq!(
            axis(Vec2::new(8.0, 0.0), STICK_RADIUS),
            Vec2::ZERO,
            "0.17 of the radius is inside repame-input's 0.22 dead zone"
        );
        let half = axis(Vec2::new(23.0, 0.0), STICK_RADIUS);
        assert!(half.x > 0.0 && half.x < 0.5, "half deflection is partial, got {half:?}");
        let full = axis(Vec2::new(46.0, 0.0), STICK_RADIUS);
        assert!((full.x - 1.0).abs() < 1e-6, "full deflection is unit, got {full:?}");
        let beyond = axis(Vec2::new(200.0, 0.0), STICK_RADIUS);
        assert!((beyond.length() - 1.0).abs() < 1e-6, "clamped to the unit disc");
        assert_eq!(beyond.y, 0.0);
    }

    #[test]
    fn axis_keeps_the_direction() {
        let up_left = axis(Vec2::new(-46.0, -46.0), STICK_RADIUS);
        assert!(up_left.x < 0.0 && up_left.y < 0.0);
        assert!((up_left.length() - 1.0).abs() < 1e-6);
        assert!((up_left.x - up_left.y).abs() < 1e-6, "the diagonal stays symmetric");
    }

    #[test]
    fn the_gate_follows_the_flow_predicates() {
        let (pad, _queue) = pad();
        assert!(!pad.enabled(), "closed until a frame says otherwise");
        pad.begin_frame(playing());
        assert!(pad.enabled());
        pad.begin_frame(paused());
        assert!(!pad.enabled(), "a pause panel closes the gate");
        pad.begin_frame(FlowState::new(Flow::Playing, Transition::Running, Suspension::Live));
        assert!(!pad.enabled(), "a transition closes the gate");
        pad.begin_frame(FlowState::new(Flow::Playing, Transition::Idle, Suspension::Suspended));
        assert!(!pad.enabled(), "a suspend closes the gate");
        pad.begin_frame(playing());
        assert!(pad.enabled());
    }

    #[test]
    fn the_gate_can_be_read_from_a_sim_world() {
        use repame_sim::Sim;

        let (pad, _queue) = pad();
        let mut sim = Sim::with_default_step();
        // No flow installed: the pad stays closed rather than live.
        pad.begin_frame_in(&sim.world);
        assert!(!pad.enabled());

        crate::flow::install(&mut sim, Flow::Playing);
        pad.begin_frame_in(&sim.world);
        assert!(pad.enabled());

        sim.set_state(Flow::Modal);
        sim.tick();
        pad.begin_frame_in(&sim.world);
        assert!(!pad.enabled(), "a modal freezes the overlay");
    }

    #[test]
    fn the_closed_flow_state_runs_nothing() {
        assert!(!FlowState::CLOSED.run_live());
        assert!(FlowState::CLOSED.menu());
        assert!(FlowState::CLOSED.panel_owns_input());
    }

    #[test]
    fn closing_the_gate_releases_the_stick() {
        let (pad, _queue) = pad();
        pad.begin_frame(playing());
        {
            let mut inner = pad.inner.borrow_mut();
            inner.active = true;
            inner.raw = Vec2::new(46.0, 0.0);
        }
        assert!(pad.active());
        assert!((pad.stick().length() - 1.0).abs() < 1e-6);
        pad.begin_frame(paused());
        assert!(!pad.active());
        assert_eq!(pad.stick(), Vec2::ZERO);
    }

    #[test]
    fn the_stick_is_dead_zoned_in_design_units() {
        let (pad, _queue) = pad();
        pad.begin_frame(playing());
        {
            let mut inner = pad.inner.borrow_mut();
            inner.active = true;
            inner.raw = Vec2::new(46.0, 0.0);
        }
        let stick = pad.stick();
        assert!((stick.x - 1.0).abs() < 1e-6);
        {
            let mut inner = pad.inner.borrow_mut();
            inner.raw = Vec2::new(5.0, 0.0);
        }
        assert_eq!(pad.stick(), Vec2::ZERO, "inside the dead zone");
    }

    #[test]
    fn the_view_is_empty_while_the_gate_is_closed() {
        let (pad, _queue) = pad();
        pad.begin_frame(paused());
        let view = pad.view([960.0, 540.0]);
        assert!(view.children.is_empty(), "no stick under a modal");
        pad.begin_frame(playing());
        let view = pad.view([960.0, 540.0]);
        assert_eq!(view.children.len(), 2, "the touch zone and the ring layer");
    }

    #[test]
    fn a_tap_button_pushes_into_the_shared_queue() {
        let (pad, queue) = pad();
        pad.begin_frame(playing());
        let tap = pad.tap([960.0, 540.0], "arrow_upward", [820.0, 400.0], 48.0, Act::Jump);
        let on_click = tap.modifier.on_click.clone().expect("enabled taps click");
        on_click();
        let dive = pad.tap([960.0, 540.0], "arrow_downward", [820.0, 480.0], 48.0, Act::Dive);
        dive.modifier.on_click.clone().expect("enabled taps click")();
        assert_eq!(queue.drain(), vec![Act::Jump, Act::Dive]);
        assert!(!tap.modifier.focusable.unwrap_or(true), "taps stay out of keyboard traversal");
    }

    #[test]
    fn a_tap_button_is_disabled_with_the_gate() {
        let (pad, queue) = pad();
        pad.begin_frame(paused());
        let tap = pad.tap([960.0, 540.0], "arrow_upward", [820.0, 400.0], 48.0, Act::Jump);
        assert!(tap.modifier.on_click.is_none());
        assert!(tap.modifier.disabled);
        assert!(queue.is_empty());
    }

    #[test]
    fn a_zone_fraction_outside_the_unit_interval_is_rejected() {
        let (pad, _queue) = pad();
        assert_eq!(TouchPad::new(pad.kit().clone()).with_zone(0.0).err(), Some(TouchError::ZoneOutOfRange));
        assert_eq!(TouchPad::new(pad.kit().clone()).with_zone(1.5).err(), Some(TouchError::ZoneOutOfRange));
        assert_eq!(TouchPad::new(pad.kit().clone()).with_zone(f32::NAN).err(), Some(TouchError::ZoneOutOfRange));
        let half = TouchPad::new(pad.kit().clone()).with_zone(0.5).expect("half the width");
        assert!(!half.enabled(), "a fresh pad is closed until a frame opens it");
    }

    #[test]
    fn the_pads_own_driver_is_the_kits_driver() {
        let (pad, _queue) = pad();
        assert_eq!(pad.driver().wrap(), Wrap::Cycle);
    }

    #[test]
    fn a_contact_snapshot_can_seed_the_stick_on_the_sim_side() {
        // The pad itself reads attacks through repose's pointer path; the sim
        // side can still take a contact and ask for the same stick curve.
        let contact = TouchContact {
            id: 3,
            start: Vec2::new(80.0, 400.0),
            pos: Vec2::new(126.0, 400.0),
            just_pressed: true,
        };
        let (_, delta) = drag(contact.start, contact.pos, STICK_RADIUS);
        let value = axis(delta, STICK_RADIUS);
        assert!((value.x - 1.0).abs() < 1e-6, "a full drag reads as full deflection");
        assert_eq!(value.y, 0.0);
    }

    #[test]
    fn pad_state_does_not_move_the_stick_by_itself() {
        // A reminder encoded as a test: the touch pad is the only thing that
        // writes the stick, so a gamepad snapshot cannot leak into it.
        let pad_state = GamepadState::default();
        let (pad, _queue) = pad();
        pad.begin_frame(playing());
        assert_eq!(pad.stick(), Vec2::ZERO);
        assert_eq!(pad_state.left_stick, Vec2::ZERO);
    }
}
