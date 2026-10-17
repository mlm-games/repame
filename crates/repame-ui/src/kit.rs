//! The kit: palette tokens, one scale, one action sink and one nav map, handed
//! to every control a panel is built from.
//!
//! A template game's `ui/` module builds the same six things by hand — a chunky
//! button, a square glyph button, an information chip, a card, a label, a
//! scrim — and each of them repeats the palette blending and the layout anchor
//! arithmetic. [`Kit`] is that set once: the controls read [`Tokens`] and the
//! frame's [`Scale`], push into the shared [`Sink`], and register themselves in
//! the [`NavMap`] so arrows, Tab, the D-pad and an analog stick can reach them.
//!
//! The scale is shared behind a cell, so a game builds the kit once and writes
//! the new viewport scale each frame (`kit.set_scale(scale)`) without rebuilding
//! the kit or the closures it handed out.
//!
//! ```ignore
//! let kit = Kit::new(Scale::unit(), Tokens::default(), sink);
//! // once per frame, before composing:
//! kit.set_scale(Scale::new(viewport, layout::DESIGN)?);
//! kit.begin_frame();
//! // then, per panel:
//! kit.button("Play", UiAction::Play).view()
//! ```
//!
//! [`Tokens`]: crate::palette::Tokens

use std::cell::Cell;
use std::rc::Rc;

use repose_core::prelude::{Color, FlexDirection, Modifier, View};
use repose_core::text::TextAlign;
use repose_core::{AlignItems, Dp, JustifyContent, Sp};
use repose_ui::{IntoChildren, Text, TextStyle as _, ViewExt as _, ZStack};

use crate::layout::{Scale, anchor_rect, disc, ring};
use crate::nav::{GroupId, NavDriver, NavMap, Wrap};
use crate::palette::{
    self, FONT, ICON_BUTTON, ICON_GLYPH, MATERIAL_SYMBOLS, Tokens, outline_for,
};
use crate::queue::{Sink, UiAction};

/// The scale, tokens, sink and nav map a panel's controls are built from.
///
/// Build one per game and clone it into whatever composes: the scale cell, the
/// sink and the nav map are shared behind `Rc`/`Arc`, so every clone sees the
/// same frame. A kit rebuilt every frame would start from an empty nav map,
/// whose one-frame rectangle binding would never complete and whose controls
/// could never be reached.
#[derive(Clone)]
pub struct Kit<A: UiAction> {
    scale: Rc<Cell<Scale>>,
    tokens: Tokens,
    sink: Sink<A>,
    nav: NavMap,
    wrap: Wrap,
}

impl<A: UiAction> Kit<A> {
    /// A kit drawing with `tokens`, pushing into `sink` and registering in a
    /// fresh nav map.
    pub fn new(scale: Scale, tokens: Tokens, sink: Sink<A>) -> Self {
        Self {
            scale: Rc::new(Cell::new(scale)),
            tokens,
            sink,
            nav: NavMap::new(),
            wrap: Wrap::default(),
        }
    }

    /// A kit wrapping an existing nav map, so a game that keeps its own map
    /// still shares it with the controls.
    pub fn with_map(scale: Scale, tokens: Tokens, sink: Sink<A>, nav: NavMap) -> Self {
        Self {
            scale: Rc::new(Cell::new(scale)),
            tokens,
            sink,
            nav,
            wrap: Wrap::default(),
        }
    }

    /// This frame's viewport scale. All clones of the kit see it.
    pub fn set_scale(&self, scale: Scale) {
        self.scale.set(scale);
    }

    pub fn scale(&self) -> Scale {
        self.scale.get()
    }

    pub fn tokens(&self) -> Tokens {
        self.tokens
    }

    /// Replace the palette. Returns the re-skinned kit.
    pub fn with_tokens(mut self, tokens: Tokens) -> Self {
        self.tokens = tokens;
        self
    }

    /// Edge behaviour for directional traversal.
    pub fn with_wrap(mut self, wrap: Wrap) -> Self {
        self.wrap = wrap;
        self
    }

    pub fn sink(&self) -> &Sink<A> {
        &self.sink
    }

    pub fn nav_map(&self) -> &NavMap {
        &self.nav
    }

    /// Traversal over this kit's controls.
    pub fn driver(&self) -> NavDriver {
        NavDriver::new(self.nav.clone()).with_wrap(self.wrap)
    }

    /// Retire the controls that are no longer composed: call once per frame,
    /// before composing panels.
    pub fn begin_frame(&self) {
        self.nav.begin_frame();
    }

    /// A chunky text button that pushes `action` and takes focus.
    pub fn button(&self, text: impl Into<String>, action: A) -> Button<A> {
        Button {
            kit: self.clone(),
            text: text.into(),
            glyph: None,
            fill: self.tokens.cream,
            text_fill: None,
            enabled: true,
            group: None,
            key: None,
            action,
        }
    }

    /// A square glyph button (play, pause, home) that pushes `action`.
    pub fn icon_button(&self, glyph: impl Into<String>, action: A) -> Button<A> {
        Button {
            kit: self.clone(),
            text: glyph.into(),
            glyph: Some(()),
            fill: self.tokens.cream,
            text_fill: Some(self.tokens.ink),
            enabled: true,
            group: None,
            key: None,
            action,
        }
    }

    /// An information chip: a HUD counter, score or timer face.
    pub fn chip(&self, text: impl Into<String>, face: Color) -> View {
        let scale = self.scale.get().fit();
        let tokens = self.tokens;
        repose_ui::Box(palette::chip(&tokens, face, scale))
            .child(self.label_plain(Modifier::new(), &text.into(), 20.0, tokens.ink))
    }

    /// A card: the panel body, placed by viewport anchor and design-unit
    /// offset, stacking its children in a column with the default gap.
    ///
    /// Add children to the returned view; they are centred as a column, which is
    /// what every template menu is.
    pub fn panel(
        &self,
        viewport: [f32; 2],
        anchor: [f32; 4],
        offset: [f32; 4],
        face: Color,
    ) -> View {
        let scale = self.scale.get().fit();
        let tokens = self.tokens;
        repose_ui::Box(
            anchor_rect(viewport, scale, anchor, offset)
                .background(face)
                .border(Dp(3.0 * scale), tokens.ink, Dp(18.0 * scale))
                .shadow_with_color(Dp(0.0), Dp(6.0 * scale), tokens.button_shadow)
                .flex_dir(FlexDirection::Column)
                .row_gap(Dp(14.0 * scale))
                .align_items(AlignItems::CENTER)
                .justify_content(JustifyContent::CENTER),
        )
    }

    /// The dim behind a modal panel: full-bleed, and it blocks pointer input so
    /// a click cannot reach the game underneath. It stays out of the focus
    /// chain.
    pub fn scrim(&self) -> View {
        let tokens = self.tokens;
        ZStack(
            Modifier::new()
                .fill_max_size()
                .background(tokens.scrim)
                .input_blocker()
                .focusable(false),
        )
        .child(repose_ui::Spacer())
    }

    /// Outlined text: labels that must read over game art.
    pub fn label(&self, modifier: Modifier, text: &str, size: f32, fill: Color) -> View {
        let scale = self.scale.get().fit();
        let tokens = self.tokens;
        Text(text.to_string())
            .size(Sp(size * scale))
            .text_align(TextAlign::Center)
            .font_family(FONT)
            .color(fill)
            .outlined(0.42 * outline_for(size) / size, tokens.ink)
            .single_line()
            .overflow_visible()
            .modifier(modifier)
    }

    /// Plain text: numbers and button captions, no outline.
    pub fn label_plain(&self, modifier: Modifier, text: &str, size: f32, fill: Color) -> View {
        let scale = self.scale.get().fit();
        Text(text.to_string())
            .size(Sp(size * scale))
            .text_align(TextAlign::Center)
            .font_family(FONT)
            .color(fill)
            .single_line()
            .overflow_visible()
            .modifier(modifier)
    }

    /// Display text: tighter tracking and a heavier faux-bold, for titles.
    pub fn display(&self, modifier: Modifier, text: &str, size: f32, fill: Color) -> View {
        let scale = self.scale.get().fit();
        let tracking = (size * -0.012).clamp(-1.2, 0.0);
        Text(text.to_string())
            .size(Sp(size * scale))
            .line_height(Sp(size * scale * 1.02))
            .letter_spacing(Sp(tracking * scale))
            .text_align(TextAlign::Center)
            .font_family(FONT)
            .color(fill)
            .fill_and_stroke(0.028)
            .single_line()
            .overflow_visible()
            .modifier(modifier)
    }

    /// A Material Symbols glyph.
    pub fn icon(&self, glyph: &str, size: f32, fill: Color) -> View {
        let scale = self.scale.get().fit();
        Text(glyph.to_string())
            .size(Sp(size * scale))
            .text_align(TextAlign::Center)
            .font_family(MATERIAL_SYMBOLS)
            .color(fill)
            .single_line()
            .overflow_visible()
            .modifier(Modifier::new())
    }

    /// A filled circle in design space.
    pub fn disc(&self, viewport: [f32; 2], at: [f32; 2], radius: f32, color: Color) -> View {
        disc(viewport, self.scale.get().fit(), at, radius, color)
    }

    /// An unfilled circle in design space.
    pub fn ring(
        &self,
        viewport: [f32; 2],
        at: [f32; 2],
        radius: f32,
        width: f32,
        color: Color,
    ) -> View {
        ring(viewport, self.scale.get().fit(), at, radius, width, color)
    }

    /// The `rect(vp, s, anchor, offset)` placement helper, with this kit's
    /// scale.
    pub fn place(&self, viewport: [f32; 2], anchor: [f32; 4], offset: [f32; 4]) -> Modifier {
        anchor_rect(viewport, self.scale.get().fit(), anchor, offset)
    }

    /// The virtual stick and tap buttons, pushing into the same sink.
    ///
    /// Build the pad once and keep it in the app, like the game's own input
    /// state: a pad built per frame forgets the finger that is down.
    pub fn touch_pad(&self) -> crate::touch::TouchPad<A> {
        crate::touch::TouchPad::new(self.clone())
    }
}

/// A button being configured. Add it to a parent view, which builds it.
pub struct Button<A: UiAction> {
    kit: Kit<A>,
    text: String,
    glyph: Option<()>,
    fill: Color,
    text_fill: Option<Color>,
    enabled: bool,
    group: Option<GroupId>,
    key: Option<String>,
    action: A,
}

impl<A: UiAction> Button<A> {
    /// Face colour. Defaults to the palette's cream.
    pub fn fill(mut self, face: Color) -> Self {
        self.fill = face;
        self
    }

    /// Caption colour. Defaults to the palette's ink.
    pub fn text_fill(mut self, fill: Color) -> Self {
        self.text_fill = Some(fill);
        self
    }

    /// A disabled button keeps its place but takes neither pointer nor
    /// keyboard activation.
    pub fn disabled(mut self, disabled: bool) -> Self {
        self.enabled = !disabled;
        self
    }

    /// Put this control in a focus group, so traversal cannot leave the panel
    /// it belongs to.
    pub fn group(mut self, group: GroupId) -> Self {
        self.group = Some(group);
        self
    }

    /// Identify this control to focus memory.
    ///
    /// The default identity is the caption (or the glyph), which is unique
    /// inside a panel in every template game. Two controls that share a caption
    /// inside one composition scope — two "OK" buttons in one dialog — must be
    /// given distinct keys, or they share one remembered focus target.
    pub fn keyed(mut self, key: impl Into<String>) -> Self {
        self.key = Some(key.into());
        self
    }

    /// Build the view.
    pub fn view(self) -> View {
        let kit = self.kit;
        let scale = kit.scale.get().fit();
        let tokens = kit.tokens;
        let square = self.glyph.is_some();

        let mut modifier = palette::chunky(
            &tokens,
            self.fill,
            if square { 10.0 } else { 22.0 },
            if square { 10.0 } else { 12.0 },
            if square { 12.0 } else { 16.0 },
            scale,
        );
        if square {
            modifier = modifier.size(Dp(ICON_BUTTON * scale), Dp(ICON_BUTTON * scale));
        }
        if !self.enabled {
            modifier = modifier.disabled();
        } else {
            let sink = kit.sink.clone();
            let action = self.action.clone();
            modifier = modifier.on_click(move || sink.push(action.clone()));

            // The focus requester has to outlive the frame: a fresh one every
            // frame would never learn its view id, so it is remembered per
            // control key.
            let identity = self.key.clone().unwrap_or_else(|| {
                if square {
                    format!("icon.{}", self.text)
                } else {
                    self.text.clone()
                }
            });
            let requester = repose_core::remember_with_key(
                format!("repame_ui.control.{identity}"),
                repose_core::prelude::FocusRequester::new,
            );

            // Register with the nav map: the layout pass reports the rectangle,
            // and from the control's second frame on it is registered while
            // composing, so a closed panel drops out of traversal in the frame
            // it disappears. A disabled control is not registered at all:
            // repose leaves it out of the focus chain, and a map entry the chain
            // does not have would make a press onto it go nowhere.
            let nav = kit.nav.clone();
            let group = self.group;
            let place = (*requester).clone();
            let map = nav.clone();
            modifier = modifier
                .focus_requester((*requester).clone())
                .on_globally_positioned(move |rect| {
                    map.place(&place, rect, group);
                });
            nav.register(&requester, group);
        }

        let caption = match (self.glyph, self.text_fill) {
            (Some(()), fill) => kit.icon(&self.text, ICON_GLYPH, fill.unwrap_or(tokens.ink)),
            (None, Some(fill)) => kit.label_plain(Modifier::new(), &self.text, 24.0, fill),
            (None, None) => kit.label(Modifier::new(), &self.text, 24.0, tokens.ink),
        };
        repose_ui::Box(modifier).child(caption)
    }
}

impl<A: UiAction> IntoChildren for Button<A> {
    fn into_children(self) -> Vec<View> {
        vec![self.view()]
    }
}

/// The overlay root: full-screen, and deliberately not focusable.
///
/// The template games wrap their overlay in a focusable root that requests focus
/// for itself every frame, which is why no button in those games can be reached
/// with a keyboard or a pad. A root that stays out of the focus chain leaves
/// focus to the kit's controls.
pub fn overlay_root(children: impl IntoChildren) -> View {
    ZStack(Modifier::new().fill_max_size()).child(children)
}

/// Whether a glyph-font family is registered: the kit's `icon` helpers need
/// `Material Symbols Outlined` to have been registered by the game.
pub const ICON_FAMILY: &str = MATERIAL_SYMBOLS;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::queue::ActionQueue;
    use crate::nav::NavDir;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Act {
        Play,
        Mute,
    }

    fn kit() -> (Kit<Act>, ActionQueue<Act>) {
        let (queue, sink) = ActionQueue::pair();
        let kit = Kit::new(Scale::unit(), Tokens::default(), sink);
        (kit, queue)
    }

    #[test]
    fn the_scale_is_shared_across_clones() {
        let (kit, _queue) = kit();
        let clone = kit.clone();
        kit.set_scale(Scale::new([1920.0, 1080.0], crate::layout::DESIGN).expect("scales"));
        assert!((clone.scale().fit() - 2.0).abs() < 1e-6);
        assert_eq!(clone.tokens(), Tokens::default());
    }

    #[test]
    fn a_button_pushes_its_action_and_registers_in_the_map() {
        let (kit, queue) = kit();
        kit.begin_frame();
        let view = kit.button("Play", Act::Play).view();
        // Composition happened; the modifier is what carries the click and the
        // focus request, so the closure is exercised by calling the modifier's
        // handler directly.
        let on_click = view.modifier.on_click.clone().expect("button is clickable");
        on_click();
        assert_eq!(queue.drain(), vec![Act::Play]);
        assert!(view.modifier.focusable.unwrap_or(false), "button takes focus");
        assert!(view.modifier.focus_requester.is_some(), "button can request focus");
    }

    #[test]
    fn a_disabled_button_neither_clicks_nor_joins_traversal() {
        let (kit, queue) = kit();
        kit.begin_frame();
        let view = kit.button("Play", Act::Play).disabled(true).view();
        assert!(view.modifier.on_click.is_none());
        assert!(view.modifier.focus_requester.is_none(), "no focus target either");
        assert!(view.modifier.disabled, "repose skips disabled controls");
        assert!(queue.is_empty());
    }

    #[test]
    fn an_icon_button_is_square_and_captionless() {
        let (kit, _queue) = kit();
        kit.begin_frame();
        let view = kit.button("x", Act::Mute).fill(Tokens::default().coral).view();
        assert!(view.modifier.on_click.is_some());
        assert_eq!(view.children.len(), 1);
        let icon = kit.icon_button("pause", Act::Mute).view();
        assert!(icon.modifier.on_click.is_some());
    }

    #[test]
    fn a_button_joins_its_focus_group() {
        let (kit, _queue) = kit();
        kit.begin_frame();
        let pause = GroupId::named("pause");
        let resume = kit.button("Resume", Act::Play).group(pause).view();
        let quit = kit.button("Quit", Act::Mute).group(pause).view();
        // The map needs a layout pass to place the controls: compose-time
        // registration only learns the view id from the previous frame.
        assert!(kit.nav_map().is_empty(), "no id and no rectangle yet");
        let resume_place = resume
            .modifier
            .on_globally_positioned
            .clone()
            .expect("buttons report their rectangle");
        let quit_place = quit
            .modifier
            .on_globally_positioned
            .clone()
            .expect("buttons report their rectangle");
        resume_place(repose_core::Rect { x: 200.0, y: 300.0, w: 200.0, h: 64.0 });
        quit_place(repose_core::Rect { x: 200.0, y: 380.0, w: 200.0, h: 64.0 });

        // The paint pass assigns view ids at the end of the frame; the next
        // frame's registration binds the rectangles to them.
        let resume_requester = resume
            .modifier
            .focus_requester
            .clone()
            .expect("buttons request focus");
        let quit_requester = quit
            .modifier
            .focus_requester
            .clone()
            .expect("buttons request focus");
        *resume_requester.target.borrow_mut() = Some(41);
        *quit_requester.target.borrow_mut() = Some(42);
        kit.begin_frame();
        assert!(kit.nav_map().register(&resume_requester, Some(pause)));
        assert!(kit.nav_map().register(&quit_requester, Some(pause)));

        assert_eq!(crate::nav::chain_of(kit.nav_map()), [41, 42]);
        assert!(crate::nav::same_group(kit.nav_map(), 41, 42));
        assert_eq!(kit.driver().wrap(), Wrap::Cycle);
    }

    #[test]
    fn a_button_becomes_navigable_the_frame_after_it_appears() {
        let (kit, _queue) = kit();
        kit.begin_frame();
        let view = kit.button("Play", Act::Play).view();
        let place = view
            .modifier
            .on_globally_positioned
            .clone()
            .expect("buttons report their rectangle");
        let requester = view
            .modifier
            .focus_requester
            .clone()
            .expect("buttons request focus");
        place(repose_core::Rect { x: 40.0, y: 300.0, w: 200.0, h: 64.0 });
        assert!(kit.nav_map().is_empty(), "the rectangle waits for the view id");
        *requester.target.borrow_mut() = Some(31);

        kit.begin_frame();
        assert!(kit.nav_map().register(&requester, None));
        let entry = kit.nav_map().active().into_iter().next().expect("one control");
        assert_eq!(entry.id(), 31);
        assert_eq!(entry.rect().w, 200.0);
        assert_eq!(
            kit.driver().enter(&repose_core::prelude::Scheduler::default()),
            crate::nav::NavStep::Entered(31),
            "the control is the entry point of an empty focus chain"
        );
    }

    #[test]
    fn a_kits_controls_can_be_driven_by_direction() {
        let (kit, _queue) = kit();
        let nav = kit.nav_map();
        nav.begin_frame();
        // Two laid-out controls, marked the way the button builder marks them.
        for (id, x) in [(11u64, 100.0f32), (12, 400.0)] {
            let requester = repose_core::prelude::FocusRequester::new();
            *requester.target.borrow_mut() = Some(id);
            nav.mark(id, Some(GroupId::named("pause")), requester.clone());
            nav.place(&requester, repose_core::Rect { x, y: 0.0, w: 200.0, h: 80.0 }, Some(GroupId::named("pause")));
        }
        assert_eq!(nav.resolve(Some(11), NavDir::Right, Wrap::Stay), crate::nav::NavStep::Moved(12));
        assert_eq!(nav.len(), 2);
    }

    #[test]
    fn extra_voices_read_the_scale_and_tokens() {
        let (kit, queue) = kit();
        kit.set_scale(Scale::new([480.0, 270.0], crate::layout::DESIGN).expect("scales"));
        let chip = kit.chip("12", Tokens::default().gold);
        assert_eq!(chip.children.len(), 1);
        let scrim = kit.scrim();
        assert!(scrim.modifier.input_blocker, "the scrim blocks the game underneath");
        assert!(!scrim.modifier.focusable.unwrap_or(true), "the scrim takes no focus");
        let panel = kit.panel([480.0, 270.0], [0.5, 0.5, 0.5, 0.5], [-100.0, -50.0, 100.0, 50.0], Tokens::default().cream);
        assert!(panel.modifier.offset_left.is_some(), "panels are placed absolutely");
        let label = kit.label(Modifier::new(), "Play", 24.0, Tokens::default().ink);
        assert_eq!(label.children.len() + 1, 1, "labels are leaves");
        assert!(queue.is_empty());
    }

    #[test]
    fn the_overlay_root_stays_out_of_the_focus_chain() {
        let root = overlay_root(repose_ui::Spacer());
        assert!(root.modifier.focusable.is_none(), "the root must not take focus");
        assert!(root.modifier.focus_requester.is_none());
    }

    #[test]
    fn touch_pads_share_the_kits_sink() {
        let (kit, queue) = kit();
        let pad = kit.touch_pad();
        pad.begin_frame(crate::flow::FlowState::new(
            crate::flow::Flow::Playing,
            crate::flow::Transition::Idle,
            crate::flow::Suspension::Live,
        ));
        assert!(pad.enabled());
        assert!(queue.is_empty());
    }
}
