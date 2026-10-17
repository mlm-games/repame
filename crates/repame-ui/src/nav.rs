//! Focus traversal over kit controls, built on repose's own focus machinery.
//!
//! Repose already implements spatial traversal (`repose_core::runtime::
//! spatial_focus_next`, the function behind `focus.rs`'s action handling),
//! `FocusRequester` with the platform's request queue and capture lock, and
//! focus groups. What the games are missing is three things this module
//! supplies, without re-implementing any of it:
//!
//! 1. **A layout to traverse.** [`NavMap`] collects the controls the kit
//!    composes — their view ids, design-space rectangles and focus groups — from
//!    `Modifier::on_globally_positioned`, i.e. from the layout pass that already
//!    hit-tests pointers. No second hit test, no second rectangle. The layout
//!    pass reports a rectangle before the paint pass assigns the control's view
//!    id, so each control's first rectangle is held for one frame and bound to
//!    its id by [`NavMap::register`]; controls are navigable from their second
//!    frame on.
//! 2. **An entry point.** Nothing ever moves focus *to* a control in the
//!    template games: the overlay root asks for focus for itself and keeps it.
//!    [`NavDriver::enter`] hands focus to the first control when nothing in the
//!    map is focused.
//! 3. **Edge behaviour and analog input.** [`Wrap`] decides whether a
//!    directional press at the layout edge wraps around the panel or stays put;
//!    [`StickNav`] turns an analog stick into discrete presses with a repeat
//!    delay, reusing `repame_input`'s dead zone.
//!
//! Focus moves by *requesting* it (`FocusRequester::try_request_focus`), so the
//! platform's capture lock keeps working: a modal that captured focus keeps it.
//! Keyboard (Tab/arrows, Space/Enter to activate) and gamepad D-pad/buttons
//! already reach a focused control through the platform's key dispatch; analog
//! sticks do not, which is why [`StickNav`] exists.
//!
//! # The frame
//!
//! ```ignore
//! // Once per frame, before composing:
//! kit.begin_frame();
//! // Compose panels. Every kit button registers itself while composing, and the
//! // layout pass that follows fills in its rectangle.
//! let root = overlay_root((hud, panel));
//! // Traversal, anywhere in the frame: it resolves against the controls of the
//! // last frame that were composed (this frame's, once the panels are built).
//! let driver = kit.driver();
//! driver.enter(sched);
//! for dir in dpad_dirs(&pad).into_iter().chain(stick.step(pad.left_stick, dt)) {
//!     driver.drive(sched, dir);
//! }
//! ```
//!
//! Nothing in the kit touches `Scheduler::focused` directly: the request goes
//! on the platform's focus queue, which the platform drains at the end of the
//! frame's layout, so the focused control is the one the player sees that frame.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use glam::Vec2;
use repose_core::Rect;
use repose_core::prelude::{FocusDirection, FocusRequester, Scheduler};
use repose_core::runtime::{FocusManager, HitRegion, spatial_focus_next};

/// A traversal request in kit terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NavDir {
    /// Next control in chain order: Tab.
    Next,
    /// Previous control in chain order: Shift-Tab.
    Previous,
    Left,
    Right,
    Up,
    Down,
}

impl NavDir {
    /// Every direction, chain order first.
    pub const ALL: [NavDir; 6] = [
        NavDir::Next,
        NavDir::Previous,
        NavDir::Left,
        NavDir::Right,
        NavDir::Up,
        NavDir::Down,
    ];

    /// The repose focus direction this request runs as.
    pub fn focus_direction(self) -> FocusDirection {
        match self {
            NavDir::Next => FocusDirection::Next,
            NavDir::Previous => FocusDirection::Previous,
            NavDir::Left => FocusDirection::Left,
            NavDir::Right => FocusDirection::Right,
            NavDir::Up => FocusDirection::Up,
            NavDir::Down => FocusDirection::Down,
        }
    }

    /// The opposite direction.
    pub fn reverse(self) -> Self {
        match self {
            NavDir::Next => NavDir::Previous,
            NavDir::Previous => NavDir::Next,
            NavDir::Left => NavDir::Right,
            NavDir::Right => NavDir::Left,
            NavDir::Up => NavDir::Down,
            NavDir::Down => NavDir::Up,
        }
    }

    /// Whether the direction moves along the chain rather than through the
    /// layout.
    pub fn is_chain(self) -> bool {
        matches!(self, NavDir::Next | NavDir::Previous)
    }

    /// The axis a wrapped press travels along, and its sign.
    fn axis(self) -> Option<(Axis, f32)> {
        match self {
            NavDir::Left => Some((Axis::X, -1.0)),
            NavDir::Right => Some((Axis::X, 1.0)),
            NavDir::Up => Some((Axis::Y, -1.0)),
            NavDir::Down => Some((Axis::Y, 1.0)),
            NavDir::Next | NavDir::Previous => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Axis {
    X,
    Y,
}

/// A focus group: controls that traversal stays inside once one of them is
/// focused.
///
/// This mirrors repose's `focus_group_id` (the same field travels in the hit
/// regions handed to `spatial_focus_next`), so every control of a modal panel
/// carrying one group traps focus in that panel without interception logic.
/// Tag groups by name so behaviour does not depend on build order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GroupId(u64);

impl GroupId {
    /// A group id derived from a stable name: FNV-1a over the tag's bytes.
    pub fn named(tag: &str) -> Self {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in tag.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
        }
        Self(hash)
    }

    /// A group id from a value the game already has.
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// What happens at the layout edge when a directional press finds nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Wrap {
    /// Focus stays put: the panel reads as a bounded box.
    Stay,
    /// Focus wraps around the panel, keeping the direction of travel (right at
    /// the right edge lands on the leftmost control of the nearest row).
    #[default]
    Cycle,
}

/// What a traversal request did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NavStep {
    /// Focus moved to this control.
    Moved(u64),
    /// Nothing was focused, so the first control took focus: the entry point.
    Entered(u64),
    /// Focus was already on a control of this map.
    Already(u64),
    /// The press found nowhere to go: a layout edge under [`Wrap::Stay`], or a
    /// panel with a single control.
    Edge,
    /// The platform refused the request: another layer captured focus.
    Held,
    /// No control is in the map.
    Empty,
}

impl NavStep {
    /// The control this step ended on, when it ended on one.
    pub fn target(self) -> Option<u64> {
        match self {
            NavStep::Moved(id) | NavStep::Entered(id) | NavStep::Already(id) => Some(id),
            NavStep::Edge | NavStep::Held | NavStep::Empty => None,
        }
    }
}

/// One focusable control as the layout pass placed it.
#[derive(Clone)]
pub struct NavEntry {
    id: u64,
    rect: Rect,
    group: Option<GroupId>,
    requester: FocusRequester,
}

impl NavEntry {
    /// The control's view id: the same id repose puts in its focus chain.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The control's rectangle in design units.
    pub fn rect(&self) -> Rect {
        self.rect
    }

    pub fn group(&self) -> Option<GroupId> {
        self.group
    }

    /// Ask the platform for this control's focus.
    pub fn request_focus(&self) -> bool {
        self.requester.try_request_focus()
    }
}

struct Entry {
    rect: Option<Rect>,
    group: Option<GroupId>,
    requester: FocusRequester,
    /// Frame stamp: an entry stays alive by being marked or placed again.
    frame: u64,
}

/// A rectangle observed before its control had a view id.
///
/// The paint pass reports `on_globally_positioned` before `set_focus_requester`
/// assigns the control's view id, so a control's very first observation arrives
/// anonymous. It is bound to the id on the next frame, keyed by the
/// `FocusRequester` the control owns: one allocation for the life of the
/// control, so it identifies it across frames. The focus group is not carried
/// here: registration supplies it.
struct Pending {
    key: usize,
    rect: Rect,
    frame: u64,
}

/// Identity of a control's focus requester, stable across frames.
fn requester_key(requester: &FocusRequester) -> usize {
    Rc::as_ptr(&requester.target) as usize
}

#[derive(Default)]
struct Registry {
    frame: u64,
    order: Vec<u64>,
    entries: HashMap<u64, Entry>,
    pending: Vec<Pending>,
}

/// The controls the kit composed this frame, with their laid-out rectangles.
///
/// Clone the map into every control: it is one shared registry behind an `Rc`.
/// The game calls [`NavMap::begin_frame`] once per frame before composing, which
/// retires controls that were not recomposed, so a panel stops being navigable
/// in the same frame it disappears.
///
/// A control registers itself while composing ([`NavMap::register`]) and reports
/// its rectangle from the layout callback ([`NavMap::place`]). Traversal
/// resolves against whatever is registered, so a call made before the panels are
/// composed in a frame resolves against the previous frame's layout — the
/// controls the player was looking at.
#[derive(Clone, Default)]
pub struct NavMap {
    registry: Rc<RefCell<Registry>>,
}

impl NavMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Start a frame: every control must be marked or placed again to stay in
    /// the map. Call before composing the panel.
    pub fn begin_frame(&self) {
        let mut registry = self.registry.borrow_mut();
        registry.frame += 1;
        let frame = registry.frame;
        registry.entries.retain(|_, entry| entry.frame + 1 >= frame);
        let mut order = std::mem::take(&mut registry.order);
        order.retain(|id| registry.entries.contains_key(id));
        registry.order = order;
        registry
            .pending
            .retain(|pending| pending.frame + 1 >= frame);
    }

    /// Forget every control, including the ones marked this frame.
    pub fn clear(&self) {
        let mut registry = self.registry.borrow_mut();
        registry.entries.clear();
        registry.order.clear();
        registry.pending.clear();
    }

    /// Register a control for this frame, deriving its view id from its focus
    /// requester.
    ///
    /// This is what a kit control calls while composing. It returns `false` on
    /// the control's first frame, before the first layout pass has assigned the
    /// id; the rectangle that layout reported is kept and bound on the next
    /// frame, so a control becomes navigable one frame after it first appears.
    pub fn register(&self, requester: &FocusRequester, group: Option<GroupId>) -> bool {
        let Some(id) = *requester.target.borrow() else {
            return false;
        };
        let mut registry = self.registry.borrow_mut();
        let frame = registry.frame;
        let key = requester_key(requester);
        let rect = match registry.pending.iter().position(|pending| pending.key == key) {
            Some(index) => Some(registry.pending.remove(index).rect),
            None => None,
        };
        match registry.entries.get_mut(&id) {
            Some(entry) => {
                entry.group = group;
                entry.requester = requester.clone();
                entry.frame = frame;
                if let Some(rect) = rect {
                    entry.rect = Some(rect);
                }
            }
            None => {
                registry.order.push(id);
                registry.entries.insert(
                    id,
                    Entry {
                        rect,
                        group,
                        requester: requester.clone(),
                        frame,
                    },
                );
            }
        }
        true
    }

    /// Mark a control whose view id the caller already has.
    ///
    /// Kit controls go through [`register`](Self::register); this is for a game
    /// that composes its own control and reads the id from the layout.
    pub fn mark(&self, id: u64, group: Option<GroupId>, requester: FocusRequester) {
        let mut registry = self.registry.borrow_mut();
        let frame = registry.frame;
        match registry.entries.get_mut(&id) {
            Some(entry) => {
                entry.group = group;
                entry.requester = requester;
                entry.frame = frame;
            }
            None => {
                registry.order.push(id);
                registry.entries.insert(
                    id,
                    Entry {
                        rect: None,
                        group,
                        requester,
                        frame,
                    },
                );
            }
        }
    }

    /// Record a control's laid-out rectangle, from
    /// `Modifier::on_globally_positioned`.
    ///
    /// A rectangle that arrives before the control's view id is known is held
    /// for it and bound by the next [`register`](Self::register); after that,
    /// the callback only fires when the rectangle changes, so this is a cheap
    /// update. Returns whether the rectangle was recorded against a control.
    pub fn place(&self, requester: &FocusRequester, rect: Rect, group: Option<GroupId>) -> bool {
        let mut registry = self.registry.borrow_mut();
        let frame = registry.frame;
        let Some(id) = *requester.target.borrow() else {
            let key = requester_key(requester);
            match registry.pending.iter_mut().find(|pending| pending.key == key) {
                Some(pending) => {
                    pending.rect = rect;
                    pending.frame = frame;
                }
                None => registry.pending.push(Pending { key, rect, frame }),
            }
            return false;
        };
        match registry.entries.get_mut(&id) {
            Some(entry) => {
                entry.rect = Some(rect);
                entry.group = group;
                entry.frame = frame;
            }
            None => {
                registry.order.push(id);
                registry.entries.insert(
                    id,
                    Entry {
                        rect: Some(rect),
                        group,
                        requester: requester.clone(),
                        frame,
                    },
                );
            }
        }
        let key = requester_key(requester);
        registry.pending.retain(|pending| pending.key != key);
        true
    }

    /// Controls composed and laid out this frame, in composition order.
    pub fn active(&self) -> Vec<NavEntry> {
        let registry = self.registry.borrow();
        registry
            .order
            .iter()
            .filter_map(|id| {
                let entry = registry.entries.get(id)?;
                Some(NavEntry {
                    id: *id,
                    rect: entry.rect?,
                    group: entry.group,
                    requester: entry.requester.clone(),
                })
            })
            .collect()
    }

    /// The first control in composition order.
    pub fn first(&self) -> Option<u64> {
        self.active().first().map(NavEntry::id)
    }

    pub fn len(&self) -> usize {
        self.active().len()
    }

    pub fn is_empty(&self) -> bool {
        self.active().is_empty()
    }

    /// Whether `id` is a control on screen this frame.
    pub fn contains(&self, id: u64) -> bool {
        self.active().iter().any(|entry| entry.id() == id)
    }

    /// Ask the platform to focus `id`. `false` when there is no such control or
    /// the request was refused (captured focus, not laid out yet).
    pub fn request(&self, id: u64) -> bool {
        self.entry(id).is_some_and(|entry| entry.request_focus())
    }

    /// Pin focus to `id` until [`release`](Self::release): what a modal panel
    /// does when it opens, so traversal cannot escape to what is underneath.
    pub fn capture(&self, id: u64) -> bool {
        self.entry(id).is_some_and(|entry| entry.requester.capture_focus())
    }

    /// Release a capture taken by [`capture`](Self::capture).
    ///
    /// Focus is cleared, so the caller asks for the next panel's focus with
    /// [`NavDriver::enter`] once the modal is gone. `false` when `id` is not a
    /// control of this frame's map.
    pub fn release(&self, id: u64) -> bool {
        match self.entry(id) {
            Some(entry) => {
                entry.requester.free_focus();
                true
            }
            None => false,
        }
    }

    fn entry(&self, id: u64) -> Option<NavEntry> {
        self.active().into_iter().find(|entry| entry.id == id)
    }

    /// Resolve one traversal request.
    pub fn resolve(&self, from: Option<u64>, dir: NavDir, wrap: Wrap) -> NavStep {
        let entries = self.active();
        if entries.is_empty() {
            return NavStep::Empty;
        }

        // No focus to move from, or focus outside the map (a root that grabbed
        // it, a control of a panel that just closed): enter at the first control
        // instead of doing nothing.
        let Some(current) = from.filter(|id| entries.iter().any(|entry| entry.id == *id)) else {
            return NavStep::Entered(entries[0].id);
        };

        let (chain, regions) = regions_of(&entries);
        let step = |next: u64| {
            if next == current {
                NavStep::Edge
            } else {
                NavStep::Moved(next)
            }
        };
        if let Some(next) = spatial_focus_next(&chain, &regions, Some(current), dir.focus_direction())
        {
            return step(next);
        }
        if dir.is_chain() {
            // Chain order already wraps; reaching here means the chain holds one
            // control.
            return NavStep::Edge;
        }
        match wrap {
            Wrap::Stay => NavStep::Edge,
            Wrap::Cycle => self.wrapped(&entries, current, dir).map_or(NavStep::Edge, step),
        }
    }

    /// Retry a directional press as if the panel repeated past the edge.
    ///
    /// The candidates that are *not* ahead of the focus in the direction of
    /// travel are translated across the panel by its span, so right at the right
    /// edge sees the leftmost controls of the nearest row through the same
    /// spatial scoring and perpendicular tolerance the unwrapped pass uses. The
    /// focused control's own rectangle is the reference the scoring measures
    /// from, so it is left where it is.
    ///
    /// If the unwrapped pass found nothing, no candidate was in the direction
    /// *and* inside the perpendicular band, so translating the ones that are
    /// behind cannot hide a nearer candidate.
    fn wrapped(&self, entries: &[NavEntry], current: u64, dir: NavDir) -> Option<u64> {
        let (axis, sign) = dir.axis()?;
        let span = self.span(entries, axis);
        if span <= 0.0 {
            return None;
        }
        let (chain, mut regions) = regions_of(entries);
        for (region, entry) in regions.iter_mut().zip(entries) {
            if entry.id != current && !is_ahead(entry.rect, entries, current, axis, sign) {
                match axis {
                    Axis::X => region.rect.x += sign * span,
                    Axis::Y => region.rect.y += sign * span,
                }
            }
        }
        spatial_focus_next(&chain, &regions, Some(current), dir.focus_direction())
    }

    /// Extent of the panel along `axis`.
    fn span(&self, entries: &[NavEntry], axis: Axis) -> f32 {
        let (min, max) = match axis {
            Axis::X => (
                entries.iter().map(|entry| entry.rect.x).fold(f32::INFINITY, f32::min),
                entries
                    .iter()
                    .map(|entry| entry.rect.x + entry.rect.w)
                    .fold(f32::NEG_INFINITY, f32::max),
            ),
            Axis::Y => (
                entries.iter().map(|entry| entry.rect.y).fold(f32::INFINITY, f32::min),
                entries
                    .iter()
                    .map(|entry| entry.rect.y + entry.rect.h)
                    .fold(f32::NEG_INFINITY, f32::max),
            ),
        };
        max - min
    }
}

fn regions_of(entries: &[NavEntry]) -> (Vec<u64>, Vec<HitRegion>) {
    let chain = entries.iter().map(NavEntry::id).collect();
    let regions = entries
        .iter()
        .map(|entry| HitRegion {
            id: entry.id,
            rect: entry.rect,
            focusable: true,
            focus_group_id: entry.group.map(GroupId::raw),
            ..HitRegion::default()
        })
        .collect();
    (chain, regions)
}

/// Whether `rect` sits ahead of the focused control in the direction of travel,
/// by centre and along the travel axis.
fn is_ahead(rect: Rect, entries: &[NavEntry], current: u64, axis: Axis, sign: f32) -> bool {
    let Some(focused) = entries.iter().find(|entry| entry.id == current) else {
        return false;
    };
    let (ahead, focus) = match axis {
        Axis::X => (rect.x + rect.w / 2.0, focused.rect.x + focused.rect.w / 2.0),
        Axis::Y => (rect.y + rect.h / 2.0, focused.rect.y + focused.rect.h / 2.0),
    };
    (ahead - focus) * sign > 0.0
}

/// Focus traversal paired with a map and a wrap policy: what a game's frame
/// calls once per nav press.
#[derive(Clone)]
pub struct NavDriver {
    map: NavMap,
    wrap: Wrap,
}

impl NavDriver {
    pub fn new(map: NavMap) -> Self {
        Self {
            map,
            wrap: Wrap::default(),
        }
    }

    pub fn with_wrap(mut self, wrap: Wrap) -> Self {
        self.wrap = wrap;
        self
    }

    pub fn wrap(&self) -> Wrap {
        self.wrap
    }

    pub fn map(&self) -> &NavMap {
        &self.map
    }

    /// Resolve `dir` from the scheduler's current focus and request the move.
    ///
    /// The scheduler keeps its old focus until the platform drains the request
    /// queue after this frame's layout, which is the same cadence keyboard focus
    /// has, so one direction never resolves twice against a stale focus.
    pub fn drive(&self, sched: &Scheduler, dir: NavDir) -> NavStep {
        let step = self.map.resolve(sched.focused, dir, self.wrap);
        match step.target() {
            Some(id) if self.map.request(id) => step,
            Some(_) => NavStep::Held,
            None => step,
        }
    }

    /// Enter the panel: pause, menu and modal screens call this on the frame
    /// they open, and safely every frame after.
    ///
    /// Focus already on a control of the map is left alone
    /// ([`NavStep::Already`]), so this never fights the player's own traversal.
    pub fn enter(&self, sched: &Scheduler) -> NavStep {
        if let Some(id) = sched.focused.filter(|id| self.map.contains(*id)) {
            return NavStep::Already(id);
        }
        self.drive(sched, NavDir::Next)
    }
}

/// Analog stick to discrete presses, with a first-press then repeat cadence.
///
/// Pads reach the platform as buttons, so the D-pad and the face buttons
/// already work; the analog stick does not, and a menu navigated with a stick
/// needs edges rather than a level. Deflection is read through
/// [`repame_input::apply_stick`], so the kit's stick and the gameplay sticks
/// share one dead zone.
#[derive(Clone, Copy, Debug)]
pub struct StickNav {
    /// Deflection, after the dead zone, a stick must reach to count as a press.
    threshold: f32,
    /// Delay before the press repeats, in seconds.
    first_delay: f32,
    /// Delay between repeats, in seconds.
    interval: f32,
    held: Option<NavDir>,
    timer: f32,
}

impl Default for StickNav {
    fn default() -> Self {
        Self::new()
    }
}

impl StickNav {
    pub fn new() -> Self {
        Self {
            threshold: 0.55,
            first_delay: 0.35,
            interval: 0.14,
            held: None,
            timer: 0.0,
        }
    }

    /// Set the press threshold, the first-repeat delay and the repeat interval.
    ///
    /// The threshold is clamped to `0.05..=1.0` and the interval to a minimum of
    /// one step at 240 Hz, so a mis-set value cannot make the stick emit twice
    /// per frame.
    pub fn with_cadence(mut self, threshold: f32, first_delay: f32, interval: f32) -> Self {
        self.threshold = if threshold.is_finite() {
            threshold.clamp(0.05, 1.0)
        } else {
            0.55
        };
        self.first_delay = if first_delay.is_finite() {
            first_delay.max(0.0)
        } else {
            0.35
        };
        self.interval = if interval.is_finite() {
            interval.max(1.0 / 240.0)
        } else {
            0.14
        };
        self
    }

    /// The direction the stick is holding down, if any.
    pub fn held(&self) -> Option<NavDir> {
        self.held
    }

    /// Forget the held direction: call when a panel opens or closes so the next
    /// frame starts from a clean edge.
    pub fn release(&mut self) {
        self.held = None;
        self.timer = 0.0;
    }

    /// Advance by `dt_secs` and return a press, if this frame produced one.
    ///
    /// `raw` is the stick's raw axis pair. A diagonal picks the dominant axis;
    /// ties break horizontally.
    pub fn step(&mut self, raw: Vec2, dt_secs: f32) -> Option<NavDir> {
        let dir = direction_of(repame_input::apply_stick(raw), self.threshold);
        match (self.held, dir) {
            (Some(held), Some(dir)) if held == dir => {
                self.timer -= dt_secs.max(0.0);
                if self.timer > 0.0 {
                    return None;
                }
                // Re-arm from now rather than banking presses, so a long frame
                // moves focus once.
                self.timer = self.interval;
                Some(dir)
            }
            (_, None) => {
                self.held = None;
                self.timer = 0.0;
                None
            }
            (_, Some(dir)) => {
                self.held = Some(dir);
                self.timer = self.first_delay;
                Some(dir)
            }
        }
    }
}

fn direction_of(stick: Vec2, threshold: f32) -> Option<NavDir> {
    if stick.length() < threshold {
        return None;
    }
    if stick.x.abs() >= stick.y.abs() {
        Some(if stick.x > 0.0 { NavDir::Right } else { NavDir::Left })
    } else {
        Some(if stick.y > 0.0 { NavDir::Down } else { NavDir::Up })
    }
}

/// The directions a pad's D-pad edges asked for this frame.
pub fn dpad_dirs(pad: &repame_input::GamepadState) -> Vec<NavDir> {
    let mut dirs = Vec::new();
    if pad.dpad_left_pressed {
        dirs.push(NavDir::Left);
    }
    if pad.dpad_right_pressed {
        dirs.push(NavDir::Right);
    }
    if pad.dpad_up_pressed {
        dirs.push(NavDir::Up);
    }
    if pad.dpad_down_pressed {
        dirs.push(NavDir::Down);
    }
    dirs
}

/// Whether the pad's confirm button was pressed this frame.
///
/// The platform maps the south button to keyboard activation (Space), so a game
/// that feeds its pad to the platform does not need this. A game driving its own
/// pad state uses it to activate the focused control.
pub fn pad_confirm(pad: &repame_input::GamepadState) -> bool {
    pad.south_pressed
}

/// The focus chain a map would hand the platform, in order.
pub fn chain_of(map: &NavMap) -> Vec<u64> {
    map.active().iter().map(NavEntry::id).collect()
}

/// Whether `id` is inside the same focus group as `anchor`.
pub fn same_group(map: &NavMap, anchor: u64, id: u64) -> bool {
    let entries = map.active();
    let group_of =
        |id: u64| entries.iter().find(|entry| entry.id == id).and_then(NavEntry::group);
    match (group_of(anchor), group_of(id)) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

/// The traversal repose would perform over a plain chain and rectangle list:
/// the raw entry point, for a game with its own layout.
pub fn resolve_in(
    chain: &[u64],
    regions: &[HitRegion],
    from: Option<u64>,
    dir: NavDir,
) -> Option<u64> {
    spatial_focus_next(chain, regions, from, dir.focus_direction())
}

/// Chain-order traversal with wrap, the rule repose's tab handling uses.
pub fn chain_step(chain: &[u64], from: Option<u64>, dir: NavDir) -> Option<u64> {
    if chain.is_empty() {
        return None;
    }
    let mut manager = FocusManager::new(chain.to_vec(), from);
    manager.move_tab(dir == NavDir::Previous)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three columns by two rows, 200 x 100 each, 10 apart.
    fn grid() -> NavMap {
        let map = NavMap::new();
        map.begin_frame();
        for row in 0..2u64 {
            for column in 0..3u64 {
                let id = row * 10 + column + 1;
                let requester = FocusRequester::new();
                *requester.target.borrow_mut() = Some(id);
                map.mark(id, None, requester.clone());
                map.place(
                    &requester,
                    Rect {
                        x: column as f32 * 210.0,
                        y: row as f32 * 110.0,
                        w: 200.0,
                        h: 100.0,
                    },
                    None,
                );
            }
        }
        map
    }

    fn ids(map: &NavMap) -> Vec<u64> {
        chain_of(map)
    }

    fn drain_requests() {
        while repose_core::runtime::take_focus_request().is_some() {}
    }

    #[test]
    fn controls_are_navigable_in_composition_order() {
        let map = grid();
        assert_eq!(ids(&map), [1, 2, 3, 11, 12, 13]);
        assert_eq!(map.first(), Some(1));
        assert_eq!(map.len(), 6);
    }

    #[test]
    fn nothing_focused_enters_at_the_first_control() {
        let map = grid();
        assert_eq!(map.resolve(None, NavDir::Right, Wrap::Stay), NavStep::Entered(1));
        assert_eq!(map.resolve(None, NavDir::Next, Wrap::Stay), NavStep::Entered(1));
        assert_eq!(map.resolve(None, NavDir::Up, Wrap::Stay), NavStep::Entered(1));
    }

    #[test]
    fn focus_outside_the_map_enters_instead_of_stalling() {
        let map = grid();
        // 999 is the game's overlay root: focusable, in repose's chain, and not
        // a kit control.
        assert_eq!(
            map.resolve(Some(999), NavDir::Right, Wrap::Stay),
            NavStep::Entered(1)
        );
    }

    #[test]
    fn spatial_moves_follow_the_layout() {
        let map = grid();
        assert_eq!(map.resolve(Some(1), NavDir::Right, Wrap::Stay), NavStep::Moved(2));
        assert_eq!(map.resolve(Some(2), NavDir::Right, Wrap::Stay), NavStep::Moved(3));
        assert_eq!(map.resolve(Some(1), NavDir::Down, Wrap::Stay), NavStep::Moved(11));
        assert_eq!(map.resolve(Some(12), NavDir::Up, Wrap::Stay), NavStep::Moved(2));
        assert_eq!(map.resolve(Some(3), NavDir::Left, Wrap::Stay), NavStep::Moved(2));
        assert_eq!(map.resolve(Some(11), NavDir::Right, Wrap::Stay), NavStep::Moved(12));
    }

    #[test]
    fn the_layout_edge_stays_put_when_wrap_is_stay() {
        let map = grid();
        assert_eq!(map.resolve(Some(3), NavDir::Right, Wrap::Stay), NavStep::Edge);
        assert_eq!(map.resolve(Some(1), NavDir::Left, Wrap::Stay), NavStep::Edge);
        assert_eq!(map.resolve(Some(1), NavDir::Up, Wrap::Stay), NavStep::Edge);
        assert_eq!(map.resolve(Some(11), NavDir::Down, Wrap::Stay), NavStep::Edge);
    }

    #[test]
    fn horizontal_wrap_stays_in_the_nearest_row() {
        let map = grid();
        assert_eq!(map.resolve(Some(3), NavDir::Right, Wrap::Cycle), NavStep::Moved(1));
        assert_eq!(map.resolve(Some(1), NavDir::Left, Wrap::Cycle), NavStep::Moved(3));
        assert_eq!(map.resolve(Some(13), NavDir::Right, Wrap::Cycle), NavStep::Moved(11));
        assert_eq!(map.resolve(Some(11), NavDir::Left, Wrap::Cycle), NavStep::Moved(13));
        // From the middle of a row the wrapped pass is not reached: the
        // neighbours are there.
        assert_eq!(map.resolve(Some(2), NavDir::Left, Wrap::Cycle), NavStep::Moved(1));
    }

    #[test]
    fn vertical_wrap_stays_in_the_column() {
        let map = grid();
        assert_eq!(map.resolve(Some(1), NavDir::Up, Wrap::Cycle), NavStep::Moved(11));
        assert_eq!(map.resolve(Some(11), NavDir::Down, Wrap::Cycle), NavStep::Moved(1));
        assert_eq!(map.resolve(Some(13), NavDir::Down, Wrap::Cycle), NavStep::Moved(3));
        assert_eq!(map.resolve(Some(3), NavDir::Up, Wrap::Cycle), NavStep::Moved(13));
    }

    #[test]
    fn a_single_control_has_nowhere_to_go() {
        let map = NavMap::new();
        map.begin_frame();
        let requester = FocusRequester::new();
        *requester.target.borrow_mut() = Some(7);
        map.mark(7, None, requester.clone());
        map.place(&requester, Rect { x: 0.0, y: 0.0, w: 100.0, h: 40.0 }, None);
        assert_eq!(map.resolve(Some(7), NavDir::Right, Wrap::Cycle), NavStep::Edge);
        assert_eq!(map.resolve(Some(7), NavDir::Down, Wrap::Cycle), NavStep::Edge);
        assert_eq!(map.resolve(Some(7), NavDir::Next, Wrap::Cycle), NavStep::Edge);
        assert_eq!(map.resolve(Some(7), NavDir::Previous, Wrap::Cycle), NavStep::Edge);
    }

    #[test]
    fn chain_order_wraps_for_tab_and_shift_tab() {
        let map = grid();
        assert_eq!(map.resolve(Some(1), NavDir::Next, Wrap::Stay), NavStep::Moved(2));
        assert_eq!(map.resolve(Some(13), NavDir::Next, Wrap::Stay), NavStep::Moved(1));
        assert_eq!(map.resolve(Some(1), NavDir::Previous, Wrap::Stay), NavStep::Moved(13));
        assert_eq!(map.resolve(Some(2), NavDir::Previous, Wrap::Stay), NavStep::Moved(1));
    }

    #[test]
    fn a_focus_group_traps_traversal() {
        let map = NavMap::new();
        map.begin_frame();
        let pause = GroupId::named("pause");
        // A HUD button outside the panel, then two panel buttons in a group.
        for (id, x, group) in [
            (1u64, 0.0f32, None),
            (2, 300.0, Some(pause)),
            (3, 600.0, Some(pause)),
        ] {
            let requester = FocusRequester::new();
            *requester.target.borrow_mut() = Some(id);
            map.mark(id, group, requester.clone());
            map.place(&requester, Rect { x, y: 0.0, w: 200.0, h: 100.0 }, group);
        }
        assert_eq!(map.resolve(Some(2), NavDir::Right, Wrap::Stay), NavStep::Moved(3));
        assert_eq!(
            map.resolve(Some(3), NavDir::Right, Wrap::Stay),
            NavStep::Edge,
            "focus does not escape the modal group"
        );
        assert_eq!(map.resolve(Some(3), NavDir::Next, Wrap::Stay), NavStep::Moved(2));
        assert!(same_group(&map, 2, 3));
        assert!(!same_group(&map, 1, 2));
    }

    #[test]
    fn a_closed_panel_is_no_longer_navigable() {
        let map = grid();
        map.begin_frame();
        // Only the HUD survives: nothing composed the grid this frame.
        let requester = FocusRequester::new();
        *requester.target.borrow_mut() = Some(50);
        map.mark(50, None, requester.clone());
        map.place(&requester, Rect { x: 0.0, y: 0.0, w: 40.0, h: 40.0 }, None);
        assert_eq!(ids(&map), [50]);
        assert_eq!(
            map.resolve(Some(1), NavDir::Right, Wrap::Stay),
            NavStep::Entered(50)
        );
    }

    #[test]
    fn an_empty_map_reports_empty() {
        let map = NavMap::new();
        map.begin_frame();
        assert_eq!(map.resolve(None, NavDir::Next, Wrap::Cycle), NavStep::Empty);
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        assert!(!map.request(1));
        assert!(!map.capture(1));
        assert!(!map.release(1));
    }

    #[test]
    fn an_unbound_control_is_not_navigable() {
        let map = NavMap::new();
        map.begin_frame();
        let requester = FocusRequester::new();
        assert!(!map.register(&requester, None), "no view id yet");
        assert!(!map.place(&requester, Rect { x: 0.0, y: 0.0, w: 10.0, h: 10.0 }, None));
        assert!(map.is_empty(), "an unbound rectangle is not a control yet");
    }

    #[test]
    fn the_first_rectangle_is_bound_when_the_view_id_arrives() {
        let map = NavMap::new();
        let requester = FocusRequester::new();
        let rect = Rect { x: 30.0, y: 40.0, w: 200.0, h: 60.0 };

        // Frame 1: layout reports the rectangle before paint assigns the id.
        map.begin_frame();
        assert!(!map.place(&requester, rect, None));
        assert!(!map.register(&requester, None));
        assert!(map.is_empty());

        // Frame 2: the id is known, and the observed rectangle comes with it.
        map.begin_frame();
        *requester.target.borrow_mut() = Some(4);
        assert!(map.place(&requester, rect, None));
        assert!(map.register(&requester, None));
        assert_eq!(ids(&map), [4]);
        assert_eq!(map.active()[0].rect(), rect);

        // Frame 3: layout only reports a change, and registration keeps it.
        map.begin_frame();
        assert!(map.register(&requester, None));
        assert_eq!(map.active()[0].rect(), rect);
        let moved = Rect { x: 60.0, y: 40.0, w: 200.0, h: 60.0 };
        assert!(map.place(&requester, moved, None));
        assert_eq!(map.active()[0].rect(), moved);
    }

    #[test]
    fn a_control_that_stops_composing_takes_its_pending_rectangle_with_it() {
        let map = NavMap::new();
        let requester = FocusRequester::new();
        map.begin_frame();
        assert!(!map.place(&requester, Rect { x: 0.0, y: 0.0, w: 10.0, h: 10.0 }, None));
        // Two frames without the control: the held rectangle is dropped.
        map.begin_frame();
        map.begin_frame();
        *requester.target.borrow_mut() = Some(9);
        assert!(map.register(&requester, None), "registering works, binding does not");
        assert!(map.is_empty(), "the stale rectangle was dropped");
    }

    #[test]
    fn requesting_focus_queues_the_platform_request() {
        drain_requests();
        let map = grid();
        assert!(map.request(2));
        assert!(!map.request(99), "an unknown control is not requested");
        assert_eq!(
            repose_core::runtime::take_focus_request(),
            Some(2),
            "the request reaches the platform's queue"
        );
        drain_requests();
    }

    #[test]
    fn a_captured_modal_refuses_other_targets() {
        drain_requests();
        let map = grid();
        assert!(map.capture(2));
        assert_eq!(map.resolve(Some(2), NavDir::Left, Wrap::Stay), NavStep::Moved(1));
        let driver = NavDriver::new(map.clone());
        assert_eq!(
            driver.drive(&scheduler_focused_on(2), NavDir::Left),
            NavStep::Held,
            "focus is pinned to the captured control"
        );
        assert!(map.release(2));
        drain_requests();
    }

    fn scheduler_focused_on(id: u64) -> Scheduler {
        let mut sched = Scheduler::default();
        sched.focused = Some(id);
        sched
    }

    #[test]
    fn the_driver_enters_and_drives_against_a_live_scheduler() {
        drain_requests();
        let driver = NavDriver::new(grid()).with_wrap(Wrap::Stay);
        let mut sched = Scheduler::default();

        assert_eq!(driver.enter(&sched), NavStep::Entered(1));
        // The platform applies the request at the next frame head.
        sched.focused = Some(1);
        assert_eq!(driver.enter(&sched), NavStep::Already(1));
        assert_eq!(driver.drive(&sched, NavDir::Right), NavStep::Moved(2));

        sched.focused = Some(2);
        assert_eq!(driver.drive(&sched, NavDir::Down), NavStep::Moved(12));

        sched.focused = Some(12);
        assert_eq!(driver.drive(&sched, NavDir::Left), NavStep::Moved(11));
        assert_eq!(driver.drive(&sched, NavDir::Up), NavStep::Moved(1));
        assert_eq!(NavStep::Moved(1).target(), Some(1));
        assert_eq!(NavStep::Edge.target(), None);
        drain_requests();
    }

    #[test]
    fn stick_nav_ignores_rest_and_small_deflection() {
        let mut nav = StickNav::new();
        assert_eq!(nav.step(Vec2::ZERO, 0.1), None);
        assert_eq!(nav.step(Vec2::new(0.15, 0.0), 0.1), None, "inside the dead zone");
        assert_eq!(nav.step(Vec2::new(0.35, 0.0), 0.1), None, "below the press threshold");
        assert_eq!(nav.held(), None);
    }

    #[test]
    fn stick_nav_presses_once_then_repeats() {
        let mut nav = StickNav::new().with_cadence(0.5, 0.35, 0.14);
        assert_eq!(nav.step(Vec2::new(1.0, 0.0), 0.016), Some(NavDir::Right));
        for _ in 0..20 {
            assert_eq!(nav.step(Vec2::new(1.0, 0.0), 0.016), None);
        }
        assert_eq!(nav.step(Vec2::new(1.0, 0.0), 0.05), Some(NavDir::Right));
        assert_eq!(nav.step(Vec2::new(1.0, 0.0), 0.02), None);
        assert_eq!(nav.step(Vec2::new(1.0, 0.0), 0.14), Some(NavDir::Right));
        assert_eq!(nav.step(Vec2::ZERO, 0.5), None);
        assert_eq!(nav.held(), None);
        assert_eq!(nav.step(Vec2::new(1.0, 0.0), 0.016), Some(NavDir::Right));
    }

    #[test]
    fn stick_nav_picks_a_side_and_reverses_on_change() {
        let mut nav = StickNav::new();
        assert_eq!(nav.step(Vec2::new(0.0, -1.0), 0.016), Some(NavDir::Up));
        assert_eq!(nav.step(Vec2::new(0.0, 1.0), 0.016), Some(NavDir::Down));
        assert_eq!(nav.step(Vec2::new(-1.0, 0.0), 0.016), Some(NavDir::Left));
        assert_eq!(nav.step(Vec2::new(1.0, 0.0), 0.016), Some(NavDir::Right));
        assert_eq!(nav.step(Vec2::new(1.0, 0.6), 0.016), Some(NavDir::Right));
        assert_eq!(nav.step(Vec2::new(0.2, 1.0), 0.016), Some(NavDir::Down));
    }

    #[test]
    fn stick_nav_release_forgets_the_held_direction() {
        let mut nav = StickNav::new();
        assert_eq!(nav.step(Vec2::new(1.0, 0.0), 0.016), Some(NavDir::Right));
        nav.release();
        assert_eq!(nav.held(), None);
        assert_eq!(nav.step(Vec2::new(1.0, 0.0), 0.016), Some(NavDir::Right));
    }

    #[test]
    fn a_mis_set_cadence_is_clamped_into_a_sane_band() {
        let mut nav = StickNav::new().with_cadence(9.0, -5.0, 0.0);
        assert_eq!(nav.step(Vec2::new(1.0, 0.0), 0.0), Some(NavDir::Right));
        assert_eq!(
            nav.step(Vec2::new(1.0, 0.0), 0.0),
            Some(NavDir::Right),
            "the repeat interval clamps to one step at 240 Hz, so a held stick repeats"
        );
        assert_eq!(
            nav.step(Vec2::new(0.9, 0.0), 0.0),
            None,
            "the threshold clamps to full deflection"
        );
    }

    #[test]
    fn dpad_edges_map_straight_through() {
        let mut pad = repame_input::GamepadState::default();
        assert!(dpad_dirs(&pad).is_empty());
        assert!(!pad_confirm(&pad));
        pad.dpad_left_pressed = true;
        pad.dpad_up_pressed = true;
        pad.south_pressed = true;
        assert_eq!(dpad_dirs(&pad), [NavDir::Left, NavDir::Up]);
        assert!(pad_confirm(&pad));
    }

    #[test]
    fn group_ids_are_stable_and_name_derived() {
        assert_eq!(GroupId::named("pause"), GroupId::named("pause"));
        assert_ne!(GroupId::named("pause"), GroupId::named("death"));
        assert_eq!(GroupId::from_raw(7).raw(), 7);
        // FNV-1a of the empty string is the offset basis.
        assert_eq!(GroupId::named("").raw(), 0xcbf2_9ce4_8422_2325);
    }

    #[test]
    fn directions_round_trip() {
        for dir in NavDir::ALL {
            assert_eq!(dir.reverse().reverse(), dir);
        }
        assert_eq!(NavDir::Next.reverse(), NavDir::Previous);
        assert_eq!(NavDir::Previous.reverse(), NavDir::Next);
        assert_eq!(NavDir::Left.reverse(), NavDir::Right);
        assert_eq!(NavDir::Up.reverse(), NavDir::Down);
        assert!(NavDir::Next.is_chain());
        assert!(!NavDir::Down.is_chain());
    }

    #[test]
    fn the_raw_repose_helpers_are_usable_without_a_map() {
        let chain = [1, 2, 3];
        let regions: Vec<HitRegion> = (0..3)
            .map(|index| HitRegion {
                id: index + 1,
                rect: Rect {
                    x: index as f32 * 100.0,
                    y: 0.0,
                    w: 90.0,
                    h: 40.0,
                },
                focusable: true,
                ..HitRegion::default()
            })
            .collect();
        assert_eq!(resolve_in(&chain, &regions, Some(1), NavDir::Right), Some(2));
        assert_eq!(chain_step(&chain, Some(1), NavDir::Next), Some(2));
        assert_eq!(chain_step(&chain, Some(3), NavDir::Next), Some(1));
        assert_eq!(chain_step(&[], Some(1), NavDir::Next), None);
    }
}
