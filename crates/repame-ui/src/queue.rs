//! One ordered action queue: the view pushes, the sim drains.
//!
//! All thirteen hand-rolled copies in the games are the same type — a
//! `Sink = Arc<Mutex<Vec<UiAction>>>` plus a `drain_actions(&mut World)` free
//! function plus a boolean `Intents` resource that the drained actions are
//! folded into. This module is that whole pattern once, split into two pieces
//! with fixed semantics:
//!
//! * [`ActionQueue<A>`] and [`Sink<A>`] are two handles on one buffer. The sink
//!   goes into click closures; the queue stays where the frame can drain it.
//!   Order is preserved across any interleaving of pushes and drains, and a
//!   drained batch never contains an action that was pushed after the drain
//!   started.
//! * [`Intents<A>`] is the ordered edge view of the drained batch: ask
//!   [`Intents::take`] for one occurrence of an action and it is consumed, so a
//!   boolean flag per action is no longer needed. Payload-carrying actions go
//!   through [`Intents::take_matching`].
//!
//! Neither type has to be an ECS resource, so this crate carries no `bevy_ecs`
//! dependency: a game puts the handles inside the resource wrapper it already
//! declares (all thirteen copies already have one). A sim system reads them
//! through that resource exactly as it reads any other:
//!
//! ```ignore
//! fn absorb_ui(ui: Res<UiActions>, mut intents: ResMut<UiIntents>) {
//!     repame_ui::absorb(&ui.0, &mut intents.0);
//! }
//! ```
//!
//! or, without a resource at all, a closure system holding the queue:
//!
//! ```ignore
//! let queue = ActionQueue::<UiAction>::new();
//! let sink = queue.sink();
//! sim.add_system(move || {
//!     for action in queue.drain() {
//!         // answer it
//!     }
//! });
//! ```

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

/// An action a UI control can raise.
///
/// Any `Clone + PartialEq + Debug + Send + Sync + 'static` value qualifies, so
/// a game's fieldless `UiAction` enum is one without ceremony. `PartialEq` is
/// what makes [`Intents`] consumption work; `Send`/`Sync` is what lets the sink
/// live in a click closure while the queue is drained from the sim.
pub trait UiAction: Clone + PartialEq + fmt::Debug + Send + Sync + 'static {}

impl<T> UiAction for T where T: Clone + PartialEq + fmt::Debug + Send + Sync + 'static {}

type Shared<A> = Arc<Mutex<Vec<A>>>;

/// Lock the buffer, keeping the queued actions if a pusher panicked while
/// holding the lock.
///
/// Recovering the guard is deliberate: the queued actions are plain data and a
/// lost queue would strand the player in whatever screen pushed last. A
/// poisoned mutex is reported by the panic itself, not by this function.
fn lock<A>(shared: &Shared<A>) -> MutexGuard<'_, Vec<A>> {
    shared.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The push side of an [`ActionQueue`]. Hand it to click closures.
#[derive(Clone)]
pub struct Sink<A: UiAction> {
    shared: Shared<A>,
}

impl<A: UiAction> Sink<A> {
    /// Queue one action. Pushing from the sim side is allowed; order is still
    /// the buffer's.
    pub fn push(&self, action: A) {
        lock(&self.shared).push(action);
    }

    /// Queue several actions in argument order.
    pub fn extend(&self, actions: impl IntoIterator<Item = A>) {
        lock(&self.shared).extend(actions);
    }

    /// Actions waiting to be drained.
    pub fn pending(&self) -> usize {
        lock(&self.shared).len()
    }

    /// Take everything that was queued before this call.
    pub fn drain(&self) -> Vec<A> {
        drain_buffer(&self.shared)
    }
}

impl<A: UiAction> fmt::Debug for Sink<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sink").field("pending", &self.pending()).finish()
    }
}

/// The drain side of an action queue: the same buffer as its [`Sink`].
#[derive(Clone)]
pub struct ActionQueue<A: UiAction> {
    shared: Shared<A>,
}

impl<A: UiAction> ActionQueue<A> {
    pub fn new() -> Self {
        Self {
            shared: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// A fresh queue and the sink for the view that pushes into it.
    pub fn pair() -> (Self, Sink<A>) {
        let queue = Self::new();
        let sink = queue.sink();
        (queue, sink)
    }

    /// The push handle. Cloning it into every control is the intended use.
    pub fn sink(&self) -> Sink<A> {
        Sink {
            shared: Arc::clone(&self.shared),
        }
    }

    /// Queue one action from the drain side.
    pub fn push(&self, action: A) {
        lock(&self.shared).push(action);
    }

    /// Actions waiting to be drained.
    pub fn pending(&self) -> usize {
        lock(&self.shared).len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending() == 0
    }

    /// Take everything queued before this call, in push order.
    pub fn drain(&self) -> Vec<A> {
        drain_buffer(&self.shared)
    }

    /// Take everything queued before this call, appending to `out`.
    pub fn drain_into(&self, out: &mut Vec<A>) {
        let mut buffer = lock(&self.shared);
        out.append(&mut buffer);
    }

    /// Drain into an intent buffer. Returns how many actions were absorbed.
    pub fn absorb(&self, intents: &mut Intents<A>) -> usize {
        intents.extend(self.drain())
    }
}

impl<A: UiAction> Default for ActionQueue<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: UiAction> fmt::Debug for ActionQueue<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ActionQueue")
            .field("pending", &self.pending())
            .finish()
    }
}

fn drain_buffer<A>(shared: &Shared<A>) -> Vec<A> {
    std::mem::take(&mut *lock(shared))
}

/// The actions drained for the current tick, consumed by the systems that
/// answer them.
///
/// Order is the order the controls were clicked. [`take`](Self::take) consumes
/// one occurrence and reports whether it was there, which is what replaces a
/// per-action boolean: `if intents.retry` becomes `if intents.take(&UiAction::Retry)`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Intents<A: UiAction> {
    pending: Vec<A>,
}

impl<A: UiAction> Intents<A> {
    pub fn new() -> Self {
        Self { pending: Vec::new() }
    }

    /// Every pending action in order.
    pub fn requests(&self) -> &[A] {
        &self.pending
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// Forget every pending action without answering any of them.
    pub fn clear(&mut self) {
        self.pending.clear();
    }

    /// Add one request.
    pub fn request(&mut self, action: A) {
        self.pending.push(action);
    }

    /// Add `actions` in argument order. Returns how many were added.
    pub fn extend(&mut self, actions: impl IntoIterator<Item = A>) -> usize {
        let before = self.pending.len();
        self.pending.extend(actions);
        self.pending.len() - before
    }

    /// Whether an occurrence is pending, without consuming it.
    pub fn requested(&self, action: &A) -> bool {
        self.pending.contains(action)
    }

    /// How many occurrences are pending.
    pub fn count(&self, action: &A) -> usize {
        self.pending.iter().filter(|pending| *pending == action).count()
    }

    /// Consume one occurrence, reporting whether it was there.
    pub fn take(&mut self, action: &A) -> bool {
        match self.pending.iter().position(|pending| pending == action) {
            Some(index) => {
                self.pending.remove(index);
                true
            }
            None => false,
        }
    }

    /// Consume the first occurrence matching `predicate`, keeping the rest.
    ///
    /// The way to handle actions that carry a payload, e.g. a selected index:
    /// `intents.take_matching(|action| matches!(action, UiAction::Pick(_)))`.
    pub fn take_matching(&mut self, mut predicate: impl FnMut(&A) -> bool) -> Option<A> {
        let index = self.pending.iter().position(|action| predicate(action))?;
        Some(self.pending.remove(index))
    }

    /// Every pending action in order, leaving the buffer empty.
    pub fn take_all(&mut self) -> Vec<A> {
        std::mem::take(&mut self.pending)
    }
}

impl<A: UiAction> FromIterator<A> for Intents<A> {
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self {
        Self {
            pending: iter.into_iter().collect(),
        }
    }
}

/// Fold a drained batch into an intent buffer.
///
/// The call site a sim system uses when it drains the queue itself:
/// `repame_ui::absorb(&ui.0, &mut intents.0)`.
pub fn absorb<A: UiAction>(queue: &ActionQueue<A>, intents: &mut Intents<A>) -> usize {
    queue.absorb(intents)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Act {
        Play,
        Pause,
        Pick(u8),
    }

    #[test]
    fn drains_in_push_order() {
        let (queue, sink) = ActionQueue::pair();
        sink.push(Act::Play);
        sink.push(Act::Pause);
        sink.push(Act::Play);
        assert_eq!(queue.drain(), vec![Act::Play, Act::Pause, Act::Play]);
    }

    #[test]
    fn drain_empties_the_buffer_once() {
        let (queue, sink) = ActionQueue::pair();
        sink.push(Act::Play);
        assert_eq!(queue.drain(), vec![Act::Play]);
        assert!(queue.drain().is_empty());
        assert!(queue.is_empty());
        assert_eq!(queue.pending(), 0);
    }

    #[test]
    fn a_batch_never_contains_what_arrives_after_the_drain_started() {
        let (queue, sink) = ActionQueue::pair();
        sink.push(Act::Play);
        let batch = queue.drain();
        sink.push(Act::Pause);
        assert_eq!(batch, vec![Act::Play]);
        assert_eq!(queue.drain(), vec![Act::Pause]);
    }

    #[test]
    fn clones_push_into_the_same_buffer() {
        let (queue, sink) = ActionQueue::pair();
        let clone = sink.clone();
        clone.push(Act::Play);
        assert_eq!(sink.pending(), 1);
        assert_eq!(queue.drain(), vec![Act::Play]);
    }

    #[test]
    fn drain_into_appends_to_the_caller_buffer() {
        let (queue, sink) = ActionQueue::pair();
        let mut out = vec![Act::Play];
        sink.push(Act::Pause);
        queue.drain_into(&mut out);
        assert_eq!(out, vec![Act::Play, Act::Pause]);
    }

    #[test]
    fn a_panic_while_pushing_keeps_the_queued_actions() {
        let (queue, sink) = ActionQueue::pair();
        sink.push(Act::Play);
        let shared = Arc::clone(&queue.shared);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _guard = shared.lock().unwrap();
            panic!("pusher panicked while holding the queue");
        }));
        sink.push(Act::Pause);
        assert_eq!(queue.drain(), vec![Act::Play, Act::Pause]);
    }

    #[test]
    fn intents_consume_one_occurrence_at_a_time() {
        let mut intents = Intents::new();
        intents.extend([Act::Play, Act::Pause, Act::Play]);
        assert_eq!(intents.count(&Act::Play), 2);
        assert!(intents.requested(&Act::Play));
        assert!(intents.take(&Act::Play));
        assert!(intents.take(&Act::Play));
        assert!(!intents.take(&Act::Play));
        assert!(!intents.requested(&Act::Play));
        assert!(intents.take(&Act::Pause));
        assert!(intents.is_empty());
    }

    #[test]
    fn take_matching_keeps_the_other_requests() {
        let mut intents = Intents::new();
        intents.extend([Act::Play, Act::Pick(3), Act::Pick(7)]);
        assert_eq!(
            intents.take_matching(|action| matches!(action, Act::Pick(_))),
            Some(Act::Pick(3))
        );
        assert_eq!(intents.requests(), [Act::Play, Act::Pick(7)]);
        assert_eq!(intents.take_matching(|action| matches!(action, Act::Pick(_))), Some(Act::Pick(7)));
        assert!(intents.take_matching(|action| matches!(action, Act::Pick(_))).is_none());
    }

    #[test]
    fn absorb_reports_how_many_actions_arrived() {
        let (queue, sink) = ActionQueue::pair();
        let mut intents = Intents::new();
        assert_eq!(queue.absorb(&mut intents), 0);
        sink.extend([Act::Play, Act::Pause]);
        assert_eq!(queue.absorb(&mut intents), 2);
        assert_eq!(intents.requests(), [Act::Play, Act::Pause]);
        assert_eq!(intents.take_all(), vec![Act::Play, Act::Pause]);
        assert!(intents.is_empty());
    }
}
