//! Typed game states with enter/exit/transition schedules.
//!
//! A state set is any small value type (usually an enum) that is `Eq + Hash`;
//! one `State<S>` resource holds the current value and one `NextState<S>`
//! holds a pending change. Requesting a transition from any system is a write
//! to `NextState<S>`; the sim applies it once per fixed step, *before* the
//! step's main schedule, so a transition never interleaves with gameplay.
//!
//! Apply order for `A -> B` matches the reference (bevy_state):
//!
//! 1. `OnExit(A)`   - the `State<S>` resource still reads `A`
//! 2. `State<S> = B`
//! 3. `OnTransition { exited: A, entered: B }`
//! 4. `OnEnter(B)`
//!
//! If an enter/exit system requests another transition, it is resolved on the
//! same step, up to [`MAX_CHAIN`] links; beyond that the chain is dropped and
//! the last state stands. That bound stops a self-triggering transition from
//! spinning the step forever.
//!
//! Determinism: transitions run inside the fixed step and in registration
//! order, so a replay that requests the same transitions at the same ticks
//! resolves them identically. Do not request transitions from a thread.

use std::collections::HashMap;
use std::hash::Hash;

use bevy_ecs::prelude::*;
use bevy_ecs::schedule::{IntoScheduleConfigs, Schedule};
use bevy_ecs::system::ScheduleSystem;

/// A state set: a plain value used as a game state.
///
/// Usually a fieldless enum. The blanket impl covers any `Clone + Eq + Hash`
/// value type with no `'static` borrows.
pub trait States: Clone + Eq + Hash + Send + Sync + 'static {}

impl<T> States for T where T: Clone + Eq + Hash + Send + Sync + 'static {}

/// The current value of a state set.
#[derive(Resource)]
pub struct State<S: States>(S);

impl<S: States> State<S> {
    pub fn new(value: S) -> Self {
        Self(value)
    }

    pub fn get(&self) -> &S {
        &self.0
    }
}

/// A pending transition. Writing here requests a change; the sim applies it
/// before the next step's main schedule.
#[derive(Resource)]
pub struct NextState<S: States>(Option<S>);

impl<S: States> NextState<S> {
    pub fn new() -> Self {
        Self(None)
    }

    /// Request a transition. A request equal to the current state is ignored.
    pub fn set(&mut self, next: S) {
        self.0 = Some(next);
    }

    /// The pending request, if any.
    pub fn pending(&self) -> Option<&S> {
        self.0.as_ref()
    }

    fn take(&mut self) -> Option<S> {
        self.0.take()
    }
}

impl<S: States> Default for NextState<S> {
    fn default() -> Self {
        Self(None)
    }
}

/// Transition schedules, keyed by the state they fire for.
#[derive(Resource)]
pub(crate) struct StateSchedules<S: States> {
    on_enter: HashMap<S, Schedule>,
    on_exit: HashMap<S, Schedule>,
    on_transition: HashMap<(S, S), Schedule>,
}

impl<S: States> Default for StateSchedules<S> {
    fn default() -> Self {
        Self {
            on_enter: HashMap::new(),
            on_exit: HashMap::new(),
            on_transition: HashMap::new(),
        }
    }
}

/// Most transition links a single step resolves before giving up.
pub const MAX_CHAIN: u32 = 8;

/// Applies pending transitions. Registered once per state set; runs at the
/// head of each fixed step.
pub(crate) fn apply<S: States>(world: &mut World) {
    for _ in 0..MAX_CHAIN {
        let Some(next) = world.resource_mut::<NextState<S>>().take() else {
            break;
        };
        let current = world.resource::<State<S>>().get().clone();
        if next == current {
            continue;
        }
        world.resource_scope(|world, mut schedules: Mut<StateSchedules<S>>| {
            if let Some(schedule) = schedules.on_exit.get_mut(&current) {
                schedule.run(world);
            }
            world.resource_mut::<State<S>>().0 = next.clone();
            if let Some(schedule) = schedules.on_transition.get_mut(&(current, next.clone())) {
                schedule.run(world);
            }
            if let Some(schedule) = schedules.on_enter.get_mut(&next) {
                schedule.run(world);
            }
        });
    }
}

/// Run condition: only run the configured systems while the state is `state`.
///
/// ```ignore
/// sim.add_chained_systems(update_walk.run_if(in_state(Mode::Walk)));
/// ```
pub fn in_state<S: States>(state: S) -> impl FnMut(Res<State<S>>) -> bool + Clone + Send + Sync + 'static {
    move |current: Res<State<S>>| current.get() == &state
}

pub(crate) fn init_schedules<S: States>(world: &mut World) {
    world.insert_resource(StateSchedules::<S>::default());
}

pub(crate) fn add_enter<S, M>(
    world: &mut World,
    state: S,
    systems: impl IntoScheduleConfigs<ScheduleSystem, M>,
) where
    S: States,
{
    world
        .resource_mut::<StateSchedules<S>>()
        .on_enter
        .entry(state)
        .or_default()
        .add_systems(systems);
}

pub(crate) fn add_exit<S, M>(
    world: &mut World,
    state: S,
    systems: impl IntoScheduleConfigs<ScheduleSystem, M>,
) where
    S: States,
{
    world
        .resource_mut::<StateSchedules<S>>()
        .on_exit
        .entry(state)
        .or_default()
        .add_systems(systems);
}

pub(crate) fn add_transition<S, M>(
    world: &mut World,
    exited: S,
    entered: S,
    systems: impl IntoScheduleConfigs<ScheduleSystem, M>,
) where
    S: States,
{
    world
        .resource_mut::<StateSchedules<S>>()
        .on_transition
        .entry((exited, entered))
        .or_default()
        .add_systems(systems);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    enum Mode {
        Menu,
        Play,
        Paused,
    }

    #[test]
    fn transition_runs_exit_then_enter_in_order() {
        use std::sync::{Arc, Mutex};

        let log: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
        let mut sim = crate::Sim::with_default_step();
        sim.add_state(Mode::Menu);

        let (a, b, c) = (log.clone(), log.clone(), log.clone());
        sim.on_exit(Mode::Menu, move || a.lock().unwrap().push("exit:menu"));
        sim.on_enter(Mode::Play, move || b.lock().unwrap().push("enter:play"));
        sim.on_transition(Mode::Menu, Mode::Play, move || {
            c.lock().unwrap().push("transition:menu->play")
        });

        sim.set_state(Mode::Play);
        sim.tick();
        assert_eq!(
            *log.lock().unwrap(),
            ["exit:menu", "transition:menu->play", "enter:play"]
        );
        assert_eq!(*sim.world.resource::<State<Mode>>().get(), Mode::Play);
    }

    #[test]
    fn exit_sees_old_state_and_enter_sees_new_state() {
        use std::sync::{Arc, Mutex};

        let seen: Arc<Mutex<Vec<Mode>>> = Arc::new(Mutex::new(Vec::new()));
        let mut sim = crate::Sim::with_default_step();
        sim.add_state(Mode::Menu);

        let e = seen.clone();
        sim.on_exit(Mode::Menu, move |s: Res<State<Mode>>| e.lock().unwrap().push(*s.get()));
        let n = seen.clone();
        sim.on_enter(Mode::Play, move |s: Res<State<Mode>>| n.lock().unwrap().push(*s.get()));

        sim.set_state(Mode::Play);
        sim.tick();
        assert_eq!(*seen.lock().unwrap(), [Mode::Menu, Mode::Play]);
    }

    #[test]
    fn a_transition_requested_on_enter_chains_on_the_same_step() {
        let mut sim = crate::Sim::with_default_step();
        sim.add_state(Mode::Menu);
        sim.on_enter(Mode::Play, |mut next: ResMut<NextState<Mode>>| {
            next.set(Mode::Paused)
        });
        sim.set_state(Mode::Play);
        sim.tick();
        assert_eq!(*sim.world.resource::<State<Mode>>().get(), Mode::Paused);
    }

    #[test]
    fn self_triggering_transition_is_bounded() {
        let mut sim = crate::Sim::with_default_step();
        sim.add_state(Mode::Menu);
        sim.on_enter(Mode::Play, |mut next: ResMut<NextState<Mode>>| {
            next.set(Mode::Play)
        });
        sim.set_state(Mode::Play);
        sim.tick();
        assert_eq!(*sim.world.resource::<State<Mode>>().get(), Mode::Play);
        assert!(sim.world.resource::<NextState<Mode>>().pending().is_none());
    }

    #[test]
    fn set_to_current_state_runs_no_schedules() {
        use std::sync::{Arc, Mutex};
        let fired = Arc::new(Mutex::new(false));
        let mut sim = crate::Sim::with_default_step();
        sim.add_state(Mode::Menu);
        let f = fired.clone();
        sim.on_enter(Mode::Menu, move || *f.lock().unwrap() = true);
        sim.set_state(Mode::Menu);
        sim.tick();
        assert!(!*fired.lock().unwrap());
    }

    #[test]
    fn in_state_condition_gates_systems() {
        use std::sync::atomic::{AtomicU32, Ordering};
        static HITS: AtomicU32 = AtomicU32::new(0);

        let mut sim = crate::Sim::with_default_step();
        sim.add_state(Mode::Menu);
        sim.add_chained_systems(
            (|| {
                HITS.fetch_add(1, Ordering::SeqCst);
            })
            .run_if(in_state(Mode::Menu)),
        );
        sim.tick();
        assert_eq!(HITS.load(Ordering::SeqCst), 1);
        sim.set_state(Mode::Play);
        sim.tick();
        assert_eq!(HITS.load(Ordering::SeqCst), 1, "gated out after switching");
    }
}
