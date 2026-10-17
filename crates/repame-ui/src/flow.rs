//! Menu and run flow on `repame_sim::state`, not on a per-game `Phase` enum.
//!
//! Every template-derived game hand-rolls the same three things: a `Phase` enum
//! (`Menu` / `Playing` / `Break(...)`), a `Panel` enum for which menu is
//! frontmost, and a `RunState` that re-derives the same four questions —
//! *playing*, *modal*, *transitional*, *suspended*. This module is the flow
//! half of that once, as three state sets with enter/exit/transition schedules:
//!
//! * [`Flow`] — the screen: front end, live run, player pause, rule-driven
//!   modal.
//! * [`Transition`] — a wipe/curtain is running over the run.
//! * [`Suspension`] — the platform took the game away (app backgrounded, window
//!   gone); the run is frozen but not paused.
//!
//! Because these are real states, `OnEnter(Flow::Paused)` and
//! `OnExit(Transition::Running)` are where a game opens its pause panel and
//! stops its wipe sound, instead of an `if` inside the frame loop. Which panel
//! is frontmost stays a game resource: the kit cannot know the game's panels.
//!
//! [`FlowState`] is the readout the per-game `RunState` was re-deriving. It is
//! plain data, so the view side can hold one and the sim side can build one from
//! the world. `run_live` and `panel_owns_input` are also run conditions.

use repame_sim::Sim;
use repame_sim::bevy_ecs::prelude::{Res, World};
use repame_sim::state::State;

/// The screen the game is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Flow {
    /// Front end: title, options, level select. No run in flight.
    Menu,
    /// A run is live and answers input.
    Playing,
    /// The player froze the run; a pause panel owns input.
    Paused,
    /// A game rule froze the run; a modal panel owns input (death screen,
    /// revive offer, upgrade pick, results).
    Modal,
}

impl Flow {
    /// Short name for logs and diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            Flow::Menu => "menu",
            Flow::Playing => "playing",
            Flow::Paused => "paused",
            Flow::Modal => "modal",
        }
    }

    /// Whether a run is still in flight behind whatever is on screen.
    ///
    /// `Menu` is the only flow with nothing to resume.
    pub fn keeps_run(self) -> bool {
        !matches!(self, Flow::Menu)
    }
}

/// Whether a wipe, curtain or other full-screen transition is running.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Transition {
    Idle,
    Running,
}

/// Whether the platform has suspended gameplay.
///
/// Distinct from [`Flow::Paused`]: a suspend is not a player decision, so it
/// must not open a pause panel or resume into one. Set it from the app's
/// foreground/background listener.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Suspension {
    Live,
    Suspended,
}

/// Register the flow state sets on a sim, before any `on_enter`/`on_exit`/
/// `on_transition` call for them.
///
/// ```ignore
/// repame_ui::flow::install(&mut sim, Flow::Menu);
/// sim.on_enter(Flow::Playing, start_run);
/// ```
pub fn install(sim: &mut Sim, start: Flow) -> &mut Sim {
    sim.add_state(start);
    sim.add_state(Transition::Idle);
    sim.add_state(Suspension::Live);
    sim
}

/// Who is answering input this frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputOwner {
    /// Gameplay: sticks, buttons, the touch overlay.
    Game,
    /// A panel: menus, the pause screen, a modal.
    Panel,
}

/// The flow distinctions a per-game `RunState` re-derives, as one value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlowState {
    flow: Flow,
    transition: Transition,
    suspension: Suspension,
}

impl FlowState {
    /// The front end with nothing running: what a game that has not installed
    /// the flow reads as, so a mis-wired frame freezes the game instead of
    /// running it behind a menu.
    pub const CLOSED: FlowState =
        FlowState::new(Flow::Menu, Transition::Idle, Suspension::Live);

    pub const fn new(flow: Flow, transition: Transition, suspension: Suspension) -> Self {
        Self {
            flow,
            transition,
            suspension,
        }
    }

    /// Read the flow state sets out of a sim world.
    ///
    /// `None` when [`install`] was never called, so a caller cannot mistake
    /// "not wired" for `Flow::Menu`.
    pub fn from_world(world: &World) -> Option<Self> {
        Some(Self {
            flow: *world.get_resource::<State<Flow>>()?.get(),
            transition: *world.get_resource::<State<Transition>>()?.get(),
            suspension: *world.get_resource::<State<Suspension>>()?.get(),
        })
    }

    pub const fn flow(&self) -> Flow {
        self.flow
    }

    pub const fn transition(&self) -> Transition {
        self.transition
    }

    pub const fn suspension(&self) -> Suspension {
        self.suspension
    }

    /// The screen is the front end.
    pub const fn menu(&self) -> bool {
        matches!(self.flow, Flow::Menu)
    }

    /// A run is live, whatever overlays are up.
    pub const fn in_run(&self) -> bool {
        matches!(self.flow, Flow::Playing)
    }

    /// The player paused the run.
    pub const fn paused(&self) -> bool {
        matches!(self.flow, Flow::Paused)
    }

    /// A blocking panel owns the screen: the pause panel or a game modal.
    ///
    /// The condition the game's own overlay and the ECS gameplay systems both
    /// gate on, so a modal owned by the sim cannot be half-drawn.
    pub const fn modal(&self) -> bool {
        matches!(self.flow, Flow::Paused | Flow::Modal)
    }

    /// A wipe or curtain is running.
    pub const fn transitional(&self) -> bool {
        matches!(self.transition, Transition::Running)
    }

    /// The platform suspended gameplay.
    pub const fn suspended(&self) -> bool {
        matches!(self.suspension, Suspension::Suspended)
    }

    /// The world should advance this step: a live run, no transition, no
    /// suspend. The gameplay schedule's gate.
    pub const fn run_live(&self) -> bool {
        self.in_run() && !self.transitional() && !self.suspended()
    }

    /// A panel (menu, pause, modal) owns the screen.
    pub const fn panel_owns_input(&self) -> bool {
        self.menu() || self.modal()
    }

    /// The game answers sticks and buttons: a live run that is neither
    /// transitioning nor suspended.
    ///
    /// The touch overlay's gate.
    pub const fn game_has_input(&self) -> bool {
        self.run_live()
    }

    pub const fn input_owner(&self) -> InputOwner {
        if self.panel_owns_input() {
            InputOwner::Panel
        } else {
            InputOwner::Game
        }
    }
}

/// Run condition: the world should advance this step.
///
/// ```ignore
/// sim.add_chained_systems(movement.run_if(flow::run_live));
/// ```
pub fn run_live(
    flow: Res<State<Flow>>,
    transition: Res<State<Transition>>,
    suspension: Res<State<Suspension>>,
) -> bool {
    FlowState::new(*flow.get(), *transition.get(), *suspension.get()).run_live()
}

/// Run condition: a menu, pause or modal panel owns the screen.
pub fn panel_owns_input(
    flow: Res<State<Flow>>,
    transition: Res<State<Transition>>,
    suspension: Res<State<Suspension>>,
) -> bool {
    FlowState::new(*flow.get(), *transition.get(), *suspension.get()).panel_owns_input()
}

#[cfg(test)]
mod tests {
    use super::*;
    use repame_sim::bevy_ecs::schedule::IntoScheduleConfigs;
    use repame_sim::in_state;

    const FLOWS: [Flow; 4] = [Flow::Menu, Flow::Playing, Flow::Paused, Flow::Modal];
    const TRANSITIONS: [Transition; 2] = [Transition::Idle, Transition::Running];
    const SUSPENSIONS: [Suspension; 2] = [Suspension::Live, Suspension::Suspended];

    /// One row per `(flow, transition, suspension)`: menu, in_run, paused,
    /// modal, transitional, suspended, run_live, panel_owns_input, owner.
    fn expected(
        flow: Flow,
        transition: Transition,
        suspension: Suspension,
    ) -> (bool, bool, bool, bool, bool, bool, bool, bool, InputOwner) {
        let transitional = transition == Transition::Running;
        let suspended = suspension == Suspension::Suspended;
        (
            flow == Flow::Menu,
            flow == Flow::Playing,
            flow == Flow::Paused,
            matches!(flow, Flow::Paused | Flow::Modal),
            transitional,
            suspended,
            flow == Flow::Playing && !transitional && !suspended,
            matches!(flow, Flow::Menu | Flow::Paused | Flow::Modal),
            if matches!(flow, Flow::Menu | Flow::Paused | Flow::Modal) {
                InputOwner::Panel
            } else {
                InputOwner::Game
            },
        )
    }

    #[test]
    fn the_predicates_agree_with_the_truth_table() {
        for flow in FLOWS {
            for transition in TRANSITIONS {
                for suspension in SUSPENSIONS {
                    let state = FlowState::new(flow, transition, suspension);
                    let row = expected(flow, transition, suspension);
                    let case = format!("({flow:?}, {transition:?}, {suspension:?})");
                    assert_eq!(state.menu(), row.0, "menu {case}");
                    assert_eq!(state.in_run(), row.1, "in_run {case}");
                    assert_eq!(state.paused(), row.2, "paused {case}");
                    assert_eq!(state.modal(), row.3, "modal {case}");
                    assert_eq!(state.transitional(), row.4, "transitional {case}");
                    assert_eq!(state.suspended(), row.5, "suspended {case}");
                    assert_eq!(state.run_live(), row.6, "run_live {case}");
                    assert_eq!(state.panel_owns_input(), row.7, "panel_owns_input {case}");
                    assert_eq!(state.input_owner(), row.8, "owner {case}");
                    assert_eq!(state.game_has_input(), row.6, "game_has_input {case}");
                }
            }
        }
    }

    #[test]
    fn a_suspend_does_not_become_a_pause() {
        let suspended = FlowState::new(Flow::Playing, Transition::Idle, Suspension::Suspended);
        assert!(suspended.suspended());
        assert!(!suspended.paused());
        assert!(!suspended.modal());
        assert!(!suspended.run_live());
        assert_eq!(suspended.input_owner(), InputOwner::Game);
    }

    #[test]
    fn menu_is_the_only_flow_without_a_run_behind_it() {
        assert!(!Flow::Menu.keeps_run());
        assert!(Flow::Playing.keeps_run());
        assert!(Flow::Paused.keeps_run());
        assert!(Flow::Modal.keeps_run());
    }

    #[test]
    fn state_schedules_fire_in_enter_exit_order() {
        use std::sync::{Arc, Mutex};

        let log: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
        let mut sim = Sim::with_default_step();
        install(&mut sim, Flow::Menu);

        let (a, b, c) = (log.clone(), log.clone(), log.clone());
        sim.on_exit(Flow::Menu, move || a.lock().unwrap().push("exit:menu"));
        sim.on_enter(Flow::Playing, move || b.lock().unwrap().push("enter:playing"));
        sim.on_transition(Flow::Menu, Flow::Playing, move || {
            c.lock().unwrap().push("menu->playing")
        });

        sim.set_state(Flow::Playing);
        let state = FlowState::from_world(&sim.world).expect("install registered the flow");
        assert!(state.menu(), "the transition applies at the step head");
        sim.tick();
        assert_eq!(
            *log.lock().unwrap(),
            ["exit:menu", "menu->playing", "enter:playing"]
        );
        let state = FlowState::from_world(&sim.world).expect("install registered the flow");
        assert!(state.run_live());
    }

    #[test]
    fn run_live_gates_a_system_through_the_sim() {
        use std::sync::atomic::{AtomicU32, Ordering};
        static TICKS: AtomicU32 = AtomicU32::new(0);

        let mut sim = Sim::with_default_step();
        install(&mut sim, Flow::Menu);
        sim.add_chained_systems(
            (|| {
                TICKS.fetch_add(1, Ordering::SeqCst);
            })
            .run_if(run_live),
        );

        sim.tick();
        assert_eq!(TICKS.load(Ordering::SeqCst), 0, "menu does not advance the world");

        sim.set_state(Flow::Playing);
        sim.tick();
        assert_eq!(TICKS.load(Ordering::SeqCst), 1);

        sim.set_state(Transition::Running);
        sim.tick();
        assert_eq!(TICKS.load(Ordering::SeqCst), 1, "a wipe freezes the world");

        sim.set_state(Transition::Idle);
        sim.set_state(Suspension::Suspended);
        sim.tick();
        assert_eq!(TICKS.load(Ordering::SeqCst), 1, "a suspend freezes the world");

        sim.set_state(Suspension::Live);
        sim.tick();
        assert_eq!(TICKS.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn states_work_through_in_state_too() {
        use std::sync::atomic::{AtomicU32, Ordering};
        static HITS: AtomicU32 = AtomicU32::new(0);

        let mut sim = Sim::with_default_step();
        install(&mut sim, Flow::Menu);
        sim.add_chained_systems(
            (|| {
                HITS.fetch_add(1, Ordering::SeqCst);
            })
            .run_if(in_state(Flow::Playing)),
        );
        sim.set_state(Flow::Playing);
        sim.tick();
        assert_eq!(HITS.load(Ordering::SeqCst), 1);
        sim.set_state(Flow::Paused);
        sim.tick();
        assert_eq!(HITS.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_missing_install_reads_as_none() {
        let sim = Sim::with_default_step();
        assert!(FlowState::from_world(&sim.world).is_none());
    }
}
