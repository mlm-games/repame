//! The suspend gate: one decision per frame, out of the run state the game
//! knows and the facts the portal knows.
//!
//! The games that ship a portal integration hand-roll this at every call site
//! that touches the clock:
//!
//! ```ignore
//! let suspended = self.platform.suspends_gameplay() && !run.transitional();
//! let live = run.playing() && !run.transitional() && !suspended;
//! self.sim.time_scale = if suspended { 0.0 } else { 1.0 };
//! if live { schedule.run(&mut world); }
//! ```
//!
//! [`Suspend`] is those three lines once, as a pure function of both halves, so
//! the truth table is testable without a backend and every call site reads the
//! same way. [`Suspend::apply`] is the only place `Sim::time_scale` is written.
//!
//! `repame-fx`'s `SimDriver::frame(step_fx, blocked, tick_sim, clear_edges)`
//! encodes the other half of the same split: it sequences an ungated fx step
//! against a gated gameplay tick and takes the block as an argument.
//! [`Suspend::suspended`] is that argument (the gate decides, the driver
//! sequences), so the two compose instead of overlapping. `SimDriver` is
//! stateless and knows nothing about run state, ads, modals, mute or lifecycle,
//! which is exactly what this module is.

use std::fmt;

use repame_sim::Sim;

/// What only the game knows, settled per frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GameState {
    /// The run is playing: not on a menu, not in a break, not paused. The host
    /// is told gameplay is running while this holds and nothing else does:
    /// see [`Suspend::host_gameplay`].
    pub playing: bool,
    /// A transition is in flight: a run transition, or the game's own break
    /// screen while it waits for an ad.
    ///
    /// The clock runs on through it whichever way the rest of the gate falls,
    /// because a transition that cannot tick can never unblock itself; the
    /// transition is what ends it. Gameplay systems stay off.
    pub transitional: bool,
    /// A game-side modal owns the screen and nothing animates under it (a pause
    /// menu, a settings panel). A modal that animates is a transition instead.
    pub modal: bool,
    /// The player muted the game in its own settings. Audio only: a mute is not
    /// a reason to stop the world.
    pub muted: bool,
}

/// What the portal knows, as a plain value so the truth table is testable
/// without a backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PortalView {
    /// An ad was accepted and has not finished or timed out: hold gameplay even
    /// before the host opens it, or the game plays under the ad's own load.
    pub ad_pending: bool,
    /// An ad is on screen.
    pub ad_visible: bool,
    /// The host paused the game (a platform pause, not the game's own menu).
    pub host_paused: bool,
    /// The page or app is in the background.
    pub backgrounded: bool,
    /// The host's audio switch is on. `false` means the host's own mute, which
    /// is not the player's, see [`GameState::muted`].
    pub audio_enabled: bool,
}

impl Default for PortalView {
    /// Nothing known, audio on: the view of a game with no host, which is also
    /// the one that suspends nothing.
    fn default() -> Self {
        Self {
            ad_pending: false,
            ad_visible: false,
            host_paused: false,
            backgrounded: false,
            audio_enabled: true,
        }
    }
}

/// Why the clock is stopped, outermost first.
///
/// Several can hold at once (an ad opens in a backgrounded tab) and the first
/// one that applies is the reported cause: the outermost reason is the one
/// that explains why the game is neither running nor visible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cause {
    /// The page or app is in the background.
    Backgrounded,
    /// The host paused the game.
    HostPaused,
    /// An ad is on screen, or is on its way.
    Ad,
    /// A game-side modal owns the screen.
    Modal,
}

impl fmt::Display for Cause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Backgrounded => "the app is in the background",
            Self::HostPaused => "the host paused the game",
            Self::Ad => "an ad is on screen",
            Self::Modal => "a menu owns the screen",
        })
    }
}

/// One frame's answer to "does gameplay run, does the world tick, does sound
/// play".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Suspend {
    suspended: bool,
    live: bool,
    silenced: bool,
    host_gameplay: bool,
    cause: Option<Cause>,
}

impl Suspend {
    /// Settle one frame from both halves of the gate.
    pub fn resolve(game: GameState, portal: PortalView) -> Self {
        // An ad holds the game from the moment the host accepts it: from then
        // on the frame the game draws is about to be covered.
        let ad = portal.ad_pending || portal.ad_visible;
        // The host took the screen away: it paused us, or the app is not on
        // screen at all.
        let away = portal.host_paused || portal.backgrounded;
        let cause = if portal.backgrounded {
            Some(Cause::Backgrounded)
        } else if portal.host_paused {
            Some(Cause::HostPaused)
        } else if ad {
            Some(Cause::Ad)
        } else if game.modal {
            Some(Cause::Modal)
        } else {
            None
        };
        // Gameplay systems run only when a run is playing and nothing holds it
        // back: a transition, a game-side modal, an ad or a host-side pause
        // covers the screen the same way, and running spawners, timers and
        // input edges under any of them is a bug the games kept shipping.
        let live = game.playing && !game.transitional && !game.modal && !ad && !away;
        Self {
            // A transition runs on the raw clock whichever way this falls: the
            // transition itself is what ends it.
            suspended: cause.is_some() && !game.transitional,
            live,
            // Sound stops for the player's own mute and whenever the game is
            // not the thing on screen.
            silenced: game.muted || !portal.audio_enabled || ad || away,
            // The host counts impressions from time it believes is gameplay. A
            // transition or a modal is the game's own screen and is not the
            // host's business, so neither is reported: what is reported is
            // exactly when gameplay systems run.
            host_gameplay: live,
            cause: if game.transitional { None } else { cause },
        }
    }

    /// The clock is stopped this frame, so the sim must not tick. This is the
    /// `blocked` argument of `repame-fx`'s `SimDriver::frame`.
    pub fn suspended(&self) -> bool {
        self.suspended
    }

    /// Gameplay systems run this frame (the games' `live` flag): a run is
    /// playing and nothing covers it, not a transition, a modal, an ad or the
    /// host. The world still ticks under a transition and the clock still runs,
    /// so this is not the negation of [`Suspend::suspended`].
    pub fn live(&self) -> bool {
        self.live
    }

    /// Audio must be silent this frame.
    pub fn silenced(&self) -> bool {
        self.silenced
    }

    /// Whether the host should count gameplay as running this frame: the same
    /// answer as [`Suspend::live`], because a transition or a modal is the
    /// game's own screen and impression counts are contractual.
    pub fn host_gameplay(&self) -> bool {
        self.host_gameplay
    }

    /// The outermost reason the clock is stopped, `None` when it runs.
    pub fn cause(&self) -> Option<Cause> {
        self.cause
    }

    /// The clock this decision means: `0.0` while suspended, `1.0` otherwise.
    ///
    /// A stopped clock is not a banked one: `Sim::step` drops the wall time it
    /// would have accumulated, so resuming after a long ad does not run a burst
    /// of catch-up ticks.
    pub fn time_scale(&self) -> f32 {
        if self.suspended { 0.0 } else { 1.0 }
    }

    /// Write the decision to the sim clock.
    pub fn apply(&self, sim: &mut Sim) {
        sim.time_scale = self.time_scale();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn game(playing: bool, transitional: bool, modal: bool, muted: bool) -> GameState {
        GameState {
            playing,
            transitional,
            modal,
            muted,
        }
    }

    fn portal(
        ad_pending: bool,
        ad_visible: bool,
        host_paused: bool,
        backgrounded: bool,
    ) -> PortalView {
        PortalView {
            ad_pending,
            ad_visible,
            host_paused,
            backgrounded,
            ..Default::default()
        }
    }

    /// Every combination of the five gate inputs and the four game flags, held
    /// to the rules the field docs state.
    #[test]
    fn the_truth_table_is_the_same_rule_everywhere() {
        for bits in 0..512u32 {
            let game_state = game(bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0);
            let view = PortalView {
                ad_pending: bits & 16 != 0,
                ad_visible: bits & 32 != 0,
                host_paused: bits & 64 != 0,
                backgrounded: bits & 128 != 0,
                audio_enabled: bits & 256 != 0,
            };
            let decision = Suspend::resolve(game_state, view);

            let ad = view.ad_pending || view.ad_visible;
            let away = view.host_paused || view.backgrounded;
            let live =
                game_state.playing && !game_state.transitional && !game_state.modal && !ad && !away;
            let held = ad || away || game_state.modal;
            let reported = {
                if view.backgrounded {
                    Some(Cause::Backgrounded)
                } else if view.host_paused {
                    Some(Cause::HostPaused)
                } else if ad {
                    Some(Cause::Ad)
                } else if game_state.modal {
                    Some(Cause::Modal)
                } else {
                    None
                }
            };

            assert_eq!(
                decision.suspended(),
                held && !game_state.transitional,
                "clock at {bits:#011b}"
            );
            assert_eq!(
                decision.cause(),
                if game_state.transitional {
                    None
                } else {
                    reported
                },
                "cause at {bits:#011b}"
            );
            assert_eq!(decision.live(), live, "gameplay at {bits:#011b}");
            assert_eq!(
                decision.silenced(),
                game_state.muted || !view.audio_enabled || ad || away,
                "sound at {bits:#011b}"
            );
            assert_eq!(
                decision.host_gameplay(),
                live,
                "host gameplay at {bits:#011b}"
            );
            assert_eq!(
                decision.time_scale(),
                if decision.suspended() { 0.0 } else { 1.0 },
                "clock scale at {bits:#011b}"
            );
        }
    }

    #[test]
    fn a_transition_keeps_the_clock_for_every_holder() {
        for view in [
            portal(true, false, false, false),
            portal(false, true, false, false),
            portal(false, false, true, false),
            portal(false, false, false, true),
        ] {
            let decision = Suspend::resolve(game(true, true, false, false), view);
            assert!(
                !decision.suspended(),
                "a transition must be able to finish: {view:?}"
            );
            assert_eq!(decision.time_scale(), 1.0);
            assert!(!decision.live(), "gameplay stays off under a transition");
        }
    }

    #[test]
    fn a_modal_keeps_the_clock_and_the_transition_rule_is_not_a_modal() {
        let menu = Suspend::resolve(game(false, false, true, false), PortalView::default());
        assert!(menu.suspended());
        assert_eq!(menu.cause(), Some(Cause::Modal));
        assert_eq!(menu.time_scale(), 0.0);
        assert!(!menu.live());
    }

    #[test]
    fn the_default_view_suspends_nothing_and_silences_nothing() {
        let decision = Suspend::resolve(GameState::default(), PortalView::default());
        assert!(!decision.suspended() && !decision.silenced());
        assert_eq!(decision.time_scale(), 1.0);
        assert!(!decision.live(), "a default state is not a playing run");
        assert!(!decision.host_gameplay());
    }

    #[test]
    fn apply_writes_the_clock_and_a_stopped_clock_is_dropped_not_banked() {
        let mut sim = Sim::with_default_step();
        Suspend::resolve(
            game(true, false, false, false),
            portal(true, false, false, false),
        )
        .apply(&mut sim);
        assert_eq!(sim.time_scale, 0.0);
        Suspend::resolve(game(true, false, false, false), PortalView::default()).apply(&mut sim);
        assert_eq!(sim.time_scale, 1.0);
    }
}
