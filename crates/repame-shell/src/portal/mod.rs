//! Portal integration: ads, cloud saves, host pause, lifecycle mute and the
//! host-facing gameplay notices, in one place.
//!
//! A portal is a host a game runs inside (Playgama, Yandex Games, CrazyGames,
//! and the rest). Six games in this family ship a byte-identical
//! `sim/platform.rs` and an `app/platform.rs` that differs between two of them
//! only in three log strings: ads with timeouts, cloud-save push and pull, the
//! host's pause, the lifecycle mute, and a `time_scale` gate copy-pasted into
//! every frame loop that touches the clock. This module is that code, once.
//!
//! # The three pieces
//!
//! - [`Portal`] is one backend: the adapter that speaks to a host. The Playgama
//!   one lives in `portal::playgama` behind the `playgama` feature and is off by
//!   default. [`NoPortal`] is the desktop backend, and it refuses every request
//!   with a typed reason instead of pretending.
//! - [`PortalHost`] is the state machine the game owns. It asks the backend,
//!   absorbs what the backend reports, owns the ad deadline and the rate limit
//!   the host enforces, and answers one question per frame: does gameplay run,
//!   does the world tick, does sound play.
//! - [`Suspend`] is that answer, and [`Suspend::apply`] is the only place
//!   `Sim::time_scale` is written.
//!
//! # The no-fabrication rule
//!
//! A host that cannot serve an ad or store a cloud save says so **now**, with a
//! typed [`Refusal`]: nothing waits on a callback that will not come, and
//! nothing reports a success it did not get. Every request is answered on the
//! frame it was made, and everything that was taken ends in an [`AdOutcome`] or
//! a cloud answer:
//!
//! - an ad that is not supported, or that another break is already in the way
//!   of, is refused before it starts ([`AdRefused`]);
//! - an ad the host accepted and never ended is released by its deadline, and
//!   reported as [`AdOutcome::TimedOut`]: a break the host forgot cannot hold
//!   the player;
//! - a rewarded ad pays only when the host reported the reward point, never on
//!   the strength of having asked ([`AdOutcome::pays`]);
//! - an empty or non-text cloud payload is refused rather than stored, because
//!   the platform reads an empty value as a delete;
//! - with no host at all, [`NoPortal`] still answers every request, with
//!   [`Refusal::NoPortal`], rather than going quiet.
//!
//! # What a game writes
//!
//! ```ignore
//! // Once, at boot: the web after the SDK loaded, the desktop plain.
//! let mut portal = PortalHost::pull_first(Box::new(backend));
//! portal.loaded();
//!
//! // Once per frame, in this order.
//! portal.update(dt);                                 // one clock for ad deadlines
//! let gate = portal.gate(GameState {
//!     playing: run.playing(),
//!     transitional: run.transitional(),
//!     modal: false,
//!     muted: save.data.muted,
//! });
//! gate.apply(&mut sim);                              // the one writer of time_scale
//! if gate.live() { schedule.run(&mut sim.world); }
//! portal.ready();                                    // first playable frame: game_ready
//! for report in portal.take_reports() { /* ad outcomes, cloud answers */ }
//! ```
//!
//! The gate replaces three hand-written lines that the games keep at every call
//! site that touches the clock (`live`, `suspended`, `time_scale`), and
//! [`Suspend::live`] replaces the fourth that decides whether the gameplay
//! schedule runs at all.
//!
//! `repame-fx`'s `SimDriver::frame(step_fx, blocked, tick_sim, clear_edges)`
//! encodes the sequencing half of the same problem (step ungated fx first, then
//! tick gameplay only when open, and roll the input edges when it is not).
//! [`Suspend::suspended`] is exactly its `blocked` argument, so the two
//! compose: the gate decides, the driver sequences. The driver has no opinion
//! about run state, ads, modals, mute or lifecycle, and it does not own the
//! clock, which is why the gate is here and not there.

mod gate;
mod host;
pub mod lifecycle;
mod none;
#[cfg(feature = "playgama")]
pub mod playgama;
mod protocol;

pub use gate::{Cause, GameState, PortalView, Suspend};
pub use host::{AdOutcome, AdRefused, CloudRefused, PortalHost, PortalReport};
pub use none::NoPortal;
pub use protocol::{
    AdEnding, AdKind, AdSchedule, Availability, HostNotice, INTERSTITIAL_TIMEOUT, Portal,
    PortalEvent, PortalRequest, PortalStatus, REWARDED_TIMEOUT, Refusal,
};
