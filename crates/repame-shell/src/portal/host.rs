//! The shell's portal state machine: the one object a game drives each frame.
//!
//! A game never talks to a host directly. It hands [`PortalHost`] a backend
//! (Playgama on the web, [`NoPortal`] on the desktop), drives it once per
//! frame, and reads two things back: the frame's [`Suspend`] gate and the
//! [`PortalReport`]s the host decided since the last call.
//!
//! ```ignore
//! // Once, at boot.
//! let mut portal = PortalHost::pull_first(backend);   // or PortalHost::none()
//! portal.loaded();                                    // the assets are up
//!
//! // Once per frame, before the sim steps.
//! portal.update(dt);                                  // one clock for ad deadlines
//! let gate = portal.gate(GameState {
//!     playing: run.playing(),
//!     transitional: run.transitional(),
//!     modal: false,
//!     muted: save.data.muted,
//! });
//! gate.apply(&mut sim);                               // the one writer of time_scale
//! if gate.live() {
//!     schedule.run(&mut sim.world);
//! }
//! portal.ready();                                     // first playable frame: game_ready
//! for report in portal.take_reports() {
//!     match report {
//!         PortalReport::AdFinished { kind: AdKind::Rewarded, outcome } if outcome.pays(kind) => revive(),
//!         PortalReport::CloudPulled(Ok(Some(bytes))) => save.apply_cloud(&bytes),
//!         _ => {}
//!     }
//! }
//! ```
//!
//! What the host owns and the game does not:
//!
//! - one ad at a time, and the deadline that releases the game when the host
//!   never ends it ([`INTERSTITIAL_TIMEOUT`], [`REWARDED_TIMEOUT`]);
//! - the reward rule (a rewarded ad pays only on [`AdOutcome::Completed`],
//!   which needs the host's reward point, never the ask);
//! - the interstitial rate limit, mirrored from the game's bridge config so a
//!   refused ad is a typed [`AdRefused::TooSoon`] instead of a request the host
//!   ignores;
//! - the lifecycle read and the host-facing gameplay notices.
//!
//! The shell never holds the player: a break the host never ends is released on
//! its deadline and reported as [`AdOutcome::TimedOut`], and a request the host
//! cannot serve is refused on the spot. Both are visible to the game: nothing
//! here can wait on a callback that will not come, and nothing reports a
//! success it did not get.
//!
//! [`INTERSTITIAL_TIMEOUT`]: super::INTERSTITIAL_TIMEOUT
//! [`REWARDED_TIMEOUT`]: super::REWARDED_TIMEOUT

use std::fmt;

use web_time::Duration;

use super::gate::{GameState, PortalView, Suspend};
use super::lifecycle;
use super::none::NoPortal;
use super::protocol::{
    AdEnding, AdKind, AdSchedule, HostNotice, Portal, PortalEvent, PortalRequest, PortalStatus,
    Refusal,
};

/// Why an ad the game asked for is not on its way.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdRefused {
    /// Another break is already running: one at a time.
    Busy,
    /// The host's interstitial rate limit has not elapsed. `remaining` is how
    /// much of it is left.
    TooSoon { remaining: Duration },
    /// The portal would not take the request at all.
    Refusal(Refusal),
}

impl fmt::Display for AdRefused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Busy => f.write_str("another break is already running"),
            Self::TooSoon { remaining } => {
                write!(f, "the host's interstitial delay has {remaining:?} left")
            }
            Self::Refusal(reason) => write!(f, "{reason}"),
        }
    }
}

/// Why a cloud write did not go to the host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CloudRefused {
    /// No cloud read has resolved yet, and this host was built with
    /// [`PortalHost::pull_first`]: the first push of a session must not race
    /// the read that would have said what the player already has.
    BeforePull,
    /// The portal would not take the write at all.
    Refusal(Refusal),
}

impl fmt::Display for CloudRefused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BeforePull => f.write_str("no cloud read has resolved yet"),
            Self::Refusal(reason) => write!(f, "{reason}"),
        }
    }
}

/// What an ad did, after the shell has matched the host's events to the
/// request that asked for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdOutcome {
    /// The ad reached its end. An interstitial is complete once the host
    /// reports it closed; a rewarded ad is complete once it reported the
    /// reward, whatever ends it afterwards. The reward is due.
    Completed,
    /// The ad closed before it completed: a rewarded ad was closed by the
    /// player before the host reported the reward. Nothing is owed.
    Skipped,
    /// The host never reported the ad's end before its deadline, so the shell
    /// released the game. Nothing is owed, and the host's later word on that ad
    /// is ignored: a break cannot be held open on a report that may never come.
    TimedOut { after: Duration },
    /// The host could not play the ad: it gave up before the video ran.
    /// Nothing is owed, and asking again is the game's call.
    Failed,
}

impl AdOutcome {
    /// Whether a `kind` ad pays a reward on this outcome.
    pub fn pays(&self, kind: AdKind) -> bool {
        kind.is_rewarded() && matches!(self, Self::Completed)
    }

    /// Whether the ad reached its end, whatever its kind.
    pub fn completed(&self) -> bool {
        matches!(self, Self::Completed)
    }
}

impl fmt::Display for AdOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Completed => f.write_str("completed"),
            Self::Skipped => f.write_str("skipped"),
            Self::TimedOut { after } => write!(f, "timed out after {after:?}"),
            Self::Failed => f.write_str("failed"),
        }
    }
}

/// One thing the shell decided, for the game to drain once per frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PortalReport {
    /// An ad opened and covers the screen. One the game did not ask for is
    /// reported too: the host can break in on its own.
    AdOpened { kind: AdKind },
    /// The ad finished. A rewarded ad pays only when `outcome.pays(kind)`.
    AdFinished { kind: AdKind, outcome: AdOutcome },
    /// The host's audio switch moved.
    AudioEnabled(bool),
    /// The host paused or resumed the game.
    HostPaused(bool),
    /// A cloud read landed: `Ok(None)` when the host had nothing stored under
    /// the key, which is not a failure.
    CloudPulled(Result<Option<Vec<u8>>, Refusal>),
    /// A cloud write landed, or the reason it did not. A refused write leaves
    /// the game's save dirty, and pushing again is the game's call.
    CloudStored(Result<(), Refusal>),
    /// A gameplay notice the host would not take. The host counts impressions
    /// from these, so a dropped one is reported rather than lost.
    NoticeDropped { notice: HostNotice, reason: Refusal },
}

/// One ad in flight, from the moment the host took the request.
#[derive(Clone, Copy, Debug)]
struct Pending {
    kind: AdKind,
    /// Time left before the shell releases the game.
    deadline: Duration,
    /// The host reported the reward point.
    rewarded: bool,
    /// The host reported the ad on screen.
    visible: bool,
}

/// The shell's half of a portal: requests out, host facts in, one gate.
pub struct PortalHost {
    portal: Box<dyn Portal>,
    reports: Vec<PortalReport>,
    pending: Option<Pending>,
    audio_enabled: bool,
    host_paused: bool,
    backgrounded: bool,
    /// Time since the game loaded, for the host's interstitial rate limit.
    since_interstitial: Duration,
    /// An interstitial has been on screen, so the rate limit between two of
    /// them now applies instead of the initial one.
    interstitial_shown: bool,
    schedule: AdSchedule,
    /// Whether the host is currently told gameplay is running, so only a change
    /// sends a notice.
    announced_gameplay: bool,
    /// The load finished and `game_ready` is waiting for the first playable
    /// frame.
    ready_pending: bool,
    ready_sent: bool,
    /// A cloud read has resolved: see [`PortalHost::cloud_ready`] for what
    /// counts.
    cloud_read: bool,
    /// Refuse cloud pushes until a read resolved.
    pull_first: bool,
}

impl PortalHost {
    /// A host over `portal`. A cloud push may be sent as soon as the game has
    /// something to store.
    pub fn new(portal: Box<dyn Portal>) -> Self {
        lifecycle::watch();
        let status = portal.status();
        Self {
            portal,
            reports: Vec::new(),
            pending: None,
            audio_enabled: status.audio_enabled,
            host_paused: status.host_paused,
            backgrounded: lifecycle::backgrounded(),
            since_interstitial: Duration::ZERO,
            interstitial_shown: false,
            schedule: status.interstitial_schedule,
            announced_gameplay: false,
            ready_pending: false,
            ready_sent: false,
            cloud_read: false,
            pull_first: false,
        }
    }

    /// A host over `portal` that refuses cloud pushes until a read has
    /// resolved.
    ///
    /// A game that loads its save and autosaves on every change wants this: a
    /// push that lands before the read would store a fresh default over the
    /// save the player came back for.
    pub fn pull_first(portal: Box<dyn Portal>) -> Self {
        Self {
            pull_first: true,
            ..Self::new(portal)
        }
    }

    /// The host a build without a portal gets: every request is refused with
    /// [`Refusal::NoPortal`], nothing hangs, and the gate suspends only for what
    /// the game itself reports.
    pub fn none() -> Self {
        Self::new(Box::new(NoPortal))
    }

    /// The backend's name, for logs.
    pub fn label(&self) -> &str {
        self.portal.label()
    }

    /// What the host can do right now.
    pub fn status(&self) -> PortalStatus {
        self.portal.status()
    }

    /// The portal's half of the gate.
    pub fn view(&self) -> PortalView {
        PortalView {
            ad_pending: self.pending.is_some(),
            ad_visible: self.ad_visible(),
            host_paused: self.host_paused,
            backgrounded: self.backgrounded,
            audio_enabled: self.audio_enabled,
        }
    }

    /// The load finished. The host is told on the first playable frame, see
    /// [`PortalHost::ready`], because a host that removes its loading overlay
    /// on `game_ready` must remove it over a frame the player can see.
    pub fn loaded(&mut self) {
        self.ready_pending = true;
    }

    /// The first playable frame is up: send `game_ready`, once.
    ///
    /// Returns whether this call sent it. `false` means there was nothing to
    /// send: a second call, or one made before [`PortalHost::loaded`]. The host
    /// rejects a second `game_ready`, and nothing in the bridge can report that
    /// rejection, so this never sends one.
    pub fn ready(&mut self) -> Result<bool, Refusal> {
        if !self.ready_pending || self.ready_sent {
            return Ok(false);
        }
        self.send(HostNotice::GameReady)?;
        self.ready_sent = true;
        self.ready_pending = false;
        Ok(true)
    }

    /// Tell the host something itself. The gate sends the gameplay notices on
    /// its own; this is for a game that has something else to say.
    pub fn notify(&mut self, notice: HostNotice) -> Result<(), Refusal> {
        self.send(notice)
    }

    /// Advance the portal by one frame: read the lifecycle, hand the backend's
    /// inbox over, and release an ad that outlived its deadline.
    ///
    /// Call once per frame, suspended or not: the ad deadline has to run while
    /// the game is held by the ad.
    pub fn update(&mut self, dt: Duration) {
        self.backgrounded = lifecycle::backgrounded();
        self.since_interstitial += dt;
        let events = self.portal.poll();
        for event in events {
            self.absorb(event);
        }
        let mut released = None;
        if let Some(pending) = &mut self.pending {
            pending.deadline = pending.deadline.saturating_sub(dt);
            if pending.deadline.is_zero() {
                released = Some(pending.kind);
            }
        }
        if let Some(kind) = released {
            log::warn!(
                "{}: a {kind} ad did not finish within {:?}; releasing the game",
                self.portal.label(),
                kind.timeout()
            );
            self.finish(
                kind,
                AdOutcome::TimedOut {
                    after: kind.timeout(),
                },
            );
        }
    }

    /// Settle this frame's gate, and tell the host when the answer changed.
    pub fn gate(&mut self, game: GameState) -> Suspend {
        let decision = Suspend::resolve(game, self.view());
        if decision.host_gameplay() != self.announced_gameplay {
            self.announced_gameplay = decision.host_gameplay();
            let notice = if decision.host_gameplay() {
                HostNotice::GameplayStarted
            } else {
                HostNotice::GameplayStopped
            };
            let _ = self.send(notice);
        }
        decision
    }

    /// Everything the shell decided since the last call.
    pub fn take_reports(&mut self) -> Vec<PortalReport> {
        std::mem::take(&mut self.reports)
    }

    /// Ask the host for an ad. `placement` names a placement from the game's
    /// config; `None` asks for the one the config declares.
    ///
    /// A taken request holds gameplay the next time the gate is settled, see
    /// [`PortalHost::ad_pending`], and ends in a
    /// [`PortalReport::AdFinished`] whichever way the host takes it.
    ///
    /// A refused rewarded ad pays nothing. A game that wants to be generous
    /// where no ad can run (the games' desktop rule: an offline rewarded break
    /// always pays out) grants the reward from that refusal itself, because the
    /// policy is the game's and not the portal's.
    pub fn show_ad(&mut self, kind: AdKind, placement: Option<&str>) -> Result<(), AdRefused> {
        if self.pending.is_some() {
            return Err(AdRefused::Busy);
        }
        if kind == AdKind::Interstitial {
            let wait = if self.interstitial_shown {
                self.schedule.minimum_delay
            } else {
                self.schedule.initial_delay
            };
            if self.since_interstitial < wait {
                return Err(AdRefused::TooSoon {
                    remaining: wait - self.since_interstitial,
                });
            }
        }
        let placement = placement.map(str::to_string);
        self.portal
            .request(PortalRequest::ShowAd { kind, placement })
            .map_err(AdRefused::Refusal)?;
        self.pending = Some(Pending {
            kind,
            deadline: kind.timeout(),
            rewarded: false,
            visible: false,
        });
        Ok(())
    }

    /// Whether an ad is in flight: taken, and not finished or released yet.
    pub fn ad_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Whether the host reported an ad on screen.
    pub fn ad_visible(&self) -> bool {
        self.pending.is_some_and(|pending| pending.visible)
    }

    /// The kind of the ad in flight, when there is one.
    pub fn ad_kind(&self) -> Option<AdKind> {
        self.pending.map(|pending| pending.kind)
    }

    /// Ask for the stored save. The answer arrives as
    /// [`PortalReport::CloudPulled`].
    pub fn cloud_pull(&mut self) -> Result<(), Refusal> {
        self.portal.request(PortalRequest::CloudPull)
    }

    /// Store this payload under the game's cloud key. The answer arrives as
    /// [`PortalReport::CloudStored`].
    ///
    /// The payload is opaque here, and the game's save type stays out of the
    /// portal: what goes up is what comes back from
    /// [`PortalReport::CloudPulled`], byte for byte, so a game hands over its
    /// own serialized save and parses it back itself.
    pub fn cloud_push(&mut self, payload: &[u8]) -> Result<(), CloudRefused> {
        if self.pull_first && !self.cloud_read {
            return Err(CloudRefused::BeforePull);
        }
        self.portal
            .request(PortalRequest::CloudPush {
                payload: payload.to_vec(),
            })
            .map_err(CloudRefused::Refusal)
    }

    /// Whether a cloud read has resolved: the host answered, nothing stored
    /// counting as an answer, or it has no cloud at all.
    ///
    /// A read the host refused is not an answer: it may have been transient,
    /// and the save it would have returned may be up there.
    pub fn cloud_ready(&self) -> bool {
        self.cloud_read
    }

    fn send(&mut self, notice: HostNotice) -> Result<(), Refusal> {
        let outcome = self.portal.request(PortalRequest::Notify(notice));
        if let Err(reason) = &outcome {
            self.reports.push(PortalReport::NoticeDropped {
                notice,
                reason: reason.clone(),
            });
        }
        outcome
    }

    fn absorb(&mut self, event: PortalEvent) {
        match event {
            PortalEvent::AudioEnabled(enabled) => {
                self.audio_enabled = enabled;
                self.reports.push(PortalReport::AudioEnabled(enabled));
            }
            PortalEvent::HostPaused(paused) => {
                self.host_paused = paused;
                self.reports.push(PortalReport::HostPaused(paused));
            }
            PortalEvent::AdOpened(kind) => self.ad_opened(kind),
            PortalEvent::AdRewarded => self.ad_rewarded(),
            PortalEvent::AdEnded { kind, ending } => self.ad_ended(kind, ending),
            PortalEvent::CloudPulled(outcome) => {
                match &outcome {
                    Ok(_) => self.cloud_read = true,
                    // A cloud that does not exist cannot be overwritten, so it
                    // counts as resolved: a game that pushes anyway gets the
                    // backend's own answer instead of `BeforePull`.
                    Err(Refusal::NoPortal | Refusal::NotSupported) => self.cloud_read = true,
                    // A host refusal may be transient, and a push now could
                    // store over a save that was there all along.
                    Err(_) => {}
                }
                self.reports.push(PortalReport::CloudPulled(outcome));
            }
            PortalEvent::CloudStored(outcome) => {
                self.reports.push(PortalReport::CloudStored(outcome));
            }
        }
    }

    fn ad_opened(&mut self, kind: AdKind) {
        if kind == AdKind::Interstitial {
            // The impression is what the SDK's minimum delay counts from.
            self.interstitial_shown = true;
            self.since_interstitial = Duration::ZERO;
        }
        match &mut self.pending {
            Some(pending) => pending.visible = true,
            None => {
                // A break the game did not ask for still covers the frame, so
                // it is held like an asked-for one, and it is still released by
                // a deadline rather than held forever.
                log::info!(
                    "{}: a {kind} ad opened without a request; holding gameplay until it ends",
                    self.portal.label()
                );
                self.pending = Some(Pending {
                    kind,
                    deadline: kind.timeout(),
                    rewarded: false,
                    visible: true,
                });
            }
        }
        self.reports.push(PortalReport::AdOpened { kind });
    }

    fn ad_rewarded(&mut self) {
        match &mut self.pending {
            Some(pending) if pending.kind.is_rewarded() => pending.rewarded = true,
            Some(pending) => log::warn!(
                "{}: a {} ad reported a reward; nothing is owed",
                self.portal.label(),
                pending.kind
            ),
            None => log::warn!(
                "{}: a reward was reported with no ad running; nothing is owed",
                self.portal.label()
            ),
        }
    }

    fn ad_ended(&mut self, kind: AdKind, ending: AdEnding) {
        let Some(pending) = self.pending.take() else {
            log::warn!(
                "{}: an ad ended with none running; ignored",
                self.portal.label()
            );
            return;
        };
        if pending.kind != kind {
            log::warn!(
                "{}: a {kind} ad ended while a {} ad was running; releasing the running one",
                self.portal.label(),
                pending.kind
            );
        }
        let outcome = match ending {
            // The reward point is the host's promise, so whatever ends the ad
            // afterwards the reward stands. Without it, a rewarded ad that just
            // closed paid nothing.
            AdEnding::Closed if pending.kind.is_rewarded() && !pending.rewarded => {
                AdOutcome::Skipped
            }
            AdEnding::Closed => AdOutcome::Completed,
            AdEnding::Failed if pending.rewarded => AdOutcome::Completed,
            AdEnding::Failed => AdOutcome::Failed,
        };
        self.finish(pending.kind, outcome);
    }

    fn finish(&mut self, kind: AdKind, outcome: AdOutcome) {
        self.pending = None;
        self.reports
            .push(PortalReport::AdFinished { kind, outcome });
    }
}

impl Default for PortalHost {
    fn default() -> Self {
        Self::none()
    }
}
