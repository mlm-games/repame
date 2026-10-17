//! The wire between the shell and a portal: what the shell asks for, what a
//! portal reports back, and the typed answers for the cases where the host
//! cannot deliver.
//!
//! The split is deliberate. A backend reports what the **host** said (an ad
//! opened, the audio switch moved, a cloud read landed) and never what it
//! means; [`PortalHost`](super::PortalHost) decides what it means (whether an
//! interrupted ad pays, whether the clock runs, what the game is told). A new
//! backend is then a translation layer, and a test backend only has to produce
//! host facts.
//!
//! Nothing here is fire-and-forget on the way in: a request is either taken
//! (`Ok`) or refused **now**, with a [`Refusal`] that says why, and everything
//! taken ends in a [`PortalEvent`]. A portal that cannot serve a call says so
//! instead of letting the game wait for an answer that never comes.

use std::fmt;

use web_time::Duration;

/// How long the shell waits for an interstitial before releasing the game.
///
/// The games' `BREAK_TIMEOUT`: a host that never reports the ad's end must not
/// freeze the game, so the deadline ends the break whichever way it went.
pub const INTERSTITIAL_TIMEOUT: Duration = Duration::from_secs(10);

/// How long the shell waits for a rewarded video before releasing the game.
///
/// The games' `REWARD_TIMEOUT`, longer than [`INTERSTITIAL_TIMEOUT`] because a
/// rewarded video is watched to the end on purpose.
pub const REWARDED_TIMEOUT: Duration = Duration::from_secs(30);

/// The two ads a portal serves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AdKind {
    /// A full-screen ad between rounds. Nothing is owed either way, and the
    /// host may also refuse it by its own rate limit.
    Interstitial,
    /// A video the player chose to watch for a reward. The reward is due only
    /// once the host reports the reward point, never on the strength of having
    /// asked for one.
    Rewarded,
}

impl AdKind {
    /// The deadline for this kind, see [`INTERSTITIAL_TIMEOUT`] and
    /// [`REWARDED_TIMEOUT`].
    pub fn timeout(self) -> Duration {
        match self {
            Self::Interstitial => INTERSTITIAL_TIMEOUT,
            Self::Rewarded => REWARDED_TIMEOUT,
        }
    }

    /// Whether this kind pays a reward. Only [`AdKind::Rewarded`] does.
    pub fn is_rewarded(self) -> bool {
        matches!(self, Self::Rewarded)
    }
}

impl fmt::Display for AdKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Interstitial => "interstitial",
            Self::Rewarded => "rewarded",
        })
    }
}

/// Why a portal would not take a request, or cannot hand back what it has.
///
/// Every one of these is an answer the caller gets on the frame it asked: a
/// host without ads or without a cloud produces one of these rather than a
/// callback that never fires.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// Nothing is listening: no portal backend is installed (a plain desktop
    /// build), or the bridge that was handed over is not talking to a live
    /// host.
    NoPortal,
    /// The host has the feature but reports it unsupported here, which is the
    /// SDK's `is_interstitial_supported` / `is_rewarded_supported` being false
    /// and its bare `Promise.reject()` for the rest.
    NotSupported,
    /// A request of the same kind is already in flight and the SDK drops a
    /// second one instead of queueing it, so a request nobody would answer is
    /// refused instead of sent.
    InFlight,
    /// The host refused the call on its own terms. `code` is the code text the
    /// platform sent, known to the bridge or not; it is `None` when the
    /// platform rejected without naming one.
    Host {
        code: Option<String>,
        message: String,
    },
    /// The bytes are not text the host can keep, or the value stored under the
    /// key is not text. A platform's cloud keeps text, and a payload it cannot
    /// hold is refused rather than mangled.
    NotText,
    /// The payload is empty. The platform's cloud reads an empty value as a
    /// **delete**, so storing nothing would drop the player's save.
    EmptyPayload,
}

impl Refusal {
    /// The host's own code, when it gave one.
    pub fn code(&self) -> Option<&str> {
        match self {
            Self::Host { code, .. } => code.as_deref(),
            _ => None,
        }
    }
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoPortal => f.write_str("no portal is installed"),
            Self::NotSupported => f.write_str("the host does not support this"),
            Self::InFlight => f.write_str("a request of the same kind is already in flight"),
            Self::Host {
                code: Some(code),
                message,
            } => write!(f, "{code}: {message}"),
            Self::Host {
                code: None,
                message,
            } => f.write_str(message),
            Self::NotText => f.write_str("the cloud keeps text, and these bytes are not text"),
            Self::EmptyPayload => {
                f.write_str("an empty payload is a delete for the platform's cloud")
            }
        }
    }
}

/// Whether a portal can serve one feature, with the host's own reason when it
/// cannot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Availability {
    Available,
    Unavailable(Refusal),
}

impl Availability {
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available)
    }

    /// Why it is unavailable, or `None` when it is available.
    pub fn reason(&self) -> Option<&Refusal> {
        match self {
            Self::Available => None,
            Self::Unavailable(reason) => Some(reason),
        }
    }
}

impl fmt::Display for Availability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Available => f.write_str("available"),
            Self::Unavailable(reason) => write!(f, "unavailable: {reason}"),
        }
    }
}

/// The interstitial rate limit the host enforces, as the game's bridge config
/// declares it (`minimumDelayBetweenInterstitial`, `initialInterstitialDelay`).
///
/// The shell mirrors the host's rule so a game gets [`AdRefused::TooSoon`] and
/// the time left, instead of a request the host ignores while the shell waits
/// out the ad deadline for an ad that was never going to play.
///
/// [`AdRefused::TooSoon`]: super::AdRefused::TooSoon
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdSchedule {
    /// No interstitial before this much time has passed since the game loaded.
    pub initial_delay: Duration,
    /// No two interstitials closer together than this.
    pub minimum_delay: Duration,
}

/// A notice the host tracks gameplay by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostNotice {
    /// The first playable frame is up. The SDK removes its loading overlay on
    /// this one and rejects a second, so it is sent exactly once.
    GameReady,
    /// Gameplay is live; the host counts impressions from here.
    GameplayStarted,
    /// Gameplay is not live: the run ended, or an ad, a host pause or the
    /// background stopped it.
    GameplayStopped,
}

impl fmt::Display for HostNotice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::GameReady => "game_ready",
            Self::GameplayStarted => "gameplay_started",
            Self::GameplayStopped => "gameplay_stopped",
        })
    }
}

/// One request the shell hands a portal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PortalRequest {
    /// Tell the host where gameplay stands.
    Notify(HostNotice),
    /// Show an ad. The outcome arrives as [`PortalEvent::AdEnded`], or as the
    /// shell's deadline when the host never ends it.
    ShowAd {
        kind: AdKind,
        /// `None` asks for the placement the game's config declares.
        placement: Option<String>,
    },
    /// Read the stored save.
    CloudPull,
    /// Write the stored save.
    CloudPush { payload: Vec<u8> },
}

/// The host's last word on an ad, before the shell turns it into an outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdEnding {
    /// The ad ended (the SDK's `closed`).
    Closed,
    /// The host gave up on the ad (the SDK's `failed`).
    Failed,
}

/// One thing a portal learned from the host.
///
/// This is the backend's side of the wire: host facts, in the host's own
/// terms. The shell owns the translation into what the game is told, see
/// [`PortalReport`](super::PortalReport).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PortalEvent {
    /// The host's audio switch moved. The value at start is in
    /// [`PortalStatus`], because a host only reports changes.
    AudioEnabled(bool),
    /// The host paused or resumed the game.
    HostPaused(bool),
    /// An ad opened. One the game did not ask for still covers the frame, so
    /// it still holds gameplay.
    AdOpened(AdKind),
    /// A rewarded ad reached its reward point. The host owes the reward from
    /// here whatever ends the ad.
    AdRewarded,
    /// The ad ended, in the host's own words.
    AdEnded { kind: AdKind, ending: AdEnding },
    /// A cloud read landed. `Ok(None)` is "the host has nothing stored under
    /// the key", which is not a refusal.
    CloudPulled(Result<Option<Vec<u8>>, Refusal>),
    /// A cloud write landed, or the reason it did not: the game should keep
    /// the save dirty and push again.
    CloudStored(Result<(), Refusal>),
}

/// What a portal can do, and the host facts it knows without being asked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortalStatus {
    /// `false` when nothing is listening: a desktop build, or a bridge that
    /// never came up. Everything else in here is then a refusal for the same
    /// reason, so a game can hide a rewarded button without probing.
    pub hosted: bool,
    /// The host's audio switch. Read at start as well as subscribed to: the
    /// hosts only report changes.
    pub audio_enabled: bool,
    /// The host paused the game. Read at start for the same reason.
    pub host_paused: bool,
    pub interstitial: Availability,
    pub rewarded: Availability,
    pub cloud: Availability,
    /// The interstitial rate limit the host enforces.
    pub interstitial_schedule: AdSchedule,
}

impl Default for PortalStatus {
    /// Nothing is listening and every feature refuses for that reason, which
    /// is what a build without a portal reports.
    fn default() -> Self {
        Self {
            hosted: false,
            audio_enabled: true,
            host_paused: false,
            interstitial: Availability::Unavailable(Refusal::NoPortal),
            rewarded: Availability::Unavailable(Refusal::NoPortal),
            cloud: Availability::Unavailable(Refusal::NoPortal),
            interstitial_schedule: AdSchedule::default(),
        }
    }
}

/// One portal backend: the adapter between the shell and one host.
///
/// A backend translates host callbacks into [`PortalEvent`]s and queues them
/// for [`Portal::poll`]; it decides nothing. Requests are taken or refused on
/// the spot, so a caller never waits on a host that has nothing to answer.
///
/// The listeners a backend subscribes to belong to the host, so they outlive
/// the call that installed them: a backend must own what its listeners touch
/// (`Rc`), and must be the only writer. A backend is installed for the life of
/// the app, see [`PortalHost::new`](super::PortalHost::new).
pub trait Portal {
    /// Hand one request to the host.
    fn request(&mut self, request: PortalRequest) -> Result<(), Refusal>;

    /// Everything the host said since the last call. Answers may have arrived
    /// on this frame, or on any frame since the previous poll.
    fn poll(&mut self) -> Vec<PortalEvent>;

    /// What the host can do and what it says without being asked. Read this at
    /// start: a host reports *changes*, so a value only ever read from events
    /// would be the wrong one until the host happens to change it.
    fn status(&self) -> PortalStatus;

    /// A name for logs: the platform and language behind the bridge, or why
    /// there is no host at all.
    fn label(&self) -> &str;
}
