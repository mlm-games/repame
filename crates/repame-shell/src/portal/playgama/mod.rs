//! The Playgama backend: [`playgama_bridge`] translated into [`Portal`].
//!
//! Behind the `playgama` feature, and off by default: a plain desktop build has
//! no bridge to talk to, and the bridge crate's desktop stand-ins answer every
//! call immediately, which a rewarded ad would read as a free reward. For the
//! same reason [`Playgama::attach`] refuses a bridge that is not talking to the
//! live SDK ([`Bridge::is_web`] is false), which is what the games did by
//! falling back to offline when the bridge reported its mock backend.
//!
//! The backend owns the listeners the SDK hands out and queues what they report
//! in a [`RefCell`], because a host callback fires outside the frame loop and
//! may fire from inside a `Portal::request` call (the editor stand-ins answer
//! synchronously). [`PortalHost`] drains that queue once per frame.
//!
//! It is also the only caller of the bridge's cloud storage. The SDK keeps one
//! call per operation in flight and **drops** a second one, whose callback then
//! never runs, so the backend holds that slot itself and refuses the second ask
//! with [`Refusal::InFlight`]: a dropped request that nobody will answer is
//! worse than a refusal on the frame it was made.
//!
//! Wiring, on the web, from a game's entry point:
//!
//! ```ignore
//! wasm_bindgen_futures::spawn_local(async move {
//!     let backend: Box<dyn Portal> = match playgama::boot(config, CLOUD_KEY).await {
//!         Ok(portal) => Box::new(portal),
//!         Err(error) => {
//!             log::warn!("running without a platform: {error}");
//!             Box::new(NoPortal)
//!         }
//!     };
//!     run(PortalHost::pull_first(backend));
//! });
//! ```
//!
//! [`PortalHost`]: super::PortalHost
//! [`Portal`]: super::Portal

mod config;

pub use config::{ConfigError, PlacementList, PlaygamaConfig};

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use playgama_bridge::{
    Bridge, BridgeError, BridgeResult, InterstitialState, PlatformMessage, RewardedState,
    StorageValue,
};

use super::protocol::{
    AdEnding, AdKind, Availability, HostNotice, Portal, PortalEvent, PortalRequest, PortalStatus,
    Refusal,
};

/// The Playgama host, through one initialized bridge.
pub struct Playgama {
    bridge: Rc<Bridge>,
    inbox: Rc<RefCell<Vec<PortalEvent>>>,
    /// A `storage.get` is outstanding. The SDK drops a second one rather than
    /// queueing it, so a second ask would never be answered.
    pulling: Rc<Cell<bool>>,
    /// A `storage.set` is outstanding, same rule.
    pushing: Rc<Cell<bool>>,
    config: PlaygamaConfig,
    cloud_key: String,
    label: String,
}

impl Playgama {
    /// Talk to the host behind `bridge`, reading the game's bridge config and
    /// storing the cloud save under `cloud_key`.
    ///
    /// `Err(Refusal::NoPortal)` when the bridge is not talking to a live host:
    /// a desktop build, or a page where the SDK never came up. The caller
    /// installs [`NoPortal`](super::NoPortal) then, which refuses
    /// everything with that same reason.
    pub fn attach(
        bridge: Bridge,
        config: PlaygamaConfig,
        cloud_key: impl Into<String>,
    ) -> Result<Self, Refusal> {
        if !bridge.is_web() {
            return Err(Refusal::NoPortal);
        }
        let cloud_key = cloud_key.into();
        let label = label(&bridge);
        let inbox = Rc::new(RefCell::new(Vec::new()));
        let bridge = Rc::new(bridge);
        listen(&bridge, &inbox);
        Ok(Self {
            bridge,
            inbox,
            pulling: Rc::new(Cell::new(false)),
            pushing: Rc::new(Cell::new(false)),
            config,
            cloud_key,
            label,
        })
    }

    /// The config this backend was attached with.
    pub fn config(&self) -> &PlaygamaConfig {
        &self.config
    }

    fn show_interstitial(&self, placement: Option<String>) -> Result<(), Refusal> {
        if !self.bridge.advertisement.is_interstitial_supported() {
            return Err(Refusal::NotSupported);
        }
        let placement = self.resolve(AdKind::Interstitial, placement);
        self.bridge
            .advertisement
            .show_interstitial(placement.as_deref());
        Ok(())
    }

    fn show_rewarded(&self, placement: Option<String>) -> Result<(), Refusal> {
        if !self.bridge.advertisement.is_rewarded_supported() {
            return Err(Refusal::NotSupported);
        }
        let placement = self.resolve(AdKind::Rewarded, placement);
        self.bridge
            .advertisement
            .show_rewarded(placement.as_deref());
        Ok(())
    }

    /// The placement to send: the one asked for when the config declares it,
    /// otherwise the config's own. An undeclared name is sent anyway, because
    /// the SDK would not refuse it, but it is worth a warning: the host
    /// substitutes its fallback, so the game does not get the placement it
    /// named.
    fn resolve(&self, kind: AdKind, placement: Option<String>) -> Option<String> {
        let Some(placement) = placement else {
            return self.config.placement(kind).map(str::to_string);
        };
        if !self.config.is_declared(kind, &placement) {
            log::warn!(
                "{}: the {kind} config does not declare the placement {placement:?}; the host will use {:?}",
                self.label,
                self.config.placement(kind)
            );
        }
        Some(placement)
    }

    fn cloud_pull(&mut self) -> Result<(), Refusal> {
        if self.pulling.replace(true) {
            return Err(Refusal::InFlight);
        }
        let key = self.cloud_key.clone();
        let inbox = Rc::clone(&self.inbox);
        let pulling = Rc::clone(&self.pulling);
        self.bridge.storage.get(
            &[&key],
            // The stored text is handed over as it is and parsed by the game:
            // the portal does not know the save type.
            false,
            Box::new(move |outcome| {
                let event = PortalEvent::CloudPulled(read(outcome));
                inbox.borrow_mut().push(event);
                // The SDK releases its own slot before the callback runs, so a
                // handler may ask again from here.
                pulling.set(false);
            }),
        );
        Ok(())
    }

    fn cloud_push(&mut self, payload: Vec<u8>) -> Result<(), Refusal> {
        let Ok(text) = String::from_utf8(payload) else {
            return Err(Refusal::NotText);
        };
        if text.is_empty() {
            // `storage.set` deletes the key for an empty string: a save that
            // would delete itself is refused instead.
            return Err(Refusal::EmptyPayload);
        }
        if self.pushing.replace(true) {
            return Err(Refusal::InFlight);
        }
        let key = self.cloud_key.clone();
        let inbox = Rc::clone(&self.inbox);
        let pushing = Rc::clone(&self.pushing);
        self.bridge.storage.set(
            &[(&key, StorageValue::Str(text))],
            Box::new(move |outcome| {
                let event = PortalEvent::CloudStored(outcome.map_err(refusal));
                inbox.borrow_mut().push(event);
                pushing.set(false);
            }),
        );
        Ok(())
    }
}

impl Portal for Playgama {
    fn request(&mut self, request: PortalRequest) -> Result<(), Refusal> {
        match request {
            PortalRequest::Notify(notice) => {
                // Fire and forget, as the SDK takes it: a second `game_ready`
                // is rejected inside the SDK's own promise, which nothing here
                // can hold. `PortalHost` sends it once instead.
                self.bridge.platform.send_message(wire(notice));
                Ok(())
            }
            PortalRequest::ShowAd { kind, placement } => match kind {
                AdKind::Interstitial => self.show_interstitial(placement),
                AdKind::Rewarded => self.show_rewarded(placement),
            },
            PortalRequest::CloudPull => self.cloud_pull(),
            PortalRequest::CloudPush { payload } => self.cloud_push(payload),
        }
    }

    fn poll(&mut self) -> Vec<PortalEvent> {
        std::mem::take(&mut self.inbox.borrow_mut())
    }

    fn status(&self) -> PortalStatus {
        let advertisement = &self.bridge.advertisement;
        PortalStatus {
            hosted: true,
            // Read, not just subscribed to: the hosts only report changes.
            audio_enabled: self.bridge.platform.is_audio_enabled(),
            host_paused: self.bridge.platform.is_paused(),
            interstitial: available(advertisement.is_interstitial_supported()),
            rewarded: available(advertisement.is_rewarded_supported()),
            // The bridge has no "cloud supported" flag and the SDK keeps the
            // platform's storage availability to itself: asking is the way to
            // find out, and the refusal is typed when the platform says no.
            cloud: Availability::Available,
            interstitial_schedule: self.config.interstitial_schedule,
        }
    }

    fn label(&self) -> &str {
        &self.label
    }
}

/// The listeners the host calls. The bridge's signals are append-only and
/// nothing here is ever removed, so each handler owns what it touches and
/// survives the call that installed it.
fn listen(bridge: &Rc<Bridge>, inbox: &Rc<RefCell<Vec<PortalEvent>>>) {
    let push = Rc::clone(inbox);
    bridge
        .platform
        .audio_state_changed()
        .connect(move |enabled| {
            push.borrow_mut().push(PortalEvent::AudioEnabled(enabled));
        });
    let push = Rc::clone(inbox);
    bridge
        .platform
        .pause_state_changed()
        .connect(move |paused| {
            push.borrow_mut().push(PortalEvent::HostPaused(paused));
        });
    let push = Rc::clone(inbox);
    bridge
        .advertisement
        .interstitial_state_changed()
        .connect(move |state| {
            let event = match state {
                Some(InterstitialState::Opened) => {
                    Some(PortalEvent::AdOpened(AdKind::Interstitial))
                }
                Some(InterstitialState::Closed) => Some(PortalEvent::AdEnded {
                    kind: AdKind::Interstitial,
                    ending: AdEnding::Closed,
                }),
                Some(InterstitialState::Failed) => Some(PortalEvent::AdEnded {
                    kind: AdKind::Interstitial,
                    ending: AdEnding::Failed,
                }),
                // `Loading`, and a state from a newer SDK than this crate
                // knows, say nothing about an ad the game is holding, so
                // neither ends one.
                Some(InterstitialState::Loading) | None => None,
            };
            if let Some(event) = event {
                push.borrow_mut().push(event);
            }
        });
    let push = Rc::clone(inbox);
    bridge
        .advertisement
        .rewarded_state_changed()
        .connect(move |state| {
            let event = match state {
                Some(RewardedState::Opened) => Some(PortalEvent::AdOpened(AdKind::Rewarded)),
                Some(RewardedState::Rewarded) => Some(PortalEvent::AdRewarded),
                Some(RewardedState::Closed) => Some(PortalEvent::AdEnded {
                    kind: AdKind::Rewarded,
                    ending: AdEnding::Closed,
                }),
                Some(RewardedState::Failed) => Some(PortalEvent::AdEnded {
                    kind: AdKind::Rewarded,
                    ending: AdEnding::Failed,
                }),
                Some(RewardedState::Loading) | None => None,
            };
            if let Some(event) = event {
                push.borrow_mut().push(event);
            }
        });
}

/// One key's worth of cloud storage as the portal's opaque payload. The bridge
/// is asked for the text as stored, so what a game pushed is what it reads
/// back.
fn read(outcome: BridgeResult<Vec<StorageValue>>) -> Result<Option<Vec<u8>>, Refusal> {
    match outcome {
        Ok(values) => match values.into_iter().next() {
            // A missing key reads as null; an empty value is the SDK's own
            // delete, so neither is a stored save.
            None | Some(StorageValue::Null) => Ok(None),
            Some(StorageValue::Str(text)) if text.is_empty() => Ok(None),
            Some(StorageValue::Str(text)) => Ok(Some(text.into_bytes())),
            // The platform kept something that is not text: it cannot be the
            // game's save, and handing back other bytes would be worse than
            // saying so.
            Some(_) => Err(Refusal::NotText),
        },
        Err(error) => Err(refusal(error)),
    }
}

/// A bridge rejection as the portal's typed refusal.
fn refusal(error: BridgeError) -> Refusal {
    let message = error.message().to_string();
    match error.raw_code() {
        // A rejection with no code at all is the SDK's unsupported feature,
        // which the bridge documents as `code()` being `None`.
        None if error.code().is_none() => Refusal::NotSupported,
        None => Refusal::Host {
            code: error.code().map(|code| code.as_str().to_string()),
            message,
        },
        Some(code) => Refusal::Host {
            code: Some(code.to_string()),
            message,
        },
    }
}

fn available(supported: bool) -> Availability {
    if supported {
        Availability::Available
    } else {
        Availability::Unavailable(Refusal::NotSupported)
    }
}

fn wire(notice: HostNotice) -> PlatformMessage {
    match notice {
        HostNotice::GameReady => PlatformMessage::GameReady,
        HostNotice::GameplayStarted => PlatformMessage::GameplayStarted,
        HostNotice::GameplayStopped => PlatformMessage::GameplayStopped,
    }
}

/// The platform behind the bridge, for logs: `playgama yandex [ru]`.
fn label(bridge: &Bridge) -> String {
    let id = bridge
        .platform
        .id()
        .unwrap_or_else(|| "unknown".to_string());
    let language = bridge
        .platform
        .language()
        .map_or_else(String::new, |language| format!(" [{language}]"));
    format!("playgama {id}{language}")
}

/// Why the web boot did not end in a portal.
#[cfg(target_arch = "wasm32")]
#[derive(Clone, Debug, PartialEq)]
pub enum BootError {
    /// The SDK script never came up. The game should run without a host.
    Load(playgama_bridge::LoadError),
    /// The SDK came up but is not talking to a live host.
    Attach(Refusal),
}

#[cfg(target_arch = "wasm32")]
impl std::fmt::Display for BootError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Load(error) => write!(f, "{error}"),
            Self::Attach(reason) => write!(f, "the playgama bridge attached no host: {reason}"),
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl std::error::Error for BootError {}

/// Load the SDK onto the page, initialize it, and attach: the whole web boot,
/// so a game's entry point keeps only its own loading screen.
///
/// The caller drives the future (`wasm_bindgen_futures::spawn_local`), which is
/// what lets the game keep painting while the script lands. Loading progress is
/// reported on the SDK's own 0 to 100 scale: `0` before the script is asked
/// for, `100` once `initialize` has resolved.
///
/// Off the web there is nothing to load, so this does not exist there: a
/// desktop build installs [`NoPortal`](super::NoPortal).
#[cfg(target_arch = "wasm32")]
pub async fn boot(
    config: PlaygamaConfig,
    cloud_key: impl Into<String>,
) -> Result<Playgama, BootError> {
    playgama_bridge::set_game_loading_progress(0.0);
    let bridge = playgama_bridge::load(playgama_bridge::LoadOptions::default())
        .await
        .map_err(BootError::Load)?;
    playgama_bridge::set_game_loading_progress(100.0);
    Playgama::attach(bridge, config, cloud_key).map_err(BootError::Attach)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bridge_that_is_not_on_the_web_is_not_a_portal() {
        // The bridge crate serves its desktop stand-ins wherever the page has
        // no live SDK, and those answer ads for free: attaching to one would
        // report a reward the player never watched.
        assert!(matches!(
            Playgama::attach(Bridge::new(), PlaygamaConfig::default(), "save"),
            Err(Refusal::NoPortal)
        ));
    }

    #[test]
    fn a_stored_value_that_is_not_text_is_refused_rather_than_handed_back() {
        assert_eq!(read(Ok(vec![StorageValue::Int(7)])), Err(Refusal::NotText));
        assert_eq!(read(Ok(vec![StorageValue::Null])), Ok(None));
        assert_eq!(read(Ok(vec![])), Ok(None));
        assert_eq!(
            read(Ok(vec![StorageValue::Str(r#"{"best":3}"#.to_string())])),
            Ok(Some(br#"{"best":3}"#.to_vec()))
        );
    }

    #[test]
    fn a_coded_rejection_keeps_the_platforms_code_and_a_bare_one_is_unsupported() {
        let coded = refusal(BridgeError::new(
            Some(playgama_bridge::ErrorCode::StorageQuotaExceeded),
            Some("Storage quota exceeded".to_string()),
        ));
        assert_eq!(
            coded,
            Refusal::Host {
                code: Some("STORAGE_QUOTA_EXCEEDED".to_string()),
                message: "Storage quota exceeded".to_string(),
            }
        );
        assert_eq!(refusal(BridgeError::rejected()), Refusal::NotSupported);
    }
}
