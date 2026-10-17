//! Portal contract: what the gate answers, how an ad ends, what survives a
//! cloud round trip, and what a build with no host says.
//!
//! Pure and compile-checked like the rest of the shell's tests: no host, no
//! wall clock, no runtime. The backend is a script this test drives, which is
//! the injection seam the portal integrations carry as a `#[cfg(test)] fn
//! inject` inside the app: here it is the public [`Portal`] trait, so the
//! public surface is what gets exercised.

use std::cell::RefCell;
use std::rc::Rc;

use repame_shell::portal::{
    AdEnding, AdKind, AdOutcome, AdRefused, AdSchedule, Cause, CloudRefused, GameState, HostNotice,
    Portal, PortalEvent, PortalHost, PortalReport, PortalRequest, PortalStatus, REWARDED_TIMEOUT,
    Refusal,
};
use web_time::Duration;

/// What a test portal does: take requests (or refuse them with one reason),
/// queue the host events the test decided on, and keep the cloud save it was
/// handed.
struct Script {
    refusal: Option<Refusal>,
    /// What a cloud read resolves to, when the request itself is taken.
    read_refusal: Option<Refusal>,
    status: PortalStatus,
    says: Vec<PortalEvent>,
    asked: Vec<PortalRequest>,
    stored: Option<Vec<u8>>,
}

impl Script {
    fn hosted() -> Self {
        Self {
            refusal: None,
            read_refusal: None,
            status: PortalStatus {
                hosted: true,
                ..Default::default()
            },
            says: Vec::new(),
            asked: Vec::new(),
            stored: None,
        }
    }

    fn says(&mut self, event: PortalEvent) {
        self.says.push(event);
    }
}

/// The handle a test keeps while the host owns the backend.
#[derive(Clone)]
struct Handle(Rc<RefCell<Script>>);

impl Handle {
    fn new(script: Script) -> (Self, Rc<RefCell<Script>>) {
        let shared = Rc::new(RefCell::new(script));
        (Self(Rc::clone(&shared)), shared)
    }
}

impl Portal for Handle {
    fn request(&mut self, request: PortalRequest) -> Result<(), Refusal> {
        self.0.borrow_mut().request(request)
    }

    fn poll(&mut self) -> Vec<PortalEvent> {
        self.0.borrow_mut().poll()
    }

    fn status(&self) -> PortalStatus {
        self.0.borrow().status()
    }

    fn label(&self) -> &str {
        "scripted"
    }
}

impl Portal for Script {
    fn request(&mut self, request: PortalRequest) -> Result<(), Refusal> {
        self.asked.push(request.clone());
        if let Some(reason) = &self.refusal {
            return Err(reason.clone());
        }
        // A real backend reports these from the host's callback; a script
        // answers on the spot, which is the same thing one frame early.
        match request {
            PortalRequest::CloudPull => {
                let event = match &self.read_refusal {
                    Some(reason) => PortalEvent::CloudPulled(Err(reason.clone())),
                    None => PortalEvent::CloudPulled(Ok(self.stored.clone())),
                };
                self.says.push(event);
            }
            PortalRequest::CloudPush { payload } => {
                self.stored = Some(payload);
                self.says.push(PortalEvent::CloudStored(Ok(())));
            }
            PortalRequest::Notify(_) | PortalRequest::ShowAd { .. } => {}
        }
        Ok(())
    }

    fn poll(&mut self) -> Vec<PortalEvent> {
        std::mem::take(&mut self.says)
    }

    fn status(&self) -> PortalStatus {
        self.status.clone()
    }

    fn label(&self) -> &str {
        "scripted"
    }
}

#[test]
fn an_ad_holds_the_clock_and_the_hosts_gameplay_count_until_it_ends() {
    let (backend, script) = Handle::new(Script::hosted());
    let mut host = PortalHost::new(Box::new(backend));
    let playing = GameState {
        playing: true,
        ..Default::default()
    };

    let running = host.gate(playing);
    assert!(running.live() && !running.suspended());
    assert!(running.host_gameplay());
    assert_eq!(
        script.borrow().asked,
        vec![PortalRequest::Notify(HostNotice::GameplayStarted)]
    );

    host.show_ad(AdKind::Interstitial, Some("game_over"))
        .expect("the host takes the request");
    assert!(host.ad_pending());
    let held = host.gate(playing);
    assert!(held.suspended(), "an accepted ad holds gameplay at once");
    assert_eq!(held.cause(), Some(Cause::Ad));
    assert_eq!(held.time_scale(), 0.0);
    assert!(!held.live());
    assert!(!held.host_gameplay(), "the host stops counting impressions");
    assert_eq!(
        script.borrow().asked.last(),
        Some(&PortalRequest::Notify(HostNotice::GameplayStopped))
    );

    script
        .borrow_mut()
        .says(PortalEvent::AdOpened(AdKind::Interstitial));
    script.borrow_mut().says(PortalEvent::AdEnded {
        kind: AdKind::Interstitial,
        ending: AdEnding::Closed,
    });
    host.update(Duration::ZERO);
    assert!(host.ad_visible());
    assert_eq!(
        host.take_reports(),
        vec![
            PortalReport::AdOpened {
                kind: AdKind::Interstitial
            },
            PortalReport::AdFinished {
                kind: AdKind::Interstitial,
                outcome: AdOutcome::Completed,
            },
        ]
    );

    let resumed = host.gate(playing);
    assert!(resumed.live() && !resumed.suspended() && resumed.host_gameplay());
    assert_eq!(
        script.borrow().asked.last(),
        Some(&PortalRequest::Notify(HostNotice::GameplayStarted))
    );
    assert!(!host.ad_pending() && !host.ad_visible());
}

#[test]
fn a_rewarded_ad_pays_only_for_its_reward_point_and_never_outlives_its_deadline() {
    let (backend, script) = Handle::new(Script::hosted());
    let mut host = PortalHost::new(Box::new(backend));

    // Closed before the host reported the reward: nothing is owed, and the
    // deadline never runs out on a break nobody is holding.
    host.show_ad(AdKind::Rewarded, Some("revive"))
        .expect("the host takes the request");
    script
        .borrow_mut()
        .says(PortalEvent::AdOpened(AdKind::Rewarded));
    script.borrow_mut().says(PortalEvent::AdEnded {
        kind: AdKind::Rewarded,
        ending: AdEnding::Closed,
    });
    host.update(Duration::from_secs(1));
    let skipped = host.take_reports();
    assert_eq!(
        skipped.last(),
        Some(&PortalReport::AdFinished {
            kind: AdKind::Rewarded,
            outcome: AdOutcome::Skipped,
        })
    );
    assert!(
        !AdOutcome::Skipped.pays(AdKind::Rewarded),
        "a rewarded ad closed early cannot pay"
    );

    // The reward point is what pays, whatever ends the ad afterwards.
    host.show_ad(AdKind::Rewarded, Some("revive"))
        .expect("the host takes the request");
    script
        .borrow_mut()
        .says(PortalEvent::AdOpened(AdKind::Rewarded));
    script.borrow_mut().says(PortalEvent::AdRewarded);
    script.borrow_mut().says(PortalEvent::AdEnded {
        kind: AdKind::Rewarded,
        ending: AdEnding::Closed,
    });
    host.update(Duration::ZERO);
    let paid = host.take_reports();
    let outcome = match paid.last() {
        Some(PortalReport::AdFinished { kind, outcome }) => {
            assert_eq!(*kind, AdKind::Rewarded);
            outcome.clone()
        }
        other => panic!("expected a finished rewarded ad, got {other:?}"),
    };
    assert!(outcome.pays(AdKind::Rewarded), "the reward is due");
    assert!(
        !outcome.pays(AdKind::Interstitial),
        "only the rewarded kind pays"
    );

    // A host that never ends the ad: the deadline releases the game, and the
    // host's late report cannot revive the break or pay a reward.
    host.show_ad(AdKind::Rewarded, Some("revive"))
        .expect("the host takes the request");
    script
        .borrow_mut()
        .says(PortalEvent::AdOpened(AdKind::Rewarded));
    host.update(Duration::from_secs(1));
    assert!(host.ad_visible() && host.ad_pending());
    host.update(REWARDED_TIMEOUT - Duration::from_secs(2));
    assert!(host.ad_pending(), "still inside the deadline");
    host.update(Duration::from_secs(1));
    assert_eq!(
        host.take_reports(),
        vec![PortalReport::AdFinished {
            kind: AdKind::Rewarded,
            outcome: AdOutcome::TimedOut {
                after: REWARDED_TIMEOUT
            },
        }]
    );
    assert!(!host.ad_pending() && !host.ad_visible());

    script.borrow_mut().says(PortalEvent::AdRewarded);
    script.borrow_mut().says(PortalEvent::AdEnded {
        kind: AdKind::Rewarded,
        ending: AdEnding::Closed,
    });
    host.update(Duration::from_secs(1));
    assert_eq!(
        host.take_reports(),
        Vec::new(),
        "a released break stays released"
    );
}

#[test]
fn a_cloud_payload_round_trips_through_the_abstraction() {
    let (backend, script) = Handle::new(Script::hosted());
    let mut host = PortalHost::pull_first(Box::new(backend));

    // Before a read resolves, a push is refused: a fresh default save must not
    // race the save the player came back for.
    assert_eq!(
        host.cloud_push(br#"{"best":3}"#),
        Err(CloudRefused::BeforePull)
    );

    // Nothing stored yet is an answer, not a failure.
    host.cloud_pull().expect("the read is taken");
    host.update(Duration::ZERO);
    assert_eq!(
        host.take_reports(),
        vec![PortalReport::CloudPulled(Ok(None))]
    );
    assert!(host.cloud_ready(), "an empty cloud is still an answer");

    let save = br#"{"best":3}"#.to_vec();
    host.cloud_push(&save).expect("the write is taken");
    host.update(Duration::ZERO);
    assert_eq!(host.take_reports(), vec![PortalReport::CloudStored(Ok(()))]);
    assert_eq!(script.borrow().stored.as_deref(), Some(save.as_slice()));

    // A fresh host reads back exactly what was pushed: the portal never learns
    // the save type.
    let (backend, _) = Handle::new(Script {
        stored: script.borrow().stored.clone(),
        ..Script::hosted()
    });
    let mut second = PortalHost::new(Box::new(backend));
    second.cloud_pull().expect("the read is taken");
    second.update(Duration::ZERO);
    assert_eq!(
        second.take_reports(),
        vec![PortalReport::CloudPulled(Ok(Some(save)))]
    );
    assert!(second.cloud_ready());

    // A read the host refused may be transient, and a push now could store over
    // a save that was there all along.
    let (backend, _) = Handle::new(Script {
        read_refusal: Some(Refusal::Host {
            code: Some("STORAGE_NOT_AVAILABLE".to_string()),
            message: "Storage not available".to_string(),
        }),
        ..Script::hosted()
    });
    let mut busy = PortalHost::pull_first(Box::new(backend));
    busy.cloud_pull().expect("the read is taken");
    busy.update(Duration::ZERO);
    assert!(!busy.cloud_ready());
    assert_eq!(busy.cloud_push(b"save"), Err(CloudRefused::BeforePull));

    // A cloud that does not exist cannot be overwritten, so it counts as
    // resolved and the push gets the backend's own answer.
    let (backend, _) = Handle::new(Script {
        read_refusal: Some(Refusal::NoPortal),
        ..Script::hosted()
    });
    let mut absent = PortalHost::pull_first(Box::new(backend));
    absent.cloud_pull().expect("the read is taken");
    absent.update(Duration::ZERO);
    assert!(absent.cloud_ready());
    assert_eq!(absent.cloud_push(b"save"), Ok(()));
}

#[test]
fn a_refused_request_leaves_the_portal_alone() {
    let script = Script {
        refusal: Some(Refusal::Host {
            code: Some("STORAGE_NOT_SUPPORTED".to_string()),
            message: "Storage not supported".to_string(),
        }),
        ..Script::hosted()
    };
    let (backend, _) = Handle::new(script);
    let mut host = PortalHost::new(Box::new(backend));

    assert_eq!(
        host.show_ad(AdKind::Rewarded, None),
        Err(AdRefused::Refusal(Refusal::Host {
            code: Some("STORAGE_NOT_SUPPORTED".to_string()),
            message: "Storage not supported".to_string(),
        }))
    );
    assert!(!host.ad_pending(), "a refused ad holds nothing");
    assert_eq!(
        host.cloud_pull(),
        Err(Refusal::Host {
            code: Some("STORAGE_NOT_SUPPORTED".to_string()),
            message: "Storage not supported".to_string(),
        })
    );
    assert_eq!(
        host.cloud_push(b"save"),
        Err(CloudRefused::Refusal(Refusal::Host {
            code: Some("STORAGE_NOT_SUPPORTED".to_string()),
            message: "Storage not supported".to_string(),
        }))
    );
}

#[test]
fn the_interstitial_rate_limit_is_a_typed_refusal_not_a_dead_wait() {
    let schedule = AdSchedule {
        initial_delay: Duration::from_secs(120),
        minimum_delay: Duration::from_secs(120),
    };
    let script = Script {
        status: PortalStatus {
            hosted: true,
            interstitial_schedule: schedule,
            ..Default::default()
        },
        ..Script::hosted()
    };
    let (backend, script) = Handle::new(script);
    let mut host = PortalHost::new(Box::new(backend));

    assert_eq!(
        host.show_ad(AdKind::Interstitial, Some("game_over")),
        Err(AdRefused::TooSoon {
            remaining: Duration::from_secs(120)
        }),
        "nothing may interrupt the first two minutes"
    );
    // A rewarded ad is not rate limited: it is the player's own choice.
    host.show_ad(AdKind::Rewarded, None).expect("taken");
    script.borrow_mut().says(PortalEvent::AdEnded {
        kind: AdKind::Rewarded,
        ending: AdEnding::Failed,
    });
    host.update(Duration::from_secs(120));
    assert_eq!(
        host.take_reports(),
        vec![PortalReport::AdFinished {
            kind: AdKind::Rewarded,
            outcome: AdOutcome::Failed,
        }]
    );

    host.show_ad(AdKind::Interstitial, None)
        .expect("the initial delay has elapsed");
    script
        .borrow_mut()
        .says(PortalEvent::AdOpened(AdKind::Interstitial));
    script.borrow_mut().says(PortalEvent::AdEnded {
        kind: AdKind::Interstitial,
        ending: AdEnding::Closed,
    });
    host.update(Duration::from_secs(1));
    assert_eq!(
        host.show_ad(AdKind::Interstitial, None),
        Err(AdRefused::TooSoon {
            remaining: Duration::from_secs(120)
        }),
        "the between-ads clock runs from the impression"
    );
    let asked = script.borrow().asked.len();
    assert_eq!(
        host.show_ad(AdKind::Interstitial, None),
        Err(AdRefused::TooSoon {
            remaining: Duration::from_secs(120)
        })
    );
    assert_eq!(
        script.borrow().asked.len(),
        asked,
        "a refused call never reaches the host"
    );
    host.update(Duration::from_secs(120));
    host.show_ad(AdKind::Interstitial, None)
        .expect("the between-ads delay has elapsed");
}

#[test]
fn the_no_portal_backend_refuses_every_request_and_says_why() {
    let mut host = PortalHost::none();
    assert_eq!(host.label(), "no portal (desktop)");

    let status = host.status();
    assert!(!status.hosted);
    assert_eq!(status.interstitial.reason(), Some(&Refusal::NoPortal));
    assert_eq!(status.rewarded.reason(), Some(&Refusal::NoPortal));
    assert_eq!(status.cloud.reason(), Some(&Refusal::NoPortal));
    assert!(status.audio_enabled, "nothing is muting the game");

    assert_eq!(
        host.show_ad(AdKind::Interstitial, None),
        Err(AdRefused::Refusal(Refusal::NoPortal))
    );
    assert_eq!(
        host.show_ad(AdKind::Rewarded, Some("revive")),
        Err(AdRefused::Refusal(Refusal::NoPortal))
    );
    assert!(!host.ad_pending(), "a refused ad holds nothing");
    assert_eq!(host.cloud_pull(), Err(Refusal::NoPortal));
    assert_eq!(
        host.cloud_push(b"save"),
        Err(CloudRefused::Refusal(Refusal::NoPortal))
    );
    assert_eq!(
        host.notify(HostNotice::GameplayStarted),
        Err(Refusal::NoPortal)
    );

    // A notice nobody took is reported, never swallowed.
    host.loaded();
    assert_eq!(host.ready(), Err(Refusal::NoPortal));
    assert_eq!(
        host.take_reports(),
        vec![PortalReport::NoticeDropped {
            notice: HostNotice::GameReady,
            reason: Refusal::NoPortal,
        }]
    );
    assert_eq!(
        host.ready(),
        Err(Refusal::NoPortal),
        "a notice that never landed is not marked sent"
    );
    host.take_reports();

    // The gate still works without a host: a desktop build runs.
    let gate = host.gate(GameState {
        playing: true,
        ..Default::default()
    });
    assert!(gate.live() && !gate.suspended());
    assert!(!gate.silenced());
    assert_eq!(
        host.take_reports(),
        vec![PortalReport::NoticeDropped {
            notice: HostNotice::GameplayStarted,
            reason: Refusal::NoPortal,
        }]
    );
}

#[test]
fn game_ready_is_sent_once_and_only_on_the_first_playable_frame() {
    let (backend, script) = Handle::new(Script::hosted());
    let mut host = PortalHost::new(Box::new(backend));

    assert_eq!(
        host.ready(),
        Ok(false),
        "nothing to announce before the load"
    );
    host.loaded();
    assert_eq!(host.ready(), Ok(true));
    assert_eq!(
        script.borrow().asked,
        vec![PortalRequest::Notify(HostNotice::GameReady)]
    );
    assert_eq!(
        host.ready(),
        Ok(false),
        "the host rejects a second game_ready, so it is never sent"
    );
    assert_eq!(script.borrow().asked.len(), 1);
}
