//! App lifecycle: foreground or background, as the single fact the gate reads.
//!
//! The platform reports real visibility transitions (Android
//! `resumed`/`suspended`, a desktop window hidden or occluded, the web's
//! `visibilitychange`), and a game that is not on screen must stop playing and
//! go quiet. Every game in the portal family carries the same pair of
//! functions for this, plus the same static flag behind them; this is that
//! pair, once.
//!
//! [`watch`] is idempotent, and it reads the state the app is already in before
//! it subscribes, because the platform only reports *changes*: a game that
//! boots straight into the background (an Android reopen, a hidden tab) would
//! otherwise read as foreground until the player came back and left again.

use std::sync::Once;
use std::sync::atomic::{AtomicBool, Ordering};

use repose_platform::AppLifecycle;

/// Foreground until the platform says otherwise, and on every platform that
/// never reports a lifecycle at all.
static FOREGROUND: AtomicBool = AtomicBool::new(true);

static WATCH: Once = Once::new();

/// Subscribe to the app's lifecycle. Idempotent: calling it twice installs one
/// listener, so a game may call it at boot and anything else may call it again.
pub fn watch() {
    WATCH.call_once(|| {
        if let Some(state) = repose_platform::current_lifecycle() {
            FOREGROUND.store(state == AppLifecycle::Foreground, Ordering::Relaxed);
        }
        // The listeners outlive this call and are never removed (the platform
        // has no unsubscribe for them), so everything they touch is a static.
        repose_platform::add_lifecycle_listener(Box::new(|state| {
            FOREGROUND.store(state == AppLifecycle::Foreground, Ordering::Relaxed);
        }));
    });
}

/// Whether the app is in the foreground.
///
/// `true` before [`watch`] runs and wherever no platform lifecycle exists, so a
/// game that never watches runs as it always did.
pub fn foreground() -> bool {
    FOREGROUND.load(Ordering::Relaxed)
}

/// Whether the app is in the background: the value the gate takes for
/// [`PortalView::backgrounded`](super::PortalView::backgrounded).
pub fn backgrounded() -> bool {
    !foreground()
}
