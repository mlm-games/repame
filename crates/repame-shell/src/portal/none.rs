//! The backend a build without a portal installs.
//!
//! It is a real implementation, not a stub. Every request is refused on the
//! spot with [`Refusal::NoPortal`], so a game that asks for an ad gets an
//! answer on that frame instead of waiting for a host that is not there, and
//! [`NoPortal::status`] says the same thing before anything is asked at all:
//! `hosted: false` and every feature `Unavailable(NoPortal)`. A game can hide
//! its rewarded button, its cloud banner and its "watch to revive" offer from
//! that, and everything it does ask is a warning in a log rather than silence.

use super::protocol::{Portal, PortalEvent, PortalRequest, PortalStatus, Refusal};

/// Nothing is listening. Installed by [`PortalHost::none`], which is also the
/// default host.
///
/// [`PortalHost::none`]: super::PortalHost::none
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoPortal;

impl Portal for NoPortal {
    fn request(&mut self, _request: PortalRequest) -> Result<(), Refusal> {
        Err(Refusal::NoPortal)
    }

    fn poll(&mut self) -> Vec<PortalEvent> {
        Vec::new()
    }

    fn status(&self) -> PortalStatus {
        PortalStatus::default()
    }

    fn label(&self) -> &str {
        "no portal (desktop)"
    }
}
