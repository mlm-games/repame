//! `playgama-bridge-config.json`, the Rust view of the file the games ship.
//!
//! The SDK reads this file itself (`initialize()` looks for it next to the
//! page), and so does the shell: the Rust side needs the placement ids and the
//! interstitial rate limit before anything is asked, so a request can name the
//! placement the config declares and a host rate limit becomes a typed
//! [`AdRefused::TooSoon`](super::super::AdRefused) instead of a request the
//! host drops while the shell waits out the ad deadline.
//!
//! Unknown keys are ignored, not rejected: the file carries the SDK's own
//! sections (`device`, `payments`, `leaderboards`, `sendAnalyticsEvents`) and a
//! `$schema` marker, and the shell has no opinion about any of them.

use std::fmt;

use serde::Deserialize;
use web_time::Duration;

use super::super::protocol::{AdKind, AdSchedule};

/// The placements and rate limit of one `playgama-bridge-config.json`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PlaygamaConfig {
    pub interstitial: PlacementList,
    pub rewarded: PlacementList,
    /// `advertisement.initialInterstitialDelay` and
    /// `advertisement.minimumDelayBetweenInterstitial`.
    pub interstitial_schedule: AdSchedule,
}

/// The placements one ad kind declares.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PlacementList {
    /// `advertisement.<kind>.placements[].id`, in file order.
    pub placements: Vec<String>,
    /// `placementFallback`: the placement the SDK substitutes for one the
    /// config does not declare.
    pub fallback: Option<String>,
}

impl PlaygamaConfig {
    /// Read the config a game ships. `text` is the file itself, so a game
    /// embeds it with `include_str!` and the shell reads what the SDK will
    /// read.
    pub fn from_json(text: &str) -> Result<Self, ConfigError> {
        let wire: Wire = serde_json::from_str(text).map_err(|error| ConfigError::Parse {
            detail: error.to_string(),
        })?;
        Ok(Self {
            interstitial: wire.advertisement.interstitial.into(),
            rewarded: wire.advertisement.rewarded.into(),
            interstitial_schedule: AdSchedule {
                initial_delay: delay(
                    wire.advertisement.initial_interstitial_delay,
                    "initialInterstitialDelay",
                )?,
                minimum_delay: delay(
                    wire.advertisement.minimum_delay_between_interstitial,
                    "minimumDelayBetweenInterstitial",
                )?,
            },
        })
    }

    /// The placement to ask for when the game does not name one: the declared
    /// fallback, or the first declared placement. `None` when the config
    /// declares neither, which leaves the choice to the host.
    pub fn placement(&self, kind: AdKind) -> Option<&str> {
        let list = self.list(kind);
        list.fallback
            .as_deref()
            .or_else(|| list.placements.first().map(String::as_str))
    }

    /// Whether the config declares `placement` for `kind`.
    ///
    /// The SDK accepts an undeclared placement and substitutes the config's
    /// fallback for it, so a game that names one is not refused; the backend
    /// warns instead, because the name it asked for is not the one it gets.
    pub fn is_declared(&self, kind: AdKind, placement: &str) -> bool {
        self.list(kind)
            .placements
            .iter()
            .any(|declared| declared == placement)
    }

    fn list(&self, kind: AdKind) -> &PlacementList {
        match kind {
            AdKind::Interstitial => &self.interstitial,
            AdKind::Rewarded => &self.rewarded,
        }
    }
}

/// Why a config could not be read.
#[derive(Clone, Debug, PartialEq)]
pub enum ConfigError {
    /// The text is not the JSON the file is written in, or a placement entry
    /// has no `id`.
    Parse { detail: String },
    /// A delay is negative or not a finite number of seconds.
    Delay { field: &'static str, value: f64 },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse { detail } => write!(f, "the bridge config is not readable: {detail}"),
            Self::Delay { field, value } => {
                write!(f, "{field} is {value}, which is not a delay")
            }
        }
    }
}

impl std::error::Error for ConfigError {}

#[derive(Deserialize)]
struct Wire {
    #[serde(default)]
    advertisement: WireAdvertisement,
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct WireAdvertisement {
    #[serde(default)]
    minimum_delay_between_interstitial: Option<f64>,
    #[serde(default)]
    initial_interstitial_delay: Option<f64>,
    #[serde(default)]
    interstitial: WirePlacements,
    #[serde(default)]
    rewarded: WirePlacements,
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct WirePlacements {
    #[serde(default)]
    placements: Vec<WirePlacement>,
    #[serde(default)]
    placement_fallback: Option<String>,
}

#[derive(Deserialize)]
struct WirePlacement {
    id: String,
}

impl From<WirePlacements> for PlacementList {
    fn from(wire: WirePlacements) -> Self {
        Self {
            placements: wire
                .placements
                .into_iter()
                .map(|placement| placement.id)
                .collect(),
            fallback: wire.placement_fallback,
        }
    }
}

/// A seconds field as a duration. Absent is zero, which is the SDK's own
/// default for both delays.
fn delay(value: Option<f64>, field: &'static str) -> Result<Duration, ConfigError> {
    let Some(value) = value else {
        return Ok(Duration::ZERO);
    };
    if !value.is_finite() || value < 0.0 {
        return Err(ConfigError::Delay { field, value });
    }
    Ok(Duration::from_secs_f64(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The file the games ship, byte for byte.
    const SHIPPED: &str = r#"{
	"$schema": "https://bridge.playgama.com/v2/stable/schema.json",
	"advertisement": {
		"minimumDelayBetweenInterstitial": 120,
		"initialInterstitialDelay": 120,
		"interstitial": {
			"preloadOnStart": "game_over",
			"placementFallback": "game_over",
			"placements": [
				{ "id": "game_over" }
			]
		},
		"rewarded": {
			"preloadOnStart": "revive",
			"placementFallback": "revive",
			"placements": [
				{ "id": "revive" }
			]
		},
		"useBuiltInErrorPopup": true
	},
	"device": {
		"useBuiltInOrientationPopup": true,
		"supportedOrientations": ["landscape"]
	},
	"payments": [],
	"leaderboards": [],
	"sendAnalyticsEvents": true
}"#;

    #[test]
    fn the_shipped_config_reads_its_placements_and_its_rate_limit() {
        let config = PlaygamaConfig::from_json(SHIPPED).expect("the shipped file parses");
        assert_eq!(config.placement(AdKind::Interstitial), Some("game_over"));
        assert_eq!(config.placement(AdKind::Rewarded), Some("revive"));
        assert!(config.is_declared(AdKind::Rewarded, "revive"));
        assert!(!config.is_declared(AdKind::Rewarded, "bonus"));
        assert_eq!(
            config.interstitial_schedule,
            AdSchedule {
                initial_delay: Duration::from_secs(120),
                minimum_delay: Duration::from_secs(120),
            }
        );
    }

    #[test]
    fn a_config_without_an_advertisement_section_declares_nothing() {
        let config = PlaygamaConfig::from_json("{}").expect("an empty object parses");
        assert_eq!(config, PlaygamaConfig::default());
        assert_eq!(config.placement(AdKind::Interstitial), None);
        assert_eq!(config.interstitial_schedule, AdSchedule::default());
    }

    #[test]
    fn a_missing_fallback_uses_the_first_declared_placement() {
        let config = PlaygamaConfig::from_json(
            r#"{"advertisement": {"rewarded": {"placements": [{"id": "first"}, {"id": "second"}]}}}"#,
        )
        .expect("parses");
        assert_eq!(config.placement(AdKind::Rewarded), Some("first"));
        assert_eq!(config.rewarded.placements.len(), 2);
    }

    #[test]
    fn a_delay_that_is_not_a_delay_is_reported_field_by_field() {
        let error =
            PlaygamaConfig::from_json(r#"{"advertisement": {"initialInterstitialDelay": -5}}"#)
                .expect_err("a negative delay is not a duration");
        assert_eq!(
            error,
            ConfigError::Delay {
                field: "initialInterstitialDelay",
                value: -5.0,
            }
        );
        assert!(PlaygamaConfig::from_json("not json").is_err());
        assert!(
            PlaygamaConfig::from_json(r#"{"advertisement": {"rewarded": {"placements": [{}]}}}"#)
                .is_err()
        );
    }
}
