//! Action sets, binding paths, and the values a sync produces.
//!
//! An action is a named logical input (`"sprint"`, `"thumbstick"`); a binding
//! says which physical control on which controller layout feeds it. The path
//! grammar is OpenXR's:
//!
//! ```text
//! binding  ::= [ profile ] '/user/' user-path '/input/' input [ '/' component ]
//! profile  ::= '/interaction_profiles/' vendor '/' name
//! ```
//!
//! so `/interaction_profiles/oculus/touch_controller/user/hand/left/input/thumbstick`
//! is one complete binding, and `/user/hand/right/input/select/click` is the same
//! binding minus its profile. Both parse; only a full path can be registered with
//! a runtime, because the profile is what a runtime binds against.
//!
//! A component may be left off (`/input/select`). Per the spec that suggests
//! *every* component of that input, so a declared path without a component
//! covers a query with one, and [`InputPath::covers`] implements exactly that
//! one-directional rule.

use std::fmt;

use glam::Vec2;

use crate::error::PathError;
use crate::pose::Hand;
use crate::{XrError, XrResult};

/// The value type an action carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ActionKind {
    /// A press.
    Boolean,
    /// A 0..1 analog value.
    Float,
    /// A two-axis stick or trackpad.
    Vector2,
}

/// What an action was worth on the last sync.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActionValue {
    /// A press.
    Bool(bool),
    /// An analog value.
    Float(f32),
    /// A two-axis value.
    Vector2(Vec2),
}

impl ActionValue {
    /// The type of this value.
    pub fn kind(&self) -> ActionKind {
        match self {
            ActionValue::Bool(_) => ActionKind::Boolean,
            ActionValue::Float(_) => ActionKind::Float,
            ActionValue::Vector2(_) => ActionKind::Vector2,
        }
    }

    /// True when every component is finite, so the value can be trusted.
    pub fn is_finite(&self) -> bool {
        match self {
            ActionValue::Bool(_) => true,
            ActionValue::Float(value) => value.is_finite(),
            ActionValue::Vector2(value) => value.is_finite(),
        }
    }
}

/// One action plus what it was worth on the last sync.
#[derive(Clone, Debug, PartialEq)]
pub struct ActionState {
    /// Name of the owning set.
    pub set: String,
    /// Name of the action.
    pub action: String,
    /// Type the action was declared with.
    pub kind: ActionKind,
    /// Value the runtime reported.
    pub value: ActionValue,
    /// Whether the runtime considers the action bound and driven right now.
    pub active: bool,
}

impl ActionState {
    /// The value as a press, or [`XrError::TypeMismatch`].
    pub fn as_bool(&self) -> XrResult<bool> {
        match self.value {
            ActionValue::Bool(value) => Ok(value),
            _ => self.mismatch(ActionKind::Boolean),
        }
    }

    /// The value as an analog reading, or [`XrError::TypeMismatch`].
    pub fn as_float(&self) -> XrResult<f32> {
        match self.value {
            ActionValue::Float(value) => Ok(value),
            _ => self.mismatch(ActionKind::Float),
        }
    }

    /// The value as a two-axis reading, or [`XrError::TypeMismatch`].
    pub fn as_vector2(&self) -> XrResult<Vec2> {
        match self.value {
            ActionValue::Vector2(value) => Ok(value),
            _ => self.mismatch(ActionKind::Vector2),
        }
    }

    fn mismatch<T>(&self, expected: ActionKind) -> XrResult<T> {
        Err(XrError::TypeMismatch {
            action: self.action.clone(),
            expected,
            actual: self.kind,
        })
    }
}

/// Find one action's state by set and action name.
pub fn find_action<'a>(
    states: &'a [ActionState],
    set: &str,
    action: &str,
) -> XrResult<&'a ActionState> {
    states
        .iter()
        .find(|state| state.set == set && state.action == action)
        .ok_or_else(|| XrError::UnknownAction {
            set: set.to_string(),
            action: action.to_string(),
        })
}

/// One action's reading from one subaction path, as a sync collected it.
///
/// A reading is only meaningful when [`ActionReading::active`] is set: the spec
/// leaves an action's current state undefined while it is not active, so an
/// inactive reading always carries its type's zero value rather than whatever
/// the runtime last had there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActionReading {
    /// Whether the runtime considers the action bound and driven right now.
    pub active: bool,
    /// What the action was worth. The type's zero value whenever `active` is
    /// false.
    pub value: ActionValue,
}

impl ActionReading {
    /// The reading of an action nothing is driving: inactive, at `kind`'s zero
    /// value (`false`, `0.0`, or the zero vector).
    pub fn inactive(kind: ActionKind) -> Self {
        Self {
            active: false,
            value: match kind {
                ActionKind::Boolean => ActionValue::Bool(false),
                ActionKind::Float => ActionValue::Float(0.0),
                ActionKind::Vector2 => ActionValue::Vector2(Vec2::ZERO),
            },
        }
    }

    /// The reading of an action a runtime answered for. An inactive answer, or
    /// an active one carrying a non-finite value, becomes
    /// [`ActionReading::inactive`]: neither is a reading an app can use, and
    /// neither is passed on as one.
    pub fn reported(kind: ActionKind, active: bool, value: ActionValue) -> Self {
        if active && value.is_finite() {
            return Self {
                active: true,
                value,
            };
        }
        if active {
            log::warn!(
                "xr: a runtime reported a non-finite {kind:?} reading for an action; \
                 reporting the action as inactive"
            );
        }
        Self::inactive(kind)
    }
}

/// Combine one action's per-subaction readings into the single reading the app
/// sees, in the order the readings were collected.
///
/// A reading that is not active never drives the result, and the combined
/// reading is inactive when none is. Among active readings, boolean and analog
/// actions follow the same rule: the largest magnitude wins, so a boolean
/// action is true when any active reading is, and a float or vector2 action
/// takes its strongest active reading. A tie goes to the earlier reading, which
/// is what makes the result depend only on binding declaration order.
pub fn combine(kind: ActionKind, readings: &[ActionReading]) -> ActionReading {
    let mut best: Option<ActionValue> = None;
    for reading in readings.iter().filter(|reading| reading.active) {
        let better = match best {
            None => true,
            Some(current) => magnitude(&reading.value) > magnitude(&current),
        };
        if better {
            best = Some(reading.value);
        }
    }
    match best {
        Some(value) => ActionReading {
            active: true,
            value,
        },
        None => ActionReading::inactive(kind),
    }
}

/// How strongly a reading drives its action: `false` is none, `true` is full,
/// and the analog kinds are their own magnitude. One rule over this covers all
/// three kinds.
fn magnitude(value: &ActionValue) -> f32 {
    match value {
        ActionValue::Bool(value) => {
            if *value {
                1.0
            } else {
                0.0
            }
        }
        ActionValue::Float(value) => value.abs(),
        ActionValue::Vector2(value) => value.length(),
    }
}

/// Which device a binding path addresses.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum UserPath {
    /// `/user/head`.
    Head,
    /// `/user/hand/{left,right}`.
    Hand(Hand),
    /// `/user/gamepad`.
    Gamepad,
    /// Any other `/user/...` device, kept as its segments without `user/`.
    Custom(String),
}

impl UserPath {
    /// Parse the segments after `user/`.
    pub fn parse(segments: &[&str]) -> Result<Self, PathError> {
        match segments {
            [] => Err(PathError::UnknownUser),
            ["head"] => Ok(UserPath::Head),
            ["gamepad"] => Ok(UserPath::Gamepad),
            ["hand"] => Err(PathError::UnknownUser),
            ["hand", "left"] => Ok(UserPath::Hand(Hand::Left)),
            ["hand", "right"] => Ok(UserPath::Hand(Hand::Right)),
            _ => Ok(UserPath::Custom(segments.join("/"))),
        }
    }

    /// The `user/...` text, without a leading slash.
    pub fn as_str(&self) -> String {
        match self {
            UserPath::Head => "head".to_string(),
            UserPath::Hand(Hand::Left) => "hand/left".to_string(),
            UserPath::Hand(Hand::Right) => "hand/right".to_string(),
            UserPath::Gamepad => "gamepad".to_string(),
            UserPath::Custom(rest) => rest.clone(),
        }
    }
}

impl fmt::Display for UserPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.as_str())
    }
}

/// A user path plus the input and component it addresses.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct InputPath {
    /// Device the input lives on.
    pub user: UserPath,
    /// Input name, e.g. `"select"`, `"thumbstick"`, `"trigger"`.
    pub input: String,
    /// Component of that input, e.g. `"click"`, `"value"`, `"x"`. `None` means
    /// every component.
    pub component: Option<String>,
}

impl InputPath {
    /// Build from parts.
    pub fn new(user: UserPath, input: impl Into<String>, component: Option<String>) -> Self {
        Self {
            user,
            input: input.into(),
            component,
        }
    }

    /// Parse an absolute `/user/...` path, optionally with a profile prefix.
    pub fn parse(text: &str) -> XrResult<Self> {
        Ok(parse_binding_path(text)?.input)
    }

    /// True when a binding on this path feeds a query for `wanted`: same device
    /// and input, and a declared component covers the queried one.
    pub fn covers(&self, wanted: &InputPath) -> bool {
        if self.user != wanted.user || self.input != wanted.input {
            return false;
        }
        match (&self.component, &wanted.component) {
            (None, _) => true,
            (Some(declared), Some(queried)) => declared == queried,
            (Some(_), None) => false,
        }
    }

    /// The `/user/...` text, with the leading slash.
    pub fn as_path(&self) -> String {
        match &self.component {
            Some(component) => format!("/user/{}/input/{}/{}", self.user, self.input, component),
            None => format!("/user/{}/input/{}", self.user, self.input),
        }
    }
}

impl fmt::Display for InputPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.as_path())
    }
}

/// A controller layout, identified by its OpenXR interaction profile path.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct InteractionProfile {
    /// Vendor segment, e.g. `oculus`, `khr`, or `ext`.
    pub vendor: String,
    /// Profile name, e.g. `touch_controller`.
    pub name: String,
}

impl InteractionProfile {
    /// Khronos' lowest-common-denominator controller.
    pub const SIMPLE_CONTROLLER: &'static str = "/interaction_profiles/khr/simple_controller";
    /// HTC Vive wand.
    pub const VIVE_CONTROLLER: &'static str = "/interaction_profiles/htc/vive_controller";
    /// Windows Mixed Reality controller.
    pub const MOTION_CONTROLLER: &'static str = "/interaction_profiles/microsoft/motion_controller";
    /// Meta Touch.
    pub const TOUCH_CONTROLLER: &'static str = "/interaction_profiles/oculus/touch_controller";
    /// Valve Index controllers.
    pub const INDEX_CONTROLLER: &'static str = "/interaction_profiles/valve/index_controller";

    /// Profiles every runtime is expected to understand, used as the fallback
    /// set when an app declares none of its own.
    pub const COMMON: [&'static str; 5] = [
        Self::SIMPLE_CONTROLLER,
        Self::VIVE_CONTROLLER,
        Self::MOTION_CONTROLLER,
        Self::TOUCH_CONTROLLER,
        Self::INDEX_CONTROLLER,
    ];

    /// Build from vendor and name.
    pub fn new(vendor: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            vendor: vendor.into(),
            name: name.into(),
        }
    }

    /// Parse `/interaction_profiles/<vendor>/<name>`.
    pub fn parse(text: &str) -> Result<Self, PathError> {
        let segments = split(text)?;
        match segments.as_slice() {
            ["interaction_profiles", vendor, name] if !vendor.is_empty() && !name.is_empty() => {
                Ok(Self::new(*vendor, *name))
            }
            _ => Err(PathError::IncompleteProfile),
        }
    }

    /// The full `/interaction_profiles/...` path.
    pub fn as_path(&self) -> String {
        format!(
            "/interaction_profiles/{}/{}",
            self.vendor, self.name
        )
    }

    /// True when `text` is this profile's path.
    pub fn matches(&self, text: &str) -> bool {
        self.as_path() == text
    }
}

impl fmt::Display for InteractionProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.as_path())
    }
}

/// One binding: `action` is fed by `input` on `profile`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Binding {
    /// Action this feeds.
    pub action: String,
    /// Layout the input belongs to.
    pub profile: InteractionProfile,
    /// The input itself.
    pub input: InputPath,
}

/// A binding path that parsed, profile optional.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParsedBindingPath {
    /// Layout the binding targets, absent when the text had no profile prefix.
    pub profile: Option<InteractionProfile>,
    /// The input addressed.
    pub input: InputPath,
}

impl fmt::Display for ParsedBindingPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.profile {
            Some(profile) => write!(f, "{profile}{}", self.input.as_path()),
            None => f.write_str(&self.input.as_path()),
        }
    }
}

/// Split a path into segments, rejecting anything that is not a well-formed
/// OpenXR path.
fn split(text: &str) -> Result<Vec<&str>, PathError> {
    if text.is_empty() {
        return Err(PathError::Empty);
    }
    let Some(rest) = text.strip_prefix('/') else {
        return Err(PathError::NoLeadingSlash);
    };
    if rest.is_empty() {
        return Err(PathError::EmptySegment);
    }
    let segments: Vec<&str> = rest.split('/').collect();
    if segments.iter().any(|segment| segment.is_empty()) {
        return Err(PathError::EmptySegment);
    }
    Ok(segments)
}

/// Build the error for a malformed path.
fn bad_path(text: &str, reason: PathError) -> XrError {
    XrError::BadPath {
        path: text.to_string(),
        reason,
    }
}

/// Parse a full binding path, with or without an interaction profile prefix.
pub fn parse_binding_path(text: &str) -> XrResult<ParsedBindingPath> {
    let segments = split(text).map_err(|reason| bad_path(text, reason))?;

    let (profile, rest) = match segments.first() {
        Some(&"interaction_profiles") => {
            let profile = InteractionProfile::parse(text).map_err(|reason| bad_path(text, reason))?;
            (Some(profile), &segments[3..])
        }
        _ => (None, segments.as_slice()),
    };

    let (user_segments, remainder) = match rest.first() {
        Some(&"user") => {
            let tail = &rest[1..];
            if tail.is_empty() {
                return Err(bad_path(text, PathError::UnknownUser));
            }
            // The user path is everything up to `input`; extension devices like
            // vive trackers have several segments of their own.
            match tail.iter().position(|segment| *segment == "input") {
                Some(0) => return Err(bad_path(text, PathError::UnknownUser)),
                Some(index) => (&tail[..index], &tail[index + 1..]),
                None => (tail, &tail[tail.len()..]),
            }
        }
        _ => return Err(bad_path(text, PathError::NotAUserPath)),
    };

    if user_segments.is_empty() {
        return Err(bad_path(text, PathError::UnknownUser));
    }
    let user = UserPath::parse(user_segments).map_err(|reason| bad_path(text, reason))?;

    let Some((input, component)) = remainder.split_first() else {
        return Err(bad_path(text, PathError::MissingInput));
    };
    if input.is_empty() {
        return Err(bad_path(text, PathError::MissingName));
    }
    let component = match component {
        [] => None,
        [name] if !name.is_empty() => Some((*name).to_string()),
        _ => return Err(bad_path(text, PathError::MissingName)),
    };

    Ok(ParsedBindingPath {
        profile,
        input: InputPath::new(user, *input, component),
    })
}

/// A named action inside a set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Action {
    /// Name the runtime and the app use.
    pub name: String,
    /// Human-readable name shown in the runtime's binding UI.
    pub localized_name: String,
    /// Value type.
    pub kind: ActionKind,
}

/// A group of actions a runtime can bind at once, with the priorities that
/// decide which set wins when two drive the same input.
#[derive(Clone, Debug, PartialEq)]
pub struct ActionSet {
    /// Name the runtime and the app use.
    pub name: String,
    /// Human-readable name shown in the runtime's binding UI.
    pub localized_name: String,
    /// Higher priority wins.
    pub priority: u32,
    actions: Vec<Action>,
    bindings: Vec<Binding>,
}

impl ActionSet {
    /// An empty set.
    pub fn new(
        name: impl Into<String>,
        localized_name: impl Into<String>,
        priority: u32,
    ) -> Self {
        Self {
            name: name.into(),
            localized_name: localized_name.into(),
            priority,
            actions: Vec::new(),
            bindings: Vec::new(),
        }
    }

    /// Declare an action.
    pub fn add_action(
        &mut self,
        name: impl Into<String>,
        localized_name: impl Into<String>,
        kind: ActionKind,
    ) -> &mut Self {
        self.actions.push(Action {
            name: name.into(),
            localized_name: localized_name.into(),
            kind,
        });
        self
    }

    /// Bind an action to a path. The action must already be declared and the
    /// path must carry an interaction profile.
    pub fn add_binding(&mut self, action: &str, path: &str) -> XrResult<&mut Self> {
        let parsed = parse_binding_path(path)?;
        let Some(profile) = parsed.profile else {
            return Err(XrError::BadPath {
                path: path.to_string(),
                reason: PathError::MissingProfile,
            });
        };
        if self.action(action).is_none() {
            return Err(XrError::UnknownAction {
                set: self.name.clone(),
                action: action.to_string(),
            });
        }
        self.bindings.push(Binding {
            action: action.to_string(),
            profile,
            input: parsed.input,
        });
        Ok(self)
    }

    /// Every declared action.
    pub fn actions(&self) -> &[Action] {
        &self.actions
    }

    /// Every declared binding.
    pub fn bindings(&self) -> &[Binding] {
        &self.bindings
    }

    /// One action by name.
    pub fn action(&self, name: &str) -> Option<&Action> {
        self.actions.iter().find(|action| action.name == name)
    }

    /// Every binding declared against one controller layout, paired with the
    /// action it feeds. This is what a backend hands its runtime.
    pub fn bindings_for(&self, profile: &str) -> Vec<(&Action, &InputPath)> {
        self.bindings
            .iter()
            .filter(|binding| binding.profile.matches(profile))
            .filter_map(|binding| {
                self.action(&binding.action)
                    .map(|action| (action, &binding.input))
            })
            .collect()
    }

    /// Every controller layout this set binds against, in declaration order and
    /// without repeats.
    pub fn profiles(&self) -> Vec<&InteractionProfile> {
        let mut out: Vec<&InteractionProfile> = Vec::new();
        for binding in &self.bindings {
            if !out.iter().any(|seen| **seen == binding.profile) {
                out.push(&binding.profile);
            }
        }
        out
    }

    /// Every device path one action is bound on, in binding declaration order
    /// and without repeats.
    ///
    /// This is what a backend reads one action subaction per: an action bound
    /// on both hands is read once for the left and once for the right, and the
    /// readings combine into the single state an app sees
    /// ([`crate::combine`]). An action with no bindings has no device paths, and
    /// nothing can drive it.
    pub fn subaction_paths(&self, action: &str) -> Vec<&UserPath> {
        let mut out: Vec<&UserPath> = Vec::new();
        for binding in self.bindings.iter().filter(|binding| binding.action == action) {
            if !out.iter().any(|seen| **seen == binding.input.user) {
                out.push(&binding.input.user);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROUND_TRIP: [&str; 10] = [
        "/user/hand/left/input/select/click",
        "/user/hand/right/input/squeeze/value",
        "/user/hand/left/input/thumbstick",
        "/user/hand/right/input/trigger/touch",
        "/user/head/input/pose",
        "/user/gamepad/input/menu/click",
        "/user/vive_tracker_htcx/role/left_foot/input/select/click",
        "/interaction_profiles/oculus/touch_controller/user/hand/left/input/thumbstick",
        "/interaction_profiles/khr/simple_controller/user/hand/right/input/select/click",
        "/interaction_profiles/ext/hand_interaction_ext/user/hand/left/input/aim/activate",
    ];

    #[test]
    fn binding_paths_round_trip() {
        for text in ROUND_TRIP {
            let parsed = parse_binding_path(text).unwrap_or_else(|e| panic!("{text}: {e}"));
            assert_eq!(parsed.to_string(), text);
            assert_eq!(InputPath::parse(text).unwrap(), parsed.input);
        }

        let full = parse_binding_path(ROUND_TRIP[7]).unwrap();
        assert_eq!(
            full.profile,
            Some(InteractionProfile::new("oculus", "touch_controller"))
        );
        assert_eq!(full.input.user, UserPath::Hand(Hand::Left));
        assert_eq!(full.input.input, "thumbstick");
        assert_eq!(full.input.component, None);

        let bare = parse_binding_path(ROUND_TRIP[0]).unwrap();
        assert!(bare.profile.is_none());
        assert_eq!(bare.input.component.as_deref(), Some("click"));

        let tracker = parse_binding_path(ROUND_TRIP[6]).unwrap();
        assert_eq!(
            tracker.input.user,
            UserPath::Custom("vive_tracker_htcx/role/left_foot".to_string())
        );

        assert_eq!(
            InteractionProfile::parse(InteractionProfile::INDEX_CONTROLLER)
                .unwrap()
                .as_path(),
            InteractionProfile::INDEX_CONTROLLER
        );
        assert_eq!(InteractionProfile::COMMON.len(), 5);
    }

    #[test]
    fn malformed_binding_paths_are_rejected() {
        let cases: [(&str, PathError); 9] = [
            ("", PathError::Empty),
            ("user/hand/left/input/select", PathError::NoLeadingSlash),
            ("/", PathError::EmptySegment),
            ("/user/hand/left/", PathError::EmptySegment),
            ("/user/hand/left", PathError::MissingInput),
            ("/user//left/input/select", PathError::EmptySegment),
            ("/hand/left/input/select", PathError::NotAUserPath),
            ("/user/hand/input/select", PathError::UnknownUser),
            (
                "/interaction_profiles/khr/simple_controller",
                PathError::NotAUserPath,
            ),
        ];
        for (text, reason) in cases {
            match parse_binding_path(text) {
                Err(XrError::BadPath { path, reason: got }) => {
                    assert_eq!(path, text);
                    assert_eq!(got, reason, "for {text}");
                }
                other => panic!("{text} parsed as {other:?}"),
            }
        }

        assert!(matches!(
            parse_binding_path("/interaction_profiles/khr/user/hand/left/input/select"),
            Err(XrError::BadPath {
                reason: PathError::IncompleteProfile,
                ..
            })
        ));
        assert!(matches!(
            parse_binding_path("/user/hand/left/input/select/click/extra"),
            Err(XrError::BadPath {
                reason: PathError::MissingName,
                ..
            })
        ));
    }

    #[test]
    fn componentless_bindings_cover_every_component() {
        let declared = InputPath::parse("/user/hand/left/input/select").unwrap();
        let click = InputPath::parse("/user/hand/left/input/select/click").unwrap();
        let touch = InputPath::parse("/user/hand/left/input/select/touch").unwrap();
        let other_hand = InputPath::parse("/user/hand/right/input/select/click").unwrap();
        let other_input = InputPath::parse("/user/hand/left/input/menu/click").unwrap();

        assert!(declared.covers(&declared));
        assert!(declared.covers(&click));
        assert!(declared.covers(&touch));
        assert!(!declared.covers(&other_hand));
        assert!(!declared.covers(&other_input));
        assert!(click.covers(&click));
        assert!(!click.covers(&declared));
        assert!(!click.covers(&touch));
    }

    #[test]
    fn action_sets_match_bindings_per_profile_and_reject_bad_ones() {
        let mut set = ActionSet::new("gameplay", "Gameplay", 0);
        set.add_action("jump", "Jump", ActionKind::Boolean)
            .add_action("move", "Move", ActionKind::Vector2)
            .add_action("grip", "Grip", ActionKind::Float);
        set.add_binding(
            "jump",
            "/interaction_profiles/khr/simple_controller/user/hand/left/input/select/click",
        )
        .unwrap();
        set.add_binding(
            "move",
            "/interaction_profiles/oculus/touch_controller/user/hand/left/input/thumbstick",
        )
        .unwrap();
        set.add_binding(
            "grip",
            "/interaction_profiles/oculus/touch_controller/user/hand/right/input/squeeze/value",
        )
        .unwrap();

        assert!(matches!(
            set.add_binding("jump", "/user/hand/left/input/select/click"),
            Err(XrError::BadPath {
                reason: PathError::MissingProfile,
                ..
            })
        ));
        assert!(matches!(
            set.add_binding("sprint", "/interaction_profiles/khr/simple_controller/user/head/input/pose"),
            Err(XrError::UnknownAction { .. })
        ));
        assert!(matches!(
            set.add_binding("jump", "/user/hand/left/input/select"),
            Err(XrError::BadPath { .. })
        ));

        assert_eq!(set.bindings().len(), 3);
        let touch = set.bindings_for(InteractionProfile::TOUCH_CONTROLLER);
        assert_eq!(touch.len(), 2);
        assert_eq!(touch[0].0.name, "move");
        assert_eq!(touch[1].0.name, "grip");
        assert_eq!(set.bindings_for(InteractionProfile::VIVE_CONTROLLER).len(), 0);
        assert_eq!(set.profiles().len(), 2);
        assert_eq!(set.action("move").unwrap().kind, ActionKind::Vector2);

        let state = ActionState {
            set: "gameplay".to_string(),
            action: "move".to_string(),
            kind: ActionKind::Vector2,
            value: ActionValue::Vector2(glam::vec2(0.5, -1.0)),
            active: true,
        };
        assert_eq!(state.as_vector2().unwrap(), glam::vec2(0.5, -1.0));
        assert!(matches!(
            state.as_bool(),
            Err(XrError::TypeMismatch {
                expected: ActionKind::Boolean,
                actual: ActionKind::Vector2,
                ..
            })
        ));
        assert_eq!(find_action(&[state.clone()], "gameplay", "move").unwrap().active, true);
        assert!(matches!(
            find_action(&[state], "gameplay", "grip"),
            Err(XrError::UnknownAction { .. })
        ));
    }

    #[test]
    fn subaction_paths_come_from_the_actions_own_bindings_in_declaration_order() {
        let mut set = ActionSet::new("gameplay", "Gameplay", 0);
        set.add_action("jump", "Jump", ActionKind::Boolean)
            .add_action("shot", "Shot", ActionKind::Boolean);
        set.add_binding(
            "jump",
            "/interaction_profiles/khr/simple_controller/user/hand/right/input/select/click",
        )
        .unwrap();
        set.add_binding(
            "jump",
            "/interaction_profiles/oculus/touch_controller/user/hand/left/input/select/click",
        )
        .unwrap();
        // The same hand again on another profile: one subaction, not two.
        set.add_binding(
            "jump",
            "/interaction_profiles/valve/index_controller/user/hand/left/input/trigger/click",
        )
        .unwrap();
        set.add_binding(
            "shot",
            "/interaction_profiles/oculus/touch_controller/user/hand/right/input/select/click",
        )
        .unwrap();

        assert_eq!(
            set.subaction_paths("jump"),
            vec![&UserPath::Hand(Hand::Right), &UserPath::Hand(Hand::Left)]
        );
        assert_eq!(
            set.subaction_paths("shot"),
            vec![&UserPath::Hand(Hand::Right)]
        );
        assert!(set.subaction_paths("sprint").is_empty());
    }

    #[test]
    fn readings_combine_by_the_largest_active_magnitude_in_order() {
        fn reading(active: bool, value: ActionValue) -> ActionReading {
            ActionReading { active, value }
        }

        // Boolean: any active reading makes the action active, and an active
        // true is stronger than an active false.
        let readings = [
            reading(true, ActionValue::Bool(false)),
            reading(true, ActionValue::Bool(true)),
            reading(false, ActionValue::Bool(true)),
        ];
        assert_eq!(
            combine(ActionKind::Boolean, &readings),
            reading(true, ActionValue::Bool(true))
        );
        // A boolean nobody is driving is inactive, not whatever it last read.
        assert_eq!(
            combine(
                ActionKind::Boolean,
                &[reading(false, ActionValue::Bool(true))]
            ),
            ActionReading::inactive(ActionKind::Boolean)
        );

        // Float: the strongest active reading; a tie goes to the earlier one.
        let tied = [
            reading(true, ActionValue::Float(0.5)),
            reading(true, ActionValue::Float(-0.9)),
            reading(true, ActionValue::Float(0.9)),
        ];
        assert_eq!(
            combine(ActionKind::Float, &tied),
            reading(true, ActionValue::Float(-0.9))
        );
        // An inactive reading never drives, however large it says it is.
        let idle = [
            reading(false, ActionValue::Float(1.0)),
            reading(true, ActionValue::Float(0.25)),
        ];
        assert_eq!(
            combine(ActionKind::Float, &idle),
            reading(true, ActionValue::Float(0.25))
        );
        // A resting stick is active at zero, which is not the same as inactive.
        let resting = [
            reading(false, ActionValue::Float(1.0)),
            reading(true, ActionValue::Float(0.0)),
        ];
        assert_eq!(
            combine(ActionKind::Float, &resting),
            reading(true, ActionValue::Float(0.0))
        );

        // Vector2: length decides, direction does not.
        let sticks = [
            reading(true, ActionValue::Vector2(glam::vec2(0.5, 0.0))),
            reading(true, ActionValue::Vector2(glam::vec2(0.0, -0.75))),
        ];
        assert_eq!(
            combine(ActionKind::Vector2, &sticks),
            reading(true, ActionValue::Vector2(glam::vec2(0.0, -0.75)))
        );

        // Nothing to combine is the type's zero value, inactive.
        assert_eq!(combine(ActionKind::Float, &[]), ActionReading::inactive(ActionKind::Float));
        assert_eq!(
            combine(ActionKind::Vector2, &[]),
            ActionReading::inactive(ActionKind::Vector2)
        );
    }

    #[test]
    fn an_inactive_action_reads_at_its_types_zero_value() {
        assert_eq!(
            ActionReading::inactive(ActionKind::Boolean),
            ActionReading {
                active: false,
                value: ActionValue::Bool(false)
            }
        );
        assert_eq!(
            ActionReading::inactive(ActionKind::Float),
            ActionReading {
                active: false,
                value: ActionValue::Float(0.0)
            }
        );
        assert_eq!(
            ActionReading::inactive(ActionKind::Vector2),
            ActionReading {
                active: false,
                value: ActionValue::Vector2(glam::Vec2::ZERO)
            }
        );
    }

    #[test]
    fn non_finite_readings_are_reported_as_inactive() {
        assert_eq!(
            ActionReading::reported(ActionKind::Float, true, ActionValue::Float(f32::NAN)),
            ActionReading::inactive(ActionKind::Float)
        );
        assert_eq!(
            ActionReading::reported(
                ActionKind::Vector2,
                true,
                ActionValue::Vector2(glam::vec2(f32::INFINITY, 0.0))
            ),
            ActionReading::inactive(ActionKind::Vector2)
        );
        // Inactive readings are never carried, even a finite one.
        assert_eq!(
            ActionReading::reported(ActionKind::Float, false, ActionValue::Float(1.0)),
            ActionReading::inactive(ActionKind::Float)
        );
        assert_eq!(
            ActionReading::reported(ActionKind::Boolean, true, ActionValue::Bool(true)),
            ActionReading { active: true, value: ActionValue::Bool(true) }
        );
    }
}
