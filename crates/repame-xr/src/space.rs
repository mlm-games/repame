//! Reference spaces: the origin convention poses are expressed in.

/// Where the XR play area's origin sits and how much of it is trusted.
///
/// The three spaces mirror the OpenXR reference space types and their WebXR
/// counterparts, so a game can be written once against this enum and run on
/// either backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ReferenceSpace {
    /// Centred on the viewer at the moment the session started, `-z` where the
    /// viewer looked. Sitting or standing still, no floor, no bounds.
    #[default]
    View,
    /// Floor-less play space centred where the viewer started. What a seated or
    /// room-scale title usually wants.
    Local,
    /// Centred on the physical floor and fixed to the room, so the origin does
    /// not move when the session restarts.
    Stage,
}

impl ReferenceSpace {
    /// All three, floor spaces after head spaces.
    pub const ALL: [ReferenceSpace; 3] = [
        ReferenceSpace::View,
        ReferenceSpace::Local,
        ReferenceSpace::Stage,
    ];

    /// Name this space carries in the OpenXR API.
    pub fn openxr_name(self) -> &'static str {
        match self {
            ReferenceSpace::View => "View",
            ReferenceSpace::Local => "Local",
            ReferenceSpace::Stage => "Stage",
        }
    }

    /// `XRReferenceSpaceType` this space maps to in WebXR.
    pub fn webxr_name(self) -> &'static str {
        match self {
            ReferenceSpace::View => "viewer",
            ReferenceSpace::Local => "local",
            ReferenceSpace::Stage => "local-floor",
        }
    }

    /// True when the origin is on the physical floor rather than at the viewer.
    pub fn is_floor_relative(self) -> bool {
        matches!(self, ReferenceSpace::Stage)
    }
}
