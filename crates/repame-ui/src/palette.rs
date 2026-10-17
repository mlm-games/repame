//! Palette tokens and the chip/button shapes the per-game palettes redraw.
//!
//! Eleven template-line games define their own colour constants and eight to
//! ten of them re-derive the same blending and the same two shapes. Two of them
//! (`kickback` and `rimshot`) are byte-identical, so [`Tokens`] takes that pair
//! as its default: warm ink, cream chips, a gold accent, and the six touch
//! colours the virtual stick draws with.
//!
//! Tokens are named by role, not by hue, so a game re-skins the kit by writing
//! a different [`Tokens`] value (`Kit::with_tokens`) without touching a control.
//! Art colours — sand, water, the player's teal — stay in the game: they are
//! game content, not chrome.
//!
//! [`mix`] and [`lighten`] are the two blending functions eleven copies share.
//! [`hand_cut`] is the optional irregular corner radii the identical pair uses
//! so a row of chips does not read as a row of copies.

use repose_core::prelude::{Color, Modifier, StateColors};
use repose_core::{AlignItems, Dp, JustifyContent, PaddingValues};

/// The font the template line's chrome is drawn in. The game registers the
/// face; the kit only names it.
pub const FONT: &str = "Fredoka";

/// Font family for glyph buttons (Material Symbols Outlined).
pub const MATERIAL_SYMBOLS: &str = "Material Symbols Outlined";

/// Side of a square glyph button in design units.
pub const ICON_BUTTON: f32 = 46.0;

/// Glyph size inside [`ICON_BUTTON`], in design units.
pub const ICON_GLYPH: f32 = 26.0;

/// Named colours for every kit control.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tokens {
    /// Outlines, text fill and hard shadows: the darkest chrome colour.
    pub ink: Color,
    /// Card and button face.
    pub cream: Color,
    /// Primary accent (win, revive, the transition disc).
    pub gold: Color,
    /// The accent's shaded half.
    pub gold_dark: Color,
    /// Attention accent (danger, counter).
    pub coral: Color,
    /// Background sky at the top of the frame.
    pub sky: Color,
    /// Background sky at the horizon.
    pub sky_dark: Color,
    /// Secondary accent.
    pub cyan: Color,
    /// Secondary accent, hot.
    pub pink: Color,
    /// Positive/success accent.
    pub green: Color,
    /// Full-screen dim behind a modal panel.
    pub scrim: Color,
    /// The hard button shadow.
    pub button_shadow: Color,
    /// Virtual stick base fill.
    pub touch_bg: Color,
    /// Virtual stick ring.
    pub touch_ring: Color,
    /// Virtual stick base shadow.
    pub touch_shadow: Color,
    /// Stick knob fill.
    pub touch_knob: Color,
    /// Stick knob edge.
    pub touch_knob_dark: Color,
    /// Stick centre pip.
    pub touch_center: Color,
}

impl Tokens {
    /// The byte-identical `kickback`/`rimshot` palette.
    pub const KICKBACK: Tokens = Tokens {
        ink: Color(0x2B, 0x1A, 0x14, 0xFF),
        cream: Color(0xFF, 0xF3, 0xDC, 0xFF),
        gold: Color(0xFF, 0xE0, 0x66, 0xFF),
        gold_dark: Color(0xD9, 0xA5, 0x1E, 0xFF),
        coral: Color(0xFF, 0x5F, 0xA2, 0xFF),
        sky: Color(0xF2, 0xA6, 0x5A, 0xFF),
        sky_dark: Color(0xE8, 0x89, 0x4A, 0xFF),
        cyan: Color(0x7F, 0xF0, 0xE4, 0xFF),
        pink: Color(0xFF, 0x8A, 0xC8, 0xFF),
        green: Color(0x8B, 0xD9, 0x6E, 0xFF),
        scrim: Color(0x2B, 0x1A, 0x14, 0x77),
        button_shadow: Color(0x2B, 0x1A, 0x14, 0x40),
        touch_bg: Color(0xFF, 0xF3, 0xDC, 0x99),
        touch_ring: Color(0x2B, 0x1A, 0x14, 0xFF),
        touch_shadow: Color(0x2B, 0x1A, 0x14, 0x2E),
        touch_knob: Color(0xFF, 0xFF, 0xFF, 0xFF),
        touch_knob_dark: Color(0xE8, 0xC4, 0x9E, 0xFF),
        touch_center: Color(0x2B, 0x1A, 0x14, 0xFF),
    };

    /// The cool sky/cream palette of the plain template, for a game that wants
    /// the kit's chrome without the sunset shift.
    pub const TEMPLATE: Tokens = Tokens {
        ink: Color(0x1E, 0x2A, 0x4A, 0xFF),
        cream: Color(0xFF, 0xF8, 0xEA, 0xFF),
        gold: Color(0xFF, 0xC9, 0x3C, 0xFF),
        gold_dark: Color(0xE0, 0x9B, 0x1F, 0xFF),
        coral: Color(0xF4, 0x71, 0x6A, 0xFF),
        sky: Color(0x7F, 0xC9, 0xF2, 0xFF),
        sky_dark: Color(0x5F, 0xAC, 0xDE, 0xFF),
        cyan: Color(0x46, 0xC8, 0xF5, 0xFF),
        pink: Color(0xFF, 0x9E, 0xB1, 0xFF),
        green: Color(0x7B, 0xD8, 0x8F, 0xFF),
        scrim: Color(0x1E, 0x2A, 0x4A, 0x66),
        button_shadow: Color(0x1E, 0x2A, 0x4A, 0x33),
        touch_bg: Color(0xFF, 0xF8, 0xEA, 0x99),
        touch_ring: Color(0x1E, 0x2A, 0x4A, 0xFF),
        touch_shadow: Color(0x1E, 0x2A, 0x4A, 0x22),
        touch_knob: Color(0xFF, 0xFF, 0xFF, 0xFF),
        touch_knob_dark: Color(0xC9, 0xD8, 0xEE, 0xFF),
        touch_center: Color(0x1E, 0x2A, 0x4A, 0xFF),
    };
}

impl Default for Tokens {
    fn default() -> Self {
        Self::KICKBACK
    }
}

/// Linear blend between two colours. `t` is clamped to `0..=1`, so `0` is `a`
/// and `1` is `b`.
pub fn mix(a: Color, b: Color, t: f32) -> Color {
    let t = t.clamp(0.0, 1.0);
    let blend = |from: u8, to: u8| -> u8 {
        (from as f32 + (to as f32 - from as f32) * t) as u8
    };
    Color(
        blend(a.0, b.0),
        blend(a.1, b.1),
        blend(a.2, b.2),
        blend(a.3, b.3),
    )
}

/// Blend a colour toward white.
pub fn lighten(color: Color, t: f32) -> Color {
    mix(color, Color(0xFF, 0xFF, 0xFF, 0xFF), t)
}

/// Blend a colour toward black.
pub fn darken(color: Color, t: f32) -> Color {
    mix(color, Color(0x00, 0x00, 0x00, 0xFF), t)
}

/// The five interaction tints a face colour needs.
///
/// Repose resolves a control's fill from these, which is why one helper is
/// enough for every button in the kit.
pub fn state_colors(face: Color, ink: Color) -> StateColors {
    StateColors {
        default: face,
        hovered: lighten(face, 0.08),
        focused: lighten(face, 0.04),
        pressed: mix(face, ink, 0.16),
        disabled: mix(face, ink, 0.30),
        dragged: mix(face, ink, 0.16),
    }
}

/// Irregular per-corner radii, pulled `±34%` around `radius` from `seed`.
///
/// The identical pair of games uses this so neighbouring chips do not share a
/// silhouette. Keep a per-control seed stable (a counter over the panel, or a
/// hash of the label) or the corners will jitter every frame.
pub fn hand_cut(radius: f32, seed: u32) -> [Dp; 4] {
    let wobble = |k: u32| -> Dp {
        let bits = seed.wrapping_mul(0x9E37_79B9 ^ k.wrapping_mul(0x85EB_CA6B));
        let frac = ((bits >> 16) & 0xFF) as f32 / 255.0 - 0.5;
        Dp(radius * (1.0 + frac * 0.34))
    };
    [wobble(1), wobble(2), wobble(3), wobble(4)]
}

/// The chunky game button: hard ink outline, hard shadow, centred content,
/// focusable.
///
/// `radius` is the corner radius in design units. A caller that wants the
/// irregular silhouette of [`hand_cut`] builds its own modifier and calls
/// [`Modifier::border_radii`].
pub fn chunky(
    tokens: &Tokens,
    face: Color,
    pad_x: f32,
    pad_y: f32,
    radius: f32,
    scale: f32,
) -> Modifier {
    Modifier::new()
        .state_colors(state_colors(face, tokens.ink))
        .border(Dp(3.0 * scale), tokens.ink, Dp(radius * scale))
        .padding_values(PaddingValues {
            left: Dp(pad_x * scale),
            right: Dp(pad_x * scale),
            top: Dp(pad_y * scale),
            bottom: Dp(pad_y * scale),
        })
        .shadow_with_color(Dp(0.0), Dp(5.0 * scale), tokens.button_shadow)
        .align_items(AlignItems::CENTER)
        .justify_content(JustifyContent::CENTER)
        .focusable(true)
}

/// The small rounded information chip (HUD counters, score, timer).
pub fn chip(tokens: &Tokens, face: Color, scale: f32) -> Modifier {
    Modifier::new()
        .background(face)
        .border(Dp(2.5 * scale), tokens.ink, Dp(14.0 * scale))
        .padding_values(PaddingValues {
            left: Dp(14.0 * scale),
            right: Dp(14.0 * scale),
            top: Dp(6.0 * scale),
            bottom: Dp(6.0 * scale),
        })
}

/// Text outline width in em-units for a `size`-point label: about a third of
/// the size in design units, never thinner than 2.5.
pub fn outline_for(size: f32) -> f32 {
    (size / 3.0).max(2.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: Color = Color(0x10, 0x20, 0x30, 0x40);
    const B: Color = Color(0x80, 0x90, 0xA0, 0xFF);

    #[test]
    fn mix_endpoints_are_the_inputs() {
        assert_eq!(mix(A, B, 0.0), A);
        assert_eq!(mix(A, B, 1.0), B);
        assert_eq!(mix(A, B, -3.0), A, "clamped below");
        assert_eq!(mix(A, B, 9.0), B, "clamped above");
    }

    #[test]
    fn mix_midpoint_is_the_average() {
        assert_eq!(mix(A, B, 0.5), Color(0x48, 0x58, 0x68, 0x9F));
    }

    #[test]
    fn lighten_and_darken_reach_white_and_black() {
        assert_eq!(lighten(A, 0.0), A);
        assert_eq!(lighten(A, 1.0), Color(0xFF, 0xFF, 0xFF, 0x40));
        assert_eq!(darken(A, 1.0), Color(0x00, 0x00, 0x00, 0x40));
    }

    #[test]
    fn state_colors_keep_the_face_as_default_and_shade_the_rest() {
        let colors = state_colors(A, Color(0x00, 0x00, 0x00, 0xFF));
        assert_eq!(colors.default, A);
        assert_eq!(colors.hovered, lighten(A, 0.08));
        assert_eq!(colors.focused, lighten(A, 0.04));
        assert_eq!(colors.pressed, mix(A, Color(0x00, 0x00, 0x00, 0xFF), 0.16));
        assert_eq!(colors.disabled, mix(A, Color(0x00, 0x00, 0x00, 0xFF), 0.30));
    }

    #[test]
    fn hand_cut_stays_within_the_documented_band() {
        for seed in [0u32, 1, 7, 0xDEAD_BEEF, 42] {
            for corner in hand_cut(16.0, seed) {
                assert!(
                    (16.0 * 0.66..=16.0 * 1.34).contains(&corner.0),
                    "seed {seed} produced {corner:?}"
                );
            }
        }
    }

    #[test]
    fn hand_cut_is_stable_for_a_seed() {
        assert_eq!(hand_cut(12.0, 99), hand_cut(12.0, 99));
        assert_ne!(hand_cut(12.0, 98), hand_cut(12.0, 99));
    }

    #[test]
    fn the_default_tokens_are_the_palette_pair() {
        assert_eq!(Tokens::default(), Tokens::KICKBACK);
        assert_ne!(Tokens::KICKBACK.ink, Tokens::TEMPLATE.ink);
    }
}
