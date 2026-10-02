//! Console texture formats: decoding, palette indirection, mip chains and
//! conversion to the RGBA8 the renderer uploads.
//!
//! The renderer only ever samples RGBA8, so every other format is resolved
//! here — at import time, once per page — instead of in a shader. Palette
//! formats carry their TLUT through [`TextureSource`] so a page keeps the
//! indirection the game declared, and [`texel_from_rgba`] lets a game bake a
//! palette texture without re-deriving the packing.

/// Texel layouts a console-era page can hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TexFormat {
    I4,
    I8,
    Ia4,
    Ia8,
    Rgb565,
    Rgb5A3,
    Rgba8,
    R4,
    /// Palette-indexed, indirection resolved through a [`Palette`].
    Indexed {
        bits: PaletteBits,
    },
    /// Palette-indexed with a 3-bit alpha in the top bits.
    IndexedAlpha {
        bits: PaletteBits,
    },
}

/// Index width of a palettized format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PaletteBits {
    I4,
    I8,
}

impl PaletteBits {
    pub const fn per_row(self, width: usize) -> usize {
        match self {
            Self::I4 => width.div_ceil(2),
            Self::I8 => width,
        }
    }
}

impl TexFormat {
    /// Bytes per texel before palettization.
    pub const fn bytes_per_texel(self) -> usize {
        match self {
            Self::I4 => 0,
            Self::I8 => 1,
            Self::Ia4 => 1,
            Self::Ia8 => 1,
            Self::R4 => 0,
            Self::Rgb565 => 2,
            Self::Rgb5A3 => 2,
            Self::Rgba8 => 4,
            Self::Indexed { .. } | Self::IndexedAlpha { .. } => 0,
        }
    }

    pub const fn is_indexed(self) -> bool {
        matches!(self, Self::Indexed { .. } | Self::IndexedAlpha { .. })
    }

    pub const fn bits_per_index(self) -> Option<u8> {
        match self {
            Self::Indexed { bits } | Self::IndexedAlpha { bits } => Some(match bits {
                PaletteBits::I4 => 4,
                PaletteBits::I8 => 8,
            }),
            _ => None,
        }
    }

    /// Byte size of a `width` x `height` image, row-aligned.
    pub fn image_size(self, width: usize, height: usize) -> usize {
        let row = self.row_bytes(width);
        row * height
    }

    pub fn row_bytes(self, width: usize) -> usize {
        match self {
            Self::I4
            | Self::R4
            | Self::Indexed {
                bits: PaletteBits::I4,
            } => width.div_ceil(2),
            Self::IndexedAlpha {
                bits: PaletteBits::I4,
            } => width,
            _ => self.bytes_per_texel() * width,
        }
    }

    /// True when the format carries its own alpha channel.
    pub const fn has_alpha(self) -> bool {
        matches!(
            self,
            Self::Ia4 | Self::Ia8 | Self::Rgba8 | Self::Rgb5A3 | Self::IndexedAlpha { .. }
        )
    }
}

/// A decoded color lookup table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Palette {
    pub colors: Vec<[u8; 3]>,
    pub bits: PaletteBits,
}

impl Palette {
    pub fn new(colors: Vec<[u8; 3]>, bits: PaletteBits) -> Self {
        Self { colors, bits }
    }

    pub fn len(&self) -> usize {
        self.colors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.colors.is_empty()
    }

    pub fn entry_count(&self) -> usize {
        match self.bits {
            PaletteBits::I4 => 16,
            PaletteBits::I8 => 256,
        }
    }

    /// Color for an index; black past the end so a short table cannot panic.
    pub fn color(&self, index: u8) -> [u8; 3] {
        self.colors
            .get(usize::from(index))
            .copied()
            .unwrap_or([0, 0, 0])
    }
}

/// A page plus the palette it indirection-resolves through.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextureSource {
    pub format: TexFormat,
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
    pub palette: Option<Palette>,
}

/// Why texel decoding failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TexError {
    Truncated,
    DimensionMismatch,
    MissingPalette,
    PaletteOutOfRange,
}

/// Decodes a palette's packed texel data into RGBA8.
pub fn decode_palette(
    data: &[u8],
    format: TexFormat,
    palette: &Palette,
    width: usize,
    height: usize,
) -> Result<Vec<u8>, TexError> {
    if data.len() < format.image_size(width, height) {
        return Err(TexError::Truncated);
    }
    let mut out = vec![0_u8; width * height * 4];
    for y in 0..height {
        for x in 0..width {
            let index = index_at(data, format, y * width + x);
            if usize::from(index) >= palette.len() {
                return Err(TexError::PaletteOutOfRange);
            }
            let color = palette.color(index);
            let at = (y * width + x) * 4;
            out[at..at + 3].copy_from_slice(&color);
            out[at + 3] = if matches!(format, TexFormat::IndexedAlpha { .. }) {
                (index >> 5) << 5
            } else {
                0xFF
            };
        }
    }
    Ok(out)
}

fn index_at(data: &[u8], format: TexFormat, linear: usize) -> u8 {
    match format {
        TexFormat::Indexed {
            bits: PaletteBits::I4,
        }
        | TexFormat::I4 => {
            let byte = data[linear / 2];
            if linear.is_multiple_of(2) {
                byte >> 4
            } else {
                byte & 0xF
            }
        }
        TexFormat::Indexed {
            bits: PaletteBits::I8,
        }
        | TexFormat::I8 => data[linear],
        TexFormat::IndexedAlpha {
            bits: PaletteBits::I4,
        }
        | TexFormat::IndexedAlpha {
            bits: PaletteBits::I8,
        } => data[linear] & 0xF,
        _ => 0,
    }
}

/// Expands one packed texel to linear-ish RGBA8 bytes.
pub fn rgba_from_texel(bytes: &[u8], format: TexFormat) -> [u8; 4] {
    match format {
        TexFormat::I4 | TexFormat::I8 | TexFormat::R4 => {
            let v = match format {
                TexFormat::I4 => (bytes[0] >> 4) & 0xF,
                _ => bytes[0],
            };
            let max = if format == TexFormat::I4 { 15.0 } else { 255.0 };
            let level = ((v as f32 / max) * 255.0).round() as u8;
            [level, level, level, 0xFF]
        }
        TexFormat::Ia4 => {
            let packed = bytes[0];
            let index = (packed >> 4) & 0xF;
            let alpha = (packed & 0xF) * 0x11;
            let level = (index as f32 / 15.0 * 255.0).round() as u8;
            [level, level, level, alpha]
        }
        TexFormat::Ia8 => [bytes[0], bytes[0], bytes[0], bytes[1]],
        TexFormat::Rgb565 => {
            let packed = u16::from_be_bytes([bytes[0], bytes[1]]);
            let r = (packed >> 11) & 0x1F;
            let g = (packed >> 5) & 0x3F;
            let b = packed & 0x1F;
            [
                (u32::from(r) * 255 / 31) as u8,
                (u32::from(g) * 255 / 63) as u8,
                (u32::from(b) * 255 / 31) as u8,
                0xFF,
            ]
        }
        TexFormat::Rgb5A3 => {
            let packed = u16::from_be_bytes([bytes[0], bytes[1]]);
            if packed & 0x8000 != 0 {
                let r = (packed >> 10) & 0x1F;
                let g = (packed >> 5) & 0x1F;
                let b = packed & 0x1F;
                [
                    (u32::from(r) * 255 / 31) as u8,
                    (u32::from(g) * 255 / 31) as u8,
                    (u32::from(b) * 255 / 31) as u8,
                    0xFF,
                ]
            } else {
                [
                    ((packed >> 8) & 0xF) as u8 * 0x11,
                    ((packed >> 4) & 0xF) as u8 * 0x11,
                    (packed & 0xF) as u8 * 0x11,
                    ((packed >> 15) & 0x1) as u8,
                ]
            }
        }
        TexFormat::Rgba8 => [bytes[0], bytes[1], bytes[2], bytes[3]],
        TexFormat::Indexed { .. } | TexFormat::IndexedAlpha { .. } => {
            [bytes[0], bytes[0], bytes[0], 0xFF]
        }
    }
}

/// Packs one RGBA8 texel back into a source format, for palette baking.
pub fn texel_from_rgba(rgba: [u8; 4], format: TexFormat) -> Vec<u8> {
    match format {
        TexFormat::I4 | TexFormat::R4 => vec![rgba[0] >> 4],
        TexFormat::I8 => vec![rgba[0]],
        TexFormat::Ia4 => {
            let level = rgba[0] >> 4;
            vec![(level << 4) | (rgba[3] >> 4)]
        }
        TexFormat::Ia8 => vec![rgba[0], rgba[3]],
        TexFormat::Rgb565 => {
            let r = (u32::from(rgba[0]) * 31 / 255) as u16 & 0x1F;
            let g = (u32::from(rgba[1]) * 63 / 255) as u16 & 0x3F;
            let b = (u32::from(rgba[2]) * 31 / 255) as u16 & 0x1F;
            let packed = (r << 11) | (g << 5) | b;
            packed.to_be_bytes().to_vec()
        }
        TexFormat::Rgb5A3 => {
            if rgba[3] >= 0xE0 {
                let r = (u32::from(rgba[0]) * 31 / 255) & 0x1F;
                let g = (u32::from(rgba[1]) * 31 / 255) & 0x1F;
                let b = (u32::from(rgba[2]) * 31 / 255) & 0x1F;
                (((r << 10) | (g << 5) | b | 0x8000) as u16)
                    .to_be_bytes()
                    .to_vec()
            } else {
                let r = (u32::from(rgba[0]) >> 4) & 0xF;
                let g = (u32::from(rgba[1]) >> 4) & 0xF;
                let b = (u32::from(rgba[2]) >> 4) & 0xF;
                let a = (u32::from(rgba[3]) >> 5) & 0x1;
                (((r << 12) | (g << 8) | (b << 4) | a) as u16)
                    .to_be_bytes()
                    .to_vec()
            }
        }
        TexFormat::Rgba8 => rgba.to_vec(),
        TexFormat::Indexed { bits } | TexFormat::IndexedAlpha { bits } => match bits {
            PaletteBits::I4 => vec![rgba[0] >> 4],
            PaletteBits::I8 => vec![rgba[0]],
        },
    }
}

/// Decodes a full page to RGBA8, resolving palettes and 4-bit row packing.
pub fn decode_texels(source: &TextureSource) -> Result<Vec<u8>, TexError> {
    if source.width == 0 || source.height == 0 {
        return Err(TexError::DimensionMismatch);
    }
    if source.format.is_indexed() {
        let palette = source.palette.as_ref().ok_or(TexError::MissingPalette)?;
        return decode_palette(
            &source.data,
            source.format,
            palette,
            source.width,
            source.height,
        );
    }
    let row = source.format.row_bytes(source.width);
    if source.data.len() < row * source.height {
        return Err(TexError::Truncated);
    }
    let bpt = source.format.bytes_per_texel();
    if bpt == 0 {
        return decode_palette(
            &source.data,
            source.format,
            &Palette::new(vec![], PaletteBits::I4),
            source.width,
            source.height,
        );
    }
    let mut out = vec![0_u8; source.width * source.height * 4];
    for y in 0..source.height {
        for x in 0..source.width {
            let at = y * row + x * bpt;
            let rgba = rgba_from_texel(&source.data[at..at + bpt], source.format);
            let out_at = (y * source.width + x) * 4;
            out[out_at..out_at + 4].copy_from_slice(&rgba);
        }
    }
    Ok(out)
}

/// Nearest-neighbor index of the source texel a mip level samples.
fn resample_half_index(x: usize, y: usize) -> (usize, usize) {
    (x * 2, y * 2)
}

/// Box-filters an RGBA8 image down one mip level.
pub fn resample_half(rgba: &[u8], width: usize, height: usize) -> Vec<u8> {
    if width < 2 || height < 2 {
        return rgba.to_vec();
    }
    let (w, h) = (width / 2, height / 2);
    let mut out = vec![0_u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let mut sum = [0_u32; 4];
            for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let (sx, sy) = resample_half_index(x, y);
                let at = ((sy + dy) * width + sx + dx) * 4;
                if at + 3 < rgba.len() {
                    for channel in 0..4 {
                        sum[channel] += u32::from(rgba[at + channel]);
                    }
                }
            }
            let out_at = (y * w + x) * 4;
            for channel in 0..4 {
                out[out_at + channel] = (sum[channel] / 4) as u8;
            }
        }
    }
    out
}

/// Full mip chain down to 1x1, largest first.
pub fn mip_chain(rgba: &[u8], width: usize, height: usize) -> Vec<(usize, usize, Vec<u8>)> {
    let mut levels = vec![(width, height, rgba.to_vec())];
    let (mut w, mut h) = (width, height);
    while w > 1 || h > 1 {
        let next = resample_half(&levels[levels.len() - 1].2, w, h);
        w = (w / 2).max(1);
        h = (h / 2).max(1);
        levels.push((w, h, next));
    }
    levels
}

/// Finds the palette entry closest to an RGBA8 color, for palette baking.
pub fn palettize_rgba(rgba: [u8; 3], palette: &Palette) -> Option<u8> {
    if palette.is_empty() {
        return None;
    }
    let mut best = 0_usize;
    let mut best_distance = u32::MAX;
    for (index, entry) in palette.colors.iter().enumerate() {
        let dr = i32::from(entry[0]) - i32::from(rgba[0]);
        let dg = i32::from(entry[1]) - i32::from(rgba[1]);
        let db = i32::from(entry[2]) - i32::from(rgba[2]);
        let distance = (dr * dr + dg * dg + db * db) as u32;
        if distance < best_distance {
            best_distance = distance;
            best = index;
        }
    }
    Some(best as u8)
}

/// Runtime replacement hook: pages a game swapped in after the first upload.
#[derive(Clone, Debug, Default)]
pub struct TextureOverrides {
    entries: std::collections::HashMap<u32, Vec<u8>>,
}

impl TextureOverrides {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, page: u32, rgba: Vec<u8>) {
        self.entries.insert(page, rgba);
    }

    pub fn remove(&mut self, page: u32) {
        self.entries.remove(&page);
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn get(&self, page: u32) -> Option<&[u8]> {
        self.entries.get(&page).map(|data| data.as_slice())
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
