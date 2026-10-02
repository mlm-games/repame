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
            Self::Ia8 => 2,
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

    /// Console tile a format is stored in: block width, block height, block
    /// bytes. `None` for the engine's linear formats.
    pub const fn tile(self) -> Option<(usize, usize, usize)> {
        match self {
            Self::I4
            | Self::Indexed {
                bits: PaletteBits::I4,
            } => Some((8, 8, 32)),
            Self::I8
            | Self::Ia4
            | Self::Indexed {
                bits: PaletteBits::I8,
            } => Some((8, 4, 32)),
            Self::Ia8 | Self::Rgb565 | Self::Rgb5A3 => Some((4, 4, 32)),
            Self::Rgba8 => Some((4, 4, 64)),
            Self::R4 | Self::IndexedAlpha { .. } => None,
        }
    }

    /// Byte size of a `width` x `height` image.
    ///
    /// Tiled formats charge whole blocks, so a partial block at the right or
    /// bottom edge still costs its full tile; linear formats stay row-aligned.
    pub fn image_size(self, width: usize, height: usize) -> usize {
        match self.tile() {
            Some((block_w, block_h, block_bytes)) => {
                width.div_ceil(block_w) * height.div_ceil(block_h) * block_bytes
            }
            None => self.row_bytes(width) * height,
        }
    }

    /// Bytes per row of the linear storage; tiled formats use [`Self::tile`].
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
            Self::IndexedAlpha {
                bits: PaletteBits::I8,
            } => width * 2,
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

/// Visits every texel of a page in storage order with the byte offset of its
/// index or texel bytes. Tiled formats walk whole blocks, padding included.
fn walk_texels(
    format: TexFormat,
    width: usize,
    height: usize,
    mut visit: impl FnMut(usize, usize, usize),
) {
    let Some((block_w, block_h, block_bytes)) = format.tile() else {
        let row = format.row_bytes(width);
        for y in 0..height {
            for x in 0..width {
                let offset = match format {
                    TexFormat::R4 => y * row + x / 2,
                    _ => y * row + x * format.row_bytes(1),
                };
                visit(x, y, offset);
            }
        }
        return;
    };
    let block_row = block_bytes / block_h;
    let mut block = 0;
    for block_y in 0..height.div_ceil(block_h) {
        for block_x in 0..width.div_ceil(block_w) {
            let rows = block_h.min(height - block_y * block_h);
            let cols = block_w.min(width - block_x * block_w);
            for row in 0..rows {
                for col in 0..cols {
                    let offset = match format {
                        TexFormat::I4
                        | TexFormat::Indexed {
                            bits: PaletteBits::I4,
                        } => block + row * block_row + col / 2,
                        TexFormat::Rgba8 => block + row * 8 + col * 2,
                        TexFormat::Ia8 | TexFormat::Rgb565 | TexFormat::Rgb5A3 => {
                            block + row * block_row + col * 2
                        }
                        _ => block + row * block_row + col,
                    };
                    visit(block_x * block_w + col, block_y * block_h + row, offset);
                }
            }
            block += block_bytes;
        }
    }
}

/// Replicates a 3-bit value across the byte: 0 -> 0, 7 -> 255.
const fn expand3(value: u8) -> u8 {
    (value << 5) | (value << 2) | (value >> 1)
}

/// Index and alpha a texel stores; `alpha` is opaque for plain indexes.
fn texel_index(format: TexFormat, data: &[u8], x: usize, offset: usize) -> (u8, u8) {
    match format {
        TexFormat::Indexed {
            bits: PaletteBits::I4,
        }
        | TexFormat::I4 => {
            let byte = data[offset];
            let value = if x.is_multiple_of(2) {
                byte >> 4
            } else {
                byte & 0xF
            };
            (value, 0xFF)
        }
        TexFormat::Indexed {
            bits: PaletteBits::I8,
        }
        | TexFormat::I8 => (data[offset], 0xFF),
        TexFormat::IndexedAlpha {
            bits: PaletteBits::I4,
        } => {
            let byte = data[offset];
            (byte & 0xF, expand3(byte >> 5))
        }
        TexFormat::IndexedAlpha {
            bits: PaletteBits::I8,
        } => (data[offset], data[offset + 1]),
        _ => (0, 0xFF),
    }
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
    let mut error = None;
    walk_texels(format, width, height, |x, y, offset| {
        if error.is_some() {
            return;
        }
        let (index, alpha) = texel_index(format, data, x, offset);
        if usize::from(index) >= palette.len() {
            error = Some(TexError::PaletteOutOfRange);
            return;
        }
        let at = (y * width + x) * 4;
        out[at..at + 3].copy_from_slice(&palette.color(index));
        out[at + 3] = alpha;
    });
    match error {
        Some(err) => Err(err),
        None => Ok(out),
    }
}

/// Expands one packed texel to linear-ish RGBA8 bytes.
///
/// Bytes arrive in the format's own storage order: intensity formats are
/// gray with their level as alpha, IA4 keeps alpha in the high nibble, IA8
/// stores alpha first, RGBA8 the A/R then G/B planes.
pub fn rgba_from_texel(bytes: &[u8], format: TexFormat) -> [u8; 4] {
    match format {
        TexFormat::I4 | TexFormat::I8 | TexFormat::R4 => {
            let level = match format {
                TexFormat::I4 | TexFormat::R4 => ((bytes[0] >> 4) & 0xF) * 0x11,
                _ => bytes[0],
            };
            [level, level, level, level]
        }
        TexFormat::Ia4 => {
            let alpha = ((bytes[0] >> 4) & 0xF) * 0x11;
            let level = (bytes[0] & 0xF) * 0x11;
            [level, level, level, alpha]
        }
        TexFormat::Ia8 => [bytes[1], bytes[1], bytes[1], bytes[0]],
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
                    expand3(((packed >> 12) & 0x7) as u8),
                ]
            }
        }
        TexFormat::Rgba8 => [bytes[1], bytes[2], bytes[3], bytes[0]],
        TexFormat::Indexed { .. } | TexFormat::IndexedAlpha { .. } => {
            [bytes[0], bytes[0], bytes[0], 0xFF]
        }
    }
}

/// Packs one RGBA8 texel back into a source format, for palette baking.
pub fn texel_from_rgba(rgba: [u8; 4], format: TexFormat) -> Vec<u8> {
    match format {
        TexFormat::I4 | TexFormat::R4 => vec![(rgba[0] >> 4) << 4],
        TexFormat::I8 => vec![rgba[0]],
        TexFormat::Ia4 => vec![((rgba[3] >> 4) << 4) | (rgba[0] >> 4)],
        TexFormat::Ia8 => vec![rgba[3], rgba[0]],
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
                let a = (u32::from(rgba[3]) * 7 / 255) & 0x7;
                (((a << 12) | (r << 8) | (g << 4) | b) as u16)
                    .to_be_bytes()
                    .to_vec()
            }
        }
        TexFormat::Rgba8 => vec![rgba[3], rgba[0], rgba[1], rgba[2]],
        TexFormat::Indexed { bits } | TexFormat::IndexedAlpha { bits } => match bits {
            PaletteBits::I4 => vec![rgba[0] >> 4],
            PaletteBits::I8 => vec![rgba[0]],
        },
    }
}

/// Decodes a full page to RGBA8, resolving palettes and block tiles.
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
    if source.data.len() < source.format.image_size(source.width, source.height) {
        return Err(TexError::Truncated);
    }
    let mut out = vec![0_u8; source.width * source.height * 4];
    walk_texels(
        source.format,
        source.width,
        source.height,
        |x, y, offset| {
            let bytes: [u8; 4] = match source.format {
                TexFormat::I4 | TexFormat::R4 => {
                    let byte = source.data[offset];
                    let normalized = if x.is_multiple_of(2) {
                        byte & 0xF0
                    } else {
                        (byte & 0xF) << 4
                    };
                    [normalized, 0, 0, 0]
                }
                TexFormat::Rgba8 => [
                    source.data[offset],
                    source.data[offset + 1],
                    source.data[offset + 32],
                    source.data[offset + 33],
                ],
                format => {
                    let size = match format {
                        TexFormat::Ia8 | TexFormat::Rgb565 | TexFormat::Rgb5A3 => 2,
                        _ => format.bytes_per_texel().max(1),
                    };
                    let mut bytes = [0_u8; 4];
                    bytes[..size].copy_from_slice(&source.data[offset..offset + size]);
                    bytes
                }
            };
            let at = (y * source.width + x) * 4;
            out[at..at + 4].copy_from_slice(&rgba_from_texel(&bytes, source.format));
        },
    );
    Ok(out)
}

/// Box-filters an RGBA8 image down one mip level.
///
/// Each destination texel averages the source texels covering it, so odd
/// dimensions keep their last row or column instead of dropping it — the
/// result is always `(width / 2).max(1)` x `(height / 2).max(1)`.
pub fn resample_half(rgba: &[u8], width: usize, height: usize) -> Vec<u8> {
    let (w, h) = ((width / 2).max(1), (height / 2).max(1));
    if w == width && h == height {
        return rgba.to_vec();
    }
    let mut out = vec![0_u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let mut sum = [0_u32; 4];
            let mut count = 0_u32;
            for dy in 0..2 {
                for dx in 0..2 {
                    let (sx, sy) = (x * 2 + dx, y * 2 + dy);
                    if sx >= width || sy >= height {
                        continue;
                    }
                    let at = (sy * width + sx) * 4;
                    if at + 3 >= rgba.len() {
                        continue;
                    }
                    count += 1;
                    for channel in 0..4 {
                        sum[channel] += u32::from(rgba[at + channel]);
                    }
                }
            }
            let out_at = (y * w + x) * 4;
            for channel in 0..4 {
                out[out_at + channel] = if count == 0 {
                    0
                } else {
                    (sum[channel] / count) as u8
                };
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
