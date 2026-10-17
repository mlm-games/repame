//! BC1, BC3 and BC4 block decoders.
//!
//! The fixed-point interpolation is the one used by the public-domain `bcdec`
//! reference (and therefore by Godot's `Image::decompress`), so the decoded
//! texels match the reference bit for bit. BC2, BC5, BC6H and BC7 are
//! recognised by the containers but have no codec here; the registry refuses
//! them with a typed error rather than approximating.

use crate::codec::{BlockCodec, TextureFormat};
use crate::error::ContentError;

/// All block codecs this crate implements.
pub(crate) fn builtins() -> Vec<Box<dyn BlockCodec>> {
    vec![Box::new(Bc1), Box::new(Bc3), Box::new(Bc4)]
}

/// BC1 / DXT1, including the 1-bit-alpha mode.
#[derive(Clone, Copy, Debug, Default)]
pub struct Bc1;

impl BlockCodec for Bc1 {
    fn format(&self) -> TextureFormat {
        TextureFormat::Bc1
    }

    fn decode_block(&self, block: &[u8]) -> Result<[[u8; 4]; 16], ContentError> {
        let (c0, c1, indices) = color_block(block)?;
        let palette = color_palette(c0, c1, false);
        Ok(expand_indices(&palette, indices))
    }
}

/// BC3 / DXT5: BC4 alpha block plus a colour block that always uses 4 colours.
#[derive(Clone, Copy, Debug, Default)]
pub struct Bc3;

impl BlockCodec for Bc3 {
    fn format(&self) -> TextureFormat {
        TextureFormat::Bc3
    }

    fn decode_block(&self, block: &[u8]) -> Result<[[u8; 4]; 16], ContentError> {
        let alpha = alpha_block(block.get(0..8).ok_or_else(|| short_block("bc3", 16, block.len()))?)?;
        let (c0, c1, indices) = color_block(
            block.get(8..16).ok_or_else(|| short_block("bc3", 16, block.len()))?,
        )?;
        let palette = color_palette(c0, c1, true);
        let colors = expand_indices(&palette, indices);
        let mut texels = [[0u8; 4]; 16];
        for (index, texel) in texels.iter_mut().enumerate() {
            let color = colors[index];
            *texel = [color[0], color[1], color[2], alpha[index]];
        }
        Ok(texels)
    }
}

/// BC4 / ATI1 / RGTC1: one smooth alpha channel, replicated to RGB on decode.
#[derive(Clone, Copy, Debug, Default)]
pub struct Bc4;

impl BlockCodec for Bc4 {
    fn format(&self) -> TextureFormat {
        TextureFormat::Bc4
    }

    fn decode_block(&self, block: &[u8]) -> Result<[[u8; 4]; 16], ContentError> {
        let alpha = alpha_block(block.get(0..8).ok_or_else(|| short_block("bc4", 8, block.len()))?)?;
        let mut texels = [[0u8; 4]; 16];
        for (index, texel) in texels.iter_mut().enumerate() {
            let value = alpha[index];
            *texel = [value, value, value, 255];
        }
        Ok(texels)
    }
}

fn short_block(codec: &str, expected: usize, actual: usize) -> ContentError {
    ContentError::format(
        std::path::Path::new("<block>"),
        format!("{codec} block needs {expected} bytes, got {actual}"),
    )
}

/// Unpack the two 5/6/5 endpoints and the 16 two-bit colour indices.
fn color_block(block: &[u8]) -> Result<(u16, u16, u32), ContentError> {
    if block.len() < 8 {
        return Err(short_block("bc colour", 8, block.len()));
    }
    let c0 = u16::from_le_bytes([block[0], block[1]]);
    let c1 = u16::from_le_bytes([block[2], block[3]]);
    let indices = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    Ok((c0, c1, indices))
}

fn expand_indices(palette: &[[u8; 4]; 4], indices: u32) -> [[u8; 4]; 16] {
    let mut texels = [[0u8; 4]; 16];
    for (index, texel) in texels.iter_mut().enumerate() {
        *texel = palette[((indices >> (2 * index)) & 0x3) as usize];
    }
    texels
}

fn raw565(value: u16) -> (u32, u32, u32) {
    (
        u32::from((value >> 11) & 0x1F),
        u32::from((value >> 5) & 0x3F),
        u32::from(value & 0x1F),
    )
}

fn mix(value: u32, multiplier: u32, bias: u32, shift: u32) -> u8 {
    ((value * multiplier + bias) >> shift) as u8
}

fn color_palette(c0: u16, c1: u16, opaque: bool) -> [[u8; 4]; 4] {
    let (r0, g0, b0) = raw565(c0);
    let (r1, g1, b1) = raw565(c1);
    let mut palette = [[0u8; 4]; 4];
    palette[0] = [
        mix(r0, 527, 23, 6),
        mix(g0, 259, 33, 6),
        mix(b0, 527, 23, 6),
        255,
    ];
    palette[1] = [
        mix(r1, 527, 23, 6),
        mix(g1, 259, 33, 6),
        mix(b1, 527, 23, 6),
        255,
    ];
    if c0 > c1 || opaque {
        palette[2] = [
            mix(2 * r0 + r1, 351, 61, 7),
            mix(2 * g0 + g1, 2763, 1039, 11),
            mix(2 * b0 + b1, 351, 61, 7),
            255,
        ];
        palette[3] = [
            mix(r0 + 2 * r1, 351, 61, 7),
            mix(g0 + 2 * g1, 2763, 1039, 11),
            mix(b0 + 2 * b1, 351, 61, 7),
            255,
        ];
    } else {
        palette[2] = [
            mix(r0 + r1, 1053, 125, 8),
            mix(g0 + g1, 4145, 1019, 11),
            mix(b0 + b1, 1053, 125, 8),
            255,
        ];
        palette[3] = [0, 0, 0, 0];
    }
    palette
}

/// Decode the 8-byte BC3/BC4 alpha block into 16 alpha values.
fn alpha_block(block: &[u8]) -> Result<[u8; 16], ContentError> {
    if block.len() < 8 {
        return Err(short_block("bc alpha", 8, block.len()));
    }
    let a0 = u32::from(block[0]);
    let a1 = u32::from(block[1]);
    let mut palette = [0u8; 8];
    palette[0] = a0 as u8;
    palette[1] = a1 as u8;
    if a0 > a1 {
        palette[2] = ((6 * a0 + a1 + 1) / 7) as u8;
        palette[3] = ((5 * a0 + 2 * a1 + 1) / 7) as u8;
        palette[4] = ((4 * a0 + 3 * a1 + 1) / 7) as u8;
        palette[5] = ((3 * a0 + 4 * a1 + 1) / 7) as u8;
        palette[6] = ((2 * a0 + 5 * a1 + 1) / 7) as u8;
        palette[7] = ((a0 + 6 * a1 + 1) / 7) as u8;
    } else {
        palette[2] = ((4 * a0 + a1 + 1) / 5) as u8;
        palette[3] = ((3 * a0 + 2 * a1 + 1) / 5) as u8;
        palette[4] = ((2 * a0 + 3 * a1 + 1) / 5) as u8;
        palette[5] = ((a0 + 4 * a1 + 1) / 5) as u8;
        palette[6] = 0;
        palette[7] = 255;
    }
    let mut indices = 0u64;
    for (index, byte) in block[2..8].iter().enumerate() {
        indices |= u64::from(*byte) << (8 * index);
    }
    let mut alpha = [0u8; 16];
    for (index, value) in alpha.iter_mut().enumerate() {
        *value = palette[((indices >> (3 * index)) & 0x7) as usize];
    }
    Ok(alpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bc1_block_decodes_to_exact_texels() {
        // c0 = 0xFFFF (white) > c1 = 0x0000: 4-colour mode.
        // Index byte 0x1B = 0b00011011 -> per row [3, 2, 1, 0].
        let block = [0xFF, 0xFF, 0x00, 0x00, 0x1B, 0x1B, 0x1B, 0x1B];
        let texels = Bc1.decode_block(&block).expect("bc1");
        let row = [
            [85, 85, 85, 255],
            [170, 170, 170, 255],
            [0, 0, 0, 255],
            [255, 255, 255, 255],
        ];
        for texel_y in 0..4 {
            assert_eq!(&texels[texel_y * 4..texel_y * 4 + 4], &row);
        }
    }

    #[test]
    fn bc1_three_color_mode_pins_reference_fixed_point() {
        // c0 = 0x0000 <= c1 = 0xFFFF: 3-colour plus transparent mode.
        // 2*c2 is the reference's fixed-point average, which rounds to
        // (128, 127, 128) rather than a channel-wise 127/128 blend.
        let block = [0x00, 0x00, 0xFF, 0xFF, 0x1B, 0x1B, 0x1B, 0x1B];
        let texels = Bc1.decode_block(&block).expect("bc1");
        let row = [
            [0, 0, 0, 0],
            [128, 127, 128, 255],
            [255, 255, 255, 255],
            [0, 0, 0, 255],
        ];
        for texel_y in 0..4 {
            assert_eq!(&texels[texel_y * 4..texel_y * 4 + 4], &row);
        }
    }

    #[test]
    fn bc3_block_decodes_alpha_and_color() {
        // a0 = 255 > a1 = 0 with the 6 index bytes 0x88,0xC6,0xFA packed as
        // base-8 digits 0..7 twice: alpha palette [255, 0, 218, 182, 145, 109, 73, 36].
        let mut block = [0u8; 16];
        block[0] = 255;
        block[1] = 0;
        block[2..8].copy_from_slice(&[0x88, 0xC6, 0xFA, 0x88, 0xC6, 0xFA]);
        block[8..16].copy_from_slice(&[0xFF, 0xFF, 0x00, 0x00, 0x1B, 0x1B, 0x1B, 0x1B]);
        let texels = Bc3.decode_block(&block).expect("bc3");
        let expected = [
            [85, 85, 85, 255],
            [170, 170, 170, 0],
            [0, 0, 0, 218],
            [255, 255, 255, 182],
            [85, 85, 85, 145],
            [170, 170, 170, 109],
            [0, 0, 0, 73],
            [255, 255, 255, 36],
            [85, 85, 85, 255],
            [170, 170, 170, 0],
            [0, 0, 0, 218],
            [255, 255, 255, 182],
            [85, 85, 85, 145],
            [170, 170, 170, 109],
            [0, 0, 0, 73],
            [255, 255, 255, 36],
        ];
        assert_eq!(texels, expected);
    }

    #[test]
    fn bc4_block_decodes_to_exact_texels() {
        let mut block = [0u8; 8];
        block[0] = 255;
        block[1] = 0;
        block[2..8].copy_from_slice(&[0x88, 0xC6, 0xFA, 0x88, 0xC6, 0xFA]);
        let texels = Bc4.decode_block(&block).expect("bc4");
        let palette = [255u8, 0, 218, 182, 145, 109, 73, 36];
        let mut expected = [[0u8; 4]; 16];
        for (index, texel) in expected.iter_mut().enumerate() {
            let value = palette[index % 8];
            *texel = [value, value, value, 255];
        }
        assert_eq!(texels, expected);
    }
}
