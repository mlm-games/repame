//! BC7 block decoder.
//!
//! A port of `bcdec_bc7` from the public-domain `bcdec` v0.97 reference, the
//! header vendored as Godot's `thirdparty/misc/bcdec.h` and used by
//! `modules/bcdec/image_decompress_bcdec.cpp`. All eight modes decode with the
//! `ARB_texture_compression_bptc` partition, rotation, P-bit and
//! index-selection layouts, and the reserved modes become transparent black
//! exactly as the reference does. Endpoints, indices and the 128-bit bitstream
//! live in fixed stack arrays, so one block costs no allocation: a bitstream
//! read, up to six endpoints and sixteen interpolations, with no fast path
//! beyond that single sequential pass.

use crate::codec::bc::short_block;
use crate::codec::{BlockCodec, TextureFormat};
use crate::error::ContentError;

/// BC7 / BPTC: all eight modes.
#[derive(Clone, Copy, Debug, Default)]
pub struct Bc7;

impl BlockCodec for Bc7 {
    fn format(&self) -> TextureFormat {
        TextureFormat::Bc7
    }

    fn decode_block(&self, block: &[u8]) -> Result<[[u8; 4]; 16], ContentError> {
        let block: &[u8; 16] = block
            .get(0..16)
            .and_then(|slice| slice.try_into().ok())
            .ok_or_else(|| short_block("bc7", 16, block.len()))?;
        Ok(decode(block))
    }
}

/// Endpoint precision per mode, colour side and alpha side; zero alpha bits
/// means the mode is opaque, as in the mode table of the BPTC extension.
const COLOR_BITS: [u32; 8] = [4, 6, 5, 7, 5, 7, 7, 5];
const ALPHA_BITS: [u32; 8] = [0, 0, 0, 0, 6, 8, 7, 5];
/// Modes carrying one P-bit per endpoint; mode 1 shares two, handled apart.
const MODE_PBITS: u32 = 0b1100_1011;
/// The three colour components, in the component-major order the block
/// stores endpoint values in.
const COLOR_COMPONENTS: [usize; 3] = [0, 1, 2];

const WEIGHTS_2: [i32; 4] = [0, 21, 43, 64];
const WEIGHTS_3: [i32; 8] = [0, 9, 18, 27, 37, 46, 55, 64];
const WEIGHTS_4: [i32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

fn decode(block: &[u8; 16]) -> [[u8; 4]; 16] {
    let mut bits = BitReader::new(block);

    // The mode is the position of the first one bit.
    let mut mode = 0usize;
    while mode < 8 && bits.read_bit() == 0 {
        mode += 1;
    }
    if mode == 8 {
        return [[0, 0, 0, 0]; 16];
    }

    let num_partitions = match mode {
        0 | 2 => 3,
        1 | 3 | 7 => 2,
        _ => 1,
    };
    let partition = match mode {
        0 => bits.read_bits(4) as usize,
        1 | 2 | 3 | 7 => bits.read_bits(6) as usize,
        _ => 0,
    };
    let num_endpoints = num_partitions * 2;

    let mut rotation = 0usize;
    let mut index_selection = false;
    if mode == 4 || mode == 5 {
        rotation = bits.read_bits(2) as usize;
        if mode == 4 {
            index_selection = bits.read_bit() != 0;
        }
    }

    let color_bits = COLOR_BITS[mode];
    let alpha_bits = ALPHA_BITS[mode];
    let has_pbits = MODE_PBITS & (1 << mode) != 0;

    let mut endpoints = [[0i32; 4]; 6];
    for component in COLOR_COMPONENTS {
        for endpoint in &mut endpoints[..num_endpoints] {
            endpoint[component] = bits.read_bits(color_bits) as i32;
        }
    }
    if alpha_bits > 0 {
        for endpoint in &mut endpoints[..num_endpoints] {
            endpoint[3] = bits.read_bits(alpha_bits) as i32;
        }
    }

    if has_pbits {
        for endpoint in &mut endpoints[..num_endpoints] {
            for value in endpoint.iter_mut() {
                *value <<= 1;
            }
        }
        if mode == 1 {
            let p0 = bits.read_bit() as i32;
            let p1 = bits.read_bit() as i32;
            for (index, endpoint) in endpoints[..4].iter_mut().enumerate() {
                let pbit = if index < 2 { p0 } else { p1 };
                for component in COLOR_COMPONENTS {
                    endpoint[component] |= pbit;
                }
            }
        } else {
            for endpoint in &mut endpoints[..num_endpoints] {
                let pbit = bits.read_bit() as i32;
                for value in endpoint.iter_mut() {
                    *value |= pbit;
                }
            }
        }
    }

    for endpoint in &mut endpoints[..num_endpoints] {
        let color_precision = color_bits + u32::from(has_pbits);
        for component in COLOR_COMPONENTS {
            endpoint[component] = expand(endpoint[component], color_precision);
        }
        endpoint[3] = if alpha_bits > 0 {
            expand(endpoint[3], alpha_bits + u32::from(has_pbits))
        } else {
            255
        };
    }

    let index_bits = match mode {
        0 | 1 => 3,
        6 => 4,
        _ => 2,
    };
    let index2_bits = match mode {
        4 => 3,
        5 => 2,
        _ => 0,
    };
    let weights: &[i32] = match index_bits {
        2 => &WEIGHTS_2,
        3 => &WEIGHTS_3,
        _ => &WEIGHTS_4,
    };
    let weights2: &[i32] = if index2_bits == 2 {
        &WEIGHTS_2
    } else {
        &WEIGHTS_3
    };

    let mut indices = [[0u8; 4]; 4];
    for (y, row) in indices.iter_mut().enumerate() {
        for (x, value) in row.iter_mut().enumerate() {
            let pattern = partition_set(num_partitions, partition, y, x);
            let texel_bits = index_bits - u32::from(pattern & 0x80 != 0);
            *value = bits.read_bits(texel_bits) as u8;
        }
    }

    let mut texels = [[0u8; 4]; 16];
    for (y, row) in indices.iter().enumerate() {
        for (x, index) in row.iter().enumerate() {
            let pattern = partition_set(num_partitions, partition, y, x);
            let first = (pattern & 0x03) as usize * 2;
            let index = *index as usize;
            let (r, g, b, a) = if index2_bits == 0 {
                (
                    interpolate(endpoints[first][0], endpoints[first + 1][0], weights, index),
                    interpolate(endpoints[first][1], endpoints[first + 1][1], weights, index),
                    interpolate(endpoints[first][2], endpoints[first + 1][2], weights, index),
                    interpolate(endpoints[first][3], endpoints[first + 1][3], weights, index),
                )
            } else {
                let texel_bits = if y == 0 && x == 0 {
                    index2_bits - 1
                } else {
                    index2_bits
                };
                let index2 = bits.read_bits(texel_bits) as usize;
                if index_selection {
                    (
                        interpolate(endpoints[first][0], endpoints[first + 1][0], weights2, index2),
                        interpolate(endpoints[first][1], endpoints[first + 1][1], weights2, index2),
                        interpolate(endpoints[first][2], endpoints[first + 1][2], weights2, index2),
                        interpolate(endpoints[first][3], endpoints[first + 1][3], weights, index),
                    )
                } else {
                    (
                        interpolate(endpoints[first][0], endpoints[first + 1][0], weights, index),
                        interpolate(endpoints[first][1], endpoints[first + 1][1], weights, index),
                        interpolate(endpoints[first][2], endpoints[first + 1][2], weights, index),
                        interpolate(endpoints[first][3], endpoints[first + 1][3], weights2, index2),
                    )
                }
            };
            let (r, g, b, a) = match rotation {
                1 => (a, g, b, r),
                2 => (r, a, b, g),
                3 => (r, g, a, b),
                _ => (r, g, b, a),
            };
            texels[y * 4 + x] = [r as u8, g as u8, b as u8, a as u8];
        }
    }
    texels
}

fn partition_set(num_partitions: usize, partition: usize, y: usize, x: usize) -> u8 {
    if num_partitions == 1 {
        if y == 0 && x == 0 {
            0x80
        } else {
            0
        }
    } else {
        PARTITION_SETS[num_partitions - 2][partition][y][x]
    }
}

fn expand(value: i32, precision: u32) -> i32 {
    let value = value << (8 - precision);
    value | (value >> precision)
}

fn interpolate(a: i32, b: i32, weights: &[i32], index: usize) -> i32 {
    (a * (64 - weights[index]) + b * weights[index] + 32) >> 6
}

/// The 128-bit block, read LSB-first from the low word while refilling it from
/// the high word, exactly like `bcdec__bitstream_read_bits`.
struct BitReader {
    low: u64,
    high: u64,
}

impl BitReader {
    fn new(block: &[u8; 16]) -> Self {
        Self {
            low: u64::from_le_bytes([
                block[0], block[1], block[2], block[3], block[4], block[5], block[6], block[7],
            ]),
            high: u64::from_le_bytes([
                block[8], block[9], block[10], block[11], block[12], block[13], block[14],
                block[15],
            ]),
        }
    }

    fn read_bits(&mut self, count: u32) -> u32 {
        let mask = (1u64 << count) - 1;
        let bits = (self.low & mask) as u32;
        self.low = (self.low >> count) | ((self.high & mask) << (64 - count));
        self.high >>= count;
        bits
    }

    fn read_bit(&mut self) -> u32 {
        self.read_bits(1)
    }
}

/// Partition patterns for the two- and three-subset modes, exactly as
/// `bcdec`'s `partition_sets[2][64][4][4]`; 0x80 marks a fix-up texel
/// whose primary and secondary indices are stored one bit short.
static PARTITION_SETS: [[[[u8; 4]; 4]; 64]; 2] = [
    [ // two-subset partitions
        [[128,   0,   1,   1], [  0,   0,   1,   1], [  0,   0,   1,   1], [  0,   0,   1, 129]], // partition  0
        [[128,   0,   0,   1], [  0,   0,   0,   1], [  0,   0,   0,   1], [  0,   0,   0, 129]], // partition  1
        [[128,   1,   1,   1], [  0,   1,   1,   1], [  0,   1,   1,   1], [  0,   1,   1, 129]], // partition  2
        [[128,   0,   0,   1], [  0,   0,   1,   1], [  0,   0,   1,   1], [  0,   1,   1, 129]], // partition  3
        [[128,   0,   0,   0], [  0,   0,   0,   1], [  0,   0,   0,   1], [  0,   0,   1, 129]], // partition  4
        [[128,   0,   1,   1], [  0,   1,   1,   1], [  0,   1,   1,   1], [  1,   1,   1, 129]], // partition  5
        [[128,   0,   0,   1], [  0,   0,   1,   1], [  0,   1,   1,   1], [  1,   1,   1, 129]], // partition  6
        [[128,   0,   0,   0], [  0,   0,   0,   1], [  0,   0,   1,   1], [  0,   1,   1, 129]], // partition  7
        [[128,   0,   0,   0], [  0,   0,   0,   0], [  0,   0,   0,   1], [  0,   0,   1, 129]], // partition  8
        [[128,   0,   1,   1], [  0,   1,   1,   1], [  1,   1,   1,   1], [  1,   1,   1, 129]], // partition  9
        [[128,   0,   0,   0], [  0,   0,   0,   1], [  0,   1,   1,   1], [  1,   1,   1, 129]], // partition 10
        [[128,   0,   0,   0], [  0,   0,   0,   0], [  0,   0,   0,   1], [  0,   1,   1, 129]], // partition 11
        [[128,   0,   0,   1], [  0,   1,   1,   1], [  1,   1,   1,   1], [  1,   1,   1, 129]], // partition 12
        [[128,   0,   0,   0], [  0,   0,   0,   0], [  1,   1,   1,   1], [  1,   1,   1, 129]], // partition 13
        [[128,   0,   0,   0], [  1,   1,   1,   1], [  1,   1,   1,   1], [  1,   1,   1, 129]], // partition 14
        [[128,   0,   0,   0], [  0,   0,   0,   0], [  0,   0,   0,   0], [  1,   1,   1, 129]], // partition 15
        [[128,   0,   0,   0], [  1,   0,   0,   0], [  1,   1,   1,   0], [  1,   1,   1, 129]], // partition 16
        [[128,   1, 129,   1], [  0,   0,   0,   1], [  0,   0,   0,   0], [  0,   0,   0,   0]], // partition 17
        [[128,   0,   0,   0], [  0,   0,   0,   0], [129,   0,   0,   0], [  1,   1,   1,   0]], // partition 18
        [[128,   1, 129,   1], [  0,   0,   1,   1], [  0,   0,   0,   1], [  0,   0,   0,   0]], // partition 19
        [[128,   0, 129,   1], [  0,   0,   0,   1], [  0,   0,   0,   0], [  0,   0,   0,   0]], // partition 20
        [[128,   0,   0,   0], [  1,   0,   0,   0], [129,   1,   0,   0], [  1,   1,   1,   0]], // partition 21
        [[128,   0,   0,   0], [  0,   0,   0,   0], [129,   0,   0,   0], [  1,   1,   0,   0]], // partition 22
        [[128,   1,   1,   1], [  0,   0,   1,   1], [  0,   0,   1,   1], [  0,   0,   0, 129]], // partition 23
        [[128,   0, 129,   1], [  0,   0,   0,   1], [  0,   0,   0,   1], [  0,   0,   0,   0]], // partition 24
        [[128,   0,   0,   0], [  1,   0,   0,   0], [129,   0,   0,   0], [  1,   1,   0,   0]], // partition 25
        [[128,   1, 129,   0], [  0,   1,   1,   0], [  0,   1,   1,   0], [  0,   1,   1,   0]], // partition 26
        [[128,   0, 129,   1], [  0,   1,   1,   0], [  0,   1,   1,   0], [  1,   1,   0,   0]], // partition 27
        [[128,   0,   0,   1], [  0,   1,   1,   1], [129,   1,   1,   0], [  1,   0,   0,   0]], // partition 28
        [[128,   0,   0,   0], [  1,   1,   1,   1], [129,   1,   1,   1], [  0,   0,   0,   0]], // partition 29
        [[128,   1, 129,   1], [  0,   0,   0,   1], [  1,   0,   0,   0], [  1,   1,   1,   0]], // partition 30
        [[128,   0, 129,   1], [  1,   0,   0,   1], [  1,   0,   0,   1], [  1,   1,   0,   0]], // partition 31
        [[128,   1,   0,   1], [  0,   1,   0,   1], [  0,   1,   0,   1], [  0,   1,   0, 129]], // partition 32
        [[128,   0,   0,   0], [  1,   1,   1,   1], [  0,   0,   0,   0], [  1,   1,   1, 129]], // partition 33
        [[128,   1,   0,   1], [  1,   0, 129,   0], [  0,   1,   0,   1], [  1,   0,   1,   0]], // partition 34
        [[128,   0,   1,   1], [  0,   0,   1,   1], [129,   1,   0,   0], [  1,   1,   0,   0]], // partition 35
        [[128,   0, 129,   1], [  1,   1,   0,   0], [  0,   0,   1,   1], [  1,   1,   0,   0]], // partition 36
        [[128,   1,   0,   1], [  0,   1,   0,   1], [129,   0,   1,   0], [  1,   0,   1,   0]], // partition 37
        [[128,   1,   1,   0], [  1,   0,   0,   1], [  0,   1,   1,   0], [  1,   0,   0, 129]], // partition 38
        [[128,   1,   0,   1], [  1,   0,   1,   0], [  1,   0,   1,   0], [  0,   1,   0, 129]], // partition 39
        [[128,   1, 129,   1], [  0,   0,   1,   1], [  1,   1,   0,   0], [  1,   1,   1,   0]], // partition 40
        [[128,   0,   0,   1], [  0,   0,   1,   1], [129,   1,   0,   0], [  1,   0,   0,   0]], // partition 41
        [[128,   0, 129,   1], [  0,   0,   1,   0], [  0,   1,   0,   0], [  1,   1,   0,   0]], // partition 42
        [[128,   0, 129,   1], [  1,   0,   1,   1], [  1,   1,   0,   1], [  1,   1,   0,   0]], // partition 43
        [[128,   1, 129,   0], [  1,   0,   0,   1], [  1,   0,   0,   1], [  0,   1,   1,   0]], // partition 44
        [[128,   0,   1,   1], [  1,   1,   0,   0], [  1,   1,   0,   0], [  0,   0,   1, 129]], // partition 45
        [[128,   1,   1,   0], [  0,   1,   1,   0], [  1,   0,   0,   1], [  1,   0,   0, 129]], // partition 46
        [[128,   0,   0,   0], [  0,   1, 129,   0], [  0,   1,   1,   0], [  0,   0,   0,   0]], // partition 47
        [[128,   1,   0,   0], [  1,   1, 129,   0], [  0,   1,   0,   0], [  0,   0,   0,   0]], // partition 48
        [[128,   0, 129,   0], [  0,   1,   1,   1], [  0,   0,   1,   0], [  0,   0,   0,   0]], // partition 49
        [[128,   0,   0,   0], [  0,   0, 129,   0], [  0,   1,   1,   1], [  0,   0,   1,   0]], // partition 50
        [[128,   0,   0,   0], [  0,   1,   0,   0], [129,   1,   1,   0], [  0,   1,   0,   0]], // partition 51
        [[128,   1,   1,   0], [  1,   1,   0,   0], [  1,   0,   0,   1], [  0,   0,   1, 129]], // partition 52
        [[128,   0,   1,   1], [  0,   1,   1,   0], [  1,   1,   0,   0], [  1,   0,   0, 129]], // partition 53
        [[128,   1, 129,   0], [  0,   0,   1,   1], [  1,   0,   0,   1], [  1,   1,   0,   0]], // partition 54
        [[128,   0, 129,   1], [  1,   0,   0,   1], [  1,   1,   0,   0], [  0,   1,   1,   0]], // partition 55
        [[128,   1,   1,   0], [  1,   1,   0,   0], [  1,   1,   0,   0], [  1,   0,   0, 129]], // partition 56
        [[128,   1,   1,   0], [  0,   0,   1,   1], [  0,   0,   1,   1], [  1,   0,   0, 129]], // partition 57
        [[128,   1,   1,   1], [  1,   1,   1,   0], [  1,   0,   0,   0], [  0,   0,   0, 129]], // partition 58
        [[128,   0,   0,   1], [  1,   0,   0,   0], [  1,   1,   1,   0], [  0,   1,   1, 129]], // partition 59
        [[128,   0,   0,   0], [  1,   1,   1,   1], [  0,   0,   1,   1], [  0,   0,   1, 129]], // partition 60
        [[128,   0, 129,   1], [  0,   0,   1,   1], [  1,   1,   1,   1], [  0,   0,   0,   0]], // partition 61
        [[128,   0, 129,   0], [  0,   0,   1,   0], [  1,   1,   1,   0], [  1,   1,   1,   0]], // partition 62
        [[128,   1,   0,   0], [  0,   1,   0,   0], [  0,   1,   1,   1], [  0,   1,   1, 129]], // partition 63
    ],
    [ // three-subset partitions
        [[128,   0,   1, 129], [  0,   0,   1,   1], [  0,   2,   2,   1], [  2,   2,   2, 130]], // partition  0
        [[128,   0,   0, 129], [  0,   0,   1,   1], [130,   2,   1,   1], [  2,   2,   2,   1]], // partition  1
        [[128,   0,   0,   0], [  2,   0,   0,   1], [130,   2,   1,   1], [  2,   2,   1, 129]], // partition  2
        [[128,   2,   2, 130], [  0,   0,   2,   2], [  0,   0,   1,   1], [  0,   1,   1, 129]], // partition  3
        [[128,   0,   0,   0], [  0,   0,   0,   0], [129,   1,   2,   2], [  1,   1,   2, 130]], // partition  4
        [[128,   0,   1, 129], [  0,   0,   1,   1], [  0,   0,   2,   2], [  0,   0,   2, 130]], // partition  5
        [[128,   0,   2, 130], [  0,   0,   2,   2], [  1,   1,   1,   1], [  1,   1,   1, 129]], // partition  6
        [[128,   0,   1,   1], [  0,   0,   1,   1], [130,   2,   1,   1], [  2,   2,   1, 129]], // partition  7
        [[128,   0,   0,   0], [  0,   0,   0,   0], [129,   1,   1,   1], [  2,   2,   2, 130]], // partition  8
        [[128,   0,   0,   0], [  1,   1,   1,   1], [129,   1,   1,   1], [  2,   2,   2, 130]], // partition  9
        [[128,   0,   0,   0], [  1,   1, 129,   1], [  2,   2,   2,   2], [  2,   2,   2, 130]], // partition 10
        [[128,   0,   1,   2], [  0,   0, 129,   2], [  0,   0,   1,   2], [  0,   0,   1, 130]], // partition 11
        [[128,   1,   1,   2], [  0,   1, 129,   2], [  0,   1,   1,   2], [  0,   1,   1, 130]], // partition 12
        [[128,   1,   2,   2], [  0, 129,   2,   2], [  0,   1,   2,   2], [  0,   1,   2, 130]], // partition 13
        [[128,   0,   1, 129], [  0,   1,   1,   2], [  1,   1,   2,   2], [  1,   2,   2, 130]], // partition 14
        [[128,   0,   1, 129], [  2,   0,   0,   1], [130,   2,   0,   0], [  2,   2,   2,   0]], // partition 15
        [[128,   0,   0, 129], [  0,   0,   1,   1], [  0,   1,   1,   2], [  1,   1,   2, 130]], // partition 16
        [[128,   1,   1, 129], [  0,   0,   1,   1], [130,   0,   0,   1], [  2,   2,   0,   0]], // partition 17
        [[128,   0,   0,   0], [  1,   1,   2,   2], [129,   1,   2,   2], [  1,   1,   2, 130]], // partition 18
        [[128,   0,   2, 130], [  0,   0,   2,   2], [  0,   0,   2,   2], [  1,   1,   1, 129]], // partition 19
        [[128,   1,   1, 129], [  0,   1,   1,   1], [  0,   2,   2,   2], [  0,   2,   2, 130]], // partition 20
        [[128,   0,   0, 129], [  0,   0,   0,   1], [130,   2,   2,   1], [  2,   2,   2,   1]], // partition 21
        [[128,   0,   0,   0], [  0,   0, 129,   1], [  0,   1,   2,   2], [  0,   1,   2, 130]], // partition 22
        [[128,   0,   0,   0], [  1,   1,   0,   0], [130,   2, 129,   0], [  2,   2,   1,   0]], // partition 23
        [[128,   1,   2, 130], [  0, 129,   2,   2], [  0,   0,   1,   1], [  0,   0,   0,   0]], // partition 24
        [[128,   0,   1,   2], [  0,   0,   1,   2], [129,   1,   2,   2], [  2,   2,   2, 130]], // partition 25
        [[128,   1,   1,   0], [  1,   2, 130,   1], [129,   2,   2,   1], [  0,   1,   1,   0]], // partition 26
        [[128,   0,   0,   0], [  0,   1, 129,   0], [  1,   2, 130,   1], [  1,   2,   2,   1]], // partition 27
        [[128,   0,   2,   2], [  1,   1,   0,   2], [129,   1,   0,   2], [  0,   0,   2, 130]], // partition 28
        [[128,   1,   1,   0], [  0, 129,   1,   0], [  2,   0,   0,   2], [  2,   2,   2, 130]], // partition 29
        [[128,   0,   1,   1], [  0,   1,   2,   2], [  0,   1, 130,   2], [  0,   0,   1, 129]], // partition 30
        [[128,   0,   0,   0], [  2,   0,   0,   0], [130,   2,   1,   1], [  2,   2,   2, 129]], // partition 31
        [[128,   0,   0,   0], [  0,   0,   0,   2], [129,   1,   2,   2], [  1,   2,   2, 130]], // partition 32
        [[128,   2,   2, 130], [  0,   0,   2,   2], [  0,   0,   1,   2], [  0,   0,   1, 129]], // partition 33
        [[128,   0,   1, 129], [  0,   0,   1,   2], [  0,   0,   2,   2], [  0,   2,   2, 130]], // partition 34
        [[128,   1,   2,   0], [  0, 129,   2,   0], [  0,   1, 130,   0], [  0,   1,   2,   0]], // partition 35
        [[128,   0,   0,   0], [  1,   1, 129,   1], [  2,   2, 130,   2], [  0,   0,   0,   0]], // partition 36
        [[128,   1,   2,   0], [  1,   2,   0,   1], [130,   0, 129,   2], [  0,   1,   2,   0]], // partition 37
        [[128,   1,   2,   0], [  2,   0,   1,   2], [129, 130,   0,   1], [  0,   1,   2,   0]], // partition 38
        [[128,   0,   1,   1], [  2,   2,   0,   0], [  1,   1, 130,   2], [  0,   0,   1, 129]], // partition 39
        [[128,   0,   1,   1], [  1,   1, 130,   2], [  2,   2,   0,   0], [  0,   0,   1, 129]], // partition 40
        [[128,   1,   0, 129], [  0,   1,   0,   1], [  2,   2,   2,   2], [  2,   2,   2, 130]], // partition 41
        [[128,   0,   0,   0], [  0,   0,   0,   0], [130,   1,   2,   1], [  2,   1,   2, 129]], // partition 42
        [[128,   0,   2,   2], [  1, 129,   2,   2], [  0,   0,   2,   2], [  1,   1,   2, 130]], // partition 43
        [[128,   0,   2, 130], [  0,   0,   1,   1], [  0,   0,   2,   2], [  0,   0,   1, 129]], // partition 44
        [[128,   2,   2,   0], [  1,   2, 130,   1], [  0,   2,   2,   0], [  1,   2,   2, 129]], // partition 45
        [[128,   1,   0,   1], [  2,   2, 130,   2], [  2,   2,   2,   2], [  0,   1,   0, 129]], // partition 46
        [[128,   0,   0,   0], [  2,   1,   2,   1], [130,   1,   2,   1], [  2,   1,   2, 129]], // partition 47
        [[128,   1,   0, 129], [  0,   1,   0,   1], [  0,   1,   0,   1], [  2,   2,   2, 130]], // partition 48
        [[128,   2,   2, 130], [  0,   1,   1,   1], [  0,   2,   2,   2], [  0,   1,   1, 129]], // partition 49
        [[128,   0,   0,   2], [  1, 129,   1,   2], [  0,   0,   0,   2], [  1,   1,   1, 130]], // partition 50
        [[128,   0,   0,   0], [  2, 129,   1,   2], [  2,   1,   1,   2], [  2,   1,   1, 130]], // partition 51
        [[128,   2,   2,   2], [  0, 129,   1,   1], [  0,   1,   1,   1], [  0,   2,   2, 130]], // partition 52
        [[128,   0,   0,   2], [  1,   1,   1,   2], [129,   1,   1,   2], [  0,   0,   0, 130]], // partition 53
        [[128,   1,   1,   0], [  0, 129,   1,   0], [  0,   1,   1,   0], [  2,   2,   2, 130]], // partition 54
        [[128,   0,   0,   0], [  0,   0,   0,   0], [  2,   1, 129,   2], [  2,   1,   1, 130]], // partition 55
        [[128,   1,   1,   0], [  0, 129,   1,   0], [  2,   2,   2,   2], [  2,   2,   2, 130]], // partition 56
        [[128,   0,   2,   2], [  0,   0,   1,   1], [  0,   0, 129,   1], [  0,   0,   2, 130]], // partition 57
        [[128,   0,   2,   2], [  1,   1,   2,   2], [129,   1,   2,   2], [  0,   0,   2, 130]], // partition 58
        [[128,   0,   0,   0], [  0,   0,   0,   0], [  0,   0,   0,   0], [  2, 129,   1, 130]], // partition 59
        [[128,   0,   0, 130], [  0,   0,   0,   1], [  0,   0,   0,   2], [  0,   0,   0, 129]], // partition 60
        [[128,   2,   2,   2], [  1,   2,   2,   2], [  0,   2,   2,   2], [129,   2,   2, 130]], // partition 61
        [[128,   1,   0, 129], [  2,   2,   2,   2], [  2,   2,   2,   2], [  2,   2,   2, 130]], // partition 62
        [[128,   1,   1, 129], [  2,   0,   1,   1], [130,   2,   0,   1], [  2,   2,   2,   0]], // partition 63
    ],
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::unhex;

    /// One 16-byte block and the 64 bytes `bcdec_bc7` emits for it, one block
    /// per mode plus the all-zero and all-one blocks. The strings are copied
    /// from the harness that compiles the vendored `bcdec.h` (v0.97) at
    /// `Godot/thirdparty/misc/bcdec.h` and decodes these exact bytes, so a
    /// mismatch here is a mismatch with the declared reference.
    const REFERENCE_BLOCKS: &[(&str, &str)] = &[
        // Mode 0
        (
            "b5b7d6c30e06467a02b015bf5452b453",
            "d26d5affde6b29ffb573d6ffb573d6ff761630ff761630ff680f34ff680f34ffca0814ff522965ffe70000ff522965ff522965ff90183bffad1027ffca0814ff",
        ),
        (
            "754836382ae2b74c46409adf86093115",
            "3c2f5dff3f3b60ff4a5a6bff3c2f5dff5f1028ff5f1028ff211021ffa0102fff7dd411ff18ff29ff49ea1dff7dd411ff30f523ff95ca0cff49ea1dff18ff29ff",
        ),
        // Mode 1
        (
            "b6b7d6c30e06467a02b015bf5452b453",
            "cf40cfffcf40cfffc84d98ffc144b1ffd0557fffd75e66ffbe45b4ffcf40cfffee7819ffd0557fffdf3aebff8b575dff8b575dffae4b98ffe67032ffee7819ff",
        ),
        (
            "764836382ae2b74c46409adf86093115",
            "33823fff2a9637ff642064ff642064ff34c13bff34c13bff0efb12ff34c13bff0efb12ff14f119ff27d42eff0efb12ff3d6f46ff33823fff514755ff20a930ff",
        ),
        // Mode 2
        (
            "b4b7d6c30e06467a02b015bf5452b453",
            "de6300ff98286aff98286affd92043ffde6300ff98286aff563a7cffde6300ffd92043ff563a7cff98286affd60063ff8158dcff4c33b7ff4c33b7ffb57bffff",
        ),
        (
            "744836382ae2b74c46409adf86093115",
            "212110ff212110ff95acc1ff62c6b2ff212110ff62c6b2ff31dea5ff6539beff31dea5ff95acc1ff79429bff6539beff95acc1ff6539beff8c4a7bff8c4a7bff",
        ),
        // Mode 3
        (
            "b8b7d6c30e06467a02b015bf5452b453",
            "83403affae5177ffae5177ff86482aff6d646fffae5177ffae5177ff5a3000ff5382b7ff5382b7ff6d646fffd761b1ff83403aff5382b7ff5382b7ff86482aff",
        ),
        (
            "784836382ae2b74c46409adf86093115",
            "2b352dff251123ff251123ff377f41ff251123ff2b352dff251123ff969466ff251123ff315b37ff83954dff969466ff315b37ff969466ff709634ff709634ff",
        ),
        // Mode 4
        (
            "b0b7d6c30e06467a02b015bf5452b453",
            "6c8c57b9618c57b9776a4bb4613939ad6c7c51b6829d5dbb825a45b2618c57b96c8c57b9618c57b9619d5dbb618c57b9617c51b6773939ad6c6a4bb4828c57b9",
        ),
        (
            "704836382ae2b74c46409adf86093115",
            "5d733f49797c7c7c5d73dc495d73fb49797cbd7c5d73bd49797c3f7c426b9f1894843fad426b3f18797c9f7c426b2018426b7c18426b5e18797cbd7c426b2018",
        ),
        // Mode 5
        (
            "a0b7d6c30e06467a02b015bf5452b453",
            "6e91c11e6e95c11e6e95c11e6e95c11e6e9ac11e619196a96795ac625a9581ed619196a9619596a96e9ec11e619a96a95a9e81ed5a9181ed6795ac626795ac62",
        ),
        (
            "604836382ae2b74c46409adf86093115",
            "4eb77fa94ec144912dadbec172c144914ec1449172c144912dadbec12dc144914eb77fa92da3f9d993c144912da3f9d94ea3f9d94ea3f9d94eadbec12db77fa9",
        ),
        // Mode 6
        (
            "c0b7d6c30e06467a02b015bf5452b453",
            "db467a51de3c8246de3c8246bfaf24c6d06f587fdb467a51b4d802f4bfaf24c6d3656074d06f587fd852705ed06f587fd3656074bfaf24c6d55c6869d06f587f",
        ),
        (
            "404836382ae2b74c46409adf86093115",
            "3e769fb04772a5ae20828ab64772a5ae825acfa2775ec7a4b246f0989d4ee29c5b6ab3aa6e62c0a6775ec7a420828ab6297e90b43e769fb0506eabac297e90b4",
        ),
        // Mode 7
        (
            "80b7d6c30e06467a02b015bf5452b453",
            "d4315a60d64c6c5e1800108a4b080baf7f1005d6d3184961d4315a604b080baf1800108a4b080bafd7657d5dd64c6c5ed7657d5d1800108a4b080bafd4315a60",
        ),
        (
            "804836382ae2b74c46409adf86093115",
            "9c457e91cf559686cf55968634244da6cf5596869c457e91cf5596866734659ccf5596866734659c9c457e919fe61d976734659c6734659cc3f330cbc3f330cb",
        ),
        // All zero: no mode bit, the reference emits transparent black.
        (
            "00000000000000000000000000000000",
            "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        ),
        // All one: mode 0, every field set, opaque.
        (
            "ffffffffffffffffffffffffffffffff",
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        ),
    ];

    #[test]
    fn bc7_blocks_match_the_bcdec_reference() {
        for (vector, (input, expected)) in REFERENCE_BLOCKS.iter().enumerate() {
            let texels = Bc7.decode_block(&unhex::<16>(input)).expect("bc7");
            let expected = unhex::<64>(expected);
            for texel in 0..16 {
                assert_eq!(
                    &texels[texel],
                    &expected[texel * 4..texel * 4 + 4],
                    "vector {vector} texel {texel}"
                );
            }
        }
    }

    /// Partition-sweep digests: for each partition-carrying mode, all 64
    /// partitions of a payload forced to select that partition, decoded and
    /// hashed in the same hex-text FNV-1a order the reference harness uses.
    const SWEEP: &[(usize, u64)] = &[
        (0, 0x7349_236b_4038_3f95),
        (1, 0xf70c_23e5_9095_3c61),
        (2, 0x8528_0fcb_8143_9e2d),
        (3, 0xd4dc_5c25_a62a_5dc3),
        (7, 0xf830_7db6_7c04_5a27),
    ];

    #[test]
    fn bc7_partition_tables_match_the_reference_sweep() {
        for (mode, expected) in SWEEP {
            let mut digest = 0xcbf2_9ce4_8422_2325;
            for partition in 0..64 {
                let block = block_for_mode(0x9e37_79b9_7f4a_7c15 + *mode as u64, *mode, Some(partition));
                let texels = Bc7.decode_block(&block).expect("bc7");
                let mut flat = [0u8; 64];
                for (index, texel) in texels.iter().enumerate() {
                    flat[index * 4..index * 4 + 4].copy_from_slice(texel);
                }
                digest = fnv1a_hex(digest, &flat);
            }
            assert_eq!(digest, *expected, "mode {mode}");
        }
    }

    #[test]
    fn bc7_short_block_is_a_typed_format_error() {
        let error = Bc7.decode_block(&[0u8; 15]).expect_err("short block");
        assert_eq!(error.kind(), crate::error::ContentErrorKind::Format);
        assert!(error.message().contains("bc7"), "{}", error.message());
    }

    fn fnv1a_hex(hash: u64, bytes: &[u8]) -> u64 {
        const DIGITS: &[u8; 16] = b"0123456789abcdef";
        let mut hash = hash;
        for byte in bytes {
            for nibble in [byte >> 4, byte & 0x0f] {
                hash ^= u64::from(DIGITS[nibble as usize]);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        hash
    }

    /// The reference harness's block builder: an xorshift payload, then the
    /// unary mode prefix and an optional partition field written LSB-first.
    fn block_for_mode(seed: u64, mode: usize, partition: Option<usize>) -> [u8; 16] {
        let mut random = XorShift(seed);
        let mut block = [0u8; 16];
        for byte in &mut block {
            *byte = random.next() as u8;
        }
        set_bits(&mut block, 0, mode + 1, 1 << mode);
        if let Some(partition) = partition {
            let count = if mode == 0 { 4 } else { 6 };
            set_bits(&mut block, mode + 1, count, partition as u32);
        }
        block
    }

    fn set_bits(block: &mut [u8; 16], offset: usize, count: usize, value: u32) {
        for bit in 0..count {
            let index = offset + bit;
            let mask = 1u8 << (index & 7);
            if (value >> bit) & 1 != 0 {
                block[index >> 3] |= mask;
            } else {
                block[index >> 3] &= !mask;
            }
        }
    }

    struct XorShift(u64);

    impl XorShift {
        fn next(&mut self) -> u64 {
            let mut value = self.0;
            value ^= value << 13;
            value ^= value >> 7;
            value ^= value << 17;
            self.0 = value;
            value
        }
    }
}
