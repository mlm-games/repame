//! Texture containers and block codecs.
//!
//! Two layers:
//!
//! * **Containers** ([`dds`], [`ktx2`]) parse level/size/format/mips into a
//!   [`TextureInfo`] without touching the pixel payload.
//! * **Codecs** ([`BlockCodecRegistry`]) turn the payload into texels. BC1, BC3
//!   and BC4 plus the uncompressed paths decode; every other recognised format
//!   (BC2, BC5, BC6H, BC7, ETC/EAC, ASTC, Basis supercompression, half/float
//!   formats) is refused with [`ContentErrorKind::Unsupported`], never
//!   approximated.
//!
//! [`ContentErrorKind::Unsupported`]: crate::ContentErrorKind::Unsupported

pub mod bc;
pub mod dds;
pub mod ktx2;

use std::path::Path;

use crate::error::ContentError;

/// `"DDS "` in little-endian byte order.
pub const DDS_MAGIC: u32 = 0x2053_4444;
/// KTX 2 identifier.
pub const KTX2_IDENTIFIER: [u8; 12] = [
    0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
];
/// KTX 1 identifier.
pub const KTX1_IDENTIFIER: [u8; 12] = [
    0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
];

/// A recognised texture pixel format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    /// The container did not identify a format (`VK_FORMAT_UNDEFINED`).
    Undefined,
    Bc1,
    Bc2,
    Bc3,
    Bc4,
    Bc5,
    Bc6hUf16,
    Bc6hSf16,
    Bc7,
    Etc1Rgb8,
    Etc2Rgb8,
    Etc2Rgb8A1,
    Etc2Rgba8,
    EacR11,
    EacRg11,
    EacR11S,
    EacRg11S,
    Astc4x4,
    Astc5x5,
    Astc6x6,
    Astc8x8,
    R8,
    Rg8,
    Rgb8,
    Rgba8,
    Bgr8,
    Bgra8,
    Rgbx8,
    Bgrx8,
    La8,
    Rgb565,
    Bgr565,
    Rgba4444,
    Bgra4444,
    Rgb10a2,
    Bgr10a2,
    Rgb9e5,
    R16,
    Rg16,
    Rgb16,
    Rgba16,
    R16F,
    Rg16F,
    Rgb16F,
    Rgba16F,
    R32F,
    Rg32F,
    Rgb32F,
    Rgba32F,
}

impl TextureFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Undefined => "undefined",
            Self::Bc1 => "bc1",
            Self::Bc2 => "bc2",
            Self::Bc3 => "bc3",
            Self::Bc4 => "bc4",
            Self::Bc5 => "bc5",
            Self::Bc6hUf16 => "bc6h_uf16",
            Self::Bc6hSf16 => "bc6h_sf16",
            Self::Bc7 => "bc7",
            Self::Etc1Rgb8 => "etc1_rgb8",
            Self::Etc2Rgb8 => "etc2_rgb8",
            Self::Etc2Rgb8A1 => "etc2_rgb8_a1",
            Self::Etc2Rgba8 => "etc2_rgba8",
            Self::EacR11 => "eac_r11",
            Self::EacRg11 => "eac_rg11",
            Self::EacR11S => "eac_r11_signed",
            Self::EacRg11S => "eac_rg11_signed",
            Self::Astc4x4 => "astc_4x4",
            Self::Astc5x5 => "astc_5x5",
            Self::Astc6x6 => "astc_6x6",
            Self::Astc8x8 => "astc_8x8",
            Self::R8 => "r8",
            Self::Rg8 => "rg8",
            Self::Rgb8 => "rgb8",
            Self::Rgba8 => "rgba8",
            Self::Bgr8 => "bgr8",
            Self::Bgra8 => "bgra8",
            Self::Rgbx8 => "rgbx8",
            Self::Bgrx8 => "bgrx8",
            Self::La8 => "la8",
            Self::Rgb565 => "rgb565",
            Self::Bgr565 => "bgr565",
            Self::Rgba4444 => "rgba4444",
            Self::Bgra4444 => "bgra4444",
            Self::Rgb10a2 => "rgb10a2",
            Self::Bgr10a2 => "bgr10a2",
            Self::Rgb9e5 => "rgb9e5",
            Self::R16 => "r16",
            Self::Rg16 => "rg16",
            Self::Rgb16 => "rgb16",
            Self::Rgba16 => "rgba16",
            Self::R16F => "r16f",
            Self::Rg16F => "rg16f",
            Self::Rgb16F => "rgb16f",
            Self::Rgba16F => "rgba16f",
            Self::R32F => "r32f",
            Self::Rg32F => "rg32f",
            Self::Rgb32F => "rgb32f",
            Self::Rgba32F => "rgba32f",
        }
    }

    /// Whether the payload is 4x4 block-compressed.
    pub fn is_block_compressed(self) -> bool {
        matches!(
            self,
            Self::Bc1
                | Self::Bc2
                | Self::Bc3
                | Self::Bc4
                | Self::Bc5
                | Self::Bc6hUf16
                | Self::Bc6hSf16
                | Self::Bc7
                | Self::Etc1Rgb8
                | Self::Etc2Rgb8
                | Self::Etc2Rgb8A1
                | Self::Etc2Rgba8
                | Self::EacR11
                | Self::EacRg11
                | Self::EacR11S
                | Self::EacRg11S
                | Self::Astc4x4
                | Self::Astc5x5
                | Self::Astc6x6
                | Self::Astc8x8
        )
    }

    /// Bytes per compressed block, or per pixel when uncompressed.
    pub fn bytes_per_block(self) -> usize {
        match self {
            Self::Undefined => 1,
            Self::Bc1
            | Self::Bc4
            | Self::Etc1Rgb8
            | Self::Etc2Rgb8
            | Self::Etc2Rgb8A1
            | Self::EacR11
            | Self::EacR11S => 8,
            Self::Bc2
            | Self::Bc3
            | Self::Bc5
            | Self::Bc6hUf16
            | Self::Bc6hSf16
            | Self::Bc7
            | Self::Etc2Rgba8
            | Self::EacRg11
            | Self::EacRg11S
            | Self::Astc4x4
            | Self::Astc5x5
            | Self::Astc6x6
            | Self::Astc8x8 => 16,
            Self::R8 => 1,
            Self::Rg8 | Self::La8 | Self::Rgb565 | Self::Bgr565 | Self::Rgba4444 | Self::Bgra4444
            | Self::R16 | Self::R16F => 2,
            Self::Rgb8 | Self::Bgr8 => 3,
            Self::Rgba8
            | Self::Bgra8
            | Self::Rgbx8
            | Self::Bgrx8
            | Self::Rgb10a2
            | Self::Bgr10a2
            | Self::Rgb9e5
            | Self::Rg16
            | Self::Rg16F
            | Self::R32F => 4,
            Self::Rgb16 | Self::Rgb16F => 6,
            Self::Rgba16 | Self::Rgba16F | Self::Rg32F => 8,
            Self::Rgb32F => 12,
            Self::Rgba32F => 16,
        }
    }

    /// Bytes occupied by one `width x height` image of this format.
    pub fn surface_bytes(self, width: u32, height: u32) -> usize {
        if self.is_block_compressed() {
            let blocks_x = width.div_ceil(4) as usize;
            let blocks_y = height.div_ceil(4) as usize;
            blocks_x.max(1) * blocks_y.max(1) * self.bytes_per_block()
        } else {
            width as usize * height as usize * self.bytes_per_block()
        }
    }
}

/// Texture shape, as declared by the container.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureKind {
    D2,
    D3,
    Cube,
    D2Array,
    CubeArray,
}

/// One image chunk (a cube face, array slice or volume) at one mip level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureLevel {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    /// Faces or array slices stored in this chunk.
    pub layers: u32,
    /// Byte offset of the chunk inside the container.
    pub offset: u64,
    /// Stored (possibly supercompressed) byte length.
    pub size: u64,
    /// Decoded byte length; equal to `size` when not supercompressed.
    pub uncompressed_size: u64,
}

/// Parsed container header: format, dimensions and the level table.
///
/// `levels` is in file order. For cube maps and arrays the codecs lay out one
/// image's whole mip chain at a time, so the first `mip_levels` entries are the
/// first image and every `decode_level(index)` reads the first image of `index`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextureInfo {
    pub format: TextureFormat,
    pub kind: TextureKind,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub layers: u32,
    pub mip_levels: u32,
    pub levels: Vec<TextureLevel>,
    pub srgb: bool,
    /// Container-specific supercompression scheme; 0 means none.
    pub supercompression: u32,
    /// Offset of the first level's payload.
    pub data_offset: u64,
}

impl TextureInfo {
    pub fn level(&self, index: usize) -> Option<&TextureLevel> {
        self.levels.get(index)
    }

    pub fn is_supercompressed(&self) -> bool {
        self.supercompression != 0
    }
}

/// Decoded texel data. Block formats decode to [`TextureFormat::Rgba8`];
/// uncompressed formats are returned in their stored layout, unchanged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedImage {
    pub format: TextureFormat,
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

/// One 4x4 block codec.
pub trait BlockCodec: Send + Sync {
    fn format(&self) -> TextureFormat;

    /// Decode one 4x4 block into 16 RGBA texels, row-major.
    fn decode_block(&self, block: &[u8]) -> Result<[[u8; 4]; 16], ContentError>;
}

/// Block codecs, keyed by [`TextureFormat`].
pub struct BlockCodecRegistry {
    codecs: Vec<Box<dyn BlockCodec>>,
}

impl BlockCodecRegistry {
    /// A registry with BC1, BC3 and BC4 registered.
    pub fn new() -> Self {
        Self::with_builtins()
    }

    pub fn empty() -> Self {
        Self { codecs: Vec::new() }
    }

    pub fn with_builtins() -> Self {
        let mut registry = Self::empty();
        for codec in bc::builtins() {
            registry.codecs.push(codec);
        }
        registry
    }

    /// Add a codec. Fails when one is already registered for the format.
    pub fn register(&mut self, codec: Box<dyn BlockCodec>) -> Result<(), String> {
        if self.codec(codec.format()).is_some() {
            return Err(format!(
                "block codec for `{}` is already registered",
                codec.format().as_str()
            ));
        }
        self.codecs.push(codec);
        Ok(())
    }

    pub fn codec(&self, format: TextureFormat) -> Option<&dyn BlockCodec> {
        self.codecs
            .iter()
            .map(Box::as_ref)
            .find(|codec| codec.format() == format)
    }

    pub fn formats(&self) -> impl Iterator<Item = TextureFormat> + '_ {
        self.codecs.iter().map(|codec| codec.format())
    }

    /// Decode the first image of mip `level`.
    pub fn decode_level(
        &self,
        info: &TextureInfo,
        level: usize,
        bytes: &[u8],
        path: &Path,
    ) -> Result<DecodedImage, ContentError> {
        if info.is_supercompressed() {
            return Err(ContentError::unsupported(
                path,
                format!(
                    "supercompression scheme {} ({}) requires a Basis transcoder",
                    info.supercompression,
                    ktx2::supercompression_name(info.supercompression)
                ),
            ));
        }
        let info_level = info.level(level).ok_or_else(|| {
            ContentError::format(path, format!("texture has no mip level {level}"))
        })?;
        let start = usize::try_from(info_level.offset)
            .map_err(|_| ContentError::format(path, "texture level offset overflows"))?;
        let end = start
            .checked_add(usize::try_from(info_level.size).map_err(|_| {
                ContentError::format(path, "texture level size overflows")
            })?)
            .ok_or_else(|| ContentError::format(path, "texture level range overflows"))?;
        let data = bytes.get(start..end).ok_or_else(|| {
            ContentError::format(path, format!("texture mip level {level} is truncated"))
        })?;

        let expected = info
            .format
            .surface_bytes(info_level.width, info_level.height);
        if data.len() < expected {
            return Err(ContentError::format(
                path,
                format!(
                    "texture mip level {level} holds {} bytes, need {expected}",
                    data.len()
                ),
            ));
        }
        if !info.format.is_block_compressed() {
            return Ok(DecodedImage {
                format: info.format,
                width: info_level.width,
                height: info_level.height,
                data: data[..expected].to_vec(),
            });
        }

        let codec = self.codec(info.format).ok_or_else(|| {
            ContentError::unsupported(
                path,
                format!(
                    "no block codec for `{}`; convert the texture to BC1, BC3, BC4 or an uncompressed format",
                    info.format.as_str()
                ),
            )
        })?;
        let width = info_level.width;
        let height = info_level.height;
        let block_bytes = info.format.bytes_per_block();
        let blocks_x = width.div_ceil(4) as usize;
        let blocks_y = height.div_ceil(4) as usize;
        let mut out = vec![0u8; width as usize * height as usize * 4];
        for block_y in 0..blocks_y {
            for block_x in 0..blocks_x {
                let block_start = (block_y * blocks_x + block_x) * block_bytes;
                let block = &data[block_start..block_start + block_bytes];
                let texels = codec.decode_block(block).map_err(|error| {
                    ContentError::format(path, format!("decode block: {}", error.message()))
                })?;
                for texel_y in 0..4usize {
                    let y = block_y * 4 + texel_y;
                    if y >= height as usize {
                        continue;
                    }
                    for texel_x in 0..4usize {
                        let x = block_x * 4 + texel_x;
                        if x >= width as usize {
                            continue;
                        }
                        let texel = texels[texel_y * 4 + texel_x];
                        let target = (y * width as usize + x) * 4;
                        out[target..target + 4].copy_from_slice(&texel);
                    }
                }
            }
        }
        Ok(DecodedImage {
            format: TextureFormat::Rgba8,
            width,
            height,
            data: out,
        })
    }

    /// Parse the container and decode mip 0 of its first image.
    pub fn decode(&self, bytes: &[u8], path: &Path) -> Result<DecodedImage, ContentError> {
        let info = parse_container(bytes, path)?;
        self.decode_level(&info, 0, bytes, path)
    }
}

impl Default for BlockCodecRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse a DDS or KTX2 container header.
pub fn parse_container(bytes: &[u8], path: &Path) -> Result<TextureInfo, ContentError> {
    if bytes.starts_with(&KTX2_IDENTIFIER) {
        ktx2::parse(bytes, path)
    } else if bytes.starts_with(&KTX1_IDENTIFIER) {
        Err(ContentError::unsupported(
            path,
            "KTX 1 containers are not supported; convert to KTX 2",
        ))
    } else if bytes.len() >= 4 && read_u32_le(bytes, 0, path)? == DDS_MAGIC {
        dds::parse(bytes, path)
    } else {
        Err(ContentError::unsupported(
            path,
            "unrecognised texture container; expected DDS or KTX 2",
        ))
    }
}

pub(crate) fn read_u32_le(bytes: &[u8], offset: usize, path: &Path) -> Result<u32, ContentError> {
    let end = offset + 4;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| ContentError::format(path, format!("container is truncated at byte {offset}")))?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

pub(crate) fn read_u64_le(bytes: &[u8], offset: usize, path: &Path) -> Result<u64, ContentError> {
    let end = offset + 8;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| ContentError::format(path, format!("container is truncated at byte {offset}")))?;
    Ok(u64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}

/// Number of mip levels a `width x height` image can hold.
pub(crate) fn max_mip_levels(width: u32, height: u32) -> u32 {
    32 - width.max(height).max(1).leading_zeros()
}
