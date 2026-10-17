//! DDS container header parsing.
//!
//! Parses the 128-byte Microsoft DDS header (the `DDS_HEADER` plus the
//! `"DX10 "` extension when present) into a [`DdsHeader`] and a [`TextureInfo`]
//! level table, mirroring Godot's `modules/dds`. Formats and layouts the header
//! can express but that this crate cannot decode are still represented in the
//! header, so the refusal happens in the codec registry with a typed error.

use std::path::Path;

use crate::codec::{
    TextureFormat, TextureInfo, TextureKind, TextureLevel, max_mip_levels, read_u32_le,
};
use crate::error::ContentError;

const HEADER_SIZE: u32 = 124;
const PIXEL_FORMAT_SIZE: u32 = 32;
const HEADER_BYTES: usize = 128;
const DX10_HEADER_BYTES: usize = 148;

pub(crate) const DDSD_MIPMAPCOUNT: u32 = 0x0002_0000;
pub(crate) const DDPF_ALPHAPIXELS: u32 = 0x0000_0001;
pub(crate) const DDPF_ALPHAONLY: u32 = 0x0000_0002;
pub(crate) const DDPF_FOURCC: u32 = 0x0000_0004;
pub(crate) const DDSC2_CUBEMAP: u32 = 0x0000_0200;
pub(crate) const DDSC2_VOLUME: u32 = 0x0020_0000;

/// The `DDS_PIXELFORMAT` sub-structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DdsPixelFormat {
    pub size: u32,
    pub flags: u32,
    pub four_cc: u32,
    pub rgb_bit_count: u32,
    pub red_mask: u32,
    pub green_mask: u32,
    pub blue_mask: u32,
    pub alpha_mask: u32,
}

/// The optional `"DX10 "` header extension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dx10Header {
    pub dxgi_format: u32,
    pub dimension: u32,
    pub misc_flags: u32,
    pub array_size: u32,
    pub misc_flags2: u32,
}

/// A parsed DDS header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DdsHeader {
    pub flags: u32,
    pub height: u32,
    pub width: u32,
    pub pitch_or_linear_size: u32,
    pub depth: u32,
    pub mipmap_count: u32,
    pub pixel_format: DdsPixelFormat,
    pub caps: u32,
    pub caps2: u32,
    pub caps3: u32,
    pub caps4: u32,
    pub dx10: Option<Dx10Header>,
    /// The format the header resolves to.
    pub format: TextureFormat,
    pub srgb: bool,
}

impl DdsHeader {
    /// Mip levels actually present, honouring `DDSD_MIPMAPCOUNT`.
    pub fn levels_declared(&self) -> u32 {
        if self.flags & DDSD_MIPMAPCOUNT != 0 {
            self.mipmap_count.max(1)
        } else {
            1
        }
    }
}

/// Parse the DDS header without looking at the payload.
pub fn parse_header(bytes: &[u8], path: &Path) -> Result<DdsHeader, ContentError> {
    if bytes.len() < 4 || read_u32_le(bytes, 0, path)? != crate::codec::DDS_MAGIC {
        return Err(ContentError::format(path, "not a DDS container"));
    }
    if bytes.len() < HEADER_BYTES {
        return Err(ContentError::format(
            path,
            format!("DDS header needs {HEADER_BYTES} bytes, got {}", bytes.len()),
        ));
    }
    let size = read_u32_le(bytes, 4, path)?;
    if size != HEADER_SIZE {
        return Err(ContentError::format(
            path,
            format!("DDS header declares {size} bytes, expected {HEADER_SIZE}"),
        ));
    }
    let flags = read_u32_le(bytes, 8, path)?;
    let pixel_format = DdsPixelFormat {
        size: read_u32_le(bytes, 76, path)?,
        flags: read_u32_le(bytes, 80, path)?,
        four_cc: read_u32_le(bytes, 84, path)?,
        rgb_bit_count: read_u32_le(bytes, 88, path)?,
        red_mask: read_u32_le(bytes, 92, path)?,
        green_mask: read_u32_le(bytes, 96, path)?,
        blue_mask: read_u32_le(bytes, 100, path)?,
        alpha_mask: read_u32_le(bytes, 104, path)?,
    };
    if pixel_format.size != PIXEL_FORMAT_SIZE {
        return Err(ContentError::format(
            path,
            format!(
                "DDS pixel format declares {} bytes, expected {PIXEL_FORMAT_SIZE}",
                pixel_format.size
            ),
        ));
    }

    let mut dx10 = None;
    let format;
    let mut srgb = false;
    if pixel_format.flags & DDPF_FOURCC != 0 {
        if pixel_format.four_cc == four_cc(b"DX10") {
            if bytes.len() < DX10_HEADER_BYTES {
                return Err(ContentError::format(path, "DDS DX10 header is truncated"));
            }
            let extension = Dx10Header {
                dxgi_format: read_u32_le(bytes, 128, path)?,
                dimension: read_u32_le(bytes, 132, path)?,
                misc_flags: read_u32_le(bytes, 136, path)?,
                array_size: read_u32_le(bytes, 140, path)?,
                misc_flags2: read_u32_le(bytes, 144, path)?,
            };
            let mapped = dxgi_format(extension.dxgi_format).ok_or_else(|| {
                ContentError::unsupported(
                    path,
                    format!("DDS DXGI format {} is not implemented", extension.dxgi_format),
                )
            })?;
            format = mapped.0;
            srgb = mapped.1;
            dx10 = Some(extension);
        } else {
            let mapped = four_cc_format(pixel_format.four_cc).ok_or_else(|| {
                ContentError::unsupported(
                    path,
                    format!(
                        "DDS FourCC `{}` is not implemented",
                        display_four_cc(pixel_format.four_cc)
                    ),
                )
            })?;
            format = mapped.0;
            srgb = mapped.1;
        }
    } else {
        format = masked_format(&pixel_format, path)?;
    }
    if format == TextureFormat::Undefined {
        return Err(ContentError::unsupported(path, "DDS header has no pixel format"));
    }

    Ok(DdsHeader {
        flags,
        height: read_u32_le(bytes, 12, path)?,
        width: read_u32_le(bytes, 16, path)?,
        pitch_or_linear_size: read_u32_le(bytes, 20, path)?,
        depth: read_u32_le(bytes, 24, path)?,
        mipmap_count: read_u32_le(bytes, 28, path)?,
        pixel_format,
        caps: read_u32_le(bytes, 108, path)?,
        caps2: read_u32_le(bytes, 112, path)?,
        caps3: read_u32_le(bytes, 116, path)?,
        caps4: read_u32_le(bytes, 120, path)?,
        dx10,
        format,
        srgb,
    })
}

/// Parse the header and build the level table, checking the payload is present.
pub fn parse(bytes: &[u8], path: &Path) -> Result<TextureInfo, ContentError> {
    let header = parse_header(bytes, path)?;
    if header.width == 0 || header.height == 0 {
        return Err(ContentError::format(path, "DDS dimensions must be non-zero"));
    }
    let data_offset = if header.dx10.is_some() {
        DX10_HEADER_BYTES
    } else {
        HEADER_BYTES
    };

    let array_size = header
        .dx10
        .map(|extension| extension.array_size.max(1))
        .unwrap_or(1);
    let is_cubemap = header.caps2 & DDSC2_CUBEMAP != 0
        || header
            .dx10
            .is_some_and(|extension| extension.misc_flags & 0x4 != 0);
    let is_volume = header.caps2 & DDSC2_VOLUME != 0
        || header
            .dx10
            .is_some_and(|extension| extension.dimension == 4);

    // Godot's loader reads one image's whole mip chain at a time, so cube
    // faces and array slices are laid out layer-major.
    let (kind, images, layers) = if is_volume {
        (TextureKind::D3, 1u32, 1u32)
    } else if is_cubemap && array_size > 1 {
        (TextureKind::CubeArray, 6 * array_size, 6 * array_size)
    } else if is_cubemap {
        (TextureKind::Cube, 6, 6)
    } else if array_size > 1 {
        (TextureKind::D2Array, array_size, array_size)
    } else {
        (TextureKind::D2, 1, 1)
    };
    let depth = if is_volume { header.depth.max(1) } else { 1 };

    let mip_levels = header.levels_declared();
    let max_levels = max_mip_levels(header.width, header.height);
    if mip_levels > max_levels {
        return Err(ContentError::format(
            path,
            format!(
                "DDS declares {mip_levels} mip levels but {}x{} allows {max_levels}",
                header.width, header.height
            ),
        ));
    }

    let mut levels = Vec::with_capacity((images * mip_levels) as usize);
    let mut offset = data_offset as u64;
    for _image in 0..images {
        for level in 0..mip_levels {
            let width = (header.width >> level).max(1);
            let height = (header.height >> level).max(1);
            let level_depth = if is_volume { (depth >> level).max(1) } else { 1 };
            let size = header.format.surface_bytes(width, height) as u64 * u64::from(level_depth);
            levels.push(TextureLevel {
                width,
                height,
                depth: level_depth,
                layers: 1,
                offset,
                size,
                uncompressed_size: size,
            });
            offset += size;
        }
    }

    if (bytes.len() as u64) < offset {
        return Err(ContentError::format(
            path,
            format!("DDS payload needs {offset} bytes, got {}", bytes.len()),
        ));
    }

    Ok(TextureInfo {
        format: header.format,
        kind,
        width: header.width,
        height: header.height,
        depth,
        layers,
        mip_levels,
        levels,
        srgb: header.srgb,
        supercompression: 0,
        data_offset: data_offset as u64,
    })
}

/// Map a DXGI format to a texture format and whether it is sRGB.
pub(crate) fn dxgi_format(value: u32) -> Option<(TextureFormat, bool)> {
    use TextureFormat as F;
    let format = match value {
        0 => (F::Undefined, false),
        2 => (F::Rgba32F, false),
        6 => (F::Rgb32F, false),
        10 => (F::Rgba16F, false),
        11 => (F::Rgba16, false),
        16 => (F::Rg32F, false),
        24 => (F::Bgr10a2, false),
        28 => (F::Rgba8, false),
        29 => (F::Rgba8, true),
        34 => (F::Rg16F, false),
        35 => (F::Rg16, false),
        41 => (F::R32F, false),
        49 => (F::Rg8, false),
        54 => (F::R16F, false),
        56 => (F::R16, false),
        61 => (F::R8, false),
        65 => (F::R8, false),
        67 => (F::Rgb9e5, false),
        71 => (F::Bc1, false),
        72 => (F::Bc1, true),
        74 => (F::Bc2, false),
        75 => (F::Bc2, true),
        77 => (F::Bc3, false),
        78 => (F::Bc3, true),
        80 => (F::Bc4, false),
        83 => (F::Bc5, false),
        85 => (F::Bgr565, false),
        87 => (F::Bgra8, false),
        95 => (F::Bc6hUf16, false),
        96 => (F::Bc6hSf16, false),
        98 => (F::Bc7, false),
        99 => (F::Bc7, true),
        115 => (F::Bgra4444, false),
        _ => return None,
    };
    Some(format)
}

fn four_cc(value: &[u8; 4]) -> u32 {
    u32::from_le_bytes(*value)
}

/// Map a legacy FourCC to a texture format and whether it is sRGB.
pub(crate) fn four_cc_format(value: u32) -> Option<(TextureFormat, bool)> {
    use TextureFormat as F;
    let format = match value {
        v if v == four_cc(b"DXT1") => (F::Bc1, false),
        v if v == four_cc(b"DXT2") || v == four_cc(b"DXT3") => (F::Bc2, false),
        v if v == four_cc(b"DXT4") || v == four_cc(b"DXT5") => (F::Bc3, false),
        v if v == four_cc(b"ATI1") || v == four_cc(b"BC4U") => (F::Bc4, false),
        v if v == four_cc(b"ATI2") || v == four_cc(b"BC5U") || v == four_cc(b"A2XY") => {
            (F::Bc5, false)
        }
        36 => (F::Rgba16, false),
        111 => (F::R16F, false),
        112 => (F::Rg16F, false),
        113 => (F::Rgba16F, false),
        114 => (F::R32F, false),
        115 => (F::Rg32F, false),
        116 => (F::Rgba32F, false),
        _ => return None,
    };
    Some(format)
}

fn display_four_cc(value: u32) -> String {
    value
        .to_le_bytes()
        .iter()
        .map(|byte| {
            if byte.is_ascii_graphic() {
                char::from(*byte)
            } else {
                '?'
            }
        })
        .collect()
}

/// Map the bitmasked legacy pixel formats Godot also understands.
pub(crate) fn masked_format(
    pixel_format: &DdsPixelFormat,
    path: &Path,
) -> Result<TextureFormat, ContentError> {
    use TextureFormat as F;
    let bits = pixel_format.rgb_bit_count;
    let (red, green, blue, alpha) = (
        pixel_format.red_mask,
        pixel_format.green_mask,
        pixel_format.blue_mask,
        pixel_format.alpha_mask,
    );
    let with_alpha = pixel_format.flags & DDPF_ALPHAPIXELS != 0;
    let format = if with_alpha {
        match (bits, red, green, blue, alpha) {
            (32, 0x0000_00ff, 0x0000_ff00, 0x00ff_0000, 0xff00_0000) => F::Rgba8,
            (32, 0x00ff_0000, 0x0000_ff00, 0x0000_00ff, 0xff00_0000) => F::Bgra8,
            (32, 0x0000_03ff, 0x000f_fc00, 0x3ff0_0000, 0xc000_0000) => F::Rgb10a2,
            (32, 0x3ff0_0000, 0x000f_fc00, 0x0000_03ff, 0xc000_0000) => F::Bgr10a2,
            (16, 0x0000_000f, 0x0000_00f0, 0x0000_0f00, 0x0000_f000) => F::Rgba4444,
            (16, 0x0000_0f00, 0x0000_00f0, 0x0000_000f, 0x0000_f000) => F::Bgra4444,
            (16, 0x0000_00ff, 0x0000_0000, 0x0000_0000, 0x0000_ff00) => F::La8,
            _ => {
                return Err(ContentError::unsupported(
                    path,
                    format!("DDS bitmasked pixel format (bits={bits}, r={red:#x}, g={green:#x}, b={blue:#x}, a={alpha:#x}) is not implemented"),
                ));
            }
        }
    } else if pixel_format.flags & DDPF_ALPHAONLY != 0 {
        if bits == 8 && alpha == 0xff {
            F::R8
        } else {
            return Err(ContentError::unsupported(
                path,
                "DDS alpha-only pixel format is not implemented",
            ));
        }
    } else {
        match (bits, red, green, blue) {
            (24, 0x0000_00ff, 0x0000_ff00, 0x00ff_0000) => F::Rgb8,
            (24, 0x00ff_0000, 0x0000_ff00, 0x0000_00ff) => F::Bgr8,
            (32, 0x0000_00ff, 0x0000_ff00, 0x00ff_0000) => F::Rgbx8,
            (32, 0x00ff_0000, 0x0000_ff00, 0x0000_00ff) => F::Bgrx8,
            (16, 0x0000_f800, 0x0000_07e0, 0x0000_001f) => F::Rgb565,
            (16, 0x0000_001f, 0x0000_07e0, 0x0000_f800) => F::Bgr565,
            (16, 0x0000_ffff, 0, 0) => F::R16,
            (16, 0x0000_00ff, 0x0000_ff00, 0) => F::Rg16,
            (8, 0x0000_00ff, 0, 0) => F::R8,
            _ => {
                return Err(ContentError::unsupported(
                    path,
                    format!("DDS bitmasked pixel format (bits={bits}, r={red:#x}, g={green:#x}, b={blue:#x}) is not implemented"),
                ));
            }
        }
    };
    Ok(format)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{DDS_MAGIC, TextureFormat, TextureKind, TextureLevel};

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    /// An uncompressed 8x8 RGBA8 DDS: the legacy 128-byte header plus 256 bytes
    /// of payload, one mip level.
    fn rgba8_dds() -> Vec<u8> {
        let mut bytes = Vec::new();
        push_u32(&mut bytes, DDS_MAGIC);
        push_u32(&mut bytes, 124); // header size
        push_u32(&mut bytes, 0x0000_1007); // CAPS|HEIGHT|WIDTH|PIXELFORMAT
        push_u32(&mut bytes, 8); // height
        push_u32(&mut bytes, 8); // width
        push_u32(&mut bytes, 32); // pitch
        push_u32(&mut bytes, 0); // depth
        push_u32(&mut bytes, 1); // mipmap count
        for _ in 0..11 {
            push_u32(&mut bytes, 0); // reserved
        }
        push_u32(&mut bytes, 32); // pixel format size
        push_u32(&mut bytes, 0x41); // RGB | ALPHAPIXELS
        push_u32(&mut bytes, 0); // four cc
        push_u32(&mut bytes, 32); // bit count
        push_u32(&mut bytes, 0x0000_00ff); // red mask
        push_u32(&mut bytes, 0x0000_ff00); // green mask
        push_u32(&mut bytes, 0x00ff_0000); // blue mask
        push_u32(&mut bytes, 0xff00_0000); // alpha mask
        push_u32(&mut bytes, 0x1000); // caps
        push_u32(&mut bytes, 0); // caps2
        push_u32(&mut bytes, 0); // caps3
        push_u32(&mut bytes, 0); // caps4
        assert_eq!(bytes.len(), 128);
        bytes.resize(128 + 8 * 8 * 4, 0x7f);
        bytes
    }

    #[test]
    fn rgba8_header_parses_to_exact_fields() {
        let path = Path::new("t.dds");
        let bytes = rgba8_dds();
        let header = parse_header(&bytes, path).expect("header");
        assert_eq!(header.flags, 0x0000_1007);
        assert_eq!(header.width, 8);
        assert_eq!(header.height, 8);
        assert_eq!(header.pitch_or_linear_size, 32);
        assert_eq!(header.depth, 0);
        assert_eq!(header.mipmap_count, 1);
        assert_eq!(header.levels_declared(), 1);
        assert_eq!(header.caps, 0x1000);
        assert_eq!(header.caps2, 0);
        assert_eq!(header.dx10, None);
        assert_eq!(header.format, TextureFormat::Rgba8);
        assert!(!header.srgb);
        assert_eq!(
            header.pixel_format,
            DdsPixelFormat {
                size: 32,
                flags: 0x41,
                four_cc: 0,
                rgb_bit_count: 32,
                red_mask: 0x0000_00ff,
                green_mask: 0x0000_ff00,
                blue_mask: 0x00ff_0000,
                alpha_mask: 0xff00_0000,
            }
        );

        let info = parse(&bytes, path).expect("container");
        assert_eq!(info.format, TextureFormat::Rgba8);
        assert_eq!(info.kind, TextureKind::D2);
        assert_eq!(info.width, 8);
        assert_eq!(info.height, 8);
        assert_eq!(info.depth, 1);
        assert_eq!(info.layers, 1);
        assert_eq!(info.mip_levels, 1);
        assert_eq!(info.data_offset, 128);
        assert_eq!(
            info.levels,
            vec![TextureLevel {
                width: 8,
                height: 8,
                depth: 1,
                layers: 1,
                offset: 128,
                size: 256,
                uncompressed_size: 256,
            }]
        );

        let decoded = crate::codec::BlockCodecRegistry::default()
            .decode(&bytes, path)
            .expect("decode");
        assert_eq!(decoded.format, TextureFormat::Rgba8);
        assert_eq!(decoded.data, vec![0x7f; 256]);
    }
}
