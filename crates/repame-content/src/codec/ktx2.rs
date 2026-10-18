//! KTX 2 container header parsing.
//!
//! Parses the 80-byte KTX 2 header plus its level index (one 24-byte entry per
//! mip level, always present) into a [`Ktx2Header`] and a [`TextureInfo`],
//! following the Khronos layout also used by Godot's `modules/ktx`. KTX 1 was
//! superseded; it is refused by [`super::parse_container`] with a typed error.

use std::path::Path;

use crate::codec::{
    TextureFormat, TextureInfo, TextureKind, TextureLevel, read_u32_le, read_u64_le,
};
use crate::error::ContentError;

/// KTX 2 header size, identifier included.
pub const HEADER_BYTES: usize = 80;
/// KTX 2 level index entry size.
pub const LEVEL_INDEX_ENTRY_BYTES: usize = 24;

/// KTX 2 supercompression schemes (`supercompressionScheme` field).
pub const SUPERCOMPRESSION_NONE: u32 = 0;

/// A parsed KTX 2 header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ktx2Header {
    pub vk_format: u32,
    pub type_size: u32,
    pub pixel_width: u32,
    pub pixel_height: u32,
    pub pixel_depth: u32,
    pub layer_count: u32,
    pub face_count: u32,
    pub level_count: u32,
    pub supercompression_scheme: u32,
    pub dfd_byte_offset: u32,
    pub dfd_byte_length: u32,
    pub kvd_byte_offset: u32,
    pub kvd_byte_length: u32,
    pub sgd_byte_offset: u64,
    pub sgd_byte_length: u64,
    /// Offset of the level index, i.e. the end of the header.
    pub level_index_offset: u64,
    /// Format resolved from `vk_format`.
    pub format: TextureFormat,
    pub srgb: bool,
}

impl Ktx2Header {
    /// `levelCount == 0` requests mipmap generation; the file holds one level.
    pub fn generate_mipmaps(&self) -> bool {
        self.level_count == 0
    }

    /// Mip levels physically present in the file.
    pub fn mip_levels(&self) -> u32 {
        self.level_count.max(1)
    }
}

/// Parse the KTX 2 header without reading the level index or payload.
pub fn parse_header(bytes: &[u8], path: &Path) -> Result<Ktx2Header, ContentError> {
    if !bytes.starts_with(&crate::codec::KTX2_IDENTIFIER) {
        return Err(ContentError::format(path, "not a KTX 2 container"));
    }
    if bytes.len() < HEADER_BYTES {
        return Err(ContentError::format(
            path,
            format!("KTX 2 header needs {HEADER_BYTES} bytes, got {}", bytes.len()),
        ));
    }
    let vk_format = read_u32_le(bytes, 12, path)?;
    let (format, srgb) = vk_format_to_texture(vk_format).ok_or_else(|| {
        ContentError::unsupported(path, format!("KTX 2 vkFormat {vk_format} is not implemented"))
    })?;
    Ok(Ktx2Header {
        vk_format,
        type_size: read_u32_le(bytes, 16, path)?,
        pixel_width: read_u32_le(bytes, 20, path)?,
        pixel_height: read_u32_le(bytes, 24, path)?,
        pixel_depth: read_u32_le(bytes, 28, path)?,
        layer_count: read_u32_le(bytes, 32, path)?,
        face_count: read_u32_le(bytes, 36, path)?,
        level_count: read_u32_le(bytes, 40, path)?,
        supercompression_scheme: read_u32_le(bytes, 44, path)?,
        dfd_byte_offset: read_u32_le(bytes, 48, path)?,
        dfd_byte_length: read_u32_le(bytes, 52, path)?,
        kvd_byte_offset: read_u32_le(bytes, 56, path)?,
        kvd_byte_length: read_u32_le(bytes, 60, path)?,
        sgd_byte_offset: read_u64_le(bytes, 64, path)?,
        sgd_byte_length: read_u64_le(bytes, 72, path)?,
        level_index_offset: HEADER_BYTES as u64,
        format,
        srgb,
    })
}

/// Build the level table from the header and its level index.
///
/// KTX 2 stores each level's byte offset, byte length and uncompressed length
/// explicitly, so this is a copy plus range checks against `container_len`
/// (the whole container, not just the bytes read so far). `index` must hold at
/// least `24 * mip levels` bytes. A level that runs past the container, or
/// whose range reaches back into the level index or overlaps another level, is
/// a typed [`ContentErrorKind::Format`](crate::ContentErrorKind) error, never a
/// panic or a silently short read.
///
/// The index is read in level order but no byte order is assumed: per the
/// specification the Mip Level Array stores the smallest level first and the
/// base level last, so a conforming file's offsets are descending. The ranges
/// only have to be disjoint and after the index.
pub fn parse_levels(
    header: &Ktx2Header,
    index: &[u8],
    container_len: u64,
    path: &Path,
) -> Result<Vec<TextureLevel>, ContentError> {
    if header.pixel_width == 0 || header.pixel_height == 0 {
        return Err(ContentError::format(path, "KTX 2 dimensions must be non-zero"));
    }
    if header.face_count != 1 && header.face_count != 6 {
        return Err(ContentError::format(
            path,
            format!("KTX 2 faceCount {} is invalid", header.face_count),
        ));
    }
    let mip_levels = header.mip_levels();
    let index_len = mip_levels as usize * LEVEL_INDEX_ENTRY_BYTES;
    if index.len() < index_len {
        return Err(ContentError::format(path, "KTX 2 level index is truncated"));
    }
    // Level data begins after the index. No level's range may reach into the
    // index or overlap another level's range: either would have this table,
    // and every decode downstream, read header or neighbour bytes as texels.
    // The ranges are not required to be in any order (the specification stores
    // the smallest level first, so offsets commonly descend).
    let data_start = header.level_index_offset + index_len as u64;
    let index_read = |base: usize| -> Result<u64, ContentError> {
        index
            .get(base..base + 8)
            .and_then(|slice| <[u8; 8]>::try_from(slice).ok())
            .map(u64::from_le_bytes)
            .ok_or_else(|| ContentError::format(path, "KTX 2 level index is truncated"))
    };
    let depth = header.pixel_depth.max(1);
    let layers = header.layer_count.max(1) * header.face_count;
    let mut levels = Vec::with_capacity(mip_levels as usize);
    let mut spans = Vec::with_capacity(mip_levels as usize);
    for level in 0..mip_levels {
        let base = level as usize * LEVEL_INDEX_ENTRY_BYTES;
        let offset = index_read(base)?;
        let size = index_read(base + 8)?;
        let uncompressed_size = index_read(base + 16)?;
        let end = offset
            .checked_add(size)
            .ok_or_else(|| ContentError::format(path, "KTX 2 level range overflows"))?;
        if end > container_len {
            return Err(ContentError::format(
                path,
                format!("KTX 2 level {level} extends past the end of the container"),
            ));
        }
        if offset < data_start {
            return Err(ContentError::format(
                path,
                format!(
                    "KTX 2 level {level} starts at byte {offset}, inside the {} byte level index that ends at {data_start}",
                    index_len
                ),
            ));
        }
        spans.push((offset, end, level));
        levels.push(TextureLevel {
            width: (header.pixel_width >> level).max(1),
            height: (header.pixel_height >> level).max(1),
            depth: (depth >> level).max(1),
            layers,
            offset,
            size,
            uncompressed_size,
        });
    }
    spans.sort_unstable();
    for pair in spans.windows(2) {
        let (first_offset, first_end, _) = pair[0];
        let (offset, end, level) = pair[1];
        if offset < first_end {
            return Err(ContentError::format(
                path,
                format!(
                    "KTX 2 level {level} (bytes {offset}..{end}) overlaps the level at bytes {first_offset}..{first_end}"
                ),
            ));
        }
    }
    Ok(levels)
}

/// Assemble a [`TextureInfo`] from a parsed header and its level table.
pub(crate) fn info(header: &Ktx2Header, levels: Vec<TextureLevel>) -> TextureInfo {
    let depth = header.pixel_depth.max(1);
    let layers = header.layer_count.max(1) * header.face_count;
    let kind = if header.face_count == 6 {
        if header.layer_count > 1 {
            TextureKind::CubeArray
        } else {
            TextureKind::Cube
        }
    } else if header.pixel_depth > 0 {
        TextureKind::D3
    } else if header.layer_count > 1 {
        TextureKind::D2Array
    } else {
        TextureKind::D2
    };
    TextureInfo {
        format: header.format,
        kind,
        width: header.pixel_width,
        height: header.pixel_height,
        depth,
        layers,
        mip_levels: header.mip_levels(),
        data_offset: levels.first().map(|level| level.offset).unwrap_or(0),
        levels,
        srgb: header.srgb,
        supercompression: header.supercompression_scheme,
    }
}

/// Parse the header and its level index into a [`TextureInfo`].
pub fn parse(bytes: &[u8], path: &Path) -> Result<TextureInfo, ContentError> {
    let header = parse_header(bytes, path)?;
    let index_end = HEADER_BYTES + header.mip_levels() as usize * LEVEL_INDEX_ENTRY_BYTES;
    if bytes.len() < index_end {
        return Err(ContentError::format(path, "KTX 2 level index is truncated"));
    }
    let levels = parse_levels(
        &header,
        &bytes[HEADER_BYTES..index_end],
        bytes.len() as u64,
        path,
    )?;
    Ok(info(&header, levels))
}

/// Map a Vulkan format to a texture format and whether it is sRGB.
pub(crate) fn vk_format_to_texture(value: u32) -> Option<(TextureFormat, bool)> {
    use TextureFormat as F;
    let format = match value {
        0 => (F::Undefined, false),
        2 => (F::Bgra4444, false),
        4 => (F::Rgb565, false),
        9 => (F::R8, false),
        13 => (F::R8, false),
        16 => (F::Rg8, false),
        23 => (F::Rgb8, false),
        29 => (F::Rgb8, true),
        37 => (F::Rgba8, false),
        43 => (F::Rgba8, true),
        44 => (F::Bgra8, false),
        70 => (F::R16, false),
        74 => (F::R16, false),
        91 => (F::Rgba16, false),
        97 => (F::Rgba16F, false),
        100 => (F::R32F, false),
        123 => (F::Rgb9e5, false),
        131 => (F::Bc1, false),
        132 => (F::Bc1, true),
        133 => (F::Bc1, false),
        134 => (F::Bc1, true),
        135 => (F::Bc2, false),
        136 => (F::Bc2, true),
        137 => (F::Bc3, false),
        138 => (F::Bc3, true),
        139 => (F::Bc4, false),
        141 => (F::Bc5, false),
        143 => (F::Bc6hUf16, false),
        144 => (F::Bc6hSf16, false),
        145 => (F::Bc7, false),
        146 => (F::Bc7, true),
        147 => (F::Etc2Rgb8, false),
        148 => (F::Etc2Rgb8, true),
        151 => (F::Etc2Rgba8, false),
        152 => (F::Etc2Rgba8, true),
        153 => (F::EacR11, false),
        155 => (F::EacRg11, false),
        157 => (F::Astc4x4, false),
        158 => (F::Astc4x4, true),
        165 => (F::Astc6x6, false),
        166 => (F::Astc6x6, true),
        171 => (F::Astc8x8, false),
        172 => (F::Astc8x8, true),
        _ => return None,
    };
    Some(format)
}

/// Supercompression scheme name, for diagnostics.
pub fn supercompression_name(scheme: u32) -> &'static str {
    match scheme {
        SUPERCOMPRESSION_NONE => "none",
        1 => "basis_lz",
        2 => "zstd",
        3 => "zlib",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{KTX2_IDENTIFIER, TextureFormat};

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    /// A one-level 16x8 BC1 KTX 2: 80-byte header, 24-byte level index, a
    /// 28-byte DFD placeholder this parser leaves opaque, then the 64-byte
    /// payload of eight copies of the BC1 golden block.
    fn bc1_ktx2() -> Vec<u8> {
        let block = [0xFFu8, 0xFF, 0x00, 0x00, 0x1B, 0x1B, 0x1B, 0x1B];
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&KTX2_IDENTIFIER);
        push_u32(&mut bytes, 131); // VK_FORMAT_BC1_RGB_UNORM_BLOCK
        push_u32(&mut bytes, 1); // typeSize
        push_u32(&mut bytes, 16); // pixelWidth
        push_u32(&mut bytes, 8); // pixelHeight
        push_u32(&mut bytes, 0); // pixelDepth
        push_u32(&mut bytes, 0); // layerCount
        push_u32(&mut bytes, 1); // faceCount
        push_u32(&mut bytes, 1); // levelCount
        push_u32(&mut bytes, 0); // supercompressionScheme
        push_u32(&mut bytes, 104); // dfdByteOffset
        push_u32(&mut bytes, 28); // dfdByteLength
        push_u32(&mut bytes, 0); // kvdByteOffset
        push_u32(&mut bytes, 0); // kvdByteLength
        push_u64(&mut bytes, 0); // sgdByteOffset
        push_u64(&mut bytes, 0); // sgdByteLength
        assert_eq!(bytes.len(), HEADER_BYTES);
        push_u64(&mut bytes, 132); // level index: byteOffset
        push_u64(&mut bytes, 64); // byteLength
        push_u64(&mut bytes, 64); // uncompressedByteLength
        assert_eq!(bytes.len(), 104);
        bytes.resize(132, 0); // DFD placeholder
        for _ in 0..8 {
            bytes.extend_from_slice(&block);
        }
        assert_eq!(bytes.len(), 196);
        bytes
    }

    #[test]
    fn bc1_header_parses_to_exact_fields() {
        let path = Path::new("t.ktx2");
        let bytes = bc1_ktx2();
        let header = parse_header(&bytes, path).expect("header");
        assert_eq!(header.vk_format, 131);
        assert_eq!(header.type_size, 1);
        assert_eq!(header.pixel_width, 16);
        assert_eq!(header.pixel_height, 8);
        assert_eq!(header.pixel_depth, 0);
        assert_eq!(header.layer_count, 0);
        assert_eq!(header.face_count, 1);
        assert_eq!(header.level_count, 1);
        assert_eq!(header.supercompression_scheme, SUPERCOMPRESSION_NONE);
        assert_eq!(header.dfd_byte_offset, 104);
        assert_eq!(header.dfd_byte_length, 28);
        assert_eq!(header.sgd_byte_offset, 0);
        assert_eq!(header.level_index_offset, 80);
        assert_eq!(header.format, TextureFormat::Bc1);
        assert!(!header.srgb);
        assert!(!header.generate_mipmaps());

        let info = parse(&bytes, path).expect("container");
        assert_eq!(info.format, TextureFormat::Bc1);
        assert_eq!(info.kind, TextureKind::D2);
        assert_eq!(
            (info.width, info.height, info.depth, info.layers, info.mip_levels),
            (16, 8, 1, 1, 1)
        );
        assert_eq!(info.data_offset, 132);
        assert_eq!(
            info.levels,
            vec![TextureLevel {
                width: 16,
                height: 8,
                depth: 1,
                layers: 1,
                offset: 132,
                size: 64,
                uncompressed_size: 64,
            }]
        );

        let decoded = crate::codec::BlockCodecRegistry::default()
            .decode(&bytes, path)
            .expect("decode");
        assert_eq!(decoded.format, TextureFormat::Rgba8);
        assert_eq!((decoded.width, decoded.height), (16, 8));
        assert_eq!(decoded.data.len(), 16 * 8 * 4);
        let row = [85u8, 85, 85, 255, 170, 170, 170, 255, 0, 0, 0, 255, 255, 255, 255, 255];
        assert_eq!(&decoded.data[0..16], &row);
        assert_eq!(&decoded.data[16..32], &row);
    }

    #[test]
    fn bc7_is_refused_with_a_typed_error() {
        let mut bytes = bc1_ktx2();
        bytes[12..16].copy_from_slice(&145u32.to_le_bytes()); // VK_FORMAT_BC7_UNORM_BLOCK
        let path = Path::new("t.ktx2");
        let info = parse(&bytes, path).expect("container");
        assert_eq!(info.format, TextureFormat::Bc7);
        let error = crate::codec::BlockCodecRegistry::default()
            .decode(&bytes, path)
            .expect_err("bc7 not implemented");
        assert!(error.is_unsupported());
        assert_eq!(error.kind(), crate::error::ContentErrorKind::Unsupported);
    }

    /// A two-level 8x8 BC1 KTX 2 with both levels stored base-first (ascending
    /// offsets, a valid non-overlapping layout): 80-byte header, two 24-byte
    /// index entries, then level 0 (32 bytes) and level 1 (8 bytes).
    fn bc1_ktx2_mipped() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&KTX2_IDENTIFIER);
        push_u32(&mut bytes, 131); // VK_FORMAT_BC1_RGB_UNORM_BLOCK
        push_u32(&mut bytes, 1); // typeSize
        push_u32(&mut bytes, 8); // pixelWidth
        push_u32(&mut bytes, 8); // pixelHeight
        push_u32(&mut bytes, 0); // pixelDepth
        push_u32(&mut bytes, 0); // layerCount
        push_u32(&mut bytes, 1); // faceCount
        push_u32(&mut bytes, 2); // levelCount
        push_u32(&mut bytes, 0); // supercompressionScheme
        push_u32(&mut bytes, 0); // dfdByteOffset
        push_u32(&mut bytes, 0); // dfdByteLength
        push_u32(&mut bytes, 0); // kvdByteOffset
        push_u32(&mut bytes, 0); // kvdByteLength
        push_u64(&mut bytes, 0); // sgdByteOffset
        push_u64(&mut bytes, 0); // sgdByteLength
        assert_eq!(bytes.len(), HEADER_BYTES);
        push_u64(&mut bytes, 128); // level 0 byteOffset
        push_u64(&mut bytes, 32); // level 0 byteLength
        push_u64(&mut bytes, 32); // level 0 uncompressedByteLength
        push_u64(&mut bytes, 160); // level 1 byteOffset
        push_u64(&mut bytes, 8); // level 1 byteLength
        push_u64(&mut bytes, 8); // level 1 uncompressedByteLength
        assert_eq!(bytes.len(), HEADER_BYTES + 2 * LEVEL_INDEX_ENTRY_BYTES);
        bytes.resize(128 + 32 + 8, 0x11);
        bytes
    }

    /// The same chain in the specification's Mip Level Array order: the
    /// smallest level first and the base level last, so the index entries
    /// descend byte-wise (level 0 at 136, level 1 at 128), the layout every
    /// mipped KTX 2 writer emits.
    fn bc1_ktx2_spec_order() -> Vec<u8> {
        let mut bytes = bc1_ktx2_mipped();
        set_level_offset(&mut bytes, 0, 136);
        set_level_offset(&mut bytes, 1, 128);
        bytes
    }

    fn set_level_offset(bytes: &mut [u8], level: usize, offset: u64) {
        let base = HEADER_BYTES + level * LEVEL_INDEX_ENTRY_BYTES;
        bytes[base..base + 8].copy_from_slice(&offset.to_le_bytes());
    }

    #[test]
    fn a_malformed_level_index_never_reads_header_bytes_as_texels() {
        let path = Path::new("t.ktx2");
        // The base-first layout parses: level 0 starts exactly at the end of
        // the index, so the check must not refuse the boundary.
        let info = parse(&bc1_ktx2_mipped(), path).expect("base-first layout");
        assert_eq!(info.mip_levels, 2);
        assert_eq!(info.levels[0].offset, 128);
        assert_eq!(info.levels[1].offset, 160);
        assert_eq!(info.data_offset, 128);

        // A level whose range reaches into the 48-byte index (80..128).
        for offset in [0u64, 96] {
            let mut bytes = bc1_ktx2_mipped();
            set_level_offset(&mut bytes, 0, offset);
            let error = parse(&bytes, path).expect_err("offset inside the index");
            assert_eq!(error.kind(), crate::error::ContentErrorKind::Format);
            assert!(error.message().contains("level index"), "{}", error.message());
        }

        // The reviewed case: level 1 at offset 0 with length 8 fits inside the
        // container, so before this check it decoded header bytes as BC1.
        let mut bytes = bc1_ktx2_mipped();
        set_level_offset(&mut bytes, 1, 0);
        let error = parse(&bytes, path).expect_err("level 1 inside the index");
        assert_eq!(error.kind(), crate::error::ContentErrorKind::Format);
        assert!(error.message().contains("level index"), "{}", error.message());
        // The streamed path shares the table, so it refuses the same index.
        let streamed = crate::codec::stream::MipReader::open(bytes.as_slice(), path)
            .err()
            .expect("streamed open");
        assert_eq!(streamed.kind(), crate::error::ContentErrorKind::Format);
    }

    #[test]
    fn the_smallest_level_first_layout_parses_and_levels_read_where_the_index_says() {
        let path = Path::new("t.ktx2");
        // The specification's Mip Level Array order: level 1 (8 bytes) first,
        // the 8x8 base level (32 bytes) last, so `levels[0].byteOffset` is the
        // larger one. A reader that insisted on ascending offsets would refuse
        // every conforming mipped file, which this test pins down.
        let bytes = bc1_ktx2_spec_order();
        let info = parse(&bytes, path).expect("smallest-first layout");
        assert_eq!(info.levels[0].offset, 136);
        assert_eq!(info.levels[0].size, 32);
        assert_eq!(info.levels[1].offset, 128);
        assert_eq!(info.levels[1].size, 8);

        // Each level decodes its own range, in the level it was asked for.
        let reader = crate::codec::stream::MipReader::open(bytes.as_slice(), path).expect("open");
        assert_eq!(reader.level_range(1).expect("range"), 128..136);
        let level = reader
            .read_level(&crate::codec::BlockCodecRegistry::default(), 1)
            .expect("level 1 decodes");
        assert_eq!((level.width, level.height), (4, 4));
        assert_eq!(level.data.len(), 4 * 4 * 4);
    }

    #[test]
    fn overlapping_level_ranges_are_refused() {
        let path = Path::new("t.ktx2");
        // Level 1 fully inside level 0's 128..160 range.
        let mut bytes = bc1_ktx2_mipped();
        set_level_offset(&mut bytes, 1, 140);
        let error = parse(&bytes, path).expect_err("levels overlap");
        assert_eq!(error.kind(), crate::error::ContentErrorKind::Format);
        assert!(error.message().contains("overlaps"), "{}", error.message());

        // Two levels starting at the same byte.
        let mut bytes = bc1_ktx2_mipped();
        set_level_offset(&mut bytes, 1, 128);
        let error = parse(&bytes, path).expect_err("levels share a start");
        assert_eq!(error.kind(), crate::error::ContentErrorKind::Format);
        assert!(error.message().contains("overlaps"), "{}", error.message());

        // Level 1 starting one byte before level 0 ends.
        let mut bytes = bc1_ktx2_mipped();
        set_level_offset(&mut bytes, 1, 159);
        let error = parse(&bytes, path).expect_err("levels overlap by one byte");
        assert_eq!(error.kind(), crate::error::ContentErrorKind::Format);
        assert!(error.message().contains("overlaps"), "{}", error.message());

        // The streamed path shares the table, so it refuses the same index.
        let streamed = crate::codec::stream::MipReader::open(bytes.as_slice(), path)
            .err()
            .expect("streamed open");
        assert_eq!(streamed.kind(), crate::error::ContentErrorKind::Format);
    }
}
