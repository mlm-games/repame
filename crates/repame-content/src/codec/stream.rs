//! Per-mip texture I/O: open a container header, read one level.
//!
//! The whole-container path ([`BlockCodecRegistry::decode`]) reads a file into
//! memory and decodes mip 0. A streamed texture wants the opposite: open the
//! header (and, for KTX 2, the level index), learn every level's stored byte
//! range, and read exactly the level that was asked for — nothing else.
//!
//! [`MipReader`] is that reader, over any [`ByteRangeSource`]: a plain file
//! ([`FileRangeSource`]), an in-memory slice, or a package entry
//! ([`PackageRangeSource`](crate::package::PackageRangeSource)). The same code
//! serves loose assets and packed archives. Only the requested level is read
//! and only that level is decoded; a recognised-but-unimplemented format stays
//! typed [`Unsupported`](crate::ContentErrorKind::Unsupported), and a level
//! table whose levels overrun the payload fails with a typed format error
//! instead of a short read.
//!
//! [`BlockCodecRegistry::decode`]: super::BlockCodecRegistry::decode

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::codec::{
    BlockCodecRegistry, DecodedImage, DDS_MAGIC, KTX1_IDENTIFIER, KTX2_IDENTIFIER, TextureInfo,
    dds, ktx2, read_u32_le,
};
use crate::error::{ContentError, io_error};

/// Longest container header this crate parses: the DDS header plus its
/// optional `"DX10 "` extension.
const HEADER_PROBE_BYTES: u64 = 148;

/// A random-access byte source: a file, a slice, or a package entry.
///
/// Reads are exact: a range that runs past the end of the source is an error,
/// never a short read that a caller might mistake for valid bytes.
pub trait ByteRangeSource {
    /// Total byte length of the container.
    fn byte_len(&self) -> Result<u64, ContentError>;

    /// Read exactly `length` bytes at `offset`.
    fn read_range(&self, offset: u64, length: usize) -> Result<Vec<u8>, ContentError>;
}

impl<T> ByteRangeSource for &T
where
    T: ByteRangeSource + ?Sized,
{
    fn byte_len(&self) -> Result<u64, ContentError> {
        (**self).byte_len()
    }

    fn read_range(&self, offset: u64, length: usize) -> Result<Vec<u8>, ContentError> {
        (**self).read_range(offset, length)
    }
}

impl ByteRangeSource for [u8] {
    fn byte_len(&self) -> Result<u64, ContentError> {
        Ok(self.len() as u64)
    }

    fn read_range(&self, offset: u64, length: usize) -> Result<Vec<u8>, ContentError> {
        let start = usize::try_from(offset)
            .map_err(|_| ContentError::format("<bytes>", "texture byte offset overflows"))?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| ContentError::format("<bytes>", "texture byte range overflows"))?;
        let bytes = self
            .get(start..end)
            .ok_or_else(|| ContentError::format("<bytes>", "texture byte range is out of bounds"))?;
        Ok(bytes.to_vec())
    }
}

impl ByteRangeSource for Vec<u8> {
    fn byte_len(&self) -> Result<u64, ContentError> {
        Ok(self.len() as u64)
    }

    fn read_range(&self, offset: u64, length: usize) -> Result<Vec<u8>, ContentError> {
        self.as_slice().read_range(offset, length)
    }
}

/// A [`ByteRangeSource`] over a plain file.
///
/// The file stays open for the lifetime of the source; each read seeks and
/// reads exactly the requested range, so a 4K mip chain is never loaded whole
/// to fetch one level.
pub struct FileRangeSource {
    path: PathBuf,
    file: Mutex<File>,
    len: u64,
}

impl FileRangeSource {
    /// Open `path` and read its length.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ContentError> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|error| io_error(path, "open texture", error))?;
        let len = file
            .metadata()
            .map_err(|error| io_error(path, "stat texture", error))?
            .len();
        Ok(Self {
            path: path.to_path_buf(),
            file: Mutex::new(file),
            len,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl ByteRangeSource for FileRangeSource {
    fn byte_len(&self) -> Result<u64, ContentError> {
        Ok(self.len)
    }

    fn read_range(&self, offset: u64, length: usize) -> Result<Vec<u8>, ContentError> {
        let end = offset
            .checked_add(length as u64)
            .ok_or_else(|| ContentError::format(&self.path, "texture byte range overflows"))?;
        if end > self.len {
            return Err(ContentError::format(
                &self.path,
                format!(
                    "texture byte range {offset}..{end} is past the {} byte file",
                    self.len
                ),
            ));
        }
        let mut file = self.file.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        file.seek(SeekFrom::Start(offset))
            .map_err(|error| io_error(&self.path, format!("seek to byte {offset}"), error))?;
        let mut bytes = vec![0u8; length];
        file.read_exact(&mut bytes)
            .map_err(|error| io_error(&self.path, format!("read {length} bytes at {offset}"), error))?;
        Ok(bytes)
    }
}

/// Opens a container header and reads single mip levels.
///
/// [`open`](MipReader::open) reads only the header and level table: 148 bytes
/// for DDS, 80 bytes plus the 24-byte-per-level index for KTX 2. Every later
/// read touches exactly one level's stored bytes, which
/// [`read_level`](MipReader::read_level) decodes through the registry. The
/// whole-texture path ([`BlockCodecRegistry::decode`]) is unaffected.
///
/// [`BlockCodecRegistry::decode`]: super::BlockCodecRegistry::decode
pub struct MipReader<S: ByteRangeSource> {
    source: S,
    path: PathBuf,
    info: TextureInfo,
}

impl<S: ByteRangeSource> MipReader<S> {
    /// Read the header (and the KTX 2 level index) and build the level table.
    pub fn open(source: S, path: &Path) -> Result<Self, ContentError> {
        let len = source.byte_len()?;
        if len < 4 {
            return Err(ContentError::format(
                path,
                format!("texture container needs at least 4 bytes, got {len}"),
            ));
        }
        let head = source.read_range(0, len.min(HEADER_PROBE_BYTES) as usize)?;
        let info = if head.starts_with(&KTX2_IDENTIFIER) {
            open_ktx2(&source, &head, path, len)?
        } else if head.starts_with(&KTX1_IDENTIFIER) {
            return Err(ContentError::unsupported(
                path,
                "KTX 1 containers are not supported; convert to KTX 2",
            ));
        } else if read_u32_le(&head, 0, path)? == DDS_MAGIC {
            open_dds(&head, path, len)?
        } else {
            return Err(ContentError::unsupported(
                path,
                "unrecognised texture container; expected DDS or KTX 2",
            ));
        };
        Ok(Self {
            source,
            path: path.to_path_buf(),
            info,
        })
    }

    /// The parsed container header and level table.
    pub fn info(&self) -> &TextureInfo {
        &self.info
    }

    /// The container this reader was opened on.
    pub fn source(&self) -> &S {
        &self.source
    }

    /// Stored byte range of one level, without reading it.
    pub fn level_range(&self, level: usize) -> Result<Range<u64>, ContentError> {
        let info_level = self
            .info
            .level(level)
            .ok_or_else(|| missing_level(&self.path, level))?;
        Ok(info_level.stored_range())
    }

    /// Read one level's stored bytes, and nothing else.
    pub fn level_bytes(&self, level: usize) -> Result<Vec<u8>, ContentError> {
        let info_level = self
            .info
            .level(level)
            .ok_or_else(|| missing_level(&self.path, level))?;
        let length = usize::try_from(info_level.size).map_err(|_| {
            ContentError::format(&self.path, format!("texture mip level {level} is too large to read"))
        })?;
        self.source.read_range(info_level.offset, length)
    }

    /// Read and decode one level's first image.
    ///
    /// Unrequested levels are neither read nor decoded. Supercompressed KTX 2
    /// payloads and formats without a registered codec stay typed
    /// [`Unsupported`](crate::ContentErrorKind::Unsupported).
    pub fn read_level(
        &self,
        registry: &BlockCodecRegistry,
        level: usize,
    ) -> Result<DecodedImage, ContentError> {
        let bytes = self.level_bytes(level)?;
        registry.decode_level_data(&self.info, level, &bytes, &self.path)
    }
}

fn missing_level(path: &Path, level: usize) -> ContentError {
    ContentError::format(path, format!("texture has no mip level {level}"))
}

fn open_dds(head: &[u8], path: &Path, len: u64) -> Result<TextureInfo, ContentError> {
    let header = dds::parse_header(head, path)?;
    let data_offset = dds::data_offset(&header);
    let payload_len = len.checked_sub(data_offset).ok_or_else(|| {
        ContentError::format(
            path,
            format!("DDS header needs {data_offset} bytes, got {len}"),
        )
    })?;
    let levels = dds::parse_levels(&header, payload_len, path)?;
    Ok(dds::info(&header, levels))
}

fn open_ktx2<S: ByteRangeSource>(
    source: &S,
    head: &[u8],
    path: &Path,
    len: u64,
) -> Result<TextureInfo, ContentError> {
    let header = ktx2::parse_header(head, path)?;
    let index_len = header.mip_levels() as u64 * ktx2::LEVEL_INDEX_ENTRY_BYTES as u64;
    let index_end = ktx2::HEADER_BYTES as u64 + index_len;
    if index_end > len {
        return Err(ContentError::format(path, "KTX 2 level index is truncated"));
    }
    let index = if head.len() as u64 >= index_end {
        head[ktx2::HEADER_BYTES..index_end as usize].to_vec()
    } else {
        source.read_range(ktx2::HEADER_BYTES as u64, index_len as usize)?
    };
    let levels = ktx2::parse_levels(&header, &index, len, path)?;
    Ok(ktx2::info(&header, levels))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{TextureFormat, TextureKind};
    use std::sync::Mutex;

    /// A source that records every range it is asked for, so a test can prove
    /// a level read touches only that level's payload bytes.
    struct RecordingSource {
        bytes: Vec<u8>,
        reads: Mutex<Vec<(u64, usize)>>,
    }

    impl RecordingSource {
        fn new(bytes: Vec<u8>) -> Self {
            Self {
                bytes,
                reads: Mutex::new(Vec::new()),
            }
        }

        fn reads(&self) -> Vec<(u64, usize)> {
            self.reads
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone()
        }
    }

    impl ByteRangeSource for RecordingSource {
        fn byte_len(&self) -> Result<u64, ContentError> {
            Ok(self.bytes.len() as u64)
        }

        fn read_range(&self, offset: u64, length: usize) -> Result<Vec<u8>, ContentError> {
            self.reads
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push((offset, length));
            self.bytes.as_slice().read_range(offset, length)
        }
    }

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    /// A four-level BC1 DDS (8x8 down to 1x1, DXT1 fourcc): levels are 32, 8,
    /// 8 and 8 bytes (one 4x4 block each below 8x8), payload offset 128.
    fn bc1_dds_mipped() -> Vec<u8> {
        let mut bytes = Vec::new();
        push_u32(&mut bytes, DDS_MAGIC);
        push_u32(&mut bytes, 124); // header size
        push_u32(&mut bytes, 0x0002_1007); // CAPS|HEIGHT|WIDTH|PIXELFORMAT|MIPMAPCOUNT
        push_u32(&mut bytes, 8); // height
        push_u32(&mut bytes, 8); // width
        push_u32(&mut bytes, 32); // pitch
        push_u32(&mut bytes, 0); // depth
        push_u32(&mut bytes, 4); // mipmap count
        for _ in 0..11 {
            push_u32(&mut bytes, 0); // reserved
        }
        push_u32(&mut bytes, 32); // pixel format size
        push_u32(&mut bytes, 0x0000_0004); // FOURCC
        push_u32(&mut bytes, u32::from_le_bytes(*b"DXT1")); // four cc
        push_u32(&mut bytes, 0); // bit count
        push_u32(&mut bytes, 0); // masks
        push_u32(&mut bytes, 0);
        push_u32(&mut bytes, 0);
        push_u32(&mut bytes, 0);
        push_u32(&mut bytes, 0x4010_0008); // caps
        push_u32(&mut bytes, 0); // caps2
        push_u32(&mut bytes, 0); // caps3
        push_u32(&mut bytes, 0); // caps4
        push_u32(&mut bytes, 0); // reserved2
        assert_eq!(bytes.len(), 128);
        for level in 0..4u32 {
            let blocks = if level == 0 { 4 } else { 1 };
            for _ in 0..blocks {
                // Opaque white: color0 = white, indices = 0.
                bytes.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0]);
            }
        }
        assert_eq!(bytes.len(), 128 + 56);
        bytes
    }

    /// A two-level, 8x8 BC1 KTX 2 with the standard chain layout: 80-byte
    /// header, 48-byte level index, a 48-byte DFD placeholder the header
    /// declares, then both levels' payloads at 176 and 208.
    fn bc1_ktx2_mipped() -> Vec<u8> {
        const DFD_BYTES: usize = 48;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&KTX2_IDENTIFIER);
        push_u32(&mut bytes, 131); // VK_FORMAT_BC1_RGB_UNORM_BLOCK
        push_u32(&mut bytes, 1);
        push_u32(&mut bytes, 8); // width
        push_u32(&mut bytes, 8); // height
        push_u32(&mut bytes, 0);
        push_u32(&mut bytes, 0);
        push_u32(&mut bytes, 1);
        push_u32(&mut bytes, 2); // level count
        push_u32(&mut bytes, 0); // supercompression
        let index_end = ktx2::HEADER_BYTES + 2 * ktx2::LEVEL_INDEX_ENTRY_BYTES;
        push_u32(&mut bytes, index_end as u32); // dfd offset
        push_u32(&mut bytes, DFD_BYTES as u32); // dfd length
        push_u32(&mut bytes, 0);
        push_u32(&mut bytes, 0);
        push_u64(&mut bytes, 0);
        push_u64(&mut bytes, 0);
        assert_eq!(bytes.len(), ktx2::HEADER_BYTES);
        let level0_offset = (index_end + DFD_BYTES) as u64;
        push_u64(&mut bytes, level0_offset);
        push_u64(&mut bytes, 32);
        push_u64(&mut bytes, 32);
        push_u64(&mut bytes, level0_offset + 32);
        push_u64(&mut bytes, 8);
        push_u64(&mut bytes, 8);
        assert_eq!(bytes.len(), index_end);
        bytes.resize(level0_offset as usize, 0); // DFD placeholder
        // Opaque white: color0 0xFFFF > color1, all indices 0.
        const WHITE_BLOCK: [u8; 8] = [0xFF, 0xFF, 0x00, 0x00, 0, 0, 0, 0];
        for _ in 0..4 {
            bytes.extend_from_slice(&WHITE_BLOCK);
        }
        bytes.extend_from_slice(&WHITE_BLOCK);
        assert_eq!(bytes.len(), 216);
        bytes
    }

    #[test]
    fn dds_reader_reads_one_level_and_only_that_level() {
        let bytes = bc1_dds_mipped();
        let source = RecordingSource::new(bytes.clone());
        let reader = MipReader::open(source, Path::new("t.dds")).expect("open");
        assert_eq!(reader.info().format, TextureFormat::Bc1);
        assert_eq!(reader.info().kind, TextureKind::D2);
        assert_eq!(reader.info().mip_levels, 4);
        assert_eq!(reader.level_range(3).expect("level 3 range"), 176..184);

        let registry = BlockCodecRegistry::default();
        let decoded = reader.read_level(&registry, 3).expect("decode mip 3");
        assert_eq!((decoded.width, decoded.height), (1, 1));
        assert_eq!(decoded.data, vec![255, 255, 255, 255]);
        // Header probe, then exactly the 8 payload bytes of level 3.
        assert_eq!(
            reader.source().reads(),
            vec![(0, 148), (176, 8)],
            "only the requested level is read"
        );
        assert_eq!(
            reader.level_bytes(1).expect("level 1").len(),
            8,
            "level bytes come from the level table"
        );
    }

    #[test]
    fn dds_reader_matches_the_whole_file_path() {
        let bytes = bc1_dds_mipped();
        let whole = BlockCodecRegistry::default()
            .decode_level(
                &crate::codec::parse_container(&bytes, Path::new("t.dds")).expect("container"),
                2,
                &bytes,
                Path::new("t.dds"),
            )
            .expect("whole-file decode");
        let reader = MipReader::open(bytes.as_slice(), Path::new("t.dds")).expect("open");
        assert_eq!(
            reader
                .read_level(&BlockCodecRegistry::default(), 2)
                .expect("per-level decode"),
            whole
        );
    }

    #[test]
    fn ktx2_level_index_round_trips_through_the_reader() {
        let bytes = bc1_ktx2_mipped();
        let reader = MipReader::open(bytes.as_slice(), Path::new("t.ktx2")).expect("open");
        assert_eq!(reader.info().mip_levels, 2);
        assert_eq!(reader.level_range(0).expect("level 0 range"), 176..208);
        assert_eq!(reader.level_range(1).expect("level 1 range"), 208..216);
        let registry = BlockCodecRegistry::default();
        let level0 = reader.read_level(&registry, 0).expect("level 0");
        assert_eq!((level0.width, level0.height), (8, 8));
        assert_eq!(level0.data.len(), 8 * 8 * 4);
        let level1 = reader.read_level(&registry, 1).expect("level 1");
        assert_eq!((level1.width, level1.height), (4, 4));
        // One white BC1 block (color0 0xFFFF > color1, all indices 0).
        assert_eq!(level1.data, vec![255u8, 255, 255, 255].repeat(16));
        assert!(reader.read_level(&registry, 2).is_err(), "no third level");
    }

    #[test]
    fn ktx2_reader_reads_only_the_requested_level() {
        let source = RecordingSource::new(bc1_ktx2_mipped());
        let reader = MipReader::open(source, Path::new("t.ktx2")).expect("open");
        reader
            .read_level(&BlockCodecRegistry::default(), 1)
            .expect("level 1");
        // 148-byte header probe (index included), then level 1's 8 bytes.
        assert_eq!(reader.source().reads(), vec![(0, 148), (208, 8)]);
    }

    #[test]
    fn overrunning_level_table_is_a_typed_error() {
        let header = dds::parse_header(&bc1_dds_mipped(), Path::new("t.dds")).expect("header");
        let error = dds::parse_levels(&header, 40, Path::new("t.dds")).expect_err("overruns");
        assert_eq!(error.kind(), crate::ContentErrorKind::Format);
        assert!(error.message().contains("payload"), "{}", error.message());
        // A truncated container fails at open, not at read.
        let mut truncated = bc1_dds_mipped();
        truncated.truncate(150);
        assert!(MipReader::open(truncated.as_slice(), Path::new("t.dds")).is_err());
    }

    #[test]
    fn unsupported_formats_stay_typed_unsupported() {
        let mut bytes = bc1_ktx2_mipped();
        bytes[12..16].copy_from_slice(&143u32.to_le_bytes()); // VK_FORMAT_BC6H_UF16_BLOCK
        let reader = MipReader::open(bytes.as_slice(), Path::new("t.ktx2")).expect("open");
        assert_eq!(reader.info().format, TextureFormat::Bc6hUf16);
        let error = reader
            .read_level(&BlockCodecRegistry::default(), 0)
            .expect_err("bc6h has no codec");
        assert!(error.is_unsupported());
    }

    #[test]
    fn package_entries_feed_per_level_reads() {
        use crate::package::{Compression, Package, PackageWriter};

        let dds = bc1_dds_mipped();
        let path = Path::new("assets/t.ktx2");
        let mut writer = PackageWriter::new();
        writer
            .add("tex/stored.dds", dds.clone(), Compression::Stored)
            .expect("add stored");
        writer
            .add("tex/deflated.dds", dds.clone(), Compression::Deflated)
            .expect("add deflated");
        let archive = writer.finish(path).expect("finish");
        let package = Package::read(archive, path).expect("read");

        let registry = BlockCodecRegistry::default();
        for name in ["tex/stored.dds", "tex/deflated.dds"] {
            let source = package.range_source(name, path).expect("range source");
            assert_eq!(source.byte_len().expect("len"), 184);
            let reader = MipReader::open(source, path).expect("open");
            let level = reader.read_level(&registry, 2).expect("level 2");
            assert_eq!((level.width, level.height), (2, 2), "{name}");
            assert_eq!(level.data, vec![255u8, 255, 255, 255].repeat(4), "{name}");
        }
    }

    #[test]
    fn file_source_reads_an_exact_range() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("repame-mip-reader-{}.dds", std::process::id()));
        std::fs::write(&path, bc1_dds_mipped()).expect("write fixture");
        let source = FileRangeSource::open(&path).expect("open");
        assert_eq!(source.byte_len().expect("len"), 184);
        let reader = MipReader::open(source, &path).expect("open reader");
        let level = reader
            .read_level(&BlockCodecRegistry::default(), 1)
            .expect("level 1");
        assert_eq!((level.width, level.height), (4, 4));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn slice_source_rejects_out_of_range_reads() {
        let bytes = bc1_dds_mipped();
        assert!(bytes.as_slice().read_range(180, 8).is_err());
        assert!(bytes.as_slice().read_range(0, 1).is_ok());
    }
}
