//! Packed content: a ZIP container plus a PCK-like directory index.
//!
//! [`PackageWriter`] writes a ZIP archive with stored or deflated entries;
//! [`Package`] reads one from either its central directory and extracts
//! entries, checking sizes and CRC-32. [`PckDirectory`] is the Godot-PCK-like
//! lookup table (path → compression, offset, sizes, CRC) that ships beside or
//! inside a packed archive so content can be resolved without walking the ZIP.
//!
//! Stored and deflated (method 0 and 8) are implemented. Every other method,
//! ZIP64 and multi-disk archives are refused with a typed error.

use std::collections::BTreeMap;
use std::io::{Cursor, Read, Write};
use std::path::Path;

use flate2::Compression as FlateCompression;
use flate2::read::DeflateDecoder;
use flate2::write::DeflateEncoder;
use serde::{Deserialize, Serialize};

use crate::error::ContentError;

pub const ZIP_LOCAL_HEADER_SIGNATURE: u32 = 0x0403_4b50;
pub const ZIP_CENTRAL_HEADER_SIGNATURE: u32 = 0x0201_4b50;
pub const ZIP_END_OF_CENTRAL_DIRECTORY_SIGNATURE: u32 = 0x0605_4b50;
pub const ZIP_LOCAL_HEADER_BYTES: u64 = 30;
pub const ZIP_CENTRAL_HEADER_BYTES: u64 = 46;
pub const ZIP_END_OF_CENTRAL_DIRECTORY_BYTES: usize = 22;
pub const ZIP_UTF8_FLAG: u16 = 0x0800;
pub const ZIP_ENCRYPTED_FLAG: u16 = 0x0001;
pub const ZIP_MAX_ENTRIES: usize = 0xFFFF;
pub const ZIP_MAX_VALUE: u64 = 0xFFFF_FFFF;

/// The PCK-like directory framing magic, `"PCK1"` little-endian.
pub const PCK_MAGIC: u32 = 0x314b_4350;
pub const PCK_VERSION: u32 = 1;

const DOS_TIME_1980: u16 = 0;
const DOS_DATE_1980: u16 = 0x0021;

/// Compression method of one archive entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Compression {
    Stored,
    Deflated,
}

impl Compression {
    pub fn method(self) -> u16 {
        match self {
            Self::Stored => 0,
            Self::Deflated => 8,
        }
    }

    pub fn from_method(method: u16) -> Option<Self> {
        match method {
            0 => Some(Self::Stored),
            8 => Some(Self::Deflated),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stored => "stored",
            Self::Deflated => "deflated",
        }
    }
}

/// One entry of a ZIP archive's central directory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackageEntry {
    pub name: String,
    pub compression: Compression,
    pub crc32: u32,
    pub compressed_size: u64,
    pub size: u64,
    /// Offset of the local file header.
    pub header_offset: u64,
    /// Offset of the payload, i.e. past the local header's name and extra fields.
    pub data_offset: u64,
}

/// CRC-32 (IEEE 802.3), the checksum both ZIP and PCK use.
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for byte in bytes {
        crc = CRC32_TABLE[((crc ^ u32::from(*byte)) & 0xff) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

const CRC32_TABLE: [u32; 256] = build_crc32_table();

const fn build_crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut index = 0;
    while index < 256 {
        let mut value = index as u32;
        let mut bit = 0;
        while bit < 8 {
            value = if value & 1 != 0 {
                0xEDB8_8320 ^ (value >> 1)
            } else {
                value >> 1
            };
            bit += 1;
        }
        table[index] = value;
        index += 1;
    }
    table
}

/// A readable ZIP archive with its entries indexed.
#[derive(Clone, Debug)]
pub struct Package {
    bytes: Vec<u8>,
    entries: BTreeMap<String, PackageEntry>,
    order: Vec<String>,
}

impl Package {
    /// Index a ZIP archive. `source` only names the container in errors.
    pub fn read(bytes: Vec<u8>, source: &Path) -> Result<Self, ContentError> {
        let eocd = find_end_of_central_directory(&bytes, source)?;
        let entries_on_disk = u32_at(&bytes, eocd + 8, source)?;
        let entry_count = u32_at(&bytes, eocd + 10, source)?;
        let directory_size = u32_at(&bytes, eocd + 12, source)?;
        let directory_offset = u32_at(&bytes, eocd + 16, source)?;
        let disk = u32_at(&bytes, eocd + 4, source)?;
        if disk != 0 || entries_on_disk != entry_count {
            return Err(ContentError::unsupported(
                source,
                "multi-disk ZIP archives are not supported",
            ));
        }
        if entry_count == u32::from(u16::MAX) || directory_offset == u32::MAX {
            return Err(ContentError::unsupported(
                source,
                "ZIP64 archives are not supported",
            ));
        }
        let directory_end = directory_offset as u64 + u64::from(directory_size);
        if directory_end > bytes.len() as u64 || eocd as u64 + 22 > bytes.len() as u64 {
            return Err(ContentError::format(source, "ZIP central directory is out of range"));
        }

        let mut entries = BTreeMap::new();
        let mut order = Vec::with_capacity(entry_count as usize);
        let mut cursor = directory_offset as usize;
        for index in 0..entry_count {
            if u32_at(&bytes, cursor, source)? != ZIP_CENTRAL_HEADER_SIGNATURE {
                return Err(ContentError::format(
                    source,
                    format!("ZIP central directory entry {index} has a bad signature"),
                ));
            }
            let flags = u16_at(&bytes, cursor + 8, source)?;
            let method = u16_at(&bytes, cursor + 10, source)?;
            let crc = u32_at(&bytes, cursor + 16, source)?;
            let compressed_size = u32_at(&bytes, cursor + 20, source)? as u64;
            let size = u32_at(&bytes, cursor + 24, source)? as u64;
            let name_length = u16_at(&bytes, cursor + 28, source)? as usize;
            let extra_length = u16_at(&bytes, cursor + 30, source)? as usize;
            let comment_length = u16_at(&bytes, cursor + 32, source)? as usize;
            let header_offset = u32_at(&bytes, cursor + 42, source)? as u64;
            if flags & ZIP_ENCRYPTED_FLAG != 0 {
                return Err(ContentError::unsupported(
                    source,
                    "encrypted ZIP entries are not supported",
                ));
            }
            let compression = Compression::from_method(method).ok_or_else(|| {
                ContentError::unsupported(
                    source,
                    format!("ZIP compression method {method} is not implemented"),
                )
            })?;
            let name_start = cursor + ZIP_CENTRAL_HEADER_BYTES as usize;
            let name_end = name_start + name_length;
            let name_bytes = bytes
                .get(name_start..name_end)
                .ok_or_else(|| ContentError::format(source, "ZIP entry name is truncated"))?;
            let name = std::str::from_utf8(name_bytes)
                .map_err(|error| ContentError::format(source, format!("ZIP entry name is not UTF-8: {error}")))?
                .to_string();

            let header = header_offset as usize;
            if u32_at(&bytes, header, source)? != ZIP_LOCAL_HEADER_SIGNATURE {
                return Err(ContentError::format(
                    source,
                    format!("ZIP entry `{name}` has a bad local header"),
                ));
            }
            let local_name_length = u16_at(&bytes, header + 26, source)? as usize;
            let local_extra_length = u16_at(&bytes, header + 28, source)? as usize;
            let data_offset =
                header_offset + ZIP_LOCAL_HEADER_BYTES + local_name_length as u64 + local_extra_length as u64;
            let data_end = data_offset + compressed_size;
            if data_end > bytes.len() as u64 {
                return Err(ContentError::format(
                    source,
                    format!("ZIP entry `{name}` payload is truncated"),
                ));
            }

            cursor = name_end + extra_length + comment_length;
            if order.iter().any(|existing| existing == &name) {
                return Err(ContentError::format(
                    source,
                    format!("ZIP archive repeats entry `{name}`"),
                ));
            }
            order.push(name.clone());
            entries.insert(
                name.clone(),
                PackageEntry {
                    name,
                    compression,
                    crc32: crc,
                    compressed_size,
                    size,
                    header_offset,
                    data_offset,
                },
            );
        }

        Ok(Self {
            bytes,
            entries,
            order,
        })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Entries in archive order.
    pub fn entries(&self) -> impl Iterator<Item = &PackageEntry> {
        self.order.iter().filter_map(|name| self.entries.get(name))
    }

    pub fn entry(&self, name: &str) -> Option<&PackageEntry> {
        self.entries.get(name)
    }

    /// Extract one entry, verifying its size and CRC-32.
    pub fn read_file(&self, name: &str, source: &Path) -> Result<Vec<u8>, ContentError> {
        let entry = self
            .entries
            .get(name)
            .ok_or_else(|| ContentError::new(source, format!("package has no entry `{name}`")))?;
        self.read_entry(entry, source)
    }

    pub fn read_entry(&self, entry: &PackageEntry, source: &Path) -> Result<Vec<u8>, ContentError> {
        let start = entry.data_offset as usize;
        let end = start + entry.compressed_size as usize;
        let stored = self.bytes.get(start..end).ok_or_else(|| {
            ContentError::format(source, format!("package entry `{}` is truncated", entry.name))
        })?;
        let data = match entry.compression {
            Compression::Stored => stored.to_vec(),
            Compression::Deflated => {
                let mut decoder = DeflateDecoder::new(Cursor::new(stored));
                let mut data = Vec::with_capacity(entry.size as usize);
                decoder.read_to_end(&mut data).map_err(|error| {
                    ContentError::format(
                        source,
                        format!("inflate package entry `{}`: {error}", entry.name),
                    )
                })?;
                data
            }
        };
        if data.len() as u64 != entry.size {
            return Err(ContentError::format(
                source,
                format!(
                    "package entry `{}` expands to {} bytes, expected {}",
                    entry.name,
                    data.len(),
                    entry.size
                ),
            ));
        }
        let checksum = crc32(&data);
        if checksum != entry.crc32 {
            return Err(ContentError::format(
                source,
                format!(
                    "package entry `{}` fails CRC-32 (expected {:08x}, got {checksum:08x})",
                    entry.name, entry.crc32
                ),
            ));
        }
        Ok(data)
    }
}

/// Builds a ZIP archive in memory.
#[derive(Default)]
pub struct PackageWriter {
    entries: Vec<(String, Vec<u8>, Compression)>,
}

impl PackageWriter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue one entry. Names are POSIX-style content paths.
    pub fn add(
        &mut self,
        name: impl Into<String>,
        data: impl Into<Vec<u8>>,
        compression: Compression,
    ) -> Result<(), ContentError> {
        let name = name.into();
        validate_entry_name(&name)?;
        if self.entries.iter().any(|entry| entry.0 == name) {
            return Err(ContentError::new(
                "<package>",
                format!("package repeats entry `{name}`"),
            ));
        }
        self.entries.push((name, data.into(), compression));
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Serialize the archive.
    pub fn finish(self, source: &Path) -> Result<Vec<u8>, ContentError> {
        if self.entries.len() >= ZIP_MAX_ENTRIES {
            return Err(ContentError::unsupported(
                source,
                "more entries than a non-ZIP64 archive can index",
            ));
        }
        let mut out: Vec<u8> = Vec::new();
        let mut directory: Vec<(String, u32, u64, u64, u64, Compression)> = Vec::new();

        for (name, data, compression) in &self.entries {
            let offset = out.len() as u64;
            let checksum = crc32(data);
            let payload = match compression {
                Compression::Stored => data.clone(),
                Compression::Deflated => {
                    let mut encoder =
                        DeflateEncoder::new(Vec::new(), FlateCompression::default());
                    encoder.write_all(data).map_err(|error| {
                        ContentError::new(source, format!("deflate package entry `{name}`: {error}"))
                    })?;
                    encoder.finish().map_err(|error| {
                        ContentError::new(source, format!("deflate package entry `{name}`: {error}"))
                    })?
                }
            };
            if payload.len() as u64 > ZIP_MAX_VALUE || data.len() as u64 > ZIP_MAX_VALUE {
                return Err(ContentError::unsupported(
                    source,
                    format!("package entry `{name}` needs ZIP64"),
                ));
            }
            push_u32(&mut out, ZIP_LOCAL_HEADER_SIGNATURE);
            push_u16(&mut out, 20);
            push_u16(&mut out, ZIP_UTF8_FLAG);
            push_u16(&mut out, compression.method());
            push_u16(&mut out, DOS_TIME_1980);
            push_u16(&mut out, DOS_DATE_1980);
            push_u32(&mut out, checksum);
            push_u32(&mut out, payload.len() as u32);
            push_u32(&mut out, data.len() as u32);
            push_u16(&mut out, name.len() as u16);
            push_u16(&mut out, 0);
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&payload);
            directory.push((
                name.clone(),
                checksum,
                payload.len() as u64,
                data.len() as u64,
                offset,
                *compression,
            ));
        }

        let directory_offset = out.len() as u64;
        for (name, checksum, compressed_size, size, offset, compression) in &directory {
            if *offset > ZIP_MAX_VALUE {
                return Err(ContentError::unsupported(
                    source,
                    format!("package entry `{name}` needs ZIP64"),
                ));
            }
            push_u32(&mut out, ZIP_CENTRAL_HEADER_SIGNATURE);
            push_u16(&mut out, 20);
            push_u16(&mut out, 20);
            push_u16(&mut out, ZIP_UTF8_FLAG);
            push_u16(&mut out, compression.method());
            push_u16(&mut out, DOS_TIME_1980);
            push_u16(&mut out, DOS_DATE_1980);
            push_u32(&mut out, *checksum);
            push_u32(&mut out, *compressed_size as u32);
            push_u32(&mut out, *size as u32);
            push_u16(&mut out, name.len() as u16);
            push_u16(&mut out, 0);
            push_u16(&mut out, 0);
            push_u16(&mut out, 0);
            push_u16(&mut out, 0);
            push_u32(&mut out, 0);
            push_u32(&mut out, *offset as u32);
            out.extend_from_slice(name.as_bytes());
        }
        let directory_size = out.len() as u64 - directory_offset;
        if directory_size > ZIP_MAX_VALUE || directory_offset > ZIP_MAX_VALUE {
            return Err(ContentError::unsupported(source, "package needs ZIP64"));
        }

        push_u32(&mut out, ZIP_END_OF_CENTRAL_DIRECTORY_SIGNATURE);
        push_u16(&mut out, 0);
        push_u16(&mut out, 0);
        push_u16(&mut out, directory.len() as u16);
        push_u16(&mut out, directory.len() as u16);
        push_u32(&mut out, directory_size as u32);
        push_u32(&mut out, directory_offset as u32);
        push_u16(&mut out, 0);
        Ok(out)
    }
}

/// One [`PckDirectory`] record: where an entry lives inside a packed archive.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PckEntry {
    pub path: String,
    pub compression: Compression,
    pub header_offset: u64,
    pub data_offset: u64,
    pub size: u64,
    pub compressed_size: u64,
    pub crc32: u32,
}

/// Godot-PCK-like directory: content path → location inside a packed archive.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PckDirectory {
    pub version: u32,
    pub entries: Vec<PckEntry>,
}

impl PckDirectory {
    /// Build the index of a readable archive.
    pub fn from_package(package: &Package) -> Self {
        Self {
            version: PCK_VERSION,
            entries: package
                .entries()
                .map(|entry| PckEntry {
                    path: entry.name.clone(),
                    compression: entry.compression,
                    header_offset: entry.header_offset,
                    data_offset: entry.data_offset,
                    size: entry.size,
                    compressed_size: entry.compressed_size,
                    crc32: entry.crc32,
                })
                .collect(),
        }
    }

    pub fn entry(&self, path: &str) -> Option<&PckEntry> {
        self.entries.iter().find(|entry| entry.path == path)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Serialize the index: magic, version, entry count, index length, then the
    /// RON-encoded records.
    pub fn write(&self, source: &Path) -> Result<Vec<u8>, ContentError> {
        let index = ron::ser::to_string(self)
            .map_err(|error| ContentError::new(source, format!("serialize PCK index: {error}")))?;
        let mut out = Vec::with_capacity(16 + index.len());
        push_u32(&mut out, PCK_MAGIC);
        push_u32(&mut out, PCK_VERSION);
        push_u32(&mut out, self.entries.len() as u32);
        push_u32(&mut out, index.len() as u32);
        out.extend_from_slice(index.as_bytes());
        Ok(out)
    }

    /// Parse an index written by [`PckDirectory::write`].
    pub fn read(bytes: &[u8], source: &Path) -> Result<Self, ContentError> {
        if bytes.len() < 16 {
            return Err(ContentError::format(source, "PCK index is truncated"));
        }
        if u32_at(bytes, 0, source)? != PCK_MAGIC {
            return Err(ContentError::format(source, "not a PCK index"));
        }
        let version = u32_at(bytes, 4, source)?;
        if version != PCK_VERSION {
            return Err(ContentError::unsupported(
                source,
                format!("PCK index version {version} is not implemented"),
            ));
        }
        let count = u32_at(bytes, 8, source)? as usize;
        let length = u32_at(bytes, 12, source)? as usize;
        let index = bytes
            .get(16..16 + length)
            .ok_or_else(|| ContentError::format(source, "PCK index body is truncated"))?;
        let text = std::str::from_utf8(index)
            .map_err(|error| ContentError::format(source, format!("PCK index is not UTF-8: {error}")))?;
        let directory: Self = ron::from_str(text)
            .map_err(|error| ContentError::format(source, format!("invalid RON: {error}")))?;
        if directory.entries.len() != count {
            return Err(ContentError::format(
                source,
                format!(
                    "PCK index declares {count} entries but holds {}",
                    directory.entries.len()
                ),
            ));
        }
        Ok(directory)
    }
}

fn validate_entry_name(name: &str) -> Result<(), ContentError> {
    if name.is_empty() || name.starts_with('/') || name.contains('\\') {
        return Err(ContentError::new(
            "<package>",
            format!("invalid package entry name `{name}`"),
        ));
    }
    if name
        .split('/')
        .any(|segment| segment == ".." || segment.is_empty())
    {
        return Err(ContentError::new(
            "<package>",
            format!("invalid package entry name `{name}`"),
        ));
    }
    Ok(())
}

fn find_end_of_central_directory(bytes: &[u8], source: &Path) -> Result<usize, ContentError> {
    let minimum = ZIP_END_OF_CENTRAL_DIRECTORY_BYTES;
    if bytes.len() < minimum {
        return Err(ContentError::format(source, "not a ZIP archive"));
    }
    let limit = bytes.len().saturating_sub(minimum + 0xFFFF);
    for offset in (limit..=bytes.len() - minimum).rev() {
        if u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) == ZIP_END_OF_CENTRAL_DIRECTORY_SIGNATURE
        {
            return Ok(offset);
        }
    }
    Err(ContentError::format(
        source,
        "ZIP end-of-central-directory record is missing",
    ))
}

fn u16_at(bytes: &[u8], offset: usize, source: &Path) -> Result<u16, ContentError> {
    let slice = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| ContentError::format(source, "ZIP archive is truncated"))?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn u32_at(bytes: &[u8], offset: usize, source: &Path) -> Result<u32, ContentError> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| ContentError::format(source, "ZIP archive is truncated"))?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_round_trips_stored_and_deflated_entries() {
        let path = Path::new("pack.repapck");
        let text = b"repame content pack".repeat(64);
        let mut writer = PackageWriter::new();
        writer
            .add("scenes/main.ron", b"(format: 1)".to_vec(), Compression::Stored)
            .expect("add stored");
        writer
            .add("assets/notes.txt", text.clone(), Compression::Deflated)
            .expect("add deflated");
        let archive = writer.finish(path).expect("finish");

        let package = Package::read(archive, path).expect("read");
        assert_eq!(package.len(), 2);
        let names: Vec<&str> = package.entries().map(|entry| entry.name.as_str()).collect();
        assert_eq!(names, vec!["scenes/main.ron", "assets/notes.txt"]);
        assert_eq!(
            package.entry("scenes/main.ron").map(|entry| entry.compression),
            Some(Compression::Stored)
        );
        assert_eq!(
            package.entry("assets/notes.txt").map(|entry| entry.compression),
            Some(Compression::Deflated)
        );
        assert_eq!(
            package.read_file("scenes/main.ron", path).expect("stored"),
            b"(format: 1)"
        );
        assert_eq!(
            package.read_file("assets/notes.txt", path).expect("deflated"),
            text
        );
        assert!(package.entry("assets/notes.txt").expect("entry").compressed_size < text.len() as u64);
        assert_eq!(
            package.entry("assets/notes.txt").map(|entry| entry.crc32),
            Some(crc32(&text))
        );

        let directory = PckDirectory::from_package(&package);
        assert_eq!(directory.len(), 2);
        let index_bytes = directory.write(path).expect("write index");
        let parsed = PckDirectory::read(&index_bytes, path).expect("read index");
        assert_eq!(parsed, directory);
    }
}
