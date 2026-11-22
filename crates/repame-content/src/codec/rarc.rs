//! The RARC archive: the filesystem volume a GameCube game mounts.
//!
//! A RARC file is a header, a table block and the file data. The header says
//! how long the file is and where its data starts, the table block holds the
//! directory table, the file entry table and the string table the names live
//! in. Every offset in the table block is relative to the block's own start.
//!
//! Retail: `tww/include/JSystem/JKernel/JKRArchive.h:10-31` for the header
//! and the table block, `:52-80` for the two entry tables, and
//! `tww/src/JSystem/JKernel/JKRMemArchive.cpp:116-124` for where each table
//! sits inside the file. The lookups are `JKRArchive::findFsResource`
//! (`tww/src/JSystem/JKernel/JKRArchivePri.cpp:109-132`),
//! `JKRArchive::findNameResource` (`:143-155`) and
//! `JKRArchive::findIdResource` (`:169-186`).

use std::path::{Path, PathBuf};

use crate::codec::yaz0;
use crate::error::{ContentError, ContentErrorKind};

/// `"RARC"` in byte order.
pub const RARC_MAGIC: [u8; 4] = *b"RARC";

/// The archive header: signature, file length, header length, data offset and
/// data length (`SArcHeader`).
const HEADER_BYTES: usize = 0x20;
/// One directory entry (`JKRArchive::SDIDirEntry`).
const DIR_ENTRY_BYTES: usize = 0x10;
/// One file entry (`JKRArchive::SDIFileEntry`).
const FILE_ENTRY_BYTES: usize = 0x14;
/// The entry flag that marks a directory.
const FLAG_DIRECTORY: u8 = 0x02;
/// The entry flag that marks a file id as usable.
const FLAG_FILE_ID: u8 = 0x01;
/// The data offset a directory entry carries when it has no directory of its
/// own, which is how the root directory is named.
const NO_DIRECTORY: u32 = 0xFFFF_FFFF;

/// One directory of the tree: the range of file entries it holds.
#[derive(Debug)]
struct DirNode {
    first: usize,
    count: usize,
}

/// One entry of the file table.
#[derive(Debug)]
struct FileEntry {
    id: u16,
    name: String,
    flags: u8,
    offset: u32,
    size: u32,
}

impl FileEntry {
    fn is_directory(&self) -> bool {
        self.flags & FLAG_DIRECTORY != 0
    }

    fn has_file_id(&self) -> bool {
        self.flags & FLAG_FILE_ID != 0
    }
}

/// A mounted RARC volume: a tree of named byte ranges over one buffer.
#[derive(Debug)]
pub struct Archive {
    source: PathBuf,
    data: Vec<u8>,
    data_start: usize,
    dirs: Vec<DirNode>,
    files: Vec<FileEntry>,
}

impl Archive {
    /// Open an archive file, expanding it first when it is a Yaz0 compressed
    /// volume, which is how the retail ripper reads one
    /// (`tww/src/JSystem/JKernel/JKRDvdRipper.cpp:131-159`).
    pub fn open(data: Vec<u8>, source: &Path) -> Result<Self, ContentError> {
        let data = yaz0::expand(data, source)?;
        Self::parse(data, source)
    }

    /// Parse a RARC volume. `source` only names the archive in errors.
    pub fn parse(data: Vec<u8>, source: &Path) -> Result<Self, ContentError> {
        if data.len() < HEADER_BYTES {
            return Err(ContentError::format(
                source,
                format!("archive is {} bytes, shorter than its header", data.len()),
            ));
        }
        if !is_rarc(&data) {
            return Err(ContentError::format(source, "not a RARC archive"));
        }
        let header_length = read_u32(&data, 0x08, source)?;
        if header_length != HEADER_BYTES as u32 {
            return Err(ContentError::format(
                source,
                format!("archive header is {header_length} bytes, expected {HEADER_BYTES}"),
            ));
        }
        let file_length = read_u32(&data, 0x04, source)? as usize;
        if file_length != data.len() {
            return Err(ContentError::format(
                source,
                format!("archive is {} bytes, its header says {file_length}", data.len()),
            ));
        }
        let relative = |at: usize| -> Result<usize, ContentError> {
            HEADER_BYTES
                .checked_add(read_u32(&data, at, source)? as usize)
                .ok_or_else(|| ContentError::format(source, "archive offset overflows"))
        };
        let data_start = relative(0x0C)?;
        let data_length = read_u32(&data, 0x10, source)? as usize;
        if data_start.checked_add(data_length) != Some(data.len()) {
            return Err(ContentError::format(
                source,
                "archive data region does not match its header",
            ));
        }

        let dir_count = read_u32(&data, 0x20, source)? as usize;
        let dir_offset = relative(0x24)?;
        let file_count = read_u32(&data, 0x28, source)? as usize;
        let file_offset = relative(0x2C)?;
        let string_length = read_u32(&data, 0x30, source)? as usize;
        let string_offset = relative(0x34)?;
        if string_offset.checked_add(string_length) != Some(data_start) {
            return Err(ContentError::format(
                source,
                "archive string table does not match its header",
            ));
        }
        if dir_count == 0 {
            return Err(ContentError::format(source, "archive has no root directory"));
        }
        let dir_end = dir_offset
            .checked_add(
                dir_count
                    .checked_mul(DIR_ENTRY_BYTES)
                    .ok_or_else(|| ContentError::format(source, "archive directory table overflows"))?,
            )
            .ok_or_else(|| ContentError::format(source, "archive directory table overflows"))?;
        if dir_end > data.len() {
            return Err(ContentError::format(
                source,
                "archive directory table is out of range",
            ));
        }
        let file_end = file_offset
            .checked_add(
                file_count
                    .checked_mul(FILE_ENTRY_BYTES)
                    .ok_or_else(|| ContentError::format(source, "archive entry table overflows"))?,
            )
            .ok_or_else(|| ContentError::format(source, "archive entry table overflows"))?;
        if file_end > data.len() {
            return Err(ContentError::format(
                source,
                "archive entry table is out of range",
            ));
        }

        let string_at = |at: usize| -> Result<String, ContentError> {
            let start = string_offset
                .checked_add(at)
                .filter(|&start| start < data_start)
                .ok_or_else(|| ContentError::format(source, "archive name offset is out of range"))?;
            let end = data[start..data_start]
                .iter()
                .position(|&byte| byte == 0)
                .map(|relative| start + relative)
                .ok_or_else(|| ContentError::format(source, "archive name is not terminated"))?;
            std::str::from_utf8(&data[start..end])
                .map(str::to_owned)
                .map_err(|_| ContentError::format(source, "archive name is not UTF-8"))
        };

        let mut dirs = Vec::with_capacity(dir_count);
        for index in 0..dir_count {
            let at = dir_offset + index * DIR_ENTRY_BYTES;
            if index == 0 && &data[at..at + 4] != b"ROOT" {
                return Err(ContentError::format(source, "archive root marker is missing"));
            }
            let name = string_at(read_u32(&data, at + 4, source)? as usize)?;
            let hash = read_u16(&data, at + 8, source)?;
            if !hash_matches(&name, hash) {
                return Err(ContentError::format(
                    source,
                    format!("archive directory `{name}` does not match its name hash"),
                ));
            }
            let count = usize::from(read_u16(&data, at + 10, source)?);
            let first = read_u32(&data, at + 12, source)? as usize;
            let end = first
                .checked_add(count)
                .ok_or_else(|| ContentError::format(source, "archive directory range overflows"))?;
            if end > file_count {
                return Err(ContentError::format(
                    source,
                    "archive directory range is out of bounds",
                ));
            }
            dirs.push(DirNode { first, count });
        }

        let mut files = Vec::with_capacity(file_count);
        for index in 0..file_count {
            let at = file_offset + index * FILE_ENTRY_BYTES;
            let id = read_u16(&data, at, source)?;
            let hash = read_u16(&data, at + 2, source)?;
            let flags_and_name = read_u32(&data, at + 4, source)?;
            let flags = (flags_and_name >> 24) as u8;
            let name = string_at((flags_and_name & 0x00FF_FFFF) as usize)?;
            if !hash_matches(&name, hash) {
                return Err(ContentError::format(
                    source,
                    format!("archive entry `{name}` does not match its name hash"),
                ));
            }
            let offset = read_u32(&data, at + 8, source)?;
            let size = read_u32(&data, at + 12, source)?;
            if read_u32(&data, at + 16, source)? != 0 {
                return Err(ContentError::format(
                    source,
                    format!("archive entry `{name}` carries a data pointer"),
                ));
            }
            let entry = FileEntry {
                id,
                name: name.clone(),
                flags,
                offset,
                size,
            };
            if entry.is_directory() {
                if offset != NO_DIRECTORY && offset as usize >= dir_count {
                    return Err(ContentError::format(
                        source,
                        format!("archive directory `{name}` is out of range"),
                    ));
                }
            } else {
                let start = data_start
                    .checked_add(offset as usize)
                    .ok_or_else(|| ContentError::format(source, "archive entry offset overflows"))?;
                let end = start
                    .checked_add(size as usize)
                    .ok_or_else(|| ContentError::format(source, "archive entry range overflows"))?;
                if end > data.len() {
                    return Err(ContentError::format(
                        source,
                        format!("archive entry `{name}` runs past the end of the archive"),
                    ));
                }
            }
            files.push(entry);
        }

        Ok(Self {
            source: source.to_path_buf(),
            data,
            data_start,
            dirs,
            files,
        })
    }

    /// The archive as the errors name it.
    pub fn source(&self) -> &Path {
        &self.source
    }

    /// The names of the files the archive holds, in table order.
    pub fn entries(&self) -> impl Iterator<Item = &str> {
        self.files
            .iter()
            .filter(|entry| !entry.is_directory())
            .map(|entry| entry.name.as_str())
    }

    /// The bytes of one file, named by its path in the tree.
    pub fn read(&self, name: &str) -> Result<&[u8], ContentError> {
        if let Some(entry) = self
            .files
            .iter()
            .find(|entry| !entry.is_directory() && entry.name == name)
        {
            return self.slice(entry);
        }
        let mut dir_index = 0_usize;
        let mut parts = name.split('/').peekable();
        loop {
            let part = parts.next().ok_or_else(|| self.missing(name))?;
            let dir = self
                .dirs
                .get(dir_index)
                .ok_or_else(|| self.missing(name))?;
            let found = (dir.first..dir.first + dir.count)
                .filter_map(|index| self.files.get(index))
                .find(|entry| entry.name == part)
                .ok_or_else(|| self.missing(name))?;
            if parts.peek().is_none() {
                if found.is_directory() {
                    return Err(self.missing(name));
                }
                return self.slice(found);
            }
            if !found.is_directory() {
                return Err(self.missing(name));
            }
            dir_index = if found.offset == NO_DIRECTORY {
                0
            } else {
                found.offset as usize
            };
            if dir_index >= self.dirs.len() {
                return Err(ContentError::format(
                    &self.source,
                    format!("archive `{name}` names a directory that is out of range"),
                ));
            }
        }
    }

    /// The bytes of the file whose id is `id`.
    ///
    /// Retail: `JKRArchive::findIdResource`
    /// (`tww/src/JSystem/JKernel/JKRArchivePri.cpp:169-186`) reads the entry
    /// at the id's own index first and then the table, and accepts an entry
    /// only when its id flag is set.
    pub fn read_id(&self, id: u16) -> Result<&[u8], ContentError> {
        if id == u16::MAX {
            return Err(self.missing_id(id));
        }
        if let Some(entry) = self.files.get(usize::from(id)) {
            if entry.id == id && entry.has_file_id() && !entry.is_directory() {
                return self.slice(entry);
            }
        }
        let found = self
            .files
            .iter()
            .find(|entry| entry.id == id && entry.has_file_id() && !entry.is_directory())
            .ok_or_else(|| self.missing_id(id))?;
        self.slice(found)
    }

    fn slice(&self, entry: &FileEntry) -> Result<&[u8], ContentError> {
        let start = self
            .data_start
            .checked_add(entry.offset as usize)
            .ok_or_else(|| ContentError::format(&self.source, "archive entry offset overflows"))?;
        let end = start
            .checked_add(entry.size as usize)
            .ok_or_else(|| ContentError::format(&self.source, "archive entry range overflows"))?;
        self.data.get(start..end).ok_or_else(|| {
            ContentError::format(
                &self.source,
                format!("archive entry `{}` runs past the end of the archive", entry.name),
            )
        })
    }

    fn missing(&self, name: &str) -> ContentError {
        ContentError::of(
            &self.source,
            ContentErrorKind::Missing,
            format!("archive has no entry `{name}`"),
        )
    }

    fn missing_id(&self, id: u16) -> ContentError {
        ContentError::of(
            &self.source,
            ContentErrorKind::Missing,
            format!("archive has no entry with id {id:#06x}"),
        )
    }
}

/// Whether these bytes start with the RARC magic.
pub fn is_rarc(bytes: &[u8]) -> bool {
    bytes.starts_with(&RARC_MAGIC[..])
}

/// Whether a name matches the hash the archive stores beside it.
///
/// Retail: `JKRArchive::CArcName::store`
/// (`tww/src/JSystem/JKernel/JKRArchivePri.cpp:189-203`) folds the name to
/// lower case and hashes it as `hash * 3 + character`, and
/// `JKRArchive::isSameName` (`:44-50`) compares the hash before the name. The
/// hash of the name as it is written is accepted as well, so an archive built
/// without the fold still reads.
fn hash_matches(name: &str, stored: u16) -> bool {
    let mut folded = 0_u32;
    let mut raw = 0_u32;
    for byte in name.bytes() {
        folded = (folded * 3 + u32::from(byte.to_ascii_lowercase())) & 0xFFFF;
        raw = (raw * 3 + u32::from(byte)) & 0xFFFF;
    }
    stored == folded as u16 || stored == raw as u16
}

fn read_u16(data: &[u8], at: usize, source: &Path) -> Result<u16, ContentError> {
    let end = at
        .checked_add(2)
        .ok_or_else(|| ContentError::format(source, "archive offset overflows"))?;
    let slice = data
        .get(at..end)
        .ok_or_else(|| ContentError::format(source, "archive is truncated"))?;
    Ok(u16::from_be_bytes([slice[0], slice[1]]))
}

fn read_u32(data: &[u8], at: usize, source: &Path) -> Result<u32, ContentError> {
    let end = at
        .checked_add(4)
        .ok_or_else(|| ContentError::format(source, "archive offset overflows"))?;
    let slice = data
        .get(at..end)
        .ok_or_else(|| ContentError::format(source, "archive is truncated"))?;
    Ok(u32::from_be_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The name hash of `a.bdl`, the value the archive stores beside it.
    const A_BDL_HASH: u16 = 10389;

    /// A one file archive: the root directory, one entry named `a.bdl` and the
    /// four bytes it holds. The layout is the retail one: a `SArcHeader`
    /// (`tww/include/JSystem/JKernel/JKRArchive.h:10-23`), the `SArcDataInfo`
    /// block that follows it, then the directory table, the file table and the
    /// string table, whose offsets are relative to the end of the header.
    fn archive_bytes() -> Vec<u8> {
        const DIR_TABLE: usize = 0x40;
        const FILE_TABLE: usize = 0x50;
        const STRING_TABLE: usize = 0x64;
        const DATA: usize = 0x6B;
        const TOTAL: usize = 0x6F;

        let mut data = vec![0u8; TOTAL];
        data[0x00..0x04].copy_from_slice(&RARC_MAGIC);
        data[0x04..0x08].copy_from_slice(&(TOTAL as u32).to_be_bytes());
        data[0x08..0x0C].copy_from_slice(&(HEADER_BYTES as u32).to_be_bytes());
        data[0x0C..0x10].copy_from_slice(&((DATA - HEADER_BYTES) as u32).to_be_bytes());
        data[0x10..0x14].copy_from_slice(&4u32.to_be_bytes());

        data[0x20..0x24].copy_from_slice(&1u32.to_be_bytes());
        data[0x24..0x28].copy_from_slice(&((DIR_TABLE - HEADER_BYTES) as u32).to_be_bytes());
        data[0x28..0x2C].copy_from_slice(&1u32.to_be_bytes());
        data[0x2C..0x30].copy_from_slice(&((FILE_TABLE - HEADER_BYTES) as u32).to_be_bytes());
        data[0x30..0x34].copy_from_slice(&7u32.to_be_bytes());
        data[0x34..0x38].copy_from_slice(&((STRING_TABLE - HEADER_BYTES) as u32).to_be_bytes());

        data[DIR_TABLE..DIR_TABLE + 4].copy_from_slice(b"ROOT");
        data[DIR_TABLE + 4..DIR_TABLE + 8].copy_from_slice(&0u32.to_be_bytes());
        data[DIR_TABLE + 8..DIR_TABLE + 10].copy_from_slice(&0u16.to_be_bytes());
        data[DIR_TABLE + 10..DIR_TABLE + 12].copy_from_slice(&1u16.to_be_bytes());
        data[DIR_TABLE + 12..DIR_TABLE + 16].copy_from_slice(&0u32.to_be_bytes());

        data[FILE_TABLE..FILE_TABLE + 2].copy_from_slice(&7u16.to_be_bytes());
        data[FILE_TABLE + 2..FILE_TABLE + 4].copy_from_slice(&A_BDL_HASH.to_be_bytes());
        data[FILE_TABLE + 4..FILE_TABLE + 8].copy_from_slice(&0x0100_0001u32.to_be_bytes());
        data[FILE_TABLE + 8..FILE_TABLE + 12].copy_from_slice(&0u32.to_be_bytes());
        data[FILE_TABLE + 12..FILE_TABLE + 16].copy_from_slice(&4u32.to_be_bytes());
        data[FILE_TABLE + 16..FILE_TABLE + 20].copy_from_slice(&0u32.to_be_bytes());

        data[STRING_TABLE..STRING_TABLE + 7].copy_from_slice(b"\0a.bdl\0");
        data[DATA..DATA + 4].copy_from_slice(b"WXYZ");
        data
    }

    #[test]
    fn an_archive_hands_out_the_entry_it_holds() {
        let source = Path::new("volume.arc");
        let archive = Archive::parse(archive_bytes(), source).expect("parse");
        assert_eq!(archive.entries().collect::<Vec<&str>>(), vec!["a.bdl"]);
        assert_eq!(archive.read("a.bdl").expect("entry"), b"WXYZ".as_slice());
        assert_eq!(archive.read_id(7).expect("id"), b"WXYZ".as_slice());
        assert_eq!(
            archive.read("gone.bdl").expect_err("missing").kind(),
            ContentErrorKind::Missing
        );
        assert_eq!(
            archive.read_id(9).expect_err("missing id").kind(),
            ContentErrorKind::Missing
        );
    }
}
