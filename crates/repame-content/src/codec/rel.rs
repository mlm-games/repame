//! The REL relocatable object: the sections a GameCube module carries.
//!
//! The header names the section table and the bss size, the table holds one
//! eight byte entry per section - the offset it starts at and its size - and
//! the first section's text starts right behind the table. The six sections
//! are the ones a module is made of: text, ctors, dtors, rodata, data and
//! bss.

use std::path::{Path, PathBuf};

use crate::codec::yaz0;
use crate::error::{ContentError, ContentErrorKind};

/// The sections a relocatable object holds, in table order.
const SECTION_COUNT: usize = 6;
/// One section table entry: the offset and the size.
const SECTION_ENTRY_BYTES: usize = 8;
/// The slot the text section sits in, the first slot being the null section.
const TEXT_SLOT: usize = 1;
/// The slot the bss section sits in.
const BSS_SLOT: usize = 6;
/// Where the header names the number of section table entries.
const ENTRY_COUNT_AT: usize = 0x0C;
/// Where the header names the section table.
const SECTION_TABLE_AT: usize = 0x10;
/// Where the header names the bss size.
const BSS_SIZE_AT: usize = 0x20;
/// The names of the sections, in table order.
const SECTION_NAMES: [&str; SECTION_COUNT] = ["text", "ctors", "dtors", "rodata", "data", "bss"];

/// One section of a relocatable object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelSection {
    pub name: &'static str,
    pub offset: usize,
    pub size: usize,
}

/// A relocatable object: its sections over the bytes it was parsed from.
#[derive(Debug)]
pub struct Rel {
    source: PathBuf,
    data: Vec<u8>,
    sections: [Option<RelSection>; SECTION_COUNT],
}

impl Rel {
    /// Open a relocatable object, expanding it first when it is a Yaz0
    /// compressed file.
    pub fn open(data: Vec<u8>, source: &Path) -> Result<Self, ContentError> {
        let data = yaz0::expand(data, source)?;
        Self::parse(data, source)
    }

    /// Parse a relocatable object. `source` only names the object in errors.
    pub fn parse(data: Vec<u8>, source: &Path) -> Result<Self, ContentError> {
        let entries = read_u32(&data, ENTRY_COUNT_AT, source)? as usize;
        let table = read_u32(&data, SECTION_TABLE_AT, source)? as usize;
        if entries <= BSS_SLOT {
            return Err(ContentError::format(
                source,
                format!("relocatable object names {entries} sections, expected more than {BSS_SLOT}"),
            ));
        }
        let text_start = table
            .checked_add(
                entries
                    .checked_mul(SECTION_ENTRY_BYTES)
                    .ok_or_else(|| ContentError::format(source, "section table overflows"))?,
            )
            .ok_or_else(|| ContentError::format(source, "section table overflows"))?;
        if text_start > data.len() {
            return Err(ContentError::format(
                source,
                "section table runs past the end of the object",
            ));
        }
        let bss_size = read_u32(&data, BSS_SIZE_AT, source)? as usize;
        let mut sections = [None; SECTION_COUNT];
        let mut filled = text_start;
        for slot in TEXT_SLOT..=BSS_SLOT {
            let at = table
                .checked_add(slot * SECTION_ENTRY_BYTES)
                .ok_or_else(|| ContentError::format(source, "section table overflows"))?;
            let stored = read_u32(&data, at, source)? as usize;
            let size = read_u32(&data, at + 4, source)? as usize;
            if slot == BSS_SLOT && (stored != 0 || size != bss_size) {
                return Err(ContentError::format(
                    source,
                    "bss section disagrees with the header",
                ));
            }
            if size == 0 {
                continue;
            }
            let offset = if slot == TEXT_SLOT {
                if stored != text_start && stored != text_start + 1 {
                    return Err(ContentError::format(source, "text section is misplaced"));
                }
                text_start
            } else if slot == BSS_SLOT {
                0
            } else {
                if stored < filled {
                    return Err(ContentError::format(source, "sections are out of order"));
                }
                stored
            };
            if slot != BSS_SLOT {
                let end = offset
                    .checked_add(size)
                    .ok_or_else(|| ContentError::format(source, "section range overflows"))?;
                if end > data.len() {
                    return Err(ContentError::format(
                        source,
                        "section runs past the end of the object",
                    ));
                }
            }
            filled = offset.saturating_add(size);
            sections[slot - TEXT_SLOT] = Some(RelSection {
                name: SECTION_NAMES[slot - TEXT_SLOT],
                offset,
                size,
            });
        }
        Ok(Self {
            source: source.to_path_buf(),
            data,
            sections,
        })
    }

    /// The object as the errors name it.
    pub fn source(&self) -> &Path {
        &self.source
    }

    /// The section named `name`, when the object carries one.
    pub fn section(&self, name: &str) -> Option<&RelSection> {
        SECTION_NAMES
            .iter()
            .position(|section| *section == name)
            .and_then(|index| self.sections[index].as_ref())
    }

    /// The bytes of a symbol: `size` bytes at `offset` inside `section`.
    pub fn symbol(&self, section: &str, offset: usize, size: usize) -> Result<&[u8], ContentError> {
        let found = self.section(section).ok_or_else(|| self.missing(section))?;
        if found.offset == 0 {
            return Err(ContentError::format(
                &self.source,
                format!("section `{section}` holds no file data"),
            ));
        }
        let end = offset
            .checked_add(size)
            .ok_or_else(|| ContentError::format(&self.source, "symbol range overflows"))?;
        if end > found.size {
            return Err(ContentError::format(
                &self.source,
                format!("symbol runs past the end of section `{section}`"),
            ));
        }
        let start = found
            .offset
            .checked_add(offset)
            .ok_or_else(|| ContentError::format(&self.source, "symbol offset overflows"))?;
        let stop = start
            .checked_add(size)
            .ok_or_else(|| ContentError::format(&self.source, "symbol range overflows"))?;
        self.data.get(start..stop).ok_or_else(|| {
            ContentError::format(
                &self.source,
                format!("symbol runs past the end of the object"),
            )
        })
    }

    /// The `count` floats a symbol holds, read big endian.
    pub fn f32s(&self, section: &str, offset: usize, count: usize) -> Result<Vec<f32>, ContentError> {
        let size = count
            .checked_mul(4)
            .ok_or_else(|| ContentError::format(&self.source, "symbol range overflows"))?;
        let bytes = self.symbol(section, offset, size)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|word| f32::from_bits(u32::from_be_bytes([word[0], word[1], word[2], word[3]])))
            .collect())
    }

    fn missing(&self, section: &str) -> ContentError {
        ContentError::of(
            &self.source,
            ContentErrorKind::Missing,
            format!("object has no section `{section}`"),
        )
    }
}

fn read_u32(data: &[u8], at: usize, source: &Path) -> Result<u32, ContentError> {
    let end = at
        .checked_add(4)
        .ok_or_else(|| ContentError::format(source, "offset overflows"))?;
    let slice = data
        .get(at..end)
        .ok_or_else(|| ContentError::format(source, "object is truncated"))?;
    Ok(u32::from_be_bytes([slice[0], slice[1], slice[2], slice[3]]))
}
