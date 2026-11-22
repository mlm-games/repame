//! Yaz0: the compression the retail disc ripper expands before it mounts a
//! volume.
//!
//! The stream is a sixteen byte header - the magic, the expanded size and two
//! reserved words - followed by groups of one flag byte and eight payloads. A
//! set flag bit is a literal byte, a clear flag bit is a back reference whose
//! distance and length are read from two bytes.
//!
//! Retail: `tww/src/JSystem/JKernel/JKRDecomp.cpp:206-269`
//! (`JKRDecomp::decodeSZS`) is the expansion the archive loader runs, and
//! `tww/src/JSystem/JKernel/JKRDvdRipper.cpp:269-382`
//! (`decompSZS_subroutine`) is the same expansion streamed off the disc. The
//! header is `SYaz0Header`
//! (`tww/include/JSystem/JKernel/JKRDvdRipper.h:13-16`) and the expanded size
//! is read from it big endian
//! (`tww/include/JSystem/JKernel/JKRDecomp.h:69-71`).

use std::path::Path;

use crate::error::ContentError;

/// `"Yaz0"` in byte order.
pub const YAZ0_MAGIC: [u8; 4] = *b"Yaz0";

/// The header before the compressed stream.
const HEADER_BYTES: usize = 0x10;

/// Whether these bytes start with the Yaz0 magic.
///
/// Retail: `JKRDecomp::checkCompressed`
/// (`tww/src/JSystem/JKernel/JKRDecomp.cpp:271-284`) and the ripper's own
/// check (`tww/src/JSystem/JKernel/JKRDvdRipper.cpp:276-278`) both test the
/// four bytes for `Yaz0`.
pub fn is_yaz0(bytes: &[u8]) -> bool {
    bytes.starts_with(&YAZ0_MAGIC[..])
}

/// Expand the bytes when they are a Yaz0 stream and leave them as they are
/// otherwise, which is what the retail ripper does with every file it reads
/// (`tww/src/JSystem/JKernel/JKRDvdRipper.cpp:131-159`).
pub fn expand(bytes: Vec<u8>, source: &Path) -> Result<Vec<u8>, ContentError> {
    if is_yaz0(&bytes) {
        decode(&bytes, source)
    } else {
        Ok(bytes)
    }
}

/// Expand a Yaz0 stream into the bytes it holds. `source` only names the
/// stream in errors.
pub fn decode(bytes: &[u8], source: &Path) -> Result<Vec<u8>, ContentError> {
    if bytes.len() < HEADER_BYTES {
        return Err(ContentError::format(
            source,
            format!(
                "Yaz0 stream is {} bytes, shorter than its header",
                bytes.len()
            ),
        ));
    }
    if !is_yaz0(bytes) {
        return Err(ContentError::format(source, "not a Yaz0 stream"));
    }
    let size = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let mut out = Vec::new();
    out.try_reserve_exact(size)
        .map_err(|_| ContentError::format(source, "Yaz0 expanded size does not fit"))?;
    let mut cursor = HEADER_BYTES;
    while out.len() < size {
        let flags = *bytes
            .get(cursor)
            .ok_or_else(|| ContentError::format(source, "Yaz0 stream is truncated"))?;
        cursor += 1;
        for bit in (0..8).rev() {
            if out.len() >= size {
                break;
            }
            if flags & (1 << bit) != 0 {
                let byte = *bytes
                    .get(cursor)
                    .ok_or_else(|| ContentError::format(source, "Yaz0 stream is truncated"))?;
                cursor += 1;
                out.push(byte);
            } else {
                let first = *bytes
                    .get(cursor)
                    .ok_or_else(|| ContentError::format(source, "Yaz0 stream is truncated"))?;
                let second = *bytes
                    .get(cursor + 1)
                    .ok_or_else(|| ContentError::format(source, "Yaz0 stream is truncated"))?;
                cursor += 2;
                let distance = (usize::from(first & 0x0F) << 8) | usize::from(second);
                let mut length = usize::from(first >> 4);
                if length == 0 {
                    length =
                        usize::from(*bytes.get(cursor).ok_or_else(|| {
                            ContentError::format(source, "Yaz0 stream is truncated")
                        })?) + 0x12;
                    cursor += 1;
                } else {
                    length += 2;
                }
                let start = out.len().checked_sub(distance + 1).ok_or_else(|| {
                    ContentError::format(
                        source,
                        "Yaz0 back reference points before the stream start",
                    )
                })?;
                for step in 0..length {
                    let byte = out.get(start + step).copied().ok_or_else(|| {
                        ContentError::format(source, "Yaz0 back reference runs past the stream end")
                    })?;
                    out.push(byte);
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ContentErrorKind;

    /// A stream that expands to `abcdabcd`: four literal bytes and a back
    /// reference that copies them again.
    fn stream() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&YAZ0_MAGIC);
        bytes.extend_from_slice(&8u32.to_be_bytes());
        bytes.extend_from_slice(&[0; 8]);
        bytes.push(0xF0);
        bytes.extend_from_slice(b"abcd");
        bytes.push(0x20);
        bytes.push(0x03);
        bytes
    }

    #[test]
    fn a_yaz0_stream_expands_to_the_bytes_it_holds() {
        let source = Path::new("stream.yaz0");
        assert_eq!(
            decode(&stream(), source).expect("expand"),
            b"abcdabcd".to_vec()
        );
        assert!(is_yaz0(&stream()));
        assert!(!is_yaz0(b"RARC"));
    }

    #[test]
    fn a_stream_that_is_not_yaz0_is_refused() {
        let source = Path::new("stream.yaz0");
        let error = decode(b"RARC", source).expect_err("bad magic");
        assert_eq!(error.kind(), ContentErrorKind::Format);
    }
}
