//! Hand-rolled binary codec: the documented, versioned repame wire format.
//!
//! # Encoding rules
//!
//! Every scalar is little-endian and fixed width, so a reader can be written
//! against this document alone and stays byte-compatible across targets
//! (unlike `usize`/`isize`, which are never encoded).
//!
//! | Rust type | bytes | layout |
//! |---|---|---|
//! | `u8`, `bool` | 1 | value; bool is `0` or `1`, anything else is an error |
//! | `u16` | 2 | little-endian |
//! | `u32`, `i32` | 4 | little-endian |
//! | `u64`, `i64` | 8 | little-endian |
//! | `f32` | 4 | `to_bits()` little-endian; NaN payload bits survive |
//! | `f64` | 8 | `to_bits()` little-endian |
//! | `Vec<u8>`, `&[u8]`, `&str`, `String` | 4 + n | `u32` byte length, then n bytes; strings are UTF-8 |
//! | `Vec<T>` | 4 + n | `u32` element count, then each element |
//! | `BTreeMap<K, V>` | 4 + n | `u32` entry count in key order, then key/value pairs |
//! | `Option<T>` | 1 + n | `0` = none, `1` = some value |
//!
//! Collections are capped at [`MAX_COLLECTION_LEN`] elements on decode and at
//! `u32::MAX` bytes on encode; exceeding either is a [`CodecError::Oversize`].
//! Because `Encode` is fallible, no length is ever silently truncated. The
//! negotiated `WIRE_VERSION` byte is not written by the codec itself; it leads
//! every packet and frame (see [`crate::protocol`]), so a version mismatch is
//! caught before any field is parsed.

use std::collections::BTreeMap;

use crate::error::{CodecError, NetError};

/// Same wire format as [`crate::WIRE_VERSION`]; re-exported for discoverability.
pub use crate::WIRE_VERSION;

/// Largest collection (element or byte count) a decode accepts.
pub const MAX_COLLECTION_LEN: u32 = 1 << 24;

/// Serialize `self` into the wire format.
pub trait Encode {
    /// Append `self` to `out`. Never partially writes on error.
    fn encode(&self, out: &mut Writer) -> Result<(), CodecError>;
}

/// Deserialize a value from the wire format.
pub trait Decode: Sized {
    /// Read one value from `input`.
    fn decode(input: &mut Reader<'_>) -> Result<Self, CodecError>;
}

/// Growable wire buffer.
#[derive(Debug, Default, Clone)]
pub struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    /// Empty buffer.
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Empty buffer with room for `capacity` bytes.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buf: Vec::with_capacity(capacity),
        }
    }

    /// Bytes written so far.
    pub fn as_slice(&self) -> &[u8] {
        &self.buf
    }

    /// Bytes written so far.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Whether nothing has been written.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Take the buffer.
    pub fn finish(self) -> Vec<u8> {
        self.buf
    }

    /// One byte.
    pub fn put_u8(&mut self, value: u8) {
        self.buf.push(value);
    }

    /// `0` or `1`.
    pub fn put_bool(&mut self, value: bool) {
        self.buf.push(u8::from(value));
    }

    /// Little-endian `u16`.
    pub fn put_u16(&mut self, value: u16) {
        self.buf.extend_from_slice(&value.to_le_bytes());
    }

    /// Little-endian `u32`.
    pub fn put_u32(&mut self, value: u32) {
        self.buf.extend_from_slice(&value.to_le_bytes());
    }

    /// Little-endian `u64`.
    pub fn put_u64(&mut self, value: u64) {
        self.buf.extend_from_slice(&value.to_le_bytes());
    }

    /// Little-endian two's complement `i32`.
    pub fn put_i32(&mut self, value: i32) {
        self.buf.extend_from_slice(&value.to_le_bytes());
    }

    /// Little-endian two's complement `i64`.
    pub fn put_i64(&mut self, value: i64) {
        self.buf.extend_from_slice(&value.to_le_bytes());
    }

    /// Bit pattern, little-endian; NaN payloads survive round-trips.
    pub fn put_f32(&mut self, value: f32) {
        self.buf.extend_from_slice(&value.to_bits().to_le_bytes());
    }

    /// Bit pattern, little-endian.
    pub fn put_f64(&mut self, value: f64) {
        self.buf.extend_from_slice(&value.to_bits().to_le_bytes());
    }

    /// `u32` length prefix then the bytes.
    pub fn put_bytes(&mut self, bytes: &[u8]) -> Result<(), CodecError> {
        let len = u32::try_from(bytes.len()).map_err(|_| CodecError::Oversize {
            field: "bytes",
            size: bytes.len(),
            max: u32::MAX as usize,
        })?;
        self.put_u32(len);
        self.buf.extend_from_slice(bytes);
        Ok(())
    }

    /// Length-prefixed UTF-8 bytes.
    pub fn put_str(&mut self, value: &str) -> Result<(), CodecError> {
        self.put_bytes(value.as_bytes())
    }
}

/// Cursor over wire bytes.
#[derive(Debug, Clone)]
pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    /// Read `buf` from position `0`.
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// Bytes not yet read.
    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// Whether every byte has been read.
    pub fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    /// Consume `n` bytes.
    pub fn take(&mut self, n: usize) -> Result<&'a [u8], CodecError> {
        if self.remaining() < n {
            return Err(CodecError::UnexpectedEof {
                needed: n,
                remaining: self.remaining(),
            });
        }
        let out = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    fn take_array<const N: usize>(&mut self) -> Result<[u8; N], CodecError> {
        let mut out = [0u8; N];
        out.copy_from_slice(self.take(N)?);
        Ok(out)
    }

    /// One byte.
    pub fn get_u8(&mut self) -> Result<u8, CodecError> {
        Ok(self.take(1)?[0])
    }

    /// One bool; any byte but `0`/`1` is an error.
    pub fn get_bool(&mut self) -> Result<bool, CodecError> {
        match self.get_u8()? {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(CodecError::InvalidBool(other)),
        }
    }

    /// Little-endian `u16`.
    pub fn get_u16(&mut self) -> Result<u16, CodecError> {
        Ok(u16::from_le_bytes(self.take_array()?))
    }

    /// Little-endian `u32`.
    pub fn get_u32(&mut self) -> Result<u32, CodecError> {
        Ok(u32::from_le_bytes(self.take_array()?))
    }

    /// Little-endian `u64`.
    pub fn get_u64(&mut self) -> Result<u64, CodecError> {
        Ok(u64::from_le_bytes(self.take_array()?))
    }

    /// Little-endian `i32`.
    pub fn get_i32(&mut self) -> Result<i32, CodecError> {
        Ok(i32::from_le_bytes(self.take_array()?))
    }

    /// Little-endian `i64`.
    pub fn get_i64(&mut self) -> Result<i64, CodecError> {
        Ok(i64::from_le_bytes(self.take_array()?))
    }

    /// Bit-preserving `f32`.
    pub fn get_f32(&mut self) -> Result<f32, CodecError> {
        Ok(f32::from_bits(u32::from_le_bytes(self.take_array()?)))
    }

    /// Bit-preserving `f64`.
    pub fn get_f64(&mut self) -> Result<f64, CodecError> {
        Ok(f64::from_bits(u64::from_le_bytes(self.take_array()?)))
    }

    /// Length-prefixed bytes; the slice borrows the input buffer.
    pub fn get_bytes(&mut self) -> Result<&'a [u8], CodecError> {
        let len = self.get_u32()? as usize;
        self.take(len)
    }

    /// Length-prefixed UTF-8 string.
    pub fn get_string(&mut self) -> Result<String, CodecError> {
        let bytes = self.get_bytes()?;
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| CodecError::InvalidUtf8)
    }

    /// Collection length guard shared by `Vec`/`BTreeMap` decode.
    pub fn get_len(&mut self, field: &'static str) -> Result<usize, CodecError> {
        let len = self.get_u32()?;
        if len > MAX_COLLECTION_LEN {
            return Err(CodecError::Oversize {
                field,
                size: len as usize,
                max: MAX_COLLECTION_LEN as usize,
            });
        }
        Ok(len as usize)
    }

    /// Fail if any byte is left, so a decoder cannot ignore trailing data.
    pub fn expect_end(&self) -> Result<(), CodecError> {
        if self.is_empty() {
            Ok(())
        } else {
            Err(CodecError::TrailingBytes(self.remaining()))
        }
    }
}

impl Encode for u8 {
    fn encode(&self, out: &mut Writer) -> Result<(), CodecError> {
        out.put_u8(*self);
        Ok(())
    }
}

impl Encode for bool {
    fn encode(&self, out: &mut Writer) -> Result<(), CodecError> {
        out.put_bool(*self);
        Ok(())
    }
}

impl Encode for u16 {
    fn encode(&self, out: &mut Writer) -> Result<(), CodecError> {
        out.put_u16(*self);
        Ok(())
    }
}

impl Encode for u32 {
    fn encode(&self, out: &mut Writer) -> Result<(), CodecError> {
        out.put_u32(*self);
        Ok(())
    }
}

impl Encode for u64 {
    fn encode(&self, out: &mut Writer) -> Result<(), CodecError> {
        out.put_u64(*self);
        Ok(())
    }
}

impl Encode for i32 {
    fn encode(&self, out: &mut Writer) -> Result<(), CodecError> {
        out.put_i32(*self);
        Ok(())
    }
}

impl Encode for i64 {
    fn encode(&self, out: &mut Writer) -> Result<(), CodecError> {
        out.put_i64(*self);
        Ok(())
    }
}

impl Encode for f32 {
    fn encode(&self, out: &mut Writer) -> Result<(), CodecError> {
        out.put_f32(*self);
        Ok(())
    }
}

impl Encode for f64 {
    fn encode(&self, out: &mut Writer) -> Result<(), CodecError> {
        out.put_f64(*self);
        Ok(())
    }
}

impl Encode for str {
    fn encode(&self, out: &mut Writer) -> Result<(), CodecError> {
        out.put_str(self)
    }
}

impl Encode for String {
    fn encode(&self, out: &mut Writer) -> Result<(), CodecError> {
        out.put_str(self)
    }
}

impl<T: Encode> Encode for Vec<T> {
    fn encode(&self, out: &mut Writer) -> Result<(), CodecError> {
        let len = u32::try_from(self.len()).map_err(|_| CodecError::Oversize {
            field: "Vec",
            size: self.len(),
            max: u32::MAX as usize,
        })?;
        out.put_u32(len);
        for item in self {
            item.encode(out)?;
        }
        Ok(())
    }
}

impl<T: Decode> Decode for Vec<T> {
    fn decode(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        let len = input.get_len("Vec")?;
        let mut out = Vec::new();
        for _ in 0..len {
            out.push(T::decode(input)?);
        }
        Ok(out)
    }
}

impl<K: Encode, V: Encode> Encode for BTreeMap<K, V> {
    fn encode(&self, out: &mut Writer) -> Result<(), CodecError> {
        let len = u32::try_from(self.len()).map_err(|_| CodecError::Oversize {
            field: "BTreeMap",
            size: self.len(),
            max: u32::MAX as usize,
        })?;
        out.put_u32(len);
        for (key, value) in self {
            key.encode(out)?;
            value.encode(out)?;
        }
        Ok(())
    }
}

impl<K: Decode + Ord, V: Decode> Decode for BTreeMap<K, V> {
    fn decode(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        let len = input.get_len("BTreeMap")?;
        let mut out = BTreeMap::new();
        for _ in 0..len {
            let key = K::decode(input)?;
            let value = V::decode(input)?;
            out.insert(key, value);
        }
        Ok(out)
    }
}

impl<T: Encode> Encode for Option<T> {
    fn encode(&self, out: &mut Writer) -> Result<(), CodecError> {
        match self {
            None => {
                out.put_u8(0);
                Ok(())
            }
            Some(value) => {
                out.put_u8(1);
                value.encode(out)
            }
        }
    }
}

impl<T: Decode> Decode for Option<T> {
    fn decode(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        match input.get_u8()? {
            0 => Ok(None),
            1 => Ok(Some(T::decode(input)?)),
            tag => Err(CodecError::UnknownTag(tag)),
        }
    }
}

impl Decode for u8 {
    fn decode(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        input.get_u8()
    }
}

impl Decode for bool {
    fn decode(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        input.get_bool()
    }
}

impl Decode for u16 {
    fn decode(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        input.get_u16()
    }
}

impl Decode for u32 {
    fn decode(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        input.get_u32()
    }
}

impl Decode for u64 {
    fn decode(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        input.get_u64()
    }
}

impl Decode for i32 {
    fn decode(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        input.get_i32()
    }
}

impl Decode for i64 {
    fn decode(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        input.get_i64()
    }
}

impl Decode for f32 {
    fn decode(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        input.get_f32()
    }
}

impl Decode for f64 {
    fn decode(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        input.get_f64()
    }
}

impl Decode for String {
    fn decode(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        input.get_string()
    }
}

/// Encode one value into a fresh buffer.
pub fn encode_to_vec<T: Encode + ?Sized>(value: &T) -> Result<Vec<u8>, NetError> {
    let mut writer = Writer::new();
    value.encode(&mut writer)?;
    Ok(writer.finish())
}

/// Decode one value, rejecting trailing bytes.
pub fn decode_from_slice<T: Decode>(bytes: &[u8]) -> Result<T, NetError> {
    let mut reader = Reader::new(bytes);
    let value = T::decode(&mut reader)?;
    reader.expect_end()?;
    Ok(value)
}
