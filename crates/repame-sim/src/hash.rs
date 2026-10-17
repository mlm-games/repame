//! The hash a determinism check folds: FNV-1a 64.
//!
//! FNV-1a 64 is what the games' hand-rolled session digests already used, so
//! the engine and a game agree on a digest and nothing new is pulled in for it.
//! It is a change detector, not a cryptographic hash: state nobody folds does
//! not move it, and a crafted collision is possible. Use it to find *where* two
//! runs differ, not to prove they cannot.
//!
//! Determinism rule: every value is folded as its bytes, little-endian, one byte
//! at a time, in the order the caller wrote it. `usize`/`isize` fold as
//! `u64`/`i64`, so a 32-bit wasm build folds the same bytes as a 64-bit desktop
//! build for the same value, and a float folds its `to_bits`, so the digest is
//! bit-exact rather than approximate (`0.0` and `-0.0` differ, NaN payloads
//! included). Strings are length-prefixed, so two short strings cannot fold
//! like one long one. Unlike `std`'s `DefaultHasher`, nothing here is seeded
//! per process: a digest computed in one run means the same thing in the next.

use std::hash::Hasher;

/// FNV-1a 64 offset basis: the digest of no bytes at all.
const OFFSET: u64 = 0xCBF2_9CE4_8422_2325;

/// FNV-1a 64 prime.
const PRIME: u64 = 0x0000_0100_0000_01B3;

/// FNV-1a 64 over `bytes`.
///
/// `const`, so a name can fold to a number at compile time; [`Hash64`] folds the
/// same bytes when it is written to.
pub const fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut state = OFFSET;
    let mut index = 0;
    while index < bytes.len() {
        state ^= bytes[index] as u64;
        state = state.wrapping_mul(PRIME);
        index += 1;
    }
    state
}

/// A streaming FNV-1a 64 digest: write values in a fixed order, read
/// [`digest`](Self::digest).
///
/// It is also a [`Hasher`], so a type whose `Hash` implementation is yours can be
/// folded through `Hash::hash`. The two paths agree on every primitive: this type
/// overrides every integer writer, so `u32` folds as four little-endian bytes
/// either way. They differ on `str`, which the harness's own [`str`](Self::str)
/// length-prefixes while `Hash`'s stable default writes the bytes and a `0xff`;
/// both are pinned, pick one and stay with it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hash64 {
    state: u64,
}

impl Hash64 {
    /// A digest of nothing: the FNV-1a offset basis.
    pub const fn new() -> Self {
        Self { state: OFFSET }
    }

    /// What has been written so far.
    ///
    /// Writing more changes it, so read it where the stream ends.
    pub const fn digest(&self) -> u64 {
        self.state
    }

    /// Raw bytes, as given: no length prefix, no padding.
    pub fn bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.state ^= *byte as u64;
            self.state = self.state.wrapping_mul(PRIME);
        }
    }

    /// A string as its length then its UTF-8 bytes.
    pub fn str(&mut self, value: &str) {
        self.u64(value.len() as u64);
        self.bytes(value.as_bytes());
    }

    /// One byte.
    pub fn u8(&mut self, value: u8) {
        self.bytes(&[value]);
    }

    /// A 16-bit value, little-endian.
    pub fn u16(&mut self, value: u16) {
        self.bytes(&value.to_le_bytes());
    }

    /// A 32-bit value, little-endian.
    pub fn u32(&mut self, value: u32) {
        self.bytes(&value.to_le_bytes());
    }

    /// A 64-bit value, little-endian.
    pub fn u64(&mut self, value: u64) {
        self.bytes(&value.to_le_bytes());
    }

    /// A 128-bit value, little-endian.
    pub fn u128(&mut self, value: u128) {
        self.bytes(&value.to_le_bytes());
    }

    /// A signed value as its two's-complement bits.
    pub fn i8(&mut self, value: i8) {
        self.u8(value as u8);
    }

    /// A signed value as its two's-complement bits.
    pub fn i16(&mut self, value: i16) {
        self.u16(value as u16);
    }

    /// A signed value as its two's-complement bits.
    pub fn i32(&mut self, value: i32) {
        self.u32(value as u32);
    }

    /// A signed value as its two's-complement bits.
    pub fn i64(&mut self, value: i64) {
        self.u64(value as u64);
    }

    /// A length or index, folded as `u64` so the width is not target-dependent.
    pub fn usize(&mut self, value: usize) {
        self.u64(value as u64);
    }

    /// A length or index, folded as `i64` so the width is not target-dependent.
    pub fn isize(&mut self, value: isize) {
        self.i64(value as i64);
    }

    /// A float by its bits, so the fold is exact.
    pub fn f32(&mut self, value: f32) {
        self.u32(value.to_bits());
    }

    /// A float by its bits, so the fold is exact.
    pub fn f64(&mut self, value: f64) {
        self.u64(value.to_bits());
    }

    /// A bool as one byte, `0` or `1`.
    pub fn bool(&mut self, value: bool) {
        self.u8(value as u8);
    }
}

impl Default for Hash64 {
    fn default() -> Self {
        Self::new()
    }
}

impl Hasher for Hash64 {
    fn finish(&self) -> u64 {
        self.digest()
    }

    fn write(&mut self, bytes: &[u8]) {
        self.bytes(bytes);
    }

    fn write_u8(&mut self, value: u8) {
        self.u8(value);
    }

    fn write_u16(&mut self, value: u16) {
        self.u16(value);
    }

    fn write_u32(&mut self, value: u32) {
        self.u32(value);
    }

    fn write_u64(&mut self, value: u64) {
        self.u64(value);
    }

    fn write_u128(&mut self, value: u128) {
        self.u128(value);
    }

    fn write_usize(&mut self, value: usize) {
        self.usize(value);
    }

    fn write_i8(&mut self, value: i8) {
        self.i8(value);
    }

    fn write_i16(&mut self, value: i16) {
        self.i16(value);
    }

    fn write_i32(&mut self, value: i32) {
        self.i32(value);
    }

    fn write_i64(&mut self, value: i64) {
        self.i64(value);
    }

    fn write_i128(&mut self, value: i128) {
        self.u128(value as u128);
    }

    fn write_isize(&mut self, value: isize) {
        self.isize(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_published_fnv1a_vectors_hold() {
        // The FNV reference vectors, and the empty digest that is the basis.
        assert_eq!(fnv1a64(b""), 0xCBF2_9CE4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xAF63_DC4C_8601_EC8C);
        assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_F739_67E8);
        // The one the engine itself depends on: `WorldSeed::named("loot")`.
        assert_eq!(fnv1a64(b"loot"), 0xCDEC_E8AD_70D5_84B7);
    }

    #[test]
    fn streaming_agrees_with_the_one_shot_hash() {
        for bytes in [b"".as_slice(), b"a", b"loot", b"repame-sim"] {
            let mut hash = Hash64::new();
            hash.bytes(bytes);
            assert_eq!(hash.digest(), fnv1a64(bytes), "{bytes:?}");
        }
    }

    #[test]
    fn a_written_integer_equals_its_little_endian_bytes() {
        let mut by_value = Hash64::new();
        by_value.u32(0x0403_0201);
        by_value.u8(0x05);
        let mut by_bytes = Hash64::new();
        by_bytes.bytes(&[0x01, 0x02, 0x03, 0x04, 0x05]);
        assert_eq!(by_value.digest(), by_bytes.digest());

        let mut wide = Hash64::new();
        wide.u64(u64::MAX);
        let mut wide_bytes = Hash64::new();
        wide_bytes.bytes(&[0xFF; 8]);
        assert_eq!(wide.digest(), wide_bytes.digest());
    }

    #[test]
    fn lengths_prefix_and_signs_keep_their_bits() {
        // "ab" must not fold like "a" followed by "b" with no separator.
        let mut joined = Hash64::new();
        joined.str("ab");
        let mut split = Hash64::new();
        split.str("a");
        split.str("b");
        assert_ne!(joined.digest(), split.digest());

        let mut signed = Hash64::new();
        signed.i32(-1);
        let mut unsigned = Hash64::new();
        unsigned.u32(u32::MAX);
        assert_eq!(signed.digest(), unsigned.digest());

        let mut plus_zero = Hash64::new();
        plus_zero.f32(0.0);
        let mut minus_zero = Hash64::new();
        minus_zero.f32(-0.0);
        assert_ne!(plus_zero.digest(), minus_zero.digest(), "bits, not values");
    }

    #[test]
    fn the_hasher_path_matches_the_write_methods() {
        use std::hash::Hash;

        let mut through_hash = Hash64::new();
        7u32.hash(&mut through_hash);
        true.hash(&mut through_hash);
        (-3i64).hash(&mut through_hash);

        let mut through_writes = Hash64::new();
        through_writes.u32(7);
        through_writes.bool(true);
        through_writes.i64(-3);
        assert_eq!(through_hash.digest(), through_writes.digest());
    }

    #[test]
    fn a_digest_is_the_same_in_the_next_run() {
        // No per-process seed: two hashers folded the same way agree, and a
        // std `DefaultHasher` comparison is deliberately absent here.
        let mut first = Hash64::new();
        first.str("repame-sim");
        let mut second = Hash64::new();
        second.str("repame-sim");
        assert_eq!(first.digest(), second.digest());
        assert_eq!(first, second);
        assert_eq!(Hash64::default().digest(), fnv1a64(b""));
    }
}
