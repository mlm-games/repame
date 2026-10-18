//! Simulation randomness a replay can trust, with no dependency behind it.
//!
//! [`SimRng`] is xoshiro256\*\* (Blackman & Vigna, 2018) seeded through
//! SplitMix64, and both algorithms are defined *in this file* as wrapping
//! `u64` arithmetic: no `rand`, no `SmallRng`, no libm, no target-dependent
//! rounding. Upgrading a dependency can therefore never move a draw, which
//! matters because `SmallRng` is documented as not stable across `rand`
//! releases while a replay is exactly the claim the crate makes.
//!
//! The 4-word state is `serde`-serialisable, so a save carries the stream
//! position; a save plus an intent tape (see [`crate::tape`]) re-runs the game.
//!
//! Two ways to get a stream:
//!
//! - [`SimRng::seeded`] for the one stream a game owns outright, inserted as a
//!   resource.
//! - [`WorldSeed`] to derive a stream per system or per entity. Deriving is a
//!   pure function of `(seed, tag)`, so adding a system does not shift the
//!   streams of the others, and the same tag always yields the same stream.
//!
//! Sampling method names follow the `rand` API the games already call
//! (`random_range`, `random`, `random_bool`), so porting a draw site is a
//! change of `use` line rather than a rewrite. The floating-point and integer
//! construction is pinned here, once, in the trait's provided methods: there is
//! no second implementation of `unit_f32` for a different target to disagree
//! with.

use core::ops::{Range, RangeInclusive};

use bevy_ecs::prelude::*;
use serde::{Deserialize, Serialize};

/// SplitMix64 increment: 2^64 divided by the golden ratio.
const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
/// SplitMix64 finalizer multiplier 1.
const SPLITMIX_MUL1: u64 = 0xBF58_476D_1CE4_E5B9;
/// SplitMix64 finalizer multiplier 2.
const SPLITMIX_MUL2: u64 = 0x94D0_49BB_1331_11EB;
/// Domain separator for [`WorldSeed`] tags, so tag 0 is not the bare stream.
const TAG_SALT: u64 = 0xD1B5_4A32_D192_ED03;

/// SplitMix64's finalizer: a bijective avalanche mix of one word.
const fn finalize(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(SPLITMIX_MUL1);
    z = (z ^ (z >> 27)).wrapping_mul(SPLITMIX_MUL2);
    z ^ (z >> 31)
}

/// One seed expanded into the xoshiro256\*\* state: four SplitMix64 outputs.
///
/// The four counters `seed + k * GAMMA`, `k` in `1..=4`, are distinct, so at
/// most one of them is zero, and `finalize` reaches zero only at zero. The four
/// words are therefore never all zero: the xoshiro fixed point is out of reach
/// from any seed, and no guard is needed.
const fn expand(seed: u64) -> [u64; 4] {
    let mut counter = seed;
    let mut state = [0; 4];
    let mut word = 0;
    while word < 4 {
        counter = counter.wrapping_add(SPLITMIX_GAMMA);
        state[word] = finalize(counter);
        word += 1;
    }
    state
}

/// The sim's random stream: xoshiro256\*\* holding four words of state.
///
/// Insert one as a resource and every draw the sim makes is reproducible from
/// the state it holds. `Copy`, so a system can take a snapshot of the stream
/// (record its [`state_words`](Self::state_words)) as cheaply as a value read.
///
/// Do not build a *series* of seeds out of arithmetic: `seeded(s)` and
/// `seeded(s + GAMMA)`, or any small multiple of `GAMMA`, expand to states that
/// share three of four words, and correlated streams follow. Derive a stream
/// per system or entity from [`WorldSeed`] instead; it mixes the tag before
/// expanding.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SimRng {
    state: [u64; 4],
}

impl SimRng {
    /// A stream seeded by `seed` through SplitMix64.
    ///
    /// Any seed is valid, including `0`. Two runs of the same seed produce the
    /// same stream on every target.
    pub const fn seeded(seed: u64) -> Self {
        Self {
            state: expand(seed),
        }
    }

    /// A stream from four raw state words, for a save that stores the words
    /// itself instead of the serialised resource.
    ///
    /// All-zero words would make xoshiro emit zeros forever, so that one input
    /// is remapped to `0`'s seeded stream rather than left as a trap.
    pub const fn from_words(mut words: [u64; 4]) -> Self {
        if words[0] == 0 && words[1] == 0 && words[2] == 0 && words[3] == 0 {
            words = expand(0);
        }
        Self { state: words }
    }

    /// The four state words: the whole stream position, for a save.
    ///
    /// `from_words(state_words())` resumes exactly where the stream stood.
    pub const fn state_words(&self) -> [u64; 4] {
        self.state
    }
}

impl Rng for SimRng {
    /// The next 64 bits: multiply, rotate, multiply, then the state update.
    fn next_u64(&mut self) -> u64 {
        let result = self.state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }
}

/// The world's seed, kept in the world so any system can derive its own stream.
///
/// ```ignore
/// world.init_resource::<WorldSeed>();
/// world.resource_mut::<WorldSeed>().0 = WorldSeed::new(menu_seed);
///
/// // in a system:
/// let mut loot = world_seed.stream(LOOT_TAG);
/// let mut spread = world_seed.stream(SPREAD_TAG);
/// ```
///
/// Prefer [`named`](Self::named) over hand-numbered tags for per-system streams,
/// and carry a stored tag value over an enum's discriminant: a name survives an
/// enum being reordered, a tag derived from `as u64` does not.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldSeed(u64);

impl WorldSeed {
    /// A world seed. Every `u64` is a different world, `0` included.
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The seed as it was given, for menus and for the save file.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// A stream derived from the seed and a tag.
    ///
    /// The tag is mixed before the words are expanded, so a run of tags
    /// (`0, 1, 2, ...`) does not produce the `GAMMA`-shifted seed series that
    /// [`SimRng::seeded`] warns about, and `stream(0)` is not the world's bare
    /// seeded stream.
    pub fn stream(&self, tag: u64) -> SimRng {
        SimRng::seeded(finalize(self.0 ^ finalize(tag ^ TAG_SALT)))
    }

    /// A stream derived from the seed and a name.
    ///
    /// The name is folded with [`fnv1a64`](crate::hash::fnv1a64) first, so the
    /// stream is stable across runs and machines.
    pub fn named(&self, name: &str) -> SimRng {
        self.stream(crate::hash::fnv1a64(name.as_bytes()))
    }

    /// A stream for one entity, the same for that entity's whole life.
    ///
    /// Derived from the seed and the entity's id bits, so it does not depend on
    /// when the entity was spawned or on how many draws happened before. The id
    /// bits include the generation, so a recycled slot gives a fresh stream.
    pub fn stream_for_entity(&self, entity: Entity) -> SimRng {
        self.stream(entity.to_bits())
    }
}

/// Draws from a [`SimRng`]: one required method, everything else pinned here.
///
/// A system writes `rng: ResMut<SimRng>` and calls these methods, or takes
/// `rng: &mut impl Rng` to stay generic over the stream. `&mut R` is a `Rng`
/// when `R` is, so a mutable borrow passes straight through.
///
/// This is not `rand::Rng`: that trait's generators cannot be passed here
/// without depending on `rand` (a foreign trait over a foreign type cannot be
/// bridged from this crate), and `rand` is the dependency this module exists to
/// retire. A game that still needs a `rand` generator somewhere can wrap it in
/// its own newtype and implement this trait's one method, `next_u64`.
pub trait Rng {
    /// The next 64 bits of the stream. Everything else is built from these.
    fn next_u64(&mut self) -> u64;

    /// The next 32 bits, taken from the top half of a draw: the low bits of an
    /// xoshiro word carry more linear structure than the high ones.
    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// A value of `T`, uniform over its whole domain.
    ///
    /// `random::<f32>()` and `random::<f64>()` are uniform in `0..1`.
    /// `random::<u64>()` is one raw draw. Narrower integers take the top bits of
    /// a draw. `usize`/`isize` are deliberately not sampleable in full: a
    /// 64-bit desktop draw and a 32-bit wasm draw would not agree, so use a
    /// fixed-width type and cast.
    fn random<T: Sample>(&mut self) -> T {
        T::sample(self)
    }

    /// A value uniform in `range`, e.g. `random_range(0..enemies.len())`,
    /// `random_range(0.0..TAU)`, `random_range(-8..=8)`.
    ///
    /// Integer ranges are half-open (or inclusive as written) and unbiased.
    /// Float ranges are half-open: `0.0..=1.0` behaves as `0.0..1.0`, as in
    /// `rand`. Panics on an empty or reversed range, and on a range with a NaN
    /// bound, because a silent fallback would hide a sim bug.
    fn random_range<R: SampleRange>(&mut self, range: R) -> R::Output {
        range.sample_from(self)
    }

    /// One bit, exactly half the time, from the top bit of a draw.
    fn bool(&mut self) -> bool {
        (self.next_u64() >> 63) == 1
    }

    /// True with probability `p`, clamped to `0..=1`: `p <= 0` never, `p >= 1`
    /// always. Panics on NaN.
    ///
    /// A clamped probability consumes no draw, so a degenerate call cannot shift
    /// the stream.
    fn random_bool(&mut self, p: f64) -> bool {
        assert!(!p.is_nan(), "random_bool probability must not be NaN");
        if p <= 0.0 {
            return false;
        }
        if p >= 1.0 {
            return true;
        }
        self.unit_f64() < p
    }

    /// Uniform in `0..1`, built from the top 24 bits of one draw.
    ///
    /// The result is one of the `2^24` values `k / 2^24`, exactly representable
    /// and identical on every target; it never returns `1.0`.
    fn unit_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform in `0..1`, built from the top 53 bits of one draw.
    fn unit_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in `0..bound`, `0` for `bound == 0` (which consumes no draw).
    ///
    /// Wide-multiply rejection (Lemire): each draw maps onto `bound * 2^64`
    /// outcomes, and the underflow region is redrawn, so no modulo bias
    /// survives. The number of draws varies with the stream, deterministically
    /// for a given tape.
    fn below(&mut self, bound: u32) -> u32 {
        self.below_u64(bound as u64) as u32
    }

    /// [`below`](Self::below) for a 64-bit bound, e.g. an index into a huge
    /// table or a hit roll over a wide damage range.
    fn below_u64(&mut self, bound: u64) -> u64 {
        if bound == 0 {
            return 0;
        }
        let mut draw = self.next_u64();
        let mut product = (draw as u128) * (bound as u128);
        let mut low = product as u64;
        if low < bound {
            let threshold = bound.wrapping_neg() % bound;
            while low < threshold {
                draw = self.next_u64();
                product = (draw as u128) * (bound as u128);
                low = product as u64;
            }
        }
        (product >> 64) as u64
    }

    /// A uniform index in `0..len`, or `None` when `len == 0`.
    ///
    /// This is the pick that does not shuffle: it consumes one draw and depends
    /// on nothing but `len`, so it is stable whether the caller's slice is the
    /// game's own `Vec` or a temporary built in a different order. A shuffle is
    /// not a substitute for a pick: a permutation depends on the starting order
    /// of the whole slice and on every draw it consumes, so picking by shuffling
    /// changes the moment a sibling system's slice order does, and it is why
    /// [`shuffle`](Self::shuffle) is pinned here rather than open-coded.
    fn pick_index(&mut self, len: usize) -> Option<usize> {
        if len == 0 {
            return None;
        }
        Some(self.below_u64(len as u64) as usize)
    }

    /// A uniform element of `items`, or `None` when it is empty.
    fn pick<'a, T>(&mut self, items: &'a [T]) -> Option<&'a T> {
        self.pick_index(items.len()).map(|index| &items[index])
    }

    /// An index in `0..weights.len()` with probability proportional to its
    /// weight, or `None` when no weight is positive.
    ///
    /// Negative, NaN and infinite weights count as zero. One draw: the weights
    /// are summed in index order, and the running sum is `f32` in that same
    /// order, so equal weights taste the same in every replay. Zero-weight
    /// entries are skipped when the target lands exactly on zero.
    fn weighted_index(&mut self, weights: &[f32]) -> Option<usize> {
        let total: f32 = weights.iter().copied().map(effective_weight).sum();
        if !(total > 0.0) {
            return None;
        }
        let target = self.unit_f32() * total;
        let mut acc = 0.0f32;
        let mut last = None;
        for (index, weight) in weights.iter().enumerate() {
            if effective_weight(*weight) == 0.0 {
                continue;
            }
            last = Some(index);
            acc += *weight;
            if acc > target {
                return Some(index);
            }
        }
        last
    }

    /// Fisher-Yates, backwards: for `i` from `len - 1` down to `1`, swap `i`
    /// with `below(i + 1)` (so an element may stay put).
    ///
    /// A permutation is the one draw whose result depends on the *whole*
    /// sequence consumed and on the starting order, so it is pinned here rather
    /// than open-coded: a forward scan, a coin-flip swap, or an inclusive bound
    /// all produce a different permutation *and* a different number of draws,
    /// which silently moves every later draw in the run. `partial_shuffle` for a
    /// weighted draw without replacement is deliberately absent: sampling a
    /// weighted index and removing the winner biases the survivors, and a
    /// correct one needs a documented algorithm of its own.
    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below_u64(i as u64 + 1) as usize;
            items.swap(i, j);
        }
    }
}

impl<T: Rng + ?Sized> Rng for &mut T {
    fn next_u64(&mut self) -> u64 {
        (**self).next_u64()
    }
}

/// A weight that counts: finite and positive. Everything else is zero.
fn effective_weight(weight: f32) -> f32 {
    if weight.is_finite() && weight > 0.0 {
        weight
    } else {
        0.0
    }
}

/// One uniform offset in `0..span`.
///
/// A span of `2^64` means the range has no `u64`-sized length (the full domain
/// of `u64`/`i64`), so it takes a raw draw instead of a bounded one.
fn offset_in<R: Rng + ?Sized>(rng: &mut R, span: u128) -> u128 {
    if span == 1u128 << 64 {
        rng.next_u64() as u128
    } else {
        rng.below_u64(span as u64) as u128
    }
}

/// Zero-width or unit-width float, for the range impls.
trait Unit: Sized {
    fn unit<R: Rng + ?Sized>(rng: &mut R) -> Self;
}

impl Unit for f32 {
    fn unit<R: Rng + ?Sized>(rng: &mut R) -> f32 {
        rng.unit_f32()
    }
}

impl Unit for f64 {
    fn unit<R: Rng + ?Sized>(rng: &mut R) -> f64 {
        rng.unit_f64()
    }
}

/// A range [`Rng::random_range`] can sample.
///
/// Implemented for `Range` and `RangeInclusive` of `f32`, `f64`, and the
/// integer types `u8`..`u64`, `usize`, `i8`..`i64`, `isize`. Sealed on
/// purpose: which primitive a range draws from, and how many draws it takes,
/// is part of the replay contract, so it is not something a game may redefine.
pub trait SampleRange: sealed::Sealed {
    /// The sampled value.
    type Output;

    /// Draw one value from the range.
    #[doc(hidden)]
    fn sample_from<R: Rng + ?Sized>(self, rng: &mut R) -> Self::Output;
}

/// A type [`Rng::random`] can draw.
///
/// Implemented for `bool`, `f32`, `f64`, and the fixed-width integers
/// `u8`..`u64` and `i8`..`i64`. Sealed for the same reason as
/// [`SampleRange`]: the width of the draw is part of the replay contract.
pub trait Sample: sealed::Sealed {
    /// Draw one value.
    #[doc(hidden)]
    fn sample<R: Rng + ?Sized>(rng: &mut R) -> Self;
}

mod sealed {
    /// Implemented for the types this module already samples, so a game cannot
    /// add a `Sample`/`SampleRange` impl with a different draw layout.
    pub trait Sealed {}
}

macro_rules! impl_uint_range {
    ($($ty:ty),* $(,)?) => {$(
        impl sealed::Sealed for Range<$ty> {}
        impl SampleRange for Range<$ty> {
            type Output = $ty;

            fn sample_from<R: Rng + ?Sized>(self, rng: &mut R) -> $ty {
                assert!(
                    self.start < self.end,
                    "random_range needs a non-empty range, got {}..{}",
                    self.start,
                    self.end
                );
                let span = self.end as u128 - self.start as u128;
                self.start.wrapping_add(offset_in(rng, span) as $ty)
            }
        }

        impl sealed::Sealed for RangeInclusive<$ty> {}
        impl SampleRange for RangeInclusive<$ty> {
            type Output = $ty;

            fn sample_from<R: Rng + ?Sized>(self, rng: &mut R) -> $ty {
                let start = *self.start();
                let end = *self.end();
                assert!(
                    start <= end,
                    "random_range needs a non-empty range, got {start}..={end}"
                );
                let span = end as u128 - start as u128 + 1;
                start.wrapping_add(offset_in(rng, span) as $ty)
            }
        }
    )*};
}

macro_rules! impl_int_range {
    ($($ty:ty),* $(,)?) => {$(
        impl sealed::Sealed for Range<$ty> {}
        impl SampleRange for Range<$ty> {
            type Output = $ty;

            fn sample_from<R: Rng + ?Sized>(self, rng: &mut R) -> $ty {
                assert!(
                    self.start < self.end,
                    "random_range needs a non-empty range, got {}..{}",
                    self.start,
                    self.end
                );
                let span = (self.end as i128 - self.start as i128) as u128;
                self.start.wrapping_add(offset_in(rng, span) as $ty)
            }
        }

        impl sealed::Sealed for RangeInclusive<$ty> {}
        impl SampleRange for RangeInclusive<$ty> {
            type Output = $ty;

            fn sample_from<R: Rng + ?Sized>(self, rng: &mut R) -> $ty {
                let start = *self.start();
                let end = *self.end();
                assert!(
                    start <= end,
                    "random_range needs a non-empty range, got {start}..={end}"
                );
                let span = (end as i128 - start as i128 + 1) as u128;
                start.wrapping_add(offset_in(rng, span) as $ty)
            }
        }
    )*};
}

macro_rules! impl_float_range {
    ($($ty:ty),* $(,)?) => {$(
        impl sealed::Sealed for Range<$ty> {}
        impl SampleRange for Range<$ty> {
            type Output = $ty;

            fn sample_from<R: Rng + ?Sized>(self, rng: &mut R) -> $ty {
                assert!(
                    self.start < self.end,
                    "random_range needs an ordered, non-empty range, got {}..{}",
                    self.start,
                    self.end
                );
                self.start + (self.end - self.start) * <$ty as Unit>::unit(rng)
            }
        }

        impl sealed::Sealed for RangeInclusive<$ty> {}
        impl SampleRange for RangeInclusive<$ty> {
            type Output = $ty;

            fn sample_from<R: Rng + ?Sized>(self, rng: &mut R) -> $ty {
                let start = *self.start();
                let end = *self.end();
                assert!(
                    start < end,
                    "random_range needs an ordered, non-empty range, got {start}..={end}"
                );
                start + (end - start) * <$ty as Unit>::unit(rng)
            }
        }
    )*};
}

impl_uint_range!(u8, u16, u32, u64, usize);
impl_int_range!(i8, i16, i32, i64, isize);
impl_float_range!(f32, f64);

/// The full domain of a fixed-width integer, a uniform draw.
///
/// `u64` and `i64` take one raw draw; narrower types take the top bits of a
/// draw. Signed types reinterpret those bits rather than drawing a magnitude.
macro_rules! impl_sample_small {
    ($(($ty:ident, $unsigned:ty)),* $(,)?) => {$(
        impl sealed::Sealed for $ty {}
        impl Sample for $ty {
            fn sample<R: Rng + ?Sized>(rng: &mut R) -> $ty {
                let bits = rng.next_u64() >> (64 - core::mem::size_of::<$unsigned>() * 8);
                (bits as $unsigned) as $ty
            }
        }
    )*};
}

impl sealed::Sealed for u64 {}
impl Sample for u64 {
    fn sample<R: Rng + ?Sized>(rng: &mut R) -> u64 {
        rng.next_u64()
    }
}

impl sealed::Sealed for i64 {}
impl Sample for i64 {
    fn sample<R: Rng + ?Sized>(rng: &mut R) -> i64 {
        rng.next_u64() as i64
    }
}

impl_sample_small!(
    (u32, u32),
    (u16, u16),
    (u8, u8),
    (i32, u32),
    (i16, u16),
    (i8, u8)
);

impl sealed::Sealed for bool {}
impl Sample for bool {
    fn sample<R: Rng + ?Sized>(rng: &mut R) -> bool {
        rng.bool()
    }
}

impl sealed::Sealed for f32 {}
impl Sample for f32 {
    fn sample<R: Rng + ?Sized>(rng: &mut R) -> f32 {
        rng.unit_f32()
    }
}

impl sealed::Sealed for f64 {}
impl Sample for f64 {
    fn sample<R: Rng + ?Sized>(rng: &mut R) -> f64 {
        rng.unit_f64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: u64 = 0x0123_4567_89AB_CDEF;

    #[test]
    fn core_matches_the_published_xoshiro256starstar_vectors() {
        // rand_xoshiro 0.7.0's `reference` test: state words 1, 2, 3, 4, as the
        // reference implementation produces for that seed.
        let mut rng = SimRng::from_words([1, 2, 3, 4]);
        let want = [
            11520,
            0,
            1509978240,
            1215971899390074240,
            1216172134540287360,
            607988272756665600,
            16172922978634559625,
            8476171486693032832,
            10595114339597558777,
            2904607092377533576,
        ];
        for value in want {
            assert_eq!(rng.next_u64(), value);
        }
    }

    #[test]
    fn seeded_stream_is_pinned_word_for_word() {
        let mut rng = SimRng::seeded(SEED);
        let want = [
            11728116837925579837,
            431261241542867727,
            7088239201150201886,
            1991960781942118182,
            16071698363884655823,
            4123588241367215080,
            13086776817198750337,
            9243408305086729258,
        ];
        for value in want {
            assert_eq!(rng.next_u64(), value);
        }
        assert_eq!(SimRng::seeded(0).state_words()[0], 0xE220_A839_7B1D_CDAF);
    }

    #[test]
    fn seeded_zero_is_a_stream_not_a_fixed_point() {
        let mut rng = SimRng::seeded(0);
        for _ in 0..8 {
            assert_ne!(rng.next_u64(), 0);
        }
        // A single zero word is legal: only the all-zero state traps xoshiro.
        let mut mostly_zero = SimRng::from_words([0, 1, 2, 3]);
        assert_ne!(mostly_zero.next_u64(), 0);
        // And the trap itself is remapped rather than left to emit zeros.
        assert_eq!(
            SimRng::from_words([0; 4]).state_words(),
            SimRng::seeded(0).state_words()
        );
    }

    #[test]
    fn sampling_is_pinned_value_for_value() {
        let mut rng = SimRng::seeded(SEED);
        assert_eq!(rng.next_u32(), 2730664992);
        assert_eq!(rng.next_u32(), 100410832);

        let mut rng = SimRng::seeded(SEED);
        assert_eq!(rng.unit_f32(), 0.63578248_f32);

        let mut rng = SimRng::seeded(SEED);
        assert_eq!(rng.unit_f64(), 0.6357824877421369_f64);
        assert!(rng.unit_f64() < 1.0);

        let mut rng = SimRng::seeded(SEED);
        assert_eq!(
            [rng.below(6), rng.below(6), rng.below(6), rng.below(6)],
            [3, 0, 2, 0]
        );
        assert_eq!(SimRng::seeded(SEED).below(0), 0);
        assert_eq!(SimRng::seeded(SEED).below_u64(1 << 40), 699050238008);

        let mut rng = SimRng::seeded(SEED);
        assert_eq!(
            [rng.bool(), rng.bool(), rng.bool(), rng.bool()],
            [true, false, false, false]
        );
        assert_eq!(SimRng::seeded(SEED).random::<u8>(), 162);
        assert_eq!(SimRng::seeded(SEED).random::<i32>(), -1564302304);
        assert_eq!(SimRng::seeded(SEED).random::<u64>(), 11728116837925579837);
    }

    #[test]
    fn ranges_are_pinned_and_agree_with_their_primitives() {
        let mut rng = SimRng::seeded(SEED);
        assert_eq!(
            [
                rng.random_range(0..6),
                rng.random_range(0..6),
                rng.random_range(0..6),
                rng.random_range(0..6),
                rng.random_range(0..6),
                rng.random_range(0..6),
                rng.random_range(0..6),
                rng.random_range(0..6),
            ],
            [3, 0, 2, 0, 5, 1, 4, 3]
        );
        let mut rng = SimRng::seeded(SEED);
        assert_eq!(
            [
                rng.random_range(2..=7),
                rng.random_range(2..=7),
                rng.random_range(2..=7),
                rng.random_range(2..=7),
            ],
            [5, 2, 4, 2]
        );
        let mut rng = SimRng::seeded(SEED);
        assert_eq!(
            rng.random_range(0.0..1.0_f32),
            SimRng::seeded(SEED).unit_f32()
        );
        let mut rng = SimRng::seeded(SEED);
        // The suffix matters: a bare `-1.0..1.0` infers `f64` and samples
        // `unit_f64`, whose result is not these bits. These are the f32 path:
        // `-1.0 + 2.0 * unit_f32`, every operator f32.
        assert_eq!(
            [
                rng.random_range(-1.0_f32..1.0_f32),
                rng.random_range(-1.0_f32..1.0_f32)
            ],
            [0.27156496047973633, -0.9532426595687866]
        );
        // The same range in `f64` is its own pin, not the f32 values widened:
        // `unit_f64` reads the top 53 bits of each draw, `unit_f32` the top 24.
        let mut rng = SimRng::seeded(SEED);
        assert_eq!(
            [
                rng.random_range(-1.0_f64..1.0_f64),
                rng.random_range(-1.0_f64..1.0_f64)
            ],
            [0.2715649754842737, -0.9532425625010428]
        );

        // The widest ranges are the ones where a naive span computation
        // overflows: the full domains of `u64` and `i64` both still sample, one
        // raw draw each.
        assert_eq!(
            SimRng::seeded(SEED).random_range(u64::MIN..=u64::MAX),
            11728116837925579837
        );
        assert_eq!(
            SimRng::seeded(SEED).random_range(i64::MIN..=i64::MAX),
            2504744801070804029
        );
        // A one-wide range consumes its draw and always lands on it.
        assert_eq!(SimRng::seeded(SEED).random_range(0..1_u32), 0);
    }

    #[test]
    fn a_pick_is_one_unbiased_draw_and_a_shuffle_is_pinned() {
        let mut rng = SimRng::seeded(SEED);
        let want = [3, 0, 1, 0, 4, 1, 3, 2];
        for index in want {
            assert_eq!(rng.pick_index(5), Some(index));
        }
        assert_eq!(rng.pick_index(0), None);
        assert_eq!(rng.pick(&[] as &[u8]), None);
        assert_eq!(rng.pick(&[10, 11, 12, 13, 14]), Some(&14));

        let mut deck = [0, 1, 2, 3, 4, 5, 6, 7];
        SimRng::seeded(SEED).shuffle(&mut deck);
        assert_eq!(deck, [7, 1, 4, 3, 6, 2, 0, 5]);

        let mut empty: [u8; 0] = [];
        SimRng::seeded(SEED).shuffle(&mut empty);
        let mut single = [9];
        SimRng::seeded(SEED).shuffle(&mut single);
        assert_eq!(single, [9]);
    }

    #[test]
    fn below_uses_its_whole_range_without_modulo_bias() {
        let mut rng = SimRng::seeded(7);
        let mut faces = [0usize; 6];
        for _ in 0..60_000 {
            faces[rng.below(6) as usize] += 1;
        }
        for (face, count) in faces.iter().enumerate() {
            assert!(
                (9_200..=10_800).contains(count),
                "face {face} came up {count} times in 60_000 draws"
            );
        }
    }

    #[test]
    fn weighted_index_follows_its_weights() {
        let mut rng = SimRng::seeded(2);
        assert_eq!(rng.weighted_index(&[1.0, 2.0, 5.0]), Some(0));
        let mut counts = [0usize; 3];
        let mut rng = SimRng::seeded(11);
        for _ in 0..30_000 {
            counts[rng.weighted_index(&[1.0, 2.0, 5.0]).unwrap()] += 1;
        }
        for (index, want) in [(0, 3_750), (1, 7_500), (2, 18_750)] {
            let got = counts[index];
            assert!(
                (want * 92 / 100..want * 108 / 100).contains(&got),
                "index {index} came up {got} times, expected about {want}"
            );
        }
        let mut rng = SimRng::seeded(SEED);
        let before = rng.state_words();
        assert_eq!(rng.weighted_index(&[0.0, 0.0]), None);
        assert_eq!(rng.weighted_index(&[]), None);
        assert_eq!(
            rng.state_words(),
            before,
            "a refused weight consumes no draw"
        );
        assert_eq!(
            SimRng::seeded(SEED).weighted_index(&[f32::NAN, -1.0, 2.0]),
            Some(2)
        );
    }

    #[test]
    fn a_save_resumes_the_identical_stream() {
        let mut original = SimRng::seeded(SEED);
        for _ in 0..13 {
            original.next_u64();
        }
        let mut restored = SimRng::from_words(original.state_words());
        assert_eq!(restored.state_words(), original.state_words());
        assert_eq!(
            restored.state_words(),
            [
                0x22F8_BA83_280F_7AD2,
                0x5DD0_0A53_80BC_904C,
                0x54FF_F2A6_2C07_7DDF,
                0x1C14_3DD2_89E7_2E17
            ]
        );
        for _ in 0..8 {
            assert_eq!(restored.next_u64(), original.next_u64());
        }
    }

    #[test]
    fn random_bool_clamps_and_consumes_nothing_for_a_certainty() {
        let mut rng = SimRng::seeded(SEED);
        assert!(!rng.random_bool(0.0));
        assert!(rng.random_bool(1.0));
        assert!(!rng.random_bool(-1.0));
        assert!(rng.random_bool(2.0));
        assert_eq!(rng.state_words(), SimRng::seeded(SEED).state_words());
        assert!(rng.random_bool(0.75), "0.63578.. is below 0.75");
    }

    #[test]
    #[should_panic(expected = "must not be NaN")]
    fn random_bool_rejects_nan() {
        SimRng::seeded(SEED).random_bool(f64::NAN);
    }

    #[test]
    #[should_panic(expected = "non-empty range")]
    fn an_empty_range_is_a_bug_not_a_default() {
        SimRng::seeded(SEED).random_range(4..4);
    }

    #[test]
    #[should_panic(expected = "non-empty range")]
    fn a_reversed_float_range_is_a_bug() {
        SimRng::seeded(SEED).random_range(1.0..=0.0);
    }

    #[test]
    fn a_seed_series_is_the_documented_hazard_derivation_avoids() {
        // Seeded streams a GAMMA apart share three words: that is why a game
        // must not build seeds by adding a constant.
        let a = SimRng::seeded(SEED).state_words();
        let b = SimRng::seeded(SEED.wrapping_add(SPLITMIX_GAMMA)).state_words();
        let shared = a.iter().filter(|word| b.contains(word)).count();
        assert_eq!(shared, 3, "the hazard this module documents is real");

        // Derived tags are mixed first, so no tag pair reproduces it.
        let seed = WorldSeed::new(0xDEAD_BEEF);
        for tag in 0..64u64 {
            let near = seed.stream(tag.wrapping_add(1)).state_words();
            let far = seed.stream(tag).state_words();
            assert!(far.iter().all(|word| !near.contains(word)));
        }
    }

    #[test]
    fn world_seed_streams_are_distinct_and_additive_changes_do_not_shift_them() {
        let seed = WorldSeed::new(0xDEAD_BEEF);
        let mut born = false;
        let mut seen = std::collections::HashSet::new();
        for tag in 0..64 {
            if tag == 32 {
                born = true;
            }
            let mut rng = seed.stream(tag);
            for _ in 0..64 {
                assert!(seen.insert(rng.next_u64()), "two streams collided");
            }
        }
        assert!(born);
        // Deriving is pure: the same tag reads the same stream whatever else ran.
        assert_eq!(
            seed.stream(7).next_u64(),
            WorldSeed::new(0xDEAD_BEEF).stream(7).next_u64()
        );
        assert_eq!(seed.stream(0).next_u64(), 0xC400_5AD9_14EA_B622);
        assert_eq!(seed.stream(7).next_u64(), 0x8BBB_8B00_BC67_7185);
        let base = SimRng::seeded(seed.get()).next_u64();
        assert_ne!(
            seed.stream(0).next_u64(),
            base,
            "tag 0 must not be the bare seeded stream"
        );
    }

    #[test]
    fn named_and_entity_streams_are_stable_and_apart_from_the_bare_stream() {
        let seed = WorldSeed::new(0xDEAD_BEEF);
        assert_eq!(seed.named("loot").next_u64(), 0x68F0_61C5_2FF8_2575);
        assert_eq!(seed.named("loot").next_u64(), seed.named("loot").next_u64());
        assert_ne!(seed.named("loot").next_u64(), seed.named("Loot").next_u64());

        let mut world = World::new();
        let entity = world.spawn_empty().id();
        assert_eq!(
            seed.stream_for_entity(entity).next_u64(),
            seed.stream(entity.to_bits()).next_u64()
        );
        assert_eq!(
            seed.stream_for_entity(entity).next_u64(),
            seed.stream_for_entity(entity).next_u64()
        );
        assert_ne!(
            seed.stream_for_entity(entity).next_u64(),
            seed.stream(0).next_u64()
        );
    }

    #[test]
    fn a_resource_can_be_seeded_and_serialised() {
        fn assert_serde<T: Serialize + serde::de::DeserializeOwned>() {}
        assert_serde::<SimRng>();
        assert_serde::<WorldSeed>();

        let mut world = World::new();
        world.insert_resource(SimRng::seeded(SEED));
        world.insert_resource(WorldSeed::new(SEED));
        assert_eq!(
            world.resource_mut::<SimRng>().random_range(0..6),
            SimRng::seeded(SEED).random_range(0..6)
        );
        assert_eq!(world.resource::<WorldSeed>().get(), SEED);
    }

    #[test]
    fn a_mutable_borrow_is_a_stream() {
        fn draw(rng: &mut impl Rng) -> u64 {
            rng.next_u64()
        }
        let mut rng = SimRng::seeded(SEED);
        let mut borrow = &mut rng;
        let through_borrow = draw(&mut borrow);
        assert_eq!(through_borrow, SimRng::seeded(SEED).next_u64());
    }
}
