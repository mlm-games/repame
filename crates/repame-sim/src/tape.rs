//! The intent tape, the canonical world hash, and the replay that checks them.
//!
//! A run is reproducible when three things are pinned: the world it starts from
//! (the seed and the spawn history), the intent fed in on every tick, and the
//! code. [`Tape`] holds the input side of that as data, one input per tick, plus
//! the canonical world hash every tick left behind, so a replay is checked tick
//! by tick instead of only at its end. [`Tape::verify`] replays a tape and names
//! the first tick whose hash differs, which is the harness the crate docs ask
//! for: run one trace twice, hash the world per tick, diff.
//!
//! # The hashing order rule
//!
//! [`WorldHasher::hash`] folds, in this order, and never in the order `bevy_ecs`
//! hands things back:
//!
//! 1. A world tag, then the sorted ids of every live entity that is not a
//!    resource entity.
//! 2. Each registered component, in registration order: how many entities match,
//!    then one `(entity id, value hash)` pair per match, sorted ascending by
//!    entity id.
//! 3. Each registered resource, in registration order: a presence byte, then the
//!    value hash when it exists.
//!
//! `bevy_ecs` iterates archetypes and tables in creation order and rows in the
//! order entities entered them, so two runs that reach the same state through a
//! different spawn/despawn history iterate differently (the warning at the top
//! of the crate docs). Sorting is what makes the digest describe state rather
//! than layout: a divergence report has to name a gameplay difference, not a
//! table reordering. Entity ids are folded because they *are* state: identical
//! spawn histories produce identical ids, so a run that spawned one entity too
//! many changes the digest the moment it does.
//!
//! Resource entities are excluded: `bevy_ecs` stores each resource on an entity
//! of its own and `iter_entities` yields those too, but one is bookkeeping, not
//! gameplay state. The harness itself inserts [`Intent`] on the replay side, and
//! a recording world that never had it (a game with its own action resources)
//! would fold one entity fewer, so a tape could not reproduce its own recording.
//! A registered resource is still folded, by its entry in step 3.
//!
//! What the digest does not cover: state nobody registered, and anything the
//! process gives one run and not the other (wall-clock time, thread-scheduled
//! work). The registered set is part of the replay contract, so register the
//! same components and resources in the same order on both sides.

use std::fmt;
use std::hash::Hash;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use bevy_ecs::resource::IsResource;
use serde::{Deserialize, Serialize};

use crate::Sim;
use crate::hash::Hash64;

/// Domain tag: the first thing a world digest folds.
const WORLD_TAG: &str = "repame-sim/world";

/// The input for the tick about to run.
///
/// One convention the tape can drive without glue: a game's input system reads
/// this resource, the app fills it from the device before each tick, and a
/// replay fills it from the recording. A game that keeps its own action
/// resources instead hands [`Tape::replay_with`] a closure that copies the
/// intent where the game expects it.
#[derive(Resource)]
pub struct Intent<I>(Option<I>);

impl<I> Intent<I> {
    /// This tick's input, if the app or the tape provided one.
    pub fn get(&self) -> Option<&I> {
        self.0.as_ref()
    }

    /// Set this tick's input.
    pub fn set(&mut self, input: I) {
        self.0 = Some(input);
    }

    /// Drop the input, the state a live app starts a tick in before polling.
    pub fn clear(&mut self) {
        self.0 = None;
    }
}

impl<I> Default for Intent<I> {
    fn default() -> Self {
        Self(None)
    }
}

/// A recorded run: one intent per tick, plus the canonical world hash each tick
/// left behind, both in tick order.
///
/// A tape is the artifact a bug report carries: the seed, what the player asked
/// for on every tick, and what the world looked like. Serialisable, so it can be
/// written next to the save it came from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tape<I> {
    seed: u64,
    intents: Vec<I>,
    hashes: Vec<u64>,
}

impl<I> Tape<I> {
    /// An empty tape for a run seeded with `seed`.
    ///
    /// The seed is carried because a tape without it is not replayable: the
    /// factory a replay is handed reads [`seed`](Self::seed) to rebuild the same
    /// starting world.
    pub const fn new(seed: u64) -> Self {
        Self {
            seed,
            intents: Vec::new(),
            hashes: Vec::new(),
        }
    }

    /// The seed the recording run started from.
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// How many ticks are recorded. What a replay runs.
    pub fn ticks(&self) -> u64 {
        self.intents.len() as u64
    }

    /// Whether no tick has been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.intents.is_empty()
    }

    /// Every recorded intent, in tick order.
    pub fn intents(&self) -> &[I] {
        &self.intents
    }

    /// Every recorded intent, to edit a tape before replaying it.
    ///
    /// Rewriting a tick's intent while keeping the recorded hashes is how a
    /// divergence is bisected: the replay then reports that tick (or says the
    /// edit changed nothing), instead of a hunt through a whole session.
    pub fn intents_mut(&mut self) -> &mut [I] {
        &mut self.intents
    }

    /// Every recorded hash, in tick order. Shorter than the tape when the intent
    /// tail was never hashed.
    pub fn hashes(&self) -> &[u64] {
        &self.hashes
    }

    /// The intent of `tick`, or `None` past the end.
    pub fn intent(&self, tick: u64) -> Option<&I> {
        self.intents.get(tick as usize)
    }

    /// The recorded world hash of `tick`, or `None` if that tick was not hashed.
    pub fn hash(&self, tick: u64) -> Option<u64> {
        self.hashes.get(tick as usize).copied()
    }

    /// Appends one tick's intent. Returns the tick it landed on.
    ///
    /// Call it before the tick runs, so the tape is in the order the game saw.
    pub fn record_intent(&mut self, intent: I) -> u64 {
        let tick = self.intents.len() as u64;
        self.intents.push(intent);
        tick
    }

    /// Appends the canonical world hash of the tick just recorded, in tick
    /// order: tick `n`'s hash is the world after tick `n` ran.
    ///
    /// A tape may stop hashing early (an intent-only tail cannot be verified,
    /// see [`verify`](Self::verify)), but never hashes twice for one tick, and
    /// never more hashes than ticks.
    ///
    /// # Panics
    ///
    /// If every recorded intent already has a hash. That is a recording loop
    /// hashing twice for one tick, which would misalign every later comparison.
    pub fn record_hash(&mut self, world_hash: u64) {
        assert!(
            self.hashes.len() < self.intents.len(),
            "record_hash with no tick to belong to: {} ticks, {} hashes",
            self.intents.len(),
            self.hashes.len()
        );
        self.hashes.push(world_hash);
    }

    /// One tick of a replay: the recorded intent into [`Intent`], then `commit`,
    /// then the tick, then the canonical hash of the world it left. Returns that
    /// hash.
    fn step(
        &self,
        sim: &mut Sim,
        hasher: &WorldHasher,
        tick: usize,
        commit: &mut impl FnMut(&mut World, &I),
    ) -> u64
    where
        I: Clone + Send + Sync + 'static,
    {
        let intent = &self.intents[tick];
        sim.world.resource_mut::<Intent<I>>().set(intent.clone());
        commit(&mut sim.world, intent);
        sim.tick();
        hasher.hash(&sim.world)
    }
}

impl<I> Tape<I>
where
    I: Clone + Send + Sync + 'static,
{
    /// Replays the tape into a fresh world and returns the canonical hash of
    /// every tick, in tick order.
    ///
    /// `setup` is called once per replay and must rebuild the same starting
    /// world every time, so a caller can replay one tape twice and diff the two
    /// logs itself, or compare them against a golden list.
    pub fn replay<F>(&self, setup: F) -> Vec<u64>
    where
        F: FnMut() -> Setup,
    {
        self.replay_with(setup, |_, _| {})
    }

    /// [`replay`](Self::replay), running `commit` on each tick's intent after
    /// [`Intent`] is set and before the tick runs.
    ///
    /// This is the hook for a game whose systems read their own action
    /// resources:
    ///
    /// ```ignore
    /// let commit = |world: &mut World, input: &Press| {
    ///     world.resource_mut::<Keys>().0 = *input;
    /// };
    /// let hashes = tape.replay_with(|| fresh(seed), commit);
    /// ```
    pub fn replay_with<F, C>(&self, mut setup: F, mut commit: C) -> Vec<u64>
    where
        F: FnMut() -> Setup,
        C: FnMut(&mut World, &I),
    {
        let Setup { mut sim, hasher } = setup();
        sim.world.insert_resource(Intent::<I>::default());
        let mut hashes = Vec::with_capacity(self.intents.len());
        for tick in 0..self.intents.len() {
            hashes.push(self.step(&mut sim, &hasher, tick, &mut commit));
        }
        hashes
    }

    /// Replays the tape and compares every tick's hash against the recording,
    /// returning the first tick that differs.
    ///
    /// The replay stops at the divergence: a world that already disagrees is not
    /// worth stepping further, and a game that would panic on it panics here
    /// instead of reporting the tick it went wrong at. Pair this with
    /// [`record_intent`](Self::record_intent) and
    /// [`record_hash`](Self::record_hash).
    pub fn verify<F>(&self, setup: F) -> Result<(), VerifyError>
    where
        F: FnMut() -> Setup,
    {
        self.verify_with(setup, |_, _| {})
    }

    /// [`verify`](Self::verify), running `commit` on each tick's intent after
    /// [`Intent`] is set and before the tick runs.
    pub fn verify_with<F, C>(&self, mut setup: F, mut commit: C) -> Result<(), VerifyError>
    where
        F: FnMut() -> Setup,
        C: FnMut(&mut World, &I),
    {
        if self.hashes.len() != self.intents.len() {
            return Err(VerifyError::UnhashedTicks {
                ticks: self.intents.len() as u64,
                hashes: self.hashes.len() as u64,
            });
        }
        let Setup { mut sim, hasher } = setup();
        sim.world.insert_resource(Intent::<I>::default());
        let mut replayed = Vec::with_capacity(self.intents.len());
        for tick in 0..self.intents.len() {
            let actual = self.step(&mut sim, &hasher, tick, &mut commit);
            let expected = self.hashes[tick];
            replayed.push(actual);
            if expected != actual {
                return Err(VerifyError::Diverged {
                    divergence: Divergence {
                        tick: tick as u64,
                        expected,
                        actual,
                    },
                    replayed,
                });
            }
        }
        Ok(())
    }
}

/// A fresh world to replay into, with the hasher that folds it.
///
/// A tape is checked by building the same starting world twice: the recording
/// side built it once, every replay builds it again. That factory returns this.
pub struct Setup {
    /// The sim to step, with the game's systems registered.
    pub sim: Sim,
    /// What the digest covers. Register the same components and resources, in
    /// the same order, as the recording run did.
    pub hasher: WorldHasher,
}

/// The canonical world hash: which state the digest covers, and in what order.
///
/// The fold order is pinned in the module docs. An empty hasher digests the
/// entity list alone, which is a real check in a game whose state *is* its
/// entities, but most callers register the components and resources that decide
/// gameplay:
///
/// ```ignore
/// let mut hasher = WorldHasher::new();
/// hasher.register::<Height>(|height, hash| hash.f32(height.0));
/// hasher.register_resource::<Score>(|score, hash| hash.u32(score.0));
/// ```
#[derive(Clone, Default)]
pub struct WorldHasher {
    entries: Vec<Arc<dyn Fn(&World, &mut Hash64) + Send + Sync>>,
}

impl WorldHasher {
    /// A hasher that folds the entity list and nothing else.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Registers a component: one `(entity, value)` fold per entity that has it.
    ///
    /// Write the value's fields in a fixed order. A float folds its bits
    /// (`hash.f32(value)`), so the sign of zero and a NaN payload are covered
    /// exactly; a field left out is a field the digest cannot report.
    pub fn register<C: Component>(
        &mut self,
        hash_value: impl Fn(&C, &mut Hash64) + Send + Sync + 'static,
    ) -> &mut Self {
        self.entries
            .push(Arc::new(move |world: &World, hash: &mut Hash64| {
                // A component this world never registered matches no entity, so
                // it folds the same zero count an empty query folds.
                let Some(mut query) = world.try_query::<(Entity, &C)>() else {
                    hash.u64(0);
                    return;
                };
                let mut rows: Vec<(u64, u64)> = Vec::new();
                for (entity, value) in query.iter(world) {
                    let mut row = Hash64::new();
                    hash_value(value, &mut row);
                    rows.push((entity.to_bits(), row.digest()));
                }
                rows.sort_unstable_by_key(|(entity, _)| *entity);
                hash.u64(rows.len() as u64);
                for (entity, value) in rows {
                    hash.u64(entity);
                    hash.u64(value);
                }
            }));
        self
    }

    /// [`register`](Self::register) for a component whose own `Hash`
    /// implementation is yours.
    ///
    /// The fold is whatever `Hash` writes, and every integer writer is pinned by
    /// [`Hash64`], but a `Hash` impl from `std` or from a dependency may change
    /// its byte layout in a later release. For a digest a save file or a golden
    /// test pins, write the fields explicitly with
    /// [`register`](Self::register).
    pub fn register_hashed<C: Component + Hash>(&mut self) -> &mut Self {
        self.register::<C>(|value, hash| value.hash(hash))
    }

    /// Registers a resource: a presence byte, then the value hash when it
    /// exists.
    ///
    /// Presence is state of its own, so a resource missing from one run and
    /// present in the other folds differently even when its payload is a zero.
    pub fn register_resource<R: Resource>(
        &mut self,
        hash_value: impl Fn(&R, &mut Hash64) + Send + Sync + 'static,
    ) -> &mut Self {
        self.entries.push(Arc::new(
            move |world: &World, hash: &mut Hash64| match world.get_resource::<R>() {
                Some(value) => {
                    let mut presence = Hash64::new();
                    hash_value(value, &mut presence);
                    hash.u8(1);
                    hash.u64(presence.digest());
                }
                None => hash.u8(0),
            },
        ));
        self
    }

    /// [`register_resource`](Self::register_resource) for a resource whose own
    /// `Hash` implementation is yours.
    pub fn register_hashed_resource<R: Resource + Hash>(&mut self) -> &mut Self {
        self.register_resource::<R>(|value, hash| value.hash(hash))
    }

    /// The canonical digest of `world`: one number that moves whenever any
    /// registered state moves.
    ///
    /// Folds in the order the module docs pin, which is not `bevy_ecs`'s
    /// iteration order, so a run that reaches the same state through a different
    /// spawn/despawn history digests the same. Resource entities are left out of
    /// the entity list for the reason the module docs give; registered resources
    /// are folded by their entries instead.
    pub fn hash(&self, world: &World) -> u64 {
        let mut hash = Hash64::new();
        hash.str(WORLD_TAG);
        let mut entities: Vec<u64> = world
            .iter_entities()
            .filter(|entity| !entity.contains::<IsResource>())
            .map(|entity| entity.id().to_bits())
            .collect();
        entities.sort_unstable();
        hash.usize(entities.len());
        for bits in entities {
            hash.u64(bits);
        }
        for entry in &self.entries {
            entry(world, &mut hash);
        }
        hash.digest()
    }
}

/// The first tick a replay disagreed with the recording at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Divergence {
    /// Tick the two runs first differed on. Ticks before it hashed the same, so
    /// the cause is in what this tick's systems read or in the intent it took:
    /// [`Tape::intent`] has that tick's input, and the recording's own hash log
    /// is one [`Tape::hash`] call away.
    pub tick: u64,
    /// Hash the recording left for this tick.
    pub expected: u64,
    /// Hash this replay produced for it.
    pub actual: u64,
}

impl fmt::Display for Divergence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "tick {}: recorded {:#018x}, replayed {:#018x}",
            self.tick, self.expected, self.actual
        )
    }
}

/// Why a tape could not be checked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    /// The tape's hash log does not have one entry per tick, so a replay cannot
    /// be checked tick for tick. An intent-only tape (replayed for its visuals,
    /// say) lands here.
    UnhashedTicks {
        /// Ticks recorded.
        ticks: u64,
        /// Hashes recorded.
        hashes: u64,
    },
    /// The replay's world differed from the recording's.
    Diverged {
        /// The first tick that differed, with both hashes.
        divergence: Divergence,
        /// Every hash this replay produced, up to and including the diverging
        /// tick: how far the two runs agreed before this.
        replayed: Vec<u64>,
    },
}

impl VerifyError {
    /// The first divergent tick, when the failure was a divergence rather than a
    /// tape that cannot be checked.
    pub fn tick(&self) -> Option<u64> {
        match self {
            Self::Diverged { divergence, .. } => Some(divergence.tick),
            Self::UnhashedTicks { .. } => None,
        }
    }
}

impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnhashedTicks { ticks, hashes } => write!(
                f,
                "tape has {ticks} ticks and {hashes} hashes; verify needs one hash per tick"
            ),
            Self::Diverged { divergence, .. } => write!(f, "diverged at {divergence}"),
        }
    }
}

impl std::error::Error for VerifyError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::{Rng, SimRng, WorldSeed};

    /// One tick of recorded input: what a player could ask for.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
    struct Press {
        /// Hold the accelerator.
        up: bool,
        /// Spawn one body.
        spawn: bool,
    }

    /// A body's height, folded into the digest.
    #[derive(Component, Clone, Copy, Debug)]
    struct Height(f32);

    /// A spawned body's speed, folded into the digest.
    #[derive(Component, Clone, Copy, Debug)]
    struct Speed(u32);

    /// Score, folded into the digest.
    #[derive(Resource, Debug, Default, PartialEq, Eq)]
    struct Score(u32);

    /// One tick of gameplay: the only place a tick's intent is read.
    fn play(
        intent: Res<Intent<Press>>,
        mut score: ResMut<Score>,
        mut rng: ResMut<SimRng>,
        mut bodies: Query<&mut Height>,
        mut commands: Commands,
    ) {
        let Some(press) = intent.get() else {
            return;
        };
        if press.up {
            score.0 += 1;
            for mut height in &mut bodies {
                height.0 += 1.0;
            }
        }
        if press.spawn {
            let speed: u32 = rng.random_range(0_u32..4);
            commands.spawn(Speed(speed));
        }
    }

    /// What a replay must agree on: the bodies, the score, and the stream.
    fn digest() -> WorldHasher {
        let mut hasher = WorldHasher::new();
        hasher.register::<Height>(|height, hash| hash.f32(height.0));
        hasher.register::<Speed>(|speed, hash| hash.u32(speed.0));
        hasher.register_resource::<Score>(|score, hash| hash.u32(score.0));
        hasher.register_resource::<SimRng>(|rng, hash| {
            for word in rng.state_words() {
                hash.u64(word);
            }
        });
        hasher
    }

    /// A fresh world for one run, exactly as the recording run built it.
    fn fresh(seed: u64) -> Setup {
        let mut sim = Sim::with_default_step();
        sim.world.insert_resource(WorldSeed::new(seed));
        sim.world.insert_resource(SimRng::seeded(seed));
        sim.world.insert_resource(Intent::<Press>::default());
        sim.world.init_resource::<Score>();
        sim.world.spawn(Height(0.0));
        sim.add_system(play);
        Setup {
            sim,
            hasher: digest(),
        }
    }

    /// The greedy policy the tape is recorded from: a recorded tape exercises
    /// states a hand-written script cannot know about.
    fn press(tick: u64, score: u32) -> Press {
        Press {
            up: (score + tick as u32) % 3 != 0,
            spawn: tick % 7 == 3,
        }
    }

    fn record(seed: u64, ticks: u64) -> Tape<Press> {
        let Setup { mut sim, hasher } = fresh(seed);
        let mut tape = Tape::new(seed);
        for tick in 0..ticks {
            let input = press(tick, sim.world.resource::<Score>().0);
            tape.record_intent(input);
            sim.world.resource_mut::<Intent<Press>>().set(input);
            sim.tick();
            tape.record_hash(hasher.hash(&sim.world));
        }
        tape
    }

    #[test]
    fn a_recorded_run_verifies_hash_for_hash() {
        let tape = record(7, 60);
        assert_eq!(tape.seed(), 7);
        assert_eq!(tape.ticks(), 60);
        assert_eq!(tape.hashes().len(), 60);
        assert_eq!(tape.hash(0), Some(tape.hashes()[0]));
        assert_eq!(tape.hash(60), None, "past the end there is nothing");

        // The recording's own hashes are what an identical replay produces, and
        // two replays of one tape agree tick for tick.
        assert_eq!(tape.replay(|| fresh(7)), tape.hashes().to_vec());
        assert_eq!(tape.replay(|| fresh(7)), tape.replay(|| fresh(7)));
        assert!(tape.verify(|| fresh(7)).is_ok());
    }

    #[test]
    fn a_different_seed_diverges_at_its_first_tick() {
        let tape = record(7, 60);
        let error = tape
            .verify(|| fresh(8))
            .expect_err("a different seed is a different run");
        assert_eq!(error.tick(), Some(0), "the stream itself is already other");
        match error {
            VerifyError::Diverged {
                divergence,
                replayed,
            } => {
                assert_eq!(replayed.len(), 1, "the replay stops at the divergence");
                assert_eq!(replayed[0], divergence.actual);
                assert_ne!(divergence.expected, divergence.actual);
                assert_eq!(divergence.expected, tape.hashes()[0]);
            }
            other => panic!("expected a divergence, got {other}"),
        }
    }

    #[test]
    fn a_perturbed_tick_is_the_reported_tick() {
        let tape = record(7, 40);
        let mut poisoned = tape.clone();
        let edited_tick = 12;
        let before = poisoned.intents()[edited_tick];
        poisoned.intents_mut()[edited_tick] = Press {
            up: !before.up,
            spawn: before.spawn,
        };

        let error = poisoned
            .verify(|| fresh(7))
            .expect_err("the edited tick must change the world");
        assert_eq!(error.tick(), Some(edited_tick as u64));
        match error {
            VerifyError::Diverged {
                divergence,
                replayed,
            } => {
                assert_eq!(divergence.tick, edited_tick as u64);
                assert_eq!(divergence.expected, tape.hashes()[edited_tick]);
                assert_eq!(divergence.actual, replayed[edited_tick]);
                assert_ne!(divergence.expected, divergence.actual);
                assert_eq!(
                    replayed[..edited_tick],
                    tape.hashes()[..edited_tick],
                    "every tick before the edit agreed"
                );
            }
            other => panic!("expected a divergence, got {other}"),
        }
    }

    #[test]
    fn an_edited_tick_that_changes_nothing_still_verifies() {
        let tape = record(7, 40);
        let mut edited = tape.clone();
        // The first tick that pressed nothing: re-recording it as nothing is
        // the one edit that cannot change the world.
        let quiet = edited
            .intents()
            .iter()
            .position(|press| !press.up && !press.spawn)
            .expect("the policy leaves a tick quiet");
        edited.intents_mut()[quiet] = Press {
            up: false,
            spawn: false,
        };
        assert!(edited.verify(|| fresh(7)).is_ok());
    }

    #[test]
    fn a_tape_is_serialisable() {
        // A tape travels: written next to the save it came from, or attached to
        // a bug report. The crate carries no format, only the bounds.
        fn assert_serde<T: Serialize + serde::de::DeserializeOwned>() {}
        assert_serde::<Tape<Press>>();
    }

    #[test]
    fn a_tape_without_hashes_replays_but_cannot_be_verified() {
        let mut tape = Tape::<Press>::new(7);
        assert!(tape.is_empty());
        for tick in 0..8u64 {
            let input = press(tick, tick as u32);
            tape.record_intent(input);
            assert_eq!(tape.intent(tick), Some(&input));
        }
        assert_eq!(tape.ticks(), 8);
        assert_eq!(tape.replay(|| fresh(7)).len(), 8, "an intent tape replays");
        let error = tape
            .verify(|| fresh(7))
            .expect_err("with no hashes there is nothing to check");
        assert_eq!(
            error,
            VerifyError::UnhashedTicks {
                ticks: 8,
                hashes: 0
            }
        );
        assert_eq!(error.tick(), None);
    }

    /// A game that keeps its own action resources, the way games predating the
    /// tape do: `commit` is what feeds it.
    #[derive(Resource, Default)]
    struct Keys(Press);

    fn play_keys(keys: Res<Keys>, mut score: ResMut<Score>) {
        if keys.0.up {
            score.0 += 1;
        }
    }

    fn keys_world(seed: u64) -> Setup {
        let mut sim = Sim::with_default_step();
        sim.world.insert_resource(SimRng::seeded(seed));
        sim.world.init_resource::<Keys>();
        sim.world.init_resource::<Score>();
        sim.add_system(play_keys);
        let mut hasher = WorldHasher::new();
        hasher.register_resource::<Keys>(|keys, hash| hash.bool(keys.0.up));
        hasher.register_resource::<Score>(|score, hash| hash.u32(score.0));
        Setup { sim, hasher }
    }

    #[test]
    fn a_commit_closure_feeds_a_game_with_its_own_input_resources() {
        let mut fresh = || keys_world(3);
        let Setup { mut sim, hasher } = fresh();
        let mut tape = Tape::new(3);
        for tick in 0..12u64 {
            let input = Press {
                up: tick % 2 == 0,
                spawn: false,
            };
            tape.record_intent(input);
            sim.world.resource_mut::<Keys>().0 = input;
            sim.tick();
            tape.record_hash(hasher.hash(&sim.world));
        }

        let commit = |world: &mut World, input: &Press| world.resource_mut::<Keys>().0 = *input;
        assert_eq!(tape.replay_with(&mut fresh, commit), tape.hashes().to_vec());
        assert!(tape.verify_with(&mut fresh, commit).is_ok());
        assert!(
            tape.verify(&mut fresh).is_err(),
            "without the commit the game never sees the input"
        );
    }

    #[test]
    #[should_panic(expected = "no tick to belong to")]
    fn record_hash_refuses_a_second_hash_for_one_tick() {
        let mut tape = Tape::<Press>::new(7);
        tape.record_intent(Press {
            up: true,
            spawn: false,
        });
        tape.record_hash(1);
        tape.record_hash(2);
    }

    #[test]
    fn intent_carries_one_tick_of_input() {
        let mut intent = Intent::<Press>::default();
        assert_eq!(intent.get(), None);
        intent.set(Press {
            up: true,
            spawn: false,
        });
        assert_eq!(
            intent.get(),
            Some(&Press {
                up: true,
                spawn: false
            })
        );
        intent.clear();
        assert_eq!(intent.get(), None);
    }

    /// Rows are folded in entity order, not in the order `bevy_ecs` yields them.
    /// The assertion on the raw iteration orders is what keeps this test honest:
    /// if `bevy_ecs` ever stops iteration from depending on history, the test
    /// says so instead of passing for the wrong reason.
    #[test]
    fn the_digest_does_not_depend_on_iteration_order() {
        #[derive(Component)]
        struct Mark;

        fn rows(world: &mut World) -> Vec<u64> {
            let mut query = world.query::<(Entity, &Height)>();
            query
                .iter(world)
                .map(|(entity, _)| entity.to_bits())
                .collect()
        }

        let mut a = World::new();
        let first = a.spawn(Height(1.0)).id();
        a.spawn(Height(2.0));

        let mut b = World::new();
        let also_first = b.spawn(Height(1.0)).id();
        b.spawn(Height(2.0));
        // Moving the first row out of the table and back leaves it at the end:
        // same entity, same value, different position in `bevy_ecs`'s order.
        b.entity_mut(also_first).insert(Mark);
        b.entity_mut(also_first).remove::<Mark>();

        assert_eq!(first, also_first, "both worlds allocate the same ids");
        assert_ne!(
            rows(&mut a),
            rows(&mut b),
            "the hazard this guards against must be real"
        );

        let mut hasher = WorldHasher::new();
        hasher.register::<Height>(|height, hash| hash.f32(height.0));
        assert_eq!(hasher.hash(&a), hasher.hash(&b));
    }

    #[test]
    fn a_missing_resource_is_not_a_zero_valued_resource() {
        let mut hasher = WorldHasher::new();
        hasher.register_resource::<Score>(|score, hash| hash.u32(score.0));

        let mut present = World::new();
        present.insert_resource(Score(0));
        let absent = World::new();
        assert_ne!(hasher.hash(&present), hasher.hash(&absent));
    }

    #[test]
    fn a_registered_component_moves_the_digest() {
        let mut hasher = WorldHasher::new();
        hasher.register::<Height>(|height, hash| hash.f32(height.0));

        let mut one = World::new();
        one.spawn(Height(1.0));
        let mut other = World::new();
        other.spawn(Height(1.5));
        assert_ne!(hasher.hash(&one), hasher.hash(&other));

        let mut twin = World::new();
        twin.spawn(Height(1.0));
        assert_eq!(hasher.hash(&one), hasher.hash(&twin));

        // State nobody registered is state the digest cannot report: two worlds
        // with the same entities and the same registered components digest the
        // same even though an unregistered component differs.
        let mut quiet = World::new();
        quiet.spawn(Speed(1));
        let mut loud = World::new();
        loud.spawn(Speed(2));
        assert_eq!(hasher.hash(&quiet), hasher.hash(&loud));
    }

    #[test]
    fn register_hashed_folds_a_derived_hash() {
        #[derive(Component, Hash)]
        struct Tally {
            wins: u32,
            losses: u32,
        }

        let mut hasher = WorldHasher::new();
        hasher.register_hashed::<Tally>();

        let mut one = World::new();
        one.spawn(Tally { wins: 1, losses: 2 });
        let mut twin = World::new();
        twin.spawn(Tally { wins: 1, losses: 2 });
        let mut other = World::new();
        other.spawn(Tally { wins: 1, losses: 3 });
        assert_eq!(hasher.hash(&one), hasher.hash(&twin));
        assert_ne!(hasher.hash(&one), hasher.hash(&other));
    }

    #[test]
    fn an_empty_hasher_still_sees_the_entity_list() {
        let hasher = WorldHasher::new();
        let mut one = World::new();
        one.spawn_empty();
        assert_eq!(hasher.hash(&World::new()), hasher.hash(&World::new()));
        assert_ne!(hasher.hash(&one), hasher.hash(&World::new()));
    }
}
