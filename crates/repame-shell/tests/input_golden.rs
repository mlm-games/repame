//! Golden input contracts: codec, held repair, focus cancel, lifecycle.
//!
//! Pins desired behavior across `repame-input` + `repame-shell` so the
//! NT migration hazards cannot regress silently:
//! - codec: `Key` vs `Physical` survive save/load as distinct entries.
//! - repair: a missed key-up drops the staged level, never synthesizes edges.
//! - cancel: focus loss clears clicks, mouse edges, RMB latch, touch.
//! - lifecycle: `consume` expires at tick end; physical capture binds
//!   physical and drives the action; `replace_map` swaps evaluation.

use std::collections::HashSet;

use glam::Vec2;
use repame_input::{
    ActionMap, ActionState, Binding, KeymapDevice, KeymapEntry, RemapSession,
    decode_keymap_entry, encode_keymap_entry,
};
use repame_shell::Staging;
use repose_core::input::{Key, Modifiers, PhysicalKey, PointerButton};
use repose_core::runtime::Scheduler;
use repose_core::shortcuts::KeyChord;

fn chord(key: Key) -> KeymapEntry {
    KeymapEntry::Key(KeyChord::new(key, Modifiers::default()))
}

fn focused_scheduler() -> Scheduler {
    let mut sched = Scheduler::new();
    sched.window_focused = true;
    sched
}

#[test]
fn codec_keeps_logical_and_physical_distinct() {
    let unambiguous = [
        chord(Key::Character('w')),
        chord(Key::Character('1')),
        KeymapEntry::Physical(PhysicalKey::KeyW),
        KeymapEntry::Physical(PhysicalKey::Digit1),
        KeymapEntry::Physical(PhysicalKey::Minus),
        KeymapEntry::Mouse(PointerButton::Primary),
        KeymapEntry::Mouse(PointerButton::Secondary),
        KeymapEntry::Pad(repose_core::input::GamepadButton::South),
        KeymapEntry::Axis {
            axis: repose_core::input::GamepadAxis::LeftTrigger,
            threshold: 0.5,
        },
        KeymapEntry::None,
    ];
    for entry in unambiguous {
        let text = encode_keymap_entry(&entry);
        assert_eq!(decode_keymap_entry(&text), entry, "round trip {text:?}");
    }

    let named_keys = [
        Key::Space,
        Key::Tab,
        Key::Enter,
        Key::Escape,
        Key::ArrowUp,
        Key::ArrowDown,
        Key::ArrowLeft,
        Key::ArrowRight,
        Key::ShiftLeft,
        Key::ShiftRight,
    ];
    let named_positions = [
        PhysicalKey::Space,
        PhysicalKey::Tab,
        PhysicalKey::Enter,
        PhysicalKey::Escape,
        PhysicalKey::ArrowUp,
        PhysicalKey::ArrowDown,
        PhysicalKey::ArrowLeft,
        PhysicalKey::ArrowRight,
        PhysicalKey::ShiftLeft,
        PhysicalKey::ShiftRight,
        PhysicalKey::KeyW,
    ];
    for (key, position) in named_keys.into_iter().zip(named_positions) {
        let logical = chord(key.clone());
        let physical = KeymapEntry::Physical(position);
        let ltext = encode_keymap_entry(&logical);
        let ptext = encode_keymap_entry(&physical);
        assert_ne!(ltext, ptext, "wire forms must differ for {key:?}");
        assert_eq!(decode_keymap_entry(&ltext), logical, "logical {ltext:?}");
        assert_eq!(decode_keymap_entry(&ptext), physical, "physical {ptext:?}");
    }

    let shifted = KeymapEntry::Key(KeyChord::new(
        Key::Space,
        Modifiers {
            shift: true,
            ..Modifiers::default()
        },
    ));
    let text = encode_keymap_entry(&shifted);
    assert_eq!(decode_keymap_entry(&text), shifted, "modifiers survive");

    assert_eq!(
        decode_keymap_entry("Space"),
        chord(Key::Space),
        "legacy bare Space is the logical default (NT Swap)"
    );
    assert_eq!(
        decode_keymap_entry("phys:Space"),
        KeymapEntry::Physical(PhysicalKey::Space)
    );
    assert_eq!(
        decode_keymap_entry("KeyW"),
        KeymapEntry::Physical(PhysicalKey::KeyW),
        "legacy bare KeyW stays physical"
    );
}

#[test]
fn missed_key_release_drops_level_without_synthesizing_edge() {
    let mut staging = Staging::default();
    staging.window_focused = true;
    staging.stage_physical(PhysicalKey::KeyW, true, false);
    assert!(staging.held.contains(&PhysicalKey::KeyW));
    assert_eq!(staging.take_edges().len(), 1);

    staging.feed_polled(&focused_scheduler());
    assert!(
        !staging.held.contains(&PhysicalKey::KeyW),
        "missed W release must drop the level"
    );
    assert!(
        staging.take_edges().is_empty(),
        "repair must never synthesize press edges"
    );
}

#[test]
fn missed_mouse_release_drops_level() {
    let mut staging = Staging::default();
    staging.window_focused = true;
    staging.pick_down(PointerButton::Primary);
    assert!(staging.lmb_held);

    staging.feed_polled(&focused_scheduler());
    assert!(!staging.lmb_held, "missed button-up must drop LMB");
}

#[test]
fn focus_loss_cancels_all_pending_input() {
    let mut staging = Staging::default();
    staging.window_focused = true;
    staging.stage_physical(PhysicalKey::KeyW, true, false);
    staging.take_edges();
    staging.pick_down(PointerButton::Primary);
    staging.pick_down(PointerButton::Secondary);
    staging.stage_click(Vec2::new(10.0, 10.0), [10.0, 10.0], 1.0);
    staging.touch_down(7, Vec2::new(100.0, 100.0));

    staging.set_window_focused(false);

    assert!(staging.held.is_empty());
    assert!(staging.take_edges().is_empty());
    assert!(!staging.lmb_held && !staging.rmb_held);
    assert!(staging.clicks.is_empty(), "staged clicks must not survive alt-tab");
    assert!(
        staging.mouse_edges.is_empty(),
        "mouse edges must not survive alt-tab"
    );
    assert!(
        !staging.take_rmb_down(),
        "RMB latch must not survive alt-tab"
    );
    assert!(
        staging.touch_active.is_empty(),
        "touch contacts must not survive alt-tab"
    );
    assert!(
        staging.capture_pending_physical.is_none()
            && staging.capture_pending_key.is_none()
            && staging.capture_pending_mouse.is_none()
    );
}

#[test]
fn consume_expires_at_tick_end() {
    let mut map = ActionMap::new();
    let space = KeyChord::new(Key::Space, Modifiers::default());
    map.bind("jump", Binding::Key(space.clone()));
    let mut state = ActionState::new(map);
    state.key(&space, true);
    assert!(state.consume(&"jump"));
    assert!(!state.pressed(&"jump"));
    state.end_tick();
    assert!(state.pressed(&"jump"), "consume must not leak past tick end");
}

#[test]
fn physical_capture_binds_physical_and_drives_action() {
    let mut session: RemapSession<&str> = RemapSession::default();
    session.begin("jump", KeymapDevice::KeyboardMouse);
    assert!(session.resolve_physical(PhysicalKey::KeyZ));
    assert_eq!(
        session.map.keyboard(&"jump"),
        KeymapEntry::Physical(PhysicalKey::KeyZ)
    );

    let map = session.map.to_action_map();
    assert!(
        map.bindings_for(&"jump")
            .contains(&Binding::Physical(PhysicalKey::KeyZ))
    );
    let mut state: ActionState<&str> = ActionState::new(map);
    state.physical(PhysicalKey::KeyZ, true);
    assert!(state.pressed(&"jump"));
    assert!(state.just_pressed(&"jump"));
    state.end_tick();
    state.physical(PhysicalKey::KeyZ, false);
    assert!(!state.pressed(&"jump"));
    assert!(state.just_released(&"jump"));
}

#[test]
fn replace_map_swaps_live_evaluation() {
    let space = KeyChord::new(Key::Space, Modifiers::default());
    let mut first = ActionMap::new();
    first.bind("jump", Binding::Key(space.clone()));
    let mut state: ActionState<&str> = ActionState::new(first);
    state.key(&space, true);
    assert!(state.pressed(&"jump"));
    state.key(&space, false);
    state.end_tick();

    let chord_e = KeyChord::new(Key::Character('e'), Modifiers::default());
    let mut second = ActionMap::new();
    second.bind("jump", Binding::Key(chord_e.clone()));
    state.replace_map(second);
    state.key(&space, true);
    assert!(!state.pressed(&"jump"), "old binding must go dead");
    assert!(!state.just_pressed(&"jump"));
    state.key(&space, false);
    state.key(&chord_e, true);
    assert!(state.pressed(&"jump"));
    assert!(state.just_pressed(&"jump"));
}

#[test]
fn reconcile_never_synthesizes_edges_for_levels_only_repair() {
    let mut staged: HashSet<u8> = HashSet::from([1]);
    let polled: HashSet<PhysicalKey> = HashSet::from([PhysicalKey::Space]);
    repame_input::reconcile_held(
        &mut staged,
        &polled,
        |_| Some(2u8),
        |code| {
            if *code == 1 {
                Some(&[][..])
            } else {
                Some(&[PhysicalKey::Space][..])
            }
        },
    );
    assert!(!staged.contains(&1));
    assert!(staged.contains(&2));
}
