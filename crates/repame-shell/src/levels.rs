use std::collections::HashSet;
use std::hash::Hash;

use repose_core::input::PhysicalKey;
use repose_core::runtime::Scheduler;

pub fn apply_scheduler_levels<C>(
    staged_held: &mut HashSet<C>,
    lmb_held: &mut bool,
    rmb_held: &mut bool,
    window_focused: &mut bool,
    sched: &Scheduler,
    code_for: impl Fn(PhysicalKey) -> Option<C>,
    keys_for: impl Fn(&C) -> Option<&'static [PhysicalKey]>,
) where
    C: Eq + Hash + Clone,
{
    if !sched.window_focused {
        staged_held.clear();
        *lmb_held = false;
        *rmb_held = false;
        *window_focused = false;
        return;
    }
    *window_focused = true;
    repame_input::reconcile_held(staged_held, &sched.held_keys, code_for, keys_for);
    if !sched.mouse_primary {
        *lmb_held = false;
    }
    if !sched.mouse_secondary {
        *rmb_held = false;
    }
}
