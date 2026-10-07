//! First-gesture hook for host policies that gate audio behind one.
//!
//! Browsers refuse to start an `AudioContext` outside a user gesture, so an
//! engine opened at boot comes up suspended. Games call
//! [`on_first_user_gesture`] with [`repame_audio::Audio::unlock`] and get a
//! resumed device; there is no DOM code in the game.

/// Set once a real gesture has been seen, so the hook fires exactly once even
/// though the listeners stay attached.
static SEEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Run `f` on the first user gesture on this page.
///
/// The listener is attached to `window` for `pointerdown`, `touchstart` and
/// `keydown`, so mouse, touch and keyboard all arm the device. It stays
/// attached for the page's life; [`SEEN`] makes every later call a no-op, which
/// keeps the closure alive cheaply instead of tearing listeners down mid-gesture.
///
/// Off wasm this does nothing, so callers need no cfg.
pub fn on_first_user_gesture(f: impl Fn() + 'static) {
    #[cfg(target_arch = "wasm32")]
    install(f);
    #[cfg(not(target_arch = "wasm32"))]
    let _ = f;
}

#[cfg(target_arch = "wasm32")]
fn install(f: impl Fn() + 'static) {
    use std::sync::atomic::Ordering;

    use wasm_bindgen::closure::Closure;
    use wasm_bindgen::JsCast;

    let global = js_sys::global();
    let Ok(add) = js_sys::Reflect::get(&global, &wasm_bindgen::JsValue::from_str(
        "addEventListener",
    )) else {
        return;
    };
    let Some(add) = add.dyn_ref::<js_sys::Function>() else {
        return;
    };
    // Leaked into the page on purpose: the closure must outlive this call for
    // the listener to stay callable, and one 24-byte handler for the life of
    // the page is cheaper than the bookkeeping to remove it.
    let handler = Closure::wrap(Box::new(move || {
        if !SEEN.swap(true, Ordering::SeqCst) {
            f();
        }
    }) as Box<dyn FnMut()>)
    .into_js_value();
    for event in ["pointerdown", "touchstart", "keydown"] {
        let _ = add.call2(
            &global,
            &wasm_bindgen::JsValue::from_str(event),
            &handler,
        );
    }
}

/// Whether a user gesture has been seen. Games that would rather poll than
/// register a callback can drive [`unlock`](repame_audio::Audio::unlock)
/// from their frame loop; it is a cheap atomic load.
pub fn user_gesture_seen() -> bool {
    SEEN.load(std::sync::atomic::Ordering::SeqCst)
}