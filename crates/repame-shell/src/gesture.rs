//! First-gesture hook for host policies that gate audio behind one.
//!
//! Browsers refuse to start an `AudioContext` outside a user gesture, so an
//! engine opened at boot comes up suspended. Games call
//! [`on_first_user_gesture`] with [`repame_audio::Audio::unlock`] and get a
//! resumed device; there is no DOM code in the game.

/// Set once a real gesture has been seen, so the hook fires exactly once even
/// though the listeners stay attached.
static SEEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Guards listener installation, so polling and callback registration can
/// both happen in either order.
#[cfg(target_arch = "wasm32")]
static INSTALLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Late-bound callback, so a game that registered [`on_first_user_gesture`]
/// after something already triggered [`install`] still gets its call.
#[cfg(target_arch = "wasm32")]
static CALLBACK: std::sync::Mutex<Option<Box<dyn Fn() + Send + Sync>>> =
    std::sync::Mutex::new(None);

/// Run `f` on the first user gesture on this page.
///
/// The listener is attached to `window` for `pointerdown`, `touchstart` and
/// `keydown`, so mouse, touch and keyboard all arm the device. It stays
/// attached for the page's life; [`SEEN`] makes every later call a no-op, which
/// keeps the closure alive cheaply instead of tearing listeners down mid-gesture.
///
/// Off wasm this does nothing, so callers need no cfg.
pub fn on_first_user_gesture(f: impl Fn() + Send + Sync + 'static) {
    #[cfg(target_arch = "wasm32")]
    {
        let mut slot = CALLBACK.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_none() {
            *slot = Some(Box::new(f));
        }
        install();
    }
    #[cfg(not(target_arch = "wasm32"))]
    let _ = f;
}

#[cfg(not(target_arch = "wasm32"))]
fn install() {}

/// Whether a user gesture has been seen.
///
/// Installs the listeners on first call, so a game that prefers polling its
/// frame loop over registering a callback needs no separate setup step.
pub fn user_gesture_seen() -> bool {
    install();
    SEEN.load(std::sync::atomic::Ordering::SeqCst)
}

#[cfg(target_arch = "wasm32")]
fn install() {
    use std::sync::atomic::Ordering;

    use wasm_bindgen::closure::Closure;
    use wasm_bindgen::JsCast;

    if INSTALLED.swap(true, Ordering::SeqCst) {
        return;
    }
    let global = js_sys::global();
    let Ok(add) = js_sys::Reflect::get(
        &global,
        &wasm_bindgen::JsValue::from_str("addEventListener"),
    ) else {
        return;
    };
    let Some(add) = add.dyn_ref::<js_sys::Function>() else {
        return;
    };
    // Leaked into the page on purpose: the closure must outlive this call for
    // the listener to stay callable, and one 24-byte handler for the life of
    // the page is cheaper than the bookkeeping to remove it.
    let handler = Closure::wrap(Box::new(|| {
        if !SEEN.swap(true, Ordering::SeqCst)
            && let Some(f) = CALLBACK.lock().unwrap_or_else(|e| e.into_inner()).take()
        {
            f();
        }
    }) as Box<dyn FnMut()>)
    .into_js_value();
    for event in ["pointerdown", "touchstart", "keydown"] {
        let _ = add.call2(&global, &wasm_bindgen::JsValue::from_str(event), &handler);
    }
}