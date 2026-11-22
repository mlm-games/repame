//! Automatic pipeline-cache persistence through `repame-save`.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use repame_save::{AppId, PipelineCacheStore};
use repose_render_wgpu::CallbackResources;

use crate::{load_pipeline_cache, pipeline_cache_data};

fn loaded() -> &'static Mutex<HashSet<String>> {
    static LOADED: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    LOADED.get_or_init(|| Mutex::new(HashSet::new()))
}

fn cache_key(device: &wgpu::Device, id: &str) -> Option<String> {
    wgpu::util::pipeline_cache_key(&device.adapter_info()).map(|key| format!("{key}_{id}"))
}

/// Loads the seed once per app/adapter/batch, before pipelines are created.
pub fn prepare(id: &str, app: AppId, device: &wgpu::Device, resources: &mut CallbackResources) {
    let Some(key) = cache_key(device, id) else {
        return;
    };
    let identity = format!("{app:?}|{key}");
    if !loaded()
        .lock()
        .map(|mut set| set.insert(identity))
        .unwrap_or(true)
    {
        return;
    }
    if let Some(bytes) = PipelineCacheStore::new(app).load(&key) {
        load_pipeline_cache(id, bytes, resources);
    }
}

fn write_cache(app: AppId, key: String, bytes: Vec<u8>, label: String) {
    if let Err(err) = PipelineCacheStore::new(app).save(&key, &bytes) {
        log::warn!("pipeline cache[{label}]: save failed: {err}");
    }
}

/// Starts a background save and returns its handle so shutdown can join it.
/// On wasm there are no threads, so the write is synchronous.
pub fn flush(
    id: &str,
    app: AppId,
    device: &wgpu::Device,
    resources: &CallbackResources,
) -> Option<std::thread::JoinHandle<()>> {
    let key = cache_key(device, id)?;
    let bytes = pipeline_cache_data(id, resources)?;
    let label = id.to_string();
    #[cfg(target_arch = "wasm32")]
    {
        write_cache(app, key, bytes, label);
        None
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        Some(std::thread::spawn(move || write_cache(app, key, bytes, label)))
    }
}

/// Synchronous save for shutdown hooks, where process exit would otherwise
/// kill a detached writer.
pub fn flush_blocking(id: &str, app: AppId, device: &wgpu::Device, resources: &CallbackResources) {
    let Some(key) = cache_key(device, id) else {
        return;
    };
    let Some(bytes) = pipeline_cache_data(id, resources) else {
        return;
    };
    write_cache(app, key, bytes, id.to_string());
}
