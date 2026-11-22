//! Shutdown-time pipeline-cache persistence.

use repame_save::AppId;
use repose_render_wgpu::CallbackResources;

/// Flushes every named scene batch's pipeline cache on background threads.
/// Call this before dropping the device, from a `Quit` shutdown hook.
pub fn flush_pipeline_caches(
    app: AppId,
    device: &wgpu::Device,
    resources: &CallbackResources,
    ids: &[String],
) {
    for id in ids {
        repame_view3d::pipeline_cache_io::flush_blocking(id, app, device, resources);
    }
}
