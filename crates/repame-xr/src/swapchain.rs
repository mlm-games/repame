//! The render seam: runtime-owned eye images handed to the renderer, and the
//! handles the renderer must give back when a session is created.
//!
//! This crate never touches a GPU. A backend acquires a runtime-owned image per
//! eye and describes it as [`EyeImage`]; the app's render layer — `repame-view3d`
//! on wgpu, or whatever replaces it — implements [`StereoSubmit`], imports the
//! image through the handle, and draws. Packing is the runtime's business, so
//! the image carries its format, size, sample count and the viewport that eye
//! occupies instead of assuming two half-width images.
//!
//! [`SwapchainImageHandle`] holds a raw native token because that is exactly
//! what the runtime owns. It is only ever filled in from a real image the
//! backend enumerated; no backend synthesizes one.

use std::fmt;

use crate::stereo::{BlendMode, Eye, PixelFormat};

/// Which graphics API a native image token belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GraphicsApi {
    /// A `VkImage`.
    Vulkan,
    /// An `ID3D11Texture2D`.
    D3D11,
    /// An `ID3D12Resource`.
    D3D12,
    /// A desktop OpenGL texture name.
    OpenGl,
    /// An OpenGL ES texture name.
    OpenGlEs,
    /// An `MTLTexture`.
    Metal,
    /// A WebGL texture name.
    WebGl,
}

impl GraphicsApi {
    /// Short name, for diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            GraphicsApi::Vulkan => "vulkan",
            GraphicsApi::D3D11 => "d3d11",
            GraphicsApi::D3D12 => "d3d12",
            GraphicsApi::OpenGl => "opengl",
            GraphicsApi::OpenGlEs => "opengles",
            GraphicsApi::Metal => "metal",
            GraphicsApi::WebGl => "webgl",
        }
    }
}

/// One image the runtime owns, addressed the way the API it came from wants it.
///
/// For a layered API the token is the image and [`SwapchainImageHandle::array_index`]
/// the layer inside it, so a stereo swapchain with two layers yields one token
/// and two indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SwapchainImageHandle {
    /// API the token belongs to.
    pub api: GraphicsApi,
    /// `VkImage`, `ID3D12Resource`, `MTLTexture` or `GLuint` value.
    pub handle: u64,
    /// Array layer inside `handle`.
    pub array_index: u32,
}

/// What the renderer must draw for one eye this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EyeImage {
    /// Which eye the image belongs to.
    pub eye: Eye,
    /// Image size in pixels.
    pub size: [u32; 2],
    /// Image pixel format.
    pub format: PixelFormat,
    /// Sample count: 1 for a plain image, higher for a multisampled one.
    pub sample_count: u32,
    /// `[x, y, width, height]` of this eye inside the image.
    pub viewport: [u32; 4],
    /// Native token the renderer imports.
    pub handle: SwapchainImageHandle,
    /// Backend-internal swapchain identity, so a driver knows which of its
    /// resources to sample.
    pub swapchain: u64,
    /// Index of the image inside the swapchain, for image-content tracking.
    pub image_index: u32,
}

impl EyeImage {
    /// True when every field describes something drawable.
    pub fn is_usable(&self) -> bool {
        self.size[0] > 0
            && self.size[1] > 0
            && self.viewport[2] > 0
            && self.viewport[3] > 0
            && self.sample_count > 0
    }
}

/// The image shape a session settled on, so an app can allocate matching
/// resources before the first frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SwapchainFormat {
    /// Pixel format every eye image uses.
    pub format: PixelFormat,
    /// Sample count every eye image uses.
    pub sample_count: u32,
    /// Compositing mode the session runs in.
    pub blend_mode: BlendMode,
}

/// Raw Vulkan handles a session needs. Every field is an object the render layer
/// already created; this crate neither owns nor destroys any of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VulkanBinding {
    /// `VkInstance`, as a pointer value.
    pub instance: u64,
    /// `VkPhysicalDevice`, as a pointer value.
    pub physical_device: u64,
    /// `VkDevice`, as a pointer value.
    pub device: u64,
    /// Queue family the queue below belongs to.
    pub queue_family_index: u32,
    /// Queue the runtime will submit on.
    pub queue_index: u32,
}

/// The graphics device a session renders with. The renderer supplies this; XR
/// cannot pick one on the app's behalf.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphicsBinding {
    /// Vulkan, the backend `repame-view3d` uses on Linux, Windows and Android.
    Vulkan(VulkanBinding),
}

impl GraphicsBinding {
    /// API this binding is for.
    pub fn api(&self) -> GraphicsApi {
        match self {
            GraphicsBinding::Vulkan(_) => GraphicsApi::Vulkan,
        }
    }
}

/// A stereo frame ready to draw: the view configuration plus one image per eye.
#[derive(Clone, Copy, Debug)]
pub struct StereoFrame<'a> {
    /// Views and projections for this frame.
    pub config: &'a crate::ViewConfig,
    /// Per-eye images, in acquisition order.
    pub images: &'a [EyeImage],
}

impl StereoFrame<'_> {
    /// Hand every eye whose view is tracked to `sink`.
    ///
    /// An eye the runtime is not tracking has no view matrix, so it is skipped
    /// rather than drawn from a stand-in pose; the runtime shows the last frame
    /// it accepted for that eye.
    pub fn submit<S: StereoSubmit>(&self, sink: &mut S) -> Result<(), S::Error> {
        sink.begin_stereo(self)?;
        for image in self.images {
            if let Some(view) = self.config.view(image.eye)
                && view.view.is_some()
            {
                sink.submit_eye(image, view)?;
            }
        }
        sink.end_stereo()
    }

    /// Eyes in this frame that have both an image and a tracked view.
    pub fn drawable_eyes(&self) -> Vec<Eye> {
        self.images
            .iter()
            .filter(|image| {
                self.config
                    .view(image.eye)
                    .is_some_and(|view| view.view.is_some())
            })
            .map(|image| image.eye)
            .collect()
    }
}

/// What a render layer implements so a backend can drive it.
///
/// The backend holds the images only for the duration of the frame, so the
/// implementation must finish with them before [`StereoSubmit::end_stereo`]
/// returns.
pub trait StereoSubmit {
    /// Error the render layer reports.
    type Error;

    /// Called once before any eye. Default does nothing.
    fn begin_stereo(&mut self, _frame: &StereoFrame<'_>) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Draw one eye's image, viewed through `view`.
    fn submit_eye(
        &mut self,
        image: &EyeImage,
        view: &crate::ViewSetup,
    ) -> Result<(), Self::Error>;

    /// Called once after the last eye. Default does nothing.
    fn end_stereo(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// Result of driving one frame through [`crate::XrSession::frame`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameOutcome {
    /// True when the runtime asked for a frame and images were handed over.
    pub rendered: bool,
    /// The runtime's own `should_render` for this frame.
    pub should_render: bool,
    /// Predicted display time, in nanoseconds on the runtime's monotonic clock.
    pub display_time_ns: i64,
    /// Views the runtime located.
    pub views: usize,
}

/// Either the runtime or the render layer failed.
#[derive(Clone, Debug, PartialEq)]
pub enum FrameError<E> {
    /// An XR runtime call failed.
    Xr(crate::XrError),
    /// The render layer failed.
    Sink(E),
}

impl<E> From<crate::XrError> for FrameError<E> {
    fn from(error: crate::XrError) -> Self {
        FrameError::Xr(error)
    }
}

impl<E: fmt::Display> fmt::Display for FrameError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FrameError::Xr(error) => write!(f, "{error}"),
            FrameError::Sink(error) => write!(f, "xr render layer: {error}"),
        }
    }
}
