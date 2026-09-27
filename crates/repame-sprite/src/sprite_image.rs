use std::fmt;
use std::sync::Arc;

use repose_core::locals::effective_density_scale;
use repose_core::{
    Color, ControlVisual, ControlVisualState, Dp, ImageAlignment, ImageFilter, ImageFit,
    ImageHandle, ImageHandleGuard, ImagePaintStyle, ImageSourceRect, Modifier, Rect,
    RenderContext, Scene, SceneNode, View,
};
use repose_ui::Box as UiBox;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpriteImageFrameDef {
    pub source_rect: ImageSourceRect,
    pub origin: [f32; 2],
    pub duration: f32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpriteImageDef {
    pub frames: Vec<SpriteImageFrameDef>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpriteImageError {
    EmptyDefinition,
    InvalidTextureSize,
    InvalidFrame(usize),
    InvalidTiming(usize),
    InvalidPixelBuffer,
}

impl fmt::Display for SpriteImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDefinition => f.write_str("sprite image definition has no frames"),
            Self::InvalidTextureSize => {
                f.write_str("sprite image texture dimensions must be nonzero")
            }
            Self::InvalidFrame(index) => write!(f, "sprite image frame {index} is invalid"),
            Self::InvalidTiming(index) => {
                write!(f, "sprite image frame {index} has invalid timing")
            }
            Self::InvalidPixelBuffer => f.write_str("sprite image RGBA buffer has the wrong size"),
        }
    }
}

impl std::error::Error for SpriteImageError {}

impl SpriteImageDef {
    pub fn new(frames: Vec<SpriteImageFrameDef>) -> Self {
        Self { frames }
    }

    pub fn single(frame_size: [u32; 2], origin: [f32; 2]) -> Self {
        Self {
            frames: vec![SpriteImageFrameDef {
                source_rect: ImageSourceRect::new(0, 0, frame_size[0], frame_size[1]),
                origin,
                duration: 0.0,
            }],
        }
    }

    pub fn horizontal_strip(
        frames: u32,
        frame_size: [u32; 2],
        origin: [f32; 2],
        fps: f32,
    ) -> Result<Self, SpriteImageError> {
        let duration = if fps > 0.0 { 1.0 / fps } else { 0.0 };
        if !fps.is_finite() || !duration.is_finite() {
            return Err(SpriteImageError::InvalidTiming(0));
        }
        let frames = (0..frames)
            .map(|frame| {
                let x = frame.saturating_mul(frame_size[0]);
                SpriteImageFrameDef {
                    source_rect: ImageSourceRect::new(x, 0, frame_size[0], frame_size[1]),
                    origin,
                    duration,
                }
            })
            .collect();
        Ok(Self { frames })
    }
}

struct SpriteImageAssetInner {
    guard: ImageHandleGuard,
    texture_size: [u32; 2],
    frames: Arc<[SpriteImageFrameDef]>,
}

#[derive(Clone)]
pub struct SpriteImageAsset {
    inner: Arc<SpriteImageAssetInner>,
}

impl fmt::Debug for SpriteImageAsset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpriteImageAsset")
            .field("handle", &self.handle())
            .field("texture_size", &self.texture_size())
            .field("frame_count", &self.frame_count())
            .finish()
    }
}

impl SpriteImageAsset {
    pub fn from_encoded(
        context: &RenderContext,
        bytes: impl Into<Vec<u8>>,
        texture_size: [u32; 2],
        definition: SpriteImageDef,
        srgb: bool,
    ) -> Result<Self, SpriteImageError> {
        Self::validate(texture_size, &definition)?;
        let guard = ImageHandleGuard::new(context);
        context.set_image_encoded(*guard, bytes.into(), srgb);
        Ok(Self::from_parts(guard, texture_size, definition))
    }

    pub fn from_rgba8(
        context: &RenderContext,
        texture_size: [u32; 2],
        rgba: Vec<u8>,
        definition: SpriteImageDef,
        srgb: bool,
    ) -> Result<Self, SpriteImageError> {
        Self::validate(texture_size, &definition)?;
        let expected = usize::try_from(texture_size[0])
            .ok()
            .and_then(|width| {
                usize::try_from(texture_size[1])
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or(SpriteImageError::InvalidPixelBuffer)?;
        if rgba.len() != expected {
            return Err(SpriteImageError::InvalidPixelBuffer);
        }
        let guard = ImageHandleGuard::new(context);
        context.set_image_rgba8(*guard, texture_size[0], texture_size[1], rgba, srgb);
        Ok(Self::from_parts(guard, texture_size, definition))
    }

    fn from_parts(
        guard: ImageHandleGuard,
        texture_size: [u32; 2],
        definition: SpriteImageDef,
    ) -> Self {
        Self {
            inner: Arc::new(SpriteImageAssetInner {
                guard,
                texture_size,
                frames: definition.frames.into(),
            }),
        }
    }

    fn validate(
        texture_size: [u32; 2],
        definition: &SpriteImageDef,
    ) -> Result<(), SpriteImageError> {
        if texture_size[0] == 0 || texture_size[1] == 0 {
            return Err(SpriteImageError::InvalidTextureSize);
        }
        if definition.frames.is_empty() {
            return Err(SpriteImageError::EmptyDefinition);
        }
        for (index, frame) in definition.frames.iter().enumerate() {
            let Some(right) = frame.source_rect.x.checked_add(frame.source_rect.width) else {
                return Err(SpriteImageError::InvalidFrame(index));
            };
            let Some(bottom) = frame.source_rect.y.checked_add(frame.source_rect.height) else {
                return Err(SpriteImageError::InvalidFrame(index));
            };
            if frame.source_rect.is_empty()
                || right > texture_size[0]
                || bottom > texture_size[1]
                || !frame.origin[0].is_finite()
                || !frame.origin[1].is_finite()
            {
                return Err(SpriteImageError::InvalidFrame(index));
            }
            if !frame.duration.is_finite() || frame.duration < 0.0 {
                return Err(SpriteImageError::InvalidTiming(index));
            }
        }
        Ok(())
    }

    pub fn handle(&self) -> ImageHandle {
        *self.inner.guard
    }

    pub fn texture_size(&self) -> [u32; 2] {
        self.inner.texture_size
    }

    pub fn frame_count(&self) -> usize {
        self.inner.frames.len()
    }

    pub fn frame(&self, frame: i32) -> Option<SpriteImage> {
        if self.inner.frames.is_empty() {
            return None;
        }
        let index = frame.rem_euclid(self.inner.frames.len() as i32) as usize;
        let def = self.inner.frames[index];
        Some(SpriteImage {
            asset: self.clone(),
            source_rect: def.source_rect,
            size: [def.source_rect.width as f32, def.source_rect.height as f32],
            origin: def.origin,
            duration: def.duration,
        })
    }
}

#[derive(Clone)]
pub struct SpriteImage {
    asset: SpriteImageAsset,
    source_rect: ImageSourceRect,
    size: [f32; 2],
    origin: [f32; 2],
    duration: f32,
}

impl fmt::Debug for SpriteImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpriteImage")
            .field("handle", &self.handle())
            .field("source_rect", &self.source_rect)
            .field("size", &self.size)
            .field("origin", &self.origin)
            .field("duration", &self.duration)
            .finish()
    }
}

impl SpriteImage {
    pub fn handle(&self) -> ImageHandle {
        self.asset.handle()
    }

    pub fn source_rect(&self) -> ImageSourceRect {
        self.source_rect
    }

    pub fn size(&self) -> [f32; 2] {
        self.size
    }

    pub fn origin(&self) -> [f32; 2] {
        self.origin
    }

    pub fn anchor(&self) -> [f32; 2] {
        if self.size[0] <= 0.0 || self.size[1] <= 0.0 {
            [0.0, 0.0]
        } else {
            [self.origin[0] / self.size[0], self.origin[1] / self.size[1]]
        }
    }

    pub fn duration(&self) -> f32 {
        self.duration
    }

    pub fn view(&self, modifier: Modifier) -> View {
        let has_size = modifier.size.is_some()
            || modifier.width.is_some()
            || modifier.height.is_some()
            || modifier.fill_max.is_some()
            || modifier.fill_max_w.is_some()
            || modifier.fill_max_h.is_some();
        let modifier = if has_size {
            modifier
        } else {
            modifier.size(Dp(self.size[0]), Dp(self.size[1]))
        };
        let visual = self.control_visual(Color::WHITE);
        UiBox(modifier.painter(move |scene, rect, alpha| {
            visual.paint(
                scene,
                rect,
                ControlVisualState {
                    alpha,
                    ..ControlVisualState::default()
                },
            );
        }))
    }

    pub fn control_visual(&self, tint: Color) -> ControlVisual {
        let _asset = self.asset.clone();
        let handle = self.handle();
        let source = self.source_rect;
        ControlVisual::custom(move |scene, rect, state| {
            let _ = &_asset;
            let alpha = state.alpha.clamp(0.0, 1.0);
            let tint = Color(tint.0, tint.1, tint.2, (tint.3 as f32 * alpha) as u8);
            scene.nodes.push(SceneNode::Image {
                rect,
                handle,
                tint,
                style: ImagePaintStyle {
                    fit: ImageFit::FillBounds,
                    filter: ImageFilter::Nearest,
                    source_rect: Some(source),
                    alignment: ImageAlignment::Center,
                },
            });
        })
    }

    pub fn rotated_control_visual(&self, tint: Color, quarter_turns: i32) -> ControlVisual {
        let quarter_turns = quarter_turns.rem_euclid(4);
        if quarter_turns == 0 {
            return self.control_visual(tint);
        }
        let _asset = self.asset.clone();
        let handle = self.handle();
        let source = self.source_rect;
        ControlVisual::custom(
            move |scene: &mut Scene, rect: Rect, state: ControlVisualState| {
                let _ = &_asset;
                let angle = quarter_turns as f32 * std::f32::consts::FRAC_PI_2;
                let (sin, cos) = angle.sin_cos();
                let center_x = rect.x + rect.w * 0.5;
                let center_y = rect.y + rect.h * 0.5;
                let transform = repose_core::Transform {
                    translate_x: center_x - (cos * center_x - sin * center_y),
                    translate_y: center_y - (sin * center_x + cos * center_y),
                    rotate: angle,
                    origin_x: 0.0,
                    origin_y: 0.0,
                    ..repose_core::Transform::identity()
                };
                let image_rect = if quarter_turns % 2 == 1 {
                    Rect {
                        x: center_x - rect.h * 0.5,
                        y: center_y - rect.w * 0.5,
                        w: rect.h,
                        h: rect.w,
                    }
                } else {
                    rect
                };
                let alpha = state.alpha.clamp(0.0, 1.0);
                let tint = Color(tint.0, tint.1, tint.2, (tint.3 as f32 * alpha) as u8);
                scene.nodes.push(SceneNode::PushTransform { transform });
                scene.nodes.push(SceneNode::Image {
                    rect: image_rect,
                    handle,
                    tint,
                    style: ImagePaintStyle {
                        fit: ImageFit::FillBounds,
                        filter: ImageFilter::Nearest,
                        source_rect: Some(source),
                        alignment: ImageAlignment::Center,
                    },
                });
                scene.nodes.push(SceneNode::PopTransform);
            },
        )
    }

    pub fn prefix_control_visual(&self, tint: Color) -> ControlVisual {
        let _asset = self.asset.clone();
        let handle = self.handle();
        let source = self.source_rect;
        let natural_width = self.size[0].max(1.0);
        ControlVisual::custom(
            move |scene: &mut Scene, rect: Rect, state: ControlVisualState| {
                let _ = &_asset;
                let density = effective_density_scale().max(1e-6);
                let logical_width = rect.w / density;
                let visible = if logical_width.is_finite() && logical_width > 0.0 {
                    ((logical_width / natural_width).clamp(0.0, 1.0) * source.width as f32)
                        .floor()
                        .max(1.0) as u32
                } else {
                    source.width
                };
                let visible = visible.min(source.width);
                let alpha = state.alpha.clamp(0.0, 1.0);
                let tint = Color(tint.0, tint.1, tint.2, (tint.3 as f32 * alpha) as u8);
                scene.nodes.push(SceneNode::Image {
                    rect,
                    handle,
                    tint,
                    style: ImagePaintStyle {
                        fit: ImageFit::FillBounds,
                        filter: ImageFilter::Nearest,
                        source_rect: Some(ImageSourceRect::new(
                            source.x,
                            source.y,
                            visible,
                            source.height,
                        )),
                        alignment: ImageAlignment::Center,
                    },
                });
            },
        )
    }

    pub fn subimage(&self, source_rect: ImageSourceRect) -> Option<Self> {
        let right = source_rect.x.checked_add(source_rect.width)?;
        let bottom = source_rect.y.checked_add(source_rect.height)?;
        let [width, height] = self.asset.texture_size();
        if source_rect.is_empty() || right > width || bottom > height {
            return None;
        }
        Some(Self {
            asset: self.asset.clone(),
            source_rect,
            size: [source_rect.width as f32, source_rect.height as f32],
            origin: self.origin,
            duration: self.duration,
        })
    }

    pub fn relative_subimage(&self, x: u32, y: u32, width: u32, height: u32) -> Option<Self> {
        let source_x = self.source_rect.x.checked_add(x)?;
        let source_y = self.source_rect.y.checked_add(y)?;
        self.subimage(ImageSourceRect::new(source_x, source_y, width, height))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use repose_core::{ControlVisualState, RenderCommand};

    #[test]
    fn sprite_frame_becomes_source_rect_control_visual() {
        let context = RenderContext::new();
        let definition = SpriteImageDef::horizontal_strip(2, [2, 2], [1.0, 1.0], 8.0)
            .expect("strip should be valid");
        let (handle, visual) = {
            let asset = SpriteImageAsset::from_rgba8(
                &context,
                [4, 2],
                vec![255; 4 * 2 * 4],
                definition,
                true,
            )
            .expect("asset should be valid");
            let frame = asset.frame(1).expect("frame should exist");
            (frame.handle(), frame.control_visual(Color::WHITE))
        };
        let _ = context.drain();
        let mut scene = Scene::default();
        visual.paint(
            &mut scene,
            Rect {
                x: 3.0,
                y: 5.0,
                w: 20.0,
                h: 10.0,
            },
            ControlVisualState::default(),
        );
        let Some(SceneNode::Image {
            rect,
            style:
                ImagePaintStyle {
                    source_rect, filter, ..
                },
            ..
        }) = scene.nodes.first()
        else {
            panic!("expected image node");
        };
        assert_eq!(rect.w, 20.0);
        assert_eq!(*source_rect, Some(ImageSourceRect::new(2, 0, 2, 2)));
        assert_eq!(*filter, ImageFilter::Nearest);
        assert!(context.drain().is_empty());
        drop(visual);
        assert!(matches!(
            context.drain().as_slice(),
            [RenderCommand::RemoveImage { handle: removed }] if *removed == handle
        ));
    }

    #[test]
    fn rotated_visual_swaps_quarter_turn_rect() {
        let context = RenderContext::new();
        let definition = SpriteImageDef::single([8, 2], [4.0, 1.0]);
        let visual =
            SpriteImageAsset::from_rgba8(&context, [8, 2], vec![255; 8 * 2 * 4], definition, true)
                .expect("asset should be valid")
                .frame(0)
                .expect("frame should exist")
                .rotated_control_visual(Color::WHITE, 1);
        let mut scene = Scene::default();
        visual.paint(
            &mut scene,
            Rect {
                x: 10.0,
                y: 20.0,
                w: 20.0,
                h: 80.0,
            },
            ControlVisualState::default(),
        );
        assert_eq!(scene.nodes.len(), 3);
        assert!(matches!(
            scene.nodes[1],
            SceneNode::Image { rect, .. } if rect == Rect { x: -20.0, y: 50.0, w: 80.0, h: 20.0 }
        ));
        assert!(matches!(&scene.nodes[2], SceneNode::PopTransform));
    }

    #[test]
    fn prefix_visual_uses_logical_control_width() {
        let context = RenderContext::new();
        let definition = SpriteImageDef::horizontal_strip(1, [2, 2], [1.0, 1.0], 8.0)
            .expect("strip should be valid");
        let visual =
            SpriteImageAsset::from_rgba8(&context, [2, 2], vec![255; 2 * 2 * 4], definition, true)
                .expect("asset should be valid")
                .frame(0)
                .expect("frame should exist")
                .prefix_control_visual(Color::WHITE);
        let mut scene = Scene::default();
        repose_core::locals::with_density(repose_core::locals::Density { scale: 2.0 }, || {
            visual.paint(
                &mut scene,
                Rect {
                    x: 0.0,
                    y: 0.0,
                    w: 2.0,
                    h: 2.0,
                },
                ControlVisualState::default(),
            );
        });
        let Some(SceneNode::Image {
            style: ImagePaintStyle { source_rect, .. },
            ..
        }) = scene.nodes.first()
        else {
            panic!("expected image node");
        };
        assert_eq!(*source_rect, Some(ImageSourceRect::new(0, 0, 1, 2)));
    }
}
