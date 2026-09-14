//! glTF embedded-image decode: buffer-view bytes in, tight sRGB RGBA8 out.
//!
//! The static importer ([`import_slice`](super::gltf::import_slice)) copies
//! uvs through but leaves `texture_page` at 0 — it cannot decode images
//! (it only sees buffer data). This module closes that half of the seam
//! for the common case: `.glb` (and inline-buffer `.gltf`) files whose
//! base-color images live in buffer views. External URIs and data URIs
//! stay game-side (the game fetches/reads the file and hands bytes to
//! [`decode_image_bytes`]); the importer reports them as
//! [`TextureImage::External`] so the gap is visible, not silent.
//!
//! Decoding uses the same loaders the `gltf` crate uses internally
//! (`image` PNG/JPEG): what the file claims determines the loader, with a
//! magic-bytes fallback for mislabeled views. Output is always tight
//! `w * h * 4` sRGB bytes in top-first row-major order, ready for a
//! [`SceneUpload`](super::render::SceneUpload).

use gltf::image::Source;

/// One base-color image referenced by the file, in primitive-visit order.
#[derive(Clone, Debug)]
pub enum TextureImage {
    /// Decoded RGBA8 pixels (top-first row-major, `w * h * 4` bytes).
    Decoded {
        /// Index into the source document's image list (stable per file).
        image: usize,
        width: u32,
        height: u32,
        rgba: Vec<u8>,
    },
    /// The image lives outside the slice (external URI or data URI):
    /// the game resolves it and decodes with [`decode_image_bytes`].
    External {
        /// Index into the source document's image list.
        image: usize,
    },
}

/// Why an image produced no pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageSkip {
    /// The buffer view is out of range (corrupt file — skipped, logged).
    BadView,
    /// No loader claims these bytes (unknown encoding, empty view).
    UnsupportedEncoding,
    /// The loader rejected the bytes (truncated/corrupt payload).
    DecodeFailed,
}

/// Slice a buffer view's bytes out of the import buffers. `None` when the
/// image is URI-based, or the view index or byte range is out of range
/// (corrupt file).
///
/// `Image::source()` unwraps the MIME label for buffer views (the spec
/// requires it, so a MIME-less view is invalid glTF). Rather than panic
/// the host app on such files, the call runs under `catch_unwind` and a
/// MIME-less view resolves as undecodable-bytes (`None` MIME → magic-only
/// decode still attempted when the view range itself is readable).
fn view_bytes<'d, 'b>(
    doc: &'d gltf::Document,
    buffers: &'b [gltf::buffer::Data],
    image: usize,
) -> Option<(Option<String>, &'b [u8])> {
    let img = doc.images().nth(image)?;
    // `source()` borrows the document with the document's own lifetime;
    // reborrow through a raw read of the same data instead of holding the
    // guard across the unwind boundary.
    let source = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| img.source())).ok()?;
    let Source::View { view, mime_type } = source else {
        return None;
    };
    let parent = buffers.get(view.buffer().index())?;
    let begin = view.offset();
    let end = begin.checked_add(view.length())?;
    let bytes: &'b [u8] = parent.0.get(begin..end)?;
    Some((Some(mime_type.to_string()), bytes))
}

/// Decode one image's bytes to tight sRGB RGBA8. Accepts PNG/JPEG by magic
/// bytes (the file's MIME label only selects between the two when both
/// fail to match — mislabeled views still decode). Anything else is
/// [`ImageSkip::UnsupportedEncoding`]; loader rejections are
/// [`ImageSkip::DecodeFailed`].
pub fn decode_image_bytes(
    bytes: &[u8],
    mime_type: Option<&str>,
) -> Result<(u32, u32, Vec<u8>), ImageSkip> {
    use image::ImageFormat;
    // Magic first (authoritative), MIME as the tiebreak: a PNG-labeled
    // JPEG still decodes, and unlabeled views (inline buffers without a
    // MIME) decode by content.
    let format = image::guess_format(bytes)
        .ok()
        .or(match mime_type {
            Some("image/png") => Some(ImageFormat::Png),
            Some("image/jpeg") => Some(ImageFormat::Jpeg),
            _ => None,
        })
        .filter(|f| matches!(f, ImageFormat::Png | ImageFormat::Jpeg))
        .ok_or(ImageSkip::UnsupportedEncoding)?;
    let decoded =
        image::load_from_memory_with_format(bytes, format).map_err(|_| ImageSkip::DecodeFailed)?;
    let rgba = decoded.to_rgba8();
    Ok((rgba.width(), rgba.height(), rgba.into_raw()))
}

/// Decode every buffer-view base-color image in `doc`/`buffers` (the pair
/// [`gltf::import_slice`] returns, or [`gltf::import_buffers`] over a
/// [`gltf::Gltf::from_slice_without_validation`] parse). External/data-URI
/// images surface as [`TextureImage::External`]; undecodable views log a
/// warning and are skipped (the material keeps its tint — never a panic,
/// never a hole in the returned order: indices still line up with the
/// document's list).
pub fn decode_document_images(
    doc: &gltf::Document,
    buffers: &[gltf::buffer::Data],
) -> Vec<TextureImage> {
    let mut out = Vec::new();
    for (index, image) in doc.images().enumerate() {
        // `source()` unwraps the MIME label (spec-required on buffer
        // views); a MIME-less view panics inside the `gltf` crate, so probe
        // it under `catch_unwind` and treat the panic as "not a URI, view
        // unreadable" — the view-bytes path below still attempts a
        // magic-only decode when the range itself is readable.
        let is_uri = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            matches!(image.source(), Source::Uri { .. })
        }))
        .unwrap_or(false);
        if is_uri {
            out.push(TextureImage::External { image: index });
            continue;
        }
        match view_bytes(doc, buffers, index) {
            Some((mime, bytes)) => match decode_image_bytes(bytes, mime.as_deref()) {
                Ok((width, height, rgba)) => out.push(TextureImage::Decoded {
                    image: index,
                    width,
                    height,
                    rgba,
                }),
                Err(skip) => {
                    log::warn!("gltf textures: skipping image {index} ({skip:?})");
                }
            },
            None => {
                log::warn!("gltf textures: skipping image {index} (BadView)");
            }
        }
    }
    out
}

/// Convenience over [`gltf::import_slice`]'s parse: decode every embedded
/// base-color image in `bytes`. Unlike `gltf::import_slice` (which fails the
/// whole import when any image is bad or external), each image resolves
/// independently here: external/data-URI images surface as
/// [`TextureImage::External`] and undecodable views skip with a warning, so
/// one corrupt texture never sinks the file's geometry or its other images.
/// No filesystem access (slice imports resolve no paths — same rule as the
/// geometry importer).
pub fn decode_slice_images(bytes: &[u8]) -> Result<Vec<TextureImage>, gltf::Error> {
    let gltf = gltf::Gltf::from_slice_without_validation(bytes)?;
    let buffers = gltf::import_buffers(&gltf.document, None, gltf.blob)?;
    Ok(decode_document_images(&gltf.document, &buffers))
}

/// One packed texture page: the array layer plus the placed rect the
/// group's uvs were scaled into.
#[derive(Clone, Copy, Debug)]
pub struct PlacedPage {
    /// Array layer (also the group's `texture_page`).
    pub page: u32,
    /// Placed image size in pixels (downscaled to fit `layer_size` when
    /// the source is larger; never upscaled).
    pub placed_w: u32,
    pub placed_h: u32,
}

/// A fully textured import: geometry with resolved pages, uploads for the
/// batch array, and the image→page map for skinned meshes (whose pages the
/// game assigns via
/// [`SkinnedMesh::assign_page`](super::skin::SkinnedMesh::assign_page)).
#[derive(Clone, Debug)]
pub struct TexturedImport {
    /// Meshes with `texture_page` assigned and uvs scaled into the placed
    /// rect. Groups whose base image had no pixels keep their tint with
    /// uvs stripped (visible, untextured — never an invisible discard).
    pub meshes: Vec<super::gltf::ImportedMesh>,
    /// Uploads for the batch texture array (feed `Frame3d::uploads` once;
    /// the game builds its `BatchDesc` with `layers = pages.len()`).
    pub uploads: Vec<super::render::SceneUpload>,
    /// Document-image-index → packed page (decoded images only).
    pub pages: std::collections::HashMap<usize, PlacedPage>,
    /// Layer edge the pages pack against (echo of the argument).
    pub layer_size: u32,
}

/// Import `.glb` (or inline-buffer `.gltf`) bytes with base-color textures:
/// geometry through the shared scene-graph walk (same output as
/// [`import_slice`](super::gltf::import_slice), but parsed without image
/// decoding so external/corrupt images never fail the mesh), pixels via
/// [`decode_slice_images`], linked through each primitive's material.
///
/// Packing is one image per layer in document order (stable: the same file
/// always packs the same way). Images larger than `layer_size` downscale
/// (aspect-preserving, `Triangle` filter); smaller images pack native-size
/// at the layer origin with uvs scaled to the placed rect (no stretching,
/// no wasted resampling). `layer_size` 0 falls back to 1 (degenerate, logs).
///
/// Groups whose base image is external, undecodable, or absent keep their
/// material tint with uvs stripped and a warning — the mesh draws flat
/// instead of discarding against an empty page (see the hazard note on
/// [`MeshGroup`](super::mesh::MeshGroup) `uvs`).
pub fn import_slice_textured(bytes: &[u8], layer_size: u32) -> Result<TexturedImport, gltf::Error> {
    let layer_size = layer_size.max(1);
    if layer_size == 1 {
        log::warn!("gltf textures: layer_size 0 falls back to 1");
    }
    // Parse WITHOUT image decoding (unlike `gltf::import_slice`, which
    // fails the whole file on a bad/external image): geometry and pixels
    // resolve independently, so one corrupt texture never sinks the mesh.
    let gltf = gltf::Gltf::from_slice_without_validation(bytes)?;
    let buffers = gltf::import_buffers(&gltf.document, None, gltf.blob)?;
    let mut meshes = super::gltf::import_document(&gltf.document, &buffers);
    let images = decode_document_images(&gltf.document, &buffers);
    let mut uploads = Vec::new();
    let mut pages = std::collections::HashMap::new();
    for (page, image) in images.iter().enumerate() {
        let TextureImage::Decoded {
            image,
            width,
            height,
            rgba,
        } = image
        else {
            continue;
        };
        let (placed_w, placed_h, pixels) = fit_to_layer(*width, *height, rgba, layer_size);
        uploads.push(super::render::SceneUpload {
            page: page as u32,
            x: 0,
            y: 0,
            w: placed_w,
            h: placed_h,
            rgba: pixels,
        });
        pages.insert(
            *image,
            PlacedPage {
                page: page as u32,
                placed_w,
                placed_h,
            },
        );
    }
    // Link groups: resolved pages assign + scale uvs; unresolved strip.
    for mesh in &mut meshes {
        for group in &mut mesh.groups {
            match group.base_image.and_then(|i| pages.get(&i)) {
                Some(placed) => {
                    group.texture_page = placed.page;
                    if !group.uvs.is_empty() {
                        let sx = placed.placed_w as f32 / layer_size as f32;
                        let sy = placed.placed_h as f32 / layer_size as f32;
                        for uv in &mut group.uvs {
                            uv[0] *= sx;
                            uv[1] *= sy;
                        }
                    }
                }
                None => {
                    if group.base_image.is_some() && !group.uvs.is_empty() {
                        log::warn!(
                            "gltf textures: no pixels for image {} — stripping uvs (tint fallback)",
                            group.base_image.unwrap_or(usize::MAX)
                        );
                        group.uvs.clear();
                    }
                }
            }
        }
    }
    Ok(TexturedImport {
        meshes,
        uploads,
        pages,
        layer_size,
    })
}

/// Fit `w`x`h` RGBA into a `layer_size` layer: downscale aspect-preserving
/// when larger (never upscale — small images upload native and uvs scale
/// instead, so no resampling blur on pixel-art sources).
fn fit_to_layer(w: u32, h: u32, rgba: &[u8], layer_size: u32) -> (u32, u32, Vec<u8>) {
    let w = w.max(1);
    let h = h.max(1);
    if w <= layer_size && h <= layer_size {
        return (w, h, rgba.to_vec());
    }
    let scale = (layer_size as f32 / w as f32).min(layer_size as f32 / h as f32);
    let (nw, nh) = (
        ((w as f32 * scale) as u32).max(1),
        ((h as f32 * scale) as u32).max(1),
    );
    let src = match image::RgbaImage::from_raw(w, h, rgba.to_vec()) {
        Some(img) => img,
        None => return (w.min(layer_size), h.min(layer_size), rgba.to_vec()),
    };
    let resized = image::imageops::resize(&src, nw, nh, image::imageops::FilterType::Triangle);
    (nw, nh, resized.into_raw())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal PNG (1x1 red) encoded at test time: no binary fixtures in
    /// the tree, and the encoder output is a real loader input.
    fn red_png() -> Vec<u8> {
        let mut out = Vec::new();
        let encoder = image::codecs::png::PngEncoder::new(&mut out);
        use image::ImageEncoder;
        encoder
            .write_image(&[255, 0, 0, 255], 1, 1, image::ExtendedColorType::Rgba8)
            .expect("encode 1x1 red");
        out
    }

    /// Wrap raw bytes as a buffer-view image: JSON declares one image
    /// pointing at view 0 with the given MIME (or none).
    fn view_gltf(payload: &[u8], mime: Option<&str>) -> Vec<u8> {
        let mime_json = mime
            .map(|m| format!(r#","mimeType":"{m}""#))
            .unwrap_or_default();
        let json = format!(
            r#"{{"asset":{{"version":"2.0"}},"images":[{{"bufferView":0{mime_json}}}],"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":{}}}],"buffers":[{{"byteLength":{}}}]}}"#,
            payload.len(),
            payload.len()
        );
        let json_bytes = json.as_bytes();
        let json_pad = (4 - json_bytes.len() % 4) % 4;
        let bin_pad = (4 - payload.len() % 4) % 4;
        let total = 12 + 8 + json_bytes.len() + json_pad + 8 + payload.len() + bin_pad;
        let mut glb = Vec::with_capacity(total);
        glb.extend_from_slice(&0x46546C67u32.to_le_bytes());
        glb.extend_from_slice(&2u32.to_le_bytes());
        glb.extend_from_slice(&(total as u32).to_le_bytes());
        glb.extend_from_slice(&((json_bytes.len() + json_pad) as u32).to_le_bytes());
        glb.extend_from_slice(&0x4E4F534Au32.to_le_bytes());
        glb.extend_from_slice(json_bytes);
        glb.extend_from_slice(&vec![0x20u8; json_pad]);
        glb.extend_from_slice(&((payload.len() + bin_pad) as u32).to_le_bytes());
        glb.extend_from_slice(&0x004E4942u32.to_le_bytes());
        glb.extend_from_slice(payload);
        glb.extend_from_slice(&vec![0u8; bin_pad]);
        glb
    }

    /// Minimal JSON-only glTF container (no buffers).
    fn wrap_json(json: &str) -> Vec<u8> {
        let json_bytes = json.as_bytes();
        let json_pad = (4 - json_bytes.len() % 4) % 4;
        let total = 12 + 8 + json_bytes.len() + json_pad;
        let mut glb = Vec::with_capacity(total);
        glb.extend_from_slice(&0x46546C67u32.to_le_bytes()); // "glTF"
        glb.extend_from_slice(&2u32.to_le_bytes());
        glb.extend_from_slice(&(total as u32).to_le_bytes());
        glb.extend_from_slice(&((json_bytes.len() + json_pad) as u32).to_le_bytes());
        glb.extend_from_slice(&0x4E4F534Au32.to_le_bytes()); // "JSON"
        glb.extend_from_slice(json_bytes);
        glb.extend_from_slice(&vec![0x20u8; json_pad]);
        glb
    }

    #[test]
    fn png_view_decodes_to_rgba() {
        let png = red_png();
        let glb = view_gltf(&png, Some("image/png"));
        let images = decode_slice_images(&glb).expect("parses");
        assert_eq!(images.len(), 1);
        match &images[0] {
            TextureImage::Decoded {
                width,
                height,
                rgba,
                ..
            } => {
                assert_eq!((*width, *height), (1, 1));
                assert_eq!(rgba.as_slice(), &[255, 0, 0, 255]);
            }
            TextureImage::External { .. } => panic!("buffer view must decode"),
        }
    }

    #[test]
    fn magic_beats_mime() {
        let png = red_png();
        // Mislabeled as JPEG: content still wins.
        let images = decode_slice_images(&view_gltf(&png, Some("image/jpeg"))).expect("parses");
        assert!(matches!(images[0], TextureImage::Decoded { .. }));
    }

    #[test]
    fn mime_less_view_skips() {
        // Buffer-view images without a MIME label are invalid glTF (the
        // spec requires it, and `gltf::Image::source` unwraps it): the
        // importer skips instead of panicking the host app.
        let png = red_png();
        let images = decode_slice_images(&view_gltf(&png, None)).expect("parses");
        assert!(images.is_empty(), "mime-less view skips: {images:?}");
    }

    #[test]
    fn garbage_view_skips_without_panic() {
        let glb = view_gltf(b"not an image at all", Some("image/png"));
        let images = decode_slice_images(&glb).expect("parses");
        assert!(images.is_empty(), "undecodable view skips: {images:?}");
        assert_eq!(
            decode_image_bytes(b"nope", None),
            Err(ImageSkip::UnsupportedEncoding)
        );
        assert_eq!(
            decode_image_bytes(&[0x89, b'P', b'N', b'G', 0, 0], Some("image/png")),
            Err(ImageSkip::DecodeFailed)
        );
    }

    #[test]
    fn external_uri_reports_instead_of_resolving() {
        let doc_json = r#"{"asset":{"version":"2.0"},"images":[{"uri":"tex/out.png"}]}"#;
        let glb = wrap_json(doc_json);
        let images = decode_slice_images(&glb).expect("parses");
        assert_eq!(images.len(), 1);
        assert!(matches!(images[0], TextureImage::External { image: 0 }));
    }

    /// Quad + material with a base-color texture backed by an embedded 2x1
    /// PNG: positions/normals/uvs/indices in view 0-3, PNG bytes in view 4.
    /// Uvs are full 0..1 (pre-flip: v=0 top in glTF space).
    fn textured_quad_gltf(png: &[u8], tex_coord: u32, with_material: bool) -> Vec<u8> {
        let mut bin: Vec<u8> = Vec::new();
        for p in [
            [-1f32, 0.0, -1.0],
            [-1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
        ] {
            bin.extend_from_slice(bytemuck::cast_slice(&p));
        }
        for _ in 0..4 {
            bin.extend_from_slice(bytemuck::cast_slice(&[0f32, 1.0, 0.0]));
        }
        // Two UV sets: set 0 full-quad, set 1 half-quad (proves texcoord
        // selection reads the right set).
        for uv in [[0f32, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]] {
            bin.extend_from_slice(bytemuck::cast_slice(&uv));
        }
        for uv in [[0f32, 0.0], [0.0, 0.5], [0.5, 0.5], [0.5, 0.0]] {
            bin.extend_from_slice(bytemuck::cast_slice(&uv));
        }
        for i in [0u16, 1, 2, 0, 2, 3] {
            bin.extend_from_slice(bytemuck::cast_slice(&[i]));
        }
        let geo_len = bin.len();
        bin.extend_from_slice(png);
        let json = format!(
            r#"{{"asset":{{"version":"2.0"}},"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2,"TEXCOORD_1":3}},"indices":4{mat}}}]}}],"materials":[{{"pbrMetallicRoughness":{{"baseColorFactor":[0.5,0.5,0.5,1.0],"baseColorTexture":{{"index":0,"texCoord":{tex_coord}}}}}}}],"textures":[{{"source":0,"sampler":0}}],"samplers":[{{}}],"images":[{{"bufferView":5,"mimeType":"image/png"}}],"buffers":[{{"byteLength":{}}}],"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":48}},{{"buffer":0,"byteOffset":48,"byteLength":48}},{{"buffer":0,"byteOffset":96,"byteLength":32}},{{"buffer":0,"byteOffset":128,"byteLength":32}},{{"buffer":0,"byteOffset":160,"byteLength":12}},{{"buffer":0,"byteOffset":{geo_len},"byteLength":{}}}],"accessors":[{{"bufferView":0,"componentType":5126,"count":4,"type":"VEC3","min":[-1.0,0.0,-1.0],"max":[1.0,0.0,1.0]}},{{"bufferView":1,"componentType":5126,"count":4,"type":"VEC3"}},{{"bufferView":2,"componentType":5126,"count":4,"type":"VEC2"}},{{"bufferView":3,"componentType":5126,"count":4,"type":"VEC2"}},{{"bufferView":4,"componentType":5123,"count":6,"type":"SCALAR"}}]}}"#,
            bin.len(),
            png.len(),
            mat = if with_material {
                r#","material":0"#
            } else {
                ""
            },
        );
        let json_bytes = json.as_bytes();
        let json_pad = (4 - json_bytes.len() % 4) % 4;
        let bin_pad = (4 - bin.len() % 4) % 4;
        let total = 12 + 8 + json_bytes.len() + json_pad + 8 + bin.len() + bin_pad;
        let mut glb = Vec::with_capacity(total);
        glb.extend_from_slice(&0x46546C67u32.to_le_bytes());
        glb.extend_from_slice(&2u32.to_le_bytes());
        glb.extend_from_slice(&(total as u32).to_le_bytes());
        glb.extend_from_slice(&((json_bytes.len() + json_pad) as u32).to_le_bytes());
        glb.extend_from_slice(&0x4E4F534Au32.to_le_bytes());
        glb.extend_from_slice(json_bytes);
        glb.extend_from_slice(&vec![0x20u8; json_pad]);
        glb.extend_from_slice(&((bin.len() + bin_pad) as u32).to_le_bytes());
        glb.extend_from_slice(&0x004E4942u32.to_le_bytes());
        glb.extend_from_slice(&bin);
        glb.extend_from_slice(&vec![0u8; bin_pad]);
        glb
    }

    /// 2x1 PNG: left red, right green (top-first row-major).
    fn red_green_png() -> Vec<u8> {
        let mut out = Vec::new();
        let encoder = image::codecs::png::PngEncoder::new(&mut out);
        use image::ImageEncoder;
        encoder
            .write_image(
                &[255, 0, 0, 255, 0, 255, 0, 255],
                2,
                1,
                image::ExtendedColorType::Rgba8,
            )
            .expect("encode 2x1");
        out
    }

    #[test]
    fn textured_import_links_pages_uploads_and_uvs() {
        let glb = textured_quad_gltf(&red_green_png(), 0, true);
        let imported = import_slice_textured(&glb, 4).expect("textured import parses");
        assert_eq!(imported.layer_size, 4);
        // One decoded image → one page, upload 2x1 native (no upscale).
        assert_eq!(imported.pages.len(), 1);
        let placed = imported.pages[&0];
        assert_eq!((placed.page, placed.placed_w, placed.placed_h), (0, 2, 1));
        assert_eq!(imported.uploads.len(), 1);
        let up = &imported.uploads[0];
        assert_eq!((up.page, up.x, up.y, up.w, up.h), (0, 0, 0, 2, 1));
        assert_eq!(up.rgba.as_slice(), &[255, 0, 0, 255, 0, 255, 0, 255]);
        // Group: page assigned, uvs scaled into the placed rect
        // (2/4 wide, 1/4 tall), tint kept from the base-color factor.
        // (Import flips v first, then the pack scales: [1,1] → [1,0] →
        // [0.5,0.0].)
        assert_eq!(imported.meshes.len(), 1);
        let g = &imported.meshes[0].groups[0];
        assert_eq!(g.base_image, Some(0));
        assert_eq!(g.texture_page, 0);
        assert_eq!(g.uvs.len(), g.positions.len());
        assert_eq!(g.uvs[0], [0.0, 0.25]);
        assert_eq!(g.uvs[2], [0.5, 0.0]);
        assert!(g.colors.iter().all(|c| *c == [0.5, 0.5, 0.5]));
    }

    #[test]
    fn texcoord_selects_the_material_set() {
        // texCoord 1 → the half-quad set: [0.5,0.5] flips to [0.5,0.5]
        // (v=0.5 is self-mirroring) then scales to [0.25,0.125].
        let glb = textured_quad_gltf(&red_green_png(), 1, true);
        let imported = import_slice_textured(&glb, 4).expect("parses");
        let g = &imported.meshes[0].groups[0];
        assert_eq!(g.uvs[2], [0.25, 0.125]);
    }

    #[test]
    fn missing_texture_strips_uvs_keeps_tint() {
        // No material → untextured group: uvs exist (copied through) but no
        // base image, so the textured import leaves them alone (nothing to
        // strip — no texture was ever intended).
        let glb = textured_quad_gltf(&red_green_png(), 0, false);
        let imported = import_slice_textured(&glb, 4).expect("parses");
        let g = &imported.meshes[0].groups[0];
        assert_eq!(g.base_image, None);
        assert_eq!(imported.uploads.len(), 1, "image still decodes");
        assert_eq!(g.texture_page, 0);
    }

    #[test]
    fn external_image_strips_uvs_keeps_tint_visible() {
        // Material with a base texture whose image is an external URI:
        // decode reports External, the group strips uvs (tint fallback)
        // instead of discarding against an empty page.
        let png = red_green_png();
        let mut bin: Vec<u8> = Vec::new();
        for p in [
            [-1f32, 0.0, -1.0],
            [-1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
        ] {
            bin.extend_from_slice(bytemuck::cast_slice(&p));
        }
        for uv in [[0f32, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]] {
            bin.extend_from_slice(bytemuck::cast_slice(&uv));
        }
        for i in [0u16, 1, 2, 0, 2, 3] {
            bin.extend_from_slice(bytemuck::cast_slice(&[i]));
        }
        let _ = png;
        let json = format!(
            r#"{{"asset":{{"version":"2.0"}},"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"TEXCOORD_0":1}},"indices":2,"material":0}}]}}],"materials":[{{"pbrMetallicRoughness":{{"baseColorTexture":{{"index":0}}}}}}],"textures":[{{"source":0}}],"images":[{{"uri":"tex/out.png"}}],"buffers":[{{"byteLength":{}}}],"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":48}},{{"buffer":0,"byteOffset":48,"byteLength":32}},{{"buffer":0,"byteOffset":80,"byteLength":12}}],"accessors":[{{"bufferView":0,"componentType":5126,"count":4,"type":"VEC3","min":[-1.0,0.0,-1.0],"max":[1.0,0.0,1.0]}},{{"bufferView":1,"componentType":5126,"count":4,"type":"VEC2"}},{{"bufferView":2,"componentType":5123,"count":6,"type":"SCALAR"}}]}}"#,
            bin.len()
        );
        let json_bytes = json.as_bytes();
        let json_pad = (4 - json_bytes.len() % 4) % 4;
        let bin_pad = (4 - bin.len() % 4) % 4;
        let total = 12 + 8 + json_bytes.len() + json_pad + 8 + bin.len() + bin_pad;
        let mut glb = Vec::with_capacity(total);
        glb.extend_from_slice(&0x46546C67u32.to_le_bytes());
        glb.extend_from_slice(&2u32.to_le_bytes());
        glb.extend_from_slice(&(total as u32).to_le_bytes());
        glb.extend_from_slice(&((json_bytes.len() + json_pad) as u32).to_le_bytes());
        glb.extend_from_slice(&0x4E4F534Au32.to_le_bytes());
        glb.extend_from_slice(json_bytes);
        glb.extend_from_slice(&vec![0x20u8; json_pad]);
        glb.extend_from_slice(&((bin.len() + bin_pad) as u32).to_le_bytes());
        glb.extend_from_slice(&0x004E4942u32.to_le_bytes());
        glb.extend_from_slice(&bin);
        glb.extend_from_slice(&vec![0u8; bin_pad]);
        let imported = import_slice_textured(&glb, 4).expect("parses");
        assert!(imported.uploads.is_empty());
        assert!(imported.pages.is_empty());
        let g = &imported.meshes[0].groups[0];
        assert_eq!(g.base_image, Some(0));
        assert!(g.uvs.is_empty(), "uvs stripped: {:?}", g.uvs);
        assert!(g.colors.iter().all(|c| *c == [1.0, 1.0, 1.0]));
    }

    #[test]
    fn oversize_image_downscales_aspect_preserving() {
        // 2x1 source into a 1px layer: scale 0.5 → 1x1... 1x1 min clamps.
        // Use layer 1 directly: placed must fit and stay proportional.
        let glb = textured_quad_gltf(&red_green_png(), 0, true);
        let imported = import_slice_textured(&glb, 1).expect("parses");
        let placed = imported.pages[&0];
        assert!(placed.placed_w <= 1 && placed.placed_h <= 1);
        assert_eq!((placed.placed_w, placed.placed_h), (1, 1));
        let up = &imported.uploads[0];
        assert_eq!((up.w, up.h), (1, 1));
        assert_eq!(up.rgba.len(), 4);
    }

    #[test]
    fn skinned_page_assignment_scales_bind_uvs_once() {
        use super::super::skin::SkinnedMesh;
        let mut mesh = SkinnedMesh {
            uvs: vec![[0.0, 1.0], [1.0, 0.0]],
            base_image: Some(0),
            ..Default::default()
        };
        mesh.assign_page(2, 2, 1, 4);
        assert_eq!(mesh.texture_page, 2);
        assert_eq!(mesh.uvs, vec![[0.0, 0.25], [0.5, 0.0]]);
    }
}
