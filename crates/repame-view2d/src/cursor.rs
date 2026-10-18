//! Hardware-cursor frames: one strip in, one magnified RGBA frame out.
//!
//! The strip is a horizontal sheet of `frames` cells of `cell` size; the
//! caller uploads the returned pixels as a cursor image. [`CursorFrame::key`]
//! is the identity of the decoded frame, so a caller can cache the upload and
//! only redo it when the key changes.

/// One decoded + magnified hardware-cursor frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CursorFrame {
    pub rgba: Vec<u8>,
    pub size: [u16; 2],
    pub hotspot: [u16; 2],
    pub key: (i32, [u32; 3], u64, u32),
}

pub fn cursor_frame(
    strip: &[u8],
    strip_w: u32,
    strip_h: u32,
    frames: u32,
    frame: i32,
    cell: (u32, u32),
    origin: (f32, f32),
    tint: [f32; 4],
    mag: u32,
    strip_hash: u64,
) -> Option<CursorFrame> {
    let frames = frames.max(1) as i32;
    let frame = frame.clamp(0, frames - 1);
    let (cw, ch) = (cell.0.max(1), cell.1.max(1));
    if strip_w < cw * (frame as u32 + 1) || strip_h < ch {
        return None;
    }
    let mag = mag.clamp(1, 8);
    let tint_u8 = [
        (tint[0].clamp(0.0, 1.0) * 255.0) as u32,
        (tint[1].clamp(0.0, 1.0) * 255.0) as u32,
        (tint[2].clamp(0.0, 1.0) * 255.0) as u32,
    ];
    let key = (frame, tint_u8, strip_hash, mag);
    let fx = (frame as u32 * cw) as usize;
    let (dw, dh) = (cw * mag, ch * mag);
    if dw > 2048 || dh > 2048 {
        return None;
    }
    let mut px = Vec::with_capacity((dw * dh * 4) as usize);
    for row in 0..ch {
        let start = ((row * strip_w) as usize + fx) * 4;
        let end = start + (cw as usize) * 4;
        let cell = strip.get(start..end)?;
        let mut mag_row = Vec::with_capacity((cw * mag * 4) as usize);
        for pix in cell.chunks_exact(4) {
            let t = [
                (pix[0] as u32 * tint_u8[0] / 255) as u8,
                (pix[1] as u32 * tint_u8[1] / 255) as u8,
                (pix[2] as u32 * tint_u8[2] / 255) as u8,
                pix[3],
            ];
            for _ in 0..mag {
                mag_row.extend_from_slice(&t);
            }
        }
        for _ in 0..mag {
            px.extend_from_slice(&mag_row);
        }
    }
    let (w16, h16) = (dw.min(2048) as u16, dh.min(2048) as u16);
    Some(CursorFrame {
        rgba: px,
        size: [w16, h16],
        hotspot: [
            (origin.0 * mag as f32)
                .round()
                .clamp(0.0, w16.saturating_sub(1) as f32) as u16,
            (origin.1 * mag as f32)
                .round()
                .clamp(0.0, h16.saturating_sub(1) as f32) as u16,
        ],
        key,
    })
}
