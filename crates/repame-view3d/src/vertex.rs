//! Explicit vertex layouts: the formats a console-era mesh declares, and the
//! packing rules that turn them into a GPU buffer.
//!
//! [`MeshGroup`](crate::MeshGroup) stays the fast path for the common
//! position/normal/color/uv set. A group that also carries a [`VertexLayout`]
//! is a raw-attribute group: the game owns the bytes and the engine only
//! validates the declaration, so formats a fixed layout cannot express
//! (packed normals, 4-bit channels, matrix indices) stay expressible without
//! per-format game-side re-encoding.

/// Scalar component of a vertex attribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VertexFormat {
    F32,
    F16,
    U8,
    S8,
    U16,
    S16,
    /// 4-bit unsigned, expanded to 0..1.
    U8x4,
    U8x4Pair,
    /// 4-bit signed, expanded to -1..1.
    S8x4,
    S8x4Pair,
    /// 10-bit fixed point, expanded to 0..1.
    U10x10x10x2,
    /// Normalized signed 2.10, expanded to -1..1.
    S10x10x10x2,
}

impl VertexFormat {
    /// Bytes the format occupies: scalar component width, or the packed
    /// word four-bit formats share.
    pub const fn size(self) -> u32 {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::U16 | Self::S16 => 2,
            Self::U8x4 | Self::S8x4 => 2,
            Self::U8x4Pair | Self::S8x4Pair => 4,
            Self::U10x10x10x2 | Self::S10x10x10x2 => 4,
            Self::U8 | Self::S8 => 1,
        }
    }

    /// Scalar channels a format expands to.
    pub const fn components(self) -> u32 {
        match self {
            Self::F32 | Self::F16 | Self::U8 | Self::S8 | Self::U16 | Self::S16 => 1,
            Self::U8x4 | Self::S8x4 => 4,
            Self::U8x4Pair | Self::S8x4Pair => 8,
            Self::U10x10x10x2 | Self::S10x10x10x2 => 4,
        }
    }

    /// Signed components expand into -1..1, unsigned into 0..1.
    pub const fn is_signed(self) -> bool {
        matches!(
            self,
            Self::S8 | Self::S16 | Self::S8x4 | Self::S8x4Pair | Self::S10x10x10x2
        )
    }
}

/// What an attribute's components mean, so a shader can be lowered for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VertexSemantic {
    Position,
    Normal,
    Tangent,
    Binormal,
    Color,
    Uv(u8),
    /// Index into a transform matrix array.
    MatrixIndex(u8),
    Weight(u8),
    /// Free-form game attribute.
    Custom(u8),
}

/// One attribute inside a vertex.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VertexAttribute {
    pub semantic: VertexSemantic,
    pub format: VertexFormat,
    /// Element count: scalar formats count components (`F32` x3 for
    /// positions), packed formats count words.
    pub count: u8,
    pub offset: u32,
}

impl VertexAttribute {
    pub const fn new(semantic: VertexSemantic, format: VertexFormat, count: u8) -> Self {
        Self {
            semantic,
            format,
            count,
            offset: 0,
        }
    }

    pub fn byte_size(&self) -> u32 {
        self.format.size() * u32::from(self.count)
    }
}

/// Index buffer element width.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IndexFormat {
    U8,
    U16,
    U32,
}

impl IndexFormat {
    pub const fn size(self) -> u32 {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
            Self::U32 => 4,
        }
    }

    /// Narrowest format that can index `vertex_count` vertices.
    pub fn for_vertex_count(vertex_count: usize) -> Self {
        if vertex_count <= 0x100 {
            Self::U8
        } else if vertex_count <= 0x1_0000 {
            Self::U16
        } else {
            Self::U32
        }
    }

    pub fn byte_size(self, index_count: usize) -> usize {
        self.size() as usize * index_count
    }
}

/// Why a layout declaration was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutError {
    Empty,
    StrideOverflow,
    AttributeOutOfRange,
    OverlappingAttributes,
    DuplicateSemantic(VertexSemantic),
    TooManyUvSets,
    AttributeTooLarge,
}

/// A validated vertex declaration with byte offsets resolved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VertexLayout {
    pub stride: u32,
    pub attributes: Vec<VertexAttribute>,
}

impl VertexLayout {
    /// Packs attributes in declaration order, each aligned to its own width.
    ///
    /// `align` raises the floor for every attribute, so a game can force a
    /// wider stride than the formats need.
    pub fn pack(attributes: Vec<VertexAttribute>, align: u32) -> Result<Self, LayoutError> {
        if attributes.is_empty() {
            return Err(LayoutError::Empty);
        }
        let align = align.max(1);
        let mut offset = 0_u32;
        let mut packed = Vec::with_capacity(attributes.len());
        for mut attribute in attributes {
            let size = attribute.byte_size();
            if size == 0 || size > 16 {
                return Err(LayoutError::AttributeTooLarge);
            }
            let width = size.min(4).next_multiple_of(align);
            offset = offset.next_multiple_of(width);
            attribute.offset = offset;
            offset += size;
            packed.push(attribute);
        }
        let stride = offset.next_multiple_of(align);
        if stride > 0xFFFF {
            return Err(LayoutError::StrideOverflow);
        }
        let layout = Self {
            stride,
            attributes: packed,
        };
        layout.check_overlap()?;
        Ok(layout)
    }

    /// The common console vertex: 3 floats position, packed normal and
    /// binormal, 2 packed colors, 2 half uvs — 24 bytes.
    pub fn console_standard() -> Self {
        let attributes = vec![
            VertexAttribute::new(VertexSemantic::Position, VertexFormat::F32, 3),
            VertexAttribute::new(VertexSemantic::Normal, VertexFormat::S8x4, 1),
            VertexAttribute::new(VertexSemantic::Binormal, VertexFormat::S8x4, 1),
            VertexAttribute::new(VertexSemantic::Color, VertexFormat::U8x4Pair, 1),
            VertexAttribute::new(VertexSemantic::Uv(0), VertexFormat::F16, 2),
        ];
        Self::pack(attributes, 1).expect("standard layout packs")
    }

    fn check_overlap(&self) -> Result<(), LayoutError> {
        let mut sorted: Vec<&VertexAttribute> = self.attributes.iter().collect();
        sorted.sort_by_key(|attribute| attribute.offset);
        for pair in sorted.windows(2) {
            let (left, right) = (pair[0], pair[1]);
            if left.offset + left.byte_size() > right.offset {
                return Err(LayoutError::OverlappingAttributes);
            }
        }
        for (index, attribute) in self.attributes.iter().enumerate() {
            if attribute.offset + attribute.byte_size() > self.stride {
                return Err(LayoutError::AttributeOutOfRange);
            }
            for other in &self.attributes[index + 1..] {
                if other.semantic == attribute.semantic {
                    return Err(LayoutError::DuplicateSemantic(attribute.semantic));
                }
            }
        }
        let highest_uv_set = self
            .attributes
            .iter()
            .filter_map(|attribute| match attribute.semantic {
                VertexSemantic::Uv(set) => Some(set),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        if highest_uv_set >= 8 {
            return Err(LayoutError::TooManyUvSets);
        }
        Ok(())
    }

    pub fn attribute(&self, semantic: VertexSemantic) -> Option<&VertexAttribute> {
        self.attributes
            .iter()
            .find(|attribute| attribute.semantic == semantic)
    }

    pub fn uv_sets(&self) -> u8 {
        self.attributes
            .iter()
            .filter_map(|attribute| match attribute.semantic {
                VertexSemantic::Uv(set) => Some(1_u8 << set),
                _ => None,
            })
            .fold(0_u8, |acc, bit| acc | bit)
            .count_ones() as u8
    }

    /// Total upload size for a vertex count.
    pub fn buffer_size(&self, vertex_count: usize) -> usize {
        self.stride as usize * vertex_count
    }
}

/// Expands one packed component to a float in the format's declared range.
pub fn expand_component(raw: u32, format: VertexFormat, component: u32) -> f32 {
    match format {
        VertexFormat::F32 => f32::from_bits(raw),
        VertexFormat::F16 => f16_to_f32(raw as u16),
        VertexFormat::U8 => (raw & 0xFF) as f32 / 255.0,
        VertexFormat::S8 => (((raw & 0xFF) as i8) as f32 / 127.0).clamp(-1.0, 1.0),
        VertexFormat::U16 => (raw & 0xFFFF) as f32 / 65_535.0,
        VertexFormat::S16 => (((raw & 0xFFFF) as i16) as f32 / 32_767.0).clamp(-1.0, 1.0),
        VertexFormat::U8x4 | VertexFormat::U8x4Pair => {
            ((raw >> (component * 4)) & 0xF) as f32 / 15.0
        }
        VertexFormat::S8x4 | VertexFormat::S8x4Pair => {
            let nibble = (raw >> (component * 4)) & 0xF;
            (((nibble as f32) - 8.0) / 7.0).clamp(-1.0, 1.0)
        }
        VertexFormat::U10x10x10x2 => expand_10_10_10_2(raw, component, false),
        VertexFormat::S10x10x10x2 => expand_10_10_10_2(raw, component, true),
    }
}

/// Extracts a 10-bit field out of a packed 10/10/10/2 word.
///
/// The three 10-bit fields are two's complement when `signed`; the trailing
/// 2-bit field is unsigned in both variants.
pub fn expand_10_10_10_2(raw: u32, component: u32, signed: bool) -> f32 {
    let (shift, mask) = match component {
        0 => (0, 0x3FF),
        1 => (10, 0x3FF),
        2 => (20, 0x3FF),
        _ => (30, 0x3),
    };
    let value = (raw >> shift) & mask;
    if component == 3 {
        return value as f32 / 3.0;
    }
    if signed {
        let twos = ((value << 22) as i32) >> 22;
        ((twos as f32) / 511.0).clamp(-1.0, 1.0)
    } else {
        value as f32 / 1023.0
    }
}

/// Decodes an IEEE half word into its f32 value.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exponent = (bits >> 10) & 0x1F;
    let fraction = bits & 0x3FF;
    let value = match exponent {
        0 => f32::from(fraction) * 2.0_f32.powi(-24),
        0x1F => {
            if fraction == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => (1.0 + f32::from(fraction) / 1024.0) * 2.0_f32.powi(i32::from(exponent) - 15),
    };
    if sign == 1 { -value } else { value }
}
