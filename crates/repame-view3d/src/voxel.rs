//! Voxel chunk mesher: game-owned grid in, [`MeshGroup`]s out.

use std::collections::{HashMap, HashSet};

use super::mesh::{MeshGroup, Rgb, shade_for_dir};

/// Chunk edge in cells. Matches the rustbox `CHUNK_SIZE` convention (and the
/// [`ChunkCache`](crate::ChunkCache) key convention).
pub const CHUNK_SIZE: i32 = 16;

/// Voxel block shape (mirrors the rustbox `BlockShape` set the mesher was
/// proven against; games map their own shapes onto these).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum VoxelShape {
    /// Full 1x1x1 cube.
    #[default]
    Full,
    /// Bottom half slab (y in `[0, 0.5]`).
    Half,
    /// Top half slab (y in `[0.5, 1]`).
    TopHalf,
    /// 45-degree ramp rising toward local +X.
    Slope,
    /// 45-degree ramp rising toward local -X.
    DSlope,
    /// Quarter corner ramp rising toward local +X,+Z.
    Corner,
    /// Peak sloping down toward local +X,+Z.
    OuterCorner,
    /// Full-height quarter against local -X/-Z, cut along `x + z = 1`.
    VerticalSlope,
    /// 1x1x0.5 slab against the local -Z face.
    VerticalSlab,
    /// Thin top slab (y in `[1 - THIN, 1]`).
    Thin,
}

/// Thickness of [`VoxelShape::Thin`] slabs (matches `THIN_HEIGHT`).
pub const THIN_HEIGHT: f32 = 0.16;

/// One voxel cell. `kind` is a game id (`classify`/`is_solid`/tint resolve
/// it); `rot` is yaw steps (90 degrees each, same convention as the rustbox
/// renderer: local verts rotate into the world by `rot * 90` degrees).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Cell {
    pub kind: u32,
    pub shape: VoxelShape,
    pub rot: u8,
    pub waterlogged: bool,
}

impl Cell {
    pub fn full(kind: u32) -> Self {
        Self {
            kind,
            shape: VoxelShape::Full,
            rot: 0,
            waterlogged: false,
        }
    }
}

/// Caller-owned grid. Must answer across chunk borders (neighbor lookups
/// cross them, that is what fixes seam holes). `None` = air.
pub trait VoxelSource {
    fn get(&self, cell: [i32; 3]) -> Option<Cell>;
}

impl VoxelSource for HashMap<[i32; 3], Cell> {
    fn get(&self, cell: [i32; 3]) -> Option<Cell> {
        self.get(&cell).copied()
    }
}

/// Face classification for culling (mirrors `rustbox-mesh` `FaceKind`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FaceKind {
    /// Air, spawn markers, pulse-off blocks: emits nothing, occludes nothing.
    Empty,
    /// Solid geometry: emits faces, occludes neighbors.
    Opaque,
    /// Water: separate transparent pass, occludes like opaque against water.
    Fluid,
}

/// Mesher parameters. `classify`/`is_solid` mirror the rustbox split:
/// `classify` answers visibility (pulse-off reads `Empty` here), `is_solid`
/// answers physics solidity (pulse blocks stay solid while hidden).
pub struct ChunkMeshInput<F, G> {
    /// Visibility per kind (`Empty` skips the cell entirely).
    pub classify: F,
    /// Solidity per kind (occlusion + water-cover tests).
    pub is_solid: G,
    /// Base tint per kind (linear RGB; missing kinds read white).
    pub kind_tint: HashMap<u32, Rgb>,
    /// Kindxshape pairs replaced by pack models (skipped, like the rustbox
    /// overlay path).
    pub skip_overlay: HashSet<(u32, VoxelShape)>,
    /// Global water plane Y (`None` = no plane). Cells below it (or
    /// waterlogged) take the submerged tint on the exact path.
    pub water_level: Option<i32>,
    /// Tint for submerged opaque faces.
    pub submerged_tint: Rgb,
    /// Alpha for the water pass (rustbox ships 0.72).
    pub water_alpha: f32,
    /// Emit normals + shade-baked tints (lit path). Off = flat exact colors.
    pub lit: bool,
    /// Emit full-quad 0..1 uvs (game assigns pages after upload). Off =
    /// untextured (tint only).
    pub textured: bool,
}

impl ChunkMeshInput<fn(u32) -> FaceKind, fn(u32) -> bool> {
    fn all_opaque(_: u32) -> FaceKind {
        FaceKind::Opaque
    }

    fn all_solid(_: u32) -> bool {
        true
    }
}

impl Default for ChunkMeshInput<fn(u32) -> FaceKind, fn(u32) -> bool> {
    fn default() -> Self {
        Self {
            classify: Self::all_opaque,
            is_solid: Self::all_solid,
            kind_tint: HashMap::new(),
            skip_overlay: HashSet::new(),
            water_level: None,
            submerged_tint: [0.55, 0.62, 0.78],
            water_alpha: 0.72,
            lit: false,
            textured: false,
        }
    }
}

/// One chunk's built geometry: one [`MeshGroup`] per opaque kind (bind one
/// material per group downstream) plus the transparent water group.
/// Ready for [`ChunkCache::store`](super::chunk::ChunkCache::store).
#[derive(Clone, Debug, Default)]
pub struct ChunkMeshOutput {
    pub opaque: HashMap<u32, MeshGroup>,
    pub water: MeshGroup,
    /// Greedy-merged quads (subset of opaque), stats/tests.
    pub merged_quads: usize,
    /// Exact fallback quads (shaped cells, submerged Fulls), stats/tests.
    pub fallback_quads: usize,
}

impl ChunkMeshOutput {
    pub fn is_empty(&self) -> bool {
        self.water.is_empty() && self.opaque.values().all(|g| g.is_empty())
    }

    pub fn tri_count(&self) -> usize {
        self.water.tri_count() + self.opaque.values().map(|g| g.tri_count()).sum::<usize>()
    }

    /// Every group (opaque per kind, then water when non-empty), for
    /// [`ChunkCache::store`](super::chunk::ChunkCache::store).
    pub fn groups(&self) -> Vec<&MeshGroup> {
        let mut out: Vec<&MeshGroup> = self.opaque.values().collect();
        if !self.water.is_empty() {
            out.push(&self.water);
        }
        out
    }
}

pub const DIRS: [[i32; 3]; 6] = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
];

/// Rotate a world-space face dir into a block's shape-local frame (inverse
/// of the renderer's local-to-world yaw).
pub fn world_dir_to_local(dir: [i32; 3], rot: u8) -> [i32; 3] {
    match rot % 4 {
        0 => dir,
        1 => [-dir[2], dir[1], dir[0]],
        2 => [-dir[0], dir[1], -dir[2]],
        3 => [dir[2], dir[1], -dir[0]],
        _ => dir,
    }
}

/// Does a neighbor fully cover the face between us and it? Conservative:
/// only axis-aligned Full neighbors occlude; slabs occlude only the side
/// they fill; slopes/corners never occlude (no hole-punching);
/// `VerticalSlab` only its own slab plane.
pub fn neighbor_occludes(neighbor: &Cell, face_dir_world: [i32; 3], neighbor_solid: bool) -> bool {
    if !neighbor_solid {
        return false;
    }
    let local_dir = world_dir_to_local(face_dir_world, neighbor.rot);
    match neighbor.shape {
        VoxelShape::Full => true,
        VoxelShape::Half => local_dir == [0, -1, 0],
        VoxelShape::TopHalf => local_dir == [0, 1, 0],
        VoxelShape::Thin => local_dir == [0, 1, 0],
        VoxelShape::VerticalSlab => local_dir == [0, 0, -1],
        _ => false,
    }
}

/// Can our face toward `dir` be culled at all? Shaped sides are never culled
/// (seams can't punch holes); only box faces and slab caps participate.
fn our_side_full(shape: VoxelShape, dir: [i32; 3]) -> bool {
    match shape {
        VoxelShape::Full => true,
        VoxelShape::Half => dir != [0, 1, 0],
        VoxelShape::TopHalf => dir != [0, -1, 0],
        VoxelShape::Thin => dir != [0, -1, 0],
        VoxelShape::VerticalSlab => true,
        _ => dir == [0, -1, 0],
    }
}

fn tint_of(
    input: &ChunkMeshInput<impl Fn(u32) -> FaceKind, impl Fn(u32) -> bool>,
    kind: u32,
) -> Rgb {
    input
        .kind_tint
        .get(&kind)
        .copied()
        .unwrap_or([1.0, 1.0, 1.0])
}

fn is_submerged<F, G>(cell: [i32; 3], block: &Cell, input: &ChunkMeshInput<F, G>) -> bool {
    if block.waterlogged {
        return true;
    }
    if let Some(wl) = input.water_level {
        return cell[1] < wl;
    }
    false
}

fn chunk_origin(cpos: [i32; 3]) -> [i32; 3] {
    [
        cpos[0] * CHUNK_SIZE,
        cpos[1] * CHUNK_SIZE,
        cpos[2] * CHUNK_SIZE,
    ]
}

/// Build one chunk's meshes from a [`VoxelSource`].
pub fn build_chunk_mesh<S, F, G>(
    grid: &S,
    cpos: [i32; 3],
    input: &ChunkMeshInput<F, G>,
    output: &mut ChunkMeshOutput,
) where
    S: VoxelSource,
    F: Fn(u32) -> FaceKind,
    G: Fn(u32) -> bool,
{
    let origin = chunk_origin(cpos);
    let mut full_cells: HashMap<u32, Vec<[i32; 3]>> = HashMap::new();
    let mut fallback: Vec<([i32; 3], Cell)> = Vec::new();

    for lx in 0..CHUNK_SIZE {
        for ly in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                let cell = [origin[0] + lx, origin[1] + ly, origin[2] + lz];
                let Some(block) = grid.get(cell) else {
                    continue;
                };
                if (input.classify)(block.kind) == FaceKind::Empty {
                    continue;
                }
                if (input.classify)(block.kind) == FaceKind::Fluid {
                    continue; // water pass below
                }
                if input.skip_overlay.contains(&(block.kind, block.shape)) {
                    continue;
                }
                if block.shape == VoxelShape::Full
                    && block.rot == 0
                    && !is_submerged(cell, &block, input)
                {
                    full_cells.entry(block.kind).or_default().push(cell);
                } else {
                    fallback.push((cell, block));
                }
            }
        }
    }

    for (kind, cells) in full_cells {
        let cells: Vec<[i32; 3]> = cells
            .into_iter()
            .filter(|cell| {
                grid.get(*cell).is_none_or(|b| {
                    !input.skip_overlay.contains(&(b.kind, b.shape))
                        && (input.classify)(b.kind) == FaceKind::Opaque
                })
            })
            .collect();
        if cells.is_empty() {
            continue;
        }
        let mesh = output.opaque.entry(kind).or_insert_with(|| MeshGroup {
            depth_test: true,
            ..Default::default()
        });
        let before = mesh.tri_count();
        greedy_full_faces(grid, &cells, input, mesh);
        output.merged_quads += mesh.tri_count() - before;
    }

    for (cell, block) in &fallback {
        let mut tint = tint_of(input, block.kind);
        if is_submerged(*cell, block, input) {
            tint = input.submerged_tint;
        }
        let mesh = output
            .opaque
            .entry(block.kind)
            .or_insert_with(|| MeshGroup {
                depth_test: true,
                ..Default::default()
            });
        let before = mesh.tri_count();
        push_shaped_faces(grid, *cell, block, tint, input, mesh);
        output.fallback_quads += mesh.tri_count() - before;
    }

    for lx in 0..CHUNK_SIZE {
        for ly in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                let cell = [origin[0] + lx, origin[1] + ly, origin[2] + lz];
                let Some(block) = grid.get(cell) else {
                    continue;
                };
                if (input.classify)(block.kind) != FaceKind::Fluid {
                    continue;
                }
                for dir in DIRS {
                    let ncell = [cell[0] + dir[0], cell[1] + dir[1], cell[2] + dir[2]];
                    let covered = match grid.get(ncell) {
                        None => false,
                        Some(nb) => {
                            (input.classify)(nb.kind) == FaceKind::Fluid
                                || ((input.is_solid)(nb.kind)
                                    && (input.classify)(nb.kind) == FaceKind::Opaque)
                        }
                    };
                    if covered {
                        continue;
                    }
                    let f = [cell[0] as f32, cell[1] as f32, cell[2] as f32];
                    push_water_quad(&mut output.water, f, dir, input);
                }
            }
        }
    }

    output.opaque.retain(|_, m| !m.is_empty());
}

/// Push one quad into a group through the matching [`MeshGroup`] helper
/// (flat / lit / textured / lit+textured, one style per group, never
/// mixed). `shade` bakes the directional contrast; `normal` is the exact
/// face normal (axis faces) or geometric normal (shaped fallback).
#[allow(clippy::too_many_arguments)]
fn push_face(
    group: &mut MeshGroup,
    quad: [[f32; 3]; 4],
    tint: Rgb,
    dir: [i32; 3],
    normal: [f32; 3],
    lit: bool,
    textured: bool,
) {
    let shaded = {
        let s = shade_for_dir(dir);
        [tint[0] * s, tint[1] * s, tint[2] * s]
    };
    let uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let [a, b, c, d] = quad;
    match (lit, textured) {
        (false, false) => group.push_quad(a, b, c, d, shaded),
        (true, false) => group.push_quad_lit(a, b, c, d, shaded, normal),
        (false, true) => group.push_quad_textured(a, b, c, d, shaded, uvs),
        (true, true) => group.push_quad_lit_textured(a, b, c, d, shaded, normal, uvs),
    }
}

fn full_face_quad(origin: [f32; 3], dir: [i32; 3]) -> [[f32; 3]; 4] {
    let [x, y, z] = origin;
    match dir {
        [1, 0, 0] => [
            [x + 1.0, y, z],
            [x + 1.0, y + 1.0, z],
            [x + 1.0, y + 1.0, z + 1.0],
            [x + 1.0, y, z + 1.0],
        ],
        [-1, 0, 0] => [
            [x, y, z + 1.0],
            [x, y + 1.0, z + 1.0],
            [x, y + 1.0, z],
            [x, y, z],
        ],
        [0, 1, 0] => [
            [x, y + 1.0, z],
            [x, y + 1.0, z + 1.0],
            [x + 1.0, y + 1.0, z + 1.0],
            [x + 1.0, y + 1.0, z],
        ],
        [0, -1, 0] => [
            [x, y, z],
            [x + 1.0, y, z],
            [x + 1.0, y, z + 1.0],
            [x, y, z + 1.0],
        ],
        [0, 0, 1] => [
            [x, y, z + 1.0],
            [x + 1.0, y, z + 1.0],
            [x + 1.0, y + 1.0, z + 1.0],
            [x, y + 1.0, z + 1.0],
        ],
        _ => [
            [x, y, z],
            [x, y + 1.0, z],
            [x + 1.0, y + 1.0, z],
            [x + 1.0, y, z],
        ],
    }
}

/// Greedy-merge Full faces: per direction, per slice perpendicular to it, a
/// 16x16 occupancy mask of unoccluded faces, then maximal-rect expansion.
fn greedy_full_faces<S, F, G>(
    grid: &S,
    cells: &[[i32; 3]],
    input: &ChunkMeshInput<F, G>,
    mesh: &mut MeshGroup,
) where
    S: VoxelSource,
    F: Fn(u32) -> FaceKind,
    G: Fn(u32) -> bool,
{
    let tint = cells
        .first()
        .and_then(|c| grid.get(*c))
        .map(|b| tint_of(input, b.kind))
        .unwrap_or([1.0, 1.0, 1.0]);

    for dir in DIRS {
        let mut slices: HashMap<i32, Vec<[i32; 3]>> = HashMap::new();
        for &cell in cells {
            let ncell = [cell[0] + dir[0], cell[1] + dir[1], cell[2] + dir[2]];
            let occluded = match grid.get(ncell) {
                None => false,
                Some(nb) => {
                    (input.classify)(nb.kind) == FaceKind::Opaque
                        && (input.is_solid)(nb.kind)
                        && neighbor_occludes(&nb, [-dir[0], -dir[1], -dir[2]], true)
                }
            };
            if occluded {
                continue;
            }
            let slice = match dir {
                [1, 0, 0] | [-1, 0, 0] => cell[0],
                [0, 1, 0] | [0, -1, 0] => cell[1],
                _ => cell[2],
            };
            slices.entry(slice).or_default().push(cell);
        }

        for (_slice, members) in slices {
            let mut mask = [[false; 16]; 16];
            for &cell in &members {
                let (u, v) = mask_coords(cell, dir);
                if (0..16).contains(&u) && (0..16).contains(&v) {
                    mask[v as usize][u as usize] = true;
                }
            }
            let mut visited = [[false; 16]; 16];
            for v in 0..16usize {
                for u in 0..16usize {
                    if !mask[v][u] || visited[v][u] {
                        continue;
                    }
                    let mut w = 1;
                    while u + w < 16 && mask[v][u + w] && !visited[v][u + w] {
                        w += 1;
                    }
                    let mut h = 1;
                    'outer: while v + h < 16 {
                        for k in 0..w {
                            if !mask[v + h][u + k] || visited[v + h][u + k] {
                                break 'outer;
                            }
                        }
                        h += 1;
                    }
                    for dv in 0..h {
                        for du in 0..w {
                            visited[v + dv][u + du] = true;
                        }
                    }
                    let (origin, eu, ev) = merged_quad_frame(&members, dir, u, v, w, h);
                    let quad = [
                        origin,
                        [origin[0] + eu[0], origin[1] + eu[1], origin[2] + eu[2]],
                        [
                            origin[0] + eu[0] + ev[0],
                            origin[1] + eu[1] + ev[1],
                            origin[2] + eu[2] + ev[2],
                        ],
                        [origin[0] + ev[0], origin[1] + ev[1], origin[2] + ev[2]],
                    ];
                    let normal = [dir[0] as f32, dir[1] as f32, dir[2] as f32];
                    push_face(mesh, quad, tint, dir, normal, input.lit, input.textured);
                }
            }
        }
    }
}

/// Mask coords (u,v) for a cell on a face perpendicular to `dir`.
fn mask_coords(cell: [i32; 3], dir: [i32; 3]) -> (i32, i32) {
    let lx = cell[0].rem_euclid(CHUNK_SIZE);
    let ly = cell[1].rem_euclid(CHUNK_SIZE);
    let lz = cell[2].rem_euclid(CHUNK_SIZE);
    match dir {
        [1, 0, 0] | [-1, 0, 0] => (lz, ly),
        [0, 1, 0] | [0, -1, 0] => (lx, lz),
        _ => (lx, ly),
    }
}

/// World-space frame for a merged rect: origin + edge vectors.
/// Reconstructs from the slice members' plane coordinate so chunks away
/// from the origin stay exact.
fn merged_quad_frame(
    members: &[[i32; 3]],
    dir: [i32; 3],
    u: usize,
    v: usize,
    w: usize,
    h: usize,
) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let any = members[0];
    let ox = any[0].div_euclid(CHUNK_SIZE) * CHUNK_SIZE;
    let oy = any[1].div_euclid(CHUNK_SIZE) * CHUNK_SIZE;
    let oz = any[2].div_euclid(CHUNK_SIZE) * CHUNK_SIZE;
    match dir {
        [1, 0, 0] => {
            let px = members.iter().map(|c| c[0]).max().unwrap_or(ox) + 1;
            (
                [px as f32, oy as f32 + v as f32, oz as f32 + u as f32],
                [0.0, 0.0, w as f32],
                [0.0, h as f32, 0.0],
            )
        }
        [-1, 0, 0] => {
            let px = members.iter().map(|c| c[0]).min().unwrap_or(ox);
            (
                [px as f32, oy as f32 + v as f32, oz as f32 + u as f32],
                [0.0, 0.0, w as f32],
                [0.0, h as f32, 0.0],
            )
        }
        [0, 1, 0] => {
            let py = members.iter().map(|c| c[1]).max().unwrap_or(oy) + 1;
            (
                [ox as f32 + u as f32, py as f32, oz as f32 + v as f32],
                [w as f32, 0.0, 0.0],
                [0.0, 0.0, h as f32],
            )
        }
        [0, -1, 0] => {
            let py = members.iter().map(|c| c[1]).min().unwrap_or(oy);
            (
                [ox as f32 + u as f32, py as f32, oz as f32 + v as f32],
                [w as f32, 0.0, 0.0],
                [0.0, 0.0, h as f32],
            )
        }
        [0, 0, 1] => {
            let pz = members.iter().map(|c| c[2]).max().unwrap_or(oz) + 1;
            (
                [ox as f32 + u as f32, oy as f32 + v as f32, pz as f32],
                [w as f32, 0.0, 0.0],
                [0.0, h as f32, 0.0],
            )
        }
        _ => {
            let pz = members.iter().map(|c| c[2]).min().unwrap_or(oz);
            (
                [ox as f32 + u as f32, oy as f32 + v as f32, pz as f32],
                [w as f32, 0.0, 0.0],
                [0.0, h as f32, 0.0],
            )
        }
    }
}

/// Exact per-face quads for shaped (non-Full) cells: Half/TopHalf/Thin as
/// true boxes; other shapes as full-cube faces with rotation-aware culling
/// (conservative, never punches holes).
fn push_shaped_faces<S, F, G>(
    grid: &S,
    cell: [i32; 3],
    block: &Cell,
    tint: Rgb,
    input: &ChunkMeshInput<F, G>,
    mesh: &mut MeshGroup,
) where
    S: VoxelSource,
    F: Fn(u32) -> FaceKind,
    G: Fn(u32) -> bool,
{
    let (y0, y1): (f32, f32) = match block.shape {
        VoxelShape::Half => (0.0, 0.5),
        VoxelShape::TopHalf => (0.5, 1.0),
        VoxelShape::Thin => (1.0 - THIN_HEIGHT, 1.0),
        _ => (0.0, 1.0),
    };
    let f = [cell[0] as f32, cell[1] as f32, cell[2] as f32];
    for dir in DIRS {
        let ncell = [cell[0] + dir[0], cell[1] + dir[1], cell[2] + dir[2]];
        let occluded = match grid.get(ncell) {
            None => false,
            Some(nb) => {
                (input.classify)(nb.kind) == FaceKind::Opaque
                    && (input.is_solid)(nb.kind)
                    && neighbor_occludes(&nb, [-dir[0], -dir[1], -dir[2]], true)
                    && our_side_full(block.shape, dir)
            }
        };
        if occluded {
            continue;
        }
        let quad = shaped_face_quad(f, dir, y0, y1);
        let normal = [dir[0] as f32, dir[1] as f32, dir[2] as f32];
        push_face(mesh, quad, tint, dir, normal, input.lit, input.textured);
    }
}

fn shaped_face_quad(f: [f32; 3], dir: [i32; 3], y0: f32, y1: f32) -> [[f32; 3]; 4] {
    let [x, y, z] = f;
    match dir {
        [1, 0, 0] => [
            [x + 1.0, y + y0, z],
            [x + 1.0, y + y1, z],
            [x + 1.0, y + y1, z + 1.0],
            [x + 1.0, y + y0, z + 1.0],
        ],
        [-1, 0, 0] => [
            [x, y + y0, z + 1.0],
            [x, y + y1, z + 1.0],
            [x, y + y1, z],
            [x, y + y0, z],
        ],
        [0, 1, 0] => [
            [x, y + y1, z],
            [x, y + y1, z + 1.0],
            [x + 1.0, y + y1, z + 1.0],
            [x + 1.0, y + y1, z],
        ],
        [0, -1, 0] => [
            [x, y + y0, z],
            [x + 1.0, y + y0, z],
            [x + 1.0, y + y0, z + 1.0],
            [x, y + y0, z + 1.0],
        ],
        [0, 0, 1] => [
            [x, y + y0, z + 1.0],
            [x + 1.0, y + y0, z + 1.0],
            [x + 1.0, y + y1, z + 1.0],
            [x, y + y1, z + 1.0],
        ],
        _ => [
            [x, y + y0, z],
            [x, y + y1, z],
            [x + 1.0, y + y1, z],
            [x + 1.0, y + y0, z],
        ],
    }
}

fn push_water_quad<F, G>(
    water: &mut MeshGroup,
    f: [f32; 3],
    dir: [i32; 3],
    input: &ChunkMeshInput<F, G>,
) {
    if water.positions.is_empty() && water.indices.is_empty() {
        water.transparent = true;
        water.depth_test = true;
        water.alpha = input.water_alpha.clamp(0.0, 1.0);
    }
    let quad = full_face_quad(f, dir);
    let normal = [dir[0] as f32, dir[1] as f32, dir[2] as f32];
    let tint = [1.0, 1.0, 1.0];
    push_face(water, quad, tint, dir, normal, input.lit, input.textured);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Default input shape used across the voxel tests.
    type TestInput = ChunkMeshInput<fn(u32) -> FaceKind, fn(u32) -> bool>;

    const GRASS: u32 = 1;
    const STONE: u32 = 2;
    const WATER: u32 = 9;

    fn classify(kind: u32) -> FaceKind {
        match kind {
            WATER => FaceKind::Fluid,
            0 => FaceKind::Empty,
            _ => FaceKind::Opaque,
        }
    }

    fn is_solid(_: u32) -> bool {
        true
    }

    fn input() -> TestInput {
        ChunkMeshInput {
            classify,
            is_solid,
            kind_tint: HashMap::from([(GRASS, [0.3, 0.7, 0.2]), (STONE, [0.5, 0.5, 0.5])]),
            skip_overlay: HashSet::new(),
            water_level: None,
            submerged_tint: [0.55, 0.62, 0.78],
            water_alpha: 0.72,
            lit: false,
            textured: false,
        }
    }

    fn grid_of(cells: &[([i32; 3], Cell)]) -> HashMap<[i32; 3], Cell> {
        cells.iter().cloned().collect()
    }

    #[test]
    fn greedy_merges_flat_ground() {
        let mut cells = Vec::new();
        for x in 0..8 {
            for z in 0..8 {
                cells.push(([x, 0, z], Cell::full(GRASS)));
            }
        }
        let grid = grid_of(&cells);
        let mut out = ChunkMeshOutput::default();
        build_chunk_mesh(&grid, [0, 0, 0], &input(), &mut out);
        let mesh = &out.opaque[&GRASS];
        let quads = mesh.tri_count() / 2;
        assert!(
            quads <= 12,
            "expected greedy merge, got {quads} quads ({} tris)",
            mesh.tri_count()
        );
        assert!(out.merged_quads > 0);
        assert!(super::super::chunk::validate_group(mesh));
    }

    #[test]
    fn interior_faces_culled() {
        let mut cells = Vec::new();
        for x in 0..2 {
            for y in 0..2 {
                for z in 0..2 {
                    cells.push(([x, y, z], Cell::full(STONE)));
                }
            }
        }
        let grid = grid_of(&cells);
        let mut out = ChunkMeshOutput::default();
        build_chunk_mesh(&grid, [0, 0, 0], &input(), &mut out);
        assert_eq!(out.opaque[&STONE].tri_count() / 2, 6);
    }

    #[test]
    fn rotation_aware_occlusion() {
        let slab = Cell {
            kind: STONE,
            shape: VoxelShape::VerticalSlab,
            rot: 0,
            waterlogged: false,
        };
        assert!(neighbor_occludes(&slab, [0, 0, -1], true));
        assert!(!neighbor_occludes(&slab, [0, 0, 1], true));
        let slab180 = Cell { rot: 2, ..slab };
        assert!(!neighbor_occludes(&slab180, [0, 0, -1], true));
        assert!(neighbor_occludes(&slab180, [0, 0, 1], true));
        let slope = Cell {
            shape: VoxelShape::Slope,
            ..slab
        };
        assert!(!neighbor_occludes(&slope, [1, 0, 0], true));
        assert!(!neighbor_occludes(&slope, [0, 1, 0], true));
    }

    #[test]
    fn empty_kind_culls() {
        assert_eq!(classify(0), FaceKind::Empty);
        let grid = grid_of(&[([0, 0, 0], Cell::full(0))]);
        let mut out = ChunkMeshOutput::default();
        build_chunk_mesh(&grid, [0, 0, 0], &input(), &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn shaped_cells_take_exact_fallback() {
        let grid = grid_of(&[(
            [0, 0, 0],
            Cell {
                kind: STONE,
                shape: VoxelShape::Slope,
                rot: 0,
                waterlogged: false,
            },
        )]);
        let mut out = ChunkMeshOutput::default();
        build_chunk_mesh(&grid, [0, 0, 0], &input(), &mut out);
        let mesh = &out.opaque[&STONE];
        assert!(out.fallback_quads > 0);
        assert_eq!(out.merged_quads, 0);
        assert!(mesh.tri_count() > 0);
        assert!(super::super::chunk::validate_group(mesh));
    }

    #[test]
    fn half_slab_is_a_true_box() {
        let grid = grid_of(&[(
            [0, 0, 0],
            Cell {
                kind: STONE,
                shape: VoxelShape::Half,
                rot: 0,
                waterlogged: false,
            },
        )]);
        let mut out = ChunkMeshOutput::default();
        build_chunk_mesh(&grid, [0, 0, 0], &input(), &mut out);
        let mesh = &out.opaque[&STONE];
        assert_eq!(mesh.tri_count() / 2, 6);
        let top = mesh
            .positions
            .iter()
            .map(|p| p[1])
            .fold(f32::NEG_INFINITY, f32::max);
        assert!((top - 0.5).abs() < 1e-6, "slab top at 0.5: {top}");
    }

    #[test]
    fn water_emits_transparent_group() {
        let grid = grid_of(&[([0, 0, 0], Cell::full(WATER))]);
        let mut out = ChunkMeshOutput::default();
        build_chunk_mesh(&grid, [0, 0, 0], &input(), &mut out);
        assert!(out.opaque.is_empty());
        assert!(out.water.transparent);
        assert!((out.water.alpha - 0.72).abs() < 1e-6);
        assert_eq!(out.water.tri_count() / 2, 6);
    }

    #[test]
    fn water_covered_faces_cull() {
        let grid = grid_of(&[
            ([0, 0, 0], Cell::full(WATER)),
            ([0, 1, 0], Cell::full(STONE)),
        ]);
        let mut out = ChunkMeshOutput::default();
        build_chunk_mesh(&grid, [0, 0, 0], &input(), &mut out);
        assert_eq!(out.water.tri_count() / 2, 5, "top face culled");
    }

    #[test]
    fn untextured_default_has_no_uvs() {
        let mut cells = Vec::new();
        for x in 0..2 {
            for z in 0..2 {
                cells.push(([x, 0, z], Cell::full(GRASS)));
            }
        }
        let grid = grid_of(&cells);
        let mut out = ChunkMeshOutput::default();
        build_chunk_mesh(&grid, [0, 0, 0], &input(), &mut out);
        assert!(out.opaque[&GRASS].uvs.is_empty());
        assert!(out.opaque[&GRASS].normals.is_empty());
    }

    #[test]
    fn lit_build_carries_normals_and_shade() {
        let mut cells = Vec::new();
        for x in 0..2 {
            for z in 0..2 {
                cells.push(([x, 0, z], Cell::full(GRASS)));
            }
        }
        let grid = grid_of(&cells);
        let mut inp = input();
        inp.lit = true;
        let mut out = ChunkMeshOutput::default();
        build_chunk_mesh(&grid, [0, 0, 0], &inp, &mut out);
        let mesh = &out.opaque[&GRASS];
        assert_eq!(mesh.normals.len(), mesh.positions.len());
        assert!(mesh.colors.contains(&[0.3, 0.7, 0.2]));
        assert!(mesh.colors.iter().any(|c| c[0] < 0.3));
    }

    #[test]
    fn seam_faces_cull_across_chunks() {
        let grid = grid_of(&[
            ([15, 0, 0], Cell::full(STONE)),
            ([16, 0, 0], Cell::full(STONE)),
        ]);
        let mut left = ChunkMeshOutput::default();
        build_chunk_mesh(&grid, [0, 0, 0], &input(), &mut left);
        let mut right = ChunkMeshOutput::default();
        build_chunk_mesh(&grid, [1, 0, 0], &input(), &mut right);
        let total = left.tri_count() / 2 + right.tri_count() / 2;
        assert_eq!(total, 10, "12 faces minus 2 shared: {total}");
    }
}
