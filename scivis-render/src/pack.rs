//! A volume laid out the way the GPU will hold it, built on the CPU.
//!
//! Everything a 3D upload needs that costs time in proportion to the volume
//! is done here, by whoever builds the volume (a background thread, in the
//! FastTIFF viewer), so the upload itself is a plain copy:
//!
//! * **Channels packed into shared textures.** Up to four channels share one
//!   RGBA texture (a fifth and sixth go in a second), so one texture read
//!   returns every channel at a point instead of one read per channel. Texels
//!   are stored in the backend's [`TexelFormat`], already converted.
//! * **A mip chain**, each level half the last per axis. A level holds the
//!   **maximum** of the texels it covers, not their mean: the levels are read by
//!   the maximum-intensity projection when a voxel is smaller than a pixel, and
//!   averaging would dim exactly the thin bright structures (a dendrite, a
//!   bead) a MIP is for.
//! * **Per-block maxima** ([`BrickMaxima`]): the largest value of each channel
//!   in each `BRICK`-voxel cube, from which the renderer works out which parts
//!   of the volume can contribute nothing under the current contrast window
//!   ([`occupancy`]) and lets rays jump over them.
//!
//! Values are in **texture units** throughout: normalized `0..=1` for integer
//! sources (`raw/255`, `raw/65535`), the raw value for float sources, which is
//! what the shaders' per-channel window is expressed in.

use crate::{VolumeKind, VolumeRender, MAX_CHANNELS};

/// Side of the cubes [`BrickMaxima`] summarises, in level-0 voxels.
pub const BRICK: u32 = 16;

/// Mip levels a volume gets at most, level 0 included. Three halvings is as
/// far as a level can be read while the renderer's one-brick margin still
/// covers what its sample spans (see [`occupancy`]).
pub const MAX_LEVELS: u32 = 4;

/// Channels one packed texture holds.
const PER_TEXTURE: usize = 4;

/// How one stored component is encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TexelFormat {
    /// `round(v * 255)` in a byte.
    Unorm8,
    /// `round(v * 65535)` in a native-endian `u16`.
    Unorm16,
    /// An IEEE half float, native-endian.
    Float16,
    /// An `f32`, native-endian.
    Float32,
}

impl TexelFormat {
    /// Bytes one component takes.
    pub fn bytes(self) -> usize {
        match self {
            TexelFormat::Unorm8 => 1,
            TexelFormat::Unorm16 | TexelFormat::Float16 => 2,
            TexelFormat::Float32 => 4,
        }
    }

    /// Whether this format stores values normalized to `0..=1`, which is only
    /// right for integer sources.
    pub fn is_unorm(self) -> bool {
        matches!(self, TexelFormat::Unorm8 | TexelFormat::Unorm16)
    }

    /// Store `v` (texture units) into `dst`, which is [`bytes`](Self::bytes) long.
    #[inline]
    pub fn encode(self, v: f32, dst: &mut [u8]) {
        match self {
            TexelFormat::Unorm8 => dst[0] = unorm(v, 255.0) as u8,
            TexelFormat::Unorm16 => {
                dst[..2].copy_from_slice(&(unorm(v, 65535.0) as u16).to_ne_bytes())
            }
            TexelFormat::Float16 => {
                dst[..2].copy_from_slice(&half::f16::from_f32(v).to_bits().to_ne_bytes())
            }
            TexelFormat::Float32 => dst[..4].copy_from_slice(&v.to_ne_bytes()),
        }
    }

    /// The lowest finite value this format stores, in texture units: what a
    /// mip texel holds where everything below it is NaN or minus infinity.
    /// Finite, so filtering never multiplies it into a NaN, and below any
    /// window, so it shows as nothing — as the NaN it stands for does.
    pub fn lowest(self) -> f32 {
        match self {
            TexelFormat::Unorm8 | TexelFormat::Unorm16 => 0.0,
            TexelFormat::Float16 => half::f16::MIN.to_f32(),
            TexelFormat::Float32 => f32::MIN,
        }
    }

    /// Read back what [`encode`](Self::encode) stored.
    #[inline]
    pub fn decode(self, src: &[u8]) -> f32 {
        match self {
            TexelFormat::Unorm8 => src[0] as f32 / 255.0,
            TexelFormat::Unorm16 => u16::from_ne_bytes([src[0], src[1]]) as f32 / 65535.0,
            TexelFormat::Float16 => {
                half::f16::from_bits(u16::from_ne_bytes([src[0], src[1]])).to_f32()
            }
            TexelFormat::Float32 => f32::from_ne_bytes([src[0], src[1], src[2], src[3]]),
        }
    }
}

/// `v` in `0..=1` as an integer of `0..=scale`, rounded. NaN stores as 0.
#[inline]
fn unorm(v: f32, scale: f32) -> f32 {
    if v.is_nan() {
        return 0.0;
    }
    (v.clamp(0.0, 1.0) * scale).round()
}

/// The kind every channel of a volume is stored as: float if any channel is,
/// else 16-bit if any is, else 8-bit. A stack's channels share a sample type
/// in practice; this only decides the rare mixed case, conservatively.
pub fn common_kind(kinds: &[VolumeKind]) -> VolumeKind {
    if kinds.contains(&VolumeKind::F32) {
        VolumeKind::F32
    } else if kinds.contains(&VolumeKind::U16) {
        VolumeKind::U16
    } else {
        VolumeKind::U8
    }
}

/// One packed texture: which channels it holds and how many components it has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureLayout {
    /// The first channel it holds; the others follow in component order.
    pub first_channel: usize,
    /// Channels it holds (1..=4).
    pub channels: usize,
    /// Components per texel: 1, 2 or 4. Three channels take an RGBA texture
    /// with the fourth component unused, as GPUs have no three-component
    /// sampled formats.
    pub components: usize,
}

/// Where every channel of a volume lives, and how big it all is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VolumeLayout {
    pub dims: (u32, u32, u32),
    pub channels: usize,
    pub format: TexelFormat,
    /// One or two textures.
    pub textures: Vec<TextureLayout>,
    /// Mip levels, level 0 included.
    pub levels: u32,
}

impl VolumeLayout {
    /// The layout for `channels` channels (clamped to `1..=MAX_CHANNELS`) of
    /// `dims` voxels stored as `format`.
    pub fn new(dims: (u32, u32, u32), channels: usize, format: TexelFormat) -> VolumeLayout {
        let channels = channels.clamp(1, MAX_CHANNELS);
        let mut textures = Vec::new();
        let mut first = 0;
        while first < channels {
            let n = (channels - first).min(PER_TEXTURE);
            textures.push(TextureLayout {
                first_channel: first,
                channels: n,
                components: match n {
                    1 => 1,
                    2 => 2,
                    _ => 4,
                },
            });
            first += n;
        }
        let dims = (dims.0.max(1), dims.1.max(1), dims.2.max(1));
        VolumeLayout {
            dims,
            channels,
            format,
            textures,
            levels: mip_levels(dims),
        }
    }

    /// The size of level `level`: each axis halved per level, rounded down, at
    /// least 1 — the rule GPUs use for mip chains.
    pub fn level_dims(&self, level: u32) -> (u32, u32, u32) {
        level_dims(self.dims, level)
    }

    /// Bytes of one texel of texture `t`.
    pub fn texel_bytes(&self, t: usize) -> usize {
        self.textures[t].components * self.format.bytes()
    }

    /// Bytes of level `level` of texture `t`.
    pub fn level_bytes(&self, t: usize, level: u32) -> usize {
        let (w, h, d) = self.level_dims(level);
        w as usize * h as usize * d as usize * self.texel_bytes(t)
    }

    /// Bytes of one z-slice of level 0 of texture `t`.
    pub fn slice_bytes(&self, t: usize) -> usize {
        self.dims.0 as usize * self.dims.1 as usize * self.texel_bytes(t)
    }

    /// Every byte the GPU will hold for this volume, every level of every
    /// texture.
    pub fn total_bytes(&self) -> usize {
        (0..self.textures.len())
            .map(|t| {
                (0..self.levels)
                    .map(|l| self.level_bytes(t, l))
                    .sum::<usize>()
            })
            .sum()
    }

    /// Which texture channel `c` is in, and which component of it.
    pub fn place(&self, c: usize) -> (usize, usize) {
        (c / PER_TEXTURE, c % PER_TEXTURE)
    }
}

/// How many mip levels a volume of `dims` gets.
pub fn mip_levels(dims: (u32, u32, u32)) -> u32 {
    let largest = dims.0.max(dims.1).max(dims.2).max(1);
    (32 - largest.leading_zeros()).min(MAX_LEVELS)
}

/// Bytes the GPU holds per level-0 voxel for `channels` channels in `format`,
/// mip chain included (it adds at most an eighth, then a sixty-fourth, ...),
/// for a budget to divide by.
pub fn bytes_per_voxel(channels: usize, format: TexelFormat) -> f64 {
    let layout = VolumeLayout::new((1, 1, 1), channels, format);
    let components: usize = layout.textures.iter().map(|t| t.components).sum();
    // 1 + 1/8 + 1/64 + 1/512: the most three halvings can add.
    components as f64 * format.bytes() as f64 * (1.0 + 1.0 / 8.0 + 1.0 / 64.0 + 1.0 / 512.0)
}

fn level_dims(dims: (u32, u32, u32), level: u32) -> (u32, u32, u32) {
    (
        (dims.0 >> level).max(1),
        (dims.1 >> level).max(1),
        (dims.2 >> level).max(1),
    )
}

/// The largest value of each channel in each [`BRICK`]-voxel cube of level 0.
#[derive(Clone, Debug, PartialEq)]
pub struct BrickMaxima {
    /// Bricks per axis.
    pub dims: (u32, u32, u32),
    /// Per channel, `dims.0 * dims.1 * dims.2` maxima, x fastest. A brick
    /// whose samples are all NaN holds negative infinity.
    pub max: Vec<Vec<f32>>,
}

impl BrickMaxima {
    /// Bricks per axis for a level-0 volume of `dims`.
    pub fn grid(dims: (u32, u32, u32)) -> (u32, u32, u32) {
        (
            dims.0.div_ceil(BRICK).max(1),
            dims.1.div_ceil(BRICK).max(1),
            dims.2.div_ceil(BRICK).max(1),
        )
    }
}

/// A volume ready to upload: every level of every texture, plus its brick
/// maxima.
#[derive(Clone, Debug)]
pub struct PackedVolume {
    pub layout: VolumeLayout,
    /// Per texture, per level (level 0 first), the texels, x fastest, then y,
    /// then z, components interleaved.
    pub levels: Vec<Vec<Vec<u8>>>,
    pub bricks: BrickMaxima,
}

impl PackedVolume {
    /// A volume of `layout` with level 0 zeroed and nothing else yet: fill
    /// level 0 (see [`level0_mut`](Self::level0_mut)), then call
    /// [`finish`](Self::finish).
    pub fn new(layout: VolumeLayout) -> PackedVolume {
        let levels = (0..layout.textures.len())
            .map(|t| vec![vec![0u8; layout.level_bytes(t, 0)]])
            .collect();
        let bricks = BrickMaxima {
            dims: BrickMaxima::grid(layout.dims),
            max: Vec::new(),
        };
        PackedVolume {
            layout,
            levels,
            bricks,
        }
    }

    /// Level 0 of every texture, to fill. Texture `t`'s z-slice `z` is the
    /// `layout.slice_bytes(t)`-byte chunk at `z`, so a builder can hand
    /// disjoint slices to parallel workers with `chunks_mut`.
    pub fn level0_mut(&mut self) -> Vec<&mut [u8]> {
        self.levels
            .iter_mut()
            .map(|l| l[0].as_mut_slice())
            .collect()
    }

    /// Pack channels given one buffer each — `kind` samples, native-endian,
    /// `dims` voxels, x fastest — and finish. The straightforward way in, for
    /// a caller that already has whole channels (tests, small volumes); a
    /// builder writing slices in parallel fills [`level0_mut`] instead.
    ///
    /// [`level0_mut`]: Self::level0_mut
    pub fn from_channels(
        dims: (u32, u32, u32),
        channels: &[(VolumeKind, &[u8])],
        format: TexelFormat,
    ) -> PackedVolume {
        let layout = VolumeLayout::new(dims, channels.len(), format);
        let mut packed = PackedVolume::new(layout.clone());
        let voxels = dims.0 as usize * dims.1 as usize * dims.2 as usize;
        for (c, (kind, bytes)) in channels.iter().enumerate().take(layout.channels) {
            let (t, k) = layout.place(c);
            let stride = layout.texel_bytes(t);
            let fb = format.bytes();
            let dst = &mut packed.levels[t][0];
            for i in 0..voxels {
                let v = read_sample(*kind, bytes, i);
                let at = i * stride + k * fb;
                format.encode(v, &mut dst[at..at + fb]);
            }
        }
        packed.finish();
        packed
    }

    /// Whether every buffer is the size the layout says — every level of
    /// every texture, and the brick maxima of every channel — which is what
    /// an upload hands the GPU. Fails for a volume that was never
    /// [`finish`](Self::finish)ed, or whose fields were changed after.
    pub fn check(&self) -> Result<(), String> {
        let layout = &self.layout;
        // The textures are what the backends upload them as only in the
        // packing `VolumeLayout::new` chooses: three components, say, would
        // be read as four.
        let packing = VolumeLayout::new(layout.dims, layout.channels, layout.format);
        if (packing.dims, packing.channels, &packing.textures)
            != (layout.dims, layout.channels, &layout.textures)
        {
            return Err("the layout is not one VolumeLayout::new makes".into());
        }
        if layout.levels == 0 || layout.levels > mip_levels(layout.dims) {
            return Err(format!(
                "{} mip levels for a volume that has {}",
                layout.levels,
                mip_levels(layout.dims)
            ));
        }
        if self.levels.len() != layout.textures.len() {
            return Err(format!(
                "levels for {} textures, where the layout has {}",
                self.levels.len(),
                layout.textures.len()
            ));
        }
        for (t, levels) in self.levels.iter().enumerate() {
            if levels.len() != layout.levels as usize {
                return Err(format!(
                    "texture {t} has {} of its {} levels: finish() builds the rest",
                    levels.len(),
                    layout.levels
                ));
            }
            for (level, data) in levels.iter().enumerate() {
                let want = layout.level_bytes(t, level as u32);
                if data.len() != want {
                    return Err(format!(
                        "level {level} of texture {t} is {} bytes, where its size takes {want}",
                        data.len()
                    ));
                }
            }
        }
        let grid = BrickMaxima::grid(layout.dims);
        let bricks = grid.0 as usize * grid.1 as usize * grid.2 as usize;
        if self.bricks.dims != grid
            || self.bricks.max.len() != layout.channels
            || self.bricks.max.iter().any(|m| m.len() != bricks)
        {
            return Err("the brick maxima do not match the volume: finish() builds them".into());
        }
        Ok(())
    }

    /// Build the mip chain and the brick maxima from level 0.
    pub fn finish(&mut self) {
        let layout = &self.layout;
        let format = layout.format;
        for (t, tex) in layout.textures.iter().enumerate() {
            let levels = &mut self.levels[t];
            levels.truncate(1);
            for level in 1..layout.levels {
                let src_dims = layout.level_dims(level - 1);
                let dst_dims = layout.level_dims(level);
                let next = max_reduce(
                    &levels[level as usize - 1],
                    src_dims,
                    dst_dims,
                    tex.components,
                    format,
                );
                levels.push(next);
            }
        }
        self.bricks = brick_maxima(layout, &self.levels);
    }
}

/// Sample `i` of a `kind` buffer, in texture units.
#[inline]
pub fn read_sample(kind: VolumeKind, bytes: &[u8], i: usize) -> f32 {
    match kind {
        VolumeKind::U8 => bytes[i] as f32 / 255.0,
        VolumeKind::U16 => u16::from_ne_bytes([bytes[2 * i], bytes[2 * i + 1]]) as f32 / 65535.0,
        VolumeKind::F32 => f32::from_ne_bytes([
            bytes[4 * i],
            bytes[4 * i + 1],
            bytes[4 * i + 2],
            bytes[4 * i + 3],
        ]),
    }
}

/// The source texels a destination index covers along one axis: its two
/// texels, and on an odd-sized axis the last destination also takes the
/// leftover, so the maximum misses nothing.
#[inline]
fn cover(i: u32, dst: u32, src: u32) -> std::ops::Range<u32> {
    let lo = (2 * i).min(src - 1);
    let mut hi = (2 * i + 2).min(src);
    if i + 1 == dst {
        hi = src;
    }
    lo..hi.max(lo + 1)
}

/// The next mip level down: each texel the maximum of the ones it covers,
/// component by component. NaN never wins; a texel over nothing but NaN and
/// minus infinity holds [`TexelFormat::lowest`].
fn max_reduce(
    src: &[u8],
    src_dims: (u32, u32, u32),
    dst_dims: (u32, u32, u32),
    components: usize,
    format: TexelFormat,
) -> Vec<u8> {
    let fb = format.bytes();
    let texel = components * fb;
    let (sw, sh, sd) = src_dims;
    let (dw, dh, dd) = dst_dims;
    let mut out = vec![0u8; dw as usize * dh as usize * dd as usize * texel];
    let slab = dw as usize * dh as usize * texel;
    let fill = |z: u32, dst: &mut [u8]| {
        let zs = cover(z, dd, sd);
        for y in 0..dh {
            let ys = cover(y, dh, sh);
            for x in 0..dw {
                let xs = cover(x, dw, sw);
                for k in 0..components {
                    let mut best = f32::NEG_INFINITY;
                    for sz in zs.clone() {
                        for sy in ys.clone() {
                            let row = (sz as usize * sh as usize + sy as usize) * sw as usize;
                            for sx in xs.clone() {
                                let at = (row + sx as usize) * texel + k * fb;
                                let v = format.decode(&src[at..at + fb]);
                                if v > best {
                                    best = v;
                                }
                            }
                        }
                    }
                    if best == f32::NEG_INFINITY {
                        best = format.lowest();
                    }
                    let at = (y as usize * dw as usize + x as usize) * texel + k * fb;
                    format.encode(best, &mut dst[at..at + fb]);
                }
            }
        }
    };
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(slab)
            .enumerate()
            .for_each(|(z, dst)| fill(z as u32, dst));
    }
    #[cfg(not(feature = "rayon"))]
    for (z, dst) in out.chunks_mut(slab).enumerate() {
        fill(z as u32, dst);
    }
    out
}

/// The largest value of every channel in every brick of level 0.
fn brick_maxima(layout: &VolumeLayout, levels: &[Vec<Vec<u8>>]) -> BrickMaxima {
    let grid = BrickMaxima::grid(layout.dims);
    let (w, h, d) = layout.dims;
    let (bx, by, bz) = grid;
    let fb = layout.format.bytes();
    let cells = bx as usize * by as usize * bz as usize;
    let empty = || vec![vec![f32::NEG_INFINITY; cells]; layout.channels];
    // Fold one z-slice of level 0 into a grid of maxima. Each thread keeps a
    // whole (small) grid of its own and the grids are merged at the end, so the
    // work splits by slice however few brick layers there are.
    let slice = |mut grid: Vec<Vec<f32>>, z: u32| -> Vec<Vec<f32>> {
        let layer = (z / BRICK) as usize * bx as usize * by as usize;
        for y in 0..h {
            let row = layer + (y / BRICK) as usize * bx as usize;
            let base = (z as usize * h as usize + y as usize) * w as usize;
            for (c, grid_c) in grid.iter_mut().enumerate() {
                let (t, k) = layout.place(c);
                let texel = layout.texel_bytes(t);
                let data = &levels[t][0];
                for x in 0..w {
                    let at = (base + x as usize) * texel + k * fb;
                    let v = layout.format.decode(&data[at..at + fb]);
                    let slot = &mut grid_c[row + (x / BRICK) as usize];
                    if v > *slot {
                        *slot = v;
                    }
                }
            }
        }
        grid
    };
    let merge = |mut a: Vec<Vec<f32>>, b: Vec<Vec<f32>>| -> Vec<Vec<f32>> {
        for (a_c, b_c) in a.iter_mut().zip(b) {
            for (x, y) in a_c.iter_mut().zip(b_c) {
                if y > *x {
                    *x = y;
                }
            }
        }
        a
    };
    #[cfg(feature = "rayon")]
    let max = {
        use rayon::prelude::*;
        (0..d)
            .into_par_iter()
            .fold(empty, slice)
            .reduce(empty, merge)
    };
    #[cfg(not(feature = "rayon"))]
    let max = {
        let _ = merge;
        (0..d).fold(empty(), slice)
    };
    BrickMaxima { dims: grid, max }
}

/// What decides whether a part of the volume can be seen: the per-channel
/// contrast windows (in texture units, as `(min, max)` pairs), which channels
/// are on, the compositing mode and the isosurface threshold.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Visibility {
    pub num_channels: usize,
    pub windows: [f32; MAX_CHANNELS * 2],
    pub enabled: [bool; MAX_CHANNELS],
    pub render: VolumeRender,
    pub iso: f32,
}

impl Visibility {
    /// What `params` makes visible.
    pub fn of(params: &crate::VolumeParams) -> Visibility {
        let mut enabled = [false; MAX_CHANNELS];
        for (e, &v) in enabled.iter_mut().zip(&params.enabled) {
            *e = v > 0.5;
        }
        Visibility {
            num_channels: params.num_channels.clamp(0, MAX_CHANNELS as i32) as usize,
            windows: params.windows,
            enabled,
            render: match params.render_mode {
                1 => VolumeRender::Alpha,
                2 => VolumeRender::Surface,
                _ => VolumeRender::Mip,
            },
            iso: params.iso,
        }
    }
}

/// Which bricks can contribute to the picture under `vis`: one byte per brick
/// of `bricks`, 255 where a ray must sample, 0 where it may skip.
///
/// A brick contributes when some enabled channel's maximum in it would come
/// through that channel's window above zero — for the maximum projection and
/// the volume rendering, where a windowed zero adds nothing — or would reach
/// the threshold, for the isosurface. Every other brick holds only samples the
/// shaders would turn into nothing, so skipping them changes no pixel.
///
/// Then **grown by one brick** in every direction: a sample near a brick's
/// face is filtered from voxels across it, and a mip level spans several, so a
/// ray may only skip a brick whose whole neighbourhood is empty. One brick
/// ([`BRICK`] voxels) covers trilinear and tricubic filtering at every level
/// up to [`MAX_LEVELS`].
pub fn occupancy(bricks: &BrickMaxima, vis: &Visibility) -> Vec<u8> {
    let (bx, by, bz) = bricks.dims;
    let n = bx as usize * by as usize * bz as usize;
    let surface = vis.render == VolumeRender::Surface;
    // A threshold at or below zero is met by every sample, windowed to zero or
    // not: nothing can be skipped.
    if surface && vis.iso <= 0.0 {
        return vec![255; n];
    }
    let windows: Vec<(usize, f32, f32)> = (0..vis.num_channels.min(bricks.max.len()))
        .filter(|&c| vis.enabled[c])
        .map(|c| {
            let lo = vis.windows[2 * c];
            let hi = vis.windows[2 * c + 1];
            // The shaders' own windowing: (v - lo) / max(hi - lo, 1e-6).
            (c, lo, (hi - lo).max(1e-6))
        })
        .collect();
    // Filtering (trilinear, the B-spline, the max-reduced levels) only ever
    // blends, so no sample in a brick's neighbourhood exceeds its maximum, and
    // windowing is monotonic: the maximum decides. For the projection and the
    // volume rendering the test is a sign, exact in floating point; the
    // threshold comparison is the shaders' expression with a margin for a GPU
    // dividing a few units in the last place differently.
    let mut seed = vec![0u8; n];
    for (i, s) in seed.iter_mut().enumerate() {
        let hot = windows.iter().any(|&(c, lo, span)| {
            let m = bricks.max[c][i];
            if surface {
                (m - lo) / span + 1e-5 >= vis.iso
            } else {
                m > lo
            }
        });
        if hot {
            *s = 255;
        }
    }
    dilate(&seed, bricks.dims)
}

/// Every cell within one cell (26 neighbours) of a set one, set.
fn dilate(src: &[u8], (bx, by, bz): (u32, u32, u32)) -> Vec<u8> {
    let (bx, by, bz) = (bx as i64, by as i64, bz as i64);
    let at = |x: i64, y: i64, z: i64| ((z * by + y) * bx + x) as usize;
    let mut out = vec![0u8; src.len()];
    for z in 0..bz {
        for y in 0..by {
            for x in 0..bx {
                if src[at(x, y, z)] == 0 {
                    continue;
                }
                for dz in -1..=1 {
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let (nx, ny, nz) = (x + dx, y + dy, z + dz);
                            if (0..bx).contains(&nx)
                                && (0..by).contains(&ny)
                                && (0..bz).contains(&nz)
                            {
                                out[at(nx, ny, nz)] = 255;
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
#[path = "pack_tests.rs"]
mod tests;
