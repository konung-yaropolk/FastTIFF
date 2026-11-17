//! The scene the GPU tests draw: a 2D image and a small volume, each of
//! `MAX_CHANNELS` channels, with the LUTs, windows and views to show them.
//!
//! `parity.rs` draws it through both backends and compares the pictures;
//! `gl.rs` draws it through the glow backend alone, on whatever GL the
//! machine has, to see that every draw is one the driver accepts. Every
//! channel there is, because each shader writes its per-channel code out by
//! hand, once per channel: a channel the scene leaves out is a line of each
//! copy that nothing draws, and so a drift between the copies nothing sees.

#![cfg(not(target_arch = "wasm32"))]

use scivis_render::{
    brightest_lut_t, ChannelKind, ChannelUniform, ImageRenderer, Lut, VolumeInterp, VolumeKind,
    VolumeParams, VolumeRender, VolumeRenderer, MAX_CHANNELS,
};

/// The share of an RGBA picture's pixels that are not all but black: a
/// picture that is not lit at all was not drawn, whatever it matches.
pub fn lit(rgba: &[u8]) -> f64 {
    let pixels = rgba.chunks(4);
    let count = pixels.len().max(1);
    pixels.filter(|px| px[..3].iter().any(|&c| c > 16)).count() as f64 / count as f64
}

/// A LUT from a function of the position along it, 0..=1.
pub fn lut(f: impl Fn(f32) -> [f32; 3]) -> Lut {
    std::array::from_fn(|i| f(i as f32 / 255.0).map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8))
}

// --- 2D composite --------------------------------------------------------

pub const W: u32 = 61;
pub const H: u32 = 37;

pub enum Samples {
    U8(Vec<u8>),
    U16(Vec<u16>),
    F32(Vec<f32>),
}

pub struct Channel {
    pub samples: Samples,
    pub lut: Lut,
    /// Its window; `enabled` is the view's to decide (see [`View::off`]).
    pub min: f32,
    pub max: f32,
}

fn samples<T>(f: impl Fn(u32, u32) -> T) -> Vec<T> {
    (0..H)
        .flat_map(|y| (0..W).map(move |x| (x, y)))
        .map(|(x, y)| f(x, y))
        .collect()
}

/// `MAX_CHANNELS` channels of every kind (8-bit, 16-bit, float with negative
/// values), each through a LUT of its own, kept dim enough that their sum
/// seldom clips, and windowed into the middle of its range so that both
/// clamps show.
pub fn image() -> Vec<Channel> {
    let channel = |samples, lut, min, max| Channel {
        samples,
        lut,
        min,
        max,
    };
    let channels = vec![
        channel(
            Samples::U16(samples(|x, y| ((x * 997 + y * 313) % 4096 * 16) as u16)),
            lut(|t| [0.55 * t, 0.2 * t, 0.0]),
            8000.0,
            52000.0,
        ),
        channel(
            Samples::U8(samples(|x, y| ((x * 4 + y * 7) % 256) as u8)),
            lut(|t| [0.0, 0.55 * t, 0.3 * t * t]),
            30.0,
            220.0,
        ),
        channel(
            Samples::F32(samples(|x, y| {
                (x as f32 * 0.3).sin() * (y as f32 * 0.2).cos() * 100.0 - 20.0
            })),
            // Not monotonic: the LUT is read where the window puts it.
            lut(|t| [(t * 6.0).sin().abs() * 0.45, 0.0, 0.4 * t]),
            -60.0,
            70.0,
        ),
        channel(
            Samples::U16(samples(|x, y| (x * y * 50) as u16)),
            lut(|t| [0.3 * t, 0.3 * t, 0.3 * t]),
            0.0,
            65535.0,
        ),
        channel(
            Samples::F32(samples(|x, y| (x as f32 - y as f32) * 0.5)),
            lut(|t| [0.15 * t, 0.3 * t, 0.45 * t]),
            -10.0,
            25.0,
        ),
        channel(
            Samples::U8(samples(|x, y| ((x * 13 + y * 5) % 256) as u8)),
            // Not black at 0: below its window the channel still adds light.
            lut(|t| [0.4 * t * t, 0.1, 0.35 * (1.0 - t)]),
            40.0,
            200.0,
        ),
    ];
    assert_eq!(channels.len(), MAX_CHANNELS);
    channels
}

/// One way of showing the image: the target's size, the part of the image
/// that fills it, and the channel switched off.
pub struct View {
    pub name: &'static str,
    pub size: (u32, u32),
    pub uv_offset: [f32; 2],
    pub uv_scale: [f32; 2],
    /// A different one in each view, so that every channel is drawn in at
    /// least two views and the switched-off path is taken in all three.
    pub off: usize,
}

/// Sizes chosen so that no pixel centre falls on a texel edge and the box
/// filter's tap count is not on a rounding edge, where the two backends
/// could round the other way and both be right.
pub const VIEWS: [View; 3] = [
    View {
        name: "magnified",
        size: (150, 91),
        uv_offset: [0.0, 0.0],
        uv_scale: [1.0, 1.0],
        off: 3,
    },
    View {
        name: "minified",
        size: (25, 15),
        uv_offset: [0.0, 0.0],
        uv_scale: [1.0, 1.0],
        off: 5,
    },
    View {
        name: "panned",
        size: (97, 53),
        uv_offset: [0.2, 0.3],
        uv_scale: [0.5, 0.4],
        off: 0,
    },
];

/// The uniforms `view` shows `channels` with.
pub fn uniforms(channels: &[Channel], view: &View) -> Vec<ChannelUniform> {
    channels
        .iter()
        .enumerate()
        .map(|(c, channel)| ChannelUniform {
            min: channel.min,
            max: channel.max,
            enabled: c != view.off,
            is_float: matches!(channel.samples, Samples::F32(_)),
        })
        .collect()
}

/// Upload `channels` to the renderer `r`, of either backend, and set it to
/// show `view`.
pub fn show_image(r: &mut impl ImageRenderer, channels: &[Channel], view: &View) {
    let kinds: Vec<ChannelKind> = channels
        .iter()
        .map(|c| match c.samples {
            Samples::U8(_) => ChannelKind::Int8,
            Samples::U16(_) => ChannelKind::Int16,
            Samples::F32(_) => ChannelKind::Float,
        })
        .collect();
    r.ensure_size(W, H, &kinds);
    for (c, channel) in channels.iter().enumerate() {
        r.upload_lut(c, &channel.lut);
        match &channel.samples {
            Samples::U8(s) => r.upload_channel_u8(c, W, H, s),
            Samples::U16(s) => r.upload_channel_u16(c, W, H, s),
            Samples::F32(s) => r.upload_channel_f32(c, W, H, s),
        }
    }
    r.set_params(
        &uniforms(channels, view),
        channels.len() as u32,
        view.uv_offset,
        view.uv_scale,
    );
}

// --- 3D volume -----------------------------------------------------------

pub const DIMS: (u32, u32, u32) = (40, 36, 32);

/// A smooth bump of radius `r` voxels around `c`, `peak` at its centre and
/// zero outside: smooth, so that sampling it a little apart reads a little
/// apart.
fn bump((x, y, z): (u32, u32, u32), c: [f32; 3], r: f32, peak: f32) -> f32 {
    let d2 = (x as f32 - c[0]).powi(2) + (y as f32 - c[1]).powi(2) + (z as f32 - c[2]).powi(2);
    let s = (1.0 - d2 / (r * r)).max(0.0);
    peak * s * s
}

/// `MAX_CHANNELS` channels of 8-bit voxels, so that the second packed texture
/// (channels 4 and 5) is read as well as the first. Each a bump of its own,
/// in a part of the box of its own, that keeps at least three voxels clear of
/// every face, where the backends differ by design (`parity.rs`).
pub fn volume() -> Vec<(VolumeKind, Vec<u8>)> {
    let (w, h, d) = DIMS;
    let channel = |c: [f32; 3], r: f32, peak: f32| {
        let mut voxels = Vec::with_capacity((w * h * d) as usize);
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    voxels.push(bump((x, y, z), c, r, peak).round() as u8);
                }
            }
        }
        (VolumeKind::U8, voxels)
    };
    let channels = vec![
        channel([14.0, 14.0, 13.0], 10.0, 250.0),
        channel([27.0, 12.0, 14.0], 9.0, 220.0),
        channel([24.0, 23.0, 19.0], 9.0, 240.0),
        channel([12.0, 24.0, 19.0], 8.0, 200.0),
        channel([27.0, 22.0, 12.0], 9.0, 230.0),
        channel([17.0, 18.0, 19.0], 9.0, 210.0),
    ];
    assert_eq!(channels.len(), MAX_CHANNELS);
    channels
}

/// [`volume`] with every channel's samples as `kind`, standing for the same
/// normalized values: `v * 257` as 16-bit, `v / 255` as float. So each kind
/// draws the 8-bit volume's picture with the same windows, and a backend that
/// stores one kind otherwise than it says shows (`caps.rs`).
pub fn volume_as(kind: VolumeKind) -> Vec<(VolumeKind, Vec<u8>)> {
    volume()
        .into_iter()
        .map(|(_, voxels)| {
            let bytes = match kind {
                VolumeKind::U8 => voxels,
                VolumeKind::U16 => voxels
                    .iter()
                    .flat_map(|&v| (v as u16 * 257).to_ne_bytes())
                    .collect(),
                VolumeKind::F32 => voxels
                    .iter()
                    .flat_map(|&v| (v as f32 / 255.0).to_ne_bytes())
                    .collect(),
            };
            (kind, bytes)
        })
        .collect()
}

pub fn volume_luts() -> [Lut; MAX_CHANNELS] {
    [
        lut(|t| [0.6 * t, 0.3 * t, 0.05]),
        lut(|t| [0.0, 0.6 * t, 0.5 * t * t]),
        lut(|t| [0.5 * t * t, 0.1 * t, 0.5 * t]),
        lut(|t| [0.4 * t, 0.4 * t, 0.4 * t]),
        // Brightest partway along, as a contrast-stretched palette is: its
        // isosurface albedo is not its top entry.
        lut(|t| [0.6 * (t * 3.0).sin().abs(), 0.2 * t, 0.0]),
        lut(|t| [0.1 * t, 0.35 * t, 0.6 * t]),
    ]
}

/// Each channel's window, in the units of the sampled texture (raw / 255).
const VOLUME_WINDOWS: [[f32; 2]; MAX_CHANNELS] = [
    [0.05, 0.9],
    [0.1, 0.85],
    [0.0, 0.7],
    [0.15, 0.95],
    [0.08, 0.8],
    [0.2, 0.9],
];

/// Every channel on.
pub const ALL: [bool; MAX_CHANNELS] = [true; MAX_CHANNELS];

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    v.map(|c| c / n)
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// The volume seen from off its axes, `distance` away, in `mode`, with the
/// channels `on` switched on.
pub fn volume_params(mode: VolumeRender, distance: f32, on: [bool; MAX_CHANNELS]) -> VolumeParams {
    let eye = normalize([0.9, -0.6, 1.5]).map(|c| c * distance);
    let forward = normalize(eye.map(|c| -c));
    let right = normalize(cross(forward, [0.0, 1.0, 0.0]));
    let up = cross(right, forward);
    let luts = volume_luts();
    VolumeParams {
        num_channels: MAX_CHANNELS as i32,
        windows: VOLUME_WINDOWS.as_flattened().try_into().unwrap(),
        enabled: on.map(|on| if on { 1.0 } else { 0.0 }),
        is_float: [0.0; MAX_CHANNELS],
        albedo_t: std::array::from_fn(|c| brightest_lut_t(&luts[c])),
        render_mode: mode,
        density: 40.0,
        brightness: 1.25,
        iso: 0.2,
        eye,
        forward,
        right,
        up,
        tan_half_fov: 25f32.to_radians().tan(),
        aspect: 1.0,
        box_he: [0.5, 0.45, 0.4],
        max_steps: 0,
    }
}

/// Upload the volume to `r`, of either backend, and set it to render as
/// `mode` with `interp`, from `distance` away, with the channels `on`.
pub fn show_volume(
    r: &mut impl VolumeRenderer,
    mode: VolumeRender,
    interp: VolumeInterp,
    distance: f32,
    on: [bool; MAX_CHANNELS],
) {
    show_volume_as(r, VolumeKind::U8, mode, interp, distance, on);
}

/// [`show_volume`], with the volume's samples as `kind` ([`volume_as`]).
pub fn show_volume_as(
    r: &mut impl VolumeRenderer,
    kind: VolumeKind,
    mode: VolumeRender,
    interp: VolumeInterp,
    distance: f32,
    on: [bool; MAX_CHANNELS],
) {
    let (w, h, d) = DIMS;
    r.upload_volumes(w, h, d, &volume_as(kind))
        .expect("the volume uploads");
    for (c, lut) in volume_luts().iter().enumerate() {
        r.upload_lut(c, lut);
    }
    r.set_volume_interp(interp);
    r.set_volume_params(volume_params(mode, distance, on));
}
