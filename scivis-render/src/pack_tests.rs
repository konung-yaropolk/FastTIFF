use super::*;

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

/// Every voxel of channel `c` at level `level`, decoded.
fn channel(p: &PackedVolume, c: usize, level: u32) -> Vec<f32> {
    let (t, k) = p.layout.place(c);
    let texel = p.layout.texel_bytes(t);
    let fb = p.layout.format.bytes();
    let (w, h, d) = p.layout.level_dims(level);
    let data = &p.levels[t][level as usize];
    (0..(w * h * d) as usize)
        .map(|i| p.layout.format.decode(&data[i * texel + k * fb..]))
        .collect()
}

#[test]
fn channels_are_packed_four_to_a_texture() {
    let comps = |n: usize| -> Vec<(usize, usize, usize)> {
        VolumeLayout::new((4, 4, 4), n, TexelFormat::Unorm16)
            .textures
            .iter()
            .map(|t| (t.first_channel, t.channels, t.components))
            .collect()
    };
    assert_eq!(comps(1), [(0, 1, 1)]);
    assert_eq!(comps(2), [(0, 2, 2)]);
    assert_eq!(comps(3), [(0, 3, 4)], "no three-component formats");
    assert_eq!(comps(4), [(0, 4, 4)]);
    assert_eq!(comps(5), [(0, 4, 4), (4, 1, 1)]);
    assert_eq!(comps(6), [(0, 4, 4), (4, 2, 2)]);
    assert_eq!(comps(9), comps(6), "clamped to MAX_CHANNELS");
}

#[test]
fn encodings_round_trip() {
    let mut buf = [0u8; 4];
    for f in [
        TexelFormat::Unorm8,
        TexelFormat::Unorm16,
        TexelFormat::Float16,
        TexelFormat::Float32,
    ] {
        // Values each format holds exactly.
        let exact: &[f32] = match f {
            TexelFormat::Unorm8 => &[0.0, 51.0 / 255.0, 1.0],
            TexelFormat::Unorm16 => &[0.0, 12345.0 / 65535.0, 1.0],
            _ => &[0.0, 0.25, 0.5, 1.0],
        };
        for &v in exact {
            f.encode(v, &mut buf);
            assert_eq!(f.decode(&buf), v, "{f:?} {v}");
        }
    }
    // Normalized formats clamp; NaN stores as zero.
    TexelFormat::Unorm16.encode(2.0, &mut buf);
    assert_eq!(TexelFormat::Unorm16.decode(&buf), 1.0);
    TexelFormat::Unorm8.encode(f32::NAN, &mut buf);
    assert_eq!(TexelFormat::Unorm8.decode(&buf), 0.0);
    // Floats keep their range.
    TexelFormat::Float32.encode(-1234.5, &mut buf);
    assert_eq!(TexelFormat::Float32.decode(&buf), -1234.5);
}

/// Every 16-bit sample survives the trip through normalized units exactly.
#[test]
fn sixteen_bit_samples_are_stored_exactly() {
    let samples: Vec<u16> = (0..=65535u32).step_by(7).map(|v| v as u16).collect();
    let n = samples.len() as u32;
    let bytes = u16_bytes(&samples);
    let p = PackedVolume::from_channels(
        (n, 1, 1),
        &[(VolumeKind::U16, &bytes)],
        TexelFormat::Unorm16,
    );
    let level0 = &p.levels[0][0];
    let back: Vec<u16> = level0
        .chunks_exact(2)
        .map(|c| u16::from_ne_bytes([c[0], c[1]]))
        .collect();
    assert_eq!(back, samples);
}

#[test]
fn two_channels_interleave_in_one_texture() {
    let a = u16_bytes(&[0, 1000, 2000, 3000]);
    let b = u16_bytes(&[65535, 4, 5, 6]);
    let p = PackedVolume::from_channels(
        (2, 2, 1),
        &[(VolumeKind::U16, &a), (VolumeKind::U16, &b)],
        TexelFormat::Unorm16,
    );
    assert_eq!(p.levels.len(), 1);
    let raw: Vec<u16> = p.levels[0][0]
        .chunks_exact(2)
        .map(|c| u16::from_ne_bytes([c[0], c[1]]))
        .collect();
    assert_eq!(raw, [0, 65535, 1000, 4, 2000, 5, 3000, 6]);
}

#[test]
fn levels_halve_down_to_one_and_stop_at_the_cap() {
    let l = VolumeLayout::new((1366, 1366, 69), 2, TexelFormat::Unorm16);
    assert_eq!(l.levels, MAX_LEVELS);
    assert_eq!(l.level_dims(1), (683, 683, 34));
    assert_eq!(l.level_dims(3), (170, 170, 8));
    assert_eq!(
        VolumeLayout::new((1, 1, 1), 1, TexelFormat::Unorm8).levels,
        1
    );
    assert_eq!(
        VolumeLayout::new((2, 1, 1), 1, TexelFormat::Unorm8).levels,
        2
    );
    assert_eq!(
        VolumeLayout::new((5, 3, 1), 1, TexelFormat::Unorm8).levels,
        3
    );
    // The budget's per-voxel figure covers what the layout allocates.
    let per = bytes_per_voxel(2, TexelFormat::Unorm16);
    let voxels = 1366.0 * 1366.0 * 69.0;
    assert!(l.total_bytes() as f64 <= per * voxels);
    assert!(l.total_bytes() as f64 > 4.0 * voxels);
}

/// A level keeps the brightest texel it covers, including the odd one left
/// over on an odd-sized axis, all the way down.
#[test]
fn every_level_keeps_the_brightest_voxel() {
    let (w, h, d) = (7u32, 5u32, 3u32);
    for spot in [(0, 0, 0), (6, 4, 2), (6, 0, 1), (3, 2, 0)] {
        let mut v = vec![0.1f32; (w * h * d) as usize];
        v[((spot.2 * h + spot.1) * w + spot.0) as usize] = 0.9;
        let bytes = f32_bytes(&v);
        let p = PackedVolume::from_channels(
            (w, h, d),
            &[(VolumeKind::F32, &bytes)],
            TexelFormat::Float32,
        );
        for level in 0..p.layout.levels {
            let c = channel(&p, 0, level);
            let top = c.iter().cloned().fold(f32::MIN, f32::max);
            assert_eq!(top, 0.9, "spot {spot:?}, level {level}");
            assert!(c.iter().all(|&x| x == 0.1 || x == 0.9));
        }
    }
}

#[test]
fn bricks_hold_each_channels_maximum() {
    // 40 x 20 x 3: bricks of 16 make a 3 x 2 x 1 grid.
    let (w, h, d) = (40u32, 20u32, 3u32);
    let n = (w * h * d) as usize;
    let mut a = vec![0u16; n];
    let mut b = vec![100u16; n];
    // Channel 0: one bright voxel in brick (2, 1, 0).
    a[((2 * h + 18) * w + 35) as usize] = 50_000;
    // Channel 1: one in brick (0, 0, 0).
    b[((h + 3) * w + 2) as usize] = 9_000;
    let p = PackedVolume::from_channels(
        (w, h, d),
        &[
            (VolumeKind::U16, &u16_bytes(&a)),
            (VolumeKind::U16, &u16_bytes(&b)),
        ],
        TexelFormat::Unorm16,
    );
    assert_eq!(p.bricks.dims, (3, 2, 1));
    let unit = |v: u16| v as f32 / 65535.0;
    assert_eq!(p.bricks.max[0][3 + 2], unit(50_000));
    assert_eq!(p.bricks.max[0][0], 0.0);
    assert_eq!(p.bricks.max[1][0], unit(9_000));
    assert_eq!(p.bricks.max[1][1], unit(100));
}

#[test]
fn an_all_nan_brick_is_never_the_maximum() {
    let v = vec![f32::NAN; 8];
    let p = PackedVolume::from_channels(
        (2, 2, 2),
        &[(VolumeKind::F32, &f32_bytes(&v))],
        TexelFormat::Float32,
    );
    assert_eq!(p.bricks.max[0][0], f32::NEG_INFINITY);
}

/// A coarse level over nothing but NaN is as dark as the NaN below it, under
/// any window: 0.0 would show as a fifth of the way up a window from -0.5.
/// And finite, so filtering it with a neighbour cannot make a NaN.
#[test]
fn a_level_over_nothing_but_nan_holds_the_lowest_value() {
    let v = vec![f32::NAN; 8];
    for (format, lowest) in [
        (TexelFormat::Float16, -65504.0),
        (TexelFormat::Float32, f32::MIN),
    ] {
        let p =
            PackedVolume::from_channels((2, 2, 2), &[(VolumeKind::F32, &f32_bytes(&v))], format);
        assert_eq!(channel(&p, 0, 1), vec![lowest], "{format:?}");
        assert_eq!(format.lowest(), lowest);
    }
    // Integer data has no NaN, and its floor is zero as before.
    assert_eq!(TexelFormat::Unorm16.lowest(), 0.0);
}

#[test]
fn only_a_finished_volume_of_the_sizes_its_layout_says_passes() {
    let layout = VolumeLayout::new((20, 9, 5), 5, TexelFormat::Unorm16);
    let mut p = PackedVolume::new(layout);
    assert!(p.check().is_err(), "no mip chain and no bricks yet");
    p.finish();
    assert_eq!(p.check(), Ok(()));

    let mut short = p.clone();
    short.levels[1][2].pop();
    assert!(short.check().is_err(), "a level a byte short");

    let mut bricks = p.clone();
    bricks.bricks.max.pop();
    assert!(bricks.check().is_err(), "a channel without brick maxima");

    let mut levels = p.clone();
    levels.layout.levels += 1;
    assert!(levels.check().is_err(), "more levels than the size has");

    // Three components with buffers to match: the GPU would read four.
    let mut three = VolumeLayout::new((8, 8, 8), 3, TexelFormat::Unorm8);
    three.textures[0].components = 3;
    let mut odd = PackedVolume::new(three);
    odd.finish();
    assert!(
        odd.check().is_err(),
        "a packing VolumeLayout::new does not make"
    );
}

fn vis(render: VolumeRender, windows: &[(f32, f32)], enabled: &[bool]) -> Visibility {
    let mut w = [0.0; MAX_CHANNELS * 2];
    let mut e = [false; MAX_CHANNELS];
    for (c, &(lo, hi)) in windows.iter().enumerate() {
        w[2 * c] = lo;
        w[2 * c + 1] = hi;
        e[c] = enabled[c];
    }
    Visibility {
        num_channels: windows.len(),
        windows: w,
        enabled: e,
        render,
        iso: 0.5,
    }
}

/// A 5 x 1 x 1 brick grid with one value per brick, for the occupancy tests.
fn row(values: &[f32]) -> BrickMaxima {
    BrickMaxima {
        dims: (values.len() as u32, 1, 1),
        max: vec![values.to_vec()],
    }
}

#[test]
fn a_brick_below_the_window_is_skipped_and_its_neighbours_kept() {
    // Only brick 2 rises above the window's bottom (0.3).
    let b = row(&[0.1, 0.2, 0.31, 0.3, 0.0]);
    let occ = occupancy(&b, &vis(VolumeRender::Mip, &[(0.3, 0.8)], &[true]));
    // Brick 2, grown by one each side.
    assert_eq!(occ, [0, 255, 255, 255, 0]);
    // Alpha compositing adds nothing for a windowed zero either.
    let occ = occupancy(&b, &vis(VolumeRender::Alpha, &[(0.3, 0.8)], &[true]));
    assert_eq!(occ, [0, 255, 255, 255, 0]);
}

#[test]
fn the_isosurface_skips_what_cannot_reach_its_threshold() {
    // Window 0.2..0.6, threshold halfway: windowed 0.5 at raw 0.4.
    let b = row(&[0.39, 0.1, 0.1, 0.4, 0.1]);
    let occ = occupancy(&b, &vis(VolumeRender::Surface, &[(0.2, 0.6)], &[true]));
    assert_eq!(occ, [0, 0, 255, 255, 255]);
    // A threshold of zero is met everywhere.
    let mut v = vis(VolumeRender::Surface, &[(0.2, 0.6)], &[true]);
    v.iso = 0.0;
    assert_eq!(occupancy(&b, &v), [255; 5]);
}

#[test]
fn a_channel_that_is_off_shows_nothing() {
    let b = BrickMaxima {
        dims: (3, 1, 1),
        max: vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.9]],
    };
    let both = vis(VolumeRender::Mip, &[(0.1, 1.0), (0.1, 1.0)], &[true, true]);
    assert_eq!(occupancy(&b, &both), [0, 255, 255]);
    let first_only = vis(VolumeRender::Mip, &[(0.1, 1.0), (0.1, 1.0)], &[true, false]);
    assert_eq!(occupancy(&b, &first_only), [0, 0, 0]);
}

#[test]
fn growth_reaches_every_neighbour_in_three_dimensions() {
    let mut max = vec![0.0f32; 27];
    max[13] = 1.0; // the centre of a 3 x 3 x 3 grid
    let b = BrickMaxima {
        dims: (3, 3, 3),
        max: vec![max],
    };
    let occ = occupancy(&b, &vis(VolumeRender::Mip, &[(0.5, 1.0)], &[true]));
    assert!(occ.iter().all(|&o| o == 255));
}
