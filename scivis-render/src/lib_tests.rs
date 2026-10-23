use super::*;

fn params(aspect: f32) -> VolumeParams {
    VolumeParams {
        num_channels: 1,
        windows: [0.0; MAX_CHANNELS * 2],
        enabled: [0.0; MAX_CHANNELS],
        is_float: [0.0; MAX_CHANNELS],
        albedo_t: [1.0; MAX_CHANNELS],
        render_mode: 0,
        density: 1.0,
        brightness: 1.0,
        iso: 0.5,
        eye: [0.1, -0.2, 2.0],
        // A camera turned off the axes, so a mistake in any component shows.
        forward: [0.267_261_24, 0.534_522_5, -0.801_783_7],
        right: [0.948_683_3, 0.0, 0.316_227_76],
        up: [-0.169_030_85, 0.845_154_3, 0.507_092_6],
        tan_half_fov: 0.414_213_57,
        aspect,
        box_he: [0.5, 0.4, 0.2],
    }
}

/// The ray the shaders cast for the pixel `(px, py)` of a `w` x `h` target.
fn ray(p: &VolumeParams, (w, h): (u32, u32), (px, py): (u32, u32)) -> [f32; 3] {
    let nx = 2.0 * (px as f32 + 0.5) / w as f32 - 1.0;
    let ny = 1.0 - 2.0 * (py as f32 + 0.5) / h as f32;
    let r: [f32; 3] = std::array::from_fn(|i| {
        p.forward[i] + nx * p.aspect * p.tan_half_fov * p.right[i] + ny * p.tan_half_fov * p.up[i]
    });
    let n = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
    [r[0] / n, r[1] / n, r[2] / n]
}

#[test]
fn the_tiles_cover_the_picture_once() {
    let (w, h) = (2500, 1100);
    let tiles = capture_tiles(w, h);
    assert_eq!(tiles.len(), 3 * 2);
    let mut covered = vec![0u8; (w * h) as usize];
    for &(x, y, tw, th) in &tiles {
        assert!(tw <= CAPTURE_TILE && th <= CAPTURE_TILE && tw > 0 && th > 0);
        for yy in y..y + th {
            for xx in x..x + tw {
                covered[(yy * w + xx) as usize] += 1;
            }
        }
    }
    assert!(covered.iter().all(|&c| c == 1));
    assert_eq!(capture_tiles(100, 50), vec![(0, 0, 100, 50)]);
}

/// Every pixel of every tile casts the ray its pixel of the whole picture
/// would: tiled, the picture is the same picture.
#[test]
fn a_tile_sees_exactly_its_part_of_the_picture() {
    let full = (2500, 1100);
    let p = params(full.0 as f32 / full.1 as f32);
    for tile in capture_tiles(full.0, full.1) {
        let (x, y, w, h) = tile;
        let t = tile_params(&p, full, tile);
        for &(u, v) in &[
            (0, 0),
            (w - 1, 0),
            (0, h - 1),
            (w - 1, h - 1),
            (w / 2, h / 3),
        ] {
            let a = ray(&t, (w, h), (u, v));
            let b = ray(&p, full, (x + u, y + v));
            let d = (0..3).map(|i| (a[i] - b[i]).abs()).fold(0.0, f32::max);
            assert!(d < 1e-5, "tile {tile:?} pixel ({u}, {v}): {a:?} vs {b:?}");
        }
    }
}

/// The picture's shape comes from its size, as a capture's does, not from
/// the params the view on screen had.
#[test]
fn the_whole_picture_as_one_tile_is_the_picture_at_its_own_aspect() {
    let p = params(0.5);
    let t = tile_params(&p, (300, 200), (0, 0, 300, 200));
    assert!((t.aspect - 1.5).abs() < 1e-6);
    assert!((t.tan_half_fov - p.tan_half_fov).abs() < 1e-7);
    assert_eq!(t.forward, p.forward);
}
