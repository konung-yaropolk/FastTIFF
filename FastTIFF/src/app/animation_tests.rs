use super::*;
use fasttiff_plugin_api::Pose;

/// The movie size every test frame offers.
const SIZE: (u32, u32) = (4, 3);

fn turn(n: usize) -> Animation {
    Animation::cycle(n, 30.0, |phase| Pose::rotation([0.0, 1.0, 0.0], phase))
}

fn frame<'a>(cam: &'a CameraState, now: f64) -> Frame<'a> {
    Frame {
        now,
        in_volume: true,
        generation: 7,
        camera: cam,
        pressed: false,
        ready: true,
        movie_size: Some(SIZE),
        capture_busy: false,
        shot: None,
    }
}

/// A render of `size` whose pixels are all `rgb`.
fn render_of(size: (u32, u32), rgb: [u8; 3]) -> VolumeImage {
    VolumeImage {
        width: size.0,
        height: size.1,
        rgba: [rgb[0], rgb[1], rgb[2], 255].repeat((size.0 * size.1) as usize),
    }
}

#[test]
fn the_pose_follows_the_clock_and_loops() {
    assert_eq!(pose_index(0.0, 30.0, 10), 0);
    assert_eq!(pose_index(1.0 / 30.0 + 1e-9, 30.0, 10), 1);
    assert_eq!(
        pose_index(0.34, 30.0, 10),
        0,
        "ten frames at 30/s loop after a third of a second"
    );
    assert_eq!(pose_index(-1.0, 30.0, 10), 0);
    assert_eq!(pose_index(f64::NAN, 30.0, 10), 0);
    assert_eq!(pose_index(5.0, 30.0, 0), 0);
}

/// Every way the user can move the view moves something this compares, and the
/// renderer's own per-frame cache does not.
#[test]
fn every_camera_move_is_noticed_and_the_renderers_cache_is_not() {
    let cam = CameraState::default();
    let mark = CameraMark::of(&cam);
    assert!(!mark.moved(&cam));

    let mut cache = cam;
    cache.box_he = [0.1, 0.2, 0.3];
    assert!(!mark.moved(&cache), "the box cache is not the user's");

    let moves: [fn(&mut CameraState); 7] = [
        |c| c.orbit(5.0, 0.0),
        |c| c.orbit(0.0, 5.0),
        |c| c.dolly(10.0),
        |c| c.pan(3.0, 0.0, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.01),
        |c| c.wheel_fly(1.0, 1.0),
        |c| c.nav = NavMode::Maya,
        |c| c.orbit_point = OrbitPoint::ScreenCenter,
    ];
    for (i, m) in moves.iter().enumerate() {
        let mut c = cam;
        m(&mut c);
        assert!(mark.moved(&c), "move {i} went unnoticed");
    }
    let mut reset = cam;
    reset.orbit(20.0, 10.0);
    let moved_mark = CameraMark::of(&reset);
    reset.reset();
    assert!(moved_mark.moved(&reset), "Reset is a move too");
}

#[test]
fn playing_starts_when_the_volume_is_ready_and_keeps_time() {
    let cam = CameraState::default();
    let a = turn(30);
    let poses: Vec<_> = a.poses.iter().map(|p| p.rotation).collect();
    let mut r = Running::new(3, a, &cam, 7, None);
    assert_eq!(r.total(), 30);

    // Still building: pose 0, and the clock has not started.
    let mut f = frame(&cam, 5.0);
    f.ready = false;
    assert!(matches!(r.step(f), Step::Show { pose, capture: None, .. } if pose == poses[0]));

    // Ready at t = 10: that is time zero.
    assert!(matches!(r.step(frame(&cam, 10.0)), Step::Show { pose, .. } if pose == poses[0]));
    // Half a second later, fifteen frames on.
    match r.step(frame(&cam, 10.5 + 1e-6)) {
        Step::Show {
            pose,
            capture,
            repaint_after,
        } => {
            assert_eq!(pose, poses[15]);
            assert_eq!(capture, None, "playing never renders off screen");
            assert!((repaint_after - 1.0 / 30.0).abs() < 1e-9);
        }
        other => panic!("{other:?}"),
    }
    // A second and a bit: round again.
    assert!(
        matches!(r.step(frame(&cam, 11.0 + 1e-6)), Step::Show { pose, .. } if pose == poses[0])
    );
}

#[test]
fn it_stops_when_the_user_takes_the_view_back() {
    let cam = CameraState::default();
    let new = || Running::new(0, turn(4), &cam, 7, None);

    let mut f = frame(&cam, 1.0);
    f.in_volume = false;
    assert_eq!(new().step(f), Step::End(Ended::LeftVolume));

    let mut f = frame(&cam, 1.0);
    f.generation = 8;
    assert_eq!(new().step(f), Step::End(Ended::StackChanged));

    let mut f = frame(&cam, 1.0);
    f.pressed = true;
    assert_eq!(new().step(f), Step::End(Ended::Interacted));

    let mut moved = cam;
    moved.orbit(4.0, 0.0);
    assert_eq!(new().step(frame(&moved, 1.0)), Step::End(Ended::Interacted));
}

/// A recording under way, the poses it will show, and where its frames land.
type Recorder = (
    Running,
    Vec<[[f32; 3]; 3]>,
    std::sync::mpsc::Receiver<Vec<u8>>,
);

/// A recording into memory, so the tests need no files and no threads.
fn recording(n: usize) -> Recorder {
    let cam = CameraState::default();
    let a = turn(n);
    let poses: Vec<_> = a.poses.iter().map(|p| p.rotation).collect();
    let (sink, rx) = FrameSink::for_test();
    let r = Running::new(
        0,
        a,
        &cam,
        7,
        Some((Destination::Download("t.tif".into()), sink)),
    );
    (r, poses, rx)
}

#[test]
fn a_recording_renders_every_pose_exactly_once_in_order() {
    let cam = CameraState::default();
    let (mut r, poses, rx) = recording(3);

    // Warming up: the first pose is shown and nothing is rendered yet.
    for _ in 0..WARMUP_FRAMES {
        assert!(
            matches!(r.step(frame(&cam, 0.0)), Step::Show { pose, capture: None, .. } if pose == poses[0])
        );
    }
    // Then pose 0 is rendered, at the movie's size.
    assert_eq!(
        r.step(frame(&cam, 0.0)),
        Step::Show {
            pose: poses[0],
            capture: Some(SIZE),
            repaint_after: 0.0
        }
    );
    // Nothing back yet: hold the pose, do not ask again.
    let mut f = frame(&cam, 0.0);
    f.capture_busy = true;
    assert!(matches!(r.step(f), Step::Show { pose, capture: None, .. } if pose == poses[0]));
    for (i, colour) in [[10u8, 20, 30], [40, 50, 60], [70, 80, 90]]
        .into_iter()
        .enumerate()
    {
        let mut f = frame(&cam, 0.0);
        f.shot = Some(Ok(render_of(SIZE, colour)));
        let step = r.step(f);
        let got = rx.try_recv().expect("the frame went to the sink");
        assert_eq!(got.len(), 4 * 3 * 3, "the render, as RGB");
        assert!(
            got.chunks(3).all(|p| p == colour),
            "frame {i} is not pose {i}'s render"
        );
        if i < 2 {
            assert_eq!(
                step,
                Step::Show {
                    pose: poses[i + 1],
                    capture: Some(SIZE),
                    repaint_after: 0.0
                }
            );
        } else {
            assert_eq!(step, Step::Recorded);
        }
    }
    assert_eq!(r.recording.as_ref().unwrap().done, 3);
}

/// A render left over from a recording that ended is not a frame of the next.
#[test]
fn a_render_nobody_asked_for_is_dropped() {
    let cam = CameraState::default();
    let (mut r, poses, rx) = recording(2);
    let mut f = frame(&cam, 0.0);
    f.shot = Some(Ok(render_of(SIZE, [9, 9, 9])));
    assert!(matches!(r.step(f), Step::Show { pose, capture: None, .. } if pose == poses[0]));
    assert!(rx.try_recv().is_err(), "the stray render was kept");
}

/// The size is fixed by the first frame: the view resizing afterwards does not
/// change it, since every frame of the movie has to be the same size.
#[test]
fn the_movie_keeps_the_size_it_started_at() {
    let cam = CameraState::default();
    let (mut r, poses, _rx) = recording(3);
    for _ in 0..=WARMUP_FRAMES {
        r.step(frame(&cam, 0.0));
    }
    let mut f = frame(&cam, 0.0);
    f.shot = Some(Ok(render_of(SIZE, [1, 2, 3])));
    f.movie_size = Some((40, 30));
    assert_eq!(
        r.step(f),
        Step::Show {
            pose: poses[1],
            capture: Some(SIZE),
            repaint_after: 0.0
        }
    );
    // And a render of another size is refused rather than written.
    let mut f = frame(&cam, 0.0);
    f.shot = Some(Ok(render_of((5, 3), [1, 2, 3])));
    assert!(matches!(r.step(f), Step::End(Ended::Failed(why)) if why.contains("size")));
}

#[test]
fn a_render_that_failed_ends_the_recording() {
    let cam = CameraState::default();
    let (mut r, _, _rx) = recording(3);
    for _ in 0..=WARMUP_FRAMES {
        r.step(frame(&cam, 0.0));
    }
    let mut f = frame(&cam, 0.0);
    f.shot = Some(Err("out of memory".into()));
    assert_eq!(r.step(f), Step::End(Ended::Failed("out of memory".into())));
}

/// While the GPU or the writer is busy the next pose waits, and is not skipped.
#[test]
fn the_next_pose_waits_for_a_busy_renderer() {
    let cam = CameraState::default();
    let (mut r, poses, _rx) = recording(3);
    for _ in 0..WARMUP_FRAMES {
        r.step(frame(&cam, 0.0));
    }
    let mut f = frame(&cam, 0.0);
    f.capture_busy = true;
    assert!(
        matches!(r.step(f), Step::Show { pose, capture: None, repaint_after } if pose == poses[0] && repaint_after > 0.0)
    );
    assert!(matches!(
        r.step(frame(&cam, 0.0)),
        Step::Show {
            capture: Some(SIZE),
            ..
        }
    ));
}

#[test]
fn the_movie_is_the_views_shape_with_every_stack_pixel_in_view() {
    // Zoomed out: the stack in view needs 1000 rows, the view is 400 x 300 on
    // screen, so the movie is the view's shape at 1000 tall.
    assert_eq!(movie_size((400.0, 300.0), 1000.0, 8192), Some((1333, 1000)));
    // Zoomed in, the stack pixels in view fewer than the screen's: never
    // smaller than the view is on screen.
    assert_eq!(movie_size((400.0, 300.0), 120.0, 8192), Some((400, 300)));
    // Too large for the GPU: scaled down, keeping the shape.
    assert_eq!(
        movie_size((400.0, 300.0), 30_000.0, 4000),
        Some((4000, 3000))
    );
    // No view, no movie.
    assert_eq!(movie_size((0.0, 300.0), 1000.0, 8192), None);
    assert_eq!(movie_size((f32::NAN, 300.0), 1000.0, 8192), None);
    // A detail measure that went wrong leaves the view's own size.
    assert_eq!(
        movie_size((400.0, 300.0), f32::INFINITY, 8192),
        Some((400, 300))
    );
}

#[test]
fn rgb_is_the_picture_without_its_alpha() {
    let img = VolumeImage {
        width: 2,
        height: 1,
        rgba: vec![1, 2, 3, 255, 4, 5, 6, 255],
    };
    assert_eq!(rgb_of(&img), [1, 2, 3, 4, 5, 6]);
}

/// The movie reads back as what was recorded: an RGB stack, one frame per
/// pose, at the rate it was played, marked to loop.
#[test]
fn the_movie_reads_back_as_an_rgb_stack_at_the_animations_rate() {
    let (w, h) = (5usize, 4usize);
    let mut m = MovieWriter::new(std::io::Cursor::new(Vec::new()), 30.0);
    let frames: Vec<Vec<u8>> = (0..3u8)
        .map(|f| {
            (0..w * h * 3)
                .map(|i| (i as u8).wrapping_mul(7).wrapping_add(f * 40))
                .collect()
        })
        .collect();
    for f in &frames {
        m.push(f, (w, h)).expect("frame");
    }
    assert_eq!(m.frames, 3);
    let bytes = m.finish().expect("finish").into_inner();
    let tiff = fast_tiff_lib::TiffStack::from_bytes(bytes).expect("a TIFF");
    assert_eq!(tiff.frames.len(), 3);
    assert_eq!(tiff.meta.fps, Some(30.0));
    assert_eq!(tiff.meta.loop_playback, Some(true));
    for (i, (info, want)) in tiff.frames.iter().zip(&frames).enumerate() {
        assert_eq!((info.width as usize, info.height as usize), (w, h));
        assert_eq!(info.samples_per_pixel, 3);
        for c in 0..3 {
            let got =
                fast_tiff_lib::read_plane_u8(&tiff.data, info, tiff.byte_order, c).expect("decode");
            let expect: Vec<u8> = want.iter().skip(c).step_by(3).copied().collect();
            assert_eq!(got, expect, "frame {i} channel {c}");
        }
    }
}

#[test]
fn a_movie_with_no_frames_is_refused() {
    let m = MovieWriter::new(std::io::Cursor::new(Vec::new()), 30.0);
    assert!(m.finish().is_err());
}

/// On the desktop the frames go to a file as they come, the file is a movie
/// when the recording finishes, and nothing is left behind when it does not.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn a_recording_to_a_file_finishes_or_leaves_nothing() {
    let dir = std::env::temp_dir().join(format!("fasttiff-anim-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("turn.tif");

    let mut sink = FrameSink::to_file(path.clone(), 25.0);
    sink.push(vec![9u8; 2 * 2 * 3], (2, 2)).unwrap();
    sink.push(vec![7u8; 2 * 2 * 3], (2, 2)).unwrap();
    let mut saving = sink.finish(Destination::File(path.clone()));
    let result = loop {
        if let Some(r) = saving.poll() {
            break r;
        }
        std::thread::yield_now();
    };
    assert_eq!(result, Ok(2));
    let tiff = fast_tiff_lib::TiffStack::from_bytes(std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(tiff.frames.len(), 2);
    assert_eq!(tiff.meta.fps, Some(25.0));

    let abandoned = dir.join("abandoned.tif");
    let mut sink = FrameSink::to_file(abandoned.clone(), 25.0);
    sink.push(vec![1u8; 2 * 2 * 3], (2, 2)).unwrap();
    // The writer notices the recording went away and removes what it wrote;
    // `abandon` waits for it to have done so.
    sink.abandon();
    assert!(!abandoned.exists(), "a half movie was left behind");
    let _ = std::fs::remove_dir_all(dir);
}

/// A 4D stack whose movie plays on rebuilds its volume at every timepoint. A
/// build in flight once the animation is under way must not send it back to
/// its first pose, or the turn would judder at the movie's frame rate.
#[test]
fn a_rebuild_in_flight_does_not_send_a_playing_animation_back_to_its_start() {
    let cam = CameraState::default();
    let a = turn(30);
    let poses: Vec<_> = a.poses.iter().map(|p| p.rotation).collect();
    let mut r = Running::new(0, a, &cam, 7, None);
    r.step(frame(&cam, 10.0));
    let mut f = frame(&cam, 10.5 + 1e-6);
    f.ready = false;
    assert!(matches!(r.step(f), Step::Show { pose, .. } if pose == poses[15]));
}
