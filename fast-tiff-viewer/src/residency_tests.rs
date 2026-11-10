//! One test per rule of [`Residency::reconcile`]: what each change to the
//! display model drops of what a renderer holds, and what it leaves.
//!
//! None of this needs a GPU. A sync fills the residency as it uploads; these
//! fill it by hand with everything a sync of a frame past the texture limit
//! would leave there, change the model the way a frontend does, and reconcile
//! as the next sync's first step would.

use super::*;
use crate::prefetch::Decoded;
use crate::stack::Stack;
use crate::window::Built;
use crate::{Viewer, Window};
use fast_tiff_lib::{SampleType, StackMetaWrite, TiffWriter, WriterOptions};

const W: u32 = 32;
const H: u32 = 16;

/// Two channels by three timepoints of 16-bit samples, as written: a stack
/// whose dimensions can be reassigned (2 x 3 becomes 3 x 2).
fn movie_bytes() -> Vec<u8> {
    let opts = WriterOptions::new(W, H, SampleType::U16).metadata(StackMetaWrite::new(2, 1));
    let mut w = TiffWriter::new(std::io::Cursor::new(Vec::new()), opts).unwrap();
    for plane in 0..6u16 {
        w.write_frame_u16(&vec![plane * 1000; (W * H) as usize])
            .unwrap();
    }
    w.finish().unwrap().into_inner()
}

fn movie() -> Stack {
    let stack = Stack::from_bytes(movie_bytes(), "movie.tif".into(), false).unwrap();
    let dims = stack.display.dims();
    assert_eq!((dims.channels, dims.slices, dims.frames), (2, 1, 3));
    stack
}

/// A window of a quarter of the frame at full resolution, as a zoomed-in view
/// of a frame past the texture limit keeps resident.
const WINDOW: Roi = Roi {
    x: 0,
    y: 0,
    w: W / 2,
    h: H / 2,
    stride: 1,
};

/// The whole frame at stride 2: the overview such a view falls back on.
fn overview_of(stack: &Stack) -> Overview {
    let roi = Roi {
        x: 0,
        y: 0,
        w: W,
        h: H,
        stride: 2,
    };
    let (tw, th) = roi.texture_size();
    let built = Built {
        frame_index: stack.frame_index,
        roi,
        channels: vec![0, 1],
        planes: (0..2)
            .map(|_| Decoded::U16(vec![7; (tw * th) as usize]))
            .collect(),
    };
    Overview::capture(built, W, H, stack.display.revision(), usize::MAX).expect("spans the frame")
}

/// Everything a sync of `stack`'s frame leaves resident: the frame's pixels
/// cut to a window, the LUTs, an overview, a finer window asked for, and a
/// volume built and uploaded.
fn filled(stack: &Stack) -> Residency {
    let mut r = Residency::default();
    r.reconcile(&stack.display, stack.frame_index);
    assert!(
        r.volume.take_release(),
        "a fresh residency gives back what the renderer held"
    );
    let revision = stack.display.revision();
    r.uploaded(WINDOW, stack.frame_index, revision);
    r.luts = Some(stack.display.lut_revision());
    r.overview = Some(overview_of(stack));
    let finer = Roi {
        stride: 1,
        ..WINDOW
    };
    r.pending_window = Some((stack.frame_index, finer, revision));
    r.volume.built_frame = Some(0);
    r.volume.uploaded_generation = Some(r.volume.generation);
    r.volume.budget_cap = Some(1 << 30);
    r
}

/// What a residency holds, in a form one assertion can compare.
#[derive(Debug, PartialEq)]
struct Held {
    roi: bool,
    frame: bool,
    luts: bool,
    overview: bool,
    pending_window: bool,
    volume: bool,
}

fn held(r: &Residency, stack: &Stack) -> Held {
    Held {
        roi: r.roi.is_some(),
        frame: r.frame.is_some(),
        luts: r.luts.is_some(),
        overview: r.overview.is_some(),
        pending_window: r.pending_window.is_some(),
        volume: r.volume.built_for(stack.display.revision().layout),
    }
}

const EVERYTHING: Held = Held {
    roi: true,
    frame: true,
    luts: true,
    overview: true,
    pending_window: true,
    volume: true,
};

/// Reconciling against the model it was filled under drops nothing, however
/// often: a sync's first step costs nothing when nothing changed.
#[test]
fn an_unchanged_model_keeps_everything() {
    let stack = movie();
    let mut r = filled(&stack);
    for _ in 0..3 {
        r.reconcile(&stack.display, stack.frame_index);
        assert_eq!(held(&r, &stack), EVERYTHING);
    }
    assert!(!r.volume.take_release(), "nothing is given back twice");
}

/// A channel switched on or off drops what is made of the channels that were
/// on — the frame's pixels, the overview, the window asked for — and nothing
/// that covers every channel regardless: the LUTs, the window the textures are
/// cut to, the volume.
#[test]
fn a_channel_toggle_drops_what_depends_on_the_enabled_channels_only() {
    let mut stack = movie();
    let mut r = filled(&stack);
    let generation = r.volume.generation;
    assert!(stack.display.set_enabled(1, false));
    r.reconcile(&stack.display, stack.frame_index);
    assert_eq!(
        held(&r, &stack),
        Held {
            frame: false,
            overview: false,
            pending_window: false,
            ..EVERYTHING
        }
    );
    assert_eq!(r.volume.generation, generation, "the volume is not rebuilt");
    assert!(!r.volume.take_release());
}

/// Reassigning the axes moves which IFD every display channel reads, so
/// everything uploaded is the wrong planes: all of it goes, the volume's GPU
/// memory is given back, and it is built again. The volume budget learned
/// from running out of memory is this renderer's, and stays.
///
/// The bands the document decoded under the old mapping go too, at the next
/// build: every build first makes the cache that of the layout in effect.
#[test]
fn a_dimension_override_drops_everything_uploaded() {
    let mut stack = movie();
    let before = stack.display.revision();
    stack.services.bands.hold_layout(before.layout);
    stack
        .services
        .bands
        .put(0, vec![0], 0, (0, W), vec![Decoded::U16(vec![7; 64])]);
    let mut r = filled(&stack);
    let generation = r.volume.generation;

    crate::dimensions::apply_dimension_override(&mut stack, 3, 1, 2);
    let after = stack.display.revision();
    assert_ne!(after.layout, before.layout);
    r.reconcile(&stack.display, stack.frame_index);

    assert_eq!(
        held(&r, &stack),
        Held {
            roi: false,
            frame: false,
            luts: false,
            overview: false,
            pending_window: false,
            volume: false,
        }
    );
    assert_ne!(
        r.volume.generation, generation,
        "a build in flight is stale"
    );
    assert_eq!(r.volume.requested, None);
    assert!(r.volume.take_release(), "the old volume's memory goes back");
    assert_eq!(r.volume.budget_cap, Some(1 << 30));

    stack.services.bands.hold_layout(after.layout);
    assert!(
        stack.services.bands.is_empty(),
        "nothing decoded under the old mapping may survive"
    );
}

/// A LUT picked marks the renderer's copy of the LUTs, and only that: nothing
/// decoded depends on a LUT. Picking the LUT already shown changes nothing.
#[test]
fn a_lut_change_marks_only_the_luts() {
    let mut stack = movie();
    let mut r = filled(&stack);
    let same = *stack.display.lut(0).unwrap();
    assert!(!stack.display.set_lut(0, same));
    r.reconcile(&stack.display, stack.frame_index);
    assert_eq!(held(&r, &stack), EVERYTHING);

    let revision = stack.display.revision();
    assert!(stack
        .display
        .set_lut(0, fast_tiff_lib::default_composite_lut(2)));
    assert_eq!(
        stack.display.revision(),
        revision,
        "no decode revision moves"
    );
    r.reconcile(&stack.display, stack.frame_index);
    assert_eq!(
        held(&r, &stack),
        Held {
            luts: false,
            ..EVERYTHING
        }
    );
}

/// The pseudocolour toggle and the single-channel LUT selector reach the
/// renderer the same way, through the LUT revision.
#[test]
fn every_lut_control_moves_the_lut_revision() {
    let mut stack = movie();
    let before = stack.display.lut_revision();
    crate::channels::refresh_pseudocolor(&mut stack, true);
    let tinted = stack.display.lut_revision();
    assert_ne!(tinted, before, "pseudocolour on");
    crate::channels::refresh_pseudocolor(&mut stack, true);
    assert_eq!(
        stack.display.lut_revision(),
        tinted,
        "and on again is no change"
    );

    let mut viewer = Viewer::new();
    viewer
        .load_bytes(movie_bytes(), "movie.tif".into())
        .unwrap();
    viewer.set_dimension_order(1, 1, 6);
    let stack = viewer.stack.as_ref().unwrap();
    assert!(crate::channels::gray_lut_applicable(stack));
    let before = stack.display.lut_revision();
    viewer.set_gray_lut(2);
    let after = viewer.stack.as_ref().unwrap().display.lut_revision();
    assert_ne!(after, before, "a colormap picked");
}

/// A contrast window goes to the shader with every frame: nothing resident
/// depends on it.
#[test]
fn a_contrast_window_drops_nothing() {
    let mut stack = movie();
    let mut r = filled(&stack);
    assert!(stack.display.set_window(0, Window::new(10.0, 20.0)));
    r.reconcile(&stack.display, stack.frame_index);
    assert_eq!(held(&r, &stack), EVERYTHING);
}

/// Another frame on screen drops the frame's pixels and what was of that
/// frame — the overview, the window asked for — and keeps the shape of what is
/// resident, which is what the next window is planned from. The volume is
/// left to the 3D view, which keeps the last timepoint's on screen until the
/// next lands.
#[test]
fn a_frame_change_drops_the_frame_on_the_gpu() {
    let mut stack = movie();
    let mut r = filled(&stack);
    stack.frame_index = 1;
    r.reconcile(&stack.display, stack.frame_index);
    assert_eq!(
        held(&r, &stack),
        Held {
            frame: false,
            overview: false,
            pending_window: false,
            ..EVERYTHING
        }
    );
    assert_eq!(r.roi, Some(WINDOW));

    // Back to the frame that was uploaded: what went does not come back, since
    // the textures have been written since.
    stack.frame_index = 0;
    r.reconcile(&stack.display, stack.frame_index);
    assert!(r.frame.is_none());
}

/// A smaller rebuild after the GPU ran out of memory is the same volume: a
/// build in flight is dropped, but what a frontend compares to tell that a
/// different stack is on show (an animation stops for that) does not move. A
/// dimension override does move it, and the volume stops counting as built
/// the moment the axes are reassigned, before any sync.
#[test]
fn a_rebuild_is_the_same_volume_and_an_override_another() {
    let mut viewer = Viewer::new();
    viewer
        .load_bytes(movie_bytes(), "movie.tif".into())
        .unwrap();
    viewer.residency = filled(viewer.stack.as_ref().unwrap());
    viewer.residency.volume.requested = Some((viewer.residency.volume.generation, 0));
    assert!(viewer.volume_built());
    assert!(viewer.volume_building());
    let identity = viewer.volume_identity();

    let generation = viewer.residency.volume.generation;
    viewer.residency.volume.rebuild();
    assert_ne!(viewer.residency.volume.generation, generation);
    assert!(!viewer.volume_built() && !viewer.volume_building());
    assert_eq!(viewer.volume_identity(), identity, "the same volume");

    viewer.residency = filled(viewer.stack.as_ref().unwrap());
    assert!(viewer.volume_built());
    viewer.set_dimension_order(3, 1, 2);
    assert_ne!(viewer.volume_identity(), identity, "another volume");
    assert!(!viewer.volume_built(), "not of the stack as it now stands");
}

/// A new document starts from a fresh residency: nothing the renderer holds is
/// of it, the last document's volume is given back at the first sync, and the
/// volume budget starts over. Even the same file opened again: its revisions
/// are new, so a residency kept by mistake would match nothing either.
#[test]
fn a_new_document_starts_a_fresh_residency() {
    let mut viewer = Viewer::new();
    viewer
        .load_bytes(movie_bytes(), "movie.tif".into())
        .unwrap();
    let first = viewer.stack.as_ref().unwrap().display.revision();
    viewer.residency = filled(viewer.stack.as_ref().unwrap());

    viewer
        .load_bytes(movie_bytes(), "movie.tif".into())
        .unwrap();
    let stack = viewer.stack.as_ref().unwrap();
    assert_ne!(stack.display.revision(), first);
    let r = &mut viewer.residency;
    assert_eq!(
        (r.roi, r.frame, r.luts, r.pending_window),
        (None, None, None, None)
    );
    assert!(r.overview.is_none());
    assert_eq!(
        (
            r.volume.built_frame,
            r.volume.uploaded_generation,
            r.volume.budget_cap
        ),
        (None, None, None)
    );
    assert!(
        r.volume.take_release(),
        "the last document's volume goes back"
    );

    // And one kept by mistake would drop everything at its first reconcile.
    let mut kept = filled(viewer.stack.as_ref().unwrap());
    let old = movie();
    kept.reconcile(&old.display, old.frame_index);
    assert_eq!(
        held(&kept, &old),
        Held {
            roi: false,
            frame: false,
            luts: false,
            overview: false,
            pending_window: false,
            volume: false,
        }
    );
}

/// What a sync is asked to send the renderer before anything else, and
/// whether it is to fill the frame's textures again, for each change to the
/// model. The sync acts on nothing else of the reconcile, so this is what
/// stands between a change and a stale texture, answered without a renderer.
#[test]
fn a_sync_is_asked_for_what_each_change_left_stale() {
    let nothing = Pending {
        release_volume: false,
        upload_luts: false,
    };
    let everything = Pending {
        release_volume: true,
        upload_luts: true,
    };
    let mut stack = movie();
    let sync = |r: &mut Residency, s: &Stack| r.begin_sync(&s.display, s.frame_index);

    // A fresh residency: all of it, and the volume it knows nothing of given
    // back, once. The LUTs are asked for until they are recorded as sent.
    let mut r = Residency::default();
    assert_eq!(sync(&mut r, &stack), everything);
    assert!(r.needs_upload(&WINDOW));
    assert_eq!(
        sync(&mut r, &stack),
        Pending {
            upload_luts: true,
            ..nothing
        }
    );
    r.luts_uploaded(&stack.display);
    assert_eq!(sync(&mut r, &stack), nothing);

    // Unchanged: nothing, and the textures hold the window on screen; a view
    // leaving it re-cuts them.
    let mut r = filled(&stack);
    assert_eq!(sync(&mut r, &stack), nothing);
    assert!(!r.needs_upload(&WINDOW));
    assert!(r.needs_upload(&Roi { x: W / 2, ..WINDOW }));

    // A contrast window: nothing, it goes to the shader with every frame.
    assert!(stack.display.set_window(0, Window::new(10.0, 20.0)));
    assert_eq!(sync(&mut r, &stack), nothing);
    assert!(!r.needs_upload(&WINDOW));

    // A channel switched on: the frame, which has nothing in that texture yet.
    assert!(stack.display.set_enabled(1, false));
    let mut r = filled(&stack);
    assert!(stack.display.set_enabled(1, true));
    assert_eq!(sync(&mut r, &stack), nothing);
    assert!(r.needs_upload(&WINDOW));

    // Another frame: the frame.
    let mut r = filled(&stack);
    stack.frame_index = 1;
    assert_eq!(sync(&mut r, &stack), nothing);
    assert!(r.needs_upload(&WINDOW));

    // A LUT picked: the LUTs, and not the frame.
    let mut r = filled(&stack);
    assert!(stack
        .display
        .set_lut(0, fast_tiff_lib::default_composite_lut(2)));
    assert_eq!(
        sync(&mut r, &stack),
        Pending {
            upload_luts: true,
            ..nothing
        }
    );
    assert!(!r.needs_upload(&WINDOW));

    // A dimension override: everything, the volume's memory given back once.
    let mut r = filled(&stack);
    crate::dimensions::apply_dimension_override(&mut stack, 3, 1, 2);
    assert_eq!(sync(&mut r, &stack), everything);
    assert!(r.needs_upload(&WINDOW));
    assert!(!sync(&mut r, &stack).release_volume);
}

/// The display model needs no worker: a document from bytes runs no thread
/// until a view asks for one, and can be reinterpreted, toggled and recoloured
/// as it stands.
#[test]
fn a_document_runs_no_worker_until_a_view_needs_one() {
    let mut stack = movie();
    let idle = |s: &Stack| {
        s.services.prefetch.is_none()
            && s.services.window_worker.is_none()
            && s.services.volume_builder.is_none()
    };
    assert!(idle(&stack));
    crate::dimensions::apply_dimension_override(&mut stack, 3, 1, 2);
    stack.display.set_enabled(0, false);
    crate::channels::refresh_pseudocolor(&mut stack, true);
    assert!(idle(&stack));
    assert_eq!(stack.display.channel_count(), 3);
}
