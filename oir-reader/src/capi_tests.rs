//! The C interface against the Rust one it wraps: the same answers from the
//! same files, and every refusal a status with a reason rather than a crash.

use super::*;
use crate::fixture::{self, Scratch};
use crate::{acquisition_parts, read, read_bytes};
use std::ffi::CStr;
use std::ptr::{null, null_mut};

fn last_error() -> String {
    unsafe { CStr::from_ptr(oir_last_error()) }
        .to_string_lossy()
        .into_owned()
}

fn c(s: &str) -> CString {
    CString::new(s).unwrap()
}

/// A text call, made the way the header says to: once for the length, once
/// with exactly the room it needs — checking the convention along the way.
fn text(call: impl Fn(*mut c_char, usize) -> isize) -> String {
    let n = call(null_mut(), 0);
    assert!(n >= 0, "status {}: {}", -n, last_error());
    let n = n as usize;
    let mut buf = vec![0xAAu8; n + 2];
    assert_eq!(call(buf.as_mut_ptr().cast(), n + 1), n as isize);
    assert_eq!(buf[n], 0, "NUL-terminated");
    assert_eq!(buf[n + 1], 0xAA, "nothing written past cap");
    String::from_utf8(buf[..n].to_vec()).unwrap()
}

/// Read through `oir_read`, insisting it works.
fn read_c(path: &std::path::Path, options: Option<&OirOptions>) -> *mut OirAcquisition {
    let p = c(path.to_str().unwrap());
    let mut a = null_mut();
    let o = options.map_or(null(), |o| o as *const _);
    let status = unsafe { oir_read(p.as_ptr(), o, None, null_mut(), &mut a) };
    assert_eq!(status, OIR_OK, "{}", last_error());
    assert!(!a.is_null());
    a
}

/// Read through `oir_read_bytes`, returning the status and the handle.
fn read_bytes_c(parts: &[&[u8]]) -> (c_int, *mut OirAcquisition) {
    let ptrs: Vec<*const u8> = parts.iter().map(|p| p.as_ptr()).collect();
    let lens: Vec<usize> = parts.iter().map(|p| p.len()).collect();
    let mut a = std::ptr::dangling_mut::<OirAcquisition>();
    let status = unsafe {
        oir_read_bytes(
            ptrs.as_ptr(),
            lens.as_ptr(),
            parts.len(),
            None,
            null_mut(),
            &mut a,
        )
    };
    if status != OIR_OK {
        assert!(a.is_null(), "a failed read hands back nothing");
        assert!(!last_error().is_empty());
    }
    (status, a)
}

fn shape_c(a: *const OirAcquisition) -> OirShape {
    let mut s = std::mem::MaybeUninit::<OirShape>::uninit();
    assert_eq!(unsafe { oir_shape(a, s.as_mut_ptr()) }, OIR_OK);
    unsafe { s.assume_init() }
}

fn record_c(a: *const OirAcquisition) -> OirRecordInfo {
    let mut r = std::mem::MaybeUninit::<OirRecordInfo>::uninit();
    assert_eq!(unsafe { oir_record_info(a, r.as_mut_ptr()) }, OIR_OK);
    unsafe { r.assume_init() }
}

fn part_c(a: *const OirAcquisition, i: usize) -> OirPartInfo {
    let mut p = std::mem::MaybeUninit::<OirPartInfo>::uninit();
    assert_eq!(
        unsafe { oir_part_info(a, i, p.as_mut_ptr()) },
        OIR_OK,
        "{}",
        last_error()
    );
    unsafe { p.assume_init() }
}

/// NaN-aware: "not stated" on both sides is agreement.
fn same(c: f64, rust: Option<f64>) -> bool {
    match rust {
        Some(v) => c == v,
        None => c.is_nan(),
    }
}

/// Every plane, the shape, and plane_index, against the Rust acquisition.
fn assert_planes_match(a: *const OirAcquisition, want: &Acquisition) {
    let s = shape_c(a);
    let w = want.shape();
    assert_eq!(
        (s.width, s.height, s.channels, s.slices, s.frames),
        (w.width, w.height, w.channels, w.slices, w.frames)
    );
    assert_eq!(s.bytes_per_sample, w.sample_type.bytes());
    assert_eq!(s.plane_len, w.plane_len());
    assert_eq!(s.plane_count, w.plane_count());
    assert_eq!(s.plane_count, want.planes().len());
    let bytes = s.plane_len * s.bytes_per_sample;
    for i in 0..s.plane_count {
        let mut copy = vec![0u8; bytes];
        assert_eq!(
            unsafe { oir_plane_copy(a, i, copy.as_mut_ptr().cast(), bytes) },
            OIR_OK
        );
        let mut data = null();
        assert_eq!(unsafe { oir_plane_data(a, i, &mut data) }, OIR_OK);
        match want.planes() {
            Planes::U8(p) => {
                assert_eq!(s.sample_type, OIR_SAMPLE_U8);
                assert_eq!(copy, p[i]);
                let view = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), s.plane_len) };
                assert_eq!(view, p[i].as_slice());
            }
            Planes::U16(p) => {
                assert_eq!(s.sample_type, OIR_SAMPLE_U16);
                let samples: Vec<u16> = copy
                    .chunks_exact(2)
                    .map(|b| u16::from_ne_bytes([b[0], b[1]]))
                    .collect();
                assert_eq!(samples, p[i]);
                let view = unsafe { std::slice::from_raw_parts(data.cast::<u16>(), s.plane_len) };
                assert_eq!(view, p[i].as_slice());
            }
        }
    }
    for t in 0..s.frames {
        for z in 0..s.slices {
            for ch in 0..s.channels {
                let mut i = usize::MAX;
                assert_eq!(unsafe { oir_plane_index(a, ch, z, t, &mut i) }, OIR_OK);
                assert_eq!(Some(i), w.plane_index(ch, z, t));
            }
        }
    }
    assert_eq!(unsafe { oir_dropped_planes(a) }, want.dropped_planes());
}

/// The parts, against the Rust acquisition's.
fn assert_parts_match(a: *const OirAcquisition, want: &Acquisition) {
    assert_eq!(unsafe { oir_part_count(a) }, want.parts().len());
    for (i, p) in want.parts().iter().enumerate() {
        let info = part_c(a, i);
        assert_eq!(info.size, p.size());
        let read_path = match p.read_path() {
            ReadPath::Mapped => OIR_READ_MAPPED,
            ReadPath::Buffered => OIR_READ_BUFFERED,
            ReadPath::Memory => OIR_READ_MEMORY,
        };
        assert_eq!(info.read_path, read_path);
        let block_source = match p.block_source() {
            BlockSource::Index => OIR_BLOCKS_INDEX,
            BlockSource::Walk => OIR_BLOCKS_WALK,
        };
        assert_eq!(info.block_source, block_source);
        assert_eq!(info.block_count, p.block_count());
        assert_eq!(info.planes_added, p.planes_added());
        let path = text(|b, n| unsafe { oir_part_path(a, i, b, n) });
        assert_eq!(
            path,
            p.path()
                .map_or(String::new(), |p| p.to_string_lossy().into_owned())
        );
        let map_error = text(|b, n| unsafe { oir_part_map_error(a, i, b, n) });
        assert_eq!(map_error, p.map_error().unwrap_or(""));
    }
}

/// A split acquisition read by path through C is the one Rust reads: every
/// part, every plane, and the listing of the files.
#[test]
fn a_split_acquisition_read_through_c_matches_the_rust_one() {
    let dir = Scratch::new("capi-split");
    let first = dir.write("rec.oir", &fixture::part_file(0, 2, true));
    dir.write("rec_00001", &fixture::part_file(2, 2, false));
    dir.write("rec_00002.oir", &fixture::part_file(4, 1, false));
    let want = read(&first).expect("read");

    let a = read_c(&first, None);
    assert_planes_match(a, &want);
    assert_parts_match(a, &want);
    assert_eq!(unsafe { oir_part_count(a) }, 3);
    let want_path = if crate::MMAP_SUPPORTED {
        OIR_READ_MAPPED
    } else {
        OIR_READ_BUFFERED
    };
    assert_eq!(part_c(a, 1).read_path, want_path);
    unsafe { oir_acquisition_free(a) };

    // Buffered, as asked.
    let mut o = OirOptions { access: -1 };
    assert_eq!(unsafe { oir_options_default(&mut o) }, OIR_OK);
    assert_eq!(o.access, OIR_ACCESS_AUTO);
    o.access = OIR_ACCESS_BUFFERED;
    let a = read_c(&first, Some(&o));
    assert_planes_match(a, &want);
    for i in 0..3 {
        assert_eq!(part_c(a, i).read_path, OIR_READ_BUFFERED);
    }
    unsafe { oir_acquisition_free(a) };

    // The listing, without reading.
    let p = c(first.to_str().unwrap());
    let listed = acquisition_parts(&first);
    let mut count = 0;
    assert_eq!(
        unsafe { oir_acquisition_part_count(p.as_ptr(), &mut count) },
        OIR_OK
    );
    assert_eq!(count, listed.len());
    for (i, want) in listed.iter().enumerate() {
        let got = text(|b, n| unsafe { oir_acquisition_part(p.as_ptr(), i, b, n) });
        assert_eq!(got, want.to_str().unwrap());
    }
    assert_eq!(
        unsafe { oir_acquisition_part(p.as_ptr(), listed.len(), null_mut(), 0) },
        -(OIR_INVALID_ARGUMENT as isize)
    );
    assert!(!last_error().is_empty());

    // The same parts from memory.
    let bytes: Vec<Vec<u8>> = listed.iter().map(|p| std::fs::read(p).unwrap()).collect();
    let refs: Vec<&[u8]> = bytes.iter().map(Vec::as_slice).collect();
    let (status, a) = read_bytes_c(&refs);
    assert_eq!(status, OIR_OK, "{}", last_error());
    let from_memory = read_bytes(&refs).expect("read");
    assert_planes_match(a, &from_memory);
    assert_parts_match(a, &from_memory);
    assert_eq!(part_c(a, 0).read_path, OIR_READ_MEMORY);
    unsafe { oir_acquisition_free(a) };
}

/// The record through C — numbers, text, channels, events, the FluoView
/// export — is the Rust record.
#[test]
fn the_record_through_c_is_the_rust_one() {
    let f = fixture::recorded_file();
    let want = read_bytes(&[&f]).expect("read");
    let r = want.record();
    let (status, a) = read_bytes_c(&[&f]);
    assert_eq!(status, OIR_OK, "{}", last_error());
    assert_planes_match(a, &want);

    let info = record_c(a);
    assert!(same(info.pixel_x_um, r.pixel_x_um()));
    assert!(same(info.pixel_y_um, r.pixel_y_um()));
    assert!(same(info.z_step_um, r.z_step_um()));
    assert!(same(info.frame_interval_s, want.frame_interval_s()));
    assert!(same(info.duration_s, r.duration_s(want.shape().frames)));
    assert_eq!(info.width, r.width().unwrap_or(0));
    assert_eq!(info.height, r.height().unwrap_or(0));
    assert_eq!(info.slices, r.slices().unwrap_or(0));
    assert_eq!(info.roundtrip, r.roundtrip().map_or(-1, c_int::from));
    assert_eq!(info.free_run, r.free_run());
    // The fixture states these, so the comparison above is not NaN == NaN.
    assert_eq!(info.pixel_x_um, 0.621480569402239);
    assert_eq!(info.frame_interval_s, 1.0);

    assert_eq!(info.frame_time_count, r.frame_times_ms().len());
    assert_eq!(info.frame_time_count, 2);
    let mut times = vec![f64::NAN; info.frame_time_count];
    assert_eq!(
        unsafe { oir_frame_times(a, times.as_mut_ptr(), times.len()) },
        OIR_OK
    );
    assert_eq!(times, r.frame_times_ms());

    // Every text field, by its number, against the accessor of that name.
    let fields = [
        r.file_version(),
        r.created(),
        r.system_name(),
        r.system_version(),
        r.objective(),
        r.magnification(),
        r.numerical_aperture(),
        r.scanner(),
        r.sampling_speed_us(),
        r.sequential_mode(),
        r.integration(),
        r.integration_count(),
        r.rotation_deg(),
        r.pan_x_um(),
        r.pan_y_um(),
        r.zoom(),
        r.mirror_turret(),
        r.laser_wavelength_nm(),
        r.laser_transmissivity_pct(),
    ];
    assert_eq!(
        fields.len() as c_int,
        OIR_FIELD_LASER_TRANSMISSIVITY_PCT + 1
    );
    for (field, want) in fields.iter().enumerate() {
        let got = text(|b, n| unsafe { oir_record_string(a, field as c_int, b, n) });
        assert_eq!(got, want.unwrap_or(""), "field {field}");
    }
    assert_eq!(
        text(|b, n| unsafe { oir_record_string(a, OIR_FIELD_OBJECTIVE, b, n) }),
        "XLUMPLFLN20XW"
    );
    assert_eq!(
        text(|b, n| unsafe { oir_record_string(a, OIR_FIELD_SYSTEM_NAME, b, n) }),
        "FVMPE-RS"
    );

    assert_eq!(info.channel_count, r.channels().len());
    assert_eq!(info.channel_count, 1);
    for (i, ch) in r.channels().iter().enumerate() {
        let fields = [
            ch.name(),
            ch.detector(),
            ch.barrier_filter(),
            ch.dichroic(),
            ch.dye(),
            ch.emission_nm(),
            ch.bits(),
            ch.pmt_voltage(),
            ch.label(),
        ];
        assert_eq!(fields.len() as c_int, OIR_CHANNEL_LABEL + 1);
        for (field, want) in fields.iter().enumerate() {
            let got = text(|b, n| unsafe { oir_channel_string(a, i, field as c_int, b, n) });
            assert_eq!(got, want.unwrap_or(""), "channel {i} field {field}");
        }
    }
    assert_eq!(
        text(|b, n| unsafe { oir_channel_string(a, 0, OIR_CHANNEL_NAME, b, n) }),
        "CH1"
    );

    assert_eq!(info.event_count, r.events().len());
    assert_eq!(info.event_count, 1);
    for (i, e) in r.events().iter().enumerate() {
        assert_eq!(text(|b, n| unsafe { oir_event_name(a, i, b, n) }), e.name());
        let mut ms = f64::NAN;
        assert_eq!(unsafe { oir_event_time_ms(a, i, &mut ms) }, OIR_OK);
        assert_eq!(ms, e.time_ms());
    }

    let name = c("rec.oir");
    let fluoview = text(|b, n| unsafe { oir_fluoview_text(a, name.as_ptr(), b, n) });
    assert_eq!(fluoview, want.fluoview_text("rec.oir").expect("a record"));
    assert!(fluoview.contains("XLUMPLFLN20XW"));
    unsafe { oir_acquisition_free(a) };

    // A file whose record says nothing: no text, NaN and 0 for the numbers.
    let bare = fixture::stack_file(4, 2, &["t001_0_1_uid"], 8);
    let (status, a) = read_bytes_c(&[&bare]);
    assert_eq!(status, OIR_OK);
    let info = record_c(a);
    assert!(info.pixel_x_um.is_nan() && info.frame_interval_s.is_nan());
    assert_eq!(
        (info.roundtrip, info.event_count, info.frame_time_count),
        (-1, 0, 0)
    );
    assert_eq!(
        text(|b, n| unsafe { oir_record_string(a, OIR_FIELD_OBJECTIVE, b, n) }),
        ""
    );
    assert_eq!(unsafe { oir_frame_times(a, null_mut(), 0) }, OIR_OK);
    unsafe { oir_acquisition_free(a) };
}

/// What `snprintf` does, this does: cut to fit, NUL after, the whole length
/// returned — and never half a character.
#[test]
fn text_follows_snprintf() {
    let f = fixture::recorded_file();
    let (_, a) = read_bytes_c(&[&f]);
    let mut buf = [0x55u8; 8];
    let objective = |b: &mut [u8], cap| unsafe {
        oir_record_string(a, OIR_FIELD_OBJECTIVE, b.as_mut_ptr().cast(), cap)
    };
    assert_eq!(objective(&mut buf, 5), 13);
    assert_eq!(&buf[..6], b"XLUM\0\x55");
    assert_eq!(objective(&mut buf, 1), 13);
    assert_eq!(buf[0], 0);
    // No room at all: nothing written, the length still told.
    buf[0] = 0x55;
    assert_eq!(objective(&mut buf, 0), 13);
    assert_eq!(buf[0], 0x55);
    // A buffer that is not there, said to have room, is refused.
    let refused = unsafe { oir_record_string(a, OIR_FIELD_OBJECTIVE, null_mut(), 4) };
    assert_eq!(refused, -(OIR_INVALID_ARGUMENT as isize));
    assert!(last_error().contains("NULL"), "{}", last_error());
    unsafe { oir_acquisition_free(a) };

    // "héllo": the é is two bytes, and cutting at 2 would split it.
    let mut buf = [0x55u8; 4];
    assert_eq!(unsafe { put_text("héllo", buf.as_mut_ptr().cast(), 3) }, 6);
    assert_eq!(&buf[..3], b"h\0\x55");
    assert_eq!(unsafe { put_text("héllo", buf.as_mut_ptr().cast(), 4) }, 6);
    assert_eq!(&buf[..4], "hé\0".as_bytes());
}

/// An 8-bit acquisition comes through as bytes; a plane the acquisition did
/// not finish is counted; a part with no index is walked and says so.
#[test]
fn eight_bit_unfinished_and_walked_files_match_the_rust_reads() {
    let corpus = fixture::corpus();
    let get = |name: &str| corpus.iter().find(|(n, _)| *n == name).unwrap().1.clone();
    for name in [
        "eight_bit",
        "incomplete_trailing_plane",
        "no_index",
        "axes",
        "axes_from_names",
    ] {
        let f = get(name);
        let want = read_bytes(&[&f]).unwrap_or_else(|e| panic!("{name}: {e}"));
        let (status, a) = read_bytes_c(&[&f]);
        assert_eq!(status, OIR_OK, "{name}: {}", last_error());
        assert_planes_match(a, &want);
        assert_parts_match(a, &want);
        unsafe { oir_acquisition_free(a) };
    }
    let (_, a) = read_bytes_c(&[&get("eight_bit")]);
    assert_eq!(shape_c(a).sample_type, OIR_SAMPLE_U8);
    assert_eq!(shape_c(a).bytes_per_sample, 1);
    unsafe { oir_acquisition_free(a) };
    let (_, a) = read_bytes_c(&[&get("incomplete_trailing_plane")]);
    assert_eq!(unsafe { oir_dropped_planes(a) }, 1);
    unsafe { oir_acquisition_free(a) };
    let (_, a) = read_bytes_c(&[&get("no_index")]);
    assert_eq!(part_c(a, 0).block_source, OIR_BLOCKS_WALK);
    unsafe { oir_acquisition_free(a) };
    let (_, a) = read_bytes_c(&[&get("axes")]);
    let s = shape_c(a);
    assert_eq!((s.channels, s.slices, s.frames), (2, 1, 3));
    unsafe { oir_acquisition_free(a) };
}

/// Each of the reader's errors comes back as its own status.
#[test]
fn the_readers_errors_are_their_statuses() {
    let corpus = fixture::corpus();
    let get = |name: &str| corpus.iter().find(|(n, _)| *n == name).unwrap().1.clone();
    for (name, status) in [
        ("not_oir", OIR_NOT_OIR),
        ("empty", OIR_NOT_OIR),
        ("no_planes", OIR_NO_PLANES),
        ("metadata_only", OIR_NO_PLANES),
        ("no_frame_size", OIR_MALFORMED),
    ] {
        let f = get(name);
        let kind = read_bytes(&[&f]).expect_err(name).kind();
        let (got, a) = read_bytes_c(&[&f]);
        assert_eq!(got, status, "{name} ({kind:?}): {}", last_error());
        assert!(a.is_null());
    }
    // No parts at all is an acquisition with no planes, as in Rust.
    let mut a = std::ptr::dangling_mut::<OirAcquisition>();
    assert_eq!(
        unsafe { oir_read_bytes(null(), null(), 0, None, null_mut(), &mut a) },
        OIR_NO_PLANES
    );
    assert!(a.is_null());

    // A file that is not there names itself, and the system's reason.
    let dir = Scratch::new("capi-missing");
    let missing = dir.path().join("nothing.oir");
    let p = c(missing.to_str().unwrap());
    let mut a = std::ptr::dangling_mut::<OirAcquisition>();
    let status = unsafe { oir_read(p.as_ptr(), null(), None, null_mut(), &mut a) };
    assert_eq!(status, OIR_IO_ERROR);
    assert!(a.is_null());
    assert!(last_error().contains("nothing.oir"), "{}", last_error());
    assert!(
        last_error().contains("could not open the file"),
        "{}",
        last_error()
    );
    // ... but is still the first file of its own acquisition.
    let mut count = 0;
    assert_eq!(
        unsafe { oir_acquisition_part_count(p.as_ptr(), &mut count) },
        OIR_OK
    );
    assert_eq!(count, 1);
}

unsafe extern "C" fn count_and_go_on(fraction: f32, user: *mut c_void) -> c_int {
    let seen = &mut *user.cast::<Vec<f32>>();
    seen.push(fraction);
    1
}

unsafe extern "C" fn stop_at_once(_fraction: f32, user: *mut c_void) -> c_int {
    *user.cast::<usize>() += 1;
    0
}

/// More blocks than the reader asks the callback about every (1,024): 300
/// frames in four blocks each.
fn long_file() -> Vec<u8> {
    let names: Vec<String> = (1..=300).map(|i| format!("t{i:03}_0_1_uid")).collect();
    let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    fixture::stack_file(4, 4, &refs, 16)
}

#[test]
fn the_callback_reports_and_can_stop_a_read() {
    let dir = Scratch::new("capi-progress");
    let first = dir.write("rec.oir", &long_file());
    dir.write("rec_00001.oir", &fixture::part_file(300, 2, false));
    let p = c(first.to_str().unwrap());

    let mut seen: Vec<f32> = Vec::new();
    let mut a = null_mut();
    let user = (&mut seen as *mut Vec<f32>).cast();
    let status = unsafe { oir_read(p.as_ptr(), null(), Some(count_and_go_on), user, &mut a) };
    assert_eq!(status, OIR_OK, "{}", last_error());
    assert!(!seen.is_empty());
    assert_eq!(*seen.last().unwrap(), 1.0);
    unsafe { oir_acquisition_free(a) };

    let mut calls = 0usize;
    let mut a = std::ptr::dangling_mut::<OirAcquisition>();
    let user = (&mut calls as *mut usize).cast();
    let status = unsafe { oir_read(p.as_ptr(), null(), Some(stop_at_once), user, &mut a) };
    assert_eq!(status, OIR_CANCELLED);
    assert!(a.is_null(), "a stopped read hands back nothing");
    assert_eq!(calls, 1);

    let f = long_file();
    let (ptr, len) = (f.as_ptr(), f.len());
    let mut calls = 0usize;
    let user = (&mut calls as *mut usize).cast();
    let status = unsafe { oir_read_bytes(&ptr, &len, 1, Some(stop_at_once), user, &mut a) };
    assert_eq!(status, OIR_CANCELLED);
    assert!(a.is_null());
}

/// What C can say and Rust cannot comes back as a status and a reason.
#[test]
fn bad_arguments_are_refused_with_a_reason() {
    let refused = |status: c_int, what: &str| {
        assert_eq!(status, OIR_INVALID_ARGUMENT, "{what}");
        assert!(!last_error().is_empty(), "{what}");
    };
    let refused_len = |len: isize, what: &str| {
        assert_eq!(len, -(OIR_INVALID_ARGUMENT as isize), "{what}");
        assert!(!last_error().is_empty(), "{what}");
    };

    // Reading.
    let dir = Scratch::new("capi-refusals");
    let first = dir.write("rec.oir", &fixture::part_file(0, 2, true));
    let p = c(first.to_str().unwrap());
    let mut a = std::ptr::dangling_mut::<OirAcquisition>();
    refused(
        unsafe { oir_read(p.as_ptr(), null(), None, null_mut(), null_mut()) },
        "out NULL",
    );
    refused(
        unsafe { oir_read(null(), null(), None, null_mut(), &mut a) },
        "path NULL",
    );
    assert!(a.is_null());
    let not_utf8 = [b'a', 0xFF, 0];
    refused(
        unsafe { oir_read(not_utf8.as_ptr().cast(), null(), None, null_mut(), &mut a) },
        "not UTF-8",
    );
    for access in [-1, 2, 99] {
        let o = OirOptions { access };
        refused(
            unsafe { oir_read(p.as_ptr(), &o, None, null_mut(), &mut a) },
            "access",
        );
        assert!(last_error().contains("OIR_ACCESS_"), "{}", last_error());
    }
    let len = 16usize;
    refused(
        unsafe { oir_read_bytes(null(), &len, 1, None, null_mut(), &mut a) },
        "parts NULL",
    );
    let part: *const u8 = null();
    refused(
        unsafe { oir_read_bytes(&part, &len, 1, None, null_mut(), &mut a) },
        "part NULL, len 16",
    );
    let f = fixture::recorded_file();
    let ptr = f.as_ptr();
    refused(
        unsafe { oir_read_bytes(&ptr, &len, usize::MAX, None, null_mut(), &mut a) },
        "n huge",
    );
    let huge = usize::MAX;
    refused(
        unsafe { oir_read_bytes(&ptr, &huge, 1, None, null_mut(), &mut a) },
        "len huge",
    );
    refused(
        unsafe { oir_read_bytes(&ptr, &len, 1, None, null_mut(), null_mut()) },
        "out NULL",
    );
    refused(
        unsafe { oir_options_default(null_mut()) },
        "options out NULL",
    );
    let mut count = 0;
    refused(
        unsafe { oir_acquisition_part_count(null(), &mut count) },
        "parts of NULL",
    );
    refused(
        unsafe { oir_acquisition_part_count(p.as_ptr(), null_mut()) },
        "count NULL",
    );
    refused_len(
        unsafe { oir_acquisition_part(null(), 0, null_mut(), 0) },
        "part of NULL",
    );

    // A NULL handle, everywhere.
    let none: *const OirAcquisition = null();
    let mut shape = std::mem::MaybeUninit::<OirShape>::uninit();
    let mut info = std::mem::MaybeUninit::<OirRecordInfo>::uninit();
    let mut part = std::mem::MaybeUninit::<OirPartInfo>::uninit();
    let mut index = 0usize;
    let mut data = null();
    let mut buf = [0u8; 16];
    let mut ms = 0.0;
    refused(unsafe { oir_shape(none, shape.as_mut_ptr()) }, "shape");
    refused(
        unsafe { oir_plane_index(none, 0, 0, 0, &mut index) },
        "plane_index",
    );
    refused(
        unsafe { oir_plane_copy(none, 0, buf.as_mut_ptr().cast(), 16) },
        "plane_copy",
    );
    refused(unsafe { oir_plane_data(none, 0, &mut data) }, "plane_data");
    refused(
        unsafe { oir_record_info(none, info.as_mut_ptr()) },
        "record_info",
    );
    refused(
        unsafe { oir_frame_times(none, null_mut(), 0) },
        "frame_times",
    );
    refused(
        unsafe { oir_event_time_ms(none, 0, &mut ms) },
        "event_time_ms",
    );
    refused(
        unsafe { oir_part_info(none, 0, part.as_mut_ptr()) },
        "part_info",
    );
    refused_len(
        unsafe { oir_record_string(none, 0, null_mut(), 0) },
        "record_string",
    );
    refused_len(
        unsafe { oir_channel_string(none, 0, 0, null_mut(), 0) },
        "channel_string",
    );
    refused_len(
        unsafe { oir_event_name(none, 0, null_mut(), 0) },
        "event_name",
    );
    refused_len(
        unsafe { oir_fluoview_text(none, c("x").as_ptr(), null_mut(), 0) },
        "fluoview_text",
    );
    refused_len(
        unsafe { oir_part_path(none, 0, null_mut(), 0) },
        "part_path",
    );
    refused_len(
        unsafe { oir_part_map_error(none, 0, null_mut(), 0) },
        "part_map_error",
    );
    assert_eq!(unsafe { oir_dropped_planes(none) }, 0);
    assert_eq!(unsafe { oir_part_count(none) }, 0);

    // A live handle, and what is past its end, too small, or not a value.
    let (_, a) = read_bytes_c(&[&f]);
    let s = shape_c(a);
    let bytes = s.plane_len * s.bytes_per_sample;
    let mut plane = vec![0u8; bytes];
    refused(unsafe { oir_shape(a, null_mut()) }, "shape out NULL");
    refused(
        unsafe { oir_plane_index(a, s.channels, 0, 0, &mut index) },
        "channel past the end",
    );
    refused(
        unsafe { oir_plane_index(a, 0, s.slices, 0, &mut index) },
        "slice past the end",
    );
    refused(
        unsafe { oir_plane_index(a, 0, 0, s.frames, &mut index) },
        "frame past the end",
    );
    refused(
        unsafe { oir_plane_index(a, 0, 0, 0, null_mut()) },
        "index NULL",
    );
    refused(
        unsafe { oir_plane_copy(a, s.plane_count, plane.as_mut_ptr().cast(), bytes) },
        "plane past the end",
    );
    refused(
        unsafe { oir_plane_copy(a, 0, plane.as_mut_ptr().cast(), bytes - 1) },
        "plane buffer short",
    );
    assert!(
        plane.iter().all(|&b| b == 0),
        "a refused copy writes nothing"
    );
    refused(
        unsafe { oir_plane_copy(a, 0, null_mut(), bytes) },
        "plane out NULL",
    );
    refused(
        unsafe { oir_plane_data(a, s.plane_count, &mut data) },
        "plane data past the end",
    );
    assert!(data.is_null());
    refused(unsafe { oir_plane_data(a, 0, null_mut()) }, "data NULL");
    refused(unsafe { oir_record_info(a, null_mut()) }, "record out NULL");
    let mut times = [0.0f64; 1];
    refused(
        unsafe { oir_frame_times(a, times.as_mut_ptr(), 1) },
        "frame times short",
    );
    refused(
        unsafe { oir_frame_times(a, null_mut(), 2) },
        "frame times NULL",
    );
    refused_len(
        unsafe { oir_record_string(a, -1, null_mut(), 0) },
        "field -1",
    );
    refused_len(
        unsafe { oir_record_string(a, OIR_FIELD_LASER_TRANSMISSIVITY_PCT + 1, null_mut(), 0) },
        "field past the end",
    );
    refused_len(
        unsafe { oir_channel_string(a, 1, OIR_CHANNEL_NAME, null_mut(), 0) },
        "channel past the end",
    );
    refused_len(
        unsafe { oir_channel_string(a, 0, OIR_CHANNEL_LABEL + 1, null_mut(), 0) },
        "channel field",
    );
    refused_len(
        unsafe { oir_event_name(a, 1, null_mut(), 0) },
        "event past the end",
    );
    refused(
        unsafe { oir_event_time_ms(a, 1, &mut ms) },
        "event time past the end",
    );
    refused(
        unsafe { oir_event_time_ms(a, 0, null_mut()) },
        "time_ms NULL",
    );
    refused_len(
        unsafe { oir_fluoview_text(a, null(), null_mut(), 0) },
        "file_name NULL",
    );
    refused(
        unsafe { oir_part_info(a, 1, part.as_mut_ptr()) },
        "part past the end",
    );
    refused(unsafe { oir_part_info(a, 0, null_mut()) }, "part out NULL");
    refused_len(
        unsafe { oir_part_path(a, 1, null_mut(), 0) },
        "part path past the end",
    );
    refused_len(
        unsafe { oir_part_map_error(a, 1, null_mut(), 0) },
        "map error past the end",
    );
    unsafe { oir_acquisition_free(a) };
    unsafe { oir_acquisition_free(null_mut()) };
}

#[test]
fn the_signature_check_is_the_crates() {
    let f = fixture::recorded_file();
    assert!(unsafe { oir_is_oir(f.as_ptr(), f.len()) });
    assert!(unsafe { oir_is_oir(f.as_ptr(), 16) });
    assert!(!unsafe { oir_is_oir(f.as_ptr(), 15) });
    assert!(!unsafe { oir_is_oir(b"II*\0 not one".as_ptr(), 12) });
    assert!(!unsafe { oir_is_oir(null(), 16) });
    // More than is there is never read: only the signature is.
    assert!(unsafe { oir_is_oir(fixture::MAGIC.as_ptr(), 16) });
}

/// A panic is a status, not an unwind into C.
#[test]
fn a_panic_comes_back_as_failed() {
    assert_eq!(guarded(|| panic!("on purpose")), OIR_FAILED);
    assert!(last_error().contains("on purpose"), "{}", last_error());
    assert_eq!(guarded_len(|| panic!("again")), -(OIR_FAILED as isize));
    assert!(last_error().contains("again"), "{}", last_error());
}

#[test]
fn the_version_and_build_are_the_crates() {
    let v = unsafe { CStr::from_ptr(oir_version()) };
    assert_eq!(v.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
    assert_eq!(oir_mmap_supported(), crate::MMAP_SUPPORTED);
}

/// The fields a `typedef struct Name { ... } Name;` in the header declares,
/// in order.
fn header_fields<'a>(header: &'a str, name: &str) -> Vec<&'a str> {
    let body = header
        .split(&format!("typedef struct {name} {{"))
        .nth(1)
        .and_then(|rest| rest.split(&format!("}} {name};")).next())
        .unwrap_or_else(|| panic!("the {name} struct"));
    body.lines()
        .filter_map(|l| l.split(';').next())
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with("/*"))
        .map(|l| l.split_whitespace().last().unwrap())
        .collect()
}

/// The header is written by hand, so it is read here and held against this
/// module: every constant's value, each structure's fields in order, and the
/// functions it declares. The struct literals list every field without `..`,
/// so a field added to a Rust structure stops this from compiling until the
/// list — and the header — are brought along.
#[test]
fn the_header_matches_this_module() {
    let header = include_str!("../include/oir_reader.h");

    let defines: Vec<(&str, i64)> = header
        .lines()
        .filter_map(|l| l.trim().strip_prefix("#define OIR_"))
        .filter(|l| !l.starts_with("READER_H"))
        .map(|l| {
            let mut it = l.split_whitespace();
            let name = it.next().unwrap();
            (name, it.next().and_then(|v| v.parse().ok()).expect(name))
        })
        .collect();
    let want: &[(&str, c_int)] = &[
        ("OK", OIR_OK),
        ("CANCELLED", OIR_CANCELLED),
        ("INVALID_ARGUMENT", OIR_INVALID_ARGUMENT),
        ("UNAVAILABLE", OIR_UNAVAILABLE),
        ("FAILED", OIR_FAILED),
        ("IO_ERROR", OIR_IO_ERROR),
        ("NOT_OIR", OIR_NOT_OIR),
        ("MALFORMED", OIR_MALFORMED),
        ("NO_PLANES", OIR_NO_PLANES),
        ("ACCESS_AUTO", OIR_ACCESS_AUTO),
        ("ACCESS_BUFFERED", OIR_ACCESS_BUFFERED),
        ("SAMPLE_U8", OIR_SAMPLE_U8),
        ("SAMPLE_U16", OIR_SAMPLE_U16),
        ("READ_MAPPED", OIR_READ_MAPPED),
        ("READ_BUFFERED", OIR_READ_BUFFERED),
        ("READ_MEMORY", OIR_READ_MEMORY),
        ("BLOCKS_INDEX", OIR_BLOCKS_INDEX),
        ("BLOCKS_WALK", OIR_BLOCKS_WALK),
        ("FIELD_FILE_VERSION", OIR_FIELD_FILE_VERSION),
        ("FIELD_CREATED", OIR_FIELD_CREATED),
        ("FIELD_SYSTEM_NAME", OIR_FIELD_SYSTEM_NAME),
        ("FIELD_SYSTEM_VERSION", OIR_FIELD_SYSTEM_VERSION),
        ("FIELD_OBJECTIVE", OIR_FIELD_OBJECTIVE),
        ("FIELD_MAGNIFICATION", OIR_FIELD_MAGNIFICATION),
        ("FIELD_NUMERICAL_APERTURE", OIR_FIELD_NUMERICAL_APERTURE),
        ("FIELD_SCANNER", OIR_FIELD_SCANNER),
        ("FIELD_SAMPLING_SPEED_US", OIR_FIELD_SAMPLING_SPEED_US),
        ("FIELD_SEQUENTIAL_MODE", OIR_FIELD_SEQUENTIAL_MODE),
        ("FIELD_INTEGRATION", OIR_FIELD_INTEGRATION),
        ("FIELD_INTEGRATION_COUNT", OIR_FIELD_INTEGRATION_COUNT),
        ("FIELD_ROTATION_DEG", OIR_FIELD_ROTATION_DEG),
        ("FIELD_PAN_X_UM", OIR_FIELD_PAN_X_UM),
        ("FIELD_PAN_Y_UM", OIR_FIELD_PAN_Y_UM),
        ("FIELD_ZOOM", OIR_FIELD_ZOOM),
        ("FIELD_MIRROR_TURRET", OIR_FIELD_MIRROR_TURRET),
        ("FIELD_LASER_WAVELENGTH_NM", OIR_FIELD_LASER_WAVELENGTH_NM),
        (
            "FIELD_LASER_TRANSMISSIVITY_PCT",
            OIR_FIELD_LASER_TRANSMISSIVITY_PCT,
        ),
        ("CHANNEL_NAME", OIR_CHANNEL_NAME),
        ("CHANNEL_DETECTOR", OIR_CHANNEL_DETECTOR),
        ("CHANNEL_BARRIER_FILTER", OIR_CHANNEL_BARRIER_FILTER),
        ("CHANNEL_DICHROIC", OIR_CHANNEL_DICHROIC),
        ("CHANNEL_DYE", OIR_CHANNEL_DYE),
        ("CHANNEL_EMISSION_NM", OIR_CHANNEL_EMISSION_NM),
        ("CHANNEL_BITS", OIR_CHANNEL_BITS),
        ("CHANNEL_PMT_VOLTAGE", OIR_CHANNEL_PMT_VOLTAGE),
        ("CHANNEL_LABEL", OIR_CHANNEL_LABEL),
    ];
    assert_eq!(defines.len(), want.len(), "{defines:?}");
    for (name, value) in want {
        let found = defines.iter().find(|(n, _)| n == name);
        assert_eq!(found.map(|(_, v)| *v), Some(*value as i64), "OIR_{name}");
    }

    let _ = OirOptions { access: 0 };
    assert_eq!(header_fields(header, "OirOptions"), ["access"]);
    let _ = OirShape {
        width: 0,
        height: 0,
        channels: 0,
        slices: 0,
        frames: 0,
        sample_type: 0,
        bytes_per_sample: 0,
        plane_len: 0,
        plane_count: 0,
    };
    assert_eq!(
        header_fields(header, "OirShape"),
        [
            "width",
            "height",
            "channels",
            "slices",
            "frames",
            "sample_type",
            "bytes_per_sample",
            "plane_len",
            "plane_count"
        ]
    );
    let _ = OirRecordInfo {
        pixel_x_um: 0.0,
        pixel_y_um: 0.0,
        z_step_um: 0.0,
        frame_interval_s: 0.0,
        duration_s: 0.0,
        width: 0,
        height: 0,
        slices: 0,
        frame_time_count: 0,
        channel_count: 0,
        event_count: 0,
        roundtrip: 0,
        free_run: false,
    };
    assert_eq!(
        header_fields(header, "OirRecordInfo"),
        [
            "pixel_x_um",
            "pixel_y_um",
            "z_step_um",
            "frame_interval_s",
            "duration_s",
            "width",
            "height",
            "slices",
            "frame_time_count",
            "channel_count",
            "event_count",
            "roundtrip",
            "free_run"
        ]
    );
    let _ = OirPartInfo {
        size: 0,
        read_path: 0,
        block_source: 0,
        block_count: 0,
        planes_added: 0,
    };
    assert_eq!(
        header_fields(header, "OirPartInfo"),
        [
            "size",
            "read_path",
            "block_source",
            "block_count",
            "planes_added"
        ]
    );

    // Functions: what the header declares is what this module exports.
    let mut declared: Vec<&str> = header
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|w| w.starts_with("oir_") && header.contains(&format!("{w}(")))
        .collect();
    declared.sort_unstable();
    declared.dedup();
    let mut exported = Vec::from([
        "oir_last_error",
        "oir_version",
        "oir_mmap_supported",
        "oir_is_oir",
        "oir_options_default",
        "oir_read",
        "oir_read_bytes",
        "oir_acquisition_free",
        "oir_shape",
        "oir_plane_index",
        "oir_dropped_planes",
        "oir_plane_copy",
        "oir_plane_data",
        "oir_record_info",
        "oir_frame_times",
        "oir_record_string",
        "oir_channel_string",
        "oir_event_name",
        "oir_event_time_ms",
        "oir_fluoview_text",
        "oir_part_count",
        "oir_part_info",
        "oir_part_path",
        "oir_part_map_error",
        "oir_acquisition_part_count",
        "oir_acquisition_part",
    ]);
    exported.sort_unstable();
    assert_eq!(declared, exported);
    // And each of those is a function of this module, with the C signature
    // the header gives it: a mismatch here does not compile.
    let _: [*const (); 26] = [
        oir_last_error as extern "C" fn() -> *const c_char as *const (),
        oir_version as extern "C" fn() -> *const c_char as *const (),
        oir_mmap_supported as extern "C" fn() -> bool as *const (),
        oir_is_oir as unsafe extern "C" fn(*const u8, usize) -> bool as *const (),
        oir_options_default as unsafe extern "C" fn(*mut OirOptions) -> c_int as *const (),
        oir_read
            as unsafe extern "C" fn(
                *const c_char,
                *const OirOptions,
                OirProgress,
                *mut c_void,
                *mut *mut OirAcquisition,
            ) -> c_int as *const (),
        oir_read_bytes
            as unsafe extern "C" fn(
                *const *const u8,
                *const usize,
                usize,
                OirProgress,
                *mut c_void,
                *mut *mut OirAcquisition,
            ) -> c_int as *const (),
        oir_acquisition_free as unsafe extern "C" fn(*mut OirAcquisition) as *const (),
        oir_shape as unsafe extern "C" fn(*const OirAcquisition, *mut OirShape) -> c_int
            as *const (),
        oir_plane_index
            as unsafe extern "C" fn(*const OirAcquisition, usize, usize, usize, *mut usize) -> c_int
            as *const (),
        oir_dropped_planes as unsafe extern "C" fn(*const OirAcquisition) -> usize as *const (),
        oir_plane_copy
            as unsafe extern "C" fn(*const OirAcquisition, usize, *mut c_void, usize) -> c_int
            as *const (),
        oir_plane_data
            as unsafe extern "C" fn(*const OirAcquisition, usize, *mut *const c_void) -> c_int
            as *const (),
        oir_record_info as unsafe extern "C" fn(*const OirAcquisition, *mut OirRecordInfo) -> c_int
            as *const (),
        oir_frame_times as unsafe extern "C" fn(*const OirAcquisition, *mut f64, usize) -> c_int
            as *const (),
        oir_record_string
            as unsafe extern "C" fn(*const OirAcquisition, c_int, *mut c_char, usize) -> isize
            as *const (),
        oir_channel_string
            as unsafe extern "C" fn(
                *const OirAcquisition,
                usize,
                c_int,
                *mut c_char,
                usize,
            ) -> isize as *const (),
        oir_event_name
            as unsafe extern "C" fn(*const OirAcquisition, usize, *mut c_char, usize) -> isize
            as *const (),
        oir_event_time_ms as unsafe extern "C" fn(*const OirAcquisition, usize, *mut f64) -> c_int
            as *const (),
        oir_fluoview_text
            as unsafe extern "C" fn(
                *const OirAcquisition,
                *const c_char,
                *mut c_char,
                usize,
            ) -> isize as *const (),
        oir_part_count as unsafe extern "C" fn(*const OirAcquisition) -> usize as *const (),
        oir_part_info
            as unsafe extern "C" fn(*const OirAcquisition, usize, *mut OirPartInfo) -> c_int
            as *const (),
        oir_part_path
            as unsafe extern "C" fn(*const OirAcquisition, usize, *mut c_char, usize) -> isize
            as *const (),
        oir_part_map_error
            as unsafe extern "C" fn(*const OirAcquisition, usize, *mut c_char, usize) -> isize
            as *const (),
        oir_acquisition_part_count as unsafe extern "C" fn(*const c_char, *mut usize) -> c_int
            as *const (),
        oir_acquisition_part
            as unsafe extern "C" fn(*const c_char, usize, *mut c_char, usize) -> isize
            as *const (),
    ];
}

/// A NULL output is refused as NULL, not as a buffer that is too small.
#[test]
fn a_null_output_says_null() {
    let f = fixture::recorded_file();
    let (status, a) = read_bytes_c(&[&f]);
    assert_eq!(status, OIR_OK, "{}", last_error());
    let shape = shape_c(a);
    let bytes = shape.plane_len * shape.bytes_per_sample;
    assert_eq!(
        unsafe { oir_plane_copy(a, 0, null_mut(), bytes) },
        OIR_INVALID_ARGUMENT
    );
    assert!(last_error().contains("NULL"), "{}", last_error());
    assert_eq!(
        unsafe { oir_frame_times(a, null_mut(), 1000) },
        OIR_INVALID_ARGUMENT
    );
    assert!(last_error().contains("NULL"), "{}", last_error());
    unsafe { oir_acquisition_free(a) };
}
