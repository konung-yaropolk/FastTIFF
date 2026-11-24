//! What playing a compressed movie allocates: nothing frame-sized, once the
//! first frames have been shown.
//!
//! A test binary of its own because it replaces the global allocator with one
//! that counts, which would count every other suite's allocations too. The
//! tests take a lock so that only one of them plays at a time, and they count
//! what every thread allocates — the read-ahead worker's and a parallel
//! decode's included.
//!
//! The decode behind the view used to allocate and zero-fill a buffer for
//! every channel of every frame shown, on the interface thread and on the
//! read-ahead worker alike, and the library under it a frame-sized native
//! buffer for every compressed one: four 2 MiB allocations a frame for the
//! two-channel stack here. The interface thread now decodes into buffers its
//! view keeps, and the worker into the buffers of the frames handed back to
//! it after they were shown.

use fast_tiff_viewer::fast_tiff_lib::{
    Compression, MetadataFormat, SampleType, StackMetaWrite, TiffWriter, WriterOptions,
};
use fast_tiff_viewer::scivis_render::pack::PackedVolume;
use fast_tiff_viewer::scivis_render::{
    ChannelUniform, VolumeCaps, VolumeImage, VolumeInterp, VolumeParams, VolumeUploadError,
};
use fast_tiff_viewer::{
    ChannelKind, DecodeMode, FrameInputs, ImageRenderer, Lut, Viewer, VolumeRenderer,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Counts allocations of at least [`AT_LEAST`] bytes.
struct Counting;

static AT_LEAST: AtomicUsize = AtomicUsize::new(usize::MAX);
static COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() >= AT_LEAST.load(Ordering::Relaxed) {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if layout.size() >= AT_LEAST.load(Ordering::Relaxed) {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size >= AT_LEAST.load(Ordering::Relaxed) {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// A frame of the movie: 1024 x 1024, which is as large as a frame has to be
/// for the library to decode it on several cores.
const SIDE: u32 = 1024;
const CHANNELS: usize = 2;
const TIMEPOINTS: usize = 12;

/// One channel's plane: what "frame-sized" means here.
const PLANE_BYTES: usize = SIDE as usize * SIDE as usize * 2;

/// A renderer that keeps nothing but a staging buffer per channel, as a GPU
/// backend's queue does: an upload is a copy into it.
struct Staging {
    planes: Vec<Vec<u8>>,
}

impl Staging {
    fn new() -> Self {
        Staging {
            planes: vec![vec![0; PLANE_BYTES]; CHANNELS],
        }
    }

    fn upload(&mut self, channel: usize, bytes: &[u8]) {
        if let Some(plane) = self.planes.get_mut(channel) {
            let n = plane.len().min(bytes.len());
            plane[..n].copy_from_slice(&bytes[..n]);
        }
    }
}

impl ImageRenderer for Staging {
    fn max_2d_texture_size(&self) -> u32 {
        16384
    }
    fn ensure_size(&mut self, _: u32, _: u32, _: &[ChannelKind]) {}
    fn upload_channel_u8(&mut self, channel: usize, _: u32, _: u32, samples: &[u8]) {
        self.upload(channel, samples);
    }
    fn upload_channel_u16(&mut self, channel: usize, _: u32, _: u32, samples: &[u16]) {
        self.upload(channel, bytemuck::cast_slice(samples));
    }
    fn upload_channel_f32(&mut self, channel: usize, _: u32, _: u32, samples: &[f32]) {
        self.upload(channel, bytemuck::cast_slice(samples));
    }
    fn upload_lut(&mut self, _: usize, _: &Lut) {}
    fn set_params(&mut self, _: &[ChannelUniform], _: u32, _: [f32; 2], _: [f32; 2]) {}
}

impl VolumeRenderer for Staging {
    fn volume_caps(&self) -> VolumeCaps {
        VolumeCaps {
            max_3d_size: 2048,
            unorm16: true,
            float32: true,
        }
    }
    fn volume_memory_available(&self) -> Option<u64> {
        None
    }
    fn upload_packed_volume(&mut self, _: &PackedVolume) -> Result<(), VolumeUploadError> {
        Ok(())
    }
    fn release_volume(&mut self) {}
    fn take_late_upload_error(&mut self) -> Option<(VolumeUploadError, usize)> {
        None
    }
    fn volume_pending(&mut self) -> bool {
        false
    }
    fn set_volume_interp(&mut self, _: VolumeInterp) {}
    fn set_volume_acceleration(&mut self, _: bool, _: bool) {}
    fn set_volume_target(&mut self, _: u32, _: u32, _: u32) {}
    fn set_volume_params(&mut self, _: VolumeParams) {}
    fn capture_volume(&mut self, _: u32, _: u32) -> Result<(), String> {
        Ok(())
    }
    fn take_volume_capture(&mut self) -> Option<Result<VolumeImage, String>> {
        None
    }
}

/// A two-channel movie, compressed with the predictor, in ImageJ's channel
/// order: what a two-photon recording is. Written once for both tests.
fn movie() -> Vec<u8> {
    static MOVIE: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    MOVIE.get_or_init(write_movie).clone()
}

fn write_movie() -> Vec<u8> {
    let compression = if cfg!(feature = "codec-zstd") {
        Compression::Zstd
    } else {
        Compression::Lzw
    };
    let meta = StackMetaWrite::new(CHANNELS, 1);
    let opts = WriterOptions::new(SIDE, SIDE, SampleType::U16)
        .compression(compression)
        .predictor(true)
        .rows_per_strip(64)
        .metadata(meta)
        .metadata_format(MetadataFormat::ImageJ);
    let mut w = TiffWriter::new(std::io::Cursor::new(Vec::new()), opts).unwrap();
    let n = (SIDE * SIDE) as usize;
    for plane in 0..CHANNELS * TIMEPOINTS {
        let px: Vec<u16> = (0..n)
            .map(|i| ((i % 977) * 31 + (i / 1024) * 7 + plane * 113) as u16)
            .collect();
        w.write_frame_u16(&px).unwrap();
    }
    w.finish().unwrap().into_inner()
}

/// Frame-sized allocations made while `play` runs.
fn frame_sized(play: impl FnOnce()) -> usize {
    COUNT.store(0, Ordering::SeqCst);
    AT_LEAST.store(PLANE_BYTES / 2, Ordering::SeqCst);
    play();
    AT_LEAST.store(usize::MAX, Ordering::SeqCst);
    COUNT.load(Ordering::SeqCst)
}

fn opened() -> Viewer {
    let mut viewer = Viewer::new();
    viewer
        .load_bytes(movie(), "movie.tif".into())
        .expect("the movie opens");
    // The parallel regime, forced: a frame this size decodes on several
    // cores, which is where a decode makes the most of its own buffers.
    viewer.decode_mode = DecodeMode::Threaded;
    let doc = viewer.document().expect("open");
    assert_eq!(doc.display().dims().channels, CHANNELS, "two channels");
    viewer
}

fn show(viewer: &mut Viewer, renderer: &mut Staging, frame: usize) {
    viewer.document_mut().expect("open").set_frame(frame);
    viewer.sync(renderer, &FrameInputs::default());
}

/// Scrubbing through the movie, every frame decoded on the interface thread:
/// after the first, none allocates a buffer for a channel.
#[test]
fn showing_frames_inline_allocates_nothing_frame_sized() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut viewer = opened();
    let mut renderer = Staging::new();
    show(&mut viewer, &mut renderer, 0);
    show(&mut viewer, &mut renderer, 1);
    let n = frame_sized(|| {
        for t in 2..TIMEPOINTS {
            show(&mut viewer, &mut renderer, t);
        }
    });
    assert_eq!(
        n,
        0,
        "frame-sized allocations over {} frames",
        TIMEPOINTS - 2
    );
}

/// Playing the movie, every frame decoded ahead by the read-ahead worker and
/// taken from it: after the first two, the worker decodes each into the
/// buffers of one already shown. (Without threads there is no worker, and a
/// played frame is decoded as a scrubbed one is.)
#[cfg(feature = "threads")]
#[test]
fn playing_through_the_read_ahead_allocates_nothing_frame_sized() {
    use std::time::{Duration, Instant};
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut viewer = opened();
    let mut renderer = Staging::new();
    viewer.playback.play();
    // Each frame is shown once the worker has the next one ready, so that it
    // is the read-ahead that is measured rather than the race between it and
    // the interface thread.
    let next = |viewer: &mut Viewer, renderer: &mut Staging, t: usize| {
        show(viewer, renderer, t);
        let start = Instant::now();
        while !fast_tiff_viewer::testing::read_ahead_ready(viewer) {
            assert!(
                start.elapsed() < Duration::from_secs(20),
                "the read-ahead never answered for frame {}",
                t + 1
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    };
    for t in 0..3 {
        next(&mut viewer, &mut renderer, t);
    }
    let n = frame_sized(|| {
        for t in 3..TIMEPOINTS - 1 {
            next(&mut viewer, &mut renderer, t);
        }
    });
    assert_eq!(
        n,
        0,
        "frame-sized allocations over {} frames",
        TIMEPOINTS - 4
    );
}
