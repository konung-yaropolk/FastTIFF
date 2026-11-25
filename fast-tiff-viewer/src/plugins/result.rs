//! Turning what a plugin produced into a document the app can show.
//!
//! A plugin's result is written as a TIFF and opened through the ordinary
//! reader, rather than built into a [`Document`](crate::Document) field by field.
//! That costs an encode and a decode of the result, and it is still the right
//! answer: a plugin result is a *real* document from the moment it exists.
//! Channel setup, auto-contrast, LUTs, the c/z/t interpretation, the
//! histogram, 3D, "Save as…", opening it in a second window: all of it works
//! with no new code, and none of it can drift from what an opened file does,
//! because it is the same path.
//!
//! # Written as it arrives
//!
//! The file is written a plane at a time, by a [`ResultWriter`], as the planes
//! arrive — which, for a plugin that streams, is while it is still running
//! (see `fasttiff_plugin_api::ImageStream`). On the desktop it goes to a file
//! beside the source (when it has a source path), which the new
//! window opens through a memory map like any other file; in a browser it goes
//! to memory, there being no file system ([`ResultDest::Memory`]).
//!
//! This replaced building the whole result in memory first. A stabilised
//! recording was held as the plugin's planes, copied into the host's, and
//! encoded into a second whole copy, every step after the plugin's own
//! progress had reached the end: an 8 GiB result needed some 16 GiB to land,
//! and more again while it was handed to a new window. Now the host holds one
//! plane of it at a time and the rest is a file the operating system pages.
//!
//! A result that does arrive whole — an [`ImageResult`], the right shape for a
//! projection or a single plane — goes through the same writer, each plane let
//! go once it is written, so both kinds produce the same bytes.
//!
//! # Whose file it is, and when it goes
//!
//! The receiving document owns its temporary TIFF, including when it opens
//! in another process. Closing or replacing it releases the file. A user save
//! can promote that TIFF directly when its display metadata is still current.

#[cfg(not(target_arch = "wasm32"))]
mod guardian;
#[cfg(not(target_arch = "wasm32"))]
mod scratch;
use fast_tiff_lib::{
    DisplayMode as LibDisplayMode, SampleType, StackMetaWrite, TiffWriter, WriterOptions,
};
use fasttiff_plugin_api::{
    DisplayMode, ImageResult, ImageSpec, Lut, PixelType, PlaneData, PlaneRef, StackInfo,
};
use std::io::{Cursor, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// What a plugin is told when it declares a result after pushing planes of
/// another: one wording whether it is behind the C boundary or not.
pub(crate) const SECOND_RESULT: &str =
    "the plugin declared a second result after pushing planes of the first";

/// Where a result is written while it arrives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResultDest {
    /// A scratch directory with no alternate volume.
    #[cfg(not(target_arch = "wasm32"))]
    Dir(PathBuf),
    /// A TIFF beside this original file, including results derived from an import.
    #[cfg(not(target_arch = "wasm32"))]
    BesideSource(PathBuf),
    /// A buffer in memory, sized for the result up front. A browser's only
    /// choice.
    Memory,
}

impl ResultDest {
    /// Write next to the original document. A document without a filesystem
    /// source keeps its result in memory.
    pub fn for_source(source: Option<&Path>) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(path) = source {
            return Self::BesideSource(path.to_owned());
        }
        #[cfg(target_arch = "wasm32")]
        let _ = source;
        Self::Memory
    }
}

/// A temporary TIFF owned by a writer or an open document.
///
/// What makes "a cancelled run leaves nothing behind" true without every exit
/// path having to remember it: a writer that is dropped half-written — the
/// plugin stopped, failed or panicked — takes its file with it. A document
/// showing the result holds one too (see [`Derived`]), so closing it is what
/// deletes the file.
///
/// Best effort, always: a file still mapped by someone cannot be deleted on
/// Windows, so a mapped file can need a retry.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
pub struct ScratchFile {
    path: PathBuf,
    keep: bool,
    mapping: std::sync::OnceLock<std::sync::Weak<fast_tiff_lib::TiffStack>>,
    guardian: Option<guardian::Guardian>,
}

#[cfg(not(target_arch = "wasm32"))]
static OWNED: std::sync::Mutex<std::collections::BTreeSet<PathBuf>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());

#[cfg(not(target_arch = "wasm32"))]
impl ScratchFile {
    /// After the desktop event loop exits, wait for any remaining mappings
    /// or retrying deletions. Only paths explicitly owned by this process
    /// participate; a successful handoff removed the parent's registration.
    pub fn cleanup_on_exit() {
        // The UI has been dropped and the plugin workers joined. Any guards
        // still living belong to a finishing load or to a deletion waiting
        // for the read-ahead worker to release its mapping.
        while !OWNED.lock().unwrap().is_empty() {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }

    pub(crate) fn track_mapping(&self, tiff: &std::sync::Arc<fast_tiff_lib::TiffStack>) {
        let _ = self.mapping.set(std::sync::Arc::downgrade(tiff));
    }

    fn own(path: PathBuf, guardian: Option<guardian::Guardian>) -> Self {
        OWNED.lock().unwrap().insert(path.clone());
        Self {
            path,
            keep: false,
            mapping: Default::default(),
            guardian,
        }
    }

    /// The file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn disarm(&mut self) {
        OWNED.lock().unwrap().remove(&self.path);
        self.keep = true;
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for ScratchFile {
    fn drop(&mut self) {
        if self.keep {
            return;
        }
        if remove_scratch(&self.path) {
            OWNED.lock().unwrap().remove(&self.path);
            return;
        }
        // Still open somewhere. For a document that is closing, that is a
        // worker of its own — the read-ahead thread lets go of its handle on
        // the mapping a moment after the window does, and Windows will not
        // delete a mapped file. So try again for a few seconds, off this
        // thread, allowing other processes time to release their handles.
        let path = std::mem::take(&mut self.path);
        let mapping = self.mapping.take();
        let retry_path = path.clone();
        let retry = std::thread::Builder::new()
            .name("fasttiff-scratch".to_owned())
            .spawn(move || {
                // A long plugin may still read the previous document after
                // another file opens. Wait for our actual readers, however
                // long they need, before counting retries for outside locks.
                while mapping.as_ref().is_some_and(|m| m.strong_count() > 0) {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
                for _ in 0..50 {
                    if remove_scratch(&retry_path) {
                        OWNED.lock().unwrap().remove(&retry_path);
                        return;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
                log::warn!("could not delete temporary TIFF {}", retry_path.display());
                OWNED.lock().unwrap().remove(&retry_path);
            });
        if let Err(e) = retry {
            log::warn!("could not schedule cleanup of {}: {e}", path.display());
            OWNED.lock().unwrap().remove(&path);
        }
    }
}

/// A source directory always belongs to the user, even when empty.
#[cfg(not(target_arch = "wasm32"))]
fn remove_scratch(path: &Path) -> bool {
    match std::fs::remove_file(path) {
        Ok(()) => true,
        Err(e) => e.kind() == std::io::ErrorKind::NotFound,
    }
}

/// What a document that is a plugin's result knows about itself beyond its
/// pixels: the document it was made from, and — on the desktop — the scratch
/// file it is read from, which it owns.
///
/// Held by the [`Document`](crate::Document) the result was opened as, so the file
/// is deleted when the document is closed or replaced, the stack's own mapping
/// of it having gone first.
#[derive(Debug, Default)]
pub struct Derived {
    pub(crate) name: Option<PathBuf>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) initial_display: Option<Box<crate::display::Display>>,
    pub(crate) importer_id: Option<String>,
    made_from: Option<PathBuf>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) scratch: Option<std::sync::Arc<ScratchFile>>,
}

impl Derived {
    /// Provenance of an imported document whose converted TIFF lives in memory.
    pub fn imported(path: PathBuf, importer_id: String) -> Self {
        Self {
            importer_id: Some(importer_id),
            made_from: Some(path),
            ..Default::default()
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn open_file(mut self, file: ScratchFile, name: PathBuf) -> std::io::Result<crate::LoadSource> {
        #[cfg(feature = "mmap")]
        let from = {
            let from = crate::LoadSource::Path(file.path.clone());
            self.scratch = Some(std::sync::Arc::new(file));
            from
        };
        #[cfg(not(feature = "mmap"))]
        let from = crate::LoadSource::Bytes(std::fs::read(file.path())?.into(), name.clone());
        self.name = Some(name);
        Ok(crate::LoadSource::Derived(Box::new(from), self))
    }

    /// Take ownership only of a scratch result explicitly handed off by a parent viewer.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn handed_off(
        path: PathBuf,
        name: PathBuf,
        made_from: Option<PathBuf>,
    ) -> std::io::Result<crate::LoadSource> {
        if scratch_name(&path).is_none() || name.file_name() != Some(name.as_os_str()) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "invalid FastTIFF scratch handoff",
            ));
        }
        Self {
            made_from,
            ..Default::default()
        }
        .open_file(ScratchFile::own(path, None), name)
    }

    /// The file the result was made from, when it was made from one: where
    /// "Save as…" starts, rather than the folder of scratch files the result
    /// happens to be read from.
    pub fn made_from(&self) -> Option<&Path> {
        self.made_from.as_deref()
    }

    /// The scratch file the document is read from and will delete, when it
    /// is read from one.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn scratch(&self) -> Option<&Path> {
        self.scratch.as_deref().map(ScratchFile::path)
    }
}

/// What the TIFF writer writes into.
enum Out {
    #[cfg(not(target_arch = "wasm32"))]
    File(Box<std::io::BufWriter<scratch::DiskOutput>>),
    Memory(Cursor<Vec<u8>>),
}

impl Write for Out {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            Out::File(f) => f.write(buf),
            Out::Memory(c) => c.write(buf),
        }
    }
    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            Out::File(f) => f.write_all(buf),
            Out::Memory(c) => c.write_all(buf),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            Out::File(f) => f.flush(),
            Out::Memory(c) => c.flush(),
        }
    }
}

impl Seek for Out {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            Out::File(f) => f.seek(pos),
            Out::Memory(c) => c.seek(pos),
        }
    }
}

/// A result being written, plane by plane.
///
/// Declared once with an [`ImageSpec`], then given each plane in `xyczt`
/// order; each is checked against the declaration, written, and let go. The
/// metadata — calibration, names, colours, the source's LUTs — can change
/// until it is finished ([`redescribe`](Self::redescribe)): the file's
/// description is written last, with its directory.
pub(crate) struct ResultWriter {
    w: TiffWriter<Out>,
    spec: ImageSpec,
    /// `spec.plane_count()`, which [`ImageSpec::validate`] has checked can be
    /// counted.
    planes: usize,
    /// The source's tables, kept so a description that changes after the
    /// first plane is written with them as the first one was.
    luts: Vec<Lut>,
}

impl ResultWriter {
    /// Start writing a result declared as `spec`, to `dest`.
    ///
    /// `luts` are the tables the source was being shown with when the run
    /// started; a result that kept the source's channels is written with
    /// them, so it opens looking like what it came from.
    pub(crate) fn begin(spec: ImageSpec, luts: &[Lut], dest: &ResultDest) -> anyhow::Result<Self> {
        spec.validate()
            .map_err(|e| anyhow::anyhow!("the plugin returned an unusable image: {e}"))?;
        let planes = spec
            .plane_count()
            .ok_or_else(|| anyhow::anyhow!("the plugin declared more planes than can exist"))?;
        let opts = writer_options(&spec, luts);
        let out = match dest {
            #[cfg(not(target_arch = "wasm32"))]
            ResultDest::Dir(_) | ResultDest::BesideSource(_) => Out::File(Box::new(
                std::io::BufWriter::new(scratch::allocate(dest, &spec)?),
            )),
            ResultDest::Memory => in_memory(&spec, planes)?,
        };
        let w = TiffWriter::new(out, opts)?;
        Ok(ResultWriter {
            w,
            spec,
            planes,
            luts: luts.to_vec(),
        })
    }

    /// Change what the result says about itself — its colours, its
    /// metadata, its channel names, its name — before it is finished.
    ///
    /// The file's description is written when it is finished, so nothing
    /// already written changes: a plugin may describe its result at any point
    /// up to then, as the C contract has always allowed. The shape cannot
    /// change — those planes are in the file already — and a `spec` that
    /// changes it is refused.
    pub(crate) fn redescribe(&mut self, spec: ImageSpec) -> anyhow::Result<()> {
        let s = &self.spec;
        if (
            spec.width,
            spec.height,
            spec.channels,
            spec.slices,
            spec.frames,
        ) != (s.width, s.height, s.channels, s.slices, s.frames)
            || spec.pixel_type != s.pixel_type
        {
            anyhow::bail!(
                "the plugin changed its result's shape after declaring it: {}x{} {:?} \
                 {}x{}x{} became {}x{} {:?} {}x{}x{}",
                s.width,
                s.height,
                s.pixel_type,
                s.channels,
                s.slices,
                s.frames,
                spec.width,
                spec.height,
                spec.pixel_type,
                spec.channels,
                spec.slices,
                spec.frames
            );
        }
        // Its file is named after it already, and so is the window that
        // will show it.
        if spec.name != s.name {
            anyhow::bail!(
                "the plugin renamed its result after declaring it, from {:?} to {:?}",
                s.name,
                spec.name
            );
        }
        self.w.set_metadata(stack_meta(&spec, &self.luts))?;
        self.spec = spec;
        Ok(())
    }

    /// Replace the result's display lookup tables before it is finished.
    /// The IFD directory is written last, so this updates both a file-backed
    /// and an in-memory writer without touching pixels already written.
    pub(crate) fn set_luts(&mut self, luts: Vec<Lut>) -> anyhow::Result<()> {
        self.luts = luts;
        self.w.set_metadata(stack_meta(&self.spec, &self.luts))?;
        Ok(())
    }

    /// Write the next plane.
    pub(crate) fn push(&mut self, plane: PlaneRef<'_>) -> anyhow::Result<()> {
        let index = self.w.frames_written();
        let expect = self.planes;
        if index >= expect {
            anyhow::bail!("the plugin pushed more than the {expect} planes it declared");
        }
        self.spec
            .check_plane(index, plane)
            .map_err(|e| anyhow::anyhow!("the plugin returned an unusable image: {e}"))?;
        match plane {
            PlaneRef::U8(v) => self.w.write_frame_bytes(v),
            PlaneRef::U16(v) => self.w.write_frame_bytes(&le_bytes_u16(v)),
            PlaneRef::F32(v) => self.w.write_frame_bytes(&le_bytes_f32(v)),
        }?;
        Ok(())
    }

    /// Planes written so far.
    pub(crate) fn pushed(&self) -> usize {
        self.w.frames_written()
    }

    /// The file it is being written to, if it is a file.
    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn path(&self) -> Option<&Path> {
        match self.w.get_ref() {
            Out::File(f) => Some(f.get_ref().owner.path()),
            Out::Memory(_) => None,
        }
    }

    /// Write the directory that makes the planes a TIFF, and close it.
    ///
    /// Refused, and the file deleted, when fewer planes arrived than were
    /// declared — the words are the ones a returned image that miscounted
    /// gets.
    pub(crate) fn finish(self) -> anyhow::Result<ResultTiff> {
        let ResultWriter {
            w,
            spec,
            planes,
            luts: _,
        } = self;
        let pushed = w.frames_written();
        if pushed != planes {
            return Err(anyhow::anyhow!(
                "the plugin returned an unusable image: {}",
                spec.count_mismatch(pushed)
            ));
        }
        let data = match w.finish()? {
            #[cfg(not(target_arch = "wasm32"))]
            Out::File(f) => {
                // Flushed by `finish`; closed here, before anyone opens it.
                let output = f.into_inner().map_err(|e| e.into_error())?;
                ResultData::File(output.owner)
            }
            Out::Memory(c) => ResultData::Memory(c.into_inner()),
        };
        Ok(ResultTiff { spec, data })
    }
}

impl std::fmt::Debug for ResultWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResultWriter")
            .field("name", &self.spec.name)
            .field("pushed", &self.pushed())
            .field("of", &self.planes)
            .finish()
    }
}

/// Where a finished result's bytes are.
#[derive(Debug)]
enum ResultData {
    #[cfg(not(target_arch = "wasm32"))]
    File(ScratchFile),
    Memory(Vec<u8>),
}

/// A finished result: a real TIFF, in a file or in memory, and what it was
/// declared as.
///
/// A file is deleted when this is dropped, unless it was handed on — to a new
/// window by supervising its handoff, to a document in this
/// one by [`into_document`](Self::into_document), or to where the plugin
/// asked by [`save_as`](Self::save_as).
#[derive(Debug)]
pub struct ResultTiff {
    spec: ImageSpec,
    data: ResultData,
}

impl ResultTiff {
    /// What the result was declared as.
    pub fn spec(&self) -> &ImageSpec {
        &self.spec
    }

    /// What the plugin called it.
    pub fn name(&self) -> &str {
        &self.spec.name
    }

    /// The file it was written to, when it was written to one.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn path(&self) -> Option<&Path> {
        match &self.data {
            ResultData::File(f) => Some(&f.path),
            ResultData::Memory(_) => None,
        }
    }

    /// The bytes, when it is held in memory.
    pub fn bytes(&self) -> Option<&[u8]> {
        match &self.data {
            #[cfg(not(target_arch = "wasm32"))]
            ResultData::File(_) => None,
            ResultData::Memory(b) => Some(b),
        }
    }

    /// Ask the cleanup process to launch a viewer and supervise its ownership.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open_in_new_window(&mut self) -> std::io::Result<()> {
        let ResultData::File(f) = &mut self.data else {
            return Err(std::io::Error::other("no temporary TIFF to open"));
        };
        if f.keep {
            return Err(std::io::Error::other("result already handed off"));
        }
        f.guardian
            .as_mut()
            .ok_or_else(|| std::io::Error::other("no scratch supervisor"))?
            .open_window()?;
        // The supervisor now follows the receiving viewer, even if this one exits.
        f.disarm();
        Ok(())
    }

    /// Hand the file on to whoever opens it next — a new window's process —
    /// and say where it is. A result held in memory comes back as it was.
    ///
    /// Simulate the receiving process in library tests without launching a viewer.
    #[cfg(all(test, not(target_arch = "wasm32")))]
    fn into_kept_path(self) -> Result<PathBuf, Box<ResultTiff>> {
        match self.data {
            ResultData::File(mut f) => {
                f.disarm();
                Ok(std::mem::take(&mut f.path))
            }
            data => Err(Box::new(ResultTiff {
                spec: self.spec,
                data,
            })),
        }
    }

    /// The whole file, in memory: a file is read and then deleted.
    pub fn into_bytes(self) -> std::io::Result<Vec<u8>> {
        match self.data {
            #[cfg(not(target_arch = "wasm32"))]
            ResultData::File(f) => std::fs::read(&f.path),
            ResultData::Memory(b) => Ok(b),
        }
    }

    /// The name a document showing it goes by: the result's own, as a file.
    fn document_name(&self) -> PathBuf {
        format!("{}.tif", file_stem(&self.spec.name)).into()
    }

    /// Open it in this window, as a document made from `made_from` — the file
    /// of the document it was computed from, if that had one.
    ///
    /// A file is memory-mapped where this build maps files, and becomes the
    /// document's to delete (see [`Derived`]); otherwise the bytes are read
    /// and the file goes at once. Either way the document is named as the
    /// result, says it came from `made_from` rather than from a scratch
    /// folder, and does not offer the scratch file to plugins as its source.
    pub fn into_document(self, made_from: Option<PathBuf>) -> std::io::Result<crate::LoadSource> {
        let name = self.document_name();
        self.into_named_document(name, made_from, None)
    }

    /// Load a conversion under its display name, retaining importer provenance.
    pub fn into_named_document(
        self,
        name: PathBuf,
        made_from: Option<PathBuf>,
        importer_id: Option<String>,
    ) -> std::io::Result<crate::LoadSource> {
        let mut derived = Derived {
            importer_id,
            made_from,
            ..Default::default()
        };
        let from = match self.data {
            #[cfg(not(target_arch = "wasm32"))]
            ResultData::File(file) => return derived.open_file(file, name),
            ResultData::Memory(bytes) => crate::LoadSource::Bytes(bytes.into(), name.clone()),
        };
        derived.name = Some(name);
        Ok(crate::LoadSource::Derived(Box::new(from), derived))
    }

    /// Put it at `path`, replacing what is there: moved when it was written
    /// to a file on the same volume, copied across volumes.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn save_as(self, path: &Path) -> std::io::Result<()> {
        check_save_name(path)?;
        match self.data {
            // Dropping `f` afterwards deletes what is left: the original
            // after a copy, or nothing after a move.
            ResultData::File(f) => {
                if std::fs::rename(&f.path, path).is_ok() {
                    return Ok(());
                }
                // Another volume, most likely: the temporary directory and the
                // destination need not share one.
                std::fs::copy(&f.path, path).map(|_| ())
            }
            ResultData::Memory(bytes) => std::fs::write(path, bytes),
        }
    }

    /// Read the result back as an [`ImageResult`]: its declaration, and its
    /// planes as they are in the file.
    ///
    /// For tests and tools that want the pixels in hand — the app never does
    /// this, it opens the file. The planes are the file's own samples, so a
    /// result read back is exactly the result that was written.
    pub fn to_image(&self) -> anyhow::Result<ImageResult> {
        let tiff = match &self.data {
            #[cfg(not(target_arch = "wasm32"))]
            ResultData::File(f) => fast_tiff_lib::TiffStack::from_bytes(std::fs::read(&f.path)?)?,
            ResultData::Memory(b) => fast_tiff_lib::TiffStack::from_bytes(b.clone())?,
        };
        let mut planes = Vec::with_capacity(tiff.frames.len());
        let mut raw = Vec::new();
        for frame in &tiff.frames {
            fast_tiff_lib::read_frame_bytes_into(
                &tiff.data,
                frame,
                tiff.byte_order,
                fast_tiff_lib::ByteOrder::Little,
                &mut raw,
            )?;
            planes.push(match self.spec.pixel_type {
                PixelType::U8 => PlaneData::U8(raw.clone()),
                PixelType::U16 | PixelType::I16 => PlaneData::U16(
                    raw.as_chunks::<2>()
                        .0
                        .iter()
                        .map(|&b| u16::from_le_bytes(b))
                        .collect(),
                ),
                PixelType::F32 => PlaneData::F32(
                    raw.as_chunks::<4>()
                        .0
                        .iter()
                        .map(|&b| f32::from_le_bytes(b))
                        .collect(),
                ),
            });
        }
        Ok(self.spec.clone().with_planes(planes))
    }
}

/// A file name from a result's name, with anything a file system would object
/// to replaced. Also what a new window is titled, so it is needed even where
/// no file is written.
pub fn file_stem(name: &str) -> String {
    let stem: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || "-_. ".contains(c) {
                c
            } else {
                '_'
            }
        })
        .collect();
    let stem = stem.trim().trim_matches('.').to_string();
    if stem.is_empty() {
        "result".to_string()
    } else {
        stem
    }
}

/// What every result's directory is named starting with, so the [`sweep`]
/// touches nothing it did not make.
#[cfg(not(target_arch = "wasm32"))]
const RUN_PREFIX: &str = "run-";

/// Return the document name from one of our unique temporary filenames.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn scratch_name(path: &Path) -> Option<&str> {
    let mut parts = path
        .file_name()?
        .to_str()?
        .strip_prefix(".fasttiff-")?
        .strip_suffix(".temp")?
        .splitn(4, '-');
    for _ in 0..3 {
        let field = parts.next()?;
        if field.is_empty() || !field.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
    }
    parts.next().filter(|n| n.len() > 4 && n.ends_with(".tif"))
}

/// Scratch paths may still be owned by a different open document.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn check_save_name(path: &Path) -> std::io::Result<()> {
    // Reserve the spelling on case-insensitive filesystems too.
    let lower = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_ascii_lowercase();
    if scratch_name(Path::new(&lower)).is_some() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Choose a normal TIFF filename when saving",
        ));
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn create_file(
    root: &Path,
    stem: &str,
    source: Option<&Path>,
) -> std::io::Result<(std::fs::File, ScratchFile)> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let pid = std::process::id();
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    // Leave room for the unique prefix and suffix within filesystem component
    // limits. The window receives the full display name separately.
    let short: String = stem
        .chars()
        .scan(0usize, |bytes, c| {
            *bytes += c.len_utf8();
            (*bytes <= 160).then_some(c)
        })
        .collect();
    loop {
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = root.join(format!(".fasttiff-{pid}-{n}-{stamp}-{short}.tif.temp"));
        match guardian::create(&path, stem, source) {
            Ok((file, guard)) => return Ok((file, ScratchFile::own(path, guard))),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
}

/// Remove old crash remnants and legacy results in the system scratch folder.
/// A live result is protected by its mapping on Windows and its lease on Unix.
/// Age is that of the newest thing in a directory, so a result still being
/// written — its file's time moves as it grows — is never taken for an old
/// one. Files of the flat layout this replaced, one `<name>.tif` per result
/// straight in `root`, go by the same rule.
#[cfg(not(target_arch = "wasm32"))]
pub fn sweep(root: &Path, age: std::time::Duration) -> usize {
    let Some(cutoff) = std::time::SystemTime::now().checked_sub(age) else {
        return 0;
    };
    let modified = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    let ours = format!("{RUN_PREFIX}{}-", std::process::id());
    let Ok(entries) = std::fs::read_dir(root) else {
        return 0;
    };
    let mut removed = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        if kind.is_file() && name.ends_with(".tif") {
            if modified(&path).is_some_and(|t| t < cutoff) {
                let _ = std::fs::remove_file(&path);
            }
            continue;
        }
        if !kind.is_dir() || !name.starts_with(RUN_PREFIX) || name.starts_with(&ours) {
            continue;
        }
        let inside: Vec<PathBuf> = std::fs::read_dir(&path)
            .map(|d| d.flatten().map(|e| e.path()).collect())
            .unwrap_or_default();
        let newest = inside
            .iter()
            .filter_map(|p| modified(p))
            .chain(modified(&path))
            .max();
        if newest.is_none_or(|t| t >= cutoff) {
            continue;
        }
        #[cfg(unix)]
        let _lease = {
            let Ok(dir) = std::fs::File::open(&path) else {
                continue;
            };
            if rustix::fs::flock(&dir, rustix::fs::FlockOperation::NonBlockingLockExclusive)
                .is_err()
            {
                // In use, or the filesystem cannot establish exclusivity.
                continue;
            }
            dir
        };
        for p in &inside {
            let _ = std::fs::remove_file(p);
        }
        if std::fs::remove_dir(&path).is_ok() {
            removed += 1;
        }
    }
    removed
}

/// Somewhere in memory for a whole result, sized for it up front.
///
/// The buffer would otherwise double its way to the finished size, and every
/// doubling copies everything written so far. A generous IFD per plane, plus
/// the header: over-reserving a little costs one allocation that is never
/// grown. Reserved with `try_reserve_exact`, so a result too big for the
/// machine is an error saying so rather than an abort.
fn in_memory(spec: &ImageSpec, planes: usize) -> anyhow::Result<Out> {
    let room = spec
        .plane_len()
        .checked_mul(spec.pixel_type.bytes())
        .and_then(|plane| planes.checked_mul(plane))
        .and_then(|pixels| pixels.checked_add(planes.checked_mul(512)?))
        .and_then(|all| all.checked_add(4096))
        .ok_or_else(|| anyhow::anyhow!("the result is too large to hold in memory"))?;
    let mut buf = Vec::new();
    buf.try_reserve_exact(room).map_err(|e| {
        anyhow::anyhow!(
            "the result needs {:.1} MiB, which cannot be held in memory: {e}",
            room as f64 / (1u64 << 20) as f64
        )
    })?;
    Ok(Out::Memory(Cursor::new(buf)))
}

/// Encode a plugin's result as a TIFF in memory.
///
/// The planes are written in the order the contract states — channel fastest,
/// then Z, then time — which is the order the reader expects, so the result
/// re-opens with the axes it declared.
pub fn to_tiff_bytes(image: &ImageResult, info: Option<&StackInfo>) -> anyhow::Result<Vec<u8>> {
    to_tiff_bytes_reporting(image, info, &[], &mut |_| true)
        .transpose()
        .unwrap_or_else(|| Err(anyhow::anyhow!("cancelled")))
}

/// [`to_tiff_bytes`], taking the result: each plane is dropped as soon as it
/// has been written, so encoding holds the result about once, plus the file
/// it is becoming, rather than both in full.
pub fn into_tiff_bytes(image: ImageResult, info: Option<&StackInfo>) -> anyhow::Result<Vec<u8>> {
    into_tiff_bytes_reporting(image, info, &[], &mut |_| true)
        .transpose()
        .unwrap_or_else(|| Err(anyhow::anyhow!("cancelled")))
}

/// [`to_tiff_bytes`], reporting how far through the planes it is, and
/// stopping when `on_progress` returns `false` (which returns `Ok(None)`).
pub fn to_tiff_bytes_reporting(
    image: &ImageResult,
    info: Option<&StackInfo>,
    luts: &[Lut],
    on_progress: &mut dyn FnMut(f32) -> bool,
) -> anyhow::Result<Option<Vec<u8>>> {
    image
        .validate()
        .map_err(|e| anyhow::anyhow!("the plugin returned an unusable image: {e}"))?;
    let mut spec = image.spec();
    override_metadata(&mut spec, info);
    let planes = image.planes.iter().map(PlaneData::as_plane_ref);
    write_all(spec, planes, luts, &ResultDest::Memory, on_progress)?
        .map(ResultTiff::into_bytes)
        .transpose()
        .map_err(Into::into)
}

/// [`to_tiff_bytes_reporting`], taking the result and dropping each plane once
/// it is written — see [`into_tiff_bytes`].
pub(crate) fn into_tiff_bytes_reporting(
    image: ImageResult,
    info: Option<&StackInfo>,
    luts: &[Lut],
    on_progress: &mut dyn FnMut(f32) -> bool,
) -> anyhow::Result<Option<Vec<u8>>> {
    Ok(
        match write_result(image, info, luts, &ResultDest::Memory, on_progress)? {
            Some(r) => Some(r.into_bytes()?),
            None => None,
        },
    )
}

/// Write a whole result to `dest`, letting each plane go once it is written.
///
/// What a result that came back whole goes through, so it is written exactly
/// as one that was streamed: the same writer, the same file, the same bytes.
/// `info`, when given, is the metadata to write in place of the result's own.
/// `on_progress` returns `false` to stop, which returns `Ok(None)` and leaves
/// no file.
pub fn write_result(
    image: ImageResult,
    info: Option<&StackInfo>,
    luts: &[Lut],
    dest: &ResultDest,
    on_progress: &mut dyn FnMut(f32) -> bool,
) -> anyhow::Result<Option<ResultTiff>> {
    image
        .validate()
        .map_err(|e| anyhow::anyhow!("the plugin returned an unusable image: {e}"))?;
    let (mut spec, planes) = image.into_parts();
    override_metadata(&mut spec, info);
    let total = planes.len().max(1);
    let mut w = ResultWriter::begin(spec, luts, dest)?;
    // By value, so each plane is freed as soon as it has been written.
    for (i, plane) in planes.into_iter().enumerate() {
        if !on_progress(i as f32 / total as f32) {
            return Ok(None);
        }
        w.push(plane.as_plane_ref())?;
    }
    w.finish().map(Some)
}

/// [`write_result`] over borrowed planes.
fn write_all<'a>(
    spec: ImageSpec,
    planes: impl ExactSizeIterator<Item = PlaneRef<'a>>,
    luts: &[Lut],
    dest: &ResultDest,
    on_progress: &mut dyn FnMut(f32) -> bool,
) -> anyhow::Result<Option<ResultTiff>> {
    let total = planes.len().max(1);
    let mut w = ResultWriter::begin(spec, luts, dest)?;
    for (i, plane) in planes.enumerate() {
        if !on_progress(i as f32 / total as f32) {
            return Ok(None);
        }
        w.push(plane)?;
    }
    w.finish().map(Some)
}

/// An explicit caller's metadata wins over the result's own.
fn override_metadata(spec: &mut ImageSpec, info: Option<&StackInfo>) {
    if let Some(info) = info {
        spec.metadata = Some(info.clone());
    }
}

/// The writer options for a result: its sample type, and the metadata that
/// goes in its header.
fn writer_options(spec: &ImageSpec, luts: &[Lut]) -> WriterOptions {
    let sample = match spec.pixel_type {
        PixelType::U8 => SampleType::U8,
        PixelType::U16 => SampleType::U16,
        PixelType::I16 => SampleType::I16,
        PixelType::F32 => SampleType::F32,
    };
    WriterOptions::new(spec.width, spec.height, sample).metadata(stack_meta(spec, luts))
}

/// The metadata a result's file describes it with: what [`writer_options`]
/// starts the writer with, and what [`ResultWriter::redescribe`] replaces it
/// with.
fn stack_meta(spec: &ImageSpec, luts: &[Lut]) -> StackMetaWrite {
    let mut meta = StackMetaWrite::new(spec.channels.max(1), spec.slices.max(1));
    if let Some(info) = &spec.metadata {
        meta = meta.mode(match info.mode {
            DisplayMode::Composite => LibDisplayMode::Composite,
            DisplayMode::Color => LibDisplayMode::Color,
            DisplayMode::Grayscale => LibDisplayMode::Grayscale,
        });
        if let Some(u) = &info.unit {
            meta = meta.unit(u.clone());
        }
        if let Some(s) = info.frame_interval_s {
            meta = meta.frame_interval_s(s);
        }
        // Spacing and calibration are the whole reason an importer bothers to
        // report metadata: without them a measurement made on the result is in
        // pixels and frames rather than microns and seconds.
        if let (Some(w), Some(h)) = (info.spacing.x, info.spacing.y) {
            meta = meta.pixel_size(w, h);
        }
        if let Some(z) = info.spacing.z {
            meta = meta.spacing(z);
        }
        if let Some((c0, c1)) = info.calibration {
            meta = meta.calibration(c0, c1);
        }
        for (i, name) in info.channel_names.iter().enumerate() {
            meta = meta.channel(name.clone(), fast_tiff_lib::metadata::composite_color(i));
        }
        // The source file's own `ImageDescription`, carried into tag 270 of the
        // written one. An importer that read a vendor format puts the vendor's
        // metadata here, and this is the only thing standing between that and
        // it being lost on conversion.
        if let Some(d) = &info.description {
            if !d.trim().is_empty() {
                meta = meta.trailing(d.clone());
            }
        }
    } else if spec.channels > 1 {
        // A multi-channel result with nothing said about it is far more useful
        // composited than shown one channel at a time.
        meta = meta.mode(LibDisplayMode::Composite);
    }

    // Per-channel colour, when the plugin asked for one. It has to go in as a
    // full LUT rather than as `channel(name, color)`: the latter feeds the OME
    // dialect, and what an ImageJ-format file (and this viewer) reads is the
    // binary LUT block.
    if !spec.channel_colors.is_empty() {
        meta = meta.mode(LibDisplayMode::Composite);
        for c in 0..spec.channels.max(1) {
            let color = spec
                .channel_colors
                .get(c)
                .copied()
                .unwrap_or_else(|| fast_tiff_lib::metadata::composite_color(c));
            meta = meta.channel_lut(fast_tiff_lib::color_ramp_lut(color));
        }
    } else if luts.len() == spec.channels.max(1) {
        // The tables the source is being shown with, carried into a result that
        // kept its channels — so a stabilised copy of a recording opens looking
        // like the recording, its built-in LUT and all, rather than in the
        // viewer's default grey.
        //
        // A whole table rather than a colour: a file's own LUT need not be a
        // ramp at all — Fire, a segmentation palette — and rebuilding one from
        // a single colour would throw that away. `channel_colors` still wins
        // when the plugin set it: a plugin that states its colours has an
        // opinion about what its own result means, which outranks what the
        // source happened to look like. The display mode is left alone, these
        // being the source's tables and not a reason to composite a grayscale
        // stack.
        for lut in luts {
            meta = meta.channel_lut(*lut);
        }
    }
    meta
}

/// A plane's samples as the little-endian bytes the file stores them in.
///
/// Borrowed rather than built, on every platform this ships for: the samples
/// are already laid out the way the file wants them, so there is nothing to do.
/// Building a fresh `Vec` per plane — one sample at a time, through an
/// iterator that cannot say how long it will be, so the buffer grew as it went
/// — copied the entire result a second time. On a long recording that was the
/// bulk of the wait after the bar had reached the end.
macro_rules! le_bytes {
    ($name:ident, $ty:ty) => {
        fn $name(v: &[$ty]) -> std::borrow::Cow<'_, [u8]> {
            #[cfg(target_endian = "little")]
            {
                std::borrow::Cow::Borrowed(bytemuck::cast_slice(v))
            }
            // A big-endian host has to swap, and then there is a copy to make.
            // Nothing this runs on today takes this branch.
            #[cfg(target_endian = "big")]
            {
                let mut out = Vec::with_capacity(std::mem::size_of::<$ty>() * v.len());
                for s in v {
                    out.extend_from_slice(&s.to_le_bytes());
                }
                std::borrow::Cow::Owned(out)
            }
        }
    };
}
le_bytes!(le_bytes_u16, u16);
le_bytes!(le_bytes_f32, f32);

/// Encode a result and open it as a stack, exactly as a file would be opened.
pub fn to_stack(
    image: &ImageResult,
    info: Option<&StackInfo>,
    apply_pseudocolor: bool,
) -> anyhow::Result<crate::document::Document> {
    let bytes = to_tiff_bytes(image, info)?;
    let name = info
        .map(|i| i.name.clone())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| image.name.clone());
    crate::document::Document::from_bytes(bytes, name.into(), apply_pseudocolor)
}

#[cfg(test)]
#[path = "result_tests.rs"]
mod tests;
