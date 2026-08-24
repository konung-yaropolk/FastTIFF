//! Opening a stack without stopping the interface.
//!
//! Opening is not cheap: the IFD chain has to be walked end to end, and then
//! every channel decodes a frame to find its display range. On a large stack
//! that is seconds, and done on the thread drawing the interface it is seconds
//! of a window that does not repaint, does not respond, and on Windows acquires
//! a "not responding" title bar. The work has nothing to do with the interface,
//! so it belongs on a thread of its own.
//!
//! What the frontend gets back is a handle it polls once a frame: a stage to
//! show while the load runs, and the finished stack when it lands.

use crate::stack::Stack;
use std::path::PathBuf;

/// How far an open has got.
///
/// Two stages because the work has two shapes. Walking the IFD chain has no
/// knowable length — the chain ends when it ends — so it can only be reported
/// as happening. The contrast scans are one per channel and countable, so they
/// can be reported as a fraction. A bar that claimed a percentage for the first
/// would be inventing it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadStage {
    /// Reading the file and indexing its planes.
    Reading,
    /// Measuring each channel's display range.
    Contrast { done: usize, total: usize },
}

impl LoadStage {
    /// Wording for a progress readout.
    pub fn label(self) -> &'static str {
        match self {
            LoadStage::Reading => "Reading file…",
            LoadStage::Contrast { .. } => "Measuring channels…",
        }
    }

    /// How far along, when that is knowable. `None` means "running, length
    /// unknown" — a caller should show motion rather than a proportion.
    ///
    /// A single channel counts as unknowable, not as 0%. There is one scan and
    /// it is either running or finished, so a bar can only sit at zero for the
    /// whole of it and then vanish — which reads as stuck, and is the one thing
    /// a progress indicator must never do. Most files are single-channel, so
    /// this is the common case rather than a corner of one.
    pub fn fraction(self) -> Option<f32> {
        match self {
            LoadStage::Reading => None,
            LoadStage::Contrast { total, .. } if total <= 1 => None,
            LoadStage::Contrast { done, total } => Some(done as f32 / total as f32),
        }
    }
}

/// Where the bytes of a stack come from.
///
/// The two hosts get files by different routes — a path from a dialog or argv,
/// or bytes from a browser picker — and this is where they meet, so everything
/// downstream of it is shared.
pub enum LoadSource {
    /// A path on disk. Only where there is a filesystem to name.
    #[cfg(feature = "mmap")]
    Path(PathBuf),
    /// The file's bytes, plus a name to show for them.
    Bytes(Vec<u8>, PathBuf),
}

impl LoadSource {
    /// The name to show for this file while it loads and once it is open.
    pub fn name(&self) -> PathBuf {
        match self {
            #[cfg(feature = "mmap")]
            LoadSource::Path(p) => p.clone(),
            LoadSource::Bytes(_, name) => name.clone(),
        }
    }

    /// Open it here and now, reporting progress as it goes.
    ///
    /// The body of the worker thread, and the whole of the load on a target
    /// with no threads to spawn.
    pub fn load(
        self,
        apply_pseudocolor: bool,
        on_stage: &mut dyn FnMut(LoadStage),
    ) -> anyhow::Result<Stack> {
        match self {
            #[cfg(feature = "mmap")]
            LoadSource::Path(path) => Stack::open_reporting(path, apply_pseudocolor, on_stage),
            LoadSource::Bytes(bytes, name) => {
                Stack::from_bytes_reporting(bytes, name, apply_pseudocolor, on_stage)
            }
        }
    }
}

#[cfg(feature = "threads")]
mod threaded {
    use super::{LoadSource, LoadStage};
    use crate::stack::Stack;
    use std::path::PathBuf;
    use std::sync::mpsc::{channel, Receiver};
    use std::sync::{Arc, Mutex};

    /// A load in flight on a worker thread.
    pub struct Loading {
        rx: Receiver<anyhow::Result<Stack>>,
        stage: Arc<Mutex<LoadStage>>,
        name: PathBuf,
    }

    impl Loading {
        /// Start `source` loading on its own thread.
        ///
        /// `None` when the thread will not spawn, which the caller should treat
        /// as "load it here instead" rather than as a failure to open — a
        /// frozen interface is much better than no picture.
        pub fn spawn(source: LoadSource, apply_pseudocolor: bool) -> Option<Self> {
            let name = source.name();
            let (tx, rx) = channel();
            let stage = Arc::new(Mutex::new(LoadStage::Reading));
            let stage_worker = Arc::clone(&stage);
            std::thread::Builder::new()
                .name("fasttiff-load".to_owned())
                .spawn(move || {
                    let result = source.load(apply_pseudocolor, &mut |s| {
                        if let Ok(mut slot) = stage_worker.lock() {
                            *slot = s;
                        }
                    });
                    // A closed receiver means the frontend gave up on this load
                    // (another file was opened over it); dropping the result is
                    // then exactly right.
                    let _ = tx.send(result);
                })
                .ok()?;
            Some(Loading { rx, stage, name })
        }

        /// How far it has got, for a progress readout.
        pub fn stage(&self) -> LoadStage {
            self.stage.lock().map(|s| *s).unwrap_or(LoadStage::Reading)
        }

        /// The file being loaded.
        pub fn name(&self) -> &PathBuf {
            &self.name
        }

        /// The finished stack, once there is one. `None` while it is still
        /// running — never blocks.
        pub fn take(&mut self) -> Option<anyhow::Result<Stack>> {
            match self.rx.try_recv() {
                Ok(result) => Some(result),
                // Disconnected without a result means the worker died, which
                // for a panic in decode is the one outcome that must not leave
                // the interface waiting for ever.
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    Some(Err(anyhow::anyhow!("the loader stopped unexpectedly")))
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => None,
            }
        }
    }
}

#[cfg(feature = "threads")]
pub use threaded::Loading;
