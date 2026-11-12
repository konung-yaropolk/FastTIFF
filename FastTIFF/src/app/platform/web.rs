//! The browser: one canvas in a page, no filesystem, no threads; downloads
//! and tabs instead.
//!
//! What [`Platform`] asks for, done the browser's way. The decisions are not
//! here — they are [`Target`]'s, and this applies `Target::Browser`'s.

use super::{Delivery, Platform, Purpose, Target, NOT_IN_MEMORY};
use crate::app::{Document, Opened, PanelLayout, View2d};
use fast_tiff_viewer::plugins::{Handover, Registry, ResultTiff};
use fast_tiff_viewer::save::SaveSource;
use fast_tiff_viewer::LoadSource;
use fasttiff_plugin_api::FileType;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32};
use std::sync::mpsc::{channel, Receiver, Sender};

/// The browser host.
pub(crate) struct Browser {
    /// Where the asynchronous picker sends what it was given, for
    /// [`take_incoming_files`](Platform::take_incoming_files) to collect.
    picked_tx: Sender<Opened>,
    picked: Receiver<Opened>,
}

impl Platform for Browser {
    const WEB: bool = true;

    fn new() -> Self {
        let (picked_tx, picked) = channel();
        Browser { picked_tx, picked }
    }

    fn registry() -> Registry {
        // The built-ins alone: nothing to scan, and no `dlopen` to load a
        // library with — the only difference between the two Plugins menus.
        Registry::new()
    }

    /// Results are kept in memory here, so there is nothing to sweep.
    fn sweep_old_results() {}

    /// A drop event here carries the bytes rather than a path. They are egui's
    /// `Arc<[u8]>`, and the document keeps that very buffer. A copy would be a
    /// whole-file memcpy on the interface thread, and would keep the file in
    /// the heap twice while the load runs inline on this frame — a file's
    /// worth less room for the load's own work.
    ///
    /// It does not lower a drop's peak, which is set before the app sees the
    /// file: eframe reads it into a `Vec` and turns that into the `Arc` by
    /// copying, so two copies are alive for a moment whatever is done here,
    /// and the largest file a drop can open is what it was. A drop held once
    /// would need eframe to hand its `Vec` over, or a source that reads the
    /// file a range at a time.
    fn take_incoming_files(&mut self, ctx: &egui::Context) -> Vec<Opened> {
        let mut opened: Vec<Opened> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| {
                    let bytes = std::sync::Arc::clone(f.bytes.as_ref()?);
                    Some(Opened::Bytes(
                        fast_tiff_lib::Bytes::Shared(bytes),
                        f.name.clone(),
                    ))
                })
                .collect()
        });
        opened.extend(self.picked.try_iter());
        opened
    }

    /// An asynchronous picker, which posts what it is given back to
    /// [`take_incoming_files`](Platform::take_incoming_files). TIFF only:
    /// a file an importer reads would reach the reader as it is, since an
    /// importer reads a path and a browser has none to give it.
    fn pick_file(&self, ctx: &egui::Context, _readable: &[FileType]) -> Option<Opened> {
        let tx = self.picked_tx.clone();
        let ctx = ctx.clone();
        wasm_bindgen_futures::spawn_local(async move {
            if let Some(handle) = rfd::AsyncFileDialog::new()
                .add_filter("TIFF", &["tif", "tiff"])
                .pick_file()
                .await
            {
                let name = handle.file_name();
                let _ = tx.send(Opened::Bytes(handle.read().await.into(), name));
            }
            // Wake the UI whether or not a file was chosen.
            ctx.request_repaint();
        });
        None
    }

    fn path_source(path: PathBuf) -> Result<LoadSource, String> {
        Err(format!(
            "{} cannot be opened: a browser opens no file by its path",
            path.display()
        ))
    }

    /// A browser cannot be asked for a place, only given a name; see
    /// [`Target::place`], which never asks here.
    fn ask_path(
        _purpose: Purpose,
        _name: Option<&str>,
        _dir: Option<&Path>,
        _filters: &[FileType],
    ) -> Option<PathBuf> {
        None
    }

    fn download(bytes: &[u8], name: &str, mime: &str) -> Result<(), String> {
        crate::web_save::download(bytes, name, mime)
    }

    fn write_tiff(
        _source: &SaveSource,
        path: &Path,
        _report: &mut dyn FnMut(f32) -> bool,
    ) -> Result<(), String> {
        Err(format!(
            "a browser has no file to write {} into",
            path.display()
        ))
    }

    fn write_result(_result: ResultTiff, path: &Path) -> Result<(), String> {
        Err(format!(
            "a browser has no file to write {} into",
            path.display()
        ))
    }

    /// Here and now: see [`Platform::spawn_job`] for why, and what it costs.
    fn spawn_job(work: impl FnOnce() + Send + 'static) {
        work();
    }

    fn wake_for_progress() {}

    /// A tab, not a process — see `crate::web_open` for why a blob and not a
    /// message, and why the plugin had to run synchronously for this call to
    /// be allowed. A refused popup is the expected failure, not an exceptional
    /// one: the result comes back and is shown in this tab, with the reason in
    /// the status line.
    fn deliver_new_document(
        result: ResultTiff,
        name: String,
        _progress: &AtomicU32,
        _cancel: &AtomicBool,
    ) -> Handover {
        let how = Target::Browser.delivery(false, result.bytes().map(<[u8]>::len), |_| true);
        let opened = match (how, result.bytes()) {
            (Delivery::SendBytes, Some(bytes)) => {
                crate::web_open::open_bytes_in_new_tab(bytes, &name)
            }
            (Delivery::Refuse(why), _) => Err(why.to_string()),
            // Nothing here is held in a file.
            (Delivery::MapFile | Delivery::SendBytes, _) => Err(NOT_IN_MEMORY.to_string()),
        };
        match opened {
            Ok(()) => Handover::Opened(name),
            Err(why) => Handover::Refused(Box::new(result), why),
        }
    }

    /// The canvas is sized by the page, and `ViewportCommand` means nothing
    /// there: the picture is fitted to the canvas instead (see
    /// [`Opening::FitCanvas`](super::Opening::FitCanvas)).
    fn manage_window(
        &mut self,
        _ctx: &egui::Context,
        _view: &mut View2d,
        _panel: &mut PanelLayout,
        _document: Option<&Document>,
        _toolbar_h: f32,
        _bar_h: f32,
    ) {
    }

    /// Not closed over here, as it has never been: the browser's own console
    /// has the panic, and the page can be reloaded.
    fn gpu_lost() -> Option<String> {
        None
    }

    fn plugin_folder() -> Option<PathBuf> {
        None
    }

    fn reveal_folder(_dir: &Path) -> std::io::Result<()> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "a browser has no file manager",
        ))
    }
}
