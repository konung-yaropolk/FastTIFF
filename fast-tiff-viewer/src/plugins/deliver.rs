//! What happens to a plugin's result once its run is over.
//!
//! A run ends with an [`Outcome`]; a window needs something it can apply in a
//! frame — a message, a chart, a document to open. Turning the one into the
//! other is the same on every platform up to the very last step: a result
//! that came back whole is written out (the [`ResultWriter`] a streamed one
//! went through while the run was still going), a cancelled run's result is
//! thrown away, a replacement for a document that has since been closed goes
//! to a window of its own. Only how a window is opened, and how a file is
//! saved, differ — a process or a browser tab, a path or a download — and that
//! last step is the [`Deliver`] the app supplies.
//!
//! This lived in the app, as three copies of the same encode-and-hand-over,
//! one per kind of outcome, with the platform differences inside each. It is
//! here because it needs the stack's host and not the window, and so the
//! desktop and the browser share one copy of it.
//!
//! [`ResultWriter`]: super::ResultWriter

use super::{write_result, ResultDest, ResultTiff, StackHost};
use fasttiff_plugin_api::{Animation, HostContext, ImageResult, Outcome, Plot, StreamTarget};

/// What a plugin's run came to, once everything that can be done off the
/// interface thread has been done.
///
/// The writing and the handover happen on the worker precisely because they
/// are not small: done on the interface thread after the result came back,
/// they froze the window with the bar sitting at 100%.
#[derive(Debug)]
pub enum Product {
    Nothing,
    Cancelled,
    Message(String),
    Plot(Box<Plot>),
    /// A window is already opening for it; this is what to call it.
    ///
    /// A derived image is a second thing, not a correction to the first: the
    /// point of a projection or a derivative is to be looked at beside what it
    /// came from, so a result opens in a window of its own unless the plugin
    /// was asked to replace.
    NewWindow(String),
    /// It could not be handed to a window of its own — this is why — so it
    /// comes back to be shown in this one rather than lost.
    Inline {
        result: ResultTiff,
        why: String,
    },
    /// Written where the plugin asked; this is where.
    Saved(String),
    /// Takes the place of the document it was run on, because the plugin asked
    /// for that rather than for a window. Not a failure path: nothing went
    /// wrong, and there is nothing to apologise for in the status line.
    Replace(ResultTiff),
    /// Move the 3D view as the plugin declared.
    Animate(Box<Animation>),
    /// Stop the plugin's own animation.
    StopAnimation,
    /// Have the user draw a rectangle, with this prompt in the status line,
    /// then run the plugin again with it.
    SelectRegion(String),
}

/// How a window of its own took the result.
#[derive(Debug)]
pub enum Handover {
    /// It is opening, under this name.
    Opened(String),
    /// It could not be, for this reason; the result is handed back to be
    /// shown here instead.
    Refused(Box<ResultTiff>, String),
    /// The user stopped it part-way.
    Cancelled,
}

/// The platform's half of delivering a result: the job's progress readout,
/// and the last step, which is different everywhere.
pub trait Deliver {
    /// Start a named phase of the work after the run, its progress starting
    /// again from nothing — "Encoding result", "Opening a new window".
    fn phase(&mut self, label: &str);

    /// How far through the phase, `0.0..=1.0`. `false` asks to stop.
    fn report(&mut self, fraction: f32) -> bool;

    /// Whether the window still shows the document the run started on. A
    /// replacement for one that has since been closed goes to a new window,
    /// as a new document would: replacing what the user opened meanwhile
    /// would throw that away and show a result of something else in its place.
    fn same_document(&self) -> bool;

    /// Show `result` in a window of its own: a new process on the desktop, a
    /// new tab in a browser.
    fn open_in_new_window(&mut self, result: ResultTiff) -> Handover;

    /// Put `result` where the plugin asked — `path` on the desktop, a download
    /// of that name in a browser — and say where it went.
    fn save(&mut self, result: ResultTiff, path: &str) -> Result<String, String>;
}

/// Turn what a run returned into something the interface can apply, writing
/// and handing over its result on the way.
///
/// `host` is the run's own: it holds a result the plugin streamed, the
/// colour tables a result is written with, and where a result is written. Any
/// result it holds that the outcome does not deliver is thrown away, its file
/// with it.
pub fn finish(
    outcome: Outcome,
    host: &mut StackHost,
    to: &mut dyn Deliver,
) -> Result<Product, String> {
    let (result, target) = match outcome {
        Outcome::NewDocument(image) => (
            write(image, host, to, "Encoding result")?,
            StreamTarget::NewDocument,
        ),
        Outcome::ReplaceDocument(image) => (
            write(image, host, to, "Encoding result")?,
            StreamTarget::ReplaceDocument,
        ),
        Outcome::SaveToFile { image, path } => (
            write(image, host, to, "Writing result")?,
            StreamTarget::SaveToFile { path },
        ),
        // Written already, a plane at a time while the plugin ran; finishing
        // it is writing its directory, which is the size of a page.
        Outcome::Streamed { target } => {
            // A plugin is not obliged to notice a stop, and the host discards
            // the result of a run the user stopped — which is what writing a
            // whole result checked for, plane by plane, before it was streamed.
            // Before it is finished: a stopped run's result is not worth its
            // directory, and dropping it unfinished deletes its file.
            if host.cancelled() {
                return Ok(without_result(host, Product::Cancelled));
            }
            let result = host
                .take_result()
                .ok_or("the plugin said it streamed a result, and none arrived")?
                .map_err(|e| format!("{e:#}"))?;
            (Some(result), target)
        }
        Outcome::Nothing => return Ok(without_result(host, Product::Nothing)),
        Outcome::Cancelled => return Ok(without_result(host, Product::Cancelled)),
        Outcome::Message(m) => return Ok(without_result(host, Product::Message(m))),
        Outcome::Plot(p) => return Ok(without_result(host, Product::Plot(p))),
        // Nothing to write: the poses are the whole result, and the work of
        // carrying them out happens frame by frame on the interface thread.
        Outcome::Animate(a) => return Ok(without_result(host, Product::Animate(a))),
        Outcome::StopAnimation => return Ok(without_result(host, Product::StopAnimation)),
        Outcome::SelectRegion { prompt } => {
            return Ok(without_result(host, Product::SelectRegion(prompt)))
        }
    };
    let Some(result) = result else {
        return Ok(Product::Cancelled);
    };
    Ok(match target {
        StreamTarget::ReplaceDocument if to.same_document() => Product::Replace(result),
        StreamTarget::NewDocument | StreamTarget::ReplaceDocument => {
            to.phase("Opening a new window");
            match to.open_in_new_window(result) {
                Handover::Opened(name) => Product::NewWindow(name),
                Handover::Refused(result, why) => Product::Inline {
                    result: *result,
                    why,
                },
                Handover::Cancelled => Product::Cancelled,
            }
        }
        StreamTarget::SaveToFile { path } => {
            to.phase("Writing result");
            Product::Saved(to.save(result, &path)?)
        }
    })
}

/// An outcome that is not an image: whatever the plugin streamed is not what
/// the run ended with, so it goes, and its file with it.
fn without_result(host: &mut StackHost, product: Product) -> Product {
    host.discard_result();
    product
}

/// Write a result that came back whole, each plane let go once it is written,
/// to where the host writes the ones that are streamed. `None` when the user
/// stopped it.
fn write(
    image: Box<ImageResult>,
    host: &mut StackHost,
    to: &mut dyn Deliver,
    label: &str,
) -> Result<Option<ResultTiff>, String> {
    // A plugin that streamed a result and then returned another has its first
    // one thrown away, not left on disk.
    host.discard_result();
    to.phase(label);
    let dest = host.result_dest().cloned().unwrap_or(ResultDest::Memory);
    write_result(*image, None, &host.view().luts, &dest, &mut |f| {
        to.report(f)
    })
    .map_err(|e| format!("{e:#}"))
}

#[cfg(test)]
#[path = "deliver_tests.rs"]
mod tests;
