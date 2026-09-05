//! Stimulus-locked derivative maps, as one magenta/green overlay.
//!
//! # What it computes
//!
//! A two-photon recording is stimulated on a repeating pattern: within each
//! *epoch* there are N *steps* of equal length, and each step either fires a
//! given stimulator or does not. For every step that fires, this finds the
//! response window after it in every epoch, takes the positive part of the
//! temporal derivative over that window, sums it, and averages across epochs.
//! The result is one image per step — a map of where the tissue responded to
//! that stimulus condition.
//!
//! Those images are then shown as **channels of a single document**, the first
//! magenta and the second green, which is the comparison the maps exist for:
//! what responded to both conditions appears white, what responded to only one
//! keeps its colour.
//!
//! # Where the timing comes from
//!
//! Not from the dialog. The trigger time, the frame count and the recording
//! duration are read out of `ImageDescription` (tag 270) — the acquisition's
//! own record, which FastTIFF's OIR importer copies there verbatim. Only lines
//! matching the patterns in [`meta`] are read and everything else in the
//! description is ignored, which matters because that description is a mixture:
//! ImageJ's own `key=value` block, the instrument's export, and whatever a
//! previous tool left behind.
//!
//! Nothing here has to be told when the stimulus started, and nothing can
//! disagree with the file about it.
//!
//! # The epoch count
//!
//! Not a parameter either: it is the largest number of whole epochs that fits
//! between the trigger and the end of the recording, response window included.
//! A recording stopped early therefore yields fewer epochs rather than an error
//! or a set of averages quietly containing a truncated one.

use fasttiff_plugin::api::{
    HostContext, ImageResult, Outcome, ParamDecl, ParamKind, Params, PixelType, Plane, PlaneData,
    Plugin, PluginError, PluginInfo,
};

pub mod filter;
pub mod meta;

/// The Gaussian width used for the derivative, in pixels and frames.
///
/// 2.3 rather than ImageJ's usual 1.0: these recordings are noisy enough that
/// a narrower kernel turns shot noise into spurious rises, and the positive
/// part of the derivative sums noise rather than cancelling it.
const DEFAULT_SIGMA: f64 = 2.3;

/// Magenta and green, in that order.
///
/// The convention this reproduces, and worth keeping rather than reaching for
/// red/green: magenta and green are separable for the most common form of
/// colour blindness, and their overlap is white rather than a muddy yellow.
const OVERLAY_COLORS: [[u8; 3]; 2] = [[255, 0, 255], [0, 255, 0]];

#[derive(Default)]
pub struct Derivatives;

impl Plugin for Derivatives {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("dev.fasttiff.derivatives", "Stimulus Derivatives…")
            .menu_path("Analysis")
            .version(env!("CARGO_PKG_VERSION"))
            .author("FastTIFF")
            .description("Average stimulus-locked derivative maps into a magenta/green overlay.")
    }

    fn params(&self, _host: &dyn HostContext) -> Vec<ParamDecl> {
        vec![
            ParamDecl::new(
                "pattern",
                "Stimulation pattern",
                ParamKind::Text {
                    default: "10,01".into(),
                },
            )
            .help("One row per stimulator, comma-separated; 1 fires, 0 does not."),
            ParamDecl::new(
                "step_duration",
                "Step duration (s)",
                ParamKind::Float {
                    default: 10.0,
                    min: 0.001,
                    max: 3600.0,
                },
            ),
            ParamDecl::new(
                "resp_duration",
                "Response window (s)",
                ParamKind::Float {
                    default: 0.8,
                    min: 0.001,
                    max: 3600.0,
                },
            )
            .help("Must be long enough to contain the response peak."),
            ParamDecl::new(
                "trig_number",
                "Trigger event",
                ParamKind::Int {
                    default: 1,
                    min: 1,
                    max: 1024,
                },
            )
            .help("Which event marker in the file's metadata starts the sequence."),
            ParamDecl::new(
                "start_from_epoch",
                "First epoch",
                ParamKind::Int {
                    default: 1,
                    min: 0,
                    max: 100_000,
                },
            )
            .help("Epochs before this are ignored."),
            ParamDecl::new(
                "frame_lag",
                "Frame lag",
                ParamKind::Int {
                    default: -1,
                    min: -1000,
                    max: 1000,
                },
            )
            .help("Shifts the response window, to line the derivative up with the stimulus."),
            ParamDecl::new(
                "sync_coef",
                "Clock correction",
                ParamKind::Float {
                    default: -0.003,
                    min: -0.5,
                    max: 0.5,
                },
            )
            .help("Fractional correction to the sampling interval, for stimulator/scanner drift."),
            ParamDecl::new(
                "sigma",
                "Gaussian sigma",
                ParamKind::Float {
                    default: DEFAULT_SIGMA,
                    min: 0.1,
                    max: 20.0,
                },
            ),
            ParamDecl::new(
                "channel",
                "Source channel",
                ParamKind::Int {
                    default: 0,
                    min: 0,
                    max: 64,
                },
            ),
        ]
    }

    fn run(&mut self, host: &mut dyn HostContext, params: &Params) -> Result<Outcome, PluginError> {
        let plan = Plan::build(host, params)?;
        host.log(&plan.summary());

        let info = host.image();
        let mut channels = Vec::with_capacity(plan.steps.len());
        let total = plan.steps.len() * plan.epochs;
        let mut done = 0usize;

        for &step in &plan.steps {
            // One average per firing step, over every epoch that fits.
            let mut acc = vec![0f32; info.plane_len()];
            for epoch in 0..plan.epochs {
                if !host.progress(done as f32 / total.max(1) as f32) {
                    return Ok(Outcome::Cancelled);
                }
                let (start, end) = plan.window(step, epoch);
                let planes = plan.read_window(host, start, end)?;
                let img = filter::positive_derivative_sum(
                    &planes,
                    info.width as usize,
                    info.height as usize,
                    plan.sigma,
                );
                for (a, v) in acc.iter_mut().zip(&img) {
                    *a += v;
                }
                done += 1;
            }
            let inv = 1.0 / plan.epochs as f32;
            for a in &mut acc {
                *a *= inv;
            }
            channels.push(PlaneData::F32(acc));
        }

        let name = format!(
            "{}-derivatives-{}ep",
            host.stack_info().name.trim_end_matches(".tif"),
            plan.epochs
        );
        Ok(Outcome::NewDocument(Box::new(ImageResult {
            width: info.width,
            height: info.height,
            channels: channels.len(),
            slices: 1,
            frames: 1,
            pixel_type: PixelType::F32,
            planes: channels,
            // Magenta first, green second, then the host's defaults for a
            // pattern with more than two firing steps.
            channel_colors: OVERLAY_COLORS
                .iter()
                .copied()
                .take(plan.steps.len())
                .collect(),
            name,
        })))
    }
}

/// Everything resolved before a single pixel is read.
///
/// Built in one place so the arithmetic can be tested without a stack, and so
/// that a recording that cannot support the requested analysis fails before it
/// has spent a minute filtering.
struct Plan {
    /// Indices of the steps that fire, in pattern order.
    steps: Vec<usize>,
    /// How many whole epochs fit after the trigger.
    epochs: usize,
    start_from_epoch: usize,
    trigger_s: f64,
    step_duration: f64,
    resp_duration: f64,
    epoch_duration: f64,
    /// Frames per second after the clock correction.
    fps: f64,
    /// Duration after the clock correction.
    duration_s: f64,
    frame_lag: i64,
    sigma: f64,
    channel: usize,
    frames: usize,
}

impl Plan {
    fn build(host: &dyn HostContext, params: &Params) -> Result<Plan, PluginError> {
        let info = host.image();
        let description = host.stack_info().description.as_deref().unwrap_or("");
        let timing = meta::Timing::parse(description);

        let spf = timing.seconds_per_frame().ok_or_else(|| {
            PluginError::unsupported(
                "this file's metadata does not state the recording's duration, so the \
                 stimulus timing cannot be worked out. Tag 270 needs the acquisition's \
                 own `T Dimension` line — FastTIFF's OIR importer puts it there.",
            )
        })?;
        let trig_number = params.int("trig_number", 1).max(1) as usize;
        let (event_name, trigger_s) =
            timing.events.get(trig_number - 1).cloned().ok_or_else(|| {
                PluginError::unsupported(format!(
                    "this file's metadata lists {} event marker(s), so there is no \
                     trigger {trig_number} to start from",
                    timing.events.len()
                ))
            })?;
        let _ = event_name;

        let steps = parse_pattern(params.text("pattern", "10,01"))?;
        let n_steps = steps.len();
        let firing: Vec<usize> = (0..n_steps).filter(|i| steps[*i]).collect();
        if firing.is_empty() {
            return Err(PluginError::failed(
                "the stimulation pattern never fires, so there is nothing to average",
            ));
        }

        // The clock correction stretches the frame clock relative to the
        // stimulator's, which is what keeps late epochs from drifting off their
        // response windows in a long recording.
        let sync = params.float("sync_coef", -0.003);
        let spf_adj = spf * (1.0 + sync);
        if !(spf_adj.is_finite() && spf_adj > 0.0) {
            return Err(PluginError::failed(
                "the clock correction leaves a sampling interval of zero or less",
            ));
        }
        let duration_s = timing.duration_s.unwrap_or(0.0) * (1.0 + sync);

        let step_duration = params.float("step_duration", 10.0);
        let resp_duration = params.float("resp_duration", 0.8);
        let epoch_duration = step_duration * n_steps as f64;
        let start_from_epoch = params.int("start_from_epoch", 1).max(0) as usize;

        let mut plan = Plan {
            steps: firing,
            epochs: 0,
            start_from_epoch,
            trigger_s,
            step_duration,
            resp_duration,
            epoch_duration,
            fps: 1.0 / spf_adj,
            duration_s,
            frame_lag: params.int("frame_lag", -1),
            sigma: params.float("sigma", DEFAULT_SIGMA),
            channel: params.int("channel", 0).max(0) as usize,
            frames: info.frames.max(1),
        };
        plan.epochs = plan.fit_epochs();

        if plan.epochs == 0 {
            return Err(PluginError::unsupported(format!(
                "no whole epoch fits between the trigger at {trigger_s:.3} s and the end \
                 of the {duration_s:.3} s recording: one epoch needs {:.3} s plus a \
                 {resp_duration:.3} s response window",
                plan.epoch_duration
            )));
        }
        // The derivative needs at least two frames to be a derivative at all.
        let (start, end) = plan.window(plan.steps[0], 0);
        if end - start < 2 {
            return Err(PluginError::unsupported(format!(
                "a {resp_duration:.3} s response window is {} frame(s) at this \
                 sampling rate ({:.3} s/frame); it needs at least 2 to differentiate",
                end - start,
                spf_adj
            )));
        }
        Ok(plan)
    }

    /// How many whole epochs fit after the trigger, response window included.
    ///
    /// The last step of the last epoch is what has to fit, not the epoch's
    /// start — an epoch whose final response runs off the end of the recording
    /// would otherwise be averaged in against a window of reflected edge
    /// frames.
    fn fit_epochs(&self) -> usize {
        let last_shift = self.step_duration * self.steps.last().copied().unwrap_or(0) as f64;
        let mut n = 0usize;
        while n < 1_000_000 {
            let epoch = (self.start_from_epoch + n) as f64;
            let end =
                self.trigger_s + epoch * self.epoch_duration + last_shift + self.resp_duration;
            if end > self.duration_s {
                break;
            }
            // The frame the window ends on must exist too, after the lag.
            if self.sec_to_frame(end) + self.frame_lag >= self.frames as i64 {
                break;
            }
            n += 1;
        }
        n
    }

    /// The frame a timestamp falls on, floored — the analysis's own convention.
    fn sec_to_frame(&self, t: f64) -> i64 {
        if t > self.duration_s {
            self.frames as i64
        } else {
            (t * self.fps).floor() as i64
        }
    }

    /// The `[start, end)` frame window for one step of one epoch.
    fn window(&self, step: usize, epoch: usize) -> (usize, usize) {
        let at = self.trigger_s
            + (self.start_from_epoch + epoch) as f64 * self.epoch_duration
            + self.step_duration * step as f64;
        let start = self.sec_to_frame(at) + self.frame_lag;
        let end = self.sec_to_frame(at + self.resp_duration) + self.frame_lag;
        let start = start.clamp(0, self.frames as i64) as usize;
        let end = end.clamp(0, self.frames as i64) as usize;
        (start.min(end), end)
    }

    fn read_window(
        &self,
        host: &mut dyn HostContext,
        start: usize,
        end: usize,
    ) -> Result<Vec<Vec<f32>>, PluginError> {
        let mut planes = Vec::with_capacity(end - start);
        let mut buf = Vec::new();
        for t in start..end {
            host.read_plane_f32(Plane::new(self.channel, 0, t), &mut buf)?;
            planes.push(std::mem::take(&mut buf));
        }
        Ok(planes)
    }

    fn summary(&self) -> String {
        let (s, e) = self.window(self.steps[0], 0);
        format!(
            "trigger {:.3} s, {} firing step(s) of {:.1} s, {} epoch(s) from #{}; \
             first window frames {s}..{e}",
            self.trigger_s,
            self.steps.len(),
            self.step_duration,
            self.epochs,
            self.start_from_epoch
        )
    }
}

/// `"10,01"` → which steps fire at all.
///
/// Rows are stimulators and columns are steps, so a step fires when *any* row
/// has a 1 in that column. The distinction between "stimulator A", "B" and
/// "both" is what the row layout records, and it survives into the channel
/// order: the columns are taken left to right, so the first firing step is the
/// magenta channel and the second the green one.
fn parse_pattern(text: &str) -> Result<Vec<bool>, PluginError> {
    let rows: Vec<&str> = text
        .split([',', ';', '/'])
        .map(str::trim)
        .filter(|r| !r.is_empty())
        .collect();
    if rows.is_empty() {
        return Err(PluginError::failed("the stimulation pattern is empty"));
    }
    let width = rows[0].chars().count();
    if width == 0 || width > 1024 {
        return Err(PluginError::failed(
            "each row of the stimulation pattern must have between 1 and 1024 steps",
        ));
    }
    let mut fires = vec![false; width];
    for row in &rows {
        if row.chars().count() != width {
            return Err(PluginError::failed(format!(
                "the stimulation pattern's rows are different lengths ({width} and {})",
                row.chars().count()
            )));
        }
        for (i, c) in row.chars().enumerate() {
            match c {
                '1' => fires[i] = true,
                '0' => {}
                other => {
                    return Err(PluginError::failed(format!(
                        "the stimulation pattern may only contain 0 and 1, not {other:?}"
                    )))
                }
            }
        }
    }
    Ok(fires)
}

fasttiff_plugin::export_plugin! { plugins: [Derivatives] }

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
