//! The toolbar: open, save, copy, the Plugins menu, the 2D/3D switch, and
//! what the open file is.

use super::*;

impl ViewerApp {
    /// Lay the toolbar out, queueing what is clicked.
    ///
    /// Only reads the app: everything a click asks for is a [`UiCommand`],
    /// applied once the panel is done ([`Phase::Toolbar`] for most; the
    /// 2D/3D switch and the settings toggle at their own phases).
    pub(super) fn toolbar(&self, ui: &mut egui::Ui, cmds: &Commands) -> egui::InnerResponse<()> {
        let current_view_mode = self.core.view_mode;
        let can_show_volume = self.core.can_show_volume();
        egui::Panel::top("toolbar").show_inside(ui, |ui| {
            // Wrapped, as the bottom bar's controls are (`scrub_bar`): a
            // window as narrow as a small picture (a 320-pixel image opens a
            // 320-point window), or a phone's page, has less room than the
            // row needs, and its tail (the frame counter, the time) ran on
            // past the right edge, out of sight. Each item moves down whole,
            // a label included, rather than breaking mid-text. On the desktop
            // the window is sized around the rows the toolbar wraps to at
            // the window's width (see `manage_window`), so the picture keeps
            // its size.
            ui.horizontal_wrapped(|ui| {
                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                if ui
                    .button(RichText::new(ICON_OPEN).size(ICON_SIZE))
                    .on_hover_text("Open a file…")
                    .clicked()
                {
                    cmds.push(UiCommand::Open);
                }
                // Save and Copy both act on the open image, so with nothing
                // open they are *hidden* rather than greyed out. A disabled
                // control is a promise that some reachable state will enable
                // it; "open a file first" is not that state, it is the whole
                // program not having started yet, and two dead buttons beside
                // Open say nothing Open does not already say.
                //
                // Busy is the opposite case and stays greyed: there *is* an
                // image, the button *will* work, and the only thing missing is
                // the few seconds a write or an import is taking. That is a
                // promise worth making, so the disabled hover text names it.
                if self.core.document().is_some() {
                    // Named for what each platform actually does: see
                    // `Target::save_label`.
                    let save_label = TARGET.save_label("as TIFF");
                    if ui
                        .add_enabled(
                            self.job.is_none(),
                            egui::Button::new(RichText::new(ICON_SAVE).size(ICON_SIZE)),
                        )
                        .on_hover_text(save_label.as_str())
                        .on_disabled_hover_text(format!("{save_label} — busy"))
                        .clicked()
                    {
                        cmds.push(UiCommand::Save);
                    }
                    // What a copy *is* differs by view mode, so the tooltip
                    // says which one the user will get rather than a generic
                    // "copy": in 2-D the frame at its own size, in 3-D the view
                    // as it is framed on screen. See `copy_view`.
                    let copy_label = if self.core.view_mode == ViewMode::Volume {
                        "Copy the 3D view as an image"
                    } else {
                        "Copy this frame as an image"
                    };
                    if ui
                        .add_enabled(
                            self.job.is_none(),
                            egui::Button::new(RichText::new(ICON_COPY).size(ICON_SIZE)),
                        )
                        .on_hover_text(format!("{copy_label}  (Ctrl/Cmd+C)"))
                        .on_disabled_hover_text(format!("{copy_label} — busy"))
                        .clicked()
                    {
                        cmds.push(UiCommand::Copy(Via::Button));
                    }
                }
                {
                    let available = self.plugins.as_ref().filter(|_| self.job.is_none());
                    let has_stack = self.core.document().is_some();
                    let folder = TARGET.has_plugin_folder();
                    match plugins_ui::plugins_menu(ui, available, has_stack, folder) {
                        plugins_ui::MenuAction::Export(i) => cmds.push(UiCommand::Export(i)),
                        plugins_ui::MenuAction::Run(i) => cmds.push(UiCommand::StartPlugin(i)),
                        plugins_ui::MenuAction::PointCounter => {
                            cmds.push(UiCommand::OpenPointCounter)
                        }
                        // Offered only where there is a folder, so on the web
                        // the variant is simply never sent.
                        plugins_ui::MenuAction::OpenPluginFolder => {
                            cmds.push(UiCommand::OpenPluginFolder)
                        }
                        plugins_ui::MenuAction::None => {}
                    }
                }
                // 2D/3D switch, right next to Open, and the render-settings
                // button beside it. Both need a volume to act on, which needs at
                // least two frames to stack into one.
                //
                // Hidden rather than greyed out when there is none. A disabled
                // control is a promise that it will work under some condition
                // the user can reach, and offers a tooltip to say which; these
                // cannot become available for the open file no matter what is
                // clicked, so a permanently dead 2D/3D pair next to Open reads
                // as breakage. The toolbar is also short enough that the gap is
                // no loss. The file-info group below supplies the trailing
                // separator, so this block owns only its own leading ones.
                if can_show_volume {
                    ui.separator();
                    if ui
                        .selectable_label(current_view_mode == ViewMode::Movie, "2D")
                        .on_hover_text("Movie (2D) view")
                        .clicked()
                    {
                        cmds.push(UiCommand::SetViewMode(ViewMode::Movie));
                    }
                    if ui
                        .selectable_label(current_view_mode == ViewMode::Volume, "3D")
                        .on_hover_text("Volume (3D) view — drag to rotate, scroll to zoom")
                        .clicked()
                    {
                        cmds.push(UiCommand::SetViewMode(ViewMode::Volume));
                    }
                    ui.separator();
                    if ui
                        .button(RichText::new(ICON_SETTINGS).size(ICON_SIZE))
                        .on_hover_text("3D render settings")
                        .clicked()
                    {
                        cmds.push(UiCommand::ToggleWindow(WindowId::RENDER_SETTINGS));
                    }
                }
                if self.core.document().is_none() {
                    // Nothing open yet: show the version + active render backend
                    // in the space the file info will later occupy.
                    separated_label(
                        ui,
                        RichText::new(format!(
                            "FastTIFF v{}, Renderer: {}",
                            env!("CARGO_PKG_VERSION"),
                            render::BACKEND
                        )),
                    );
                }
                if let Some(loaded) = self.core.document() {
                    let meta = &loaded.tiff().meta;
                    // What the viewer is actually showing, which is not the
                    // file's raw claim whenever an axis was reclassified.
                    let dims = loaded.display().dims();
                    // Reflect a toggle click made earlier in this same toolbar
                    // (the 2D/3D buttons run before this), so the layout updates
                    // on the click frame rather than one frame later.
                    let in_3d = cmds.view_mode().unwrap_or(current_view_mode) == ViewMode::Volume;
                    // In 3D the frame axis becomes the volume's depth, so the
                    // frame counter/time are only meaningful when the stack also
                    // has a separate time axis (the channels+Z+time case, where
                    // `slices > 1`); otherwise hide them.
                    let hide_frame_info = in_3d && dims.slices <= 1;

                    // Zoom is a 2D-only concept — hidden entirely in 3D.
                    if !in_3d {
                        // Up to 2 decimals, trailing zeros trimmed: 3.1%, 33.3%,
                        // 100%, 3200% — so the fractional small zooms read correctly.
                        //
                        // The level being glided *to*, not the value part-way
                        // there: the number is what the wheel selected, and
                        // running it through the intermediate frames turns a
                        // readout into a blur nobody can read.
                        let pct = format!("{:.2}", self.view.model.zoom_settled() * 100.0);
                        let pct = pct.trim_end_matches('0').trim_end_matches('.');
                        separated_label(ui, RichText::new(format!("{pct}%")).monospace())
                            .on_hover_text("Zoom (Ctrl+scroll to change)");
                    }
                    let channels_desc = match loaded.display().layout() {
                        ChannelLayout::Cmyk => {
                            // Say what was read *and* what is shown: the file holds
                            // four ink plates, the viewer shows three converted
                            // components, and a printing user will want to know the
                            // K plate is folded in rather than dropped.
                            //
                            // ASCII arrow on purpose: the bundled font has
                            // no U+2192, so a real arrow renders as a tofu
                            // box. The multiplication sign already on this
                            // status line IS in the font, which makes that
                            // gap easy to assume away.
                            "CMYK->RGB".to_string()
                        }
                        ChannelLayout::Rgb => "RGB".to_string(),
                        ChannelLayout::Intensity | ChannelLayout::Palette => {
                            format!("{} channel(s)", dims.channels)
                        }
                    };
                    let bits = loaded
                        .tiff()
                        .frames
                        .first()
                        .map(|f| f.bits_per_sample)
                        .unwrap_or(0);
                    let (px_w, px_h) = loaded
                        .tiff()
                        .frames
                        .first()
                        .map_or((0, 0), |f| (f.width, f.height));
                    separated_label(ui, format!("{px_w}×{px_h} px, {bits}-bit, {channels_desc}"));
                    // The same frame in real units, when the file says how big a
                    // pixel is. Next to the pixel count rather than replacing it:
                    // the two answer different questions — how much data there is
                    // against how much specimen it covers — and a micrograph is
                    // routinely discussed in both.
                    if let Some(physical) = physical_size(
                        meta.pixel_width,
                        meta.pixel_height,
                        meta.unit.as_deref(),
                        px_w,
                        px_h,
                    ) {
                        separated_label(ui, physical).on_hover_text(
                            "Physical size, from the pixel calibration in the file's metadata \
                             (OME PhysicalSize, or the TIFF resolution tags).",
                        );
                    }

                    // Say so when the picture on screen is not the picture in
                    // the file. This only appears for frames past the GPU
                    // texture limit, which is rare and enormous — but silently
                    // showing a reduced image as if it were the data would be
                    // the kind of wrong a viewer must never be.
                    let gpu_stride = self.core.gpu_stride();
                    if gpu_stride > 1 {
                        separated_label(
                            ui,
                            RichText::new(format!("1/{gpu_stride} scale"))
                                .color(Color32::from_rgb(230, 170, 60)),
                        )
                        .on_hover_text(subsampled_note(gpu_stride));
                    }

                    if !hide_frame_info {
                        let frame_digits = dims.frames.to_string().len();
                        separated_label(
                            ui,
                            RichText::new(format!(
                                "Frame {:>frame_digits$} / {}",
                                loaded.frame() + 1,
                                dims.frames
                            ))
                            .monospace(),
                        );
                        if let Some(interval) = meta.frame_interval_s {
                            let max_time = dims.frames.saturating_sub(1) as f64 * interval;
                            let time_width = format!("{max_time:.3}").len();
                            let current_time = loaded.frame() as f64 * interval;
                            separated_label(
                                ui,
                                RichText::new(format!("t = {current_time:>time_width$.3}s"))
                                    .monospace(),
                            );
                        }
                    }
                }
            });
        })
    }
}

/// How wide a toolbar separator is, in points: egui's own default, given
/// explicitly so that [`separated_label`] measures what it draws.
const SEPARATOR_SPACING: f32 = 6.0;

/// `text` as a label in the wrapping toolbar row, after a separator that is
/// left out where the row breaks before the label, so that no row ends on a
/// separator or starts with one. The label is laid out first, for its width;
/// where it and the separator do not both fit, the row is ended there and
/// the label starts the next.
///
/// Only the labels, which run from the zoom readout on: the few controls
/// ahead of them fit on the first row of any window the desktop allows
/// (`MIN_WINDOW`, 256 points), so their separators do not meet a break
/// there.
fn separated_label(ui: &mut egui::Ui, text: impl Into<egui::WidgetText>) -> egui::Response {
    let galley = text.into().into_galley(
        ui,
        Some(egui::TextWrapMode::Extend),
        f32::INFINITY,
        egui::FontSelection::Default,
    );
    // The separator, the gap after it, and half a point for the layout's
    // rounding: a break where the pair would just have fitted costs nothing,
    // and a separator left at a row's end is what this is for.
    let separator = SEPARATOR_SPACING + ui.spacing().item_spacing.x + 0.5;
    if ui.available_size_before_wrap().x >= separator + galley.size().x {
        ui.add(egui::Separator::default().spacing(SEPARATOR_SPACING));
    } else {
        ui.end_row();
    }
    ui.label(galley)
}

/// What the "1/N scale" badge says on hover, for a frame shown at one pixel
/// in `stride` on each axis.
fn subsampled_note(stride: u32) -> String {
    format!(
        "This frame is larger than one GPU texture can hold, so it is shown \
         subsampled {stride}x on each axis. Pixel values and measurements read \
         from it are sampled, not exact."
    )
}

#[cfg(test)]
#[path = "toolbar_tests.rs"]
mod tests;
