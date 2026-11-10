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
            ui.horizontal(|ui| {
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
                if self.core.document.is_some() {
                    // Named for what each platform actually does. The desktop
                    // asks where to put the file, so its label ends in an
                    // ellipsis; the browser cannot ask and hands the file to the
                    // download manager instead, so promising a dialog there
                    // would be a lie.
                    let save_label = if cfg!(target_arch = "wasm32") {
                        "Download as TIFF"
                    } else {
                        "Save as TIFF…"
                    };
                    if ui
                        .add_enabled(
                            self.job.is_none(),
                            egui::Button::new(RichText::new(ICON_SAVE).size(ICON_SIZE)),
                        )
                        .on_hover_text(save_label)
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
                    let has_stack = self.core.document.is_some();
                    match plugins_ui::plugins_menu(ui, available, has_stack) {
                        plugins_ui::MenuAction::Run(i) => cmds.push(UiCommand::StartPlugin(i)),
                        plugins_ui::MenuAction::PointCounter => cmds.push(UiCommand::OpenPointCounter),
                        // The entry that produces this is itself desktop-only,
                        // so on the web the variant is simply never sent.
                        #[cfg(not(target_arch = "wasm32"))]
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
                if self.core.document.is_none() {
                    // Nothing open yet: show the version + active render backend
                    // in the space the file info will later occupy.
                    ui.separator();
                    ui.label(
                        RichText::new(format!(
                            "FastTIFF v{}, Renderer: {}",
                            env!("CARGO_PKG_VERSION"),
                            render::BACKEND
                    )));

                }
                if let Some(loaded) = &self.core.document {
                    let meta = &loaded.tiff.meta;
                    // What the viewer is actually showing, which is not the
                    // file's raw claim whenever an axis was reclassified.
                    let dims = loaded.display.dims();
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
                        ui.separator();
                        // Up to 2 decimals, trailing zeros trimmed: 3.1%, 33.3%,
                        // 100%, 3200% — so the fractional small zooms read correctly.
                        //
                        // The level being glided *to*, not the value part-way
                        // there: the number is what the wheel selected, and
                        // running it through the intermediate frames turns a
                        // readout into a blur nobody can read.
                        let pct = format!("{:.2}", self.view.zoom_settled() * 100.0);
                        let pct = pct.trim_end_matches('0').trim_end_matches('.');
                        ui.label(RichText::new(format!("{pct}%")).monospace())
                            .on_hover_text("Zoom (Ctrl+scroll to change)");
                    }
                    ui.separator();
                    let channels_desc = if loaded.display.cmyk() {
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
                    } else if loaded.display.rgb() {
                        "RGB".to_string()
                    } else {
                        format!("{} channel(s)", dims.channels)
                    };
                    let bits = loaded.tiff.frames.first().map(|f| f.bits_per_sample).unwrap_or(0);
                    let (px_w, px_h) = loaded
                        .tiff
                        .frames
                        .first()
                        .map_or((0, 0), |f| (f.width, f.height));
                    ui.label(format!("{px_w}×{px_h} px, {bits}-bit, {channels_desc}"));
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
                        ui.separator();
                        ui.label(physical).on_hover_text(
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
                        ui.separator();
                        ui.label(
                            RichText::new(format!("1/{gpu_stride} scale"))
                                .color(Color32::from_rgb(230, 170, 60)),
                        )
                        .on_hover_text(format!(
                            "This frame is larger than one GPU texture can hold, so it is                              shown subsampled {}x on each axis. Pixel values and measurements                              read from it are sampled, not exact.",
                            gpu_stride
                        ));
                    }

                    if !hide_frame_info {
                        ui.separator();
                        let frame_digits = dims.frames.to_string().len();
                        ui.label(
                            RichText::new(format!("Frame {:>frame_digits$} / {}", loaded.frame() + 1, dims.frames))
                                .monospace(),
                        );
                        if let Some(interval) = meta.frame_interval_s {
                            ui.separator();
                            let max_time = dims.frames.saturating_sub(1) as f64 * interval;
                            let time_width = format!("{max_time:.3}").len();
                            let current_time = loaded.frame() as f64 * interval;
                            ui.label(RichText::new(format!("t = {current_time:>time_width$.3}s")).monospace());
                        }
                    }
                }
            });
        })
    }
}
