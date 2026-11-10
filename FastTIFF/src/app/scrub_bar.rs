//! The bottom bar: playback and frame navigation, the collapsible channels
//! and contrast panel, and the status line.

use super::*;

/// What laying the bottom bar out leaves for the rest of the frame, beside the
/// commands its controls queued.
pub(super) struct Bar {
    pub(super) response: egui::InnerResponse<()>,
    /// What the info row drew — see [`PanelLayout::note_info_row`].
    pub(super) drawn_info: Option<String>,
    /// Whether the scrubber is held, which suspends the playback clock.
    pub(super) scrubber_held: bool,
}

impl Bar {
    pub(super) fn height(&self) -> f32 {
        self.response.response.rect.height()
    }
}

impl ViewerApp {
    /// Lay the bottom bar out, queueing what its controls ask for.
    ///
    /// The frame-navigation buttons, the scrubber, the arrow keys and the
    /// contrast sliders edit the document in place, as the bar holds it; the
    /// rest is a [`UiCommand`] applied once the bar is done.
    pub(super) fn scrub_bar(&mut self, ui: &mut egui::Ui, cmds: &Commands) -> Bar {
        // In 3D the arrow keys rotate the volume (handled in the central panel),
        // so the movie's arrow-scrub and wheel-scrub paths must stand down.
        let view_is_volume = self.core.view_mode == ViewMode::Volume;

        let panel_expanded = self.panel.expanded;
        let is_playing = self.core.playback.playing;
        let pseudocolor_on = self.core.apply_pseudocolor;
        // Whether the scrubber is being held this frame; handed back with the
        // bar, for the playback clock, which runs once the bar is done.
        let mut scrubber_held = false;
        let mut playback_fps = self.core.playback.fps;
        let mut decode_mode = self.core.decode_mode;
        let mut large_image_mode = self.core.large_image_mode;
        // Read here, before the stack is borrowed for the controls: what the
        // last sync found the frame to be on this device.
        let over_texture_limit = self.core.over_texture_limit();
        let current_status = self.core.status.clone();
        // Green only for the message it was recorded for; see `good_status`.
        let status_is_good = current_status.is_some() && current_status == self.good_status;

        // The readout is the same whichever slow thing is running: a worker —
        // import, plugin, save, export — or the core opening a file on its own.
        // What differs is only the words and whether there is a fraction to
        // draw, so that is all that is worked out here.
        //
        // The worker comes first because it *is* first: a file is imported and
        // then opened, so while both could be set the worker is the one still
        // happening.
        //
        // The file's name is deliberately not part of this. It was shown beside
        // the label, but it is already in the window title, so it bought a
        // second copy of the same string at the cost of the width the bar now
        // spreads into.
        let working = self.job.as_ref().map(|job| (job.label(), job.fraction()));
        let progress = working.or_else(|| {
            let stage = self.core.load_stage()?;
            Some((stage.label().to_string(), stage.fraction()))
        });
        let load_stage = progress.is_some();
        // Filled in below by the bar itself, if it draws a status line.
        let mut drawn_info: Option<String> = None;
        // Whether the info row offers a stop button: only for work this app is
        // running. The core's own load has no cancel to press.
        #[cfg(not(target_arch = "wasm32"))]
        let can_stop = self.job.is_some();
        #[cfg(target_arch = "wasm32")]
        let can_stop = false;
        // The bottom line of the panel: what is running, and the last thing
        // that happened. One row for both.
        //
        // The progress readout used to be a row of its own above everything
        // else, which pushed the picture up for the length of every load — and
        // a load is exactly when the picture is being looked at. Sharing the
        // status line costs nothing: they are both one line of chrome saying
        // what the program is up to, and they are rarely both interesting at
        // once.
        //
        // Returns what it drew, which is what the window's height bookkeeping
        // compares against — a row that appears has to come out of the window
        // rather than out of the canvas.
        let info_row = |ui: &mut egui::Ui, status: Option<(&str, bool)>| -> Option<String> {
            if progress.is_none() && status.is_none() {
                return None;
            }
            let key = info_key(
                status.map(|(s, _)| s),
                progress.as_ref().map(|(label, _)| label.as_str()),
            );
            ui.separator();
            ui.horizontal(|ui| {
                // The status goes first so the bar can have the rest of the
                // row. An `egui::ProgressBar` with no `desired_width` takes
                // `available_size_before_wrap().x`, so whatever is placed
                // before it decides how wide it ends up — and anything placed
                // after it would be squeezed out of the row entirely.
                if let Some((status, good)) = status {
                    let color = if good { STATUS_DONE } else { STATUS_NOTE };
                    ui.label(RichText::new(status).color(color).small());
                }
                if let Some((label, fraction)) = &progress {
                    // Also before the bar, and for the same reason.
                    if can_stop && ui.small_button("✖").on_hover_text("Stop").clicked() {
                        cmds.push(UiCommand::StopJob);
                    }
                    ui.add(progress_bar(label, *fraction));
                }
            });
            Some(key)
        };
        let response = egui::Panel::bottom("scrub_bar").show_inside(ui, |ui| {
            let Some(loaded) = &mut self.core.document else {
                if !load_stage {
                    ui.label(RichText::new("© 2026 SciWare LLC").weak());
                }
                // Still worth a row with nothing open: this is where the first
                // file's progress is reported, and where a plugin's "open an
                // image first" lands.
                drawn_info = info_row(ui, current_status.as_deref().map(|s| (s, status_is_good)));
                return;
            };
            ui.add_space(4.0);

            ui.horizontal(|ui| {
                let max_frame = loaded.display.dims().frames.saturating_sub(1);
                let has_multiple_frames = loaded.display.dims().frames > 1;
                // In 3D the frame axis is the volume's depth, so play/step/scrub
                // are meaningless unless the stack has a separate time axis
                // (`slices > 1`). Grey them out otherwise.
                let frame_nav_enabled =
                    has_multiple_frames && !(view_is_volume && loaded.display.dims().slices <= 1);

                let toggle_size = egui::vec2(20.0, 20.0);
                let toggle_response = ui
                    .add_sized(toggle_size, egui::Button::new(""))
                    .on_hover_text("Show/hide channel & contrast settings");
                if toggle_response.clicked() {
                    cmds.push(UiCommand::TogglePanel);
                }
                let icon_color = ui.style().interact(&toggle_response).fg_stroke.color;
                let r = toggle_response.rect.shrink(6.0);
                let triangle = if panel_expanded {
                    vec![r.left_bottom(), r.right_bottom(), r.center_top()]
                } else {
                    vec![r.left_top(), r.right_top(), r.center_bottom()]
                };
                ui.painter().add(egui::Shape::convex_polygon(triangle, icon_color, egui::Stroke::NONE));

                // Play/pause looped movie. Painted (triangle / two bars) rather
                // than using glyphs, since the default font may not carry the
                // ▶/⏸ characters.
                ui.add_enabled_ui(frame_nav_enabled, |ui| {
                    let play_resp = ui
                        .add_sized(egui::vec2(22.0, 20.0), egui::Button::new(""))
                        .on_hover_text("Play/pause looped movie");
                    if play_resp.clicked() {
                        cmds.push(UiCommand::TogglePlay);
                    }
                    let color = ui.style().interact(&play_resp).fg_stroke.color;
                    let r = play_resp.rect.shrink(5.0);
                    if is_playing {
                        let bar = r.width() * 0.32;
                        let left = egui::Rect::from_min_max(r.left_top(), egui::pos2(r.left() + bar, r.bottom()));
                        let right = egui::Rect::from_min_max(egui::pos2(r.right() - bar, r.top()), r.right_bottom());
                        ui.painter().rect_filled(left, 0.0, color);
                        ui.painter().rect_filled(right, 0.0, color);
                    } else {
                        let tri = vec![r.left_top(), r.left_bottom(), egui::pos2(r.right(), r.center().y)];
                        ui.painter().add(egui::Shape::convex_polygon(tri, color, egui::Stroke::NONE));
                    }
                });

                ui.add_enabled_ui(frame_nav_enabled, |ui| {
                    if ui.button("|<").on_hover_text("First frame").clicked() {
                        loaded.set_frame(0);
                    }
                    if ui.button("<").on_hover_text("Previous frame (←)").clicked() {
                        loaded.set_frame(loaded.frame().saturating_sub(1));
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button(">|").on_hover_text("Last frame").clicked() {
                            loaded.set_frame(max_frame);
                        }
                        if ui.button(">").on_hover_text("Next frame (→)").clicked() {
                            loaded.set_frame(loaded.frame() + 1);
                        }

                        let remaining = ui.available_width();
                        if has_multiple_frames {
                            ui.spacing_mut().slider_width = remaining.max(40.0);
                            let mut frame = loaded.frame();
                            let scrub = ui.add(
                                egui::Slider::new(&mut frame, 0..=max_frame)
                                    .show_value(false)
                                    .trailing_fill(true),
                            );
                            loaded.set_frame(frame);
                            // Held, not merely dragged: the frame under the
                            // pointer should hold still from the moment it is
                            // grabbed, before any movement, since that is
                            // already the frame being looked at.
                            scrubber_held =
                                scrub.is_pointer_button_down_on() || scrub.dragged();
                        } else {
                            // Single-frame stack: there's nothing to scrub, so draw
                            // a flat, handleless track instead of a slider parked at
                            // zero (the whole row is already disabled above).
                            let (rect, _) = ui.allocate_exact_size(
                                egui::vec2(remaining.max(40.0), 18.0),
                                egui::Sense::hover(),
                            );
                            let y = rect.center().y;
                            let track = egui::Rect::from_min_max(
                                egui::pos2(rect.left(), y - 2.0),
                                egui::pos2(rect.right(), y + 2.0),
                            );
                            ui.painter().rect_filled(track, 2.0, ui.visuals().widgets.inactive.bg_fill);
                        }
                    });
                });
            });

            // Not while a text field has the caret, where the arrows move it: a
            // group's name being edited in the point counter's window, which in
            // a browser shares this viewport's keyboard.
            if !view_is_volume && !ui.ctx().text_edit_focused() {
                ui.input(|i| {
                    // Shift jumps ~5% of the stack at a time (min 1 frame) instead
                    // of 1, matching the Shift+wheel fast-scroll step.
                    let step = if i.modifiers.shift {
                        ((loaded.display.dims().frames as f64 * FAST_SCROLL_RATE).round() as usize).max(1)
                    } else {
                        1
                    };
                    if i.key_pressed(egui::Key::ArrowRight) {
                        loaded.set_frame(loaded.frame() + step);
                    }
                    if i.key_pressed(egui::Key::ArrowLeft) {
                        loaded.set_frame(loaded.frame().saturating_sub(step));
                    }
                });
            }

            if panel_expanded {
                ui.separator();
                // Wrapping, not clipping: this row holds up to six independent
                // control groups, and which of them are present depends on the
                // file. Narrow the window — or open the web build, whose chrome
                // is drawn half again as large — and the tail of the row used to
                // simply vanish past the edge with nothing to indicate it was
                // there. Wrapping spends panel height, which the layout has, to
                // buy back reachability, which it did not.
                ui.horizontal_wrapped(|ui| {
                    // Whether a control group already sits in the row — each
                    // optional group below draws its leading separator only if
                    // so, so hiding a group never leaves an orphaned separator.
                    let mut row_has_items = false;

                    // The channels-vs-time guess (and its override) is
                    // meaningless for RGB, where the "channels" are fixed color
                    // planes — so the dropdown and pseudocolor toggle are hidden
                    // there.
                    if !loaded.display.rgb() {
                        ui.label("Dimension order:");
                        let c = loaded.display.dims().channels;
                        let z = loaded.display.dims().slices;
                        let f = loaded.display.dims().frames;
                        // When the file has a real Z axis (as loaded — see
                        // `has_z_axis`), offer every assignment of the three
                        // counts to the three roles; otherwise just the
                        // channels/time swap (Z passes through untouched).
                        // sort+dedup collapses duplicates when counts are equal
                        // and keeps the list order stable across
                        // reinterpretations.
                        let show_z = loaded.display.has_z_axis;
                        let mut options: Vec<(usize, usize, usize)> = if show_z {
                            vec![(c, z, f), (c, f, z), (z, c, f), (z, f, c), (f, c, z), (f, z, c)]
                        } else {
                            vec![(c, z, f), (f, z, c)]
                        };
                        options.sort_unstable();
                        options.dedup();
                        let dim_label = |oc: usize, oz: usize, of: usize| {
                            if show_z {
                                format!("c: {oc}  z: {oz}  t: {of}")
                            } else if view_is_volume {
                                // Without a separate Z axis, 3D uses the frame
                                // axis as the volume's depth — so what reads as
                                // time in 2D is genuinely Z here.
                                format!("c: {oc}  z: {of}")
                            } else {
                                format!("c: {oc}  t: {of}")
                            }
                        };
                        egui::ComboBox::from_id_salt("dim_override")
                            .selected_text(dim_label(c, z, f))
                            .show_ui(ui, |ui| {
                                for (oc, oz, of) in options {
                                    if ui
                                        .selectable_label((oc, oz, of) == (c, z, f), dim_label(oc, oz, of))
                                        .clicked()
                                    {
                                        cmds.push(UiCommand::SetDimensionOrder(oc, oz, of));
                                    }
                                }
                            });

                        // Optional channel palette — only for multi-channel
                        // grayscale stacks that carry no colors of their own.
                        if pseudocolor_applicable(loaded) {
                            ui.separator();
                            let mut on = pseudocolor_on;
                            if ui
                                .checkbox(&mut on, "Apply pseudocolor")
                                .on_hover_text("Tint channels ch1 = red, ch2 = green, ch3 = blue, …")
                                .changed()
                            {
                                cmds.push(UiCommand::SetPseudocolor(on));
                            }
                        }
                        row_has_items = true;
                    }

                    // Editable playback rate (seeded from metadata `fps=`, else
                    // 30). Only shown when there's a playable time axis: several
                    // frames in 2D; in 3D the frame axis is the volume's depth,
                    // so time only exists for 4D stacks (`slices > 1`) — matches
                    // the play/scrub controls' enable logic above.
                    let fps_playable = loaded.display.dims().frames > 1
                        && !(view_is_volume && loaded.display.dims().slices <= 1);
                    if fps_playable {
                        if row_has_items {
                            ui.separator();
                        }
                        ui.add(
                            egui::DragValue::new(&mut playback_fps)
                                .speed(0.5)
                                .range(0.1..=1000.0)
                                .max_decimals(2)
                                .suffix(" fps"),
                        )
                        .on_hover_text("Playback speed (frames per second)");
                        row_has_items = true;
                    }

                    // LUT selector for a single channel: show it through a
                    // channel color or a perceptual colormap
                    // (magma/plasma/inferno/viridis/turbo). When the file carries
                    // its own LUT the list leads with "Built-in LUT" (the
                    // default). Hidden for RGB, composite, and multi-channel
                    // stacks — there the per-channel colors / pseudocolor toggle
                    // already handle coloring.
                    if gray_lut_applicable(loaded) {
                        if row_has_items {
                            ui.separator();
                        }
                        ui.label("LUT:");
                        let sel = loaded.display.gray_lut_sel;
                        egui::ComboBox::from_id_salt("gray_lut")
                            .selected_text(gray_lut_sel_name(&loaded.display, sel))
                            .show_ui(ui, |ui| {
                                for opt in 0..gray_lut_count(&loaded.display) {
                                    // Tint each entry with its LUT's low (dark) end
                                    // — the color the darkest samples map to. A
                                    // grayscale/black low end (grayscale + the pure
                                    // channel-color ramps) keeps the default text color.
                                    let name = gray_lut_sel_name(&loaded.display, opt);
                                    let text = match tint_color(gray_lut_sel_tint(&loaded.display, opt)) {
                                        Some(c) => RichText::new(name).color(c),
                                        None => RichText::new(name),
                                    };
                                    if ui.selectable_label(opt == sel, text).clicked() {
                                        cmds.push(UiCommand::SetGrayLut(opt));
                                    }
                                }
                            })
                            .response
                            .on_hover_text("Display this channel through a color LUT or colormap");
                        row_has_items = true;
                    }

                    // CPU decode parallelism: Auto threads only when playback
                    // can't keep up; Serial/Threaded force it off/on. Threaded
                    // decode only ever kicks in for compressed frames (parallel
                    // strip decompression) or wide 32-/64-bit frames (parallel
                    // per-pixel rescale/cast). 8- and 16-bit uncompressed frames
                    // decode zero-copy or with an unthreaded copy, so the control
                    // has no effect and is hidden for them.
                    let threadable = loaded.tiff.frames.first().is_some_and(|f| {
                        f.compression != fast_tiff_lib::Compression::None || f.bits_per_sample >= 32
                    });
                    if threadable {
                        if row_has_items {
                            ui.separator();
                        }
                        ui.label("Decode:");
                        egui::ComboBox::from_id_salt("decode_mode")
                            .selected_text(decode_mode.label())
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut decode_mode, DecodeMode::Auto, "Auto")
                                    .on_hover_text("Single-threaded until playback drops frames, then multi-threaded");
                                ui.selectable_value(&mut decode_mode, DecodeMode::Serial, "Serial")
                                    .on_hover_text("Always single-threaded (lowest total CPU)");
                                ui.selectable_value(&mut decode_mode, DecodeMode::Threaded, "Threaded")
                                    .on_hover_text("Always multi-threaded for large frames (spreads across cores)");
                            });
                        row_has_items = true;
                    }

                    // How a frame too large to become a texture is kept on
                    // screen. Only for such frames — for every ordinary image
                    // both settings do exactly the same thing, and an option
                    // that changes nothing is worse than no option. The flag is
                    // set during the GPU sync, since only there is the device's
                    // texture limit known.
                    if over_texture_limit {
                        if row_has_items {
                            ui.separator();
                        }
                        ui.label("Large image:");
                        egui::ComboBox::from_id_salt("large_image_mode")
                            .selected_text(match large_image_mode {
                                LargeImageMode::Preload => "Preload",
                                LargeImageMode::Tiled => "Tiled",
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut large_image_mode, LargeImageMode::Tiled, "Tiled")
                                    .on_hover_text(
                                        "Load only the region on screen, at whatever resolution the zoom \
                                         calls for. Uses far less memory, but every zoom step decodes afresh, \
                                         so zooming costs CPU and can stutter.",
                                    );
                                ui.selectable_value(&mut large_image_mode, LargeImageMode::Preload, "Preload")
                                    .on_hover_text(
                                        "Keep one reduced copy of the whole image in memory, plus full \
                                         resolution for whatever is on screen. Smooth to pan and zoom, and \
                                         less moiré when zoomed out, at the cost of the memory that copy \
                                         occupies.",
                                    );
                            })
                            .response
                            .on_hover_text(
                                "This image is larger than the graphics card can hold as one texture, \
                                 so only part of it is resident at a time.",
                            );
                        row_has_items = true;
                    }

                    // The two pop-up toggles. They flow with the rest of the
                    // row rather than being pinned to its right edge: once the
                    // row can wrap there is no single right edge to pin to, and
                    // a button that jumps to the end of whichever line happens
                    // to be last is harder to find than one that simply follows
                    // the controls.
                    if row_has_items {
                        ui.separator();
                    }
                    if histogram_button(ui).clicked() {
                        cmds.push(UiCommand::ToggleWindow(WindowId::HISTOGRAM));
                    }
                    if ui
                        .button(RichText::new("( i )").size(12.0))
                        .on_hover_text("See metadata")
                        .clicked()
                    {
                        cmds.push(UiCommand::ToggleWindow(WindowId::METADATA));
                    }
                });
                if !loaded.display.rgb() {
                    ui.label(
                        RichText::new(
                            "Channels are guessed automatically (6 or fewer = channels, more = time); \
                             use Dimension order if that guess is wrong for this file.",
                        )
                        .small()
                        .weak(),
                    );
                }

                contrast_controls(ui, loaded, ContrastLayout::Inline, None);
            }
            // The triple-axis note explains that 2D freezes Z at its first
            // slice — but the 3D view *does* use Z (as the volume depth), so
            // showing it there would be wrong. When `triple_axis_warning` is
            // set, the status IS that note (`compute_status` short-circuits on
            // it), so this suppresses exactly the right message.
            let suppressed = view_is_volume && loaded.display.triple_axis_warning;
            drawn_info = info_row(
                ui,
                current_status
                    .as_deref()
                    .filter(|_| !suppressed)
                    .map(|s| (s, status_is_good)),
            );
            ui.add_space(4.0);
        });

        // The settings edited in place above, as the commands that set them.
        if playback_fps != self.core.playback.fps {
            cmds.push(UiCommand::SetFps(playback_fps));
        }
        if decode_mode != self.core.decode_mode {
            cmds.push(UiCommand::SetDecodeMode(decode_mode));
        }
        if large_image_mode != self.core.large_image_mode {
            cmds.push(UiCommand::SetLargeImageMode(large_image_mode));
        }
        Bar {
            response,
            drawn_info,
            scrubber_held,
        }
    }
}
