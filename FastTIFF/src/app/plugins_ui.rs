//! The Plugins menu and the dialog a plugin declares.
//!
//! The plugin never draws: it hands over a list of [`ParamDecl`]s and this
//! renders them. That is what lets a `.dll` compiled by someone else's
//! toolchain have a dialog at all — see `fasttiff_plugin_api::params` for why
//! passing an `&mut egui::Ui` across that boundary is not an option.

use egui::RichText;
use fast_tiff_viewer::plugins::Registry;
use fasttiff_plugin_api::{ParamDecl, ParamKind, ParamValue, Params, PluginInfo};

/// What the menu asked for.
pub(super) enum MenuAction {
    /// Run this plugin, showing its dialog first if it declared one.
    Run(usize),
    /// Open the folder plugins are installed into.
    OpenPluginFolder,
    None,
}

/// The Plugins menu button and its contents.
pub(super) fn plugins_menu(ui: &mut egui::Ui, registry: Option<&Registry>) -> MenuAction {
    // `None` means an import has the registry — see `ViewerApp::plugins`. The
    // button stays, greyed, rather than vanishing: a toolbar that loses an item
    // for the duration of a long read is a toolbar whose other buttons move
    // under the pointer while it is being read.
    let icon = || RichText::new(super::ICON_PLUGINS).size(super::ICON_SIZE);
    let Some(registry) = registry else {
        ui.add_enabled(false, egui::Button::new(icon()))
            .on_disabled_hover_text("Plugins — busy importing a file");
        return MenuAction::None;
    };
    let mut action = MenuAction::None;
    ui.menu_button(icon(), |ui| {
        if registry.is_empty() {
            ui.label(RichText::new("No plugins installed").italics());
        } else {
            for (path, items) in registry.grouped() {
                if path.is_empty() {
                    for (i, info) in items {
                        if entry(ui, info).clicked() {
                            action = MenuAction::Run(i);
                            ui.close();
                        }
                    }
                } else {
                    ui.menu_button(path, |ui| {
                        for (i, info) in items {
                            if entry(ui, info).clicked() {
                                action = MenuAction::Run(i);
                                ui.close();
                            }
                        }
                    });
                }
            }
        }

        // Importers do not appear as menu entries — they run from opening a
        // file — but a user who installed one needs to see that it is there,
        // otherwise a plugin that silently failed to load looks identical to
        // one that is working and simply has not been triggered.
        if !registry.importers().is_empty() {
            ui.separator();
            ui.label(RichText::new("Importable formats:").strong());
            for e in registry.importers() {
                let exts: Vec<String> = e
                    .file_types
                    .iter()
                    .flat_map(|t| t.extensions.iter().map(|x| format!(".{x}")))
                    .collect();
                ui.label(
                    RichText::new(format!("{}  ({})", e.info.name, exts.join(" ")))
                        .weak()
                        .small(),
                )
                .on_hover_text(&e.info.description);
            }
        }

        // Exporters are not menu entries either — they run from the Save-as
        // dialog — and are listed for the same reason importers are: an
        // installed plugin that silently failed to load looks exactly like one
        // that is working and has not been triggered.
        if !registry.exporters().is_empty() {
            ui.separator();
            ui.label(RichText::new("Exportable formats:").strong());
            for e in registry.exporters() {
                let exts: Vec<String> = e
                    .file_types
                    .iter()
                    .flat_map(|t| t.extensions.iter().map(|x| format!(".{x}")))
                    .collect();
                ui.label(
                    RichText::new(format!("{}  ({})", e.info.name, exts.join(" ")))
                        .weak()
                        .small(),
                )
                .on_hover_text(&e.info.description);
            }
        }

        ui.separator();
        if ui
            .button("Open plugin folder…")
            .on_hover_text("Where to put a plugin so FastTIFF finds it")
            .clicked()
        {
            action = MenuAction::OpenPluginFolder;
            ui.close();
        }

        // A plugin the user installed and cannot find is worse than one that
        // says why it will not run, so problems are shown rather than logged.
        if !registry.problems.is_empty() {
            ui.separator();
            ui.label(RichText::new("Problems").color(egui::Color32::from_rgb(220, 120, 60)));
            for p in &registry.problems {
                ui.label(RichText::new(p).small().weak());
            }
        }
    });
    action
}

fn entry(ui: &mut egui::Ui, info: &PluginInfo) -> egui::Response {
    let r = ui.button(&info.name);
    if info.description.is_empty() {
        r
    } else {
        r.on_hover_text(&info.description)
    }
}

/// Draw a plugin's declared dialog. Returns `Some(true)` to run, `Some(false)`
/// to cancel, `None` while it is still open.
pub(super) fn dialog(
    ctx: &egui::Context,
    title: &str,
    decls: &[ParamDecl],
    values: &mut Params,
) -> Option<bool> {
    let mut outcome = None;
    let mut open = true;
    egui::Window::new(title)
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .show(ctx, |ui| {
            // The controls scroll; the buttons below do not. A plugin decides
            // how many controls it wants and some want twenty — the
            // deconvolution dialog is the union of what ImageJ's
            // deconvolution plugins ask for — and on a laptop screen a column
            // that long pushes Run off the bottom of the display, where the
            // window being centred and not resizable leaves no way to reach
            // it. Keeping the buttons outside the scrolled region is the
            // whole fix: however long the dialog, it can always be answered.
            egui::ScrollArea::vertical()
                .max_height(ctx.content_rect().height() * 0.7)
                .auto_shrink([false, true])
                .show(ui, |ui| groups(ui, decls, values));
            ui.separator();
            ui.horizontal(|ui| {
                if ui.button("Run").clicked() {
                    outcome = Some(true);
                }
                if ui.button("Cancel").clicked() {
                    outcome = Some(false);
                }
            });
        });
    // Closing the window with its X is a cancel.
    if !open && outcome.is_none() {
        outcome = Some(false);
    }
    outcome
}

/// Draw the declarations, split into groups at each
/// [`ParamKind::Section`].
///
/// A grid per group rather than one grid with headings inside it, for two
/// reasons. A rule drawn inside a two-column grid is as wide as the column it
/// lands in, which is not what a heading rule is for; and the label column of
/// a group about the objective has no reason to line up with the label column
/// of a group about sampling — letting each group size its own columns is
/// what stops one long label in one group from indenting every control in the
/// dialog.
fn groups(ui: &mut egui::Ui, decls: &[ParamDecl], values: &mut Params) {
    let mut group = 0usize;
    let mut rest = decls;
    // Anything before the first section heading is a group of its own, so a
    // plugin that declares no sections draws exactly as it did before.
    while !rest.is_empty() {
        let head = match &rest[0].kind {
            ParamKind::Section => {
                let d = &rest[0];
                rest = &rest[1..];
                Some(d)
            }
            _ => None,
        };
        let end = rest
            .iter()
            .position(|d| matches!(d.kind, ParamKind::Section))
            .unwrap_or(rest.len());
        let (here, after) = rest.split_at(end);
        rest = after;

        if let Some(d) = head {
            if group > 0 {
                ui.add_space(10.0);
            }
            let r = ui.strong(&d.label);
            if let Some(h) = &d.help {
                r.on_hover_text(h);
            }
            ui.separator();
        }
        if !here.is_empty() {
            egui::Grid::new(("plugin_params", group))
                .num_columns(2)
                .spacing([12.0, 6.0])
                .show(ui, |ui| {
                    for d in here {
                        control(ui, d, values);
                        ui.end_row();
                    }
                });
        }
        group += 1;
    }
}

/// One declared control.
fn control(ui: &mut egui::Ui, d: &ParamDecl, values: &mut Params) {
    match &d.kind {
        ParamKind::Label => {
            ui.label(&d.label);
            ui.label("");
        }
        // `groups` takes the section headings out before this is reached, so
        // one arriving here is a declaration list that was rendered some
        // other way. Drawing it as a heading is still the right answer.
        ParamKind::Section => {
            ui.strong(&d.label);
            ui.label("");
        }
        ParamKind::Int { default, min, max } => {
            label(ui, d);
            let mut v = values.int(&d.key, *default);
            // The declared range is a contract, so the widget cannot leave it.
            let (lo, hi) = (*min.min(max), *max.max(min));
            if ui
                .add(egui::Slider::new(&mut v, lo..=hi).clamping(egui::SliderClamping::Always))
                .changed()
            {
                values.set(d.key.clone(), ParamValue::Int(v));
            }
        }
        ParamKind::Float { default, min, max } => {
            label(ui, d);
            let mut v = values.float(&d.key, *default);
            let (lo, hi) = (min.min(*max), max.max(*min));
            // A slider over a range that spans orders of magnitude is not a
            // control. A regularisation weight declared `0.000001..=1` with a
            // default of `0.001` sits a thousandth of the way along a linear
            // track: every useful value is in the first pixel of it, and the
            // rest of the track is values nobody wants. Laid out
            // logarithmically the same declaration gives even resolution over
            // every decade.
            //
            // Decided from the declaration rather than added to it: a plugin
            // asking for a positive quantity spanning a thousandfold has said
            // everything that is needed, and a `logarithmic` flag would have
            // to cross the C ABI — where `FtParamDecl` is frozen — to be
            // worth anything to the plugins that are not compiled in.
            let decades = lo > 0.0 && hi / lo >= 1000.0;
            let slider = egui::Slider::new(&mut v, lo..=hi)
                .clamping(egui::SliderClamping::Always)
                .logarithmic(decades);
            if ui.add(slider).changed() {
                values.set(d.key.clone(), ParamValue::Float(v));
            }
        }
        ParamKind::Bool { default } => {
            label(ui, d);
            let mut v = values.bool(&d.key, *default);
            if ui.checkbox(&mut v, "").changed() {
                values.set(d.key.clone(), ParamValue::Bool(v));
            }
        }
        ParamKind::Choice { default, options } => {
            label(ui, d);
            let mut sel = values
                .choice(&d.key, *default)
                .min(options.len().saturating_sub(1));
            let shown = options.get(sel).cloned().unwrap_or_default();
            // A choice of one is not a choice. It is still shown, because the
            // value says what is about to happen — a projection dialog offering
            // only "T (frames)" is telling you the stack has no Z to flatten —
            // but it is not offered, so it cannot read as a decision the user
            // failed to make.
            ui.add_enabled_ui(options.len() > 1, |ui| {
                egui::ComboBox::from_id_salt(&d.key)
                    .selected_text(shown)
                    .show_ui(ui, |ui| {
                        for (i, o) in options.iter().enumerate() {
                            if ui.selectable_label(i == sel, o).clicked() {
                                sel = i;
                                values.set(d.key.clone(), ParamValue::Choice(i));
                            }
                        }
                    });
            });
        }
        ParamKind::Text { default } => {
            label(ui, d);
            let mut v = values.text(&d.key, default).to_string();
            if ui.text_edit_singleline(&mut v).changed() {
                values.set(d.key.clone(), ParamValue::Text(v));
            }
        }
        ParamKind::Path { default, save } => {
            label(ui, d);
            ui.horizontal(|ui| {
                let mut v = values.text(&d.key, default).to_string();
                if ui.text_edit_singleline(&mut v).changed() {
                    values.set(d.key.clone(), ParamValue::Path(v.clone()));
                }
                if ui.button("…").clicked() {
                    let picked = if *save {
                        rfd::FileDialog::new().save_file()
                    } else {
                        rfd::FileDialog::new().pick_file()
                    };
                    if let Some(p) = picked {
                        values.set(d.key.clone(), ParamValue::Path(p.display().to_string()));
                    }
                }
            });
        }
    }
}

fn label(ui: &mut egui::Ui, d: &ParamDecl) {
    let r = ui.label(&d.label);
    if let Some(h) = &d.help {
        r.on_hover_text(h);
    }
}
