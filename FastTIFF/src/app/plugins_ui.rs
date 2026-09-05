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
pub(super) fn plugins_menu(ui: &mut egui::Ui, registry: &Registry) -> MenuAction {
    let mut action = MenuAction::None;
    ui.menu_button("Plugins", |ui| {
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
            ui.label(RichText::new("File formats").strong());
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
            egui::Grid::new("plugin_params")
                .num_columns(2)
                .spacing([12.0, 6.0])
                .show(ui, |ui| {
                    for d in decls {
                        control(ui, d, values);
                        ui.end_row();
                    }
                });
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

/// One declared control.
fn control(ui: &mut egui::Ui, d: &ParamDecl, values: &mut Params) {
    match &d.kind {
        ParamKind::Label => {
            ui.label(&d.label);
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
            if ui
                .add(egui::Slider::new(&mut v, lo..=hi).clamping(egui::SliderClamping::Always))
                .changed()
            {
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
