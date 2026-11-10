//! The file-metadata pop-up, opened from the bottom bar's "( i )".

use super::*;

use egui::RichText;
use std::sync::Arc;

impl WindowId {
    pub(in crate::app) const METADATA: WindowId = WindowId("file_metadata");
}

/// The window. It reads the document and keeps nothing of its own.
pub(super) struct MetadataWindow;

impl AppWindow for MetadataWindow {
    fn id(&self) -> WindowId {
        WindowId::METADATA
    }

    fn show(&mut self, ctx: &egui::Context, open: &mut bool, host: &mut Host<'_>) {
        match &host.core.document {
            Some(loaded) => metadata_window(ctx, open, loaded),
            None => *open = false,
        }
    }

    /// Up on the desktop. Each pop-up is its own window there, and what it
    /// shows is "the open file" rather than one particular file — it is
    /// rebuilt from the live document each frame, so leaving it open simply
    /// repoints it at the new one. That is the behaviour worth having when
    /// comparing files: park the histogram on a second monitor and step
    /// through a folder, rather than reopening it after every load.
    ///
    /// In the browser the pop-ups are windows *inside* the canvas, laid over
    /// the image and sized for the channel count that is going away, so there
    /// they close.
    fn outlives_document(&self, _core: &Viewer, web: bool) -> bool {
        !web
    }
}

fn metadata_window(ctx: &egui::Context, open: &mut bool, loaded: &Document) {
    let tiff = &loaded.tiff;
    let spec = dialog::Dialog {
        id: WindowId::METADATA.0,
        title: "File metadata",
        size: egui::vec2(360.0, 600.0),
        scroll: true,
        fit_height: false,
        resizable: true,
        // Read *about* the picture rather than used to change it, so it
        // stays an ordinary window a user can put behind the app.
        always_on_top: false,
    };
    dialog::show(ctx, spec, open, |ui| {
        fn kv(ui: &mut egui::Ui, k: &str, v: impl Into<String>) {
            ui.label(RichText::new(k).strong());
            ui.label(v.into());
            ui.end_row();
        }

        ui.heading("File");
        egui::Grid::new("meta_file")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                kv(ui, "Size", human_bytes(tiff.data.len() as u64));
                let container = match tiff.flavor {
                    fast_tiff_lib::TiffFlavor::Classic => "classic TIFF",
                    fast_tiff_lib::TiffFlavor::Big => "BigTIFF",
                };
                kv(ui, "Container", container);
                let order = match tiff.byte_order {
                    fast_tiff_lib::ByteOrder::Little => "little-endian (II)",
                    fast_tiff_lib::ByteOrder::Big => "big-endian (MM)",
                };
                kv(ui, "Byte order", order);
                kv(ui, "Planes (IFDs)", tiff.frames.len().to_string());
                let meta_format = match tiff.meta.source_format {
                    fast_tiff_lib::MetadataFormat::ImageJ => "ImageJ",
                    fast_tiff_lib::MetadataFormat::Ome => "OME-XML",
                    _ => "—",
                };
                kv(ui, "Metadata", meta_format);
            });

        if let Some(f) = tiff.frames.first() {
            ui.add_space(12.0);
            ui.heading("Frame format");
            egui::Grid::new("meta_frame")
                .num_columns(2)
                .striped(true)
                .show(ui, |ui| {
                    kv(ui, "Dimensions", format!("{} x {} px", f.width, f.height));
                    let format = match f.sample_format {
                        fast_tiff_lib::SampleFormat::UnsignedInt => "unsigned integer",
                        fast_tiff_lib::SampleFormat::SignedInt => "signed integer",
                        fast_tiff_lib::SampleFormat::Float => "IEEE float",
                    };
                    kv(
                        ui,
                        "Pixel type",
                        format!("{}-bit {format}", f.bits_per_sample),
                    );
                    kv(ui, "Samples/pixel", {
                        let model = if f.is_rgb() {
                            Some("RGB")
                        } else if f.is_cmyk() {
                            Some("CMYK")
                        } else {
                            None
                        };
                        match (model, f.is_planar()) {
                            (Some(m), false) => format!("{} (chunky {m})", f.samples_per_pixel),
                            (Some(m), true) => format!("{} (planar {m})", f.samples_per_pixel),
                            (None, true) => format!("{} (planar)", f.samples_per_pixel),
                            (None, false) => f.samples_per_pixel.to_string(),
                        }
                    });
                    let photometric = match f.photometric {
                        0 => "0 (WhiteIsZero)".into(),
                        1 => "1 (BlackIsZero)".into(),
                        2 => "2 (RGB)".into(),
                        3 => "3 (palette)".into(),
                        // Separated. InkSet says *which* inks; only set 1 is
                        // CMYK, and only that one the viewer converts. Anything
                        // else stays raw ink planes, so label it honestly
                        // rather than calling every separated file CMYK.
                        5 if f.ink_set == 1 => "5 (CMYK)".into(),
                        5 => format!("5 (separated, InkSet {})", f.ink_set),
                        other => format!("{other}"),
                    };
                    kv(ui, "Photometric", photometric);
                    let compression = match f.compression {
                        fast_tiff_lib::Compression::None => "uncompressed".into(),
                        fast_tiff_lib::Compression::Lzw => "LZW".into(),
                        fast_tiff_lib::Compression::PackBits => "PackBits".into(),
                        fast_tiff_lib::Compression::Deflate => "Deflate (zip)".into(),
                        fast_tiff_lib::Compression::Zstd => "ZSTD".into(),
                        other => format!("{other:?}"),
                    };
                    kv(ui, "Compression", compression);
                    let predictor = match f.predictor {
                        1 => "none".into(),
                        2 => "2 (horizontal differencing)".into(),
                        3 => "3 (floating-point)".into(),
                        other => format!("{other}"),
                    };
                    kv(ui, "Predictor", predictor);
                    kv(
                        ui,
                        "Strips/frame",
                        format!(
                            "{} ({} rows/strip)",
                            f.strip_offsets.len(),
                            f.rows_per_strip
                        ),
                    );
                    let bpf = f.width as u64
                        * f.height as u64
                        * f.samples_per_pixel as u64
                        * (f.bits_per_sample as u64 / 8);
                    kv(ui, "Decoded frame", human_bytes(bpf));
                });
        }

        ui.add_space(12.0);
        egui::CollapsingHeader::new("ImageDescription (tag 270)")
            .default_open(true)
            .show(ui, |ui| match &tiff.description {
                Some(desc) => {
                    // Read-only TextEdit: selectable + copyable.
                    let mut text = desc.as_str();
                    ui.add(
                        egui::TextEdit::multiline(&mut text)
                            .font(egui::TextStyle::Monospace)
                            .desired_width(f32::INFINITY)
                            .desired_rows(desc.lines().count().clamp(2, 16)),
                    );
                }
                None => {
                    ui.label(RichText::new("(this file carries no ImageDescription)").weak());
                }
            });

        ui.add_space(12.0);
        egui::CollapsingHeader::new("OME-TIFF metadata")
            .default_open(false)
            .show(ui, |ui| ome_metadata(ui, tiff));

        ui.add_space(12.0);
        ui.heading("ImageJ metadata");
        let meta = &tiff.meta;
        egui::Grid::new("meta_ij")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                kv(
                    ui,
                    "Dimensions",
                    format!(
                        "{} channel(s) x {} slice(s) x {} frame(s)",
                        meta.channels, meta.slices, meta.frames
                    ),
                );
                // Everything in this panel is the *file's* own metadata, left
                // exactly as parsed. The viewer's interpretation is separate
                // (see `fast_tiff_viewer::display`) and routinely differs — a
                // mislabeled `channels=100` is really a frame count, and the
                // user can reassign the axes by hand. Show that only when the
                // two disagree, so the panel stays quiet for ordinary files.
                let shown = loaded.display.dims();
                if (shown.channels, shown.slices, shown.frames)
                    != (meta.channels, meta.slices, meta.frames)
                {
                    kv(
                        ui,
                        "Shown as",
                        format!(
                            "{} channel(s) x {} slice(s) x {} frame(s)",
                            shown.channels, shown.slices, shown.frames
                        ),
                    );
                }
                let mode = match meta.mode {
                    fast_tiff_lib::DisplayMode::Grayscale => "grayscale",
                    fast_tiff_lib::DisplayMode::Composite => "composite",
                    fast_tiff_lib::DisplayMode::Color => "color",
                };
                kv(ui, "Display mode", mode);
                if let Some(unit) = &meta.unit {
                    kv(ui, "Unit", unit.clone());
                }
                if let Some(fi) = meta.frame_interval_s {
                    kv(ui, "Frame interval", format!("{fi} s"));
                }
                if let Some(fps) = meta.fps {
                    kv(ui, "Playback fps", fps.to_string());
                }
                if let Some(spacing) = meta.spacing {
                    kv(ui, "Z spacing", spacing.to_string());
                }
                if let Some(looped) = meta.loop_playback {
                    kv(ui, "Loop playback", looped.to_string());
                }
                if let Some((c0, c1)) = meta.calibration {
                    kv(ui, "Calibration", format!("value = {c0} + {c1} x raw"));
                }
                for (i, cd) in meta.channel_display.iter().enumerate() {
                    let range = match cd.range {
                        Some((lo, hi)) => format!("{lo} .. {hi}"),
                        None => "auto-contrast".into(),
                    };
                    kv(ui, &format!("Ch {} display range", i + 1), range);
                }
            });
    });
}

type OmeElement = fast_tiff_lib::metadata::ome::XmlElement;

/// The OME-XML an OME-TIFF keeps in its ImageDescription, as tables: each
/// element's attributes, a table for a run of same-named elements (a `Plane`
/// for every plane, a `Channel` for every channel), and a section for anything
/// nested deeper.
///
/// Parsed when the section is first opened, and once per file: the XML of a
/// long recording carries a `Plane` per plane and runs to megabytes, and this
/// window is drawn every frame.
fn ome_metadata(ui: &mut egui::Ui, tiff: &Arc<fast_tiff_lib::TiffStack>) {
    let Some(desc) = tiff
        .description
        .as_deref()
        .filter(|d| fast_tiff_lib::metadata::detect(Some(d)) == fast_tiff_lib::MetadataFormat::Ome)
    else {
        ui.label(RichText::new("(this file carries no OME-XML)").weak());
        return;
    };
    match ome_tree(ui.ctx(), tiff, desc).as_ref() {
        Some(root) => {
            ui.label(RichText::new(&root.name).strong());
            ome_element(ui, root, egui::Id::new("ome_tree"), 0);
        }
        None => {
            ui.label(
                RichText::new(
                    "(the OME-XML is not well-formed, so it cannot be laid out as a table; \
                     ImageDescription above shows it as text)",
                )
                .weak(),
            );
        }
    }
}

/// The parsed tree of `desc`, from the last time it was parsed if that was
/// this file's. Keyed by where the stack and its description live, so a file
/// opened in its place is parsed afresh.
fn ome_tree(
    ctx: &egui::Context,
    tiff: &Arc<fast_tiff_lib::TiffStack>,
    desc: &str,
) -> Arc<Option<OmeElement>> {
    type Cached = ((usize, usize, usize), Arc<Option<OmeElement>>);
    let key = (
        Arc::as_ptr(tiff) as usize,
        desc.as_ptr() as usize,
        desc.len(),
    );
    let id = egui::Id::new("ome_tree_cache");
    if let Some((k, tree)) = ctx.data(|d| d.get_temp::<Cached>(id)) {
        if k == key {
            return tree;
        }
    }
    let tree = Arc::new(fast_tiff_lib::metadata::ome::xml_tree(desc));
    ctx.data_mut(|d| d.insert_temp::<Cached>(id, (key, tree.clone())));
    tree
}

fn ome_attr<'a>(el: &'a OmeElement, key: &str) -> Option<&'a str> {
    el.attributes
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
}

/// What a section is called: the element, and its `Name` or `ID` when it has
/// one, which is what tells one `Image` or `Objective` from the next.
fn ome_title(el: &OmeElement) -> String {
    match ome_attr(el, "Name").or_else(|| ome_attr(el, "ID")) {
        Some(tag) => format!("{} \u{2014} {tag}", el.name),
        None => el.name.clone(),
    }
}

/// A value, wrapped at a fixed width. A `schemaLocation` or an annotation runs
/// to hundreds of characters, and unwrapped it widens the window for good —
/// an egui window grows to fit and never shrinks back.
fn ome_value(ui: &mut egui::Ui, text: &str) {
    ui.scope(|ui| {
        ui.set_max_width(260.0);
        ui.add(egui::Label::new(text).wrap());
    });
}

/// One element: its attributes and text-only children as a grid, then its
/// other children grouped by name.
fn ome_element(ui: &mut egui::Ui, el: &OmeElement, id: egui::Id, depth: usize) {
    // Children by name, in the order each name first appears.
    let mut groups: Vec<(&str, Vec<&OmeElement>)> = Vec::new();
    for child in &el.children {
        match groups.iter_mut().find(|(name, _)| *name == child.name) {
            Some((_, members)) => members.push(child),
            None => groups.push((&child.name, vec![child])),
        }
    }
    // A lone child that is only a value, like `AcquisitionDate`, reads better
    // as a row than as a section of one line.
    let is_value = |members: &[&OmeElement]| {
        members.len() == 1 && members[0].attributes.is_empty() && members[0].children.is_empty()
    };
    let values: Vec<_> = groups.iter().filter(|(_, m)| is_value(m)).collect();
    if !el.attributes.is_empty() || !el.text.is_empty() || !values.is_empty() {
        egui::Grid::new(id.with("attributes"))
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                for (k, v) in &el.attributes {
                    ui.label(RichText::new(k).strong());
                    ome_value(ui, v);
                    ui.end_row();
                }
                for (name, members) in &values {
                    ui.label(RichText::new(*name).strong());
                    ome_value(ui, &members[0].text);
                    ui.end_row();
                }
                if !el.text.is_empty() {
                    ui.label(RichText::new("(text)").weak());
                    ome_value(ui, &el.text);
                    ui.end_row();
                }
            });
    }

    for (name, members) in groups.iter().filter(|(_, m)| !is_value(m)) {
        let gid = id.with(name);
        if let [only] = members.as_slice() {
            egui::CollapsingHeader::new(ome_title(only))
                .id_salt(gid)
                .default_open(depth < 2)
                .show(ui, |ui| ome_element(ui, only, gid, depth + 1));
        } else if members.iter().all(|m| m.children.is_empty()) {
            egui::CollapsingHeader::new(format!("{name} ({})", members.len()))
                .id_salt(gid)
                .default_open(false)
                .show(ui, |ui| ome_table(ui, members, gid));
        } else {
            egui::CollapsingHeader::new(format!("{name} ({})", members.len()))
                .id_salt(gid)
                .default_open(false)
                .show(ui, |ui| {
                    for (i, m) in members.iter().enumerate() {
                        let mid = gid.with(i);
                        egui::CollapsingHeader::new(format!("{i}: {}", ome_title(m)))
                            .id_salt(mid)
                            .default_open(false)
                            .show(ui, |ui| ome_element(ui, m, mid, depth + 2));
                    }
                });
        }
    }
}

/// A run of same-named elements with nothing nested in them: a row each and
/// a column per attribute any of them has. Only the rows in view are laid
/// out, because a recording's `Plane` table can be tens of thousands long.
fn ome_table(ui: &mut egui::Ui, members: &[&OmeElement], id: egui::Id) {
    use egui_extras::{Column, TableBuilder};

    let mut columns: Vec<&str> = Vec::new();
    for m in members {
        for (k, _) in &m.attributes {
            if !columns.contains(&k.as_str()) {
                columns.push(k);
            }
        }
    }
    let any_text = members.iter().any(|m| !m.text.is_empty());
    let row_height = ui.spacing().interact_size.y;
    let spaced = row_height + ui.spacing().item_spacing.y;
    // A cell keeps its value on one line; a wide table scrolls sideways.
    let cell = |ui: &mut egui::Ui, text: &str| {
        ui.add(egui::Label::new(text).extend());
    };
    egui::ScrollArea::horizontal()
        .id_salt(id.with("sideways"))
        .show(ui, |ui| {
            TableBuilder::new(ui)
                .id_salt(id.with("table"))
                .striped(true)
                .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
                .max_scroll_height(320.0)
                // Inside the window's own scroll area the space left is
                // whatever is below this point, and a table falls back to its
                // minimum there: two or three rows. A dozen, or all of them.
                .min_scrolled_height(spaced * members.len().min(12) as f32)
                .column(Column::auto())
                .columns(
                    Column::auto().resizable(true),
                    columns.len() + usize::from(any_text),
                )
                .header(row_height, |mut header| {
                    header.col(|ui| {
                        ui.strong("#");
                    });
                    for c in &columns {
                        header.col(|ui| {
                            ui.strong(*c);
                        });
                    }
                    if any_text {
                        header.col(|ui| {
                            ui.strong("(text)");
                        });
                    }
                })
                .body(|body| {
                    body.rows(row_height, members.len(), |mut row| {
                        let m = members[row.index()];
                        let i = row.index();
                        row.col(|ui| {
                            ui.label(RichText::new(i.to_string()).weak());
                        });
                        for c in &columns {
                            row.col(|ui| cell(ui, ome_attr(m, c).unwrap_or("")));
                        }
                        if any_text {
                            row.col(|ui| cell(ui, &m.text));
                        }
                    });
                });
        });
}

/// `1234567` -> `"1.2 MiB (1234567 bytes)"`.
fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{n} B")
    } else {
        format!("{:.1} {} ({n} bytes)", v, UNITS[u])
    }
}
