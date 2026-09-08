//! The acquisition record an OIR carries in its own XML, written out in the
//! form the acquisition software exports beside the file as a `.txt`.
//!
//! # Why this exists
//!
//! An OIR's metadata reaches a converted TIFF through `ImageDescription`
//! (tag 270), and the file's own XML cannot go there as it stands. In one real
//! acquisition it is four megabytes, of which **3.1 MB is three 65,536-entry
//! display lookup tables written out as XML**, plus 350 KB of drawing overlays,
//! plus the binary padding that sits between the documents in a block. The part
//! a person would want is under 40 KB of it. A second file has 299 more
//! documents, one per frame. Nothing reads that as a description.
//!
//! So it is *translated*: every value that has a place in the exported record
//! is written into one, and everything else — the lookup tables, the overlays,
//! the per-frame duplicates — is dropped, being display state and drawing
//! geometry rather than a record of how the data was acquired.
//!
//! # Why that format
//!
//! Because things downstream parse it. A converted file's description is the
//! only place an analysis can learn when the stimulus fired, and the stimulus
//! plugins already read the shape the instrument exports. Inventing a second
//! dialect here would mean every reader learning both, for no gain: this is a
//! record of an acquisition, and the acquisition's own software already has a
//! way of writing one down.
//!
//! What is *not* done is reading that `.txt`. It is an optional, detachable
//! second copy of what the container already holds — it goes missing, it gets
//! renamed, it is left behind when the `.oir` is moved, and because the export
//! carries a series number it can belong to a different acquisition in the same
//! folder. Everything here comes from the file being opened, so the record
//! cannot end up describing some other recording's pixels.
//!
//! The numbers that matter (pixel size, z-step, frame timing, the event markers
//! a stimulus experiment is aligned to) also reach [`StackInfo`] as values, not
//! only as text — see [`Record`].
//!
//! [`StackInfo`]: fasttiff_plugin_api::StackInfo
//!
//! # How it was checked
//!
//! The mapping is not guesswork about what the elements mean: it was diffed
//! against the vendor's own export. For an acquisition that has both, the text
//! built here from the XML reproduces **46 of the 53 lines of that export, byte
//! for byte and in the same order**, and contradicts none of them. The seven it
//! does not emit are values the XML does not carry — the path on the
//! acquisition machine, `Primary Dimensions`, `Region Mode`, `Find Mode`,
//! `ADM`, and `Laser ND Filter`. Omitting a key is a gap; emitting a wrong
//! value for it would be worse, so nothing is emitted that the oracle did not
//! confirm.
//!
//! One value is *derived* rather than copied for that reason: the XML's
//! integration count is whatever was last configured and stays there when
//! integration is switched off, so a file stating `None` still carried a count
//! of 5 where the export says 0.
//!
//! # Layout of the metadata
//!
//! An OIR's metadata blocks each hold one or more XML documents laid end to
//! end, separated by binary padding, and the interesting ones are:
//!
//! ```text
//!   lsmimage:imageProperties   the acquisition: optics, scanner, laser,
//!                              detectors, axes, per-channel calibration
//!   event:eventList            the stimulus/TTL markers, with times in ms
//!   lsmframe:frameProperties   one per frame, carrying its timestamp
//!   fileinfo:fileInfomation    the file format version (spelling theirs)
//!   base:imageDefinition       frame size and sample depth
//!   lsmimage:lsmChannel        one recorded channel
//! ```
//!
//! Both `imageProperties` and `eventList` appear more than once: an OIR is
//! written as it is acquired, so there is an early copy and a final one. The
//! early `eventList` is *empty* — the events had not happened yet — which is
//! why [`Record::parse`] takes the last non-empty one rather than the first
//! match. Reading the first would silently produce a recording with no
//! stimulus markers at all.

use quick_xml::events::Event;
// Aliased: `Reader` below is this module's own, and reads better unqualified
// at its call site than quick-xml's does here.
use quick_xml::Reader as XmlReader;

/// Document roots this module reads. Everything else in an OIR's metadata is
/// display state (`lut:LUT`), drawing geometry (`overlay:contents`) or empty
/// scaffolding (`annotation:annotationStore`), and the LUT documents alone are
/// most of the bytes — so they are rejected on the root element's name, before
/// anything parses them.
const KNOWN_ROOTS: &[&str] = &[
    "imageProperties",
    "eventList",
    "frameProperties",
    "fileInfomation",
    "imageDefinition",
    "lsmChannel",
];

/// Bounds on a document from an untrusted file, so a malformed or hostile one
/// fails on a limit rather than on memory. Both are far beyond any real
/// document this reads: the largest is `imageProperties` at ~1,300 elements.
const MAX_DEPTH: usize = 64;
const MAX_ELEMENTS: usize = 50_000;

// ------------------------------------------------------------- a tiny DOM

/// One XML element, keyed by *local* name — the OIR schema puts nearly every
/// element in a different namespace prefix (`base:`, `commonimage:`,
/// `lsmparam:`, …) and none of the paths below are ambiguous without them.
#[derive(Debug, Default)]
pub(super) struct El {
    name: String,
    attrs: Vec<(String, String)>,
    text: String,
    kids: Vec<El>,
}

impl El {
    /// Parse one document. `None` when it is malformed or exceeds the bounds
    /// above — a document that does not parse contributes nothing rather than
    /// failing the import, exactly as a missing one would.
    fn parse(xml: &str) -> Option<El> {
        let mut reader = XmlReader::from_str(xml);
        let mut stack: Vec<El> = vec![El::default()];
        let mut count = 0usize;
        loop {
            let event = reader.read_event();
            // `<a/>` arrives as `Empty` and never sends a matching `End`.
            let empty = matches!(event, Ok(Event::Empty(_)));
            match event {
                Ok(Event::Start(ref e)) | Ok(Event::Empty(ref e)) => {
                    count += 1;
                    if count > MAX_ELEMENTS || stack.len() > MAX_DEPTH {
                        return None;
                    }
                    let el = El {
                        name: String::from_utf8_lossy(e.local_name().as_ref()).into_owned(),
                        attrs: e
                            .attributes()
                            .flatten()
                            .map(|a| {
                                (
                                    String::from_utf8_lossy(a.key.local_name().as_ref())
                                        .into_owned(),
                                    a.unescape_value().unwrap_or_default().into_owned(),
                                )
                            })
                            .collect(),
                        ..El::default()
                    };
                    // An empty element is finished the moment it arrives.
                    // Pushing it would leave the stack one deeper for the rest
                    // of the document and nest every following sibling inside
                    // it — which is how `<opticalResolution/>` would end up
                    // owning the calibration that follows it.
                    if empty {
                        stack.last_mut()?.kids.push(el);
                    } else {
                        stack.push(el);
                    }
                }
                Ok(Event::End(_)) => {
                    // The root sentinel must survive: a document with a stray
                    // closing tag would otherwise pop it and panic below.
                    if stack.len() <= 1 {
                        continue;
                    }
                    let done = stack.pop()?;
                    stack.last_mut()?.kids.push(done);
                }
                Ok(Event::Text(t)) => {
                    if let Ok(s) = t.unescape() {
                        stack.last_mut()?.text.push_str(s.trim());
                    }
                }
                Ok(Event::Eof) => break,
                Err(_) => return None,
                _ => {}
            }
        }
        // Unclosed elements are folded back in rather than discarded: a
        // document truncated by an interrupted acquisition still describes
        // everything before the cut.
        while stack.len() > 1 {
            let done = stack.pop()?;
            stack.last_mut()?.kids.push(done);
        }
        stack.pop()?.kids.into_iter().next()
    }

    /// The value of an attribute, by local name.
    fn attr(&self, name: &str) -> Option<&str> {
        self.attrs
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    /// Every element at a `/`-separated path of local names, in document order.
    fn all(&self, path: &str) -> Vec<&El> {
        let mut cur = vec![self];
        for step in path.split('/') {
            let mut next = Vec::new();
            for el in cur {
                next.extend(el.kids.iter().filter(|k| k.name == step));
            }
            cur = next;
            if cur.is_empty() {
                break;
            }
        }
        cur
    }

    /// The first element at `path`.
    fn at(&self, path: &str) -> Option<&El> {
        self.all(path).into_iter().next()
    }

    /// The text of the first element at `path`, if it is not empty.
    fn text_at(&self, path: &str) -> Option<&str> {
        self.at(path)
            .map(|e| e.text.as_str())
            .filter(|s| !s.is_empty())
    }

    /// The text of the first element at `path`, parsed as a finite number.
    fn num_at(&self, path: &str) -> Option<f64> {
        self.text_at(path)?
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite())
    }
}

/// The `width`/`height` an `imageDefinition` states, if both are sane.
fn frame_size(el: &El) -> Option<(u32, u32)> {
    let n = |k: &str| {
        el.num_at(k)
            .filter(|v| *v >= 1.0 && *v <= 1e6)
            .map(|v| v as u32)
    };
    Some((n("width")?, n("height")?))
}

/// Whether a document is one of the [`KNOWN_ROOTS`] — decided on its root
/// element's name alone, which is what keeps a 525 KB lookup table from ever
/// being parsed, carried, or written into a TIFF tag.
pub(super) fn is_known_document(xml: &str) -> bool {
    root_name(xml).is_some_and(|r| KNOWN_ROOTS.contains(&r))
}

/// The local name of a document's root element, without parsing the document.
fn root_name(xml: &str) -> Option<&str> {
    let mut rest = xml;
    loop {
        let open = rest.find('<')?;
        rest = &rest[open + 1..];
        // Skip the declaration, comments and processing instructions.
        if rest.starts_with('?') || rest.starts_with('!') {
            continue;
        }
        let end = rest
            .find(|c: char| c.is_whitespace() || c == '>' || c == '/')
            .unwrap_or(rest.len());
        let name = &rest[..end];
        return Some(name.rsplit(':').next().unwrap_or(name));
    }
}

// ------------------------------------------------------------- the record

/// One channel as the acquisition recorded it.
#[derive(Debug, Default)]
pub(super) struct Channel {
    /// The acquisition's own name for it, `CH1`.
    pub(super) name: Option<String>,
    /// The detector's name, `RNDD3G` — which is what the exported record calls
    /// the "Channel Name", so it is what is written under that key.
    pub(super) device: Option<String>,
    /// Barrier filter, `BA575-645`.
    filter: Option<String>,
    /// Emission dichroic mirror, `SDM570`.
    dichroic: Option<String>,
    /// The dye the channel was configured for, `Alexa Fluor 488`.
    dye: Option<String>,
    /// That dye's emission peak in nm, as the file states it.
    emission_nm: Option<String>,
    bits: Option<String>,
    voltage: Option<String>,
}

impl Channel {
    /// What to call this channel in a one-line label: the detector name if the
    /// file gives one, else the acquisition's own `CHn`.
    pub(super) fn label(&self) -> Option<String> {
        self.device.clone().or_else(|| self.name.clone())
    }
}

/// Everything this reader takes from an OIR's own XML.
///
/// Every field is optional and independently sourced: a file that states none
/// of them yields an empty record and no text, rather than a failed import.
/// The pixels never depend on any of it.
#[derive(Debug, Default)]
pub(super) struct Record {
    file_version: Option<String>,
    created: Option<String>,
    system_name: Option<String>,
    system_version: Option<String>,

    pub(super) width: Option<u32>,
    pub(super) height: Option<u32>,
    /// Microns per pixel, at full precision — the text record rounds these to
    /// three decimals, and the structured metadata should not inherit that.
    pub(super) pixel_x: Option<f64>,
    pub(super) pixel_y: Option<f64>,

    /// Every frame timestamp the acquisition recorded, in milliseconds, sorted
    /// and without duplicates — a multi-channel acquisition writes one document
    /// per channel per frame and they carry the same timestamp.
    ///
    /// Kept rather than reduced to a single duration, because the duration
    /// depends on how many frames are being asked about: see
    /// [`Record::duration_s`].
    times: Vec<f64>,
    /// Whether the timelapse ran free (no interval was set), which is the only
    /// interval state this can tell apart.
    free_run: bool,
    pub(super) slices: Option<usize>,
    pub(super) z_step: Option<f64>,

    // Everything below is carried as the *text the file states*, not as a
    // number this reads and prints again. Reformatting is how a record starts
    // disagreeing with the instrument that wrote it: rendering these to one
    // decimal turned an objective's numerical aperture of 1.05 into 1.1 and a
    // resonant scanner's 0.067 us/pixel dwell into 0.1, both of them wrong, and
    // wrong in a way that looks like a plausible number rather than like a bug.
    // Nothing here is arithmetic — these values are passed through — so passing
    // them through is also the only way to be sure they are unaltered.
    objective: Option<String>,
    magnification: Option<String>,
    numerical_aperture: Option<String>,
    scanner: Option<String>,
    /// `false` is FluoView's "Oneway".
    roundtrip: Option<bool>,
    sampling_speed_us: Option<String>,
    sequential: Option<String>,
    integration: Option<String>,
    integration_count: Option<String>,
    rotation_deg: Option<String>,
    pan_x: Option<String>,
    pan_y: Option<String>,
    zoom: Option<String>,
    mirror_turret: Option<String>,
    laser_nm: Option<String>,
    laser_transmissivity: Option<String>,

    pub(super) channels: Vec<Channel>,
    /// `(contents, milliseconds)` for each event marker, in acquisition order.
    pub(super) events: Vec<(String, f64)>,
}

/// One section's rows, in the order they are written.
///
/// A section that ends up with no rows is dropped entirely rather than written
/// as a bare header: `"[Acquisition]"` with nothing under it is not a record of
/// anything, and reads like a value that went missing on the way here.
#[derive(Default)]
struct Rows(Vec<(&'static str, String)>);

impl Rows {
    /// Add a row, or nothing at all when the file did not state the value.
    fn set(&mut self, key: &'static str, value: Option<String>) {
        if let Some(v) = value {
            self.0.push((key, v));
        }
    }
}

/// One `"key"\t"value"` line, as FluoView writes them.
fn row(key: &str, value: &str) -> String {
    format!("\"{key}\"\t\"{value}\"\n")
}

/// Collects an acquisition's metadata documents, a file at a time.
///
/// A long recording is split across several files, and **each part carries its
/// own frame timestamps** — 2,030 of them per part in the four-part recording
/// this was built against. Reading only the first would time the first part and
/// leave the rest of the recording unaccounted for, so every part is absorbed
/// here; taking them one at a time is what keeps one part's documents in memory
/// rather than all of them.
#[derive(Default)]
pub(super) struct Reader {
    props: Option<El>,
    /// The frame size as stated somewhere other than the acquisition record,
    /// which is where it is normally read from.
    size: Option<(u32, u32)>,
    file_version: Option<String>,
    events: Vec<(String, f64)>,
    times: Vec<f64>,
}

impl Reader {
    /// Take in one file's documents.
    ///
    /// Later documents win over earlier ones for `imageProperties` and the
    /// event list, because an OIR is written as it is acquired: the first copy
    /// describes the acquisition that was *about* to happen, the last one the
    /// acquisition that did. The rule holds across parts as well as within one,
    /// since every part's final copy restates the whole acquisition.
    pub(super) fn absorb(&mut self, docs: &[String]) {
        for doc in docs {
            let Some(root) = root_name(doc) else { continue };
            if !KNOWN_ROOTS.contains(&root) {
                continue;
            }
            let Some(el) = El::parse(doc) else { continue };
            match root {
                "imageProperties" => self.props = Some(el),
                "imageDefinition" => self.size = self.size.or_else(|| frame_size(&el)),
                "fileInfomation" => {
                    if let Some(v) = el.text_at("version") {
                        self.file_version = Some(v.to_string());
                    }
                }
                "eventList" => {
                    let events: Vec<(String, f64)> = el
                        .all("event")
                        .iter()
                        .filter_map(|e| {
                            let ms = e.num_at("time")?;
                            Some((e.text_at("name").unwrap_or("").to_string(), ms))
                        })
                        .collect();
                    // The early copy is empty; only a list with something in it
                    // replaces what is already known.
                    if !events.is_empty() {
                        self.events = events;
                    }
                }
                "frameProperties" => {
                    for axis in el.all("axisValue") {
                        if axis.text_at("axisType") == Some("TIMELAPSE") {
                            if let Some(ms) = axis.num_at("position") {
                                self.times.push(ms);
                            }
                        }
                    }
                    // A frame states its own size, which is the last place left
                    // to learn it in a file whose acquisition record is missing
                    // or unreadable — and the import cannot proceed without it.
                    if let Some(d) = el.at("imageDefinition") {
                        self.size = self.size.or_else(|| frame_size(d));
                    }
                }
                _ => {}
            }
        }
    }

    /// The record, once every part has been absorbed.
    pub(super) fn finish(self) -> Record {
        let mut r = Record {
            file_version: self.file_version,
            events: self.events,
            times: self.times,
            ..Record::default()
        };
        if let Some(p) = &self.props {
            r.read_properties(p);
        }
        // Only where the acquisition record did not say. It describes the
        // acquisition as a whole; the others describe one frame of it, and an
        // acquisition whose frames disagree with it is one this reader has
        // never seen.
        if let Some((w, h)) = self
            .size
            .filter(|_| r.width.is_none() || r.height.is_none())
        {
            r.width.get_or_insert(w);
            r.height.get_or_insert(h);
        }
        r.times
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        r.times.dedup();
        r
    }
}

impl Record {
    /// Every document at once. The importer reads a part at a time through
    /// [`Reader`]; this is the shape the tests are written in, where the whole
    /// acquisition is a handful of strings.
    #[cfg(test)]
    pub(super) fn parse(docs: &[String]) -> Record {
        let mut reader = Reader::default();
        reader.absorb(docs);
        reader.finish()
    }

    /// How long a recording of `frames` frames took, in seconds.
    ///
    /// `None` when the timestamps do not reach the end of it. That is not
    /// pedantry about completeness — it is the difference between a duration
    /// and a fraction of one. A four-part acquisition whose later parts went
    /// unread states 7,213 frames and carries the first 2,030 timestamps; the
    /// last of *those* is the length of part one, and reporting it as the
    /// recording's length would put the frame rate out by a factor of three and
    /// misalign every stimulus-locked analysis downstream — silently, because a
    /// frame rate wrong by a constant factor produces maps that still look like
    /// maps.
    ///
    /// # Why the count of timestamps settles nothing
    ///
    /// A real acquisition rarely has exactly one timestamp per kept frame, and
    /// it can miss in both directions at once. One 4,913-frame recording
    /// carries 4,912 of them: two documents absent from the middle, and one
    /// *extra* past the end for the partial final frame this reader drops. Its
    /// timestamps are therefore one short of its frames while actually running
    /// one frame *past* them, so neither "index the frame asked about" nor
    /// "take the last timestamp" gives the right answer — the first runs out,
    /// the second overshoots by a frame.
    ///
    /// What does settle it is the clock. Frames arrive on a fixed period — the
    /// gaps are 133.333333 ms, 4,909 times out of 4,911, and exactly double
    /// that for the two with a document missing — so the last kept frame is at
    /// `(frames - 1) * period`, whatever happened to the documents in between.
    /// The timestamps are then needed only to answer the one question that is
    /// really being asked: did the recording run that long? The last of them
    /// falls on frame `round(last / period)`, and if that reaches the last kept
    /// frame, it did.
    pub(super) fn duration_s(&self, frames: usize) -> Option<f64> {
        if frames == 0 {
            return None;
        }
        let ms = if self.times.len() >= frames {
            // A timestamp for every frame: read the one asked about rather than
            // modelling it. Indexing is exact where the clock is only a very
            // good fit, and it is right even for a recording whose frames were
            // not evenly spaced at all.
            *self.times.get(frames - 1)?
        } else {
            let period = self.frame_period()?;
            let last = *self.times.last()?;
            // `- 0.5` rather than an exact reach: the last timestamp lands on
            // its own frame, and asking it to be at or past the last frame's
            // slot allows for the rounding in either.
            let reached = last / period >= (frames - 1) as f64 - 0.5;
            reached.then(|| (frames - 1) as f64 * period)?
        };
        (ms.is_finite() && ms > 0.0).then_some(ms / 1000.0)
    }

    /// The typical gap between frames, in milliseconds.
    ///
    /// The median rather than the mean, so that the double-width gaps left by a
    /// missing document — the very thing this exists to see past — cannot move
    /// it. `None` for a recording of one frame, which has no gap to measure.
    fn frame_period(&self) -> Option<f64> {
        let mut gaps: Vec<f64> = self.times.windows(2).map(|w| w[1] - w[0]).collect();
        if gaps.is_empty() {
            return None;
        }
        gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Some(gaps[gaps.len() / 2]).filter(|p| p.is_finite() && *p > 0.0)
    }

    /// Seconds between frames, over the gaps rather than over the frame count —
    /// which is how the `T Dimension` line this writes is meant to be read
    /// back, so the value and the text cannot disagree by a frame.
    pub(super) fn frame_interval_s(&self, frames: usize) -> Option<f64> {
        if frames < 2 {
            return None;
        }
        Some(self.duration_s(frames)? / (frames - 1) as f64)
    }

    /// Everything that comes out of the `imageProperties` document.
    fn read_properties(&mut self, p: &El) {
        self.created = p.text_at("general/creationDateTime").map(str::to_string);
        self.system_name = p.text_at("system/systemName").map(str::to_string);
        self.system_version = p.text_at("system/systemVersion").map(str::to_string);
        self.width = p.num_at("imageInfo/width").map(|v| v as u32);
        self.height = p.num_at("imageInfo/height").map(|v| v as u32);

        // `imageInfo` is what was *recorded*; `acquisition` is what was
        // configured. They differ — a two-detector microscope with one channel
        // enabled lists two channels under `acquisition` and one under
        // `imageInfo` — and the recorded set is the one that matches the planes
        // in the file. The configured entries are still needed: they carry the
        // detector name and the PMT voltage, matched back by channel id.
        let configured = p.all("acquisition/phase/group/channel");
        let pmts = p.all("acquisition/imagingParam/pmt");
        let lightpaths = p.all("acquisition/configuration/productData/externalDetectorLightpath");

        for c in p.all("imageInfo/phase/group/channel") {
            let id = c.attr("id");
            let conf = configured
                .iter()
                .find(|e| e.attr("id") == id && id.is_some());
            let detector = conf.and_then(|e| e.attr("detectorId"));
            self.channels.push(Channel {
                name: c.text_at("name").map(str::to_string),
                device: conf
                    .and_then(|e| e.text_at("deviceName"))
                    .map(str::to_string),
                filter: c.text_at("productData/filterName").map(str::to_string),
                dichroic: lightpaths
                    .iter()
                    .find(|lp| detector.is_some() && lp.attr("detectorId") == detector)
                    .and_then(|lp| lp.text_at("filterCube/dichroicMirror/name"))
                    .map(str::to_string),
                dye: conf.and_then(|e| e.text_at("dyeName")).map(str::to_string),
                emission_nm: conf
                    .and_then(|e| e.text_at("dyeData/emissionWavelength"))
                    .map(str::to_string),
                bits: c.text_at("imageDefinition/bitCounts").map(str::to_string),
                voltage: pmts
                    .iter()
                    .find(|e| e.attr("channelId") == id && id.is_some())
                    .and_then(|e| e.text_at("voltage"))
                    .map(str::to_string),
            });
        }
        // Pixel size is per channel in the schema and identical across them in
        // every file seen; the first recorded channel speaks for the frame.
        if let Some(first) = p.at("imageInfo/phase/group/channel") {
            self.pixel_x = first.num_at("length/x").filter(|v| *v > 0.0);
            self.pixel_y = first.num_at("length/y").filter(|v| *v > 0.0);
        }

        for axis in p.all("imageInfo/axis") {
            let size = axis.num_at("maxSize").unwrap_or(0.0) as usize;
            match axis.text_at("axis") {
                Some("TIMELAPSE") if size > 0 => {
                    // A step of zero is FluoView's "FreeRun": no interval was
                    // asked for, the scanner ran as fast as it could. A
                    // non-zero step is an interval this cannot name a unit for,
                    // so it says nothing rather than guessing one.
                    self.free_run = axis.num_at("step").unwrap_or(0.0) == 0.0;
                }
                Some("ZSTACK") if size > 0 => {
                    self.slices = Some(size);
                    self.z_step = axis.num_at("step").filter(|v| *v > 0.0);
                }
                _ => {}
            }
        }

        if let Some(obj) = p.at("acquisition/microscopeConfiguration/objectiveLens") {
            self.objective = obj.text_at("name").map(str::to_string);
            self.magnification = obj.text_at("magnification").map(str::to_string);
            self.numerical_aperture = obj.text_at("naValue").map(str::to_string);
        }
        self.scanner = p
            .text_at("acquisition/configuration/scannerType")
            .map(str::to_string);
        // The settings block for the scanner actually in use. A FluoView system
        // carries one per scanner it could have used, and the resonant
        // scanner's numbers are thirty times the galvanometer's — picking the
        // wrong block would state a pixel dwell time that never happened.
        if let Some(settings) = p
            .all("acquisition/scannerSettings")
            .into_iter()
            .find(|s| s.attr("type").is_some() && s.attr("type") == self.scanner.as_deref())
        {
            self.roundtrip = match settings.text_at("param/speed/roundtrip") {
                Some("true") => Some(true),
                Some("false") => Some(false),
                _ => None,
            };
            self.sampling_speed_us = settings.text_at("param/speed/speed").map(str::to_string);
        }
        self.sequential = p
            .text_at("acquisition/imagingParam/method/sequentialType")
            .map(str::to_string);
        self.integration = p
            .text_at("acquisition/imagingParam/method/integration/type")
            .map(str::to_string);
        // The count the file holds is whatever was last configured, and it
        // stays there when integration is switched off — a file that says
        // `None` still carried a count of 5. FluoView writes 0 in that case,
        // and 0 is what "no integration" means, so this follows it rather than
        // repeating a number that describes nothing.
        self.integration_count = match self.integration.as_deref() {
            Some("None") => Some("0".to_string()),
            Some(_) => p
                .text_at("acquisition/imagingParam/method/integration/count")
                .map(str::to_string),
            None => None,
        };
        if let Some(area) = p.at("acquisition/imagingParam/area") {
            self.rotation_deg = area.text_at("rotation").map(str::to_string);
            self.pan_x = area.text_at("xpan").map(str::to_string);
            self.pan_y = area.text_at("ypan").map(str::to_string);
            self.zoom = area.text_at("zoom").map(str::to_string);
        }
        self.mirror_turret = p
            .text_at("acquisition/productData/lsmMicroscopePhase/lightpath/firstMirrorCube/name")
            .map(str::to_string);
        self.laser_nm = p
            .text_at("acquisition/phase/imagingMainLaser/wavelength")
            .map(str::to_string);
        self.laser_transmissivity = p
            .text_at("acquisition/imagingParam/mainLaser/transmissivity")
            .map(str::to_string);
    }

    /// Whether anything was found worth writing down.
    fn is_empty(&self) -> bool {
        self.width.is_none()
            && self.channels.is_empty()
            && self.events.is_empty()
            && self.created.is_none()
    }

    /// Render the record as FluoView writes it: `"key"\t"value"` lines under
    /// `"[Section]"` headers.
    ///
    /// The shape is not decoration. It is the format the acquisition software
    /// itself exports, so a converted file's tag 270 and that export can be
    /// read by one parser rather than two.
    ///
    /// `frames` is the number of time points in the stack being imported, not
    /// the number the acquisition record claims. They differ: an interrupted
    /// acquisition leaves a partial final frame, which this reader drops and the
    /// record still counts. What goes into the file has to describe the file — a
    /// `T Dimension` stating one more frame than the stack holds puts the frame
    /// interval out by one gap for anything that divides one by the other.
    ///
    /// `None` when the file carried nothing recognizable, so the caller writes
    /// no description at all rather than an empty scaffold of section headers.
    pub(super) fn to_text(&self, file_name: &str, frames: usize) -> Option<String> {
        if self.is_empty() {
            return None;
        }
        let mut doc: Vec<(String, Rows)> = Vec::new();

        let mut general = Rows::default();
        general.set("Name", Some(file_name.to_string()));
        general.set("Scan Mode", self.scan_mode(frames));
        general.set("Date", self.created.as_deref().and_then(fluoview_date));
        general.set("File Version", self.file_version.clone());
        general.set("System Name", self.system_name.clone());
        general.set("System Version", self.system_version.clone());
        doc.push(("[General]".into(), general));

        let mut dims = Rows::default();
        // `512, 0.0 - 318.198 [um], 0.621 [um/pixel]` — the extent is the frame
        // size times the pixel size, which is how the export states it.
        for (key, size, pixel) in [
            ("X Dimension", self.width, self.pixel_x),
            ("Y Dimension", self.height, self.pixel_y),
        ] {
            dims.set(
                key,
                size.zip(pixel).map(|(n, p)| {
                    format!("{n}, 0.0 - {:.3} [um], {p:.3} [um/pixel]", n as f64 * p)
                }),
            );
        }
        dims.set(
            "Channel Dimension",
            (!self.channels.is_empty()).then(|| format!("{} [Ch]", self.channels.len())),
        );
        dims.set(
            "Z Dimension",
            self.slices.zip(self.z_step).map(|(n, step)| {
                format!(
                    "{n}, 0.0 - {:.1} [um], {step:.1} [um/slice]",
                    n.saturating_sub(1) as f64 * step
                )
            }),
        );
        dims.set(
            "T Dimension",
            Some(frames)
                .filter(|n| *n > 1)
                .map(|n| match self.duration_s(n) {
                    Some(s) => format!(
                        "{n}, 0.000 - {s:.3} [s]{}",
                        if self.free_run {
                            ", Interval FreeRun"
                        } else {
                            ""
                        }
                    ),
                    // Without the timestamps there is no range to state, and
                    // inventing one would hand every reader a frame rate the file
                    // never claimed. The count alone is still worth writing: a
                    // reader looking for the duration finds no range and reports it
                    // as unknown, which is exactly the situation.
                    None => format!("{n} [T]"),
                }),
        );
        doc.push(("[Dimensions]".into(), dims));

        let mut image = Rows::default();
        image.set(
            "Image Size",
            self.width
                .zip(self.height)
                .map(|(w, h)| format!("{w} * {h} [pixel]")),
        );
        if let (Some(w), Some(h), Some(px), Some(py)) =
            (self.width, self.height, self.pixel_x, self.pixel_y)
        {
            image.set(
                "Image Size(Unit Converted)",
                Some(format!(
                    "{:.3} [um] * {:.3} [um]",
                    w as f64 * px,
                    h as f64 * py
                )),
            );
        }
        doc.push(("[Image]".into(), image));

        let mut acq = Rows::default();
        acq.set("Objective Lens", self.objective.clone());
        acq.set(
            "Objective Lens Mag.",
            self.magnification.as_ref().map(|m| format!("{m}X")),
        );
        acq.set("Objective Lens NA", self.numerical_aperture.clone());
        acq.set("Scan Device", self.scanner.clone());
        acq.set(
            "Scan Direction",
            self.roundtrip
                .map(|rt| if rt { "Roundtrip" } else { "Oneway" }.to_string()),
        );
        acq.set(
            "Sampling Speed",
            self.sampling_speed_us
                .as_ref()
                .map(|s| format!("{s} [us/pixel]")),
        );
        acq.set("Sequential Mode", self.sequential.clone());
        acq.set("Integration Type", self.integration.clone());
        acq.set("Integration Count", self.integration_count.clone());
        for (key, value, unit) in [
            ("Rotation", &self.rotation_deg, "deg"),
            ("Pan X", &self.pan_x, "um"),
            ("Pan Y", &self.pan_y, "um"),
        ] {
            acq.set(key, value.as_ref().map(|v| format!("{v} [{unit}]")));
        }
        acq.set("Zoom", self.zoom.as_ref().map(|z| format!("x{z}")));
        acq.set("MirrorTurret 1", self.mirror_turret.clone());
        doc.push(("[Acquisition]".into(), acq));

        for (i, c) in self.channels.iter().enumerate() {
            let mut ch = Rows::default();
            ch.set("Channel Name", c.label());
            ch.set("Dye Name", c.dye.clone());
            ch.set(
                "Emission WaveLength",
                c.emission_nm.as_ref().map(|nm| format!("{nm} [nm]")),
            );
            ch.set(
                "PMT Voltage",
                c.voltage.as_ref().map(|v| format!("{v} [V]")),
            );
            ch.set("BF Name", c.filter.clone());
            ch.set("Emission DM Name", c.dichroic.clone());
            ch.set("Bits/Pixel", c.bits.as_ref().map(|b| format!("{b} [bits]")));
            ch.set(
                "Laser Wavelength",
                self.laser_nm.as_ref().map(|nm| format!("{nm} [nm]")),
            );
            ch.set(
                "Laser Transmissivity",
                self.laser_transmissivity
                    .as_ref()
                    .map(|t| format!("{t} [%]")),
            );
            doc.push((format!("[Channel {}]", i + 1), ch));
        }

        // Last, and in this order: an event is read as the three lines
        // `[Event n]`, `Event Contents`, `Event Timer`, which is how the
        // export writes them and how they are read back out.
        for (i, (contents, ms)) in self.events.iter().enumerate() {
            let mut ev = Rows::default();
            ev.set("Event Contents", Some(contents.clone()));
            ev.set("Event Timer", Some(format!("{ms:.6}[ms]")));
            doc.push((format!("[Event {}]", i + 1), ev));
        }

        let mut out = String::new();
        for (name, rows) in doc {
            // A header with nothing under it is not metadata; it reads like a
            // value that went missing on the way here.
            if rows.0.is_empty() {
                continue;
            }
            out.push_str(&row(&name, ""));
            for (key, value) in rows.0 {
                out.push_str(&row(key, &value));
            }
        }
        (!out.is_empty()).then_some(out)
    }

    /// `XY`, `XYT`, `XYZ`, `XYZT` — the axes that were actually recorded, for a
    /// stack of `frames` time points.
    fn scan_mode(&self, frames: usize) -> Option<String> {
        self.width?;
        let mut mode = String::from("XY");
        if self.slices.is_some_and(|n| n > 1) {
            mode.push('Z');
        }
        if frames > 1 {
            mode.push('T');
        }
        Some(mode)
    }
}

/// `2025-07-28T15:36:14.925-04:00` → `07/28/2025 03:36:14.925 PM`, the way the
/// export states it. `None` if the timestamp is not in the shape the schema
/// uses, since a half-converted date is worse than none.
fn fluoview_date(iso: &str) -> Option<String> {
    let b = iso.as_bytes();
    // yyyy-mm-ddThh:mm:ss
    if b.len() < 19 || b[4] != b'-' || b[7] != b'-' || b[10] != b'T' || b[13] != b':' {
        return None;
    }
    let num = |a: usize, z: usize| iso.get(a..z)?.parse::<u32>().ok();
    let (y, mo, d) = (num(0, 4)?, num(5, 7)?, num(8, 10)?);
    let (h, mi, s) = (num(11, 13)?, num(14, 16)?, num(17, 19)?);
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) || h > 23 || mi > 59 || s > 60 {
        return None;
    }
    // Milliseconds, kept to three digits and defaulted rather than omitted, so
    // the field is the same width whether or not the clock reported them.
    let frac = iso
        .get(19..)
        .filter(|f| f.starts_with('.'))
        .map(|f| {
            f[1..]
                .chars()
                .take_while(char::is_ascii_digit)
                .collect::<String>()
        })
        .unwrap_or_default();
    let ms: String = frac.chars().chain("000".chars()).take(3).collect();
    let (hour12, meridiem) = match h {
        0 => (12, "AM"),
        1..=11 => (h, "AM"),
        12 => (12, "PM"),
        _ => (h - 12, "PM"),
    };
    Some(format!(
        "{mo:02}/{d:02}/{y:04} {hour12:02}:{mi:02}:{s:02}.{ms} {meridiem}"
    ))
}

#[cfg(test)]
#[path = "meta_tests.rs"]
mod tests;
