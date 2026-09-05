//! Reading the acquisition record out of `ImageDescription` (tag 270).
//!
//! FastTIFF's OIR importer copies the FluoView sidecar `.txt` into tag 270
//! verbatim, and the ImageJ writer appends it after its own `key=value` block.
//! So the description of a converted file is a mixture: ImageJ's keys, then a
//! `"key"\t"value"` export, and possibly whatever else a previous tool left
//! behind.
//!
//! This reads the three things the analysis needs and **ignores every line that
//! does not match**. That is the important property, not an aside: the same
//! description carries `images=298` and `slices=1` from ImageJ, XML from other
//! sources, and free text, and a parser that guessed at unmatched lines would
//! find numbers that mean something else entirely.

/// What the acquisition record says about time.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Timing {
    /// Frames the acquisition states — not necessarily how many are in the
    /// stack, which is why the caller checks one against the other.
    pub frames: Option<usize>,
    /// Seconds from the first frame to the last.
    pub duration_s: Option<f64>,
    /// `(contents, seconds)` for each event marker, in file order.
    pub events: Vec<(String, f64)>,
}

impl Timing {
    /// Seconds per frame, as the acquisition states them.
    ///
    /// Duration over frame *count*, not over the gaps between frames. That is
    /// what the analysis this reproduces does, and the two differ by 0.3% at
    /// 300 frames — small, but it accumulates into a shift of a third of a
    /// frame by the end of a recording, which is the sort of thing the
    /// `sync_coef` correction exists to absorb. Changing it here would silently
    /// change every result.
    pub fn seconds_per_frame(&self) -> Option<f64> {
        match (self.frames, self.duration_s) {
            (Some(n), Some(d)) if n > 0 && d.is_finite() && d > 0.0 => Some(d / n as f64),
            _ => None,
        }
    }

    /// Parse whatever of it is present. Never fails: a description with none of
    /// these leaves every field empty, and the caller says what was missing.
    pub fn parse(description: &str) -> Timing {
        let mut t = Timing::default();
        let lines: Vec<&str> = description.lines().collect();

        for (i, line) in lines.iter().enumerate() {
            // `"T Dimension"\t"298, 0.000 - 322.701 [s], Interval FreeRun"`
            if let Some(value) = tagged_value(line, "T Dimension") {
                t.frames = first_number(value).map(|v| v as usize).filter(|n| *n > 0);
                t.duration_s = range_end(value);
            }
            // An event is three lines: the `[Event N]` header, its contents,
            // and its timer. Read forwards from the header rather than
            // matching the value lines on their own — `"Event Timer"` also
            // appears in sections that are not events.
            if is_section(line, "Event ") {
                let contents = lines
                    .get(i + 1)
                    .and_then(|l| tagged_value(l, "Event Contents"))
                    .unwrap_or("")
                    .to_string();
                if let Some(ms) = lines
                    .get(i + 2)
                    .and_then(|l| tagged_value(l, "Event Timer"))
                    .and_then(first_number)
                {
                    t.events.push((contents, ms / 1000.0));
                }
            }
        }
        t
    }
}

/// The value of a `"key"\t"value"` line, if that is what this line is.
fn tagged_value<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let (k, v) = line.split_once('\t')?;
    (k.trim().trim_matches('"') == key).then(|| v.trim().trim_matches('"'))
}

/// Whether the line is a `"[Section…]"` header.
fn is_section(line: &str, prefix: &str) -> bool {
    line.split('\t')
        .next()
        .map(|k| k.trim().trim_matches('"'))
        .is_some_and(|k| k.starts_with('[') && k[1..].starts_with(prefix))
}

/// The first number in `s`, ignoring everything around it.
pub fn first_number(s: &str) -> Option<f64> {
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let start = i;
        if b[i] == b'-' || b[i] == b'+' {
            i += 1;
        }
        let digits = i;
        while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'.') {
            i += 1;
        }
        if i > digits {
            if let Ok(v) = s[start..i].parse::<f64>() {
                return Some(v);
            }
        }
        i = (i + 1).max(start + 1);
    }
    None
}

/// The end of a `0.000 - 322.701 [s]` range: the number after the dash.
fn range_end(s: &str) -> Option<f64> {
    let (_, rest) = s.split_once('-')?;
    first_number(rest).filter(|v| v.is_finite() && *v > 0.0)
}

#[cfg(test)]
#[path = "meta_tests.rs"]
mod tests;
