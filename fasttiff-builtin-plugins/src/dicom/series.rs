//! Gathering a slice series from a folder, and putting it in order.
//!
//! A scanner writes one DICOM per slice. Opening one of them shows one slice,
//! so this collects its siblings — and the ordering is the part worth being
//! careful about, because there are three tags that look like they say what
//! order the slices go in and only one of them does.
//!
//! # Why not `InstanceNumber`
//!
//! `(0020,0013)` is a counter, and what it counts is up to whatever wrote the
//! file. It is usually acquisition order, which is usually spatial order — and
//! on an interleaved acquisition it is not, because the scanner collected
//! every other slice on the way down and filled in the rest on the way back.
//! Sorting by it then produces a volume that is interleaved: scrolling through
//! it jumps alternate slices and the 3-D view is a comb. It looks like a
//! plausible volume of a different anatomy.
//!
//! # Why not the filename
//!
//! `IM_2` sorts after `IM_10` in any lexicographic order, so a series numbered
//! without zero-padding comes out shuffled. Numeric parsing fixes that case and
//! not the general one, since the number in the name is whatever the exporter
//! felt like putting there.
//!
//! # What does work
//!
//! `ImagePositionPatient` `(0020,0032)` is the position of the slice's first
//! voxel in the patient's frame, in millimetres, and
//! `ImageOrientationPatient` `(0020,0037)` gives the two direction cosines of
//! the image plane. Their cross product is the slice normal, and each slice's
//! **distance along that normal** is a real coordinate: it is monotonic through
//! the stack by construction, whatever order the scanner wrote the files in,
//! and it is in the units the spacing is in.
//!
//! That is what this sorts by. `InstanceNumber` is the fallback for a file that
//! does not carry the geometry, and the filename is the fallback for that.

use super::parse::{Found, Tag};
use super::read_one;
use fasttiff_plugin_api::ImportHost;
use std::path::{Path, PathBuf};

pub(super) const SERIES_UID: Tag = (0x0020, 0x000E);
pub(super) const INSTANCE_NUMBER: Tag = (0x0020, 0x0013);
pub(super) const IMAGE_POSITION: Tag = (0x0020, 0x0032);
pub(super) const IMAGE_ORIENTATION: Tag = (0x0020, 0x0037);

/// How many files in a folder will be looked at before giving up on gathering.
///
/// A study folder holds hundreds; a download folder can hold hundreds of
/// thousands, and every candidate costs a file open. The cap is what keeps
/// opening one DICOM that happens to live somewhere busy from stalling the
/// window — and when it is hit, the file is read on its own and the user is
/// told, rather than a partial series being assembled silently.
const MAX_CANDIDATES: usize = 20_000;

/// Where a slice sits along the stack, and how sure we are of it.
///
/// Ordered so that the better keys sort as better: every slice with a real
/// position is compared on that, and the ones without fall back together.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Along {
    /// Millimetres along the slice normal. The real answer.
    Normal(f64),
    /// `InstanceNumber`, for a file with no geometry.
    Instance(f64),
    /// Nothing to go on; the filename decides.
    Unknown,
}

/// The slice normal from the six direction cosines, as `(row × column)`.
///
/// `ImageOrientationPatient` is the row direction then the column direction,
/// each a unit vector in patient coordinates. Their cross product is
/// perpendicular to the image plane, which is the axis a stack of parallel
/// slices advances along.
fn normal(orientation: &[f64]) -> Option<[f64; 3]> {
    let o: &[f64; 6] = orientation.first_chunk::<6>()?;
    let (r, c) = ([o[0], o[1], o[2]], [o[3], o[4], o[5]]);
    let n = [
        r[1] * c[2] - r[2] * c[1],
        r[2] * c[0] - r[0] * c[2],
        r[0] * c[1] - r[1] * c[0],
    ];
    // A degenerate orientation — two parallel cosines, or a vector of zeros
    // from a file that wrote the tag empty — gives a normal of no length, and
    // projecting onto it would make every slice's position zero. That is worse
    // than having no geometry at all, because it looks like geometry.
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    (len > 1e-6 && len.is_finite()).then(|| [n[0] / len, n[1] / len, n[2] / len])
}

/// Where this slice sits along the stack.
fn along(f: &Found) -> Along {
    let position = f.nums(IMAGE_POSITION);
    let orientation = f.nums(IMAGE_ORIENTATION);
    if let (Some(p), Some(n)) = (position.first_chunk::<3>(), normal(&orientation)) {
        let d = p[0] * n[0] + p[1] * n[1] + p[2] * n[2];
        if d.is_finite() {
            return Along::Normal(d);
        }
    }
    match f.num(INSTANCE_NUMBER) {
        Some(v) if v.is_finite() => Along::Instance(v),
        _ => Along::Unknown,
    }
}

/// A gathered candidate.
struct Candidate {
    path: PathBuf,
    along: Along,
}

/// Every DICOM in `path`'s folder belonging to the same series, in slice order.
///
/// Always returns at least `path` itself, so a caller never has to handle an
/// empty series: a folder that cannot be listed, or holds nothing else, simply
/// yields the one file that was asked for.
pub(super) fn gather(path: &Path, first: &Found, host: &mut dyn ImportHost) -> Vec<PathBuf> {
    let only = || vec![path.to_path_buf()];

    // Without a series UID there is nothing to match siblings against.
    // Gathering on "every DICOM in the folder" instead would merge a
    // localiser, a T1 and a diffusion run into one stack of mixed geometry.
    let Some(uid) = first.text(SERIES_UID) else {
        host.log(
            "this file has no Series Instance UID, so its folder cannot be \
             searched for the rest of the series — reading it alone",
        );
        return only();
    };
    let Some(dir) = path.parent() else {
        return only();
    };
    let Ok(entries) = std::fs::read_dir(dir) else {
        return only();
    };

    let mut found = Vec::new();
    let mut seen = 0usize;
    let mut other_series = 0usize;
    for entry in entries.flatten() {
        seen += 1;
        if seen > MAX_CANDIDATES {
            host.log(&format!(
                "this folder holds more than {MAX_CANDIDATES} files, which is \
                 too many to search for the series — reading the chosen file \
                 alone"
            ));
            return only();
        }
        let p = entry.path();
        if !p.is_file() {
            continue;
        }
        // The file being opened is read by the caller, which already has it in
        // memory; it is added below so it is in the list exactly once however
        // the directory enumerates it.
        if p == path {
            continue;
        }
        // Read every file, not only the ones named `.dcm`. A scanner export is
        // as likely to be `IM_0001` or `1.2.840.…` with no extension at all,
        // and requiring the extension would collect a tenth of a series and
        // show it as the whole thing — which is the failure mode with no
        // symptom. A file that is not DICOM is rejected by `read_one` on its
        // magic, which costs one open.
        let Ok(slice) = read_one(&p) else {
            continue;
        };
        match slice.found.text(SERIES_UID) {
            Some(u) if u == uid => found.push(Candidate {
                path: p,
                along: along(&slice.found),
            }),
            Some(_) => other_series += 1,
            None => {}
        }
    }

    found.push(Candidate {
        path: path.to_path_buf(),
        along: along(first),
    });

    // Filename as the last resort, applied first so that the sort below is
    // stable on top of something meaningful rather than on directory order —
    // which is not even reproducible between two runs on some filesystems.
    found.sort_by(|a, b| a.path.cmp(&b.path));
    found.sort_by(|a, b| order(a.along, b.along));

    if found.len() > 1 {
        host.log(&format!(
            "gathered {} slices of this series{}",
            found.len(),
            match other_series {
                0 => String::new(),
                n => format!(", leaving {n} file(s) from other series"),
            }
        ));
        if let Some(what) = ordered_by(&found) {
            host.log(what);
        }
    }
    found.into_iter().map(|c| c.path).collect()
}

/// Compare two positions along the stack.
///
/// `total_cmp`, not `partial_cmp`: the values are read from a file and a NaN
/// would make the comparison inconsistent, which is a contract violation
/// `sort_by` is entitled to panic on. `Along::along` filters non-finite values
/// out already, and this is the belt to that braces — the cost is nothing and
/// the failure it prevents is a crash on a malformed file.
fn order(a: Along, b: Along) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a, b) {
        (Along::Normal(x), Along::Normal(y)) => x.total_cmp(&y),
        (Along::Instance(x), Along::Instance(y)) => x.total_cmp(&y),
        (Along::Unknown, Along::Unknown) => Ordering::Equal,
        // A slice with real geometry sorts before one without, so a series
        // where one file is missing its tags keeps the rest in proper order and
        // puts the odd one at the end, instead of having the whole order
        // decided by the worst file in it.
        (Along::Normal(_), _) => Ordering::Less,
        (_, Along::Normal(_)) => Ordering::Greater,
        (Along::Instance(_), _) => Ordering::Less,
        (_, Along::Instance(_)) => Ordering::Greater,
    }
}

/// What the order was actually decided by, for the log.
///
/// Worth saying out loud. "Ordered by instance number" is the user's cue that
/// an interleaved acquisition may have come out interleaved, which is the one
/// failure of this module that produces a plausible-looking volume.
fn ordered_by(found: &[Candidate]) -> Option<&'static str> {
    let any = |p: fn(&Along) -> bool| found.iter().any(|c| p(&c.along));
    if any(|a| matches!(a, Along::Normal(_))) {
        if any(|a| !matches!(a, Along::Normal(_))) {
            Some(
                "ordered by position along the slice normal; some files carried \
                 no position and are at the end",
            )
        } else {
            None // The normal case, and not worth a line.
        }
    } else if any(|a| matches!(a, Along::Instance(_))) {
        Some(
            "these files carry no slice position, so they are ordered by \
             instance number — an interleaved acquisition may come out \
             interleaved",
        )
    } else {
        Some("these files carry no slice order at all, so they are in name order")
    }
}

#[cfg(test)]
#[path = "series_tests.rs"]
pub(crate) mod tests;
