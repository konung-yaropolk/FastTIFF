//! What a file handed to a browser's downloads is called.
//!
//! A tab cannot ask where a file goes, so the name is decided here rather
//! than by the user, and the browser decides where. String handling, and the
//! part most likely to be wrong on a file called `scan.ome.tif` or on one
//! called nothing at all — so it is compiled, and tested, on every target,
//! although only [`Target::Browser`](super::Target::Browser) uses the answers.

/// The name to suggest for a download of `shown_as`.
///
/// `shown_as` is what the window is titled with, which for a browser document
/// is the name it arrived under and for a plugin's result is the name the
/// plugin gave it. Either way it is the name the user is looking at, so it is
/// the one to offer back — with a `.tif` on it, because what is being written
/// is a TIFF whatever the source was called.
pub(in crate::app) fn suggested_name(shown_as: &str) -> String {
    let trimmed = shown_as.trim();
    // Everything before the first dot, not the last: `scan.ome.tif` is an
    // OME-TIFF and its stem is `scan`, where taking the last would leave
    // `scan.ome` and produce `scan.ome.tif` again — which is right by accident
    // here and wrong for `scan.ome.tiff`.
    let stem = trimmed.split('/').next_back().unwrap_or(trimmed);
    let stem = stem.split('\\').next_back().unwrap_or(stem);
    let stem = stem.split('.').next().unwrap_or(stem).trim();
    if stem.is_empty() {
        "stack.tif".to_string()
    } else {
        format!("{stem}.tif")
    }
}

/// The download name for a path a *plugin* chose.
///
/// Unlike [`suggested_name`], this keeps whatever the plugin called the file,
/// extension and all. A plugin that asks the host to write `derivative.tif`
/// gets `derivative.tif` on both platforms, which is the point: the name is the
/// plugin's output, not the host's suggestion, and a web build that quietly
/// renamed it would make the same plugin produce two different results.
///
/// Only the last component survives. A browser download cannot choose a
/// directory, and a `/` left in the `download` attribute is ignored by some
/// browsers and rewritten by others — so the directory is dropped deliberately
/// rather than passed along to be mangled.
pub(in crate::app) fn name_from_path(path: &str) -> String {
    let last = path
        .split(['/', '\\'])
        .next_back()
        .unwrap_or(path)
        .trim()
        .trim_end_matches('.');
    if last.is_empty() {
        "result.tif".to_string()
    } else {
        last.to_string()
    }
}

#[cfg(test)]
#[path = "naming_tests.rs"]
mod tests;
