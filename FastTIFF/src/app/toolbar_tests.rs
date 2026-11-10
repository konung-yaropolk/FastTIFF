//! What the toolbar says, where it is worked out apart from the layout.

use super::*;

/// The badge's hover text is one paragraph of prose. Its literal is split
/// over lines, and a line that loses its `\` continuation keeps the next
/// line's indent: a run of spaces in the middle of a sentence.
#[test]
fn the_subsampled_note_reads_as_one_paragraph() {
    let note = subsampled_note(4);
    assert!(note.contains("subsampled 4x on each axis"), "{note}");
    assert!(!note.contains("  "), "a run of spaces: {note:?}");
    assert!(!note.contains('\n'), "a line break: {note:?}");
    assert!(note.ends_with("not exact."), "{note}");
}
