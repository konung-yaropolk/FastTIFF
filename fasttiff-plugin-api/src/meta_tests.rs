use super::*;

fn stem_of(name: &str) -> String {
    StackInfo {
        name: name.into(),
        ..Default::default()
    }
    .stem()
    .to_string()
}

#[test]
fn the_stem_is_the_name_without_its_extension() {
    assert_eq!(stem_of("cells.tif"), "cells");
    assert_eq!(stem_of("cells.TIFF"), "cells");
    assert_eq!(stem_of("scan.oir"), "scan");
    assert_eq!(stem_of("run.2.tif"), "run.2");
}

#[test]
fn two_part_extensions_go_whole() {
    assert_eq!(stem_of("cells.ome.tif"), "cells");
    assert_eq!(stem_of("cells.OME.TIFF"), "cells");
    assert_eq!(stem_of("brain.nii.gz"), "brain");
}

/// Nothing that only looks like an extension is taken off.
#[test]
fn a_name_without_an_extension_is_left_alone() {
    assert_eq!(stem_of("cells"), "cells");
    assert_eq!(stem_of(".hidden"), ".hidden");
    assert_eq!(stem_of("trial 1.5 mM"), "trial 1.5 mM");
    assert_eq!(stem_of("data.long_ext"), "data.long_ext");
    assert_eq!(stem_of(".ome.tif"), ".ome");
    assert_eq!(stem_of(""), "");
}
