//! The container: block keys, the walk, the arithmetic the move pinned, and the
//! metadata documents the one pass collects.

use super::*;
use crate::fixture::{self, Builder};
use crate::legacy;
use crate::source::Memory;

#[test]
fn plane_keys_sort_numerically_and_exclude_non_planes() {
    assert_eq!(plane_key("REF_LSM0_abc_0"), None);
    assert_eq!(plane_key("thumbnail"), None);
    assert_eq!(plane_key(""), None);
    assert_eq!(plane_key("t"), None);

    let k2 = plane_key("t2_0_1_uid").expect("t2");
    let k10 = plane_key("t10_0_1_uid").expect("t10");
    // `t002` already sorts correctly; the padding is what makes the *later*
    // fields sort numerically too.
    assert!(plane_key("t002_0_1_x") < plane_key("t010_0_1_x"));
    assert!(k2 < k10, "keys must order by number");
    assert!(plane_key("t001_0_2_x") < plane_key("t001_0_10_x"));
}

/// The walk stops at a block that is not all there, rather than emitting an
/// offset for it.
///
/// Checked on the walk's own output. End to end the half-written block is
/// harmless either way — the chunk read fails, the plane stays short and is
/// dropped — so an end-to-end test cannot tell the two apart, and the rule the
/// walk's doc comment states would go unpinned.
#[test]
fn the_walk_stops_at_a_block_that_is_not_all_there() {
    let f = fixture::stack_file(4, 4, &["t001_0_1_uid"], 16);
    let at = fixture::index_at(&f);
    let whole = walk_blocks(&mut Memory(&f[..at])).unwrap();
    assert!(whole.len() > 2, "the fixture should have several blocks");

    // One byte short of the last block's final byte.
    let cut = walk_blocks(&mut Memory(&f[..at - 1])).unwrap();
    assert_eq!(
        cut.len(),
        whole.len() - 1,
        "the block that is one byte short must not be offered"
    );
    assert_eq!(cut, whole[..whole.len() - 1]);
}

/// The guard on the metadata-only fixture the read tests rely on: it has to be
/// walkable, or the test that uses it never reaches the rule it is about — it
/// is the plane count that must decide, not the luck of a fixture too small to
/// walk.
#[test]
fn the_metadata_only_fixture_can_be_walked() {
    let f = fixture::metadata_only_file();
    assert!(
        walk_blocks(&mut Memory(&f)).unwrap().len() >= 4,
        "the fixture must be walkable, or it cannot exercise the guard"
    );
}

/// An index entry within eight of `u64::MAX` wraps past zero when the reader
/// checks it, and is kept — to be found later not to be in the file. That is
/// what the reader always did as a release build; a debug build used to panic
/// on the addition instead.
#[test]
fn an_index_entry_at_the_top_of_the_range_is_kept_without_panicking() {
    let mut f = fixture::stack_file(4, 4, &["t001_0_1_uid"], 16);
    let at = fixture::index_at(&f) + fixture::INDEX_HEADER + 8;
    f[at..at + 8].copy_from_slice(&(u64::MAX - 3).to_le_bytes());

    let (offsets, found) = block_offsets(&mut Memory(&f)).unwrap();
    let (old, old_found) = legacy::part_blocks(&f).unwrap();
    assert_eq!(found, BlockSource::Index);
    assert_eq!(old_found, legacy::Found::Index);
    assert_eq!(offsets, old);
    assert!(offsets.contains(&(u64::MAX - 3)), "{offsets:x?}");
    // And reading it is no more than a block skipped.
    crate::read_bytes(&[&f]).expect("one unreadable index entry is not fatal");
}

/// A descriptor whose name length is `u32::MAX` names nothing that is there,
/// and is skipped — on a 32-bit target the end of that name used to wrap.
#[test]
fn a_descriptor_with_an_impossible_name_length_is_skipped() {
    let mut b = Builder::new();
    b.xml(&fixture::size_record(4, 2));
    let mut d = Vec::new();
    d.extend(0u32.to_le_bytes());
    d.extend(16u32.to_le_bytes());
    d.extend(u32::MAX.to_le_bytes());
    d.extend(b"t001_0_1_uid");
    b.block(fixture::TYPE_DESCRIPTOR, &d);
    b.block(4, &[9u8; 16]);
    b.plane_chunk("t001_0_1_uid", 0, &[1u8; 16]);
    let f = b.finish();

    let acq = crate::read_bytes(&[&f]).expect("the good chunk still reads");
    assert_eq!(acq.planes().as_u16().unwrap()[0], vec![0x0101u16; 8]);
    let old = legacy::read(&[&f], false).expect("and did before");
    assert!(matches!(old.planes[0], legacy::PlaneData::U16(ref v) if *v == vec![0x0101u16; 8]));
}

/// A record stating an absurd slice count multiplies past `usize::MAX`
/// against the channel count. The wrapped product fails to divide the planes,
/// so the record is ignored, as it always was in a release build — where a
/// debug build used to panic on the multiplication.
#[test]
fn an_absurd_slice_count_in_the_record_is_ignored_without_panicking() {
    let names = [
        "t001_0_1_uid",
        "t002_0_1_uid",
        "t003_0_1_uid",
        "t004_0_1_uid",
    ];
    let f = fixture::channels_file_with(
        4,
        4,
        &names,
        16,
        2,
        "<commonimage:axis><commonimage:axis>ZSTACK</commonimage:axis>\
         <commonimage:step>1.0</commonimage:step>\
         <commonimage:maxSize>1e30</commonimage:maxSize></commonimage:axis>",
    );
    let acq = crate::read_bytes(&[&f]).expect("the planes are fine");
    assert_eq!(acq.record().slices(), Some(usize::MAX));
    let s = acq.shape();
    assert_eq!((s.channels, s.slices, s.frames), (1, 1, 4));
    let old = legacy::read(&[&f], false).unwrap();
    assert_eq!((old.channels, old.slices, old.frames), (1, 1, 4));
}

/// Every block a fixture holds, laid out so that metadata documents sit where
/// the one pass is most likely to miss them: in the data block of a
/// descriptor pair, which the descriptor loop steps over, and as the very last
/// block, which it never reaches.
fn xml_in_awkward_places() -> Vec<u8> {
    let mut b = Builder::new();
    b.xml(&fixture::size_record(4, 2));
    b.plane_chunk("t001_0_1_uid", 0, &[1u8; 16]);
    let event = "<?xml version=\"1.0\"?><event:eventList><event:event>\
                 <event:name>DRS</event:name><event:time>12.5</event:time>\
                 </event:event></event:eventList>";
    // A descriptor whose data block is a document. Named as a reference
    // image, so the bytes are not also a plane.
    b.plane_chunk("REF_LSM0_x_0", 0, event.as_bytes());
    b.plane_chunk("t002_0_1_uid", 0, &[2u8; 16]);
    // Two documents in one block, with padding between them, as a real
    // acquisition puts ten.
    let mut two = Vec::new();
    two.extend(b"\0\0\0\0padding");
    two.extend(fixture::size_record(4, 2).as_bytes());
    two.extend([0u8; 13]);
    two.extend(
        b"<?xml version=\"1.0\"?><fileinfo:fileInfomation><fileinfo:version>2.1.2.3\
          </fileinfo:version></fileinfo:fileInfomation>",
    );
    two.extend([0xFFu8; 7]);
    b.block(5, &two);
    // The last block of all.
    b.xml(
        "<?xml version=\"1.0\"?><lsmframe:frameProperties><commonframe:axisValue>\
         <commonframe:axisType>TIMELAPSE</commonframe:axisType>\
         <commonframe:position>40.0</commonframe:position></commonframe:axisValue>\
         </lsmframe:frameProperties>",
    );
    b.finish()
}

/// The documents the one pass collects are the documents the old separate
/// sweep found, in the same order, over the same block list — for every
/// fixture, and for every length each could be cut to.
#[test]
fn the_one_pass_collects_exactly_what_the_separate_sweep_did() {
    let mut corpus = fixture::corpus();
    corpus.push(("xml_in_awkward_places", xml_in_awkward_places()));
    let mut compared = 0usize;
    for (name, f) in &corpus {
        for cut in (0..=f.len()).rev() {
            let bytes = &f[..cut];
            let Ok((offsets, _)) = block_offsets(&mut Memory(bytes)) else {
                continue;
            };
            let mut planes = Reassembly::default();
            let mut docs = Vec::new();
            if read_part(
                &mut Memory(bytes),
                &offsets,
                &mut planes,
                &mut docs,
                &mut |_| true,
            )
            .is_err()
            {
                continue;
            }
            assert_eq!(
                docs,
                legacy::documents(bytes, &offsets),
                "{name} cut to {cut} bytes"
            );
            compared += 1;
        }
    }
    assert!(compared > 1000, "only {compared} cases compared");

    // And the awkward fixture's documents are really all there.
    let f = xml_in_awkward_places();
    let (offsets, _) = block_offsets(&mut Memory(&f)).unwrap();
    let mut docs = Vec::new();
    read_part(
        &mut Memory(&f),
        &offsets,
        &mut Reassembly::default(),
        &mut docs,
        &mut |_| true,
    )
    .unwrap();
    let roots: Vec<&str> = docs
        .iter()
        .map(|d| {
            let open = d[2..].find('<').unwrap() + 3;
            let close = open + d[open..].find('>').unwrap();
            &d[open..close]
        })
        .collect();
    assert_eq!(
        roots,
        [
            "lsmimage:imageProperties",
            "event:eventList",
            "lsmimage:imageProperties",
            "fileinfo:fileInfomation",
            "lsmframe:frameProperties"
        ]
    );
}

/// Documents split from the padding of the block they share: each from its
/// declaration to its own last `>`.
#[test]
fn documents_are_cut_free_of_the_padding_between_them() {
    let text = "<?xml a?><x>1</x>\u{0}\u{0}junk<?xml b?><y/>trailing";
    assert_eq!(
        split_documents(text),
        ["<?xml a?><x>1</x>", "<?xml b?><y/>"]
    );
    assert!(split_documents("no declaration here").is_empty());
    assert!(split_documents("<?xml with no close").is_empty());
}
