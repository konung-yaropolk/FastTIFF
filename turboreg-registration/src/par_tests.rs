//! The same four shapes the crate actually uses, exercised through whichever
//! implementation this build selected.
//!
//! These are the only tests in the crate that are *about* the shim rather than
//! about registration, and they are worth having for one reason: with `threads`
//! on they assert that rayon's prelude still covers every name the call sites
//! need, and with it off they assert that the sequential stand-ins type-check
//! and agree. The numbers are deliberately trivial — what is under test is that
//! the expression compiles and preserves order, not the arithmetic.

use super::*;

#[test]
fn mapping_over_a_slice_keeps_index_order() {
    let v: Vec<u32> = (0..64).collect();
    let doubled: Vec<u32> = v.par_iter().map(|&x| x * 2).collect();
    assert_eq!(doubled, (0..64).map(|x| x * 2).collect::<Vec<u32>>());
}

/// The shape in `nonrigid::measure_blocks_batch`: two slices chunked the same
/// way and zipped against a per-worker scratch set. If chunking did not line up
/// index for index, a frame would be measured against another frame's shift.
#[test]
fn chunks_zip_against_each_other_and_against_a_scratch_set() {
    let frames: Vec<u32> = (0..9).collect();
    let shifts: Vec<u32> = (100..109).collect();
    let mut sets: Vec<u32> = vec![0; 3];
    let chunk = frames.len().div_ceil(sets.len()).max(1);

    let paired: Vec<Vec<(u32, u32)>> = frames
        .par_chunks(chunk)
        .zip(shifts.par_chunks(chunk))
        .zip(sets.par_iter_mut())
        .map(|((f, s), set)| {
            *set = f.len() as u32;
            f.iter().zip(s).map(|(&a, &b)| (a, b)).collect()
        })
        .collect();

    let flat: Vec<(u32, u32)> = paired.into_iter().flatten().collect();
    assert_eq!(
        flat,
        (0..9).map(|i| (i, i + 100)).collect::<Vec<(u32, u32)>>(),
        "every frame must stay paired with its own shift"
    );
    assert_eq!(sets, vec![3, 3, 3], "each set saw its own chunk");
}

/// The shape in `pipeline::measure_batch`: a chunk is handed one FFT plan and
/// the results of the whole chunk are flattened back into frame order.
#[test]
fn flat_map_iter_flattens_in_order() {
    let v: Vec<u32> = (0..10).collect();
    let out: Vec<u32> = v
        .par_chunks(3)
        .flat_map_iter(|chunk| chunk.iter().map(|&x| x + 1).collect::<Vec<_>>())
        .collect();
    assert_eq!(out, (1..11).collect::<Vec<u32>>());
}

/// The shape in `pipeline::apply_batch`: planes corrected in place, each by the
/// shift belonging to its own frame.
#[test]
fn mutating_in_place_zipped_against_a_second_slice() {
    let mut planes: Vec<u32> = vec![0; 8];
    let shifts: Vec<u32> = (0..8).map(|i| i * 10).collect();
    planes
        .par_iter_mut()
        .zip(shifts.par_iter())
        .for_each(|(p, &s)| *p = s + 1);
    assert_eq!(planes, (0..8).map(|i| i * 10 + 1).collect::<Vec<u32>>());
}

/// A hint, so the only thing to hold it to is that it is usable as a divisor.
#[test]
fn the_worker_count_is_at_least_one() {
    assert!(workers() >= 1);
    #[cfg(not(feature = "threads"))]
    assert_eq!(workers(), 1, "without threads there is exactly one worker");
}

/// `map_in_order` returns results in input order with the flag on or off —
/// the property every ordered reduction in this crate rests on.
#[test]
fn ordered_map_keeps_input_order_either_way() {
    let items: Vec<u64> = (0..1000).collect();
    let seq = map_in_order(&items, false, |&x| x * x + 1);
    let par = map_in_order(&items, true, |&x| x * x + 1);
    assert_eq!(seq, par);
    assert_eq!(seq[0], 1);
    assert_eq!(seq[999], 999 * 999 + 1);
}

/// Chunked writes cover every element exactly once, chunk indices count from
/// zero, and the last chunk is the short one.
#[test]
fn chunked_writes_cover_the_buffer_exactly_once() {
    for parallel in [false, true] {
        let mut buf = vec![0u32; 103];
        for_each_chunk_mut(&mut buf, 10, parallel, |i, c| {
            for (j, v) in c.iter_mut().enumerate() {
                *v += (i * 10 + j) as u32 + 1;
            }
        });
        let want: Vec<u32> = (1..=103).collect();
        assert_eq!(buf, want, "parallel = {parallel}");
    }
    // A chunk size of zero is treated as one rather than panicking.
    let mut buf = vec![0u8; 4];
    for_each_chunk_mut(&mut buf, 0, true, |i, c| c[0] = i as u8);
    assert_eq!(buf, vec![0, 1, 2, 3]);
}

/// The paired form hands chunk `i` of both buffers to the same call.
#[test]
fn paired_chunks_stay_in_step() {
    for parallel in [false, true] {
        let mut a = vec![0usize; 50];
        let mut b = vec![0usize; 50];
        for_each_chunk_pair_mut(&mut a, &mut b, 7, parallel, |i, ca, cb| {
            ca.iter_mut().for_each(|v| *v = i);
            cb.iter_mut().for_each(|v| *v = i * 2);
        });
        for k in 0..50 {
            assert_eq!(b[k], a[k] * 2, "chunk mismatch at {k}");
            assert_eq!(a[k], k / 7);
        }
    }
}
