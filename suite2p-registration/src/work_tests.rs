use super::*;

/// At the default sample size the pick is a substantial share of building the
/// reference — the share that used to go unreported entirely.
#[test]
fn the_pick_is_a_real_share_of_the_reference() {
    let share = pick(300) / reference(300, 8);
    assert!(
        (0.2..0.8).contains(&share),
        "the pick is {share:.2} of the reference work"
    );
}

/// The pick grows as the square of the sample; the passes grow linearly.
#[test]
fn the_pick_grows_faster_than_the_passes() {
    let small = pick(100) / reference(100, 8);
    let large = pick(1000) / reference(1000, 8);
    assert!(large > small, "{small} then {large}");
}

#[test]
fn an_empty_sample_is_no_work() {
    assert_eq!(pick(0), 0.0);
    assert_eq!(reference(0, 8), 0.0);
}
