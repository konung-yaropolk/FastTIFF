use super::*;

#[test]
fn the_view_is_reduced_while_it_moves_and_sharp_once_it_settles() {
    let moving = crate::render::MOVING_DOWNSCALE;
    assert_eq!(downscale_after(None, moving), 1, "never moved");
    assert_eq!(downscale_after(Some(0.0), moving), moving);
    assert_eq!(downscale_after(Some(0.1), moving), moving);
    assert_eq!(
        downscale_after(Some(super::super::VOLUME_SETTLE_S), moving),
        1
    );
    assert_eq!(downscale_after(Some(5.0), moving), 1);
    // A clock that ran backwards is not a movement.
    assert_eq!(downscale_after(Some(-1.0), moving), 1);
}

#[test]
fn the_resolution_while_moving_is_the_users_choice() {
    // Full, 50% and 25% in the settings.
    assert_eq!(downscale_after(Some(0.0), 1), 1);
    assert_eq!(downscale_after(Some(0.0), 2), 2);
    assert_eq!(downscale_after(Some(0.0), 4), 4);
    // Nothing smaller than full, whatever is asked for.
    assert_eq!(downscale_after(Some(0.0), 0), 1);
}
