//! How bright the 3-D render comes out, which is a different question in each
//! compositing mode.
//!
//! Three modes, three answers, and the reason they cannot share one is that
//! they do not accumulate the same thing:
//!
//! * **Max intensity** takes the brightest sample along each ray and sums the
//!   channels. The result is independent of how far the ray travelled, so there
//!   is nothing to compensate — but summing channels, and taking a *maximum*
//!   rather than a mean, tends to come out hotter than the 2-D view of the same
//!   data. So this mode gets a plain gain, with a default below 1 because that
//!   is where the picture actually looks right.
//! * **Volume** is emission-absorption: each sample contributes
//!   `1 - exp(-density · intensity · dt)` of opacity, so the total depends on
//!   the density the user is dragging. Dragging it therefore changes the
//!   *brightness* as well as the solidity, which is not what the control is for.
//!   [`autolevel`] cancels that.
//! * **Surface** returns a shaded hit and never accumulates. It has its own
//!   fixed shading constants and no gain.
//!
//! Kept apart from the shader and from the interface for the usual reason: it
//! is arithmetic with a feel, it is tuned by reading numbers rather than by
//! watching a picture, and it can be held to its promises without a GPU.

/// The max-intensity gain a fresh view starts at.
///
/// Below 1 deliberately. A MIP sums the per-channel maxima along each ray, and
/// a maximum over a few hundred samples is a high-order statistic — on anything
/// with noise in it, the brightest sample along a ray is well above the tissue
/// it passes through. The result reads as blown out next to the same data in
/// 2-D, which is the complaint this default answers.
///
/// 0.78 rather than a round number because it is the point at which a 16-bit
/// MRI with its ordinary auto-contrast window stops clipping its highlights.
pub const MIP_DEFAULT: f32 = 0.78;

/// Checked here rather than in a test because both sides are constants: a test
/// would be asserting something the compiler already knows, and clippy says so.
/// The range is the claim — dim a little, not a lot — and a build that breaks it
/// should not link.
const _: () = assert!(
    MIP_DEFAULT > 0.5 && MIP_DEFAULT < 1.0,
    "the max-intensity default should dim a little, not a lot"
);

/// The density a fresh view starts at, and the point [`autolevel`] holds
/// brightness steady *at*.
///
/// Having the reference be the default is what makes this change invisible
/// until the slider is moved: at the default density the gain is exactly 1, so
/// a volume opens looking as it always has, and only dragging shows the
/// difference.
pub const DENSITY_DEFAULT: f32 = 100.0;

/// How far [`autolevel`] may move the brightness, in either direction.
///
/// This is the one number here with a real trade behind it, and the trade is
/// worth writing down because the obvious choice — enough gain to flatten the
/// response completely — is the wrong one.
///
/// The correction assumes a [typical thickness](TYPICAL_THICKNESS). A volume
/// that is much *more* opaque than typical was already near-saturated across
/// the whole slider, so correcting it as though it were not overshoots. The cap
/// is what bounds that overshoot. Measured across the slider's 1..1000, as a
/// ratio between the brightest and darkest result:
///
/// ```text
///   cap    typical (τ=0.2)   thin (τ=0.02)   opaque (τ=2.0)
///   none         5.5x            50.5x            1.2x
///   2.5          2.2x            20.2x            2.2x
///   3.0          1.8x            16.8x            2.6x
///   6.0          1.0x             9.2x            4.8x
/// ```
///
/// 2.5 is the balanced row: it more than halves the variation on an ordinary
/// volume while keeping the overshoot on an opaque one to about the same
/// figure. Flattening the typical case completely would nearly double the harm
/// to the opaque one, which is a bad trade for a control whose whole purpose is
/// to stop the brightness moving.
///
/// It is also why the manual gain exists beside this: a volume the assumption
/// does not fit has a slider of its own rather than a worse automatic answer.
const MAX_GAIN: f32 = 2.5;

/// A volume's representative optical thickness, in the shader's own units.
///
/// This is the one tuned number here, and it is tuned rather than derived
/// because the real value depends on the data: it is the sum of windowed
/// intensity along a ray, which for a head MRI is not what it is for a sparse
/// vessel tree. What it controls is only *where the knee sits* — the curve has
/// the same shape either way — so being approximately right flattens the
/// response across the slider, and being exactly right is not available without
/// sampling the volume on the host.
///
/// 0.2 comes from the geometry: the box is one unit across, a ray crosses it in
/// about one unit of `dt` summed, and the mean windowed intensity of the part
/// of a scan that is not air runs about a fifth of full scale.
const TYPICAL_THICKNESS: f32 = 0.2;

/// The gain that holds the Volume mode's brightness steady as `density` moves.
///
/// # Where this comes from
///
/// The shader accumulates opacity front to back as
/// `acc = 1 - exp(-density · τ)`, where τ is the ray's integrated intensity.
/// The picture's brightness is roughly `acc` times the colour of what the ray
/// passed through — so brightness rises from nearly nothing at low density and
/// saturates once `density · τ` is large. That is exactly the complaint: the
/// volume goes dark at one end of the slider and stops changing at the other,
/// and in between the brightness is doing the moving rather than the solidity.
///
/// Dividing out that factor leaves the density controlling only how solid the
/// volume looks:
///
/// ```text
///     gain(d) = (1 - exp(-d_ref · τ)) / (1 - exp(-d · τ))
/// ```
///
/// which is 1 at the reference density by construction.
///
/// # What it cannot do
///
/// Hold brightness *exactly* constant, because τ is the data's and this uses a
/// typical value. On a volume much thinner or thicker than typical the
/// compensation is partial — flatter than no compensation, not flat. Saying so
/// is the honest version; measuring τ per volume would mean sampling it on the
/// host every time the slider moved.
pub fn autolevel(density: f32) -> f32 {
    if !density.is_finite() || density <= 0.0 {
        return 1.0;
    }
    let opacity = |d: f32| 1.0 - (-d * TYPICAL_THICKNESS).exp();
    let reference = opacity(DENSITY_DEFAULT);
    let here = opacity(density);
    if here <= 1e-6 {
        return MAX_GAIN;
    }
    (reference / here).clamp(1.0 / MAX_GAIN, MAX_GAIN)
}

#[cfg(test)]
#[path = "brightness_tests.rs"]
mod tests;
