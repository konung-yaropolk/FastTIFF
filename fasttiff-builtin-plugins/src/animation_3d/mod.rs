//! 3D animation: turn the volume in the 3D view, and save one turn as a movie.
//!
//! Three motions, all about the same point: the middle of the stretch of
//! volume under the centre of the view, as the view is when Play is pressed.
//! So the volume turns in place in the middle of the frame, even after a pan or
//! a zoom has moved it away from its own centre:
//!
//! * **Rotate about X** — a tumble about the image's horizontal axis, top over
//!   bottom.
//! * **Rotate about Y** — a turntable about the image's vertical axis, the
//!   classic way to show a z-stack has depth.
//! * **Precession** — the volume's z axis, the axis it is stacked along,
//!   tilted by an angle (15° unless set otherwise) and swung round in a circle.
//!   The volume wobbles rather than turns: it leans towards each side in turn
//!   and never spins about its own axis, so a z-stack seen from the front stays
//!   the right way up while its depth shows. The same motion ImageJ's
//!   rocking projections approximate, done continuously.
//!
//! The dialog is a panel rather than a form: Play (which becomes Stop while the
//! animation runs), Save, and the settings both use. The view is the user's the
//! moment they want it back — a drag, the wheel, a key, or going back to 2D
//! stops the animation, and the host does that by itself.
//!
//! # One turn, no seam
//!
//! A full cycle is cut into as many frames as the speed takes to cover it at
//! [`FPS`] frames a second, and the cycle's end — which is its start again — is
//! not one of them. Played, it loops without a stutter; saved, it is a movie in
//! which every frame is a different orientation and the last leads straight
//! back into the first. See [`fasttiff_plugin_api::animation`].
//!
//! # What it is written against
//!
//! Nothing but the plugin contract's animation API: this plugin never touches
//! the camera or the renderer. It computes the poses and the host plays them,
//! so it would work unchanged as a shared library.

use fasttiff_plugin_api::{
    Animation, HostContext, MovieResolution, Outcome, ParamDecl, ParamKind, Params, Plugin,
    PluginError, PluginInfo, Pose,
};

/// The frame rate the animation is computed at — on screen, and in the movie a
/// save writes, so the two run at the same speed.
pub const FPS: f32 = 30.0;

/// The motions, in the order the dialog offers them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Motion {
    RotateX,
    RotateY,
    Precession,
}

impl Motion {
    const ALL: [Motion; 3] = [Motion::RotateX, Motion::RotateY, Motion::Precession];

    fn label(self) -> &'static str {
        match self {
            Motion::RotateX => "Rotate about X",
            Motion::RotateY => "Rotate about Y",
            Motion::Precession => "Precession",
        }
    }

    fn from_choice(i: usize) -> Motion {
        Motion::ALL.get(i).copied().unwrap_or(Motion::RotateY)
    }

    /// The pose at `phase` radians into the cycle.
    ///
    /// For precession, `tilt` is how far the z axis leans, in radians. The
    /// lean is a turn about an axis lying in the image plane and pointing at
    /// `phase`, so the z axis sweeps a cone of that half-angle while the volume
    /// itself never turns about z.
    pub fn pose(self, phase: f32, tilt: f32) -> Pose {
        match self {
            Motion::RotateX => Pose::rotation([1.0, 0.0, 0.0], phase),
            Motion::RotateY => Pose::rotation([0.0, 1.0, 0.0], phase),
            Motion::Precession => Pose::rotation([phase.cos(), phase.sin(), 0.0], tilt),
        }
    }
}

/// Defaults, and the bounds the dialog declares.
const DEFAULT_MOTION: usize = 1;
const DEFAULT_SPEED: f64 = 30.0;
const SPEED: (f64, f64) = (1.0, 360.0);
const DEFAULT_TILT: f64 = 15.0;
const TILT: (f64, f64) = (1.0, 80.0);
/// The sizes a saved movie can be, in the order the dialog offers them.
const RESOLUTIONS: [(&str, MovieResolution); 2] = [
    ("Window resolution", MovieResolution::Window),
    ("Max resolution", MovieResolution::Max),
];
/// Max: the most a saved movie can show, which is what saving one is for.
const DEFAULT_RESOLUTION: usize = 1;

/// The size the dialog asks a saved movie to be.
fn resolution(params: &Params) -> MovieResolution {
    let i = params.choice("resolution", DEFAULT_RESOLUTION);
    RESOLUTIONS
        .get(i)
        .unwrap_or(&RESOLUTIONS[DEFAULT_RESOLUTION])
        .1
}

/// How many frames one cycle takes at `speed` degrees a second.
///
/// At least two: a "cycle" of one frame is a still picture, and a saved movie
/// of it would be a single frame claiming to be an animation.
pub fn frames_per_cycle(speed_deg_per_s: f64) -> usize {
    let speed = speed_deg_per_s.clamp(SPEED.0, SPEED.1);
    ((360.0 / speed) * FPS as f64).round().max(2.0) as usize
}

/// One cycle of `motion` at `speed` degrees a second.
pub fn animation(motion: Motion, speed_deg_per_s: f64, tilt_deg: f64) -> Animation {
    let tilt = (tilt_deg.clamp(TILT.0, TILT.1) as f32).to_radians();
    Animation::cycle(frames_per_cycle(speed_deg_per_s), FPS, |phase| {
        motion.pose(phase, tilt)
    })
}

/// Play, save, or stop an animation of the volume in the 3D view.
#[derive(Default)]
pub struct Animation3d;

impl Plugin for Animation3d {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("dev.fasttiff.animation-3d", "3D animation")
            .menu_path("Stack")
            .version(env!("CARGO_PKG_VERSION"))
            .author("FastTIFF")
            .description(
                "Turn the volume in the 3D view about X or Y, or wobble it in a precession; \
                 play it, or save one turn as a TIFF movie.",
            )
    }

    fn params(&self, host: &dyn HostContext) -> Vec<ParamDecl> {
        let pending = host.pending_params();
        let motion = Motion::from_choice(pending.choice("motion", DEFAULT_MOTION));
        let speed = pending.float("speed", DEFAULT_SPEED);
        let frames = frames_per_cycle(speed);

        let mut d = vec![ParamDecl::new(
            "motion",
            "Motion",
            ParamKind::Choice {
                default: DEFAULT_MOTION,
                options: Motion::ALL.iter().map(|m| m.label().to_string()).collect(),
            },
        )
        .help(
            "About the volume's own axes: X along the image's width, Y along its height. \
             Precession leans the Z axis — the one the stack is stacked along — and swings \
             it round a cone.",
        )];
        if motion == Motion::Precession {
            d.push(
                ParamDecl::new(
                    "tilt",
                    "Tilt (°)",
                    ParamKind::Float {
                        default: DEFAULT_TILT,
                        min: TILT.0,
                        max: TILT.1,
                    },
                )
                .help("How far the Z axis leans from where it started."),
            );
        }
        d.push(
            ParamDecl::new(
                "speed",
                "Speed (°/s)",
                ParamKind::Float {
                    default: DEFAULT_SPEED,
                    min: SPEED.0,
                    max: SPEED.1,
                },
            )
            .help("How fast the volume turns, or the tilted axis goes round."),
        );
        d.push(ParamDecl::new(
            "cycle",
            format!(
                "One cycle: {frames} frames, {:.1} s at {FPS:.0} frames/s",
                frames as f64 / FPS as f64
            ),
            ParamKind::Label,
        ));
        // One button that is Play or Stop, as the animation is stopped or
        // running. Two keys rather than one with a changing label, so a press
        // that arrives after the state has changed underneath it — the user
        // grabbed the view a moment before clicking Stop — still means what
        // the button said when it was clicked.
        d.push(if host.animating() {
            ParamDecl::new("stop", "Stop", ParamKind::Action)
                .help("Stop, and put the view back where it was.")
        } else {
            ParamDecl::new("play", "Play", ParamKind::Action)
                .help("Animate the 3D view. Drag it, scroll, or go back to 2D to stop.")
        });
        d.push(
            ParamDecl::new(
                "resolution",
                "Save in",
                ParamKind::Radio {
                    default: DEFAULT_RESOLUTION,
                    options: RESOLUTIONS.iter().map(|(l, _)| l.to_string()).collect(),
                },
            )
            .help(
                "Window resolution: the size of the 3D view on screen. Max resolution: \
                 large enough that every pixel of the stack in view gets a pixel of its \
                 own (up to 8192 on the long side), and never smaller than the view.",
            ),
        );
        d.push(
            ParamDecl::new("save", "Save animation as TIFF…", ParamKind::Action).help(
                "Render one cycle, every frame once, and write it as an RGB movie at the \
                 resolution chosen above.",
            ),
        );
        d
    }

    fn run(&mut self, host: &mut dyn HostContext, params: &Params) -> Result<Outcome, PluginError> {
        if params.pressed("stop") {
            return Ok(Outcome::StopAnimation);
        }
        // The same rule the host's 2D/3D switch follows: a volume needs at
        // least two planes to stack.
        if host.image().frames < 2 {
            return Err(PluginError::unsupported(
                "this stack has a single plane along its depth, so there is no volume to \
                 turn — the 3D view needs at least two",
            ));
        }
        let motion = Motion::from_choice(params.choice("motion", DEFAULT_MOTION));
        let a = animation(
            motion,
            params.float("speed", DEFAULT_SPEED),
            params.float("tilt", DEFAULT_TILT),
        );
        // No button pressed — a test, a macro runner, a host that predates
        // buttons — plays, which is what the plugin is for.
        Ok(Outcome::Animate(Box::new(if params.pressed("save") {
            a.save_tiff("").resolution(resolution(params))
        } else {
            a
        })))
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
