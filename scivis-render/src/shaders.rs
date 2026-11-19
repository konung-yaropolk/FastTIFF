//! The shader sources both backends compile, assembled from the constants
//! they share with the Rust side.
//!
//! Every shader is written twice: in WGSL for the wgpu backend
//! (`src/shaders/*.wgsl`) and in GLSL 1.40 for the glow backend
//! (`src/shaders/*.glsl`). Both copies, and the Rust code that fills their
//! uniforms, have to agree on a handful of numbers: how many channels there
//! are, the cap on ray-march steps, the isosurface's look, the codes the mode
//! uniforms carry. Each was once a literal in each copy and in Rust, so
//! changing [`MAX_CHANNELS`] compiled and then failed at run time — a uniform
//! of the wrong size is fatal to wgpu, and glow drops a uniform it cannot find
//! without a word.
//!
//! Now each is defined once, in Rust, and written into both languages' sources
//! as a `const` ahead of the shader's own text (see [`SHARED`]). No shader file
//! defines one. What follows from them on its own does: the uniform arrays, the
//! LUT rows, the composite's per-channel texture bindings. What cannot — the
//! per-channel code, which both languages make us unroll by hand — is counted
//! against [`MAX_CHANNELS`] by the shader tests (`tests/it/wgsl.rs`,
//! `tests/it/glsl.rs`), so a change there fails a test instead of drawing
//! wrong.
//!
//! The sources are put together once per pipeline, when a backend is made;
//! nothing here runs per frame. They are public so that a host, or a test, can
//! validate exactly what is compiled.
//!
//! # Why the GLSL is not written from the WGSL
//!
//! naga, in the tree through wgpu, can write it: GLSL 1.40 with each uniform
//! struct as a `std140` block laid out as the WGSL's (and the Rust's), once
//! the `#version 140 core` line it writes is replaced (profiles came with
//! 1.50; glslang and Mesa refuse it). That was built and measured (Nov 2026,
//! Quadro P620, OpenGL 3.3 core, driver 512.15): it drew what wgpu draws, but
//! slower than this GLSL on the same bindings — the ray-march 1.3 to 1.9
//! times (a MIP of a 256 x 256 x 128 x 6 volume at 1280 x 720, 35 ms a frame
//! here, 60 there; a surface 9.5 ms, 18.5), and six 2048 x 2048 channels
//! composited whole into a 1080p view a quarter (4.2 ms, 5.2). Not the WGSL's
//! doing: on Vulkan, wgpu runs it within 5% of this (the MIP in 36 ms), but
//! for the surface (11.4 ms). The driver compiles naga's
//! GLSL worse: a branch on a uniform inside the march loop costs it even
//! untaken (one added to this GLSL's sampling made its MIP a fifth slower),
//! and naga writes a `for` loop as a `while` with a flag (the composite's
//! 5.2 ms are 4.5 with its loops written back as `for`). Programs
//! specialized per render and sampling mode (WGSL `override` constants, set
//! by a build script) took the MIP and the alpha DVR from 35 and 33 ms to 26
//! and 25, but left the cubic MIP and the surface slower than this (78 ms
//! against 70, 11.3 against 9.5). And naga as a build-dependency adds some
//! 27 s to a clean build of this crate. Those, measured on Mesa as well (the
//! driver the glow build ships to, and one not measured here), are what to
//! beat before the GLSL here goes.

use crate::{VolumeInterp, VolumeRender, MAX_CHANNELS, SHADER_MAX_STEPS};
use std::fmt::Write as _;

/// The albedo an isosurface is scaled to. 1.0 is a blown-out white for a
/// plain grayscale stack; lower reads as a softer gray. (Blender's classic
/// default base colour is 0.8.)
///
/// A compile-time constant rather than a uniform: it is a fixed look, not
/// per-frame state, so it costs no uniform slot.
pub const SURFACE_GRAY: f32 = 0.6;

/// How an isosurface's shading splits between ambient light and a headlight
/// at the camera: `shade = (1 - SURFACE_CONTRAST) + SURFACE_CONTRAST * diffuse`.
/// 0.0 is flat and unshaded; 1.0 falls all the way to black at silhouettes.
pub const SURFACE_CONTRAST: f32 = 0.85;

/// The opacity at which alpha compositing stops marching: nothing further
/// along the ray can show through more than half a percent.
pub const ALPHA_OPAQUE: f32 = 0.995;

/// The most texels the 2D composite averages along each axis of an output
/// pixel when the image is shown smaller than its size (a box filter against
/// the shimmer of plain nearest sampling). Capped so a far zoom-out costs at
/// most this squared reads a pixel.
pub const MAX_MINIFY_TAPS: f32 = 4.0;

/// The textures the composite's fragment shader binds: two a channel (of
/// which it reads the one the channel's kind uses), and the LUTs.
const COMPOSITE_TEXTURES: usize = 2 * MAX_CHANNELS + 1;

// 16 is as many as a GL 3 context or a WebGPU device has to let one fragment
// shader sample (`GL_MAX_TEXTURE_IMAGE_UNITS`,
// `maxSampledTexturesPerShaderStage`).
const _: () = assert!(
    COMPOSITE_TEXTURES <= 16,
    "MAX_CHANNELS is more than the composite shader can sample everywhere"
);

/// The bindings of the composite pipeline's group 0. The per-channel textures
/// come last, so that only their own numbers move with [`MAX_CHANNELS`].
pub mod composite_bindings {
    use crate::MAX_CHANNELS;

    /// The uniform buffer, `Params` in `composite.wgsl`.
    pub const PARAMS: u32 = 0;
    /// The LUTs, one layer per channel.
    pub const LUT: u32 = 1;
    /// The LUTs' sampler.
    pub const SAMPLER: u32 = 2;
    /// Channel `c`'s integer texture is at `INT_TEXTURE + c`.
    pub const INT_TEXTURE: u32 = 3;
    /// Channel `c`'s float texture is at `FLOAT_TEXTURE + c`.
    pub const FLOAT_TEXTURE: u32 = INT_TEXTURE + MAX_CHANNELS as u32;
}

/// A value both shading languages are given under one name.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shared {
    Int(i32),
    Float(f32),
}

/// Every constant written into the shaders, by the name they use for it.
///
/// One list for both languages, so a constant cannot reach one and not the
/// other. The codes are those [`VolumeRender::shader_mode`] and
/// [`VolumeInterp::shader_mode`] give the uniforms.
pub const SHARED: [(&str, Shared); 12] = [
    ("MAX_CHANNELS", Shared::Int(MAX_CHANNELS as i32)),
    ("MAX_STEPS", Shared::Int(SHADER_MAX_STEPS)),
    ("MODE_MIP", Shared::Int(VolumeRender::Mip.shader_mode())),
    ("MODE_ALPHA", Shared::Int(VolumeRender::Alpha.shader_mode())),
    (
        "MODE_SURFACE",
        Shared::Int(VolumeRender::Surface.shader_mode()),
    ),
    (
        "INTERP_NEAREST",
        Shared::Int(VolumeInterp::Nearest.shader_mode()),
    ),
    (
        "INTERP_LINEAR",
        Shared::Int(VolumeInterp::Linear.shader_mode()),
    ),
    (
        "INTERP_CUBIC",
        Shared::Int(VolumeInterp::Cubic.shader_mode()),
    ),
    ("SURFACE_GRAY", Shared::Float(SURFACE_GRAY)),
    ("SURFACE_CONTRAST", Shared::Float(SURFACE_CONTRAST)),
    ("ALPHA_OPAQUE", Shared::Float(ALPHA_OPAQUE)),
    ("MAX_MINIFY_TAPS", Shared::Float(MAX_MINIFY_TAPS)),
];

/// A literal both languages read as the same `f32`: Rust's shortest form that
/// round-trips, which always has a `.` or an exponent.
fn float_literal(v: f32) -> String {
    assert!(v.is_finite(), "a shader constant must be finite, not {v}");
    format!("{v:?}")
}

#[derive(Clone, Copy)]
enum Lang {
    Wgsl,
    Glsl,
}

/// The constants of [`SHARED`] as declarations in `lang`.
///
/// In WGSL they are left untyped (abstract), so that `MAX_CHANNELS` sizes an
/// array and compares with an `i32` or a `u32` alike.
fn prelude(lang: Lang) -> String {
    let mut s = String::from(
        "// Written in by scivis-render's `shaders` module: the constants the\n\
         // shaders share with each other and with the Rust side.\n",
    );
    for (name, value) in SHARED {
        let _ = match (lang, value) {
            (Lang::Wgsl, Shared::Int(v)) => writeln!(s, "const {name} = {v};"),
            (Lang::Wgsl, Shared::Float(v)) => writeln!(s, "const {name} = {};", float_literal(v)),
            (Lang::Glsl, Shared::Int(v)) => writeln!(s, "const int {name} = {v};"),
            (Lang::Glsl, Shared::Float(v)) => {
                writeln!(s, "const float {name} = {};", float_literal(v))
            }
        };
    }
    s
}

/// The 2D composite, for the wgpu backend: `composite.wgsl` after the shared
/// constants and the per-channel texture bindings.
///
/// Core WebGPU has no arrays of textures, so each channel's two textures are
/// bindings of their own, `ch{c}_tex` and `ch{c}_ftex`, numbered from
/// [`composite_bindings`] — declared here, for every channel there is, rather
/// than listed by hand in the shader.
pub fn composite_wgsl() -> String {
    use composite_bindings::{FLOAT_TEXTURE, INT_TEXTURE};
    let mut s = prelude(Lang::Wgsl);
    for c in 0..MAX_CHANNELS as u32 {
        let _ = writeln!(
            s,
            "@group(0) @binding({}) var ch{c}_tex: texture_2d<u32>;",
            INT_TEXTURE + c
        );
    }
    for c in 0..MAX_CHANNELS as u32 {
        let _ = writeln!(
            s,
            "@group(0) @binding({}) var ch{c}_ftex: texture_2d<f32>;",
            FLOAT_TEXTURE + c
        );
    }
    s + include_str!("shaders/composite.wgsl")
}

/// The volume ray-march, for the wgpu backend.
pub fn volume_wgsl() -> String {
    prelude(Lang::Wgsl) + include_str!("shaders/volume.wgsl")
}

/// The pass that scales a reduced render of the 3D view up, for the wgpu
/// backend.
pub fn blit_wgsl() -> String {
    prelude(Lang::Wgsl) + include_str!("shaders/blit.wgsl")
}

/// The GLSL version every glow program is written for: what a desktop GL 3.x
/// context accepts — the one egui_glow and most native GL hosts make — and
/// the first with `usampler2D`, `texelFetch` and `gl_VertexID`.
pub const GLSL_VERSION: &str = "#version 140";

/// One glow program's two stages, each a complete GLSL source.
#[derive(Clone, Debug)]
pub struct GlslProgram {
    /// Which program, for messages.
    pub name: &'static str,
    pub vertex: String,
    pub fragment: String,
}

fn glsl(stage: &str) -> String {
    // The `#version` line must come first; the constants follow it.
    format!("{GLSL_VERSION}\n{}{stage}", prelude(Lang::Glsl))
}

/// The 2D composite, for the glow backend.
pub fn composite_glsl() -> GlslProgram {
    GlslProgram {
        name: "composite",
        vertex: glsl(include_str!("shaders/composite.vert.glsl")),
        fragment: glsl(include_str!("shaders/composite.frag.glsl")),
    }
}

/// The volume ray-march, for the glow backend.
pub fn volume_glsl() -> GlslProgram {
    GlslProgram {
        name: "volume",
        vertex: glsl(include_str!("shaders/volume.vert.glsl")),
        fragment: glsl(include_str!("shaders/volume.frag.glsl")),
    }
}

/// The scale-up pass of the reduced 3D view, for the glow backend.
pub fn blit_glsl() -> GlslProgram {
    GlslProgram {
        name: "blit",
        vertex: glsl(include_str!("shaders/blit.vert.glsl")),
        fragment: glsl(include_str!("shaders/blit.frag.glsl")),
    }
}

/// Every glow program, for a test to compile.
pub fn glsl_programs() -> [GlslProgram; 3] {
    [composite_glsl(), volume_glsl(), blit_glsl()]
}

/// Every wgpu shader by name, for a test to validate.
pub fn wgsl_modules() -> [(&'static str, String); 3] {
    [
        ("composite", composite_wgsl()),
        ("volume", volume_wgsl()),
        ("blit", blit_wgsl()),
    ]
}
