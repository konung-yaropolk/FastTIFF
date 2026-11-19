//! The glow programs on the machine's own OpenGL driver, built and drawn
//! with by the backend itself.
//!
//! `glsl.rs` holds the sources to the GLSL specification with no driver; this
//! is the other half, which only a real context can show: that the driver
//! compiles and links what the backend hands it, that each uniform the
//! backend sets is one the linked program has, and that each call the
//! backend makes to draw is one the driver accepts. GL drops a call it
//! refuses, and draws on: a uniform set with the function for another type
//! (`glUniform1i` on a `float`) is `GL_INVALID_OPERATION` and the uniform
//! stays zero, which glslang cannot see, as the mismatch is between the
//! shader and the Rust that drives it. So every test here checks
//! `glGetError` after what it ran, and draws the scene of `scene.rs` in
//! every mode the backend has.
//!
//! A uniform the program does not have (misspelt, or optimized away) GL
//! answers a lookup of with nothing, and drops every later set of it without
//! even an error. The backend asserts on such a lookup in a debug build;
//! release builds (release.yml's `cargo test --release`) compile that check
//! out, and there `the_backend_builds_its_programs_on_the_driver` sees only
//! compile, link and GL errors.
//!
//! Skipped, with a message, where no GL 3.1 context can be made, and failed
//! there instead where the environment asks for one (see `headless.rs`).

#![cfg(not(target_arch = "wasm32"))]

use crate::headless::{self, assert_no_error};
use crate::scene::{self, show_image, show_volume, ALL, VIEWS};
use glow::HasContext as _;
use scivis_render::glow_backend::ImageRenderResources;
use scivis_render::shaders::{self, GlslProgram};
use scivis_render::{VolumeInterp, VolumeRender, VolumeRenderer as _, MAX_CHANNELS};
use std::sync::Arc;

/// `program` compiled and linked on `gl`, the way the backend does it.
unsafe fn link(gl: &glow::Context, program: &GlslProgram) -> glow::NativeProgram {
    let linked = gl.create_program().expect("create program");
    let mut stages = Vec::new();
    for (kind, which, src) in [
        (glow::VERTEX_SHADER, "vertex", &program.vertex),
        (glow::FRAGMENT_SHADER, "fragment", &program.fragment),
    ] {
        let shader = gl.create_shader(kind).expect("create shader");
        gl.shader_source(shader, src);
        gl.compile_shader(shader);
        assert!(
            gl.get_shader_compile_status(shader),
            "the {} {which} shader does not compile: {}",
            program.name,
            gl.get_shader_info_log(shader)
        );
        gl.attach_shader(linked, shader);
        stages.push(shader);
    }
    gl.link_program(linked);
    assert!(
        gl.get_program_link_status(linked),
        "the {} program does not link: {}",
        program.name,
        gl.get_program_info_log(linked)
    );
    for shader in stages {
        gl.delete_shader(shader);
    }
    linked
}

/// The backend builds all three programs on this driver and finds every
/// uniform it sets in them.
#[test]
fn the_backend_builds_its_programs_on_the_driver() {
    let _gpu = headless::exclusive();
    let Some(gl) = headless::gl() else { return };
    let renderer = ImageRenderResources::new(Arc::clone(&gl.gl));
    assert_no_error(&gl.gl, "building the programs");
    drop(renderer);
}

/// That at least a fiftieth of a picture is lit: a draw the driver dropped
/// leaves the clear colour.
fn assert_lit(what: &str, rgba: &[u8]) {
    let lit = scene::lit(rgba);
    assert!(
        lit > 0.02,
        "{what}: {:.1}% of the picture is lit; nothing was drawn",
        lit * 100.0
    );
}

/// Every draw the backend makes, of the scene in every view and mode, is one
/// the driver accepts: every uniform set with the function for its type,
/// every texture bound where its sampler reads, and a picture at the end.
/// Small pictures, so that a software rasterizer (Mesa's llvmpipe, on the
/// Linux CI runner) draws them all in a moment.
#[test]
fn the_backend_draws_every_mode_without_a_gl_error() {
    let _gpu = headless::exclusive();
    let Some(gl) = headless::gl() else { return };
    let gl = &gl.gl;
    let mut r = ImageRenderResources::new(Arc::clone(gl));
    assert_no_error(gl, "building the programs");

    let channels = scene::image();
    for view in &VIEWS {
        show_image(&mut r, &channels, view);
        assert_no_error(gl, "uploading the image");
        let what = format!("the composite, {}", view.name);
        let rgba = headless::draw_offscreen(gl, view.size, || r.paint(gl));
        assert_no_error(gl, &what);
        assert_lit(&what, &rgba);
    }

    let modes = [
        VolumeRender::Mip,
        VolumeRender::Alpha,
        VolumeRender::Surface,
    ];
    let interps = [
        VolumeInterp::Nearest,
        VolumeInterp::Linear,
        VolumeInterp::Cubic,
    ];
    for mode in modes {
        for interp in interps {
            show_volume(&mut r, mode, interp, 1.5, ALL);
            assert_no_error(gl, "uploading the volume");
            r.capture_volume(32, 24).expect("a capture starts");
            let what = format!("the volume, {mode:?}, {interp:?}");
            let image = r
                .take_volume_capture()
                .expect("GL renders at once")
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_no_error(gl, &what);
            assert_lit(&what, &image.rgba);
        }
    }

    // On screen, as the app draws it: at full size, and reduced and scaled
    // up (the blit program) as while the view moves.
    for downscale in [1, 2] {
        r.set_volume_target(48, 32, downscale);
        let what = format!("the volume painted, reduced {downscale} times");
        let rgba = headless::draw_offscreen(gl, (48, 32), || r.paint_volume(gl));
        assert_no_error(gl, &what);
        assert_lit(&what, &rgba);
    }
}

/// Every uniform array of the glow programs is a per-channel one, and the
/// driver has it `MAX_CHANNELS` long. A driver may report an array only as
/// long as its last element read, so on one that does, a shorter array is
/// per-channel code that stops short (the count itself is `unrolled.rs`'s).
#[test]
fn every_uniform_array_is_max_channels_long() {
    let _gpu = headless::exclusive();
    let Some(gl) = headless::gl() else { return };
    let gl = &*gl.gl;
    for program in shaders::glsl_programs() {
        // SAFETY: the context is current on this thread.
        unsafe {
            let linked = link(gl, &program);
            for index in 0..gl.get_active_uniforms(linked) {
                let uniform = gl
                    .get_active_uniform(linked, index)
                    .expect("an active uniform");
                // GL names an array by its first element.
                if uniform.name.ends_with("[0]") {
                    assert_eq!(
                        uniform.size, MAX_CHANNELS as i32,
                        "the {} program's `{}`",
                        program.name, uniform.name
                    );
                }
            }
            gl.delete_program(linked);
        }
    }
}
