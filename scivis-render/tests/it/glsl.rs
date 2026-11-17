//! Offline GLSL validation: every glow program, as the backend assembles it,
//! compiled and linked by glslang, Khronos' reference GLSL compiler — with no
//! GPU, no driver and no window, so on every machine CI runs on.
//!
//! The glow backend is the Linux release renderer, and its GLSL was checked by
//! nothing before a driver compiled it at startup: a typo was a panic, or with
//! a forgiving driver a blank canvas, on someone else's machine. A driver is
//! also a poor judge of GLSL 1.40. NVIDIA's accepts much that the version
//! forbids and Mesa's does not, so a shader that works here can fail there.
//! glslang holds the source to the version it declares, as written in the
//! specification, which is the one judgement every driver has to accept.
//!
//! glslang links each stage on its own, though: it reports a function
//! declared and never defined, but not a fragment input that the vertex
//! shader has no output for, which GLSL 1.40 also makes a link error (and
//! which NVIDIA's driver lets through as well, where Mesa's refuses). That
//! half of linking, the interface between the stages, is checked here from
//! the declarations themselves (`check_interface`).
//!
//! What a reference compiler cannot see is what a real context does with the
//! program: the uniforms it reports, the picture it draws. `gl.rs` compiles
//! the same programs on the machine's own driver where there is one, and
//! draws with them.

#![cfg(not(target_arch = "wasm32"))]

use crate::unrolled::{
    assert_every_channel, assert_every_channel_arg, assert_one_channel_a_statement,
    assert_reads_where_packed, Arg,
};
use glslang::{
    Compiler, CompilerOptions, ShaderInput, ShaderMessage, ShaderSource, ShaderStage,
    SourceLanguage, Target,
};
use scivis_render::shaders::{self, GlslProgram, GLSL_VERSION};

/// Compile `program`'s two stages and link them, panicking with glslang's log
/// on the first error.
fn compile_and_link(program: &GlslProgram) {
    let compiler = Compiler::acquire().expect("glslang did not initialize");
    // Plain GLSL for an OpenGL driver: no SPIR-V target and no client API,
    // so glslang applies the rules of the `#version` the source declares and
    // nothing else (`Target::OpenGL` would add those of GL SPIR-V).
    let options = CompilerOptions {
        source_language: SourceLanguage::GLSL,
        target: Target::None(None),
        version_profile: None,
        messages: ShaderMessage::DEFAULT,
    };
    let sources = [
        (ShaderStage::Vertex, "vertex", &program.vertex),
        (ShaderStage::Fragment, "fragment", &program.fragment),
    ];
    let stages: Vec<_> = sources
        .iter()
        .map(|(stage, which, src)| {
            let source = ShaderSource::from(src.as_str());
            let input = ShaderInput::new(
                &source,
                *stage,
                &options,
                None::<&[(&str, Option<&str>)]>,
                None,
            )
            .unwrap_or_else(|e| panic!("{} {which}: {e}", program.name));
            compiler.create_shader(input).unwrap_or_else(|e| {
                panic!(
                    "the {} {which} shader does not compile: {e}{}",
                    program.name,
                    quote_lines(&e.to_string(), src)
                )
            })
        })
        .collect();
    let mut linked = compiler.create_program();
    for stage in &stages {
        linked.add_shader(stage);
    }
    linked
        .link()
        .unwrap_or_else(|e| panic!("the {} program does not link: {e}", program.name));
    check_interface(program);
}

/// The variables of storage `storage` (`in` or `out`) that `src` declares at
/// global scope, each as (its interpolation qualifiers and type, its name):
/// `out vec2 v_ndc;` is `("vec2", "v_ndc")` and `flat in int k;` is
/// `("flat int", "k")`. A precision qualifier (`in highp vec2 x;`) is legal in
/// 1.40 and need not match between stages, so it is left out of the type.
///
/// A line whose statement starts with the storage keyword (after any
/// interpolation qualifiers) is a declaration, and one this cannot read —
/// several names, an array, two statements on a line, a block comment after
/// it — panics rather than being skipped: a check that skips what it cannot
/// read passes a mismatch it never saw. A function parameter written on a
/// line of its own ends in `,` or `)` and is not one.
fn interface(src: &str, storage: &str) -> Vec<(String, String)> {
    const QUALIFIERS: [&str; 4] = ["flat", "smooth", "noperspective", "centroid"];
    const PRECISION: [&str; 3] = ["highp", "mediump", "lowp"];
    let ident = |s: &str| !s.is_empty() && s.chars().all(|c| c == '_' || c.is_ascii_alphanumeric());
    let mut found = Vec::new();
    for (n, line) in src.lines().enumerate() {
        let code = line.split("//").next().unwrap_or_default().trim();
        let tokens: Vec<&str> = code.split_whitespace().collect();
        let Some(at) = tokens.iter().position(|t| !QUALIFIERS.contains(t)) else {
            continue;
        };
        if tokens[at] != storage || code.ends_with(',') || code.ends_with(')') {
            continue;
        }
        let read = code.strip_suffix(';').and_then(|decl| {
            let tokens: Vec<&str> = decl.split_whitespace().collect();
            let rest = match &tokens[at + 1..] {
                [p, rest @ ..] if PRECISION.contains(p) => rest,
                rest => rest,
            };
            let [ty, name] = rest else {
                return None;
            };
            (ident(ty) && ident(name)).then(|| {
                let mut qualified = tokens[..at].to_vec();
                qualified.push(ty);
                (qualified.join(" "), name.to_string())
            })
        });
        match read {
            Some(declared) => found.push(declared),
            None => panic!(
                "line {}, `{}`: a shader `{storage}` this check cannot read; write it as \
                 `[qualifiers] {storage} [precision] type name;`, alone on its line, or \
                 teach interface() the new shape",
                n + 1,
                line.trim()
            ),
        }
    }
    found
}

/// Whether `src` assigns to `name` (`name = …`, or a swizzle of it,
/// `name.xy = …`) outside a `//` comment: as a whole word, and not a
/// comparison (`name == …`).
fn writes(src: &str, name: &str) -> bool {
    let word = |c: char| c == '_' || c.is_ascii_alphanumeric();
    src.lines()
        .map(|line| line.split("//").next().unwrap_or_default())
        .any(|code| {
            code.match_indices(name).any(|(i, _)| {
                if code[..i].chars().next_back().is_some_and(word) {
                    return false;
                }
                let after = &code[i + name.len()..];
                if after.starts_with(word) {
                    return false;
                }
                let after = after
                    .strip_prefix('.')
                    .map_or(after, |s| {
                        s.trim_start_matches(|c: char| c.is_ascii_alphabetic())
                    })
                    .trim_start();
                after.starts_with('=') && !after.starts_with("==")
            })
        })
}

/// That every input of `program`'s fragment shader is an output of its
/// vertex shader, of the same type and interpolation, and one the vertex
/// shader writes. GLSL 1.40 makes a fragment input with no output to match a
/// link error; glslang leaves it to the driver, and a forgiving driver draws
/// with the input undefined.
fn check_interface(program: &GlslProgram) {
    let outputs = interface(&program.vertex, "out");
    for (ty, name) in interface(&program.fragment, "in") {
        let Some((out_ty, _)) = outputs.iter().find(|(_, n)| *n == name) else {
            panic!(
                "the {} program does not link: its fragment shader reads `in {ty} {name}`, \
                 which its vertex shader has no output for",
                program.name
            );
        };
        assert_eq!(
            *out_ty, ty,
            "the {} program does not link: `{name}` is `{out_ty}` out of its vertex shader \
             and `{ty}` into its fragment shader",
            program.name
        );
        assert!(
            writes(&program.vertex, &name),
            "the {} program's vertex shader declares `out {ty} {name}` and never writes it",
            program.name
        );
    }
}

/// The lines of `src` that glslang's `log` names (`ERROR: 0:LINE: ...`),
/// quoted: its numbers count the constants written in ahead of the file, so
/// they are not the file's own.
fn quote_lines(log: &str, src: &str) -> String {
    let lines: Vec<&str> = src.lines().collect();
    let mut quoted = String::new();
    let mut named = Vec::new();
    for entry in log.lines() {
        let Some(rest) = entry.split_once(": 0:").map(|(_, r)| r) else {
            continue;
        };
        let Some(n) = rest.split(':').next().and_then(|n| n.parse::<usize>().ok()) else {
            continue;
        };
        if named.contains(&n) {
            continue;
        }
        named.push(n);
        if let Some(line) = n.checked_sub(1).and_then(|i| lines.get(i)) {
            quoted.push_str(&format!(
                "
  line {n}: {}",
                line.trim()
            ));
        }
    }
    quoted
}

#[test]
fn every_program_compiles_and_links() {
    for program in shaders::glsl_programs() {
        compile_and_link(&program);
    }
}

/// glslang must be holding the shaders to GLSL 1.40, or the test above proves
/// less than it says: a feature from a later version has to fail.
#[test]
fn a_later_feature_fails_to_compile() {
    let mut program = shaders::blit_glsl();
    assert!(program.fragment.starts_with(GLSL_VERSION));
    // `layout(location)` on a fragment output is GLSL 3.30.
    program.fragment = program.fragment.replace(
        "out vec4 frag_color;",
        "layout(location = 0) out vec4 frag_color;",
    );
    let failed = std::panic::catch_unwind(|| compile_and_link(&program));
    assert!(
        failed.is_err(),
        "glslang accepted a GLSL 3.30 feature in a 1.40 shader"
    );
}

/// The interface check must be holding the stages to each other, or the test
/// above proves less than it says: a fragment input the vertex shader has no
/// output for (as when one side of a rename is missed) has to fail, and so
/// does one of another type.
#[test]
fn a_stage_mismatch_fails_to_link() {
    let mut renamed = shaders::volume_glsl();
    assert!(renamed.vertex.contains("out vec2 v_ndc;"));
    renamed.vertex = renamed.vertex.replace("v_ndc", "v_ndcx");
    let mut retyped = shaders::blit_glsl();
    assert!(retyped.vertex.contains("out vec2 v_uv;"));
    retyped.vertex = retyped
        .vertex
        .replace("out vec2 v_uv;", "out vec3 v_uv;")
        .replace("v_uv = c;", "v_uv = vec3(c, 0.0);");
    // The reviewer's case: a rename behind a precision qualifier, which the
    // first version of the check could not read and so skipped.
    let mut qualified = shaders::blit_glsl();
    assert!(qualified.fragment.contains("in vec2 v_uv;"));
    qualified.fragment = qualified
        .fragment
        .replace("in vec2 v_uv;", "in highp vec2 v_uvx;")
        .replace("src, v_uv,", "src, v_uvx,");
    // Declared but only compared, or written only in a comment: not written.
    let mut unwritten = shaders::blit_glsl();
    assert!(unwritten.vertex.contains("v_uv = c;"));
    unwritten.vertex = unwritten
        .vertex
        .replace("v_uv = c;", "// v_uv = c;\n    if (v_uv == c) {}");
    for (mismatch, program) in [
        ("a rename", renamed),
        ("a type", retyped),
        ("a rename under a precision qualifier", qualified),
        ("an output never written", unwritten),
    ] {
        let failed = std::panic::catch_unwind(|| compile_and_link(&program));
        assert!(
            failed.is_err(),
            "the {} program linked with {mismatch} in only one of its stages",
            program.name
        );
    }
}

/// A declaration the interface check cannot read is an error, not a pass.
#[test]
fn an_interface_the_check_cannot_read_is_refused() {
    for shape in [
        "in vec2 v_uv, v_extra;",
        "in vec2 v_uv[2];",
        "in vec2 v_uv; in vec2 v_extra;",
        "in vec2 v_uv; /* the picture's uv */",
    ] {
        let src = format!("#version 140\n{shape}\nvoid main() {{}}\n");
        let failed = std::panic::catch_unwind(|| interface(&src, "in"));
        assert!(
            failed.is_err(),
            "`{shape}` was read, or skipped, without a word"
        );
    }
    // What it does read, and what is not a declaration at all.
    let src = "flat in int k;\nin highp vec2 x;\nvoid f(\n    in vec2 a,\n    out vec2 b)\n{}\n";
    assert_eq!(
        interface(src, "in"),
        [
            ("flat int".to_string(), "k".to_string()),
            ("vec2".to_string(), "x".to_string())
        ]
    );
    assert!(writes("    v_uv = c;", "v_uv"));
    assert!(writes("    v_uv.xy = c;", "v_uv"));
    assert!(!writes("    prev_v_uv = c;", "v_uv"));
    assert!(!writes("    v_uvx = c;", "v_uv"));
    assert!(!writes("    if (v_uv == c) {}", "v_uv"));
    assert!(!writes("    // v_uv = c;", "v_uv"));
}

/// GLSL's character set is ASCII, comments included (non-ASCII in a comment
/// is allowed only from 4.20): a stray dash from prose is an error to a
/// driver that keeps to that.
#[test]
fn sources_are_ascii() {
    for program in shaders::glsl_programs() {
        for (which, src) in [("vertex", &program.vertex), ("fragment", &program.fragment)] {
            if let Some((line, text)) = src.lines().enumerate().find(|(_, l)| !l.is_ascii()) {
                panic!(
                    "the {} {which} shader is not ASCII, line {}: {text}",
                    program.name,
                    line + 1
                );
            }
        }
    }
}

/// The per-channel lines written out by hand: see `unrolled.rs`.
#[test]
fn per_channel_code_covers_every_channel() {
    let what = "composite.frag.glsl";
    let composite = include_str!("../../src/shaders/composite.frag.glsl");
    for array in ["ch_tex[", "ch_ftex[", "ch_params["] {
        assert_every_channel(what, composite, array, "]");
    }
    assert_every_channel_arg(what, composite, "apply_channel", Arg::At(1));
    assert_one_channel_a_statement(what, composite);
    let what = "volume.frag.glsl";
    let volume = include_str!("../../src/shaders/volume.frag.glsl");
    assert_every_channel(what, volume, "on(", ")");
    assert_every_channel(what, volume, "ch_albedo[", "]");
    assert_every_channel_arg(what, volume, "lut_col", Arg::Last);
    assert_reads_where_packed(what, volume, "norm_ch(");
    assert_one_channel_a_statement(what, volume);
}
