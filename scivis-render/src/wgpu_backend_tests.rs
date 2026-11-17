//! The uniform structs against the WGSL structs they are copied into.
//!
//! `CompositeParams` and its `ChannelParams` (shared with the glow backend, in
//! `uniforms`) and `VolParamsGpu` go to the GPU byte for byte
//! (`bytemuck::bytes_of`), into buffers the shaders read as `Params`,
//! `ChannelParams` and `VolParams`. Nothing but these tests ties the two
//! declarations together. A field added on one side only still compiles and
//! validates; then a buffer of the wrong size is fatal to wgpu at the first
//! draw, or one of the right size is read a field out of step. So each Rust
//! struct is compared with naga's layout of the shader's, field by field:
//! name, offset, size and type, and the size of the whole.

use super::*;
use crate::uniforms::ChannelParams;
use std::mem::{offset_of, size_of, size_of_val};

/// A type's shape, spelled alike for Rust and WGSL: scalars by name, vectors
/// and arrays both as `[element; n]` (the sizes and offsets tell the two
/// apart), structs by their WGSL name. Equal sizes and offsets say the bytes
/// line up; equal shapes say they are read as what they were written as —
/// a `u32` read as an `f32` is the same four bytes.
trait Shape {
    fn shape() -> String;
}

impl Shape for f32 {
    fn shape() -> String {
        "f32".into()
    }
}

impl Shape for u32 {
    fn shape() -> String {
        "u32".into()
    }
}

impl Shape for i32 {
    fn shape() -> String {
        "i32".into()
    }
}

impl<T: Shape, const N: usize> Shape for [T; N] {
    fn shape() -> String {
        format!("[{}; {N}]", T::shape())
    }
}

impl Shape for ChannelParams {
    fn shape() -> String {
        "ChannelParams".into()
    }
}

fn shape_of<T: Shape>(_: &T) -> String {
    T::shape()
}

/// One field of a struct: name, offset, size, shape.
type Field = (String, usize, usize, String);

/// The fields of the uniform struct `$ty`, in order. The pattern names every
/// field with no `..`, so a field added to the struct and not to the list
/// here is a compile error, not a field the test does not look at.
macro_rules! fields {
    ($ty:ident { $($field:ident),* $(,)? }) => {{
        let value: $ty = bytemuck::Zeroable::zeroed();
        let $ty { $($field: _),* } = value;
        vec![$((
            stringify!($field).to_string(),
            offset_of!($ty, $field),
            size_of_val(&value.$field),
            shape_of(&value.$field),
        )),*]
    }};
}

fn shape_in(module: &naga::Module, ty: naga::Handle<naga::Type>) -> String {
    let scalar = |s: &naga::Scalar| match (s.kind, s.width) {
        (naga::ScalarKind::Float, 4) => "f32".to_string(),
        (naga::ScalarKind::Uint, 4) => "u32".to_string(),
        (naga::ScalarKind::Sint, 4) => "i32".to_string(),
        other => format!("{other:?}"),
    };
    let t = &module.types[ty];
    match &t.inner {
        naga::TypeInner::Scalar(s) => scalar(s),
        naga::TypeInner::Vector { size, scalar: s } => format!("[{}; {}]", scalar(s), *size as u8),
        naga::TypeInner::Array {
            base,
            size: naga::ArraySize::Constant(n),
            ..
        } => format!("[{}; {n}]", shape_in(module, *base)),
        naga::TypeInner::Struct { .. } => t.name.clone().unwrap_or_default(),
        other => format!("{other:?}"),
    }
}

/// The struct `name` of the WGSL `src`, as naga lays it out: its size and its
/// members.
fn wgsl_struct(src: &str, name: &str) -> (usize, Vec<Field>) {
    let module =
        naga::front::wgsl::parse_str(src).expect("the shader parses (see tests/it/wgsl.rs)");
    let mut layouter = naga::proc::Layouter::default();
    layouter
        .update(module.to_ctx())
        .expect("the shader's types lay out");
    let (handle, ty) = module
        .types
        .iter()
        .find(|(_, t)| t.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("the shader has no struct `{name}`"));
    let naga::TypeInner::Struct { members, span } = &ty.inner else {
        panic!("`{name}` is not a struct");
    };
    assert_eq!(layouter[handle].size, *span);
    let fields = members
        .iter()
        .map(|m| {
            (
                m.name.clone().unwrap_or_default(),
                m.offset as usize,
                layouter[m.ty].size as usize,
                shape_in(&module, m.ty),
            )
        })
        .collect();
    (*span as usize, fields)
}

/// `rust` (the struct `rust_name`, `rust_size` bytes) is laid out exactly as
/// the shader's struct `wgsl_name` in `src`.
fn assert_same_layout(
    rust_name: &str,
    rust_size: usize,
    rust: Vec<Field>,
    src: &str,
    wgsl_name: &str,
) {
    let (wgsl_size, wgsl) = wgsl_struct(src, wgsl_name);
    let table = |fields: &[Field]| {
        fields
            .iter()
            .map(|(n, o, s, t)| format!("    {o:>4} +{s:<4} {n}: {t}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    assert!(
        rust == wgsl && rust_size == wgsl_size,
        "`{rust_name}` ({rust_size} bytes) is not laid out as `{wgsl_name}` in the shader \
         ({wgsl_size} bytes): add, remove or reorder fields on both sides alike\n  \
         {rust_name}:\n{}\n  {wgsl_name}:\n{}",
        table(&rust),
        table(&wgsl),
    );
}

#[test]
fn composite_uniform_is_laid_out_as_the_shader_reads_it() {
    let src = crate::shaders::composite_wgsl();
    assert_same_layout(
        "ChannelParams",
        size_of::<ChannelParams>(),
        fields!(ChannelParams {
            min_max,
            enabled,
            is_float,
        }),
        &src,
        "ChannelParams",
    );
    assert_same_layout(
        "CompositeParams",
        size_of::<CompositeParams>(),
        fields!(CompositeParams {
            channels,
            uv_offset,
            uv_scale,
            num_channels,
            _pad0,
            _pad1,
            _pad2,
        }),
        &src,
        "Params",
    );
}

#[test]
fn volume_uniform_is_laid_out_as_the_shader_reads_it() {
    assert_same_layout(
        "VolParamsGpu",
        size_of::<VolParamsGpu>(),
        fields!(VolParamsGpu {
            channels,
            cam_eye,
            cam_forward,
            cam_right,
            cam_up,
            box_he,
            misc,
            gain,
            modes,
            accel,
        }),
        &crate::shaders::volume_wgsl(),
        "VolParams",
    );
}
