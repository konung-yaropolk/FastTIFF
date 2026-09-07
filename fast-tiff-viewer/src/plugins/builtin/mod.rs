//! The plugins compiled into this build.
//!
//! Everything in this directory is a *plugin*. Everything in the directory
//! above it is the *interface* plugins are written against — the registry, the
//! search path, the host context, the shared-library loader. The split is worth
//! keeping sharp: a file here may only use `fasttiff_plugin_api`, exactly as a
//! third-party plugin would, so if one of these ever needs something the API
//! does not offer, that is a gap in the contract rather than a reason to reach
//! sideways into the viewer.
//!
//! Being compiled in is a *lane*, not a different kind of plugin. These
//! implement the same [`Plugin`] and [`Importer`] traits a `.dll` does, and the
//! registry cannot tell them apart except by the [`Origin`](super::Origin) tag
//! it stores alongside. They exist for four reasons:
//!
//! * The browser build can never load a shared library, and still gets a
//!   working Plugins menu.
//! * The contract gets a real consumer before any loading mechanism exists,
//!   which is how a plugin API avoids being designed in the abstract and then
//!   failing on the first real plugin.
//! * [`Invert`] is the oracle for the `.dll` lane: the same filter, run both
//!   ways, must produce byte-identical output.
//! * [`Oir`] is the importer that earns its place: a proprietary format with no
//!   spec at all, worked out from a real acquisition and checked against the
//!   vendor software's own export.
//!
//! [`Netpbm`] is a fifth thing — a worked example of the other shape the job
//! takes, a documented format implemented straight from its spec. It is behind
//! the off-by-default `netpbm-example` feature: read it when writing an
//! importer, enable it (`--features netpbm-example`) to run it, but it is not
//! part of the product and nobody opens a `.pgm` in a TIFF viewer.
//!
//! # Adding one
//!
//! Write it in a file here, using nothing but `fasttiff_plugin_api`, and add it
//! to [`all`] or [`importers`] below. That is the whole procedure — and if it
//! turns out you would rather ship it separately, moving the file into a
//! `cdylib` crate needs no changes to the plugin itself.

pub mod invert;
#[cfg(feature = "netpbm-example")]
pub mod netpbm;
pub mod oir;
pub mod zproject;

pub use invert::Invert;
#[cfg(feature = "netpbm-example")]
pub use netpbm::Netpbm;
pub use oir::Oir;
pub use zproject::ZProject;

use fasttiff_plugin_api::{Importer, Plugin};

/// The filters compiled into this build, in registration order.
pub fn all() -> Vec<Box<dyn Plugin>> {
    vec![Box::new(Invert), Box::new(ZProject)]
}

/// The importers compiled into this build, in registration order.
///
/// Order is the tie-break when two importers are equally confident about a
/// file, so it is a real decision rather than a list: the more specific format
/// goes first.
pub fn importers() -> Vec<Box<dyn Importer>> {
    #[allow(unused_mut)]
    let mut v: Vec<Box<dyn Importer>> = vec![Box::new(Oir)];
    #[cfg(feature = "netpbm-example")]
    v.push(Box::new(Netpbm));
    v
}
