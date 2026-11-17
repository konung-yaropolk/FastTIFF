//! Plugins compiled into a host, handed to it by type.

use crate::api::{Exporter, Importer, Plugin};

/// Where plugins compiled into a host are installed.
///
/// A plugin *library* is installed through its query function: the host hands
/// it an [`FtRegistrar`](crate::abi::FtRegistrar), and the function
/// [`export_plugin!`](crate::export_plugin) generated registers what the
/// library has. A plugin compiled into the host has no library to look a
/// function up in, so a set of them is handed this instead and says what it
/// has the same way, one type at a time:
///
/// ```
/// # use fasttiff_plugin::api::*;
/// # #[derive(Default)] struct Invert;
/// # impl Plugin for Invert {
/// #     fn info(&self) -> PluginInfo { PluginInfo::new("com.example.invert", "Invert") }
/// #     fn run(&mut self, _h: &mut dyn HostContext, _p: &Params)
/// #         -> Result<Outcome, PluginError> { Ok(Outcome::Nothing) }
/// # }
/// pub fn register(reg: &mut impl fasttiff_plugin::Registrar) {
///     reg.plugin::<Invert>();
/// }
/// ```
///
/// What installing one means is the host's to decide, which is why this is a
/// trait and not a function. FastTIFF's registry
/// (`fast_tiff_viewer::plugins::Registry`) installs each through the very C
/// ABI a library's goes through:
/// [`register_plugin`](crate::register_plugin) and its two siblings build the
/// descriptor and the vtable, the host's registrar checks them, and every later
/// call crosses the host table and sinks. So a plugin handed over here is held
/// to the rules a library is, and can do nothing a library cannot.
///
/// A type, not a value, for the reason [`export_plugin!`](crate::export_plugin)
/// gives: the vtable is stateless, so a value is built for every call
/// (`Default`) and dropped inside it.
///
/// Nothing is returned. A host that refuses one — an id already taken, an
/// importer with no file types — says so where it reports its other problems,
/// and the rest are still installed.
///
/// One method per kind of plugin the contract has. A kind it gains — an
/// interactive tool, say — gains a method here, and a host that cannot hold
/// one yet does not compile until it says what to do with it.
pub trait Registrar {
    /// Install a filter, an analysis, anything run against the open stack.
    fn plugin<T: Plugin + Default + 'static>(&mut self);
    /// Install a reader of a file format.
    fn importer<T: Importer + Default + 'static>(&mut self);
    /// Install a writer of a file format.
    fn exporter<T: Exporter + Default + 'static>(&mut self);
}
