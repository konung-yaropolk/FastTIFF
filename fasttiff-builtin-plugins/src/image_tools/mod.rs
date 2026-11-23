//! Built-in tools for modifying image pixels, selections, channels and stack axes.

pub mod invert;
pub use invert::Invert;
pub mod crop;
pub use crop::Crop;
pub mod stack;
pub use stack::{SliceKeeper, SliceOrderInvert, SliceRemover, SplitChannels, SplitStack};
