// Copyright (C) 2026 SciWare LLC
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version. See the LICENSE file at the root of this crate.

//! What the two registration crates share that is not an algorithm.
//!
//! `suite2p-registration` and `turboreg-registration` grew the same
//! infrastructure side by side, and the copies drifted: one shim gained ordered
//! helpers the other did not have, the other gained `map_init`. Each piece here
//! is the union of what both used, and both crates now depend on it rather than
//! keeping their own.

pub mod par;
