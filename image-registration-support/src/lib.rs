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

pub mod area;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod par;

/// Pixelwise channel mean for one timepoint. Reuse this across frames to retain its buffer.
#[derive(Default)]
pub struct ChannelMean {
    sum: Vec<f64>,
    count: usize,
}

impl ChannelMean {
    /// Include one raw channel plane. All planes of a frame must have equal length.
    pub fn push(&mut self, plane: &[f32]) {
        if self.count == 0 {
            self.sum.clear();
            self.sum.extend(plane.iter().map(|&x| f64::from(x)));
        } else {
            assert_eq!(plane.len(), self.sum.len(), "channel planes differ in size");
            for (sum, &value) in self.sum.iter_mut().zip(plane) {
                *sum += f64::from(value);
            }
        }
        self.count += 1;
    }

    /// Write the mean and begin a fresh frame. Requires at least one channel.
    pub fn finish_into(&mut self, out: &mut Vec<f32>) {
        assert!(self.count > 0);
        out.clear();
        out.extend(self.sum.iter().map(|&sum| (sum / self.count as f64) as f32));
        self.count = 0;
    }
}
