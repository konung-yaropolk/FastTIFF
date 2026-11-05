# image-registration-support

The parts of [`suite2p-registration`](../suite2p-registration) and
[`turboreg-registration`](../turboreg-registration) that are not their
algorithms, kept once instead of once per crate.

- `par`: every parallel loop of both crates goes through it. With the
  `threads` feature (the default) it is rayon's prelude plus a few ordered
  helpers; without it the same names are sequential stand-ins, and rayon is
  not compiled at all, which is what a wasm build wants.
- `area`: the largest rectangle every registered frame covers, upright or at
  an angle (the search behind a crop to the common area), given each crate's
  own test of whether a frame reads a point from inside itself.
- `gpu` (feature `gpu`): the one GPU device every registration in the process
  runs on (a host's, given once with `use_device`, or one opened on first
  use), and the error-scoped habits that keep a card shared with a renderer
  alive.

It exists for those two crates; its API follows what they need.

Licensed GPL-3.0-only, like the crates it serves.
