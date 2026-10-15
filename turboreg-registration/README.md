# turboreg-registration

Image registration by the TurboReg/StackReg pyramid method, in Rust: cubic
B-spline interpolation, a least-squares spline pyramid, and Thevenaz's inverse
Marquardt-Levenberg optimizer. Five transformation models, from translation to
bilinear.

The API mirrors [pystackreg](https://github.com/glichtner/pystackreg) so results
can be compared against the reference implementation.

## Licensing

GPL-3.0-only. The algorithm is Thevenaz, Ruttimann & Unser's, shipped by EPFL as
TurboReg/StackReg, which EPFL relicensed to GPL-3-or-later in April 2023.

This crate is **not** a port of pystackreg's code. pystackreg carries the
pre-2023 terms ("you should not redistribute it without our consent"), which are
not GPL-compatible; it lends this crate its interface and nothing else.

Please cite:

> P. Thevenaz, U.E. Ruttimann, M. Unser, "A Pyramid Approach to Subpixel
> Registration Based on Intensity", IEEE Transactions on Image Processing,
> vol. 7, no. 1, pp. 27-41, January 1998.
