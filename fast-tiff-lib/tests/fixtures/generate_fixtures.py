#!/usr/bin/env python3
"""Generates the TIFF fixture matrix that tests/libtiff_fixtures.rs decodes.

Two independent producers cross-validate fast-tiff-lib's reader:

- **tifffile** (`tff_` prefix): the de-facto scientific-Python TIFF writer —
  full parameter control (dtype, compression, predictor 2/3, byte order,
  strips, RGB, ImageJ metadata).
- **Pillow** (`pil_` prefix): its compressed-TIFF path runs the *actual
  libtiff* encoders, so these files carry genuine libtiff LZW/Deflate/PackBits
  streams.

Pixel values are pure functions of the flat sample index `g` (page-major,
row-major, chunky-interleaved), so the Rust test recomputes the expected data
without sidecar files. THE FORMULAS MUST MATCH tests/libtiff_fixtures.rs:

    u8 : (g*7 + 13) % 256
    i8 : that - 128
    u16: (g*131 + 17) % 65536
    i16: that - 32768
    u32: (g*97 + 5) % 100000          (< 2^24, so exact as f32)
    i32: that - 50000
    f32: (g % 2000) * 0.25 - 250.0    (exact in f32)

Filename grammar (the Rust test parses the first four tokens; the rest is
informational): {gen}_{dtype}_spp{s}_p{pages}_{info}.tif
Prefix `err_` = the file must FAIL to open; `ij_` = ImageJ metadata checks.

Deterministic: no RNG, fixed sizes. Rerunning overwrites the same bytes
(modulo library versions changing their container layout, which is fine — the
pixel contract is what's tested).
"""

import os
import sys

import numpy as np
import tifffile

W, H = 23, 11  # deliberately odd sizes to catch stride/rounding bugs
OUT = os.path.dirname(os.path.abspath(__file__))

written = []


def flat(dtype: str, pages: int, spp: int = 1) -> np.ndarray:
    """The shared pixel formula over the flat sample index."""
    g = np.arange(pages * H * W * spp, dtype=np.int64)
    if dtype == "u8":
        a = (g * 7 + 13) % 256
    elif dtype == "i8":
        a = (g * 7 + 13) % 256 - 128
    elif dtype == "u16":
        a = (g * 131 + 17) % 65536
    elif dtype == "i16":
        a = (g * 131 + 17) % 65536 - 32768
    elif dtype == "u32":
        a = (g * 97 + 5) % 100000
    elif dtype == "i32":
        a = (g * 97 + 5) % 100000 - 50000
    elif dtype == "f32":
        a = (g % 2000) * 0.25 - 250.0
    else:
        raise ValueError(dtype)
    np_dtype = {"u8": np.uint8, "i8": np.int8, "u16": np.uint16, "i16": np.int16,
                "u32": np.uint32, "i32": np.int32, "f32": np.float32}[dtype]
    a = a.astype(np_dtype)
    shape = (pages, H, W, spp) if spp > 1 else (pages, H, W)
    return a.reshape(shape)


def tff(name: str, dtype: str, pages: int, spp: int = 1, **kwargs):
    """Write one tifffile fixture; metadata=None keeps descriptions out."""
    path = os.path.join(OUT, name)
    arr = flat(dtype, pages, spp)
    if spp > 1:
        kwargs.setdefault("photometric", "rgb")
    try:
        tifffile.imwrite(path, arr, metadata=None, **kwargs)
        written.append(name)
    except Exception as e:  # report and continue: a partial matrix still tests
        print(f"SKIP {name}: {e}", file=sys.stderr)


# --- 1. Baselines: every dtype, uncompressed, little-endian, 2 pages ---
for dt in ["u8", "i8", "u16", "i16", "u32", "i32", "f32"]:
    tff(f"tff_{dt}_spp1_p2_none-le.tif", dt, 2)

# --- 2. Codecs, multi-strip (rows-per-strip 4 over height 11 = 3 strips) ---
tff("tff_u16_spp1_p2_lzw-rps4.tif", "u16", 2, compression="lzw", rowsperstrip=4)
tff("tff_u16_spp1_p2_zip-rps4.tif", "u16", 2, compression="zlib", rowsperstrip=4)
tff("tff_u16_spp1_p2_pb-rps4.tif", "u16", 2, compression="packbits", rowsperstrip=4)

# --- 3. Predictor 2 (integer horizontal differencing), incl. 32-bit ---
tff("tff_u8_spp1_p2_zip-pred2.tif", "u8", 2, compression="zlib", predictor=2)
tff("tff_u16_spp1_p2_lzw-pred2-rps4.tif", "u16", 2, compression="lzw", predictor=2, rowsperstrip=4)
tff("tff_u16_spp1_p2_zip-pred2.tif", "u16", 2, compression="zlib", predictor=2)
tff("tff_i16_spp1_p2_lzw-pred2.tif", "i16", 2, compression="lzw", predictor=2)
tff("tff_u32_spp1_p2_zip-pred2.tif", "u32", 2, compression="zlib", predictor=2)

# --- 4. Predictor 3 (TechNote 3 floating point) — the cross-validation the
# --- in-crate roundtrip tests can't provide on their own ---
tff("tff_f32_spp1_p2_zip-pred3.tif", "f32", 2, compression="zlib", predictor=3)
tff("tff_f32_spp1_p2_lzw-pred3.tif", "f32", 2, compression="lzw", predictor=3)

# --- 5. Big-endian files (incl. BE floating-point predictor) ---
tff("tff_u16_spp1_p2_none-be.tif", "u16", 2, byteorder=">")
tff("tff_i16_spp1_p2_none-be.tif", "i16", 2, byteorder=">")
tff("tff_u16_spp1_p2_lzw-pred2-be.tif", "u16", 2, byteorder=">", compression="lzw", predictor=2)
tff("tff_f32_spp1_p2_zip-pred3-be.tif", "f32", 2, byteorder=">", compression="zlib", predictor=3)

# --- 6. Chunky RGB ---
tff("tff_u8_spp3_p2_none.tif", "u8", 2, spp=3)
tff("tff_u8_spp3_p2_lzw-pred2.tif", "u8", 2, spp=3, compression="lzw", predictor=2)
tff("tff_u16_spp3_p2_zip-pred2.tif", "u16", 2, spp=3, compression="zlib", predictor=2)

# --- 7. ImageJ hyperstack metadata (2 channels x 3 time frames) ---
try:
    arr = flat("u16", 6).reshape(3, 2, H, W)  # TCYX; plane order = ImageJ's czt
    tifffile.imwrite(
        os.path.join(OUT, "ij_u16_spp1_p6_hyperstack.tif"),
        arr,
        imagej=True,
        metadata={"axes": "TCYX", "mode": "composite", "unit": "um",
                  "spacing": 0.5, "fps": 10.0, "loop": False},
    )
    written.append("ij_u16_spp1_p6_hyperstack.tif")
except Exception as e:
    print(f"SKIP ij fixture: {e}", file=sys.stderr)

# --- 8. Tiled file: fast-tiff-lib must refuse it with a clear error ---
tff("err_u16_spp1_p1_tiled.tif", "u16", 1, tile=(16, 16))

# --- 9. Pillow fixtures: genuine libtiff-encoded compressed streams ---
try:
    from PIL import Image

    def pil(name: str, mode: str, dtype: str, spp: int, compression: str):
        path = os.path.join(OUT, name)
        arr = flat(dtype, 1, spp)[0]
        try:
            img = Image.fromarray(arr, mode=mode)
            img.save(path, compression=compression)
            written.append(name)
        except Exception as e:
            print(f"SKIP {name}: {e}", file=sys.stderr)

    pil("pil_u8_spp1_p1_lzw.tif", "L", "u8", 1, "tiff_lzw")
    pil("pil_u8_spp1_p1_pb.tif", "L", "u8", 1, "packbits")
    pil("pil_u8_spp3_p1_zip.tif", "RGB", "u8", 3, "tiff_adobe_deflate")
    pil("pil_u16_spp1_p1_lzw.tif", "I;16", "u16", 1, "tiff_lzw")
except ImportError:
    print("SKIP pil fixtures: Pillow not installed", file=sys.stderr)

print(f"wrote {len(written)} fixtures:")
for name in sorted(written):
    print(f"  {name}")
