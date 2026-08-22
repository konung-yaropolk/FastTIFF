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


def tff_planar(name: str, dtype: str, pages: int, spp: int = 4, **kwargs):
    """Write one planar (PlanarConfiguration=2) fixture.

    tifffile reads the sample axis positionally: a page shaped (Y, X, S) is
    chunky, and forcing planarconfig on it makes tifffile reinterpret the axes
    as (S, Y, X) rather than transpose them -- which silently produces an
    11-sample 4x23 image instead of a 4-sample 23x11 one. So the array is built
    plane-major here instead.

    Reshaping the flat buffer (rather than transposing it) is deliberate: it
    leaves plane `pl` holding the contiguous run of flat sample indices
    [pl*H*W, (pl+1)*H*W) within its page, which is exactly what the Rust
    checker predicts once it sees PlanarConfiguration=2 on the frame.
    """
    path = os.path.join(OUT, name)
    arr = flat(dtype, pages, spp).reshape(pages, spp, H, W)
    kwargs.setdefault("photometric", "separated")
    try:
        tifffile.imwrite(path, arr, metadata=None, planarconfig="separate", **kwargs)
        written.append(name)
    except Exception as e:
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

# --- 4b. ZSTD (tag 50000, libtiff/GDAL extension; needs imagecodecs) ---
tff("tff_u16_spp1_p2_zstd-rps4.tif", "u16", 2, compression="zstd", rowsperstrip=4)
tff("tff_u16_spp1_p2_zstd-pred2.tif", "u16", 2, compression="zstd", predictor=2)
tff("tff_f32_spp1_p2_zstd-pred3.tif", "f32", 2, compression="zstd", predictor=3)

# --- 4c. BigTIFF (magic 43, 64-bit offsets) — small files are still valid ---
tff("tff_u16_spp1_p2_none-le-bigtiff.tif", "u16", 2, bigtiff=True)
tff("tff_u16_spp1_p2_zip-pred2-bigtiff.tif", "u16", 2, bigtiff=True, compression="zlib", predictor=2)
tff("tff_f32_spp1_p2_zstd-pred3-bigtiff.tif", "f32", 2, bigtiff=True, compression="zstd", predictor=3)
tff("tff_u16_spp1_p2_none-be-bigtiff.tif", "u16", 2, bigtiff=True, byteorder=">")

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

# --- 8. Tiled files. A tile is bounded on both axes, unlike a strip, which is
# --- what lets a window of a huge image be read without touching the rest of
# --- its rows. Tiles are stored full-size and padded at the right and bottom
# --- edges, so 23x11 in 16x16 tiles exercises padding on both axes at once.
tff("tff_u16_spp1_p1_tiled.tif", "u16", 1, tile=(16, 16))
tff("tff_u8_spp3_p1_tiled-lzw-pred2.tif", "u8", 1, spp=3, tile=(16, 16),
    compression="lzw", predictor=2)
tff("tff_f32_spp1_p1_tiled-zip-pred3.tif", "f32", 1, tile=(16, 16),
    compression="zlib", predictor=3)


# --- 8a. Tiled files with a grid more than one tile deep, which the fixture
# --- matrix above cannot express (its frames are 11 rows, shorter than a tile).
# --- Verified by tests/tiled.rs instead; the `tld_` prefix is what tells
# --- libtiff_fixtures.rs to leave them alone.
TW, TH = 100, 70  # 7 x 5 tiles of 16, with both edges partial


def tld(name: str, dtype: str, spp: int = 1, planar: bool = False, **kwargs):
    """A tiled fixture on its own, larger geometry.

    Planar needs the sample axis *first*: tifffile reads axes positionally, so
    handing it (Y, X, S) and asking for separate planes makes it reinterpret the
    shape rather than transpose it -- which silently produces a 70-sample 3x100
    image instead of a 3-sample 100x70 one.
    """
    path = os.path.join(OUT, name)
    g = np.arange(TH * TW * spp, dtype=np.int64)
    a = ((g * 7 + 13) % 256) if dtype == "u8" else ((g * 131 + 17) % 65536)
    a = a.astype(np.uint8 if dtype == "u8" else np.uint16)
    if planar:
        arr = a.reshape(spp, TH, TW)
        kwargs["planarconfig"] = "separate"
    else:
        arr = a.reshape((TH, TW, spp) if spp > 1 else (TH, TW))
    if spp > 1:
        kwargs.setdefault("photometric", "rgb")
    try:
        tifffile.imwrite(path, arr, metadata=None, tile=(16, 16), **kwargs)
        written.append(name)
    except Exception as e:
        print(f"SKIP {name}: {e}", file=sys.stderr)


tld("tld_u16_spp1_p1_grid.tif", "u16")
tld("tld_u16_spp1_p1_grid-lzw.tif", "u16", compression="lzw")
tld("tld_u8_spp3_p1_grid-pred2.tif", "u8", spp=3, compression="zlib", predictor=2)
tld("tld_u8_spp3_p1_grid-planar.tif", "u8", spp=3, planar=True)

# --- 8b. CMYK / Separated (photometric=5). Ink coverage, not light: the
# --- reader must both hand back the four raw plates unchanged AND offer the
# --- converted RGB. Chunky, planar and compressed all included, since the
# --- conversion runs after the plane gather and must not care which it was.
tff("tff_u8_spp4_p1_cmyk.tif", "u8", 1, spp=4, photometric="separated")
tff("tff_u8_spp4_p1_cmyk-lzw.tif", "u8", 1, spp=4, photometric="separated",
    compression="lzw")
tff_planar("tff_u8_spp4_p2_cmyk-planar.tif", "u8", 2, spp=4)
tff("tff_u16_spp4_p1_cmyk.tif", "u16", 1, spp=4, photometric="separated")
tff("tff_u16_spp4_p1_cmyk-zip-pred2.tif", "u16", 1, spp=4,
    photometric="separated", compression="zlib", predictor=2)

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
    # Pillow CMYK goes through the real libtiff separated-image encoder.
    pil("pil_u8_spp4_p1_cmyk-lzw.tif", "CMYK", "u8", 4, "tiff_lzw")
except ImportError:
    print("SKIP pil fixtures: Pillow not installed", file=sys.stderr)

print(f"wrote {len(written)} fixtures:")
for name in sorted(written):
    print(f"  {name}")
