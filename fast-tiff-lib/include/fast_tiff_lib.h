/*
 * fast-tiff-lib: the C interface.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * The MPL is a file-level copyleft: a C or C++ program may link this
 * library, statically or dynamically, under any licence of its own; changes
 * to the library's own files stay MPL-2.0 and have to be made available.
 *
 * Build the library (from the repository, or an unpacked crate):
 *
 *   cargo rustc -p fast-tiff-lib --release --features capi \
 *       --crate-type cdylib,staticlib
 *
 * The header matches the library it ships with, and no other: check
 * ftl_version().
 *
 * Reading, in brief: ftl_open() a file (or ftl_open_bytes() a buffer),
 * ftl_frame_count() and ftl_frame_info() for its frames (one per IFD),
 * ftl_metadata() for its shape and calibration, ftl_sample_count() to size a
 * buffer and ftl_read_u8/u16/f32() to decode a frame into it, and
 * ftl_stack_free() at the end. Writing: fill an FtlWriterOptions with
 * ftl_writer_options_default(), change what differs, ftl_writer_create(),
 * ftl_writer_write_*() each frame, ftl_writer_finish().
 *
 * Every call that can fail returns an FTL_* status; ftl_last_error() says why
 * one was not FTL_OK. A failure has the same status whichever call meets it:
 * a NULL where a pointer is needed, an index past the end or a buffer too
 * small is FTL_INVALID_ARGUMENT everywhere, and a failure inside the library
 * that no input should cause (a panic, a codec refusing what it takes
 * whatever it is, a decoder handing back the wrong size) is FTL_FAILED from
 * any call; each declaration below names the rest it can return. Strings
 * come in as NUL-terminated UTF-8. Strings go out
 * by the snprintf convention: the caller passes a buffer and its capacity in
 * bytes, the call writes as much as fits, NUL-terminated, and returns the
 * whole length, not counting the NUL; buf = NULL, cap = 0 asks for the size.
 * A double the file does not give, or the caller does not set, is NaN.
 *
 * A frame decodes to its sample planes one after another: width * height
 * values of sample 0, then of sample 1, and so on, rows top to bottom.
 */
#ifndef FAST_TIFF_LIB_H
#define FAST_TIFF_LIB_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Statuses. */
/* 0 to 4 mean the same in OIR_, S2P_ and TR_; this library's own statuses follow from 5. */
#define FTL_OK 0               /* done */
#define FTL_CANCELLED 1        /* reserved: never returned here, kept so 0-4 match the other C APIs */
#define FTL_INVALID_ARGUMENT 2 /* a pointer, size, index, buffer or option it cannot take, a frame the reader called does not read (a 16-bit frame through ftl_read_u8), or a writer used out of turn */
#define FTL_UNAVAILABLE 3      /* not in this build or on this target (ftl_open without `mmap`, ZSTD without `codec-zstd`) */
#define FTL_FAILED 4           /* something inside the library went wrong (a panic, a codec failing where no input should) */
#define FTL_IO 5               /* a file could not be opened, created, mapped, read or written */
#define FTL_NOT_TIFF 6         /* the bytes are not a TIFF at all */
#define FTL_FORMAT 7           /* a TIFF it cannot read: damaged, or using a codec, bit depth, sample format or predictor it has no code for */
#define FTL_TOO_LARGE 8        /* larger than this library or target can hold: a size past the address space, memory refused, too many frames */

/* FtlFrameInfo.sample_format: TIFF's SampleFormat codes. */
#define FTL_SAMPLE_UNSIGNED 1
#define FTL_SAMPLE_SIGNED 2
#define FTL_SAMPLE_FLOAT 3

/* Compression: TIFF's codes. FtlFrameInfo.compression may hold any other code the file used. */
#define FTL_COMPRESSION_NONE 1
#define FTL_COMPRESSION_LZW 5
#define FTL_COMPRESSION_DEFLATE 8 /* 32946 reads back as 8 */
#define FTL_COMPRESSION_PACKBITS 32773
#define FTL_COMPRESSION_ZSTD 50000 /* needs the `codec-zstd` feature (a default) */

/* FtlWriterOptions.sample_type. */
#define FTL_TYPE_U8 0
#define FTL_TYPE_I8 1
#define FTL_TYPE_U16 2 /* the default */
#define FTL_TYPE_I16 3
#define FTL_TYPE_U32 4
#define FTL_TYPE_I32 5
#define FTL_TYPE_F32 6
#define FTL_TYPE_U64 7
#define FTL_TYPE_I64 8
#define FTL_TYPE_F64 9

/* Metadata dialects: what a file used (FtlMetadata.format), what to write (FtlWriterOptions.metadata_format). */
#define FTL_METADATA_NONE 0 /* none recognised; none written */
#define FTL_METADATA_IMAGEJ 1
#define FTL_METADATA_OME 2

/* Display modes. */
#define FTL_MODE_GRAYSCALE 0
#define FTL_MODE_COMPOSITE 1
#define FTL_MODE_COLOR 2

/* FtlWriterOptions.compression_level: the library's default (Deflate 6, ZSTD 3). */
#define FTL_LEVEL_DEFAULT INT32_MIN

/* One frame's layout, as the file declares it. */
typedef struct FtlFrameInfo {
    uint32_t width;
    uint32_t height;
    uint16_t bits_per_sample;   /* 4, 8, 16, 32 or 64 */
    uint16_t samples_per_pixel;
    uint16_t sample_format;     /* FTL_SAMPLE_* */
    uint16_t compression;       /* TIFF code; FTL_COMPRESSION_* */
    uint16_t predictor;         /* 1 none, 2 horizontal, 3 floating point */
    uint16_t photometric;       /* 1 grayscale, 2 RGB, 3 palette, 5 separated (CMYK) */
    uint16_t planar_config;     /* 1 chunky, 2 planar (as stored; the readers always give planes) */
    uint16_t ink_set;           /* 1 CMYK, 2 other inks */
    uint32_t rows_per_strip;
    uint32_t tile_width;        /* 0 for a stripped frame */
    uint32_t tile_height;       /* 0 for a stripped frame */
    bool tiled;
    bool rgb;                   /* 3 or more samples that are red, green, blue */
    bool planar;                /* samples stored as whole planes */
    bool palette;               /* 4- or 8-bit indices into a colour map */
    bool cmyk;                  /* what ftl_read_rgb_u8/u16 convert */
} FtlFrameInfo;

/*
 * What the file's metadata (ImageJ or OME) says. Not given: NaN, and -1 for
 * loop_playback. channels x slices x frames are the file's own labels, in
 * ImageJ's xyczt plane order, or 1 x 1 x the frame count when it has none.
 */
typedef struct FtlMetadata {
    size_t channels;
    size_t slices;
    size_t frames;
    int format;            /* FTL_METADATA_* */
    int mode;              /* FTL_MODE_* */
    double pixel_width;    /* in the unit ftl_metadata_unit() gives */
    double pixel_height;
    double spacing;        /* Z step, same unit */
    double calibration[2]; /* a raw sample r stands for calibration[0] + calibration[1] * r */
    double fps;
    double frame_interval_s;
    int loop_playback;     /* 1, 0, or -1 for not said */
} FtlMetadata;

/* A reading of channels x slices x frames; see ftl_resolve_dimensions(). */
typedef struct FtlDims {
    size_t channels;
    size_t slices;
    size_t frames;
    bool triple_axis_warning; /* several channels with Z and time both real */
} FtlDims;

/*
 * How to write a stack. Always start from ftl_writer_options_default(): the
 * field list can grow. A double left NaN, a pointer left NULL and
 * loop_playback left -1 are not written. The fields from `mode` on are used
 * only when metadata_format is not FTL_METADATA_NONE; the pixel size also goes
 * to the resolution tags. Strings are copied by ftl_writer_create().
 */
typedef struct FtlWriterOptions {
    uint32_t width;
    uint32_t height;
    int sample_type;            /* FTL_TYPE_* */
    uint16_t samples_per_pixel; /* 1 grayscale, 3 RGB, 4+ with cmyk */
    uint16_t compression;       /* FTL_COMPRESSION_* */
    int32_t compression_level;  /* Deflate 0-9, ZSTD 1-22, or FTL_LEVEL_DEFAULT */
    uint32_t rows_per_strip;    /* 0 for the library's choice */
    bool planar;                /* frames go in plane after plane, not interleaved */
    bool cmyk;                  /* tag the samples as C, M, Y, K ink (unsigned 8/16-bit, 4+ samples) */
    bool predictor;             /* horizontal (integers) or floating-point predictor */
    bool bigtiff;               /* force BigTIFF; it is automatic past 4 GiB */
    const char *description;    /* a verbatim ImageDescription instead of metadata */
    int metadata_format;        /* FTL_METADATA_*; NONE writes no structured metadata */
    int mode;                   /* FTL_MODE_* */
    size_t channels;            /* per time point; planes go in xyczt order */
    size_t slices;              /* per time point; time points are counted at the finish */
    const char *unit;           /* e.g. "um" */
    double pixel_width;         /* both or neither */
    double pixel_height;
    double spacing;             /* Z step */
    double calibration[2];      /* c0, c1; both or neither */
    double range[2];            /* display window min, max; both or neither */
    double fps;
    double frame_interval_s;
    int loop_playback;          /* 1, 0, or -1 for not written */
    const char *const *channel_names; /* named_channels names (OME) */
    const uint8_t *channel_colors;    /* named_channels RGB triples, 3 bytes each */
    size_t named_channels;
} FtlWriterOptions;

/* An open stack. Opaque; free with ftl_stack_free(). Readable from several threads at once. */
typedef struct FtlStack FtlStack;

/* A writer. Opaque; end with ftl_writer_finish() or ftl_writer_free(). One thread at a time. */
typedef struct FtlWriter FtlWriter;

/* Why the last call on this thread that failed did; valid until the next failure. */
const char *ftl_last_error(void);

/* The library's version, "major.minor.patch". */
const char *ftl_version(void);

/* Let large decodes, and the writer's compression, use several cores. Process-wide; off by default. */
void ftl_set_parallel_decode(bool enabled);

/* ---- reading ---- */

/*
 * Open and index the file at `path`, memory-mapped. On FTL_OK `*out` is the stack, else NULL. The
 * file must not change while open. FTL_IO when it cannot be opened or mapped; FTL_NOT_TIFF,
 * FTL_FORMAT or FTL_TOO_LARGE for what is in it; FTL_UNAVAILABLE in a build without `mmap`.
 */
int ftl_open(const char *path, FtlStack **out);

/*
 * Index a TIFF held in memory; the `len` bytes are copied. On FTL_OK `*out` is the stack, else
 * NULL. FTL_NOT_TIFF, FTL_FORMAT or FTL_TOO_LARGE for what is in it.
 */
int ftl_open_bytes(const void *data, size_t len, FtlStack **out);

/* Free a stack; NULL is ignored. */
void ftl_stack_free(FtlStack *s);

/* How many frames (IFDs) the stack has; 0 for NULL. */
size_t ftl_frame_count(const FtlStack *s);

/* Whether the file is a BigTIFF; whether it is big-endian. Informational: values come back in host order. */
bool ftl_stack_is_bigtiff(const FtlStack *s);
bool ftl_stack_is_big_endian(const FtlStack *s);

/* Frame `index`'s layout. */
int ftl_frame_info(const FtlStack *s, size_t index, FtlFrameInfo *out);

/* What the file's metadata says. */
int ftl_metadata(const FtlStack *s, FtlMetadata *out);

/*
 * The calibration unit (UTF-8, so a micron is "\xC2\xB5m") and the first IFD's ImageDescription,
 * verbatim, snprintf-style: the whole length back, 0 when there is none, and a cut never splits a
 * character. A NULL stack, or buf NULL with cap > 0, is -FTL_INVALID_ARGUMENT.
 */
ptrdiff_t ftl_metadata_unit(const FtlStack *s, char *buf, size_t cap);
ptrdiff_t ftl_description(const FtlStack *s, char *buf, size_t cap);

/*
 * channels x slices x frames as a viewer reads them: classified by size rather
 * than by the file's labels (a "channels=100" is time), Z folded into time
 * unless both are real. The product never changes. A count of 0, or a product
 * past size_t, is FTL_INVALID_ARGUMENT.
 */
int ftl_resolve_dimensions(size_t channels, size_t slices, size_t frames, FtlDims *out);

/*
 * The frame holding channel c, slice z, time point t: (t * slices + z) *
 * channels + c. `dims` NULL for the file's own labels, or a resolution of them.
 * A position outside `dims`, or a frame the file does not have, is
 * FTL_INVALID_ARGUMENT.
 */
int ftl_plane_index(const FtlStack *s, const FtlDims *dims, size_t c, size_t z, size_t t, size_t *index);

/*
 * How many values ftl_read_u8/u16/f32 write for frame `index`: width * height * samples_per_pixel.
 * FTL_TOO_LARGE when that does not fit in a size_t.
 */
int ftl_sample_count(const FtlStack *s, size_t index, size_t *count);

/*
 * Decode frame `index` into `out`, which holds `len` values: at least
 * ftl_sample_count(), or FTL_INVALID_ARGUMENT. Planes one after another.
 * FTL_FORMAT for a frame it cannot decode (damaged, or a codec, bit depth or
 * sample format it has no code for), FTL_TOO_LARGE for one too large to
 * decode here, FTL_UNAVAILABLE for ZSTD in a build without `codec-zstd`, and
 * FTL_INVALID_ARGUMENT for a frame the reader called does not take.
 *
 * ftl_read_u8:  8-bit frames as they are, 4-bit palette indices one per byte; wider is
 *               FTL_INVALID_ARGUMENT.
 * ftl_read_u16: any depth for display: 8-bit widened onto 0-65535, signed offset to unsigned,
 *               32/64-bit rescaled from range[0]..range[1], or each plane's own min..max when
 *               `range` is NULL (ftl_float_minmax gives one range for the whole frame).
 *               4-bit is FTL_INVALID_ARGUMENT: ftl_read_u8 and ftl_read_f32 read it.
 * ftl_read_f32: any depth as float in the file's own units (8-bit 200 is 200.0, signed 16-bit
 *               -1 is -1.0): no widening, offset or rescale. 16-bit floating point is FTL_FORMAT,
 *               having no reader here.
 */
int ftl_read_u8(const FtlStack *s, size_t index, uint8_t *out, size_t len);
int ftl_read_u16(const FtlStack *s, size_t index, const float *range, uint16_t *out, size_t len);
int ftl_read_f32(const FtlStack *s, size_t index, float *out, size_t len);

/*
 * A CMYK frame (FtlFrameInfo.cmyk) converted to red, green and blue planes:
 * `len` at least 3 * width * height. ftl_read_rgb_u8 takes 8-bit CMYK,
 * ftl_read_rgb_u16 8- and 16-bit; anything else is FTL_INVALID_ARGUMENT. An
 * RGB frame's planes are already red, green and blue. Otherwise the statuses
 * of the readers above.
 */
int ftl_read_rgb_u8(const FtlStack *s, size_t index, uint8_t *out, size_t len);
int ftl_read_rgb_u16(const FtlStack *s, size_t index, uint16_t *out, size_t len);

/*
 * A 32/64-bit frame's smallest and largest finite value, across all its planes
 * (0 and 1 for a constant frame); NaN, NaN for 8/16-bit. Pass it as
 * ftl_read_u16's range to put every plane, or several frames, on one scale.
 * Either output may be NULL. The statuses of the readers above, for the
 * decode it needs.
 */
int ftl_float_minmax(const FtlStack *s, size_t index, float *min, float *max);

/* ---- writing ---- */

/* Fill `out` with the defaults: 16-bit grayscale, uncompressed, no metadata. Width and height are 0: set them. */
int ftl_writer_options_default(FtlWriterOptions *out);

/*
 * Create (or truncate) `path` and a writer for it. Options it refuses leave
 * no file behind. On FTL_OK `*out` is the writer, else NULL. Options it
 * cannot take are FTL_INVALID_ARGUMENT, a frame too large to address
 * FTL_TOO_LARGE, ZSTD without `codec-zstd` or a target with no filesystem
 * FTL_UNAVAILABLE, and a file it cannot create FTL_IO.
 */
int ftl_writer_create(const char *path, const FtlWriterOptions *options, FtlWriter **out);

/*
 * Append one frame: `count` = width * height * samples_per_pixel samples,
 * interleaved, or plane after plane with `planar`. Each takes its own sample
 * type only; ftl_writer_write_bytes takes any, as little-endian bytes
 * (`len` = count * bytes per sample). The wrong type or count is
 * FTL_INVALID_ARGUMENT, a failed write FTL_IO.
 */
int ftl_writer_write_u8(FtlWriter *w, const uint8_t *samples, size_t count);
int ftl_writer_write_u16(FtlWriter *w, const uint16_t *samples, size_t count);
int ftl_writer_write_f32(FtlWriter *w, const float *samples, size_t count);
int ftl_writer_write_f64(FtlWriter *w, const double *samples, size_t count);
int ftl_writer_write_bytes(FtlWriter *w, const void *data, size_t len);

/* How many frames the writer has taken; 0 for NULL. */
size_t ftl_writer_frames_written(const FtlWriter *w);

/*
 * Write the directory and metadata and close the file; it is not a TIFF until
 * this returns FTL_OK. Frees the writer whatever it returns. No frames, or a
 * frame count that does not divide into channels * slices, is
 * FTL_INVALID_ARGUMENT; a failed write FTL_IO.
 */
int ftl_writer_finish(FtlWriter *w);

/* Abandon a writer unfinished; the file left behind is not a valid TIFF. NULL is ignored. */
void ftl_writer_free(FtlWriter *w);

#ifdef __cplusplus
}
#endif

#endif /* FAST_TIFF_LIB_H */
