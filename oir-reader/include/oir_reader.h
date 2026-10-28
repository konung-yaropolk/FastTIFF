/*
 * oir-reader: the C interface.
 *
 * MPL-2.0, like the library. The licence is per file: a C or C++ program may
 * link the library, statically or dynamically, under any licence of its own;
 * changes to the library's own files stay MPL-2.0 and have to be made
 * available. See LICENSE.
 *
 * Build the library (from the repository, or an unpacked crate):
 *
 *   cargo rustc -p oir-reader --release --features capi \
 *       --crate-type cdylib,staticlib
 *
 * The header matches the library it ships with, and no other: check
 * oir_version().
 *
 * Usage, in brief: oir_read() a file (every part of a split acquisition comes
 * with it), oir_shape() for the stack, oir_plane_index() to find a plane,
 * oir_plane_copy() or oir_plane_data() for its samples, and
 * oir_acquisition_free() at the end. Every call that can fail returns an
 * OIR_* status; oir_last_error() says why one was not OIR_OK.
 *
 * Planes are row-major, `width` samples by `height` rows, of the sample type
 * in OirShape: uint8_t or uint16_t, native-endian. They are in ImageJ's xyczt
 * order: channel fastest, then slice, then time. The whole acquisition is in
 * memory once read.
 *
 * Text: paths and names passed in are NUL-terminated UTF-8. A function that
 * hands text back follows snprintf: it takes `buf` and `cap`, the room there
 * in bytes, writes as much of the text as fits and a NUL after it (cut at a
 * character boundary, never inside a UTF-8 sequence), and returns the length
 * of the whole text without the NUL. If that is >= cap, the text was cut
 * short; buf = NULL with cap = 0 asks for the length alone. A negative return
 * is minus an OIR_* status: the call could not answer, and nothing was
 * written. A value the file does not state is the empty string.
 */
#ifndef OIR_READER_H
#define OIR_READER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Statuses. */
#define OIR_OK 0               /* done */
#define OIR_CANCELLED 1        /* the progress callback asked to stop */
#define OIR_INVALID_ARGUMENT 2 /* a pointer, size, index or enum value it cannot take */
#define OIR_UNAVAILABLE 3      /* not in this build: reading a path where there is no filesystem (wasm) */
#define OIR_FAILED 4           /* the reader failed: a panic inside it */
#define OIR_IO_ERROR 5         /* a part could not be opened or read */
#define OIR_NOT_OIR 6          /* not an OIR file: no signature */
#define OIR_MALFORMED 7        /* damaged, truncated or contradictory structure or metadata */
#define OIR_NO_PLANES 8        /* no image plane: usually the parts that hold them are missing */

/* OirOptions.access: how the parts of an acquisition named by a path are read. */
#define OIR_ACCESS_AUTO 0     /* the default: memory-mapped where the build can (oir_mmap_supported), else buffered */
#define OIR_ACCESS_BUFFERED 1 /* never mapped: for network shares, or files another process may truncate */

/* OirShape.sample_type. */
#define OIR_SAMPLE_U8 0  /* uint8_t */
#define OIR_SAMPLE_U16 1 /* uint16_t, native-endian; often 10 or 12 significant bits (OIR_CHANNEL_BITS) */

/* OirPartInfo.read_path: how a part's bytes were read. */
#define OIR_READ_MAPPED 0   /* memory-mapped */
#define OIR_READ_BUFFERED 1 /* through a bounded buffer */
#define OIR_READ_MEMORY 2   /* from bytes handed to oir_read_bytes */

/* OirPartInfo.block_source: how a part's blocks were found. */
#define OIR_BLOCKS_INDEX 0 /* from its block index: a finished file */
#define OIR_BLOCKS_WALK 1  /* by walking the blocks: a file still being written, or one whose tail was lost */

/* oir_record_string fields: the record's text values, as the file states them. */
#define OIR_FIELD_FILE_VERSION 0
#define OIR_FIELD_CREATED 1 /* ISO 8601 */
#define OIR_FIELD_SYSTEM_NAME 2
#define OIR_FIELD_SYSTEM_VERSION 3
#define OIR_FIELD_OBJECTIVE 4
#define OIR_FIELD_MAGNIFICATION 5
#define OIR_FIELD_NUMERICAL_APERTURE 6
#define OIR_FIELD_SCANNER 7 /* "Resonant", "Galvano" */
#define OIR_FIELD_SAMPLING_SPEED_US 8 /* pixel dwell, us/pixel */
#define OIR_FIELD_SEQUENTIAL_MODE 9
#define OIR_FIELD_INTEGRATION 10
#define OIR_FIELD_INTEGRATION_COUNT 11
#define OIR_FIELD_ROTATION_DEG 12
#define OIR_FIELD_PAN_X_UM 13
#define OIR_FIELD_PAN_Y_UM 14
#define OIR_FIELD_ZOOM 15
#define OIR_FIELD_MIRROR_TURRET 16
#define OIR_FIELD_LASER_WAVELENGTH_NM 17
#define OIR_FIELD_LASER_TRANSMISSIVITY_PCT 18

/* oir_channel_string fields: a channel's text values, as the file states them. */
#define OIR_CHANNEL_NAME 0           /* the acquisition's own name, "CH1" */
#define OIR_CHANNEL_DETECTOR 1       /* FluoView's "Channel Name" */
#define OIR_CHANNEL_BARRIER_FILTER 2
#define OIR_CHANNEL_DICHROIC 3
#define OIR_CHANNEL_DYE 4
#define OIR_CHANNEL_EMISSION_NM 5
#define OIR_CHANNEL_BITS 6           /* significant bits per sample */
#define OIR_CHANNEL_PMT_VOLTAGE 7
#define OIR_CHANNEL_LABEL 8          /* the detector, or the name without one */

/* How to read. Always start from oir_options_default(): the field list can grow. */
typedef struct OirOptions {
    int access; /* OIR_ACCESS_* */
} OirOptions;

/* The stack the planes make up. */
typedef struct OirShape {
    uint32_t width;          /* samples per row */
    uint32_t height;         /* rows per plane */
    size_t channels;
    size_t slices;
    size_t frames;
    int sample_type;         /* OIR_SAMPLE_* */
    size_t bytes_per_sample; /* 1 or 2 */
    size_t plane_len;        /* samples per plane, width * height */
    size_t plane_count;      /* channels * slices * frames */
} OirShape;

/*
 * The acquisition record's numbers. Not stated in the file: NaN for a
 * double, 0 for a size, -1 for roundtrip. The timing is for the stack as
 * read, which may be a frame short of what the record states.
 */
typedef struct OirRecordInfo {
    double pixel_x_um;       /* microns per pixel, full precision */
    double pixel_y_um;
    double z_step_um;        /* microns between slices */
    double frame_interval_s; /* seconds between frames, from the file's timestamps */
    double duration_s;       /* seconds the stack's frames took; NaN if the timestamps stop short */
    uint32_t width;          /* the frame size as the record states it */
    uint32_t height;
    size_t slices;           /* slices a z-stack was configured for */
    size_t frame_time_count; /* timestamps oir_frame_times copies */
    size_t channel_count;    /* channels oir_channel_string reads; the record's, usually the stack's */
    size_t event_count;      /* events oir_event_name and oir_event_time_ms read */
    int roundtrip;           /* 1 the scan ran both ways, 0 one way, -1 not stated */
    bool free_run;           /* the timelapse ran free: no interval was set */
} OirRecordInfo;

/* What happened to one file of an acquisition. */
typedef struct OirPartInfo {
    uint64_t size;       /* bytes read: its length when it was opened */
    int read_path;       /* OIR_READ_* */
    int block_source;    /* OIR_BLOCKS_* */
    size_t block_count;
    size_t planes_added; /* image planes first seen in it; 0 for a part holding only metadata */
} OirPartInfo;

/* An acquisition, read, every plane in memory. Opaque; free with oir_acquisition_free(). */
typedef struct OirAcquisition OirAcquisition;

/* Progress, 0.0 to 1.0 of the bytes, on the calling thread. Return non-zero to go on, 0 to stop. */
typedef int (*OirProgress)(float fraction, void *user);

/* Why the last call on this thread that failed did; valid until the next failure. */
const char *oir_last_error(void);

/* The library's version, "major.minor.patch". */
const char *oir_version(void);

/* Whether this build memory-maps parts under OIR_ACCESS_AUTO. */
bool oir_mmap_supported(void);

/* Whether `head`, the first `len` bytes of a file, carry the OIR signature (16 bytes). False for NULL. */
bool oir_is_oir(const uint8_t *head, size_t len);

/* Fill `out` with the defaults. */
int oir_options_default(OirOptions *out);

/*
 * Read the acquisition `path` begins: it and every continuation beside it
 * (name_00001, name_00002, ... with or without .oir). `options` may be NULL
 * for the defaults, `progress` NULL for none; `user` goes to `progress`
 * untouched. On OIR_OK `*out` is the acquisition; otherwise it is NULL.
 */
int oir_read(const char *path, const OirOptions *options, OirProgress progress, void *user,
             OirAcquisition **out);

/*
 * Read an acquisition from its parts already in memory, in order: part i is
 * the lens[i] bytes at parts[i]. Only the pixels are copied; the buffers are
 * free again when this returns. On OIR_OK `*out` is the acquisition;
 * otherwise it is NULL.
 */
int oir_read_bytes(const uint8_t *const *parts, const size_t *lens, size_t n_parts,
                   OirProgress progress, void *user, OirAcquisition **out);

/* Free an acquisition; NULL is ignored. */
void oir_acquisition_free(OirAcquisition *a);

/* The stack's shape. */
int oir_shape(const OirAcquisition *a, OirShape *out);

/* Where channel c of slice z at time t is among the planes. */
int oir_plane_index(const OirAcquisition *a, size_t c, size_t z, size_t t, size_t *index);

/* Planes left out because the acquisition did not finish them; 0 for NULL. */
size_t oir_dropped_planes(const OirAcquisition *a);

/*
 * Copy plane i into `out`: plane_len samples, plane_len * bytes_per_sample
 * bytes. `out_bytes`, the room at `out`, must be at least that.
 */
int oir_plane_copy(const OirAcquisition *a, size_t i, void *out, size_t out_bytes);

/*
 * Plane i without a copy: `*data` points to its plane_len samples (uint8_t or
 * uint16_t, by the shape's sample_type), read-only, valid until the
 * acquisition is freed.
 */
int oir_plane_data(const OirAcquisition *a, size_t i, const void **data);

/* The record's numbers and the counts of its lists. */
int oir_record_info(const OirAcquisition *a, OirRecordInfo *out);

/* Copy the frame timestamps, ms, sorted: frame_time_count doubles; `len`, the room at `out`, must be at least that. */
int oir_frame_times(const OirAcquisition *a, double *out, size_t len);

/* One of the record's text values, an OIR_FIELD_*. Text convention above. */
ptrdiff_t oir_record_string(const OirAcquisition *a, int field, char *buf, size_t cap);

/* One of channel `index`'s text values, an OIR_CHANNEL_*. Text convention above. */
ptrdiff_t oir_channel_string(const OirAcquisition *a, size_t index, int field, char *buf, size_t cap);

/* Event `index`'s name ("DRS"). Text convention above. */
ptrdiff_t oir_event_name(const OirAcquisition *a, size_t index, char *buf, size_t cap);

/* When event `index` happened, ms from the start of the acquisition. */
int oir_event_time_ms(const OirAcquisition *a, size_t index, double *time_ms);

/*
 * The record as FluoView exports it beside an OIR, "key"\t"value" lines, for
 * the frames read; `file_name` is the name it states. Empty when the file
 * carried nothing recognisable. Text convention above.
 */
ptrdiff_t oir_fluoview_text(const OirAcquisition *a, const char *file_name, char *buf, size_t cap);

/* How many files the acquisition was read from; 0 for NULL. */
size_t oir_part_count(const OirAcquisition *a);

/* What happened to part `index`. */
int oir_part_info(const OirAcquisition *a, size_t index, OirPartInfo *out);

/* Part `index`'s path; empty for one read from memory. Text convention above. */
ptrdiff_t oir_part_path(const OirAcquisition *a, size_t index, char *buf, size_t cap);

/* Why part `index` could not be mapped (and was buffered instead); empty if it was not tried or worked. Text convention above. */
ptrdiff_t oir_part_map_error(const OirAcquisition *a, size_t index, char *buf, size_t cap);

/* How many files make up the acquisition `path` begins, without reading them; `path` counts even if missing. */
int oir_acquisition_part_count(const char *path, size_t *count);

/* File `index` of the acquisition `path` begins; 0 is `path`. Looks at the directory again each call. Text convention above. */
ptrdiff_t oir_acquisition_part(const char *path, size_t index, char *buf, size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* OIR_READER_H */
