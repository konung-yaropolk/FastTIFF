/*
 * turboreg-registration: the C interface.
 *
 * Copyright (C) 2026 SciWare LLC. GPL-3.0-only, like the library: a program
 * that links it, statically or dynamically, and is distributed, is a
 * derivative work and goes out under the GPL too. See LICENSE, and cite
 * Thevenaz, Ruttimann & Unser (1998), as EPFL asks.
 *
 * Build the library (from the repository, or an unpacked crate):
 *
 *   cargo rustc -p turboreg-registration --release --features capi \
 *       --crate-type cdylib,staticlib
 *
 * Add `--features capi,gpu` for the GPU backend. The header matches the
 * library it ships with, and no other: check tr_version().
 *
 * Usage, in brief: fill a TrSettings with tr_settings_default(), change what
 * differs, call tr_register() once on the channel to measure, then
 * tr_registered_apply() on every frame of every channel, and
 * tr_registered_free() at the end. Every call returns a TR_* status;
 * tr_last_error() says why one was not TR_OK.
 *
 * Frames are float, row-major, `ly` rows of `lx` columns. A stack is its
 * frames one after another.
 */
#ifndef TURBOREG_REGISTRATION_H
#define TURBOREG_REGISTRATION_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Statuses. */
#define TR_OK 0               /* done */
#define TR_CANCELLED 1        /* the progress callback asked to stop */
#define TR_INVALID_ARGUMENT 2 /* a pointer, size, index or setting it cannot take */
#define TR_UNAVAILABLE 3      /* the backend cannot run here: no GPU, a frame too large for it, no threads */
#define TR_FAILED 4           /* the registration failed: a panic, or frames the GPU could not register */
#define TR_NO_COMMON_AREA 5   /* the frames moved too far apart to share any pixel: nothing to crop to */

/* TrSettings.transformation: the model fitted, fewest degrees of freedom first.
 * Not pystackreg's numbers (2, 3, 4, 6, 8): numbered from 0 like every enum here. */
#define TR_TRANSFORMATION_TRANSLATION 0
#define TR_TRANSFORMATION_RIGID_BODY 1 /* the default */
#define TR_TRANSFORMATION_SCALED_ROTATION 2
#define TR_TRANSFORMATION_AFFINE 3
#define TR_TRANSFORMATION_BILINEAR 4 /* not with TR_REFERENCE_PREVIOUS */

/* TrSettings.reference: what each frame is registered to. */
#define TR_REFERENCE_PREVIOUS 0 /* the frame before, composed into a chain */
#define TR_REFERENCE_FIRST 1    /* the first n_frames averaged; the default */
#define TR_REFERENCE_MEAN 2     /* the mean of the whole stack */

/* TrSettings.backend: where the arithmetic runs. */
#define TR_BACKEND_SINGLE_THREAD 0
#define TR_BACKEND_MULTI_THREAD 1 /* the default */
#define TR_BACKEND_GPU 2          /* needs the `gpu` feature */

/* TrSettings.border: what a moved frame shows where it no longer covers the frame. */
#define TR_BORDER_NATIVE 0   /* the default: TurboReg's own mirrored extension */
#define TR_BORDER_EDGE 1     /* the nearest edge pixel, repeated */
#define TR_BORDER_CONSTANT 2 /* zero */
#define TR_BORDER_CROP 3     /* the largest rectangle all frames cover, at any angle, turned upright: smaller frames */

/* The registration parameters. Always start from tr_settings_default(): the field list can grow. */
typedef struct TrSettings {
    size_t n_frames;          /* frames averaged for TR_REFERENCE_FIRST; 1 is the first alone */
    size_t moving_average;    /* temporal smoothing window before measuring; 1 for none */
    size_t max_pyramid_depth; /* cap on pyramid levels; 0 for as deep as the image allows */
    int transformation;       /* TR_TRANSFORMATION_* */
    int reference;            /* TR_REFERENCE_* */
    int backend;              /* TR_BACKEND_* */
    int border;               /* TR_BORDER_*, used by tr_registered_apply */
} TrSettings;

/* A registration's result. Opaque; free with tr_registered_free(). */
typedef struct TrRegistered TrRegistered;

/* Progress, 0.0 to 1.0, on the calling thread. Return non-zero to go on, 0 to stop. */
typedef int (*TrProgress)(float fraction, void *user);

/* Why the last call on this thread that failed did; valid until the next failure. */
const char *tr_last_error(void);

/* The library's version, "major.minor.patch". */
const char *tr_version(void);

/* Fill `out` with the defaults. */
int tr_settings_default(TrSettings *out);

/* TR_OK if `backend` can run frames of ly x lx here, else TR_UNAVAILABLE and why. */
int tr_backend_available(int backend, size_t ly, size_t lx);

/*
 * Register `n_frames` frames of `ly` x `lx`, one after another in `frames`.
 * `settings` may be NULL for the defaults, `progress` NULL for none; `user`
 * goes to `progress` untouched. The frames are copied. On TR_OK `*out` is the
 * result; otherwise it is NULL. GPU runs from several threads take turns.
 */
int tr_register(const float *frames, size_t n_frames, size_t ly, size_t lx,
                const TrSettings *settings, TrProgress progress, void *user,
                TrRegistered **out);

/* Free a result; NULL is ignored. */
void tr_registered_free(TrRegistered *r);

/* How many frames the result covers; 0 for NULL. */
size_t tr_registered_frames(const TrRegistered *r);

/*
 * Frame t's transformation, eight coefficients into c[8]:
 *   x' = c[0] + c[1] x + c[2] y + c[3] x y
 *   y' = c[4] + c[5] x + c[6] y + c[7] x y
 * mapping REFERENCE coordinates to MOVING ones: the direction a resampler
 * pulls in, used as it stands. c[3] and c[7] are 0 except for bilinear.
 */
int tr_registered_matrix(const TrRegistered *r, size_t t, double *c);

/* Whether the fit gave up on frame t, which then keeps the identity. */
int tr_registered_failed(const TrRegistered *r, size_t t, bool *failed);

/* Whether there is one reference to copy: not under TR_REFERENCE_PREVIOUS. */
bool tr_registered_has_reference(const TrRegistered *r);

/* Copy the reference into `out`; `len` must be ly * lx. */
int tr_registered_reference(const TrRegistered *r, float *out, size_t len);

/* Change the border tr_registered_apply uses, a TR_BORDER_*. */
int tr_registered_set_border(TrRegistered *r, int border);

/*
 * The size of the frames tr_registered_apply writes: ly x lx, or under
 * TR_BORDER_CROP the common area's (TR_NO_COMMON_AREA when there is none).
 * Either output may be NULL.
 */
int tr_registered_output_size(const TrRegistered *r, size_t *ly, size_t *lx);

/*
 * The largest rectangle every frame covers once moved, at any angle: what
 * TR_BORDER_CROP keeps. Where its first pixel sits in the frame (y, x;
 * fractional when turned), its height and width, and its angle in radians
 * (0 upright; positive turns its rows clockwise as an image is shown, rows
 * running down). Result pixel (row, col) sits at
 *   (y + col sin(angle) + row cos(angle), x + col cos(angle) - row sin(angle)).
 * Under any border; TR_NO_COMMON_AREA when the frames share no pixel. Any
 * output may be NULL.
 */
int tr_registered_common_area(const TrRegistered *r, double *y, double *x,
                              size_t *height, size_t *width, double *angle);

/*
 * Put frame t back where it belongs. `frame` holds ly * lx floats; `out` holds
 * the output size (tr_registered_output_size), which is ly * lx unless the
 * border is TR_BORDER_CROP. They may be the same buffer. Any channel's frame t
 * can be passed: that is how a second channel moves with the first. Under
 * TR_BORDER_CROP with no common area: TR_NO_COMMON_AREA, `out` untouched.
 */
int tr_registered_apply(const TrRegistered *r, const float *frame, size_t t, float *out);

#ifdef __cplusplus
}
#endif

#endif /* TURBOREG_REGISTRATION_H */
