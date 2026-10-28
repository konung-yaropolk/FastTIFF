/*
 * suite2p-registration: the C interface.
 *
 * Copyright (C) 2026 SciWare LLC. GPL-3.0-only, like the library: a program
 * that links it, statically or dynamically, and is distributed, is a
 * derivative work and goes out under the GPL too. See LICENSE.
 *
 * Build the library (from the repository, or an unpacked crate):
 *
 *   cargo rustc -p suite2p-registration --release --features capi \
 *       --crate-type cdylib,staticlib
 *
 * Add `--features capi,gpu` for the GPU backend. The header matches the
 * library it ships with, and no other: check s2p_version().
 *
 * Usage, in brief: fill an S2pSettings with s2p_settings_default(), change
 * what differs, call s2p_register() once on the channel to measure, then
 * s2p_registered_apply() on every frame of every channel, and
 * s2p_registered_free() at the end. Every call returns an S2P_* status;
 * s2p_last_error() says why one was not S2P_OK.
 *
 * Frames are float, row-major, `ly` rows of `lx` columns. A movie is its
 * frames one after another.
 */
#ifndef SUITE2P_REGISTRATION_H
#define SUITE2P_REGISTRATION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Statuses. */
#define S2P_OK 0               /* done */
#define S2P_CANCELLED 1        /* the progress callback asked to stop */
#define S2P_INVALID_ARGUMENT 2 /* a pointer, size, index or setting it cannot take */
#define S2P_UNAVAILABLE 3      /* the backend cannot run here (see s2p_backend_available) */
#define S2P_FAILED 4           /* the registration failed: a panic, or the GPU could not finish */

/* S2pSettings.backend: where the arithmetic runs. Same answer on each. */
#define S2P_BACKEND_SINGLE_THREAD 0
#define S2P_BACKEND_MULTI_THREAD 1 /* the default */
#define S2P_BACKEND_GPU 2          /* power-of-two frames up to 1024; needs the `gpu` feature */

/* S2pSettings.border: what a moved frame shows where it no longer covers the frame. */
#define S2P_BORDER_NATIVE 0   /* the default: suite2p's own (a rigid shift wraps; a warp repeats the edge) */
#define S2P_BORDER_EDGE 1     /* the nearest edge pixel, repeated */
#define S2P_BORDER_CONSTANT 2 /* zero */

/*
 * The registration parameters, under suite2p's own names and with its
 * defaults; see the crate's documentation of `Settings` for each. Always
 * start from s2p_settings_default(): the field list can grow.
 */
typedef struct S2pSettings {
    size_t nimg_init;            /* frames sampled to build the reference */
    size_t reference_iterations; /* passes refining the reference */
    size_t batch_size;           /* frames measured together */
    size_t block_size[2];        /* non-rigid block, rows then columns */
    size_t subpixel;             /* non-rigid sub-pixel precision, 1/subpixel */
    double maxregshift;          /* largest rigid shift, fraction of the smaller side */
    double maxregshift_nr;       /* largest per-block shift, pixels */
    double smooth_sigma;         /* spatial smoothing of the correlation, pixels */
    double smooth_sigma_time;    /* temporal smoothing within a batch, frames; 0 for none */
    double spatial_taper;        /* border fade before correlating, pixels */
    double snr_thresh;           /* non-rigid peak quality below which a block is smoothed */
    double th_badframes;         /* outlier threshold for bad-frame detection */
    int32_t bidiphase;           /* fixed bidirectional offset, columns; 0 to measure or skip */
    int backend;                 /* S2P_BACKEND_* */
    int border;                  /* S2P_BORDER_* */
    bool norm_frames;            /* clip frames to the reference's range before correlating */
    bool two_step_registration;  /* register twice, the second pass on the first's output */
    bool do_bidiphase;           /* measure the bidirectional offset when bidiphase is 0 */
    bool nonrigid;               /* measure a per-block deformation on top of the rigid shift */
} S2pSettings;

/* A registration's result. Opaque; free with s2p_registered_free(). */
typedef struct S2pRegistered S2pRegistered;

/* Progress, 0.0 to 1.0, on the calling thread. Return non-zero to go on, 0 to stop. */
typedef int (*S2pProgress)(float fraction, void *user);

/* Why the last call on this thread that failed did; valid until the next failure. */
const char *s2p_last_error(void);

/* The library's version, "major.minor.patch". */
const char *s2p_version(void);

/* Fill `out` with suite2p's defaults. */
int s2p_settings_default(S2pSettings *out);

/* S2P_OK if `backend` can run frames of ly x lx here, else S2P_UNAVAILABLE and why. */
int s2p_backend_available(int backend, size_t ly, size_t lx);

/*
 * Register `n_frames` frames of `ly` x `lx`, one after another in `frames`.
 * `settings` may be NULL for the defaults, `progress` NULL for none; `user`
 * goes to `progress` untouched. The frames are copied. On S2P_OK `*out` is
 * the result; otherwise it is NULL. GPU runs from several threads take turns.
 */
int s2p_register(const float *frames, size_t n_frames, size_t ly, size_t lx,
                 const S2pSettings *settings, S2pProgress progress, void *user,
                 S2pRegistered **out);

/* Free a result; NULL is ignored. */
void s2p_registered_free(S2pRegistered *r);

/* How many frames the result covers; 0 for NULL. */
size_t s2p_registered_frames(const S2pRegistered *r);

/*
 * Frame t's rigid shift and correlation peak; outputs may be NULL. (dy, dx) is
 * where the frame's content sits relative to the reference: the reference's
 * (y, x) is the frame's (y + dy, x + dx). Bringing the frame onto the
 * reference moves it by (-dy, -dx), as s2p_registered_apply does.
 */
int s2p_registered_shift(const S2pRegistered *r, size_t t, int32_t *dy, int32_t *dx, float *corr);

/* Whether frame t looks like an outlier. Reported, never dropped. */
int s2p_registered_is_bad(const S2pRegistered *r, size_t t, bool *bad);

/* The bidirectional offset corrected for, in columns; 0 for none. */
int32_t s2p_registered_bidiphase(const S2pRegistered *r);

/* Whether a non-rigid block field was measured (s2p_registered_apply uses it). */
bool s2p_registered_is_nonrigid(const S2pRegistered *r);

/* Copy the reference into `out`; `len` must be ly * lx. */
int s2p_registered_reference(const S2pRegistered *r, float *out, size_t len);

/* Change the border s2p_registered_apply uses, an S2P_BORDER_*. */
int s2p_registered_set_border(S2pRegistered *r, int border);

/*
 * Put frame t back where it belongs: the bidirectional correction, then the
 * shift or warp. `frame` and `out` hold ly * lx floats each and may be the
 * same buffer. Any channel's frame t can be passed: that is how a second
 * channel moves with the first.
 */
int s2p_registered_apply(const S2pRegistered *r, const float *frame, size_t t, float *out);

#ifdef __cplusplus
}
#endif

#endif /* SUITE2P_REGISTRATION_H */
