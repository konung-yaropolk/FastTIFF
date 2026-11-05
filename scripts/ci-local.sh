#!/usr/bin/env bash
# The CI checks, defined once. Each CI job calls this script with the names
# of the steps it runs, so the commands live here and nowhere else; run it
# locally to get the same checks, one after another, in this checkout.
#
#   scripts/ci-local.sh            every job CI runs on every push
#   scripts/ci-local.sh --quick    skip the slow ones (web bundle, oir package)
#   scripts/ci-local.sh --fuzz     also the fuzz jobs (needs nightly + cargo-fuzz)
#   scripts/ci-local.sh --list     print the step names and stop
#   scripts/ci-local.sh NAME...    only the steps named (see --list)
#
# Runs from Git Bash on Windows, and on Linux and macOS. Every step runs even
# when an earlier one fails; each one's output goes to target/ci-local/<step>.log
# and the end prints a table and exits non-zero if anything failed.
#
# Under CI (`CI` set, as GitHub sets it) each step's output is streamed in a
# folded group instead of only going to its log.
#
# What stays in the workflows is what is not a check: installing toolchains,
# caches, the Linux job's system packages. The `ci-coverage` step fails if a
# step here is not named by any workflow, so a check cannot be added here and
# silently run only on one machine. `cargo package` gets
# `--allow-dirty` locally so the script can be run on uncommitted work, and
# not in CI, where a file an earlier step dirtied should fail it. Steps marked
# `explicit` only run when named, since they need a particular host (the macOS
# Intel cross-check needs the Apple SDK).
#
# The example plugin's Cargo.lock is not part of the workspace and a build of
# it rewrites it; it is put back afterwards if it was unmodified before.

set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)
LOGS="$ROOT/target/ci-local"
mkdir -p "$LOGS"

QUICK=0
FUZZ=0
ONLY=()
LIST=0
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK=1 ;;
        --fuzz) FUZZ=1 ;;
        --list) LIST=1 ;;
        -h|--help) awk 'NR > 1 && !/^#/ { exit } NR > 1' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) echo "unknown option: $arg" >&2; exit 2 ;;
        *) ONLY+=("$arg") ;;
    esac
done

# A misspelled name would otherwise just not run, which a CI job naming a
# second, real step would never notice.
for n in ${ONLY[@]+"${ONLY[@]}"}; do
    grep -qE "^step $n( |\$)" "$0" || { echo "no step named '$n' (see --list)" >&2; exit 2; }
done

NAMES=()
RESULTS=()
TIMES=()
FAILED=0

wanted() {
    [ ${#ONLY[@]} -eq 0 ] && return 0
    local n
    for n in "${ONLY[@]}"; do [ "$n" = "$1" ] && return 0; done
    return 1
}

# step NAME [slow|fuzz|explicit] -- COMMAND...
step() {
    local name=$1; shift
    local kind=normal
    if [ "$1" = slow ] || [ "$1" = fuzz ] || [ "$1" = explicit ]; then kind=$1; shift; fi
    [ "$1" = -- ] && shift
    if [ $LIST -eq 1 ]; then printf '  %-22s %s\n' "$name" "$*"; return; fi
    wanted "$name" || return
    if [ ${#ONLY[@]} -eq 0 ]; then
        [ "$kind" = slow ] && [ $QUICK -eq 1 ] && return
        [ "$kind" = fuzz ] && [ $FUZZ -eq 0 ] && return
        [ "$kind" = explicit ] && return
    fi
    local log="$LOGS/$name.log"
    local start=$SECONDS rc
    if [ -n "${CI:-}" ]; then
        # In CI, where nobody reads the log files: the output in full, folded
        # under the step's name, and still kept in the log.
        echo "::group::$name"
        (cd "$ROOT" && "$@") 2>&1 | tee "$log"
        rc=${PIPESTATUS[0]}
        echo "::endgroup::"
        printf '%-22s ... ' "$name"
    else
        printf '%-22s ... ' "$name"
        (cd "$ROOT" && "$@") >"$log" 2>&1
        rc=$?
    fi
    if [ "$rc" -eq 0 ]; then
        RESULTS+=(PASS)
        echo "pass ($((SECONDS - start))s)"
    else
        RESULTS+=(FAIL)
        FAILED=1
        echo "FAIL ($((SECONDS - start))s) — see $log"
        tail -n 15 "$log" | sed 's/^/    /'
    fi
    NAMES+=("$name")
    TIMES+=($((SECONDS - start)))
}

# A path dependency that also carries `version` is a requirement against the
# crate next door, so bumping that crate breaks this one, and the resolver
# reports it as a missing candidate. Only a crate that is published needs the
# field: crates.io wants every dependency of a published crate published too,
# with a version. The published ones are those publish-lib.yml publishes, so a
# pin anywhere else fails here.
no_pinned_paths() {
    local published file name bad=0
    published=$(sed -n 's/.*cargo publish -p \([a-z0-9_-]*\).*/\1/p' .github/workflows/publish-lib.yml)
    for file in $(grep -rlE '^[a-zA-Z0-9_-]+ *= *\{.*(path.*version|version.*path) *=' \
                      --include=Cargo.toml --exclude-dir=target --exclude-dir=.git .); do
        name=$(sed -n 's/^name = "\(.*\)"/\1/p' "$file" | head -n 1)
        if ! printf '%s\n' $published | grep -qx "$name"; then
            grep -nE '^[a-zA-Z0-9_-]+ *= *\{.*(path.*version|version.*path) *=' "$file" | sed "s|^|$file:|"
            bad=1
        fi
    done
    if [ $bad -ne 0 ]; then
        echo "A path dependency above also pins a version, in a crate that is not published"
        echo "(publish-lib.yml); drop the 'version' field."
        return 1
    fi
}

in_dir() { local d=$1; shift; (cd "$d" && "$@"); }

# The libraries, all features on, built with the oldest Rust they claim. The
# number and the crate list both come from the manifests: `rust-version` in the
# root `[workspace.package]`, and every member that inherits it.
msrv() {
    local v crates=()
    v=$(sed -n 's/^rust-version = "\(.*\)"/\1/p' Cargo.toml | head -n 1)
    [ -n "$v" ] || { echo "no rust-version in the root Cargo.toml"; return 1; }
    for m in $(grep -l '^rust-version.workspace = true' */Cargo.toml); do
        crates+=(-p "$(sed -n 's/^name = "\(.*\)"/\1/p' "$m" | head -n 1)")
    done
    echo "Rust $v: ${crates[*]}"
    rustup toolchain install "$v" --profile minimal --no-self-update &&
    cargo "+$v" check "${crates[@]}" --all-features
}

# Every step is named by some workflow. The workflows call this script by step
# name, so a step no workflow names would run only on whoever's machine runs
# the whole list.
every_step_in_ci() {
    local n missing=0
    for n in $(bash "$ROOT/scripts/ci-local.sh" --list | sed -n 's/^  \([a-z0-9-]*\) .*/\1/p'); do
        grep -qE "^[[:space:]]*run: bash scripts/ci-local\.sh( [a-z0-9-]+)* $n( |\$)" .github/workflows/*.yml .github/actions/*/action.yml ||
            { echo "no workflow runs the step '$n'"; missing=1; }
    done
    return $missing
}

# Comments and docs that name a file this repository does not have; see the
# script for what counts. `python3` where there is one, which on Windows can
# be a store alias that only prints an advert, so `python` there.
doc_paths() {
    local py=python3
    case "$(uname -s)" in MINGW* | MSYS* | CYGWIN*) py=python ;; esac
    "$py" scripts/check-doc-paths.py
}

# `cargo package`, as a release would: in CI on the clean checkout, locally
# with `--allow-dirty` so it can be run on uncommitted work. Several `-p` at
# once package together, so a crate whose dependency is not on crates.io yet
# is verified against the one beside it.
package() {
    if [ -n "${CI:-}" ]; then
        cargo package "$@"
    else
        cargo package --allow-dirty "$@"
    fi
}

# The example plugin is outside the workspace, so `cargo fmt --all` and the
# workspace clippy runs never see it.
example_lint() {
    cargo fmt --manifest-path plugins/example/Cargo.toml --check &&
    cargo clippy --manifest-path plugins/example/Cargo.toml --all-targets -- -D warnings
}

# The three clippy runs. A lint only sees the code its features compile: the
# default build leaves out the glow renderer, the C APIs, the GPU registration
# paths and the example importer. Every crate but the app goes in the second
# run with all its features; the app's two renderers exclude each other, so it
# gets the other one in a third.
clippy_all() {
    cargo clippy --workspace --all-targets -- -D warnings &&
    cargo clippy --workspace --exclude FastTIFF --all-features --all-targets -- -D warnings &&
    cargo clippy -p FastTIFF --no-default-features --features renderer-glow,builtin-plugins,plugin-netpbm --all-targets -- -D warnings
}

fuzz() {
    local target=$1 seconds=$2
    mkdir -p "fast-tiff-lib/fuzz/corpus/$target"
    in_dir fast-tiff-lib cargo +nightly fuzz run "$target" "fuzz/corpus/$target" fuzz/seeds -- \
        -max_total_time="$seconds" -rss_limit_mb=2048 -malloc_limit_mb=512
}

EXAMPLE_LOCK=plugins/example/Cargo.lock
lock_was_clean=0
git diff --quiet -- "$EXAMPLE_LOCK" 2>/dev/null && lock_was_clean=1

[ $LIST -eq 1 ] && echo "Steps (slow ones are skipped by --quick, fuzz ones need --fuzz, explicit ones need naming):"

step ci-coverage         -- every_step_in_ci
step pinned-paths        -- no_pinned_paths
step doc-paths           -- doc_paths
step example-build       -- cargo build --manifest-path plugins/example/Cargo.toml
step example-test        -- cargo test --manifest-path plugins/example/Cargo.toml
step example-lint        -- example_lint
step workspace-tests     -- cargo test --workspace
step capi-tests          -- cargo test -p fast-tiff-lib -p oir-reader -p suite2p-registration -p turboreg-registration --features fast-tiff-lib/capi,oir-reader/capi,suite2p-registration/capi,suite2p-registration/gpu,turboreg-registration/capi,turboreg-registration/gpu
step glow-build          -- cargo build -p FastTIFF --no-default-features --features renderer-glow,builtin-plugins
step glow-tests          -- cargo test --workspace --no-default-features --features FastTIFF/renderer-glow,FastTIFF/builtin-plugins
step lib-no-default      -- cargo test -p fast-tiff-lib --no-default-features
step msrv                -- msrv
step wasm-target         -- rustup target add wasm32-unknown-unknown
step lib-wasm            -- cargo check -p fast-tiff-lib --target wasm32-unknown-unknown --no-default-features
step viewer-wasm         -- cargo check -p fast-tiff-viewer --target wasm32-unknown-unknown --no-default-features --features backend-wgpu,builtin-plugins
step oir-no-default      -- env RUSTDOCFLAGS="-D warnings" cargo test -p oir-reader --no-default-features
step oir-wasm            -- cargo check -p oir-reader --target wasm32-unknown-unknown --all-targets
step oir-wasm-no-default -- cargo check -p oir-reader --target wasm32-unknown-unknown --no-default-features
step oir-doc             -- env RUSTDOCFLAGS="-D warnings" cargo doc -p oir-reader --no-deps
step published-package slow -- package -p c-api-support -p image-registration-support -p fast-tiff-lib -p oir-reader -p suite2p-registration -p turboreg-registration
step web-check           -- cargo clippy -p fasttiff-web-egui --target wasm32-unknown-unknown -- -D warnings
step viewer-no-threads   -- cargo test -p fast-tiff-viewer --no-default-features --features backend-wgpu,builtin-plugins,plugin-netpbm
step viewer-no-backend   -- cargo check -p fast-tiff-viewer --no-default-features
step macos-intel explicit -- env MACOSX_DEPLOYMENT_TARGET=11.0 cargo check --release --target x86_64-apple-darwin -p FastTIFF --no-default-features --features renderer-wgpu,builtin-plugins
step web-bundle slow     -- in_dir FastTIFF-web wasm-pack build --target web --out-dir pkg --release
step fmt                 -- cargo fmt --all --check
step clippy              -- clippy_all
step fuzz-open fuzz      -- fuzz tiff_open 120
step fuzz-decode fuzz    -- fuzz tiff_decode 180

[ $LIST -eq 1 ] && exit 0

if [ $lock_was_clean -eq 1 ]; then
    git checkout -- "$EXAMPLE_LOCK" 2>/dev/null
fi

echo
echo "Summary (logs in $LOGS):"
total=0
for i in "${!NAMES[@]}"; do
    printf '  %-4s %-22s %5ss\n' "${RESULTS[$i]}" "${NAMES[$i]}" "${TIMES[$i]}"
    total=$((total + TIMES[i]))
done
if [ ${#NAMES[@]} -eq 0 ]; then
    echo "  no step ran (see --list)"
    exit 2
fi
echo "  total ${total}s"
if [ $FAILED -eq 0 ]; then echo "All passed."; else echo "Something failed."; fi
exit $FAILED
