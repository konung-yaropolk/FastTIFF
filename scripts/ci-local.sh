#!/usr/bin/env bash
# Run the CI checks locally, before a merge: the same commands as
# .github/workflows/ci.yml, one after another, in this checkout.
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
# One thing differs from CI, on purpose: `cargo package -p oir-reader` gets
# `--allow-dirty`, so the script can be run on uncommitted work. CI packages a
# clean checkout. Keep the steps below in step with ci.yml: CI is the gate, and
# a check that passes here and fails there is this list falling behind.
# Left out: the macOS-only Intel cross-check (needs the Apple SDK), and the
# Linux job's system packages (the glow tests run on any OS here).
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
        -h|--help) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) echo "unknown option: $arg" >&2; exit 2 ;;
        *) ONLY+=("$arg") ;;
    esac
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

# step NAME [slow|fuzz] -- COMMAND...
step() {
    local name=$1; shift
    local kind=normal
    if [ "$1" = slow ] || [ "$1" = fuzz ]; then kind=$1; shift; fi
    [ "$1" = -- ] && shift
    if [ $LIST -eq 1 ]; then printf '  %-22s %s\n' "$name" "$*"; return; fi
    wanted "$name" || return
    if [ ${#ONLY[@]} -eq 0 ]; then
        [ "$kind" = slow ] && [ $QUICK -eq 1 ] && return
        [ "$kind" = fuzz ] && [ $FUZZ -eq 0 ] && return
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

no_pinned_paths() {
    local pinned
    pinned=$(grep -rnE '^[a-zA-Z0-9_-]+ *= *\{.*(path.*version|version.*path) *=' --include=Cargo.toml --exclude-dir=target --exclude-dir=.git . || true)
    if [ -n "$pinned" ]; then
        echo "$pinned"
        echo "A path dependency above also pins a version; drop the 'version' field"
        echo "unless the crate holding it is published."
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

# The same three runs as ci.yml's clippy step, which says why there are three.
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

[ $LIST -eq 1 ] && echo "Steps (slow ones are skipped by --quick, fuzz ones need --fuzz):"

step pinned-paths        -- no_pinned_paths
step example-build       -- cargo build --manifest-path plugins/example/Cargo.toml
step example-test        -- cargo test --manifest-path plugins/example/Cargo.toml
step workspace-tests     -- cargo test --workspace
step capi-tests          -- cargo test -p fast-tiff-lib -p oir-reader -p suite2p-registration -p turboreg-registration --features fast-tiff-lib/capi,oir-reader/capi,suite2p-registration/capi,suite2p-registration/gpu,turboreg-registration/capi,turboreg-registration/gpu
step glow-build          -- cargo build -p FastTIFF --no-default-features --features renderer-glow,builtin-plugins
step glow-tests          -- cargo test --workspace --no-default-features --features FastTIFF/renderer-glow,FastTIFF/builtin-plugins
step lib-no-default      -- cargo test -p fast-tiff-lib --no-default-features
step msrv                -- msrv
step wasm-target         -- rustup target add wasm32-unknown-unknown
step lib-wasm            -- cargo check -p fast-tiff-lib --target wasm32-unknown-unknown --no-default-features
step viewer-wasm         -- cargo check -p fast-tiff-viewer --target wasm32-unknown-unknown --no-default-features --features backend-wgpu,builtin-plugins
step oir-no-default      -- cargo test -p oir-reader --no-default-features
step oir-wasm            -- cargo check -p oir-reader --target wasm32-unknown-unknown --all-targets
step oir-wasm-no-default -- cargo check -p oir-reader --target wasm32-unknown-unknown --no-default-features
step oir-doc             -- env RUSTDOCFLAGS="-D warnings" cargo doc -p oir-reader --no-deps
step oir-package slow    -- cargo package -p oir-reader --allow-dirty
step web-check           -- cargo clippy -p fasttiff-web-egui --target wasm32-unknown-unknown -- -D warnings
step viewer-no-threads   -- cargo test -p fast-tiff-viewer --no-default-features --features backend-wgpu,builtin-plugins,plugin-netpbm
step viewer-no-backend   -- cargo check -p fast-tiff-viewer --no-default-features
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
