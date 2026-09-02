#!/usr/bin/env bash
#
# Bring a bare Ubuntu GPU box to the point where solve.jl runs, and measure
# the two rates `plan` needs to estimate a wall clock on it.
#
# Runs on the RENTED MACHINE, not on the desktop. `peregrine_remote.py
# provision` copies this up and runs it; you can also just scp it and run it
# by hand. It is idempotent -- running it again on a provisioned box checks
# everything and re-measures, which is the cheap way to confirm a box is
# still good before starting an eight-hour solve on it.
#
#   ./provision.sh [--driver] [--quick]
#
#   --driver  install the NVIDIA driver too. Only needed on a plain OS image.
#             DigitalOcean's "AI/ML ready" images already have one, and that
#             is the image to pick -- installing a driver on a just-released
#             Ubuntu is the flakiest part of this whole exercise, and it
#             needs a reboot afterwards.
#   --quick   skip the solver self-test. Do not use this on a box you have
#             not tested before: the self-test is what tells you the GPU
#             produces the same answers this project was developed against,
#             and it costs a couple of minutes against a rental you are
#             paying for by the second either way.
#
# Everything it learns lands in ~/peregrine/cloud-env.json, which the desktop
# side reads back.

set -euo pipefail

JULIA_CHANNEL="${JULIA_CHANNEL:-1.12}"
ROOT="${PEREGRINE_ROOT:-$HOME/peregrine}"
WORK="${PEREGRINE_WORK:-$HOME/work}"
ENVFILE="$ROOT/cloud-env.json"

WANT_DRIVER=0
QUICK=0
for a in "$@"; do
    case "$a" in
        --driver) WANT_DRIVER=1 ;;
        --quick)  QUICK=1 ;;
        *) echo "unknown flag: $a" >&2; exit 2 ;;
    esac
done

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }
warn() { printf '\033[1;33m   %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m   %s\033[0m\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------
# 1. The GPU
# --------------------------------------------------------------------------

say "GPU"
if [ "$WANT_DRIVER" = 1 ]; then
    note "installing the NVIDIA driver (this needs a reboot afterwards)"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq ubuntu-drivers-common
    ubuntu-drivers install --gpgpu || die "driver install failed -- rent the
   AI/ML-ready image instead of a plain OS image and skip this step"
    warn "reboot now, then run this script again WITHOUT --driver"
    exit 0
fi

command -v nvidia-smi >/dev/null 2>&1 || die "no nvidia-smi. Either this is not
   a GPU box, or the image has no driver -- re-run with --driver, or destroy
   this droplet and create one from an AI/ML-ready image."

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
note "$GPU_COUNT x $GPU_NAME, $GPU_MEM_MIB MiB each"
if [ "$GPU_COUNT" -gt 1 ]; then
    warn "the solver uses ONE GPU. You are paying for $GPU_COUNT."
fi

# --------------------------------------------------------------------------
# 2. Packages
# --------------------------------------------------------------------------

say "packages"
export DEBIAN_FRONTEND=noninteractive
NEED=""
for p in tmux curl tar; do
    command -v "$p" >/dev/null 2>&1 || NEED="$NEED $p"
done
if [ -n "$NEED" ]; then
    note "installing:$NEED"
    apt-get update -qq
    # shellcheck disable=SC2086
    apt-get install -y -qq $NEED
else
    note "tmux, curl and tar already present"
fi

# --------------------------------------------------------------------------
# 3. Julia
#
# Installed under $HOME by juliaup, then symlinked into /usr/local/bin. That
# symlink is load-bearing: a non-interactive `ssh box julia ...` returns out
# of Ubuntu's ~/.bashrc before it reaches any PATH line juliaup adds, so
# without it the desktop side can install Julia successfully and then be
# unable to find it.
# --------------------------------------------------------------------------

say "Julia"
JULIA=""
for cand in /usr/local/bin/julia "$HOME/.juliaup/bin/julia" "$(command -v julia 2>/dev/null || true)"; do
    [ -n "$cand" ] && [ -x "$cand" ] && { JULIA="$cand"; break; }
done

if [ -z "$JULIA" ]; then
    note "installing juliaup, channel $JULIA_CHANNEL"
    curl -fsSL https://install.julialang.org \
        | sh -s -- --yes --default-channel "$JULIA_CHANNEL" >/dev/null
    JULIA="$HOME/.juliaup/bin/julia"
    [ -x "$JULIA" ] || die "juliaup ran but $JULIA is not executable"
fi
if [ ! -e /usr/local/bin/julia ] && [ "$JULIA" != /usr/local/bin/julia ]; then
    ln -sf "$JULIA" /usr/local/bin/julia 2>/dev/null && JULIA=/usr/local/bin/julia || true
fi
note "$("$JULIA" --version) at $JULIA"

# --------------------------------------------------------------------------
# 4. The project
#
# `instantiate` against the checked-in Manifest.toml, so the box runs the
# same package versions the desktop does rather than whatever resolves today.
# CUDA.jl then downloads its own CUDA toolkit as an artifact -- a couple of
# GB, and the slowest step here. Only the DRIVER has to come from the image.
# --------------------------------------------------------------------------

say "solver project"
[ -f "$ROOT/solver/Project.toml" ] || die "no solver project at $ROOT -- push
   the repo first (peregrine_remote.py does this for you)"

cd "$ROOT"
note "instantiating (CUDA.jl pulls ~2-3 GB of toolkit artifacts the first time)"
"$JULIA" --project=solver -e '
    using Pkg
    Pkg.instantiate()
    Pkg.precompile()
' || die "instantiate failed"

note "checking CUDA is functional"
"$JULIA" --project=solver -e '
    using CUDA
    CUDA.functional() || error("CUDA.jl is not functional on this box")
    dev = CUDA.device()
    free, total = CUDA.memory_info()
    println("   device: ", CUDA.name(dev))
    println("   vram:   ", round(total / 2^30, digits=1), " GB total, ",
            round(free / 2^30, digits=1), " GB free")
    println("   driver: ", CUDA.driver_version(), "  runtime: ", CUDA.runtime_version())
' || die "CUDA check failed"

# --------------------------------------------------------------------------
# 5. The self-test
# --------------------------------------------------------------------------

if [ "$QUICK" = 1 ]; then
    say "self-test"
    warn "skipped (--quick). You are trusting an untested box with a paid run."
    SELFTEST="skipped"
else
    say "self-test"
    note "recovering known answers from synthetic data on this GPU"
    if "$JULIA" --project=solver --threads=auto solver/solve.jl --self-test; then
        SELFTEST="passed"
    else
        SELFTEST="FAILED"
        warn "the self-test failed on this box. Do not spend a long run on it."
    fi
fi

# --------------------------------------------------------------------------
# 6. Rates
#
# `plan` estimates a wall clock from a cell-update rate and a scratch-disk
# rate, and its defaults are measurements of the desktop's 8 GB card and its
# disk. Both are wrong here, and the disk one is easy to measure now. The
# cell rate needs a real solve, so the desktop side derives it from the first
# run's own progress stream instead of guessing.
# --------------------------------------------------------------------------

say "disk"
mkdir -p "$WORK/scratch" "$WORK/inputs" "$WORK/runs"
SCRATCH_FS=$(df -h --output=target,avail "$WORK/scratch" | tail -1)
note "scratch on $SCRATCH_FS available"

# 4 GiB, direct, so the page cache does not answer for the disk. The
# out-of-core store is read and written in big sequential runs per x plane,
# which is what this imitates.
DD_OUT=$(dd if=/dev/zero of="$WORK/scratch/.ratetest" bs=1M count=4096 \
            oflag=direct conv=fsync 2>&1 | tail -1)
rm -f "$WORK/scratch/.ratetest"
note "$DD_OUT"
DISK_RATE=$(printf '%s' "$DD_OUT" | grep -oE '[0-9.]+ [MG]B/s' | tail -1 | \
    awk '{ if ($2 == "GB/s") printf "%.0f", $1 * 1e9; else printf "%.0f", $1 * 1e6 }')
[ -n "$DISK_RATE" ] || DISK_RATE=500000000
note "disk_rate = $DISK_RATE B/s"

VRAM_BYTES=$(( GPU_MEM_MIB * 1024 * 1024 ))

cat > "$ENVFILE" <<JSON
{
  "gpu_name": "$GPU_NAME",
  "gpu_count": $GPU_COUNT,
  "vram_bytes": $VRAM_BYTES,
  "julia": "$JULIA",
  "root": "$ROOT",
  "work": "$WORK",
  "scratch_dir": "$WORK/scratch",
  "inputs_dir": "$WORK/inputs",
  "runs_dir": "$WORK/runs",
  "disk_rate": $DISK_RATE,
  "self_test": "$SELFTEST",
  "provisioned_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON

say "ready"
note "wrote $ENVFILE"
[ "$SELFTEST" = "FAILED" ] && exit 1
exit 0
