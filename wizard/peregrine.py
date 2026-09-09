#!/usr/bin/env python3
"""
Peregrine desktop wizard.

A terminal walkthrough of the whole offline pipeline:

    calibration CSV -> drivetrain regression -> field + targets
                    -> GPU value-table solve -> SD card

Stdlib only. Run with:

    py -3.12 wizard/peregrine.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from datetime import datetime, timezone

import sdcard

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FITTER = os.path.join(REPO, "calibration", "fit_drivetrain.py")
DIAGNOSER = os.path.join(REPO, "calibration", "diagnose_fit.py")
PODFINDER = os.path.join(REPO, "calibration", "find_pod_offsets.py")
SOLVER = os.path.join(REPO, "solver", "solve.jl")
SOLVER_PROJ = os.path.join(REPO, "solver")
HONING_BENCH = os.path.join(REPO, "solver", "bench", "honing.jl")
EXAMPLES = os.path.join(REPO, "solver", "examples")
SETTINGS = os.path.join(os.environ.get("LOCALAPPDATA", HERE), "Peregrine",
                        "wizard.json")

BAR_W = 44
CARD_MARKER = "card_written.json"


# --------------------------------------------------------------------------
# Terminal helpers
# --------------------------------------------------------------------------

def supports_ansi() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)
            return True
        except Exception:
            return False
    return sys.stdout.isatty()


ANSI = supports_ansi()


def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if ANSI else text


def rule(title: str = "") -> None:
    if title:
        print("\n" + c(f"== {title} " + "=" * max(0, 66 - len(title)), "1;36"))
    else:
        print(c("=" * 70, "36"))


def ask(prompt: str, default: str | None = None) -> str:
    """
    Prompt for a line.

    EOF propagates rather than being swallowed as the default: closed stdin
    means there is nobody left to answer, and returning a default forever
    would spin the menu loop.
    """
    suffix = f" [{default}]" if default is not None else ""
    # Strip a stray BOM and surrounding quotes: paths get pasted in from
    # Explorer and from editors that helpfully add both.
    v = input(c(f"  {prompt}{suffix}: ", "1")).lstrip("﻿").strip().strip('"')
    return v or (default or "")


def ask_int(prompt, default, lo=None, hi=None) -> int:
    while True:
        raw = ask(prompt, str(default))
        try:
            v = int(raw)
        except ValueError:
            print(c("    not a whole number", "31")); continue
        if lo is not None and v < lo:
            print(c(f"    must be >= {lo}", "31")); continue
        if hi is not None and v > hi:
            print(c(f"    must be <= {hi}", "31")); continue
        return v


def ask_float(prompt, default) -> float:
    while True:
        raw = ask(prompt, str(default))
        try:
            return float(raw)
        except ValueError:
            print(c("    not a number", "31"))


def confirm(prompt: str, default_yes: bool = False) -> bool:
    suffix = "(Y/n)" if default_yes else "(y/N)"
    got = ask(f"{prompt} {suffix}", "y" if default_yes else "n").lower()
    return got.startswith("y")


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


def hms(s: float) -> str:
    s = int(max(0, s))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s//60}m {s%60:02d}s"
    return f"{s//3600}h {(s%3600)//60:02d}m"


class Bar:
    """Single-line progress bar that redraws in place."""

    def __init__(self, label: str):
        self.label = label
        self.last = 0.0
        # Kept so a caller can redraw with a new note without knowing where
        # the bar had got to -- a tiled solve reports tile progress between
        # rounds, and those must not drag the bar back to zero.
        self.frac = 0.0

    def update(self, frac: float, note: str = "") -> None:
        frac = min(1.0, max(0.0, frac))
        now = time.time()
        if frac < 1.0 and now - self.last < 0.08:
            return
        self.last = now
        self.frac = frac
        filled = int(BAR_W * frac)
        bar = "#" * filled + "-" * (BAR_W - filled)
        line = f"  {self.label:<12} [{bar}] {frac*100:5.1f}%  {note}"
        sys.stdout.write("\r" + line[:150].ljust(150))
        sys.stdout.flush()

    def done(self, note: str = "") -> None:
        self.update(1.0, note)
        sys.stdout.write("\n")
        sys.stdout.flush()


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------

def load_settings() -> dict:
    try:
        with open(SETTINGS, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def save_settings(s: dict) -> None:
    os.makedirs(os.path.dirname(SETTINGS), exist_ok=True)
    with open(SETTINGS, "w", encoding="utf-8") as fh:
        json.dump(s, fh, indent=2)


class Workspace:
    """
    Everything the wizard produces, under one user-chosen directory.

    The location is the user's call -- the tables are enormous and often will
    not fit on the system drive -- so only a pointer to it is kept in
    LOCALAPPDATA.
    """

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.calib = os.path.join(self.root, "calibration")
        self.field = os.path.join(self.root, "field")
        self.runs = os.path.join(self.root, "runs")
        # Saved plans, each a self-contained directory that a later
        # `peregrine_remote.py run` can solve with no wizard session and no
        # droplet in existence at save time. See `save_job`.
        self.jobs = os.path.join(self.root, "jobs")
        for d in (self.calib, self.field, self.runs, self.jobs):
            os.makedirs(d, exist_ok=True)

    @property
    def regression(self) -> str | None:
        p = os.path.join(self.calib, "drivetrain_fit.toml")
        return p if os.path.exists(p) else None

    @property
    def field_file(self) -> str | None:
        p = os.path.join(self.field, "field.json")
        return p if os.path.exists(p) else None

    @property
    def targets_file(self) -> str | None:
        p = os.path.join(self.field, "targets.json")
        return p if os.path.exists(p) else None

    def latest_run(self) -> str | None:
        if not os.path.isdir(self.runs):
            return None
        rs = [os.path.join(self.runs, d) for d in sorted(os.listdir(self.runs))]
        rs = [r for r in rs if os.path.exists(os.path.join(r, "MANIFEST.JSON"))]
        return rs[-1] if rs else None

    # The "card written" marker lives inside the run directory, not in global
    # settings, so solving again correctly clears step 4 -- the new tables
    # have not been written anywhere yet.
    def card_record(self, run: str | None = None) -> dict | None:
        run = run or self.latest_run()
        if not run:
            return None
        try:
            with open(os.path.join(run, CARD_MARKER), encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None

    def mark_card_written(self, run: str, drive: str, files: int,
                          nbytes: int) -> None:
        rec = {
            "drive": drive.upper(),
            "written_utc": datetime.now().astimezone().isoformat(timespec="seconds"),
            "files": files,
            "bytes": nbytes,
            "verified": True,
        }
        with open(os.path.join(run, CARD_MARKER), "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2)

    def free(self) -> int:
        return shutil.disk_usage(self.root).free


# --------------------------------------------------------------------------
# Step 0 -- pod offsets (optional, but do it first)
# --------------------------------------------------------------------------

def step_pods(ws: Workspace) -> None:
    """
    Find the odometry pod offsets from a rotate-in-place log.

    Optional, but it belongs before everything else: a wrong pod offset makes
    the robot report a phantom sideways velocity whenever it turns, and the
    drivetrain fit cannot tell that apart from real dynamics. It quietly
    absorbs it and gives you a model that is wrong wherever the robot rotates.
    """
    rule("0. Pod offsets from a rotation-only run  (optional, do it first)")
    print("  Spin the robot in place -- no translation, a few revolutions each")
    print("  way -- and log it. If the offsets are right the reported position")
    print("  stays put; if they are wrong it sweeps a circle, and the radius of")
    print("  that circle is the error.")
    print()
    print(c("  Do this before step 1. A wrong offset contaminates every later", "33"))
    print(c("  run, and the drivetrain fit will absorb it silently.", "33"))
    print()

    logs = find_logs(ws)
    if not logs:
        print(c("  No calibration_log_*.csv found on a card or in the workspace.",
                "33"))
        return
    for i, (src, p_) in enumerate(logs, 1):
        print(f"   {i:2d}. [{src}] {os.path.basename(p_)}")
    print()
    i = ask_int("Which log (a rotation-only one)", 1, 1, len(logs))
    src, path = logs[i - 1]
    local = os.path.join(ws.calib, os.path.basename(path))
    if os.path.abspath(path) != os.path.abspath(local):
        shutil.copy2(path, local)
        print(f"  imported to {c(local, '36')}")

    cur = ask("Offsets currently set on the robot, 'X Y' in CM "
              "(blank if unknown)", "")
    cmd = [sys.executable, PODFINDER, local,
           "-o", os.path.join(ws.calib, "pod_offsets.json")]
    if cur:
        parts = cur.replace(",", " ").split()
        if len(parts) == 2:
            cmd += ["--current-offsets", parts[0], parts[1]]
        else:
            print(c("  could not read that as two numbers; continuing without",
                    "33"))
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        if r.stderr.strip():
            print(c(r.stderr.strip()[:800], "31"))
        return
    rec = os.path.join(ws.calib, "pod_offsets.json")
    try:
        with open(rec, encoding="utf-8") as fh:
            got = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return
    if got.get("centred"):
        print(c("  Nothing to change. Go on to step 1.", "32"))
    else:
        print(c("  Apply the change above, spin again, and re-run this step "
                "to confirm.", "33"))


# --------------------------------------------------------------------------
# Step 1 -- calibration
# --------------------------------------------------------------------------

def find_logs(ws: Workspace) -> list[tuple[str, str]]:
    """Calibration CSVs on any mounted card, plus any already imported."""
    found: list[tuple[str, str]] = []
    for v in sdcard.list_volumes():
        letter = (v.get("Letter") or "").upper()
        if not letter:
            continue
        d = os.path.join(f"{letter}:\\", "Android", "data",
                         "com.qualcomm.ftcrobotcontroller", "files", "logs")
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.startswith("calibration_log_") and f.endswith(".csv"):
                    found.append((f"card {letter}:", os.path.join(d, f)))
    for f in sorted(os.listdir(ws.calib)):
        if f.endswith(".csv"):
            found.append(("workspace", os.path.join(ws.calib, f)))
    return found


def step_calibration(ws: Workspace) -> None:
    rule("1. Calibration -> drivetrain regression")
    logs = find_logs(ws)
    if not logs:
        print(c("  No calibration_log_*.csv found on any card or in the "
                "workspace.", "33"))
        print("  Insert the SD card, or copy a CSV into "
              + c(ws.calib, "36"))
        p = ask("Path to a CSV (blank to go back)", "")
        if not p or not os.path.isfile(p):
            return
        logs = [("manual", p)]

    print()
    for i, (src, p) in enumerate(logs, 1):
        sz = os.path.getsize(p)
        print(f"   {i:2d}. [{src}] {os.path.basename(p):<38} {human(sz):>10}")
    print()
    i = ask_int("Which log", 1, 1, len(logs))
    src, path = logs[i - 1]

    local = os.path.join(ws.calib, os.path.basename(path))
    if os.path.abspath(path) != os.path.abspath(local):
        shutil.copy2(path, local)
        print(f"  imported to {c(local, '36')}")

    print()
    # The defaults here mirror the fitter's own, so pressing Enter through
    # this block gives the recommended model.
    extra: list[str] = []
    print()
    print("  Model terms. The defaults are the recommended ones -- press Enter")
    print("  through these unless you have a reason not to.")
    if not confirm("  Coulomb friction + quadratic drag (recommended)",
                   default_yes=True):
        extra += ["--no-coulomb", "--no-drag"]
    if confirm("  omega^2 centripetal term (only if the pods are NOT centred)"):
        extra.append("--omega-sq")
    if not confirm("  Constant offset term (a diagnostic; should come out ~0)",
                   default_yes=True):
        extra.append("--no-intercept")

    cmd = [sys.executable, FITTER, local, "-o", ws.calib,
           "--no-sensitivity"] + extra
    print()
    print(c("  running the regression...", "2"))
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        print(c("  regression failed:", "31"))
        print(r.stderr.strip()[:2000])
        return
    print(c(f"  wrote {os.path.join(ws.calib, 'drivetrain_fit.toml')}", "32"))


# --------------------------------------------------------------------------
# Optional helper -- regression diagnostics
# --------------------------------------------------------------------------

def step_diagnose(ws: Workspace) -> None:
    """
    Optional: plot how well the regression fits and what model form to use.

    Deliberately not one of the numbered steps -- nothing downstream needs it,
    and a run is perfectly valid without ever opening it.
    """
    rule("Diagnose the regression fit  (optional)")
    logs = [f for f in sorted(os.listdir(ws.calib)) if f.endswith(".csv")]
    if not logs:
        print(c("  No calibration CSV in the workspace -- run step 1 first.",
                "33"))
        return
    print("  Plots how well the fitted model tracks the data, and scores")
    print("  candidate model forms against each other by cross-validation, so")
    print("  you can see whether linear is really the right shape.")
    print()
    for i, f in enumerate(logs, 1):
        print(f"   {i:2d}. {f}")
    print()
    i = ask_int("Which log", 1, 1, len(logs))
    log = os.path.join(ws.calib, logs[i - 1])

    cmd = [sys.executable, DIAGNOSER, log]
    # Mirror the preprocessing of the existing fit so the report describes
    # that fit rather than a differently-preprocessed one.
    if ws.regression:
        cmd += ["--fit", ws.regression]
        print(c("  matching the preprocessing of the current regression", "2"))
    out = os.path.join(ws.calib, os.path.splitext(logs[i - 1])[0] +
                       "_diagnostics.html")
    cmd += ["-o", out]

    print(c("  analysing...", "2"))
    r = subprocess.run(cmd, capture_output=True, text=True)
    # The tool already prints the report path; do not echo it twice.
    for line in r.stdout.splitlines():
        if not line.strip().startswith("report:"):
            print(line)
    if r.returncode != 0:
        print(c("  diagnostics failed:", "31"))
        print((r.stderr or "").strip()[:1500])
        return
    print()
    print(c(f"  report: {out}", "32"))
    if confirm("Open it in a browser?"):
        import webbrowser
        webbrowser.open("file:///" + out.replace("\\", "/"))


# --------------------------------------------------------------------------
# Step 2 -- field and targets
# --------------------------------------------------------------------------

def _install(src: str, dst: str, what: str) -> bool:
    try:
        with open(src, "rb") as fh:
            raw = fh.read()
        if raw[:3] == b"\xef\xbb\xbf":
            raw = raw[3:]
        data = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        print(c(f"  {what} is not valid JSON: {e}", "31"))
        return False

    if what == "field":
        try:
            f = data["field"]
            for k in ("x_min", "y_min", "x_max", "y_max"):
                float(f[k])
            n = len(data.get("obstacles", []))
            for ob in data.get("obstacles", []):
                if len(ob["polygon"]) < 3:
                    raise ValueError(f"obstacle {ob.get('name')} has < 3 vertices")
            print(f"  field {f['x_max']-f['x_min']:.0f} x "
                  f"{f['y_max']-f['y_min']:.0f} cm, {n} obstacle(s)")

            poly = (data.get("robot") or {}).get("polygon")
            if poly:
                if len(poly) < 3:
                    raise ValueError("robot polygon has < 3 vertices")
                xs = [float(p[0]) for p in poly]
                ys = [float(p[1]) for p in poly]
                reach = max((x * x + y * y) ** 0.5 for x, y in zip(xs, ys))
                print(f"  robot {max(xs)-min(xs):.0f} x {max(ys)-min(ys):.0f} cm, "
                      f"{len(poly)} vertices, reach {reach:.1f} cm "
                      f"from the tracking point")
                # Off-centre is legitimate (the tracking point need not be the
                # chassis centre) but is worth flagging, since it is also what
                # a wrong-origin polygon looks like.
                cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
                if (cx * cx + cy * cy) ** 0.5 > 0.25 * reach:
                    print(c(f"  note: footprint centroid is ({cx:.1f}, {cy:.1f}), "
                            f"well off the tracking point -- intended?", "33"))
            else:
                print(c("  no 'robot' polygon: the robot will be treated as a "
                        "point, so obstacles must already include its size", "33"))
        except (KeyError, TypeError, ValueError, IndexError) as e:
            print(c(f"  field file is malformed: {e}", "31"))
            return False
    else:
        try:
            ts = data["targets"]
            for t in ts:
                if len(t["state"]) != 6:
                    raise ValueError(f"target {t.get('name')} state is not 6 numbers")
            print(f"  {len(ts)} target(s): "
                  + ", ".join(t.get("name", "?") for t in ts[:6])
                  + (" ..." if len(ts) > 6 else ""))
        except (KeyError, TypeError, ValueError) as e:
            print(c(f"  targets file is malformed: {e}", "31"))
            return False

    shutil.copy2(src, dst)
    print(c(f"  installed {dst}", "32"))
    return True


def step_field(ws: Workspace) -> None:
    rule("2. Field and targets")
    print("  Give the REAL obstacles, un-inflated, plus the robot's own footprint")
    print("  under 'robot'. The solver swells them per heading itself, which it")
    print("  has to do because the swept shape is not the same at every angle.")
    print("  See " + c("solver/examples/field.json", "36"))
    print()

    if confirm("Copy the bundled examples to start from?"):
        for name in ("field.json", "targets.json"):
            shutil.copy2(os.path.join(EXAMPLES, name),
                         os.path.join(ws.field, name))
        print(c(f"  copied examples into {ws.field}", "32"))
        print("  Edit them, then re-run this step to validate.")
        return

    for name, what in (("field.json", "field"), ("targets.json", "targets")):
        cur = os.path.join(ws.field, name)
        prompt = f"Path to {name}" + (" (blank to keep current)" if
                                      os.path.exists(cur) else "")
        p = ask(prompt, "")
        if not p:
            if os.path.exists(cur):
                _install(cur, cur, what)
            continue
        if not os.path.isfile(p):
            print(c(f"  no such file: {p}", "31")); continue
        _install(p, cur, what)


# --------------------------------------------------------------------------
# Step 3 -- solve
# --------------------------------------------------------------------------

# The cards a run can be planned for, and what they cost to rent by the hour.
#
# `plan` decides in-core against tiled from a VRAM budget and estimates a wall
# clock from a cell rate, so planning for a machine you do not have in front of
# you is just a matter of handing it that machine's two numbers. Both are
# overridable per entry, and `cell_rate` is deliberately absent until measured:
# `solver/cloud/benchmark.jl` fills it in, and until it has, the estimate is
# quoted at the desktop's rate and labelled as a ceiling.
#
# `headroom` matches the solver's own default -- the driver and the CUDA
# context need room that a solve may not have.
#
# The H200 is here for one reason worth knowing before you pick from this
# list: full resolution needs 100 GB to be held whole, so it is the first
# card that does not tile it. Six cents an hour over the H100 removes a
# +12.3% accuracy cost and, because the halo goes with it, comes out cheaper
# in absolute dollars. `solver/cloud/README.md` section 1 has the arithmetic.
TARGET_GPUS = {
    "local":  dict(label="this machine's card", vram_gb=None, usd_hr=0.0),
    "l40s":   dict(label="NVIDIA L40S 48 GB", vram_gb=48.0, usd_hr=1.57),
    "h100":   dict(label="NVIDIA H100 80 GB", vram_gb=80.0, usd_hr=3.39),
    "h200":   dict(label="NVIDIA H200 141 GB", vram_gb=141.0, usd_hr=3.45),
}
GPU_HEADROOM = 0.85
# Held back for the driver and the CUDA context, as the solver does.
GPU_RESERVE_GB = 1.0


def gpu_budget_bytes(vram_gb: float) -> int:
    return int((vram_gb - GPU_RESERVE_GB) * (2 ** 30) * GPU_HEADROOM)


def target_overrides(target: str, measured: dict) -> dict:
    """Config keys that make `plan` answer for `target` rather than for here.

    Only ever added for the *plan*. A solve that runs on this machine must see
    this machine's real card, so these are stripped before solving locally --
    a stale `vram_budget_bytes` would tile a grid that fits, or worse, claim
    one fits that does not.
    """
    spec = TARGET_GPUS.get(target)
    if not spec or spec["vram_gb"] is None:
        return {}
    out = {"vram_budget_bytes": gpu_budget_bytes(spec["vram_gb"])}
    rates = measured.get(target, {})
    # `cell_cost` is what the planner prices with -- two coefficients rather
    # than a rate, because how long a cell update takes depends on the
    # lookahead horizon as much as on the card. `cell_rate` is the older
    # single number, carried through for entries measured before that, and
    # `cell_rate_work` says which workload it was a rate of.
    if rates.get("cell_cost"):
        out["cell_cost"] = rates["cell_cost"]
    if rates.get("cell_rate"):
        out["cell_rate"] = float(rates["cell_rate"])
    if rates.get("cell_rate_work"):
        out["cell_rate_work"] = rates["cell_rate_work"]
    if rates.get("disk_rate"):
        out["disk_rate"] = float(rates["disk_rate"])
    return out


def julia_exe() -> str | None:
    return shutil.which("julia")


# Everything the solve needs, in the unit the thing is actually measured in.
# Nothing here is a sample count: counts are what the solver derives, not what
# a user should be asked to reason about.
DEFAULTS = {
    "clearance": 5.0,
    "wall_clearance": 5.0,
    "vmax": 170.0,
    "wmax": 8.0,
    "xy_cm": 8.0,
    "heading_deg": 22.5,
    "v_cm_s": 34.0,
    "w_rad_s": 1.6,
    "dtype": "u16",
    "iterations": 400,
    "cfl": 2.0,
    "tau_levels": 5,
    # Derived per run by the solver, not carried between them. It was a number
    # once and that was a trap: applying a recommendation on a tiled grid wrote
    # a short step into the saved answers, where it then stayed -- silently
    # costing accuracy on every later run, including in-core ones with nothing
    # to gain from it. "auto" is the only value the wizard ever writes.
    "tau_max": "auto",
    "level": 4,
    "budget_gb": 8.0,
    # After each target converges, fill in the cells it could not reach with
    # the time to get out of them, so a robot shoved into an obstacle has a
    # gradient to follow instead of `unreachable` in every direction. Written
    # into the config rather than left to the solver's default so it is
    # visible and tunable; it only ever writes cells that were going to be
    # unreachable anyway, so there is no reason to turn it off.
    "escape": True,
    "escape_iterations": 60,
    # The final-approach PID that MODEL.JSON carries beside the tables. These
    # are not gains: the gains are derived from the fit (TABLE_FORMAT.md
    # section 8.8), and these are the handful of judgements the derivation
    # takes. `honing_budget_frac` is the one worth an opinion -- how much of
    # the traction knee the approach may spend -- and the rest describe the
    # loop it will run in. Step `h` edits them; a solve only carries them.
    "honing_budget_frac": 0.5,
    "honing_bandwidth_scale": 1.0,
    "honing_loop_hz": 50.0,
    "honing_fine_cm": 2.0,
    "honing_fine_rad": 0.03,
    "honing_handoff_cm": 15.0,
    "honing_handoff_rad": 0.25,
    "honing_integral_share": 0.25,
}

# Passed through to the solver under exactly these names, so the wizard's
# saved answers and the solver config say the same thing in the same words.
HONING_KEYS = ("honing_budget_frac", "honing_bandwidth_scale",
               "honing_loop_hz", "honing_fine_cm", "honing_fine_rad",
               "honing_handoff_cm", "honing_handoff_rad",
               "honing_integral_share")


ELEM_BYTES = {"u8": 1, "u16": 2, "f16": 2, "f32": 4}


def derived_shape(span_x: float, span_y: float, d: dict) -> dict:
    """
    The sample counts a resolution implies, mirroring `axis_samples` and
    `symmetric_samples` in the solver.

    Duplicated arithmetic, deliberately, and it is three lines: it buys the
    user the size of what they are asking for *while they are asking for it*.
    Cell sizes are a much better way to say what you want than sample counts
    were, but they hide the one thing sample counts made obvious -- that the
    table goes as the product of six axes, so halving the position cell is
    eight times the space and not twice. Finding that out from `plan` twenty
    seconds later is finding out too late.
    """
    import math
    n1 = max(2, math.ceil(span_x / d["xy_cm"] - 1e-9) + 1)
    n2 = max(2, math.ceil(span_y / d["xy_cm"] - 1e-9) + 1)
    nh = max(4, round(360.0 / d["heading_deg"]))
    nv = 2 * max(1, math.ceil(d["vmax"] / d["v_cm_s"] - 1e-9)) + 1
    nw = 2 * max(1, math.ceil(d["wmax"] / d["w_rad_s"] - 1e-9)) + 1
    n = [n1, n2, nh, nv, nv, nw]
    cells = 1
    for x in n:
        cells *= x
    return {"n": n, "cells": cells,
            "bytes": cells * ELEM_BYTES.get(d["dtype"], 2),
            # 4 bytes for V plus 12 for the warm-start policy: what it would
            # take to hold the whole grid on the card at once.
            "vram": cells * 16}


def show_shape(shape: dict, ntargets: int, exact: bool) -> None:
    about = "" if exact else c("  (upper bound: the table is inset from the "
                               "wall, so the real count is a little smaller)", "2")
    print(f"    -> {' x '.join(str(x) for x in shape['n'])}"
          f" = {shape['cells']:,} cells, {human(shape['bytes'])} per table,"
          f" {human(shape['bytes'] * max(ntargets, 1))} total")
    v = shape["vram"]
    note = c("   <-- too big to hold whole; will be solved a tile at a time, "
             "which is much slower", "33") if v > 4 * 2 ** 30 else ""
    print(f"       {human(v)} of VRAM to hold it whole" + note)
    if about:
        print(about)


def gather_solver_config(ws: Workspace, run_dir: str, prev: dict,
                         span: tuple | None = None, exact: bool = False,
                         ntargets: int = 1) -> dict:
    """
    Ask for the run settings, in physical units, with the previous answers as
    the defaults.

    The previous answers matter more than they look. Tuning a grid is a loop --
    plan, look, adjust, plan again -- and re-typing fifteen answers to change
    one of them is most of what made the old version painful to use.
    """
    d = dict(DEFAULTS); d.update(prev)

    print()
    print("  The table covers every state the robot can legally be in, and the")
    print("  solver works out that extent itself: the field, inset by the")
    print("  robot's own footprint and the wall clearance. So there are no")
    print("  bounds to type in -- only how big you want one cell to be.")

    rule("Safety gaps")
    print("  A gap held around every obstacle, applied before the robot's own")
    print("  footprint is swept in, so it is plain 'keep this far away' and")
    print("  does not depend on how big the robot is.")
    d["clearance"] = ask_float("  clearance around obstacles, cm", d["clearance"])
    print()
    print("  The field wall is an obstacle too, and gets the same treatment by")
    print("  default. Give it its own number if you are willing to run closer")
    print("  to the perimeter than to a scoring structure.")
    d["wall_clearance"] = ask_float("  clearance from the field wall, cm",
                                    d["wall_clearance"])

    rule("Speed envelope")
    print("  The table says nothing outside this box -- a state beyond it is")
    print("  unreachable, not merely expensive -- so this is a real speed")
    print("  limit and not a description of the drivetrain. The fitted model")
    print("  can go far faster than either of these.")
    d["vmax"] = ask_float("  max |velocity|, cm/s", d["vmax"])
    d["wmax"] = ask_float("  max |omega|, rad/s", d["wmax"])

    rule("Resolution")
    print("  The physical size of one cell, on each axis. This is independent")
    print("  of the field and of the envelope above: change either and the")
    print("  cell stays the size you asked for while the sample count moves.")
    print("  Table size grows as the product of all six axes, so halving the")
    print("  position cell costs roughly eight times the space.")
    print()
    d["xy_cm"] = ask_float("  position cell, cm", d["xy_cm"])
    d["heading_deg"] = ask_float("  heading bin, degrees", d["heading_deg"])
    d["v_cm_s"] = ask_float("  velocity cell, cm/s", d["v_cm_s"])
    d["w_rad_s"] = ask_float("  omega cell, rad/s", d["w_rad_s"])
    if span is not None:
        print()
        show_shape(derived_shape(span[0], span[1], d), ntargets, exact)

    print()
    print("  Everything else has a sensible default: u16 tables, 400 sweeps,")
    print("  67 seeded controls, and a lookahead the plan will recommend.")
    if confirm("Change any of the solver settings?"):
        rule("Solver")
        print("  Element type: u8 (1 B, 25 ms steps to 6.3 s), u16 (2 B, 1 ms")
        print("  to 65.5 s, recommended), f16 (2 B, ~3 digits), f32 (4 B).")
        dt = ask("  dtype", d["dtype"])
        while dt not in ("u8", "u16", "f16", "f32"):
            dt = ask("  dtype (u8/u16/f16/f32)", d["dtype"])
        d["dtype"] = dt
        print()
        d["iterations"] = ask_int("  max sweeps per target", d["iterations"],
                                  1, 100000)
        print()
        print("  Lookahead. Each backup asks 'what is the best command to hold")
        print("  for a while, and what is left to do afterwards'. How long that")
        print("  while should be is not one number: at rest the robot barely")
        print("  moves however long you wait, and at speed it crosses cells")
        print("  fast. So you say how far a step should reach, in cells, and")
        print("  the solver derives the seconds for every cell itself.")
        d["cfl"] = ask_float("  cells advanced per step", d["cfl"])
        print()
        print("  Each cell then tries several step lengths around that one and")
        print("  keeps the best. 5 brackets it widely, which measured best.")
        d["tau_levels"] = ask_int("  step lengths per cell", d["tau_levels"],
                                  1, 8)
        print()
        print("  Longest step any of them may be. 'auto' is right nearly")
        print("  always: on a grid that fits the GPU it is the full 0.5 s,")
        print("  and on one that has to be tiled the solver works out the")
        print("  knee of the accuracy/time trade for that particular grid.")
        print("  A number here pins it, including on grids where a short")
        print(c("  step buys nothing and costs real accuracy.", "33"))
        raw = ask("  longest step, seconds (or 'auto')", str(d["tau_max"]))
        if raw.strip().lower() in ("auto", ""):
            d["tau_max"] = "auto"
        else:
            try:
                d["tau_max"] = float(raw)
            except ValueError:
                print(c("    not a number; leaving it on auto", "31"))
                d["tau_max"] = "auto"
        print()
        print("  Control sampling. Bang-bang puts the optimum on the boundary")
        print("  of the reachable set but not at its corners, so the boundary")
        print("  is sampled and then refined off-lattice. The lattice only")
        print("  seeds that search, so a high level costs little.")
        print("    1 = 7 controls (corners only, not recommended)")
        print("    2 = 19    3 = 39    4 = 67 (recommended)")
        d["level"] = ask_int("  control lattice level", d["level"], 1, 4)
        print()
        print("  How much space the recommended resolution is allowed to use.")
        print("  It does not cap anything -- it is what 'recommended' is")
        print("  recommended against.")
        d["budget_gb"] = ask_float("  size budget for the recommendation, GB",
                                   d["budget_gb"])

    return d


def config_from(ws: Workspace, run_dir: str, d: dict) -> dict:
    """The answers, as the JSON the solver reads."""
    return {
        "regression": ws.regression,
        "field": ws.field_file,
        "targets": ws.targets_file,
        "out_dir": run_dir,
        "grid": {
            "vmax": d["vmax"], "wmax": d["wmax"],
            "resolution": {
                "xy_cm": d["xy_cm"], "heading_deg": d["heading_deg"],
                "v_cm_s": d["v_cm_s"], "w_rad_s": d["w_rad_s"],
            },
        },
        "clearance_cm": d["clearance"],
        "wall_clearance_cm": d["wall_clearance"],
        "size_budget_bytes": int(d["budget_gb"] * 2 ** 30),
        "dtype": d["dtype"],
        "iterations": d["iterations"],
        "cfl": d["cfl"],
        "tau_levels": d["tau_levels"],
        "tau_max": d["tau_max"],
        "control_level": d["level"],
        "escape": d.get("escape", True),
        "escape_iterations": d.get("escape_iterations", 60),
        # The honing PID is derived from the same regression the tables are
        # and ships in MODEL.JSON beside them, so its settings travel in the
        # solve config. Written even when untouched: what the card claims it
        # was designed for should not depend on which wizard wrote it.
        **{k: d.get(k, DEFAULTS[k]) for k in HONING_KEYS},
        "zero_c": True,
        "backend": "auto",
        # The scratch file for a tiled solve is the whole value function --
        # tens of gigabytes. It goes in the workspace, which the user already
        # chose for exactly this reason, rather than beside the system drive's
        # temp directory.
        "scratch_dir": ws.runs,
    }


def run_plan(cfgpath: str) -> dict | None:
    # --threads=auto matters here as well as in the solve: `plan` measures
    # the dependency reach by integrating the model over the whole velocity
    # envelope, which is embarrassingly parallel and the difference between a
    # second and a quarter of a minute on every resolution the user tries.
    r = subprocess.run([julia_exe(), f"--project={SOLVER_PROJ}",
                        "--threads=auto", SOLVER, cfgpath, "plan"],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("PLAN "):
            return json.loads(line[5:])
    print(c("  plan failed:", "31"))
    print((r.stderr or r.stdout).strip()[:2000])
    return None


def show_extent(p: dict) -> None:
    """Where the table starts and stops, and why it is not the whole field."""
    b = p.get("bounds")
    if not b:
        return
    fx0, fy0, fx1, fy1 = b["field"]
    tmin, tmax = b["table_min"], b["table_max"]
    ins = b["inset_cm"]
    print(f"  field               {fx1-fx0:.0f} x {fy1-fy0:.0f} cm")
    print(f"  table covers        x {tmin[0]:.1f}..{tmax[0]:.1f}, "
          f"y {tmin[1]:.1f}..{tmax[1]:.1f} cm")
    print(f"  inset from the wall {min(ins):.1f}..{max(ins):.1f} cm  "
          + c(f"({b['cells_saved_frac']*100:.0f}% fewer cells than the "
              f"whole field)", "2"))
    print(c("  The robot cannot put its footprint through the wall, so states "
            "outside", "2"))
    print(c(f"  that span are not stored. Wall clearance "
            f"{b['wall_clearance_cm']:.1f} cm, obstacle clearance "
            f"{b['clearance_cm']:.1f} cm.", "2"))


def show_resolution(p: dict) -> None:
    """What was asked for, what the sample counts realise, and the shape."""
    r = p.get("resolution")
    if not r:
        return
    req, act, n = r["requested"], r["actual"], p["n"]
    rows = (("position", "xy_cm", "cm", 2, (n[0], n[1])),
            ("heading", "heading_deg", "deg", 2, (n[2],)),
            ("velocity", "v_cm_s", "cm/s", 1, (n[3], n[4])),
            ("omega", "w_rad_s", "rad/s", 2, (n[5],)))
    print(f"  {'axis':<10}{'asked':>9}{'actual':>10}   samples")
    for label, key, unit, dp, counts in rows:
        got = act[key]
        flag = "" if abs(got - req[key]) <= 5e-3 * max(req[key], 1e-9) \
               else c("  <-- rounded", "33")
        print(f"  {label:<10}{req[key]:>9.{dp}f}{got:>10.{dp}f} {unit:<6}"
              + " x ".join(str(x) for x in counts) + flag)
    if r["pinned_by_n"]:
        print(c("  grid.n is set in the config, so the counts are pinned and "
                "the cell size falls out of the span instead.", "33"))


def show_targets(p: dict) -> bool:
    """Per-target verdict. Returns True if every target can seed."""
    ts = p.get("targets") or []
    ok = True
    for t in ts:
        if t["ok"]:
            print(f"  {c('ok', '32')}   {t['name']}")
            continue
        ok = False
        if t["off_axes"]:
            why = ("outside the table on " + ", ".join(t["off_axes"]))
        else:
            why = "; ".join(t["blocked_by"]) or "blocked"
        print(f"  {c('BAD', '1;31')}  {t['name']}: {why}")
    if not ok:
        print(c("  A target the robot cannot be in seeds nothing, and the "
                "whole table comes out unreachable -- silently, after the "
                "full run. Move it, or lower the clearance.", "31"))
    return ok


def show_recommendation(p: dict, d: dict) -> dict:
    """
    Print what the settings ought to be, and return the ones on offer.

    Two of them, and they are different kinds of advice. The resolution is
    arithmetic against a size budget. The lookahead is a measurement: how far
    one backup reaches, what halo that forces, and what the accuracy is worth
    against the 0.5 s reference the cost curve was measured at.
    """
    rec = p.get("recommend") or {}
    offer = {}

    r = rec.get("resolution")
    if r:
        cur = p["cells"]
        if r["cells"] > cur * 1.05 or r["cells"] < cur * 0.95:
            print(f"  resolution          {r['xy_cm']:.1f} cm cells, "
                  f"{r['heading_deg']:.1f} deg bins, {r['v_cm_s']:.0f} cm/s, "
                  f"{r['w_rad_s']:.1f} rad/s")
            print(f"                      -> {'x'.join(str(x) for x in r['n'])}"
                  f" = {r['cells']:,} cells, {human(r['bytes'])} on the card")
            bound = ("the GPU, so it still holds whole"
                     if r.get("bound_by") == "vram"
                     else f"the {human(rec['size_budget_bytes'])} card budget")
            print(c(f"                      the finest that clears both "
                    f"budgets -- limited by {bound}; heading follows the "
                    f"footprint, velocity is 21 samples with zero on the grid",
                    "2"))
            offer["resolution"] = r

    # Reported, never offered. The step length the run will use is derived by
    # the solver on this plan and on the solve that follows it, from the same
    # code path, so there is nothing here for the user to accept and nothing
    # to carry into the next run. Bundling it into an "apply the recommended
    # settings?" prompt is what made a short step stick to a workspace.
    applied = rec.get("tau_applied")
    if applied is not None:
        acost = rec.get("tau_applied_cost") or 0.0
        how = "auto" if rec.get("tau_auto") else c("pinned in the config", "33")
        print(f"  longest step        {applied:.3f} s ({how})"
              + (c(f"   costs about +{acost:.0f}% on mean value against the "
                   f"{rec['tau_reference']} s reference", "33")
                 if acost > 0.5 else c("   full accuracy", "32")))
        if rec.get("tau_reason"):
            print(c(f"                      {rec['tau_reason']}", "2"))
        if rec.get("tau_pinned_waste"):
            print(c("  This grid fits the GPU whole, so there is no halo for a "
                    "short step to pay for -- the pinned value is costing "
                    "accuracy and buying nothing. Set 'longest step' back to "
                    "auto in the solver settings.", "1;31"))

    # The options table only makes sense next to a pick. `recommend_tau`
    # returns options without one in no case today, but the two come from
    # different fields and a nothing-fits grid is exactly when this is being
    # read most carefully.
    ic = rec.get("in_core")
    if ic:
        print(f"  to avoid tiling      {ic['xy_cm']:.1f} cm cells, "
              f"{ic['heading_deg']:.1f} deg bins, {ic['v_cm_s']:.0f} cm/s, "
              f"{ic['w_rad_s']:.2f} rad/s"
              + c(f"   ({ic['scale']:.2f}x coarser)", "2"))
        print(f"                      -> {'x'.join(str(x) for x in ic['n'])}"
              f" = {ic['cells']:,} cells, {human(ic['vram_bytes'])} of VRAM"
              + c("   holds whole", "32"))
        print(c("                      tiling is the difference between "
                "minutes and days, so this is usually the trade worth making",
                "2"))
        offer["in_core"] = ic

    # The table is marked against the value actually in force -- `tau_applied`,
    # what `settle` resolved and what the solve will use -- not against the raw
    # pick. When the step is pinned the two differ, and marking the pick would
    # point at a row the run is not on.
    opts = rec.get("tau_options") or []
    if applied is not None and len(opts) > 1:
        print()
        print(c("  step    reach   loaded/updated    sweeps      i/o   "
                "per round   value cost", "2"))
        for o in opts[:8]:
            mark = c(" <--", "1;32") if abs(o["tau"] - applied) < 1e-3 else ""
            print(f"  {o['tau']:>5.2f} {o['reach_cm']:>7.0f} cm "
                  f"{o['amplification']:>10.1f}x {hms(o['compute_s']):>9} "
                  f"{hms(o['io_s']):>8} {hms(o['round_s']):>11}  "
                  f"+{o['cost']:>4.0f}%" + mark)
    return offer


def _field_span(ws: Workspace) -> tuple | None:
    """The field's own width and height, for the size preview."""
    try:
        with open(ws.field_file, "rb") as fh:
            raw = fh.read()
        f = json.loads(raw.decode("utf-8-sig"))["field"]
        return (float(f["x_max"]) - float(f["x_min"]),
                float(f["y_max"]) - float(f["y_min"]))
    except (OSError, TypeError, KeyError, ValueError):
        return None


def _count_targets(ws: Workspace) -> int:
    try:
        with open(ws.targets_file, "rb") as fh:
            return len(json.loads(fh.read().decode("utf-8-sig"))["targets"])
    except (OSError, TypeError, KeyError, ValueError):
        return 1


def _cost_preview(rt: dict, target: str, measured: dict) -> None:
    """What this run would cost to rent, and how much to trust the figure.

    The estimate spends the whole iteration budget, so it is a ceiling on
    that count -- and unless the box has been benchmarked it spends it at the
    desktop card's cost, which is a guess in an unknown direction. A user who
    reads "$52" and does not know both of those will buy the wrong hours.

    The two passes are shown apart because they are priced on different
    grounds and can be wrong independently: the sweeps are the bulk and rest
    on a measurement of the card, the escape pass is a floor built on how
    much of the field is blocked. Adding them into one number hides which one
    to distrust when a run overruns -- which is exactly what happened to the
    H200 job that this split exists because of.
    """
    spec = TARGET_GPUS.get(target, {})
    hourly = spec.get("usd_hr", 0.0)

    unit_s = rt.get("unit_s") or 0.0
    esc_s = rt.get("escape_unit_s") or 0.0
    units = rt.get("units") or 0
    esc_n = rt.get("escape_sweeps") or 0
    if unit_s and units:
        word = rt.get("unit", "sweep")
        print(f"  {word}s              {units - esc_n} x {unit_s:.1f} s"
              f" = {hms(unit_s * (units - esc_n))}")
        if esc_n:
            print(f"  escape pass         {esc_n} x {esc_s:.1f} s"
                  f" = {hms(esc_s * esc_n)}"
                  + c(f"   (floor: {rt.get('blocked_frac', 0)*100:.0f}% of the "
                      f"grid is blocked and cannot be reached)", "2"))

    # The prefetch, when the tiled driver is going to use it. Reported as what
    # it saves rather than as a flag, because on a slow card it saves nothing
    # and on a fast one it is the difference between two very different bills.
    if rt.get("prefetch"):
        saved = rt.get("prefetch_saves_s", 0.0)
        if saved > 1.0:
            line = f"  prefetch            on, hiding {hms(saved)} of reads"
            if hourly:
                line += f" ({c(f'${saved / 3600 * hourly:,.2f}', '32')})"
            print(line)
    elif rt.get("prefetch_off_because"):
        why = rt["prefetch_off_because"]
        if why not in ("not tiled",):
            print(c(f"  prefetch            off -- {why}", "33"))

    if not hourly:
        return
    cost = rt["total_s"] / 3600.0 * hourly
    print(f"  {c('rental cost', '1;33')}         "
          f"{c(f'up to ${cost:,.2f}', '1;33')} at ${hourly:.2f}/hr "
          f"on {spec['label']}")
    src = rt.get("cost_source", "cost")
    if src == "default":
        print(c("                      ...but that is at THIS machine's cost "
                "per cell. The card is", "2"))
        print(c("                      faster, so the real bill is lower. "
                "Measure it once with", "2"))
        print(c("                      peregrine_remote.py benchmark and this "
                "becomes arithmetic.", "2"))
    elif src == "rate":
        # An old-style `cell_rate` says how fast the box was on one workload
        # and not which one, so the planner has to assume it was the workload
        # the retired benchmark used. When that assumption is wrong the
        # estimate is wrong by however much the two workloads differ -- on
        # the first H200 job, by 41%.
        print(c("                      ...from an old-style cell rate, which "
                "does not record what", "2"))
        print(c("                      workload it was measured on. Re-run "
                "peregrine_remote.py benchmark", "2"))
        print(c("                      on this box; it fits the cost model "
                "and this stops being a guess.", "2"))


def _pick_target(default: str) -> str:
    """Which card the plan should answer for."""
    print()
    print("  Plan for which GPU?")
    keys = list(TARGET_GPUS)
    rates = load_settings().get("gpu_rates", {})
    for i, k in enumerate(keys, 1):
        spec = TARGET_GPUS[k]
        bits = []
        if spec["vram_gb"]:
            bits.append(f"{spec['vram_gb']:.0f} GB")
        if spec["usd_hr"]:
            bits.append(f"${spec['usd_hr']:.2f}/hr")
        got = rates.get(k, {})
        if got.get("cell_cost"):
            bits.append(c(f"measured {got['cell_rate']/1e6:.0f}M cells/s",
                          "32"))
        elif got.get("cell_rate"):
            # Measured before the cost model, so the workload it was measured
            # at has to be assumed rather than known. Better than nothing and
            # worth re-measuring.
            bits.append(c(f"{got['cell_rate']/1e6:.0f}M cells/s "
                          f"(old-style -- re-benchmark)", "33"))
        elif spec["vram_gb"]:
            bits.append(c("not measured -- estimates are ceilings", "33"))
        mark = "*" if k == default else " "
        print(f"   {mark}{i}. {spec['label']:<24s} {'  '.join(bits)}")
    print(c("     Measure a rented box with: julia --project=solver -t auto "
            "solver/cloud/benchmark.jl", "2"))
    raw = ask("  choice", str(keys.index(default) + 1 if default in keys else 1))
    try:
        return keys[int(raw) - 1]
    except (ValueError, IndexError):
        return default


def _pick_where(target: str) -> str:
    """Solve here, solve on the droplet, save for later, or change settings.

    "Save for later" exists because planning and renting are on different
    clocks. A plan is worked out over an evening of re-planning at different
    resolutions; the droplet is created minutes before it is needed and
    destroyed the moment the tables land, because it bills until it is
    destroyed. Requiring the box to exist at the moment the plan is settled
    would mean renting it through the whole deliberation.
    """
    spec = TARGET_GPUS.get(target, {})
    print()
    if spec.get("vram_gb"):
        print(f"   1. Solve on the rented {spec['label']} "
              + c("(uploads and runs there -- needs the box up now)", "2"))
        print("   2. Save this plan as a job to run later "
              + c("(no droplet needed now)", "2"))
        print("   3. Solve on this machine instead")
        print("   4. Change settings")
        raw = ask("  choice", "2")
        return {"1": "remote", "2": "save",
                "3": "local"}.get(raw.strip(), "back")
    if confirm("Start solving with these settings?"):
        return "local"
    return "save" if confirm("Save this plan as a job to run later instead?",
                             default_yes=False) else "back"


# The keys of a saved job that name an input file. Frozen copies live beside
# job.json under these names, so the job does not depend on the workspace
# still holding what it held at save time.
JOB_INPUTS = {"regression": "drivetrain_fit.toml",
              "field": "field.json",
              "targets": "targets.json"}


def save_job(ws: Workspace, cfg: dict, target: str, p: dict) -> str | None:
    """Freeze a planned config into a directory that runs itself later.

    Self-contained on purpose. The alternative -- a job that points back at
    `<workspace>/field/field.json` -- reads the field as it is on the day it
    is *run*, so editing a target between planning and renting would quietly
    solve a different problem than the one that was planned, and the plan
    printed here would be a record of nothing. The three inputs are
    kilobytes; copying them costs nothing and makes the job a fact.

    The planning overrides stay in the config. `vram_budget_bytes` is the
    dangerous one and `remote_config` on the far side drops it before the box
    ever sees it, so what survives here is a faithful record of what was
    planned rather than an instruction to the solver.
    """
    print()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = ask("  job name", stamp).strip()
    # One path segment, no surprises: this becomes a directory name and is
    # typed back on a command line.
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.") or stamp
    if safe != name:
        print(c(f"  saving as '{safe}'", "2"))
    jdir = os.path.join(ws.jobs, safe)
    if os.path.isdir(jdir) and not confirm(
            f"'{safe}' already exists -- overwrite it?", default_yes=False):
        return None
    os.makedirs(jdir, exist_ok=True)

    cfg = dict(cfg)
    for key, fname in JOB_INPUTS.items():
        src = cfg.get(key)
        if not src or not os.path.isfile(src):
            print(c(f"  cannot save: config's '{key}' is not a file "
                    f"({src})", "31"))
            return None
        shutil.copy2(src, os.path.join(jdir, fname))
        # Relative, so the job survives being moved or copied to another
        # machine. `peregrine_remote.py` resolves them against the job dir.
        cfg[key] = fname
    # Where the tables come home to. Absolute, because it is the one path
    # that names a place outside the job.
    cfg["out_dir"] = os.path.join(jdir, "tables")
    cfg["scratch_dir"] = ws.runs

    with open(os.path.join(jdir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    ooc = p.get("out_of_core") or {}
    rec = p.get("recommend") or {}
    rt = p.get("runtime") or {}
    spec = TARGET_GPUS.get(target, {})
    job = {
        "name": safe,
        "saved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_gpu": target,
        "gpu_label": spec.get("label", target),
        "usd_hr": spec.get("usd_hr", 0.0),
        # What was planned, kept so `jobs` can list it and a later run can be
        # checked against it without re-planning. Advisory: the box re-plans
        # for itself, and its answer is the one that runs.
        "plan": {
            "n": p.get("n"),
            "cells": p.get("cells"),
            "n_targets": p.get("n_targets"),
            "bytes_per_target": p.get("bytes_per_target"),
            "bytes_total": p.get("bytes_total"),
            "driver": ooc.get("mode"),
            "amplification": ooc.get("amplification"),
            "store_bytes": ooc.get("store_bytes"),
            "tau_applied": rec.get("tau_applied"),
            "tau_applied_cost": rec.get("tau_applied_cost"),
            "estimate_s": rt.get("total_s"),
            # The rate the estimate used, what made it that rate, and whether
            # it came from a measurement of the target card or from this
            # desktop standing in. Kept so a run that overshoots can be
            # checked against what was assumed rather than re-derived.
            "cell_rate": rt.get("cell_rate"),
            "cell_cost": rt.get("cell_cost"),
            "cost_measured": rt.get("cost_measured"),
            "work": rt.get("work"),
            "solve_sweep_s": rt.get("unit_s"),
            "escape_sweep_s": rt.get("escape_unit_s"),
        },
    }
    with open(os.path.join(jdir, "job.json"), "w", encoding="utf-8") as fh:
        json.dump(job, fh, indent=2)

    est = rt.get("total_s") or 0.0
    hourly = spec.get("usd_hr", 0.0)
    print()
    print(c(f"  saved job '{safe}'", "1;32"))
    print(f"    {jdir}")
    if est:
        line = f"    planned {hms(est)} on {spec.get('label', target)}"
        if hourly:
            line += f", about ${est / 3600 * hourly:,.2f}"
        print(c(line, "2"))
    print()
    print("  When the droplet is up, from this repo:")
    print(c(f"    py -3.12 solver/cloud/peregrine_remote.py provision "
            f"root@<ip>", "36"))
    print(c(f"    py -3.12 solver/cloud/peregrine_remote.py run {safe} "
            f"--host root@<ip>", "1;36"))
    print(c("  The tables come home to the job's tables/ directory.", "2"))
    return jdir


def _solve_remote(cfgpath: str, run_dir: str, target: str) -> None:
    """Hand the run to solver/cloud/peregrine_remote.py.

    The config written for the plan carries this machine's idea of the rented
    card -- `vram_budget_bytes` above all. The remote driver drops that and
    lets the real card speak for itself, which is why the override is safe to
    leave in the file the plan was made from.
    """
    tool = os.path.join(REPO, "solver", "cloud", "peregrine_remote.py")
    if not os.path.isfile(tool):
        print(c("  solver/cloud/peregrine_remote.py is missing", "31")); return
    spec = TARGET_GPUS.get(target, {})
    cmd = [sys.executable, tool, "--rate", str(spec.get("usd_hr", 0.0)),
           "run", cfgpath]
    print()
    print(c("  handing the run to the droplet...", "36"))
    print(c("    " + " ".join(cmd), "2"))
    print()
    try:
        rc = subprocess.run(cmd).returncode
    except OSError as e:
        print(c(f"  could not start the remote driver: {e}", "31")); return
    if rc != 0:
        print(c(f"  remote run failed (exit {rc})", "31"))
        print("  If it could not reach the box, set one with:")
        print(c(f"    {sys.executable} {tool} host root@<ip>", "36"))
        return
    print(c(f"  tables are in {run_dir}", "32"))


def step_solve(ws: Workspace) -> None:
    rule("3. Solve value tables")
    if not ws.regression:
        print(c("  No regression yet -- run step 1 first.", "33")); return
    if not ws.field_file or not ws.targets_file:
        print(c("  Field or targets missing -- run step 2 first.", "33")); return
    if not julia_exe():
        print(c("  julia is not on PATH.", "31")); return

    run_dir = os.path.join(ws.runs, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    cfgpath = os.path.join(run_dir, "config.json")

    answers = dict(load_settings().get("solver", {}))
    # The step length is derived per run, so it is never restored from the
    # last one -- and a workspace saved by an older wizard may well have a
    # short one baked in, which is the exact failure this is undoing. Pinning
    # it stays possible, in the solver settings, for the current session.
    answers.pop("tau_max", None)
    # The field's own span, until a plan reports the real one. It is an upper
    # bound -- the table is inset from the wall -- which is the right way for
    # a preview to be wrong.
    span, span_exact = _field_span(ws), False
    ntargets = _count_targets(ws)

    st0 = load_settings()
    target = _pick_target(st0.get("target_gpu", "local"))
    st0["target_gpu"] = target
    save_settings(st0)
    measured = load_settings().get("gpu_rates", {})
    over = target_overrides(target, measured)

    ask_again = True
    while True:
        if ask_again:
            answers = gather_solver_config(ws, run_dir, answers, span,
                                           span_exact, ntargets)
        ask_again = True
        cfg = config_from(ws, run_dir, answers)
        # The plan answers for the target card; the solve config on disk does
        # not carry the override any further than that.
        cfg.update(over)
        with open(cfgpath, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)

        p = run_plan(cfgpath)
        if p is None:
            return

        ooc = p.get("out_of_core") or {}
        tiled = ooc.get("mode") == "ooc"
        free = ws.free()
        # From here on the preview uses the span the solver actually computed,
        # not the field's.
        b = p.get("bounds")
        if b:
            span = (b["table_max"][0] - b["table_min"][0],
                    b["table_max"][1] - b["table_min"][1])
            span_exact = True
        ntargets = p["n_targets"]

        rule("Table")
        show_extent(p)
        print()
        show_resolution(p)
        print()
        print(f"  cells per table     {p['cells']:,}")
        print(f"  bytes per table     {human(p['bytes_per_target'])}")
        print(f"  TOTAL on card       {c(human(p['bytes_total']), '1;33')}"
              f"   ({p['n_targets']} target(s), "
              f"{p['chunks_per_target']} chunk(s) each)")
        print(f"  workspace free      {human(free)}"
              + ("" if p["bytes_total"] < free else c("   <-- NOT ENOUGH", "31")))
        occ = p.get("occupancy")
        if occ:
            print(f"  blocked by geometry {occ['blocked_frac']*100:.0f}% of "
                  f"(x, y, heading) cells"
                  + c(f"   ({occ['best_heading_blocked_frac']*100:.0f}% at the "
                      f"easiest heading, "
                      f"{occ['worst_heading_blocked_frac']*100:.0f}% at the "
                      f"hardest)", "2"))

        rule("Targets")
        targets_ok = show_targets(p)

        rule("Machine")
        if p["gpu_available"]:
            print(f"  GPU                 needs {p['gpu_need_gb']:.2f} GB of "
                  f"{p['gpu_free_gb']:.2f} GB free"
                  + ("" if p["gpu_fits"] else c("   <-- too big to hold whole",
                                                "33")))
        else:
            print(c("  GPU                 not available; will run on CPU "
                    "(much slower)", "33"))

        # How well the position resolution matches the velocity resolution.
        # A backup only learns from a step that leaves its own cell, and the
        # distance covered while the speed changes by one velocity cell is
        # dv^2 / (2a). Far below one position cell means near-stationary
        # states are heavily smoothed and their times come out pessimistic;
        # far above means position cells are being paid for without buying
        # accuracy. Reported, not enforced -- the right answer depends on how
        # much time the robot spends near rest.
        bal = p.get("grid_balance")
        if bal is not None and bal == bal:          # not NaN
            note = ""
            if bal < 0.05:
                note = c("   coarse in position near rest", "33")
            elif bal > 4.0:
                note = c("   finer in position than the dynamics can use", "33")
            print(f"  grid balance        {bal:.3f}"
                  f"   ({p['cell_cm']:.1f} cm cells, {p['cell_cm_s']:.0f} cm/s "
                  f"cells, {p['max_accel_cm_s2']:.0f} cm/s^2)" + note)

        # A grid too big for the card is solved a tile at a time, out of a
        # scratch file. What that costs is the amplification: how many cells
        # have to be loaded per cell updated, set by how far one backup can
        # reach against how big a tile the card can hold.
        if tiled and ooc.get("fits"):
            hx, hy = ooc["halo_cells"]
            tx, ty = ooc["tile_cells"]
            amp = ooc["amplification"]
            note = c("   <-- most of the work is halo", "33") if amp > 3 else ""
            print()
            print(f"  too big to hold whole: solving {ooc['tiles_per_round']} "
                  f"tiles of {tx}x{ty} cells at a time")
            print(f"  reach               {ooc['reach_cm']:.0f} cm "
                  f"= {ooc['reach_cells']:.0f} cells  -> halo {hx}x{hy}")
            print(f"  loaded per updated  {amp:.2f}x" + note)
            print(f"  scratch file        {human(ooc['store_bytes'])} in the "
                  f"workspace, {human(ooc['io_bytes_per_round'])} moved per round")

        # How long this is going to take. It is the ceiling -- the full
        # iteration budget, which `tolerance` usually cuts short -- built from
        # two rates measured on one machine, so it separates "overnight" from
        # "next week" rather than being trusted to the hour.
        rt = p.get("runtime")
        if rt and rt.get("unit") not in (None, "none"):
            print()
            print(f"  expected run time   up to {hms(rt['total_s'])} for "
                  f"{p['n_targets']} target(s)"
                  + c("   (budget; tolerance usually stops sooner)", "2"))
            _cost_preview(rt, target, measured)
            print(f"  per {rt['unit']:<15} {hms(rt['unit_s'])} x "
                  f"{rt['units']} {rt['unit']}s per target")
            # The escape pass is in that count, priced at a full backup per
            # cell. It is not one: a cell with a route retires immediately, so
            # on a table that reaches most of its states this is a large
            # over-estimate of a small number. Broken out rather than folded
            # in silently, because it moved the total.
            esc = (rt.get("escape_rounds") if rt["unit"] == "round"
                   else rt.get("escape_sweeps"))
            if esc:
                print(f"  of which escape     {esc} {rt['unit']}s"
                      + c("   (over-priced; reachable cells retire at once)",
                          "2"))
            if rt["unit"] == "round":
                # A round pays for both halves in sequence -- load, sweep,
                # store -- so this says which half is the larger, not which
                # one "limits". Nothing is hidden behind anything else.
                lim = "disk" if rt["io_bound"] else "GPU"
                print(f"  mostly              {c(lim, '1')}  "
                      f"(sweeps {hms(rt['compute_s'])} + "
                      f"i/o {hms(rt['io_s'])} per round)")

        rule("Recommended")
        offer = show_recommendation(p, answers)
        if not offer:
            print(c("  Nothing to change -- these settings are already at the "
                    "recommendation.", "32"))

        rule()
        need = p["bytes_total"] + (ooc.get("store_bytes", 0) if tiled else 0)
        blocked = not targets_ok
        if need >= free:
            blocked = True
            print(c("  Not enough space in the workspace. Use a bigger "
                    "position cell.", "31"))
        elif tiled and not ooc.get("fits"):
            _no_tiling_advice(ooc, offer)
            blocked = True

        # Offered whether or not the settings can run, and *especially* when
        # they cannot. The recommendation comes from the same reach scan that
        # decided nothing fits, so when it names a step length it is naming one
        # that does fit -- which makes it the fix, not a polish step. Printing
        # it above a wall of red and then refusing to apply it was backwards.
        #
        # Asked as two questions rather than one because the offers conflict:
        # "coarsen until it holds whole" and "the finest that fits the card"
        # are different targets and applying both means applying whichever was
        # written last. The bigger win goes first.
        def _apply_res(r):
            answers.update(xy_cm=r["xy_cm"], heading_deg=r["heading_deg"],
                           v_cm_s=r["v_cm_s"], w_rad_s=r["w_rad_s"])

        if "in_core" in offer:
            if confirm("Coarsen to hold the whole grid on the GPU and re-plan?",
                       default_yes=True):
                _apply_res(offer["in_core"])
                ask_again = False
                continue
        # Only ever a resolution: the step length is derived, not offered.
        rest = {k: v for k, v in offer.items() if k != "in_core"}
        if rest:
            lead = ("Apply the recommended settings and re-plan?" if not blocked
                    else "Apply the recommended settings and try again?")
            if confirm(lead, default_yes=True):
                _apply_res(rest["resolution"])
                ask_again = False
                continue
        if not blocked:
            where = _pick_where(target)
            if where in ("remote", "save"):
                st = load_settings()
                st["solver"] = {k: v for k, v in answers.items()
                                if k != "tau_max"}
                save_settings(st)
            if where == "remote":
                _solve_remote(cfgpath, run_dir, target)
                return
            if where == "save":
                # Saved from `cfg`, the dict the plan above was made from,
                # rather than by re-reading cfgpath -- they are the same
                # bytes, and going through the object keeps the job tied to
                # the plan being displayed rather than to a file that a
                # later loop iteration would overwrite.
                if save_job(ws, cfg, target, p):
                    return
                # Cancelled at the overwrite prompt. Falls through to the
                # "adjust and try again" question at the bottom rather than
                # looping, so declining does not re-ask every setting.
            if where == "local":
                break
        # If the settings cannot run, going back is the only useful move, so
        # that is what Enter should do. Only when the user has declined a
        # perfectly workable plan does leaving become the likelier intent.
        if not confirm("Adjust settings and try again?", default_yes=blocked):
            return

    # Remembered across runs: the next solve starts from what worked last time
    # rather than from the defaults. Everything except the step length, which
    # is a consequence of the grid rather than a preference about it.
    st = load_settings()
    st["solver"] = {k: v for k, v in answers.items() if k != "tau_max"}
    save_settings(st)
    if over:
        # Planned for a rented card, solving on this one. The budget and the
        # rates belong to the other machine and would mis-route this run --
        # the whole class of bug this strip exists to prevent is a config that
        # claims a grid fits in VRAM this card does not have.
        cfg = config_from(ws, run_dir, answers)
        with open(cfgpath, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
        print(c("  planning override dropped -- solving against this card",
                "2"))
    _stream_solve(cfgpath, run_dir)


def _no_tiling_advice(ooc: dict, offer: dict) -> None:
    """
    Nothing fits at the configured step length. Say why, then list *every*
    lever rather than the first one that happens to apply.

    The old version was an if/elif chain, so a grid where both a shorter step
    and dropping the warm start would have worked only ever heard about the
    step -- and the step is the one that costs accuracy, so it was reliably
    naming the worst of the available fixes first.
    """
    # The smallest tile is one column of interior inside the halo, so it costs
    # (1 + 2h)^2 columns -- and h is set by the longest step, not by the grid.
    h = ooc.get("halo_cells", [0, 0])[0]
    w = ooc.get("min_window_cells", 1 + 2 * h)
    print(c(f"  No tiling fits at this step length. The reach is "
            f"{ooc.get('reach_cm', 0):.0f} cm = {ooc.get('reach_cells', 0):.0f}"
            f" cells, so the halo is {h} and the smallest tile is {w}x{w} "
            f"columns of {human(ooc.get('column_bytes', 0))} -- against "
            f"{human(ooc.get('budget_bytes', 0))} of usable VRAM.", "31"))
    print()
    print("  Three levers, cheapest first:")

    # 1. The column, which is the real constraint and the only lever that
    #    costs nothing but resolution on axes that are usually over-resolved.
    lever1 = ("  1. A coarser heading bin, velocity cell or omega cell. Those "
              "three multiply out to the column, the column sets the smallest "
              "tile, and halving it doubles the tile you can hold at the same "
              "step length. This is almost always the right fix.")
    ic = offer.get("in_core")
    if ic:
        lever1 += (f" Coarsening everything {ic['scale']:.2f}x -- "
                   f"{ic['xy_cm']:.1f} cm, {ic['heading_deg']:.1f} deg, "
                   f"{ic['v_cm_s']:.0f} cm/s, {ic['w_rad_s']:.2f} rad/s -- "
                   f"drops it to {ic['cells']:,} cells, which holds whole and "
                   f"skips tiling altogether.")
    print(c(lever1, "1;33"))

    # 2. Free, and costs only convergence rate.
    if ooc.get("warm_start_would_help"):
        print(c("  2. Set warm_start_tiled to false: it drops the resident "
                "cost from 7 bytes a cell to 4, which is enough on its own "
                "here. It costs convergence rate, not accuracy -- the "
                "self-test measured 144 rounds warm against 250 cold.", "1;33"))
    else:
        print(c("  2. warm_start_tiled is already off, or would not be enough "
                "on its own.", "2"))

    # 3. Last, because it is the only one that makes the answer worse.
    if "suggest_tau_max" in ooc:
        bits = []
        for pre, lead in (("suggest", "fits from"),
                          ("comfortable", "worth starting at")):
            if pre + "_tau_max" not in ooc:
                continue
            bits.append(f"{lead} {ooc[pre+'_tau_max']} s "
                        f"({ooc[pre+'_amplification']:.1f}x loaded per updated, "
                        f"about {hms(ooc[pre+'_total_s'])} to run, "
                        f"+{ooc[pre+'_value_cost']:.0f}% on mean value)")
        print(c("  3. A shorter 'longest step': " + "; ".join(bits) + ".",
                "1;33"))
        print(c("     This is the only lever that makes the answers worse, "
                "and it is a steep trade: 0.2 s costs about +6% on mean "
                "value, 0.1 s about +21%, 0.06 s about +56%.", "33"))
    else:
        print(c("  3. Even the shortest step will not fit, so the step is not "
                "the problem -- the column is. See lever 1.", "1;33"))


def _stream_solve(cfgpath: str, run_dir: str) -> None:
    """Run the solver, turning its PROGRESS lines into a progress bar."""
    logpath = os.path.join(run_dir, "solve.log")
    bar = Bar("solving")
    n_targets = 1
    done_targets = 0
    tiled = False
    honing = {}
    started = time.time()
    solve_started = [None]      # when the first target actually began
    esc_share = [0.0]           # fraction of a target's slot the escape owns

    def show(frac: float, note: str, eta_s: float | None = None) -> None:
        """Draw the bar, with the solver's own estimate of the time left.

        `eta_s` now covers the whole run -- the sweeps left in this target,
        the escape pass after them, and every target still to come -- priced
        from sweeps this run has actually timed. That is better than anything
        this end can work out, so it is used when it is there.

        The fallback, for a solver too old to send one, extrapolates from the
        rate the whole run has achieved since the first target began. Setup
        and occupancy are excluded from that: they happen once and would
        flatter the rest.
        """
        frac = min(1.0, max(0.0, frac))
        if solve_started[0] is None:
            solve_started[0] = time.time()
        el = time.time() - solve_started[0]
        if eta_s is not None and el > 5.0:
            eta = f"  left {hms(eta_s)}"
        elif frac > 0.01 and el > 5.0:
            eta = f"  left {hms(el * (1.0 - frac) / frac)}"
        else:
            eta = ""
        bar.update(frac, note + eta)

    print()

    proc = subprocess.Popen(
        [julia_exe(), f"--project={SOLVER_PROJ}", "--threads=auto", SOLVER,
         cfgpath, "solve"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    with open(logpath, "w", encoding="utf-8") as log:
        for line in proc.stdout:
            log.write(line)
            if not line.startswith("PROGRESS "):
                continue
            try:
                ev = json.loads(line[9:])
            except json.JSONDecodeError:
                continue
            ph = ev.get("phase")
            if ph == "setup":
                n_targets = max(1, ev.get("n_targets", 1))
                # How a target's slot on the bar divides between the two
                # passes. Without this the bar reaches the end of the slot
                # when the value iteration does and then sits there for the
                # whole escape pass -- which on the first H200 job was twelve
                # minutes of a bar that looked hung and an ETA of zero.
                it = max(1, ev.get("iterations", 1))
                esc = max(0, ev.get("escape_iterations", 0))
                esc_share[0] = esc / (it + esc)
                print(f"  {ev['cells']:,} cells, {ev['controls']} controls, "
                      f"{human(ev['bytes_per_target'])} per table")
            elif ph == "occupancy":
                print(f"  obstacles block {ev['blocked_frac']*100:.1f}% of the "
                      f"xy plane")
            elif ph == "decompose":
                # Too big to hold whole, so it is being solved a tile at a
                # time. An "iteration" below is then a round over every tile.
                tiled = True
                tx, ty = ev["tile_cells"]
                hx, hy = ev["halo_cells"]
                print(f"  tiled: {ev['tiles']} tiles of {tx}x{ty} cells, "
                      f"halo {hx}x{hy} ({ev['reach_cells']:.0f} cell reach), "
                      f"{ev['amplification']:.1f}x loaded per updated")
                print(f"  scratch file {human(ev['store_bytes'])}, "
                      f"{human(ev['resident_bytes'])} resident on the GPU")
            elif ph == "backend":
                print(f"  backend: {c(ev['backend'].upper(), '1;32')}")
                print()
            elif ph == "solve":
                within = (ev["iter"] / max(1, ev["iters"])) * (1 - esc_share[0])
                frac = (done_targets + within) / n_targets
                unit = "round" if tiled else "it"
                show(frac, f"{ev['target_name']}  {unit} "
                           f"{ev['iter']}/{ev['iters']}", ev.get("eta_s"))
            elif ph == "tile":
                # A round over a full-scale grid can be tens of minutes, so
                # the bar has to move inside one or it looks hung. The tile
                # index is a genuine fraction of the round, so this is real
                # progress rather than a spinner.
                rounds = max(1, ev.get("rounds", 1))
                within = ((ev["round"] - 1 + ev["tile"] / max(1, ev["tiles"]))
                          / rounds) * (1 - esc_share[0])
                show((done_targets + within) / n_targets,
                     f"{ev['target_name']}  round {ev['round']}/{rounds}  "
                     f"tile {ev['tile']}/{ev['tiles']}")
            elif ph == "escape":
                # Filling in the cells the solve could not reach with the time
                # to get out of them. It owns the last `esc_share` of this
                # target's slot, so the bar keeps moving through it rather
                # than sitting at the end of the slot looking hung.
                within = (1 - esc_share[0]) + esc_share[0] * (
                    ev["iter"] / max(1, ev["iters"]))
                show((done_targets + within) / n_targets,
                     f"{ev['target_name']}  escape "
                     f"{ev['iter']}/{ev['iters']}", ev.get("eta_s"))
            elif ph == "model":
                # Held rather than printed: the bar owns the line until it is
                # done with it. The honing gains are the one thing on the card
                # the user may have chosen by hand (step `h`), so the run
                # should say what actually landed there.
                honing.update(ev)
            elif ph == "encode":
                show((done_targets + 1) / n_targets,
                     f"{ev['target_name']}  writing")
            elif ph == "target_done":
                done_targets += 1
                esc = ev.get("escape_of_unreached_frac")
                extra = ("" if not esc else
                         f", {esc*100:.0f}% of the rest can escape")
                show(done_targets / n_targets,
                           f"{ev['target_name']} done "
                           f"({ev['reached_frac']*100:.0f}% reachable{extra})")
            elif ph == "done":
                bar.done(f"{ev['targets']} table(s) in {hms(time.time()-started)}")
        err = proc.stderr.read()
        log.write(err)
    proc.wait()

    if proc.returncode != 0:
        print(c(f"\n  solver failed (exit {proc.returncode}); see {logpath}", "31"))
        print(err.strip()[:1500])
        return
    print(c(f"  tables written to {run_dir}", "32"))
    if honing:
        print(c("  honing PID: omega %.3f rad/s at %.0f%% of the traction knee "
                "(bound by %s)"
                % (honing["honing_omega"], 100 * honing["honing_budget_frac"],
                   honing["honing_bound_by"]), "2"))

    print()
    print(c("  verifying against the format spec...", "2"))
    v = subprocess.run([sys.executable, os.path.join(HERE, "verify_tables.py"),
                        run_dir], capture_output=True, text=True)
    sys.stdout.write(v.stdout)
    if v.returncode != 0:
        print(c("  verification FAILED -- do not write this to the card", "31"))


# --------------------------------------------------------------------------
# Honing PID (optional)
# --------------------------------------------------------------------------

def traction_knee(ws: Workspace) -> float | None:
    """The command at which the tyres stop returning what they are asked for.

    Read from the fit rather than from a solved card, so the step works with
    nothing but step 1 done. `None` means the file did not say, and then the
    aggressiveness is shown as a bare fraction -- the solver reads the knee
    itself, so the derivation is unaffected either way.
    """
    if not ws.regression:
        return None
    try:
        with open(ws.regression, "rb") as fh:
            k = float(tomllib.load(fh)["mecanum_basis"]["traction_knee"])
    except (OSError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError):
        return None
    return k if k > 0 else None


def honing_config(ws: Workspace, d: dict, base: dict | None = None) -> dict:
    """The smallest config that derives the gains: the fit, and the settings.

    The block depends on the regression and on these settings and on nothing
    else -- not the grid, not the field, not the targets -- which is what lets
    this step answer in a moment, and run before there is a field file at all.
    """
    cfg = dict(base) if base else {"regression": ws.regression, "zero_c": True}
    cfg.update({k: d.get(k, DEFAULTS[k]) for k in HONING_KEYS})
    return cfg


def derive_honing(cfg: dict, run_dir: str | None = None) -> dict | None:
    """Ask the solver for the gains. With `run_dir`, its MODEL.JSON is rebuilt.

    The derivation stays in the solver rather than being repeated here: two
    implementations of one closed form drift apart, and the gains the card
    gets have to be the gains the wizard showed.
    """
    fd, path = tempfile.mkstemp(prefix="honing_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
        cmd = [julia_exe(), f"--project={SOLVER_PROJ}", SOLVER, path, "honing"]
        if run_dir:
            cmd.append(run_dir)
        r = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    for line in r.stdout.splitlines():
        if line.startswith("HONING "):
            return json.loads(line[7:])
    print(c("  deriving the gains failed:", "31"))
    print((r.stderr or r.stdout).strip()[:1500])
    return None


def show_honing(h: dict) -> None:
    """What the settings just bought, in the same terms the bench reports."""
    lim = h["omega_limits"]
    hc = h["config"]
    print()
    print(f"  bandwidth     omega = {c('%.4f' % h['omega'], '1;36')} rad/s"
          f"   (bound by {lim['bound_by']}: saturation "
          f"{lim['saturation']:.4f}, loop {lim['loop_rate']:.4f})")
    print(f"  command       budget {hc['budget']:.4f} of full stick,"
          f" traction gain {h['traction_gain']['value']:.4f}")
    print(f"  third poles   {[round(v, 3) for v in h['poles']['third']]}"
          f"   kd diagonal {[round(v, 4) for v in h['poles']['kd_diag']]}")
    print(f"  anti-windup   "
          f"{[round(v, 4) for v in h['integral_limit']['value']]}"
          f"   (cm*s, cm*s, rad*s)")
    print(f"  fine band     {hc['fine_cm']:.2f} cm / {hc['fine_rad']:.3f} rad"
          f"   handoff {h['handoff']['cm']:.1f} cm /"
          f" {h['handoff']['rad']:.2f} rad")
    # Past the saturation bound the proportional term leaves the budget while
    # still inside the fine band -- the one guarantee the block makes -- and
    # verify_tables rejects a card for it. Only a bandwidth scale above 1 can
    # get here, and the fix is nearly always the other knob: aggressiveness
    # raises the bound rather than stepping over it.
    if h["omega"] > lim["saturation"] * (1 + 1e-9):
        print()
        print(c("  omega is past the saturation bound: the command would",
                "31"))
        print(c("  saturate close in, which is exactly what the fine band",
                "31"))
        print(c("  promises it will not, and verify_tables refuses a card",
                "31"))
        print(c("  whose gains do that. Lower the bandwidth scale, or raise",
                "31"))
        print(c("  the aggressiveness -- that lifts the bound with it.", "31"))
    if any(v < 0 for v in h["poles"]["kd_diag"]):
        print(c("  negative derivative gain: the controller would be "
                "cancelling the robot's own damping", "31"))


def simulate_honing(cfg: dict) -> bool | None:
    """Drive the full nonlinear model with these gains and print the verdict.

    The gains are designed against a linearisation, so the only honest answer
    to "is this too aggressive" is to run the real dynamics with them. Both
    ends of the dial fail here, for opposite reasons -- too hot overshoots,
    too gentle is still short when the time is up -- and the marks it scores
    against are the ones `bench/honing.jl` gates a card with.

    Returns the verdict, or `None` if the simulation could not run. It is
    advice and not a veto: nothing on the card records it, and a robot whose
    real behaviour disagrees with the model is the user's call to make, not
    the wizard's.
    """
    fd, path = tempfile.mkstemp(prefix="honing_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
        print(c("  simulating the approach...", "2"))
        r = subprocess.run([julia_exe(), f"--project={SOLVER_PROJ}",
                            HONING_BENCH, path],
                           capture_output=True, text=True)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    # The report is the product here, verdict line and all: it marks its own
    # failures, against the same budgets that gate a card. Only its "model:"
    # line is dropped -- a workspace has one regression, and this step is
    # already about it.
    body = "\n".join(ln for ln in r.stdout.splitlines()
                     if not ln.startswith("model: ")).strip()
    print(body)
    if not body:
        print(c("  the simulation could not run:", "31"))
        print((r.stderr or "").strip()[:1500])
        return None
    return r.returncode == 0


def step_honing(ws: Workspace) -> None:
    """
    Optional: how hard the final approach pushes, and what that does to it.

    Deliberately not one of the numbered steps -- the defaults are what
    section 8.8 of TABLE_FORMAT.md was designed and measured against, and a
    run is perfectly valid without ever opening this.

    There is still one judgement in it that the fit cannot make: how much of
    the traction knee the robot may spend on the last few centimetres. Gentle
    is not automatically safe -- too little authority and the loop is still
    short of the point when the match has moved on -- so the step derives the
    gains and then offers to drive the *nonlinear* model with them, which is
    the only place either mistake actually shows up.

    Nothing downstream depends on it. The settings ride along with the rest of
    the solver answers into the next solve, and a run that is already solved
    can have its MODEL.JSON rebuilt in place, since the gains never depended
    on the tables.
    """
    rule("Honing PID for the final approach  (optional)")
    if not ws.regression:
        print(c("  No regression yet -- run step 1 first.", "33")); return
    if not julia_exe():
        print(c("  julia is not on PATH.", "31")); return

    print("  The tables stop at a handoff region, roughly 15 cm and 0.25 rad")
    print("  out; a PID owns the rest and settles on the point. Its gains are")
    print("  derived from the same fit the tables are, so there is nothing to")
    print("  tune here in the usual sense and nothing to twiddle on the field.")
    print("  What is a choice is how much command it is allowed to spend.")

    st = load_settings()
    d = dict(DEFAULTS)
    d.update({k: v for k, v in st.get("solver", {}).items()
              if k in HONING_KEYS})
    knee = traction_knee(ws)

    # One attempt: ask, derive, and see it driven. Declining to keep them
    # offers another go rather than dropping the answers on the floor --
    # finding the setting is a loop, and the simulation is what closes it.
    while True:
        rule("Aggressiveness")
        print("  A fraction of the traction knee -- the command where the tyres")
        print("  stop returning what they are asked for. Half the knee is the")
        print("  default: the approach is gentle there, and the linear design the")
        print("  gains come from is honest about itself.")
        print()
        print("  It is the whole dial. A bigger share is more command and more")
        print("  bandwidth together, because what limits the gains is the budget.")
        print(c("    0.25  gentle, and slow enough that a fit error leaves it "
                "short", "2"))
        print(c("    0.50  the default; settles in about 2 s from the corner", "2"))
        print(c("    0.90  quick, near the knee, with more overshoot to absorb",
                "2"))
        while True:
            f = ask_float("  aggressiveness (fraction of the knee)",
                          d["honing_budget_frac"])
            if f <= 0:
                print(c("    must be positive", "31")); continue
            # The octahedron is a hard limit on the command, so gains sized
            # against one outside it are designed for authority the wheels clip
            # away before the robot ever sees it.
            if knee and f * knee > 1.0:
                print(c("    that is a command of %.2f at a knee of %.2f, outside "
                        "the octahedron" % (f * knee, knee), "31")); continue
            break
        d["honing_budget_frac"] = f
        if knee:
            print(c("    -> command budget %.3f of full stick (knee %.2f)"
                    % (f * knee, knee), "2"))

        if confirm("Change the rest of the honing settings?"):
            rule("Loop")
            print("  The rate the robot will actually run this loop at. It caps")
            print("  the bandwidth at a tenth of itself: a sampled loop cannot")
            print("  track a pole much faster than that.")
            d["honing_loop_hz"] = ask_float("  loop rate, Hz", d["honing_loop_hz"])
            print()
            print("  The fine band -- the error inside which the command is")
            print("  guaranteed not to saturate. The bandwidth is sized against")
            print("  it, so a wider band buys gentler gains.")
            d["honing_fine_cm"] = ask_float("  fine band, cm", d["honing_fine_cm"])
            d["honing_fine_rad"] = ask_float("  fine band, rad",
                                             d["honing_fine_rad"])
            print()
            print("  Where the tables hand over. Recorded on the card for the")
            print("  robot side to read; it does not move the gains, which are")
            print("  sized against the fine band above.")
            d["honing_handoff_cm"] = ask_float("  handoff, cm",
                                               d["honing_handoff_cm"])
            d["honing_handoff_rad"] = ask_float("  handoff, rad",
                                                d["honing_handoff_rad"])
            print()
            print("  The share of the budget the integrator may claim before it")
            print("  is clamped. The clamp itself is not optional: without it the")
            print("  integrator winds the command out of the octahedron and the")
            print("  loop never settles.")
            d["honing_integral_share"] = ask_float("  integral share (0-1)",
                                                   d["honing_integral_share"])
            print()
            print("  A last multiplier on the bandwidth, for detuning. Below 1 is")
            print("  slower and gentler than the design would allow.")
            print(c("  Above 1 breaks the fine-band guarantee and the card fails "
                    "verification;", "33"))
            print(c("  to go faster, raise the aggressiveness instead.", "33"))
            d["honing_bandwidth_scale"] = ask_float("  bandwidth scale",
                                                    d["honing_bandwidth_scale"])

        cfg = honing_config(ws, d)
        h = derive_honing(cfg)
        if h is None:
            return
        show_honing(h)

        verdict = None
        print()
        if confirm("Drive the nonlinear model with these gains?", default_yes=True):
            print()
            verdict = simulate_honing(cfg)

        print()
        if verdict is False:
            # Keeping them anyway is allowed -- the simulation is the model's
            # opinion of itself, and the robot is the one that settles the
            # argument -- but it should not be the answer Enter gives.
            print(c("  A case above missed its budget: with these settings "
                    "the model does not", "33"))
            print(c("  land the robot where it says it will. Enter takes "
                    "another go at it.", "33"))
        if confirm("Keep these settings for the next solve?",
                   default_yes=verdict is not False):
            break
        if not confirm("Try a different setting?",
                       default_yes=verdict is False):
            print(c("  left as they were", "2"))
            return

    st = load_settings()
    solver = dict(st.get("solver", {}))
    solver.update({k: d[k] for k in HONING_KEYS})
    st["solver"] = solver
    save_settings(st)
    print(c("  saved -- the next solve writes them into MODEL.JSON", "32"))

    # The gains do not depend on the tables, so a solved run does not have to
    # be solved again to carry new ones. Offering the rebuild is the whole
    # reason this is safe to fiddle with: the alternative is waiting out a run
    # for a number that takes a millisecond.
    run = ws.latest_run()
    if not run or not os.path.exists(os.path.join(run, "MODEL.JSON")):
        return
    print()
    print(f"  The last run is already solved: {c(os.path.basename(run), '36')}")
    print("  Its gains can be rebuilt in place -- they never depended on the")
    print("  tables, only on the fit both were made from.")
    if not confirm("Rebuild that run's MODEL.JSON with these settings?",
                   default_yes=True):
        return
    runcfgpath = os.path.join(run, "config.json")
    try:
        with open(runcfgpath, encoding="utf-8") as fh:
            base = json.load(fh)
    except (OSError, json.JSONDecodeError):
        # A run that arrived without its config: rebuild from the fit this
        # workspace holds. The solver refuses the write unless that is the fit
        # the run was solved from, so guessing here cannot mis-pair them.
        base = None
        print(c("  no config.json in the run; using the workspace regression",
                "2"))
    newcfg = honing_config(ws, d, base)
    if derive_honing(newcfg, run) is None:
        return
    if base is not None:
        with open(runcfgpath, "w", encoding="utf-8") as fh:
            json.dump(newcfg, fh, indent=2)
    print(c("  rebuilt " + os.path.join(run, "MODEL.JSON"), "32"))
    if ws.card_record(run):
        print(c("  The card written from this run still has the old gains -- "
                "write it again (step 4).", "33"))


# --------------------------------------------------------------------------
# Step 4 -- SD card
# --------------------------------------------------------------------------

def step_card(ws: Workspace) -> None:
    rule("4. Write the SD card")
    run = ws.latest_run()
    if not run:
        print(c("  No solved tables yet -- run step 3 first.", "33")); return
    print(f"  source: {c(run, '36')}")
    files = sdcard.card_payload(run)
    payload = sum(os.path.getsize(s) for s, _ in files)
    print(f"  payload: {human(payload)} in {len(files)} file(s)")
    print()

    try:
        vols = sdcard.list_volumes()
    except RuntimeError as e:
        print(c(f"  {e}", "31")); return

    assessed = [sdcard.assess(v, ws.root) for v in vols]
    safe = [a for a in assessed if a["safe"]]

    print("  Drives:")
    for a in assessed:
        mark = c(" OK ", "1;32") if a["safe"] else c("  X ", "1;31")
        print(f"  {mark} {sdcard.describe(a)}")
        for r in a["reasons"]:
            print(f"         {c('- ' + r, '31')}")
    print()

    if not safe:
        print(c("  No drive is eligible. Peregrine will only write to a", "33"))
        print(c("  removable FAT32 volume that is not a system or boot disk.", "33"))
        return

    letters = [a["Letter"].upper() for a in safe]
    print(f"  Eligible: {', '.join(l + ':' for l in letters)}")
    choice = ask("Which drive (letter, blank to cancel)", "").strip().rstrip(":").upper()
    if not choice:
        return
    if choice not in letters:
        print(c(f"  {choice}: is not in the eligible list.", "31")); return

    try:
        nfiles, nbytes, top = sdcard.inventory(choice)
    except OSError as e:
        print(c(f"  cannot read {choice}: {e}", "31")); return

    target = next(a for a in safe if a["Letter"].upper() == choice)
    rule()
    print(c("  THIS WILL PERMANENTLY DELETE EVERYTHING ON THIS DRIVE", "1;31"))
    rule()
    print(f"  drive     {choice}:  {target.get('Label') or '(no label)'}  "
          f"{target.get('FriendlyName','')}")
    print(f"  size      {human(target.get('Size',0))} "
          f"({target.get('FileSystem')})")
    print(f"  contents  {nfiles:,} file(s), {human(nbytes)}")
    if top:
        print(f"  top level {', '.join(top[:10])}"
              + (" ..." if len(top) > 10 else ""))
    print(f"  writing   {human(payload)} of tables afterwards")
    if payload > target.get("Size", 0):
        print(c("  Payload is larger than the card. Aborting.", "31")); return
    rule()

    print(c(f"  Type the drive letter '{choice}' to confirm, anything else "
            f"cancels.", "1;33"))
    typed = ask("Confirm drive letter", "").strip().rstrip(":").upper()
    if typed != choice:
        print(c("  Cancelled -- nothing was changed.", "32")); return

    try:
        removed = sdcard.wipe(choice, typed, ws.root)
    except (RuntimeError, ValueError, OSError) as e:
        print(c(f"  wipe refused: {e}", "31")); return
    print(c(f"  removed {removed} top-level item(s)", "32"))

    bar = Bar("copying")

    def prog(i, n, done, total):
        bar.update(done / max(1, total), f"{i}/{n} files")

    try:
        nf, nb = sdcard.copy_image(run, choice, prog)
    except OSError as e:
        print(c(f"\n  copy failed: {e}", "31")); return
    bar.done(f"{nf} files, {human(nb)}")

    print()
    print(c("  verifying the card...", "2"))
    v = subprocess.run([sys.executable, os.path.join(HERE, "verify_tables.py"),
                        f"{choice}:\\"], capture_output=True, text=True)
    sys.stdout.write(v.stdout)
    if v.returncode == 0:
        # Only record success once the card itself has been read back and
        # verified -- a copy that completed but failed verification is not a
        # finished step.
        ws.mark_card_written(run, choice, nf, nb)
        print(c(f"\n  Card {choice}: is ready.", "1;32"))
    else:
        print(c("\n  Card verification FAILED -- step 4 left unfinished.", "1;31"))


# --------------------------------------------------------------------------
# Menu
# --------------------------------------------------------------------------

def pick_workspace(settings: dict) -> Workspace:
    cur = settings.get("workspace")
    rule("Workspace")
    print("  Where should calibration data, regressions and value tables live?")
    print("  Tables are large -- pick a drive with room.")
    if cur:
        print(f"  current: {c(cur, '36')}")
    print()
    for d in ("D:", "E:"):
        if os.path.isdir(d + "\\"):
            try:
                free = shutil.disk_usage(d + "\\").free
                print(f"    {d}  {human(free)} free")
            except OSError:
                pass
    print()
    while True:
        p = ask("Workspace directory", cur or os.path.join(
            os.environ.get("LOCALAPPDATA", HERE), "Peregrine", "data"))
        try:
            ws = Workspace(p)
        except OSError as e:
            print(c(f"  cannot use that path: {e}", "31")); continue
        settings["workspace"] = ws.root
        save_settings(settings)
        print(c(f"  using {ws.root}  ({human(ws.free())} free)", "32"))
        return ws


def status(ws: Workspace) -> None:
    def mark(ok):
        return c("[done]", "32") if ok else c("[    ]", "2")
    run = ws.latest_run()
    card = ws.card_record(run)
    print()
    print(f"  workspace  {c(ws.root, '36')}   {human(ws.free())} free")
    print()
    pods = os.path.join(ws.calib, "pod_offsets.json")
    pod_note = c("   (optional, do first)", "2")
    if os.path.exists(pods):
        try:
            with open(pods, encoding="utf-8") as fh:
                pr = json.load(fh)
            pod_note = ("   (centred, %.2f cm)" % pr["delta_magnitude"]
                        if pr.get("centred")
                        else c("   (%.2f cm off -- fix it)"
                               % pr["delta_magnitude"], "33"))
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    print(f"  {mark(os.path.exists(pods))}  0. Pod offsets from a rotation run"
          + pod_note)
    print(f"  {mark(bool(ws.regression))}  1. Calibration -> regression")
    print(f"  {mark(bool(ws.field_file and ws.targets_file))}  2. Field and targets")
    print(f"  {mark(bool(run))}  3. Solve value tables"
          + (f"   ({os.path.basename(run)})" if run else ""))
    detail = ""
    if card:
        when = card.get("written_utc", "")[:16].replace("T", " ")
        detail = f"   ({card['drive']}: {when}, {human(card.get('bytes', 0))})"
    print(f"  {mark(bool(card))}  4. Write the SD card{detail}")
    print()
    print(c("  d. diagnose the regression fit (optional)", "2"))
    print(c("  h. tune the honing PID for the final approach (optional)", "2"))
    print()
    print("  w. change workspace     q. quit")


def main() -> int:
    settings = load_settings()
    print()
    print(c("  PEREGRINE  -- desktop calibration and planning pipeline", "1;36"))

    ws_path = settings.get("workspace")
    ws = Workspace(ws_path) if ws_path and os.path.isdir(
        os.path.dirname(ws_path) or ws_path) else pick_workspace(settings)

    steps = {"0": step_pods, "1": step_calibration, "2": step_field,
             "3": step_solve, "4": step_card, "d": step_diagnose,
             "h": step_honing}
    while True:
        try:
            status(ws)
            ch = ask("Choose", "").lower()
            if ch in ("q", "quit", "exit"):
                print("  bye\n")
                return 0
            if ch == "w":
                ws = pick_workspace(settings)
                continue
            fn = steps.get(ch)
            if not fn:
                continue
            fn(ws)
        except EOFError:
            print("\n  stdin closed, exiting\n")
            return 0
        except KeyboardInterrupt:
            print(c("\n  interrupted -- nothing destructive was left half done", "33"))
        except Exception as e:  # keep the wizard alive on any step failure
            print(c(f"\n  step failed: {type(e).__name__}: {e}", "31"))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n")
        sys.exit(130)
