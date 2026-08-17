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
import shutil
import subprocess
import sys
import time
from datetime import datetime

import sdcard

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FITTER = os.path.join(REPO, "calibration", "fit_drivetrain.py")
SOLVER = os.path.join(REPO, "solver", "solve.jl")
SOLVER_PROJ = os.path.join(REPO, "solver")
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


def confirm(prompt: str) -> bool:
    return ask(f"{prompt} (y/N)", "n").lower().startswith("y")


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

    def update(self, frac: float, note: str = "") -> None:
        frac = min(1.0, max(0.0, frac))
        now = time.time()
        if frac < 1.0 and now - self.last < 0.08:
            return
        self.last = now
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
        for d in (self.calib, self.field, self.runs):
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
    extra: list[str] = []
    if not confirm("Include the omega^2 centripetal term?"):
        extra.append("--no-omega-sq")
    if not confirm("Include the constant offset term?"):
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

def julia_exe() -> str | None:
    return shutil.which("julia")


def gather_solver_config(ws: Workspace, run_dir: str) -> dict:
    print()
    print("  Grid resolution drives everything: table size grows as the product")
    print("  of all six axes, and solve time with it. Start coarse.")
    print()
    n = [
        ask_int("  x samples", 41, 2, 4096),
        ask_int("  y samples", 41, 2, 4096),
        ask_int("  heading samples", 16, 2, 512),
        ask_int("  vx samples", 11, 2, 512),
        ask_int("  vy samples", 11, 2, 512),
        ask_int("  omega samples", 11, 2, 512),
    ]
    print()
    vmax = ask_float("  max |velocity| cm/s", 150.0)
    wmax = ask_float("  max |omega| rad/s", 10.0)
    print()
    print("  Element type: u8 (1 B, 25 ms steps to 6.3 s), u16 (2 B, 1 ms to")
    print("  65.5 s, recommended), f16 (2 B, ~3 digits), f32 (4 B).")
    dtype = ask("  dtype", "u16")
    while dtype not in ("u8", "u16", "f16", "f32"):
        dtype = ask("  dtype (u8/u16/f16/f32)", "u16")
    print()
    iters = ask_int("  max iterations per target", 400, 1, 100000)
    dt = ask_float("  integration step dt (s)", 0.05)
    level = ask_int("  control refinement level (1 = bang-bang, 7 controls)", 1, 1, 4)

    return {
        "regression": ws.regression,
        "field": ws.field_file,
        "targets": ws.targets_file,
        "out_dir": run_dir,
        "grid": {"n": n, "vmax": vmax, "wmax": wmax},
        "dtype": dtype,
        "iterations": iters,
        "dt": dt,
        "control_level": level,
        "zero_c": True,
        "backend": "auto",
    }


def run_plan(cfgpath: str) -> dict | None:
    r = subprocess.run([julia_exe(), f"--project={SOLVER_PROJ}", SOLVER,
                        cfgpath, "plan"], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("PLAN "):
            return json.loads(line[5:])
    print(c("  plan failed:", "31"))
    print((r.stderr or r.stdout).strip()[:2000])
    return None


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

    while True:
        cfg = gather_solver_config(ws, run_dir)
        cfgpath = os.path.join(run_dir, "config.json")
        with open(cfgpath, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)

        p = run_plan(cfgpath)
        if p is None:
            return

        rule()
        print(f"  cells per table     {p['cells']:,}")
        print(f"  bytes per table     {human(p['bytes_per_target'])}")
        print(f"  targets             {p['n_targets']}")
        print(f"  TOTAL on card       {c(human(p['bytes_total']), '1;33')}"
              f"   ({p['chunks_per_target']} chunk(s) each)")
        free = ws.free()
        print(f"  workspace free      {human(free)}"
              + ("" if p["bytes_total"] < free else c("   <-- NOT ENOUGH", "31")))
        if p["gpu_available"]:
            print(f"  GPU                 needs {p['gpu_need_gb']:.2f} GB of "
                  f"{p['gpu_free_gb']:.2f} GB free"
                  + ("" if p["gpu_fits"] else c("   <-- WILL NOT FIT", "31")))
        else:
            print(c("  GPU                 not available; will run on CPU "
                    "(much slower)", "33"))
        rule()

        if p["bytes_total"] >= free:
            print(c("  Not enough space in the workspace. Reduce the grid.", "31"))
        elif p["gpu_available"] and not p["gpu_fits"]:
            print(c("  Grid will not fit in VRAM. Reduce the grid.", "31"))
        elif confirm("Start solving with these settings?"):
            break
        if not confirm("Adjust settings and try again?"):
            return

    _stream_solve(cfgpath, run_dir)


def _stream_solve(cfgpath: str, run_dir: str) -> None:
    """Run the solver, turning its PROGRESS lines into a progress bar."""
    logpath = os.path.join(run_dir, "solve.log")
    bar = Bar("solving")
    n_targets = 1
    done_targets = 0
    started = time.time()
    print()

    proc = subprocess.Popen(
        [julia_exe(), f"--project={SOLVER_PROJ}", SOLVER, cfgpath, "solve"],
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
                print(f"  {ev['cells']:,} cells, {ev['controls']} controls, "
                      f"{human(ev['bytes_per_target'])} per table")
            elif ph == "occupancy":
                print(f"  obstacles block {ev['blocked_frac']*100:.1f}% of the "
                      f"xy plane")
            elif ph == "backend":
                print(f"  backend: {c(ev['backend'].upper(), '1;32')}")
                print()
            elif ph == "solve":
                frac = (done_targets + ev["iter"] / max(1, ev["iters"])) / n_targets
                bar.update(frac,
                           f"{ev['target_name']}  it {ev['iter']}/{ev['iters']}  "
                           f"eta {hms(ev['eta_s'] + 0)}")
            elif ph == "encode":
                bar.update((done_targets + 1) / n_targets,
                           f"{ev['target_name']}  writing "
                           f"({ev['reached_frac']*100:.0f}% reachable)")
            elif ph == "target_done":
                done_targets += 1
                bar.update(done_targets / n_targets, f"{ev['target_name']} done")
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

    print()
    print(c("  verifying against the format spec...", "2"))
    v = subprocess.run([sys.executable, os.path.join(HERE, "verify_tables.py"),
                        run_dir], capture_output=True, text=True)
    sys.stdout.write(v.stdout)
    if v.returncode != 0:
        print(c("  verification FAILED -- do not write this to the card", "31"))


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
    print("  w. change workspace     q. quit")


def main() -> int:
    settings = load_settings()
    print()
    print(c("  PEREGRINE  -- desktop calibration and planning pipeline", "1;36"))

    ws_path = settings.get("workspace")
    ws = Workspace(ws_path) if ws_path and os.path.isdir(
        os.path.dirname(ws_path) or ws_path) else pick_workspace(settings)

    steps = {"1": step_calibration, "2": step_field, "3": step_solve,
             "4": step_card}
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
