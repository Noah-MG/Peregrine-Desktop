#!/usr/bin/env python3
"""
Run a Peregrine solve on a rented GPU box over SSH.

    py -3.12 solver/cloud/peregrine_remote.py provision root@1.2.3.4
    py -3.12 solver/cloud/peregrine_remote.py run <job> --host root@1.2.3.4

`run` is the whole job: push the working tree and the inputs, start the solve
in a detached tmux session, follow its progress with the same bar the wizard
draws, pull the tables back, verify them, and tell you what the rental cost.

**Planning and renting are on different clocks.** A plan is settled over an
evening of re-planning at different resolutions; the box is created minutes
before it is needed and destroyed as soon as the tables land, because it
bills until it is destroyed. So the wizard's step 3 can freeze a plan into a
**job** -- a directory holding `job.json`, `config.json`, and copies of the
three inputs -- and `run <name> --host <box>` solves it later with no wizard
session and nothing rented in between. `jobs` lists them.

The inputs are copies, and the config names them relatively. That makes a job
a fact rather than a pointer: editing a target in the workspace after saving
cannot quietly change what a later run solves, and a job directory can be
moved or copied to another machine intact.

**`--host` is per invocation, and naming a different box drops what was
remembered about the last one.** `rented_since` is the clock every cost line
is figured from, and a destroyed box's clock carried onto a new one reports a
machine ninety seconds old as three days of billing. The `run` record goes
the same way -- it names a directory on a box that no longer exists. Use
`forget` when the next box comes up on the same address, which is the one
case this cannot detect.

Everything here is deliberately provider-agnostic -- it needs an Ubuntu box
with an NVIDIA GPU and an SSH key, and nothing else. It works on a
DigitalOcean GPU Droplet, a RunPod pod with SSH exposed, a Lambda instance,
or a machine under someone's desk.

Two properties matter more than the convenience:

  * **The solve survives a dropped connection.** It runs under tmux with its
    output on a file; this process only follows that file. Closing the laptop
    on a six-hour run loses the progress bar and nothing else -- `attach`
    picks it back up.

  * **It pushes the WORKING TREE, not a git checkout.** The desktop repo is
    normally ahead of origin, so cloning on the box would silently solve with
    different code than the one being tested.

Commands. Every one that talks to a box takes `--host USER@IP`:

    provision [host] [--driver] [--quick]  install Julia and check the GPU
    plan      <job|config.json>            remote `plan`, printed like the wizard's
    run       <job|config.json>            push, solve, stream home, verify
    attach                                 re-follow a solve already running
    status                                 what the box is and what is running
    pull                                   fetch the last solve's tables again
    cost                                   what the current rental has run up
    jobs                                   list the plans saved for later
    host      <host>                       remember a box between commands
    forget                                 drop the remembered box and its clock
    benchmark [--seconds N]                measure this box's cell and disk rates
    --self-test                            check this script without a box

`<job>` is a saved job's name, its directory, or any `config.json`.

`--rate <usd>` sets the price per hour the cost lines are figured at. Left
off, a job is priced at the card it was planned for and everything else at
the module default, which is an L40S.

`run` and `attach` fetch each table **as the box finishes it**, rather than
all of them at the end -- a target's chunks are final the moment
`target_done` is emitted, so the download of one overlaps the compute of the
next and costs no rental time. `--stream-to DIR` sends them somewhere other
than the config's `out_dir` (an SD card, when they will not fit on the local
disk); `--no-stream` restores the fetch-everything-at-the-end behaviour.
"""

import argparse
import json
import os
import posixpath
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "wizard"))

# The wizard's terminal helpers, so a remote solve looks exactly like a local
# one. peregrine.py guards its entry point, so importing it runs nothing.
from peregrine import Bar, c, hms, human  # noqa: E402

STATE = os.path.join(os.environ.get("LOCALAPPDATA", HERE), "Peregrine",
                     "remote.json")

# Rented boxes bill by the second and stop billing only when the machine is
# destroyed, so the elapsed time of a run is a price. Default is
# DigitalOcean's on-demand single L40S; override with `--rate`.
DEFAULT_USD_PER_HOUR = 1.57

# Pushed to the box; everything else in the tree is either derived, huge, or
# the desktop's own state.
EXCLUDE = (".git", "__pycache__", ".pytest_cache", "runs", ".vscode",
           "*.pyc", "*.SCRATCH", "*.BIN")

# The config keys that name a path on the desktop and must name one on the
# box instead. Split by what happens to them: inputs are uploaded, outputs
# are directories the box creates.
INPUT_KEYS = ("regression", "field", "targets")
OUTPUT_KEYS = ("out_dir", "scratch_dir")


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------

def load_state() -> dict:
    try:
        with open(STATE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(st: dict) -> None:
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, "w", encoding="utf-8") as fh:
        json.dump(st, fh, indent=2)


def resolve_host(args, st: dict) -> str:
    """The box this invocation talks to, and the state that belongs to it.

    `--host` wins over anything remembered, because the normal life of a
    rented box is hours: created, provisioned, solved on, destroyed. The next
    one has a different address and shares nothing with it.

    **A different host clears the remembered state.** That is not tidiness.
    `rented_since` is the clock every cost line is figured from, and carrying
    a destroyed box's clock onto a new one is how a machine created a minute
    ago gets billed as three days -- which is what a stale `remote.json` did
    here. The `run` record (what `attach` and `pull` resume from) is just as
    host-specific: a remote directory on a box that no longer exists. Neither
    has any meaning once the address changes, so neither survives it.
    """
    want = getattr(args, "host", None)
    h = want or st.get("host")
    if not h:
        die("no box to talk to. Give one:\n"
            "    py -3.12 solver/cloud/peregrine_remote.py <cmd> "
            "--host root@1.2.3.4\n"
            "or remember it for this box's lifetime:\n"
            "    py -3.12 solver/cloud/peregrine_remote.py host root@1.2.3.4")
    if st.get("host") != h:
        known = st.get("host")
        st.clear()
        st["host"] = h
        st["rented_since"] = time.time()
        save_state(st)
        if known:
            print(c(f"  new box {h} -- the clock and run state for {known} "
                    f"are dropped", "2"))
    return h


def die(msg: str) -> "NoReturn":  # noqa: F821
    print(c("  " + msg.replace("\n", "\n  "), "31"))
    sys.exit(1)


# --------------------------------------------------------------------------
# SSH
#
# Windows OpenSSH has no ControlMaster, so every call is its own connection.
# That rules out polling loops -- which is why the solve appends its own exit
# sentinel to the log rather than being asked whether it has finished.
# --------------------------------------------------------------------------

SSH_OPTS = ["-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=10"]


def ssh_cmd(host: str, remote: str) -> list:
    return ["ssh", *SSH_OPTS, host, remote]


def ssh(host: str, remote: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(ssh_cmd(host, remote), capture_output=True, text=True)
    if check and r.returncode != 0:
        die(f"ssh failed ({r.returncode}): {remote}\n{r.stderr.strip()[:800]}")
    return r


def ssh_popen(host: str, remote: str) -> subprocess.Popen:
    """Line-buffered text, for following a log."""
    return subprocess.Popen(ssh_cmd(host, remote), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, bufsize=1)


def ssh_popen_binary(host: str, remote: str) -> subprocess.Popen:
    """Raw bytes, for carrying a tar stream.

    Separate from `ssh_popen` on purpose: a tar decoded as UTF-8 is a
    corrupted tar, and the failure would land at the far end of an 86 GB
    download rather than at the start of it.
    """
    return subprocess.Popen(ssh_cmd(host, remote), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)


def agent_running() -> bool:
    """Is an ssh-agent reachable? `ssh-add -l` exits 2 when it is not."""
    try:
        return subprocess.run(["ssh-add", "-l"], capture_output=True,
                              timeout=10).returncode != 2
    except (OSError, subprocess.TimeoutExpired):
        return False


def encrypted_keys() -> list:
    """Private keys under ~/.ssh that are passphrase-protected.

    Parsed rather than probed: an OpenSSH-format key names its cipher in
    clear at the head of the blob, and "none" means unencrypted. Shelling out
    to `ssh-keygen -y` to find out risks a prompt, which is the very thing
    this whole check exists to avoid hanging on.
    """
    out = []
    d = os.path.expanduser("~/.ssh")
    for name in ("id_ed25519", "id_rsa", "id_ecdsa", "id_ed25519_sk"):
        path = os.path.join(d, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                if key_cipher(fh.read()) not in (None, "none"):
                    out.append(path)
        except OSError:
            continue
    return out


def key_cipher(body: str):
    """The cipher named in a private key, or None if it cannot be read.

    "none" means the key is unencrypted. Anything else means a passphrase.
    """
    import base64
    if "BEGIN OPENSSH PRIVATE KEY" not in body:
        # Older PEM keys carry a Proc-Type/DEK-Info header instead.
        return "pem-encrypted" if "ENCRYPTED" in body else None
    try:
        b64 = "".join(l for l in body.splitlines() if "-----" not in l)
        raw = base64.b64decode(b64)
    except ValueError:
        return None
    magic = b"openssh-key-v1\x00"
    if not raw.startswith(magic):
        return None
    p = len(magic)
    n = int.from_bytes(raw[p:p + 4], "big")
    if n <= 0 or p + 4 + n > len(raw):
        return None
    return raw[p + 4:p + 4 + n].decode("ascii", "replace")


def check_ssh(host: str) -> None:
    print(f"  reaching {c(host, '36')} ...", end="", flush=True)
    r = ssh(host, "echo ok", check=False)
    if r.returncode == 0 and r.stdout.strip() == "ok":
        print(c(" ok", "32"))
        return
    print()
    err = r.stderr.strip()

    # The common first-run failure, and the one whose default message is
    # actively misleading: the key is installed and correct, but it has a
    # passphrase and there is no agent holding it. Every call here runs
    # BatchMode, which forbids the prompt, so ssh reports the same
    # "Permission denied (publickey)" it would for a missing key.
    if "publickey" in err.lower():
        locked = encrypted_keys()
        if locked and not agent_running():
            die("your key has a passphrase and no ssh-agent is running, so "
                "the\nautomated calls here cannot unlock it. (Plain `ssh "
                f"{host}` works\nbecause it can prompt you; these calls run "
                "BatchMode and cannot.)\n\n"
                "Start the agent once, as Administrator:\n"
                "    Set-Service ssh-agent -StartupType Automatic\n"
                "    Start-Service ssh-agent\n\n"
                "Then, as yourself, load the key (it asks for the passphrase "
                "once):\n"
                f"    ssh-add {locked[0]}\n\n"
                "After that this works, and keeps working across reboots.")
        if locked:
            die("an agent is running but does not seem to hold your key.\n"
                f"Load it with:\n    ssh-add {locked[0]}\n\n{err[:400]}")

    die(f"cannot ssh to {host}.\n"
        f"{err[:600]}\n\n"
        "Check the box is up, and that your public key\n"
        f"({os.path.expanduser('~/.ssh/id_ed25519.pub')}) was added to it\n"
        "when it was created.")


def remote_env(host: str) -> dict:
    """What provision.sh learned about the box, or None if never provisioned."""
    r = ssh(host, "cat ~/peregrine/cloud-env.json 2>/dev/null || true",
            check=False)
    try:
        return json.loads(r.stdout)
    except (json.JSONDecodeError, TypeError):
        return {}


# --------------------------------------------------------------------------
# Uploading
# --------------------------------------------------------------------------

def push_tree(host: str) -> None:
    """Stream the working tree up as a tar. No rsync on Git Bash or Windows."""
    print("  pushing the working tree ...", end="", flush=True)
    excl = []
    for pat in EXCLUDE:
        excl += ["--exclude", pat]
    tar = subprocess.Popen(["tar", "-cz", "-C", REPO, *excl, "."],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    dest = subprocess.Popen(
        ssh_cmd(host, "mkdir -p ~/peregrine && tar -xz -C ~/peregrine"),
        stdin=tar.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True)
    tar.stdout.close()
    _, derr = dest.communicate()
    tar.wait()
    if dest.returncode != 0:
        print()
        die(f"pushing the tree failed: {derr.strip()[:600]}")
    print(c(" ok", "32"))


def push_inputs(host: str, cfg: dict, env: dict) -> dict:
    """Upload the files the config names, and return the remote config."""
    inputs = env.get("inputs_dir", "~/work/inputs")
    ssh(host, f"mkdir -p {shlex.quote(inputs)}")
    out = dict(cfg)
    for key in INPUT_KEYS:
        local = cfg.get(key)
        if not local:
            die(f"config has no '{key}'")
        if not os.path.isfile(local):
            die(f"config's '{key}' is not a file here: {local}")
        name = os.path.basename(local)
        print(f"  pushing {name} ({human(os.path.getsize(local))}) ...",
              end="", flush=True)
        r = subprocess.run(["scp", *SSH_OPTS, local, f"{host}:{inputs}/{name}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print()
            die(f"scp of {local} failed: {r.stderr.strip()[:600]}")
        print(c(" ok", "32"))
        out[key] = posixpath.join(inputs, name)
    return out


def remote_config(cfg: dict, env: dict, stamp: str) -> dict:
    """The desktop's config with every path it names moved onto the box.

    Also carries across what provision.sh measured. `disk_rate` in
    particular: leaving the desktop's 500 MB/s in place would have `plan`
    quote an out-of-core round on a rented NVMe as roughly twice what it is,
    and the whole point of the estimate is to decide how long to rent for.
    """
    runs = env.get("runs_dir", "~/work/runs")
    out = dict(cfg)
    out["out_dir"] = posixpath.join(runs, stamp)
    out["scratch_dir"] = env.get("scratch_dir", "~/work/scratch")
    if env.get("disk_rate"):
        out["disk_rate"] = int(env["disk_rate"])
    # Never carry a desktop-sized VRAM override onto a bigger card: it would
    # tile a grid that now fits whole.
    out.pop("vram_budget_bytes", None)
    return out


def write_remote_config(host: str, rcfg: dict, env: dict, stamp: str) -> str:
    runs = env.get("runs_dir", "~/work/runs")
    rdir = posixpath.join(runs, stamp)
    ssh(host, f"mkdir -p {shlex.quote(rdir)}")
    fd, tmp = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(rcfg, fh, indent=1)
        path = posixpath.join(rdir, "config.json")
        r = subprocess.run(["scp", *SSH_OPTS, tmp, f"{host}:{path}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            die(f"scp of the config failed: {r.stderr.strip()[:600]}")
    finally:
        os.unlink(tmp)
    return path


# --------------------------------------------------------------------------
# The progress stream
#
# Lifted from the wizard's _stream_solve, with one difference that matters:
# the source is `tail -F` on a file the box is still writing, so it never
# ends by itself. The remote command appends `PEREGRINE_EXIT <code>` when the
# solver returns, and that sentinel is what ends the follow -- no polling,
# which a connection-per-call SSH could not afford anyway.
# --------------------------------------------------------------------------

EXIT_RE = re.compile(r"^PEREGRINE_EXIT (\d+)")


class ProgressReader:
    """Turns the solver's PROGRESS lines into a bar. Returns an exit code."""

    def __init__(self, label: str = "solving"):
        self.bar = Bar(label)
        self.n_targets = 1
        self.done_targets = 0
        self.tiled = False
        self.started = time.time()
        self.solve_started = None
        self.exit_code = None
        # Kept so the run can report a measured cell rate afterwards.
        self.cells = 0
        self.sweeps_per_unit = 1
        self.units_seen = 0
        # The reader's own idea of how far along the run is. Deliberately not
        # read back off the bar: the bar drops redraws less than 80 ms apart,
        # so on a fast tile stream its `frac` lags the run by an arbitrary
        # amount. That is right for a terminal and wrong for anything that
        # wants to know where the solve actually is.
        self.frac = 0.0
        # Set by `follow` when a streamer is attached. Called the moment a
        # target's chunks are final -- `target_done` is emitted after
        # `write_table` has returned and hashed them.
        self.on_target_done = None
        # Called on the `setup` event, which is the first thing that knows how
        # big the output will be.
        self.on_setup = None

    def show(self, frac: float, note: str) -> None:
        frac = min(1.0, max(0.0, frac))
        self.frac = frac
        if self.solve_started is None:
            self.solve_started = time.time()
        el = time.time() - self.solve_started
        eta = (f"  left {hms(el * (1.0 - frac) / frac)}"
               if frac > 0.01 and el > 5.0 else "")
        self.bar.update(frac, note + eta)

    def feed(self, line: str) -> bool:
        """Consume one line. Returns False once the run has finished."""
        m = EXIT_RE.match(line)
        if m:
            self.exit_code = int(m.group(1))
            return False
        if not line.startswith("PROGRESS "):
            return True
        try:
            ev = json.loads(line[9:])
        except json.JSONDecodeError:
            return True
        ph = ev.get("phase")
        if ph == "setup":
            self.n_targets = max(1, ev.get("n_targets", 1))
            self.cells = ev.get("cells", 0)
            print(f"  {ev['cells']:,} cells, {ev['controls']} controls, "
                  f"{human(ev['bytes_per_target'])} per table")
            if self.on_setup is not None:
                self.on_setup(ev)
        elif ph == "occupancy":
            print(f"  obstacles block {ev['blocked_frac']*100:.1f}% of the "
                  f"xy plane")
        elif ph == "decompose":
            self.tiled = True
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
            self.units_seen = max(self.units_seen, ev["iter"])
            frac = ((self.done_targets + ev["iter"] / max(1, ev["iters"]))
                    / self.n_targets)
            unit = "round" if self.tiled else "it"
            self.show(frac, f"{ev['target_name']}  {unit} "
                            f"{ev['iter']}/{ev['iters']}")
        elif ph == "tile":
            rounds = max(1, ev.get("rounds", 1))
            within = ((ev["round"] - 1 + ev["tile"] / max(1, ev["tiles"]))
                      / rounds)
            self.show((self.done_targets + within) / self.n_targets,
                      f"{ev['target_name']}  round {ev['round']}/{rounds}  "
                      f"tile {ev['tile']}/{ev['tiles']}")
        elif ph == "escape":
            self.show((self.done_targets + 1) / self.n_targets,
                      f"{ev['target_name']}  escape "
                      f"{ev['iter']}/{ev['iters']}")
        elif ph == "encode":
            self.show((self.done_targets + 1) / self.n_targets,
                      f"{ev['target_name']}  writing")
        elif ph == "target_done":
            self.done_targets += 1
            esc = ev.get("escape_of_unreached_frac")
            extra = ("" if not esc else
                     f", {esc*100:.0f}% of the rest can escape")
            self.show(self.done_targets / self.n_targets,
                      f"{ev['target_name']} done "
                      f"({ev['reached_frac']*100:.0f}% reachable{extra})")
            if self.on_target_done is not None:
                self.on_target_done(ev)
        elif ph == "done":
            self.bar.done(f"{ev['targets']} table(s) in "
                          f"{hms(time.time() - self.started)}")
        return True


def copy_counted(src, dst, total: int, label: str = "downloading"):
    """Shovel bytes from one pipe to another, drawing a bar. Returns (bytes, s).

    The copy goes through this process rather than being piped shell to
    shell purely so it can be counted -- at any plausible link speed the
    extra copy is far below the network cost, and a table pull is long
    enough that a silent one looks hung.

    Binary throughout. Both ends are tar streams, and a tar that has been
    through a text decode is a corrupt tar.
    """
    bar = Bar(label)
    got = 0
    t0 = time.time()
    try:
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            dst.write(chunk)
            got += len(chunk)
            el = max(1e-6, time.time() - t0)
            bar.update(got / total if total else 0.0,
                       f"{human(got)}  {human(got / el)}/s")
    finally:
        dst.close()
    el = time.time() - t0
    bar.done(f"{human(got)} in {hms(el)}")
    return got, el


class TableStreamer:
    """Fetch each target's table while the box is still solving the next one.

    A run's tables are the slow part of getting the work home -- tens of
    gigabytes at production resolution, hundreds at full -- and downloading
    them after the solve means paying for the box while a home connection
    trickles. But a target's chunks are finished long before the run is: the
    solver emits `target_done` only after `write_table` has returned and
    hashed them, so from that moment they are immutable and safe to take.

    So each target is fetched as it lands, overlapping the download with the
    remaining targets' compute. On a three-target run that hides two thirds
    of the transfer behind work you are paying for anyway.

    One worker thread and a queue rather than a download per target in
    parallel: they would only contend for the same link, and a single stream
    keeps the ordering (and the reporting) obvious. Targets finish minutes
    apart and a queue depth above one is rare.
    """

    def __init__(self, host: str, remote_dir: str, dest: str):
        self.host = host
        self.remote_dir = remote_dir
        self.dest = dest
        self.q = queue.Queue()
        self.done = {}          # target index -> bytes fetched
        self.failed = {}        # target index -> why
        self.seconds = 0.0
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    def check_space(self, total_bytes: int) -> None:
        """Warn now if the destination cannot hold what the run will make.

        Called on the `setup` event, which is seconds into the solve and the
        first moment the size is known. A warning rather than a stop: the box
        is already working, and killing a run over a disk the user may be
        about to clear would be the more expensive mistake. What matters is
        that it is said at the start and not discovered at 90%.
        """
        if total_bytes <= 0:
            return
        try:
            free = shutil.disk_usage(self.dest).free
        except OSError:
            return
        if total_bytes > free:
            print(c(f"\n  WARNING: this run will produce {human(total_bytes)} "
                    f"but {self.dest}", "1;31"))
            print(c(f"  has only {human(free)} free. Point --stream-to at a "
                    f"bigger volume (an SD card,", "1;31"))
            print(c("  say) before the first table lands, or the fetch will "
                    "fail part way.", "1;31"))
        elif total_bytes > free * 0.8:
            print(c(f"  note: {human(total_bytes)} of tables into "
                    f"{human(free)} free -- tight", "33"))

    def enqueue(self, index: int, name: str, nbytes: int) -> None:
        self.q.put((index, name, nbytes))

    def finish(self) -> None:
        """Stop accepting work and wait for what is queued."""
        self.q.put(None)
        self.thread.join()

    def _work(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                return
            index, name, nbytes = item
            try:
                got, el = self._fetch(index)
                self.done[index] = got
                self.seconds += el
                # Printed rather than drawn as a bar: the solve's bar owns the
                # live line, and two of them would fight over it.
                print(f"\n  {c('fetched', '32')} {name} "
                      f"({human(got)} in {hms(el)}, "
                      f"{human(got / max(el, 1e-6))}/s) "
                      + c("-- while the box keeps solving", "2"))
            except Exception as e:                      # noqa: BLE001
                # Never let a transfer failure stop the solve being followed.
                # The final sweep re-fetches whatever is missing.
                self.failed[index] = str(e)[:200]
                print(c(f"\n  could not fetch target {index} yet "
                        f"({str(e)[:120]}); it will be collected at the end",
                        "33"))

    def _fetch(self, index: int):
        """Pull one target's chunks. `find -print0` keeps argv out of it.

        A full-resolution target is nearly two thousand chunk files, so a
        shell glob would build a 40 KB argv. Feeding the names to tar on
        stdin sidesteps the question entirely.
        """
        os.makedirs(os.path.join(self.dest, "TABLES"), exist_ok=True)
        pat = f"T{index:02d}C*.BIN"
        remote = (f"cd {shlex.quote(self.remote_dir)} && "
                  f"find TABLES -name {shlex.quote(pat)} -print0 | "
                  f"tar -c --null -T -")
        src = ssh_popen_binary(self.host, remote)
        dst = subprocess.Popen(["tar", "-x", "-C", self.dest],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
        got = 0
        t0 = time.time()
        try:
            while True:
                chunk = src.stdout.read(1 << 20)
                if not chunk:
                    break
                dst.stdin.write(chunk)
                got += len(chunk)
        finally:
            dst.stdin.close()
            dst.wait()
            src.wait()
        if dst.returncode != 0 or got == 0:
            raise RuntimeError(f"tar exited {dst.returncode} after {got} bytes")
        return got, time.time() - t0


def follow(host: str, logpath: str, label: str = "solving",
           streamer: "TableStreamer | None" = None) -> ProgressReader:
    """Follow a running solve's log until its exit sentinel arrives."""
    pr = ProgressReader(label)
    if streamer is not None:
        pr.on_target_done = lambda ev: streamer.enqueue(
            int(ev.get("target", 0)), ev.get("target_name", "?"),
            int(ev.get("bytes", 0)))
        pr.on_setup = lambda ev: streamer.check_space(
            int(ev.get("bytes_per_target", 0))
            * max(1, int(ev.get("n_targets", 1))))
    # -n +1 so re-attaching replays the run from the start and the bar lands
    # where the run actually is rather than at zero.
    cmd = f"tail -n +1 -F {shlex.quote(logpath)}"
    proc = ssh_popen(host, cmd)
    try:
        for line in proc.stdout:
            if not pr.feed(line.rstrip("\n")):
                break
    except KeyboardInterrupt:
        proc.terminate()
        print(c("\n  detached. The solve is STILL RUNNING on the box.", "33"))
        print("  follow it again with: "
              + c("peregrine_remote.py attach", "36"))
        raise
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    return pr


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_benchmark(args, st) -> int:
    """Measure the rented box, and remember what it measured.

    This is the command that replaces the guesswork. `plan` costs every run
    from `cell_rate` and `disk_rate`, whose defaults are measurements of one
    desktop card and one desktop SSD; on any other machine every wall clock
    and every dollar it quotes is an extrapolation. A couple of minutes here
    turns them into arithmetic.

    The answer is written into the wizard's settings under the card's name,
    so the next plan for that card uses it without being asked.
    """
    host = resolve_host(args, st)
    check_ssh(host)
    env = remote_env(host)
    if not env:
        die("this box has not been provisioned. Run:\n"
            "    py -3.12 solver/cloud/peregrine_remote.py provision")
    push_tree(host)
    julia = env.get("julia", "julia")
    print()
    remote = (f"cd ~/peregrine && PEREGRINE_SCRATCH={shlex.quote(env.get('scratch_dir', '~/work/scratch'))} "
              f"{shlex.quote(julia)} --project=solver -t auto "
              f"solver/cloud/benchmark.jl --seconds {int(args.seconds)}")
    proc = subprocess.Popen(ssh_cmd(host, remote), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    last = None
    for line in proc.stdout:
        sys.stdout.write(line)
        if line.startswith("BENCHMARK "):
            last = line[10:]
    proc.wait()
    if proc.returncode != 0 or not last:
        die(f"the benchmark failed (exit {proc.returncode})")
    try:
        res = json.loads(last)
    except json.JSONDecodeError:
        die("the benchmark produced no readable result")

    key = _gpu_key(res.get("gpu", ""))
    _remember_rates(key, res)
    print(c(f"\n  saved as '{key}' -- the wizard will plan with these now",
            "32"))
    _cost_line(st.get("rented_since"), job_rate(args, None))
    return 0


def _gpu_key(name: str) -> str:
    """The wizard's name for a card, from what the driver calls it."""
    n = name.lower()
    for key in ("l40s", "h100", "h200", "a100", "l4", "a10"):
        if key in n.replace(" ", ""):
            return key
    return "".join(ch for ch in n if ch.isalnum()) or "unknown"


def _remember_rates(key: str, res: dict) -> None:
    """Write the measured rates where the wizard's planner reads them."""
    path = os.path.join(os.environ.get("LOCALAPPDATA", HERE), "Peregrine",
                        "wizard.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            settings = json.load(fh)
    except (OSError, json.JSONDecodeError):
        settings = {}
    rates = settings.setdefault("gpu_rates", {})
    rates[key] = {"cell_rate": res.get("cell_rate"),
                  "disk_rate": res.get("disk_rate"),
                  "gpu": res.get("gpu"),
                  "vram_total_bytes": res.get("vram_total_bytes"),
                  "prefetch_gain": (res.get("prefetch") or {}).get("gain"),
                  "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                time.gmtime())}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(settings, fh, indent=2)


def cmd_host(args, st) -> int:
    # Through resolve_host, so remembering a *different* box drops the old
    # one's rental clock and run record here too, not only when --host is
    # used. The two ways of naming a box must not disagree about that.
    host = resolve_host(args, st)
    check_ssh(host)
    print(f"  remembered {c(host, '36')}")
    return 0


def cmd_forget(args, st) -> int:
    """Drop everything remembered about the last box.

    Wanted after destroying a droplet, and needed when the next one comes up
    on the *same* address -- `resolve_host` cannot tell that apart from the
    box still being there, so it would carry the dead one's clock forward.
    """
    host = st.get("host")
    if not host:
        print("  nothing remembered")
        return 0
    st.clear()
    save_state(st)
    print(f"  forgot {c(host, '36')} -- its clock and run record are gone")
    print(c("  Give the next box with --host, or `host root@<ip>`.", "2"))
    return 0


def cmd_provision(args, st) -> int:
    args.host = args.host or args.host_pos
    host = resolve_host(args, st)
    st["host"] = host
    # Provisioning a box already remembered leaves its clock alone; a new one
    # got a fresh clock from resolve_host. This covers the third case: a box
    # remembered by `host` and never provisioned, so no clock was ever set.
    st.setdefault("rented_since", time.time())
    save_state(st)

    check_ssh(host)
    push_tree(host)

    flags = []
    if args.driver:
        flags.append("--driver")
    if args.quick:
        flags.append("--quick")
    remote = ("chmod +x ~/peregrine/solver/cloud/provision.sh && "
              "~/peregrine/solver/cloud/provision.sh " + " ".join(flags))
    print()
    proc = subprocess.Popen(ssh_cmd(host, remote), text=True)
    proc.wait()
    if proc.returncode != 0:
        die(f"provisioning failed (exit {proc.returncode})")

    env = remote_env(host)
    if env:
        print()
        print(f"  {c(env.get('gpu_name', '?'), '1;32')}, "
              f"{env.get('vram_bytes', 0) / 2**30:.0f} GB VRAM, "
              f"scratch at {env.get('disk_rate', 0) / 1e6:.0f} MB/s, "
              f"self-test {env.get('self_test')}")
    print(c("\n  Billing runs until the box is DESTROYED, not powered off.",
            "1;33"))
    return 0


# --------------------------------------------------------------------------
# Saved jobs
#
# A job is a directory the wizard froze a plan into: `job.json` (what was
# planned, and for which card), `config.json`, and copies of the three
# inputs. It exists because planning and renting run on different clocks --
# the plan is settled over an evening of re-planning, the box is created
# minutes before it is needed and destroyed as soon as the tables land.
#
# The inputs are *copies*, and the config names them relatively, so a job is
# a fact rather than a pointer: editing a target in the workspace after
# saving cannot quietly change what a later run solves. Resolving those
# relative names against the job directory is this side's half of that
# bargain.
# --------------------------------------------------------------------------

JOB_FILE = "job.json"


def jobs_dir() -> str | None:
    """Where the wizard keeps saved jobs, if a workspace has been chosen."""
    try:
        ws = load_wizard_settings().get("workspace")
    except Exception:
        return None
    return os.path.join(ws, "jobs") if ws else None


def load_wizard_settings() -> dict:
    path = os.path.join(os.environ.get("LOCALAPPDATA", HERE), "Peregrine",
                        "wizard.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def resolve_config(spec: str) -> tuple:
    """A config path from what the user typed, plus the job it came from.

    Three forms, most explicit first: a config file, a job directory, or a
    bare job name looked up in the workspace. The bare name is the one worth
    having -- `run overnight-full --host root@1.2.3.4` is short enough to
    type from memory next to a droplet that has been up for ninety seconds.
    """
    if os.path.isfile(spec):
        d = os.path.dirname(os.path.abspath(spec))
        jpath = os.path.join(d, JOB_FILE)
        if os.path.basename(spec) == JOB_FILE:
            return _job_at(d)
        return spec, _read_job(jpath)
    if os.path.isdir(spec) and os.path.isfile(os.path.join(spec, JOB_FILE)):
        return _job_at(spec)
    root = jobs_dir()
    if root and os.path.isfile(os.path.join(root, spec, JOB_FILE)):
        return _job_at(os.path.join(root, spec))
    known = list_jobs()
    hint = ""
    if known:
        hint = "\nsaved jobs: " + ", ".join(j["name"] for j in known)
    elif root:
        hint = (f"\nno saved jobs in {root} -- the wizard's step 3 writes "
                f"them,\nchoose 'Save this plan as a job to run later'")
    die(f"no config or saved job called '{spec}'{hint}")


def _job_at(d: str) -> tuple:
    cfg = os.path.join(d, "config.json")
    if not os.path.isfile(cfg):
        die(f"job at {d} has no config.json")
    return cfg, _read_job(os.path.join(d, JOB_FILE))


def _read_job(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def list_jobs() -> list:
    root = jobs_dir()
    if not root or not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        j = _read_job(os.path.join(root, name, JOB_FILE))
        if j is not None:
            j.setdefault("name", name)
            j["dir"] = os.path.join(root, name)
            out.append(j)
    return out


def job_rate(args, job: dict | None) -> float:
    """What to price this run at.

    An explicit `--rate` wins. Otherwise the job's own card is a far better
    default than the module constant, which is an L40S price and would
    under-bill an H200 run by half.
    """
    if getattr(args, "rate", None) is not None:
        return args.rate
    if job and job.get("usd_hr"):
        return float(job["usd_hr"])
    return DEFAULT_USD_PER_HOUR


def resolve_inputs(cfg: dict, cfg_path: str) -> dict:
    """Make the config's input paths absolute, against its own directory.

    A saved job names its inputs relatively so it can be moved or copied to
    another machine. Everything downstream -- `push_inputs` above all -- wants
    real paths, and this is the one place that knows what they are relative
    to.
    """
    base = os.path.dirname(os.path.abspath(cfg_path))
    out = dict(cfg)
    for key in INPUT_KEYS:
        v = cfg.get(key)
        if isinstance(v, str) and v and not os.path.isabs(v):
            out[key] = os.path.normpath(os.path.join(base, v))
    return out


def _print_job(j: dict, rate: float) -> None:
    p = j.get("plan") or {}
    est = p.get("estimate_s") or 0.0
    bits = []
    if p.get("cells"):
        bits.append(f"{p['cells']:,} cells")
    if p.get("driver"):
        bits.append("in core" if p["driver"] == "incore" else "tiled")
    if p.get("tau_applied") is not None:
        cost = p.get("tau_applied_cost") or 0.0
        bits.append(f"lookahead {p['tau_applied']:.3f} s"
                    + (f" (+{cost:.1f}%)" if cost > 0.5 else ""))
    print(f"  {c(j.get('name', '?'), '1;36')}   "
          + c(j.get("gpu_label", j.get("target_gpu", "?")), "2"))
    if bits:
        print("    " + ", ".join(bits))
    if est:
        print(f"    planned {hms(est)}"
              + (f", about ${est / 3600 * rate:,.2f} at ${rate:.2f}/hr"
                 if rate else ""))
    print(c(f"    saved {j.get('saved_utc', '?')}   {j.get('dir', '')}", "2"))


def cmd_jobs(args, st) -> int:
    js = list_jobs()
    if not js:
        root = jobs_dir()
        print("  no saved jobs" + (f" in {root}" if root else
                                   " -- no workspace chosen yet"))
        print(c("  The wizard's step 3 saves one: choose 'Save this plan as "
                "a job to run later'.", "2"))
        return 0
    print()
    for j in js:
        _print_job(j, job_rate(args, j))
        print()
    print(c("  run one with:  py -3.12 solver/cloud/peregrine_remote.py run "
            "<name> --host root@<ip>", "2"))
    return 0


def _prepare(host, cfg_path, st):
    """Everything both `plan` and `run` need: env, upload, remote config."""
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg = resolve_inputs(cfg, cfg_path)
    env = remote_env(host)
    if not env:
        die("this box has not been provisioned. Run:\n"
            "    py -3.12 solver/cloud/peregrine_remote.py provision")
    push_tree(host)
    cfg = push_inputs(host, cfg, env)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    rcfg = remote_config(cfg, env, stamp)
    rpath = write_remote_config(host, rcfg, env, stamp)
    return env, rcfg, rpath, stamp


def cmd_plan(args, st) -> int:
    cfg_path, job = resolve_config(args.config)
    host = resolve_host(args, st)
    check_ssh(host)
    env, rcfg, rpath, _ = _prepare(host, cfg_path, st)
    julia = env.get("julia", "julia")
    print("\n  planning on the box ...\n")
    r = ssh(host, f"cd ~/peregrine && {shlex.quote(julia)} --project=solver "
                  f"--threads=auto solver/solve.jl {shlex.quote(rpath)} plan",
            check=False)
    if r.returncode != 0:
        die(f"plan failed:\n{r.stderr.strip()[:1500]}")
    line = next((l for l in r.stdout.splitlines() if l.startswith("PLAN ")), None)
    if not line:
        die(f"no PLAN line in the output:\n{r.stdout[:1000]}")
    p = json.loads(line[5:])
    _print_plan(p, job_rate(args, job), env)
    if job:
        _compare_to_job(job, p)
    return 0


def _compare_to_job(job: dict, p: dict) -> None:
    """Say when the box disagrees with the plan the job was saved from.

    It often will, and legitimately: the desktop planned for a *model* of the
    card, the box plans for the card. A bigger VRAM than the catalogue
    assumed can move the run from tiled to in core, which changes the driver,
    the lookahead and the estimate all at once. That is good news, but it is
    news, and a job that quietly solves something other than what was
    approved is the failure worth avoiding.
    """
    was = job.get("plan") or {}
    now_mode = (p.get("out_of_core") or {}).get("mode")
    now_tau = (p.get("recommend") or {}).get("tau_applied")
    rows = []
    if was.get("driver") and now_mode and was["driver"] != now_mode:
        rows.append(("driver", was["driver"], now_mode))
    if was.get("tau_applied") is not None and now_tau is not None \
            and abs(was["tau_applied"] - now_tau) > 1e-6:
        rows.append(("lookahead", f"{was['tau_applied']:.3f} s",
                     f"{now_tau:.3f} s"))
    if was.get("cells") and p.get("cells") and was["cells"] != p["cells"]:
        rows.append(("cells", f"{was['cells']:,}", f"{p['cells']:,}"))
    if not rows:
        return
    print()
    print(c(f"  this box does not agree with the saved plan "
            f"('{job.get('name', '?')}')", "33"))
    for what, a, b in rows:
        print(f"    {what:<12} saved {a}  ->  here {b}")
    print(c("    The box's own plan is the one that runs.", "2"))


def _print_plan(p: dict, rate: float, env: dict) -> None:
    ooc = p.get("out_of_core", {})
    rt = p.get("runtime", {})
    print(f"  grid         {p['n']}  =  {p['cells']:,} cells")
    print(f"  per table    {human(p['bytes_per_target'])}   "
          f"({p['n_targets']} target(s), {human(p['bytes_total'])} total)")
    print(f"  resolution   {p['resolution']['actual']['xy_cm']:.2f} cm xy, "
          f"{p['resolution']['actual']['heading_deg']:.2f} deg, "
          f"{p['resolution']['actual']['v_cm_s']:.1f} cm/s")
    mode = ooc.get("mode", "?")
    if mode == "incore":
        print(f"  driver       {c('WHOLE GRID on the card', '1;32')} "
              f"-- no tiling, no scratch file, lookahead costs nothing")
    elif ooc.get("fits"):
        print(f"  driver       {c('TILED', '33')} -- "
              f"{ooc['tiles_per_round']} tiles, "
              f"{ooc['amplification']:.1f}x loaded per updated, "
              f"{human(ooc['store_bytes'])} scratch file")
    else:
        print(f"  driver       {c('DOES NOT FIT', '1;31')} at this lookahead")

    # The number the whole comparison turns on. A bigger card is not mainly
    # faster -- it is a card that can afford a longer lookahead, and the
    # lookahead is priced in accuracy, one-sidedly: a backup that cannot see
    # far enough settles for a worse command, so a short step only ever adds
    # time to the answer. Reported next to the wall clock so the two are
    # traded against each other rather than one at a time.
    rec = p.get("recommend", {})
    if rec:
        cost = rec.get("tau_applied_cost", 0.0)
        tag = c(f"+{cost:.1f}% mean value", "1;31" if cost >= 15
                else "33" if cost > 0.5 else "32")
        print(f"  lookahead    {rec.get('tau_applied', 0):.3f} s costs {tag} "
              f"against the {rec.get('tau_reference', 0.5)} s reference")
        want = rec.get("tau_max")
        if want and abs(want - rec.get("tau_applied", 0.0)) > 1e-6:
            print(f"               this box would rather run "
                  f"{c(f'{want:.3f} s', '36')} "
                  f"-- set the longest step back to auto")
    if rt:
        total = rt.get("total_s", 0.0)
        print(f"  estimate     {hms(total)} for all targets "
              f"({'i/o' if rt.get('io_bound') else 'compute'}-bound, "
              f"at {rt.get('cell_rate', 0)/1e6:.1f}M cells/s)")
        print(f"  {c(f'rental      ~${total / 3600 * rate:,.2f} at ${rate:.2f}/hr', '1;33')}")
        if rt.get("cell_rate", 0) == 23.4e6:
            print(c("               the cell rate is still the DESKTOP's "
                    "measurement -- this box is", "2"))
            print(c("               probably faster, so read the estimate as "
                    "a ceiling until one run", "2"))
            print(c("               has measured it.", "2"))
    dl = p["bytes_total"]
    print(f"  download     {human(dl)} to bring home "
          f"({hms(dl / (12.5e6))} at 100 Mbit/s)")


def cmd_run(args, st) -> int:
    cfg_path, job = resolve_config(args.config)
    rate = job_rate(args, job)
    host = resolve_host(args, st)
    check_ssh(host)
    if job:
        print()
        _print_job(job, rate)
    env, rcfg, rpath, stamp = _prepare(host, cfg_path, st)
    julia = env.get("julia", "julia")
    rdir = rcfg["out_dir"]
    log = posixpath.join(rdir, "solve.log")

    # Detached, so the run outlives this process and every network hiccup.
    # The sentinel is appended by the same shell that ran the solver, which
    # is what lets `follow` end without polling.
    inner = (f"cd ~/peregrine && {shlex.quote(julia)} --project=solver "
             f"--threads=auto solver/solve.jl {shlex.quote(rpath)} solve "
             f"> {shlex.quote(log)} 2>&1; "
             f"echo PEREGRINE_EXIT $? >> {shlex.quote(log)}")
    launch = (f"tmux has-session -t peregrine 2>/dev/null && "
              f"{{ echo BUSY; exit 3; }}; "
              f"touch {shlex.quote(log)}; "
              f"tmux new-session -d -s peregrine {shlex.quote(inner)}")
    r = ssh(host, launch, check=False)
    if r.returncode == 3 or "BUSY" in r.stdout:
        die("a solve is already running on this box.\n"
            "Follow it with `attach`, or stop it with\n"
            f"    ssh {host} tmux kill-session -t peregrine")
    if r.returncode != 0:
        die(f"could not start the solve: {r.stderr.strip()[:600]}")

    st["host"] = host
    st["run"] = {"remote_dir": rdir, "log": log, "stamp": stamp,
                 "config": os.path.abspath(cfg_path),
                 "job": (job or {}).get("name"),
                 "rate": rate,
                 # The config's own out_dir, not the remapped one: a job
                 # points it at the job's tables/ directory, which is where
                 # the whole point is for the tables to end up.
                 "local_out": os.path.abspath(
                     json.load(open(cfg_path, encoding="utf-8"))["out_dir"]),
                 "started": time.time(), "cells": 0}
    save_state(st)

    print(f"\n  started in tmux on {c(host, '36')} -- "
          f"this window can close safely\n")

    # Stream each table home as it is finished rather than all of them at the
    # end. The download is then paid for out of time the box is spending on
    # the next target anyway, instead of out of rented minutes at the end.
    dest = os.path.abspath(args.stream_to) if args.stream_to \
        else st["run"]["local_out"]
    streamer = None
    if not args.no_pull and not args.no_stream:
        os.makedirs(dest, exist_ok=True)
        streamer = TableStreamer(host, rdir, dest)
        if dest != st["run"]["local_out"]:
            print(f"  streaming tables to {c(dest, '36')}")
        st["run"]["local_out"] = dest
        save_state(st)

    started = time.time()
    pr = follow(host, log, streamer = streamer)
    st["run"]["cells"] = pr.cells
    save_state(st)
    if streamer is not None:
        # Anything still queued is a table the box has already finished, so
        # this is waiting on the wire and not on the GPU -- but the box is
        # billing throughout, so say what is happening rather than appearing
        # to hang.
        if not streamer.q.empty():
            print(c("\n  finishing the last table's download...", "2"))
        streamer.finish()

    if pr.exit_code not in (0, None):
        print(c(f"\n  solver failed (exit {pr.exit_code})", "31"))
        tailr = ssh(host, f"tail -40 {shlex.quote(log)}", check=False)
        print(tailr.stdout[-2000:])
        _cost_line(started, rate)
        return 1

    elapsed = time.time() - started
    _measured_rate(pr, elapsed, cfg_path)
    if args.no_pull:
        print(c("\n  --no-pull: the tables are still on the box.", "33"))
        _cost_line(st.get("rented_since", started), rate)
        return 0
    if streamer is not None and streamer.done:
        got = sum(streamer.done.values())
        print(f"\n  {c(f'{human(got)} already home', '1;32')} -- "
              f"{len(streamer.done)} of {pr.n_targets} tables arrived while "
              f"the box was still solving")
        print(c(f"  that is {hms(streamer.seconds)} of transfer that cost no "
                f"rental time (~${streamer.seconds / 3600 * rate:,.2f})",
                "2"))
    return _pull(host, st, rate, streamed = streamer)


def _measured_rate(pr: ProgressReader, elapsed: float, cfg_path: str) -> None:
    """Report the box's real cell rate, so the next `plan` estimates properly."""
    if not pr.cells or pr.done_targets < 1 or elapsed <= 0:
        return
    # Sweeps actually run, not the budget: the tolerance usually stops a
    # target early, and charging the whole budget would understate the rate.
    sweeps = pr.units_seen * pr.done_targets
    if sweeps <= 0:
        return
    rate = pr.cells * sweeps / elapsed
    print(f"\n  measured {c(f'{rate/1e6:.1f}M cell-updates/s', '1;32')} on this "
          f"box ({pr.cells:,} cells x {sweeps} sweeps in {hms(elapsed)})")
    print(f"  put {c(f'\"cell_rate\": {rate:.3g}', '36')} in "
          f"{os.path.basename(cfg_path)} and every later estimate sharpens")


def _pull(host: str, st: dict, rate: float, streamed=None) -> int:
    run = st.get("run")
    if not run:
        die("no run to pull -- this box has not solved anything from here")
    local = run["local_out"]
    os.makedirs(local, exist_ok=True)

    # Chunk files already fetched by the streamer are excluded by name, so
    # this is a sweep for the remainder -- the manifest, the model, the log,
    # and any target whose stream failed -- rather than a second copy of
    # everything. Excluding by target rather than by file keeps the argument
    # list to a handful of patterns however many chunks there are.
    excl = ""
    if streamed is not None and streamed.done:
        pats = " ".join(f"--exclude={shlex.quote(f'TABLES/T{i:02d}C*.BIN')}"
                        for i in sorted(streamed.done))
        excl = " " + pats
        print(f"\n  collecting what the stream did not take "
              f"({len(streamed.done)} table(s) already home)")

    size = ssh(host,
               f"cd {shlex.quote(run['remote_dir'])} && du -sb ."
               + (" 2>/dev/null" if not excl else " 2>/dev/null"),
               check=False).stdout.strip().split()
    try:
        total = int(size[0])
    except (ValueError, IndexError):
        total = 0
    if streamed is not None and streamed.done:
        total = max(0, total - sum(streamed.done.values()))
    else:
        print(f"\n  pulling {human(total) if total else 'the tables'} "
              f"to {c(local, '36')}")

    src = ssh_popen_binary(
        host, f"tar -c{excl} -C {shlex.quote(run['remote_dir'])} .")
    dst = subprocess.Popen(["tar", "-x", "-C", local], stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    got, secs = copy_counted(src.stdout, dst.stdin, total)
    dst.wait()
    src.wait()
    if dst.returncode != 0:
        die("unpacking the download failed -- the tables did not arrive "
            "intact.\nThe box still has them; run `pull` again.")

    # Verified here rather than on the box, deliberately: this checks the
    # bytes that actually arrived, so a truncated download fails the same
    # check a malformed table would.
    print(c("\n  verifying against the format spec...", "2"))
    v = subprocess.run([sys.executable,
                        os.path.join(REPO, "wizard", "verify_tables.py"), local],
                       capture_output=True, text=True)
    print(v.stdout[-3000:])
    if v.returncode != 0:
        print(c("  verification FAILED -- do not destroy the box yet, the "
                "tables may need re-pulling", "31"))
        print(v.stderr[-1000:])
        return 1

    _cost_line(st.get("rented_since", run["started"]), rate)
    print(c("\n  Tables are home and verified. DESTROY THE BOX NOW:", "1;33"))
    print(c(f"      doctl compute droplet delete <name>", "36"))
    print(c("      (or the Destroy tab in the control panel -- powering off "
            "does NOT stop billing)", "2"))
    return 0


def cmd_pull(args, st) -> int:
    host = resolve_host(args, st)
    check_ssh(host)
    return _pull(host, st, job_rate(args, None))


def cmd_attach(args, st) -> int:
    host = resolve_host(args, st)
    run = st.get("run")
    if not run:
        die("no run started from this desktop to attach to")
    check_ssh(host)
    print(f"  following {c(run['stamp'], '36')} on {host}\n")
    # Re-attaching replays the log from the start, so every target that has
    # already finished is announced again -- and therefore streamed now. That
    # is the behaviour worth having rather than a quirk to suppress: it is
    # exactly how a run whose connection dropped mid-solve collects the tables
    # it missed, without waiting for the whole thing to end.
    dest = os.path.abspath(args.stream_to) if args.stream_to \
        else run["local_out"]
    streamer = None
    if not args.no_pull and not args.no_stream:
        os.makedirs(dest, exist_ok=True)
        streamer = TableStreamer(host, run["remote_dir"], dest)
        run["local_out"] = dest
        save_state(st)
    pr = follow(host, run["log"], streamer = streamer)
    if streamer is not None:
        streamer.finish()
    if pr.exit_code not in (0, None):
        print(c(f"\n  solver failed (exit {pr.exit_code})", "31"))
        return 1
    if pr.exit_code == 0 and not args.no_pull:
        # The run remembered what it was being priced at, which for a job is
        # its own card rather than the module's L40S default.
        rate = args.rate if args.rate is not None \
            else run.get("rate") or DEFAULT_USD_PER_HOUR
        return _pull(host, st, rate, streamed = streamer)
    return 0


def cmd_status(args, st) -> int:
    host = resolve_host(args, st)
    check_ssh(host)
    env = remote_env(host)
    if not env:
        print(c("  not provisioned", "33"))
        return 0
    print(f"  {c(env['gpu_name'], '1;32')} x{env['gpu_count']}, "
          f"{env['vram_bytes']/2**30:.0f} GB VRAM")
    print(f"  julia {env['julia']}, self-test {env['self_test']}, "
          f"provisioned {env['provisioned_utc']}")
    print(f"  scratch {env['scratch_dir']} at {env['disk_rate']/1e6:.0f} MB/s")
    busy = ssh(host, "tmux has-session -t peregrine 2>/dev/null && echo yes "
                     "|| echo no", check=False).stdout.strip()
    print(f"  solve running: {c('yes', '1;33') if busy == 'yes' else 'no'}")
    df = ssh(host, "df -h --output=target,used,avail,pcent ~/work | tail -1",
             check=False).stdout.strip()
    print(f"  disk    {df}")
    _cost_line(st.get("rented_since"), job_rate(args, None))
    return 0


def _cost_line(since, rate: float) -> None:
    if not since:
        return
    el = time.time() - since
    print(f"\n  rented {c(hms(el), '1;33')} so far "
          f"= {c(f'${el / 3600 * rate:,.2f}', '1;33')} at ${rate:.2f}/hr")


def cmd_cost(args, st) -> int:
    # Goes through resolve_host so `cost --host <new box>` reports on that
    # box rather than silently on the last one. No SSH: this reads a clock.
    if getattr(args, "host", None):
        resolve_host(args, st)
    since = st.get("rented_since")
    if not since:
        print("  nothing rented from here yet")
        return 0
    print(f"  {c(st.get('host', '?'), '36')}")
    _cost_line(since, job_rate(args, None))
    print(c("  Counted from when this desktop first named the box, which is "
            "later than\n  the box was created if it sat idle before you got "
            "to it.", "2"))
    print(c("  Billing stops when the box is DESTROYED, not when it is "
            "powered off\n  and not when this script exits.", "2"))
    return 0


# --------------------------------------------------------------------------
# Self-test
#
# Everything that can be wrong without a box: the path rewriting, the
# exclusion list, the progress parser and its exit sentinel, and the cost
# arithmetic. The parts that need a box are the parts provision.sh checks.
# --------------------------------------------------------------------------

def self_test() -> int:
    npass = nfail = 0

    def check(name, cond, detail=""):
        nonlocal npass, nfail
        if cond:
            npass += 1
            print(f"  {c('pass', '32')}  {name}")
        else:
            nfail += 1
            print(f"  {c('FAIL', '31')}  {name}  {detail}")

    print("\n[1] the config's desktop paths all move onto the box")
    cfg = {"regression": r"D:\PeregrineWorkspace\3\calibration\drivetrain_fit.toml",
           "field": r"D:\PeregrineWorkspace\3\field\field.json",
           "targets": r"D:\PeregrineWorkspace\3\field\targets.json",
           "out_dir": r"D:\PeregrineWorkspace\3\runs\20260830_215044",
           "scratch_dir": r"D:\PeregrineWorkspace\3\runs",
           "vram_budget_bytes": 6206227742,
           "grid": {"n": [72, 72, 32, 21, 21, 32]}}
    env = {"inputs_dir": "/root/work/inputs", "runs_dir": "/root/work/runs",
           "scratch_dir": "/root/work/scratch", "disk_rate": 1600000000}
    # push_inputs does the input keys; remote_config does the rest.
    staged = dict(cfg)
    for k in INPUT_KEYS:
        staged[k] = posixpath.join(env["inputs_dir"],
                                   os.path.basename(cfg[k]))
    r = remote_config(staged, env, "20260901_120000")
    leftover = [k for k, v in r.items()
                if isinstance(v, str) and ("\\" in v or re.match(r"^[A-Za-z]:", v))]
    check("no Windows path survives", not leftover, str(leftover))
    check("out_dir is the stamped run dir",
          r["out_dir"] == "/root/work/runs/20260901_120000", r["out_dir"])
    check("scratch_dir points at the box's scratch",
          r["scratch_dir"] == "/root/work/scratch", r["scratch_dir"])
    check("the box's measured disk rate replaces the desktop default",
          r["disk_rate"] == 1600000000, str(r.get("disk_rate")))
    check("a desktop VRAM override is dropped, not carried onto a bigger card",
          "vram_budget_bytes" not in r)
    check("the grid is untouched", r["grid"]["n"] == [72, 72, 32, 21, 21, 32])

    print("\n[2] the exclusion list keeps the big and the local off the wire")
    for pat, why in ((".git", "history"), ("__pycache__", "bytecode"),
                     ("runs", "previous tables"), ("*.SCRATCH", "store files"),
                     ("*.BIN", "table chunks")):
        check(f"{pat} excluded ({why})", pat in EXCLUDE)

    print("\n[3] the progress parser drives the bar and stops on the sentinel")
    pr = ProgressReader("test")
    lines = [
        'PROGRESS {"phase":"setup","n_targets":2,"cells":2341011456,'
        '"controls":25,"bytes_per_target":4682022912}',
        'PROGRESS {"phase":"occupancy","blocked_frac":0.326}',
        'PROGRESS {"phase":"backend","backend":"cuda"}',
        'PROGRESS {"phase":"solve","target_name":"a","iter":50,"iters":400}',
        'PROGRESS {"phase":"target_done","target_name":"a","reached_frac":0.88}',
        'PROGRESS {"phase":"solve","target_name":"b","iter":200,"iters":400}',
        'PROGRESS {"phase":"target_done","target_name":"b","reached_frac":0.9}',
        'PROGRESS {"phase":"done","targets":2}',
    ]
    alive = [pr.feed(l) for l in lines]
    check("every progress line is consumed", all(alive))
    check("target count read from setup", pr.n_targets == 2, str(pr.n_targets))
    check("cell count kept for the rate measurement",
          pr.cells == 2341011456, str(pr.cells))
    check("both targets counted", pr.done_targets == 2, str(pr.done_targets))
    check("progress reaches 100%", abs(pr.frac - 1.0) < 1e-9, str(pr.frac))
    check("no exit code before the sentinel", pr.exit_code is None)
    alive_after = pr.feed("PEREGRINE_EXIT 0")
    check("the sentinel ends the follow", not alive_after)
    check("and carries the exit code", pr.exit_code == 0, str(pr.exit_code))

    pr2 = ProgressReader("test")
    pr2.feed('PROGRESS {"phase":"setup","n_targets":1,"cells":10,'
             '"controls":1,"bytes_per_target":20}')
    check("a failing solve reports its code",
          not pr2.feed("PEREGRINE_EXIT 1") and pr2.exit_code == 1)
    check("junk on the log does not crash the parser",
          pr2.feed("some julia warning") and pr2.feed("PROGRESS {not json"))

    print("\n[4] a tiled run's bar moves inside a round")
    pr3 = ProgressReader("test")
    pr3.feed('PROGRESS {"phase":"setup","n_targets":1,"cells":100,'
             '"controls":1,"bytes_per_target":200}')
    pr3.feed('PROGRESS {"phase":"decompose","tiles":16,"tile_cells":[22,22],'
             '"halo_cells":[11,11],"reach_cells":6.7,"amplification":4.0,'
             '"store_bytes":16387080192,"resident_bytes":6119928320}')
    pr3.feed('PROGRESS {"phase":"tile","target_name":"a","round":1,'
             '"rounds":100,"tile":8,"tiles":16}')
    f1 = pr3.frac
    pr3.feed('PROGRESS {"phase":"tile","target_name":"a","round":2,'
             '"rounds":100,"tile":1,"tiles":16}')
    f2 = pr3.frac
    check("tile progress is real progress, not a spinner",
          0.0 < f1 < f2, f"{f1} then {f2}")
    check("the bar's redraw throttle does not hide it from the reader",
          f2 != pr3.bar.frac or f1 == pr3.bar.frac)

    print("\n[5] the download survives bytes that are not text")
    # The pull is `tar -c` on the box piped through here into `tar -x`. A
    # table chunk is u16 little-endian, so it is full of bytes that are not
    # valid UTF-8 and full of 0x0d 0x0a pairs that a text pipe would rewrite.
    # This is that path end to end, with the same helper the pull uses.
    work = tempfile.mkdtemp(prefix="peregrine-selftest-")
    try:
        srcdir = os.path.join(work, "src")
        dstdir = os.path.join(work, "dst")
        os.makedirs(os.path.join(srcdir, "TABLES"))
        os.makedirs(dstdir)
        # Every byte value, plus the CRLF and EOF-marker sequences that a
        # text or Windows-mode pipe mangles.
        payload = bytes(range(256)) * 400 + b"\r\n\x1a\r\n" + os.urandom(4096)
        with open(os.path.join(srcdir, "TABLES", "T00C0000.BIN"), "wb") as fh:
            fh.write(payload)
        with open(os.path.join(srcdir, "MANIFEST.JSON"), "w",
                  encoding="utf-8") as fh:
            json.dump({"schema_version": 1}, fh)

        tar_c = subprocess.Popen(["tar", "-c", "-C", srcdir, "."],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
        tar_x = subprocess.Popen(["tar", "-x", "-C", dstdir],
                                 stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
        n, _ = copy_counted(tar_c.stdout, tar_x.stdin, 0, "self-test")
        tar_x.wait()
        tar_c.wait()
        check("the stream unpacks cleanly", tar_x.returncode == 0,
              f"tar -x exited {tar_x.returncode}")
        check("bytes actually moved", n > len(payload), str(n))
        landed = os.path.join(dstdir, "TABLES", "T00C0000.BIN")
        check("the table chunk arrives", os.path.isfile(landed))
        if os.path.isfile(landed):
            with open(landed, "rb") as fh:
                back = fh.read()
            check("byte for byte identical -- no text decode, no CRLF "
                  "translation", back == payload,
                  f"{len(back)} B back vs {len(payload)} B sent")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("\n[6] tables stream home as each target finishes")
    # The whole point is overlap, so this drives the real machinery -- the
    # reader's hook, the queue, the worker, and the exclusion the final sweep
    # builds -- with only the ssh call replaced by a local tar. Everything
    # except the wire is the shipping code.
    work = tempfile.mkdtemp(prefix="peregrine-stream-")
    try:
        remote = os.path.join(work, "remote")
        dest = os.path.join(work, "dest")
        os.makedirs(os.path.join(remote, "TABLES"))
        os.makedirs(dest)
        payload = {}
        for ti in range(3):
            for ck in range(2):
                nm = "T%02dC%04d.BIN" % (ti, ck)
                body = bytes(range(256)) * (20 + ti) + b"\r\n\x1a"
                with open(os.path.join(remote, "TABLES", nm), "wb") as fh:
                    fh.write(body)
                payload[nm] = body
        with open(os.path.join(remote, "MANIFEST.JSON"), "w",
                  encoding="utf-8") as fh:
            json.dump({"schema_version": 1}, fh)

        real_popen = globals()["ssh_popen_binary"]

        def fake(host, cmd):
            # The streamer's own command names the pattern; run the same
            # find | tar locally, so the argv-avoidance is exercised too.
            m = re.search(r"-name '([^']+)'", cmd)
            if m:
                return subprocess.Popen(
                    ["bash", "-c",
                     "cd %s && find TABLES -name %s -print0 | tar -c --null -T -"
                     % (shlex.quote(remote), shlex.quote(m.group(1)))],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return real_popen(host, cmd)

        globals()["ssh_popen_binary"] = fake
        try:
            streamer = TableStreamer("fake", remote, dest)
            pr = ProgressReader("test")
            pr.on_target_done = lambda ev: streamer.enqueue(
                int(ev["target"]), ev["target_name"], int(ev.get("bytes", 0)))
            pr.feed('PROGRESS {"phase":"setup","n_targets":3,"cells":100,'
                    '"controls":25,"bytes_per_target":1000}')
            for ti, nm in enumerate(("a", "b", "c")):
                pr.feed('PROGRESS {"phase":"target_done","target":%d,'
                        '"target_name":"%s","reached_frac":0.9,"bytes":1000}'
                        % (ti, nm))
            pr.feed('PROGRESS {"phase":"done","targets":3}')
            streamer.finish()
        finally:
            globals()["ssh_popen_binary"] = real_popen

        check("every target was fetched", sorted(streamer.done) == [0, 1, 2],
              "%s failed=%s" % (sorted(streamer.done), streamer.failed))
        landed = sorted(os.listdir(os.path.join(dest, "TABLES")))
        check("all six chunks arrived", landed == sorted(payload), str(landed))
        same = True
        for k, v in payload.items():
            with open(os.path.join(dest, "TABLES", k), "rb") as fh:
                same = same and fh.read() == v
        check("byte for byte identical through the stream", same)
        check("bytes were counted", sum(streamer.done.values()) > 0,
              str(streamer.done))

        # The final sweep must not re-fetch what the stream already took.
        pats = " ".join("--exclude=%s" % shlex.quote("TABLES/T%02dC*.BIN" % i)
                        for i in sorted(streamer.done))
        check("the final sweep excludes every streamed target",
              all(("TABLES/T%02dC*.BIN" % i) in pats for i in range(3)), pats)
        check("...and it is a handful of patterns, not a file list",
              len(pats.split()) == 3, str(len(pats.split())))

        # A failed fetch must be left for the sweep rather than lost or fatal.
        def broken(host, cmd):
            return subprocess.Popen(["bash", "-c", "exit 1"],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)

        globals()["ssh_popen_binary"] = broken
        try:
            s2 = TableStreamer("fake", remote, dest)
            s2.enqueue(7, "doomed", 10)
            s2.finish()
        finally:
            globals()["ssh_popen_binary"] = real_popen
        check("a failed fetch is recorded, not raised", 7 in s2.failed,
              str(s2.failed))
        check("and is absent from `done`, so the final sweep collects it",
              7 not in s2.done)

        # The space check warns; it must never stop a run that is already
        # costing money.
        s3 = TableStreamer("fake", remote, dest)
        s3.finish()
        s3.check_space(1 << 62)
        check("an impossible size warns without raising", True)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("\n[7] a passphrase-protected key is told apart from a missing one")
    # Both produce "Permission denied (publickey)" under BatchMode, and the
    # advice for each is the opposite of the other's, so guessing is worse
    # than useless.
    import base64 as _b64

    def blob(cipher: bytes) -> str:
        raw = (b"openssh-key-v1\x00" + len(cipher).to_bytes(4, "big") + cipher
               + b"\x00\x00\x00\x04none" + os.urandom(32))
        return ("-----BEGIN OPENSSH PRIVATE KEY-----\n"
                + _b64.b64encode(raw).decode() +
                "\n-----END OPENSSH PRIVATE KEY-----\n")

    check("an unencrypted key reads as 'none'",
          key_cipher(blob(b"none")) == "none")
    check("an encrypted key names its cipher",
          key_cipher(blob(b"aes256-ctr")) == "aes256-ctr")
    check("a legacy PEM key is caught by its header",
          key_cipher("-----BEGIN RSA PRIVATE KEY-----\n"
                     "Proc-Type: 4,ENCRYPTED\n") == "pem-encrypted")
    check("a public key is not mistaken for a private one",
          key_cipher("ssh-ed25519 AAAAC3Nz... bknap@desktop\n") is None)
    check("garbage does not raise", key_cipher("not a key at all") is None)
    check("a truncated blob does not raise",
          key_cipher("-----BEGIN OPENSSH PRIVATE KEY-----\n"
                     + _b64.b64encode(b"openssh-key-v1\x00\xff\xff\xff\xff"
                                      ).decode()
                     + "\n-----END OPENSSH PRIVATE KEY-----\n") is None)

    print("\n[8] the cost arithmetic")
    # Two hours of the default rate, to the cent.
    check("2 h at $1.57/hr is $3.14",
          abs(2.0 * DEFAULT_USD_PER_HOUR - 3.14) < 1e-9)

    print("\n[9] the pieces this script needs are on PATH")
    for tool in ("ssh", "scp", "tar"):
        check(f"{tool} found", shutil.which(tool) is not None)
    check("provision.sh sits next to this script",
          os.path.isfile(os.path.join(HERE, "provision.sh")))
    check("verify_tables.py is where the pull expects it",
          os.path.isfile(os.path.join(REPO, "wizard", "verify_tables.py")))

    print("\n[10] a saved job resolves, and carries its own inputs")
    with tempfile.TemporaryDirectory() as tmp:
        jdir = os.path.join(tmp, "jobs", "overnight")
        os.makedirs(jdir)
        for fname in ("drivetrain_fit.toml", "field.json", "targets.json"):
            with open(os.path.join(jdir, fname), "w", encoding="utf-8") as fh:
                fh.write("{}")
        jcfg = {"regression": "drivetrain_fit.toml", "field": "field.json",
                "targets": "targets.json",
                "out_dir": os.path.join(jdir, "tables"),
                "scratch_dir": tmp, "grid": {"n": [8, 8, 4, 3, 3, 3]}}
        with open(os.path.join(jdir, "config.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(jcfg, fh)
        with open(os.path.join(jdir, JOB_FILE), "w", encoding="utf-8") as fh:
            json.dump({"name": "overnight", "target_gpu": "h200",
                       "usd_hr": 3.45, "gpu_label": "NVIDIA H200 141 GB",
                       "plan": {"driver": "incore", "tau_applied": 0.5,
                                "cells": 15363480384, "estimate_s": 138460.0}},
                      fh)

        cfgp, job = resolve_config(jdir)
        check("a job directory resolves to its config",
              cfgp == os.path.join(jdir, "config.json"), cfgp)
        check("and brings the job with it",
              (job or {}).get("target_gpu") == "h200", str(job))
        cfgp2, job2 = resolve_config(os.path.join(jdir, JOB_FILE))
        check("naming job.json itself works too",
              cfgp2 == cfgp and job2 == job)
        cfgp3, job3 = resolve_config(os.path.join(jdir, "config.json"))
        check("naming the config directly still finds the job beside it",
              cfgp3 == cfgp and (job3 or {}).get("usd_hr") == 3.45)

        # The point of the relative names: a job that has been moved still
        # finds its own inputs, because they travelled with it.
        moved = os.path.join(tmp, "elsewhere")
        shutil.copytree(jdir, moved)
        cfgp4, _ = resolve_config(moved)
        with open(cfgp4, encoding="utf-8") as fh:
            r = resolve_inputs(json.load(fh), cfgp4)
        check("a moved job's inputs resolve against the job, not the "
              "workspace",
              all(os.path.isfile(r[k]) and os.path.dirname(r[k]) == moved
                  for k in INPUT_KEYS),
              str([r[k] for k in INPUT_KEYS]))

        # An absolute path in a hand-written config must survive untouched --
        # the relative handling is an addition, not a replacement.
        abs_cfg = dict(jcfg, field=os.path.join(jdir, "field.json"))
        r2 = resolve_inputs(abs_cfg, os.path.join(jdir, "config.json"))
        check("an absolute input path is left alone",
              r2["field"] == abs_cfg["field"], r2["field"])

    print("\n[11] the rate falls back through the job, then the constant")

    class _A:
        rate = None
    check("no --rate and no job: the module default",
          job_rate(_A(), None) == DEFAULT_USD_PER_HOUR)
    check("no --rate but a job: the job's own card",
          job_rate(_A(), {"usd_hr": 3.45}) == 3.45)
    _B = type("_B", (), {"rate": 9.99})
    check("an explicit --rate wins over the job",
          job_rate(_B(), {"usd_hr": 3.45}) == 9.99)

    print("\n[12] a new box does not inherit the last one's clock")
    # The failure this prevents: a droplet destroyed on Monday leaves
    # `rented_since` behind, and Thursday's box -- alive for ninety seconds
    # -- reports three days of billing. Every cost line in this script reads
    # that one field.
    saved, restore = STATE, None
    with tempfile.TemporaryDirectory() as tmp:
        globals()["STATE"] = os.path.join(tmp, "remote.json")
        old = {"host": "root@1.1.1.1", "rented_since": time.time() - 3 * 86400,
               "run": {"remote_dir": "/root/work/runs/x", "stamp": "x"}}
        save_state(old)
        st = load_state()
        args = type("_", (), {"host": "root@2.2.2.2"})()
        h = resolve_host(args, st)
        check("--host wins over the remembered box", h == "root@2.2.2.2", h)
        check("the dead box's rental clock is dropped",
              time.time() - st["rented_since"] < 60,
              f"{(time.time() - st['rented_since']) / 86400:.1f} days old")
        check("and so is its run record, which names a directory on it",
              "run" not in st, str(st.get("run")))
        check("the reset is persisted, not just in memory",
              load_state().get("host") == "root@2.2.2.2")

        # Same box named again: nothing is disturbed. Re-running `status`
        # must not keep restarting the clock it is reporting.
        was = st["rented_since"]
        st["run"] = {"stamp": "y"}
        save_state(st)
        st2 = load_state()
        resolve_host(type("_", (), {"host": "root@2.2.2.2"})(), st2)
        check("naming the same box again leaves the clock and run alone",
              st2["rented_since"] == was and st2.get("run", {}).get("stamp") == "y")

        # No --host at all falls back to what is remembered, unchanged.
        st3 = load_state()
        h3 = resolve_host(type("_", (), {"host": None})(), st3)
        check("no --host falls back to the remembered box",
              h3 == "root@2.2.2.2" and st3["rented_since"] == was)
    globals()["STATE"] = saved

    print(f"\n{npass} passed, {nfail} failed\n")
    return 1 if nfail else 0


# --------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Run a Peregrine solve on a rented GPU box.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true",
                   help="check this script without needing a box")
    # No module-level default. `--rate` unset means "work it out": a saved
    # job knows which card it was planned for, and that is a far better
    # answer than a constant that happens to be an L40S price.
    p.add_argument("--rate", type=float, default=None,
                   help="rental price per hour (default: the job's card, "
                        f"else {DEFAULT_USD_PER_HOUR}, DigitalOcean's "
                        "on-demand single L40S)")
    sub = p.add_subparsers(dest="cmd")

    # --host on every command that talks to a box, because a box lives for
    # hours and the next one has a different address. Declared per subparser
    # rather than on the top-level parser: a subparser's default would
    # otherwise overwrite a value given before the subcommand with None.
    def box(name, help_):
        s = sub.add_parser(name, help=help_)
        s.add_argument("--host", metavar="USER@IP",
                       help="the box to use, for this command only "
                            "(overrides, and replaces, the remembered one)")
        return s

    s = sub.add_parser("host", help="remember which box to use")
    s.add_argument("host")

    s = box("provision", "install Julia and check the GPU")
    # `provision root@1.2.3.4` is the documented shape and predates --host.
    # It needs its own dest: sharing "host" with the option above would let
    # argparse's positional default of None overwrite a --host given before
    # the subcommand. cmd_provision folds the two back together.
    s.add_argument("host_pos", nargs="?", metavar="host",
                   help="the box, positionally -- same thing as --host")
    s.add_argument("--driver", action="store_true",
                   help="install the NVIDIA driver (plain OS images only)")
    s.add_argument("--quick", action="store_true",
                   help="skip the solver self-test")

    s = box("plan", "what the box would do with this config or saved job")
    s.add_argument("config", help="a config.json, a job directory, or the "
                                  "name of a saved job")

    s = box("run", "push, solve, pull, verify")
    s.add_argument("config", help="a config.json, a job directory, or the "
                                  "name of a saved job")
    s.add_argument("--no-pull", action="store_true",
                   help="leave the tables on the box")
    s.add_argument("--no-stream", action="store_true",
                   help="fetch every table at the end instead of each one as "
                        "it is finished")
    s.add_argument("--stream-to", metavar="DIR",
                   help="write the tables here instead of the config's "
                        "out_dir -- an SD card, say, when they will not fit "
                        "on the local disk")

    s = box("attach", "re-follow a solve already running")
    s.add_argument("--no-pull", action="store_true")
    s.add_argument("--no-stream", action="store_true")
    s.add_argument("--stream-to", metavar="DIR")

    s = box("benchmark", "measure this box's cell and disk rates")
    s.add_argument("--seconds", type=float, default=25.0,
                   help="roughly how long to spend on the cell-rate timing")

    box("status", "what the box is and what it is doing")
    box("pull", "fetch the last solve's tables")
    box("cost", "what this rental has run up")

    sub.add_parser("jobs", help="list the plans saved for later")
    sub.add_parser("forget", help="drop the remembered box and its clock")

    args = p.parse_args()
    if args.self_test:
        return self_test()
    if not args.cmd:
        p.print_help()
        return 2

    st = load_state()
    fn = {"host": cmd_host, "provision": cmd_provision, "plan": cmd_plan,
          "run": cmd_run, "attach": cmd_attach, "status": cmd_status,
          "pull": cmd_pull, "cost": cmd_cost, "jobs": cmd_jobs,
          "forget": cmd_forget, "benchmark": cmd_benchmark}[args.cmd]
    return fn(args, st)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n")
        sys.exit(130)
