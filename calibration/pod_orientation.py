#!/usr/bin/env python3
"""
Check that the odometry pods point the way the drivetrain drives.

The Pinpoint reports x and y from two pods, one along the robot and one across
it. Each can be plugged into the wrong port or set to the wrong direction, and
nothing on the robot complains. The pose it reports is then mirrored or
rotated, and a calibration log recorded that way looks fine until a full solve
has been paid for and the robot drives the wrong way.

The wheel commands already say which way the robot went. The robot's mixer
(see MEC_MIX in fit_drivetrain.py) is

    FR = drive + strafe - turn      BR = drive - strafe - turn
    FL = drive - strafe + turn      BL = drive + strafe + turn

so with every motor turning forward on positive power

    +drive   moves the robot forward    -> +x on the Pinpoint
    +strafe  moves it left              -> +y
    +turn    spins it clockwise         -> -heading

Heading comes from the Pinpoint's IMU, not the pods, so a pod mistake cannot
touch it. That makes `turn` the anchor: if +turn gives -heading the motors are
as the mixer expects and +drive, +strafe must give +x, +y; if +turn gives
+heading every motor is inverted and both expectations flip with it. Either
way it is the pods, and only the pods, that are being tested.

The check is deliberately crude. It regresses body-frame acceleration on the
command (plus velocity and a constant, to soak up drag) and asks only which
signed permutation of the {x, y} axes the command gains most resemble. A pod
mistake M maps the true gains B to M.B exactly, and it is a sign flip or a
swap, never a small angle, so crude is enough. Acceleration rather than
velocity because it answers the command at once, where velocity lags it by
however long this particular drivetrain takes to spin up.

numpy only; imported by fit_drivetrain.py, diagnose_fit.py and
find_pod_offsets.py. The wizard, which is stdlib only, reads the verdict back
from drivetrain_fit.json instead.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

# Below this rms a command direction was not exercised enough to say anything.
MIN_COMMAND_RMS = 0.03
# The acceleration each command explains must be at least this, cm/s^2 rms.
# Anything less is lost in wheel slip and odometry noise.
MIN_RESPONSE = 15.0
# Likewise for turn -> heading, rad/s^2 rms, before it is trusted as the anchor.
MIN_TURN_RESPONSE = 0.3
# Each command's response must point within 45 degrees of an axis to call it.
MIN_ALIGN = math.cos(math.radians(45.0))

AXES = ("x", "y")
COMMANDS = ("drive", "strafe")

# The 8 signed permutations of two axes: every way two pods can be swapped
# and/or reversed. The identity is correct wiring.
SIGNED_PERMS = []
for _perm in itertools.permutations(range(2)):
    for _signs in itertools.product((1, -1), repeat=2):
        _m = np.zeros((2, 2))
        for _r, (_col, _s) in enumerate(zip(_perm, _signs)):
            _m[_r, _col] = _s
        SIGNED_PERMS.append(_m)


def _describe(m: np.ndarray) -> list[str]:
    """Say in words what each reported axis is actually measuring."""
    words = {0: "forward/back", 1: "sideways (left/right)"}
    out = []
    for r, ax in enumerate(AXES):
        col = int(np.argmax(np.abs(m[r])))
        rev = m[r, col] < 0
        right = col == r and not rev
        out.append("reported %s follows the robot's %s motion%s%s"
                   % (ax, words[col], ", REVERSED" if rev else "",
                      "  (correct)" if right else ""))
    return out


# The direction a body-frame response points, in the robot's own words.
_DIRECTION = {(0, 1): "forward", (0, -1): "backward",
              (1, 1): "left", (1, -1): "right"}


def _observed(G: np.ndarray) -> str:
    """What each command did, taking the pods as right.

    The other reading of a mismatch. A log compares the pods with the motors
    and cannot say which of the two is wrong: a motor in the wrong port, set
    to the wrong direction, or logged under the wrong name moves the robot
    off the mixer's directions in exactly the way a pod mistake seems to.
    Saying what the wheels did in plain directions lets someone who knows
    their pods are right see the motor mistake instead.
    """
    def where(col):
        ax = int(np.argmax(np.abs(col)))
        return _DIRECTION[(ax, 1 if col[ax] > 0 else -1)]
    return ("If the pods are right, then in this log 'drive' (all four motors "
            "forward) moved the robot %s, 'strafe' moved it %s, and '+turn' "
            "spun it %s -- where the logger's mixer expects forward, left and "
            "clockwise. A motor plugged into the wrong port, set to the wrong "
            "direction, or logged under the wrong name does exactly this, and "
            "no log can tell it apart from a pod mistake."
            % (where(G[:2, 0]), where(G[:2, 1]),
               "counter-clockwise" if G[2, 2] > 0 else "clockwise"))


# A pod check that needs no motors, so it is the one that can settle which
# side is wrong.
PUSH_TEST = ("Before changing anything, check the pods alone: with the motors "
             "off, push the robot forward by hand and watch the Pinpoint's x "
             "rise, then push it to its left and watch y rise. If both do, "
             "the pods are right and the motors are not wired, named or "
             "directed the way the calibration OpMode's mixer assumes (see "
             "the note below) -- fix that, not the pods.")


def _fixes(m: np.ndarray) -> list[str]:
    """What to change on the robot to turn `m` into the identity."""
    fixes = [PUSH_TEST]
    swapped = m[0, 0] == 0
    if swapped:
        fixes.append("Swap the two pod cables between the X and Y ports on the "
                     "Pinpoint. The pod in the X port must be the one that "
                     "rolls when the robot drives forward.")
        # After the swap, the X port carries what the Y port did.
        m = m[::-1]
    flips = [ax for r, ax in enumerate(AXES) if m[r, r] < 0]
    if flips:
        fixes.append("Reverse the %s pod%s: in odo.setEncoderDirections(x, y) "
                     "flip %s between FORWARD and REVERSED."
                     % (" and ".join(f.upper() for f in flips),
                        "s" if len(flips) > 1 else "",
                        "both arguments" if len(flips) > 1 else
                        "the %s argument" % ("first" if flips[0] == "x"
                                             else "second")))
    if swapped:
        fixes.append("setOffsets() describes the physical pods, so after "
                     "swapping the cables its two values no longer match. "
                     "Re-run step 0 (pod offsets) before anything else.")
    fixes.append("Then record a NEW calibration log. This log's positions are "
                 "mirrored or rotated, and a fit or solve built on it drives "
                 "the robot the wrong way.")
    return fixes


def check(u_mec: np.ndarray, v_body: np.ndarray, a_body: np.ndarray,
          valid: np.ndarray | None = None) -> dict:
    """
    Compare what the wheels were told to do with what the odometry saw.

    u_mec   (n,3)  commanded [drive, strafe, turn]
    v_body  (n,3)  body-frame [v_x, v_y, omega] from the odometry, cm/s
    a_body  (n,3)  body-frame [a_x, a_y, alpha] from the odometry, cm/s^2
    valid   (n,)   optional mask of samples that can be trusted

    Returns a JSON-ready dict. `status` is one of

        "ok"            pods agree with the wheels
        "mismatch"      pods are swapped and/or reversed -- clear evidence
        "suspect"       probably wrong, but the evidence is weak
        "inconclusive"  this log cannot tell (e.g. a spin-only run)
    """
    u = np.asarray(u_mec, float)[:, :3]
    v = np.asarray(v_body, float)[:, :3]
    a = np.asarray(a_body, float)[:, :3]
    ok = (np.all(np.isfinite(u), axis=1) & np.all(np.isfinite(v), axis=1)
          & np.all(np.isfinite(a), axis=1))
    if valid is not None:
        ok &= np.asarray(valid, bool)

    res = {"status": "inconclusive", "reason": "", "anchor": None,
           "gain": None, "alignment": None, "best_match": None,
           "explanation": [], "fixes": [], "notes": []}

    if ok.sum() < 30:
        res["reason"] = "too few usable samples to check"
        return res

    X = np.column_stack([u[ok], v[ok], np.ones(ok.sum())])
    Y = a[ok]
    try:
        coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    except np.linalg.LinAlgError:
        res["reason"] = "command matrix is singular"
        return res
    G = coef[:3].T                       # G[i, j] = d a_i / d u_j
    res["gain"] = G.tolist()

    urms = np.sqrt((u[ok] ** 2).mean(axis=0))
    ustd = u[ok].std(axis=0)

    # --- anchor on the heading, which the pods cannot affect ------------------
    turn_resp = abs(G[2, 2]) * ustd[2]
    if urms[2] >= MIN_COMMAND_RMS and turn_resp >= MIN_TURN_RESPONSE:
        motors = -1.0 if G[2, 2] > 0 else 1.0
        res["anchor"] = "turn"
        if motors < 0:
            res["notes"].append(
                "+turn spun the robot counter-clockwise, the opposite of what "
                "the mixer expects. If every motor is simply inverted that is "
                "harmless for the fit, and the pod check allows for it; it is "
                "also what motors in the wrong ports can look like.")
    else:
        motors = 1.0
        res["anchor"] = "assumed"
        res["notes"].append(
            "The run barely turned, so the check assumes the motors run "
            "forward on positive power rather than confirming it.")
    res["motor_sign"] = motors

    # --- which way did drive and strafe move the reported x, y? --------------
    short = [c for j, c in enumerate(COMMANDS) if urms[j] < MIN_COMMAND_RMS]
    if short:
        res["reason"] = ("this run hardly commanded %s, so the pod directions "
                         "cannot be checked from it" % " or ".join(short))
        return res

    T = motors * G[:2, :2]               # columns: response to drive, strafe
    resp = np.linalg.norm(T, axis=0) * ustd[:2]
    weak = [c for j, c in enumerate(COMMANDS) if resp[j] < MIN_RESPONSE]
    if weak:
        res["reason"] = ("the odometry barely moved in response to %s "
                         "(%.0f cm/s^2 explained) -- wheels slipping, or the "
                         "robot held still" % (" or ".join(weak), resp.min()))
        return res

    D = T / np.linalg.norm(T, axis=0)
    scores = [float(np.trace(P.T @ D)) for P in SIGNED_PERMS]
    best = SIGNED_PERMS[int(np.argmax(scores))]
    align = [float(best[:, j] @ D[:, j]) for j in range(2)]
    res["alignment"] = dict(zip(COMMANDS, align))
    res["best_match"] = best.astype(int).tolist()
    res["explanation"] = _describe(best)
    confident = min(align) >= MIN_ALIGN

    if np.array_equal(best, np.eye(2)):
        if confident:
            res["status"] = "ok"
            res["reason"] = "both pods agree with the wheel commands"
        else:
            res["status"] = "suspect"
            res["reason"] = ("the pods agree with the wheels only loosely -- "
                             "the robot moved well off the direction it was "
                             "driven")
            res["fixes"] = [
                "Check the pods are plugged in to the correct ports and set "
                "to the correct directions, and that the log is from a clean "
                "run on a flat floor, then record it again."]
        return res

    res["status"] = "mismatch" if confident else "suspect"
    res["reason"] = ("the odometry does not move the way the wheels drove: "
                     "the pods look %s" % _kind(best))
    res["fixes"] = _fixes(best)
    res["notes"].append(_observed(G))
    if np.array_equal(best, -np.eye(2)):
        res["notes"].append(
            "Both axes reversed is also what a Pinpoint mounted 180 degrees "
            "round, or a turn direction wired backwards in the mixer, would "
            "look like.")
    return res


def _kind(m: np.ndarray) -> str:
    swapped = m[0, 0] == 0
    flips = int((m.sum(axis=1) < 0).sum())
    parts = []
    if swapped:
        parts.append("swapped (X and Y in each other's ports)")
    if flips == 2:
        parts.append("both reversed")
    elif flips == 1:
        r = int(np.argmin(m.sum(axis=1)))
        parts.append("reversed on the reported %s axis" % AXES[r])
    return " and ".join(parts)


def banner(res: dict, width: int = 74) -> list[str]:
    """The warning box to print, or [] when there is nothing to warn about."""
    if res.get("status") not in ("mismatch", "suspect"):
        return []
    bar = "!" * width
    head = ("ODOMETRY PODS LOOK MISORIENTED" if res["status"] == "mismatch"
            else "ODOMETRY PODS MAY BE MISORIENTED")
    lines = [bar, "!!  " + head, "!!"]
    for chunk in _wrap(res["reason"][0].upper() + res["reason"][1:] + ".",
                       width - 5):
        lines.append("!!  " + chunk)
    if res.get("explanation"):
        lines.append("!!")
        for e in res["explanation"]:
            lines.append("!!    - " + e)
    if res.get("fixes"):
        lines.append("!!")
        lines.append("!!  How to fix it:")
        for i, f in enumerate(res["fixes"], 1):
            chunks = _wrap(f, width - 9)
            lines.append("!!    %d. %s" % (i, chunks[0]))
            lines.extend("!!       " + ch for ch in chunks[1:])
    for n in res.get("notes", []):
        lines.append("!!")
        for j, ch in enumerate(_wrap(n, width - 11)):
            lines.append(("!!  note: " if j == 0 else "!!        ") + ch)
    lines.append(bar)
    return lines


def summary(res: dict) -> str:
    """One line for a report that has room for nothing more."""
    st = res.get("status", "inconclusive")
    if st == "ok":
        return "pod orientation: OK -- " + res["reason"]
    if st == "inconclusive":
        return "pod orientation: not checked -- " + res["reason"]
    return "pod orientation: %s -- %s" % (st.upper(), res["reason"])


def _wrap(s: str, width: int) -> list[str]:
    out, line = [], ""
    for w in s.split():
        if line and len(line) + 1 + len(w) > width:
            out.append(line)
            line = w
        else:
            line = (line + " " + w) if line else w
    if line:
        out.append(line)
    return out or [""]


# --------------------------------------------------------------------------
# Self-test: every wiring mistake is recognised, and correct wiring passes
# --------------------------------------------------------------------------

def _synth(m: np.ndarray, motor_sign: float = 1.0, seed: int = 3,
           turn: bool = True, n: int = 3000, hz: float = 110.0):
    """A robot obeying a = B.u + A.v, seen through pod wiring `m`."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / hz
    tt = t[:, None]
    u = np.clip(np.sin(2 * np.pi * np.array([0.21, 0.13, 0.17]) * tt
                       + np.array([0.0, 1.3, 2.1]))
                + 0.4 * np.sin(2 * np.pi * np.array([0.53, 0.47, 0.61]) * tt),
                -1, 1)
    if not turn:
        u[:, 2] = 0.0
    # Clockwise for +turn, and plenty of cross-coupling, as on a real robot.
    B = np.array([[380.0, 60.0, 30.0], [-50.0, 250.0, -20.0],
                  [-1.0, 2.0, -40.0]]) * motor_sign
    A = np.diag([-0.8, -0.75, -1.9])
    v = np.zeros((n, 3))
    a = np.zeros((n, 3))
    for i in range(n):
        a[i] = B @ u[i] + A @ v[i]
        if i + 1 < n:
            v[i + 1] = v[i] + a[i] / hz
    for arr in (v, a):
        arr[:, :2] = arr[:, :2] @ m.T
    v += rng.normal(0, [3.0, 3.0, 0.05], v.shape)
    a += rng.normal(0, [60.0, 60.0, 2.0], a.shape)
    return u, v, a


def self_test() -> int:
    ok = True
    print()
    print("  pod orientation self-test")
    print("  " + "-" * 60)
    for m in SIGNED_PERMS:
        for motor_sign in (1.0, -1.0):
            r = check(*_synth(m, motor_sign))
            want = "ok" if np.array_equal(m, np.eye(2)) else "mismatch"
            got_m = np.array(r["best_match"]) if r["best_match"] else None
            good = r["status"] == want and np.array_equal(got_m, m)
            ok &= good
            print("  %-22s motors %+d  -> %-9s %s"
                  % (str(m.astype(int).tolist()), motor_sign, r["status"],
                     "ok" if good else "FAIL"))
    # A spin-only run has nothing to say about the translation pods.
    u, v, a = _synth(np.eye(2))
    u[:, :2] = 0.0
    r = check(u, v, a)
    good = r["status"] == "inconclusive"
    ok &= good
    print("  %-38s -> %-9s %s" % ("spin only", r["status"],
                                   "ok" if good else "FAIL"))
    # No turning: fall back to assuming the motors are the right way round.
    r = check(*_synth(np.diag([1.0, -1.0]), turn=False))
    good = r["status"] == "mismatch" and r["anchor"] == "assumed"
    ok &= good
    print("  %-38s -> %-9s %s" % ("no turning, Y reversed", r["status"],
                                   "ok" if good else "FAIL"))
    print()
    print("  %s" % ("PASS" if ok else "FAIL"))
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(self_test())
