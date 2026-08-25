#!/usr/bin/env python3
"""
Find the Pinpoint pod offsets from a rotate-in-place calibration log.

    py -3.12 calibration/find_pod_offsets.py <rotation_log.csv> [options]

If the pod offsets are right, spinning in place leaves the reported position
where it is. If they are wrong, the reported tracking point sits somewhere
other than the true centre of rotation, and spinning sweeps it around a
circle. The radius of that circle is the error.

Run this FIRST, before fitting a drivetrain model. A wrong offset injects a
rotation-dependent velocity into every sample of every later run, and the
drivetrain fit has no way to tell that apart from real dynamics -- it will
quietly absorb it and give you a model that is wrong wherever the robot turns.

Everything is in CENTIMETRES, matching the rest of Peregrine.

Requires numpy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_drivetrain as fd

# How big an offset has to be before it is worth changing anything. Below this
# the phantom velocity it creates is lost in the noise of everything else.
CENTRED_CM = 0.5
# Above this, it distorts the drivetrain fit badly enough to matter.
SERIOUS_CM = 2.0


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------

def estimate_from_position(t, x, y, h, drift=True):
    """
    Solve for the circle the tracking point sweeps.

        p(t) = p0 + vd*t + R(h) * delta

    Linear in all six unknowns, so plain least squares -- no iterative circle
    fit, and the heading supplies the phase so a full revolution is not
    required.

    The `vd*t` term matters more than it looks. A robot spinning in place
    still creeps across the floor, and without somewhere for that creep to go
    the fit charges it to `delta` instead. On a real 12.8 s spin, adding it cut
    the fit residual from 0.86 cm to 0.22 cm and halved the estimated offset.
    """
    n = len(x)
    tc = t - t.mean()
    c, s = np.cos(h), np.sin(h)
    ncol = 6 if drift else 4
    A = np.zeros((2 * n, ncol))
    A[0::2, 0] = 1.0; A[0::2, 2] = c; A[0::2, 3] = -s
    A[1::2, 1] = 1.0; A[1::2, 2] = s; A[1::2, 3] = c
    if drift:
        A[0::2, 4] = tc
        A[1::2, 5] = tc
    b = np.empty(2 * n)
    b[0::2] = x; b[1::2] = y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = b - A @ sol
    dof = max(2 * n - ncol, 1)
    cov = float(resid @ resid) / dof * np.linalg.pinv(A.T @ A)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return {
        "dx": sol[2], "dy": sol[3],
        "se_dx": se[2], "se_dy": se[3],
        "drift": (sol[4], sol[5]) if drift else (0.0, 0.0),
        "rmse": math.sqrt(float(resid @ resid) / (2 * n)),
    }


def estimate_from_velocity(w, h, vbx, vby, drift=True):
    """
    Fit the body-frame velocity a spin produces.

        v_body = R(-h) * vd  +  omega x delta

    which componentwise is

        v_x = -omega*delta_y + cos(h)*vd_x + sin(h)*vd_y
        v_y = +omega*delta_x - sin(h)*vd_x + cos(h)*vd_y

    Both components are fitted jointly because they share `delta` and `vd`.
    The drift enters through R(-h), so in the body frame it is a rotating
    signal rather than a constant -- an intercept cannot absorb it, which is
    why it has to be modelled explicitly.

    Independent of the position channel entirely, which is what makes this a
    real cross-check rather than a restatement.
    """
    n = len(w)
    c, s = np.cos(h), np.sin(h)
    ncol = 4 if drift else 2
    A = np.zeros((2 * n, ncol))
    # v_x rows: delta_x has no effect, delta_y enters as -omega
    A[0::2, 1] = -w
    # v_y rows: +omega * delta_x
    A[1::2, 0] = w
    if drift:
        A[0::2, 2] = c;  A[0::2, 3] = s
        A[1::2, 2] = -s; A[1::2, 3] = c
    b = np.empty(2 * n)
    b[0::2] = vbx; b[1::2] = vby
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = b - A @ sol
    dof = max(2 * n - ncol, 1)
    cov = float(resid @ resid) / dof * np.linalg.pinv(A.T @ A)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return {
        "dx": sol[0], "dy": sol[1],
        "se_dx": se[0], "se_dy": se[1],
        "drift": (sol[2], sol[3]) if drift else (0.0, 0.0),
        "rmse": math.sqrt(float(resid @ resid) / (2 * n)),
    }


# --------------------------------------------------------------------------
# Is this actually a rotate-in-place run?
# --------------------------------------------------------------------------

def check_run(sweep_deg, w, trans_rms, n_spin, args):
    """
    Decide whether the log is a usable spin, from the MOTION and the COMMANDS.

    Deliberately not from how well omega explains the measured velocity. A
    perfectly centred tracking point produces no rotation-induced velocity at
    all, so that test reads exactly the same as "this is not a spin" -- it
    would reject the one result you most want to see.
    """
    problems = []
    if sweep_deg < args.min_sweep:
        problems.append(
            "the robot only turned through %.0f deg; at least %.0f is needed "
            "(two or three full revolutions is ideal)"
            % (sweep_deg, args.min_sweep))
    if n_spin < args.min_samples:
        problems.append(
            "only %d samples are actually turning faster than %.2f rad/s; "
            "need %d" % (n_spin, args.min_omega, args.min_samples))
    if trans_rms > args.max_translate:
        problems.append(
            "translation was commanded (rms %.3f, limit %.3f) -- drive the "
            "robot in place, turn only" % (trans_rms, args.max_translate))
    return problems


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

def synth_spin(dx, dy, revs, drift=(0.0, 0.0), n=1500, dur=8.0, seed=4):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, dur, n)
    w = 2 * math.pi * revs / dur * (1.0 + 0.4 * np.sin(2 * math.pi * t / dur))
    h = np.cumsum(w) * (t[1] - t[0])
    c, s = np.cos(h), np.sin(h)
    x = 40.0 + drift[0] * t + c * dx - s * dy + rng.normal(0, 0.02, n)
    y = -15.0 + drift[1] * t + c * dy + s * dx + rng.normal(0, 0.02, n)
    # Body-frame velocity: the rotation term plus the drift rotated in.
    vbx = -w * dy + c * drift[0] + s * drift[1] + rng.normal(0, 2.0, n)
    vby = w * dx - s * drift[0] + c * drift[1] + rng.normal(0, 2.0, n)
    return t, x, y, h, w, vbx, vby


def self_test() -> int:
    print()
    print("=" * 74)
    print("  SELF-TEST -- recovering known offsets from synthetic spins")
    print("=" * 74)
    print("  All values in cm. The last two cases drift while spinning, which")
    print("  is what a real robot does and what used to corrupt the estimate.")
    print()
    print("  %8s %8s %8s   %9s %9s   %9s %9s"
          % ("true dx", "true dy", "drift", "pos dx", "pos dy", "vel dx", "vel dy"))
    cases = [((2.5, -1.8), 3.0, (0.0, 0.0)),
             ((0.0, 0.0), 2.5, (0.0, 0.0)),
             ((-11.2, 1.1), 2.0, (0.0, 0.0)),
             ((0.0, 0.0), 3.5, (0.6, -0.2)),
             ((1.5, -0.9), 3.0, (0.6, -0.2))]
    worst_pos = worst_vel = 0.0
    for (dx, dy), revs, dr in cases:
        t, x, y, h, w, vbx, vby = synth_spin(dx, dy, revs, dr)
        pos = estimate_from_position(t, x, y, h)
        vel = estimate_from_velocity(w, h, vbx, vby)
        worst_pos = max(worst_pos, abs(pos["dx"] - dx), abs(pos["dy"] - dy))
        worst_vel = max(worst_vel, abs(vel["dx"] - dx), abs(vel["dy"] - dy))
        print("  %8.2f %8.2f %8s   %9.3f %9.3f   %9.3f %9.3f"
              % (dx, dy, "yes" if any(dr) else "no",
                 pos["dx"], pos["dy"], vel["dx"], vel["dy"]))

    # And the case that used to be rejected outright: centred, so omega
    # explains none of the velocity.
    t, x, y, h, w, vbx, vby = synth_spin(0.0, 0.0, 3.0, (0.5, 0.0))
    pos = estimate_from_position(t, x, y, h)
    centred_ok = math.hypot(pos["dx"], pos["dy"]) < CENTRED_CM

    print()
    print("  worst error -- position %.4f cm, velocity %.4f cm"
          % (worst_pos, worst_vel))
    print("  a centred robot that drifts is reported as centred: %s"
          % ("yes" if centred_ok else "NO"))
    ok = worst_pos < 0.05 and worst_vel < 0.8 and centred_ok
    print()
    print("  %s  (position within 0.05 cm, velocity within 0.8, centred case "
          "recognised)" % ("PASS" if ok else "FAIL"))
    print()
    print("  The two estimators are held to different standards on purpose.")
    print("  Position noise is a fraction of a millimetre while the velocity")
    print("  channel carries several cm/s, so demanding equal accuracy would")
    print("  fail a good estimator for doing its best with worse data.")
    print()
    return 0 if ok else 1


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Find Pinpoint pod offsets from a rotate-in-place log. "
                    "All values in centimetres.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("log", nargs="?",
                   help="a calibration CSV of the robot spinning in place")
    p.add_argument("--self-test", action="store_true",
                   help="recover known offsets from synthetic spins, then exit")
    p.add_argument("--current-offsets", nargs=2, type=float, metavar=("X", "Y"),
                   help="the offsets currently set on the robot, in CM")
    p.add_argument("--min-omega", type=float, default=0.30,
                   help="a sample counts as turning above this, rad/s")
    p.add_argument("--min-sweep", type=float, default=180.0,
                   help="the run must turn through at least this many degrees")
    p.add_argument("--min-samples", type=int, default=150,
                   help="minimum turning samples")
    p.add_argument("--max-translate", type=float, default=0.05,
                   help="rms commanded translation allowed before the run is "
                        "judged not to be a pure spin")
    p.add_argument("--no-drift", dest="drift", action="store_false",
                   help="do not model the slow drift a spinning robot creeps "
                        "with (modelling it is almost always right)")
    p.add_argument("--segment", type=int, default=None)
    p.add_argument("-o", "--out", default=None,
                   help="write the result as JSON here")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.log:
        print("error: give a log path, or --self-test", file=sys.stderr)
        return 2
    if not os.path.exists(args.log):
        print("error: no such file: " + args.log, file=sys.stderr)
        return 2

    segs, _ = fd.load_segments(args.log)
    usable = [s for s in segs if fd.validate(s)[0]]
    if not usable:
        for s in segs:
            print("  unusable segment %d: %s" % (s.index, fd.validate(s)[1]),
                  file=sys.stderr)
        print("error: no usable segment in this log", file=sys.stderr)
        return 1
    seg = (next((s for s in usable if s.index == args.segment), None)
           if args.segment is not None else max(usable, key=lambda s: s.n))

    d = seg.raw[np.argsort(seg.raw[:, 0], kind="stable")]
    t = d[:, 0] / 1000.0
    mec = d[:, 1:5] @ fd.MEC_MIX[:3].T
    x, y = d[:, 5], d[:, 6]
    h = np.unwrap(d[:, 7])
    xv, yv, w = d[:, 8], d[:, 9], d[:, 10]

    dup = np.zeros(len(t), bool)
    dup[1:] = (np.diff(x) == 0) & (np.diff(y) == 0) & (np.diff(h) == 0)
    spin = (~dup) & (np.abs(w) > args.min_omega)

    c, s = np.cos(h), np.sin(h)
    vbx = c * xv + s * yv
    vby = -s * xv + c * yv

    sweep = math.degrees(np.ptp(h))
    trans_rms = float(np.sqrt((np.hypot(mec[:, 0], mec[:, 1]) ** 2).mean()))

    print()
    print("=" * 74)
    print("  PINPOINT POD OFFSETS          (all values in cm)")
    print("=" * 74)
    print("  log             %s" % os.path.basename(args.log))
    print("  duration        %.1f s, %d samples" % (t.max() - t.min(), len(t)))
    print("  turned through  %.0f deg  (%.2f revolutions)"
          % (sweep, sweep / 360.0))
    print("  turning samples %d  (|omega| > %.2f rad/s, peak %.2f)"
          % (spin.sum(), args.min_omega, np.abs(w).max()))
    print("  translation cmd rms %.4f  (limit %.3f)"
          % (trans_rms, args.max_translate))

    problems = check_run(sweep, w, trans_rms, int(spin.sum()), args)
    if problems:
        print()
        print("  " + "!" * 68)
        print("  !! This is not a usable rotate-in-place run:")
        for pr in problems:
            print("  !!   - " + pr)
        print("  !!")
        print("  !! Record one that only rotates: spin the robot in place,")
        print("  !! two or three revolutions each direction, no translation.")
        print("  " + "!" * 68)
        print()
        return 1

    pos = estimate_from_position(t[spin], x[spin], y[spin], h[spin], args.drift)
    vel = estimate_from_velocity(w[spin], h[spin], vbx[spin], vby[spin],
                                 args.drift)

    dx, dy = pos["dx"], pos["dy"]
    mag = math.hypot(dx, dy)
    # How far the true offset could be from what was measured.
    r95 = 1.96 * math.hypot(pos["se_dx"], pos["se_dy"])
    gap = math.hypot(dx - vel["dx"], dy - vel["dy"])

    print()
    print("  %-28s %9s %9s %10s" % ("estimator", "delta_x", "delta_y", "fit rms"))
    print("  " + "-" * 60)
    print("  %-28s %9.3f %9.3f %10.3f"
          % ("position circle (primary)", dx, dy, pos["rmse"]))
    print("  %-28s %9.3f %9.3f %10.3f"
          % ("velocity vs omega (check)", vel["dx"], vel["dy"], vel["rmse"]))
    print("  " + "-" * 60)
    print("  %-28s %9.3f %9.3f" % ("difference", dx - vel["dx"], dy - vel["dy"]))
    if args.drift:
        print("  robot also drifted at (%.2f, %.2f) cm/s while spinning, "
              "which is\n  modelled and removed rather than charged to the "
              "offset." % pos["drift"])

    print()
    print("  Offset of the reported tracking point from the true centre of")
    print("  rotation:  %.3f cm   (+/- %.3f at 95%% confidence)" % (mag, r95))
    peak_phantom = np.abs(w).max() * mag
    print("  At this run's peak %.1f rad/s that creates %.1f cm/s of phantom"
          % (np.abs(w).max(), peak_phantom))
    print("  sideways velocity whenever the robot turns.")

    warnings = []
    if gap > max(0.5, 4.0 * math.hypot(vel["se_dx"], vel["se_dy"])):
        warnings.append("the two estimators disagree by %.2f cm, more than "
                        "noise explains" % gap)

    print()
    print("=" * 74)
    print("  WHAT TO CHANGE ON THE ROBOT")
    print("=" * 74)

    centred = mag < CENTRED_CM
    corrected = None
    if centred:
        print()
        print("    Nothing. The tracking point is already centred.")
        print()
        print("    The measured offset is %.2f cm, below the %.1f cm that would"
              % (mag, CENTRED_CM))
        print("    be worth acting on. It is only %.1f cm/s of phantom velocity"
              % peak_phantom)
        print("    at full spin, which is lost among everything else.")
        print()
        print("    Leave setOffsets() as it is and move on to step 1.")
        print()
        print("    (Note the measurement is precise enough that this offset is")
        print("     statistically real -- it is simply too small to matter.")
        print("     Precision and importance are different questions.)")
    else:
        # v = omega x delta, so shifting the configured offsets by -delta
        # moves the reported tracking point onto the centre of rotation.
        adj_x, adj_y = dy, dx
        print()
        print("    The tracking point is %.2f cm off centre. Adjust the pod"
              % mag)
        print("    offsets in your Pinpoint setup by:")
        print()
        print("        X offset:  %+.2f cm" % adj_x)
        print("        Y offset:  %+.2f cm" % adj_y)
        if args.current_offsets:
            cx, cy = args.current_offsets
            corrected = [cx + adj_x, cy + adj_y]
            print()
            print("    You said the robot currently uses:")
            print()
            print("        odo.setOffsets(%.2f, %.2f, DistanceUnit.CM);"
                  % (cx, cy))
            print()
            print("    Change that line to:")
            print()
            print("        odo.setOffsets(%.2f, %.2f, DistanceUnit.CM);"
                  % (corrected[0], corrected[1]))
        else:
            print()
            print("    Add those to whatever your setOffsets() call uses now.")
            print("    Re-run with --current-offsets X Y (in cm) and this will")
            print("    print the finished line for you.")
        if mag >= SERIOUS_CM:
            print()
            print("    This is large enough to distort the drivetrain fit.")
            print("    Fix it before recording a driving run.")
        print()
        print("    Then spin again and re-run this. The offset should drop")
        print("    toward zero. If it roughly DOUBLES instead, your firmware")
        print("    takes the opposite sign -- negate both adjustments and")
        print("    reapply. One iteration settles it permanently.")

    if warnings:
        print()
        print("  " + "!" * 68)
        for wn in warnings:
            print("  !! " + wn)
        print("  " + "!" * 68)
    print()

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({
                "source": os.path.abspath(args.log),
                "units": "cm",
                "delta": [dx, dy],
                "delta_magnitude": mag,
                "confidence_95_radius": r95,
                "centred": bool(centred),
                "centred_threshold": CENTRED_CM,
                "delta_from_velocity": [vel["dx"], vel["dy"]],
                "estimator_gap": gap,
                "drift_cm_s": list(pos["drift"]),
                "adjust_offsets_by": (None if centred else {"x": dy, "y": dx}),
                "corrected_offsets": ({"x": corrected[0], "y": corrected[1]}
                                      if corrected else None),
                "samples_used": int(spin.sum()),
                "sweep_deg": sweep,
                "warnings": warnings,
            }, fh, indent=2)
        print("  wrote " + args.out)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
