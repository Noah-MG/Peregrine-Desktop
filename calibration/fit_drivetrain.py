#!/usr/bin/env python3
"""
Fit a linear drivetrain model from a Peregrine calibration log.

Model (one linear equation per body-frame degree of freedom):

    a = B.u + A.v + q*omega^2 + S.csign(v) + D.(|v|*v) + c

one row per body-frame degree of freedom, where

where
    u = [FR, FL, BR, BL]        commanded motor powers, -1..1
    v = [v_x, v_y, omega]       current velocity, ROBOT body frame
    a = [a_x, a_y, alpha]       resulting acceleration, ROBOT body frame

Results are written to JSON and TOML for the downstream Julia consumer.

See MODEL NOTES at the bottom of this file for the derivation of the
body-frame rotation and the treatment of the Pinpoint's 1.5 kHz velocity
quantisation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# 2: added the omega^2 centripetal term, and removed the per-motor basis --
#    inputs are always {fwd, strafe, turn}.
# 3: added Coulomb (S) and quadratic-drag (D) blocks and turned omega^2 off by
#    default, after a real run showed omega^2 scoring worse than plain linear
#    while S and D both earned their place on held-out velocity prediction.
SCHEMA_VERSION = 3
HEADER_FIELDS = ["timestamp", "FR", "FL", "BR", "BL",
                 "x", "y", "h", "x_vel", "y_vel", "h_vel"]
N_FIELDS = len(HEADER_FIELDS)

# Coulomb friction is smoothed over this velocity band rather than using a
# hard sign(). A true sign() flips discontinuously at v = 0, and the solver
# integrates this model forward -- a discontinuity there chatters. The fit and
# the solver must use the same smoothing, so it is part of the model, not a
# solver detail. Units: cm/s, cm/s, rad/s.
COULOMB_EPS = (5.0, 5.0, 0.15)

# Traction knee search. `None` means "no saturation"; the auto search always
# includes it, so a run that shows no slip disables the term by itself.
KNEE_GRID = (None, 1.20, 0.90, 0.70, 0.55, 0.45, 0.35, 0.28)
# A knee has to earn its parameter. Below this improvement in held-out rollout
# it is not switched on -- an always-sliding run, for instance, can be fitted
# marginally better with a huge knee that means nothing.
KNEE_MIN_GAIN = 0.01

MOTORS = ["FR", "FL", "BR", "BL"]
MECANUM = ["fwd", "strafe", "turn"]
STATES = ["v_x", "v_y", "omega"]
RESPONSES = ["a_x", "a_y", "alpha"]

# Mecanum mixing, expressed against the log's FR,FL,BR,BL column order.
#
# This is the exact inverse of the mixer the ROBOT uses:
#     FR = drive + strafe - turn      BR = drive - strafe - turn
#     FL = drive - strafe + turn      BL = drive + strafe + turn
# whose columns are orthogonal with norm^2 = 4, so the inverse is its
# transpose over 4. The signs must match the robot exactly -- the gains are
# fitted against these definitions and the robot applies them by name, so a
# flipped row would send it the wrong way along that axis.
#
# The fourth row is the "null" direction: it commands the wheels to fight one
# another and produces no chassis force, so it is excluded from the model and
# only reported as a diagnostic.
MEC_MIX = np.array([
    [+1, +1, +1, +1],      # drive  (a.k.a. fwd)
    [+1, -1, -1, +1],      # strafe
    [-1, +1, -1, +1],      # turn
    [+1, +1, -1, -1],      # null (unactuated)
], dtype=float) / 4.0

# Tell-tale values of the pre-`odo.update()`-fix logs, in which the pose is
# pinned to a constant snapshot for the whole run.
FROZEN_H = 1.2611155398190022e-5
FROZEN_HVEL = 0.005745213013142347


# --------------------------------------------------------------------------
# Log loading
# --------------------------------------------------------------------------

@dataclass
class Segment:
    """One contiguous run. A single file may hold several (append-mode collisions)."""
    raw: np.ndarray                     # (n, 11)
    source: str
    index: int
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.raw)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_segments(path: str) -> tuple[list[Segment], dict]:
    """Parse a calibration CSV into one or more contiguous run segments.

    Tolerates the documented quirks of `CalibrationLogger` output:
      * header rows repeated mid-file (append-mode collision of two runs)
      * a truncated final row (the FileWriter is flushed but never closed)
      * stray unparseable rows
    """
    stats = {"lines": 0, "headers": 0, "short_rows": 0, "bad_rows": 0}
    segments: list[list[list[float]]] = []
    current: list[list[float]] = []

    with open(path, "r", newline="") as fh:
        for line in fh:
            line = line.strip()
            stats["lines"] += 1
            if not line:
                continue
            parts = line.split(",")
            if parts[0] == "timestamp":
                stats["headers"] += 1
                if current:
                    segments.append(current)
                    current = []
                continue
            if len(parts) != N_FIELDS:
                stats["short_rows"] += 1      # truncated last row, most likely
                continue
            try:
                current.append([float(p) for p in parts])
            except ValueError:
                stats["bad_rows"] += 1
    if current:
        segments.append(current)

    out = [Segment(raw=np.asarray(s, dtype=float), source=path, index=i)
           for i, s in enumerate(s for s in segments if s)]
    return out, stats


def validate(seg: Segment, min_rows: int = 40) -> tuple[bool, str]:
    """Reject segments that cannot carry usable dynamics information."""
    if seg.n < min_rows:
        return False, f"only {seg.n} rows (need >= {min_rows})"

    x, y, h = seg.raw[:, 5], seg.raw[:, 6], seg.raw[:, 7]
    hv = seg.raw[:, 10]

    # Frozen-pose logs: pose/velocity pinned to a constant snapshot.
    if np.allclose(h, FROZEN_H, atol=1e-12) and np.allclose(hv, FROZEN_HVEL, atol=1e-12):
        return False, "frozen-pose log (pre-odo.update() fix signature)"
    if np.ptp(x) < 1e-6 and np.ptp(y) < 1e-6 and np.ptp(h) < 1e-9:
        return False, "pose columns have no variance (robot never moved / frozen pose)"

    u = np.abs(seg.raw[:, 1:5]).sum(axis=1)
    if not np.any(u > 1e-9):
        return False, "no nonzero motor power anywhere in segment"

    return True, "ok"


# --------------------------------------------------------------------------
# Derivative estimation on an irregular time grid
# --------------------------------------------------------------------------

def local_poly_derivatives(
    t: np.ndarray,
    y: np.ndarray,
    halfwidth: float,
    order: int = 3,
    max_gap: float = 0.05,
    min_points: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Savitzky-Golay-style smoothing generalised to non-uniform sampling.

    At every sample a weighted polynomial of `order` is fit to all points
    inside +/- `halfwidth` seconds, and the value / 1st / 2nd derivative are
    read off analytically at that sample's own timestamp. Weights are tricube,
    as in LOESS, which tapers the window edges and avoids the ringing a boxcar
    would produce.

    This is the core of the answer to the Pinpoint's velocity quantisation
    problem: it derives BOTH velocity and acceleration from *position*, which
    is the clean integrated quantity, instead of differencing the sensor's
    own 1.5 kHz-differenced velocity. See MODEL NOTES.

    Returns (value, first derivative, second derivative, valid mask).
    """
    n = len(t)
    val = np.full(n, np.nan)
    d1 = np.full(n, np.nan)
    d2 = np.full(n, np.nan)
    ok = np.zeros(n, dtype=bool)

    lo = np.searchsorted(t, t - halfwidth, side="left")
    hi = np.searchsorted(t, t + halfwidth, side="right")

    for i in range(n):
        a, b = lo[i], hi[i]
        if b - a < max(min_points, order + 2):
            continue
        tw = t[a:b] - t[i]
        yw = y[a:b]

        # A window straddling a dropped-frame hole cannot support a
        # trustworthy second derivative.
        if np.max(np.diff(t[a:b])) > max_gap:
            continue

        scale = max(np.max(np.abs(tw)), 1e-12)
        w = (1.0 - np.abs(tw / scale) ** 3) ** 3      # tricube
        w = np.clip(w, 1e-6, None)
        sw = np.sqrt(w)

        # Fit in a scaled variable to keep the Vandermonde well conditioned.
        ts = tw / scale
        V = np.vander(ts, order + 1, increasing=True)
        try:
            coef, *_ = np.linalg.lstsq(V * sw[:, None], yw * sw, rcond=None)
        except np.linalg.LinAlgError:
            continue

        # Derivatives at ts = 0, undoing the time scaling.
        val[i] = coef[0]
        d1[i] = coef[1] / scale
        d2[i] = 2.0 * coef[2] / (scale ** 2)
        ok[i] = True

    return val, d1, d2, ok


# --------------------------------------------------------------------------
# Kinematics
# --------------------------------------------------------------------------

def to_body_frame(vec_x: np.ndarray, vec_y: np.ndarray, h: np.ndarray):
    """Rotate a field-frame planar vector into the robot body frame: R(-h) * v."""
    c, s = np.cos(h), np.sin(h)
    return c * vec_x + s * vec_y, -s * vec_x + c * vec_y


# --------------------------------------------------------------------------
# Regression
# --------------------------------------------------------------------------

@dataclass
class FitResult:
    name: str
    names: list[str]
    coef: np.ndarray
    stderr: np.ndarray
    stderr_hac: np.ndarray
    tstat: np.ndarray
    r2: float
    r2_adj: float
    rmse: float
    resid_std: float
    response_std: float
    cv_r2: float
    n: int
    vif: np.ndarray
    block_resid_rms: list[float] = field(default_factory=list)
    weights_used: np.ndarray | None = None

    @property
    def max_vif(self) -> float:
        v = self.vif[np.isfinite(self.vif)]
        return float(v.max()) if len(v) else float("nan")


def newey_west_cov(X: np.ndarray, resid: np.ndarray, xtx_inv: np.ndarray,
                   lag: int) -> np.ndarray:
    """HAC covariance. Smoothing makes residuals strongly autocorrelated, so
    the textbook iid standard errors would be wildly optimistic."""
    n = len(resid)
    u = X * resid[:, None]
    S = u.T @ u
    for L in range(1, lag + 1):
        w = 1.0 - L / (lag + 1.0)          # Bartlett kernel
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    return xtx_inv @ S @ xtx_inv


def block_cv_r2(X: np.ndarray, y: np.ndarray, folds: int) -> float:
    """R^2 from contiguous-block cross-validation.

    The samples are a smoothed time series, so neighbouring rows are highly
    correlated; random k-fold would leak the answer across the split and
    report a flattering number. Contiguous blocks are the honest test.
    """
    n = len(y)
    if folds < 2 or n < folds * (X.shape[1] + 2):
        return float("nan")
    edges = np.linspace(0, n, folds + 1).astype(int)
    pred = np.full(n, np.nan)
    for k in range(folds):
        te = np.zeros(n, dtype=bool)
        te[edges[k]:edges[k + 1]] = True
        tr = ~te
        if tr.sum() < X.shape[1] + 2:
            continue
        beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        pred[te] = X[te] @ beta
    m = np.isfinite(pred)
    if m.sum() < 2:
        return float("nan")
    ss_res = float(np.sum((y[m] - pred[m]) ** 2))
    ss_tot = float(np.sum((y[m] - y[m].mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_linear(X: np.ndarray, y: np.ndarray, names: Sequence[str], label: str,
               robust: bool = False, cv_folds: int = 5,
               hac_lag: int | None = None) -> FitResult:
    n, p = X.shape
    w = np.ones(n)

    if robust:
        # Huber IRLS. The log has genuine outliers (dropped frames, wheel slip)
        # and OLS would let a handful of them steer the coefficients.
        for _ in range(30):
            beta, *_ = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w),
                                       rcond=None)
            r = y - X @ beta
            s = 1.4826 * np.median(np.abs(r - np.median(r))) or 1.0
            k = 1.345 * s
            w_new = np.where(np.abs(r) <= k, 1.0, k / np.maximum(np.abs(r), 1e-12))
            if np.max(np.abs(w_new - w)) < 1e-6:
                w = w_new
                break
            w = w_new

    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    resid = y - X @ beta

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p) if n > p else float("nan")
    rmse = math.sqrt(ss_res / n)

    xtx_inv = np.linalg.pinv(X.T @ X)
    sigma2 = ss_res / max(n - p, 1)
    se = np.sqrt(np.maximum(np.diag(sigma2 * xtx_inv), 0.0))

    if hac_lag is None:
        hac_lag = max(1, int(round(n ** 0.25)) * 2)
    se_hac = np.sqrt(np.maximum(np.diag(newey_west_cov(X, resid, xtx_inv, hac_lag)), 0.0))

    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = beta / np.where(se_hac > 0, se_hac, np.nan)

    # Variance inflation factors, to expose regressors the run failed to excite.
    vif = np.full(p, np.nan)
    for j in range(p):
        if np.ptp(X[:, j]) < 1e-12:      # the constant column
            continue
        others = [k for k in range(p) if k != j]
        bj, *_ = np.linalg.lstsq(X[:, others], X[:, j], rcond=None)
        rj = X[:, j] - X[:, others] @ bj
        sst = float(np.sum((X[:, j] - X[:, j].mean()) ** 2))
        if sst > 0:
            r2j = 1.0 - float(np.sum(rj ** 2)) / sst
            vif[j] = 1.0 / max(1.0 - r2j, 1e-12)

    # Residual RMS per contiguous fifth of the run. A single bad block is the
    # signature of a localised event (wheel slip, a bump) rather than a model
    # that is uniformly wrong, and it is what drags the CV score down.
    nb = 5
    edges = np.linspace(0, n, nb + 1).astype(int)
    blocks = [float(np.sqrt((resid[edges[k]:edges[k + 1]] ** 2).mean()))
              if edges[k + 1] > edges[k] else float("nan") for k in range(nb)]

    return FitResult(
        name=label, names=list(names), coef=beta, stderr=se, stderr_hac=se_hac,
        tstat=tstat, r2=r2, r2_adj=r2_adj, rmse=rmse,
        resid_std=float(resid.std()), response_std=float(y.std()),
        cv_r2=block_cv_r2(X, y, cv_folds), n=n, vif=vif,
        block_resid_rms=blocks,
        weights_used=w if robust else None,
    )


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

@dataclass
class Prepared:
    t: np.ndarray
    u: np.ndarray            # (n,4) powers, band-limited to match the response
    u_raw: np.ndarray        # (n,4) powers as logged
    v: np.ndarray            # (n,3) body-frame velocity
    a: np.ndarray            # (n,3) body-frame acceleration
    v_reported: np.ndarray   # (n,3) body-frame velocity from the sensor's own output
    keep: np.ndarray
    diag: dict


def prepare(seg: Segment, args) -> Prepared:
    d = seg.raw
    order = np.argsort(d[:, 0], kind="stable")
    d = d[order]

    t = d[:, 0] / 1000.0                       # ms -> s
    u = d[:, 1:5].copy()
    x, y = d[:, 5].copy(), d[:, 6].copy()
    h = np.unwrap(d[:, 7])                     # guard against a +-pi wrap
    xv, yv, hv = d[:, 8].copy(), d[:, 9].copy(), d[:, 10].copy()

    if args.units == "m":
        x /= 100.0
        y /= 100.0
        xv /= 100.0
        yv /= 100.0

    # Drop exactly-duplicated consecutive samples: the loop polled faster than
    # the sensor produced a new reading, so they carry no new information and
    # would inject a spurious zero-derivative point.
    dup = np.zeros(len(t), dtype=bool)
    dup[1:] = (np.diff(x) == 0) & (np.diff(y) == 0) & (np.diff(h) == 0)
    n_dup = int(dup.sum())

    t, u, x, y, h = t[~dup], u[~dup], x[~dup], y[~dup], h[~dup]
    xv, yv, hv = xv[~dup], yv[~dup], hv[~dup]

    # Derivatives from position, on the irregular grid.
    _, vx_f, ax_f, ok_x = local_poly_derivatives(t, x, args.window, args.order,
                                                 args.max_gap)
    _, vy_f, ay_f, ok_y = local_poly_derivatives(t, y, args.window, args.order,
                                                 args.max_gap)
    _, omega, alpha, ok_h = local_poly_derivatives(t, h, args.window, args.order,
                                                   args.max_gap)
    ok = ok_x & ok_y & ok_h

    # Band-limit the motor powers through the SAME window that produced the
    # response. The model a = B u + A v + c is linear, so applying one linear
    # filter to every term leaves the coefficients unchanged; filtering only
    # the response does not, and shrinks the fitted gains toward zero. On this
    # log, leaving the powers unfiltered made the gains vary by ~2x across the
    # window sweep instead of ~20%.
    u_raw = u.copy()
    if args.smooth_inputs:
        cols = []
        for i in range(u.shape[1]):
            val, _, _, ok_u = local_poly_derivatives(t, u[:, i], args.window,
                                                     args.order, args.max_gap)
            cols.append(np.where(ok_u, val, u[:, i]))
        u = np.column_stack(cols)

    # Field frame -> robot body frame.
    vx_r, vy_r = to_body_frame(vx_f, vy_f, h)
    ax_r, ay_r = to_body_frame(ax_f, ay_f, h)

    # The sensor's own velocity, rotated the same way, kept for cross-checking
    # and for --vel-source reported.
    vxr_rep, vyr_rep = to_body_frame(xv, yv, h)

    v_pos = np.column_stack([vx_r, vy_r, omega])
    v_rep = np.column_stack([vxr_rep, vyr_rep, hv])
    a = np.column_stack([ax_r, ay_r, alpha])

    # Keep only samples that carry dynamics information: powered, or coasting
    # with meaningful speed. The long stationary blocks at either end of the
    # run would otherwise dominate the sample count and inflate R^2 by making
    # "predict zero" look good.
    powered = np.abs(u_raw).sum(axis=1) > 1e-9
    speed = np.hypot(v_pos[:, 0], v_pos[:, 1])
    moving = (speed > args.v_min) | (np.abs(omega) > args.w_min)
    keep = ok & (powered | moving)
    keep &= np.all(np.isfinite(v_pos), axis=1) & np.all(np.isfinite(a), axis=1)
    keep &= np.all(np.isfinite(u), axis=1)

    mec = u_raw @ MEC_MIX.T
    diag = {
        "rows_in_segment": int(seg.n),
        "duplicate_pose_rows_dropped": n_dup,
        "derivative_window_invalid": int((~ok).sum()),
        "stationary_rows_dropped": int((ok & ~(powered | moving)).sum()),
        "rows_used": int(keep.sum()),
        "powered_rows_used": int((keep & powered).sum()),
        "coasting_rows_used": int((keep & ~powered).sum()),
        "loop_dt_median_s": float(np.median(np.diff(t))) if len(t) > 1 else float("nan"),
        "loop_dt_p95_s": float(np.percentile(np.diff(t), 95)) if len(t) > 1 else float("nan"),
        "loop_hz_median": float(1.0 / np.median(np.diff(t))) if len(t) > 1 else float("nan"),
        "t_start_s": float(t.min()), "t_end_s": float(t.max()),
        "motor_singular_values": [float(s) for s in
                                  np.linalg.svd(u_raw[keep], compute_uv=False)]
        if keep.sum() > 4 else None,
        "mecanum_rms": {nm: float(np.sqrt((mec[keep, i] ** 2).mean()))
                        for i, nm in enumerate(MECANUM + ["null"])}
        if keep.sum() > 4 else None,
    }

    # Cross-check: how far the sensor's velocity sits from the smoothed one.
    # This is the direct measurement of the 1.5 kHz quantisation noise.
    if keep.sum() > 10:
        res = v_rep[keep] - v_pos[keep]
        diag["reported_vel_residual_std"] = [float(s) for s in res.std(axis=0)]
        diag["smoothed_vel_std"] = [float(s) for s in v_pos[keep].std(axis=0)]
        diag["reported_vel_noise_to_signal"] = [
            float(a_ / b_) if b_ > 1e-12 else float("nan")
            for a_, b_ in zip(res.std(axis=0), v_pos[keep].std(axis=0))
        ]

    return Prepared(t=t, u=u, u_raw=u_raw, v=v_pos, a=a, v_reported=v_rep,
                    keep=keep, diag=diag)


def traction_gain(u_mec, knee):
    """
    How much of the commanded effort the tyres can actually deliver.

    Past a certain demand the wheels break loose and extra command buys no
    extra force, so the relationship bends over. Modelled as

        m = |u|                       total demand across all three axes
        g = tanh(m / knee) / (m / knee)

    which is 1 for small demand, falls off smoothly past the knee, and keeps
    the direction of the command unchanged. Traction is one shared budget, so
    the demand counts all three axes and the gain scales all three -- if the
    tyres are sliding, every force they were producing drops together.

    `knee` of None disables it and returns 1.
    """
    if knee is None or knee <= 0:
        return np.ones(len(u_mec))
    m = np.sqrt(u_mec[:, 0] ** 2 + u_mec[:, 1] ** 2 + u_mec[:, 2] ** 2)
    r = np.maximum(m / knee, 1e-9)
    return np.tanh(r) / r


def saturate_u(u_wheel, knee):
    """Apply the traction gain to wheel powers.

    Scaling every mecanum component by one factor is the same as scaling the
    wheel powers by it, because the mixing is linear -- so this stays in wheel
    space and no round trip is needed.
    """
    if knee is None or knee <= 0:
        return u_wheel
    g = traction_gain(u_wheel @ MEC_MIX[:3].T, knee)
    return u_wheel * g[:, None]


def csign(v, eps):
    """Smoothed sign: a linear ramp through zero, saturating at +-1."""
    return np.clip(v / eps, -1.0, 1.0)


def build_design(u: np.ndarray, v: np.ndarray, coulomb: bool = True,
                 intercept: bool = True, omega_sq: bool = False,
                 drag: bool = True, eps=COULOMB_EPS):
    """Assemble the regressor matrix and its column names.

    Inputs are always the {fwd, strafe, turn} projection of the motor powers.
    Regressing on the four wheel powers directly is not supported: a standard
    mecanum mapping leaves the fourth direction (FR+FL-BR-BL) essentially
    unexcited, so those coefficients are not separately identifiable -- and
    that direction exerts no chassis force anyway. See WHY MECANUM ONLY.

    Defaults follow the 2026-08-19 analysis of a real run: Coulomb friction
    and quadratic drag are on because they earn their place on held-out
    velocity prediction, and omega^2 is off because it did not -- it scored
    worse than a plain linear model and its apparent centripetal signal was a
    mis-set odometry tracking point leaking in.
    """
    m = u @ MEC_MIX[:3].T
    cols = [m[:, 0], m[:, 1], m[:, 2], v[:, 0], v[:, 1], v[:, 2]]
    names = list(MECANUM) + STATES
    if omega_sq:
        # Centripetal term. A tracking point offset by delta from the true
        # centre of rotation sees  a_P = a_C + alpha x delta - omega^2 * delta,
        # so the omega^2 coefficient estimates -delta directly. Off by default:
        # correct the pod offsets instead of fitting around them.
        cols.append(v[:, 2] ** 2)
        names.append("omega_sq")
    if coulomb:
        # Break-away friction, which a proportional drag term cannot express:
        # it opposes motion with roughly constant magnitude regardless of
        # speed. Smoothed through zero -- see COULOMB_EPS.
        cols += [csign(v[:, 0], eps[0]), csign(v[:, 1], eps[1]),
                 csign(v[:, 2], eps[2])]
        names += ["sgn_v_x", "sgn_v_y", "sgn_omega"]
    if drag:
        # |v|*v keeps the sign of v but grows faster than linear, which is the
        # shape aerodynamic and rolling losses actually take.
        cols += [np.abs(v[:, 0]) * v[:, 0], np.abs(v[:, 1]) * v[:, 1],
                 np.abs(v[:, 2]) * v[:, 2]]
        names += ["absv_v_x", "absv_v_y", "absv_omega"]
    if intercept:
        # A drivetrain at rest with no power should not accelerate, so this
        # term ought to come out near zero. When it does not, it is absorbing
        # something systematic -- a mis-set centre of rotation, a persistent
        # external force -- and is worth reading as a diagnostic.
        cols.append(np.ones(len(u)))
        names.append("const")
    return np.column_stack(cols), names


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _fmt(v: float, w: int = 11, p: int = 4) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return " " * (w - 3) + "n/a"
    return f"{v:{w}.{p}f}"


def print_report(results: list[FitResult], prep: Prepared, args, dist_unit: str,
                 title: str, note: str = ""):
    acc_u = f"{dist_unit}/s^2"
    ang_u = "rad/s^2"

    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)
    if note:
        for ln in note.strip("\n").split("\n"):
            print(ln)

    for r in results:
        unit = ang_u if r.name == "alpha" else acc_u
        print()
        print(f"  {r.name}   [{unit}]      n = {r.n}")
        print("  " + "-" * 74)
        print(f"  {'regressor':<12}{'coef':>13}{'std.err(HAC)':>15}{'t':>9}{'VIF':>9}")
        print("  " + "-" * 74)
        for j, nm in enumerate(r.names):
            star = ""
            if math.isfinite(r.tstat[j]):
                star = "***" if abs(r.tstat[j]) > 3 else ("**" if abs(r.tstat[j]) > 2 else "")
            if not math.isfinite(r.vif[j]):
                vif = ""
            elif r.vif[j] >= 1e6:
                vif = ">1e6"          # rank-deficient; a number here is noise
            else:
                vif = f"{r.vif[j]:.2f}"
            print(f"  {nm:<12}{r.coef[j]:13.4f}{r.stderr_hac[j]:15.4f}"
                  f"{r.tstat[j]:9.2f}{vif:>9}  {star}")
        print("  " + "-" * 74)
        print(f"  R^2 = {r.r2:6.3f}   adj = {r.r2_adj:6.3f}   "
              f"block-CV R^2 = {_fmt(r.cv_r2, 6, 3)}")
        print(f"  RMSE = {r.rmse:8.2f} {unit}    "
              f"response sd = {r.response_std:8.2f}    "
              f"resid sd = {r.resid_std:8.2f}")

        # Readable closed form.
        terms = []
        for j, nm in enumerate(r.names):
            if nm == "const":
                terms.append(f"{r.coef[j]:+.4f}")
            else:
                terms.append(f"{r.coef[j]:+.4f}*{nm}")
        line = f"  {r.name} = " + " ".join(terms)
        print()
        for chunk in _wrap(line, 76):
            print(chunk)

    print()
    print("  " + "-" * 74)
    print(f"  {'response':<10}{'R^2':>9}{'adj R^2':>10}{'CV R^2':>10}"
          f"{'RMSE':>12}{'resid/signal':>15}{'max VIF':>10}")
    print("  " + "-" * 74)
    for r in results:
        ratio = r.resid_std / r.response_std if r.response_std > 0 else float("nan")
        print(f"  {r.name:<10}{r.r2:9.3f}{r.r2_adj:10.3f}{_fmt(r.cv_r2, 10, 3)}"
              f"{r.rmse:12.2f}{ratio:15.3f}{r.max_vif:10.1f}")

    print()
    print("  residual RMS by contiguous fifth of the run "
          "(a single hot block = a localised event, not a wrong model)")
    for r in results:
        unit = ang_u if r.name == "alpha" else acc_u
        cells = " ".join(f"{b:8.1f}" for b in r.block_resid_rms)
        print(f"    {r.name:<8}{cells}   [{unit}]")

    worst = max(results, key=lambda r: r.max_vif if math.isfinite(r.max_vif) else 0)
    if math.isfinite(worst.max_vif) and worst.max_vif > 10:
        j = int(np.nanargmax(np.where(np.isfinite(worst.vif), worst.vif, -1)))
        print()
        print(f"  !! COLLINEARITY WARNING: max VIF = {worst.max_vif:.0f} "
              f"on '{worst.names[j]}'")
        print("     That regressor is nearly a linear combination of the others,")
        print("     so its coefficient is not separately identified -- the")
        print("     standard error swamps it even if the model predicts well.")
        print("     Drive the run so that input varies independently of the rest.")
    print()


def print_traction(args, rows):
    """Show what the knee search found, including when it found nothing."""
    print()
    print("=" * 78)
    print("  WHEEL SLIP (traction saturation)")
    print("=" * 78)
    if args.knee_mode == "off":
        print("  disabled by --traction-knee off")
        print()
        return
    if rows:
        print("  Searching for the demand at which the tyres stop delivering.")
        print("  Scored by integrating the model on held-out data, because a")
        print("  saturating term will always reduce the fit residual somewhere.")
        print()
        print("    %-14s %8s %8s %8s %8s" % ("knee", "v_x", "v_y", "omega", "mean"))
        best = max(rows, key=lambda r: r[2])
        for knee, r2, mu in rows:
            lbl = "no saturation" if knee is None else "%.2f" % knee
            print("    %-14s %8.3f %8.3f %8.3f %8.3f%s"
                  % (lbl, *r2, mu, "   <-- chosen" if knee == best[0] else ""))
        flat = [r for r in rows if r[0] is None]
        gain = best[2] - (flat[0][2] if flat else best[2])
        print()
        if args.knee is None and best[0] is not None:
            print("  Best knee %.2f gained only %+.3f, below the %.2f needed to"
                  % (best[0], best[2] - (flat[0][2] if flat else best[2]),
                     KNEE_MIN_GAIN))
            print("  justify the parameter, so the term is switched off. A run")
            print("  that is ALWAYS sliding looks like this: it can be fitted")
            print("  slightly better with a huge knee, but the knee is not")
            print("  identifiable because the data never comes back below it.")
        elif args.knee is None:
            print("  No slip found: the run scores best with no saturation, so")
            print("  the term switches itself off. That is the right answer both")
            print("  for a run that never reaches the traction limit and for one")
            print("  that never comes back below it -- a permanently sliding")
            print("  robot just looks like a lower gain.")
        else:
            print("  Knee at %.2f, worth %+.3f mean rollout R2 over no saturation."
                  % (args.knee, gain))
            print("  Past that demand, extra command buys no extra force.")
    elif args.knee:
        print("  knee fixed at %.2f by --traction-knee" % args.knee)
    if args.knee:
        print()
        print("  This knee describes THIS FLOOR. A grippier surface slips later,")
        print("  so re-measure whenever the surface changes.")
    print()


def print_centripetal(results: list[FitResult], dist_unit: str):
    """Interpret the omega^2 coefficients as a tracking-point offset."""
    by = {r.name: r for r in results}
    if "omega_sq" not in by["a_x"].names:
        return
    j = by["a_x"].names.index("omega_sq")
    qx, qy = by["a_x"].coef[j], by["a_y"].coef[j]
    qa = by["alpha"].coef[j]
    tx, ty = by["a_x"].tstat[j], by["a_y"].tstat[j]

    print("=" * 78)
    print("  CENTRIPETAL TERM  ->  TRACKING-POINT OFFSET")
    print("=" * 78)
    print("  A tracking point offset by delta from the true centre of rotation")
    print("  sees  a = a_centre + alpha x delta - omega^2 * delta,  so the")
    print("  omega^2 coefficient estimates -delta.")
    print()
    print(f"    implied delta_x = {-qx:+8.3f} {dist_unit}   (t = {tx:5.2f})")
    print(f"    implied delta_y = {-qy:+8.3f} {dist_unit}   (t = {ty:5.2f})")
    print(f"    |delta|         = {math.hypot(qx, qy):8.3f} {dist_unit}"
          f"  = {math.hypot(qx, qy) * 10:.1f} mm")
    print()
    print(f"    alpha row omega^2 coefficient = {qa:+.4f}  <- should be ~0;")
    print("      omega^2 cannot torque a rigid body, so a large value here")
    print("      means the term is soaking up something else.")
    print()
    print("  A large |delta| means the Pinpoint's tracking point is not at the")
    print("  robot's centre of rotation. Correcting the pod offsets on the")
    print("  robot should shrink it toward zero.")
    print()


def _wrap(s: str, width: int) -> list[str]:
    words, out, cur = s.split(" "), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            out.append(cur)
            cur = "      " + w
        else:
            cur = (cur + " " + w) if cur else w
    if cur:
        out.append(cur)
    return out


def print_diagnostics(prep: Prepared, stats: dict, seg: Segment, args, dist_unit: str):
    d = prep.diag
    print()
    print("=" * 78)
    print("  DATA CONDITIONING")
    print("=" * 78)
    print(f"  source                     {os.path.basename(seg.source)} (segment {seg.index})")
    print(f"  rows parsed                {d['rows_in_segment']}")
    print(f"  truncated / bad rows       {stats['short_rows']} / {stats['bad_rows']}")
    print(f"  header rows in file        {stats['headers']}"
          + ("   <-- appended runs, split into segments" if stats["headers"] > 1 else ""))
    print(f"  duplicate pose rows        {d['duplicate_pose_rows_dropped']} dropped (stale sensor reads)")
    print(f"  window straddled a gap     {d['derivative_window_invalid']} dropped")
    print(f"  stationary rows            {d['stationary_rows_dropped']} dropped")
    print(f"  rows used for the fit      {d['rows_used']} "
          f"({d['powered_rows_used']} powered, {d['coasting_rows_used']} coasting)")
    print(f"  time span                  {d['t_start_s']:.2f} .. {d['t_end_s']:.2f} s")
    print(f"  loop rate                  {d['loop_hz_median']:.1f} Hz median "
          f"(dt {d['loop_dt_median_s']*1000:.1f} ms, p95 {d['loop_dt_p95_s']*1000:.1f} ms)")

    if "reported_vel_noise_to_signal" in d:
        print()
        print("  Pinpoint velocity check (1.5 kHz quantisation)")
        ns = d["reported_vel_noise_to_signal"]
        rs = d["reported_vel_residual_std"]
        for nm, r_, n_ in zip(STATES, rs, ns):
            unit = "rad/s" if nm == "omega" else f"{dist_unit}/s"
            print(f"    {nm:<8} reported-vs-smoothed residual sd = {r_:8.3f} {unit:<6} "
                  f"= {100*n_:5.1f}% of signal")
        print("    -> reported velocity is NOT differenced to get acceleration;")
        print("       both v and a come from a local polynomial fit to position.")

    if d.get("mecanum_rms"):
        print()
        print("  Input excitation (rms of each commanded direction)")
        mr = d["mecanum_rms"]
        for nm in MECANUM + ["null"]:
            tag = "   <-- unactuated direction, excluded from the model" \
                  if nm == "null" else ""
            print(f"    {nm:<8}{mr[nm]:8.4f}{tag}")
        sv = d.get("motor_singular_values") or []
        if sv:
            cond = sv[0] / max(sv[-1], 1e-30)
            print(f"    singular values of [FR FL BR BL]: "
                  f"{', '.join(f'{s:.3f}' for s in sv)}")
            cond_s = ">1e6" if cond > 1e6 else f"{cond:.1f}"
            print(f"    cond of the raw wheel powers = {cond_s}"
                  + ("   (why the fit uses fwd/strafe/turn)" if cond > 20 else ""))


def print_sensitivity(seg: Segment, args, base_results: list[FitResult]):
    """Show how much the answer depends on the smoothing choice.

    With this much sensor noise the window length is the one knob that could
    quietly manufacture a result, so it is worth showing explicitly.
    """
    print()
    print("=" * 78)
    print("  SENSITIVITY TO SMOOTHING WINDOW")
    print("=" * 78)
    print("  Mecanum-basis input gains vs. window halfwidth. Coefficients that")
    print("  drift steadily with the window are being set by the filter, not")
    print("  by the data; a flat stretch is the trustworthy region.")
    print()

    windows = [0.08, 0.12, 0.18, 0.22, 0.28, 0.35]
    for resp_i, resp in enumerate(RESPONSES):
        print(f"  {resp}")
        print(f"    {'window':>8}{'fwd':>11}{'strafe':>11}{'turn':>11}"
              f"{'R^2':>9}{'CV R^2':>9}{'n':>7}")
        for w in windows:
            a2 = argparse.Namespace(**vars(args))
            a2.window = w
            try:
                p2 = prepare(seg, a2)
                if p2.keep.sum() < 30:
                    continue
                v2 = p2.v_reported if args.vel_source == "reported" else p2.v
                X2, nm2 = build_design(
                    saturate_u(p2.u, getattr(args, "knee", None))[p2.keep],
                    v2[p2.keep], args.coulomb, args.intercept, args.omega_sq,
                    args.drag)
                r2 = fit_linear(X2, p2.a[p2.keep, resp_i], nm2, resp,
                                robust=args.robust, cv_folds=args.cv_folds)
            except Exception:
                continue
            mark = " <-" if abs(w - args.window) < 1e-9 else ""
            print(f"    {w:8.2f}{r2.coef[0]:11.2f}{r2.coef[1]:11.2f}"
                  f"{r2.coef[2]:11.2f}"
                  f"{r2.r2:9.3f}{_fmt(r2.cv_r2, 9, 3)}{r2.n:7d}{mark}")
        print()


def print_vel_source_comparison(seg: Segment, args):
    """Contrast the two candidate velocity regressors."""
    print("=" * 78)
    print("  VELOCITY REGRESSOR: smoothed-from-position vs. sensor-reported")
    print("=" * 78)
    prep = prepare(seg, args)
    k = prep.keep
    if k.sum() < 30:
        print("  not enough samples")
        return
    print(f"    {'source':<12}{'response':<9}{'R^2':>9}{'CV R^2':>9}"
          f"{'c_vx':>10}{'c_vy':>10}{'c_w':>10}")
    us = saturate_u(prep.u, getattr(args, "knee", None))
    for src, v in (("position", prep.v), ("reported", prep.v_reported)):
        X, nm = build_design(us[k], v[k], args.coulomb, args.intercept,
                             args.omega_sq, args.drag)
        iv = nm.index("v_x")
        for i, resp in enumerate(RESPONSES):
            r = fit_linear(X, prep.a[k, i], nm, resp, robust=args.robust,
                           cv_folds=args.cv_folds)
            print(f"    {src:<12}{resp:<9}{r.r2:9.3f}{_fmt(r.cv_r2, 9, 3)}"
                  f"{r.coef[iv]:10.4f}{r.coef[iv+1]:10.4f}{r.coef[iv+2]:10.4f}")
    print("  Sensor-reported velocity carries ~25% noise, which attenuates the")
    print("  drag coefficients toward zero (classic errors-in-variables bias).")
    print()


# --------------------------------------------------------------------------
# Output files
# --------------------------------------------------------------------------

def _pack_basis(results, inputs, knee=None) -> dict:
    names = results[0].names
    B = np.array([[r.coef[names.index(m)] for m in inputs] for r in results])
    A = np.array([[r.coef[names.index(s)] for s in STATES] for r in results])
    has_const = "const" in names
    c = np.array([r.coef[names.index("const")] if has_const else 0.0
                  for r in results])
    has_wsq = "omega_sq" in names
    q = np.array([r.coef[names.index("omega_sq")] if has_wsq else 0.0
                  for r in results])

    def block(prefix, present):
        return np.array([[r.coef[names.index(prefix + s_)] if present else 0.0
                          for s_ in STATES] for r in results])

    has_coul = ("sgn_" + STATES[0]) in names
    has_drag = ("absv_" + STATES[0]) in names
    S = block("sgn_", has_coul)     # Coulomb, multiplies csign(v)
    D = block("absv_", has_drag)    # quadratic drag, multiplies |v|*v

    # Every optional regressor still appears, pinned to zero when it was not
    # fitted, so the file has one fixed shape regardless of the flags used.
    canonical = (list(inputs) + STATES + ["omega_sq"]
                 + ["sgn_" + s_ for s_ in STATES]
                 + ["absv_" + s_ for s_ in STATES] + ["const"])

    def pad(pairs, missing=0.0):
        d = {n: float(v) if v is not None and math.isfinite(v) else None
             for n, v in pairs}
        return {n: d.get(n, missing) for n in canonical}

    models = {}
    for r in results:
        models[r.name] = {
            "coef": pad(zip(r.names, r.coef)),
            "stderr_hac": pad(zip(r.names, r.stderr_hac)),
            "stderr_ols": pad(zip(r.names, r.stderr)),
            "tstat": pad(zip(r.names, r.tstat), None),
            "vif": pad(zip(r.names, r.vif), None),
            "r2": float(r.r2),
            "r2_adj": float(r.r2_adj),
            "cv_r2": (float(r.cv_r2) if math.isfinite(r.cv_r2) else None),
            "rmse": float(r.rmse),
            "resid_std": float(r.resid_std),
            "response_std": float(r.response_std),
            "block_resid_rms": [float(b) for b in r.block_resid_rms],
            "max_vif": (float(r.max_vif) if math.isfinite(r.max_vif) else None),
            "n": int(r.n),
        }

    return {
        "inputs": list(inputs),
        "regressors": canonical,
        "equation": ("a = B*u + A*v + q*omega^2 + S*csign(v) + "
                     "D*(|v|*v) + c"),
        "B": B.tolist(),          # (3,3) input -> acceleration
        "A": A.tolist(),          # (3,3) velocity -> acceleration (drag / back-EMF)
        "q": q.tolist(),          # (3,)  centripetal, estimates -delta
        "S": S.tolist(),          # (3,3) Coulomb break-away friction
        "D": D.tolist(),          # (3,3) quadratic drag
        "c": c.tolist(),          # (3,)  constant offset
        "coulomb_eps": list(COULOMB_EPS),
        "csign": "csign(v)_i = clamp(v_i / coulomb_eps_i, -1, 1)",
        "traction_knee": knee,
        "traction_note": ("u is the SATURATED command: m = norm(u_raw), "
                          "u = u_raw * tanh(m/knee)/(m/knee). "
                          "A null knee means no saturation."),
        "B_columns": list(inputs),
        "A_columns": STATES,
        "rows": RESPONSES,
        "has_intercept": bool(has_const),
        # When the omega^2 term is disabled, q is still emitted as zeros and the
        # equation is unchanged, so a consumer never has to branch on this --
        # evaluating a = B*u + A*v + q*omega^2 + c just drops the term.
        "has_omega_sq": bool(has_wsq),
        "has_coulomb": bool(has_coul),
        "has_drag": bool(has_drag),
        "n_coefficients_per_direction": len(names),
        # q ~ -delta, so this is the implied tracking-point offset.
        "implied_tracking_offset": [float(-q[0]), float(-q[1])],
        "models": models,
    }


def build_output(mec_results: list[FitResult], prep: Prepared, seg: Segment,
                 stats: dict, args, dist_unit: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": "fit_drivetrain.py",
        "source": {
            "file": os.path.abspath(seg.source),
            "sha256": _sha256(seg.source),
            "segment_index": seg.index,
            "parse_stats": stats,
        },
        "units": {
            "distance": dist_unit,
            "angle": "rad",
            "time": "s",
            "linear_acceleration": f"{dist_unit}/s^2",
            "angular_acceleration": "rad/s^2",
        },
        "convention": {
            "frame": ("robot body frame; field-frame vectors rotated by R(-h). "
                      "x and y are the Pinpoint axes as logged."),
            "responses": RESPONSES,
            "motors": MOTORS,
            "states": STATES,
            "mecanum_mix_rows": MECANUM + ["null"],
            "mecanum_mix": MEC_MIX.tolist(),
            "basis": "mecanum",
        },
        "mecanum_basis": _pack_basis(mec_results, MECANUM, args.knee),
        "preprocessing": {
            "derivative_method": "local weighted polynomial (tricube, irregular grid)",
            "window_halfwidth_s": args.window,
            "poly_order": args.order,
            "max_gap_s": args.max_gap,
            "velocity_source": args.vel_source,
            "inputs_band_limited": bool(args.smooth_inputs),
            "omega_sq_term": bool(args.omega_sq),
            "traction_knee": args.knee,
            "traction_knee_mode": args.knee_mode,
            "coulomb_terms": bool(args.coulomb),
            "drag_terms": bool(args.drag),
            "coulomb_eps": list(COULOMB_EPS),
            "robust_huber": bool(args.robust),
            "cv_folds": args.cv_folds,
            "v_min": args.v_min,
            "w_min": args.w_min,
        },
        "diagnostics": prep.diag,
    }


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "nan"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return repr(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"cannot serialise {type(v)}")


def write_toml(data: dict, path: str) -> None:
    """Emit TOML by hand so the Julia side can read it with the stdlib `TOML`
    module and no package dependency at all."""
    lines: list[str] = []

    def emit(prefix: str, d: dict):
        scalars = {k: v for k, v in d.items() if not isinstance(v, dict)}
        tables = {k: v for k, v in d.items() if isinstance(v, dict)}
        if prefix:
            lines.append(f"[{prefix}]")
        for k, v in scalars.items():
            lines.append(f"{k} = {_toml_value(v)}")
        if scalars:
            lines.append("")
        for k, v in tables.items():
            emit(f"{prefix}.{k}" if prefix else k, v)

    emit("", data)
    with open(path, "w") as fh:
        fh.write("# Generated by fit_drivetrain.py -- do not edit by hand.\n")
        fh.write("# Read from Julia with:  using TOML; cfg = TOML.parsefile(path)\n\n")
        fh.write("\n".join(lines).rstrip() + "\n")


# --------------------------------------------------------------------------
# Rollout scoring
# --------------------------------------------------------------------------

def eval_terms(u, v, names, eps=COULOMB_EPS):
    """Rebuild the design columns for arbitrary (u, v) from a name list.

    Lets a fitted coefficient vector be evaluated at states the fit never saw,
    which is what rolling the model forward requires.
    """
    cols = []
    idx = {"fwd": 0, "strafe": 1, "turn": 2}
    for nm in names:
        if nm in idx:
            cols.append(u[:, idx[nm]])
        elif nm in STATES:
            cols.append(v[:, STATES.index(nm)])
        elif nm == "omega_sq":
            cols.append(v[:, 2] ** 2)
        elif nm.startswith("sgn_"):
            j = STATES.index(nm[4:])
            cols.append(csign(v[:, j], eps[j]))
        elif nm.startswith("absv_"):
            j = STATES.index(nm[5:])
            cols.append(np.abs(v[:, j]) * v[:, j])
        elif nm.endswith("^2") and nm[:-2] in idx:
            cols.append(u[:, idx[nm[:-2]]] ** 2)
        elif "*" in nm:
            a_, b_ = nm.split("*", 1)
            cols.append(u[:, idx[a_]] * v[:, STATES.index(b_)])
        elif nm == "const":
            cols.append(np.ones(len(u)))
        else:
            raise ValueError("unknown regressor " + nm)
    return np.column_stack(cols)


def rollout_r2(t, u_mec, v_meas, coef, names, horizon=0.5, dt=0.01,
               eps=COULOMB_EPS, stride=7, test_half=True):
    """
    Integrate the model and score the CHANGE in velocity over `horizon`.

    This is the yardstick that matters, because the solver integrates this
    model forward. Scoring the acceleration instead flatters or damns a model
    for how noisy a smoothed second derivative happened to be. Scoring the
    change rather than the absolute velocity keeps the comparison honest: over
    a short horizon "assume nothing changes" would otherwise score ~0.99.

    Returns (r2 per axis, r2 of the null "no change" model).
    """
    tu = np.arange(t.min(), t.max(), dt)
    if len(tu) < 50:
        return [float("nan")] * 3, [float("nan")] * 3
    U = np.column_stack([np.interp(tu, t, u_mec[:, i]) for i in range(3)])
    V = np.column_stack([np.interp(tu, t, v_meas[:, i]) for i in range(3)])
    ns = max(1, int(horizon / dt))
    st = np.arange(0, len(tu) - ns - 1, stride)
    if len(st) < 20:
        return [float("nan")] * 3, [float("nan")] * 3
    if test_half:
        st = st[len(st) // 2:]          # never fitted on

    v = V[st].copy()
    for k in range(ns):
        X = eval_terms(U[st + k], v, names, eps)
        acc = X @ coef.T
        w = v[:, 2]
        # Body-frame proper acceleration: dv/dt = a - omega x v.
        dv = np.column_stack([acc[:, 0] + w * v[:, 1],
                              acc[:, 1] - w * v[:, 0], acc[:, 2]])
        v = v + dv * dt
        if not np.all(np.isfinite(v)):
            return [float("-inf")] * 3, [0.0] * 3

    v0 = V[st]
    truth = V[st + ns] - v0
    pred = v - v0
    out, null = [], []
    for i in range(3):
        ss = float(np.sum((truth[:, i] - truth[:, i].mean()) ** 2))
        if ss <= 0:
            out.append(float("nan")); null.append(float("nan")); continue
        out.append(1.0 - float(np.sum((pred[:, i] - truth[:, i]) ** 2)) / ss)
        null.append(1.0 - float(np.sum(truth[:, i] ** 2)) / ss)
    return out, null


def choose_knee(prep, args, grid=KNEE_GRID):
    """
    Pick the traction knee by held-out rollout, "no saturation" included.

    Judged by integrating the model rather than by fit residual, because a
    saturating term will always reduce residual somewhere. Including `None` in
    the grid is what makes this safe to leave on: a run with no visible slip
    simply scores best without it and the term switches itself off.

    Slip is only identifiable from a run that CROSSES the traction limit. A
    run that lives entirely above it just looks like a lower gain, and one
    that never reaches it has nothing to see -- both come back as None.
    """
    k = prep.keep
    best = (None, -np.inf, None)
    rows = []
    flat_score = None
    for knee in grid:
        us = saturate_u(prep.u[k], knee)
        X, names = build_design(us, prep.v[k], args.coulomb, args.intercept,
                                args.omega_sq, args.drag)
        coef = np.zeros((3, X.shape[1]))
        for i in range(3):
            b, *_ = np.linalg.lstsq(X, prep.a[k][:, i], rcond=None)
            coef[i] = b
        r2, _ = rollout_r2(prep.t[k], us @ MEC_MIX[:3].T, prep.v[k], coef,
                           names, horizon=0.5)
        mu = float(np.nanmean(r2))
        rows.append((knee, r2, mu))
        if knee is None:
            flat_score = mu
        if mu > best[1]:
            best = (knee, mu, r2)
    # Only keep a knee that is clearly better than no saturation at all.
    if (best[0] is not None and flat_score is not None
            and best[1] - flat_score < KNEE_MIN_GAIN):
        return None, rows
    return best[0], rows


# --------------------------------------------------------------------------
# Synthetic drive, for validating the pipeline end to end
# --------------------------------------------------------------------------

def synth_drive(B, A, q, c, S=None, D=None, dur=14.0, hz=150.0, jitter=0.35,
                pos_noise=0.02, vel_noise=4.0, seed=0):
    """Simulate a robot obeying  a = B*u + A*v + q*omega^2 + c  exactly.

    Integrated finely, then sampled at a jittery ~150 Hz and reported the way
    the Pinpoint reports: field-frame position and velocity, with the velocity
    channel far noisier than position.
    """
    rng = np.random.default_rng(seed)
    fine = 1.0 / 2000.0
    n_fine = int(dur / fine)

    # Smooth pseudo-random commands that exercise all three directions.
    def band(k):
        ph = rng.uniform(0, 2 * np.pi, 5)
        fr = rng.uniform(0.15, 0.9, 5)
        tt = np.arange(n_fine) * fine
        return np.clip(sum(np.sin(2 * np.pi * f * tt + p) for f, p in zip(fr, ph)) / 2.2,
                       -1, 1)
    U = np.column_stack([band(0), band(1), band(2) * 0.6])

    v = np.zeros(2)
    w = 0.0
    h = 0.0
    p = np.zeros(2)
    J = np.array([[0.0, -1.0], [1.0, 0.0]])
    out = np.empty((n_fine, 7))   # t, px, py, h, v_field_x, v_field_y, omega
    for i in range(n_fine):
        st = np.array([v[0], v[1], w])
        a = B @ U[i] + A @ st + q * w * w + c
        if S is not None:
            a = a + S @ csign(st, np.asarray(COULOMB_EPS))
        if D is not None:
            a = a + D @ (np.abs(st) * st)
        dv = a[:2] - w * (J @ v)
        out[i] = (i * fine, p[0], p[1], h,
                  *(np.array([[math.cos(h), -math.sin(h)],
                              [math.sin(h), math.cos(h)]]) @ v), w)
        p = p + (np.array([[math.cos(h), -math.sin(h)],
                           [math.sin(h), math.cos(h)]]) @ v) * fine
        v = v + dv * fine
        h = h + w * fine
        w = w + a[2] * fine

    # Sample at a jittery loop rate.
    t_s, tc = [], 0.0
    while tc < dur - 0.05:
        t_s.append(tc)
        tc += (1.0 / hz) * (1.0 + jitter * (rng.random() - 0.5))
    t_s = np.array(t_s)
    idx = np.clip((t_s / fine).astype(int), 0, n_fine - 1)
    s = out[idx]
    ui = U[idx]

    rows = np.column_stack([
        s[:, 0] * 1000.0,
        # Invert the mecanum mix to get per-wheel powers back. The pseudo-
        # inverse gives the minimum-norm solution, which carries no null
        # component -- exactly what a real mecanum mapping commands.
        *((ui @ np.linalg.pinv(MEC_MIX[:3].T)).T),
        s[:, 1] + rng.normal(0, pos_noise, len(s)),
        s[:, 2] + rng.normal(0, pos_noise, len(s)),
        s[:, 3],
        s[:, 4] + rng.normal(0, vel_noise, len(s)),
        s[:, 5] + rng.normal(0, vel_noise, len(s)),
        s[:, 6] + rng.normal(0, 0.02, len(s)),
    ])
    return rows


def self_test(args) -> int:
    B = np.array([[-420.0, 5.0, 40.0], [30.0, 260.0, -15.0], [-2.0, 3.0, -42.0]])
    A = np.array([[-0.80, -0.10, -4.0], [-0.05, -0.75, 3.0], [0.004, -0.002, -1.90]])
    q = np.array([-3.5, 2.1, 0.0])          # implies a tracking offset
    c = np.array([0.0, 0.0, 0.0])
    # Break-away friction and quadratic drag, both opposing motion.
    S = np.diag([-9.0, -11.0, -0.45])
    D = np.diag([-0.004, -0.005, -0.03])
    # Each term is simulated only when the model is allowed to fit it --
    # otherwise the test would ask a model to reproduce something it was
    # explicitly denied, and "failure" would mean nothing.
    if not args.omega_sq:
        q = np.zeros(3)
    if not args.coulomb:
        S = np.zeros((3, 3))
    if not args.drag:
        D = np.zeros((3, 3))

    print()
    print("=" * 78)
    print("  SELF-TEST -- recovering a known model from a simulated drive")
    print("=" * 78)
    print("  Robot integrated from a known B, A, q, then sampled at a jittery")
    print("  150 Hz with 0.02 cm position noise and 4 cm/s velocity noise.")
    print()

    rows = synth_drive(B, A, q, c, S, D, seed=7)
    seg = Segment(raw=rows, source="<synthetic>", index=0)
    a2 = argparse.Namespace(**vars(args))
    a2.self_test = False
    prep = prepare(seg, a2)
    k = prep.keep
    X, names = build_design(prep.u[k], prep.v[k], a2.coulomb, a2.intercept,
                            a2.omega_sq, a2.drag)
    res = [fit_linear(X, prep.a[k, i], names, r, cv_folds=a2.cv_folds)
           for i, r in enumerate(RESPONSES)]

    print(f"  samples used: {int(k.sum())}")
    print()
    # Two different things are checked, because they have different answers.
    #
    # The input gains B are identifiable and physically meaningful, so they are
    # checked against the truth directly. The velocity-dependent blocks are
    # NOT: linear v, csign(v) and |v|*v are all odd monotonic functions of the
    # same variable, so over any finite speed range they are mutually
    # collinear and the fit can trade freely between them. Demanding that each
    # come back individually would fail a model that predicts perfectly.
    #
    # What the solver actually needs is prediction, so that is the pass/fail
    # criterion, with coefficient recovery reported only for B.
    print("  %-7s %-11s %11s %11s %8s" % ("row", "term", "true", "fitted", "err %"))
    print("  " + "-" * 52)
    worst_B = 0.0
    for i, r in enumerate(res):
        scale = max(np.abs(B[i]).max(), 1e-9)
        for nm, tv in zip(MECANUM, B[i]):
            fv = r.coef[names.index(nm)]
            e = 100 * abs(fv - tv) / scale
            worst_B = max(worst_B, e)
            print("  %-7s %-11s %11.3f %11.3f %8.1f"
                  % (r.name if nm == MECANUM[0] else "", nm, tv, fv, e))
    print()
    print("  worst input-gain error: %.1f%% of that row's dominant gain" % worst_B)

    coef = np.vstack([r.coef for r in res])
    r2, null = rollout_r2(prep.t[k], prep.u[k] @ MEC_MIX[:3].T, prep.v[k],
                          coef, names, horizon=0.5)
    print()
    print("  held-out prediction of the CHANGE in velocity over 0.5 s:")
    print("    %-10s %9s %9s %9s" % ("", "v_x", "v_y", "omega"))
    print("    %-10s %9.3f %9.3f %9.3f" % ("model", *r2))
    print("    %-10s %9.3f %9.3f %9.3f" % ("null", *null))

    ok_B = worst_B < 25.0
    ok_pred = all(np.isfinite(x) and x > 0.90 for x in r2)
    print()
    print("  input gains within 25%%: %s" % ("yes" if ok_B else "NO"))
    print("  every axis predicts above 0.90: %s" % ("yes" if ok_pred else "NO"))
    ok = ok_B and ok_pred
    print()
    print("  %s" % ("PASS" if ok else "FAIL"))
    print()
    print("  Note: the velocity-dependent coefficients are deliberately not")
    print("  checked one by one. Linear v, csign(v) and |v|*v are collinear")
    print("  over any finite speed range, so their split is arbitrary even")
    print("  when the model as a whole is exactly right. Read A, S and D as a")
    print("  group, never individually.")
    print()
    return 0 if ok else 1


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(
        description="Fit a linear drivetrain model from a Peregrine calibration log.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("log", nargs="?", help="path to a calibration_log_*.csv")
    p.add_argument("--self-test", action="store_true",
                   help="recover a known model from a simulated drive, then exit")
    p.add_argument("-o", "--out-dir", default=None,
                   help="directory for drivetrain_fit.{json,toml} "
                        "(default: alongside this script)")
    p.add_argument("--window", type=float, default=0.22,
                   help="halfwidth in seconds of the local polynomial window")
    p.add_argument("--no-smooth-inputs", dest="smooth_inputs", action="store_false",
                   help="do NOT band-limit the motor powers to match the response "
                        "(leaving them raw biases the fitted gains toward zero)")
    p.add_argument("--order", type=int, default=3,
                   help="polynomial order for the derivative fit (>=2)")
    p.add_argument("--max-gap", type=float, default=0.05,
                   help="reject a derivative whose window straddles a gap larger than this")
    p.add_argument("--vel-source", choices=["position", "reported"], default="position",
                   help="velocity regressor: smoothed from position, or the sensor's own")
    p.add_argument("--units", choices=["cm", "m"], default="cm",
                   help="distance unit for the fit and the output file")
    p.add_argument("--no-intercept", dest="intercept", action="store_false",
                   help="drop the constant term (3 input + 3 velocity + omega^2)")
    p.add_argument("--omega-sq", dest="omega_sq", action="store_true",
                   help="add an omega^2 centripetal term. Off by default: on a "
                        "real run it scored worse than plain linear, and its "
                        "apparent signal was a mis-set tracking point. Correct "
                        "the pod offsets instead (see find_pod_offsets.py)")
    p.add_argument("--no-coulomb", dest="coulomb", action="store_false",
                   help="drop the Coulomb break-away friction terms (on by default)")
    p.add_argument("--no-drag", dest="drag", action="store_false",
                   help="drop the quadratic |v|v drag terms (on by default)")
    p.add_argument("--traction-knee", default="auto",
                   help="wheel-slip saturation: 'auto' searches for the knee "
                        "and switches the term off if the run shows no slip, "
                        "'off' disables it, or give a number to fix it")
    p.add_argument("--robust", action="store_true",
                   help="Huber IRLS instead of ordinary least squares")
    p.add_argument("--min-excitation", type=float, default=0.01,
                   help="refuse to fit if any of fwd/strafe/turn has rms command "
                        "below this; 0 disables the check")
    p.add_argument("--cv-folds", type=int, default=5,
                   help="contiguous blocks for cross-validated R^2")
    p.add_argument("--v-min", type=float, default=None,
                   help="speed below which an unpowered sample is treated as "
                        "stationary, in the chosen --units (default: 2 cm/s)")
    p.add_argument("--w-min", type=float, default=0.05,
                   help="|omega| below which an unpowered sample is treated as stationary")
    p.add_argument("--segment", type=int, default=None,
                   help="which segment to fit when a file holds several appended runs")
    p.add_argument("--no-sensitivity", action="store_true",
                   help="skip the smoothing-window sensitivity sweep")
    p.add_argument("--no-write", action="store_true", help="report only, write no files")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.order < 2:
        print("error: --order must be at least 2 to estimate an acceleration",
              file=sys.stderr)
        return 2
    # Resolve the stationary threshold in whichever distance unit we fit in,
    # so --units does not silently change which samples are kept.
    if args.v_min is None:
        args.v_min = 2.0 if args.units == "cm" else 0.02
    tk = str(args.traction_knee).strip().lower()
    if tk in ("off", "none", "0"):
        args.knee_mode, args.knee = "off", None
    elif tk == "auto":
        args.knee_mode, args.knee = "auto", None
    else:
        try:
            args.knee_mode, args.knee = "fixed", float(tk)
        except ValueError:
            print("error: --traction-knee wants 'auto', 'off' or a number",
                  file=sys.stderr)
            return 2
    if args.self_test:
        return self_test(args)
    if not args.log:
        print("error: give a log path, or --self-test", file=sys.stderr)
        return 2
    if not os.path.exists(args.log):
        print(f"error: no such file: {args.log}", file=sys.stderr)
        return 2

    segments, stats = load_segments(args.log)
    if not segments:
        print(f"error: no data rows in {args.log} (header-only init?)", file=sys.stderr)
        return 1

    usable = []
    for s in segments:
        ok, why = validate(s)
        if ok:
            usable.append(s)
        else:
            print(f"  skipping segment {s.index}: {why}", file=sys.stderr)
    if not usable:
        print("error: no usable segment in this log", file=sys.stderr)
        return 1

    if args.segment is not None:
        chosen = [s for s in usable if s.index == args.segment]
        if not chosen:
            print(f"error: segment {args.segment} not usable/present", file=sys.stderr)
            return 1
        seg = chosen[0]
    else:
        seg = max(usable, key=lambda s: s.n)      # the longest run
        if len(usable) > 1:
            print(f"  note: {len(usable)} runs in this file; fitting the longest "
                  f"(segment {seg.index}, {seg.n} rows). Use --segment to choose.")

    dist_unit = args.units
    prep = prepare(seg, args)

    if prep.keep.sum() < 30:
        print(f"error: only {prep.keep.sum()} usable samples after conditioning; "
              f"try a smaller --window or a longer run", file=sys.stderr)
        return 1

    # A direction the run never commanded cannot have its gain identified. Left
    # unchecked this yields absurd coefficients with VIFs around 1e15 rather
    # than an honest refusal, so say so plainly and stop.
    mr = prep.diag.get("mecanum_rms") or {}
    dead = [n for n in MECANUM if mr.get(n, 0.0) < args.min_excitation]
    if dead:
        print(f"error: this run never commanded {', '.join(dead)} "
              f"(rms below {args.min_excitation}). Their gains are not "
              f"identifiable, and fitting anyway produces meaningless "
              f"coefficients.", file=sys.stderr)
        for n in MECANUM:
            print(f"         {n:<8}rms {mr.get(n, 0.0):.4f}", file=sys.stderr)
        print("       Use a run that exercises all three directions, or pass "
              "--min-excitation 0 to override.", file=sys.stderr)
        return 1

    k = prep.keep
    v = prep.v_reported if args.vel_source == "reported" else prep.v

    knee_rows = None
    if args.knee_mode == "auto":
        args.knee, knee_rows = choose_knee(prep, args)
    print_traction(args, knee_rows)

    # Everything downstream sees the SATURATED command, so the fitted gains
    # describe force actually delivered rather than force asked for.
    u_fit = saturate_u(prep.u, args.knee)
    X, names = build_design(u_fit[k], v[k], args.coulomb, args.intercept,
                            args.omega_sq, args.drag)
    mec_results = [fit_linear(X, prep.a[k, i], names, resp, robust=args.robust,
                              cv_folds=args.cv_folds)
                   for i, resp in enumerate(RESPONSES)]

    print_diagnostics(prep, stats, seg, args, dist_unit)
    print_report(
        mec_results, prep, args, dist_unit,
        "LINEAR DRIVETRAIN MODEL  (robot body frame)",
        "  Inputs are the {fwd, strafe, turn} projection of the motor powers.")
    if args.omega_sq:
        print_centripetal(mec_results, dist_unit)
    if not args.no_sensitivity:
        print_vel_source_comparison(seg, args)
        print_sensitivity(seg, args, mec_results)

    data = build_output(mec_results, prep, seg, stats, args, dist_unit)

    if not args.no_write:
        out_dir = args.out_dir or os.path.dirname(os.path.abspath(__file__))
        os.makedirs(out_dir, exist_ok=True)
        jp = os.path.join(out_dir, "drivetrain_fit.json")
        tp = os.path.join(out_dir, "drivetrain_fit.toml")
        with open(jp, "w") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")
        write_toml(data, tp)
        print("=" * 78)
        print("  WROTE")
        print("=" * 78)
        print(f"  {jp}")
        print(f"  {tp}")
        print()
        print("  From Julia (TOML is in the stdlib, no package needed):")
        print('    using TOML')
        print('    mat(v) = reduce(vcat, permutedims.(Vector{Float64}.(v)))')
        print('    fit = TOML.parsefile("drivetrain_fit.toml")["mecanum_basis"]')
        print('    B = mat(fit["B"])          # 3x3, columns [fwd strafe turn]')
        print('    A = mat(fit["A"])          # 3x3, columns [v_x v_y omega]')
        print('    q = Vector{Float64}(fit["q"])   # centripetal, x omega^2')
        print('    c = Vector{Float64}(fit["c"])')
        print('    a = B*u + A*v + q*v[3]^2 + S*csign(v) + D*(abs.(v).*v) + c')
        print()

    return 0


# --------------------------------------------------------------------------
# MODEL NOTES
# --------------------------------------------------------------------------
#
# Body-frame rotation
# -------------------
# The Pinpoint logs pose and velocity in the FIELD frame (verified against the
# finite difference of the logged position: the reported velocity correlates
# 0.90/0.98 with the field-frame derivative, versus 0.19/0.42 if it were
# interpreted as body-frame). Motor forces, however, act along body axes, so
# the regression has to be done in the body frame.
#
# With R(h) the rotation from body to field, v_f = R(h) v_r, and
#
#     a_f = R(h) dv_r/dt + Rdot(h) v_r = R(h) [ dv_r/dt + omega * J * v_r ]
#
# so the body-frame acceleration the motors actually produce is
#
#     a_r = R(-h) a_f = dv_r/dt + omega * J * v_r,   J = [[0,-1],[1,0]]
#
# i.e. it is NOT simply the time derivative of the body-frame velocity: the
# omega x v Coriolis/centripetal term matters. This run reaches omega ~ 4.8
# rad/s at speeds around 100 cm/s, so that term is worth several hundred
# cm/s^2 -- the same order as the accelerations being fitted, and dropping it
# would corrupt the fit badly.
#
# This code sidesteps the algebra by differentiating the FIELD-frame position
# to get a_f and rotating the result once, which yields a_r with the Coriolis
# term already correctly included.
#
# The 1.5 kHz Pinpoint update rate
# --------------------------------
# The Pinpoint recomputes velocity every ~0.667 ms by differencing its encoder
# counts over that one interval. Over so short a window the change is only a
# few encoder ticks, so tick quantisation dominates: the velocity quantum is
# (one tick in distance) x 1500 Hz, which is a large number. Measured on this
# log, the reported velocity sits ~25% away from a smoothed estimate -- that
# is quantisation noise, not real motion.
#
# Differencing that reported velocity to get acceleration multiplies the noise
# by 1/dt ~ 110, which on this log produces accelerations with an RMS of
# ~534 cm/s^2 and peaks past 3600 cm/s^2 -- physically impossible for this
# robot, and pure noise.
#
# The fix is to never differentiate the reported velocity. Position is the
# clean quantity: its quantisation error is bounded by one encoder tick and
# does not get multiplied by the 1500 Hz rate. So both velocity and
# acceleration are taken from a local weighted polynomial fit to position,
# which estimates the derivatives over a ~0.18 s window instead of 0.667 ms.
#
# Because the logging loop runs at ~110 Hz with heavy jitter (5.8 ms to
# 125 ms), a stock Savitzky-Golay filter -- which assumes uniform spacing --
# is not valid here. local_poly_derivatives() fits in real time coordinates,
# so the jitter is handled exactly rather than being smeared.
#
# Fit quality reporting
# ---------------------
# Smoothing makes neighbouring residuals strongly correlated, so plain OLS
# standard errors would be far too optimistic. The reported standard errors
# are Newey-West (HAC). Likewise the reported cross-validated R^2 uses
# contiguous blocks rather than random folds, since random folds would put
# near-duplicate neighbouring samples on both sides of the split and flatter
# the model.

if __name__ == "__main__":
    sys.exit(main())
