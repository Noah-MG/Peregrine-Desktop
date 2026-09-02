#!/usr/bin/env python3
"""
Diagnose how well a drivetrain regression actually fits, and suggest what
shape of model to use instead.

    py -3.12 calibration/diagnose_fit.py <calibration_log.csv> [options]

Writes a self-contained HTML report with the plots embedded. The point is not
to admire the fit but to answer one question: is a linear model the right
shape for this robot, and if not, which term is missing?

It reuses `fit_drivetrain.py`'s own preprocessing, so what gets plotted is
exactly what the fitter sees. Pass `--fit drivetrain_fit.toml` to mirror the
preprocessing a particular fit was produced with, which makes the report
reproducible against that fit rather than against defaults.

Requires numpy and matplotlib.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import os
import sys
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_drivetrain as fd

FG = "#1a1a1a"
GRID = "#d8d8d8"
ACCENT = "#2563eb"
WARN = "#dc2626"
OK = "#059669"


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------

def style(ax):
    ax.grid(True, color=GRID, lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=FG, labelsize=8)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def binned(x, y, nbins=14):
    """Mean of y within equal-count bins of x, plus a standard error."""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    edges = np.linspace(0, len(xs), nbins + 1).astype(int)
    bx, by, be = [], [], []
    for i in range(nbins):
        lo, hi = edges[i], edges[i + 1]
        if hi - lo < 3:
            continue
        bx.append(xs[lo:hi].mean())
        by.append(ys[lo:hi].mean())
        be.append(ys[lo:hi].std() / max(1.0, np.sqrt(hi - lo)))
    return np.array(bx), np.array(by), np.array(be)


# --------------------------------------------------------------------------
# Candidate model families
# --------------------------------------------------------------------------

def build_terms(u, v, eps=fd.COULOMB_EPS):
    """Every candidate regressor column, grouped into model families.

    Names match `fit_drivetrain.eval_terms` so a fitted family can be rolled
    forward at states it never saw, which is what the scoring needs.
    """
    fwd, strafe, turn = u[:, 0], u[:, 1], u[:, 2]
    vx, vy, w = v[:, 0], v[:, 1], v[:, 2]

    linear = [("fwd", fwd), ("strafe", strafe), ("turn", turn),
              ("v_x", vx), ("v_y", vy), ("omega", w)]
    omega_sq = [("omega_sq", w * w)]
    coulomb = [("sgn_v_x", fd.csign(vx, eps[0])),
               ("sgn_v_y", fd.csign(vy, eps[1])),
               ("sgn_omega", fd.csign(w, eps[2]))]
    drag = [("absv_v_x", np.abs(vx) * vx), ("absv_v_y", np.abs(vy) * vy),
            ("absv_omega", np.abs(w) * w)]
    # No control-squared family, deliberately. A u^2 column is even in u, so
    # it predicts the same force for full forward and full reverse, which no
    # drivetrain does; what it actually absorbs is the saturation curvature
    # that `traction_gain` already models, and it does so in a form the fitted
    # model cannot carry (`A_uu` is structurally zero, see TABLE_FORMAT §8.2).
    # A family that cannot be shipped is not worth scoring.
    inter = []
    for un, uu in (("fwd", fwd), ("strafe", strafe), ("turn", turn)):
        for vn, vv in (("v_x", vx), ("v_y", vy), ("omega", w)):
            inter.append((un + "*" + vn, uu * vv))

    fams = {
        "linear only": linear,
        "+ Coulomb": linear + coulomb,
        "+ quadratic drag": linear + drag,
        "+ Coulomb + drag (current)": linear + coulomb + drag,
        "+ omega^2": linear + omega_sq,
        "+ omega^2 + Coulomb + drag": linear + omega_sq + coulomb + drag,
        "+ control x velocity": linear + coulomb + drag + inter,
    }
    return linear, fams


def design(terms, n):
    X = np.column_stack([t[1] for t in terms] + [np.ones(n)])
    names = [t[0] for t in terms] + ["const"]
    return X, names


def evaluate(fams, T, U, V, A, cv_folds, horizon=0.5):
    """
    Score every family the way the solver uses the model.

    Acceleration R^2 is a poor yardstick: the response is a twice-smoothed
    derivative, so a model can look bad while predicting motion well, or look
    fine while being useless. Each family is therefore fitted as usual and
    then integrated forward, scoring the CHANGE in velocity over `horizon` on
    the half of the run it was not fitted to. The null model -- "velocity does
    not change" -- is reported alongside so the numbers mean something.
    """
    out = {}
    n = A.shape[0]
    for fname, terms in fams.items():
        X, names = design(terms, n)
        row = {}
        coef = np.zeros((3, X.shape[1]))
        for i, resp in enumerate(fd.RESPONSES):
            r = fd.fit_linear(X, A[:, i], names, resp, cv_folds=cv_folds)
            coef[i] = r.coef
            row[resp] = {"r2": r.r2, "cv": r.cv_r2, "rmse": r.rmse,
                         "n_terms": X.shape[1]}
        roll, null = fd.rollout_r2(T, U, V, coef, names, horizon=horizon)
        for i, resp in enumerate(fd.RESPONSES):
            row[resp]["roll"] = roll[i]
            row[resp]["null"] = null[i]
        out[fname] = row
    return out


def curvature_scan(resid, regs):
    """
    Per regressor, how much leftover error does a quadratic in it explain?

    This is the direct answer to "which term is missing". The linear model has
    already absorbed everything linear, so whatever a squared term can still
    pick up is exactly the curvature that model cannot represent. Reported as
    the extra fraction of residual variance explained, so it is comparable
    across regressors.
    """
    out = []
    for name, x in regs:
        sd = x.std()
        if sd < 1e-12:
            continue
        z = (x - x.mean()) / sd
        ss_tot = float(np.sum((resid - resid.mean()) ** 2))
        if ss_tot <= 0:
            continue
        Xq = np.column_stack([z, z * z, np.ones(len(z))])
        bq, *_ = np.linalg.lstsq(Xq, resid, rcond=None)
        gq = 1.0 - float(np.sum((resid - Xq @ bq) ** 2)) / ss_tot
        Xl = np.column_stack([z, np.ones(len(z))])
        bl, *_ = np.linalg.lstsq(Xl, resid, rcond=None)
        gl = 1.0 - float(np.sum((resid - Xl @ bl) ** 2)) / ss_tot
        out.append((name, max(0.0, gq - gl)))
    out.sort(key=lambda t: -t[1])
    return out


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def fig_overview(A, P, t, results):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for i, resp in enumerate(fd.RESPONSES):
        unit = "rad/s^2" if resp == "alpha" else "cm/s^2"
        ax = axes[0, i]
        ax.scatter(A[:, i], P[:, i], s=4, alpha=0.25, color=ACCENT, lw=0)
        lo = min(A[:, i].min(), P[:, i].min())
        hi = max(A[:, i].max(), P[:, i].max())
        ax.plot([lo, hi], [lo, hi], color=WARN, lw=1.2, ls="--")
        r = results[resp]
        ax.set_title(resp + ": R2=%.3f  CV=%.3f" % (r["r2"], r["cv"]),
                     fontsize=10)
        ax.set_xlabel("measured [" + unit + "]")
        ax.set_ylabel("predicted [" + unit + "]")
        style(ax)

        ax = axes[1, i]
        ax.plot(t, A[:, i], lw=0.9, color=FG, label="measured")
        ax.plot(t, P[:, i], lw=0.9, color=ACCENT, label="model", alpha=0.85)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(unit)
        ax.legend(fontsize=7, frameon=False)
        style(ax)
    fig.suptitle("Fit overview: predicted against measured, and both over time",
                 fontsize=12, color=FG)
    fig.tight_layout()
    return fig_to_b64(fig)


def fig_residual_structure(R, regs):
    nr, nc = len(fd.RESPONSES), len(regs)
    fig, axes = plt.subplots(nr, nc, figsize=(2.35 * nc, 2.5 * nr),
                             squeeze=False)
    for i, resp in enumerate(fd.RESPONSES):
        for j, (name, x) in enumerate(regs):
            ax = axes[i][j]
            ax.axhline(0, color=WARN, lw=0.9, ls="--", zorder=1)
            ax.scatter(x, R[:, i], s=3, alpha=0.15, color=ACCENT, lw=0, zorder=2)
            bx, by, be = binned(x, R[:, i])
            if len(bx):
                ax.errorbar(bx, by, yerr=be, color=FG, lw=1.4, marker="o",
                            ms=3, capsize=2, zorder=3)
            if i == 0:
                ax.set_title(name, fontsize=9)
            if j == 0:
                ax.set_ylabel(resp + " residual", fontsize=8)
            style(ax)
    fig.suptitle("Residual against each regressor. A flat black line means "
                 "that variable is fully explained; a slope or a bend means "
                 "it is not.", fontsize=11, color=FG)
    fig.tight_layout()
    return fig_to_b64(fig)


def fig_families(table, ref):
    """
    Plot each family as a *change* against the current model.

    Absolute CV R2 is useless here: every family lands within a percent or
    two of the others, so a plain bar chart is seven bars all pinned at the
    right edge. What matters is whether a family beats the one in use, and by
    how much, so that is what gets drawn.
    """
    fams = [f for f in table if f != ref]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for i, resp in enumerate(fd.RESPONSES):
        ax = axes[i]
        base = table[ref][resp]["roll"]
        deltas = []
        for f in fams:
            v = table[f][resp]["roll"]
            deltas.append((v - base) if np.isfinite(v) and np.isfinite(base)
                          else 0.0)
        colors = [OK if d > 0.01 else (WARN if d < -0.01 else "#9ca3af")
                  for d in deltas]
        ax.barh(range(len(fams)), deltas, color=colors)
        ax.axvline(0, color=FG, lw=1.1)
        ax.set_yticks(range(len(fams)))
        ax.set_yticklabels(fams if i == 0 else [""] * len(fams), fontsize=8)
        ax.invert_yaxis()
        span = max(0.02, max(abs(d) for d in deltas) * 1.35)
        ax.set_xlim(-span, span)
        for j, d in enumerate(deltas):
            ax.text(d + (span * 0.03 if d >= 0 else -span * 0.03), j,
                    "%+.4f" % d, va="center",
                    ha="left" if d >= 0 else "right", fontsize=7, color=FG)
        ax.set_xlabel("change in rollout R2 vs current")
        ax.set_title("%s   (current = %.4f)" % (resp, base), fontsize=10)
        style(ax)
    fig.suptitle("Would another model form do better at predicting motion? "
                 "Bars right of the line beat the current model.",
                 fontsize=11, color=FG)
    fig.tight_layout()
    return fig_to_b64(fig)


def fig_knee(rows, chosen):
    """Rollout score against traction knee, with 'no saturation' on the left."""
    labels = ["none" if k is None else "%.2f" % k for k, _, _ in rows]
    means = [m for _, _, m in rows]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    colors = [OK if (k == chosen) else ACCENT for k, _, _ in rows]
    ax.bar(range(len(rows)), means, color=colors)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("traction knee (command magnitude at which the tyres give up)")
    ax.set_ylabel("mean rollout R2")
    lo = min(means); hi = max(means)
    ax.set_ylim(lo - 0.08 * (hi - lo + 1e-6), hi + 0.08 * (hi - lo + 1e-6))
    for i, m in enumerate(means):
        ax.text(i, m, "%.3f" % m, ha="center", va="bottom", fontsize=8, color=FG)
    style(ax)
    fig.suptitle("Wheel slip: how well the model predicts motion at each knee "
                 "(green = chosen)", fontsize=11, color=FG)
    fig.tight_layout()
    return fig_to_b64(fig)


def fig_coverage(u, v):
    pairs = [("fwd", u[:, 0], "strafe", u[:, 1]),
             ("fwd", u[:, 0], "turn", u[:, 2]),
             ("strafe", u[:, 1], "turn", u[:, 2]),
             ("speed", np.hypot(v[:, 0], v[:, 1]), "omega", v[:, 2])]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, (xn, x, yn, y) in zip(axes, pairs):
        ax.hexbin(x, y, gridsize=26, cmap="Blues", mincnt=1, linewidths=0)
        ax.set_xlabel(xn)
        ax.set_ylabel(yn)
        style(ax)
    fig.suptitle("Where the run actually sampled. Blank regions are where the "
                 "model is extrapolating rather than fitting.",
                 fontsize=11, color=FG)
    fig.tight_layout()
    return fig_to_b64(fig)


def fig_residual_diag(R, P):
    from scipy.special import erfinv
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))
    for i, resp in enumerate(fd.RESPONSES):
        ax = axes[i][0]
        ax.scatter(P[:, i], R[:, i], s=4, alpha=0.2, color=ACCENT, lw=0)
        ax.axhline(0, color=WARN, lw=0.9, ls="--")
        bx, by, _ = binned(P[:, i], R[:, i])
        if len(bx):
            ax.plot(bx, by, color=FG, lw=1.4, marker="o", ms=3)
        ax.set_xlabel("predicted")
        ax.set_ylabel(resp + " residual")
        style(ax)

        # Slow ACF decay means the error is structured in time, which points
        # at missing dynamics (a lag) rather than a missing power of a state.
        ax = axes[i][1]
        r = R[:, i] - R[:, i].mean()
        nlag = max(2, min(60, len(r) // 4))
        denom = float(r @ r) or 1.0
        acf = [float(r[k:] @ r[:len(r) - k]) / denom for k in range(nlag)]
        ax.bar(range(nlag), acf, color=ACCENT, width=0.9)
        ax.axhline(0, color=FG, lw=0.8)
        ax.set_xlabel("lag [samples]")
        ax.set_ylabel("residual ACF")
        style(ax)

        ax = axes[i][2]
        q = np.sort(r) / (r.std() or 1.0)
        pp = (np.arange(len(q)) + 0.5) / len(q)
        theo = np.sqrt(2) * erfinv(2 * pp - 1)
        ax.scatter(theo, q, s=3, alpha=0.3, color=ACCENT, lw=0)
        ax.plot([theo.min(), theo.max()], [theo.min(), theo.max()],
                color=WARN, lw=1.0, ls="--")
        ax.set_xlabel("normal quantile")
        ax.set_ylabel("residual quantile")
        style(ax)
    fig.suptitle("Residual diagnostics: structure against prediction, "
                 "correlation in time, and departure from normal",
                 fontsize=11, color=FG)
    fig.tight_layout()
    return fig_to_b64(fig)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif;
       max-width: 1180px; margin: 0 auto; padding: 32px 24px 64px;
       color: #1a1a1a; background: #fff; line-height: 1.55; }
h1 { font-size: 26px; margin-bottom: 4px; }
h2 { font-size: 19px; margin-top: 38px; border-bottom: 1px solid #e5e5e5;
     padding-bottom: 6px; }
.sub { color: #666; font-size: 13px; margin-top: 0; }
img { width: 100%; border: 1px solid #e5e5e5; border-radius: 6px;
      margin: 10px 0; }
table { border-collapse: collapse; width: 100%; font-size: 13px;
        margin: 12px 0; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #eee; }
th { background: #f7f7f7; font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.verdict { background: #f0f7ff; border-left: 4px solid #2563eb;
           padding: 14px 18px; margin: 18px 0; border-radius: 4px; }
.warn { background: #fff5f5; border-left: 4px solid #dc2626; }
.good { background: #f0fdf4; border-left: 4px solid #059669; }
.note { color: #666; font-size: 13px; }
code { background: #f4f4f4; padding: 1px 5px; border-radius: 3px;
       font-size: 12px; }
"""


def esc(s):
    return html.escape(str(s))


def build_report(ctx) -> str:
    p = []
    a = p.append
    a("<!-- generated by diagnose_fit.py -->")
    a("<style>" + CSS + "</style>")
    a("<h1>Drivetrain regression diagnostics</h1>")
    a('<p class="sub">' + esc(ctx["source"]) + " &middot; " +
      esc(ctx["generated"]) + " &middot; " + str(ctx["n"]) +
      " samples used</p>")

    for cls, text in ctx["verdicts"]:
        a('<div class="verdict ' + cls + '">' + text + "</div>")

    a("<h2>1. Does the model fit at all?</h2>")
    a("<p>Points on the dashed line are perfect predictions. Systematic bend "
      "away from it means the model has the wrong shape, not just noise.</p>")
    a('<img src="data:image/png;base64,' + ctx["fig_overview"] + '">')

    a("<h2>2. Which variable is the model getting wrong?</h2>")
    a("<p>Each panel plots the leftover error against one regressor. If the "
      "model has captured that variable properly the black line sits flat on "
      "zero. A tilt means a missing linear effect; a bend means a missing "
      "nonlinear one, and the shape of the bend tells you which term to add."
      "</p>")
    a('<img src="data:image/png;base64,' + ctx["fig_resid"] + '">')

    a("<h3>Ranked: where curvature is hiding</h3>")
    a("<p class='note'>Extra fraction of the leftover error that a squared "
      "term in each variable would explain. <b>This is a pointer for "
      "investigation, not a recommendation.</b> Estimating acceleration by "
      "smoothing leaves a signal-dependent bias that looks exactly like "
      "curvature, so a high score here regularly survives into a term that "
      "does not help at all. Section 4 is what decides; this only suggests "
      "where to look.</p>")
    a("<table><tr><th>response</th><th>regressor</th>"
      "<th class='num'>variance explained by a squared term</th></tr>")
    for resp, rows in ctx["curv"].items():
        for name, gain in rows[:4]:
            flag = " &larr;" if gain > 0.02 else ""
            a("<tr><td>" + esc(resp) + "</td><td><code>" + esc(name) +
              "</code></td><td class='num'>" + ("%.4f" % gain) + flag +
              "</td></tr>")
    a("</table>")

    a("<h2>3. Wheel slip</h2>")
    a("<p>Past a certain demand the tyres stop delivering and extra command "
      "buys no extra force. That is fitted as a knee: "
      "<code>u = u_raw &middot; tanh(m/knee)/(m/knee)</code> with "
      "<code>m = &#8214;u_raw&#8214;</code>, applied <b>before</b> the "
      "control gains. Scored by integrating the model, because a saturating "
      "term will always reduce the fit residual somewhere.</p>")
    a("<p>Everything else in this report uses the knee shown here. "
      + ("<b>No saturation</b> was applied." if ctx["knee"] is None else
         "The knee in force is <b>%.2f</b> (%s)." % (ctx["knee"], ctx["knee_mode"]))
      + "</p>")
    a('<img src="data:image/png;base64,' + ctx["fig_knee"] + '">')
    a("<p class='note'>A knee only shows up if the run actually crosses the "
      "traction limit. A run that never reaches it has nothing to find, and "
      "one that never comes back below it just looks like a lower gain "
      "&mdash; both correctly come back as no saturation. And the knee "
      "describes <b>this floor</b>: a grippier surface slips later.</p>")

    a("<h2>4. What model should you actually use?</h2>")
    a("<p class='note'>null (velocity unchanged) rollout score: "
      + ", ".join("%s %.3f" % (r, ctx["null"][r]) for r in fd.RESPONSES)
      + "</p>")
    a("<p>Each candidate family is refitted on the same data, then "
      "<b>integrated forward</b> and scored on how well it predicts the "
      "change in velocity over half a second, on the part of the run it was "
      "not fitted to. That is what the solver actually does with the model. "
      "Acceleration R2 is shown too but should not drive the decision: the "
      "response is a twice-smoothed derivative, so it can damn a model that "
      "predicts motion perfectly well. The null column is the score for "
      "assuming the velocity simply does not change.</p>")
    a('<img src="data:image/png;base64,' + ctx["fig_fams"] + '">')
    a("<table><tr><th>family</th><th class='num'>terms</th>")
    for r in fd.RESPONSES:
        a("<th class='num'>" + r + " roll</th><th class='num'>" + r + " accR2</th>")
    a("</tr>")
    for fam, row in ctx["table"].items():
        a("<tr><td>" + esc(fam) + "</td><td class='num'>" +
          str(row[fd.RESPONSES[0]]["n_terms"]) + "</td>")
        for r in fd.RESPONSES:
            rl = row[r]["roll"]
            a("<td class='num'>" +
              ("%.3f" % rl if np.isfinite(rl) else "n/a") + "</td>")
            a("<td class='num'>%.3f</td>" % row[r]["r2"])
        a("</tr>")
    a("</table>")

    a("<h2>5. Is the data good enough to tell?</h2>")
    a("<p>A model can only be trusted where the run actually went. Blank "
      "regions below are combinations the robot never tried, so any "
      "coefficient governing them is guesswork.</p>")
    a('<img src="data:image/png;base64,' + ctx["fig_cov"] + '">')

    a("<h2>6. Residual diagnostics</h2>")
    a("<p>Left: error against prediction, which should be a flat band. "
      "Middle: how correlated the error is with itself over time. Some "
      "correlation is expected because the derivatives are smoothed, but a "
      "long slow decay suggests the drivetrain has lag the static model "
      "cannot express. Right: whether the errors are normally distributed, "
      "where a heavy tail usually means a few bad samples rather than a bad "
      "model.</p>")
    a('<img src="data:image/png;base64,' + ctx["fig_diag"] + '">')

    a("<h2>Reproducing this</h2>")
    a("<pre><code>" + esc(ctx["cmd"]) + "</code></pre>")
    a("<p class='note'>Preprocessing used: " + esc(ctx["preproc"]) + "</p>")
    return "\n".join(p)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def self_test(cv_folds=5) -> int:
    """
    Check that the family comparison actually identifies the right model.

    Two synthetic robots are built: one that genuinely obeys linear + omega^2,
    and one with a real quadratic-drag term added. The comparison should pick
    the current form for the first and quadratic drag for the second. Without
    this the tool could confidently recommend nonsense.
    """
    import math

    B = np.array([[-420.0, 5.0, 40.0], [30.0, 260.0, -15.0],
                  [-2.0, 3.0, -42.0]])
    Amat = np.array([[-0.80, -0.10, -4.0], [-0.05, -0.75, 3.0],
                     [0.004, -0.002, -1.90]])
    qv = np.array([-3.5, 2.1, 0.0])

    def synth(drag, seed=11, dur=14.0, hz=150.0):
        rng = np.random.default_rng(seed)
        fine = 1.0 / 2000.0
        nf = int(dur / fine)

        def band(_):
            ph = rng.uniform(0, 2 * np.pi, 5)
            fr = rng.uniform(0.15, 0.9, 5)
            tt = np.arange(nf) * fine
            return np.clip(sum(np.sin(2 * np.pi * f * pp + p)
                               for f, p in zip(fr, ph) for pp in [tt]) / 2.2,
                           -1, 1)
        U = np.column_stack([band(0), band(1), band(2) * 0.6])
        v = np.zeros(2); w = 0.0; h = 0.0; pos = np.zeros(2)
        J = np.array([[0.0, -1.0], [1.0, 0.0]])
        out = np.empty((nf, 7))
        for i in range(nf):
            st = np.array([v[0], v[1], w])
            a = B @ U[i] + Amat @ st + qv * w * w
            if drag:
                a = a + drag * np.abs(st) * st
            dv = a[:2] - w * (J @ v)
            R = np.array([[math.cos(h), -math.sin(h)],
                          [math.sin(h), math.cos(h)]])
            out[i] = (i * fine, pos[0], pos[1], h, *(R @ v), w)
            pos = pos + (R @ v) * fine
            v = v + dv * fine
            h = h + w * fine
            w = w + a[2] * fine
        ts, tc = [], 0.0
        while tc < dur - 0.05:
            ts.append(tc)
            tc += (1.0 / hz) * (1.0 + 0.35 * (rng.random() - 0.5))
        idx = np.clip((np.array(ts) / fine).astype(int), 0, nf - 1)
        sm = out[idx]
        wheels = (U[idx] @ np.linalg.pinv(fd.MEC_MIX[:3].T)).T
        return np.column_stack([
            sm[:, 0] * 1000.0, *wheels,
            sm[:, 1] + rng.normal(0, 0.02, len(sm)),
            sm[:, 2] + rng.normal(0, 0.02, len(sm)),
            sm[:, 3],
            sm[:, 4] + rng.normal(0, 4.0, len(sm)),
            sm[:, 5] + rng.normal(0, 4.0, len(sm)),
            sm[:, 6] + rng.normal(0, 0.02, len(sm))])

    def rank(rows):
        seg = fd.Segment(raw=rows, source="<synthetic>", index=0)
        a2 = argparse.Namespace(window=0.22, order=3, max_gap=0.05,
                                units="cm", smooth_inputs=True,
                                v_min=2.0, w_min=0.05)
        prep = fd.prepare(seg, a2)
        k = prep.keep
        u = prep.u[k] @ fd.MEC_MIX[:3].T
        _, fams = build_terms(u, prep.v[k])
        tab = evaluate(fams, prep.t[k], u, prep.v[k], prep.a[k], cv_folds)
        mean = {f: float(np.nanmean([tab[f][r]["roll"] for r in fd.RESPONSES]))
                for f in tab}
        return sorted(mean.items(), key=lambda t: -t[1]), mean

    print()
    print("=" * 70)
    print("  SELF-TEST -- can the family comparison find the true model?")
    print("=" * 70)
    ok = True
    # The synthetic robots genuinely contain omega^2, and the second also has
    # quadratic drag, so the winner must be a family that can express what is
    # actually there.
    for label, drag, accept in (
            ("linear + omega^2, no drag", 0.0,
             ("+ omega^2", "+ omega^2 + Coulomb + drag")),
            ("linear + omega^2 + quadratic drag", -0.010,
             ("+ omega^2 + Coulomb + drag", "+ quadratic drag",
              "+ Coulomb + drag (current)", "+ control x velocity"))):
        ranked, mean = rank(synth(drag))
        print()
        print("  ground truth: " + label)
        for f, sc in ranked:
            print("     %-30s mean rollout R2 %7.4f%s"
                  % (f, sc, "   <== picked" if f == ranked[0][0] else ""))
        good = ranked[0][0] in accept
        ok = ok and good
        print("     -> %s" % ("PASS" if good else "FAIL, expected one of "
                              + str(accept)))
    print()
    print("  %s" % ("PASS" if ok else "FAIL"))
    print()
    print("  Note this validates the family comparison in section 3, which is")
    print("  the part that recommends a model. The curvature scan in section 2")
    print("  is more sensitive and can flag structure that the rollout score")
    print("  then rejects; that is why the report defers to this test.")
    print()
    return 0 if ok else 1


def mirror_preprocessing(fit_path, args):
    """
    Copy the preprocessing settings out of an existing drivetrain_fit.toml.

    Without this the report would be diagnosing a slightly different pipeline
    than the fit it is supposed to be judging.
    """
    import tomllib
    with open(fit_path, "rb") as fh:
        cfg = tomllib.load(fh)
    pp = cfg.get("preprocessing", {})
    # The traction knee is part of what the fit did to the data, so a report
    # that ignores it is describing a different model than the one on disk.
    args.knee = pp.get("traction_knee", None)
    args.knee_mode = "mirrored from --fit"
    args.coulomb = bool(pp.get("coulomb_terms", args.coulomb))
    args.drag = bool(pp.get("drag_terms", args.drag))
    args.omega_sq = bool(pp.get("omega_sq_term", args.omega_sq))
    args.window = float(pp.get("window_halfwidth_s", args.window))
    args.order = int(pp.get("poly_order", args.order))
    args.max_gap = float(pp.get("max_gap_s", args.max_gap))
    args.units = str(pp.get("units", "cm")) if "units" in pp else args.units
    args.smooth_inputs = bool(pp.get("inputs_band_limited", args.smooth_inputs))
    args.v_min = float(pp.get("v_min", args.v_min))
    args.w_min = float(pp.get("w_min", args.w_min))
    return cfg


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Diagnose a drivetrain regression and suggest a model form.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("log", nargs="?", help="calibration_log_*.csv to analyse")
    p.add_argument("--self-test", action="store_true",
                   help="check the model selection against synthetic robots "
                        "with a known true model, then exit")
    p.add_argument("--fit", default=None,
                   help="a drivetrain_fit.toml whose preprocessing settings "
                        "should be mirrored, so the report matches that fit")
    p.add_argument("-o", "--out", default=None,
                   help="output HTML path (default: alongside the log)")
    p.add_argument("--window", type=float, default=0.22)
    p.add_argument("--order", type=int, default=3)
    p.add_argument("--max-gap", type=float, default=0.05)
    p.add_argument("--units", choices=["cm", "m"], default="cm")
    p.add_argument("--no-smooth-inputs", dest="smooth_inputs",
                   action="store_false")
    p.add_argument("--v-min", type=float, default=2.0)
    p.add_argument("--w-min", type=float, default=0.05)
    # Model-term switches, so the shared fitter helpers can be called with
    # this namespace. Defaults match fit_drivetrain's own.
    p.add_argument("--no-coulomb", dest="coulomb", action="store_false")
    p.add_argument("--no-drag", dest="drag", action="store_false")
    p.add_argument("--omega-sq", dest="omega_sq", action="store_true")
    p.add_argument("--no-intercept", dest="intercept", action="store_false")
    p.add_argument("--traction-knee", default=None,
                   help="wheel-slip knee: a positive number, or 'auto' to "
                        "search for it. A knee is required. Default is "
                        "whatever --fit used, or auto.")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--segment", type=int, default=None)
    p.add_argument("--open", action="store_true",
                   help="open the report in a browser when done")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test(args.cv_folds)
    if not args.log:
        print("error: give a log path, or --self-test", file=sys.stderr)
        return 2
    if not os.path.exists(args.log):
        print("error: no such file: " + args.log, file=sys.stderr)
        return 2

    if args.fit:
        if not os.path.exists(args.fit):
            print("error: no such fit file: " + args.fit, file=sys.stderr)
            return 2
        mirror_preprocessing(args.fit, args)
        print("  mirroring preprocessing from " + args.fit)

    segments, stats = fd.load_segments(args.log)
    usable = [s for s in segments if fd.validate(s)[0]]
    if not usable:
        for s in segments:
            print("  unusable segment " + str(s.index) + ": " +
                  fd.validate(s)[1], file=sys.stderr)
        print("error: no usable segment in this log", file=sys.stderr)
        return 1
    seg = (next((s for s in usable if s.index == args.segment), None)
           if args.segment is not None else max(usable, key=lambda s: s.n))
    if seg is None:
        print("error: segment not found", file=sys.stderr)
        return 1

    prep = fd.prepare(seg, args)
    k = prep.keep
    if k.sum() < 50:
        print("error: only " + str(int(k.sum())) + " usable samples",
              file=sys.stderr)
        return 1

    # Resolve the traction knee before anything is fitted or plotted. The
    # command the drivetrain actually delivers is the saturated one, so every
    # residual, coefficient and rollout below has to use it too.
    knee_rows = None
    override = getattr(args, "traction_knee", None)
    if override is not None:
        tk = str(override).strip().lower()
        if tk in ("off", "none", "0"):
            print("error: the traction knee cannot be switched off -- the "
                  "model requires one. Use 'auto' or a positive number.",
                  file=sys.stderr)
            return 2
        elif tk == "auto":
            args.knee, knee_rows = fd.choose_knee(prep, args)
            args.knee_mode = "searched"
        else:
            args.knee, args.knee_mode = float(tk), "fixed (--traction-knee)"
    elif not hasattr(args, "knee"):
        args.knee, knee_rows = fd.choose_knee(prep, args)
        args.knee_mode = "searched"
    if knee_rows is None:
        # Always produce the comparison table, even when the knee came from a
        # file -- seeing what saturation buys is the point of the report.
        _, knee_rows = fd.choose_knee(prep, args)
    print("  traction knee: %s (%s)"
          % ("none" if args.knee is None else "%.2f" % args.knee,
             args.knee_mode))

    u_sat = fd.saturate_u(prep.u, args.knee)
    u_mec = u_sat[k] @ fd.MEC_MIX[:3].T
    v = prep.v[k]
    A = prep.a[k]
    t = prep.t[k]
    n = int(k.sum())
    print("  " + str(n) + " samples, " +
          "%.1f" % (t.max() - t.min()) + " s of run")

    linear, fams = build_terms(u_mec, v)
    table = evaluate(fams, t, u_mec, v, A, args.cv_folds)

    # The reference fit is the one the solver actually uses today.
    ref_name = "+ Coulomb + drag (current)"
    Xr, namesr = design(fams[ref_name], n)
    results, P, R = {}, np.zeros_like(A), np.zeros_like(A)
    for i, resp in enumerate(fd.RESPONSES):
        r = fd.fit_linear(Xr, A[:, i], namesr, resp, cv_folds=args.cv_folds)
        results[resp] = {"r2": r.r2, "cv": r.cv_r2, "rmse": r.rmse}
        P[:, i] = Xr @ r.coef
        R[:, i] = A[:, i] - P[:, i]

    regs = linear + [("omega^2", v[:, 2] ** 2)]
    curv = {resp: curvature_scan(R[:, i], regs)
            for i, resp in enumerate(fd.RESPONSES)}

    # ---- verdicts -------------------------------------------------------
    verdicts = []
    mean_cv = {f: float(np.nanmean([table[f][r]["roll"] for r in fd.RESPONSES]))
               for f in table}
    best = max(mean_cv, key=lambda f: mean_cv[f])
    cur = mean_cv[ref_name]
    gain = mean_cv[best] - cur
    # A model form has to earn its extra terms by a clear margin, not by the
    # third decimal place. Below this the difference is not distinguishable
    # from which half of the run happened to be held out.
    WORTH_SWITCHING = 0.03
    if best == ref_name or gain < WORTH_SWITCHING:
        verdicts.append(("good",
            "<b>The current model form is the right one.</b> No candidate "
            "family beat <code>" + ref_name + "</code> by a clear margin "
            "when the model is integrated forward, so the extra terms are "
            "fitting noise. Mean rollout R2 is %.3f." % cur))
    else:
        verdicts.append(("",
            "<b>Worth trying <code>" + esc(best) + "</code>.</b> "
            "It scores %.3f mean rollout R2 against %.3f for the current "
            "<code>%s</code>, a gain of %.3f."
            % (mean_cv[best], cur, ref_name, gain)))

    rollv = {r: table[ref_name][r]["roll"] for r in fd.RESPONSES}
    if args.knee is not None and knee_rows:
        flat = [r for r in knee_rows if r[0] is None]
        best = max(knee_rows, key=lambda r: r[2])
        if flat and best[0] is not None:
            verdicts.append(("", (
                "<b>Wheel slip is being modelled</b> with a knee at %.2f, "
                "worth %+.3f mean rollout R2 over no saturation. Everything "
                "below already has it applied. Note it describes this floor "
                "only." % (args.knee, best[2] - flat[0][2]))))

    worst = min(fd.RESPONSES, key=lambda r: rollv[r]
                if np.isfinite(rollv[r]) else -9)
    if np.isfinite(rollv[worst]) and rollv[worst] < 0.3:
        verdicts.append(("warn",
            "<b>" + worst + " is poorly predicted</b> (rollout R2 %.3f). "
            "Usually this means that axis was barely excited: a command held "
            "steady tells you the steady-state gain but nothing about the "
            "transient, and the transient is what these coefficients are. "
            "Check the coverage plots before trusting them." % rollv[worst]))

    # The curvature scan used to raise a verdict of its own. It no longer
    # does. It is far more sensitive than the rollout comparison, and on a
    # smoothed second derivative it reliably finds structure that adding the
    # term does not actually fix -- so it produced a steady trickle of "add
    # this coefficient" advice that was mostly noise. The ranked table in
    # section 2 is still there as a pointer for anyone investigating; it just
    # no longer presents itself as a recommendation.

    ctx = {
        "source": os.path.basename(args.log),
        "generated": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M"),
        "n": n,
        "verdicts": verdicts,
        "fig_overview": fig_overview(A, P, t, results),
        "fig_resid": fig_residual_structure(R, regs),
        "fig_fams": fig_families(table, ref_name),
        "fig_cov": fig_coverage(u_mec, v),
        "fig_diag": fig_residual_diag(R, P),
        "curv": curv,
        "table": table,
        "knee": args.knee,
        "knee_mode": args.knee_mode,
        "fig_knee": fig_knee(knee_rows, args.knee),
        "null": {r: table[ref_name][r]["null"] for r in fd.RESPONSES},
        "cmd": "py -3.12 calibration/diagnose_fit.py " + args.log +
               (" --fit " + args.fit if args.fit else ""),
        "preproc": ("window=%.3fs order=%d max_gap=%.3fs units=%s "
                    "smooth_inputs=%s v_min=%.2f w_min=%.3f"
                    % (args.window, args.order, args.max_gap, args.units,
                       args.smooth_inputs, args.v_min, args.w_min)),
    }

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.log)),
        os.path.splitext(os.path.basename(args.log))[0] + "_diagnostics.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(build_report(ctx))

    print()
    import re, textwrap
    for _, v_ in verdicts:
        # Strip tags and decode entities: these strings are written for the
        # HTML report but also get shown in the terminal.
        plain = html.unescape(re.sub("<[^>]+>", "", v_))
        for j, line in enumerate(textwrap.wrap(plain, 72)):
            print(("  * " if j == 0 else "    ") + line)
    print()
    print("  report: " + out)
    if args.open:
        import webbrowser
        webbrowser.open("file:///" + out.replace("\\", "/"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
