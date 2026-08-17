# Drivetrain calibration

Reads a `calibration_log_*.csv` written by `CalibrationLogger` on the
robot, fits a linear drivetrain model, and emits the coefficients as
JSON and TOML for the downstream Julia solver.

## Usage

```bash
py -3.12 calibration/fit_drivetrain.py path/to/calibration_log_XXXX.csv
```

Requires `numpy`. Writes `drivetrain_fit.json` and `drivetrain_fit.toml`
next to the script unless `-o/--out-dir` says otherwise.

Useful flags:

| flag | effect |
| --- | --- |
| `--window S` | halfwidth of the derivative window, seconds (default 0.22) |
| `--order N` | polynomial order for the derivative fit (default 3, min 2) |
| `--units {cm,m}` | distance unit for the fit and the output (default `cm`, matching `RobotParams.distanceUnit`) |
| `--vel-source {position,reported}` | velocity regressor: smoothed from position, or the Pinpoint's own output |
| `--no-intercept` | drop the constant term |
| `--no-omega-sq` | pin the `omega^2` centripetal term to zero (output shape unchanged) |
| `--min-excitation X` | refuse to fit if any of fwd/strafe/turn is barely commanded (default 0.01) |
| `--self-test` | recover a known model from a simulated drive, then exit |
| `--coulomb` | add `sign(v)` break-away friction terms (still linear in the parameters) |
| `--robust` | Huber IRLS instead of ordinary least squares |
| `--no-smooth-inputs` | leave the motor powers unfiltered (biases gains toward zero; diagnostic only) |
| `--segment N` | pick a run when append-mode collision put several in one file |
| `--no-sensitivity` | skip the window sweep and velocity-source comparison |
| `--no-write` | report to the terminal only |

## The model

Three linear equations, one per body-frame degree of freedom:

```
a = B*u + A*v + q*omega^2 + c
```

| symbol | meaning |
| --- | --- |
| `a` | `[a_x, a_y, alpha]`, robot body frame, `cm/s^2` and `rad/s^2` |
| `u` | `[fwd, strafe, turn]` — the mecanum projection of the wheel powers |
| `v` | `[v_x, v_y, omega]`, robot body frame |
| `B` | input gain, 3x3 |
| `A` | velocity feedback: drag and back-EMF, 3x3 |
| `q` | centripetal, times `omega^2`; estimates `-delta`, 3 |
| `c` | constant offset, 3 |

Inputs are **always** the `{fwd, strafe, turn}` projection. Regressing on the
four wheel powers directly is not supported — see *Why mecanum only* below.

### The omega² centripetal term

A tracking point offset by `delta` from the true centre of rotation sees

```
a_P = a_C + alpha x delta - omega^2 * delta
```

so the `omega^2` coefficient estimates `-delta` directly. It should come out
near `-delta_x` in the `a_x` row, `-delta_y` in the `a_y` row, and **~0 in the
`alpha` row** — `omega^2` cannot torque a rigid body, so a large value there
means the term is soaking up something else. The tool prints the implied
offset and it is written to `implied_tracking_offset`. A large `|delta|` means
the Pinpoint's tracking point is not at the robot's centre of rotation;
correcting the pod offsets on the robot should shrink it toward zero.

`--no-omega-sq` pins the term to zero. **The output shape does not change** —
`q` is still written, as zeros — so `a = B*u + A*v + q*omega^2 + c` evaluates
correctly either way and a consumer never branches. `has_omega_sq` records
which was done. The same holds for `--no-intercept` and `c`. Every optional
regressor also stays present in the per-response `coef` tables, pinned to zero.

### How many coefficients per output direction

Six are physical — 3 input (`fwd, strafe, turn`) and 3 velocity
(`v_x, v_y, omega`) — plus, by default, the `omega^2` term and a constant.

A drivetrain at rest with no power should not accelerate, so `c` ought to come
out near zero. Read it as a diagnostic: a large, statistically significant
constant means the fit is absorbing something systematic — a mis-set centre of
rotation, a persistent external force, damage biasing one direction. On the
2026-08-16 reference log it was `a_x = +66.7 cm/s²` (t = 3.8) and
`a_y = −54.0` (t = −4.9), consistent with that run's known problems.

`--no-intercept` drops it. Combined with `--no-omega-sq` that leaves exactly
the 6 physical coefficients. The output schema does not change either way —
`c` and `q` are simply written as zeros — so consuming code never branches.

On bad data the two disagree, and the disagreement lands almost entirely on
the velocity terms rather than the input gains:

| | `fwd`→`a_x` | `turn`→`alpha` | `v_x`→`a_x` | `omega`→`a_x` |
| --- | --- | --- | --- | --- |
| with intercept | −425.5 | −42.70 | −0.780 | −26.17 |
| without | −392.5 | −42.15 | −0.229 | −13.14 |

The input gains barely move; the drag terms shift 2–3x because the constant
was standing in for them. On clean data `c` should collapse toward zero and
the two fits should agree — which makes that agreement a good acceptance test
for a calibration run.

## Output schema

Identical content in both files. Julia reads the TOML with the stdlib and no
package dependency:

```julia
using TOML
mat(v) = reduce(vcat, permutedims.(Vector{Float64}.(v)))

fit = TOML.parsefile("drivetrain_fit.toml")
m   = fit["mecanum_basis"]
B   = mat(m["B"])                  # 3x3, columns [fwd strafe turn]
A   = mat(m["A"])                  # 3x3, columns [v_x v_y omega]
q   = Vector{Float64}(m["q"])      # 3, centripetal
c   = Vector{Float64}(m["c"])      # 3
a   = B*u + A*v + q*v[3]^2 + c     # v[3] is omega
```

This expression is correct whether or not `--no-omega-sq` / `--no-intercept`
were used; the disabled terms are simply zero.

Top-level keys:

| key | contents |
| --- | --- |
| `schema_version` | integer, currently `2` |
| `source` | absolute path, sha256, segment index, parse counters |
| `units` | `distance`, `angle`, `time`, and the two acceleration units |
| `convention` | frame definition, row/column orders, the mecanum mixing matrix |
| `mecanum_basis` | `B`, `A`, `q`, `c`, `B_columns`, `A_columns`, `rows`, `models`, `implied_tracking_offset` |
| `preprocessing` | every knob the fit was run with |
| `diagnostics` | row accounting, loop rate, input excitation, sensor-noise check |

`rows` is always `["a_x", "a_y", "alpha"]` and gives the row order of `B`,
`A`, `q`, and `c`. `B_columns` and `A_columns` give the column orders.

Per-response entries under `models` carry `coef`, `stderr_hac`, `stderr_ols`,
`tstat`, `vif`, `r2`, `r2_adj`, `cv_r2`, `rmse`, `resid_std`, `response_std`,
`block_resid_rms`, `max_vif`, and `n`.

Missing values are `null` in JSON and `nan` in TOML.

## What the code has to work around

### The Pinpoint's 1.5 kHz update rate

The Pinpoint recomputes velocity every ~0.667 ms by differencing encoder
counts over that single interval. Over so short a window the count changes by
only a few ticks, so tick quantisation dominates: the velocity quantum is one
tick times 1500 Hz. Measured on the reference log, the reported velocity sits
8–11% away from a smoothed estimate, and that residual is quantisation noise
rather than real motion.

Differencing that reported velocity to get acceleration multiplies the noise
by `1/dt ≈ 110`, which produces accelerations with an RMS of ~534 cm/s² and
peaks past 3600 cm/s² — physically impossible for this robot, and pure noise.

So the reported velocity is never differentiated. Position is the clean
quantity: its quantisation error is bounded by one tick and is not multiplied
by the update rate. Both velocity and acceleration come from a local weighted
polynomial fit to *position*, estimating derivatives over a ~0.22 s window
instead of 0.667 ms.

The logging loop runs at ~109 Hz with heavy jitter (5.8 ms to 125 ms), so a
stock Savitzky–Golay filter — which assumes uniform spacing — is not valid.
`local_poly_derivatives()` fits in real time coordinates, handling the jitter
exactly. Windows straddling a dropped-frame hole are discarded rather than
interpolated across.

### Rotation into the robot frame

The Pinpoint logs pose *and velocity* in the field frame. This was verified
rather than assumed: the reported velocity correlates 0.90/0.98 with the
field-frame derivative of the logged position, versus 0.19/0.42 if read as
body-frame.

Motor forces act along body axes, so the fit must happen in the body frame.
With `R(h)` the body-to-field rotation,

```
a_f = R(h)·dv_r/dt + Ṙ(h)·v_r = R(h)·[dv_r/dt + ω·J·v_r]
a_r = R(-h)·a_f    = dv_r/dt + ω·J·v_r,     J = [[0,-1],[1,0]]
```

The body-frame acceleration is therefore **not** just the derivative of the
body-frame velocity — the `ω × v` Coriolis term matters. This run reaches
ω ≈ 4.8 rad/s at ~100 cm/s, making that term worth several hundred cm/s²,
the same order as the accelerations being fitted.

The code avoids the algebra by differentiating field-frame position and
rotating once, which yields `a_r` with the Coriolis term already included.

### Band-limiting the inputs

Applying one linear filter to every term of `a = B·u + A·v + c` leaves the
coefficients unchanged; filtering only the response does not, and shrinks the
fitted gains toward zero. The motor powers are therefore pushed through the
same window that produced the response. On the reference log this took the
gains' drift across the window sweep from ~2x down to under 10%.

### Log quirks

Handled per the robot-side notes: mid-file header rows are split into separate
segments, a truncated final row is tolerated, frozen-pose logs from before the
`odo.update()` fix are rejected by signature, header-only files exit cleanly,
and exactly-duplicated consecutive pose rows are dropped.

## Why mecanum only

The four motor powers are **not** separately identifiable from a normal
driving run, so the tool does not offer that parameterisation at all.

On a reference log the singular values of `[FR FL BR BL]` were
`18.7, 14.7, 9.1, 0.30` — condition number 63, with the weakest direction
holding 0.68% of the energy. That direction is `(FR+FL-BR-BL)`, the mecanum
*null* direction, which commands the wheels to fight each other and produces
no chassis force. A driver using a standard mecanum mapping never excites it
(rms 0.006 versus 0.36 for `fwd`).

Fitting four wheel gains anyway gave max VIF ~900 and coefficients like
`+1109 ± 740` (t = 1.5), against `fwd = −425.5 ± 34.8` (t = −12.2) in the
mecanum basis. Both described the data equally well; only the mecanum one had
coefficients that meant anything individually.

A related guard: `--min-excitation` refuses to fit when the run never
commanded one of `fwd`, `strafe`, or `turn`. A pure spin log, for instance,
leaves the `fwd` and `strafe` gains completely unidentifiable, and fitting it
anyway produces coefficients in the hundreds of thousands with VIFs around
1e15 rather than an honest failure.

## Validating with `--self-test`

Simulates a robot that obeys a known `B`, `A`, `q` exactly, integrates it
finely, samples it at a jittery 150 Hz with realistic noise, and checks what
comes back. Typical result:

| block | worst error |
| --- | --- |
| input gains | ~11% |
| velocity gains | ~18% |
| `omega^2` | ~25% |

Everything is attenuated **low**, because smoothing removes signal from the
response faster than from the regressors. `omega^2` suffers extra: squaring a
smoothed `omega` drops the variance of the noise that was removed, which
biases that regressor and so shrinks its coefficient. Expect dominant gains
back within ~10%, `omega^2` within ~25%, and small cross-terms to be
unreliable. The tolerance checks that the pipeline is *sound*, not precise.

## Reading the fit quality

- **`cv_r2` is the number that matters.** It is cross-validated on *contiguous*
  blocks, not random folds. The samples are a smoothed time series, so random
  folds would put near-duplicate neighbours on both sides of the split and
  report a flattering score.
- **Standard errors are Newey–West (HAC).** Smoothing makes neighbouring
  residuals strongly correlated, so textbook OLS errors would be far too
  optimistic. `stderr_ols` is kept alongside for comparison.
- **`block_resid_rms`** breaks the residual down by fifth of the run. One hot
  block means a localised event — wheel slip, a bump — rather than a uniformly
  wrong model.
- **The window sweep** is printed on every run. A coefficient that drifts
  steadily with the window is being set by the filter rather than the data; a
  flat stretch is the trustworthy region.
