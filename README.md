# Peregrine-Desktop

The set of programs required for the off-robot side of Peregrine.

Calibration logs come off the robot's SD card; value tables go back onto it.

```
calibration CSV -> drivetrain regression -> field + targets
                -> GPU value-table solve -> SD card
```

## Quick start

Double-click **`peregrine.bat`**, or from a terminal:

```bash
peregrine.bat
```

That is the intended way in. It walks through every step and remembers
your workspace. It works from any directory, finds a suitable Python itself,
and when double-clicked keeps the window open at the end so you can read the
output.

To call it as just `peregrine` from anywhere, add this folder to your PATH
once — run this from the repository root:

```bash
setx PATH "%PATH%;%CD%"
```

The wizard is a plain Python script underneath, so this is equivalent:

```bash
py -3.12 wizard/peregrine.py
```

Everything the wizard drives can also be run directly — see *Layout*.

## Layout

| path | what |
| --- | --- |
| `wizard/peregrine.py` | terminal wizard — the front door |
| `wizard/sdcard.py` | drive detection and the **only** destructive code |
| `wizard/verify_tables.py` | checks a card image against the format spec |
| `calibration/find_pod_offsets.py` | optional, **do first**: pod offsets from a rotation run |
| `calibration/fit_drivetrain.py` | fits the drivetrain model from a driving log |
| `calibration/diagnose_fit.py` | optional: plots the fit and recommends a model form |
| `solver/solve.jl` | Julia/CUDA minimum-time value-table solver |
| `docs/TABLE_FORMAT.md` | **the contract the robot firmware reads** |

## Requirements

- Python 3.12 (`py -3.12`), with `numpy` for the calibration fitters. The
  wizard itself is stdlib only.
- Julia 1.12 with `CUDA.jl` and `JSON3` — `julia --project=solver -e 'using
  Pkg; Pkg.instantiate()'`.
- An NVIDIA GPU for the solver. It falls back to CPU, much more slowly.

## The steps

**0. Pod offsets (optional, but do it first).** From a rotate-in-place log.
If the offsets are right, spinning leaves the reported position where it is;
if they are wrong it sweeps a circle whose radius is the error. Everything is
in **cm**, and it prints the exact `setOffsets()` line to put on the robot.

This matters more than it sounds: a wrong offset makes the robot report a
phantom sideways velocity whenever it turns, and the drivetrain fit cannot
tell that apart from real dynamics. It absorbs it silently and gives a model
that is wrong wherever the robot rotates. Measured on a real run, an 11 cm
offset moved `corr(omega, v_y)` from **−0.60 to +0.02** once corrected.

Two things it handles that the obvious implementation gets wrong. A robot
spinning in place still creeps across the floor, so the fit carries a drift
term — without one, that creep is charged to the offset instead (on a real
12.8 s spin it doubled the answer and quadrupled the residual). And a
**centred** tracking point produces no rotation-induced velocity at all, which
looks identical to "this is not a spin"; the run is therefore judged from how
far the robot turned and what was commanded, never from how well the velocity
fits.

It reports "centred, nothing to change" below 0.5 cm. That threshold is about
consequence, not statistics: at this precision a 2 mm offset is
*statistically* real while being far too small to matter.

**1. Calibration.** Finds `calibration_log_*.csv` on any mounted card, imports
one, and fits

```
a = B·u + A·v + q·ω² + S·csign(v) + D·(|v|·v) + c
```

with Coulomb friction `S` and quadratic drag `D` **on** by default and `ω²`
**off**, following the 2026-08-19 analysis of a real run: `ω²` scored worse
than a plain linear model on held-out prediction, while `S` and `D` roughly
doubled it. `csign` is a *smoothed* sign, `clamp(v/ε, −1, 1)`, because the
solver integrates this model and a hard `sign()` chatters at zero.

Read `A`, `S` and `D` as a group, never individually — linear `v`, `csign(v)`
and `|v|·v` are collinear over any finite speed range, so how the fit splits
them between the three is arbitrary even when the model as a whole is right.

Option `d` in the wizard is an optional diagnostic report — nothing downstream
needs it, but it answers "is this the right model shape?" with plots and a
comparison of candidate forms. See *Checking the model form*, and
[calibration/README.md](calibration/README.md) — in particular why the fit
uses `{fwd, strafe, turn}` rather than the four wheel powers, and how the
Pinpoint's 1.5 kHz velocity quantisation is handled.

**2. Field and targets.** Two JSON files, with commented examples in
`solver/examples/`. Obstacles are the **real** physical obstacles,
un-inflated; the robot's own footprint goes under `robot` and the solver
swells them itself, once per heading bin. The solve also asks for a
**clearance** (default 5 cm), a plain safety gap held around every obstacle
before the footprint is swept in.

It has to be per heading, and that is worth knowing: the configuration-space
obstacle of a polygonal robot is **not** a polyhedron in `(x, y, h)`. Its
faces satisfy

```
n_x·x + n_y·y + A·cos h + B·sin h = c
```

which is planar only when `A = B = 0` — that is, only for a point robot. For
a real chassis the faces are curved, and the slice even changes vertex count
as it turns. What *is* exact is that every fixed-heading slice is a polygon,
so the solver rasterises one slice per heading bin and unions across each
bin so nothing slips through in between.

**3. Solve.** For each target state, a minimum-time value table
`V(s) = time to reach the target from s` over a 6D grid of
`(x, y, h, vx, vy, ω)`. Run `plan` first — the wizard does — to see table
size and VRAM fit before committing, since these tables get very large very
quickly.

**4. Write the SD card.** Wipes the card and writes the tables, plus
`MANIFEST.JSON` describing them and `MODEL.JSON` carrying the drivetrain
model itself — the online optimizer needs both. Nothing else goes on the
card.

## Checking the model form

```bash
py -3.12 calibration/diagnose_fit.py <log.csv> --fit <drivetrain_fit.toml> --open
```

Writes a self-contained HTML report. Pass `--fit` and it mirrors that fit's
preprocessing exactly, so the report describes the fit you actually have
rather than a differently-preprocessed one.

The part that answers "what regression should I be using" is the model-family
comparison: eight candidate forms refitted on the same data, then
**integrated forward** and scored on how well they predict the change in
velocity over half a second, on the half of the run they were not fitted to.

Scoring by rollout rather than by acceleration R² is deliberate, and it
changed conclusions on real data. The response the fit regresses on is a
twice-smoothed derivative, so acceleration R² can damn a model that predicts
motion perfectly well — on a real run the model reached 0.97/0.79/0.60 at
predicting motion while its `alpha` R² was 0.26 with negative CV. A null
"velocity does not change" baseline is reported alongside so the numbers mean
something.

`--self-test` checks that this works, by building two synthetic robots with a
known true model and confirming the comparison recovers it in both cases.

A caution the report repeats: the residual-vs-regressor plots are more
sensitive than the rollout comparison, and estimating acceleration by
smoothing leaves a signal-dependent bias behind that looks like curvature. The
report only recommends adding a term when the rollout score agrees.

## Safety

All destructive code is confined to `wizard/sdcard.py`, deliberately kept
short enough to read in one sitting. A wipe requires **five independent
checks to pass**:

1. Not on a disk Windows marks as system or boot.
2. Not a letter that hosts Windows, your profile, the workspace, or this repo.
3. Genuinely external — removable, or on a USB/SD/MMC bus.
4. No system directories in the root.
5. The drive letter typed back by hand.

It deletes files. It never formats, never touches partition tables, and never
runs on a path that is not the root of a verified removable volume. To see
what it would allow, without changing anything:

```bash
py -3.12 wizard/sdcard.py
```

## Verifying a card

```bash
py -3.12 wizard/verify_tables.py G:\
```

This re-implements the lookup **from `docs/TABLE_FORMAT.md`**, not from the
solver's internals, so it tests the contract the firmware depends on rather
than agreeing with whatever the solver happened to write. It checks chunk
counts, sizes, SHA-256, FAT32 file limits, that `V(target) ≈ 0`, and that
time-to-go grows with distance. The wizard runs it automatically after
solving and again after writing.
