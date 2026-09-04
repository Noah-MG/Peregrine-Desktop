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
| `solver/cloud/` | running a solve on a rented GPU, when the card here is the limit |
| `docs/TABLE_FORMAT.md` | **the contract the robot firmware reads** |

## Requirements

- Python 3.12 (`py -3.12`), with `numpy` for the calibration fitters. The
  wizard itself is stdlib only.
- Julia 1.12 with `CUDA.jl` and `JSON3` — `julia --project=solver -e 'using
  Pkg; Pkg.instantiate()'`.
- An NVIDIA GPU for the solver. It falls back to CPU, much more slowly.
- Optional: an SSH key and a rented GPU box, if the grid you want is bigger
  than the card here. See `solver/cloud/README.md` — the 8 GB card is what
  forces tiling, and tiling is what makes a long lookahead expensive, so a
  bigger card buys accuracy as well as speed. You do not need the box to
  exist while you are planning: step 3 saves a plan as a **job**, and
  `peregrine_remote.py run <job> --host root@<ip>` solves it whenever the
  box is up.

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

Three things about the Bellman backup are worth knowing, because they are
where the accuracy actually comes from:

- **The command is continuous, not a lattice point.** The admissible set is
  the octahedron `|fwd| + |strafe| + |turn| ≤ 1`, and the optimum lies on its
  surface but not at a corner — the traction term `tanh(|u|/knee)/(|u|/knee)`
  depends on the Euclidean length of the command, which varies from 1.0 at a
  corner to 0.577 at a face centre, so a face point can deliver more useful
  force than the corner beside it. The lattice now only *seeds* a pattern
  search that then moves off-lattice, and each cell keeps its answer as a
  warm start for the next sweep. `control_scan` sets how much of the lattice
  each sweep looks at; `refine_rounds` sets how far the search then refines.
- **Each cell picks its own lookahead.** A single global `dt` is wrong almost
  everywhere: a step that does not leave its own cell teaches the backup
  nothing and converges to something too pessimistic, while one that spans
  several cells smears the interpolation. `cfl` says how many grid cells one
  backup should advance and the solver derives the seconds from that and from
  how fast the robot is going; `tau_levels` is how many horizons each cell
  brackets around it. Substep count follows the horizon so a longer step is
  never a less accurate one. Set `cfl: 0` to go back to a fixed `dt`.
- **Leaving the velocity box is not free.** A state outside the grid's
  velocity envelope is rejected, exactly like a state whose footprint is
  through a wall. It used to be clamped, which priced exceeding the envelope
  at zero — and since the fitted model's terminal speed is far above a typical
  `vmax`, the minimisation took that offer. Set `vmax` and `wmax` to what you
  actually want the robot to do, because the table will not exceed them.
- **The perimeter wall is an obstacle like any other.** It gets the robot's
  own footprint swept in and the clearance held off it, per heading, exactly
  as a scoring structure does. It did not always: the boundary used to
  constrain the tracking *point* alone, so a 36 cm chassis was free to park
  with 18 cm of itself outside the field — a looser rule than the one applied
  to an obstacle standing one centimetre inboard of that wall. `wall_clearance_cm`
  defaults to `clearance_cm`; set it separately if you are willing to run
  closer to the perimeter than to a structure.
- **You set the cell size, not the sample count.** `grid.resolution` is four
  numbers in four different units — `xy_cm`, `heading_deg`, `v_cm_s`,
  `w_rad_s` — and each is honoured exactly. The extent follows from them
  rather than the other way round, so changing the field or the speed
  envelope leaves the resolution alone and moves the counts. The velocity
  axes always come out odd, which puts **zero exactly on the grid**: every
  target is a state at rest, and on an even count rest falls between two
  samples and every value near it is an interpolation.
- **The table is narrower than the field, and says so.** Its x/y span is the
  field inset by the footprint and the wall clearance — the positions the
  robot can actually occupy — which on a 366 cm field with a 36 cm chassis is
  24% fewer cells for exactly the same set of legal states. `MANIFEST.JSON`
  reports the stored span under `grid.min`/`grid.max` and the field it was cut
  from under `grid.field_bounds`.
- **An unreachable cell now carries the way out of it.** `unreachable` is the
  right answer right up until the robot *is* there — shoved into an obstacle,
  or a footprint-width from a wall at a heading that does not fit. It then
  reads `unreachable` in every direction, has no gradient to descend, and
  stops. So after each target converges, a second short pass fills those cells
  with the time to reach the nearest state that *does* have a route, and the
  robot descends that until the table starts answering normally again.

  They are still unreachable, and must still lose every comparison against a
  real route — the escape time is only for a robot already stuck. It rides in
  the codes above the value range (`escape_base` and up, or a negative value
  for a float dtype), so **the tables are exactly the same size**, and firmware
  that has not been updated reads an escape cell as a finite time worse than
  every real route and behaves as it always did. §6.1 of
  `docs/TABLE_FORMAT.md` is the decode; `escape: false` turns the pass off.

  It only ever writes cells that were going to be `unreachable`, so it cannot
  make a table worse — the self-test holds every reachable cell bit-identical
  across it. It will not route through a wall either: a cell in free space may
  not escape *into* an obstacle, only a cell already inside one may move
  through obstacle space to leave it.

**A grid too big for the card is solved a tile at a time.** At full
resolution the value function runs to tens of gigabytes, well past any
consumer GPU, so the solver keeps it in a scratch file and gives the card one
window of it at a time: load a tile plus a halo, update the middle, write the
middle back, move on. Value iteration does not care what order cells are
updated in, only that they keep being updated, so this converges to the same
answer as the whole-grid solve — verified against it in the self-test, to
within the spread the solver already has between two runs of itself.

The cut is over `x` and `y` and can only be over `x` and `y`. Heading and
`ω` are too short to cut without the halo eating them, and `vx`/`vy` cannot
be cut at all: the successor's velocity is rotated back to the field frame
through the *new* heading, so a step that turns a quarter turn carries
`(vx, 0)` to `(0, vx)` and the reach spans the whole velocity plane.

A **round** is one pass over every tile: each is loaded, swept `tile_sweeps`
times against a frozen halo, and written back. So a round is `tile_sweeps`
sweeps of the whole grid, and `iterations` — which is a budget of sweeps —
buys `iterations / tile_sweeps` rounds, keeping the two drivers comparable on
both time and quality.

The tile boundaries **move between rounds**, which is why the tile count
changes: a shifted tiling starts partly off the edge and picks up an extra
partial row and column. That was for seams — a cell frozen on a boundary this
round is mid-tile the next — and it is load-bearing. With no halo and the
tiling pinned, mean value went 1.80 s to 14.33 s in the self-test; allowed to
shift, it came back to 1.79 s.

`plan` also estimates the wall clock. It is a ceiling — the full iteration
budget, which `tolerance` usually cuts short — and it comes from two rates
measured on one machine: about 23M cell-updates/s on an 8 GB card, and about
500 MB/s of scratch. After one run, put the real numbers in `cell_rate` and
`disk_rate` and every later estimate sharpens.

A round costs its compute **plus** its I/O, not the greater of the two. The
driver loads a tile, sweeps it, stores it, and moves on — strictly in
sequence, with no prefetch and no second staging buffer — so nothing hides
behind anything else. (Writes are the exception: they land in the page cache
and the kernel flushes them behind us, so they are partly hidden in practice.)
That makes the halo a real, visible cost at every size, which is why `plan`
reports the two halves separately.

The number to watch in `plan` is **loaded per updated** — how many cells come
off disk for each one the sweep improves. It is set by how far one backup can
reach compared with how big a tile the card can hold, so the lever on it is
`tau_max`, the longest lookahead the step-length ladder may try.

`plan` no longer leaves that to you. It costs out the whole ladder on a
single shared reach scan and recommends one, with the accuracy it costs
against the 0.5 s reference the cost curve was measured at:

```
  step    reach   loaded/updated    sweeps      i/o   per round   value cost
   0.50      87 cm       29.2x      3m 47s   9m 22s     13m 09s  +   0%
   0.40      72 cm       11.4x      3m 47s   3m 50s      7m 38s  +   1%
   0.30      57 cm        6.0x      3m 47s   2m 10s      5m 58s  +   3%
   0.20      40 cm        3.7x      3m 47s   1m 27s      5m 15s  +   6% <--
   0.10      21 cm        2.2x      3m 47s   1m 00s      4m 48s  +  21%
```

The rule behind the arrow is one objective: lowest `round time × (1 + value
cost)`. Both are multiplicative penalties on the same run — one lengthens it,
the other makes every answer in it worse by a measured percentage — so the
product has a real minimum, and it sits at the knee rather than at either end.
In core there is no halo at all, so the answer is simply the reference and no
scan is run.

When the grid will not hold whole, `plan` also reports **the nearest
resolution that would** — the same cell sizes, uniformly coarsened, with the
factor named. Tiling is the difference between minutes and days, so that is
usually the trade worth making before any of the others, and the resolution
suggestion is bounded by VRAM as well as by card space so that taking it can
never move a run from in-core to tiled.

**`tau_max` defaults to `auto` and is derived on every plan and every solve**,
from the same code path so the two cannot disagree. In core it resolves to the
full 0.5 s, because there is no halo for a shorter step to pay for; tiled it
resolves to the knee for that particular grid. It is deliberately not a
remembered setting: a number carried in a config cannot track the resolution,
the field, the model or the machine, and a stale short step is invisible —
0.12 s left over from a tiled experiment costs about 17% on mean value on
every run afterwards, including ones with nothing to gain from it. Pin it to a
number if you want to, and `plan` will tell you what the pin is costing.

**A settled plan can be saved and rented for later.** Planning takes an
evening of re-planning at different resolutions; a rented box bills until it
is destroyed, so it should exist for the solve and not a minute more. Step 3
therefore offers to freeze a plan into a **job** — a directory under
`<workspace>/jobs/` holding the config and frozen copies of the three inputs
— and nothing needs to be rented at that point. When the box is up:

```bash
py -3.12 solver/cloud/peregrine_remote.py provision root@<ip>
py -3.12 solver/cloud/peregrine_remote.py run <job> --host root@<ip>
```

The inputs are copies rather than references, so editing a target afterwards
cannot quietly change what the job solves. `solver/cloud/README.md` §8.

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

## Checking the solver

```bash
julia --project=solver solver/solve.jl --self-test
```

Seven checks, and the two that matter most exist to keep the solver honest
about its own optimisations:

- On a **double integrator**, where the minimum time from rest at distance `d`
  is exactly `2·√(d/a)`, does the solver recover it? The solver knows nothing
  about that formula, so agreeing with it exercises the control set, the
  integration, the interpolation, the seeding and the iteration at once. The
  remaining error is reported split into a **fixed part** (the terminal
  approach, where the robot is slow and the interpolation is most diffusive)
  and a **per-second part** (the cruise) — a flat percentage cannot tell those
  two apart, and only the second would signal a real regression.
- Is the **fast control search ever worse** than scanning the whole lattice?
  Measured: worse on 0 of 606,368 cells, and 36% tighter on average.
- Does **widening the step-length ladder ever raise a value**? It must not —
  minimising over more horizons is only sound if the horizons are *added* to
  the existing one rather than substituted for it. Measured: 0 of 617,400
  cells raised.

The self-test also prints the minimum-time excess split into a **fixed** part
(the terminal approach, where the robot is slow and the interpolation is most
diffusive) and a **per-second** part (the cruise). Those have different causes
and only the second would signal a regression in the backup, so a single
percentage tolerance cannot police both.

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
