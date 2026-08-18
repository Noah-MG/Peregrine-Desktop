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

That is the intended way in. It walks through all four steps and remembers
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
| `calibration/fit_drivetrain.py` | fits `a = B·u + A·v + q·ω² + c` from a log |
| `solver/solve.jl` | Julia/CUDA minimum-time value-table solver |
| `docs/TABLE_FORMAT.md` | **the contract the robot firmware reads** |

## Requirements

- Python 3.12 (`py -3.12`), with `numpy` for the calibration fitters. The
  wizard itself is stdlib only.
- Julia 1.12 with `CUDA.jl` and `JSON3` — `julia --project=solver -e 'using
  Pkg; Pkg.instantiate()'`.
- An NVIDIA GPU for the solver. It falls back to CPU, much more slowly.

## The four steps

**1. Calibration.** Finds `calibration_log_*.csv` on any mounted card, imports
one, and fits the drivetrain model. See
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
