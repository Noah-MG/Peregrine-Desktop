# Peregrine value-table SD card format

The desktop solver writes this; the online optimizer on the robot reads it.

The card holds one **value table** per target. A value table answers a single
question for every state the robot could be in:

> from this state, what is the shortest possible time to reach the target?

The robot reads a few of those numbers near its current state each loop and
steers toward the smallest. All the searching happened offline.

Read this document in order. Every symbol is defined before it is used.

---

## 1. Files on the card

```
/MANIFEST.JSON            describes everything below
/MODEL.JSON               the drivetrain model, see section 8
/TABLES/T00C0000.BIN      target 00, chunk 0000
/TABLES/T00C0001.BIN      target 00, chunk 0001
/TABLES/T01C0000.BIN      target 01, chunk 0000
...
```

Chunk filenames are `T` + two-digit target + `C` + four-digit chunk + `.BIN`,
which is a strict 8.3 name. On FAT32 a long filename consumes several
directory entries; an 8.3 name consumes one.

That naming caps a target at 10000 chunks and the card at 100 targets. At the
default `chunk_elements` of 2^23 those are not close: a full-resolution grid
of `[161, 161, 64, 21, 21, 21]` is 15.4 billion cells, which is 1832 chunks
and 28.6 GB per target at `u16`. What a card that size *does* run into is
sheer file count -- a few thousand entries in `/TABLES` -- so it is worth
formatting exFAT rather than FAT32 when the tables get past a few gigabytes.

Nothing else belongs on the card.

---

## 2. The manifest, which defines every symbol

```json
{
  "schema_version": 1,
  "generated_utc": "2026-08-16T21:40:00Z",
  "generator": "peregrine-desktop",
  "regression_sha256": "...",
  "field_sha256": "...",
  "model_file": "MODEL.JSON",

  "grid": {
    "axes":  ["x", "y", "h", "vx", "vy", "w"],
    "n":     [46, 46, 24, 13, 13, 13],
    "min":   [23.0, 23.0, -3.14159265, -170.0, -170.0, -8.0],
    "max":   [343.0, 343.0, 3.14159265, 170.0, 170.0, 8.0],
    "wrap":  [false, false, true, false, false, false],
    "units": ["cm", "cm", "rad", "cm/s", "cm/s", "rad/s"],
    "frame": "field",
    "field_bounds": [0.0, 0.0, 366.0, 366.0],
    "total_cells": 111572448,
    "index_formula": "((((ix*Ny+iy)*Nh+ih)*Nvx+ivx)*Nvy+ivy)*Nw+iw"
  },

  "encoding": {
    "dtype": "u16",
    "elem_bytes": 2,
    "scale": 0.001,
    "unit": "seconds",
    "unreachable": 65535,
    "escape_base": 64512,
    "escape_codes": 1023,
    "escape_scale": 5.7448e-05,
    "escape_unit": "seconds",
    "byte_order": "little",
    "order": "row_major_c",
    "chunk_elements": 8388608,
    "chunk_shift": 23
  },

  "targets": [
    {
      "index": 0,
      "name": "score_left",
      "state": [60.0, 300.0, 0.7854, 0.0, 0.0, 0.0],
      "file_pattern": "TABLES/T00C%04d.BIN",
      "n_chunks": 14,
      "bytes": 223144896,
      "sha256": "...",
      "reached_frac": 0.88,
      "escape_frac": 0.09,
      "escape_of_unreached_frac": 0.75
    }
  ]
}
```

Every name used in the rest of this document comes from that file:

| symbol | manifest field | meaning |
| --- | --- | --- |
| `n[k]` | `grid.n` | number of samples on axis `k` |
| `min[k]`, `max[k]` | `grid.min`, `grid.max` | span of axis `k` |
| `wrap[k]` | `grid.wrap` | true if axis `k` is periodic |
| `step[k]` | *derived*, see §3 | spacing between samples on axis `k` |
| `Nx … Nw` | `grid.n[0] … n[5]` | shorthand for the six axis lengths |
| `ix … iw` | *computed*, see §3 | the six per-axis indices |
| `idx` | *computed*, see §4 | flat index into the table |
| `elem_bytes` | `encoding.elem_bytes` | bytes per stored value |
| `dtype` | `encoding.dtype` | how to decode those bytes |
| `scale` | `encoding.scale` | seconds per raw unit, integer types only |
| `unreachable` | `encoding.unreachable` | the "no route" sentinel |
| `escape_base` | `encoding.escape_base` | first raw code of the escape band (§6.1) |
| `escape_codes` | `encoding.escape_codes` | how many codes the band has; 0 = no band |
| `escape_scale` | `encoding.escape_scale` | escape seconds per code squared |
| `chunk_elements` | `encoding.chunk_elements` | values per chunk file, a power of two |
| `chunk_shift` | `encoding.chunk_shift` | `log2(chunk_elements)` |

Read these from the file rather than hard-coding them. The grid is expected to
change as the resolution gets tuned; that should not require a firmware change.

### The table is narrower than the field

`min[0..1]` and `max[0..1]` are **not** the field boundary, and in the example
above they are 23 cm inside it on every side. They span the positions the
robot can legally *occupy*: the field, inset by the robot's own footprint and
by `solver.wall_clearance_cm`. A state outside that span is one where part of
the chassis is through the perimeter wall, which is not a place the robot can
be, so no value is stored for it.

`grid.field_bounds` carries the field the inset was taken from, as
`[x_min, y_min, x_max, y_max]`. It is informational — every lookup uses
`min`/`max` — and it is there so a table can be matched back to the field it
was cut from without re-deriving the inset.

Nothing about the lookup changes. §3 clamps `x` and `y` into `min..max` as it
always has, and for a position between the table edge and the wall that clamp
now lands on the nearest state the robot could actually hold, which is the
right answer to give. The inset is heading-dependent — a rectangular chassis
needs more room across its diagonal — so the span is the union over headings
and the headings that need more are marked unreachable cell by cell, exactly
as an obstacle is.

Three fields are informational rather than needed for lookup:
`sha256` covers the concatenated logical table in index order, so it does not
change if the chunking does; `reached_frac` is the fraction of cells that got
a real answer, and a low value warns that much of the state space could not
reach that target; `escape_frac` is the fraction carrying an escape (§6.1) and
`escape_of_unreached_frac` the share of the cells *without* a route that got
one, which is the number that says whether the escape pass did its job —
`escape_frac` on its own falls simply because a table got better;
`regression_sha256` and `field_sha256` identify which drivetrain fit and field
description produced the tables.

Everything under `solver` is informational too, and the robot reads none of
it. `solver.driver` says whether the table came from a whole-grid solve or a
tiled out-of-core one, and `solver.tiling` records how it was cut if so. The
two converge to the same fixed point, so a tiled table is not a different
kind of table -- the field is there because "which driver produced this" is
the first thing worth being able to answer without guessing when a table does
look wrong.

---

## 3. Step 1 — from a robot state to six indices

A state is six numbers, always in this order:

| `k` | axis | meaning | unit |
| ---: | --- | --- | --- |
| 0 | `x` | position across the field | cm |
| 1 | `y` | position along the field | cm |
| 2 | `h` | heading | rad |
| 3 | `vx` | velocity, x component | cm/s |
| 4 | `vy` | velocity, y component | cm/s |
| 5 | `w` | angular velocity | rad/s |

All six are **field frame**, velocities included, so they can be fed straight
from odometry with no rotation. (`w` is the same in either frame.)

### Sample spacing

Two cases, distinguished by `wrap[k]`:

```
wrap[k] == false:   step[k] = (max[k] - min[k]) / (n[k] - 1)
wrap[k] == true:    step[k] =  2*pi / n[k]
```

The `n[k] - 1` is not a typo, and neither is its absence in the second case.
On a normal axis both endpoints are stored samples, so `n[k]` samples leave
`n[k] - 1` gaps between them. On the heading axis the two ends are the *same
state* — `-pi` and `+pi` are the same direction — so storing both would be a
duplicate. Heading stores `n[2]` distinct samples spanning the full circle,
which leaves `n[2]` gaps, not `n[2] - 1`.

### Index of a value

```
c = (value - min[k]) / step[k]

wrap[k] == false:   i = clamp(round(c), 0, n[k] - 1)
wrap[k] == true:    i = mod(round(c), n[k])
```

Clamping is correct behaviour on the position axes, not a fallback. `min` and
`max` bound the positions the robot can legally occupy (see §2), so a query
beyond them is a state with part of the chassis through a wall; the nearest
edge cell is the nearest state it could actually be in, and that is the best
answer available.

The velocity axes are **not** the same case. Clamping there would price
exceeding the envelope at zero, and the drivetrain really can exceed it — see
the note in §9. Treat a velocity query outside `min`/`max` as "no value
available" rather than clamping it.

`mod` on the heading axis is what makes index `n[2]` fold back to `0`. Use it
for neighbours too: index `n[2] - 1` and index `0` are adjacent, not opposite
ends.

---

## 4. Step 2 — from six indices to one flat index

```
idx = ((((ix * Ny + iy) * Nh + ih) * Nvx + ivx) * Nvy + ivy) * Nw + iw
```

This is a mixed-radix number: `iw` is the ones digit in base `Nw`, `ivy` the
next digit up, and so on, with the nesting evaluated by Horner's method. The
same expression appears in the manifest as `index_formula`, so the file and
the firmware can be checked against each other.

Two consequences worth knowing:

- `idx` runs from `0` to `total_cells - 1` with no gaps, so the table is one
  dense array.
- Incrementing `iw` by one moves one element along in the file. The last axes
  are the fastest-varying, so a small neighbourhood of states lands in a short
  contiguous stretch rather than scattered across the whole table.

---

## 5. Step 3 — from a flat index to a byte in a file

A single table can exceed the 4 GiB that FAT32 allows in one file — and FAT32
is the only filesystem the Control Hub accepts — so the array is cut into
equal **chunks** of `chunk_elements` values each, one file per chunk.

```
chunk         = idx >> chunk_shift
elem_in_chunk = idx &  (chunk_elements - 1)
byte_offset   = elem_in_chunk * elem_bytes
```

Those are exactly `idx / chunk_elements` and `idx % chunk_elements`. They can
be written as a shift and a mask because `chunk_elements` is always a power of
two, with `chunk_elements == 1 << chunk_shift`. For non-negative integers the
two forms are identical, so this is a free speed-up rather than an
approximation.

The file is `file_pattern` with `chunk` substituted, e.g. chunk 3 of target 0
is `TABLES/T00C0003.BIN`. Every chunk is completely full except the last one
of each target.

---

## 6. Step 4 — from bytes to seconds

Read `elem_bytes` bytes at `byte_offset`, little-endian, and decode according
to `dtype`:

| `dtype` | `elem_bytes` | raw type | seconds |
| --- | ---: | --- | --- |
| `u8` | 1 | unsigned byte | `raw * scale` |
| `u16` | 2 | unsigned 16-bit | `raw * scale` |
| `f16` | 2 | IEEE half | `raw` |
| `f32` | 4 | IEEE single | `raw` |

Before scaling, check for the sentinel:

```
integer dtypes:   raw >= escape_base   ->  +infinity      (see 6.1)
                  raw == unreachable   ->  +infinity
float dtypes:     isnan(raw)           ->  +infinity
                  raw < 0              ->  +infinity      (see 6.1)
```

Integers compare exactly, so `==` is safe there. Floats need `isnan`, because
`NaN == NaN` is false by definition and an equality test would never fire.

When `escape_codes` is 0 the band is absent and the two `escape` lines can be
dropped; a card solved with `escape` turned off, or written before 2026-09-04,
has no band and decodes exactly as it used to.

Unreachable must be treated as `+infinity`, not as a large finite number, so
it loses every comparison against a real route. It covers three situations —
the footprint overlapping an obstacle or the perimeter wall at that heading,
a state outside the solved envelope, or no route found within the solved
horizon — which are deliberately not distinguished, since all three mean the
same thing to the robot.

### 6.1 The escape band — unreachable, with a way out

An unreachable cell says "don't go there", which is the right answer right up
until the robot **is** there. Shoved into an obstacle by a collision, or a
footprint-width from a wall at a heading that does not fit, it reads
`unreachable` in every direction and has no gradient to descend. It stops.

So an unreachable cell can now also carry the **time to get out**: the
shortest time from it to a state that does have a real route. Decode:

```
integer dtypes:   raw >= escape_base && raw != unreachable
                      ->  escape_seconds = escape_scale * (raw - escape_base)^2

float dtypes:     raw < 0 && !isnan(raw)
                      ->  escape_seconds = -raw
```

`raw == unreachable` (and `NaN`) still means no route **and** no way out.

**These cells are still unreachable.** `escape_seconds` is not a time-to-go
and must never be compared against one, or fed to anything choosing where to
drive. It is only for a robot that has already found itself with no reachable
state anywhere nearby: descend `escape_seconds` until a cell decodes to a
finite time, then use the table normally. Two separate reads, two separate
uses — which is why §6 above returns `+infinity` for these and this section is
a second lookup rather than a different answer from the first.

**Why the band is square-law.** It has to span zero to `solver.value_cap_s`
— 60 s by default — in 1023 codes, and splitting that evenly would be 59 ms a
code. The gradient the robot descends is the difference between neighbouring
cells inside an obstacle, which at full resolution is tens of milliseconds, so
a linear band would flatten it into a plateau with nothing to follow. Squaring
puts the resolution where the values are: about 8 ms a code at a quarter-second
escape, 15 ms at one second, coarsening to 120 ms at the far end where the
number only has to mean "a long way". On the robot it is one multiply.

**Why a band at the top rather than a flag bit.** A flag bit would halve the
range, stopping real routes at 32.767 s — well inside the 60 s `value_cap`, so
they would start clipping. Codes above `value_cap / scale` are unreachable by
construction and cost nothing to take.

**Reading a new card with old firmware is safe.** Skip 6.1 entirely and an
escape cell decodes as a finite 64.5–65.5 s. `value_cap` guarantees every real
route is under 60 s, so it loses every comparison anyway and the robot behaves
exactly as it did before — it just doesn't get the recovery. The band is
additive; it is not a flag day.

The wall case is worth calling out because it is new as of 2026-08-31 and it
is common: cells within roughly a footprint of the table edge are unreachable
at the headings whose chassis would not fit there, and reachable at the
headings whose would. A robot that finds `unreachable` while hugging a wall
is being told to turn, not that the table is broken — and §6.1 now tells it
which way, since those cells carry an escape.

The default `u16` with `scale = 0.001` is plain milliseconds: exact to 1 ms up
to 64.511 s — the escape band takes the codes above that — in half the space
of `f32`. `f16` is the same size but carries
only about three significant digits, so prefer `u16` unless you specifically
want floats. `u8` with `scale = 0.025` gives 25 ms steps up to 6.35 s in one
byte, which is worth it when the horizon is short and the grid is large.

---

## 7. Worked example

Using the manifest shown in §2, and the state

```
x = 100 cm, y = 200 cm, h = 1.0 rad, vx = 50 cm/s, vy = -30 cm/s, w = 2.5 rad/s
```

**Steps** (§3), with `n = [46, 46, 24, 13, 13, 13]`. Note that `step[0]` comes
from the *table* span, 23..343, not from the 366 cm field:

```
step[0] = (343 - 23) / (46 - 1)  = 7.111111    step[3] = 340 / 12 = 28.333333
step[1] = (343 - 23) / (46 - 1)  = 7.111111    step[4] = 340 / 12 = 28.333333
step[2] = 2*pi / 24              = 0.261799    step[5] =  16 / 12 =  1.333333
```

**Indices** (§3):

```
ix  = round((100 - 23)     / 7.111111)  = round(10.8281) = 11
iy  = round((200 - 23)     / 7.111111)  = round(24.8906) = 25
ih  = round((1.0 - -pi)    / 0.261799)  = round(15.8197) = 16   (mod 24)
ivx = round((50 - -170)    / 28.333333) = round( 7.7647) =  8
ivy = round((-30 - -170)   / 28.333333) = round( 4.9412) =  5
iw  = round((2.5 - -8)     / 1.333333)  = round( 7.8750) =  8
```

**Flat index** (§4):

Evaluated from the inside out, one axis per line:

```
     11 * 46 + 25  =        531      folded in iy
    531 * 24 + 16  =      12760      folded in ih
  12760 * 13 +  8  =     165888      folded in ivx
 165888 * 13 +  5  =    2156549      folded in ivy
2156549 * 13 +  8  =   28035145      folded in iw   -> idx
```

**File and offset** (§5), with `chunk_shift = 23`:

```
chunk         = 28035145 >> 23      = 3          -> TABLES/T00C0003.BIN
elem_in_chunk = 28035145 & 8388607  = 2869321
byte_offset   = 2869321 * 2         = 5738642
```

**Decode** (§6): read 2 bytes little-endian at 5738642.

```
raw <  64512   ->  reachable, raw * 0.001 seconds
raw >= 64512   ->  unreachable; and if raw != 65535 it also carries
                   5.7448e-05 * (raw - 64512)^2 seconds to get out (§6.1)
```

So `12345` is 12.345 s to go; `64612` is unreachable with a 0.57 s escape;
`65535` is unreachable with no way out.

---

## 8. The drivetrain model

The tables say how long a route takes. They do not say how the robot moves,
and the online optimizer needs both, so `/MODEL.JSON` carries the fitted
drivetrain alongside them.

This section is the whole thing: what the file contains, the order to
evaluate it in, and the three places it is easy to get wrong.

### 8.1 What it computes

Given a **command** and a **state**, it returns the **acceleration** the robot
will produce.

| symbol | size | is | units |
| --- | ---: | --- | --- |
| `u_raw` | 3 | commanded `[fwd, strafe, turn]`, each in −1…1 | — |
| `s` | 6 | state `[x, y, h, vx, vy, w]` | cm, rad, cm/s, rad/s |
| `a` | 3 | resulting `[a_x, a_y, alpha]` | cm/s², rad/s² |

Two conventions that are not negotiable, both covered again in §8.5:

- **`vx`, `vy` in `s` must be BODY frame here.** The value tables index
  field-frame velocity. Rotate by `−h` before evaluating this model.
- **`a` is the *proper* body-frame acceleration.** Integrating it to get a
  velocity needs the Coriolis term removed — see §8.4.

### 8.2 The equation

```
u = u_raw * traction_gain(u_raw)                          (1) wheel slip

a = A_u   * u
  + A_uu  * (u .* u)
  + A_s   * s
  + A_ss  * (s .* s)
  + A_sgn * csign(s)
  + A_absv* (abs(s) .* s)
  + k                                                     (2) acceleration
```

`.*` is elementwise multiplication. Every matrix comes straight out of
`MODEL.JSON`:

| field | shape | multiplies |
| --- | --- | --- |
| `A_u` | 3×3 | `u` |
| `A_uu` | 3×3 | `u .* u` |
| `A_s` | 3×6 | `s` |
| `A_ss` | 3×6 | `s .* s` |
| `A_sgn` | 3×6 | `csign(s)` |
| `A_absv` | 3×6 | `abs(s) .* s` |
| `k` | 3 | — |

Read the shapes from the declared `state`, `control` and `output` vectors
rather than hard-coding 3 and 6.

Several blocks are always or usually zero:

- The `x, y, h` columns of every state block are **always** zero. The dynamics
  have no position dependence, and the command reaches position only through
  velocity: `u` enters as an acceleration and nothing else. Both the writer
  and `verify_tables.py` refuse a file that breaks this.
- `A_uu` is **always** zero. Control-squared terms are not part of this model
  and cannot be fitted — a `u²` column is even in `u`, so it would claim the
  same force for full forward and full reverse, and the command curvature that
  is really there is already carried by the traction knee (§8.3).
- `A_ss` is zero whenever the `omega²` term is off, which is the normal case.

**`nonzero_blocks` lists the ones actually carrying anything** — skip the rest
if you want the speed. They are all emitted regardless so the equation never
changes shape.

### 8.3 The two helper functions

**Traction gain** — wheel slip. Past a certain demand the tyres stop
delivering and extra command buys no extra force.

```
knee = MODEL.JSON -> control.saturation.knee

traction_gain(u_raw):
    m = sqrt(fwd² + strafe² + turn²)         # one shared traction budget
    r = m / knee
    if r < 1e-6:                   return 1  # avoid 0/0
    return tanh(r) / r
```

`knee` is **required and always positive**. There is no "no saturation" case
to branch on: the gains in `A_u` were fitted against the saturated command, so
a file without a knee would describe dynamics nothing evaluates. A card whose
knee is missing, null or ≤ 0 is rejected by `verify_tables.py` — treat it as a
bad card rather than defaulting the gain to 1.

It is 1 for small demand, falls off smoothly past the knee, and never changes
the *direction* of the command — only its magnitude. Because it is derived
from the total demand and scales all three components, a robot that is sliding
loses translation and rotation authority together.

**`csign`** — Coulomb friction, smoothed.

```
eps = MODEL.JSON -> csign.coulomb_eps        # 6 entries, matching s

csign(s)_i = clamp(s_i / eps_i, -1, 1)
```

This is **not** `sign()`. A true sign flips discontinuously at zero and makes
any integrator chatter there. The desktop fit uses this exact smoothed form,
so departing from it means the robot no longer matches the tables it was
given. The `x, y, h` entries of `eps` are 0 and unused; their coefficients are
zero anyway.

### 8.4 Evaluation order

```
 1. read u_raw = [fwd, strafe, turn]        the command you are about to send
 2. read the robot state, field frame
 3. rotate velocity into the BODY frame:
        vbx = +cos(h)*vx_field + sin(h)*vy_field
        vby = -sin(h)*vx_field + cos(h)*vy_field
        s   = [x, y, h, vbx, vby, w]
 4. u = u_raw * traction_gain(u_raw)                       §8.3
 5. a = A_u*u + A_uu*(u.*u) + A_s*s + A_ss*(s.*s)
        + A_sgn*csign(s) + A_absv*(abs(s).*s) + k          §8.2
 6. a is now [a_x, a_y, alpha], body frame, PROPER acceleration
```

To step a simulation forward from `a`, remove the Coriolis term:

```
    dvbx/dt = a_x   + w * vby
    dvby/dt = a_y   - w * vbx
    dw/dt   = alpha
    dh/dt   = w
    d(x,y)/dt = R(h) * [vbx, vby]           back to the field frame
```

That `± w * v` is not optional. `a` is what an accelerometer bolted to the
chassis would read; the rate of change of the body-frame velocity components
is a different quantity, and they differ by exactly `w × v`. Skipping it makes
the model wrong the moment the robot turns.

### 8.5 The three easy mistakes

**Saturating in the wrong place, or not at all.** Step 4 comes before step 5.
The gains were fitted against the *saturated* command, so they describe force
delivered rather than force requested. Feeding `u_raw` straight into `A_u`
overestimates what the robot can do near full stick — exactly where it matters.

**Frames.** The tables are field frame, this model is body frame. Rotate in at
step 3 and back out at step 6. Motor forces act along the robot's own axes,
which is precisely why the model *can* be constant matrices — a field-frame
version would need matrices that depend on heading.

**Using `k` from the wrong place.** `k` reflects what the solver actually
used. If `constant_zeroed` is true it is zero here, because the tables were
solved that way. Substituting the raw fit's constant would put the robot on
different dynamics than the tables it is steering by.

### 8.6 Where the numbers come from

Everything in §8.2 is fitted by `calibration/fit_drivetrain.py`. Worth knowing
when reading a file:

- `A_sgn` and `A_absv` are on by default; `A_ss` (the `omega²` term) is off by
  default. On a real run `omega²` scored *worse* than a plain linear model,
  and what looked like a centripetal signal turned out to be a mis-set
  odometry tracking point leaking in.
- The traction knee is fitted per run and **describes that floor**. A grippier
  surface slips later. Re-measure when the surface changes; a knee measured on
  a slippery practice floor will make the robot look weaker than it is. The
  fitter always chooses one — the search picks the best knee rather than
  deciding whether to have a knee — and prints what it bought over an
  unsaturated reference. A small gain there means the run never crossed the
  traction limit, so read the knee as an upper bound and re-measure on a run
  that does cross it.
- There is no control-squared block to fit, and no flag that adds one. See
  §8.2.
- `A_s`, `A_sgn` and `A_absv` are collinear — linear `v`, `csign(v)` and
  `|v|·v` are all odd monotonic functions of the same variable, so how the fit
  splits a given behaviour between them is somewhat arbitrary. Use them
  together; do not read one coefficient on its own as physics.

### 8.7 A complete example

A real `/MODEL.JSON`, cm units, wrapped for readability. Numbers are from a
run with Coulomb friction and drag on and `omega²` off — the normal case.

```json
{
  "schema_version": 2,
  "generator": "peregrine-desktop",
  "regression_sha256": "9f1c0a3e5b7d2e48a6c1f0b93d7e5a2c4b8f6013d29e7a5c1b0f4e8d3a6c9b72",
  "equation": "a = A_s*s + A_u*u + A_ss*(s.*s) + A_uu*(u.*u) + A_sgn*csign(s) + A_absv*(abs(s).*s) + k",
  "csign": {
    "formula": "csign(s)_i = clamp(s_i / coulomb_eps_i, -1, 1)",
    "coulomb_eps": [0.0, 0.0, 0.0, 5.0, 5.0, 0.15],
    "why": "Coulomb friction is smoothed instead of using a hard sign(), which would flip discontinuously at zero and make an integrator chatter. Use exactly this form."
  },
  "output": {
    "vector": ["a_x", "a_y", "alpha"],
    "units": ["cm/s^2", "cm/s^2", "rad/s^2"],
    "frame": "robot body"
  },
  "state": {
    "vector": ["x", "y", "h", "vx", "vy", "w"],
    "units": ["cm", "cm", "rad", "cm/s", "cm/s", "rad/s"],
    "frame_note": "IMPORTANT: vx and vy must be rotated into the ROBOT BODY frame before use here. The value tables index field-frame velocity, so rotate by -h between the two. Motor forces act along the body axes, which is why the model cannot be written with constant matrices in the field frame. x, y and h have zero coefficients throughout."
  },
  "control": {
    "vector": ["fwd", "strafe", "turn"],
    "note": "mecanum projection of the wheel powers; the admissible set is |fwd| + |strafe| + |turn| <= 1",
    "saturation": {
      "knee": 0.55,
      "formula": "m = norm(u_raw); u = u_raw * tanh(m/knee) / (m/knee)",
      "why": "Past the knee the tyres stop delivering, so extra command buys no extra force. APPLY THIS BEFORE A_u -- the gains were fitted against the saturated command. The knee is always present and positive."
    }
  },

  "A_s": [
    [0.0, 0.0, 0.0, -2.418,  0.061,  1.472],
    [0.0, 0.0, 0.0,  0.037, -3.106, -0.884],
    [0.0, 0.0, 0.0,  0.0021, 0.0009, -4.233]
  ],
  "A_u": [
    [318.44,   6.12,  -4.87],
    [ -5.33, 241.07,   3.94],
    [  0.128, -0.061, 13.706]
  ],
  "A_ss":  [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
  "A_uu":  [[0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]],
  "A_sgn": [
    [0.0, 0.0, 0.0, -21.66,  -0.42,   0.53],
    [0.0, 0.0, 0.0,  -0.31, -27.94,  -0.18],
    [0.0, 0.0, 0.0,  0.004, -0.002, -1.882]
  ],
  "A_absv": [
    [0.0, 0.0, 0.0, -0.00417,  0.00008,  0.0312],
    [0.0, 0.0, 0.0,  0.00011, -0.00583, -0.0204],
    [0.0, 0.0, 0.0,  0.0,      0.0,     -0.2461]
  ],
  "k": [0.0, 0.0, 0.0],

  "constant_zeroed": true,
  "nonzero_blocks": ["A_s", "A_u", "A_sgn", "A_absv"]
}
```

Reading it back against the rules above:

- `A_uu` is all zeros and is not in `nonzero_blocks` — as it always is.
- Every state block's first three columns (`x`, `y`, `h`) are zero, so nothing
  in the file couples the command or the state to *where* the robot is.
- `k` is zero and `constant_zeroed` is true: the solve zeroed the fitted
  constant, so the file reports what the tables were actually solved with.
- `A_ss` is present but zero because `omega²` was off. Evaluate it anyway if
  you are not consulting `nonzero_blocks`; the answer is the same.
- The off-diagonal `A_u` entries are small but not zero. That is normal — a
  real chassis is not perfectly symmetric — and they must not be rounded away.

Worked step, using §8.4 with `u_raw = [1.0, 0.0, 0.0]` from rest:

```
m = 1.0,  r = 1.0/0.55 = 1.8182,  gain = tanh(1.8182)/1.8182 = 0.5218
u = [0.5218, 0, 0]
a = A_u * u = [166.2, -2.78, 0.067]        (s = 0, so every other block drops)
```

166 cm/s², not the 318 that `A_u` alone suggests. Skipping step 4 would have
the robot plan for nearly double the acceleration it can produce.

### 8.8 Honing gains — the PID for the final approach

The value tables stop at a **handoff region**, not at the target point. Getting
the last few centimetres exactly right is a different problem: it wants
precision and a light touch, not a minimum-time policy running at full stick.
So `MODEL.JSON` also carries a ready-tuned PID for that stage, under `honing`.

**There is no tuning step.** The gains are derived from the same regression as
the rest of the file. Near the target the drivetrain linearises, and the gains
of a controller for the linearised plant have a closed form. Nothing here needs
a calibration run of its own, and nothing here is a knob to twiddle on the
field.

There is **one choice**, and it is a choice about the robot rather than about
the controller: how much of the traction knee the approach may spend, in
`config.budget_frac`. Half the knee by default. It sets the command budget and,
through it, the bandwidth — see §8.8.5 — so it is the single dial from "gentle
and slow" to "quick and close to the knee". Everything else in `config`
describes the loop the gains will run in. All of it is recorded on the card, so
a card always says what it was designed for.

The block is **additive to schema 2**. A reader written before it existed is
unaffected; a card written by an older solver simply has no `honing` key.

#### 8.8.1 Symbols

| Symbol | Where it comes from | Meaning |
| --- | --- | --- |
| `e` | you | `target - current` **pose** error, body frame: `[fwd, strafe, turn]` in cm, cm, rad |
| `Kp`, `Ki`, `Kd` | the block | 3×3 gain matrices |
| `Lambda` | the block | 3×3 linearised damping, 1/s |
| `B_eff_inv` | the block | 3×3 inverse of the authority the tyres actually deliver |
| `integral_limit` | the block | per-axis clamp on the integrator, cm·s / cm·s / rad·s |
| `omega` | the block | the closed-loop bandwidth the design was sized to, rad/s |
| `fine_band` | the block | the error inside which the command is guaranteed not to saturate |

`e` is a **pose** error, not a velocity error, and it is **body frame** — the
same frame as everything else in §8. The tables are field frame, so rotate by
`-h` first, exactly as §8.3 requires for velocities.

#### 8.8.2 The control law

```
i     = clamp(i + e*dt, -integral_limit, +integral_limit)   # only if not saturated
u_fb  = Kp*e + Ki*i + Kd*de/dt
u     = u_fb + u_ff                                        # u_ff optional, see 8.8.5
```

Then apply the §8.4 traction saturation and clip to the octahedron
`|fwd| + |strafe| + |turn| <= 1`, exactly as you would any other command.

Three things about this are not optional:

- **The integrator must be clamped**, to `integral_limit`. Without it the
  integrator winds the command straight out of the octahedron and the loop
  never settles. This is measured, not theoretical.
- **Freeze the integrator while the command is saturated.** Integrating
  against a limit you cannot exceed only buys authority that does not exist.
- **`de/dt` is the derivative of the error.** With a stationary target that is
  just `-v` in the body frame, which is what you already have; use it rather
  than differencing `e`, which is noisy.

#### 8.8.3 Why these gains

Near the target both the error and the velocity are small, and the model of
§8.2 collapses:

- quadratic drag `A_absv*(|s|.*s)` has **zero slope** at `v = 0` — it vanishes;
- the `omega²` block likewise;
- the smoothed Coulomb term is **linear** inside its band, since
  `csign(v) = v/eps` there. It contributes `A_sgn * diag(1/coulomb_eps)`.

What is left is a damped double integrator in the body frame:

```
dp/dt = v
dv/dt = B_eff*u - Lambda*v,     Lambda = -(A_s_v + A_sgn*diag(1/coulomb_eps))
```

where `A_s_v` is the velocity half of `A_s` (its x/y/h columns are zero, §8.5).

`B_eff` is **not** `A_u`. The honing command is small, but the traction knee
bites well before the octahedron edge, so what the tyres deliver is
`A_u * tanh(r)/r` at the operating point `r = budget/knee`. This is the §8.7
trap in miniature: design against raw `A_u` and you claim roughly twice the
acceleration you have, and the loop overshoots badly. `B_eff_inv` already
contains this factor — **do not apply the traction gain to the gains a second
time.** You still apply §8.4 to the final command, as always.

Both matrices are full 3×3, so the gains are too. That is still **three PID
loops** — three errors, three integrators, three derivatives — but each
output mixes all three errors. That is simply what it means for the axes to
be coupled; a mecanum chassis is not three independent robots.

#### 8.8.4 Pole placement, and why `Kd` is never negative

Multiplying by `B_eff_inv` turns the command into a requested acceleration, and
`Kd` additionally cancels `Lambda`'s off-diagonal part, so the three channels
decouple. Channel `i` then has characteristic polynomial

```
s³ + (lambda_i + kd_i)s² + kp_i*s + ki_i
```

against the drivetrain's own damping `lambda_i = Lambda[i][i]`. Placing poles
at `-p_i` and a double `-omega` gives `kd_i = p_i + 2w - lambda_i`,
`kp_i = w² + 2*p_i*w`, `ki_i = p_i*w²`. The third pole is
`p_i = max(lambda_i - 2w, w)`:

- a **well-damped** axis (`lambda_i >= 3w`) takes `kd_i = 0` and keeps the drag
  it already has, rather than paying derivative gain for it;
- a **lightly damped** axis (`lambda_i < 3w`) takes `p_i = w`, giving a triple
  pole at `-omega` and `kd_i = 3w - lambda_i > 0`.

The branches meet continuously at `lambda_i = 3w`, and `kd_i >= 0` always.

This matters. The obvious design — a triple pole at `-omega` everywhere —
demands `kd_i = 3w - lambda_i`, which on a well-damped axis is **negative**:
a controller spending command to cancel the drivetrain's own friction. It
places the poles correctly on paper and is badly fragile in practice. Simulated
against the nonlinear model it overshoots by a factor of four nominally, and by
five and a half on a robot 30% less draggy than its fit. `verify_tables.py`
fails any card whose `Kd` has a negative diagonal term.

#### 8.8.5 Bandwidth, and the fine band

`omega` is not a free parameter either. It is the smaller of two bounds:

- **saturation**: `|Kp*e|₁ <= budget` for every `e` inside `fine_band`;
- **loop rate**: `omega <= 2*pi*loop_hz/10`, since a sampled loop cannot track
  a pole much faster than a tenth of its rate.

Note the first is sized against the **fine band**, not the handoff distance.
Saturating on the way in from the handoff corner is intended and harmless —
`Kd >= 0` and the clamped integrator make it safe, and it is what makes the
approach quick. What matters is that the controller is linear and gentle once
it is *close*, which is the whole point of honing. Sizing against the handoff
corner instead costs about a factor of eight in bandwidth and the robot never
settles.

`config.budget` defaults to **half the traction knee** (`budget_frac` 0.5), so
honing runs at a fraction of full power and the linearisation stays honest
about itself (the traction gain is still ~0.92 there). Raising `budget_frac`
raises the saturation bound with it, which is why it is the dial to reach for
when the approach is too slow: it buys command and bandwidth together, and the
fine-band guarantee still holds at the new budget. `bandwidth_scale` above 1
does not: it steps over that bound rather than lifting it, and
`verify_tables.py` rejects any card whose `Kp` then leaves the budget inside
the fine band. A budget outside the octahedron
(`budget_frac` above `1/knee`) is refused outright: the wheels would clip the
command the gains were sized against.

Both ends of the dial are real mistakes, and only the nonlinear rollout of
§8.8.6 tells them apart — too hot overshoots, too gentle is still short of the
point when the time is up. At `budget_frac` 0.25 the §8.7 model misses its
landing budget on most of the robustness cases; at 0.9 it settles in 1.3 s and
passes them all, at the cost of more overshoot to absorb.

**Feedforward** is optional and worth having. `u_ff = B_eff_inv*(a_ref +
Lambda*v_ref)` asks for the command the fit says produces the reference motion,
leaving the PID only the residual. `Lambda` already contains the linearised
Coulomb term, so no separate break-away command is needed. With a stationary
target, `a_ref = v_ref = 0` and `u_ff` drops out.

#### 8.8.6 A worked example

The `honing` block that the §8.7 model produces, at the default config
(`loop_hz` 50, `budget` = 0.5 × 0.55 = 0.275, `fine_band` 2 cm / 0.03 rad):

```json
"honing": {
  "frame": "robot body",
  "law": "u = Kp*e + Ki*clamp(integral(e), -integral_limit, integral_limit) + Kd*d(e)/dt + u_ff",
  "Kp": [[ 0.0360318, -0.0012504,  0.0381302],
         [ 0.0008021,  0.0652998, -0.0391032],
         [-0.0003329,  0.0003023,  2.4441237]],
  "Ki": [[ 0.0165087, -0.0005896,  0.0187047],
         [ 0.0003675,  0.0307902, -0.0191820],
         [-0.0001525,  0.0001425,  1.1989574]],
  "Kd": [[ 5.71816e-06, -7.74879e-05,  0.0171762],
         [-1.15812e-04, -2.37000e-06, -0.0089704],
         [ 2.28362e-04,  4.01840e-05, -0.0002003]],
  "B_eff_inv": [[ 0.00339581, -0.00008590,  0.00123129],
                [ 0.00007559,  0.00448599, -0.00126271],
                [-0.00003138,  0.00002077,  0.07892471]],
  "Lambda": [[ 6.750000,  0.023000, -5.005333],
             [ 0.025000,  8.694000,  2.084000],
             [-0.002900, -0.000500, 16.779666]],
  "traction_gain": { "value": 0.9242 },
  "integral_limit": { "value": [1.3457636, 0.7269980, 0.0185283],
                      "units": ["cm*s", "cm*s", "rad*s"] },
  "omega": 1.0148450,
  "poles": { "double": 1.0148450,
             "third": [4.720310, 6.664310, 14.749976],
             "kd_diag": [0.0, 0.0, 0.0] },
  "omega_limits": { "saturation": 1.0148450, "loop_rate": 31.4159265,
                    "bound_by": "saturation" },
  "handoff": { "cm": 15.0, "rad": 0.25 },
  "fine_band": { "cm": 2.0, "rad": 0.03 },
  "config": { "loop_hz": 50.0, "budget": 0.275, "budget_frac": 0.5,
              "bandwidth_scale": 1.0, "integral_share": 0.25,
              "fine_cm": 2.0, "fine_rad": 0.03 }
}
```

Checks you can do by hand:

```
Lambda[0][0] = -(A_s[0][3] + A_sgn[0][3]/coulomb_eps[3])
             = -(-2.418 + -21.66/5) = 6.750           ✓
traction gain: r = 0.275/0.55 = 0.5, tanh(0.5)/0.5 = 0.92423       ✓
B_eff[0][0]  = 318.44 * 0.92423 = 294.31; the 3x3 inverse of B_eff
               has [0][0] = 0.00339581                             ✓
lambda_0 = 6.750 >= 3*omega = 3.045, so axis 0 is well damped:
  kd_0 = 0, p_0 = 6.750 - 2*1.014845 = 4.720310                    ✓
  kp_0 = omega^2 + 2*p_0*omega = 1.02992 + 9.58079 = 10.61071
  Kp[0][0] = B_eff_inv[0][0] * kp_0 = 0.00339581 * 10.61071
           = 0.0360318                                             ✓
saturation is tight, as it should be — omega was solved for it:
  |Kp * [2, 2, 0.03]|_1 = 0.2750 = budget                          ✓
```

Note `Kd`'s diagonal is zero here: all three axes are well damped, so the
controller adds no derivative gain at all and `Kd` carries only the small
off-diagonal terms that decouple the axes.

The honing loop against the full **nonlinear** model, from the worst-case
handoff corner (15 cm, 15 cm, 0.25 rad), settles in **1.9 s to under 0.005 cm
and 0.0003 rad**, with 3.6% overshoot, clipping the command for the first ~10%
of ticks. It stays inside a 0.1 cm final error across ±50% drag error and ±20%
authority error. Reproduce with:

```
julia --project=solver solver/bench/honing.jl --example
```

Against a real fit, pass a config instead of `--example`; only its
`regression`, `zero_c` and `honing_*` keys are read, so no field, target list
or solve is needed. The `honing_*` keys are the `config` block above, prefixed:
`honing_budget_frac`, `honing_bandwidth_scale`, `honing_loop_hz`,
`honing_fine_cm`, `honing_fine_rad`, `honing_handoff_cm`,
`honing_handoff_rad`, `honing_integral_share`. `honing_budget` pins the budget
outright instead of as a fraction of the knee.

To see what a setting does without solving anything:

```
julia --project=solver solver/solve.jl config.json honing            # print
julia --project=solver solver/solve.jl config.json honing <run_dir>  # and write
```

With a run directory it rebuilds that run's `MODEL.JSON` in place, which is
sound because the gains never depended on the tables — only on the fit both
were made from. It refuses if that is not the fit the run was solved from. The
desktop wizard drives all of this from its optional `h` step, which also runs
the rollout above.

## 9. Notes for the robot side

**The tables hand over; they do not arrive.** The value function is a
minimum-time policy to the *handoff region*, not to the target point. Drive it
with the rule below until you are inside roughly 15 cm and 0.25 rad, then hand
to the honing PID of §8.8, which owns the last stretch and settles on the point
without running at full stick. Those two stages answer different questions and
neither substitutes for the other.

**Keep chunk files open.** Reopening a file every loop cycle costs far more
than the read. Consecutive queries almost always land in the same chunk or
two, so a small cache keyed by chunk number is enough.

**Interpolate if you need a smooth value.** §3 snaps to the nearest cell,
which is fine for ranking candidate directions. For a continuous value,
interpolate between neighbouring cells — wrapping on the heading axis and
clamping on the other five, per §3.

**The table is solved with the interpolant you read it with.** As of
2026-08-25 the solver's Bellman backup uses 6D Kuhn (Freudenthal) simplex
interpolation — the same 7-vertex simplex the online optimizer locates to
recover `grad(V)`. That is deliberate: `V` is the fixed point of whichever
interpolant the backup is written with, so a table solved with 64-point
multilinear and read with simplex differences is self-consistent under an
operator nobody applies. `MANIFEST.JSON` records which was used under
`solver.interpolant`. If the robot-side gradient recovery ever changes, the
solver's `simplex` flag has to change with it.

If you *do* choose a command by minimising `T + V(state after holding it for
T)` rather than by the gradient rule, note that `T` must be a real planning
horizon, not the loop period. A minimum-time value function satisfies
`V(s) − V(s′) = T` along an optimal path: the value earned is proportional to
`T`, while from a standstill the *state* only moves as `T²`. Too small and
every command loses to standing still — a 0.02 s horizon parks the robot a
few centimetres short and holds it there indefinitely, while 0.15 s drives in
cleanly.

**The wall is an obstacle, and the table already knows it.** As of 2026-08-31
the solver sweeps the robot footprint against the perimeter exactly as it does
against every other obstacle, and holds `solver.wall_clearance_cm` off it. Two
consequences for the robot side. The stored span is smaller than the field, so
read `grid.min`/`grid.max` and do not substitute the field dimensions for them
(§2). And a wall-hugging state can be unreachable at one heading and fine at
the heading 90° from it, which is the table correctly describing a chassis
that does not fit sideways into a gap — not a hole in the solve.

**Stay inside the velocity envelope.** `grid.min` and `grid.max` bound `vx`,
`vy` and `ω`, and the table says nothing outside them — states beyond the box
are unreachable, not merely expensive. The drivetrain can exceed those speeds
(the fitted model's terminal speed is far above a typical `vmax`), so this is
a live constraint, not a theoretical one. Treat a query outside the envelope
as "no value available" and fall back, rather than clamping it to the edge.

---

## 10. Verifying a card

```bash
py -3.12 wizard/verify_tables.py G:\
```

This implements §3 to §6 directly from this document rather than from the
solver's code, and checks `MODEL.JSON` against §8, so it tests the contract instead of agreeing with whatever the
solver happened to write. It checks chunk counts and sizes, SHA-256, FAT32
file limits, that the value at each target is approximately zero, that
time-to-go grows with distance, and that the velocity axes behave as field
frame.

For the escape band (§6.1) it checks that the band stops exactly at the
sentinel, that it spans as far as `value_cap_s`, and — the property the whole
design rests on — that **no cell decodes as both reachable and escapable**. A
card where those two overlap is one where the robot cannot tell a route from
an obstacle, so that check is a failure and not a warning.

Change anything here and `verify_tables.py` needs the same change, and the
robot side will too.
