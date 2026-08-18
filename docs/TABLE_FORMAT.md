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
    "min":   [0.0, 0.0, -3.14159265, -150.0, -150.0, -10.0],
    "max":   [366.0, 366.0, 3.14159265, 150.0, 150.0, 10.0],
    "wrap":  [false, false, true, false, false, false],
    "units": ["cm", "cm", "rad", "cm/s", "cm/s", "rad/s"],
    "frame": "field",
    "total_cells": 111572448,
    "index_formula": "((((ix*Ny+iy)*Nh+ih)*Nvx+ivx)*Nvy+ivy)*Nw+iw"
  },

  "encoding": {
    "dtype": "u16",
    "elem_bytes": 2,
    "scale": 0.001,
    "unit": "seconds",
    "unreachable": 65535,
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
      "reached_frac": 0.88
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
| `chunk_elements` | `encoding.chunk_elements` | values per chunk file, a power of two |
| `chunk_shift` | `encoding.chunk_shift` | `log2(chunk_elements)` |

Read these from the file rather than hard-coding them. The grid is expected to
change as the resolution gets tuned; that should not require a firmware change.

Three fields are informational rather than needed for lookup:
`sha256` covers the concatenated logical table in index order, so it does not
change if the chunking does; `reached_frac` is the fraction of cells that got
a real answer, and a low value warns that much of the state space could not
reach that target; `regression_sha256` and `field_sha256` identify which
drivetrain fit and field description produced the tables.

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

Clamping is correct behaviour, not a fallback. A position off the field is
meaningless, and a velocity past the edge of the grid means the robot is
moving faster than the drivetrain model was ever fitted for; the nearest edge
cell is the best answer available in both cases.

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
integer dtypes:   raw == unreachable   ->  +infinity
float dtypes:     isnan(raw)           ->  +infinity
```

Integers compare exactly, so `==` is safe there. Floats need `isnan`, because
`NaN == NaN` is false by definition and an equality test would never fire.

Unreachable must be treated as `+infinity`, not as a large finite number, so
it loses every comparison against a real route. It covers three situations —
inside an obstacle, off the field, or no route found within the solved horizon
— which are deliberately not distinguished, since all three mean the same
thing to the robot.

The default `u16` with `scale = 0.001` is plain milliseconds: exact to 1 ms up
to 65.534 s, in half the space of `f32`. `f16` is the same size but carries
only about three significant digits, so prefer `u16` unless you specifically
want floats. `u8` with `scale = 0.025` gives 25 ms steps up to 6.35 s in one
byte, which is worth it when the horizon is short and the grid is large.

---

## 7. Worked example

Using the manifest shown in §2, and the state

```
x = 100 cm, y = 200 cm, h = 1.0 rad, vx = 50 cm/s, vy = -30 cm/s, w = 2.0 rad/s
```

**Steps** (§3), with `n = [46, 46, 24, 13, 13, 13]`:

```
step[0] = (366 - 0) / (46 - 1)   = 8.133333    step[3] = 300 / 12 = 25.0
step[1] = (366 - 0) / (46 - 1)   = 8.133333    step[4] = 300 / 12 = 25.0
step[2] = 2*pi / 24              = 0.261799    step[5] =  20 / 12 =  1.666667
```

**Indices** (§3):

```
ix  = round((100 - 0)      / 8.133333) = round(12.2951) = 12
iy  = round((200 - 0)      / 8.133333) = round(24.5902) = 25
ih  = round((1.0 - -pi)    / 0.261799) = round(15.8197) = 16   (mod 24)
ivx = round((50 - -150)    / 25.0)     = round( 8.0000) =  8
ivy = round((-30 - -150)   / 25.0)     = round( 4.8000) =  5
iw  = round((2.0 - -10)    / 1.666667) = round( 7.2000) =  7
```

**Flat index** (§4):

Evaluated from the inside out, one axis per line:

```
     12 * 46 + 25  =        577      folded in iy
    577 * 24 + 16  =      13864      folded in ih
  13864 * 13 +  8  =     180240      folded in ivx
 180240 * 13 +  5  =    2343125      folded in ivy
2343125 * 13 +  7  =   30460632      folded in iw   -> idx
```

**File and offset** (§5), with `chunk_shift = 23`:

```
chunk         = 30460632 >> 23      = 3          -> TABLES/T00C0003.BIN
elem_in_chunk = 30460632 & 8388607  = 5294808
byte_offset   = 5294808 * 2         = 10589616
```

**Decode** (§6): read 2 bytes little-endian at 10589616. If they read `65535`,
the state is unreachable; otherwise the answer is `raw * 0.001` seconds.

---

## 8. The drivetrain model

The tables say how long a route takes. They do not say how the robot moves,
and the online optimizer needs both, so `/MODEL.JSON` carries the fitted
drivetrain alongside them.

```
a = A_s*s + A_u*u + A_ss*(s.*s) + A_uu*(u.*u) + k
```

where `.*` is elementwise squaring, and

| symbol | shape | meaning |
| --- | --- | --- |
| `a` | 3 | output acceleration, `[a_x, a_y, alpha]` |
| `s` | 6 | state, the same `[x, y, h, vx, vy, w]` as everywhere else |
| `u` | 3 | control, `[fwd, strafe, turn]` |
| `A_s` | 3x6 | state to acceleration |
| `A_u` | 3x3 | control to acceleration |
| `A_ss` | 3x6 | squared state to acceleration |
| `A_uu` | 3x3 | squared control to acceleration |
| `k` | 3 | constant |

Read the shapes from the declared `state`, `control` and `output` vectors
rather than hard-coding 3 and 6.

Several blocks are currently all zero: the fit has no position dependence, so
the `x`, `y` and `h` columns of `A_s` and `A_ss` are zero, and there are no
control-squared terms, so `A_uu` is zero throughout. They are written out
anyway. The point is that the regression can gain or lose terms later without
the robot-side reader changing — it always multiplies the same five things and
sums them. `nonzero_blocks` lists which are actually carrying anything, if you
want to skip the rest for speed.

### Two things that will bite

**The velocities in `s` must be body frame here.** The value tables index
field-frame velocity, because that is what odometry gives you. This model is
body frame, because motor forces act along the robot's own axes — which is
exactly why it *can* be written with constant matrices. Rotate `vx, vy` by
`-h` before evaluating the model, and rotate back afterwards if you need the
result in field terms. `alpha` and `w` are the same in both frames.

**`k` reflects what the solver used, not the raw fit.** If the solve zeroed
the constant term, `k` is zero here too, and `constant_zeroed` says so.
Otherwise the robot's dynamics would disagree with the tables solved from
them.

---

## 9. Notes for the robot side

**Keep chunk files open.** Reopening a file every loop cycle costs far more
than the read. Consecutive queries almost always land in the same chunk or
two, so a small cache keyed by chunk number is enough.

**Interpolate if you need a smooth value.** §3 snaps to the nearest cell,
which is fine for ranking candidate directions. For a continuous value,
interpolate between neighbouring cells — wrapping on the heading axis and
clamping on the other five, per §3.

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

Change anything here and `verify_tables.py` needs the same change, and the
robot side will too.
