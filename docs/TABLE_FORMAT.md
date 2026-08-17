# Peregrine value-table SD card format

This is the contract between the desktop solver and the online optimizer on
the robot. The desktop side writes it; the robot side reads it. Nothing else
depends on it, and nothing else should.

---

## 1. What is on the card, and why

The solver produces, for every target you asked about, a **table of
minimum-time-to-go**. One number per state:

> If the robot were in state `s` right now, what is the shortest possible time
> to reach the target, driving as hard as the drivetrain allows and avoiding
> every obstacle?

That number is the *value function*. The robot does not re-plan from scratch
each loop; it looks up a few of these numbers around its current state and
picks the direction that decreases the value fastest. All the expensive
searching already happened on the desktop.

A state is six numbers, and that is the reason the tables are large: a table
is a six-dimensional array, so its size is the **product** of all six axis
resolutions. Doubling one axis doubles the file. Doubling all six multiplies
it by 64. That single fact drives every other decision below.

The card holds only two kinds of thing:

```
/MANIFEST.JSON            one small file describing everything
/TABLES/T00C0000.BIN      raw table data, nothing else
/TABLES/T00C0001.BIN
/TABLES/T01C0000.BIN
...
```

No logs, no configs, no leftovers. Everything except the manifest is payload.

---

## 2. The state vector

The same six numbers appear everywhere in Peregrine, always in this order:

| axis | name | meaning | unit |
| ---: | --- | --- | --- |
| 0 | `x` | position across the field | cm |
| 1 | `y` | position along the field | cm |
| 2 | `h` | heading | rad |
| 3 | `vx` | velocity, x component | cm/s |
| 4 | `vy` | velocity, y component | cm/s |
| 5 | `w` | angular velocity | rad/s |

**Everything is field frame.** Position, heading, and *both velocity
components* are measured against the field, not the robot. `w` is the same in
either frame, so the question does not arise for it.

This matters because it is a deliberate choice that costs the desktop a little
and saves the robot a lot. The drivetrain model is naturally body-frame — motor
forces push along the robot's own axes — so the solver rotates into the body
frame internally, steps the physics, and rotates back out before storing. Doing
that once per cell on a GPU is free. Doing the inverse rotation on the robot,
every loop cycle, is not. The robot reads field-frame velocity straight from
odometry and indexes the table with it, with no trigonometry at all.

### Heading wraps, the others do not

Axis 2 is periodic. It covers a full turn, `[-pi, +pi)`, in `n[2]` equal
steps:

```
step  = 2*pi / n[2]
value = -pi + i * step
```

Index `n[2]` is the same state as index `0`. When interpolating or looking at
neighbours, wrap with `mod`, never clamp — heading `+3.10` and `-3.10` rad are
neighbours, not opposite ends of the range.

Every other axis is a plain inclusive span from `min[k]` to `max[k]`:

```
step  = (max[k] - min[k]) / (n[k] - 1)
value = min[k] + i * step
```

Note the `n[k] - 1`: both endpoints are sample points. For heading it is
`n[2]`, not `n[2] - 1`, precisely because the two ends are the same state and
sampling both would be a duplicate.

### Going from a state to indices

```
i = round((value - min[k]) / step[k])          # non-wrapping axes: then clamp
i = mod(round((value - min[2]) / step[2]), n[2])   # heading
```

Clamping the non-wrapping axes is the right behaviour, not a fudge. Position
outside the field is meaningless, and velocity outside the grid means the
robot is moving faster than the model was ever fitted for — the nearest edge
cell is the best available answer in both cases.

---

## 3. Finding a cell

The six-dimensional array is flattened into one long run of numbers in
**row-major order**: the last axis varies fastest, the first slowest.

```
idx = ((((ix * Ny + iy) * Nh + ih) * Nvx + ivx) * Nvy + ivy) * Nw + iw
```

The same formula is repeated verbatim in the manifest as `index_formula`, so
the robot code and the file can be checked against each other.

Row-major with `w` last is not arbitrary. The robot's lookups are clustered
around its current state, and neighbours along the *last* axes are adjacent in
the file. Reading a small neighbourhood therefore touches a short contiguous
stretch rather than scattering across the whole table.

### Why the data is split into chunks

`idx` can run into the billions, and a single table can be far larger than any
file the card can hold — FAT32 caps a file at 4 GiB, and FAT32 is the only
filesystem the Control Hub accepts. So the flat run of numbers is cut into
equal **chunks**, each written as its own file.

The chunk size is measured in **elements, not bytes**, and is always a power of
two. That is what makes the lookup cheap: a division and a modulo become a
shift and a mask.

```
chunk         = idx >> chunk_shift
elem_in_chunk = idx &  (chunk_elements - 1)
byte_offset   = elem_in_chunk * elem_bytes
```

with `chunk_elements == 1 << chunk_shift`. Default is `2^23` elements, which
is 16 MiB per file at two bytes each. Every chunk is exactly full except the
last one of each target.

### Why the filenames look like that

```
TABLES/T00C0000.BIN
       ^^ ^^^^
       |  chunk number, 4 digits
       target number, 2 digits
```

Eight characters, then `.BIN`: a strict **8.3** name. FAT32 stores a long
filename by chaining several directory entries together, so `table_00_0000.bin`
would cost four entries where `T00C0000.BIN` costs one. With thousands of
chunks that is a real amount of the card's directory space, and it also means
the robot never has to deal with long-filename handling.

---

## 4. Reading a number

Each element is one time-to-go value. Four encodings are available, chosen
when you solve, because these tables are big enough that precision is the
cheapest thing to trade away.

| `dtype` | bytes | how to read it | "unreachable" is |
| --- | ---: | --- | --- |
| `u8` | 1 | unsigned int, `seconds = raw * scale` | `255` |
| `u16` | 2 | unsigned int, `seconds = raw * scale` | `65535` |
| `f16` | 2 | IEEE half, already seconds | `NaN` |
| `f32` | 4 | IEEE single, already seconds | `NaN` |

All little-endian.

**The default is `u16` with `scale = 0.001`** — plain milliseconds. It covers
0 to 65.534 s with exact 1 ms resolution, in half the space of `f32`. Prefer
it to `f16`, which costs the same two bytes but carries only about three
significant digits, so it is already rounding to tenths of a second by the
time you reach a minute.

`u8` with `scale = 0.025` gives 25 ms steps up to 6.35 s, in one byte. Worth it
when the horizon is genuinely short and the grid is large.

### The unreachable sentinel

A cell reads as the sentinel when the robot cannot get from that state to the
target: the state is inside an obstacle, or off the field, or no route was
found within the solved horizon.

These are deliberately **not** distinguished. All of them mean the same thing
to the robot — *do not plan through here* — so collapsing them costs nothing
and reduces the check to a single comparison in the hot loop.

```
raw == unreachable  ->  +infinity
otherwise           ->  raw * scale     (integer types)
                    ->  raw             (float types)
```

Treat unreachable as `+infinity`, not as a large finite number. It has to lose
every comparison against a real route.

---

## 5. The manifest

`/MANIFEST.JSON`, UTF-8, no BOM. Deliberately short: individual chunk
filenames are **not** listed, because they follow from `file_pattern` and
`n_chunks`. Listing several thousand of them would make the manifest larger
than it has any need to be.

```json
{
  "schema_version": 1,
  "generated_utc": "2026-08-16T21:40:00Z",
  "generator": "peregrine-desktop",
  "regression_sha256": "...",
  "field_sha256": "...",

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

Read `n`, `min`, `max`, and `wrap` from the file rather than hard-coding them.
The whole point of the manifest is that the grid can change without a firmware
change.

A few fields deserve a note:

- **`sha256`** is over the *concatenated logical table*, in index order — not
  per chunk. So it does not change if the chunk size changes, and it is the
  same value `verify_tables.py` recomputes.
- **`reached_frac`** is the fraction of cells that got a real answer. A low
  number is a warning that much of the state space could not reach that
  target within the solved horizon.
- **`regression_sha256`** and **`field_sha256`** identify exactly which
  drivetrain fit and which field description produced these tables, so a card
  can always be traced back to its inputs.

---

## 6. A complete lookup

Everything above, in the order the robot does it:

```java
// 1. state -> indices  (field-frame velocity, straight from odometry)
int ix  = clamp(round((x - min0) / step0), 0, n0 - 1);
int iy  = clamp(round((y - min1) / step1), 0, n1 - 1);
int ih  = floorMod(round((h - min2) / step2), n2);     // wraps
int ivx = clamp(round((vx - min3) / step3), 0, n3 - 1);
int ivy = clamp(round((vy - min4) / step4), 0, n4 - 1);
int iw  = clamp(round((w  - min5) / step5), 0, n5 - 1);

// 2. indices -> flat index
long idx = ((((long) ix * n1 + iy) * n2 + ih) * n3 + ivx) * n4 + ivy;
idx = idx * n5 + iw;

// 3. flat index -> file and byte offset
int  chunk  = (int) (idx >>> chunkShift);
long offset = (idx & (chunkElements - 1)) * elemBytes;

// 4. read, and decode
//    open TABLES/T<tt>C<cccc>.BIN, seek(offset), read elemBytes little-endian
double seconds = (raw == unreachable)
               ? Double.POSITIVE_INFINITY
               : raw * scale;
```

Two practical notes for the robot side:

**Keep the chunk files open.** Reopening a file every loop cycle will cost far
more than the read itself. The states queried in one cycle are close together,
so they nearly always land in the same chunk or two — a small cache of open
handles keyed by chunk number is enough.

**Interpolate if you need smoothness.** The steps above snap to the nearest
cell, which is fine for comparing candidate directions. If you want a smooth
value, interpolate between neighbouring cells — remembering to wrap on the
heading axis and clamp on the rest.

---

## 7. Checking a card

```bash
py -3.12 wizard/verify_tables.py G:\
```

This implements the lookup from *this document*, independently of the solver's
own code, so it tests the contract rather than agreeing with whatever the
solver happened to write. It checks chunk counts and sizes, SHA-256, FAT32
file limits, that the value at each target is approximately zero, and that
time-to-go grows as you move away from a target.

If you change anything in this file, change `verify_tables.py` too, and expect
the robot side to need updating — that is the whole point of writing it down.
