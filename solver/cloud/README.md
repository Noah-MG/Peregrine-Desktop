# Solving on a rented GPU

The desktop's 8 GB card is the binding constraint on how big a value table
this project can produce, and it binds twice: once on the grid, and again on
the **lookahead**, because a grid that has to be tiled pays for `tau_max`
through the halo. Renting a bigger card for a few hours relaxes both.

This directory is the whole apparatus: `provision.sh` runs on the rented box,
`peregrine_remote.py` runs here.

Plan whenever you like, in the wizard, with nothing rented. Save the plan as
a **job**. Then, on the day you actually want the compute:

```bash
py -3.12 solver/cloud/peregrine_remote.py jobs                          # what is saved
py -3.12 solver/cloud/peregrine_remote.py provision root@<ip>
py -3.12 solver/cloud/peregrine_remote.py run overnight --host root@<ip>
```

`run` pushes the working tree and the inputs, starts the solve under `tmux`,
draws the same progress bar the wizard does, pulls the tables back, verifies
them against `docs/TABLE_FORMAT.md`, and prints the bill.

`--host` goes on every command that talks to a box. Section 8 is why it is
per invocation rather than remembered.

---

## 1. What a bigger card actually buys

Not a guess — `plan` was run on the real 08-30 config with
`vram_budget_bytes` set to each card's budget, so these are this solver's own
numbers on this field and this regression.

**The 08-30 production grid**, `[72, 72, 32, 21, 21, 32]` = 2.34 billion
cells, 4.4 GB per table, three targets:

| | 8 GB desktop card | 48 GB L40S |
|---|---|---|
| driver | tiled, 16 tiles of 22x22 | **whole grid on the card** |
| halo | 11 cells, 4.0x loaded per updated | none |
| scratch file | 15.3 GB | none |
| longest lookahead that fits | 0.2 s | 0.5 s |
| accuracy cost of that | **+6.0% mean value** | none |
| estimate | 47.0 h | 33.3 h |

**Full resolution**, `[161, 161, 64, 21, 21, 21]` = 15.4 billion cells,
28.6 GB per table:

| | 8 GB desktop card | 48 GB L40S |
|---|---|---|
| at `tau_max` 0.5 | **does not fit at all** | tiled, 16 tiles of 53x53, 3.6x |
| best that does fit | 0.1 s, 40x amplification | — |
| accuracy cost | **+21.1% mean value** | none |
| estimate | 956 h (0.06 s / +56%: 369 h) | 302 h |

The second table is the real argument, and it is not about speed: at full
resolution the desktop card **cannot run the accurate scheme at any price in
time**. The 48 GB card can.

### And the 141 GB H200, which is the one to rent

Added 2026-09-04, on Noah's quote of **six cents an hour** over the H100 --
so $3.45 against $3.39, which is what `TARGET_GPUS` in `wizard/peregrine.py`
now carries. Re-check the price before renting; at full resolution those six
cents change the driver.

100 GB is what full resolution needs to be held whole (15.4e9 cells at
`cell_bytes` 7). The budgets are `(VRAM - 1 GB) * 0.85`, so:

| card | budget | in-core ceiling | full resolution |
|---|---|---|---|
| L40S 48 GB | 39.9 GiB | 6.1e9 cells | tiled |
| H100 80 GB | 67.1 GiB | 1.0e10 cells | tiled |
| **H200 141 GB** | **119.0 GiB** | **1.8e10 cells** | **whole grid on the card** |

`plan` on the full-resolution grid `[161,161,64,21,21,21]`, three targets,
every card costed at the **H100's measured 153M cell-updates/s** so the
comparison isolates the VRAM and nothing else:

| | L40S 48 GB | H100 80 GB | H200 141 GB |
|---|---|---|---|
| driver | tiled, 2.14x | tiled, 2.17x | **in core** |
| longest lookahead that fits | 0.1 s | 0.15 s | **0.5 s** |
| accuracy cost of that | +21.1% mean value | +12.3% mean value | **none** |
| scratch store on disk | 100.2 GB | 100.2 GB | **none** |
| estimate | 54.5 h | 54.6 h | **38.5 h** |
| at that card's rate | $86 | $185 | **$133** |

So against the H100 the H200 is **more accurate and cheaper in absolute
dollars** -- it removes a +12.3% error and saves about $52, because dropping
the halo removes 2.17x of redundant loading and the run gets 30% shorter.
It also removes the 100 GB scratch store, so the box needs a disk for the
tables and nothing else.

Two honesties about that table:

- **The H200's cell rate is not measured.** It is quoted at the H100's,
  which is the conservative choice: the two have the same SM count and
  clocks, and the H200's advantage is memory bandwidth (4.8 vs 3.35 TB/s),
  which this sweep is more likely to be helped by than hurt. Run
  `peregrine_remote.py benchmark` on the box before believing 38.5 h.
- **At the grid Noah is actually running** -- 2.54 cm, `[130,130,36,19,19,17]`,
  3.73e9 cells, 24.3 GB -- an H100 already holds it whole at `tau_max` 0.5,
  and `plan` returns the *identical* 9.3 h estimate for all three cards. The
  H200 buys nothing there. Its case is full resolution, and only that.

### Read those estimates as ceilings on the iteration count

They assume the full `iterations` budget. `tolerance` normally stops a target
well short — on the 35.8M-cell grid it cut 612 s of budget to a measured
171 s.

They no longer assume a cell rate. That is worth spelling out, because the
old `cell_rate` 23.4e6 was wrong in two directions at once and the errors hid
each other:

- **It was stale, in the flattering direction.** The same desktop card
  measures 51-70M cell-updates/s across grid shapes, so every card looked
  ~2.5x faster than the desktop when the honest figure for a modern card of
  that class is nearer 1x.
- **It was measured on the wrong workload, in the other direction.** The
  benchmark swept at `cfl` 1 / `tau_max` 0.2 with a synthetic drivetrain that
  has 1.7x the yaw authority of a real fit. All three make the step shorter,
  and a shorter step means fewer integration substeps per lookahead. A
  production sweep — `cfl` 2, `tau_max` 0.5, a real fit — asks for **2.2x**
  the work per cell that the benchmark's did.

The second one is the one that bit. Job `H200_med_one`, 1.78e9 cells on a
rented H200, was planned at **1 h 29 m** and took **2 h 32 m**: 400 sweeps at
20.6 s where the plan had priced them at 11.6 s, plus 12 minutes of escape
pass the plan had priced but the progress bar had not.

What replaced it: `benchmark.jl` sweeps the same grid under five horizon
regimes and fits

    seconds per active cell = a * lookaheads + b * (substeps + probes)

then levels the pair against one sweep of a production-sized grid. `plan`
computes the three counts for the actual grid, drivetrain and settings
(`sweep_work`) and multiplies. Fitting on two regimes and predicting a third
lands within 5%; the H200 job, priced with a pair matching its measured
sweep, comes out at 2 h 30 m against 2 h 32 m actual.

Two coefficients instead of one number is the whole change, and it is why
`run` no longer offers you a `cell_rate` to paste into a config: a rate is a
rate of one workload, and the next job is a different one. Run `benchmark` on
the box instead.

On the second point, one risk is already ruled out: the sweep kernel is
Float32 throughout (`Grid6`, `Model` and `Params` hold only `Float32`/`Int32`,
and `cell_update`, `interp_kuhn` and the integrators name no `Float64`). The
L40S's 1:64 FP64 rate, which would otherwise be a trap on a workstation card,
never comes into it.

---

## 2. Renting the box

### The image is the decision that goes wrong

Pick an **AI/ML-ready image**, not a plain OS image. It has the NVIDIA driver
already. Installing a driver yourself on a just-released Ubuntu is the
flakiest step in this whole exercise and it costs a reboot; `provision.sh
--driver` will attempt it, but it is a fallback, not the plan.

Everything above the driver, `provision.sh` installs: Julia via juliaup, the
solver project from the checked-in `Manifest.toml`, and CUDA.jl's own toolkit
artifacts (~2-3 GB, the slow part of a first provision).

### Sizing

- **VRAM** is what decides in-core vs tiled, at **7 bytes a cell** with warm
  start — `cell_bytes`, which is now the single place that decides it. So the
  in-core ceiling is about **6.1e9 cells on an L40S**, **1.0e10 on an H100**
  and **1.8e10 on an H200**, an 11 GB, a 19 GB and a 34 GB `u16` table
  respectively. Full resolution needs 100 GB to hold whole, so the H200 is
  the first card in this list that does not tile it.

  It used to be two numbers. `decompose` sized the whole-grid case at 7 B/cell
  while `run_solve` allocated 16 (the in-core policy was three `Float32`), so
  a grid between `budget/16` and `budget/7` was routed in core and could not
  be allocated — and Windows paged it into host RAM rather than failing, so
  only Linux would have raised it. Fixed by quantising the in-core policy to
  bytes like the tiled one: measured at **−0.4% and +0.1% on sweep rate** for
  a 2.29x VRAM saving, and no accuracy cost (self-test 15). There is also now
  a pre-flight VRAM check before the in-core allocation, mirroring the one the
  tiled path already had for disk.
- **Disk** must hold the scratch store (7 B/cell tiled) *and* the tables.
  Full resolution is a 100 GB store plus 28.6 GB per target — a 500 GB boot
  disk covers three targets with room, and nothing smaller does.
- **vCPU and RAM** barely matter. The work is on the GPU.
- **More than one GPU is money burnt.** The solver uses one.

### Destroying it is the part that costs money

Billing starts at create and stops at **destroy**. Powering the machine off
does not stop it — the GPU stays reserved. Every `run`, `status` and `cost`
prints what the rental has cost so far for exactly this reason.

```bash
doctl compute droplet delete <name>
```

---

## 3. The commands

Every command that talks to a box takes `--host USER@IP`.

| | |
|---|---|
| `provision <host> [--driver] [--quick]` | Julia, the project, CUDA check, solver self-test, disk-rate measurement |
| `plan <job\|config.json>` | what the box would do with this grid, and what it would cost |
| `run <job\|config.json>` | push, solve, stream each table home as it lands, verify, bill |
| `attach` | re-follow a solve after a dropped connection, collecting any tables it missed |
| `status` | what the box is, whether a solve is running, disk left |
| `pull` | fetch the last solve's tables again |
| `cost` | what this rental has run up |
| `jobs` | list the plans saved for later |
| `host <host>` | remember a box between commands, instead of repeating `--host` |
| `forget` | drop the remembered box and its rental clock |
| `benchmark [--seconds N]` | measure this box's cell and disk rates |
| `--self-test` | check this script with no box at all |

`<job>` is a saved job's name, its directory, or any `config.json`.

`--rate` sets the price per hour for the cost lines. Left off, a job is
priced at the card it was planned for, and everything else at DigitalOcean's
on-demand single L40S.

### Two properties worth knowing

**A dropped connection does not kill the solve.** It runs under `tmux` with
its output on a file, and `run` only *follows* that file. Ctrl-C detaches;
closing the laptop detaches; `attach` picks it back up and replays the log so
the bar lands where the run really is. This matters more than it sounds: a
multi-hour solve has no checkpoint, and losing one to a network hiccup is
losing the whole rental.

**It pushes the working tree, not a git checkout.** `main` here is normally
ahead of `origin`, so cloning on the box would quietly solve with different
code than the one being tested.

### What crosses the wire

Up: the repo minus history and derived files, plus `drivetrain_fit.toml`,
`field.json` and `targets.json` — kilobytes.

Down: the tables. This is the real friction, and it is not a cloud problem,
it is a home-connection problem. 13.1 GB for the three-target 08-30 grid,
**86 GB** at full resolution — call it two hours on a 100 Mbit link.

Most of that no longer costs rental time: each table is fetched as the box
finishes it, so only the last one is downloaded on the clock. See section 7.
You still need somewhere to put them, and `--stream-to` exists for when that
is not the local disk.

---

## 4. Is DigitalOcean the right place?

For a first attempt, yes, and for an unexciting reason: a GPU Droplet is a
plain Ubuntu VM with root SSH and a big local NVMe, which is exactly the
shape this workload wants. The out-of-core driver needs a real filesystem it
can `seek` and `unsafe_write` through at speed, and a 500 GB boot disk that
comes with the machine is simpler than any container-plus-volume arrangement.

It is not the cheapest. RunPod lists the same L40S at roughly half
DigitalOcean's rate, and Vast.ai's marketplace is cheaper still. On a $10 run
that difference is $5; on a full-resolution run it is real money. Two things
to weigh against it:

- **Interruption.** There is no resume. A preemptible or community-hosted
  instance that disappears eight hours into a solve costs the whole eight
  hours. Pay for on-demand, whoever you buy it from.
- **Disk shape.** Anything that gives you a small container disk and charges
  separately for a volume needs checking against the store size above.

Nothing in `peregrine_remote.py` is DigitalOcean-specific. It wants an Ubuntu
box, an NVIDIA GPU and your SSH key. Switching provider is `host <new-ip>`
and `provision`.

---

## 5. Planning for a card you do not own

`plan` decides in-core against tiled from a VRAM budget and costs the run from
a cell rate, so planning for a rented card is just handing it that card's two
numbers. The wizard's step 3 now asks which card to plan for, prices the run
at that card's hourly rate, and offers to upload and solve there instead of
here. The override never reaches a local solve -- it is stripped before
`_stream_solve`, because a `vram_budget_bytes` belonging to another machine is
exactly the kind of stale setting that routes a run into a driver it cannot
allocate for.

Measure a box before trusting any of it:

```bash
py -3.12 solver/cloud/peregrine_remote.py benchmark
```

That runs `solver/cloud/benchmark.jl` on the droplet -- five real in-core
solves at different lookahead horizons for the cost pair, one more at
production size to level it, a real file for the disk rate, and the same
tiled rounds both ways for what the prefetch buys -- and writes the answers
where the wizard's planner reads them. Takes a few minutes.

**This matters more than it sounds**, and for a reason that is easy to get
backwards. It is not only that an unmeasured card is quoted at the desktop's
speed. It is that a card measured the *old* way was quoted at one number for
every workload, and the workload moves the answer by more than the card does:
2.2x between the settings the old benchmark used and the settings production
runs. A box whose `gpu_rates` entry predates the cost model shows as
`old-style -- re-benchmark` in the wizard's card list, and the plan says so
too. Re-run it; it is a few minutes against an hour of misjudged rental.

## 6. The prefetch

The tiled driver can read the next tile while the GPU sweeps this one
(`prefetch: true|false|"auto"`). It costs one tile window of host RAM and
about 6% more rounds -- tile k+1 is read before tile k is written back, so
its halo can be one round stale, which is sound (V only decreases, so a stale
read is a looser bound, never a tighter one) but converges slightly slower.

What it buys is bounded by the reads, so **it is worth most exactly where the
card is fastest**. On a slow card the reads were already free.

The host buffer is the same size as the VRAM budget, so a machine whose RAM is
not comfortably larger than its card cannot prefetch a full-budget tile. That
is not the blocker it looks like -- shrinking the tile to afford the prefetch
is usually the better run. Full resolution on an L40S, at the measured desktop
cell rate:

| budget | amplification | prefetch | estimate | cost |
|---|---|---|---|---|
| 40 GB | 3.63x | off | 109.0 h | $171 |
| 30 GB | 4.84x | **on** | **88.9 h** | **$140** |
| 24 GB | 6.76x | on | 88.9 h | $140 |

An 18% win, and it saturates: at 30 GB the reads are already entirely hidden,
so the run sits on its compute floor and a smaller tile buys nothing more.
`plan` prices this -- set `vram_budget_bytes` and re-plan.

## 7. Tables come home as they are made

`run` and `attach` fetch each target's table the moment the box finishes it,
not at the end. A target's chunks are immutable from the instant `target_done`
is emitted -- the solver sends it only after `write_table` has returned and
hashed them -- so the download of one target overlaps the compute of the next
and costs **no rental time at all**. On a three-target run that hides two
thirds of the transfer behind work you are already paying for.

At full resolution that is the difference between 86 GB downloaded on the
clock and 29 GB downloaded on the clock. The run reports what it saved:

```
  fetched score_left (28.6 GB in 41m 12s, 11.8 MB/s) -- while the box keeps solving
  ...
  57.2 GB already home -- 2 of 3 tables arrived while the box was still solving
  that is 1h 22m of transfer that cost no rental time (~$4.63)
```

The final sweep then collects only the remainder -- manifest, model, log, and
any target whose stream failed -- by excluding the streamed targets by
pattern, so nothing is fetched twice and a failed transfer is retried rather
than lost.

`--stream-to DIR` puts them somewhere other than the config's `out_dir`. Use
it when the tables will not fit on the local disk: point it at the SD card and
they land in `<DIR>/TABLES/`. This is purely additive -- it creates and writes
files and never wipes anything, so preparing and verifying the card stays with
`wizard/sdcard.py` and its guard, where the only destructive code in this
project lives.

The destination's free space is checked against the run's total on the
`setup` event, seconds into the solve, and warns rather than stopping: the box
is already working by then, and killing a run over a disk you are about to
clear is the more expensive mistake.

`--no-stream` restores the old fetch-everything-at-the-end behaviour.

## 8. Saved jobs, and why the host is typed every time

Added 2026-09-04. Planning and renting run on different clocks, and the
tooling used to pretend they were the same one.

A plan is settled slowly: re-plan at 4 cm, look at the estimate, try 5, look
at what the lookahead costs, coarsen, re-plan again. That is an evening. A
box is the opposite -- created when the compute is wanted, destroyed the
moment the tables land, because **it bills until it is destroyed and not a
second less**. Wiring "solve on the rented card" into the end of the wizard's
step 3 meant either renting the box through the whole deliberation, or
re-doing the deliberation when the box was up.

So step 3 offers a fourth thing to do with a plan: save it.

```
   1. Solve on the rented NVIDIA H200 141 GB (uploads and runs there -- needs the box up now)
   2. Save this plan as a job to run later (no droplet needed now)
   3. Solve on this machine instead
   4. Change settings
```

A job is a directory under `<workspace>/jobs/<name>/`:

```
job.json             what was planned, and for which card
config.json          the solver config, inputs named relatively
drivetrain_fit.toml  \
field.json            > copies, frozen at save time
targets.json         /
tables/              where the tables come home to
```

**The inputs are copies, not references.** A job that pointed back at
`<workspace>/field/field.json` would read the field as it is on the day it is
*run*, so moving a target between planning and renting would quietly solve a
different problem than the one that was approved -- and `job.json`'s record of
the plan would be a record of nothing. The three files are kilobytes. Copying
them makes the job a fact, and makes the directory portable: it can be moved,
copied to another machine, or kept after the workspace moves on.

Then, later, with a droplet up for as long as it takes:

```bash
py -3.12 solver/cloud/peregrine_remote.py jobs
py -3.12 solver/cloud/peregrine_remote.py provision root@<ip>
py -3.12 solver/cloud/peregrine_remote.py run overnight --host root@<ip>
doctl compute droplet delete <name>
```

`run` re-plans on the box, and says so when the box disagrees with what was
saved -- a bigger VRAM than the catalogue assumed can move the run from tiled
to in core, which changes the driver, the lookahead and the estimate at once.
That is good news, but it is news. The box's own plan is the one that runs.

### `--host` is not remembered, on purpose

`host <ip>` still exists for a box you are going to use all afternoon. But
`--host` on the command is the normal way, because **naming a different box
drops everything remembered about the last one**, and that is a correctness
rule rather than tidiness:

- `rented_since` is the clock every cost line in this script is figured from.
  A droplet destroyed on Monday leaves it behind, and Thursday's box, alive
  for ninety seconds, then reports three days of billing. That is not
  hypothetical -- it is what `remote.json` here did, and the reason this
  section exists.
- The `run` record that `attach` and `pull` resume from names a remote
  directory *on that box*. On a new one it points at nothing.

Neither has any meaning once the address changes, so neither survives it.
`forget` covers the one case this cannot detect: the next box coming up on
the same IP, where nothing in the state can tell that apart from the old box
still being there.

The rental clock counts from **when this desktop first named the box**, which
is later than the box was created if it sat idle before you got to it. It is
a floor on the bill, not the bill; the provider's console is the bill.
