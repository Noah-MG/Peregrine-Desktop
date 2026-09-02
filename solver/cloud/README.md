# Solving on a rented GPU

The desktop's 8 GB card is the binding constraint on how big a value table
this project can produce, and it binds twice: once on the grid, and again on
the **lookahead**, because a grid that has to be tiled pays for `tau_max`
through the halo. Renting a bigger card for a few hours relaxes both.

This directory is the whole apparatus: `provision.sh` runs on the rented box,
`peregrine_remote.py` runs here.

```bash
py -3.12 solver/cloud/peregrine_remote.py provision root@<ip>
py -3.12 solver/cloud/peregrine_remote.py plan D:\PeregrineWorkspace\3\runs\<stamp>\config.json
py -3.12 solver/cloud/peregrine_remote.py run  D:\PeregrineWorkspace\3\runs\<stamp>\config.json
```

`run` pushes the working tree and the inputs, starts the solve under `tmux`,
draws the same progress bar the wizard does, pulls the tables back, verifies
them against `docs/TABLE_FORMAT.md`, and prints the bill. The host is
remembered, so only the first command needs it.

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

### Read those estimates as ceilings, twice over

1. They assume the full `iterations` budget. `tolerance` normally stops a
   target well short — on the 35.8M-cell grid it cut 612 s of budget to a
   measured 171 s.
2. They assume `cell_rate` 23.4e6, and **that constant is stale**. The same
   desktop card measures 51-70M cell-updates/s across three grid shapes
   (`benchmark.jl`), so it is not a shape artefact — it is roughly 2.5x
   pessimistic, and every hour and every dollar in the tables above is
   inflated by about that much. Divide them by the ratio `benchmark.jl`
   reports for whatever card you are actually planning for. Section 5 has
   the same numbers re-run at the measured rate.

So treat 33.3 h as "not more than 33.3 h, probably a good deal less", get the
real number from the first run — `run` prints the measured cell rate when it
finishes — and put it in the config as `cell_rate` before planning the next
one.

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
  in-core ceiling is about **6.1e9 cells on an L40S** and **1.0e10 on an
  H100**, an 11 GB and a 19 GB `u16` table respectively.

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

| | |
|---|---|
| `provision <host> [--driver] [--quick]` | Julia, the project, CUDA check, solver self-test, disk-rate measurement |
| `plan <config.json>` | what the box would do with this grid, and what it would cost |
| `run <config.json>` | push, solve, pull, verify, bill |
| `attach` | re-follow a solve after a dropped connection |
| `status` | what the box is, whether a solve is running, disk left |
| `pull` | fetch the last solve's tables again |
| `cost` | what this rental has run up |
| `--self-test` | check this script with no box at all |

`--rate` sets the price per hour for the cost lines; the default is
DigitalOcean's on-demand single L40S.

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
**86 GB** at full resolution — call it two hours on a 100 Mbit link. Worth
planning for before renting, not after.

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

That runs `solver/cloud/benchmark.jl` on the droplet -- a real in-core solve
for the cell rate, a real file for the disk rate, and the same tiled rounds
both ways for what the prefetch buys -- and writes the answers where the
wizard's planner reads them. Takes a couple of minutes.

**This matters more than it sounds.** The `cell_rate` default of 23.4e6 is
stale: the same desktop card now measures **51-70M cell-updates/s** across
three grid shapes, so every wall clock and every dollar the planner has been
quoting is roughly 2.5x too high.

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
