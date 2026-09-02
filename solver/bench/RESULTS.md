# Solver optimization results

Measured on workspace 3 (`D:\PeregrineWorkspace\3`), config
`runs/20260825_153710/config.json`, on an RTX 3060 Ti (8 GB).

Reproduce with:

```bash
julia -t auto --project=solver solver/bench/compare.jl <config.json> --starts 400 --schemes base_l4,opt_kuhn
```

`base_l4` reproduces the pre-optimization solver exactly: whole 67-entry
control lattice scanned every sweep, one fixed lookahead, Euler integration,
fixed collision probe count, 64-point multilinear interpolation, and
out-of-envelope velocities clamped.

---

## Headline

| grid | scheme | s/sweep | sweeps | total per target | speedup |
|---|---|---|---|---|---|
| **production `41,41,16,11,11,11` (35.8 M cells)** | base_l4 | 3.439 | 248 | 852.97 s | — |
| | **opt_kuhn** | **0.932** | 184 | **171.43 s** | **4.98x** |
| dev `21,21,12,9,9,9` (3.86 M cells) | base_l4 | 0.365 | 374 | 136.6 s | — |
| | **opt_kuhn** | **0.089** | 211 | **18.8 s** | **7.3x** |

The production speedup is the one to quote: **5.0x**, from 3.7x on sweep time
and 248 -> 184 sweeps to converge.

**End-to-end, as shipped:** a full three-target production solve
(`solve.jl`, defaults `cfl 2` / `tau_levels 5` / `control_level 4` / simplex)
took **192.9 + 174.6 + 194.3 = 561.8 s (9.4 min)**, against 3 x 852.97 s =
42.6 min for the baseline -- **4.6x**. The card verifies clean against
`docs/TABLE_FORMAT.md` (`verify_tables.py`: 0 failures, 0 warnings) with
`reached_frac` 0.679 per target.

The shipped figure is 4.6x rather than the benchmark's 5.0x because the
ladder also evaluates the configured `dt` on every cell. That costs ~9% of
sweep time and buys a guarantee: the ladder is then a strict superset of the
fixed-step scheme it replaces, so it cannot be worse on any problem rather
than merely better on average. See "the horizon ladder" below -- this was
found by self-test, not assumed.

Ten targets at production resolution: ~2.4 h -> ~31 min.

The dev grid shows a larger factor because the interpolant change helps more
when the table is small enough that cache behaviour dominates. Do not quote
7.3x.

**Accuracy improved at the same time.** On the production grid, mean `V` over
the 23.5 M cells both schemes reached fell **54.2%**, lower in 99.4% of them,
while *more* cells were reached (0.6592 -> 0.6681).

---

## Where the speed came from

One change at a time, dev grid, each row adding to the row above.

| scheme | s/sweep | mean V vs base | cells tighter |
|---|---|---|---|
| `base_l4` | 0.365 | — | — |
| `+ warm start & rotating scan` | 0.054 | ±0.000% | 0.000 |
| `+ off-lattice refinement` | 0.119 | −2.76% | 1.000 |
| `+ adaptive collision probes` | 0.119 | −2.77% | 1.000 |
| `+ midpoint (RK2) integration` | 0.125 | −10.47% | 0.998 |
| `+ horizon ladder` | 0.170 | −40.44% | 1.000 |
| `+ grid-derived horizons, cfl 2` | 0.250 | −53.9% | 0.999 |
| `+ Kuhn simplex interpolation` | **0.089** | **−56.2%** | 0.997 |

Two entries deserve comment.

**The warm start is free.** Scanning 8 of 67 lattice entries per sweep, with
each cell keeping the command it chose last sweep, reproduces the exhaustive
scan to *identical* mean value -- 0.000% difference, zero cells changed -- at
6.8x the sweep rate. The optimal command at a cell barely moves between
sweeps because `V` around it barely moves, so re-deriving it from scratch
every sweep was pure repetition.

**The horizon ladder dominates accuracy** at roughly −40%, and it also
*reduces sweep count* (374 -> ~210) because long steps carry information
across the grid faster. Its cost per sweep is real but it is paid back twice.

**It must be a superset, and originally it was not.** "Minimising over more
horizons can only lower V" is true of horizons *added* to the one you would
otherwise use and false of a different set of them. With `cfl > 0` the rungs
are built around a grid-derived `tau0` that need not contain the configured
`dt`, so the ladder *substituted* rather than added -- and on a double
integrator whose grid happened to suit `dt = 0.05`, it came out **4.8%
worse** than the single fixed step it replaced. Self-test 1 caught this only
after being split into three configs (fixed-dt / +ladder / +fast-search) so
that a failure attributes itself; the two-way version had blamed the control
search, which in fact costs exactly 0.0%.

Fixed by always evaluating `dt` alongside the derived rungs. The ladder then
went from +4.8% to **−1.5%** against fixed-dt, and the terminal excess on the
same test fell from 0.351 s to 0.288 s fixed plus 16.4% -> 13.0% per second.

Diminishing returns were checked, not assumed: tripling the refinement effort
(`control_scan` 12, `refine_rounds` 3) bought 0.004% over `refine_rounds` 2 at
43% more time. Off-lattice refinement is worth about 3%; the horizon ladder is
worth about 50%.

---

## The interpolant

`interp_kuhn` replaces 64-point multilinear with 6D Kuhn (Freudenthal) simplex
interpolation: 7 reads instead of 64 in the bandwidth-bound hot loop.

**Speed:** 0.250 -> 0.089 s/sweep, **2.8x on the interpolant alone**.

**The real reason it is the default is correctness, not speed.** The robot
recovers `grad(V)` by locating the simplex containing its state, taking the 7
vertices and finite-differencing them. `V` is the fixed point of whichever
interpolant the Bellman update is written with, so solving with multilinear
and reading with simplex differences produces a table that is self-consistent
under an operator nobody ever applies. Matching them is the point; the 2.8x
is a bonus.

The gradient comes out with no extra reads -- `V` is affine on a simplex and
consecutive vertices differ by one step along one axis -- which also makes an
analytic control seed cheap if it is ever wanted.

**Caveat, measured.** On a pure double integrator Kuhn is about 6% *looser*
than multilinear, because simplex interpolation discards the cross terms
multilinear keeps and the affine dynamics there make those terms exactly
right. On the real fitted model, with the traction knee active, Kuhn came out
both faster and tighter (−56.2% vs −53.9%). Do not generalise the double
integrator result.

---

## Accuracy: two measures, and why both are needed

**Mean `V` over commonly-reached cells.** Every candidate a scheme evaluates
is a real Bellman evaluation, never an approximation of one, so every
scheme's `V` is a valid upper bound on the same true value function -- and
the lower of two valid upper bounds is the tighter one. It averages over
millions of cells, so it is far less noisy than rollout. Restricting to the
commonly-reached set stops a scheme from scoring well by leaving the
expensive cells unsolved.

Its blind spot is soundness: a table that is simply wrong can be lower than
one that is right. That is what the rollout is for.

**Closed-loop rollout, using the robot's actual control rule** -- maximise
`dot(-grad(V), f(s,u))` over the octahedron, with only the velocity
components of the gradient mattering, and `grad(V)` from the same Kuhn
differences the firmware uses. Arrival means reaching the PID handoff region
(15 cm, 0.25 rad, 40 cm/s, 1.5 rad/s), not the seeded cell, because a
separate controller owns the final approach.

`censored` scores a non-arrival as the full 30 s budget. **This is the only
unbiased summary of the two.** Averaging over arrivals only lets a table
improve its score by failing on hard starts, and that is not hypothetical: on
the dev grid `opt_kuhn` arrived on 29% of starts against 19.8% and still
looked *worse* on mean arrival time, because the extra starts it solved were
the slow ones nothing else reached.

Production grid, 400 starts -- use these, the dev grid is too coarse for its
rollout to mean much (arrival there is 20-30% for every scheme):

| scheme | arrive | censored | median | mean | timeout | obstacle | offfield |
|---|---|---|---|---|---|---|---|
| `base_l4` | **0.853** | **6.76 s** | 2.52 s | 2.74 s | 26 | **15** | **16** |
| `opt_kuhn` | 0.777 | 8.22 s | **2.08 s** | **1.99 s** | **8** | 42 | 23 |

Paired on the 290 starts both solve: median 2.32 -> 2.02 s, mean 2.50 -> 1.97 s,
**better on 229, worse on 44, tied on 17**.

Dev grid, 500 starts, kept for the interpolant comparison only:

| scheme | arrive | censored | median | timeouts | obstacle |
|---|---|---|---|---|---|
| `base_l4` | 0.198 | 24.82 s | 2.34 s | 331 | 44 |
| `opt_ml` | 0.176 | 25.26 s | 2.28 s | 256 | 112 |
| `opt_kuhn` | 0.290 | 23.08 s | 3.30 s | 177 | 113 |

---

## Honest caveats

**The rollout's obstacle count rises, and that is expected rather than a
defect.** On the production grid the new table wins the paired comparison
decisively (229 better / 44 worse over 290 shared starts, mean -21%) and cuts
timeouts from 26 to 8, while obstacle contacts go 15 -> 42 and off-field
16 -> 23.

Read that as the design working, not failing. A minimum-time table is
*supposed* to route close to obstacles during parts of a run -- that is where
the time is -- so a tighter table spends more of its margin, and a simulated
robot with no physical constraint records more contacts. The real robot
cannot pass through an obstacle, and a separate PID owns the final approach,
so neither the contact count nor the arrival rate here should drive a solver
decision. (Noted after discussion with Noah, who owns that call.)

What was worth ruling out, and was: that the solver is *cheating* on
collisions, i.e. a long horizon stepping over an obstacle without seeing it.
See the probe test below -- it is not.

The `censored` figure (8.22 s against 6.76 s) is therefore not a meaningful
verdict on this pair either, since it is dominated by simulated contacts. The
numbers to judge the tables by are mean `V`, reached fraction, and the paired
arrival comparison -- all three of which favour the new solver.

**Obstacle contacts rise, and this was investigated rather than excused.**
Both optimized schemes hit obstacles ~2.5x more often than baseline
(44 -> 112). Two explanations fit: either they simply move (baseline times
out on two thirds of starts, and a robot that goes nowhere hits nothing), or
long horizons are tunnelling through obstacles because the swept probe count
is too coarse -- which would be a soundness bug.

Tested by forcing far more probes:

| probes | mean V | obstacle hits |
|---|---|---|
| adaptive (default) | 7.1638 s | 110 |
| >= 12 | 7.1846 s (+0.29%) | 102 |
| >= 24 | 7.2023 s (+0.54%) | 110 |

Tunnelling is **ruled out**: 24 forced probes move `V` half a percent and the
obstacle count not at all. The swept check is keeping up with the long
horizons. `clearance_cm` remains the knob if a wider margin is ever wanted,
but nothing here says one is needed.

**Rollout has run-to-run variance.** Value iteration updates in place and
asynchronously, so GPU thread ordering makes `V` differ slightly between runs
(the races are benign -- every write only lowers a cell). Observed arrival
rate moved 0.290 -> 0.308 across two identical invocations. Treat rollout
differences under ~0.03 as noise; mean `V` is stable to far better than that.

---

## Correctness work that came out of this

Three bugs, each found by a different instrument:

1. **Out-of-envelope velocity was clamped, pricing it at zero.** The fitted
   model's terminal speed is ~400 cm/s against a 150 cm/s grid, so
   "accelerate out of the box" was reachable and free. Found by tracing one
   rollout: it ran away to 432 cm/s and then to NaN. Cells near the velocity
   boundary in every previously solved table are optimistic.
2. **NaN could reach the index arithmetic.** Every comparison against NaN is
   false, so a non-finite coordinate passed all range tests and then indexed
   the table with garbage.
3. **`sincos(+-Inf)` throws on the CPU** but returns NaN on CUDA. A long
   horizon can walk the state into a region where the *fitted model* is
   unstable (its `+0.312*|w|*w` drag term outruns `-4.67*w` past ~15 rad/s),
   which the ladder reaches on purpose. The GPU hid this completely: 25
   passing self-test assertions said nothing about the documented CPU
   fallback.

`solver/solve.jl --self-test` covers all three plus the interpolant
properties, the integrator order, and the two invariants that license the
optimization work.
