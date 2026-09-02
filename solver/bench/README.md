# Benchmarks for the value solver

Two questions have to be answered together, because either one on its own is
easy to win by cheating: **how fast is a sweep**, and **how good is the table
it produces**. A scheme that skips work looks fast until you drive by its
output; a scheme that produces smaller numbers looks accurate until you check
whether the robot can actually deliver them.

```bash
julia --project=solver solver/bench/compare.jl <config.json> \
      --grid 21,21,12,9,9,9 --starts 120 --schemes base_l4,opt
```

Schemes are declared at the top of `compare.jl` as overrides on a real config
file, so every one of them solves the same grid, the same occupancy and the
same seed cells. `base_*` reproduce the original scheme — whole lattice every
sweep, one step length, Euler integration, fixed collision probes — and are
the reference the optimised schemes have to beat on both axes.

## The two accuracy measures

**Mean value over the cells every scheme reached.** This is the sharp one,
and it needs no reference table. Every candidate control a scheme evaluates
is a real Bellman evaluation rather than an approximation of one, so every
scheme's `V` is an upper bound on the same true value function — and two
valid upper bounds compare directly: the lower one is tighter, therefore more
accurate. Restricting to the commonly-reached cells stops a scheme from
scoring well by quietly leaving the expensive cells unsolved.

**Closed-loop rollout.** Mean value can only be trusted while the upper-bound
argument holds, and a policy that actually drives the robot to the target is
what tests it. Every table is scored by the same controller: minimise
`T + V(state after holding a command for T)` over a fine control set, apply
the winner for a much shorter step, repeat, and integrate finely throughout.

The control horizon `T` is 0.15 s and the applied step is 0.02 s, and those
must not be the same number. A minimum-time value function satisfies
`V(s) − V(s′) = T` along an optimal path, so the value earned by looking
ahead is proportional to `T` while the state, from a standstill, only moves
as `T²`. With `T` = 0.02 s the robot parks a few centimetres short of every
target and stops: no command moves it far enough for `V` to fall by the step
cost, so doing nothing wins. That is a property of greedy control, not of the
table, and the robot side needs to know it too — see §9 of
`docs/TABLE_FORMAT.md`.

## Diagnosing one rollout

```bash
julia --project=solver solver/bench/debug_rollout.jl <config.json> 21,21,12,9,9,9
```

Prints the value at each start state and traces a single rollout step by
step. Worth reaching for whenever the arrival rate looks wrong: a harness
that silently never arrives makes every scheme look identical. This is what
caught the velocity-envelope clamp — the trace ran away to 432 cm/s, well
past the ±150 cm/s grid, and then to NaN.

## The self-test is the other half

`julia --project=solver solver/solve.jl --self-test` checks the properties
these benchmarks assume: that the solver recovers a minimum time known in
closed form, that the integrator is the order it claims, that the fast
control search is never worse than an exhaustive scan, and that widening the
step-length ladder only ever lowers the value. Run it before trusting a
benchmark result — a fast scheme that fails test 3 or 7 is not fast, it is
wrong.

## Out-of-core runs are not benchmarked here, and that is deliberate

Tiling changes the order cells are updated in and nothing else — the same
`cell_update`, over the same grid, against the same model. So the question it
raises is not "is it faster" but "does it get the same answer", and that is
tested rather than benchmarked: `--self-test` [10] solves one grid both ways
and compares.

**Comparing them needs a yardstick, and zero is the wrong one.** The sweep
updates `V` in place and lets the resulting races run — they are benign,
every write only lowers a cell — so two runs of the *same* in-core solver do
not agree exactly either. Measured on the self-test grid:

| compared | mean \|dV\| | p99 |
| --- | --- | --- |
| in-core vs in-core, cold | 0.0092 s | 0.088 s |
| in-core vs **tiled**, cold | 0.0115 s | 0.146 s |
| in-core vs in-core, **warm start** | 0.041 s | 0.756 s |
| in-core vs **tiled**, warm start | 0.051 s | 0.911 s |
| in-core, float policy vs **byte** policy | 0.044 s | 0.799 s |

Three things to read off it. Tiling costs about as much as running the solver
twice. **Warm starting costs four times more than tiling does** — refining
the same incumbent every sweep couples each sweep to the last, so the races
that a cold run averages out are carried forward instead. And quantising the
warm-start policy to a byte a component costs nothing measurable on top,
which is what makes the out-of-core policy store affordable.

The trap here is the same shape as the rollout trap above: score a warm run
against a cold reference and the operator difference lands on the tiling's
bill. That failure looked exactly like a tiling bug for an afternoon.

**What the halo is worth** is measured the same way, by taking it away. On
that grid, with the tiling pinned, dropping the halo to nothing raises the
mean value over the reached cells from 1.80 s to 14.33 s. With the tiling
allowed to shift between rounds it comes back to 1.79 s — the shift heals a
missing halo, because a cell that lost a step this round is mid-tile the
next and `V` only ever decreases. That is a safety net and not a licence: it
works while the reach is small next to the tile, which is not the case on a
full-scale grid.
