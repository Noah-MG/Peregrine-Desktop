# Orchestration: plan, solve every target, write tables and manifest.
# Included into the PeregrineSolver module.

# The velocity envelope the table covers. These are not measurements of the
# drivetrain -- the fitted model's terminal speed is far above either -- they
# are a statement of how fast the robot is *allowed* to be planned to. Outside
# the box the table says nothing at all (see `interp`), so raising them buys
# reachable states and costs cells, and lowering them is a real speed limit.
const DEFAULT_VMAX = 170.0     # cm/s   -- a quick FTC chassis, with headroom
const DEFAULT_WMAX = 8.0       # rad/s  -- ~1.3 rev/s

"""Emit one machine-readable progress line for the Python wizard."""
function progress(; kw...)
    println("PROGRESS ", JSON3.write(Dict(pairs(kw))))
    flush(stdout)
end

getc(cfg, key, default) = haskey(cfg, key) && cfg[key] !== nothing ? cfg[key] : default

# Default resolution, in the units the axes are actually measured in. These
# reproduce the sample counts the solver shipped with -- 41 x 41 x 16 x 11 x
# 11 x 11 on a 366 cm field -- so they are a coarse starting point rather than
# a recommendation. `plan` computes a recommendation from the field, the
# footprint and a size budget; see `suggest_resolution`.
const DEFAULT_XY_CM      = 8.0
const DEFAULT_HEADING_DEG = 22.5
const DEFAULT_V_CM_S     = 34.0
const DEFAULT_W_RAD_S    = 1.6

"""
Turn a physical resolution into sample counts and the exact span that
realises it.

**The cell size is the input and it is honoured exactly.** Asking for 5 cm
pixels and getting 5.13 cm because the span did not divide by five is the
behaviour this replaces: resolution is a property of the robot and the
geometry it has to resolve, not of how wide the field happens to be. So the
count is rounded up and the *span* moves to suit, never the other way round.

Each axis rounds outward, and each in the way that suits it:

  * `x`, `y` -- `ceil(span / cell) + 1` samples, and the surplus is split
    evenly either side, so the table stays centred on the region the robot
    can actually occupy. The surplus is under one cell and lands outside the
    feasible box, where the wall rule marks it blocked anyway.
  * `h` -- periodic, so the bin has to divide the full turn exactly. The
    requested angle is the closest that does.
  * `vx`, `vy`, `w` -- `2*ceil(limit / cell) + 1` samples, which does two
    things at once: it holds the cell exactly, and it puts **zero on the
    grid**. That matters more than it looks. Every target is a state at rest,
    and rest is where the robot spends the approach; on an even count zero
    falls between two samples and every value near it is an interpolation.
    The envelope rounds up to suit, so `vmax` is a floor, never a ceiling.
"""
function axis_samples(span::Float64, cell::Float64)
    cell > 0 || error("resolution must be positive, got $cell")
    n = max(2, ceil(Int, span / cell - 1.0e-9) + 1)
    (n = n, span = (n - 1) * cell)
end

function symmetric_samples(limit::Float64, cell::Float64)
    cell > 0 || error("resolution must be positive, got $cell")
    half = max(1, ceil(Int, limit / cell - 1.0e-9))
    (n = 2 * half + 1, limit = half * cell)
end

"""The resolution block, with every axis in its own physical unit."""
function grid_resolution(cfg)
    r = getc(cfg, "resolution", Dict())
    (xy_cm = Float64(getc(r, "xy_cm", DEFAULT_XY_CM)),
     heading_deg = Float64(getc(r, "heading_deg", DEFAULT_HEADING_DEG)),
     v_cm_s = Float64(getc(r, "v_cm_s", DEFAULT_V_CM_S)),
     w_rad_s = Float64(getc(r, "w_rad_s", DEFAULT_W_RAD_S)))
end

"""
Build the grid from the field, the robot, the velocity envelope and the
resolution.

The x and y bounds are **computed, not copied from the field file**. A table
cell is a state the robot can hold, and it cannot hold a state whose
footprint is outside the field -- so the span that has to be stored is the
field inset by the footprint plus the wall clearance, at the most permissive
heading. See `feasible_bounds`; on a 366 cm field with a 36 cm chassis that
is 76% of the cells the old full-field span cost, describing exactly the same
set of legal states.

`MANIFEST.JSON` reports the span that is actually stored, which is what the
robot reads, so a table narrower than the field is not something the robot
side has to be told separately. A position between the table edge and the
wall clamps to the edge cell, per section 3 of the format spec, and that edge
cell is the nearest legal state -- which is the right answer to give.

Two escape hatches, both for reproducing an older table rather than for
ordinary use: `grid.n` sets the sample counts directly and the resolution
falls out of the span instead of the other way round, and `grid.bounds`
overrides the inset calculation with `[x_min, y_min, x_max, y_max]`.
"""
function build_grid(cfg, bounds::NTuple{4,Float64},
                    robot::Union{Nothing,Matrix{Float64}} = nothing,
                    wall_clearance::Real = 0.0, hsub::Integer = 3)
    vmax = Float64(getc(cfg, "vmax", DEFAULT_VMAX))
    wmax = Float64(getc(cfg, "wmax", DEFAULT_WMAX))
    res = grid_resolution(cfg)
    nfix = getc(cfg, "n", nothing)

    # The heading count is needed before the box, because the box is a union
    # over heading bins and the bins are what the count defines.
    nh = nfix !== nothing ? Int(nfix[3]) :
         max(4, round(Int, 360.0 / res.heading_deg))

    ov = getc(cfg, "bounds", nothing)
    fb = ov !== nothing ?
        (xlo = Float64(ov[1]), ylo = Float64(ov[2]),
         xhi = Float64(ov[3]), yhi = Float64(ov[4]), headings = nh) :
        feasible_bounds(bounds, robot, wall_clearance, nh, hsub)

    if nfix !== nothing
        n = NTuple{6,Int}(Int.(nfix))
        lo = (fb.xlo, fb.ylo, -π, -vmax, -vmax, -wmax)
        hi = (fb.xhi, fb.yhi,  π,  vmax,  vmax,  wmax)
        return Grid6(n, lo, hi)
    end

    ax = axis_samples(fb.xhi - fb.xlo, res.xy_cm)
    ay = axis_samples(fb.yhi - fb.ylo, res.xy_cm)
    av = symmetric_samples(vmax, res.v_cm_s)
    aw = symmetric_samples(wmax, res.w_rad_s)
    # Surplus split evenly, so the stored span stays centred on the feasible
    # one rather than growing off one side.
    padx = (ax.span - (fb.xhi - fb.xlo)) / 2
    pady = (ay.span - (fb.yhi - fb.ylo)) / 2

    n = (ax.n, ay.n, nh, av.n, av.n, aw.n)
    lo = (fb.xlo - padx, fb.ylo - pady, -π, -av.limit, -av.limit, -aw.limit)
    hi = (fb.xhi + padx, fb.yhi + pady,  π,  av.limit,  av.limit,  aw.limit)
    Grid6(n, lo, hi)
end

"""
Everything `build_grid` needs that does not come out of the `grid` block.

`plan` and `run_solve` both build the same grid and have to agree on it to
the last bit -- a plan that sized the table from one inset and a solve that
used another would disagree about how many cells there are. One reader, used
by both.
"""
function grid_inputs(cfg::AbstractDict, bounds::NTuple{4,Float64},
                     robot::Union{Nothing,Matrix{Float64}})
    clearance = Float64(getc(cfg, "clearance_cm", 5.0))
    (bounds = bounds, robot = robot,
     clearance = clearance,
     # The wall is an obstacle, so by default it holds the obstacle gap. It is
     # separately settable because the two are not always the same thing in
     # practice: a perimeter wall is a surface a chassis may legitimately run
     # close to, while a scoring structure usually is not.
     wall_clearance = Float64(getc(cfg, "wall_clearance_cm", clearance)),
     hsub = Int(getc(cfg, "heading_substeps", 3)))
end

build_grid(cfg, gi::NamedTuple) =
    build_grid(cfg, gi.bounds, gi.robot, gi.wall_clearance, gi.hsub)

"""
How well the grid's position resolution matches its velocity resolution.

A Bellman backup only learns from a step that leaves the cell it started in.
For a drivetrain that can pull `a`, the distance covered while the speed
changes by one velocity cell is `dv^2 / (2a)`. Compare that with the position
cell:

  * far below 1 -- resolving the velocity axis leaves position almost
    stationary, so near-stationary states are heavily diffused and their
    values come out pessimistic. This is the normal case near a target, and
    the step-length ladder is what covers it.
  * far above 1 -- the position axis is finer than the dynamics can use, and
    the cells are being paid for without buying accuracy.

Reported rather than enforced: the right resolution is a memory decision as
much as an accuracy one, and the number is only meaningful next to how much
time the robot actually spends near rest.
"""
function grid_balance(g::Grid6, m::Model)
    ax = maximum(abs.(m.B[1:3])); ay = maximum(abs.(m.B[4:6]))
    a = max(ax, ay)
    a <= 0 && return (ratio = NaN, dv_distance_cm = NaN, accel_cm_s2 = 0.0)
    dv = Float64(min(g.step[4], g.step[5]))
    d = dv * dv / (2 * Float64(a))
    (ratio = d / Float64(min(g.step[1], g.step[2])),
     dv_distance_cm = d, accel_cm_s2 = Float64(a))
end

"""
Every solver knob, read once.

`plan` and `run_solve` both need these, and they must agree exactly: the plan
report includes the dependency halo, and the halo is a function of the
horizon ladder, the CFL number and the control set. A plan computed from a
second reading of the config is a plan that can disagree with the run it is
supposed to describe, and the way that failure shows up -- a halo one cell
too small -- is silent.
"""
function solver_params(cfg::AbstractDict, g::Grid6)
    dt = Float32(getc(cfg, "dt", 0.05))
    nsub = Int(getc(cfg, "substeps", 4))
    checks = Int(getc(cfg, "sweep_checks", 3))
    nearest = Bool(getc(cfg, "nearest", false))
    iters = Int(getc(cfg, "iterations", 400))
    tol = Float64(getc(cfg, "tolerance", 1e-3))
    # The lattice now only seeds the pattern search, and each sweep looks at
    # `control_scan` entries of it rather than all of them, so a denser
    # lattice costs sweeps-to-converge rather than time-per-sweep. That makes
    # level 4 affordable where it used to be the expensive option. Corners
    # alone (level 1) remain a bad idea: they made every cell pessimistic by
    # ~1.8 s on a real model, because a plain diagonal was not in the menu.
    level = Int(getc(cfg, "control_level", 4))
    margin = Float64(getc(cfg, "margin_cm", 0.0))
    # Safety gap held around every obstacle, applied before the robot's own
    # footprint is swept in.
    clearance = Float64(getc(cfg, "clearance_cm", 5.0))
    # Half a cell, so the seed is normally the single nearest cell. The online
    # optimizer owns the real arrival test; this only has to plant the seed.
    ttol = NTuple{6,Float32}(Float32.(getc(cfg, "target_tol",
              [g.step[k] * 0.5 for k in 1:6])))
    # Unreached cells hold this finite value rather than Inf; see `interp`.
    # It also bounds what the table can express, so keep it inside the dtype's
    # range: u16 at 1 ms tops out at 65.5 s.
    cap = Float32(getc(cfg, "value_cap", 60.0))

    # Control search. `control_scan` is how many lattice entries each sweep
    # looks at; 0 means all of them, which is the old behaviour. The lattice
    # is only a seed for `refine_rounds` of pattern search, so a small scan
    # plus refinement beats a big scan on both counts -- see `cell_update`.
    scan = Int(getc(cfg, "control_scan", 8))
    rounds = Int(getc(cfg, "refine_rounds", 2))
    delta0 = Float64(getc(cfg, "refine_delta", 0.35))
    warm = Bool(getc(cfg, "warm_start", true))
    # Lookahead horizon. `cfl` is how many grid cells one backup should
    # advance; the solver derives the seconds per cell from that and from how
    # fast the robot is going, so `dt` is only the fallback for cfl = 0.
    # `tau_levels` is how many horizons each cell brackets around it -- the
    # accuracy dial the user actually sets.
    # cfl 2 / 5 rungs measured best on both the dev and production grids:
    # it is the ladder's REACH that pays, not its base, and longer steps also
    # carry information across the grid faster, so it converges in fewer
    # sweeps as well (248 -> 184 on the production grid).
    ntau = Int(getc(cfg, "tau_levels", 5))
    tau_ratio = Float64(getc(cfg, "tau_ratio", 2.0))
    cfl = Float64(getc(cfg, "cfl", 2.0))
    tau_min = Float64(getc(cfg, "tau_min", 0.004))
    # `settle` resolves "auto" before this is reached, and is the only thing
    # that should. Tolerating a leftover string here rather than throwing keeps
    # a hand-written config that says "auto" from failing deep inside the
    # parameter reader with a `Float64("auto")`; it lands on the reference,
    # which is what "auto" means everywhere the grid holds whole.
    _tm = getc(cfg, "tau_max", TAU_REF)
    tau_max = _tm isa Real ? Float64(_tm) : TAU_REF
    # Longest integration substep. Tying substeps to this rather than fixing
    # their count keeps a long horizon from also being a less accurate one.
    hmax = Float64(getc(cfg, "substep_max", dt / max(nsub, 1)))
    achecks = Bool(getc(cfg, "adaptive_checks", true))
    rk2 = Bool(getc(cfg, "rk2", true))
    # Clamping out-of-envelope speeds prices leaving the velocity box at
    # zero; see `interp`. Left available only to reproduce old tables.
    vclamp = Bool(getc(cfg, "velocity_clamp", false))
    # Kuhn simplex interpolation: 7 reads per lookup instead of 64, and --
    # the reason it is the default -- the same interpolant the robot uses to
    # recover grad(V) from the finished table. V is the fixed point of
    # whichever interpolant the backup is written with, so solving with one
    # and reading with another gives a table that is self-consistent under an
    # operator nobody applies. `simplex: false` restores multilinear.
    simplex = Bool(getc(cfg, "simplex", true))

    p = Params(dt = dt, nsub = Int32(nsub), checks = Int32(checks),
               adaptive_checks = achecks, nearest = nearest, cap = cap,
               ntau = Int32(ntau), tau_ratio = Float32(tau_ratio),
               cfl = Float32(cfl), tau_min = Float32(tau_min),
               tau_max = Float32(tau_max), hmax = Float32(hmax),
               ncoarse = Int32(scan), rounds = Int32(rounds),
               delta0 = Float32(delta0), rk2 = rk2, vclamp = vclamp,
               simplex = simplex)

    ctl_t = control_set(level)
    nctl = length(ctl_t)
    ctl_h = Float32[getindex.(ctl_t, 1); getindex.(ctl_t, 2); getindex.(ctl_t, 3)]

    (p = p, level = level, nctl = nctl, ctl_h = ctl_h, warm = warm,
     iters = iters, tol = tol, cap = cap, nearest = nearest, dt = dt,
     nsub = nsub, checks = checks, margin = margin, clearance = clearance,
     ttol = ttol, scan = scan, rounds = rounds, delta0 = delta0, ntau = ntau,
     tau_ratio = tau_ratio, cfl = cfl, tau_min = tau_min, tau_max = tau_max,
     hmax = hmax, achecks = achecks, rk2 = rk2, vclamp = vclamp,
     simplex = simplex)
end

"""
Decide how the grid is going to be solved, and on what.

Three outcomes:

  * `:incore` -- the whole grid fits in the device budget. Nothing changes;
    this is the original driver.
  * `:ooc` -- it does not, so the value function lives in a file and the GPU
    sees one tile of it at a time. See `Tiles.jl` and `OutOfCore.jl`.
  * `:cpu` -- no usable device; the caller falls back to host sweeps.

`out_of_core` in the config forces the choice (`true`/`false`), and the
default `"auto"` picks by whether it fits. Forcing it on is how the tiled
driver gets tested against the in-core one on a grid small enough to run
both.

The reach scan is only run when it is going to be used. It costs seconds, and
seconds are worth avoiding in `plan`, which the wizard calls interactively
every time the user nudges a resolution.
"""
function decompose(cfg::AbstractDict, g::Grid6, m::Model, sp)
    want = getc(cfg, "out_of_core", "auto")
    force_on  = want === true || want == "true" || want == "always"
    force_off = want === false || want == "false" || want == "never"

    budget = Int64(getc(cfg, "vram_budget_bytes", 0))
    if budget <= 0
        budget = gpu_budget(headroom = Float64(getc(cfg, "vram_headroom", 0.85)))
    end
    whole, _, _, col = _tile_bytes(g, Int(g.n[1]), Int(g.n[2]), 0, 0, sp.warm)

    if !CUDA.functional() && Int64(getc(cfg, "vram_budget_bytes", 0)) <= 0
        return (mode = :cpu, tp = nothing, budget = budget,
                whole_bytes = whole, col = col)
    end
    if force_off || (!force_on && whole <= budget)
        return (mode = :incore, tp = nothing, budget = budget,
                whole_bytes = whole, col = col)
    end

    nang = Int(getc(cfg, "halo_scan_angles", 16))
    ncmd = Int(getc(cfg, "halo_scan_commands", 128))
    margin = Int(getc(cfg, "halo_margin", 2))
    # `warm_start` is the in-core setting; tiled, the policy is bytes rather
    # than floats and competes with the tile for residency, so it has its own.
    warm = Bool(getc(cfg, "warm_start_tiled", true))

    dxy, dh = reach_extent(g, m, sp.p, sp.ctl_h, sp.nctl;
                           nangle = nang, nsample = ncmd)
    hx, hy = halo_cells(g, dxy; margin = margin)
    tp = plan_tiles(g, budget, hx, hy; warm = warm, dxy_cm = dxy, dh_rad = dh)
    tp !== nothing &&
        return (mode = :ooc, tp = tp, budget = budget, whole_bytes = whole,
                col = col, dxy_cm = dxy, hx = hx, hy = hy, advice = nothing)

    # It does not fit even at one cell of interior. That is nearly always the
    # halo rather than the grid -- the smallest possible tile is a single
    # column wrapped in `2h` columns of border, so the cost goes as `(1+2h)^2`
    # and `h` comes straight from the longest lookahead. Saying "reduce
    # something" here would be useless, so work out which something: search
    # for the longest step that would fit, and report it.
    (mode = :ooc, tp = nothing, budget = budget, whole_bytes = whole,
     col = col, dxy_cm = dxy, hx = hx, hy = hy,
     advice = fit_advice(cfg, g, m, sp, budget, warm, margin, nang, ncmd))
end

"""
When no tiling fits, work out what would.

Two levers, reported with numbers rather than named: the longest lookahead
`tau_max`, which sets the halo, and `warm_start_tiled`, which sets the bytes
per resident cell. Both ends of the first are reported -- the most accurate
step that fits at all, and the first that is actually worth starting -- since
the longest step that squeezes in is usually a bad place to run: it fits with
a single column of interior inside a wall of halo, which is arithmetically a
fit and practically a machine reading the same cells sixty times over.

`tau_options` does the search on one shared reach scan, so the whole ladder
costs about what a single rung used to.
"""
function fit_advice(cfg, g::Grid6, m::Model, sp, budget::Int64, warm::Bool,
                    margin::Int, nang::Int, ncmd::Int)
    col = ncells(g) ÷ (Int64(g.n[1]) * Int64(g.n[2]))
    # The largest halo the budget admits at all, from `(1+2h)^2` columns.
    per = cell_bytes(warm)
    maxcols = budget ÷ (col * per)
    hmax = maxcols <= 0 ? -1 : (isqrt(maxcols) - 1) ÷ 2
    ts = Int(getc(cfg, "tile_sweeps", 4))
    cost, _ = cell_cost(cfg)
    disk = Float64(getc(cfg, "disk_rate", 500e6))

    opts = filter(o -> o.tau < Float64(sp.p.tau_max),
                  tau_options(g, m, sp, budget, warm, margin, nang, ncmd,
                              Float64(per), ts, cost, disk))
    # "Worth starting" is not "lowest amplification". A round costs its
    # compute plus its I/O, and on the class of machine this was tuned on --
    # a desktop card against a SATA SSD -- the two are comparable at around
    # eight cells loaded per cell updated. Past that the reads start to
    # dominate; below it, buying amplification with `tau_max` is paying
    # accuracy for very little time. Measured on a 145x145 grid at the cell
    # size a full-scale run uses: `tau_max` 0.2 costs +6% on mean value, 0.1
    # costs +21%, and 0.06 costs +56%. That is the whole reason this
    # threshold is not tighter.
    #
    # The eight is a fixed rung rather than a crossing computed from the cost
    # pair, so on a much faster card, or a much slower disk, it is the wrong
    # rung -- it is a default for a shape of machine, and the `tau_options`
    # table beside it carries the compute/io split for any other.
    comfy = findfirst(o -> o.amplification <= 8.0, opts)
    (column_bytes = col * per, budget_bytes = budget, max_halo = hmax,
     warm_helps = warm && !warm_fits(g, budget, sp, m, margin, nang, ncmd),
     tau_max = isempty(opts) ? nothing : opts[1],
     tau_comfortable = comfy === nothing ? nothing : opts[comfy])
end

"""Would dropping the warm-start policy, and nothing else, make it fit?"""
function warm_fits(g::Grid6, budget::Int64, sp, m::Model, margin::Int,
                   nang::Int, ncmd::Int)
    dxy, dh = reach_extent(g, m, sp.p, sp.ctl_h, sp.nctl;
                           nangle = nang, nsample = ncmd)
    hx, hy = halo_cells(g, dxy; margin = margin)
    plan_tiles(g, budget, hx, hy; warm = false, dxy_cm = dxy,
               dh_rad = dh) !== nothing
end

"""
Should the tiled driver prefetch, and what does it need to?

The prefetch holds one whole tile window in host RAM so the read of the next
tile can run while the GPU sweeps this one. That buffer is the only cost, and
it is not small -- at full scale it is tens of gigabytes -- so `"auto"` says
yes only when it fits in `prefetch_ram_frac` of what the machine has free.

It also needs a second thread to read on. One thread means the read would run
on the same task that is waiting for the kernel, which is exactly the
serialisation this is trying to remove, so a single-threaded Julia gets the
old path and is told why.

Returns `(on, bytes, why)`; `why` is the reason it is off, or "" when it is on.
"""
function prefetch_plan(cfg::AbstractDict, dec, warm::Bool)
    want = getc(cfg, "prefetch", "auto")
    force_off = want === false || want == "false" || want == "never"
    force_on  = want === true || want == "true" || want == "always"
    dec.mode == :ooc && dec.tp !== nothing || return (false, Int64(0), "not tiled")
    tp = dec.tp
    win = Int64(tp.nxl) * Int64(tp.nyl) * tp.col
    bytes = win * cell_bytes(warm)
    force_off && return (false, bytes, "turned off in the config")
    if Threads.nthreads() < 2
        return (force_on, bytes,
                force_on ? "" : "julia has one thread; start it with -t auto")
    end
    force_on && return (true, bytes, "")
    frac = Float64(getc(cfg, "prefetch_ram_frac", 0.5))
    free = try
        Int64(Sys.free_memory())
    catch
        Int64(0)
    end
    free <= 0 && return (false, bytes, "cannot read free memory")
    bytes <= free * frac && return (true, bytes, "")
    # Naming the lever matters here, because the obvious reading of this
    # message is "buy more RAM" and the actual fix is usually free. The host
    # buffer is one tile window, so it is the same size as the VRAM budget:
    # a machine whose RAM is not comfortably larger than its card cannot
    # prefetch a full-budget tile. Shrinking the budget shrinks both, and a
    # smaller tile with its reads hidden can beat a larger one without --
    # `vram_budget_bytes` is the knob, and re-planning prices the trade.
    want = round(bytes / frac / 2^30, digits = 1)
    (false, bytes,
     "the window buffer is $(round(bytes / 2^30, digits = 1)) GB, which needs " *
     "$(want) GB of RAM and only $(round(free / 2^30, digits = 1)) GB is free; " *
     "lower vram_budget_bytes to shrink the tile, or raise prefetch_ram_frac " *
     "(now $(frac)) if the RAM really is spare")
end

"""
How long the whole thing is likely to take, in seconds.

**A sweep is priced from what it asks of the machine, not from a stored cell
rate.** `sweep_work` counts the lookaheads, integration substeps and swept
probes one cell update actually costs at this grid, this drivetrain and these
solver parameters; `cell_cost` says what those cost in seconds on this card.
The product is the per-cell time, and it varies by more than 3x across the
range of `cfl` and `tau_max` a real config spans -- which is why the earlier
single-rate estimate under-quoted a production H200 run by 41%: the rate had
been measured on the benchmark's short-step case and spent on a long-step one.

`disk_rate` is still a plain measurement: 500 MB/s, a SATA SSD, and it only
enters a tiled run. `cell_cost` defaults to the same desktop card. Both are
replaced by `solver/cloud/benchmark.jl` on any machine worth renting.

**The two passes are priced over different cells.** A solve sweep retires a
blocked cell before its first lookahead, so it only pays for the free ones.
The escape pass is the opposite: it works on the cells the solve could not
reach, and a cell that WAS reached costs it a load and a retire. Measured on
four occupancies spanning 0.5 to 1.0 unreached, an escape sweep costs the
unreached fraction of a full sweep times `ESCAPE_CELL_FACTOR` -- so it is
priced that way, with `blocked_frac` standing in for the unreached share.

`blocked_frac` is a floor on what the escape pass will work on rather than
the true figure: every blocked cell is unreached, and the free states that
are also unreachable are extra. On the H200 job that gap was small -- 31.0%
blocked against 34.7% unreached -- but it is a floor, and it is the only part
of this estimate that is. `ESCAPE_CELL_FACTOR` is taken at the top of its
measured range to lean the other way.

The remaining ceiling is the iteration budget: the full count is assumed
spent, though `tolerance` usually stops a target earlier.
"""
function runtime_estimate(cfg, g::Grid6, dec, iters::Int, ntargets::Int;
                          m::Union{Nothing,Model} = nothing, sp = nothing,
                          blocked_frac::Float64 = 0.0)
    cost, cost_source = cell_cost(cfg)
    disk = Float64(getc(cfg, "disk_rate", 500e6))
    cells = Float64(ncells(g))
    # Without a model and parameters there is nothing to weigh the sweep
    # with, so fall back to the workload the cost pair is anchored to. `plan`
    # always passes both; the fallback is for callers that only have a grid.
    work = (m === nothing || sp === nothing) ? REF_WORK :
           sweep_work(g, m, sp.p, sp.nctl)
    active = clamp(1.0 - blocked_frac, 0.0, 1.0)
    rate = effective_rate(work, cost)
    esc_iters = Bool(getc(cfg, "escape", true)) ?
                Int(getc(cfg, "escape_iterations", 60)) : 0
    # A solve sweep's worth of work that the escape pass does: the share of
    # the grid it backs up, times what one of its backups costs against one of
    # the solve's. See the docstring for the first and `ESCAPE_CELL_FACTOR`
    # for the second.
    esc_active = clamp(blocked_frac, 0.0, 1.0) * ESCAPE_CELL_FACTOR
    solve_iters = iters
    iters += esc_iters
    out = Dict{String,Any}(
        "cell_rate" => rate, "assumes_full_budget" => true,
        "escape_sweeps" => esc_iters, "cost_measured" => cost_source == "cost",
        "cost_source" => cost_source,
        "blocked_frac" => 1.0 - active, "escape_active_frac" => esc_active,
        # What the rate above is a rate *of*, so a surprising estimate can be
        # traced to the workload rather than only to the machine.
        "work" => Dict("lookaheads" => work.lookaheads,
                       "substeps" => work.substeps,
                       "probes" => work.probes),
        "cell_cost" => Dict("per_lookahead_s" => cost.per_lookahead_s,
                            "per_step_s" => cost.per_step_s))
    # Nothing fits, so there is no run to time. Saying "sweep" here would
    # quote the in-core cost of a solve that cannot start.
    dec.mode == :ooc && dec.tp === nothing &&
        return merge!(out, Dict{String,Any}("unit" => "none"))
    if dec.mode != :ooc
        # A blocked cell retires before its first lookahead, so a solve sweep
        # only pays for the free ones; the escape pass pays for exactly the
        # ones the solve skipped.
        per = cells * active / rate
        esc = cells * esc_active / rate
        total = per * solve_iters + esc * esc_iters
        merge!(out, Dict{String,Any}(
            "unit" => "sweep", "units" => iters, "unit_s" => per,
            "escape_unit_s" => esc,
            "per_target_s" => total,
            "total_s" => total * ntargets, "io_bound" => false))
        return out
    end
    ts = Int(getc(cfg, "tile_sweeps", 4))
    # The two passes round up to whole rounds separately, because that is how
    # they run: the escape pass starts its own round count rather than
    # continuing the solve's.
    esc_rounds = esc_iters > 0 ? max(1, cld(esc_iters, ts)) : 0
    rounds = max(1, cld(solve_iters, ts)) + esc_rounds
    out["escape_rounds"] = esc_rounds
    sb = Float64(cell_bytes(Bool(getc(cfg, "warm_start_tiled", true))))
    pf, pfb, pfwhy = prefetch_plan(cfg, dec, Bool(getc(cfg, "warm_start_tiled", true)))
    # Only the compute half of a round takes the blocked-cell discount: the
    # tiled driver loads and stores a blocked cell like any other.
    srate = rate / max(active, 1.0e-6)
    erate = rate / max(esc_active, 1.0e-6)
    r = round_seconds(cells, ts, dec.tp.amplification, sb, srate, disk;
                      prefetch = pf)
    re = round_seconds(cells, ts, dec.tp.amplification, sb, erate, disk;
                       prefetch = pf)
    per_target = r.total * (rounds - esc_rounds) + re.total * esc_rounds
    # What the prefetch is worth on this grid, for the report. Costed against
    # the same round rather than asserted, because it is entirely a function
    # of how the compute and the reads compare, and that flips with the card.
    plain = round_seconds(cells, ts, dec.tp.amplification, sb, srate, disk)
    plain_e = round_seconds(cells, ts, dec.tp.amplification, sb, erate, disk)
    plain_total = plain.total * (rounds - esc_rounds) + plain_e.total * esc_rounds
    merge!(out, Dict{String,Any}(
        "prefetch" => pf, "prefetch_bytes" => pfb,
        "prefetch_saves_s" => (plain_total - per_target) * ntargets,
        "prefetch_off_because" => pfwhy,
        "unit" => "round", "units" => rounds, "unit_s" => r.total,
        "escape_unit_s" => re.total,
        "per_target_s" => per_target,
        "total_s" => per_target * ntargets,
        # Which half is the larger, not which one "limits": a round pays for
        # both, one after the other. See `round_seconds`.
        "io_bound" => r.io > r.compute, "compute_s" => r.compute,
        "io_s" => r.io, "disk_rate" => disk))
    out
end

"""
Settle `tau_max`, then decide how the grid is going to be solved.

**`tau_max` defaults to `"auto"`, and that is the point of this function.** It
is not a preference; it is a consequence. In core it should always be the
reference, because there is no halo for it to pay for. Tiled it should be the
knee of the reach/accuracy trade, which depends on the resolution, the field,
the model and how much VRAM the machine has. A number carried in a config file
cannot track any of that, and a stale one is invisible: a `tau_max` of 0.12
left over from a tiled experiment costs about 17% on mean value forever after,
including on grids that hold whole and have nothing to gain from it.

So it is derived here, on every plan and every solve, from the same code path,
which is the other half of the point -- a plan that reported one lookahead and
a solve that used another would be a table nobody could account for.

Pin it to a number in the config and that number is honoured exactly; `plan`
then says so, and says what it is costing.

Costs two reach scans on the tiled path (one to size the tiles at the
reference, one to re-size them at the pick) plus the shared curve. In core it
costs nothing at all -- whether the grid fits whole is a byte count, not a
scan -- which is the case the interactive loop spends its time in.
"""
function settle(cfg::AbstractDict, g::Grid6, m::Model)
    want = getc(cfg, "tau_max", "auto")
    auto = want isa AbstractString && lowercase(String(want)) == "auto"
    withtau(t) = merge(Dict{String,Any}(String(k) => v for (k, v) in pairs(cfg)),
                       Dict{String,Any}("tau_max" => Float64(t)))

    if !auto
        sp = solver_params(cfg, g)
        return (sp = sp, dec = decompose(cfg, g, m, sp), auto = false,
                rec = nothing)
    end

    cfg0 = withtau(TAU_REF)
    sp0 = solver_params(cfg0, g)
    dec0 = decompose(cfg0, g, m, sp0)
    wt = Bool(getc(cfg, "warm_start_tiled", true))
    rec = recommend_tau(dec0.mode, g, m, sp0, dec0.budget, wt,
                        Int(getc(cfg, "halo_margin", 2)),
                        Int(getc(cfg, "halo_scan_angles", 16)),
                        Int(getc(cfg, "halo_scan_commands", 128)),
                        wt ? 7.0 : 4.0, Int(getc(cfg, "tile_sweeps", 4)),
                        first(cell_cost(cfg)),
                        Float64(getc(cfg, "disk_rate", 500e6)))
    # Nothing to change: in core, or no tiling fits at any step length (in
    # which case the caller reports the failure and the reference is the
    # honest thing to have been trying).
    (rec.tau_max === nothing || rec.tau_max >= TAU_REF) &&
        return (sp = sp0, dec = dec0, auto = true, rec = rec)

    cfg1 = withtau(rec.tau_max)
    sp1 = solver_params(cfg1, g)
    (sp = sp1, dec = decompose(cfg1, g, m, sp1), auto = true, rec = rec)
end

"""
Size and feasibility report, cheap enough to run before committing.

It answers four questions, and the last two are the ones worth having:

  * how big the table is, and whether it fits -- on the card, in VRAM, in the
    workspace;
  * how long the solve will take, and which of compute or disk bounds it;
  * what the settings **should** be: a resolution for the size budget, and a
    `tau_max` with the accuracy it costs, since that one is otherwise a knob
    with two opposed effects and no visible units;
  * whether the targets are states the robot can actually be in. A target off
    the table or hanging through a wall produces an entirely empty table --
    quietly, and at the end of a full-length run.
"""
function plan(cfg::AbstractDict)
    bounds, polys, robot, _ = load_field(String(cfg["field"]))
    names, states, _ = load_targets(String(cfg["targets"]))
    gi = grid_inputs(cfg, bounds, robot)
    g = build_grid(get(cfg, "grid", Dict()), gi)
    mdl, _ = load_model(String(cfg["regression"]))
    # The reach depends on the model the solver will actually integrate, so
    # the constant has to be zeroed here too if the run is going to zero it.
    if Bool(getc(cfg, "zero_c", true))
        mdl = Model(mdl.B, mdl.A, mdl.q, mdl.S, mdl.D, (0.0f0, 0.0f0, 0.0f0),
                    mdl.eps, mdl.knee)
    end
    bal = grid_balance(g, mdl)
    dtype = String(getc(cfg, "dtype", "u16"))
    haskey(DTYPES, dtype) || error("unknown dtype '$dtype'")
    eb = DTYPES[dtype].bytes
    cells = ncells(g)
    per = cells * eb
    chunk_elements = Int(getc(cfg, "chunk_elements", 1 << 23))
    ok, need, free = gpu_fits(g; warm = Bool(getc(cfg, "warm_start", true)))

    st = settle(cfg, g, mdl)
    sp = st.sp; dec = st.dec
    iters = Int(getc(cfg, "iterations", 400))
    ooc_rounds = max(1, cld(iters, Int(getc(cfg, "tile_sweeps", 4))))
    ooc = Dict{String,Any}("mode" => String(dec.mode))
    if dec.mode == :ooc
        tp = dec.tp
        if tp === nothing
            # The smallest tile the driver can hold is one column of interior
            # wrapped in the halo, so it costs `(1 + 2h)^2` columns and it is
            # `h` -- the longest lookahead -- that usually decides. `advice`
            # carries the numbers for what would fit instead.
            a = dec.advice
            ooc["fits"] = false
            ooc["reach_cm"] = dec.dxy_cm
            ooc["reach_cells"] = dec.dxy_cm / Float64(min(g.step[1], g.step[2]))
            ooc["halo_cells"] = [dec.hx, dec.hy]
            ooc["column_bytes"] = a === nothing ? dec.col * 4 : a.column_bytes
            ooc["budget_bytes"] = dec.budget
            if a !== nothing
                ooc["max_halo_cells"] = a.max_halo
                ooc["min_window_cells"] = 1 + 2 * max(dec.hx, dec.hy)
                ooc["warm_start_would_help"] = a.warm_helps
                for (pre, o) in (("suggest", a.tau_max),
                                 ("comfortable", a.tau_comfortable))
                    o === nothing && continue
                    ooc[pre * "_tau_max"] = o.tau
                    ooc[pre * "_halo_cells"] = o.halo
                    ooc[pre * "_reach_cm"] = o.reach_cm
                    ooc[pre * "_tile_cells"] = o.tile
                    ooc[pre * "_amplification"] = o.amplification
                    ooc[pre * "_value_cost"] = o.cost
                    ooc[pre * "_total_s"] =
                        o.round_s * ooc_rounds * length(names)
                end
            end
        else
            # 4 bytes a cell for V, and 3 more for the quantised warm-start
            # policy when it is kept across rounds.
            # Named apart from `eb` above: an `if` block is not a scope in
            # Julia, so reusing that name here would quietly rewrite the
            # element size the manifest and the size report are built from.
            wt = Bool(getc(cfg, "warm_start_tiled", true))
            sb = cell_bytes(wt)
            store = cells * sb
            # Bytes moved between the store and the device in one round: every
            # loaded cell read, every interior cell written back.
            per_round = Int64(round(dec.col * sb *
                (Float64(tp.nxl) * tp.nyl * cld(Int(g.n[1]), tp.wx) *
                 cld(Int(g.n[2]), tp.wy)))) + store
            ooc["fits"] = true
            ooc["halo_cells"] = [tp.hx, tp.hy]
            ooc["tile_cells"] = [tp.wx, tp.wy]
            ooc["loaded_cells"] = [tp.nxl, tp.nyl]
            ooc["tiles_per_round"] = tp.ntiles
            ooc["resident_bytes"] = tp.bytes
            ooc["amplification"] = tp.amplification
            ooc["reach_cm"] = tp.dxy_cm
            ooc["reach_rad"] = tp.dh_rad
            ooc["reach_cells"] = tp.dxy_cm / Float64(min(g.step[1], g.step[2]))
            ooc["store_bytes"] = store
            ooc["io_bytes_per_round"] = per_round
            ooc["tile_sweeps"] = Int(getc(cfg, "tile_sweeps", 4))
        end
    end

    # The occupancy the solver will use, built here so the target check can
    # ask the authoritative question -- "will this seed anything" -- instead
    # of a geometric approximation of it. Even on the biggest grids it is a
    # few tenths of a second, against a reach scan measured in seconds.
    occ = build_occupancy(g, polys, Float64(getc(cfg, "margin_cm", 0.0)),
                          robot, gi.hsub, gi.clearance;
                          bounds = bounds,
                          wall_clearance_cm = gi.wall_clearance)
    osum = occupancy_summary(g, occ)
    tchk = check_targets(g, occ, names, states, sp.ttol, bounds, polys, robot,
                         gi.clearance, gi.wall_clearance)

    # What the settings ought to be. The resolution suggestion is arithmetic;
    # the `tau_max` one runs a reach scan, and only when the grid is actually
    # tiled -- in core there is no halo, so there is nothing to trade and
    # nothing worth measuring.
    rec_budget = Float64(getc(cfg, "size_budget_bytes", 8 * 2^30))
    reach_cm = (robot === nothing || size(robot, 2) < 3) ? 0.0 :
               maximum(sqrt.(robot[1, :] .^ 2 .+ robot[2, :] .^ 2))
    gcfg = get(cfg, "grid", Dict())
    res = grid_resolution(gcfg)
    vmax = Float64(getc(gcfg, "vmax", DEFAULT_VMAX))
    wmax = Float64(getc(gcfg, "wmax", DEFAULT_WMAX))
    # Both budgets. Without the VRAM one the suggestion could recommend its
    # way from an in-core run into a tiled one, which is a far bigger cost
    # than any resolution it was buying.
    bpc = cell_bytes(Bool(getc(cfg, "warm_start", true)))
    sugg = suggest_resolution(Float64(g.step[1]) * (Int(g.n[1]) - 1),
                              Float64(g.step[2]) * (Int(g.n[2]) - 1),
                              reach_cm, vmax, wmax, eb, length(names),
                              rec_budget;
                              vram_bytes = dec.budget > 0 ? Float64(dec.budget) : Inf,
                              bytes_per_cell = bpc)
    # And, when it will not hold whole, the nearest resolution that would.
    # Tiling is the difference between minutes and days, so "how close am I to
    # not needing it" is worth answering without making the user bisect it by
    # hand. Uses the same VRAM budget `decompose` used, so the two agree.
    incore = dec.mode == :incore || dec.budget <= 0 ? nothing :
        coarsen_to_fit(Float64(g.step[1]) * (Int(g.n[1]) - 1),
                       Float64(g.step[2]) * (Int(g.n[2]) - 1), vmax, wmax,
                       (xy_cm = Float64(min(g.step[1], g.step[2])),
                        heading_deg = rad2deg(Float64(g.step[3])),
                        v_cm_s = Float64(g.step[4]),
                        w_rad_s = Float64(g.step[6])),
                       Float64(dec.budget);
                       bytes_per_cell = cell_bytes(Bool(getc(cfg, "warm_start", true))))

    wt = Bool(getc(cfg, "warm_start_tiled", true))
    # On the auto path `settle` has already run the scan and used the answer;
    # reusing it costs nothing and guarantees the report describes the run.
    rtau = st.rec !== nothing ? st.rec :
        recommend_tau(dec.mode, g, mdl, sp, dec.budget, wt,
                      Int(getc(cfg, "halo_margin", 2)),
                      Int(getc(cfg, "halo_scan_angles", 16)),
                      Int(getc(cfg, "halo_scan_commands", 128)),
                      wt ? 7.0 : 4.0,
                      Int(getc(cfg, "tile_sweeps", 4)),
                      first(cell_cost(cfg)),
                      Float64(getc(cfg, "disk_rate", 500e6)))

    Dict(
        "cells" => cells,
        "axes" => collect(AXES),
        "n" => collect(Int.(g.n)),
        # The span the table actually stores, and how it was arrived at. The
        # field is wider: the difference is the footprint plus the wall
        # clearance, which is exactly the inset the robot cannot cross.
        "bounds" => Dict(
            "field" => [bounds[1], bounds[2], bounds[3], bounds[4]],
            "table_min" => [Float64(g.lo[k]) for k in 1:6],
            "table_max" => [Float64(g.lo[k] +
                                    g.step[k] * (k == 3 ? g.n[k] : g.n[k] - 1))
                            for k in 1:6],
            "inset_cm" => [Float64(g.lo[1]) - bounds[1],
                           Float64(g.lo[2]) - bounds[2],
                           bounds[3] - Float64(g.lo[1] + g.step[1] * (g.n[1] - 1)),
                           bounds[4] - Float64(g.lo[2] + g.step[2] * (g.n[2] - 1))],
            "cells_saved_frac" => 1.0 -
                (Float64(g.step[1]) * (Int(g.n[1]) - 1) *
                 Float64(g.step[2]) * (Int(g.n[2]) - 1)) /
                max((bounds[3] - bounds[1]) * (bounds[4] - bounds[2]), 1.0e-9),
            "wall_clearance_cm" => gi.wall_clearance,
            "clearance_cm" => gi.clearance,
        ),
        # The resolution as physically set, in the unit each axis is measured
        # in, alongside what the sample counts actually realise. The two agree
        # unless `grid.n` was given, which pins the counts and lets the cell
        # size fall out of the span instead.
        "resolution" => Dict(
            "requested" => Dict("xy_cm" => res.xy_cm,
                                "heading_deg" => res.heading_deg,
                                "v_cm_s" => res.v_cm_s,
                                "w_rad_s" => res.w_rad_s),
            "actual" => Dict("xy_cm" => Float64(min(g.step[1], g.step[2])),
                             "heading_deg" => rad2deg(Float64(g.step[3])),
                             "v_cm_s" => Float64(g.step[4]),
                             "w_rad_s" => Float64(g.step[6])),
            "pinned_by_n" => getc(gcfg, "n", nothing) !== nothing,
        ),
        "occupancy" => Dict(
            "blocked_frac" => osum.frac,
            "blocked_cells" => osum.blocked,
            "xyh_cells" => osum.cells,
            "worst_heading_blocked_frac" => osum.worst_heading_frac,
            "best_heading_blocked_frac" => osum.best_heading_frac,
        ),
        "targets" => tchk,
        "targets_ok" => all(t -> t["ok"], tchk),
        "recommend" => Dict(
            "size_budget_bytes" => rec_budget,
            "resolution" => sugg === nothing ? nothing : Dict(
                "xy_cm" => sugg.xy_cm, "heading_deg" => sugg.heading_deg,
                "v_cm_s" => sugg.v_cm_s, "w_rad_s" => sugg.w_rad_s,
                "n" => sugg.n, "cells" => sugg.cells, "bytes" => sugg.bytes,
                "vram_bytes" => sugg.vram_bytes, "bound_by" => sugg.bound_by),
            "tau_max" => rtau.tau_max,
            "tau_value_cost" => rtau.cost,
            "tau_current" => rtau.current,
            "tau_current_cost" => rtau.current_cost,
            "tau_io_minor" => rtau.io_minor,
            "tau_reason" => rtau.reason,
            "tau_reference" => TAU_REF,
            # Whether the value in `solver.tau_max` was derived or pinned, and
            # what a pinned one is costing. A pinned step below the reference
            # on a grid that holds whole is pure loss, and it is the failure
            # this whole mechanism exists to make impossible.
            "tau_auto" => st.auto,
            "tau_applied" => Float64(sp.p.tau_max),
            "tau_applied_cost" => value_cost(TAU_REF, Float64(sp.p.tau_max)),
            "tau_pinned_waste" => !st.auto && dec.mode != :ooc &&
                                  Float64(sp.p.tau_max) < TAU_REF,
            # The nearest resolution that would not need tiling at all.
            "in_core" => incore === nothing ? nothing : Dict(
                "scale" => incore.scale, "xy_cm" => incore.xy_cm,
                "heading_deg" => incore.heading_deg,
                "v_cm_s" => incore.v_cm_s, "w_rad_s" => incore.w_rad_s,
                "n" => incore.n, "cells" => incore.cells,
                "vram_bytes" => incore.vram_bytes),
            "tau_options" => [Dict("tau" => o.tau, "cost" => o.cost,
                                   "round_s" => o.round_s,
                                   "compute_s" => o.compute_s,
                                   "io_s" => o.io_s,
                                   "amplification" => o.amplification,
                                   "reach_cm" => o.reach_cm)
                              for o in rtau.options],
        ),
        "dtype" => dtype,
        "elem_bytes" => eb,
        "bytes_per_target" => per,
        "n_targets" => length(names),
        "bytes_total" => per * length(names),
        "chunks_per_target" => cld(cells, chunk_elements),
        "gpu_available" => CUDA.functional(),
        "gpu_fits" => ok,
        "gpu_need_gb" => need,
        "gpu_free_gb" => free,
        # Not fitting whole is no longer a refusal: it selects the tiled
        # driver. `out_of_core.fits` is the question that can still be no.
        "out_of_core" => ooc,
        "grid_balance" => bal.ratio,
        "dv_distance_cm" => bal.dv_distance_cm,
        "max_accel_cm_s2" => bal.accel_cm_s2,
        "cell_cm" => Float64(min(g.step[1], g.step[2])),
        "cell_cm_s" => Float64(min(g.step[4], g.step[5])),
        # Ceiling on wall clock, from the configured iteration budget. The
        # model and the parameters go in with it because a sweep's cost is a
        # property of the workload as much as of the card; the occupancy
        # because a blocked cell retires before it costs anything. See
        # `runtime_estimate`.
        "runtime" => runtime_estimate(cfg, g, dec, iters, length(names);
                                      m = mdl, sp = sp,
                                      blocked_frac = osum.frac),
    )
end

# --- Honing gains -------------------------------------------------------
#
# The tables drive the robot to a handoff region, not to the target point.
# A PID owns the last few centimetres, and its gains are not tuned by hand:
# inside that region the fitted model linearises to a plant whose gains have
# a closed form, so the regression that produced `A_u` and `A_s` also
# produces the controller.

"""3x3 inverse by cofactors.

Written out rather than pulling in LinearAlgebra: the matrices here are
always exactly 3x3, and the module has no linear-algebra dependency to
justify for nine multiplies.
"""
function inv3(M)
    a, b, c = M[1][1], M[1][2], M[1][3]
    d, e, f = M[2][1], M[2][2], M[2][3]
    g, h, i = M[3][1], M[3][2], M[3][3]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    isfinite(det) && abs(det) > 1e-12 ||
        error("A_u is singular (det = $det); the drivetrain fit does not " *
              "give the three axes independent authority, so no controller " *
              "can be derived from it")
    [[(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
     [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
     [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det]]
end

mul3(X, Y) = [[sum(X[r][k] * Y[k][c] for k in 1:3) for c in 1:3] for r in 1:3]
mulv3(X, v) = [sum(X[r][k] * v[k] for k in 1:3) for r in 1:3]
scale3(s, X) = [[s * X[r][c] for c in 1:3] for r in 1:3]

"""
Derive the three honing PID controllers from the fitted model.

Near the target both the error and the velocity are small, and in that
regime the model collapses. Quadratic drag `A_absv*(abs(v).*v)` has zero
slope at `v = 0`; the omega^2 block likewise; and the smoothed Coulomb term
is *linear* inside its band, contributing a Jacobian `S * diag(1/eps)`. What
is left is a damped double integrator in the body frame:

    dp/dt = v
    dv/dt = Beff*u - Lambda*v,   Lambda = -(A + S*diag(1/eps))

`Beff` is **not** `A_u`. The command during honing is small but not
negligible, and the traction knee bites well before the octahedron edge, so
the authority the robot actually delivers is `A_u * tanh(r)/r` at the design
operating point `r = budget/knee`. Designing against raw `A_u` claims about
twice the acceleration the tyres will give and the loop overshoots badly --
this is the same trap section 8.7 of TABLE_FORMAT.md warns about, and the
reason `honing_budget` defaults to *half* the knee: there the traction gain
is still ~0.92, so the linear design is honest about itself.

The gains come out as full 3x3 matrices because `Beff` and `Lambda` are.
That is still three PID loops -- three errors, three integrators, three
derivatives -- but each output mixes all three errors, which is what it
means for the axes to be coupled.

# Pole placement

Multiplying by `inv(Beff)` turns the command into a requested acceleration,
and `Kd` additionally cancels `Lambda`'s *off-diagonal* part, so the three
channels decouple. Channel `i` then has

    s^3 + (lambda_i + kd_i)*s^2 + kp_i*s + ki_i

for the drivetrain's own damping `lambda_i = Lambda[i,i]`. Placing poles at
`-p_i` and a double `-w` gives `kd_i = p_i + 2w - lambda_i`,
`kp_i = w^2 + 2*p_i*w`, `ki_i = p_i*w^2`.

The third pole is `p_i = max(lambda_i - 2w, w)`, which is what keeps
`kd_i >= 0`:

- A **well-damped** axis (`lambda_i >= 3w`) takes `p_i = lambda_i - 2w` and
  `kd_i = 0`. The controller uses the drag the robot already has instead of
  paying derivative gain for it.
- A **lightly damped** axis (`lambda_i < 3w`) takes `p_i = w`, giving the
  triple pole at `-w` and `kd_i = 3w - lambda_i > 0`.

The two branches meet continuously at `lambda_i = 3w`. The naive triple pole
everywhere would demand `kd_i = 3w - lambda_i < 0` on a well-damped axis:
*negative* derivative gain, i.e. a controller spending command to cancel the
drivetrain's own friction. It places the poles correctly on paper and is
badly fragile in simulation -- a robot 30% less draggy than its fit
overshoots by five times -- so it is ruled out by construction.

# Bandwidth

`w` is not a free parameter. It is pinned by the command budget: requiring
the proportional term alone to stay inside the budget at the worst-case
error in the *fine band* bounds it, and a sampled loop cannot track a pole
faster than about a tenth of its rate. `norm1(Kp(w)*efine)` is increasing in `w` but
no longer a closed form once `p_i` has a max in it, so the saturation bound
is bisected rather than solved. Take the smaller of the two bounds.
"""
function honing_gains(m::Model, cfg)
    row3(t, r) = [Float64(t[(r - 1) * 3 + c]) for c in 1:3]
    B = [row3(m.B, r) for r in 1:3]
    A = [row3(m.A, r) for r in 1:3]
    S = [row3(m.S, r) for r in 1:3]
    eps = [Float64(m.eps[i]) for i in 1:3]
    knee = Float64(m.knee)

    # Coulomb linearised inside its band: csign(v) = v/eps there, so the
    # column scales by 1/eps. This is why the band is part of the model and
    # not a solver convenience -- it sets how much damping the robot sees at
    # a standstill.
    Lambda = [[-(A[r][c] + S[r][c] / eps[c]) for c in 1:3] for r in 1:3]

    # Reported, not used: the gains are sized against the fine band, so
    # where the robot chooses to hand over does not change them. It is on
    # the card so the robot side can see what the design had in mind.
    hcm  = Float64(getc(cfg, "honing_handoff_cm", 15.0))
    hrad = Float64(getc(cfg, "honing_handoff_rad", 0.25))
    hz     = Float64(getc(cfg, "honing_loop_hz", 50.0))
    # How hard the approach is allowed to push, as a fraction of the traction
    # knee. This is the aggressiveness knob, and it is a fraction of the knee
    # rather than an absolute command because the knee is what makes the
    # number mean anything: it is where the tyres stop returning what they
    # are asked for. Half the knee by default -- honing is meant to be
    # gentle, and staying well inside the knee is also what makes the linear
    # design honest (see above). Raising it raises `w_sat` with it, so the
    # loop gets faster as well as stronger, and the fine-band guarantee still
    # holds at the new budget. `honing_budget` pins the budget outright
    # instead, for a caller that would rather say it in octahedron units.
    frac   = Float64(getc(cfg, "honing_budget_frac", 0.5))
    budget = Float64(getc(cfg, "honing_budget", frac * knee))
    scale  = Float64(getc(cfg, "honing_bandwidth_scale", 1.0))
    ishare = Float64(getc(cfg, "honing_integral_share", 0.25))
    (budget > 0 && frac > 0 && hz > 0 && scale > 0 && 0 < ishare < 1) ||
        error("honing config must be positive (honing_budget=$budget, " *
              "honing_budget_frac=$frac, honing_loop_hz=$hz, " *
              "honing_bandwidth_scale=$scale, " *
              "honing_integral_share=$ishare), with the integral share below 1")
    # The octahedron is a hard limit on the command, so a budget outside it
    # is a design against authority the robot does not have: `Kp*e` would be
    # sized to a command the wheels clip away, and the fine band would stop
    # meaning what it says.
    budget <= 1 ||
        error("honing_budget is $budget, outside the octahedron " *
              "|fwd| + |strafe| + |turn| <= 1; the wheels would clip the " *
              "command the gains are sized against " *
              "(honing_budget_frac=$frac at a knee of $knee)")

    # Authority the tyres actually deliver at the design operating point.
    r_op = budget / knee
    tgain = r_op < 1e-6 ? 1.0 : tanh(r_op) / r_op
    Beff = scale3(tgain, B)
    Binv = inv3(Beff)

    lam = [Lambda[i][i] for i in 1:3]
    Loff = [[r == c ? 0.0 : Lambda[r][c] for c in 1:3] for r in 1:3]
    # The band the bandwidth is actually sized against. Saturating on the way
    # in from the handoff corner is fine and fast -- kd >= 0 and the clamped
    # integrator make it safe, and measured overshoot stays under 5%. What
    # matters is that the controller is linear and gentle once it is close,
    # which is the whole point of honing. Sizing against the handoff corner
    # instead costs a factor of eight in bandwidth and never settles.
    fcm = Float64(getc(cfg, "honing_fine_cm", 2.0))
    efine = [fcm, fcm, Float64(getc(cfg, "honing_fine_rad", 0.03))]

    # Scalar per-channel design, then back through inv(Beff).
    function gains_at(w)
        p  = [max(lam[i] - 2w, w) for i in 1:3]
        kp = [w^2 + 2 * p[i] * w for i in 1:3]
        ki = [p[i] * w^2 for i in 1:3]
        kd = [p[i] + 2w - lam[i] for i in 1:3]
        Kp = [[Binv[r][c] * kp[c] for c in 1:3] for r in 1:3]
        Ki = [[Binv[r][c] * ki[c] for c in 1:3] for r in 1:3]
        # Kd = Binv*(diag(kd) - Loff): the diagonal adds damping where the
        # drivetrain lacks it, the off-diagonal cancels axis coupling.
        Kd = [[Binv[r][c] * kd[c] - sum(Binv[r][k] * Loff[k][c] for k in 1:3)
               for c in 1:3] for r in 1:3]
        (Kp, Ki, Kd, p, kd)
    end
    prop_norm(w) = sum(abs, mulv3(gains_at(w)[1], efine))

    # Bisect the saturation bound. prop_norm is increasing in w, so bracket
    # upward from a bandwidth no loop would ever use, then halve in.
    w_hi = 1e-3
    while prop_norm(w_hi) < budget && w_hi < 1e4
        w_hi *= 2
    end
    w_lo = w_hi / 2
    for _ in 1:80
        mid = 0.5 * (w_lo + w_hi)
        prop_norm(mid) < budget ? (w_lo = mid) : (w_hi = mid)
    end
    w_sat = w_lo
    w_loop = 2pi * hz / 10
    w = scale * min(w_sat, w_loop)
    isfinite(w) && w > 0 ||
        error("honing bandwidth came out as $w; check honing_loop_hz, " *
              "honing_budget and honing_bandwidth_scale are positive")

    Kp, Ki, Kd, p, kd = gains_at(w)

    # Anti-windup. Without a clamp the integrator walks the command straight
    # out of the octahedron and the loop never settles -- measured, not
    # assumed. The limit gives the integral term at most `ishare` of the
    # budget when every axis is pinned, split evenly across the three.
    ilim = [begin
                col = sum(abs(Ki[r][c]) for r in 1:3)
                col > 0 ? ishare * budget / (3 * col) : 0.0
            end for c in 1:3]

    Dict(
        "note" => "Gains for the PID that takes over from the tables for " *
            "the final approach. Derived from this same model, so there is " *
            "no separate tuning step and nothing here to adjust on the " *
            "field. Everything that was chosen rather than derived is in " *
            "`config`, and `budget_frac` is the choice that matters: the " *
            "share of the traction knee this approach was allowed to " *
            "spend, which sets both the command budget and the bandwidth. " *
            "Additive to schema 2: a reader that does not know this block " *
            "is unaffected by it.",
        "frame" => "robot body",
        "law" => "u = Kp*e + Ki*clamp(integral(e), -integral_limit, " *
                 "integral_limit) + Kd*d(e)/dt + u_ff",
        "error" => Dict(
            "vector" => ["e_fwd", "e_strafe", "e_turn"],
            "units" => ["cm", "cm", "rad"],
            "note" => "e = target - current, rotated into the ROBOT BODY " *
                "frame, same as the state block. The tables are field " *
                "frame, so rotate by -h first.",
        ),
        "feedforward" => Dict(
            "formula" => "u_ff = B_eff_inv*(a_ref + Lambda*v_ref)",
            "why" => "Model inversion: it asks for the command the fit says " *
                "produces the reference motion, leaving the PID only the " *
                "residual to clean up. Optional -- the gains stand alone -- " *
                "but it is two matrix products and it tracks far better. " *
                "Lambda already contains the linearised Coulomb term, so no " *
                "separate break-away command is needed.",
        ),
        "Kp" => Kp, "Ki" => Ki, "Kd" => Kd,
        "B_eff_inv" => Binv, "Lambda" => Lambda,
        "traction_gain" => Dict(
            "value" => tgain,
            "why" => "B_eff = A_u * tanh(r)/r at r = budget/knee = " *
                "$(round(r_op, digits = 4)), the authority the tyres " *
                "actually deliver at the command this controller commands. " *
                "Designing against raw A_u claims roughly twice the " *
                "acceleration the robot has. B_eff_inv already contains it, " *
                "so do NOT apply it again -- but DO still apply the " *
                "section 8.4 saturation to the final command.",
        ),
        "integral_limit" => Dict(
            "value" => ilim,
            "units" => ["cm*s", "cm*s", "rad*s"],
            "why" => "Clamp each integrator to this before multiplying by " *
                "Ki, so the integral term can claim at most " *
                "$(round(ishare * 100)) percent of the command budget. " *
                "Required, not optional: without it the integrator winds " *
                "past the octahedron and the loop does not settle. Also " *
                "stop integrating while the command is saturated.",
        ),
        "omega" => w,
        "poles" => Dict(
            "double" => w,
            "third" => p,
            "kd_diag" => kd,
            "why" => "Per axis the closed loop is (s+third)(s+omega)^2. An " *
                "axis whose own damping already exceeds 3*omega takes " *
                "kd = 0 and keeps that damping; a lighter one gets the " *
                "triple pole at -omega. kd is never negative, so the " *
                "controller never cancels the drivetrain's own friction.",
        ),
        "omega_limits" => Dict(
            "saturation" => w_sat,
            "loop_rate" => w_loop,
            "bound_by" => w_sat <= w_loop ? "saturation" : "loop_rate",
            "why" => "The smaller of the two sets the bandwidth. " *
                "`saturation` keeps Kp*e inside the budget everywhere " *
                "in the fine band; `loop_rate` keeps the poles within a " *
                "tenth of the sampling rate. Outside the fine band the " *
                "command may saturate, which is intended.",
        ),
        "handoff" => Dict(
            "cm" => hcm, "rad" => hrad,
            "note" => "Where the tables hand over. Engaging further out " *
                "than this is allowed but saturates the command for longer.",
        ),
        "fine_band" => Dict(
            "cm" => efine[1], "rad" => efine[3],
            "why" => "Inside this band the command is guaranteed to stay " *
                "within the budget, so the controller is linear and gentle " *
                "exactly where precision matters. Outside it, saturating is " *
                "intended and harmless: it is what makes the approach quick.",
        ),
        "config" => Dict("loop_hz" => hz, "budget" => budget,
                         "budget_frac" => budget / knee,
                         "bandwidth_scale" => scale,
                         "integral_share" => ishare,
                         "fine_cm" => efine[1], "fine_rad" => efine[3]),
        "derivation" => "TABLE_FORMAT.md section 8.8",
    )
end

"""
Build the dynamics model the robot needs alongside the tables.

Written in a deliberately over-general form:

    a = A_s*s + A_u*u + A_ss*(s.*s) + A_uu*(u.*u)
        + A_sgn*csign(s) + A_absv*(abs(s).*s) + k

Several blocks are usually zero -- the fitted regression has no position
dependence and no control-squared terms. They are emitted anyway so that the
regression can gain or lose terms without the robot-side reader changing: it
always multiplies the same things and adds them up. `nonzero_blocks` says
which are actually carrying anything.

The matrices describe the model **as the solver actually used it**, so if
`zero_c` was set the constant here is zero too. Otherwise the robot's
dynamics would disagree with the tables that were solved from them.
"""
function model_json(m::Model, reg_path::AbstractString, zeroed_c::Bool,
                    cfg = Dict())
    row3(t, r) = [Float64(t[(r - 1) * 3 + c]) for c in 1:3]
    Am = [row3(m.A, r) for r in 1:3]      # velocity -> acceleration
    Bm = [row3(m.B, r) for r in 1:3]      # control  -> acceleration

    # State is the same 6-vector the value tables use. Position and heading do
    # not affect the dynamics, so those columns are zero.
    A_s  = [[0.0, 0.0, 0.0, Am[r][1], Am[r][2], Am[r][3]] for r in 1:3]
    # Only omega^2 is fitted, and it lands in the w column of s.*s.
    A_ss = [[0.0, 0.0, 0.0, 0.0, 0.0, Float64(m.q[r])] for r in 1:3]
    A_u  = Bm
    # Structurally zero: control-squared terms are not fitted, and there is no
    # flag that can turn them on. A u^2 column is even in u -- it claims the
    # same force for full forward and full reverse -- and command curvature is
    # already carried by the traction knee. The block is still emitted so the
    # robot-side reader never changes shape.
    A_uu = [[0.0, 0.0, 0.0] for _ in 1:3]
    Sm   = [row3(m.S, r) for r in 1:3]
    Dm   = [row3(m.D, r) for r in 1:3]
    A_sgn = [[0.0, 0.0, 0.0, Sm[r][1], Sm[r][2], Sm[r][3]] for r in 1:3]
    A_abs = [[0.0, 0.0, 0.0, Dm[r][1], Dm[r][2], Dm[r][3]] for r in 1:3]
    k    = [Float64(m.c[r]) for r in 1:3]

    # Control reaches position only through velocity, and velocity only
    # through acceleration. Enforced, not assumed: `A_u` maps the command to
    # the three accelerations and nothing else, and no block may carry an
    # x, y or h column. A regressor that quietly grew one would otherwise
    # ship as a robot that thinks its field position drives its dynamics.
    length(A_u) == 3 && all(length(r) == 3 for r in A_u) ||
        error("A_u must be 3x3: the command feeds the three accelerations " *
              "only, never a position")
    for (nm, blk) in (("A_s", A_s), ("A_ss", A_ss),
                      ("A_sgn", A_sgn), ("A_absv", A_abs))
        all(all(iszero, r[1:3]) for r in blk) ||
            error("$nm has a non-zero x/y/h column; the dynamics must not " *
                  "depend on where the robot is")
    end

    Dict(
        "schema_version" => 2,
        "generator" => "peregrine-desktop",
        "regression_sha256" => filehash(reg_path),
        "equation" => "a = A_s*s + A_u*u + A_ss*(s.*s) + A_uu*(u.*u) + " *
                      "A_sgn*csign(s) + A_absv*(abs(s).*s) + k",
        "csign" => Dict(
            "formula" => "csign(s)_i = clamp(s_i / coulomb_eps_i, -1, 1)",
            "coulomb_eps" => [0.0, 0.0, 0.0, Float64(m.eps[1]),
                              Float64(m.eps[2]), Float64(m.eps[3])],
            "why" => "Coulomb friction is smoothed instead of using a hard " *
                     "sign(), which would flip discontinuously at zero and " *
                     "make an integrator chatter. Use exactly this form.",
        ),
        "output" => Dict(
            "vector" => ["a_x", "a_y", "alpha"],
            "units" => ["cm/s^2", "cm/s^2", "rad/s^2"],
            "frame" => "robot body",
        ),
        "state" => Dict(
            "vector" => collect(AXES),
            "units" => ["cm", "cm", "rad", "cm/s", "cm/s", "rad/s"],
            "frame_note" => "IMPORTANT: vx and vy must be rotated into the " *
                "ROBOT BODY frame before use here. The value tables index " *
                "field-frame velocity, so rotate by -h between the two. " *
                "Motor forces act along the body axes, which is why the " *
                "model cannot be written with constant matrices in the " *
                "field frame. x, y and h have zero coefficients throughout.",
        ),
        "control" => Dict(
            "vector" => ["fwd", "strafe", "turn"],
            "note" => "mecanum projection of the wheel powers; the " *
                      "admissible set is |fwd| + |strafe| + |turn| <= 1",
            "saturation" => Dict(
                "knee" => Float64(m.knee),
                "formula" => "m = norm(u_raw); " *
                             "u = u_raw * tanh(m/knee) / (m/knee)",
                "why" => "Past the knee the tyres stop delivering, so extra " *
                         "command buys no extra force. APPLY THIS BEFORE " *
                         "A_u -- the gains were fitted against the saturated " *
                         "command. The knee is always present and positive.",
            ),
        ),
        "A_s" => A_s,      # 3x6
        "A_u" => A_u,      # 3x3
        "A_ss" => A_ss,    # 3x6
        "A_uu" => A_uu,    # 3x3
        "A_sgn" => A_sgn,  # 3x6, Coulomb, multiplies csign(s)
        "A_absv" => A_abs, # 3x6, quadratic drag, multiplies abs(s).*s
        "k" => k,          # 3
        "constant_zeroed" => zeroed_c,
        "nonzero_blocks" => [b for (b, nz) in (
            ("A_s", any(any(!iszero, r) for r in A_s)),
            ("A_u", any(any(!iszero, r) for r in A_u)),
            ("A_ss", any(any(!iszero, r) for r in A_ss)),
            ("A_uu", any(any(!iszero, r) for r in A_uu)),
            ("A_sgn", any(any(!iszero, r) for r in A_sgn)),
            ("A_absv", any(any(!iszero, r) for r in A_abs)),
            ("k", any(!iszero, k))) if nz],
        "honing" => honing_gains(m, cfg),
    )
end

"""
Derive the honing block on its own, and optionally write it into a solved run.

The gains depend on the regression and on the `honing_*` config, and on
nothing else -- not the grid, not the field, not the targets, not the tables.
So changing them does not need a solve, which is the whole reason this exists:
a run is hours and the gains are a millisecond, and asking for the second by
paying for the first is how a knob stops being used.

With `write_dir` the run's `MODEL.JSON` is rebuilt in place. It is rebuilt
whole rather than spliced, from the same `model_json` the solve calls, so the
file cannot drift into a shape only this path produces. The guard is the
regression hash: the tables were solved from a particular fit, and dropping a
different fit's gains beside them would describe a robot the tables do not
plan for. Everything else in the file is a function of that fit, so if the
hash matches, only the honing block can have moved.
"""
function honing_only(cfg::AbstractDict; write_dir = nothing)
    reg_path = String(cfg["regression"])
    m, _ = load_model(reg_path)
    zeroed = Bool(getc(cfg, "zero_c", true))
    if zeroed
        m = Model(m.B, m.A, m.q, m.S, m.D, (0.0f0, 0.0f0, 0.0f0),
                  m.eps, m.knee)
    end
    mj = model_json(m, reg_path, zeroed, cfg)
    if write_dir !== nothing
        path = joinpath(String(write_dir), "MODEL.JSON")
        isfile(path) ||
            error("no MODEL.JSON in $write_dir; there is nothing to update " *
                  "there, so solve the run first")
        old = readjson(path, Dict)
        String(getc(old, "regression_sha256", "")) == mj["regression_sha256"] ||
            error("$path was written from a different regression than " *
                  "$reg_path; its tables were solved from that one, so " *
                  "writing these gains beside them would ship a model the " *
                  "tables disagree with")
        open(path, "w") do io
            JSON3.pretty(io, mj)
        end
    end
    mj["honing"]
end

"""
Run value iteration for one target until it converges or runs out of sweeps.

Split out of `run_solve` so that anything measuring or testing the solver
drives exactly the same loop the CLI does. A harness that reimplements the
driver eventually measures the reimplementation instead of the solver.

`tidx` holds the 1-based flat indices of the seeded target cells. Returns
`(iterations_done, final_delta)`.
"""
function solve_value!(V, occ, g::Grid6, m::Model, tidx, ctl, nctl, p::Params;
                      iters::Int, tol::Float64, use_gpu::Bool, pol = nothing,
                      total = nothing, on_progress = nothing)
    fill!(V, p.cap)
    @views V[tidx] .= 0.0f0
    pol === nothing || fill!(pol, 0.0f0)

    # How many consecutive quiet sweeps count as converged.
    #
    # With the full lattice scanned every sweep, one quiet sweep proves no
    # control improves any cell, and that is the fixed point. With a rotating
    # slice it proves only that the controls in *this* slice do not help, so
    # the run must stay quiet for a whole rotation before the same claim
    # holds. Getting this wrong would stop early and silently ship a table
    # that a later slice would still have improved.
    quiet_needed = p.ncoarse <= Int32(0) ? 1 :
                   max(1, cld(Int(nctl) - 1, Int(p.ncoarse)))

    last_delta = Inf
    done_iters = 0
    quiet = 0
    for it in 1:iters
        d = use_gpu ?
            sweep_gpu!(V, occ, pol, g, m, ctl, nctl, p, it - 1, total;
                       rev = isodd(it)) :
            sweep_cpu!(V, occ, pol, g, m, ctl, nctl, p, it - 1;
                       rev = isodd(it))
        # The seed must be reasserted: a sweep can lower a target cell below
        # zero-cost only through interpolation noise, and letting it drift
        # would corrupt the whole basin.
        @views V[tidx] .= 0.0f0
        done_iters = it
        last_delta = d
        quiet = d <= tol ? quiet + 1 : 0
        if on_progress !== nothing && (it % 5 == 0 || it == 1 || quiet >= quiet_needed)
            on_progress(it, d)
        end
        quiet >= quiet_needed && break
    end
    (done_iters, last_delta)
end

"""
Fill in the cells the main solve could not reach, with the time to get out of
them.

Run immediately after `solve_value!` on the same `V`, for the same target,
before the table is written. Every cell with a real route is the terminal set
and is left exactly as it is; every other cell -- inside an obstacle, inside
the wall inset at that heading, or simply never reached within the horizon --
is given the shortest time to reach one of them, stored negated. See
`EscapeView` for the convention and why it is in the sign bit.

**There is no seeding step and no `fill!`.** The converged table *is* the
initial condition: the terminal set is already there, and the escape cells are
already at `cap`, which is what "no way out yet" means. That is also why this
is cheap -- it starts from the answer rather than from nothing.

**It cannot damage the table it is given.** The only write is to a cell that
failed `is_terminal`, so a cell with a real route is never touched, and a cell
that this pass fails to reach keeps the `cap` it already had and encodes as
`unreachable` exactly as it does today. Running it can add information and
cannot remove any, which is why it is on by default.

`iters` is normally far smaller than the main solve's: escape distances are
short -- a few cells to the edge of an obstacle -- so the front converges in
tens of sweeps rather than hundreds. Cells that do not converge in the budget
simply stay `unreachable`.
"""
function solve_escape!(V, occ, g::Grid6, m::Model, ctl, nctl, p::Params;
                       iters::Int, tol::Float64, use_gpu::Bool, total = nothing,
                       on_progress = nothing)
    # Same rule as `solve_value!`, and for the same reason: with a rotating
    # slice of the control lattice, one quiet sweep only proves that *this*
    # slice helps nobody.
    quiet_needed = p.ncoarse <= Int32(0) ? 1 :
                   max(1, cld(Int(nctl) - 1, Int(p.ncoarse)))

    last_delta = Inf
    done_iters = 0
    quiet = 0
    for it in 1:iters
        d = use_gpu ?
            sweep_escape_gpu!(V, occ, g, m, ctl, nctl, p, it - 1, total;
                              rev = isodd(it)) :
            sweep_escape_cpu!(V, occ, g, m, ctl, nctl, p, it - 1;
                              rev = isodd(it))
        done_iters = it
        last_delta = d
        quiet = d <= tol ? quiet + 1 : 0
        if on_progress !== nothing && (it % 5 == 0 || it == 1 || quiet >= quiet_needed)
            on_progress(it, d)
        end
        quiet >= quiet_needed && break
    end
    (done_iters, last_delta)
end

"""
Solve every target and write the card image.

Tables are written into `out_dir/TABLES`, and the manifest to
`out_dir/MANIFEST.JSON`, mirroring the layout the SD card will have so the
wizard can copy the tree across verbatim.
"""
function run_solve(cfg::AbstractDict)
    reg_path = String(cfg["regression"])
    field_path = String(cfg["field"])
    targ_path = String(cfg["targets"])
    out_dir = String(cfg["out_dir"])

    m, regcfg = load_model(reg_path)
    if Bool(getc(cfg, "zero_c", true))
        # A drivetrain at rest with no power must not accelerate. A nonzero
        # constant makes the simulated robot drift forever, so it is zeroed
        # by default -- it is a diagnostic of the fit, not real dynamics.
        m = Model(m.B, m.A, m.q, m.S, m.D, (0.0f0, 0.0f0, 0.0f0),
                  m.eps, m.knee)
    end
    bounds, polys, robot, fieldcfg = load_field(field_path)
    names, states, _ = load_targets(targ_path)

    gcfg = get(cfg, "grid", Dict())
    gi = grid_inputs(cfg, bounds, robot)
    g = build_grid(gcfg, gi)
    cells = ncells(g)

    dtype = String(getc(cfg, "dtype", "u16"))
    haskey(DTYPES, dtype) || error("unknown dtype '$dtype'")
    scale = Float64(getc(cfg, "scale", DTYPES[dtype].scale))
    chunk_elements = Int(getc(cfg, "chunk_elements", 1 << 23))
    ispow2(chunk_elements) || error("chunk_elements must be a power of two")

    # Same settlement the plan reported, from the same function, so the table
    # cannot be solved with a lookahead the plan never mentioned.
    st = settle(cfg, g, m)
    sp = st.sp
    p = sp.p; level = sp.level; nctl = sp.nctl; ctl_h = sp.ctl_h
    warm = sp.warm; iters = sp.iters; tol = sp.tol; cap = sp.cap
    nearest = sp.nearest; dt = sp.dt; nsub = sp.nsub; checks = sp.checks
    margin = sp.margin; clearance = sp.clearance; ttol = sp.ttol
    scan = sp.scan; rounds = sp.rounds; delta0 = sp.delta0; ntau = sp.ntau
    tau_ratio = sp.tau_ratio; cfl = sp.cfl; tau_min = sp.tau_min
    tau_max = sp.tau_max; hmax = sp.hmax; achecks = sp.achecks
    rk2 = sp.rk2; vclamp = sp.vclamp; simplex = sp.simplex

    # The escape pass, and the band on the card that carries what it finds.
    #
    # On by default: it only ever writes cells that came out `unreachable`, so
    # the worst it can do is spend its budget and leave the table exactly as
    # the solve left it. `escape_iterations` is small next to `iterations`
    # because escape distances are a few cells, not a field.
    do_escape = Bool(getc(cfg, "escape", true))
    esc_iters = Int(getc(cfg, "escape_iterations", 60))
    esc_tol = Float64(getc(cfg, "escape_tolerance", tol))
    escape_scale = do_escape ? escape_scale_for(dtype, cap) : 0.0
    ebase = Int(DTYPES[dtype].escape_base)
    # The band sits above every code a real value can produce. If the run has
    # been configured so that it does not, real routes would saturate into the
    # sentinel and the table would lose the top of its range silently -- so
    # refuse instead, and name the two numbers that have to move.
    if do_escape && ebase > 0 && Float64(cap) / scale >= ebase
        error("value_cap $(cap) s at scale $(scale) needs codes up to " *
              "$(round(Int, Float64(cap) / scale)), which runs into the " *
              "escape band at $ebase. Lower value_cap below " *
              "$(round(ebase * scale, digits = 2)) s, use a coarser scale, " *
              "or set escape to false")
    end
    do_escape && escape_scale <= 0.0 && dtype in ("u8", "u16") &&
        error("dtype '$dtype' has no escape band configured")

    progress(phase = "setup", cells = cells, n = collect(Int.(g.n)),
             controls = nctl, dtype = dtype,
             escape = do_escape, escape_scale = escape_scale,
             escape_base = ebase, escape_codes = escape_codes(dtype),
             bytes_per_target = cells * DTYPES[dtype].bytes,
             # Both sweep budgets, so a progress bar can give the escape pass
             # its own share of a target's slot instead of sitting at 100%
             # through it.
             iterations = iters,
             escape_iterations = do_escape ? esc_iters : 0,
             n_targets = length(names))

    hsub = gi.hsub
    # The field boundary is rasterised with the obstacles, from the same
    # footprint sweep and the same clearance rule. Before this, the boundary
    # constrained the tracking *point* only, so a 36 cm chassis was free to
    # park with half of itself outside the field -- a looser rule than the one
    # applied to an obstacle standing one centimetre inboard of that wall.
    occ_h = build_occupancy(g, polys, margin, robot, hsub, clearance;
                            bounds = bounds,
                            wall_clearance_cm = gi.wall_clearance)
    blocked = count(occ_h)
    progress(phase = "occupancy", blocked_cells = blocked,
             blocked_frac = blocked / length(occ_h),
             robot_vertices = robot === nothing ? 0 : size(robot, 2),
             heading_substeps = hsub, clearance_cm = clearance,
             wall_clearance_cm = gi.wall_clearance)

    # A target that seeds nothing produces a table that is `unreachable`
    # everywhere, and does it quietly: the sweeps converge, `delta` goes to
    # zero, and the only symptom is `reached_frac` at the very end of a run
    # that may have taken all night. Refuse before starting instead.
    tchk = check_targets(g, occ_h, names, states, ttol, bounds, polys, robot,
                         clearance, gi.wall_clearance)
    bad = filter(t -> !t["ok"], tchk)
    isempty(bad) || error(
        "these targets are states the robot cannot be in, so they would seed " *
        "nothing and the whole table would come out unreachable:\n" *
        join(["  '$(t["name"])' " *
              (isempty(t["off_axes"]) ?
               join(t["blocked_by"], "; ") :
               "is outside the table on " * join(t["off_axes"], ", ") *
               " (the table spans " *
               join([@sprintf("%s %.1f..%.1f", AXES[k], Float64(g.lo[k]),
                              Float64(g.lo[k] + g.step[k] * (g.n[k] - 1)))
                     for k in (1, 2, 4, 5, 6)], ", ") * ")")
              for t in bad], "\n"))

    backend = String(getc(cfg, "backend", "auto"))
    use_gpu = backend == "cuda" || (backend == "auto" && CUDA.functional())

    # How the grid is going to be cut up, if at all. A grid too big for the
    # card is now a decomposition rather than an error; the only remaining
    # refusal is a grid whose velocity and heading axes alone overflow it,
    # because those cannot be cut. See `decompose`.
    dec = use_gpu ? st.dec :
                    (mode = :cpu, tp = nothing, budget = Int64(0),
                     whole_bytes = Int64(0), col = Int64(0))
    ooc = dec.mode == :ooc
    if ooc && dec.tp === nothing
        a = dec.advice
        msg = "no tiling of this grid fits in " *
              "$(round(dec.budget / 2^30, digits = 2)) GB of VRAM. The " *
              "smallest tile is one column of interior inside a halo of " *
              "$(dec.hx) cells, so it needs $((1 + 2 * dec.hx)^2) columns at " *
              "$(round((a === nothing ? dec.col * 4 : a.column_bytes) / 2^20, digits = 2)) MB each"
        if a !== nothing && a.tau_max !== nothing
            msg *= ". Set tau_max to $(a.tau_max.tau) (reach " *
                   "$(round(a.tau_max.reach_cm, digits = 0)) cm, halo " *
                   "$(a.tau_max.halo), tiles $(a.tau_max.tile)x$(a.tau_max.tile), " *
                   "$(round(a.tau_max.amplification, digits = 1))x amplification, " *
                   "about +$(round(a.tau_max.cost, digits = 0))% on mean value)"
        elseif a !== nothing && a.warm_helps
            msg *= ". Set warm_start_tiled to false"
        else
            msg *= ". Reduce n[3..6] (h, vx, vy, w) -- those set the column, " *
                   "and they are the axes that cannot be tiled"
        end
        error(msg)
    end
    pf_on, pf_bytes, pf_why =
        prefetch_plan(cfg, dec, Bool(getc(cfg, "warm_start_tiled", true)))
    if ooc
        v = verify_halo(g, dec.tp)
        v.ok || error("halo of $(dec.tp.hx)x$(dec.tp.hy) cells does not cover " *
                      "a reach of $(round(dec.tp.dxy_cm, digits = 1)) cm " *
                      "($(v.need_x)x$(v.need_y) cells needed)")
        progress(phase = "decompose", tiles = dec.tp.ntiles,
                 tile_cells = [dec.tp.wx, dec.tp.wy],
                 halo_cells = [dec.tp.hx, dec.tp.hy],
                 loaded_cells = [dec.tp.nxl, dec.tp.nyl],
                 reach_cm = dec.tp.dxy_cm,
                 reach_cells = dec.tp.dxy_cm / Float64(min(g.step[1], g.step[2])),
                 amplification = dec.tp.amplification,
                 resident_bytes = dec.tp.bytes,
                 # Both scratch files: V at four bytes a cell, and the
                 # warm-start policy at three more when it is kept.
                 store_bytes = cells *
                     cell_bytes(Bool(getc(cfg, "warm_start_tiled", true))),
                 prefetch = pf_on, prefetch_bytes = pf_bytes,
                 prefetch_off_because = pf_why)
    end
    progress(phase = "backend",
             backend = ooc ? "cuda_tiled" : use_gpu ? "cuda" : "cpu")

    # The warm-start policy is three more floats per cell. In core that is a
    # good trade -- the table is a few hundred megabytes against several
    # gigabytes of VRAM, and carrying the previous sweep's command lets each
    # sweep skip most of the lattice. Tiled it costs a second store on disk,
    # at a byte per component rather than a float, and is worth it for the
    # same reason and by a wider margin: it is what keeps the round count in
    # the hundreds rather than the thousands. See `PolicyStore`.
    store = nothing
    pstore = nothing
    if ooc
        warm = Bool(getc(cfg, "warm_start_tiled", true))
        scratch = String(getc(cfg, "scratch_dir", out_dir))
        mkpath(scratch)
        # Checked before the file is created, not discovered while filling it.
        # Creating the scratch only reserves space on a filesystem that does
        # not do sparse files, so on NTFS the first sign of trouble would
        # otherwise be an IOError part way through the initial fill -- after
        # the occupancy build, which on a full-scale grid is not quick.
        # The scratch on disk holds the same two things a resident cell
        # does, in the same layout, so it is the same number.
        need = cells * cell_bytes(warm)
        free = try
            Int64(Base.Filesystem.diskstat(scratch).available)
        catch
            typemax(Int64)         # unknown; let the write find out
        end
        need > free && error(
            "the tiled solve needs $(round(need / 2^30, digits = 1)) GB of " *
            "scratch in $scratch but only $(round(free / 2^30, digits = 1)) " *
            "GB is free; point scratch_dir at a bigger volume, set " *
            "warm_start_tiled to false to drop it to " *
            "$(round(cells * 4 / 2^30, digits = 1)) GB, or reduce the grid")
        do_prefetch = pf_on
        store = open_store(cells;
                           path = joinpath(scratch, "PEREGRINE_V.SCRATCH"))
        pstore = warm ? open_policy_store(cells;
                     path = joinpath(scratch, "PEREGRINE_P.SCRATCH")) : nothing
        occ = occ_h
        V = nothing            # the tiled driver takes the store, not an array
        ctl = ctl_h
        total = nothing
        pol = nothing
    elseif use_gpu
        # Checked before anything is allocated, for the same reason the tiled
        # branch above checks the scratch before creating it: the failure it
        # replaces is a bare CUDA out-of-memory raised somewhere inside the
        # allocator, after the occupancy build, with nothing in it that says
        # what to change.
        #
        # And on Windows there is no failure at all to replace. WDDM lets an
        # allocation past the end of VRAM page into host RAM, so an oversized
        # grid runs -- at a fraction of the speed, over PCIe, with no warning.
        # That is worse than an error, and it is why this checks the number
        # rather than trusting `cudaMalloc` to object.
        need = cells * cell_bytes(warm)
        free, _ = CUDA.memory_info()
        if need > free
            error("the whole-grid solve needs " *
                  "$(round(need / 2^30, digits = 1)) GB on the card but only " *
                  "$(round(free / 2^30, digits = 1)) GB is free. Lower the " *
                  "resolution, set warm_start to false to drop it to " *
                  "$(round(cells * value_bytes_per_cell() / 2^30, digits = 1)) GB, " *
                  "or force the tiled driver with out_of_core: true. " *
                  "(On Windows this would otherwise page into host RAM and " *
                  "run, slowly, rather than fail.)")
        end
        V = CUDA.fill(cap, cells)
        occ = CuArray(occ_h)
        ctl = CuArray(ctl_h)
        total = CUDA.zeros(Float32, 1)
        # Bytes, matching the tiled driver. `cell_update` dispatches on the
        # element type through `pol_get`/`pol_set!`, so this is the whole
        # change -- and it is what makes `cell_bytes` true of both drivers
        # rather than of one. See `cell_bytes` for what the disagreement cost.
        pol = warm ? CUDA.zeros(Int8, 3 * cells) : nothing
    else
        V = fill(cap, cells)
        occ = occ_h
        ctl = ctl_h
        total = nothing
        pol = warm ? zeros(Int8, 3 * cells) : nothing
    end

    tables_dir = joinpath(out_dir, "TABLES")
    mkpath(tables_dir)
    entries = Any[]

    tile_sweeps = Int(getc(cfg, "tile_sweeps", 4))
    # `iterations` is a budget of sweeps over the grid, and a round is
    # `tile_sweeps` of them, so the tiled driver gets proportionally fewer
    # rounds. Otherwise the same config would quietly buy several times as
    # much work from one driver as from the other, and the two would not be
    # comparable on either time or quality.
    ooc_rounds = max(1, cld(iters, tile_sweeps))

    # What is left to run, in the unit the driver reports progress in.
    #
    # **The escape pass and the remaining targets are part of the run.** The
    # ETA used to count only the sweeps left in the value iteration of the
    # target in flight, so it reached zero with the escape pass and every
    # later table still to go -- on a one-target H200 job it read 0 s with
    # twelve minutes of escape sweeps left, and on a four-target job it would
    # have read zero three times. A number that says "done" three quarters of
    # the way through a rented hour is worse than no number.
    #
    # Escape sweeps are priced from the solve sweep until the pass actually
    # starts, then at their own measured cost. The conversion is the same one
    # `runtime_estimate` uses and rests on the same measurement: a solve sweep
    # pays for the free cells and an escape sweep pays for the unreached ones,
    # so the ratio is `blocked / (1 - blocked)`, using the blocked share as
    # the floor on what is unreached.
    #
    # Excludes the per-table encode and write, which is seconds to a couple
    # of minutes against sweeps measured in hours.
    blocked_frac = blocked / max(length(occ_h), 1)
    esc_ratio = ESCAPE_CELL_FACTOR * blocked_frac / max(1.0 - blocked_frac, 1.0e-6)
    solve_units = ooc ? ooc_rounds : iters
    esc_units = !do_escape ? 0 :
                ooc ? max(1, cld(esc_iters, tile_sweeps)) : esc_iters
    ntargets = length(states)
    run_eta(ti, solve_done, solve_unit_s, esc_done, esc_unit_s) =
        max(0.0, (solve_units - solve_done) * solve_unit_s) +
        max(0.0, (esc_units - esc_done) * esc_unit_s) +
        (ntargets - ti) * (solve_units * solve_unit_s + esc_units * esc_unit_s)

    # What a unit costs in the steady state, from the sweeps after the first.
    #
    # The first sweep of each kernel pays for its compilation, and pays a lot:
    # 2.96 s against 0.03 s on a small grid here, 24.2 s against 20.6 s on the
    # H200 run. Averaging it in makes the first ETA of every phase far too
    # long -- a projection of the whole remaining run at a cost that will
    # never be paid again -- so it is dropped once there is anything to
    # measure without it.
    steady(el, first_el, done) =
        done > 1 ? max(el - first_el, 0.0) / (done - 1) : el
    # And with only the compiling sweep timed there is nothing to project
    # from, so no figure is offered rather than a wrong one. Consumers fall
    # back to their own extrapolation for the one report it costs.
    maybe_eta(done, eta) = done > 1 ? eta : nothing

    try
    for (ti, s) in enumerate(states)
        tcells = target_cells(g, s, ttol)
        seeds = Int64.(tcells) .+ 1
        tidx = (use_gpu && !ooc) ? CuArray(seeds) : seeds

        t0 = time()
        first_el = Ref(0.0)      # elapsed at the first reported unit
        if ooc
            done_iters, last_delta = solve_value_ooc!(
                store, occ, g, m, seeds, ctl, nctl, p, dec.tp;
                rounds = ooc_rounds, tol = tol, tile_sweeps = tile_sweeps,
                warm = warm, pstore = pstore, prefetch = do_prefetch,
                on_progress = (rd, d) -> begin
                    el = time() - t0
                    rd == 1 && (first_el[] = el)
                    per = steady(el, first_el[], rd)
                    progress(phase = "solve", target = ti - 1,
                             target_name = names[ti], iter = rd,
                             iters = ooc_rounds, delta = d, elapsed_s = el,
                             sweep_s = per,
                             eta_s = maybe_eta(rd, run_eta(ti, rd, per, 0,
                                                           esc_ratio * per)))
                end,
                # A round over a full-scale grid is thousands of tiles and
                # tens of minutes; without this the bar would sit still for
                # all of it. Capped at about a hundred reports a round so the
                # log does not fill with them. `rounds` rides along because
                # the caller needs it to place this inside the whole run.
                on_tile = (rd, k, n) ->
                    (k % max(1, n ÷ 100) == 0 || k == n) &&
                    progress(phase = "tile", target = ti - 1,
                             target_name = names[ti], round = rd,
                             rounds = ooc_rounds, tile = k, tiles = n,
                             elapsed_s = time() - t0))
        else
            done_iters, last_delta = solve_value!(
                V, occ, g, m, tidx, ctl, nctl, p; iters = iters, tol = tol,
                use_gpu = use_gpu, pol = pol, total = total,
                on_progress = (it, d) -> begin
                    el = time() - t0
                    it == 1 && (first_el[] = el)
                    per = steady(el, first_el[], it)
                    progress(phase = "solve", target = ti - 1, target_name = names[ti],
                             iter = it, iters = iters, delta = d,
                             elapsed_s = el, sweep_s = per,
                             eta_s = maybe_eta(it, run_eta(ti, it, per, 0,
                                                           esc_ratio * per)))
                end)
        end
        # What a sweep of this target actually cost, for pricing everything
        # still to come. `done_iters` rather than the budget: `tolerance` may
        # have stopped it early, and dividing by the budget would quote a
        # sweep that never ran.
        solve_unit_s = steady(time() - t0, first_el[], max(done_iters, 1))

        # Fill in the cells that solve could not reach, with the time to get
        # out of them. This runs on the converged table, in place, and only
        # ever writes cells that were going to be `unreachable` anyway -- so
        # it is strictly additive and a budget that runs out costs nothing
        # but the sweeps. See `solve_escape!`.
        esc_iters_done = 0
        esc_delta = 0.0
        if do_escape
            te = time()
            esc_first = Ref(0.0)
            if ooc
                esc_iters_done, esc_delta = solve_escape_ooc!(
                    store, occ, g, m, ctl, nctl, p, dec.tp;
                    rounds = max(1, cld(esc_iters, tile_sweeps)), tol = esc_tol,
                    tile_sweeps = tile_sweeps, prefetch = do_prefetch,
                    on_progress = (rd, d) -> begin
                        el = time() - te
                        rd == 1 && (esc_first[] = el)
                        per = steady(el, esc_first[], rd)
                        progress(phase = "escape", target = ti - 1,
                                 target_name = names[ti], iter = rd,
                                 iters = max(1, cld(esc_iters, tile_sweeps)),
                                 delta = d, elapsed_s = el, sweep_s = per,
                                 eta_s = maybe_eta(rd,
                                     run_eta(ti, solve_units, solve_unit_s, rd, per)))
                    end)
            else
                esc_iters_done, esc_delta = solve_escape!(
                    V, occ, g, m, ctl, nctl, p; iters = esc_iters,
                    tol = esc_tol, use_gpu = use_gpu, total = total,
                    on_progress = (it, d) -> begin
                        el = time() - te
                        it == 1 && (esc_first[] = el)
                        per = steady(el, esc_first[], it)
                        progress(phase = "escape", target = ti - 1,
                                 target_name = names[ti], iter = it,
                                 iters = esc_iters, delta = d,
                                 elapsed_s = el, sweep_s = per,
                                 eta_s = maybe_eta(it,
                                     run_eta(ti, solve_units, solve_unit_s, it, per)))
                    end)
            end
        end

        # The table is encoded straight out of whatever holds V -- device
        # array, host array or the memory-mapped store -- one chunk at a time.
        # Materialising it first would mean a second full-size copy, which at
        # this scale is the difference between running and not.
        Vout = ooc ? store : use_gpu ? Array(V) : V
        progress(phase = "encode", target = ti - 1, target_name = names[ti],
                 iters_done = done_iters, delta = last_delta)

        info = write_table(tables_dir, ti - 1, Vout, dtype, scale,
                           chunk_elements, cap, escape_scale)
        reached_frac = info.reached / info.cells
        escape_frac = info.escaped / info.cells
        # Of the cells with no route, the share that now has a way out. This
        # is the number that says whether the pass did its job -- `escape_frac`
        # alone falls just because a table got better.
        dead = info.cells - info.reached
        escape_of_unreached = dead > 0 ? info.escaped / dead : 0.0
        push!(entries, Dict(
            "index" => ti - 1,
            "name" => names[ti],
            "state" => collect(Float64.(s)),
            "file_pattern" => @sprintf("TABLES/T%02dC%%04d.BIN", ti - 1),
            "n_chunks" => info.nchunks,
            "bytes" => info.bytes,
            "sha256" => info.sha256,
            "iterations" => done_iters,
            "final_delta" => last_delta,
            "reached_frac" => reached_frac,
            "escape_frac" => escape_frac,
            "escape_of_unreached_frac" => escape_of_unreached,
            "escape_iterations" => esc_iters_done,
            "escape_final_delta" => esc_delta,
        ))
        progress(phase = "target_done", target = ti - 1, target_name = names[ti],
                 chunks = info.nchunks, bytes = info.bytes,
                 reached_frac = reached_frac, escape_frac = escape_frac,
                 escape_of_unreached_frac = escape_of_unreached,
                 elapsed_s = time() - t0)
    end
    finally
        # The scratch file is the size of the value function -- tens of
        # gigabytes. Leaving one behind because a solve threw half way is how
        # a workspace fills up silently, so the cleanup is unconditional.
        keep = Bool(getc(cfg, "keep_scratch", false))
        store === nothing || close_store!(store; keep = keep)
        pstore === nothing || close_store!(pstore; keep = keep)
    end

    manifest = Dict(
        "schema_version" => 1,
        "generated_utc" => string(round(now_utc(), Second)) * "Z",
        "generator" => "peregrine-desktop",
        "regression_sha256" => filehash(reg_path),
        "field_sha256" => filehash(field_path),
        "grid" => Dict(
            "axes" => collect(AXES),
            "n" => collect(Int.(g.n)),
            "min" => [Float64(g.lo[k]) for k in 1:6],
            "max" => [Float64(g.lo[k] + g.step[k] * (k == 3 ? g.n[k] : g.n[k] - 1))
                      for k in 1:6],
            "wrap" => [false, false, true, false, false, false],
            "units" => ["cm", "cm", "rad", "cm/s", "cm/s", "rad/s"],
            "frame" => "field",
            "frame_note" => "x, y, h and the vx, vy velocity axes are all " *
                            "field frame; omega is frame independent",
            # The table is narrower than the field, on purpose: a state whose
            # footprint hangs outside the field is not one the robot can hold,
            # so storing it would be storing `unreachable`. `min`/`max` above
            # are the authority for lookup -- this is here so that a table can
            # be matched back to the field it was cut from without re-deriving
            # the inset.
            "field_bounds" => [bounds[1], bounds[2], bounds[3], bounds[4]],
            "field_note" => "grid.min/max span the positions the robot can " *
                            "legally occupy, which is the field inset by the " *
                            "footprint plus wall_clearance_cm. Clamp x and y " *
                            "into that span as section 3 of TABLE_FORMAT.md " *
                            "describes; the edge cell is the nearest legal " *
                            "state to anything beyond it",
            "total_cells" => cells,
            "index_formula" => "((((ix*Ny+iy)*Nh+ih)*Nvx+ivx)*Nvy+ivy)*Nw+iw",
        ),
        "encoding" => Dict(
            "dtype" => dtype,
            "elem_bytes" => DTYPES[dtype].bytes,
            "scale" => scale,
            "unit" => "seconds",
            "unreachable" => dtype == "u8" ? 255 :
                             dtype == "u16" ? 65535 : "NaN",
            # The escape band. A raw code at or above `escape_base` -- or, for
            # a float dtype, any negative value -- is still unreachable, and
            # additionally carries the time to reach a state that is not.
            # Zero codes means the run was solved with `escape` off, and the
            # table has none.
            "escape_base" => do_escape ? ebase : 0,
            "escape_codes" => do_escape ? escape_codes(dtype) : 0,
            "escape_scale" => do_escape ? escape_scale : 0.0,
            "escape_unit" => "seconds",
            "escape_formula" => dtype in ("u8", "u16") ?
                "raw >= escape_base -> unreachable, escape_seconds = " *
                "escape_scale * (raw - escape_base)^2" :
                "value < 0 -> unreachable, escape_seconds = -value",
            "escape_note" => "an escape cell is UNREACHABLE and must lose " *
                             "every comparison against a real route. The " *
                             "escape time is only for a robot already in one " *
                             "-- descend it to get out, then use the table " *
                             "normally. See section 6 of TABLE_FORMAT.md",
            "byte_order" => "little",
            "order" => "row_major_c",
            "chunk_elements" => chunk_elements,
            "chunk_shift" => trailing_zeros(chunk_elements),
        ),
        "solver" => Dict(
            "dt" => Float64(dt), "substeps" => nsub, "controls" => nctl,
            "control_level" => level, "iterations_max" => iters,
            "tolerance" => tol, "nearest" => nearest,
            "control_scan" => scan, "refine_rounds" => rounds,
            "refine_delta" => delta0, "warm_start" => warm,
            "escape" => do_escape, "escape_iterations" => esc_iters,
            "escape_tolerance" => esc_tol,
            "tau_levels" => ntau, "tau_ratio" => tau_ratio,
            "cfl" => cfl, "tau_min" => tau_min, "tau_max" => tau_max,
            # Derived per run unless pinned; see `settle`. Recorded because
            # "why is this table 17% slower than that one" is otherwise an
            # unanswerable question.
            "tau_max_auto" => st.auto,
            "tau_max_value_cost_pct" => value_cost(TAU_REF, Float64(tau_max)),
            "substep_max" => hmax,
            "adaptive_checks" => achecks, "rk2" => rk2,
            "simplex" => simplex,
            "interpolant" => simplex ? "kuhn_simplex_7pt" : "multilinear_64pt",
            "velocity_clamp" => vclamp,
            "integrator" => rk2 ? "midpoint_rk2" : "semi_implicit_euler",
            # How the table was produced. Provenance rather than something the
            # robot reads: a tiled solve and a whole-grid solve converge to
            # the same fixed point, so the table is the same table. It is here
            # because when a table does look wrong, "which driver made it" is
            # the first thing worth being able to answer without guessing.
            "driver" => ooc ? "tiled" : use_gpu ? "whole_grid" : "cpu",
            "tiling" => ooc ? Dict(
                "tile_cells" => [dec.tp.wx, dec.tp.wy],
                "halo_cells" => [dec.tp.hx, dec.tp.hy],
                "loaded_cells" => [dec.tp.nxl, dec.tp.nyl],
                "tiles_per_round" => dec.tp.ntiles,
                "tile_sweeps" => tile_sweeps,
                "reach_cm" => dec.tp.dxy_cm,
                "reach_rad" => dec.tp.dh_rad,
                "amplification" => dec.tp.amplification,
                "warm_start_tiled" => warm,
            ) : nothing,
            "sweep_checks" => checks, "zero_c" => Bool(getc(cfg, "zero_c", true)),
            "margin_cm" => margin, "clearance_cm" => clearance,
            "wall_clearance_cm" => gi.wall_clearance,
            "value_cap_s" => Float64(cap),
            "heading_substeps" => hsub,
            "robot_vertices" => robot === nothing ? 0 : size(robot, 2),
            "occupancy" => "per heading bin (x, y, h), obstacles and the " *
                           "field boundary alike",
            "resolution" => Dict(
                "xy_cm" => Float64(min(g.step[1], g.step[2])),
                "heading_deg" => rad2deg(Float64(g.step[3])),
                "v_cm_s" => Float64(g.step[4]),
                "w_rad_s" => Float64(g.step[6])),
        ),
        "model_file" => "MODEL.JSON",
        "targets" => entries,
    )
    open(joinpath(out_dir, "MANIFEST.JSON"), "w") do io
        JSON3.pretty(io, manifest)
    end

    mj = model_json(m, reg_path, Bool(getc(cfg, "zero_c", true)), cfg)
    open(joinpath(out_dir, "MODEL.JSON"), "w") do io
        JSON3.pretty(io, mj)
    end
    progress(phase = "model", file = "MODEL.JSON",
             nonzero_blocks = mj["nonzero_blocks"],
             honing_omega = mj["honing"]["omega"],
             honing_budget_frac = mj["honing"]["config"]["budget_frac"],
             honing_bound_by = mj["honing"]["omega_limits"]["bound_by"])
    progress(phase = "done", out_dir = out_dir, targets = length(entries))
    manifest
end

using Dates
now_utc() = Dates.now(Dates.UTC)

function filehash(path)
    open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end
