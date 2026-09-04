# Turning a config into advice: what resolution to ask for, how long the
# lookahead should be, and whether the targets are states the robot can
# actually be in. Included into the PeregrineSolver module.
#
# Everything here is advisory. Nothing in this file changes what a solve
# computes -- it changes what the user is told before committing to one. That
# distinction matters most for the reach numbers: the tile plan a solve is
# built from always comes from `reach_extent` at the configured `tau_max`,
# never from the cheaper curve the recommendation is read off.

# --------------------------------------------------------------------------
# What a shorter lookahead costs
# --------------------------------------------------------------------------

# The longest step the cost curve is anchored to. Not a hard maximum -- a
# longer one is legal, and presumably slightly better -- but it is the longest
# that has been measured, so it is what "no accuracy lost" means in every
# percentage this file reports.
const TAU_REF = 0.5

# What shortening the longest lookahead costs, in percent of mean value.
#
# Measured, not modelled: a 145x145x8x9x9x5 grid at the cell size a full-scale
# run uses, scored over the 44.5M cells every variant reached, against the
# full `tau_max` 0.5 ladder. The value function only ever gets *more*
# pessimistic as the ladder shortens -- a backup that cannot see far enough
# ahead to find the good command settles for a worse one -- so this is a
# one-sided cost, and it is not linear: the last few rungs are the ones that
# carry the long coasting moves.
const TAU_COST = ((0.5, 0.0), (0.2, 6.0), (0.1, 21.1), (0.06, 56.0))

"""Percent added to mean value by running with longest step `tau`."""
function value_cost(tau::Real)
    t = Float64(tau)
    t >= TAU_COST[1][1] && return 0.0
    for i in 1:(length(TAU_COST) - 1)
        a, b = TAU_COST[i], TAU_COST[i + 1]
        if t >= b[1]
            f = (log(t) - log(a[1])) / (log(b[1]) - log(a[1]))
            return a[2] + f * (b[2] - a[2])
        end
    end
    # Past the shortest rung measured, continue on the last slope. It is an
    # extrapolation of an already steep curve, so it understates if anything.
    a, b = TAU_COST[end - 1], TAU_COST[end]
    b[2] + (log(t) - log(b[1])) * (b[2] - a[2]) / (log(b[1]) - log(a[1]))
end

"""Percent added by moving the longest step from `from` down to `to`."""
value_cost(from::Real, to::Real) = max(0.0, value_cost(to) - value_cost(from))

"""
Seconds for one round: `sweeps` passes over every cell, plus the traffic.

**The sum, not the greater of the two.** This used to take the maximum, on the
reasoning that the loads of one tile overlap the arithmetic of the last. They
do not. `solve_value_ooc!` runs strictly sequentially -- `load_tile!`, then
`tile_sweeps` sweeps, then `store_tile!`, one tile at a time -- with no CUDA
streams, no second staging buffer and no prefetch task, and the store is
buffered `IOStream` reads rather than a mapping, so there is no kernel
readahead across tiles either. Every stage waits for the one before it.

The maximum is the model of a double-buffered pipeline, and it flattered a
driver that has never been one: with I/O comparable to compute it under-states
a round by close to a factor of two, and worse, it makes every step length
below the compute-bound point look *identical* in cost, which turned the
`tau_max` recommendation into "take the longest one that fits".

Writes are the one part that really is partly hidden -- `unsafe_write` returns
once the data is in the page cache and the kernel flushes behind us -- so the
truth sits between the two models and much nearer this one. Erring toward the
sum is also the right direction to err in: it makes a long halo look expensive,
which it is.

`amp + 1` because every loaded cell is read and every updated cell written back.
"""
function round_seconds(cells::Float64, sweeps::Int, amp::Float64, sb::Float64,
                       rate::Float64, disk::Float64; prefetch::Bool = false)
    compute = cells * sweeps / rate
    read_io = cells * sb * amp / disk
    write_io = cells * sb / disk
    io = read_io + write_io
    # With the prefetch on, the reads really are overlapped -- a background
    # task fills the next window while the kernel runs -- so for the read half
    # the maximum is the right model rather than the sum. The writes stay in
    # the sum: nothing hides them, beyond the page cache that was always
    # hiding them a little.
    #
    # Note which way this cuts. Hiding the reads helps most when compute is
    # *small* next to them, so the faster the card, the more the prefetch is
    # worth -- on a slow card the reads were already free.
    prefetch ? (total = max(compute, read_io) + write_io, compute = compute,
                io = io) :
               (total = compute + io, compute = compute, io = io)
end

# --------------------------------------------------------------------------
# Recommending a longest lookahead
# --------------------------------------------------------------------------

# The candidates. Geometric down from the reference, plus enough short rungs
# to have something to say about a grid that only fits with a tiny halo.
const TAU_LADDER = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.12, 0.1, 0.08, 0.06,
                    0.05, 0.04, 0.03, 0.02]

"""
Cost out every candidate `tau_max` on one shared reach scan.

`tau_max` is the only knob in the config whose two effects pull opposite ways
and neither of which is visible from the config file: it sets how far the
step-length ladder may look ahead, which is accuracy, and through that it
sets the halo every tile must be loaded with, which is wall clock. Costing
both on the same ladder is the only way to see where the two cross, and the
crossing is not where intuition puts it -- the halo grows as `(1 + 2h)^2`
against an interior that does not, so the last few rungs of step length cost
several times what the first few do.

Returns one entry per candidate that produces a workable tiling, longest
first, each carrying both sides: `cost` against `TAU_REF`, and `round_s` with
the compute/io split that explains it.
"""
function tau_options(g::Grid6, m::Model, sp, budget::Int64, warm::Bool,
                     margin::Int, nang::Int, ncmd::Int, sb::Float64,
                     sweeps::Int, rate::Float64, disk::Float64;
                     ladder::Vector{Float64} = TAU_LADDER)
    taus = sort(unique(Float64.(ladder)))
    curve = reach_curve(g, m, sp.p, sp.ctl_h, sp.nctl, taus;
                        nangle = nang, nsample = ncmd)
    cells = Float64(ncells(g))
    opts = NamedTuple[]
    for tau in reverse(taus)
        dxy, dh = reach_at(curve, tau)
        hx, hy = halo_cells(g, dxy; margin = margin)
        tp = plan_tiles(g, budget, hx, hy; warm = warm, dxy_cm = dxy,
                        dh_rad = dh)
        tp === nothing && continue
        r = round_seconds(cells, sweeps, tp.amplification, sb, rate, disk)
        push!(opts, (tau = tau, reach_cm = dxy, halo = hx, tile = tp.wx,
                     amplification = tp.amplification, round_s = r.total,
                     compute_s = r.compute, io_s = r.io,
                     cost = value_cost(TAU_REF, tau)))
    end
    opts
end

"""
Pick a `tau_max`, and be able to say why.

One objective, scored as `(round time) * (1 + value cost)`. Both are
multiplicative penalties on the same run -- one lengthens it, the other makes
every answer in it worse by a measured percentage -- so their product has a
meaningful minimum, and it sits at the knee of the curve rather than at either
end of it.

There used to be a second, special regime here: while `round_seconds` took the
maximum of compute and I/O, every step below the compute-bound point cost the
same, so the rule was "take the longest one that is still compute-bound" and
the halo was described as free. That was an artefact of the cost model rather
than a property of the solver -- see `round_seconds` -- and it biased the
recommendation toward longer steps than the driver can actually afford. With
the sum there is no free regime and no special case: shortening the step
always buys some time and always costs some accuracy, and the only question is
where the product turns.

In core there is no halo and no trade -- `tau_max` costs a few integration
substeps and nothing else -- so the answer is the reference and no scan runs
at all. That is also the common case while the user is still tuning
resolution, which is why the short circuit is worth having.
"""
function recommend_tau(mode::Symbol, g::Grid6, m::Model, sp, budget::Int64,
                       warm::Bool, margin::Int, nang::Int, ncmd::Int,
                       sb::Float64, sweeps::Int, rate::Float64, disk::Float64)
    cur = Float64(sp.p.tau_max)
    if mode != :ooc
        return (tau_max = TAU_REF, cost = 0.0, current = cur,
                current_cost = value_cost(TAU_REF, cur), io_minor = true,
                options = NamedTuple[],
                reason = "the whole grid fits on the card, so the lookahead " *
                         "sets no halo and costs nothing but a few " *
                         "integration substeps")
    end
    opts = tau_options(g, m, sp, budget, warm, margin, nang, ncmd, sb, sweeps,
                       rate, disk)
    isempty(opts) && return (tau_max = nothing, cost = nothing, current = cur,
                current_cost = nothing, io_minor = false, options = opts,
                reason = "no tiling of this grid fits at any step length")

    best = argmin(o -> o.round_s * (1.0 + o.cost / 100), opts)
    # Which side of the trade is actually binding at the pick, since that is
    # what tells the user whether there is anything to gain by pushing.
    minor = best.io_s <= 0.25 * best.compute_s
    (tau_max = best.tau, cost = best.cost, current = cur,
     current_cost = value_cost(TAU_REF, cur), io_minor = minor, options = opts,
     reason = minor ?
        "the halo is already a small part of a round here -- the sweeps " *
        "dominate -- so shortening the step further would trade real " *
        "accuracy for very little time" :
        "the knee of the trade: shortening the step below this saves less " *
        "time than the accuracy it costs, and lengthening it past this costs " *
        "more time than the accuracy is worth")
end

# --------------------------------------------------------------------------
# Recommending a resolution
# --------------------------------------------------------------------------

"""
A heading bin fine enough that the rasterised slice represents the bin.

The occupancy grid stores one polygon per heading bin, so the bin has to be
narrow enough that the footprint has not moved appreciably across it. The
extreme point of the footprint sits `reach` from the tracking point and
travels `reach * dtheta` across a whole bin; holding that under one position
cell is the rule, and it is calibrated rather than guessed -- at 16 bins on a
30 x 16 cm chassis the swept extent was measured to move 0.47 cm across a
half bin, which this rule calls acceptable next to a 9 cm cell and not next
to a 0.5 cm one.

Clamped at both ends: below 8 bins the heading axis stops resolving rotation
at all, and past 180 the bins are finer than the substep union that fills
them.
"""
function suggest_heading_deg(reach_cm::Real, xy_cm::Real)
    r = max(Float64(reach_cm), 1.0e-6)
    nb = clamp(ceil(Int, 2pi * r / Float64(xy_cm)), 8, 180)
    360.0 / nb
end

"""
A resolution for every axis, at or under a size budget.

The position cell is the free variable and the other three are pinned to it
by rules of very different standing, which is worth being honest about:

  * **heading** is derived -- see `suggest_heading_deg`. Geometry, calibrated
    against a measurement.
  * **velocity and spin** are a convention: a tenth of the envelope, so 21
    samples with zero exactly on the grid. There is no clean derivation for
    this one, and pretending otherwise would be worse than saying so. The
    grid-proportioning rule says a *coarser* velocity cell makes each backup
    learn more, while the terminal approach wants a finer one, and only the
    first effect has been measured. Twenty-one samples is what the grids that
    were measured actually used.
  * **position** is then the finest cell satisfying *both* budgets below,
    searched directly rather than solved for, because the sample counts round
    outward and the heading count moves with the answer.

**Two budgets, and the VRAM one is not optional.** `budget_bytes` is space on
the card; `vram_bytes` is what the GPU can hold at once. A suggestion that
respected only the first was free to recommend a grid that has to be solved a
tile at a time -- and tiling is not a small tax on this solver, it is the
difference between minutes and days. Worse, it made "apply the recommended
settings" a move that could quietly take a run from in-core to out-of-core,
which is the opposite of what a recommendation should do. So the finest cell
offered is the finest that clears both, and `bound_by` says which one stopped
it, since "you are out of card" and "you are out of VRAM" have different fixes.

Table bytes go as the cube of the position cell -- two position axes, and the
heading axis through the rule above -- so neither budget is a strong lever.
Halving the cell costs eight times the space.
"""
function suggest_resolution(span_x::Float64, span_y::Float64, reach_cm::Real,
                            vmax::Float64, wmax::Float64, elem_bytes::Int,
                            ntargets::Int, budget_bytes::Float64;
                            finest_cm::Float64 = 1.0,
                            vram_bytes::Float64 = Inf,
                            bytes_per_cell::Int = 16)
    v_cm_s = vmax / 10
    w_rad_s = wmax / 10
    nv = 21; nw = 21                      # by construction of the two above
    xy = finest_cm
    while xy <= 60.0
        hdeg = suggest_heading_deg(reach_cm, xy)
        nh = max(4, round(Int, 360.0 / hdeg))
        n1 = max(2, ceil(Int, span_x / xy - 1.0e-9) + 1)
        n2 = max(2, ceil(Int, span_y / xy - 1.0e-9) + 1)
        cells = Int64(n1) * n2 * nh * nv * nv * nw
        bytes = Float64(cells) * elem_bytes * max(ntargets, 1)
        vram = Float64(cells) * bytes_per_cell
        if bytes <= budget_bytes && vram <= vram_bytes
            return (xy_cm = xy, heading_deg = hdeg, v_cm_s = v_cm_s,
                    w_rad_s = w_rad_s, n = [n1, n2, nh, nv, nv, nw],
                    cells = cells, bytes = bytes, vram_bytes = vram,
                    bound_by = vram > 0.5 * vram_bytes ? "vram" : "card")
        end
        xy += xy < 10 ? 0.1 : 0.5
    end
    nothing
end

"""
The cell counts a resolution implies, mirroring `build_grid` exactly.

Kept next to `coarsen_to_fit` because that function's whole job is to search
over resolutions, and a search that sized its candidates differently from the
builder would recommend a grid that is not the grid you get.
"""
function shape_for(span_x::Float64, span_y::Float64, vmax::Float64,
                   wmax::Float64, xy::Float64, hdeg::Float64, v::Float64,
                   w::Float64)
    n1 = max(2, ceil(Int, span_x / xy - 1.0e-9) + 1)
    n2 = max(2, ceil(Int, span_y / xy - 1.0e-9) + 1)
    nh = max(4, round(Int, 360.0 / hdeg))
    nv = 2 * max(1, ceil(Int, vmax / v - 1.0e-9)) + 1
    nw = 2 * max(1, ceil(Int, wmax / w - 1.0e-9)) + 1
    n = [n1, n2, nh, nv, nv, nw]
    (n = n, cells = Int64(n1) * n2 * nh * nv * nv * nw)
end

"""
The finest version of *this* resolution that still holds whole on the card.

A different question from `suggest_resolution`, which answers "what fits in a
size budget on the SD card". This one answers "what fits in VRAM", which is
the question behind a grid that has just been told it will be solved a tile
at a time -- and tiling is the difference between minutes and days, so it is
worth being able to see the nearest point where it stops.

The search is **one scale factor applied to all four cell sizes**, not a free
search over four axes. That is deliberate. It keeps whatever balance the user
chose between position, heading and velocity resolution instead of quietly
substituting this file's opinion for it, and it makes the answer one number
they can reason about: "your resolution, 1.6x coarser everywhere". A free
search would find a slightly smaller grid and would do it by wrecking one
axis, which is not a trade anyone asked for.

Returns `nothing` if even a 16x coarsening will not fit, which on any real
field means the velocity envelope or the heading axis is the problem rather
than the resolution.
"""
function coarsen_to_fit(span_x::Float64, span_y::Float64, vmax::Float64,
                        wmax::Float64, res, budget_bytes::Float64;
                        bytes_per_cell::Int = 16)
    k = 1.0
    while k <= 16.0
        sh = shape_for(span_x, span_y, vmax, wmax, res.xy_cm * k,
                       res.heading_deg * k, res.v_cm_s * k, res.w_rad_s * k)
        if Float64(sh.cells) * bytes_per_cell <= budget_bytes
            return (scale = k, xy_cm = res.xy_cm * k,
                    heading_deg = res.heading_deg * k, v_cm_s = res.v_cm_s * k,
                    w_rad_s = res.w_rad_s * k, n = sh.n, cells = sh.cells,
                    vram_bytes = Float64(sh.cells) * bytes_per_cell)
        end
        k += 0.05
    end
    nothing
end

# --------------------------------------------------------------------------
# Checking the targets are states the robot can be in
# --------------------------------------------------------------------------

"""
Would this target seed anything?

The failure this exists to catch is silent and total. `target_cells` clamps
its ranges to the grid, so a target off the table seeds *no* cells; a target
inside an obstacle, or hanging through a wall, seeds cells that `cell_update`
returns `cap` for and that `lookahead` refuses to step into. Either way the
solve runs its full iteration budget and writes a table that is `unreachable`
everywhere, with a converged `delta` and nothing in the log to say why. It is
much cheaper to hear about it in `plan`.

The escape pass does not paper over this, which is worth knowing since it
exists to put numbers in unreachable cells. Its terminal set is the cells with
a real route, and a table that seeds nothing has none, so it escapes nothing
and the symptom survives intact: `reached_frac` zero, `escape_frac` zero.

The occupancy mask is the authority rather than a fresh geometry test,
because it is what the solver will actually read -- including the union over
substep angles, which is how a target legal at its own heading can still land
in a bin that is blocked. The geometry is then re-tested only to say *which*
constraint did it, since "blocked" on its own is not actionable.
"""
function check_targets(g::Grid6, occ::Vector{Bool}, names::Vector{String},
                       states::Vector{NTuple{6,Float32}},
                       ttol::NTuple{6,Float32}, bounds::NTuple{4,Float64},
                       polys::Vector{Matrix{Float64}},
                       robot::Union{Nothing,Matrix{Float64}},
                       clearance::Real, wall_clearance::Real)
    out = Dict{String,Any}[]
    for (i, s) in enumerate(states)
        seeds = target_cells(g, s, ttol)
        nfree = 0
        for c in seeds
            i1, i2, i3, _, _, _ = unflatten(g, c)
            occ[(Int64(i1) * g.n[2] + i2) * g.n[3] + i3 + 1] || (nfree += 1)
        end
        # Which axes the state is off the end of, named rather than numbered.
        off = String[]
        for k in 1:6
            k == 3 && continue            # heading wraps; never out of range
            hi = Float64(g.lo[k]) + Float64(g.step[k]) * (Int(g.n[k]) - 1)
            (Float64(s[k]) < Float64(g.lo[k]) - 1.0e-6 ||
             Float64(s[k]) > hi + 1.0e-6) && push!(off, AXES[k])
        end
        # And, when it is on the table but blocked, what is blocking it.
        why = String[]
        if isempty(off) && nfree == 0
            th = Float64(s[3])
            bxlo, bxhi, bylo, byhi = wall_box(bounds, robot, wall_clearance, th)
            (Float64(s[1]) < bxlo || Float64(s[1]) > bxhi ||
             Float64(s[2]) < bylo || Float64(s[2]) > byhi) &&
                push!(why, "the footprint crosses the field boundary")
            point_robot = robot === nothing || size(robot, 2) < 3
            body = point_robot ? zeros(2, 0) :
                   ([cos(th) -sin(th); sin(th) cos(th)] * robot) .+
                   [Float64(s[1]); Float64(s[2])]
            for (pi_, P) in enumerate(polys)
                hit = point_robot ?
                    point_too_close(Float64(s[1]), Float64(s[2]), P, clearance) :
                    polys_too_close(body, P, clearance)
                hit && push!(why, "it is within $(clearance) cm of obstacle $(pi_)")
            end
            # Legal where it stands, blocked in the bin it snaps to: the bin
            # is a union over substep angles, so a heading near the edge of
            # one can be condemned by a neighbour it never occupies.
            isempty(why) && push!(why,
                "it is legal at its own heading but its heading bin is not. " *
                "The bin spans $(round(rad2deg(Float64(g.step[3])), digits = 1)) " *
                "deg and counts as blocked if any angle in it is, so more " *
                "heading samples would separate the two")
        end
        push!(out, Dict{String,Any}(
            "index" => i - 1, "name" => names[i],
            "seed_cells" => length(seeds), "free_seed_cells" => nfree,
            "ok" => nfree > 0,
            "off_axes" => off,
            "blocked_by" => why))
    end
    out
end

"""How much of the (x, y, h) grid is wall, obstacle, or neither."""
function occupancy_summary(g::Grid6, occ::Vector{Bool})
    n = length(occ)
    nh = Int(g.n[3])
    # Per heading bin, so an anisotropic chassis shows up: a long robot on a
    # tight field has bins far more blocked than others, and that is the shape
    # of "it cannot turn round in here".
    per = [count(@view occ[(k + 1):nh:end]) / (n / nh) for k in 0:(nh - 1)]
    (blocked = count(occ), cells = n, frac = count(occ) / n,
     worst_heading_frac = maximum(per), best_heading_frac = minimum(per))
end
