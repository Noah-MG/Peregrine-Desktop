#!/usr/bin/env julia
#
# Benchmark and accuracy harness for the value solver.
#
#   julia --project=solver solver/bench/harness.jl <config.json> [options]
#
# Two questions have to be answered together, because either one alone is
# easy to win by cheating:
#
#   * how long does a sweep take, and how many sweeps to converge
#   * how good is the resulting policy
#
# The second is measured two ways, because neither alone is trustworthy.
#
# MEAN VALUE over commonly-reached cells is the sharp measure and needs no
# reference table: every candidate a scheme evaluates is a real Bellman
# evaluation, so every scheme's V is an upper bound on the same true value
# function, and the lower of two valid upper bounds is the tighter one. It
# averages over millions of cells, so it is far less noisy than rollout.
# Its one blind spot is soundness -- a table that is simply wrong can be
# lower than one that is right -- which is what the rollout is for.
#
# CLOSED-LOOP ROLLOUT drives the robot with the policy the table implies and
# times the arrival. Every scheme is rolled out with the SAME controller, so
# the comparison isolates the table. Read the arrival *reasons*, not just the
# rate: a scheme can lose arrivals for opposite causes, and the rate alone
# cannot tell a table that dawdles from one that overshoots.

include(joinpath(@__DIR__, "..", "src", "PeregrineSolver.jl"))
using .PeregrineSolver
const PS = PeregrineSolver
using JSON3, Printf, Random

# --------------------------------------------------------------------------
# Case loading
# --------------------------------------------------------------------------

"""Everything a solve needs, built once and shared between schemes."""
struct Case
    g::PS.Grid6
    m::PS.Model
    occ::Vector{Bool}
    names::Vector{String}
    states::Vector{NTuple{6,Float32}}
    cfg::Dict
end

function load_case(cfgpath::AbstractString; grid_override = nothing)
    cfg = PS.readjson(cfgpath, Dict)
    m, _ = PS.load_model(String(cfg["regression"]))
    if Bool(PS.getc(cfg, "zero_c", true))
        m = PS.Model(m.B, m.A, m.q, m.S, m.D, (0.0f0, 0.0f0, 0.0f0), m.eps, m.knee)
    end
    bounds, polys, robot, _ = PS.load_field(String(cfg["field"]))
    names, states, _ = PS.load_targets(String(cfg["targets"]))
    gcfg = Dict{String,Any}(get(cfg, "grid", Dict()))
    grid_override !== nothing && (gcfg["n"] = collect(grid_override))
    gi = PS.grid_inputs(cfg, bounds, robot)
    g = PS.build_grid(gcfg, gi)
    # Same occupancy the solve builds, field boundary included: a benchmark
    # run against a looser wall rule than the shipped one would be measuring
    # a solver nobody uses.
    occ = PS.build_occupancy(g, polys, Float64(PS.getc(cfg, "margin_cm", 0.0)),
                             robot, gi.hsub, gi.clearance;
                             bounds = bounds,
                             wall_clearance_cm = gi.wall_clearance)
    Case(g, m, occ, names, states, cfg)
end

# --------------------------------------------------------------------------
# Rollout scoring
# --------------------------------------------------------------------------

"""
Drive the robot with the policy the table implies, and report when it hands
off to the PID controller.

**This is the robot's actual rule, not a finite-horizon lookahead.** The
online optimizer picks the command maximising the dot product of `-grad(V)`
with the direction that command pushes the robot through state space -- the
Hamiltonian minimiser of the HJB equation. Control enters the dynamics only
through acceleration, so only the *velocity* components of the gradient
matter. An earlier version of this harness scored tables by minimising
`T + V(state after T)` instead, which is a different controller; any accuracy
number from it was measuring something the robot never does.

`grad(V)` comes from the same Kuhn simplex differences the robot uses, so the
harness and the firmware read the table the same way.

**The frame boundary.** The table indexes FIELD-frame velocity; the drivetrain
model is body-frame. Writing `vf = R(h) vb` and differentiating,
`d(vf)/dt = R(h) * (dvb/dt + omega x vb)`, and the regression's `a` is exactly
that bracket -- the proper acceleration. So field-frame acceleration is
`R(h) * a`, and the objective can be evaluated entirely in the body frame by
rotating the velocity gradient the same way a velocity rotates: by `-h`.

Arrival means reaching the **handoff region**, not the seeded cell. A separate
PID controller takes over for the final approach, so whether the last few
centimetres are reachable under the value table is not this benchmark's
question -- and scoring it that way made every scheme look identical, because
almost nothing lands inside a box half a grid cell wide on all six axes.

Returns `(time, reason)`. `reason` is `:arrived`, or why it failed:
`:no_grad` (no usable gradient), `:envelope`, `:offfield`, `:obstacle`,
`:timeout`, `:diverged`.
"""
function rollout(V::Vector{Float32}, g::PS.Grid6, m::PS.Model, occ::Vector{Bool},
                 s0::NTuple{6,Float32}, targ::NTuple{6,Float32},
                 ctl::Vector{NTuple{3,Float32}};
                 dt_r::Float32 = 0.02f0, nsub::Int32 = Int32(2),
                 handoff_cm::Float32 = 15.0f0, handoff_rad::Float32 = 0.25f0,
                 handoff_speed::Float32 = 40.0f0, handoff_w::Float32 = 1.5f0,
                 max_time::Float32 = 30.0f0, cap::Float32 = 60.0f0)
    x, y, h, vfx, vfy, w = s0
    t = 0.0f0
    nsteps = ceil(Int, max_time / dt_r)
    for _ in 1:nsteps
        dx = x - targ[1]; dy = y - targ[2]
        dh = h - targ[3]
        dh -= 2.0f0 * Float32(pi) * round(dh / (2.0f0 * Float32(pi)))
        if sqrt(dx * dx + dy * dy) <= handoff_cm && abs(dh) <= handoff_rad &&
           sqrt(vfx * vfx + vfy * vfy) <= handoff_speed && abs(w) <= handoff_w
            return (t, :arrived)
        end

        # Gradient of V at the current state, field frame, from the simplex
        # the state sits in -- the robot's own recovery.
        #
        # Outside the velocity envelope the table has nothing to say, so the
        # robot forces the gradient to point back inside instead of giving
        # up, and the harness has to do the same or it scores a recovery the
        # robot performs as a failure. Look the value up at the clamped state
        # and override the offending component so that descending the
        # gradient means slowing down on that axis.
        lo4 = g.lo[4]; hi4 = -lo4
        lo5 = g.lo[5]; hi5 = -lo5
        lo6 = g.lo[6]; hi6 = -lo6
        qx = clamp(vfx, lo4, hi4); qy = clamp(vfy, lo5, hi5); qw = clamp(w, lo6, hi6)
        vv, _, _, _, gvx, gvy, gw =
            PS.interp_kuhn(V, g, x, y, h, qx, qy, qw, cap)
        vv >= cap && return (Float32(Inf), :no_grad)
        # `big` only has to dominate the real gradient, which is of order
        # (seconds per cm/s); it sets a direction, not a magnitude.
        big = 1.0f3
        vfx != qx && (gvx = sign(vfx) * big)
        vfy != qy && (gvy = sign(vfy) * big)
        w   != qw && (gw  = sign(w)   * big)

        sh, ch = sincos(h)
        vbx =  ch * vfx + sh * vfy
        vby = -sh * vfx + ch * vfy
        # Rotate the velocity gradient into the body frame, exactly as a
        # velocity rotates. Then body-frame acceleration and body-frame
        # gradient are in the same frame and the dot product is meaningful.
        gbx =  ch * gvx + sh * gvy
        gby = -sh * gvx + ch * gvy

        best = -Float32(Inf); bu = (0.0f0, 0.0f0, 0.0f0)
        for u in ctl
            ax, ay, al = PS.body_accel(m, u[1], u[2], u[3], vbx, vby, w)
            # Descend V: maximise how fast the state moves down the gradient.
            sc = -(gbx * ax + gby * ay + gw * al)
            if sc > best
                best = sc; bu = u
            end
        end

        x, y, h, vbx, vby, w =
            PS.step_state_rk2(m, x, y, h, vbx, vby, w, bu[1], bu[2], bu[3],
                              dt_r, nsub)
        sh2, ch2 = sincos(h)
        vfx = ch2 * vbx - sh2 * vby
        vfy = sh2 * vbx + ch2 * vby
        t += dt_r

        (isfinite(x) && isfinite(y) && isfinite(h) &&
         isfinite(vfx) && isfinite(vfy) && isfinite(w)) ||
            return (Float32(Inf), :diverged)
        # Only a runaway counts as an envelope failure now: a brief excursion
        # is something the gradient override is expected to pull back.
        vlim4 = abs(g.lo[4]) * 1.5f0; vlim5 = abs(g.lo[5]) * 1.5f0
        wlim = abs(g.lo[6]) * 1.5f0
        (abs(vfx) > vlim4 || abs(vfy) > vlim5 || abs(w) > wlim) &&
            return (Float32(Inf), :envelope)

        gx = (x - g.lo[1]) / g.step[1]
        gy = (y - g.lo[2]) / g.step[2]
        (gx < 0 || gy < 0 || gx > Float32(g.n[1] - 1) || gy > Float32(g.n[2] - 1)) &&
            return (Float32(Inf), :offfield)
        n3f = Float32(g.n[3])
        gh = (h - g.lo[3]) / g.step[3]
        gh = gh - floor(gh / n3f) * n3f
        ci = Int32(round(gx)); cj = Int32(round(gy)); ck = Int32(round(gh)) % g.n[3]
        @inbounds occ[(Int64(ci) * g.n[2] + cj) * g.n[3] + ck + 1] &&
            return (Float32(Inf), :obstacle)
    end
    (Float32(Inf), :timeout)
end

"""
Pick rollout start states: random, collision-free, and reproducible.

Deliberately drawn from the whole reachable box rather than from a handful of
hand-picked poses, because a scheme can easily be better in the open field
and worse near an obstacle, and an average over both is what the robot
experiences.
"""
function sample_starts(g::PS.Grid6, occ::Vector{Bool}, n::Int; seed = 20260825,
                       vfrac = 0.6)
    rng = MersenneTwister(seed)
    out = NTuple{6,Float32}[]
    vmaxx = g.lo[4] + g.step[4] * (g.n[4] - 1)
    vmaxy = g.lo[5] + g.step[5] * (g.n[5] - 1)
    wmax  = g.lo[6] + g.step[6] * (g.n[6] - 1)
    while length(out) < n
        x = g.lo[1] + rand(rng) * g.step[1] * (g.n[1] - 1)
        y = g.lo[2] + rand(rng) * g.step[2] * (g.n[2] - 1)
        h = -Float32(pi) + rand(rng) * 2Float32(pi)
        vx = (2rand(rng) - 1) * vmaxx * vfrac
        vy = (2rand(rng) - 1) * vmaxy * vfrac
        w  = (2rand(rng) - 1) * wmax * vfrac
        gx = (x - g.lo[1]) / g.step[1]; gy = (y - g.lo[2]) / g.step[2]
        gh = (h - g.lo[3]) / g.step[3]
        ci = Int32(round(gx)); cj = Int32(round(gy))
        ck = Int32(round(gh)) % g.n[3]
        occ[(Int64(ci) * g.n[2] + cj) * g.n[3] + ck + 1] && continue
        push!(out, (Float32(x), Float32(y), Float32(h),
                    Float32(vx), Float32(vy), Float32(w)))
    end
    out
end

"""
Aggregate a set of rollouts.

`censored` is the headline number and the only one that is not selection
biased. Averaging arrival times over *arrivals only* rewards a table that
fails on exactly the hard starts -- it quietly drops them from its own
average -- and that is not hypothetical: a table here arrived on 33% of
starts against 18% and still looked WORSE on mean arrival time, purely
because the extra 15% it solved were the slow ones the other never reached
at all. Scoring a non-arrival as the full time budget spends every start on
every scheme and makes "arrives at all" and "arrives quickly" commensurable.

`median` sits next to the mean because the arrival distribution has a long
tail: a handful of near-timeout arrivals move a mean a long way and a median
hardly at all.
"""
function score(V, g, m, occ, targ, starts, ctl; max_time::Float32 = 30.0f0, kw...)
    ts = Vector{Float32}(undef, length(starts))
    rs = Vector{Symbol}(undef, length(starts))
    Threads.@threads for i in eachindex(starts)
        ts[i], rs[i] = rollout(V, g, m, occ, starts[i], targ, ctl;
                               max_time = max_time, kw...)
    end
    fin = filter(isfinite, ts)
    why = Dict{Symbol,Int}()
    for r in rs
        why[r] = get(why, r, 0) + 1
    end
    cens = [isfinite(t) ? t : max_time for t in ts]
    (times = ts, reasons = rs, why = why,
     arrived = length(fin) / length(ts),
     mean_time = isempty(fin) ? NaN : sum(fin) / length(fin),
     median_time = isempty(fin) ? NaN : sort(fin)[cld(length(fin), 2)],
     censored = sum(cens) / length(cens))
end

"""Render the failure tally compactly, most common first."""
function why_str(why::Dict{Symbol,Int})
    ks = sort([k for k in keys(why) if k != :arrived]; by = k -> -why[k])
    isempty(ks) ? "-" : join(("$(k) $(why[k])" for k in ks), ", ")
end

"""Paired comparison on the starts both schemes solved, plus win counts."""
function compare(a::Vector{Float32}, b::Vector{Float32})
    both = [i for i in eachindex(a) if isfinite(a[i]) && isfinite(b[i])]
    isempty(both) && return (n = 0, mean_a = NaN, mean_b = NaN, med_a = NaN,
                             med_b = NaN, wins_b = 0, losses_b = 0, ties = 0)
    da = a[both]; db = b[both]
    ma = sum(da) / length(both); mb = sum(db) / length(both)
    sa = sort(da); sb = sort(db)
    (n = length(both), mean_a = ma, mean_b = mb,
     med_a = sa[cld(length(sa), 2)], med_b = sb[cld(length(sb), 2)],
     wins_b = count(i -> b[i] < a[i] - 1.0f-4, both),
     losses_b = count(i -> b[i] > a[i] + 1.0f-4, both),
     ties = count(i -> abs(b[i] - a[i]) <= 1.0f-4, both))
end
