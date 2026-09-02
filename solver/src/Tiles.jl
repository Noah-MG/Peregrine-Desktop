# How far one Bellman backup reaches, and the tile decomposition that follows
# from it. Included into the PeregrineSolver module.

# --------------------------------------------------------------------------
# Dependency reach
# --------------------------------------------------------------------------

"""
The set of lookahead horizons `cell_update` will actually try from a cell
whose field speed is `sp` and spin is `w`.

Kept in one place because the reach analysis and the backup have to agree
exactly: a horizon the backup tries but the analysis has not accounted for is
a read outside the loaded tile, which is silently wrong rather than loudly
wrong. Mirrors the ladder in `cell_update` rung for rung -- the base `tau0`,
the geometric ladder around it, and the `dt` rung that restores the "no worse
than the fixed step" guarantee.
"""
function horizon_set(g::Grid6, m::Model, p::Params, vfx, vfy, w)
    tau0 = cfl_tau(g, m, Float32(vfx), Float32(vfy), Float32(w), p)
    ts = Float32[tau0]
    if p.ntau > Int32(1)
        tau = tau0 * 0.5f0
        for _ in Int32(1):p.ntau
            push!(ts, clamp(tau, p.tau_min, p.tau_max))
            tau *= p.tau_ratio
        end
        p.cfl > 0.0f0 && push!(ts, clamp(p.dt, p.tau_min, p.tau_max))
    end
    unique!(ts)
    ts
end

"""
Commands the reach scan has to cover.

The lattice is not enough on its own. `cell_update` runs a pattern search
that walks the incumbent off the lattice and across the octahedron surface by
arbitrary amounts, so the reach has to be measured over the *continuous*
admissible set. Sampling that surface densely and taking the worst case is
how it is covered: displacement varies smoothly with the command -- the only
non-smooth term, `csign`, is already smoothed over a band -- so a fine sample
plus the cell margin in `halo_cells` bounds it.

Coasting is included because it is always evaluated, and for a cell that is
already moving fast it is the longest step of all.
"""
function reach_commands(ctl::Vector{Float32}, nctl::Int; nsample::Int = 512)
    us = NTuple{3,Float32}[(0.0f0, 0.0f0, 0.0f0)]
    for k in 1:nctl
        push!(us, (ctl[k], ctl[k + nctl], ctl[k + 2 * nctl]))
    end
    # Deterministic quasi-uniform cover of the octahedron surface |u|_1 = 1:
    # a golden-angle spiral on the sphere, pushed onto the surface by the
    # 1-norm. Deterministic so two runs of `plan` report the same halo.
    ga = Float32(pi * (3.0 - sqrt(5.0)))
    for i in 0:(nsample - 1)
        z = 1.0f0 - 2.0f0 * Float32(i) / Float32(max(nsample - 1, 1))
        r = sqrt(max(0.0f0, 1.0f0 - z * z))
        th = ga * Float32(i)
        u1 = r * cos(th); u2 = r * sin(th); u3 = z
        s = abs(u1) + abs(u2) + abs(u3)
        s < 1.0f-6 && continue
        push!(us, (u1 / s, u2 / s, u3 / s))
    end
    unique!(us)
    us
end

"""
How far one Bellman backup can reach, in centimetres and in radians.

This is the number the whole out-of-core decomposition rests on. A tile is
swept against a frozen border, and that border has to be at least as wide as
the furthest a single backup can look -- so an under-stated reach does not
fail loudly. A read past the loaded window comes back as `cap`, which prices
a perfectly good step as unreachable, and `cap` is a legal value everywhere
else in the solver. The damage shows up as a faint lattice of seams on the
tile boundaries and as nothing else at all.

So the scan is deliberately a worst case over everything a cell could be
doing, rather than a typical case:

  * every horizon in `horizon_set`, which is the ladder the backup really
    tries, including the rungs longer than the nominal one;
  * every command in `reach_commands`, which covers the continuous
    octahedron surface rather than only the lattice the search seeds from;
  * every speed out to the *corner* of the velocity box, sampled at half a
    velocity cell -- the corner matters, because a grid point at
    `(vmax, vmax)` has speed `sqrt(2)*vmax` and is a real starting state;
  * every spin on the omega axis exactly, since that axis is short and both
    `cfl_tau` and the dynamics use it directly.

The position axes drop out entirely. Nothing in the dynamics depends on where
the robot is, so sweeping the body-frame velocity *direction* over a full
turn covers every (heading, field velocity) pair at a given speed, and the
displacement it produces is the same wherever the cell sits.

Steps the backup would reject are rejected here too, and for the same
reasons -- a diverged integration, or a successor outside the velocity
envelope -- because a step that reaches nothing must not be allowed to set
the halo.
"""
function reach_extent(g::Grid6, m::Model, p::Params, ctl::Vector{Float32},
                      nctl::Int; nangle::Int = 48, nsample::Int = 512)
    us = reach_commands(ctl, nctl; nsample = nsample)

    # The envelope `interp` tests the successor against, on the same axes and
    # in the same frame.
    v4hi = Float64(g.lo[4] + g.step[4] * (g.n[4] - 1)); v4lo = Float64(g.lo[4])
    v5hi = Float64(g.lo[5] + g.step[5] * (g.n[5] - 1)); v5lo = Float64(g.lo[5])
    v6hi = Float64(g.lo[6] + g.step[6] * (g.n[6] - 1)); v6lo = Float64(g.lo[6])
    # A step longer than the field's diagonal cannot land on the field from
    # anywhere on it, whatever its velocity. Belt and braces behind the
    # envelope test, and the thing that keeps a `vclamp` run finite.
    span = sqrt((Float64(g.step[1]) * (g.n[1] - 1))^2 +
                (Float64(g.step[2]) * (g.n[2] - 1))^2)

    # Speeds out to the far corner of the velocity box, sampled at half a
    # velocity cell. The corner matters: a grid point at (vmax, vmax) has
    # speed sqrt(2)*vmax, and that is a real starting state.
    vmax = max(Float64(g.lo[4] + g.step[4] * (g.n[4] - 1)), Float64(-g.lo[4]),
               Float64(g.lo[5] + g.step[5] * (g.n[5] - 1)), Float64(-g.lo[5]))
    spmax = sqrt(2.0) * vmax
    dsp = 0.5 * min(Float64(g.step[4]), Float64(g.step[5]))
    nsp = max(2, ceil(Int, spmax / max(dsp, 1e-6)) + 1)
    speeds = collect(range(0.0, spmax; length = nsp))
    # Spins: the w axis exactly. cfl_tau and the dynamics both use it
    # directly, and it is only n6 points.
    spins = [Float64(axisvalue(g, 6, i)) for i in 0:(Int(g.n[6]) - 1)]

    # Reduced per speed rather than per thread. Indexing a scratch buffer by
    # `threadid()` is the usual shape for this and is wrong on any recent
    # Julia: a task can migrate between threads mid-loop, and `nthreads()`
    # does not count the interactive pool, so the id can be out of range
    # before it is even unsafe.
    bxy = zeros(Float64, length(speeds))
    bh  = zeros(Float64, length(speeds))

    Threads.@threads for si in eachindex(speeds)
        t = si
        sp = speeds[si]
        for w in spins
            # cfl_tau reads the field velocity only through its magnitude, so
            # one direction fixes the horizon set for this (speed, spin).
            ts = horizon_set(g, m, p, Float32(sp), 0.0f0, Float32(w))
            for ai in 0:(nangle - 1)
                # Body-frame velocity direction. Sweeping it covers every
                # (heading, field velocity) pair of this speed, which is what
                # lets the position axes drop out of the scan entirely.
                th = 2pi * ai / nangle
                vx = Float32(sp * cos(th)); vy = Float32(sp * sin(th))
                for u in us, tau in ts
                    ns = substeps_for(tau, p)
                    nx, ny, nh, nvx, nvy, nw = p.rk2 ?
                        step_state_rk2(m, 0.0f0, 0.0f0, 0.0f0, vx, vy,
                                       Float32(w), u[1], u[2], u[3], tau, ns) :
                        step_state(m, 0.0f0, 0.0f0, 0.0f0, vx, vy,
                                   Float32(w), u[1], u[2], u[3], tau, ns)
                    # A diverged step is rejected by `lookahead` before any
                    # table read, so it reaches nothing and must not be let
                    # near the halo -- one NaN would make it the whole grid.
                    (isfinite(nx) && isfinite(ny) && isfinite(nh) &&
                     isfinite(nvx) && isfinite(nvy) && isfinite(nw)) || continue
                    d = sqrt(Float64(nx)^2 + Float64(ny)^2)
                    d > span && continue
                    if !p.vclamp
                        # The successor as `lookahead` hands it to `interp`:
                        # body velocity rotated back to the field frame
                        # through the NEW heading. Out of the envelope on any
                        # axis and the read never happens.
                        snh, cnh = sincos(nh)
                        fx = Float64(cnh * nvx - snh * nvy)
                        fy = Float64(snh * nvx + cnh * nvy)
                        (v4lo <= fx <= v4hi) || continue
                        (v5lo <= fy <= v5hi) || continue
                        (v6lo <= Float64(nw) <= v6hi) || continue
                    end
                    d > bxy[t] && (bxy[t] = d)
                    a = abs(Float64(nh))
                    a > bh[t] && (bh[t] = a)
                end
            end
        end
    end
    (maximum(bxy), maximum(bh))
end

"""
Max displacement at each of an explicit ladder of horizons.

`reach_extent` answers "how far can one backup reach at the configured
`tau_max`" and has to mirror `cell_update`'s rung set exactly to do it. This
answers a different question -- "how far at *each* horizon" -- and it is what
makes recommending a `tau_max` affordable. Every candidate `tau_max` needs a
reach, and running a separate scan per candidate repeats the whole outer
sweep over speeds, spins, directions and commands for each one; here the
outer sweep happens once and the horizons ride along on the inside, where the
marginal cost is one integration.

The result is **cumulative**: entry `i` is the largest displacement over
every horizon up to and including `taus[i]`. That is what a halo needs, since
`tau_max` caps the ladder rather than selecting a rung, and it makes the
curve monotone by construction. `taus` is sorted here rather than assumed
sorted, and returned in that order -- a cumulative maximum run down a
*descending* list quietly reports the longest step's reach for every rung,
which is conservative enough to look plausible and wrong enough to make the
recommendation useless.

Reading a reach for `tau_max = T` off this curve is conservative rather than
exact: the real rung set at `T` is a subset of the horizons at or below `T`,
so the curve can only over-state. It over-states by very little in the cases
that bind -- with the default ladder any cell whose top rung would exceed `T`
contributes `clamp(rung, ., T) = T` itself, so `T` really is in the set --
but the tile plan a solve is built from still comes from `reach_extent` at
the chosen value. This one is for choosing.
"""
function reach_curve(g::Grid6, m::Model, p::Params, ctl::Vector{Float32},
                     nctl::Int, taus::Vector{Float64};
                     nangle::Int = 16, nsample::Int = 128)
    us = reach_commands(ctl, nctl; nsample = nsample)
    taus = sort(unique(taus))
    nt = length(taus)

    v4hi = Float64(g.lo[4] + g.step[4] * (g.n[4] - 1)); v4lo = Float64(g.lo[4])
    v5hi = Float64(g.lo[5] + g.step[5] * (g.n[5] - 1)); v5lo = Float64(g.lo[5])
    v6hi = Float64(g.lo[6] + g.step[6] * (g.n[6] - 1)); v6lo = Float64(g.lo[6])
    span = sqrt((Float64(g.step[1]) * (g.n[1] - 1))^2 +
                (Float64(g.step[2]) * (g.n[2] - 1))^2)

    vmax = max(Float64(g.lo[4] + g.step[4] * (g.n[4] - 1)), Float64(-g.lo[4]),
               Float64(g.lo[5] + g.step[5] * (g.n[5] - 1)), Float64(-g.lo[5]))
    spmax = sqrt(2.0) * vmax
    dsp = 0.5 * min(Float64(g.step[4]), Float64(g.step[5]))
    nsp = max(2, ceil(Int, spmax / max(dsp, 1e-6)) + 1)
    speeds = collect(range(0.0, spmax; length = nsp))
    spins = [Float64(axisvalue(g, 6, i)) for i in 0:(Int(g.n[6]) - 1)]

    # One row per speed, reduced at the end, for the same reason
    # `reach_extent` does it: a task can migrate between threads mid-loop, so
    # indexing scratch by `threadid()` is not safe on any recent Julia.
    bxy = zeros(Float64, nt, length(speeds))
    bh  = zeros(Float64, nt, length(speeds))

    Threads.@threads for si in eachindex(speeds)
        sp = speeds[si]
        for w in spins, ai in 0:(nangle - 1)
            th = 2pi * ai / nangle
            vx = Float32(sp * cos(th)); vy = Float32(sp * sin(th))
            for u in us, ti in 1:nt
                tau = Float32(taus[ti])
                ns = substeps_for(tau, p)
                nx, ny, nh, nvx, nvy, nw = p.rk2 ?
                    step_state_rk2(m, 0.0f0, 0.0f0, 0.0f0, vx, vy,
                                   Float32(w), u[1], u[2], u[3], tau, ns) :
                    step_state(m, 0.0f0, 0.0f0, 0.0f0, vx, vy,
                               Float32(w), u[1], u[2], u[3], tau, ns)
                (isfinite(nx) && isfinite(ny) && isfinite(nh) &&
                 isfinite(nvx) && isfinite(nvy) && isfinite(nw)) || continue
                d = sqrt(Float64(nx)^2 + Float64(ny)^2)
                d > span && continue
                if !p.vclamp
                    snh, cnh = sincos(nh)
                    fx = Float64(cnh * nvx - snh * nvy)
                    fy = Float64(snh * nvx + cnh * nvy)
                    (v4lo <= fx <= v4hi) || continue
                    (v5lo <= fy <= v5hi) || continue
                    (v6lo <= Float64(nw) <= v6hi) || continue
                end
                d > bxy[ti, si] && (bxy[ti, si] = d)
                a = abs(Float64(nh))
                a > bh[ti, si] && (bh[ti, si] = a)
            end
        end
    end

    dxy = [maximum(@view bxy[ti, :]) for ti in 1:nt]
    dh  = [maximum(@view bh[ti, :])  for ti in 1:nt]
    for ti in 2:nt                       # cumulative: tau_max caps, not selects
        dxy[ti] = max(dxy[ti], dxy[ti - 1])
        dh[ti]  = max(dh[ti],  dh[ti - 1])
    end
    (taus = taus, dxy_cm = dxy, dh_rad = dh)
end

"""Reach at `tau_max` off a `reach_curve`: the last rung at or below it."""
function reach_at(curve, tau_max::Real)
    i = searchsortedlast(curve.taus, Float64(tau_max))
    i < 1 && (i = 1)
    (curve.dxy_cm[i], curve.dh_rad[i])
end

"""
Halo width, in cells, on each position axis.

Three things are added to the measured displacement:

  * `+1` because the interpolant reads the cell *above* the one the landing
    point falls in -- `bump` steps one index up on every axis it walks.
  * `+1` because the swept collision probe rounds to the nearest occupancy
    cell rather than flooring, so it can touch the cell above as well.
  * `margin`, the safety factor, for the gap between a dense sample of the
    octahedron surface and the continuum the pattern search really searches.

Too small a halo is not a graceful degradation. A read past the loaded window
is answered with `cap`, which prices a perfectly good step as unreachable,
and the damage appears as a faint lattice of seams on tile boundaries rather
than as an error -- `cap` is a legal value everywhere else in the solver.
`verify_halo` is what stops that shipping.
"""
function halo_cells(g::Grid6, dxy_cm::Real; margin::Int = 2)
    hx = ceil(Int, Float64(dxy_cm) / Float64(g.step[1])) + 2 + margin
    hy = ceil(Int, Float64(dxy_cm) / Float64(g.step[2])) + 2 + margin
    (min(hx, Int(g.n[1])), min(hy, Int(g.n[2])))
end

# --------------------------------------------------------------------------
# Tile decomposition
# --------------------------------------------------------------------------

"""
How the grid is cut up for an out-of-core solve.

The cut is over `x` and `y` only, and that is forced by the dynamics rather
than chosen for convenience:

  * `x`, `y` -- reach is bounded by the CFL horizon, and they are the two
    slowest-varying axes of the row-major index, so a tile is a set of long
    contiguous runs rather than a scatter. Cuttable.
  * `vx`, `vy` -- **not** cuttable. The successor's velocity is rotated back
    to the field frame through the *new* heading, so a step that turns by a
    quarter turn carries (vx, 0) to (0, vx): the reach spans the whole
    velocity plane, and the only halo that covers it is the entire axis.
  * `h`, `w` -- reach is bounded, but by the same CFL argument it is a large
    fraction of those axes, which are short. Cutting them would cost more in
    halo than it returns in residency. Kept whole.

`col` is the cell count of one (x, y) column: the granularity of every
transfer between the store and the device.
"""
struct TilePlan
    hx::Int              # halo, cells, x
    hy::Int              # halo, cells, y
    wx::Int              # interior extent, cells, x
    wy::Int              # interior extent, cells, y
    nxl::Int             # worst-case loaded extent, x  (wx + 2hx, clipped)
    nyl::Int             # worst-case loaded extent, y
    col::Int64           # cells per (x, y) column
    bytes::Int64         # worst-case resident bytes
    ntiles::Int
    amplification::Float64   # cells loaded per cell updated
    dxy_cm::Float64
    dh_rad::Float64
end

_loaded(w, h, n) = min(w + 2h, n)

function _tile_bytes(g::Grid6, wx::Int, wy::Int, hx::Int, hy::Int, warm::Bool)
    col = ncells(g) ÷ (Int64(g.n[1]) * Int64(g.n[2]))
    nxl = _loaded(wx, hx, Int(g.n[1]))
    nyl = _loaded(wy, hy, Int(g.n[2]))
    loaded = Int64(nxl) * Int64(nyl) * col
    # V is four bytes a cell over the loaded window. The warm-start policy is
    # three more -- one byte a command component, read in place by `pol_get` --
    # and is allocated over the same window rather than over the interior,
    # because `cell_update` addresses it with the same tile-local index it
    # addresses V with. Occupancy is a byte per (x, y, h).
    v = loaded * cell_bytes(warm)
    o = Int64(nxl) * Int64(nyl) * Int64(g.n[3])
    (v + o, nxl, nyl, col)
end

"""
Largest tile that fits in `budget` bytes.

The interior grows along both axes together, in proportion to the axis
lengths, so tiles stay roughly square *in cells* -- which is what minimises
halo area for a given interior volume. Bisecting one scalar keeps that
property automatic instead of making it a second search.
"""
function plan_tiles(g::Grid6, budget::Int64, hx::Int, hy::Int;
                    warm::Bool = false, dxy_cm = NaN, dh_rad = NaN)
    n1 = Int(g.n[1]); n2 = Int(g.n[2])
    fits(s) = begin
        wx = clamp(round(Int, s * n1), 1, n1)
        wy = clamp(round(Int, s * n2), 1, n2)
        b, _, _, _ = _tile_bytes(g, wx, wy, hx, hy, warm)
        (b <= budget, wx, wy)
    end
    wx = n1; wy = n2
    if !first(fits(1.0))
        first(fits(1.0 / max(n1, n2))) || return nothing   # not one column fits
        lo, hi = 1.0 / max(n1, n2), 1.0
        for _ in 1:48
            mid = 0.5 * (lo + hi)
            first(fits(mid)) ? (lo = mid) : (hi = mid)
        end
        _, wx, wy = fits(lo)
    end
    tile_plan(g, wx, wy, hx, hy; warm = warm, dxy_cm = dxy_cm, dh_rad = dh_rad)
end

"""A `TilePlan` for an interior size chosen by hand, rather than by budget."""
function tile_plan(g::Grid6, wx::Int, wy::Int, hx::Int, hy::Int;
                   warm::Bool = false, dxy_cm = NaN, dh_rad = NaN)
    b, nxl, nyl, col = _tile_bytes(g, wx, wy, hx, hy, warm)
    ntiles = cld(Int(g.n[1]), wx) * cld(Int(g.n[2]), wy)
    amp = (Float64(nxl) * nyl) / (Float64(wx) * wy)
    TilePlan(hx, hy, wx, wy, nxl, nyl, col, b, ntiles, amp,
             Float64(dxy_cm), Float64(dh_rad))
end

"""One tile: the cells it updates, and the cells it loads to do that."""
struct TileSpec
    ix0::Int; iy0::Int       # interior origin, global cells
    wx::Int;  wy::Int        # interior extent
    lx0::Int; ly0::Int       # loaded origin, global cells
    nxl::Int; nyl::Int       # loaded extent
end

"""
Cut the grid into tiles, with the boundaries offset by `shift` cells.

Two properties, and the driver depends on both.

**Every cell is interior to exactly one tile.** The out-of-core solve is a
reordering of the in-core one and nothing else, which stops being true the
moment a cell is updated twice or not at all -- and "not at all" is the
dangerous one, because the missed cells simply keep their last value and the
result still looks like a converged table. Self-test 9 checks this directly,
at every shift, rather than inferring it from the value produced.

**The boundaries move between rounds.** `shift` shortens the first cut, so
every later boundary slides with it and a cell frozen on a tile edge this
round is mid-tile the next. That is worth much more than it sounds: with the
tiling pinned and no halo at all, mean value in the self-test went from
1.80 s to 14.33 s, and allowing the shift alone brought it back to 1.79 s.
The halo still earns its place on the grids a full-scale run actually uses,
where the tile is barely wider than the halo and no shift can put a cell
clear of every boundary.

The loaded window is the interior grown by the halo and clipped to the grid,
so it is never wider than the `nxl` by `nyl` the `TilePlan` was sized for --
which is what the device buffers were allocated against.
"""
function tiles_of(g::Grid6, tp::TilePlan; shift::Int = 0)
    n1 = Int(g.n[1]); n2 = Int(g.n[2])
    # The first cut is short by `shift`, and every later one follows it. A
    # tile as wide as its own axis has nowhere to slide to, hence the guard:
    # `mod` by one is zero anyway, but a zero-width first tile would not
    # terminate.
    cuts(n, w, s) = begin
        out = Tuple{Int,Int}[]
        x = 0
        step = w <= 1 ? w : w - mod(s, w)
        while x < n
            width = min(step, n - x)
            push!(out, (x, width))
            x += width
            step = w
        end
        out
    end
    xs = cuts(n1, tp.wx, shift)
    ys = cuts(n2, tp.wy, shift)

    tiles = TileSpec[]
    for (x0, wx) in xs, (y0, wy) in ys
        lx0 = max(0, x0 - tp.hx)
        ly0 = max(0, y0 - tp.hy)
        nxl = min(n1, x0 + wx + tp.hx) - lx0
        nyl = min(n2, y0 + wy + tp.hy) - ly0
        push!(tiles, TileSpec(x0, y0, wx, wy, lx0, ly0, nxl, nyl))
    end
    tiles
end

"""
The tile's own grid: same cells, same spacing, origin moved to the window.

`cell_update` is written against a `Grid6` and does not know it is looking at
a window, so handing it one of these is the whole of what lets the tiled
driver reuse the in-core backup unchanged. Only the two position axes move --
the other four are never cut.

The step is carried across **bit for bit** rather than recomputed from a span
and a count, which is why this calls the inner constructor. It has to: a tile
whose cells sat a rounding apart from the grid's would interpolate against
very slightly the wrong lattice, and an error that size is far too small to
see and exactly the size that makes a table quietly wrong.
"""
@inline function subgrid(g::Grid6, x0::Integer, y0::Integer, nx::Integer, ny::Integer)
    n = (Int32(nx), Int32(ny), g.n[3], g.n[4], g.n[5], g.n[6])
    lo = (g.lo[1] + g.step[1] * Float32(x0), g.lo[2] + g.step[2] * Float32(y0),
          g.lo[3], g.lo[4], g.lo[5], g.lo[6])
    Grid6(n, lo, g.step)
end

"""
Is this halo actually wide enough for the reach it was built from?

The last check before a run starts, and it exists because the failure it
catches is invisible in the output. `halo_cells` adds a `margin` on top of
the hard requirement; this checks the hard requirement alone -- the
displacement in cells, plus the one cell the interpolant reads above the
landing point, plus the one the swept collision probe can round up to.
"""
function verify_halo(g::Grid6, tp::TilePlan)
    nx = min(ceil(Int, tp.dxy_cm / Float64(g.step[1])) + 2, Int(g.n[1]))
    ny = min(ceil(Int, tp.dxy_cm / Float64(g.step[2])) + 2, Int(g.n[2]))
    (ok = tp.hx >= nx && tp.hy >= ny, need_x = nx, need_y = ny)
end
