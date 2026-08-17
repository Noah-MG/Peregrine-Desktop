"""
    PeregrineSolver

Minimum-time value functions over a 6D robot state, solved by value iteration
on the Hamilton-Jacobi-Bellman equation.

For each target state the solver produces a table

    V(s) = minimum time to reach the target from state s

covering a grid of states, avoiding obstacles, under the drivetrain model
fitted by `calibration/fit_drivetrain.py`. The table is consumed by the
online optimizer on the robot; see `docs/TABLE_FORMAT.md`.

Why value iteration rather than one trajectory optimization per cell: the
cells are not independent problems. `V` satisfies the Bellman recursion

    V(s) = min over u of [ dt + V(step(s, u, dt)) ]

so one sweep over the whole grid advances every cell at once, and every cell
updates independently -- which is exactly the shape a GPU wants. Solving each
cell separately would repeat almost all of the same work billions of times.
"""
module PeregrineSolver

using Printf
using SHA
using JSON3

export Grid6, Model, solve_target, load_field, load_targets, load_model

const AXES = ("x", "y", "h", "vx", "vy", "w")
const INF32 = Float32(Inf)

# --------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------

"""
Uniform 6D grid over (x, y, h, vx, vy, w).

Axis 3 (heading) is periodic: it spans [-pi, pi) with step 2pi/n, and index
n wraps to 0. Every other axis spans lo..hi inclusive with step (hi-lo)/(n-1).

Kept isbits so it can be passed straight into a GPU kernel.
"""
struct Grid6
    n::NTuple{6,Int32}
    lo::NTuple{6,Float32}
    step::NTuple{6,Float32}
end

function Grid6(n::NTuple{6,Integer}, lo::NTuple{6,Real}, hi::NTuple{6,Real})
    any(n .< 2) && error("every axis needs at least 2 samples, got $n")
    st = ntuple(6) do k
        k == 3 ? Float32(2π / n[k]) : Float32((hi[k] - lo[k]) / (n[k] - 1))
    end
    lo3 = ntuple(k -> k == 3 ? Float32(-π) : Float32(lo[k]), 6)
    Grid6(Int32.(n), lo3, st)
end

ncells(g::Grid6) = prod(Int64.(g.n))

@inline axisvalue(g::Grid6, k::Int, i::Integer) = g.lo[k] + Float32(i) * g.step[k]

"""Decode a 0-based flat row-major index into 0-based per-axis subscripts."""
@inline function unflatten(g::Grid6, idx::Int64)
    n = g.n
    i6 = idx % Int64(n[6]);  r = idx ÷ Int64(n[6])
    i5 = r   % Int64(n[5]);  r ÷= Int64(n[5])
    i4 = r   % Int64(n[4]);  r ÷= Int64(n[4])
    i3 = r   % Int64(n[3]);  r ÷= Int64(n[3])
    i2 = r   % Int64(n[2]);  r ÷= Int64(n[2])
    i1 = r
    (Int32(i1), Int32(i2), Int32(i3), Int32(i4), Int32(i5), Int32(i6))
end

@inline function flatten(g::Grid6, i1, i2, i3, i4, i5, i6)::Int64
    n = g.n
    ((((Int64(i1) * n[2] + i2) * n[3] + i3) * n[4] + i4) * n[5] + i5) * n[6] + i6
end

# --------------------------------------------------------------------------
# Drivetrain model
# --------------------------------------------------------------------------

"""
Body-frame acceleration model from the calibration regression:

    a = B*u + A*v + q*w^2 + c

with u = (fwd, strafe, turn), v = (vx, vy, w) in the body frame. `a` is the
*proper* body-frame acceleration, so integrating it back to a velocity has to
subtract the Coriolis term -- see `step_state`.
"""
struct Model
    B::NTuple{9,Float32}   # row-major 3x3
    A::NTuple{9,Float32}
    q::NTuple{3,Float32}
    c::NTuple{3,Float32}
end

@inline function body_accel(m::Model, u1, u2, u3, vx, vy, w)
    B = m.B; A = m.A; q = m.q; c = m.c
    w2 = w * w
    ax = B[1]*u1 + B[2]*u2 + B[3]*u3 + A[1]*vx + A[2]*vy + A[3]*w + q[1]*w2 + c[1]
    ay = B[4]*u1 + B[5]*u2 + B[6]*u3 + A[4]*vx + A[5]*vy + A[6]*w + q[2]*w2 + c[2]
    al = B[7]*u1 + B[8]*u2 + B[9]*u3 + A[7]*vx + A[8]*vy + A[9]*w + q[3]*w2 + c[3]
    (ax, ay, al)
end

"""
Advance the state by `dt` using `nsub` semi-implicit Euler substeps.

The regression's `a` is the body-frame proper acceleration,
`a = dv/dt + w x v`, so the velocity derivative is `dv/dt = a - w x v`.
Position integrates in the field frame through R(h).
"""
@inline function step_state(m::Model, x, y, h, vx, vy, w, u1, u2, u3,
                            dt::Float32, nsub::Int32)
    hs = dt / Float32(nsub)
    @inbounds for _ in 1:nsub
        ax, ay, al = body_accel(m, u1, u2, u3, vx, vy, w)
        # dv/dt = a - w x v,  with w x v = w*(-vy, vx)
        dvx = ax + w * vy
        dvy = ay - w * vx
        sh, ch = sincos(h)
        x += (ch * vx - sh * vy) * hs
        y += (sh * vx + ch * vy) * hs
        h += w * hs
        vx += dvx * hs
        vy += dvy * hs
        w  += al * hs
    end
    (x, y, h, vx, vy, w)
end

# --------------------------------------------------------------------------
# Control set
# --------------------------------------------------------------------------

"""
Admissible controls for a mecanum drive.

Wheel powers live in [-1,1]^4, and the mixing
`fwd,strafe,turn = (FR+FL+BR+BL, -FR+FL+BR-BL, -FR+FL-BR+BL)/4`
maps that cube onto the octahedron `|fwd| + |strafe| + |turn| <= 1`.

Because the acceleration is affine in u, the minimum-time Hamiltonian is
affine in u as well, so the optimum sits at an extreme point of that
octahedron -- bang-bang. `level == 1` returns exactly those 6 vertices plus
coast, which is the cheapest set that can still represent the optimum.
Higher levels subdivide, which helps only because time is discretized.
"""
function control_set(level::Int)
    level < 1 && error("control level must be >= 1")
    vals = range(-1.0f0, 1.0f0; length = 2 * level + 1)
    out = NTuple{3,Float32}[]
    for f in vals, s in vals, t in vals
        abs(f) + abs(s) + abs(t) <= 1.0f0 + 1.0f-6 || continue
        push!(out, (Float32(f), Float32(s), Float32(t)))
    end
    unique!(out)
    # Put coast first so ties resolve toward doing nothing.
    sort!(out; by = c -> (abs(c[1]) + abs(c[2]) + abs(c[3])))
    out
end

# --------------------------------------------------------------------------
# Interpolation
# --------------------------------------------------------------------------

"""
6D multilinear interpolation of V at a continuous state.

This is the hot spot of the whole solver: 64 corner reads per lookup, with
poor locality. `nearest` trades accuracy for one read, which is a large speed
win on big grids at the cost of a more diffusive value function.

Velocity outside the grid is clamped rather than rejected -- leaving the
velocity box means the model has been pushed past where it was fitted, not
that the state is illegal. Position outside the field returns `cap`.

Unreached cells hold `cap`, a large *finite* value, rather than Inf. That
matters: during value iteration "not computed yet" and "inside an obstacle"
would otherwise be the same number, and any rule that rejects an infinite
corner also blocks the very first wave of propagation, leaving the whole
table empty. With a finite cap, blending against an unreached or blocked
corner merely pulls the estimate up, which is conservative and still
converges monotonically downward to the true value.

Values cannot leak through a wall because transitions into obstacles are
rejected by the swept check in `cell_update`, and obstacle cells sit at
`cap`. That argument needs obstacles to be at least one cell thick -- see
`margin_cm` in `build_occupancy`.
"""
@inline function interp(V, g::Grid6, x, y, h, vx, vy, w, nearest::Bool, cap::Float32)
    st = (x, y, h, vx, vy, w)
    # Fractional grid coordinates.
    f1 = (st[1] - g.lo[1]) / g.step[1]
    f2 = (st[2] - g.lo[2]) / g.step[2]
    # Heading wraps into [0, n3).
    n3 = Float32(g.n[3])
    f3 = (st[3] - g.lo[3]) / g.step[3]
    f3 = f3 - floor(f3 / n3) * n3
    f4 = (st[4] - g.lo[4]) / g.step[4]
    f5 = (st[5] - g.lo[5]) / g.step[5]
    f6 = (st[6] - g.lo[6]) / g.step[6]

    # Off the field is unreachable.
    (f1 < 0.0f0 || f1 > Float32(g.n[1] - 1)) && return cap
    (f2 < 0.0f0 || f2 > Float32(g.n[2] - 1)) && return cap
    # Past the fitted velocity envelope: clamp.
    f4 = clamp(f4, 0.0f0, Float32(g.n[4] - 1))
    f5 = clamp(f5, 0.0f0, Float32(g.n[5] - 1))
    f6 = clamp(f6, 0.0f0, Float32(g.n[6] - 1))

    if nearest
        i1 = Int32(round(f1)); i2 = Int32(round(f2)); i3 = Int32(round(f3)) % g.n[3]
        i4 = Int32(round(f4)); i5 = Int32(round(f5)); i6 = Int32(round(f6))
        @inbounds return V[flatten(g, i1, i2, i3, i4, i5, i6) + 1]
    end

    b1 = floor(f1); b2 = floor(f2); b3 = floor(f3)
    b4 = floor(f4); b5 = floor(f5); b6 = floor(f6)
    t1 = f1 - b1; t2 = f2 - b2; t3 = f3 - b3
    t4 = f4 - b4; t5 = f5 - b5; t6 = f6 - b6
    i1 = Int32(b1); i2 = Int32(b2); i3 = Int32(b3)
    i4 = Int32(b4); i5 = Int32(b5); i6 = Int32(b6)

    acc = 0.0f0
    @inbounds for d in 0:63
        c1 = (d      ) & 1; c2 = (d >> 1) & 1; c3 = (d >> 2) & 1
        c4 = (d >> 3) & 1; c5 = (d >> 4) & 1; c6 = (d >> 5) & 1
        wgt = (c1 == 1 ? t1 : 1.0f0 - t1) *
              (c2 == 1 ? t2 : 1.0f0 - t2) *
              (c3 == 1 ? t3 : 1.0f0 - t3) *
              (c4 == 1 ? t4 : 1.0f0 - t4) *
              (c5 == 1 ? t5 : 1.0f0 - t5) *
              (c6 == 1 ? t6 : 1.0f0 - t6)
        wgt == 0.0f0 && continue
        j1 = min(i1 + c1, g.n[1] - Int32(1))
        j2 = min(i2 + c2, g.n[2] - Int32(1))
        j3 = (i3 + c3) % g.n[3]                       # periodic
        j4 = min(i4 + c4, g.n[4] - Int32(1))
        j5 = min(i5 + c5, g.n[5] - Int32(1))
        j6 = min(i6 + c6, g.n[6] - Int32(1))
        acc += wgt * V[flatten(g, j1, j2, j3, j4, j5, j6) + 1]
    end
    min(acc, cap)
end

# --------------------------------------------------------------------------
# Bellman update for one cell
# --------------------------------------------------------------------------

"""
One Bellman backup. Written to be valid in both a CPU loop and a GPU kernel:
scalar math, no allocation, no dynamic dispatch.
"""
@inline function cell_update(idx::Int64, V, occ, g::Grid6, m::Model,
                             ctl, nctl::Int32, dt::Float32, nsub::Int32,
                             sweep_checks::Int32, nearest::Bool, cap::Float32)
    i1, i2, i3, i4, i5, i6 = unflatten(g, idx)

    # Occupancy is (x, y, heading): a chassis with real extent blocks
    # different cells depending on which way it is pointing.
    @inbounds occ[(Int64(i1) * g.n[2] + i2) * g.n[3] + i3 + 1] && return cap

    x  = axisvalue(g, 1, i1); y   = axisvalue(g, 2, i2)
    h  = axisvalue(g, 3, i3); vfx = axisvalue(g, 4, i4)
    vfy = axisvalue(g, 5, i5); w  = axisvalue(g, 6, i6)

    # The grid stores FIELD-frame velocity, because that is what the robot
    # already has from odometry and it saves a rotation every loop cycle. The
    # drivetrain model is body-frame, so rotate in here and back out below.
    sh0, ch0 = sincos(h)
    vx =  ch0 * vfx + sh0 * vfy
    vy = -sh0 * vfx + ch0 * vfy

    best = cap
    @inbounds for k in 1:nctl
        u1 = ctl[k]; u2 = ctl[k+nctl]; u3 = ctl[k+2*nctl]
        nx, ny, nh, nvx, nvy, nw = step_state(m, x, y, h, vx, vy, w, u1, u2, u3,
                                              dt, nsub)
        # Swept collision check: at speed the step can jump several cells, so
        # sampling only the endpoint would tunnel straight through a wall.
        blocked = false
        for sc in 1:sweep_checks
            a = Float32(sc) / Float32(sweep_checks)
            px = x + (nx - x) * a
            py = y + (ny - y) * a
            ph = h + (nh - h) * a          # the robot turns as it moves
            gx = (px - g.lo[1]) / g.step[1]
            gy = (py - g.lo[2]) / g.step[2]
            if gx < 0.0f0 || gy < 0.0f0 ||
               gx > Float32(g.n[1] - 1) || gy > Float32(g.n[2] - 1)
                blocked = true; break
            end
            n3f = Float32(g.n[3])
            gh = (ph - g.lo[3]) / g.step[3]
            gh = gh - floor(gh / n3f) * n3f
            ci = Int32(round(gx)); cj = Int32(round(gy))
            ck = Int32(round(gh)) % g.n[3]
            if occ[(Int64(ci) * g.n[2] + cj) * g.n[3] + ck + 1]
                blocked = true; break
            end
        end
        blocked && continue

        # Rotate the successor's velocity back to the field frame, using the
        # NEW heading, before looking it up in the grid.
        snh, cnh = sincos(nh)
        nvfx = cnh * nvx - snh * nvy
        nvfy = snh * nvx + cnh * nvy
        nv = interp(V, g, nx, ny, nh, nvfx, nvfy, nw, nearest, cap)
        best = min(best, min(dt + nv, cap))
    end
    best
end

include("Pipeline.jl")
include("Gpu.jl")
include("Run.jl")

end # module
