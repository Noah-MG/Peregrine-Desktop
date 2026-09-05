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

export Grid6, Model, solve_value!, load_field, load_targets, load_model

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

    u = u_raw * traction_gain(u_raw)        wheel slip, applied first
    a = B*u + A*v + q*w^2 + S*csign(v) + D*(|v|*v) + c

with u = (fwd, strafe, turn), v = (vx, vy, w) in the body frame. `a` is the
*proper* body-frame acceleration, so integrating it back to a velocity has to
subtract the Coriolis term -- see `step_state`.

The saturation comes first because the fit was made against the saturated
command: the gains describe force delivered, not force asked for.
"""
struct Model
    B::NTuple{9,Float32}   # row-major 3x3, control
    A::NTuple{9,Float32}   # velocity
    q::NTuple{3,Float32}   # omega^2
    S::NTuple{9,Float32}   # Coulomb, multiplies csign(v)
    D::NTuple{9,Float32}   # quadratic drag, multiplies |v|*v
    c::NTuple{3,Float32}   # constant
    eps::NTuple{3,Float32} # Coulomb smoothing band
    knee::Float32          # traction knee; <= 0 disables saturation
end

"""Smoothed sign: a linear ramp through zero saturating at +-1.

A hard `sign` would flip discontinuously at v = 0 and make the integration
chatter, so the fit and the solver share this smoothed form -- it is part of
the model, not a solver convenience.
"""
@inline csign(v, e) = clamp(v / e, -1.0f0, 1.0f0)

"""Fraction of commanded effort the tyres can actually deliver.

Past `knee` the wheels break loose and extra command buys no extra force. The
direction of the command is preserved; only its magnitude is folded over.
Traction is one shared budget, so the demand counts all three axes and the
result scales all three.
"""
@inline function traction_gain(u1, u2, u3, knee)
    # NaN as well as <= 0 means "no saturation". TOML has no null, so a
    # disabled knee arrives as NaN, and `NaN <= 0` is false -- without this
    # check the gain becomes NaN, every control becomes NaN, and the kernel
    # indexes the table with garbage.
    (isnan(knee) || knee <= 0.0f0) && return 1.0f0
    m = sqrt(u1 * u1 + u2 * u2 + u3 * u3)
    r = m / knee
    r < 1.0f-6 && return 1.0f0
    tanh(r) / r
end

"""Apply the traction saturation to a raw command, once.

Split out of `body_accel` because the command is constant across the
substeps of one Bellman lookahead, while the velocity is not: computing
`sqrt` and `tanh` inside the substep loop repeated the same answer `nsub`
times per control. Hoisting it is exactly value-preserving.
"""
@inline function saturate(m::Model, u1r, u2r, u3r)
    gn = traction_gain(u1r, u2r, u3r, m.knee)
    (u1r * gn, u2r * gn, u3r * gn)
end

"""Body-frame proper acceleration for an ALREADY-SATURATED command."""
@inline function body_accel_sat(m::Model, u1, u2, u3, vx, vy, w)
    B = m.B; A = m.A; q = m.q; S = m.S; D = m.D; c = m.c; e = m.eps
    w2 = w * w
    sx = csign(vx, e[1]); sy = csign(vy, e[2]); sw = csign(w, e[3])
    dx = abs(vx) * vx;    dy = abs(vy) * vy;    dw = abs(w) * w
    ax = B[1]*u1 + B[2]*u2 + B[3]*u3 + A[1]*vx + A[2]*vy + A[3]*w +
         q[1]*w2 + S[1]*sx + S[2]*sy + S[3]*sw + D[1]*dx + D[2]*dy + D[3]*dw + c[1]
    ay = B[4]*u1 + B[5]*u2 + B[6]*u3 + A[4]*vx + A[5]*vy + A[6]*w +
         q[2]*w2 + S[4]*sx + S[5]*sy + S[6]*sw + D[4]*dx + D[5]*dy + D[6]*dw + c[2]
    al = B[7]*u1 + B[8]*u2 + B[9]*u3 + A[7]*vx + A[8]*vy + A[9]*w +
         q[3]*w2 + S[7]*sx + S[8]*sy + S[9]*sw + D[7]*dx + D[8]*dy + D[9]*dw + c[3]
    (ax, ay, al)
end

"""Body-frame proper acceleration for a raw (unsaturated) command."""
@inline function body_accel(m::Model, u1r, u2r, u3r, vx, vy, w)
    # The fit was made against the saturated command, so the solver has to
    # saturate too or its dynamics will not match the tables.
    u1, u2, u3 = saturate(m, u1r, u2r, u3r)
    body_accel_sat(m, u1, u2, u3, vx, vy, w)
end

"""
Has the integration blown up?

A long lookahead can walk the state into a region where the *fitted model*
is unstable, which is not the same thing as the robot being unstable. The
quadratic drag block is fitted over the speeds the calibration run actually
visited, and extrapolated far enough its omega term changes sign and becomes
positive feedback: past roughly 15 rad/s the fitted `0.312*|w|*w` outruns the
linear `-4.67*w` and runs away. The step-length ladder deliberately tries
horizons long enough to reach that, because trying them is how it finds out
they are not worth taking.

So divergence has to be *detected* rather than prevented -- the caller
rejects a non-finite successor and that candidate simply loses. What must not
happen is reaching `sincos` with an infinite argument: CUDA returns NaN and
carries on, but the CPU throws a DomainError and kills the whole solve. The
GPU hid this bug completely; only running the CPU backend found it.
"""
@inline diverged(x, y, h, vx, vy, w) =
    !isfinite(x + y + h + vx + vy + w)

const NANSTATE = (NaN32, NaN32, NaN32, NaN32, NaN32, NaN32)

"""
Advance the state by `dt` using `nsub` semi-implicit Euler substeps.

The regression's `a` is the body-frame proper acceleration,
`a = dv/dt + w x v`, so the velocity derivative is `dv/dt = a - w x v`.
Position integrates in the field frame through R(h).
"""
@inline function step_state(m::Model, x, y, h, vx, vy, w, u1r, u2r, u3r,
                            dt::Float32, nsub::Int32)
    hs = dt / Float32(nsub)
    # Saturate once: the command does not change across substeps.
    u1, u2, u3 = saturate(m, u1r, u2r, u3r)
    @inbounds for _ in 1:nsub
        diverged(x, y, h, vx, vy, w) && return NANSTATE
        ax, ay, al = body_accel_sat(m, u1, u2, u3, vx, vy, w)
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

"""The full state derivative, for a command that is already saturated."""
@inline function state_deriv(m::Model, x, y, h, vx, vy, w, u1, u2, u3)
    ax, ay, al = body_accel_sat(m, u1, u2, u3, vx, vy, w)
    sh, ch = sincos(h)
    (ch * vx - sh * vy,        # dx
     sh * vx + ch * vy,        # dy
     w,                        # dh
     ax + w * vy,              # dvx   (dv/dt = a - w x v)
     ay - w * vx,              # dvy
     al)                       # dw
end

"""
Advance the state by `dt` using `nsub` explicit-midpoint (RK2) substeps.

Same dynamics as `step_state`, integrated to second order instead of first.
The extra cost is one more acceleration evaluation per substep, which is
arithmetic; the Bellman backup is limited by the 64 scattered reads in
`interp`, so on this kernel it is very nearly free. What it buys is real:
first-order error at this step size is not negligible next to the grid
resolution, and it is systematic rather than random, so it does not average
out over a trajectory.

It matters most exactly where the model is hardest. The Coulomb term is
smoothed over a band of only 5 cm/s, and the drivetrain crosses that band in
about 6 ms under full command -- shorter than a substep. Euler steps straight
over the transition and books the wrong friction for the whole substep;
midpoint at least samples inside it.
"""
@inline function step_state_rk2(m::Model, x, y, h, vx, vy, w, u1r, u2r, u3r,
                                dt::Float32, nsub::Int32)
    hs = dt / Float32(nsub)
    hh = 0.5f0 * hs
    u1, u2, u3 = saturate(m, u1r, u2r, u3r)
    @inbounds for _ in 1:nsub
        diverged(x, y, h, vx, vy, w) && return NANSTATE
        d1x, d1y, d1h, d1vx, d1vy, d1w =
            state_deriv(m, x, y, h, vx, vy, w, u1, u2, u3)
        mx = x + d1x * hh; my = y + d1y * hh; mh = h + d1h * hh
        mvx = vx + d1vx * hh; mvy = vy + d1vy * hh; mw = w + d1w * hh
        # The midpoint is itself fed to sincos, so it needs the same guard as
        # the state proper -- one unstable substep is enough to make it
        # infinite while the state it came from was still finite.
        diverged(mx, my, mh, mvx, mvy, mw) && return NANSTATE
        d2x, d2y, d2h, d2vx, d2vy, d2w =
            state_deriv(m, mx, my, mh, mvx, mvy, mw, u1, u2, u3)
        x  += d2x  * hs
        y  += d2y  * hs
        h  += d2h  * hs
        vx += d2vx * hs
        vy += d2vy * hs
        w  += d2w  * hs
    end
    (x, y, h, vx, vy, w)
end

# --------------------------------------------------------------------------
# Control set
# --------------------------------------------------------------------------

"""
Admissible controls for a mecanum drive.

The robot mixes a command into wheel powers with

    FR = drive + strafe - turn      BR = drive - strafe - turn
    FL = drive - strafe + turn      BL = drive + strafe + turn

and every wheel must stay in [-1, 1]. Those four rows are (1, +-1, +-1) over
all sign combinations, and the largest of `|d +- s +- t|` is `|d|+|s|+|t|`, so
the constraint is exactly the octahedron

    |drive| + |strafe| + |turn| <= 1

**Why the boundary and not just the corners.** Minimum time wants as much
useful acceleration as possible, so the optimum lies on the boundary of that
octahedron -- that is all "bang-bang" actually tells us. It would additionally
land on a *vertex* if the dynamics were affine in u, because the minimum of a
linear function over a polytope sits at a corner. They are not affine: the
traction term scales the command by `tanh(|u|/knee)/(|u|/knee)`, which depends
on the Euclidean length of u, and that length varies over the boundary -- 1.0
at a vertex against 0.577 at a face centre. Corners are therefore saturated
hardest, and a point in the middle of a face can deliver more useful force
than the corner next to it. Sampling only the corners quietly forbids those.

So the boundary is sampled: each of the eight triangular faces is covered by a
lattice of resolution `level`, plus the origin for coasting. Interior points
are not needed, because more command always means more force -- the saturation
is monotone -- so the best point in any direction is the one on the boundary.

    level 1 ->  7 controls (the corners, as before)
    level 2 -> 19    level 3 -> 39    level 4 -> 67
"""
function control_set(level::Int)
    level < 1 && error("control level must be >= 1")
    seen = Set{NTuple{3,Float32}}()
    out = NTuple{3,Float32}[]
    push!(seen, (0.0f0, 0.0f0, 0.0f0))
    push!(out, (0.0f0, 0.0f0, 0.0f0))          # coast
    L = level
    for sd in (-1, 1), ss in (-1, 1), st in (-1, 1)
        for i in 0:L, j in 0:(L - i)
            k = L - i - j
            # a + b + c == 1, so the point sits on the octahedron surface.
            a = Float32(i) / Float32(L)
            b = Float32(j) / Float32(L)
            c = Float32(k) / Float32(L)
            # Normalise -0.0 away so mirrored duplicates collapse.
            p = (a == 0 ? 0.0f0 : Float32(sd) * a,
                 b == 0 ? 0.0f0 : Float32(ss) * b,
                 c == 0 ? 0.0f0 : Float32(st) * c)
            if !(p in seen)
                push!(seen, p)
                push!(out, p)
            end
        end
    end
    # Coast first, then increasing effort, so ties resolve toward doing less.
    sort!(out; by = p -> (abs(p[1]) + abs(p[2]) + abs(p[3]),
                          p[1] * p[1] + p[2] * p[2] + p[3] * p[3]))
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

Velocity outside the grid is REJECTED, like position outside the field.

That is a correction, not a preference. Clamping instead -- charging a state
outside the envelope the value of the nearest state inside it -- prices
exceeding the envelope at exactly zero, and the minimisation finds that
immediately. The fitted model's terminal speed is around 400 cm/s, well
beyond the 150 cm/s a typical grid covers, so "accelerate out of the box" is
both reachable and free: the cells near the velocity boundary come out
optimistic, and the policy they induce drives the robot into a region where
the table says nothing and the regression was never fitted. A rollout on a
clamped table does exactly that -- observed running away to 432 cm/s and then
to NaN.

Rejecting keeps every trajectory the table endorses inside the envelope it
was solved over, which is the honest reading of what a value table covers.
It raises `V` near the velocity boundary, because it removes controls that
were previously free; that is the optimism being taken away, not accuracy
being lost. `vclamp = true` restores the old behaviour.

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

**Both halves are load-bearing, and the second one is easy to lose.** The
swept check rounds to a cell; this blends the corners of the successor's cell,
which is not the same set. A successor just short of a wall reads a corner
inside it, and if that corner ever holds something small the value leaks
across regardless of the collision check. The escape pass puts small numbers
in obstacle cells on purpose, and pays for it by hiding them again -- see
`EscapeView`. Anything else that writes an obstacle cell owes the same debt.
"""
@inline function interp(V, g::Grid6, x, y, h, vx, vy, w, nearest::Bool,
                        cap::Float32, vclamp::Bool = false)
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

    # A non-finite coordinate must be caught before anything is rounded: every
    # comparison against NaN is false, so an unchecked NaN passes all the
    # range tests below and then indexes the table with garbage.
    (isfinite(f1) && isfinite(f2) && isfinite(f3) &&
     isfinite(f4) && isfinite(f5) && isfinite(f6)) || return cap

    # Off the field is unreachable.
    (f1 < 0.0f0 || f1 > Float32(g.n[1] - 1)) && return cap
    (f2 < 0.0f0 || f2 > Float32(g.n[2] - 1)) && return cap
    # Past the velocity envelope: reject, unless explicitly asked to clamp.
    if vclamp
        f4 = clamp(f4, 0.0f0, Float32(g.n[4] - 1))
        f5 = clamp(f5, 0.0f0, Float32(g.n[5] - 1))
        f6 = clamp(f6, 0.0f0, Float32(g.n[6] - 1))
    else
        (f4 < 0.0f0 || f4 > Float32(g.n[4] - 1)) && return cap
        (f5 < 0.0f0 || f5 > Float32(g.n[5] - 1)) && return cap
        (f6 < 0.0f0 || f6 > Float32(g.n[6] - 1)) && return cap
    end

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
# Kuhn / Freudenthal simplex interpolation
# --------------------------------------------------------------------------

"""Descending compare-exchange on (value, axis) pairs."""
@inline cmpx(ta, ka, tb, kb) = ta >= tb ? (ta, ka, tb, kb) : (tb, kb, ta, ka)

"""
Sort six (t, axis) pairs into descending order by `t`.

A 12-comparator sorting network rather than a loop: it is branchless, which
is what a GPU wants, and 12 is the optimal comparator count for six inputs.
"""
@inline function sort6_desc(t1, t2, t3, t4, t5, t6)
    k1 = Int32(1); k2 = Int32(2); k3 = Int32(3)
    k4 = Int32(4); k5 = Int32(5); k6 = Int32(6)
    t1, k1, t6, k6 = cmpx(t1, k1, t6, k6)
    t2, k2, t4, k4 = cmpx(t2, k2, t4, k4)
    t3, k3, t5, k5 = cmpx(t3, k3, t5, k5)
    t2, k2, t3, k3 = cmpx(t2, k2, t3, k3)
    t4, k4, t5, k5 = cmpx(t4, k4, t5, k5)
    t1, k1, t4, k4 = cmpx(t1, k1, t4, k4)
    t3, k3, t6, k6 = cmpx(t3, k3, t6, k6)
    t1, k1, t2, k2 = cmpx(t1, k1, t2, k2)
    t3, k3, t4, k4 = cmpx(t3, k3, t4, k4)
    t5, k5, t6, k6 = cmpx(t5, k5, t6, k6)
    t2, k2, t3, k3 = cmpx(t2, k2, t3, k3)
    t4, k4, t5, k5 = cmpx(t4, k4, t5, k5)
    (t1, t2, t3, t4, t5, t6, k1, k2, k3, k4, k5, k6)
end

"""
Step one grid subscript along axis `k`.

Heading wraps; every other axis clamps at the top index. Clamping is safe
rather than merely convenient: a point sitting exactly on the last grid plane
has `t = 0` on that axis, so the sort puts it last and the vertex that would
step past the edge carries zero barycentric weight. The read still happens --
it just does not contribute.
"""
@inline function bump(g::Grid6, i1, i2, i3, i4, i5, i6, k::Int32)
    k == Int32(1) && return (min(i1 + Int32(1), g.n[1] - Int32(1)), i2, i3, i4, i5, i6)
    k == Int32(2) && return (i1, min(i2 + Int32(1), g.n[2] - Int32(1)), i3, i4, i5, i6)
    k == Int32(3) && return (i1, i2, (i3 + Int32(1)) % g.n[3], i4, i5, i6)   # periodic
    k == Int32(4) && return (i1, i2, i3, min(i4 + Int32(1), g.n[4] - Int32(1)), i5, i6)
    k == Int32(5) && return (i1, i2, i3, i4, min(i5 + Int32(1), g.n[5] - Int32(1)), i6)
    (i1, i2, i3, i4, i5, min(i6 + Int32(1), g.n[6] - Int32(1)))
end

"""Grid step along a runtime axis index, as an explicit chain.

Indexing the `NTuple` with a non-constant would make the compiler either
build a branch tree anyway or spill the tuple to local memory, which on a GPU
is the expensive outcome. Writing the chain out keeps it in registers.
"""
@inline function stepof(g::Grid6, k::Int32)
    k == Int32(1) && return g.step[1]
    k == Int32(2) && return g.step[2]
    k == Int32(3) && return g.step[3]
    k == Int32(4) && return g.step[4]
    k == Int32(5) && return g.step[5]
    g.step[6]
end

"""
6D Kuhn (Freudenthal) simplex interpolation, with the gradient for free.

Returns `(value, dV/dx, dV/dy, dV/dh, dV/dvx, dV/dvy, dV/dw)`.

**Why this and not the 64-corner multilinear.** Two independent reasons, and
the second is the important one.

*It is far cheaper.* A unit cube in 6D splits into 6! = 720 simplices, each
with 7 vertices, so a lookup reads 7 values instead of 64. The Bellman backup
is bound by exactly these scattered reads, so this is close to a 9x cut in
the hot loop.

*It is what the robot uses.* The online optimizer recovers `grad(V)` by
locating the simplex containing its state, taking the 7 vertices and
finite-differencing them. `V` is the fixed point of whichever interpolant the
Bellman update is written with -- so solving with multilinear and reading with
simplex produces a table that is self-consistent under an operator nobody
ever applies. Matching the two is a correctness argument, not a performance
one; the speed is a bonus.

**Construction.** Sort the six fractional coordinates descending,
`t_s(1) >= ... >= t_s(6)`. The simplex has vertices `v0 = floor(f)` and
`v_k = v_(k-1) + e_s(k)`, so the walk from the low corner to the high corner
takes the axes in order of how far into the cell the point sits. Barycentric
weights are the consecutive differences, `L0 = 1 - t_s(1)`,
`L_k = t_s(k) - t_s(k+1)`, `L6 = t_s(6)`; they sum to one and are
non-negative precisely because the sort is descending.

That makes the value a convex combination of stored cell values, exactly as
multilinear is, so the Bellman operator stays monotone and value iteration
still converges downward to a fixed point. The decomposition also agrees on
shared cell faces, so the interpolant is globally continuous -- neighbouring
cells triangulate their common face the same way.

The gradient falls out with no extra reads: `V` is affine on the simplex, and
consecutive vertices differ by one step along one axis, so
`dV/dx_s(k) = (V(v_k) - V(v_(k-1))) / step_s(k)`.
"""
@inline function interp_kuhn(V, g::Grid6, x, y, h, vx, vy, w,
                             cap::Float32, vclamp::Bool = false)
    z = 0.0f0
    f1 = (x - g.lo[1]) / g.step[1]
    f2 = (y - g.lo[2]) / g.step[2]
    n3 = Float32(g.n[3])
    f3 = (h - g.lo[3]) / g.step[3]
    f3 = f3 - floor(f3 / n3) * n3
    f4 = (vx - g.lo[4]) / g.step[4]
    f5 = (vy - g.lo[5]) / g.step[5]
    f6 = (w - g.lo[6]) / g.step[6]

    # NaN before anything is rounded: every comparison against NaN is false,
    # so an unchecked NaN passes all the range tests and indexes with garbage.
    (isfinite(f1) && isfinite(f2) && isfinite(f3) &&
     isfinite(f4) && isfinite(f5) && isfinite(f6)) ||
        return (cap, z, z, z, z, z, z)

    (f1 < 0.0f0 || f1 > Float32(g.n[1] - 1)) && return (cap, z, z, z, z, z, z)
    (f2 < 0.0f0 || f2 > Float32(g.n[2] - 1)) && return (cap, z, z, z, z, z, z)
    if vclamp
        f4 = clamp(f4, 0.0f0, Float32(g.n[4] - 1))
        f5 = clamp(f5, 0.0f0, Float32(g.n[5] - 1))
        f6 = clamp(f6, 0.0f0, Float32(g.n[6] - 1))
    else
        (f4 < 0.0f0 || f4 > Float32(g.n[4] - 1)) && return (cap, z, z, z, z, z, z)
        (f5 < 0.0f0 || f5 > Float32(g.n[5] - 1)) && return (cap, z, z, z, z, z, z)
        (f6 < 0.0f0 || f6 > Float32(g.n[6] - 1)) && return (cap, z, z, z, z, z, z)
    end

    b1 = floor(f1); b2 = floor(f2); b3 = floor(f3)
    b4 = floor(f4); b5 = floor(f5); b6 = floor(f6)
    i1 = Int32(b1); i2 = Int32(b2); i3 = Int32(b3) % g.n[3]
    i4 = Int32(b4); i5 = Int32(b5); i6 = Int32(b6)

    s1, s2, s3, s4, s5, s6, a1, a2, a3, a4, a5, a6 =
        sort6_desc(f1 - b1, f2 - b2, f3 - b3, f4 - b4, f5 - b5, f6 - b6)

    # Walk the simplex from the low corner to the high one, accumulating the
    # value and keeping each edge's difference for the gradient.
    @inbounds v0 = V[flatten(g, i1, i2, i3, i4, i5, i6) + 1]
    acc = (1.0f0 - s1) * v0
    prev = v0

    # Unrolled by hand. A `for k in 1:6` over tuples would index them with a
    # runtime value, and the usual result of that on a GPU is the tuple going
    # to local memory.
    @inbounds begin
        i1, i2, i3, i4, i5, i6 = bump(g, i1, i2, i3, i4, i5, i6, a1)
        v = V[flatten(g, i1, i2, i3, i4, i5, i6) + 1]
        acc += (s1 - s2) * v
        e1 = (v - prev) / stepof(g, a1); prev = v

        i1, i2, i3, i4, i5, i6 = bump(g, i1, i2, i3, i4, i5, i6, a2)
        v = V[flatten(g, i1, i2, i3, i4, i5, i6) + 1]
        acc += (s2 - s3) * v
        e2 = (v - prev) / stepof(g, a2); prev = v

        i1, i2, i3, i4, i5, i6 = bump(g, i1, i2, i3, i4, i5, i6, a3)
        v = V[flatten(g, i1, i2, i3, i4, i5, i6) + 1]
        acc += (s3 - s4) * v
        e3 = (v - prev) / stepof(g, a3); prev = v

        i1, i2, i3, i4, i5, i6 = bump(g, i1, i2, i3, i4, i5, i6, a4)
        v = V[flatten(g, i1, i2, i3, i4, i5, i6) + 1]
        acc += (s4 - s5) * v
        e4 = (v - prev) / stepof(g, a4); prev = v

        i1, i2, i3, i4, i5, i6 = bump(g, i1, i2, i3, i4, i5, i6, a5)
        v = V[flatten(g, i1, i2, i3, i4, i5, i6) + 1]
        acc += (s5 - s6) * v
        e5 = (v - prev) / stepof(g, a5); prev = v

        i1, i2, i3, i4, i5, i6 = bump(g, i1, i2, i3, i4, i5, i6, a6)
        v = V[flatten(g, i1, i2, i3, i4, i5, i6) + 1]
        acc += s6 * v
        e6 = (v - prev) / stepof(g, a6)

        # Scatter each edge difference back to the axis it stepped along.
        d1 = a1 == Int32(1) ? e1 : a2 == Int32(1) ? e2 : a3 == Int32(1) ? e3 :
             a4 == Int32(1) ? e4 : a5 == Int32(1) ? e5 : e6
        d2 = a1 == Int32(2) ? e1 : a2 == Int32(2) ? e2 : a3 == Int32(2) ? e3 :
             a4 == Int32(2) ? e4 : a5 == Int32(2) ? e5 : e6
        d3 = a1 == Int32(3) ? e1 : a2 == Int32(3) ? e2 : a3 == Int32(3) ? e3 :
             a4 == Int32(3) ? e4 : a5 == Int32(3) ? e5 : e6
        d4 = a1 == Int32(4) ? e1 : a2 == Int32(4) ? e2 : a3 == Int32(4) ? e3 :
             a4 == Int32(4) ? e4 : a5 == Int32(4) ? e5 : e6
        d5 = a1 == Int32(5) ? e1 : a2 == Int32(5) ? e2 : a3 == Int32(5) ? e3 :
             a4 == Int32(5) ? e4 : a5 == Int32(5) ? e5 : e6
        d6 = a1 == Int32(6) ? e1 : a2 == Int32(6) ? e2 : a3 == Int32(6) ? e3 :
             a4 == Int32(6) ? e4 : a5 == Int32(6) ? e5 : e6
    end

    (min(acc, cap), d1, d2, d3, d4, d5, d6)
end

# --------------------------------------------------------------------------
# Warm-start policy storage
# --------------------------------------------------------------------------

"""
Bytes a resident cell costs, by driver. **The one place that decides this.**

`V` is a `Float32`, and the warm-start policy is three `Int8` -- one per
command component -- in *both* drivers, so both answer 7.

They did not always. The in-core driver kept the policy as three `Float32`
planes, which cost 16 bytes a cell, while the tiled driver quantised to
bytes; `decompose` sized the whole-grid case with the tiled driver's number
and `run_solve` then allocated with the in-core one. A grid between
`budget / 16` and `budget / 7` was therefore routed in core and could not be
allocated -- and on Windows it did not even fail, because WDDM pages VRAM
into host RAM, so the desktop silently ran at half speed and only Linux
raised it.

Hence one function. If a driver's layout changes, this number changes with
it, and self-test 15 asserts that what the drivers *allocate* still matches
what this claims.
"""
value_bytes_per_cell() = 4
policy_bytes_per_cell() = 3
cell_bytes(warm::Bool) =
    value_bytes_per_cell() + (warm ? policy_bytes_per_cell() : 0)

"""
Read and write the warm-start policy, in either of the two layouts it is
kept in.

`Float32` in three planes is what the in-core solver uses: `V` and the policy
are both the size of the grid, the grid fits in VRAM by assumption, and
planes give perfectly coalesced access.

`Int8` interleaved per cell is what the tiled solver uses, and both halves of
that matter. **Bytes**, because out-of-core the policy is read and written
again on every round, and three bytes carry this quantity perfectly well: the
policy is not the answer, it is the seed the pattern search starts from, and
the search's own first step is `delta0` -- 0.35 by default, fifty times
coarser than the 1/127 a byte resolves. Every candidate is still evaluated
exactly by `lookahead`, so the quantisation cannot move the fixed point, only
the route to it. **Interleaved**, because a tile's window is then one
contiguous run per x plane exactly as `V`'s is, and moves between store and
device in the same handful of copies rather than three times as many.

Both layouts go through these two functions so that `cell_update` does not
have to know which it was handed.
"""
@inline function pol_get(pol::AbstractVector{Float32}, idx::Int64, n::Int64)
    @inbounds (pol[idx + 1], pol[idx + 1 + n], pol[idx + 1 + 2 * n])
end

@inline function pol_get(pol::AbstractVector{Int8}, idx::Int64, n::Int64)
    b = 3 * idx
    @inbounds (Float32(pol[b + 1]) * (1.0f0 / 127.0f0),
               Float32(pol[b + 2]) * (1.0f0 / 127.0f0),
               Float32(pol[b + 3]) * (1.0f0 / 127.0f0))
end

@inline function pol_set!(pol::AbstractVector{Float32}, idx::Int64, n::Int64,
                          u1, u2, u3)
    @inbounds pol[idx + 1] = u1
    @inbounds pol[idx + 1 + n] = u2
    @inbounds pol[idx + 1 + 2 * n] = u3
    nothing
end

@inline function pol_set!(pol::AbstractVector{Int8}, idx::Int64, n::Int64,
                          u1, u2, u3)
    q(v) = unsafe_trunc(Int8, round(clamp(v, -1.0f0, 1.0f0) * 127.0f0))
    b = 3 * idx
    @inbounds pol[b + 1] = q(u1)
    @inbounds pol[b + 2] = q(u2)
    @inbounds pol[b + 3] = q(u3)
    nothing
end

"""Value only, from whichever interpolant `p` selects."""
@inline function value_at(V, g::Grid6, x, y, h, vx, vy, w, p)
    p.simplex && return interp_kuhn(V, g, x, y, h, vx, vy, w, p.cap, p.vclamp)[1]
    interp(V, g, x, y, h, vx, vy, w, p.nearest, p.cap, p.vclamp)
end

# --------------------------------------------------------------------------
# The escape pass: a second value function, in the same array
# --------------------------------------------------------------------------

"""
`V` under the escape pass's reading of it.

**The problem this solves.** After a target converges, every cell the solve
could not reach sits at `cap`, and a robot that finds itself in one -- shoved
into an obstacle, or a footprint-width from a wall at the wrong heading --
reads `unreachable` in every direction and has no gradient to follow out.
The escape pass gives those cells the time to reach the nearest state that
*does* have a route, so there is always something to descend.

That is a second value function over the same grid, and at 28.6 GB a table it
cannot have a second array. So it lives in the sign bit of the first one:

| stored `V` | means |
| --- | --- |
| `0 <= v < cap` | a real route to the target, `v` seconds. The escape pass's terminal set, and it never writes these. |
| `v < 0` | no route, but a way out: escape takes `-v` seconds |
| `v >= cap` | no route and no way out yet |

Negation is what keeps the two separable. Without it an escape value of 0.3 s
and a real value of 0.3 s are the same bits, and the *next* sweep cannot tell
whether the cell it is reading is a boundary condition or a work in progress.

`getindex` is the whole transform: a terminal cell reads as **zero**, because
reaching one ends the escape and costs nothing more; an unsolved cell reads as
`cap`; an escape cell reads as its escape time. Wrapping `V` rather than
branching inside the interpolants is deliberate -- `interp` and `interp_kuhn`
are the hot loop and are not touched by any of this, they just index something
whose `getindex` does one compare. Constructed inside the kernel from the
device array, so it stays `isbits` and needs no `Adapt` rule.

**And it has to hide obstacle cells from everyone standing outside one.** The
argument in `interp` that values cannot leak through a wall has two halves:
the swept check rejects transitions into an obstacle, *and obstacle cells sit
at `cap`*. This pass breaks the second half on purpose -- it is what puts small
numbers in them -- and the swept check alone does not close the gap, because
interpolation reads the corners of the successor's cell rather than the cell
the collision probe rounded to. A successor half a cell short of a wall
therefore blends a corner that is inside it.

Measured, before this was here: 95,940 free cells on the far side of a slab
were handed an escape that led straight into the slab. They were cells with
genuinely no way out, and the pass invented one through the obstacle.

So a corner that is an obstacle reads as `cap` -- unless `ignore`, meaning the
cell being updated is itself inside an obstacle and is allowed to see its way
out through obstacle space. Exactly the rule `OccView` applies to transitions,
applied to the value read. The occupancy index is `idx / (Nvx*Nvy*Nw)`, one
division: occupancy is indexed by (x, y, h) and those are the slowest three
axes of the flat index, so the velocity block divides straight out.
"""
struct EscapeView{A,O} <: AbstractVector{Float32}
    V::A
    occ::O
    blk::Int64          # cells per (x, y, h) slice: Nvx * Nvy * Nw
    cap::Float32
    ignore::Bool        # the cell being updated is itself blocked
end

Base.size(e::EscapeView) = size(e.V)
Base.IndexStyle(::Type{<:EscapeView}) = IndexLinear()

Base.@propagate_inbounds function Base.getindex(e::EscapeView, i::Integer)
    if !e.ignore
        @inbounds e.occ[(Int64(i) - Int64(1)) ÷ e.blk + Int64(1)] && return e.cap
    end
    v = e.V[i]
    v < 0.0f0 ? -v : (v < e.cap ? 0.0f0 : e.cap)
end

"""
Occupancy as the escape pass reads it, which is not always as it is.

A cell inside an obstacle has to be allowed to move *through* obstacle space,
or there is no way out of one and the pass computes nothing for exactly the
states it exists to serve. A cell that is merely unreached must not: it is
standing in free space, and telling it to escape through a wall would be
worse advice than the `unreachable` it gets today.

So the rule is **you may stay in an obstacle, but you may not enter one** --
the swept check is waived only when the cell being updated is itself blocked.
One flag rather than two types, because a GPU kernel that picks between two
different occupancy representations per cell is type-unstable and will not
compile.
"""
struct OccView{A} <: AbstractVector{Bool}
    occ::A
    ignore::Bool
end

Base.size(o::OccView) = size(o.occ)
Base.IndexStyle(::Type{<:OccView}) = IndexLinear()

Base.@propagate_inbounds Base.getindex(o::OccView, i::Integer) =
    o.ignore ? false : o.occ[i]

"""
Is this cell part of the escape pass's terminal set -- a state with a real
route, which is frozen and read as zero?

`v >= 0 && v < cap`. The `v >= 0` half is what excludes an escape cell whose
value has already been written, and it is the reason the sign convention
exists at all.
"""
@inline is_terminal(v::Float32, cap::Float32) = v >= 0.0f0 && v < cap

"""
The escape time a cell currently claims, from its stored value: `-v` if one
has been found, and `cap` -- no way out yet -- otherwise.
"""
@inline escape_of(v::Float32, cap::Float32) = v < 0.0f0 ? -v : cap

"""
Swap in the escape pass's views, or don't.

By dispatch rather than by a ternary, so that the substitution is settled at
compile time and the ordinary sweep is provably unchanged: for `Val{false}`
these are the identity, they inline away, and `cell_update` indexes the bare
arrays exactly as it did before the escape pass existed. A runtime branch
would leave a `Union` in the hot loop, which on the GPU is the difference
between a register and a spill.
"""
@inline esc_v(V, occ, blk::Int64, cap::Float32, blocked::Bool, ::Val{false}) = V
@inline esc_v(V, occ, blk::Int64, cap::Float32, blocked::Bool, ::Val{true}) =
    EscapeView(V, occ, blk, cap, blocked)
@inline esc_occ(occ, blocked::Bool, ::Val{false}) = occ
@inline esc_occ(occ, blocked::Bool, ::Val{true}) = OccView(occ, blocked)

# --------------------------------------------------------------------------
# Bellman update for one cell
# --------------------------------------------------------------------------

"""
Everything the Bellman backup needs that is not the grid, the model or the
arrays. `isbits`, so it passes straight into a GPU kernel as one argument
instead of a dozen.

The defaults reproduce the original fixed scheme exactly: `ncoarse = 0` means
"scan the whole control table", `rounds = 0` disables the refinement, and
`ntau = 1` pins the lookahead to a single `dt`.
"""
struct Params
    dt::Float32            # fallback horizon when cfl <= 0
    nsub::Int32
    checks::Int32          # floor on the number of swept collision probes
    adaptive_checks::Bool  # scale the probe count with how far the step moves
    nearest::Bool
    cap::Float32
    ntau::Int32            # how many step lengths each cell tries
    tau_ratio::Float32     # ratio between consecutive step lengths
    cfl::Float32           # grid cells a step should advance (0 = use dt)
    tau_min::Float32
    tau_max::Float32
    hmax::Float32          # longest integration substep allowed
    ncoarse::Int32         # lattice controls scanned per sweep (0 = all)
    rounds::Int32          # pattern-search refinement rounds
    delta0::Float32        # first pattern-search radius
    rk2::Bool              # midpoint integration instead of Euler
    vclamp::Bool           # clamp (rather than reject) out-of-envelope speeds
    simplex::Bool          # Kuhn simplex interpolation instead of multilinear
end

Params(; dt = 0.05f0, nsub = Int32(4), checks = Int32(3),
         adaptive_checks = true, nearest = false, cap = 60.0f0,
         ntau = Int32(1), tau_ratio = 2.0f0, cfl = 0.0f0,
         tau_min = 0.004f0, tau_max = 0.5f0, hmax = 0.0125f0,
         ncoarse = Int32(0), rounds = Int32(0), delta0 = 0.35f0,
         rk2 = true, vclamp = false, simplex = true) =
    Params(Float32(dt), Int32(nsub), Int32(checks), adaptive_checks, nearest,
           Float32(cap), Int32(ntau), Float32(tau_ratio), Float32(cfl),
           Float32(tau_min), Float32(tau_max), Float32(hmax), Int32(ncoarse),
           Int32(rounds), Float32(delta0), rk2, vclamp, simplex)

"""
The lookahead horizon this cell should use, from the grid rather than a
global constant.

**Why a single `dt` cannot be right everywhere.** The Bellman backup learns
nothing from a step that does not leave the cell it started in: the
interpolated successor is then mostly the cell's own value, the update is
dominated by its own numerical diffusion, and the iteration crawls toward an
answer that is too pessimistic. Push the step the other way and the
constant-command assumption starts costing more than the diffusion it saves.
The step that balances the two is the one that advances about one grid cell
-- a CFL condition -- and *that time is different in every cell*, because it
depends on how fast the robot is already going.

At rest, position hardly moves however long the step, so the binding axis is
velocity: `cell / a_max`. At speed, position crosses a cell first. Taking the
minimum over all six axes covers both, and covers rotation, which has its
own occupancy slice to cross.

`a_max` per body axis is the largest acceleration any admissible command can
produce. Over the octahedron `|u|_1 <= 1` that is simply the largest entry of
that row of `B`, since the maximum of a linear form over the unit 1-ball is
its largest coefficient. Traction saturation only lowers it, so using the
unsaturated figure errs toward a shorter step, which is the safe direction.
"""
@inline function cfl_tau(g::Grid6, m::Model, vfx, vfy, w, p::Params)
    p.cfl <= 0.0f0 && return p.dt
    B = m.B
    ax = max(abs(B[1]), abs(B[2]), abs(B[3]))
    ay = max(abs(B[4]), abs(B[5]), abs(B[6]))
    aw = max(abs(B[7]), abs(B[8]), abs(B[9]))

    t = p.tau_max
    # Velocity axes: time to cross `cfl` velocity cells at full command.
    ax > 1.0f-6 && (t = min(t, p.cfl * g.step[4] / ax))
    ay > 1.0f-6 && (t = min(t, p.cfl * g.step[5] / ay))
    aw > 1.0f-6 && (t = min(t, p.cfl * g.step[6] / aw))
    # Position: time to cross `cfl` position cells at the current speed.
    sp = sqrt(vfx * vfx + vfy * vfy)
    sp > 1.0f-3 && (t = min(t, p.cfl * min(g.step[1], g.step[2]) / sp))
    # Heading: time to cross `cfl` heading bins at the current spin rate.
    aw2 = abs(w)
    aw2 > 1.0f-4 && (t = min(t, p.cfl * g.step[3] / aw2))
    clamp(t, p.tau_min, p.tau_max)
end

"""
Substeps for a horizon of `tau`: enough that no substep exceeds `hmax`.

Tying the substep count to the horizon rather than fixing it is what makes
the step-length ladder safe. Integration error grows with the substep, and
unlike discretisation error it can push the value BELOW the truth -- the one
direction a minimum-time table must not be wrong in, because the robot would
then be steering by a promise the drivetrain cannot keep. Holding the substep
constant means a longer horizon is not silently a less accurate one.
"""
@inline function substeps_for(tau::Float32, p::Params)
    p.hmax <= 0.0f0 && return p.nsub
    k = unsafe_trunc(Int32, min(tau / p.hmax, 1.0f4)) + Int32(1)
    max(Int32(1), min(k, Int32(64)))
end

"""
What one sweep asks of the machine, per cell that is not blocked.

**Why an estimate needs this at all.** A cell update is not a fixed amount of
arithmetic. `cell_update` evaluates a fixed NUMBER of lookaheads -- coast, the
warm start, `control_scan` lattice entries, `6 * refine_rounds` pattern probes
and the step-length ladder -- but each of those costs `substeps_for` RK2
substeps and `lookahead`'s swept collision probes, and both of those are
derived from the horizon, which `cfl_tau` derives from the grid and the model.
So the same card, sweeping the same number of cells, runs at wildly different
cell rates depending on `cfl`, `tau_max`, the cell sizes and how much yaw
authority the drivetrain has. Measured on one card: 48 ns a cell at `cfl` 1 /
`tau_max` 0.2 against 160 ns at `cfl` 4 / `tau_max` 0.5, a spread of 3.3x.

Quoting a single `cell_rate` therefore only predicts runs that resemble the
one it was measured on, and the benchmark's case did not resemble production:
it under-quoted a real H200 job by 41%, which is an hour of rental on a
two-and-a-half hour solve.

Returned as three counts rather than one number so the caller can price the
per-lookahead overhead (interpolation, the `sincos`, the successor lookup)
separately from the per-substep work, which is what `cell_cost` does.

  * `lookaheads` -- calls to `lookahead` per cell
  * `substeps`   -- integration substeps summed over those calls
  * `probes`     -- swept collision probes summed over those calls

Exact for the first two. The probe count depends on how far the step actually
moves, which is not known without running the dynamics, so it is estimated
from the same `B`-row bound `cfl_tau` uses: `speed*tau + a_max*tau^2/2`. That
over-states displacement for a command fighting drag and under-states nothing,
which is the safe direction for a time estimate.

The average is over the velocity sub-grid only, because `cfl_tau` depends on
`(vx, vy, w)` and nothing else -- so this costs `n4*n5*n6` evaluations, a few
thousand, not one per cell. `plan` can call it in an interactive loop.
"""
function sweep_work(g::Grid6, m::Model, p::Params, nctl::Integer)
    B = m.B
    ax = max(abs(B[1]), abs(B[2]), abs(B[3]))
    ay = max(abs(B[4]), abs(B[5]), abs(B[6]))
    aw = max(abs(B[7]), abs(B[8]), abs(B[9]))
    axy = Float64(max(ax, ay))
    dxy = Float64(min(g.step[1], g.step[2]))
    dh = Float64(g.step[3])
    # The lookaheads every cell runs at `tau0`, exactly as `cell_update`
    # issues them: coast, the warm start, the lattice slice, the pattern
    # search. The warm start is counted always -- it is skipped only on the
    # first sweep, and on a cell whose incumbent is coast.
    nnc = max(Int(nctl) - 1, 1)
    span = p.ncoarse <= Int32(0) ? nnc : min(Int(p.ncoarse), nnc)
    nfix = 2 + span + 6 * Int(p.rounds)

    function nprobes(tau::Float64, speed::Float64, spin::Float64)
        p.adaptive_checks || return Float64(p.checks)
        dpos = speed * tau + 0.5 * axy * tau * tau
        drot = spin * tau + 0.5 * Float64(aw) * tau * tau
        cells = max(dpos / (0.5 * dxy), drot / (0.5 * dh))
        Float64(max(Int(p.checks), min(trunc(Int, min(cells, 1.0e4)) + 1, 24)))
    end

    nL = 0.0; nS = 0.0; nP = 0.0
    ncell = 0
    tmin = Float64(p.tau_min); tmax = Float64(p.tau_max)
    for i4 in 0:(Int(g.n[4]) - 1), i5 in 0:(Int(g.n[5]) - 1),
        i6 in 0:(Int(g.n[6]) - 1)
        vfx = axisvalue(g, 4, i4); vfy = axisvalue(g, 5, i5)
        w = axisvalue(g, 6, i6)
        speed = sqrt(Float64(vfx)^2 + Float64(vfy)^2); spin = abs(Float64(w))
        tau0 = Float64(cfl_tau(g, m, vfx, vfy, w, p))

        add(t, mult) = begin
            nL += mult
            nS += mult * Float64(substeps_for(Float32(t), p))
            nP += mult * nprobes(t, speed, spin)
        end
        add(tau0, nfix)
        if p.ntau > Int32(1)
            tau = tau0 * 0.5
            for k in 1:Int(p.ntau)
                k != 2 && add(clamp(tau, tmin, tmax), 1)
                tau *= Float64(p.tau_ratio)
            end
            # `cell_update` always tries the configured `dt` as well.
            p.cfl > 0.0f0 && add(clamp(Float64(p.dt), tmin, tmax), 1)
        end
        ncell += 1
    end
    n = Float64(max(ncell, 1))
    (lookaheads = nL / n, substeps = nS / n, probes = nP / n)
end

"""
Cost of holding one control for `tau`: `tau + V(successor)`, or `cap` if the
step is blocked.

**Why the probe count is derived rather than fixed.** The swept check exists
because a fast step can cross several cells, and sampling only the endpoint
tunnels through walls. A constant three probes is right for one particular
speed and wrong either side of it: wasteful when the robot has barely moved,
and unsafe once a step spans more than about one and a half cells -- which is
exactly what a longer `tau` does on purpose. Deriving the count from the
actual displacement, in cells, makes the guarantee independent of `dt`,
`tau` and the grid resolution.

The rotation is counted too, not just the translation: a chassis pivoting in
place covers no ground at all while sweeping through headings whose occupancy
slices are completely different.
"""
@inline function lookahead(V, occ, g::Grid6, m::Model, x, y, h, vx, vy, w,
                           u1, u2, u3, tau::Float32, nsub::Int32, p::Params)
    nx, ny, nh, nvx, nvy, nw = p.rk2 ?
        step_state_rk2(m, x, y, h, vx, vy, w, u1, u2, u3, tau, nsub) :
        step_state(m, x, y, h, vx, vy, w, u1, u2, u3, tau, nsub)

    dx = nx - x; dy = ny - y; dh = nh - h
    # A non-finite successor means the integration has diverged. It must be
    # rejected explicitly: every comparison against NaN is false, so an
    # unchecked NaN sails through the range tests in `interp` and indexes the
    # table with garbage -- the same failure mode a NaN traction knee once
    # caused, from a different direction.
    (isfinite(dx) && isfinite(dy) && isfinite(dh) &&
     isfinite(nvx) && isfinite(nvy) && isfinite(nw)) || return p.cap

    nchk = p.checks
    if p.adaptive_checks
        cells_pos = sqrt(dx * dx + dy * dy) / (0.5f0 * min(g.step[1], g.step[2]))
        cells_rot = abs(dh) / (0.5f0 * g.step[3])
        # Bounded before truncating: `unsafe_trunc` past Int32 range is
        # undefined, and a diverging step can reach a large finite number
        # before it reaches Inf.
        cells = min(max(cells_pos, cells_rot), 1.0f4)
        need = unsafe_trunc(Int32, cells) + Int32(1)
        nchk = max(p.checks, min(need, Int32(24)))
    end

    @inbounds for sc in 1:nchk
        a = Float32(sc) / Float32(nchk)
        px = x + dx * a
        py = y + dy * a
        ph = h + dh * a                    # the robot turns as it moves
        gx = (px - g.lo[1]) / g.step[1]
        gy = (py - g.lo[2]) / g.step[2]
        if gx < 0.0f0 || gy < 0.0f0 ||
           gx > Float32(g.n[1] - 1) || gy > Float32(g.n[2] - 1)
            return p.cap
        end
        n3f = Float32(g.n[3])
        gh = (ph - g.lo[3]) / g.step[3]
        gh = gh - floor(gh / n3f) * n3f
        ci = Int32(round(gx)); cj = Int32(round(gy))
        ck = Int32(round(gh)) % g.n[3]
        occ[(Int64(ci) * g.n[2] + cj) * g.n[3] + ck + 1] && return p.cap
    end

    # Rotate the successor's velocity back to the field frame, using the NEW
    # heading, before looking it up in the grid.
    snh, cnh = sincos(nh)
    nvfx = cnh * nvx - snh * nvy
    nvfy = snh * nvx + cnh * nvy
    nv = value_at(V, g, nx, ny, nh, nvfx, nvfy, nw, p)
    min(tau + nv, p.cap)
end

"""
Push a command back onto the octahedron surface along its own direction.

Used by the pattern search: a perturbed command generally leaves the
admissible set, and rescaling by the 1-norm is the cheapest way back that
keeps the direction. Rescaling rather than clipping is also what lets a
perturbation walk off one face and onto a neighbouring one -- a component
that crosses zero simply changes sign, so the search is free to move between
orthants instead of being trapped on the face it started from.
"""
@inline function proj_oct(u1, u2, u3)
    s = abs(u1) + abs(u2) + abs(u3)
    s < 1.0f-6 && return (0.0f0, 0.0f0, 0.0f0)
    r = 1.0f0 / s
    (u1 * r, u2 * r, u3 * r)
end

"""
One Bellman backup. Written to be valid in both a CPU loop and a GPU kernel:
scalar math, no allocation, no dynamic dispatch.

Returns `(value, u1, u2, u3)` -- the backed-up value and the command that
achieved it, so the caller can keep it as a warm start.

**How the minimisation over `u` is done.** Scanning the whole lattice every
sweep spends the entire budget rediscovering the same answer: the optimal
command at a cell barely moves from one sweep to the next, because `V` around
it barely moves. So each sweep evaluates

  * coasting, which is always admissible and is the tie-break;
  * the command this cell chose last sweep (`pol`), the warm start;
  * a rotating slice of the lattice, `ncoarse` entries starting at an offset
    that advances every sweep, so the whole lattice is still swept -- just
    spread over several sweeps rather than all at once;
  * `rounds` of pattern search around whichever of those is currently best,
    with the radius halving each round.

The pattern search is what makes the command **continuous**: the lattice only
ever seeds it, and the refinement then moves off-lattice by arbitrary
amounts. That is worth having because the optimum genuinely is not at a
lattice point -- the traction term `tanh(|u|/knee)/(|u|/knee)` varies over
the octahedron surface, so the best command sits wherever alignment with the
value gradient balances against saturation, which no fixed lattice hits.

Every candidate is a real Bellman evaluation, never an approximation, so the
result is still an upper bound on the true value and the iteration is still
monotone. A restricted candidate set can only slow convergence down, never
move the fixed point -- which is why the stopping rule in `solve_value!` has
to see a quiet sweep for a whole rotation of the coarse offset, not just one.
"""
@inline function cell_update(idx::Int64, V0, occ0, pol, g::Grid6, m::Model,
                             ctl, nctl::Int32, p::Params, phase::Int32,
                             ::Val{ESC} = Val(false)) where {ESC}
    i1, i2, i3, i4, i5, i6 = unflatten(g, idx)

    # Occupancy is (x, y, heading): a chassis with real extent blocks
    # different cells depending on which way it is pointing.
    @inbounds blocked = occ0[(Int64(i1) * g.n[2] + i2) * g.n[3] + i3 + 1]

    # The escape pass runs this same backup over the cells the main solve
    # could not reach, against a different reading of `V` and of `occ`. Both
    # substitutions are by dispatch on a `Val`, so the ordinary sweep compiles
    # to exactly what it did before: `ESC` is false, `esc_v` and `esc_occ`
    # return their argument unchanged, and the blocked test below is the same
    # early return it always was.
    V = esc_v(V0, occ0, Int64(g.n[4]) * Int64(g.n[5]) * Int64(g.n[6]), p.cap,
              blocked, Val(ESC))
    occ = esc_occ(occ0, blocked, Val(ESC))
    if !ESC
        blocked && return (p.cap, 0.0f0, 0.0f0, 0.0f0)
    end

    x  = axisvalue(g, 1, i1); y   = axisvalue(g, 2, i2)
    h  = axisvalue(g, 3, i3); vfx = axisvalue(g, 4, i4)
    vfy = axisvalue(g, 5, i5); w  = axisvalue(g, 6, i6)

    # The grid stores FIELD-frame velocity, because that is what the robot
    # already has from odometry and it saves a rotation every loop cycle. The
    # drivetrain model is body-frame, so rotate in here and back out below.
    sh0, ch0 = sincos(h)
    vx =  ch0 * vfx + sh0 * vfy
    vy = -sh0 * vfx + ch0 * vfy

    ncell = ncells(g)

    # The horizon this cell searches at, and the substeps that keep the
    # integration error of that horizon bounded.
    tau0 = cfl_tau(g, m, vfx, vfy, w, p)
    ns0 = substeps_for(tau0, p)

    # Coasting: always admissible, and the tie-break, so it goes first.
    best = lookahead(V, occ, g, m, x, y, h, vx, vy, w,
                     0.0f0, 0.0f0, 0.0f0, tau0, ns0, p)
    b1 = 0.0f0; b2 = 0.0f0; b3 = 0.0f0

    # The warm start: what this cell chose last sweep.
    if pol !== nothing
        q1, q2, q3 = pol_get(pol, idx, ncell)
        if abs(q1) + abs(q2) + abs(q3) > 1.0f-6
            c = lookahead(V, occ, g, m, x, y, h, vx, vy, w, q1, q2, q3,
                          tau0, ns0, p)
            if c < best
                best = c; b1 = q1; b2 = q2; b3 = q3
            end
        end
    end

    # Lattice scan: the whole table, or a rotating slice of it.
    nnc = max(nctl - Int32(1), Int32(1))       # entries after coast
    span = p.ncoarse <= Int32(0) ? nnc : min(p.ncoarse, nnc)
    off  = p.ncoarse <= Int32(0) ? Int32(0) :
           Int32(mod(Int64(phase) * Int64(span), Int64(nnc)))
    @inbounds for j in Int32(0):(span - Int32(1))
        k = Int32(mod(Int64(off) + Int64(j), Int64(nnc))) + Int32(2)   # skip coast
        u1 = ctl[k]; u2 = ctl[k+nctl]; u3 = ctl[k+2*nctl]
        c = lookahead(V, occ, g, m, x, y, h, vx, vy, w, u1, u2, u3,
                      tau0, ns0, p)
        if c < best
            best = c; b1 = u1; b2 = u2; b3 = u3
        end
    end

    # Pattern search: walk the incumbent across the octahedron surface.
    delta = p.delta0
    for _ in Int32(1):p.rounds
        for d in Int32(1):Int32(6)
            s = iseven(d) ? -delta : delta
            e1 = d <= Int32(2) ? s : 0.0f0
            e2 = (d == Int32(3) || d == Int32(4)) ? s : 0.0f0
            e3 = d >= Int32(5) ? s : 0.0f0
            c1, c2, c3 = proj_oct(b1 + e1, b2 + e2, b3 + e3)
            (abs(c1) + abs(c2) + abs(c3)) < 1.0f-6 && continue
            c = lookahead(V, occ, g, m, x, y, h, vx, vy, w, c1, c2, c3,
                          tau0, ns0, p)
            if c < best
                best = c; b1 = c1; b2 = c2; b3 = c3
            end
        end
        delta *= 0.5f0     # each round looks half as far as the last
    end

    # Step-length ladder on the incumbent.
    #
    # The dynamic programming principle holds for ANY lookahead horizon, so
    # taking the minimum over several is a valid backup -- and a strictly
    # tighter one, since it can only lower the result. `cfl_tau` already
    # picked the horizon that should be about right for this cell; the ladder
    # brackets it, half as long through several times as long, because "about
    # right" is an estimate from the linearised geometry and the true best
    # horizon depends on how `V` is shaped nearby.
    #
    # The ladder is geometric and centred on `tau0` rather than climbing away
    # from it, so a cell that wants a shorter step can have one. Near the
    # target that matters: the last approach needs finer resolution than the
    # open-field cruise that got there.
    if p.ntau > Int32(1)
        tau = tau0 * 0.5f0
        for k in Int32(1):p.ntau
            if k != Int32(2)                 # k = 2 is tau0, already done
                t = min(max(tau, p.tau_min), p.tau_max)
                c = lookahead(V, occ, g, m, x, y, h, vx, vy, w, b1, b2, b3,
                              t, substeps_for(t, p), p)
                c < best && (best = c)
            end
            tau *= p.tau_ratio
        end

        # And always try the configured `dt` as well.
        #
        # "More horizons can only lower V" is true of a SUPERSET of horizons
        # and false of a different set of them. With `cfl > 0` the ladder is
        # built around a grid-derived `tau0`, which need not contain the fixed
        # `dt` a run would otherwise have used -- so the ladder can, on some
        # problems, be strictly worse than the single horizon it replaced.
        # Measured: +4.8% on a double integrator whose grid happened to suit
        # dt = 0.05 better than any rung around `tau0`.
        #
        # One extra evaluation restores the guarantee, and it is a guarantee
        # worth having: it makes the scheme provably no worse than the fixed
        # step it supersedes, on every problem rather than on average.
        if p.cfl > 0.0f0
            t = min(max(p.dt, p.tau_min), p.tau_max)
            c = lookahead(V, occ, g, m, x, y, h, vx, vy, w, b1, b2, b3,
                          t, substeps_for(t, p), p)
            c < best && (best = c)
        end
    end

    (best, b1, b2, b3)
end

include("Pipeline.jl")
include("Tiles.jl")
include("Advise.jl")
include("Gpu.jl")
include("OutOfCore.jl")
include("Run.jl")
include("SelfTest.jl")

end # module
